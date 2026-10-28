"""Private SQLite/provider admission tests. Every HTTP opener is authored; no API calls."""
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
import json
from pathlib import Path
import sqlite3
import stat
import tempfile
import threading
import unittest
from unittest.mock import patch
from urllib.error import HTTPError, URLError

from proofloop.inference_journal import AttemptJournal, JournaledMetaClient
from proofloop.meta import MetaAPIError, SAM_MODEL, SPARK_MODEL
from proofloop.smoke import SMOKE_SCHEMA


def response(value):
    return BytesIO(json.dumps(value).encode())


def completed(value=None, *, usage=None):
    if value is None:
        value = {"project": "ProofLoop", "requires_fresh_evidence": True, "result_when_occluded": "unknown"}
    return {"status": "completed", "output": [{"type": "message", "content": [
        {"type": "output_text", "text": json.dumps(value)}]}],
        "usage": usage if usage is not None else {"input_tokens": 10, "output_tokens": 20}}


class InferenceJournalTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "private" / "attempts.sqlite3"
        self.journal = AttemptJournal(self.path, request_limit=20)
        self.operation = "a" * 32

    def tearDown(self):
        self.temp.cleanup()

    def client(self, opener=None, **kwargs):
        return JournaledMetaClient("authored-secret-key-not-real", journal=self.journal,
            operation_id=self.operation, guard=lambda: None,
            opener=opener or (lambda *args, **kwargs: response(completed())), **kwargs)

    def reserve(self, *, operation=None, limit=4):
        return self.journal.reserve(operation or self.operation, "responses", {"model": SPARK_MODEL},
                                    operation_limit=limit)

    def rows(self):
        with sqlite3.connect(str(self.path)) as db:
            db.row_factory = sqlite3.Row
            return [dict(row) for row in db.execute("SELECT * FROM attempts ORDER BY started_ms,id")]

    def test_spark_success_is_reserved_before_network_and_settled_with_numeric_usage(self):
        seen = []
        def opener(request, **kwargs):
            pending = self.journal.summary(self.operation)
            self.assertEqual(1, pending["operation_attempts"])
            self.assertEqual(1, pending["outcomes"]["reserved"])
            self.assertTrue(pending["unsettled_outcome_is_unknown"])
            seen.append(json.loads(request.data))
            return response(completed())
        client = self.client(opener)
        result = client.spark_json("Authored transport test", SMOKE_SCHEMA, reasoning_effort="low")
        self.assertEqual("ProofLoop", result["parsed"]["project"])
        self.assertEqual(SPARK_MODEL, seen[0]["model"])
        summary = self.journal.summary(self.operation)
        self.assertEqual({"reserved": 0, "ok": 1, "rejected": 0, "uncertain": 0}, summary["outcomes"])
        self.assertEqual({"input_tokens": 10, "output_tokens": 20, "total_tokens": 0}, summary["reported_tokens"])
        self.assertFalse(summary["automatic_retry"])
        self.assertFalse(summary["hard_dollar_limit"])
        self.assertEqual(1, len(client.ledger))

    def test_accepted_model_list_is_also_a_counted_provider_request(self):
        client = self.client(lambda *args, **kwargs: response({"data": [{"id": SPARK_MODEL}, {"id": SAM_MODEL}]}))
        self.assertEqual([SPARK_MODEL, SAM_MODEL], client.list_models())
        self.assertEqual(1, self.journal.summary()["total_reserved_attempts"])
        row = self.rows()[0]
        self.assertEqual("models", row["endpoint"])
        self.assertIsNone(row["model"])
        self.assertEqual("ok", row["state"])

    def test_guard_runs_before_reservation_and_http_attempt(self):
        calls = []
        def guard():
            raise ValueError("The task binding changed")
        client = JournaledMetaClient("authored-key", journal=self.journal, operation_id=self.operation,
            guard=guard, opener=lambda *args, **kwargs: calls.append(1))
        with self.assertRaisesRegex(ValueError, "binding changed"):
            client.spark_json("Must not leave process", SMOKE_SCHEMA)
        self.assertEqual([], calls)
        self.assertEqual([], client.ledger)
        self.assertEqual(0, self.journal.summary()["total_reserved_attempts"])

    def test_guard_is_rechecked_before_every_call_not_only_client_creation(self):
        guards, network = [], []
        def guard():
            guards.append(1)
            if len(guards) > 1:
                raise ValueError("The inspection lease was lost")
        def opener(*args, **kwargs):
            network.append(1)
            return response(completed())
        client = JournaledMetaClient("authored-key", journal=self.journal, operation_id=self.operation,
                                    guard=guard, opener=opener)
        client.spark_json("First attempt", SMOKE_SCHEMA)
        with self.assertRaisesRegex(ValueError, "lease was lost"):
            client.spark_json("Second attempt", SMOKE_SCHEMA)
        self.assertEqual(2, len(guards))
        self.assertEqual(1, len(network))
        self.assertEqual(1, self.journal.summary(self.operation)["operation_attempts"])

    def test_incomplete_spark_retains_usage_and_consumes_allowance_without_retry(self):
        calls = []
        value = {"status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"},
                 "usage": {"input_tokens": 100, "output_tokens": 80, "total_tokens": 180,
                           "output_tokens_details": {"reasoning_tokens": 79}}}
        def opener(*args, **kwargs):
            calls.append(1)
            return response(value)
        client = self.client(opener, max_requests=1)
        with self.assertRaises(MetaAPIError) as caught:
            client.spark_json("Authored incomplete request", SMOKE_SCHEMA, reasoning_effort="low")
        self.assertEqual("incomplete_response", caught.exception.code)
        summary = self.journal.summary(self.operation)
        self.assertEqual(1, summary["outcomes"]["rejected"])
        self.assertEqual({"input_tokens": 100, "output_tokens": 80, "total_tokens": 180}, summary["reported_tokens"])
        self.assertEqual(80, client.usage_summary()["output_tokens"])
        with self.assertRaises(MetaAPIError):
            client.spark_json("No refund", SMOKE_SCHEMA)
        self.assertEqual(1, len(calls))
        self.assertEqual(1, self.journal.summary()["total_reserved_attempts"])

    def test_http_error_retains_reported_usage_and_never_stores_provider_body(self):
        body = {"error": {"code": "rate_limited", "message": "private-provider-text"},
                "usage": {"input_tokens": 7, "output_tokens": 9, "total_tokens": 16}}
        calls = []
        def opener(*args, **kwargs):
            calls.append(1)
            raise HTTPError("https://api.meta.ai/v1/responses", 429, "private reason", {}, response(body))
        client = self.client(opener)
        with self.assertRaises(MetaAPIError) as caught:
            client.spark_json("Private authored prompt", SMOKE_SCHEMA)
        self.assertEqual(429, caught.exception.status)
        self.assertEqual(1, len(calls))
        summary = self.journal.summary(self.operation)
        self.assertEqual(1, summary["outcomes"]["rejected"])
        self.assertEqual(16, summary["reported_tokens"]["total_tokens"])
        self.assertNotIn("private-provider-text", json.dumps(self.rows()))

    def test_network_timeout_or_interruption_is_uncertain_counted_and_never_retried(self):
        for failure in (URLError("authored transport failure"), TimeoutError("timeout"), KeyboardInterrupt()):
            operation = {URLError: "b", TimeoutError: "c", KeyboardInterrupt: "d"}[type(failure)] * 32
            calls, sleeps = [], []
            def opener(*args, **kwargs):
                calls.append(1)
                raise failure
            client = JournaledMetaClient("authored-key", journal=self.journal, operation_id=operation,
                guard=lambda: None, opener=opener, sleeper=lambda value: sleeps.append(value), max_requests=1)
            expected = KeyboardInterrupt if isinstance(failure, KeyboardInterrupt) else MetaAPIError
            with self.subTest(error=type(failure).__name__):
                with self.assertRaises(expected):
                    client.spark_json("Authored ambiguous request", SMOKE_SCHEMA)
                self.assertEqual(1, self.journal.summary(operation)["outcomes"]["uncertain"])
                with self.assertRaises(MetaAPIError):
                    client.spark_json("No automatic or manual allowance refund", SMOKE_SCHEMA)
                self.assertEqual(1, len(calls))
                self.assertEqual([], sleeps)

    def test_schema_rejected_generation_counts_as_completed_provider_transport_not_valid_evidence(self):
        client = self.client(lambda *args, **kwargs: response(completed({"wrong": "schema"})))
        with self.assertRaises(MetaAPIError) as caught:
            client.spark_json("Authored wrong schema", SMOKE_SCHEMA)
        self.assertEqual("schema_validation_failed", caught.exception.code)
        summary = self.journal.summary(self.operation)
        # The durable wrapper surrounds HTTP attempts, not downstream evidence
        # acceptance. 'ok' must never be presented as a verified task outcome.
        self.assertEqual(1, summary["outcomes"]["ok"])
        self.assertEqual(20, summary["reported_tokens"]["output_tokens"])
        self.assertEqual(1, summary["total_reserved_attempts"])

    def test_invalid_provider_json_is_rejected_and_counted_without_retry(self):
        calls = []
        def opener(*args, **kwargs):
            calls.append(1)
            return BytesIO(b"not-valid-json-private-body")
        client = self.client(opener)
        with self.assertRaises(MetaAPIError) as caught:
            client.spark_json("Authored invalid JSON", SMOKE_SCHEMA)
        self.assertEqual("invalid_json", caught.exception.code)
        self.assertEqual(1, len(calls))
        self.assertEqual(1, self.journal.summary(self.operation)["outcomes"]["rejected"])

    def test_reopening_does_not_refund_unsettled_or_finished_attempts(self):
        first = self.reserve(limit=2)
        second = self.reserve(limit=2)
        self.journal.settle(second, state="rejected", usage={"input_tokens": 2})
        reopened = AttemptJournal(self.path, request_limit=20)
        summary = reopened.summary(self.operation)
        self.assertEqual(2, summary["operation_attempts"])
        self.assertEqual(1, summary["outcomes"]["reserved"])
        self.assertEqual(1, summary["outcomes"]["rejected"])
        self.assertTrue(summary["unsettled_outcome_is_unknown"])
        with self.assertRaises(MetaAPIError):
            reopened.reserve(self.operation, "responses", {"model": SPARK_MODEL}, operation_limit=2)
        reopened.settle(first, state="uncertain")
        self.assertEqual(2, reopened.summary(self.operation)["operation_attempts"])

    def test_summary_keeps_one_snapshot_while_an_independent_writer_reserves_or_settles(self):
        for action in ("reserve", "settle"):
            with self.subTest(action=action):
                path = Path(self.temp.name) / (action + "-snapshot.sqlite3")
                reader = AttemptJournal(path, request_limit=20)
                writer = AttemptJournal(path, request_limit=20)
                original = reader.reserve(self.operation, "responses", {"model": SPARK_MODEL}, operation_limit=4)
                read_connect, write_connect = reader._connect, writer._connect
                writer_at_commit = threading.Event()
                reader_closed = threading.Event()
                select_transactions, futures = [], []
                operation = self.operation

                class WriterProxy:
                    def __init__(self, connection):
                        self.connection = connection

                    def __getattr__(self, name):
                        return getattr(self.connection, name)

                    def execute(self, sql, *args):
                        if action == "settle" and sql.startswith("UPDATE attempts"):
                            # Autocommit UPDATE blocks until the active DELETE-
                            # journal reader releases its snapshot. Signal before
                            # entering it, never wait for a blocked commit here.
                            writer_at_commit.set()
                        return self.connection.execute(sql, *args)

                    def commit(self):
                        writer_at_commit.set()
                        return self.connection.commit()

                def mutate():
                    if action == "reserve":
                        writer.reserve(operation, "responses", {"model": SPARK_MODEL}, operation_limit=4)
                    else:
                        writer.settle(original, state="ok", usage={"input_tokens": 77})

                with ThreadPoolExecutor(max_workers=1) as pool:
                    class ReaderProxy:
                        def __init__(self, connection):
                            self.connection = connection

                        def __getattr__(self, name):
                            return getattr(self.connection, name)

                        def execute(proxy, sql, *args):
                            if sql.startswith("SELECT"):
                                select_transactions.append(proxy.connection.in_transaction)
                            if sql == "SELECT count(*) FROM attempts":
                                # The first SELECT has already been consumed.
                                # Force a real, separate writer between the two
                                # reads; wait only for its pre-commit point.
                                futures.append(pool.submit(mutate))
                                self.assertTrue(writer_at_commit.wait(5), "Writer never reached mutation/commit")
                            return proxy.connection.execute(sql, *args)

                        def close(proxy):
                            proxy.connection.close()
                            reader_closed.set()

                    with patch.object(reader, "_connect", side_effect=lambda: ReaderProxy(read_connect())), \
                         patch.object(writer, "_connect", side_effect=lambda: WriterProxy(write_connect())):
                        snapshot = reader.summary(operation)
                        self.assertTrue(reader_closed.is_set())
                        futures[0].result(timeout=5)
                self.assertEqual([True, True], select_transactions)
                self.assertEqual(1, snapshot["total_reserved_attempts"])
                self.assertEqual(1, snapshot["operation_attempts"])
                self.assertEqual(1, snapshot["outcomes"]["reserved"])
                self.assertEqual(0, snapshot["reported_tokens"]["input_tokens"])
                latest = reader.summary(operation)
                if action == "reserve":
                    self.assertEqual(2, latest["total_reserved_attempts"])
                    self.assertEqual(2, latest["outcomes"]["reserved"])
                else:
                    self.assertEqual(1, latest["total_reserved_attempts"])
                    self.assertEqual(1, latest["outcomes"]["ok"])
                    self.assertEqual(77, latest["reported_tokens"]["input_tokens"])

    def test_global_cap_is_shared_across_independent_connections_and_operations(self):
        path = Path(self.temp.name) / "global-cap.sqlite3"
        journals = [AttemptJournal(path, request_limit=3) for _ in range(12)]
        barrier = threading.Barrier(len(journals))
        def reserve(index):
            barrier.wait()
            try:
                return journals[index].reserve(f"{index:032x}", "responses", {"model": SPARK_MODEL}, operation_limit=1)
            except MetaAPIError as error:
                return error.code
        with ThreadPoolExecutor(max_workers=len(journals)) as pool:
            results = list(pool.map(reserve, range(len(journals))))
        self.assertEqual(3, sum(result != "persistent_request_limit" for result in results))
        self.assertEqual(9, results.count("persistent_request_limit"))
        summary = journals[0].summary()
        self.assertEqual(3, summary["total_reserved_attempts"])
        self.assertEqual(0, summary["remaining_attempts"])
        self.assertEqual(3, summary["outcomes"]["reserved"])

    def test_operation_cap_is_shared_across_independent_connections_concurrently(self):
        journals = [AttemptJournal(self.path, request_limit=20) for _ in range(10)]
        barrier = threading.Barrier(len(journals))
        def reserve(index):
            barrier.wait()
            try:
                return journals[index].reserve(self.operation, "responses", {"model": SAM_MODEL}, operation_limit=2)
            except MetaAPIError as error:
                return error.code
        with ThreadPoolExecutor(max_workers=len(journals)) as pool:
            results = list(pool.map(reserve, range(len(journals))))
        self.assertEqual(2, sum(result != "persistent_request_limit" for result in results))
        self.assertEqual(8, results.count("persistent_request_limit"))
        self.assertEqual(2, self.journal.summary(self.operation)["operation_attempts"])

    def test_new_adapter_instance_cannot_reset_the_same_operation_http_allowance(self):
        calls = []
        def opener(*args, **kwargs):
            calls.append(1)
            return response(completed())
        first = self.client(opener, max_requests=1)
        first.spark_json("One admitted request", SMOKE_SCHEMA)
        second = JournaledMetaClient("another-authored-key", journal=AttemptJournal(self.path, request_limit=20),
            operation_id=self.operation, guard=lambda: None, opener=opener, max_requests=1)
        with self.assertRaises(MetaAPIError) as caught:
            second.spark_json("A new client is not a new allowance", SMOKE_SCHEMA)
        self.assertEqual("persistent_request_limit", caught.exception.code)
        self.assertEqual(1, len(calls))
        self.assertEqual([], second.ledger)

    def test_global_limit_stops_a_different_real_adapter_before_its_http_opener(self):
        path = Path(self.temp.name) / "one-global-request.sqlite3"
        calls = []
        def opener(*args, **kwargs):
            calls.append(1)
            return response(completed())
        for index in range(2):
            client = JournaledMetaClient("authored-key", journal=AttemptJournal(path, request_limit=1),
                operation_id=f"{index:032x}", guard=lambda: None, opener=opener, max_requests=1)
            if index == 0:
                client.spark_json("First operation", SMOKE_SCHEMA)
            else:
                with self.assertRaises(MetaAPIError) as caught:
                    client.spark_json("Different operation, same persistent global cap", SMOKE_SCHEMA)
                self.assertEqual("persistent_request_limit", caught.exception.code)
                self.assertEqual([], client.ledger)
        self.assertEqual(1, len(calls))

    def test_global_allowance_configuration_is_persisted_not_caller_expandable(self):
        self.reserve()
        for changed_limit in (1, 21):
            with self.subTest(limit=changed_limit), self.assertRaises(ValueError):
                AttemptJournal(self.path, request_limit=changed_limit)
        self.assertEqual(20, AttemptJournal(self.path, request_limit=20).summary()["request_limit"])

    def test_operation_allowance_configuration_is_persisted_not_caller_expandable(self):
        self.reserve(limit=1)
        for changed_limit in (2, 13):
            reopened = AttemptJournal(self.path, request_limit=20)
            with self.subTest(limit=changed_limit), self.assertRaises(ValueError):
                reopened.reserve(self.operation, "responses", {"model": SPARK_MODEL}, operation_limit=changed_limit)
        self.assertEqual(1, self.journal.summary(self.operation)["operation_attempts"])

    def test_settlement_is_one_way_and_cannot_refund_or_duplicate_usage(self):
        attempt = self.reserve()
        self.journal.settle(attempt, state="uncertain", usage={"input_tokens": 3})
        for state in ("ok", "uncertain", "rejected"):
            with self.subTest(state=state), self.assertRaises(ValueError):
                self.journal.settle(attempt, state=state, usage={"input_tokens": 999})
        with self.assertRaises(ValueError):
            self.journal.settle("f" * 32, state="ok")
        self.assertEqual(3, self.journal.summary()["reported_tokens"]["input_tokens"])
        self.assertEqual(1, self.journal.summary()["total_reserved_attempts"])

    def test_no_credentials_prompts_images_provider_text_or_nested_usage_are_stored(self):
        key, prompt = "secret-key-not-real-unique-marker", "private-prompt-unique-marker"
        image_data, body = "data:image/jpeg;base64,private-image-marker", "private-provider-body-marker"
        def opener(*args, **kwargs):
            value = completed(usage={"input_tokens": 1, "output_tokens": 2, "total_tokens": 3,
                                     "secret": body, "output_tokens_details": {"reasoning_tokens": body}})
            value["private"] = body
            return response(value)
        client = JournaledMetaClient(key, journal=self.journal, operation_id=self.operation,
                                    guard=lambda: None, opener=opener)
        client._request("POST", "responses", {"model": SPARK_MODEL,
            "input": [{"content": [{"type": "input_text", "text": prompt},
                                      {"type": "input_image", "image_url": image_data}]}]})
        serialized = json.dumps(self.rows())
        file_bytes = self.path.read_bytes()
        for marker in (key, prompt, image_data, body):
            self.assertNotIn(marker, serialized)
            self.assertNotIn(marker.encode(), file_bytes)
        row = self.rows()[0]
        self.assertRegex(row["request_sha256"], r"^[0-9a-f]{64}$")
        self.assertEqual({"input_tokens": 1, "output_tokens": 2, "total_tokens": 3}, json.loads(row["usage_json"]))
        self.assertEqual(0o600, stat.S_IMODE(self.path.stat().st_mode))

    def test_usage_sanitizer_rejects_booleans_negative_oversized_and_freeform_values(self):
        attempt = self.reserve()
        self.journal.settle(attempt, state="ok", usage={"input_tokens": True, "output_tokens": -1,
            "total_tokens": 2 ** 63, "secret": "never stored"})
        self.assertEqual("{}", self.rows()[0]["usage_json"])
        self.assertEqual({"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}, self.journal.summary()["reported_tokens"])

    def test_invalid_limits_operations_endpoints_and_payloads_do_not_reserve(self):
        for limit in (0, -1, True, 100_001, 1.5):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                AttemptJournal(Path(self.temp.name) / "not-created.sqlite3", request_limit=limit)
        for operation, endpoint, payload, limit in (
            ("short", "responses", None, 1), ("A" * 32, "responses", None, 1),
            (self.operation, "unsupported", None, 1), (self.operation, "responses", None, 0),
            (self.operation, "responses", None, 14), (self.operation, "responses", None, True),
            (self.operation, "responses", {"model": "unsupported-model"}, 1),
            (self.operation, "responses", {"model": SPARK_MODEL, "value": float("nan")}, 1),
        ):
            with self.subTest(operation=operation, endpoint=endpoint, limit=limit), self.assertRaises(ValueError):
                self.journal.reserve(operation, endpoint, payload, operation_limit=limit)
        self.assertEqual(0, self.journal.summary()["total_reserved_attempts"])

    def test_automatic_retry_configuration_is_rejected(self):
        for retries in (1, 2, True):
            with self.subTest(retries=retries), self.assertRaises(ValueError):
                self.client(retries=retries)
        self.assertEqual(0, self.journal.summary()["total_reserved_attempts"])

    def test_symlink_or_directory_cannot_be_used_as_private_journal(self):
        target = Path(self.temp.name) / "target"
        target.write_text("must remain untouched")
        link = Path(self.temp.name) / "link"
        link.symlink_to(target)
        for path in (link, Path(self.temp.name)):
            with self.subTest(path=path), self.assertRaises(ValueError):
                AttemptJournal(path)
        self.assertEqual("must remain untouched", target.read_text())


if __name__ == "__main__":
    unittest.main()
