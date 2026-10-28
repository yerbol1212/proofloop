"""Durable queue tests: authored jobs, no model/network requests."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
import threading
import unittest

from proofloop.jobs import (JobConflict, JobIntegrityError, JobLeaseError, JobNotFound,
    JobStore, JobValidationError, MAX_PAYLOAD_BYTES)


BINDING = {"session_id": "interactive-123abc", "task_hash": "a" * 64,
    "protocol_hash": "b" * 64, "revision": 4}


class JobStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "private-jobs" / "jobs.sqlite3"
        self.store = JobStore(self.path)

    def submit(self, key="request-1", payload=None, **kwargs):
        return self.store.submit("inspection", {"image": "private-base64", "capture_token": "private-token"} if payload is None else payload,
            idempotency_key=key, binding=deepcopy(BINDING), now_ms=10, **kwargs)

    def claim(self, *, dispatch=False, lease_ms=100):
        job = self.submit()
        claimed = self.store.claim_next("worker-1", now_ms=20, lease_ms=lease_ms)
        if dispatch:
            self.store.mark_dispatched(job["id"], claimed["lease"]["claim_token"], now_ms=21)
        return job, claimed["lease"]["claim_token"]

    def test_exact_retries_return_same_job_and_do_not_add_events(self):
        job = self.submit()
        retry = self.submit()
        self.assertEqual(job, retry)
        self.assertEqual(1, len(self.store.list_jobs()))
        self.assertEqual(["submitted"], [event["event_type"] for event in self.store.events(job["id"])])
        self.assertTrue(self.store.verify(job["id"])["valid"])

    def test_lookup_is_read_only_and_returns_historical_completed_reference(self):
        payload = {"image": "private-base64", "capture_token": "private-token"}
        self.assertIsNone(self.store.lookup_submission("inspection", payload,
            idempotency_key="request-1", binding=BINDING))
        self.assertEqual([], self.store.list_jobs())
        job, token = self.claim(dispatch=True)
        completed = self.store.complete(job["id"], token, {"revision": 5, "status": "pass"}, now_ms=22)
        reopened = JobStore(self.path)
        self.assertEqual(completed, reopened.lookup_submission("inspection", payload,
            idempotency_key="request-1", binding=BINDING))
        self.assertEqual(4, reopened.get(job["id"])["event_count"])
        self.assertIsNone(reopened.claim_next("another-worker", now_ms=23))
        for other_payload, binding in (({"image": "different"}, BINDING),
                (payload, {**BINDING, "revision": 9}), (payload, {**BINDING, "session_id": "new-session"})):
            with self.assertRaises(JobConflict):
                reopened.lookup_submission("inspection", other_payload, idempotency_key="request-1", binding=binding)

    def test_lookup_rejects_changed_payload_and_journal_before_returning_public_reference(self):
        payload = {"image": "private-base64", "capture_token": "private-token"}
        job = self.submit()
        with sqlite3.connect(str(self.path)) as connection:
            connection.execute("UPDATE jobs SET payload_json='{}' WHERE id=?", (job["id"],))
        with self.assertRaises(JobIntegrityError):
            self.store.lookup_submission("inspection", payload, idempotency_key="request-1", binding=BINDING)

    def test_idempotency_canonicalizes_order_but_conflicts_on_changed_payload_binding_or_kind(self):
        first = self.submit(payload={"b": 2, "a": 1})
        self.assertEqual(first, self.submit(payload={"a": 1, "b": 2}))
        with self.assertRaises(JobConflict):
            self.submit(payload={"a": 2, "b": 2})
        for field, value in (("revision", 5), ("task_hash", "c" * 64), ("protocol_hash", "c" * 64), ("session_id", "other-session")):
            binding = {**BINDING, field: value}
            with self.assertRaises(JobConflict):
                self.store.submit("inspection", {"a": 1, "b": 2}, idempotency_key="request-1", binding=binding, now_ms=10)
        with self.assertRaises(JobValidationError):
            self.store.submit("other", {}, idempotency_key="request-1", binding=BINDING, now_ms=10)
        self.assertEqual(1, self.store.get(first["id"])["event_count"])

    def test_concurrent_separate_instances_submit_once(self):
        stores = [JobStore(self.path) for _ in range(8)]
        barrier = threading.Barrier(len(stores))
        def submit(store):
            barrier.wait()
            return store.submit("inspection", {"a": 1}, idempotency_key="concurrent", binding=BINDING, now_ms=10)
        with ThreadPoolExecutor(max_workers=8) as executor:
            results = list(executor.map(submit, stores))
        self.assertEqual(1, len({job["id"] for job in results}))
        self.assertEqual(1, results[0]["event_count"])
        self.assertEqual(1, len(self.store.list_jobs()))

    def test_concurrent_claim_is_atomic_and_never_assigns_two_workers(self):
        submitted = self.submit()
        stores = [JobStore(self.path) for _ in range(8)]
        barrier = threading.Barrier(len(stores))
        def claim(pair):
            index, store = pair
            barrier.wait()
            return store.claim_next("worker-" + str(index), now_ms=20)
        with ThreadPoolExecutor(max_workers=8) as executor:
            values = list(executor.map(claim, enumerate(stores)))
        claimed = [value for value in values if value is not None]
        self.assertEqual(1, len(claimed))
        self.assertEqual(submitted["id"], claimed[0]["id"])
        self.assertEqual(1, claimed[0]["attempt"])
        self.assertTrue(self.store.verify(submitted["id"])["valid"])

    def test_claim_order_and_bounded_listing(self):
        first = self.submit("first")
        second = self.store.submit("inspection", {}, idempotency_key="second", binding=BINDING, now_ms=11)
        self.assertEqual(first["id"], self.store.claim_next("w1", now_ms=20)["id"])
        self.assertEqual(second["id"], self.store.claim_next("w2", now_ms=20)["id"])
        self.assertIsNone(self.store.claim_next("w3", now_ms=20))
        self.assertEqual(1, len(self.store.list_jobs(limit=1)))
        for limit in (True, 0, 101, -1, "2"):
            with self.assertRaises(JobValidationError):
                self.store.list_jobs(limit=limit)

    def test_success_requires_recorded_dispatch_then_seals_reference(self):
        job, token = self.claim()
        with self.assertRaises(JobConflict):
            self.store.complete(job["id"], token, {"revision": 5, "status": "pass"}, now_ms=21)
        self.store.mark_dispatched(job["id"], token, now_ms=22)
        result = {"revision": 5, "status": "pass", "receipt_id": "c" * 24, "result_sha256": "d" * 64}
        completed = self.store.complete(job["id"], token, result, now_ms=23)
        self.assertEqual("succeeded", completed["state"])
        self.assertEqual(result, completed["result"])
        self.assertIsNone(completed["lease_expires_ms"])
        self.assertEqual(["submitted", "claimed", "dispatched", "succeeded"],
            [event["event_type"] for event in self.store.events(job["id"])])
        self.assertTrue(self.store.verify(job["id"])["valid"])

    def test_dispatch_marker_is_one_time_and_never_authorizes_repeat(self):
        job, token = self.claim(dispatch=True)
        with self.assertRaises(JobConflict):
            self.store.mark_dispatched(job["id"], token, now_ms=22)
        self.assertEqual(3, self.store.get(job["id"])["event_count"])

    def test_fail_before_or_after_dispatch_preserves_safe_error_code(self):
        for dispatched in (False, True):
            store = JobStore(self.path.parent / ("dispatched.sqlite3" if dispatched else "not-dispatched.sqlite3"))
            job = store.submit("inspection", {}, idempotency_key="one", binding=BINDING, now_ms=10)
            token = store.claim_next("worker", now_ms=20)["lease"]["claim_token"]
            if dispatched:
                store.mark_dispatched(job["id"], token, now_ms=21)
            failed = store.fail(job["id"], token, "provider_error", now_ms=22)
            self.assertEqual("failed", failed["state"])
            self.assertEqual(dispatched, failed["dispatched"])
            self.assertEqual("provider_error", failed["error_code"])
            self.assertTrue(store.verify(job["id"])["valid"])

    def test_expired_before_dispatch_is_terminal_and_not_automatically_requeued(self):
        job, token = self.claim(lease_ms=10)
        self.assertEqual([], self.store.recover_expired(now_ms=29))
        recovered = JobStore(self.path).recover_expired(now_ms=30)
        self.assertEqual("interrupted", recovered[0]["state"])
        self.assertEqual("interrupted_no_dispatch", recovered[0]["error_code"])
        self.assertIsNone(self.store.claim_next("new-worker", now_ms=31))
        self.assertEqual([], self.store.recover_expired(now_ms=31))
        with self.assertRaises(JobLeaseError):
            self.store.mark_dispatched(job["id"], token, now_ms=31)
        self.assertEqual(job["id"], self.submit()["id"])
        self.assertEqual("interrupted", self.submit()["state"])
        self.assertTrue(self.store.verify(job["id"])["valid"])

    def test_expired_after_dispatch_has_unknown_external_outcome_and_fences_stale_completion(self):
        job, token = self.claim(dispatch=True, lease_ms=10)
        recovered = JobStore(self.path).recover_expired(now_ms=30)[0]
        self.assertEqual("outcome_unknown", recovered["error_code"])
        for operation in (
                lambda: self.store.complete(job["id"], token, {"revision": 5, "status": "pass"}, now_ms=31),
                lambda: self.store.fail(job["id"], token, "worker_error", now_ms=31),
                lambda: self.store.heartbeat(job["id"], token, now_ms=31)):
            with self.assertRaises(JobLeaseError):
                operation()
        self.assertTrue(self.store.verify(job["id"])["valid"])
        self.assertIsNone(self.store.claim_next("new-worker", now_ms=31))

    def test_lease_expiry_fences_even_before_recovery_and_heartbeat_extends_live_lease(self):
        job, token = self.claim(lease_ms=10)
        renewed = self.store.heartbeat(job["id"], token, now_ms=29, lease_ms=20)
        self.assertEqual(49, renewed["lease_expires_ms"])
        self.assertEqual([], self.store.recover_expired(now_ms=30))
        for operation in (
                lambda: self.store.mark_dispatched(job["id"], token, now_ms=49),
                lambda: self.store.heartbeat(job["id"], token, now_ms=49),
                lambda: self.store.fail(job["id"], token, "model_error", now_ms=49)):
            with self.assertRaises(JobLeaseError):
                operation()
        self.assertTrue(self.store.verify(job["id"])["valid"])

    def test_wrong_claim_token_cannot_mutate_a_running_job(self):
        job, token = self.claim()
        self.assertNotEqual("e" * 64, token)
        with self.assertRaises(JobLeaseError):
            self.store.mark_dispatched(job["id"], "e" * 64, now_ms=21)
        self.assertEqual(2, self.store.get(job["id"])["event_count"])

    def test_only_queued_jobs_cancel_and_remain_terminal(self):
        job = self.submit()
        cancelled = self.store.cancel(job["id"], now_ms=11)
        self.assertEqual("cancelled", cancelled["state"])
        self.assertIsNone(self.store.claim_next("worker", now_ms=20))
        with self.assertRaises(JobConflict):
            self.store.cancel(job["id"], now_ms=21)
        self.assertTrue(self.store.verify(job["id"])["valid"])
        other = self.submit("other")
        self.store.claim_next("worker", now_ms=20)
        with self.assertRaises(JobConflict):
            self.store.cancel(other["id"], now_ms=21)

    def test_public_dto_and_events_redact_payload_tokens_workers_and_paths(self):
        payload = {"image": "private-image-data", "token": "secret-model-key", "file": "/Users/example/private.jpg",
            "error": "private exception text"}
        job = self.submit(payload=payload)
        private = self.store.claim_next("sensitive-worker", now_ms=20)
        public = self.store.get(job["id"])
        serialized = json.dumps([public, self.store.list_jobs(), self.store.events(job["id"]), self.store.verify(job["id"])])
        for secret in list(payload.values()) + ["sensitive-worker", private["lease"]["claim_token"], "request-1"]:
            self.assertNotIn(secret, serialized)
        self.assertEqual(payload, private["payload"])
        self.assertEqual(hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest(), public["payload_sha256"])

    def test_invalid_inputs_do_not_mutate_queue(self):
        bindings = [{}, {**BINDING, "token": "private"}, {**BINDING, "revision": True},
            {**BINDING, "revision": -1}, {**BINDING, "task_hash": "bad"},
            {**BINDING, "protocol_hash": "/private/hash"}, {**BINDING, "session_id": "/Users/private"}]
        for binding in bindings:
            with self.assertRaises(JobValidationError):
                self.store.submit("inspection", {}, idempotency_key="request", binding=binding, now_ms=10)
        for payload in ([], {1: "bad-key"}, {"bad": float("nan")}, {"bad": b"bytes"}, {"bad": {1, 2}},
                {"large": "x" * MAX_PAYLOAD_BYTES}):
            with self.assertRaises(JobValidationError):
                self.store.submit("inspection", payload, idempotency_key="request", binding=BINDING, now_ms=10)
        for key in ("", "key with spaces", "x" * 257, "\nsecret"):
            with self.assertRaises(JobValidationError):
                self.store.submit("inspection", {}, idempotency_key=key, binding=BINDING, now_ms=10)
        for now in (True, -1, float("inf"), "10", 2 ** 63):
            with self.assertRaises(JobValidationError):
                self.store.submit("inspection", {}, idempotency_key="request", binding=BINDING, now_ms=now)
        self.assertEqual([], self.store.list_jobs())

    def test_invalid_completion_and_error_text_do_not_mutate_running_job(self):
        job, token = self.claim(dispatch=True)
        for result in ({}, {"revision": True, "status": "pass"}, {"revision": 5, "status": "error"},
                {"revision": 5, "status": "pass", "file": "/private/result.json"},
                {"revision": 5, "status": "pass", "receipt_id": "/private/receipt.json"},
                {"revision": 5, "status": "pass", "result_sha256": "wrong"}):
            with self.assertRaises(JobValidationError):
                self.store.complete(job["id"], token, result, now_ms=22)
        for error in ("/Users/private/key", "HTTP 500 Authorization secret", "outcome_unknown", "interrupted_no_dispatch", "unknown"):
            with self.assertRaises(JobValidationError):
                self.store.fail(job["id"], token, error, now_ms=22)
        self.assertEqual("running", self.store.get(job["id"])["state"])
        self.assertEqual(3, self.store.get(job["id"])["event_count"])

    def test_invalid_worker_or_lease_and_backwards_clock_are_rejected(self):
        job = self.submit()
        for worker in ("", "/private/worker", "worker token"):
            with self.assertRaises(JobValidationError):
                self.store.claim_next(worker, now_ms=20)
        for lease in (True, 0, -1, 300_001, "10"):
            with self.assertRaises(JobValidationError):
                self.store.claim_next("worker", now_ms=20, lease_ms=lease)
        with self.assertRaises(JobConflict):
            self.store.claim_next("worker", now_ms=9)
        claimed = self.store.claim_next("worker", now_ms=20)
        with self.assertRaises(JobConflict):
            self.store.mark_dispatched(job["id"], claimed["lease"]["claim_token"], now_ms=19)

    def test_event_cursor_is_bounded_and_hash_chain_links(self):
        job, token = self.claim(dispatch=True)
        self.store.heartbeat(job["id"], token, now_ms=22)
        events = self.store.events(job["id"])
        self.assertEqual("0" * 64, events[0]["previous_hash"])
        for left, right in zip(events, events[1:]):
            self.assertEqual(left["hash"], right["previous_hash"])
        self.assertEqual(events[2:3], self.store.events(job["id"], after_sequence=2, limit=1))
        for values in ({"limit": 201}, {"limit": True}, {"after_sequence": -1}, {"after_sequence": True}):
            with self.assertRaises(JobValidationError):
                self.store.events(job["id"], **values)
        self.assertTrue(self.store.verify(job["id"])["valid"])

    def test_sessions_match_policy_length_and_lease_overflow_is_rejected_before_claim(self):
        binding = {**BINDING, "session_id": "s" * 160}
        job = self.store.submit("inspection", {}, idempotency_key="long-session", binding=binding, now_ms=10)
        self.assertEqual(binding, job["binding"])
        with self.assertRaises(JobValidationError):
            self.store.submit("inspection", {}, idempotency_key="too-long", binding={**binding, "session_id": "s" * 161}, now_ms=10)
        with self.assertRaises(JobValidationError):
            self.store.claim_next("worker", now_ms=2 ** 63 - 1, lease_ms=1)
        with self.assertRaises(JobValidationError):
            self.store.events(job["id"], after_sequence=2 ** 63)
        self.assertEqual("queued", self.store.get(job["id"])["state"])

    def test_files_are_private_and_delete_journal_is_portable(self):
        self.assertEqual(0o700, self.path.parent.stat().st_mode & 0o777)
        self.assertEqual(0o600, self.path.stat().st_mode & 0o777)
        connection = sqlite3.connect(str(self.path))
        self.addCleanup(connection.close)
        self.assertEqual("delete", connection.execute("PRAGMA journal_mode").fetchone()[0])

    def test_sql_events_are_append_only(self):
        job = self.submit()
        connection = sqlite3.connect(str(self.path))
        self.addCleanup(connection.close)
        for sql in ("UPDATE job_events SET event_hash='bad' WHERE job_id=?", "DELETE FROM job_events WHERE job_id=?"):
            with self.assertRaises(sqlite3.IntegrityError):
                connection.execute(sql, (job["id"],))
        connection.rollback()
        self.assertTrue(self.store.verify(job["id"])["valid"])

    def test_verifier_detects_private_payload_and_public_projection_mutations(self):
        for field, value in (("payload_json", '{"image":"different"}'), ("payload_sha256", "d" * 64),
                ("state", "succeeded"), ("event_head", "e" * 64), ("event_count", 4),
                ("binding_json", json.dumps({**BINDING, "revision": 44}))):
            path = self.path.parent / (field + ".sqlite3")
            store = JobStore(path)
            job = store.submit("inspection", {}, idempotency_key="key", binding=BINDING, now_ms=10)
            with sqlite3.connect(str(path)) as connection:
                connection.execute("UPDATE jobs SET " + field + "=? WHERE id=?", (value, job["id"]))
            self.assertFalse(store.verify(job["id"])["valid"], field)

    def test_verifier_detects_event_rewrite_even_if_sql_guards_are_removed(self):
        job = self.submit()
        with sqlite3.connect(str(self.path)) as connection:
            connection.execute("DROP TRIGGER job_events_no_update")
            connection.execute("UPDATE job_events SET event_json=? WHERE job_id=?", ('{"edited":true}', job["id"]))
        self.assertFalse(self.store.verify(job["id"])["valid"])

    def test_tampered_journal_blocks_public_reads_and_mutations_without_echoing_content(self):
        job = self.submit()
        with sqlite3.connect(str(self.path)) as connection:
            connection.execute("DROP TRIGGER job_events_no_update")
            connection.execute("UPDATE job_events SET event_json=? WHERE job_id=?", ('{"private":"/Users/secret/API_KEY"}', job["id"]))
        for operation in (lambda: self.store.get(job["id"]), lambda: self.store.events(job["id"]),
                lambda: self.store.list_jobs(), lambda: self.store.cancel(job["id"], now_ms=11),
                lambda: self.store.claim_next("worker", now_ms=20)):
            with self.assertRaises(JobIntegrityError) as raised:
                operation()
            self.assertNotIn("/Users/secret", str(raised.exception))
            self.assertNotIn("API_KEY", str(raised.exception))

    def test_verifier_detects_resealed_impossible_transition(self):
        job = self.submit()
        with sqlite3.connect(str(self.path)) as connection:
            stored = connection.execute("SELECT event_json FROM job_events WHERE job_id=?", (job["id"],)).fetchone()
            event = json.loads(stored[0])
            event["job"]["state"] = "succeeded"
            encoded = json.dumps(event, sort_keys=True, separators=(",", ":"))
            digest = hashlib.sha256(encoded.encode()).hexdigest()
            connection.execute("DROP TRIGGER job_events_no_update")
            connection.execute("UPDATE job_events SET event_json=?,event_hash=? WHERE job_id=?", (encoded, digest, job["id"]))
            connection.execute("UPDATE jobs SET state='succeeded',event_head=? WHERE id=?", (digest, job["id"]))
        self.assertFalse(self.store.verify(job["id"])["valid"])

    def test_payload_corruption_blocks_claim_before_any_dispatch(self):
        job = self.submit()
        with sqlite3.connect(str(self.path)) as connection:
            connection.execute("UPDATE jobs SET payload_json='{}' WHERE id=?", (job["id"],))
        with self.assertRaises(JobIntegrityError):
            self.store.claim_next("worker", now_ms=20)
        self.assertEqual("queued", self.store.get(job["id"])["state"])
        self.assertEqual(1, self.store.get(job["id"])["event_count"])

    def test_missing_jobs_and_malformed_ids_are_distinct_safe_errors(self):
        for operation in (self.store.get, self.store.verify, self.store.events):
            with self.assertRaises(JobNotFound):
                operation("f" * 32)
            with self.assertRaises(JobValidationError):
                operation("/private/not-a-job")


if __name__ == "__main__":
    unittest.main()
