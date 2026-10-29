"""Versioned reasoning-default transport tests; all HTTP responses are authored."""
from io import BytesIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import URLError

from proofloop.bounded_spark import BOUNDED_SPARK_VERSION, BoundedSparkClient
from proofloop.inference_journal import AttemptJournal
from proofloop.meta import MetaAPIError, SAM_MODEL, SPARK_MODEL
from proofloop.smoke import SMOKE_SCHEMA


def response(value):
    return BytesIO(json.dumps(value).encode())


def completed(*, usage=None):
    parsed = {"project": "ProofLoop", "requires_fresh_evidence": True, "result_when_occluded": "unknown"}
    return {"model": SPARK_MODEL, "status": "completed", "output": [{"type": "message", "content": [
        {"type": "output_text", "text": json.dumps(parsed)}]}],
        "usage": usage if usage is not None else {"input_tokens": 10, "output_tokens": 20, "total_tokens": 30}}


class BoundedSparkTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.journal = AttemptJournal(self.root / "private" / "attempts.sqlite3", request_limit=20)
        self.sent = []
        self.guards = []

    def client(self, *, operation_id="a" * 32, max_requests=1, value=None, error=None, guard=None, **kwargs):
        def opener(request, **options):
            self.sent.append((request, options))
            if error is not None:
                raise error
            return response(completed() if value is None else value)
        def checked_guard():
            self.guards.append(1)
            if guard is not None:
                guard()
        return BoundedSparkClient("authored-secret-not-real", journal=self.journal, operation_id=operation_id,
            guard=checked_guard, max_requests=max_requests, opener=opener, **kwargs)

    def test_omitted_reasoning_adds_low_and_preserves_explicit_4096_cap(self):
        client = self.client()
        result = client.spark_json("Authored transport request", SMOKE_SCHEMA, max_output_tokens=4096)
        wire = json.loads(self.sent[0][0].data)
        self.assertEqual("bounded_spark_low_reasoning_v1", BOUNDED_SPARK_VERSION)
        self.assertEqual({"effort": "low"}, wire["reasoning"])
        self.assertEqual(4096, wire["max_output_tokens"])
        self.assertEqual(SPARK_MODEL, wire["model"])
        self.assertEqual("https://api.meta.ai/v1/responses", self.sent[0][0].full_url)
        self.assertFalse(wire["stream"])
        self.assertFalse(wire["store"])
        self.assertEqual("ProofLoop", result["parsed"]["project"])
        self.assertEqual(1, self.journal.summary()["total_reserved_attempts"])

    def test_explicit_none_retains_the_provider_default_without_reasoning_field(self):
        self.client().spark_json("Authored transport request", SMOKE_SCHEMA, reasoning_effort=None, max_output_tokens=2048)
        wire = json.loads(self.sent[0][0].data)
        self.assertNotIn("reasoning", wire)
        self.assertEqual(2048, wire["max_output_tokens"])

    def test_explicit_low_medium_and_high_are_preserved(self):
        for number, effort in enumerate(("low", "medium", "high"), start=1):
            client = self.client(operation_id=format(number, "032x"))
            client.spark_json("Authored transport request", SMOKE_SCHEMA, reasoning_effort=effort, max_output_tokens=4096)
            self.assertEqual({"effort": effort}, json.loads(self.sent[-1][0].data)["reasoning"])
        self.assertEqual(3, self.journal.summary()["total_reserved_attempts"])

    def test_inherited_default_output_cap_is_unchanged(self):
        self.client().spark_json("Authored transport request", SMOKE_SCHEMA)
        self.assertEqual(1200, json.loads(self.sent[0][0].data)["max_output_tokens"])

    def test_invalid_explicit_reasoning_or_output_cap_fails_before_guard_or_allowance(self):
        for kwargs in ({"reasoning_effort": "unbounded"}, {"reasoning_effort": False},
                {"max_output_tokens": 4097}, {"max_output_tokens": 15}):
            with self.assertRaises(ValueError):
                self.client().spark_json("Authored transport request", SMOKE_SCHEMA, **kwargs)
        self.assertEqual([], self.guards)
        self.assertEqual([], self.sent)
        self.assertEqual(0, self.journal.summary()["total_reserved_attempts"])

    def test_sam_request_is_unchanged_and_gets_no_reasoning_or_spark_token_cap(self):
        image = self.root / "authored-image.jpg"
        image.write_bytes(b"\xff\xd8\xffauthored-signature-only-fixture")
        value = {"model": SAM_MODEL, "status": "completed", "output": [{"type": "message", "content": [
            {"type": "output_text", "text": ""}]}], "usage": {}}
        result = self.client(value=value).segment_image(image, "fork", decode_masks=False)
        wire = json.loads(self.sent[0][0].data)
        self.assertEqual(SAM_MODEL, wire["model"])
        self.assertNotIn("reasoning", wire)
        self.assertNotIn("max_output_tokens", wire)
        self.assertEqual({"include_confidence": "true", "mask_encoding": "one_bit"}, wire["metadata"])
        self.assertEqual([], result["objects"])
        self.assertEqual(1, self.journal.summary()["outcomes"]["ok"])

    def test_guard_rejection_precedes_reservation_and_http(self):
        def stale():
            raise ValueError("Task binding changed")
        with self.assertRaisesRegex(ValueError, "binding changed"):
            self.client(guard=stale).spark_json("Authored transport request", SMOKE_SCHEMA, max_output_tokens=4096)
        self.assertEqual([1], self.guards)
        self.assertEqual([], self.sent)
        self.assertEqual(0, self.journal.summary()["total_reserved_attempts"])

    def test_incomplete_response_retains_full_usage_and_never_retries(self):
        value = {"model": SPARK_MODEL, "status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"},
            "usage": {"input_tokens": 100, "output_tokens": 4096, "total_tokens": 4196,
                "output_tokens_details": {"reasoning_tokens": 4090}}}
        client = self.client(value=value)
        with self.assertRaises(MetaAPIError) as raised:
            client.spark_json("Authored transport request", SMOKE_SCHEMA, max_output_tokens=4096)
        self.assertEqual("incomplete_response", raised.exception.code)
        self.assertEqual(value, raised.exception.response)
        self.assertEqual(4096, client.usage_summary()["output_tokens"])
        summary = self.journal.summary()
        self.assertEqual(1, summary["outcomes"]["rejected"])
        self.assertEqual(4096, summary["reported_tokens"]["output_tokens"])
        with self.assertRaises(MetaAPIError):
            client.spark_json("Do not repeat", SMOKE_SCHEMA, max_output_tokens=4096)
        self.assertEqual(1, len(self.sent))
        self.assertEqual(1, self.journal.summary()["total_reserved_attempts"])

    def test_network_failure_remains_uncertain_and_does_not_repeat_http(self):
        client = self.client(error=URLError("private network detail"))
        with self.assertRaises(MetaAPIError) as raised:
            client.spark_json("Authored transport request", SMOKE_SCHEMA)
        self.assertEqual("network_error", raised.exception.code)
        self.assertNotIn("private network", str(raised.exception))
        self.assertEqual(1, len(self.sent))
        self.assertEqual(1, self.journal.summary()["outcomes"]["uncertain"])

    def test_nonzero_retry_configuration_is_still_rejected(self):
        with self.assertRaises(ValueError):
            self.client(retries=1)
        self.assertEqual([], self.sent)
        self.assertEqual(0, self.journal.summary()["total_reserved_attempts"])

    def test_inherited_from_env_constructs_the_versioned_subclass(self):
        with patch.dict("os.environ", {"META_API_KEY": "authored-env-key-not-real"}):
            client = BoundedSparkClient.from_env(journal=self.journal, operation_id="a" * 32,
                guard=lambda: None, max_requests=1, opener=lambda *args, **kwargs: response(completed()))
        self.assertIsInstance(client, BoundedSparkClient)
        self.assertEqual("ProofLoop", client.spark_json("Authored fixture", SMOKE_SCHEMA)["parsed"]["project"])


if __name__ == "__main__":
    unittest.main()
