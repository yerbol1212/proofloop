"""Contract/security checks use mocks, never paid inference or invented live results."""

from io import BytesIO
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from urllib.error import HTTPError, URLError

from proofloop.meta import (
    MetaAPIError, MetaClient, SAM_MODEL, SPARK_MODEL,
    _NoRedirect, decode_sam_masks, load_env_file, output_text, parse_sam_boxes, validate_json,
)
from proofloop.smoke import SMOKE_SCHEMA


def response(value):
    return BytesIO(json.dumps(value).encode())


def lane(text):
    return {"status": "completed", "output": [{"type": "message", "content": [
        {"type": "output_text", "text": text}
    ]}], "usage": {"input_tokens": 10, "output_tokens": 20}}


SAM_TEXT = "<0f>2<|box;x1=1;y1=2;x2=4;y2=6;w=10;h=20;c=0.9|><|mask;x=0;y=0;c=0.9;data=5,4,!abc|>"


class AdapterTests(unittest.TestCase):
    def test_exact_spark_responses_contract(self):
        captured = []
        value = {"project": "ProofLoop", "requires_fresh_evidence": True, "result_when_occluded": "unknown"}
        def opener(request, **kwargs):
            captured.append(json.loads(request.data))
            return response(lane(json.dumps(value)))
        client = MetaClient("secret-key", opener=opener)
        result = client.spark_json("test", SMOKE_SCHEMA, max_output_tokens=128)
        self.assertEqual(result["parsed"], value)
        self.assertEqual(captured[0]["model"], SPARK_MODEL)
        self.assertTrue(captured[0]["text"]["format"]["strict"])
        self.assertEqual(captured[0]["max_output_tokens"], 128)
        self.assertFalse(captured[0]["store"])

    def test_schema_invalid_output_is_never_accepted(self):
        client = MetaClient("key", opener=lambda *a, **k: response(lane('{"project":"ProofLoop"}')))
        with self.assertRaisesRegex(MetaAPIError, "schema validation"):
            client.spark_json("test", SMOKE_SCHEMA)

    def test_http_error_redacts_key_and_echoed_request(self):
        def opener(*args, **kwargs):
            raise HTTPError("https://api.meta.ai/v1/responses", 401, "secret-key", {},
                            BytesIO(b'{"error":{"code":"invalid_api_key","message":"Bearer secret-key data:image/png;base64,abc"}}'))
        client = MetaClient("secret-key", opener=opener)
        with self.assertRaises(MetaAPIError) as caught:
            client.list_models()
        serialized = str(caught.exception) + json.dumps(client.ledger)
        self.assertNotIn("secret-key", serialized)
        self.assertNotIn("data:image", serialized)
        self.assertIn("invalid_api_key", serialized)

    def test_network_error_not_retried(self):
        attempts = []
        def opener(*args, **kwargs):
            attempts.append(1)
            raise URLError("Bearer secret-key")
        client = MetaClient("secret-key", retries=2, opener=opener)
        with self.assertRaises(MetaAPIError):
            client.list_models()
        self.assertEqual(len(attempts), 1)
        self.assertNotIn("secret-key", json.dumps(client.ledger))

    def test_rate_limit_retry_and_attempt_budget(self):
        attempts = []
        def opener(*args, **kwargs):
            attempts.append(1)
            if len(attempts) == 1:
                raise HTTPError("https://api.meta.ai/v1/models", 429, "", {}, BytesIO(b"{}"))
            return response({"data": [{"id": SPARK_MODEL}]})
        client = MetaClient("key", retries=1, max_requests=2, opener=opener, sleeper=lambda n: None)
        self.assertEqual(client.list_models(), [SPARK_MODEL])
        with self.assertRaisesRegex(MetaAPIError, "budget"):
            client.list_models()

    def test_redirect_never_forwards_credentials(self):
        with self.assertRaises(MetaAPIError):
            _NoRedirect().redirect_request(None, None, 307, "", {}, "https://evil.test")

    def test_sam_image_contract(self):
        captured = []
        def opener(req, **kwargs):
            captured.append(json.loads(req.data))
            return response(lane(SAM_TEXT))
        with tempfile.TemporaryDirectory() as temp:
            fixture = Path(temp) / "own.png"
            fixture.write_bytes(b"\x89PNG\r\n\x1a\nsynthetic-transport-only")
            result = MetaClient("key", opener=opener).segment_image(fixture, "screwdriver", decode_masks=False)
        request = captured[0]
        self.assertEqual(request["model"], SAM_MODEL)
        self.assertEqual(request["metadata"]["include_confidence"], "true")
        self.assertEqual(result["objects"][0]["object_id"], 2)
        self.assertNotIn("image_url", result["source"])
        self.assertEqual(len(result["source"]["sha256"]), 64)

    def test_failed_and_incomplete_responses_rejected(self):
        for status in ["incomplete", "failed", "cancelled", "queued", "in_progress", None]:
            client = MetaClient("key", opener=lambda *a, **k: response({"status": status}))
            with self.assertRaises(MetaAPIError):
                client.spark_json("test", SMOKE_SCHEMA)

    def test_malformed_output_collection_rejected(self):
        for output in [None, {}, [None], [{"type": "message", "content": [None]}]]:
            with self.assertRaises(MetaAPIError):
                output_text({"output": output})

    def test_missing_parser_stops_before_network(self):
        def forbidden_network(*args, **kwargs):
            self.fail("Inference must not start without its required parser")
        with patch("proofloop.meta._sam_decoder", side_effect=MetaAPIError("missing parser")):
            with self.assertRaises(MetaAPIError):
                MetaClient("key", opener=forbidden_network).segment_image("not-read.png", "screwdriver")

    def test_env_file_literal_only_and_existing_wins(self):
        with tempfile.TemporaryDirectory() as temp:
            env = Path(temp) / "env"
            env.write_text('META_API_KEY="$(never-execute)"\nHOME=wrong\n')
            with patch.dict(os.environ, {"META_API_KEY": "existing"}):
                load_env_file(env)
                self.assertEqual(os.environ["META_API_KEY"], "existing")
            with patch.dict(os.environ, {}, clear=True):
                load_env_file(env)
                self.assertEqual(os.environ["META_API_KEY"], "$(never-execute)")
                self.assertNotIn("HOME", os.environ)

    def test_budget_estimate_uses_reported_tokens_and_images(self):
        client = MetaClient("key", opener=lambda *a, **k: response(lane("")))
        client.ledger = [{"status": "ok", "model": SPARK_MODEL,
                          "usage": {"input_tokens": 1000, "output_tokens": 100}},
                         {"status": "ok", "model": SAM_MODEL}]
        self.assertAlmostEqual(client.usage_summary()["estimated_usd"], .004175)

    def test_unreported_or_malformed_usage_does_not_lose_audit_report(self):
        value = {"project": "ProofLoop", "requires_fresh_evidence": True, "result_when_occluded": "unknown"}
        for usage in [None, [], "not-reported", {"input_tokens": "bad", "output_tokens": -1},
                      {"input_tokens": True, "output_tokens": 1.5}]:
            provider_response = lane(json.dumps(value))
            provider_response["usage"] = usage
            client = MetaClient("key", opener=lambda *a, **k: response(provider_response))
            client.spark_json("test", SMOKE_SCHEMA)
            report = client.usage_summary()
            self.assertEqual(report["spark_requests_with_unreported_token_usage"], 1)
            self.assertEqual(report["estimated_usd"], 0)
            json.dumps(report)  # Audit serialization remains possible.


class SAMParserTests(unittest.TestCase):
    def test_sparse_frame_and_object_ids_and_mask_preservation(self):
        text = SAM_TEXT.replace("<0f>", "<30f>") + "\n" + SAM_TEXT.replace("<0f>", "<2f>")
        records = parse_sam_boxes(text)
        self.assertEqual([r["frame"] for r in records], [30, 2])
        self.assertEqual(records[0]["mask"]["encoded_payload"], "!abc")
        self.assertFalse(records[0]["mask"]["decoded"])

    def test_zero_matches_is_not_a_physical_absence_result(self):
        self.assertEqual(parse_sam_boxes(""), [])

    def test_bad_geometry_or_grammar_rejected(self):
        bad = ["I see a screwdriver", SAM_TEXT.replace("x2=4", "x2=40"),
               SAM_TEXT.replace(";c=0.9;data", ";c=0.8;data"), SAM_TEXT + ",",
               SAM_TEXT.replace("<0f>", "<0f>garbage"), SAM_TEXT.replace("!abc", "!"), "<0f>"]
        for text in bad:
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_sam_boxes(text)

    def test_refusal_not_treated_as_empty_text(self):
        with self.assertRaises(MetaAPIError):
            output_text({"output": [{"type": "message", "content": [{"type": "refusal"}]}]})

    def test_bool_is_not_an_integer_and_extra_fields_fail(self):
        with self.assertRaises(ValueError):
            validate_json(True, {"type": "integer"})
        with self.assertRaises(ValueError):
            validate_json({"surprise": 1}, {"type": "object", "properties": {}, "additionalProperties": False})

    def test_official_decoder_wrapper_checks_raster(self):
        objects = parse_sam_boxes(SAM_TEXT)
        raster = bytes([0, 1] * 10)
        with patch("proofloop.meta._sam_decoder", return_value=(SimpleNamespace, lambda mask: raster, ValueError)):
            decoded = decode_sam_masks(objects)
        self.assertTrue(decoded[0]["mask"]["decoded"])
        self.assertEqual(decoded[0]["mask"]["foreground_pixels"], 10)
        self.assertFalse(objects[0]["mask"]["decoded"])
        with patch("proofloop.meta._sam_decoder", return_value=(SimpleNamespace, lambda mask: b"bad", ValueError)):
            with self.assertRaises(MetaAPIError):
                decode_sam_masks(objects)

    def test_official_decoder_wrapper_rejects_invalid_mask(self):
        def invalid(mask):
            raise ValueError("private payload")
        with patch("proofloop.meta._sam_decoder", return_value=(SimpleNamespace, invalid, ValueError)):
            with self.assertRaisesRegex(MetaAPIError, "rejected"):
                decode_sam_masks(parse_sam_boxes(SAM_TEXT))

    def test_official_parser_public_example_if_installed(self):
        try:
            import meta_sam_parser
        except (ImportError, TypeError, SyntaxError):
            self.skipTest("Optional official parser requires Python >=3.10 and meta-sam-parser==0.1.0")
        # This encoded 5x5 example comes from Meta's public package README.
        text = ("<0f>2<|box;x1=1;y1=2;x2=5;y2=6;w=10;h=20|>"
                "<|mask;x=0;y=0;data=5,5,!!!!!(QO(0lu8?|>")
        decoded = decode_sam_masks(parse_sam_boxes(text))
        self.assertTrue(decoded[0]["mask"]["decoded"])
        self.assertEqual(decoded[0]["mask"]["foreground_pixels"], 10)
        # Regression: the wire marker is part of the official payload contract.
        self.assertEqual(decoded[0]["mask"]["encoded_payload"], "!!!!!(QO(0lu8?")


if __name__ == "__main__":
    unittest.main()
