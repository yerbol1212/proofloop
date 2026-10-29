"""Authored JPEG/header and HTTP fixtures, never live model or accuracy claims."""
from copy import deepcopy
import hashlib
from io import BytesIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import HTTPError, URLError

from proofloop.core import ValidationError
from proofloop.meta import MetaAPIError, MetaClient, SPARK_MODEL
from proofloop import tracking_guard as guard
from tests.test_guided import jpeg_header


def contract():
    return {"target_id": "target", "concept": "unwrapped white oval candy",
            "attributes": {"white": "white color", "oval": "oval physical shape"},
            "exclusions": {"wrapper": "printed wrapper", "battery": "metallic cylindrical battery"}}


def supported():
    return {"candidate_identity": "target", "fully_visible": True, "identity_ambiguous": False,
            "attributes": {"white": "present", "oval": "present"},
            "exclusions": {"wrapper": "absent", "battery": "absent"}}


def provider(value=None, *, status="completed", usage=None):
    return {"model": SPARK_MODEL, "status": status,
            "output": [{"type": "message", "content": [{"type": "output_text", "text": json.dumps(value or supported())}]}],
            "usage": usage or {"input_tokens": 17, "output_tokens": 29}}


def response(value):
    return BytesIO(json.dumps(value).encode())


class TrackingGuardTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "frame.jpg"
        self.source.write_bytes(jpeg_header(width=100, height=80, suffix=b"authored-not-photographic"))
        self.box = {"x1": 20, "y1": 20, "x2": 40, "y2": 40}
        def crop(executable, source, target, box):
            width, height = box["x2"] - box["x1"] + 1, box["y2"] - box["y1"] + 1
            suffix = guard.digest({"source": hashlib.sha256(source.read_bytes()).hexdigest(), "box": box}).encode()
            target.write_bytes(jpeg_header(width=width, height=height, suffix=suffix))
            target.chmod(0o600)
        for name, value in (("_crop", crop), ("_extractor_path", lambda: "authored-ffmpeg"),
                            ("_extractor_version", lambda executable: "authored-ffmpeg-version")):
            override = patch.object(guard, name, side_effect=value)
            override.start()
            self.addCleanup(override.stop)
        self.prepared = self.prepare()

    def prepare(self, *, name="prepared", **kwargs):
        return guard.prepare(self.source, self.box, contract(), self.root / name,
                             source_sha256="a" * 64, frame_index=7, track_id=1, **kwargs)

    def client(self, value=None, *, opener=None, **kwargs):
        return MetaClient("authored-key-not-real", max_requests=1,
                          opener=opener or (lambda *args, **kwargs: response(provider(value))), **kwargs)

    def test_prepare_records_exact_clip_jpeg_candidate_crop_and_frozen_configuration(self):
        prepared = self.prepared
        self.assertEqual("a" * 64, prepared["source_sha256"])
        self.assertEqual(7, prepared["frame_index"])
        self.assertEqual(1, prepared["track_id"])
        self.assertEqual("actual_sam_video_track", prepared["anchor_origin"])
        self.assertEqual(self.box, prepared["candidate_box"])
        self.assertEqual({"x1": 12, "y1": 12, "x2": 48, "y2": 48}, prepared["crop_box"])
        self.assertEqual((37, 37), (prepared["crop_width"], prepared["crop_height"]))
        self.assertEqual({"x1": 8, "y1": 8, "x2": 28, "y2": 28}, prepared["candidate_in_crop"])
        self.assertEqual({"reasoning_effort": "low", "max_output_tokens": 4096, "max_requests": 1,
                          "retries": 0, "crop_rule": guard.CROP_RULE}, prepared["config"])
        guard.verify(prepared)

    def test_support_requires_every_attribute_every_exclusion_and_explicit_full_visibility(self):
        self.assertEqual("supported", guard.evaluate(supported(), contract())["decision"])
        variants = []
        for name in ("white", "oval"):
            for value in ("absent", "uncertain"):
                parsed = supported()
                parsed["attributes"][name] = value
                variants.append((parsed, "unknown"))
        for name in ("wrapper", "battery"):
            for value in ("present", "uncertain"):
                parsed = supported()
                parsed["exclusions"][name] = value
                variants.append((parsed, "contradicted" if value == "present" else "unknown"))
        variants.extend([({**supported(), "fully_visible": False}, "unknown"),
                         ({**supported(), "identity_ambiguous": True}, "unknown"),
                         ({**supported(), "candidate_identity": "other"}, "contradicted"),
                         ({**supported(), "candidate_identity": "uncertain"}, "unknown")])
        for parsed, expected in variants:
            with self.subTest(parsed=parsed):
                self.assertEqual(expected, guard.evaluate(parsed, contract())["decision"])

    def test_classification_schema_rejects_missing_extra_or_wrongly_typed_fields(self):
        variants = [{**supported(), "ground_truth": "target"}, {**supported(), "fully_visible": "true"},
                    {**supported(), "identity_ambiguous": None}, {**supported(), "candidate_identity": "pass"}]
        for field in supported():
            variants.append({key: value for key, value in supported().items() if key != field})
        for section in ("attributes", "exclusions"):
            parsed = supported()
            parsed[section]["invented"] = "present"
            variants.append(parsed)
        for parsed in variants:
            with self.subTest(parsed=parsed), self.assertRaises(ValueError):
                guard.evaluate(parsed, contract())

    def test_contract_requires_bounded_named_attributes_and_exclusions(self):
        variants = [{**contract(), "target_id": "../private"}, {**contract(), "concept": ""},
                    {**contract(), "attributes": {}}, {**contract(), "exclusions": {}},
                    {**contract(), "attributes": {"bad-name": "unsupported identifier"}},
                    {**contract(), "exclusions": {str(index): "item" for index in range(7)}}]
        for value in variants:
            with self.subTest(contract=value), self.assertRaises(ValidationError):
                guard.schema(value)

    def test_real_adapter_success_is_bound_to_only_the_candidate_crop_and_one_low_reasoning_request(self):
        requests = []
        def opener(request, **kwargs):
            requests.append(json.loads(request.data))
            return response(provider())
        client = self.client(opener=opener)
        result = guard.challenge(client, self.prepared)
        self.assertEqual("completed", result["status"])
        self.assertEqual("supported", result["evaluation"]["decision"])
        self.assertIsNone(result["error"])
        self.assertEqual(1, result["usage"]["request_attempts"])
        self.assertEqual(17, result["usage"]["input_tokens"])
        self.assertEqual(29, result["usage"]["output_tokens"])
        payload = requests[0]
        self.assertEqual({"effort": "low"}, payload["reasoning"])
        self.assertEqual(4096, payload["max_output_tokens"])
        self.assertFalse(payload["store"])
        contents = payload["input"][0]["content"]
        self.assertEqual(["input_text", "input_image"], [part["type"] for part in contents])
        self.assertEqual(self.prepared["crop_sha256"], result["response"]["source"]["sha256"])
        self.assertNotIn(str(self.source), contents[0]["text"])
        self.assertNotIn("task_hash", contents[0]["text"])
        self.assertNotIn("expected_region", contents[0]["text"])

    def test_completed_non_target_stays_completed_classification_with_contradicted_identity(self):
        result = guard.challenge(self.client({**supported(), "candidate_identity": "other"}), self.prepared)
        self.assertEqual("completed", result["status"])
        self.assertEqual("contradicted", result["evaluation"]["decision"])
        self.assertEqual(1, result["usage"]["request_attempts"])

    def test_incomplete_provider_response_stays_unscored_and_retains_reported_usage(self):
        client = self.client(opener=lambda *args, **kwargs: response(provider(status="incomplete")))
        result = guard.challenge(client, self.prepared)
        self.assertEqual("unscored", result["status"])
        self.assertIsNone(result["evaluation"])
        self.assertEqual("incomplete_response", result["error"]["code"])
        self.assertEqual(29, result["usage"]["output_tokens"])
        self.assertEqual(1, result["usage"]["unsuccessful_spark_requests_with_reported_token_usage"])

    def test_schema_failure_stays_unscored_instead_of_becoming_unknown_or_supported(self):
        result = guard.challenge(self.client({"missing": "fields"}), self.prepared)
        self.assertEqual("unscored", result["status"])
        self.assertIsNone(result["evaluation"])
        self.assertEqual("schema_validation_failed", result["error"]["code"])
        self.assertEqual(29, result["usage"]["output_tokens"])

    def test_transport_and_http_failures_are_unscored_without_automatic_retry(self):
        for kind in ("network", "http"):
            calls = []
            def opener(*args, **kwargs):
                calls.append(1)
                if kind == "network":
                    raise URLError("authored network uncertainty")
                raise HTTPError("https://api.meta.ai/v1/responses", 429, "authored", {},
                    response({"error": {"code": "rate_limited"}, "usage": {"input_tokens": 3, "output_tokens": 5}}))
            result = guard.challenge(self.client(opener=opener), self.prepared)
            with self.subTest(kind=kind):
                self.assertEqual("unscored", result["status"])
                self.assertIsNone(result["evaluation"])
                self.assertEqual(1, len(calls))
                self.assertEqual(1, result["usage"]["request_attempts"])
                if kind == "http":
                    self.assertEqual(5, result["usage"]["output_tokens"])

    def test_source_response_status_or_model_mismatch_stays_unscored(self):
        for change in ("source", "status", "model"):
            client = self.client()
            original = client.spark_json
            def changed(*args, **kwargs):
                value = original(*args, **kwargs)
                if change == "source":
                    value["source"]["sha256"] = "d" * 64
                elif change == "status":
                    value["response"]["status"] = "incomplete"
                else:
                    value["response"]["model"] = "another-model"
                return value
            with self.subTest(change=change), patch.object(client, "spark_json", side_effect=changed):
                result = guard.challenge(client, self.prepared)
                self.assertEqual("unscored", result["status"])
                self.assertEqual("local_validation_failed", result["error"]["code"])
                self.assertIsNone(result["evaluation"])
                self.assertEqual(1, result["usage"]["request_attempts"])

    def test_wrong_budget_retries_or_used_client_stops_before_network(self):
        for field, value in (("max_requests", 2), ("retries", 1), ("ledger", [{"used": True}])):
            calls = []
            client = self.client(opener=lambda *args, **kwargs: calls.append(1))
            setattr(client, field, value)
            with self.subTest(field=field), self.assertRaises(ValidationError):
                guard.challenge(client, self.prepared)
            self.assertEqual([], calls)

    def test_hash_or_candidate_provenance_tampering_stops_before_network(self):
        for field, value in (("source_sha256", "d" * 64), ("frame_index", 8), ("track_id", 2),
                             ("candidate_box", {"x1": 21, "y1": 20, "x2": 40, "y2": 40}),
                             ("prepared_hash", "d" * 64)):
            bad = deepcopy(self.prepared)
            bad[field] = value
            client = self.client(opener=lambda *args, **kwargs: self.fail("Changed provenance reached HTTP"))
            with self.subTest(field=field), self.assertRaises(ValidationError):
                guard.challenge(client, bad)
            self.assertEqual([], client.ledger)
        for field, value in (("anchor_origin", "ground_truth_box"), ("config", {**guard.CONFIG, "retries": 1})):
            bad = deepcopy(self.prepared)
            bad[field] = value
            bad["prepared_hash"] = guard.digest({key: item for key, item in bad.items() if key != "prepared_hash"})
            with self.subTest(field=field), self.assertRaises(ValidationError):
                guard.verify(bad)

    def test_source_or_crop_byte_changes_are_rejected_before_http(self):
        for field in ("source_file", "crop_file"):
            path = Path(self.prepared[field])
            original = path.read_bytes()
            path.write_bytes(original + b"changed")
            client = self.client(opener=lambda *args, **kwargs: self.fail("Changed bytes reached HTTP"))
            with self.subTest(field=field), self.assertRaises(ValidationError):
                guard.challenge(client, self.prepared)
            self.assertEqual([], client.ledger)
            path.write_bytes(original)

    def test_rehashing_altered_geometry_offsets_dimensions_or_extractor_cannot_rebind_existing_crop(self):
        changes = ({"candidate_box": {"x1": 21, "y1": 20, "x2": 40, "y2": 40}},
                   {"crop_box": {"x1": 11, "y1": 12, "x2": 48, "y2": 48}},
                   {"candidate_in_crop": {"x1": 9, "y1": 8, "x2": 28, "y2": 28}},
                   {"crop_width": 38}, {"crop_height": 38}, {"extractor": "different-extractor"})
        for change in changes:
            bad = deepcopy(self.prepared)
            bad.update(change)
            bad["prepared_hash"] = guard.digest({key: value for key, value in bad.items() if key != "prepared_hash"})
            with self.subTest(change=change), self.assertRaises(ValidationError):
                guard.verify(bad)

    def test_consistently_rehashed_new_geometry_still_requires_replayed_crop_bytes(self):
        bad = deepcopy(self.prepared)
        bad["candidate_box"] = {"x1": 30, "y1": 30, "x2": 50, "y2": 50}
        bad["crop_box"] = {"x1": 22, "y1": 22, "x2": 58, "y2": 58}
        # Same-size translated crop and unchanged relative rectangle would pass
        # metadata-only dimension checks; exact deterministic bytes must differ.
        bad["prepared_hash"] = guard.digest({key: value for key, value in bad.items() if key != "prepared_hash"})
        with self.assertRaises(ValidationError):
            guard.verify(bad)

    def test_source_or_crop_changes_during_provider_call_cannot_return_an_accepted_result(self):
        for field in ("source_file", "crop_file"):
            path = Path(self.prepared[field])
            original = path.read_bytes()
            def opener(*args, **kwargs):
                path.write_bytes(original + b"changed-during-request")
                return response(provider())
            client = self.client(opener=opener)
            with self.subTest(field=field), self.assertRaises(ValidationError):
                guard.challenge(client, self.prepared)
            self.assertEqual(1, len(client.ledger))
            self.assertEqual(29, client.usage_summary()["output_tokens"])
            path.write_bytes(original)

    def test_invalid_candidate_geometry_stops_before_crop_or_output_directory(self):
        for box in ({"x1": 20, "y1": 20, "x2": 20, "y2": 40},
                    {"x1": -1, "y1": 20, "x2": 40, "y2": 40},
                    {"x1": 20, "y1": 20, "x2": 100, "y2": 40},
                    {"x1": True, "y1": 20, "x2": 40, "y2": 40}):
            output = self.root / "must-not-exist"
            with self.subTest(box=box), patch.object(guard, "_crop") as crop:
                with self.assertRaises(ValidationError):
                    guard.prepare(self.source, box, contract(), output, source_sha256="a" * 64, frame_index=7, track_id=1)
                crop.assert_not_called()
                self.assertFalse(output.exists())

    def test_success_does_not_mutate_preparation_or_classification_input(self):
        before, value = deepcopy(self.prepared), supported()
        original = deepcopy(value)
        result = guard.challenge(self.client(value), self.prepared)
        self.assertEqual(before, self.prepared)
        self.assertEqual(original, value)
        self.assertEqual(result, json.loads(json.dumps(result, allow_nan=False)))


if __name__ == "__main__":
    unittest.main()
