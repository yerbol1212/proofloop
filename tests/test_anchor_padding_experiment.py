"""Offline authored adapter/transport tests, not model-quality evidence."""
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

from proofloop import anchor_padding_experiment as padding, benchmark, prospective
from proofloop.core import ValidationError
from proofloop.meta import MetaAPIError, MetaClient, SAM_MODEL
from tests.test_benchmark import contract
from tests.test_guided import jpeg_header
from tests.test_prospective import Client


def original(root, name="first", transform=None):
    value = contract()
    third = deepcopy(value["cases"][0])
    third.update(id="extra", evaluation_timestamp_ms=5000)
    third["frames"] = [{"file": "extra-4000.jpg", "sha256": "0" * 64, "source_timestamp_ms": 4000},
                       {"file": "extra-5000.jpg", "sha256": "1" * 64, "source_timestamp_ms": 5000}]
    third["annotation"].update(status="unknown", checks={"glass-dispense": "unknown"})
    value["cases"].append(third)
    value["dataset_id"] = name
    for case in value["cases"]:
        case["id"] = name + "-" + case["id"]
    for index, record in enumerate(value["sources"] + [frame for case in value["cases"] for frame in case["frames"]]):
        record["file"] = "inputs/local/" + name + "/" + str(index) + (".mp4" if index == 0 else ".jpg")
        data = b"authored-video" if index == 0 else jpeg_header(suffix=(name + str(index)).encode())
        path = root / record["file"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        record["sha256"] = hashlib.sha256(data).hexdigest()
    envelope = benchmark.approve_manifest(value, benchmark.manifest_hash(value), confirm_unique_targets=True,
        review_origin="assistant_under_user_authorization", authorization_reference="PRIVATE_AUTHORIZATION")
    location = root / "artifacts/local" / (name + "-original")
    prospective.freeze(envelope, location, root=root, max_requests=12)
    def factory(budget):
        client = Client(budget)
        client.transform = transform
        return client
    prospective.run(factory, location, root=root)
    return location


class PaddingExperimentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.pilot = original(self.root)
        self.output = self.root / "artifacts/local/padding"

    def freeze(self, **kwargs):
        return padding.freeze([self.pilot], self.output, root=self.root, **kwargs)

    def run_pilot(self, factory=Client):
        return padding.run(factory, self.output, root=self.root)

    def test_fixed_padding_uses_source_image_size_and_clamps_inclusive_edges(self):
        self.assertEqual({"x1": 196, "y1": 157, "x2": 309, "y2": 296},
            padding.padded_box({"x1": 228, "y1": 189, "x2": 277, "y2": 264}, 640, 480))
        self.assertEqual({"x1": 0, "y1": 0, "x2": 99, "y2": 79},
            padding.padded_box({"x1": 0, "y1": 0, "x2": 99, "y2": 79}, 100, 80))
        for box in ({"x1": -1, "y1": 0, "x2": 2, "y2": 2},
                    {"x1": 0, "y1": 0, "x2": 100, "y2": 2}):
            with self.assertRaises(ValidationError): padding.padded_box(box, 100, 80)

    def test_freeze_keeps_all_nine_cases_and_exact_eighteen_cap_without_calls(self):
        second, third = original(self.root, "second"), original(self.root, "third")
        record = padding.freeze([self.pilot, second, third], self.output, root=self.root)
        self.assertEqual(9, len(record["cases"]))
        self.assertEqual(18, record["planned_requests"])
        self.assertIn("anchor_padding_experiment.py", record["implementation_sha256"])
        result = padding.summarize(self.output, root=self.root)
        self.assertEqual(9, sum(row["metrics"]["missing_predictions"] for row in result["metrics"]))
        self.assertEqual(0, result["usage"]["request_attempts"])
        self.assertFalse((self.output / "run-started.json").exists())

    def test_original_visibility_identity_and_single_candidate_gates_remain_withheld(self):
        def transform(value, paths):
            for frame in value["parsed"]["frames"]:
                frame["objects"][0]["fully_visible"] = False
        self.pilot = original(self.root, "withheld", transform)
        record = self.freeze()
        self.assertEqual(0, record["planned_requests"])
        result = self.run_pilot(lambda _: self.fail("Withheld frames cannot spend"))
        self.assertEqual(0, result["usage"]["request_attempts"])
        self.assertEqual(3, result["metrics"][0]["metrics"]["unknown_results"])
        self.assertTrue(all(frame["status"] == "withheld" for row in result["results"] for frame in row["frames"]))

    def test_new_sam_only_calls_use_exact_source_and_padded_token_without_text_or_labels(self):
        self.freeze()
        clients = []
        def factory(budget):
            self.assertEqual(1, budget)
            client = Client(budget)
            clients.append(client)
            return client
        result = self.run_pilot(factory)
        self.assertEqual("completed", result["status"])
        self.assertEqual(6, result["usage"]["request_attempts"])
        for client in clients:
            self.assertEqual(1, len(client.calls))
            kind, path, prompt, kwargs = client.calls[0]
            self.assertEqual("sam", kind)
            self.assertTrue(prompt.startswith("<0f>1<|box;"))
            self.assertNotIn("glass", prompt)
            self.assertNotIn("dispense", prompt)
            self.assertTrue(kwargs["decode_masks"])
        self.assertEqual(result, padding.summarize(self.output, root=self.root))
        with self.assertRaises(ValidationError): self.run_pilot()

    def test_unchanged_threshold_rejects_low_score_without_promoting_acceptance(self):
        self.freeze()
        def factory(budget):
            client = Client(budget)
            client.sam_transform = lambda result, path: result["objects"][0].update(confidence=.79)
            return client
        result = self.run_pilot(factory)
        self.assertEqual(3, result["metrics"][0]["metrics"]["unknown_results"])
        self.assertEqual(0, result["metrics"][0]["metrics"]["correct_acceptances"])
        self.assertTrue(all(row["checks"][0]["gap_reason"] in ("low_confidence", "unavailable_or_ambiguous") for row in result["results"]))

    def test_failed_request_is_missing_case_not_fabricated_unknown_and_next_cases_continue(self):
        self.freeze()
        count = []
        def factory(budget):
            count.append(budget)
            client = Client(budget)
            if len(count) == 1:
                def fail(*args, **kwargs):
                    client.ledger.append({"model": SAM_MODEL, "status": "network_error"})
                    raise MetaAPIError("PRIVATE_PROVIDER_ERROR", code="network_error")
                client.segment_image = fail
            return client
        result = self.run_pilot(factory)
        self.assertEqual(6, len(count))
        self.assertEqual("incomplete", result["status"])
        self.assertEqual(1, result["metrics"][0]["metrics"]["missing_predictions"])
        self.assertTrue(result["usage"]["unreported_charge_possible"])
        self.assertEqual("unscored", result["results"][0]["status"])
        self.assertNotIn("PRIVATE_PROVIDER_ERROR", json.dumps(result))

    def test_real_http_401_string_error_code_stops_remaining_admissions(self):
        self.freeze()
        clients = []
        def factory(budget):
            def opener(*args, **kwargs):
                raise HTTPError("https://api.meta.ai/v1/responses", 401, "authored", {},
                    io.BytesIO(b'{"error":{"code":"invalid_api_key","message":"PRIVATE_PROVIDER_TEXT"}}'))
            client = MetaClient("authored-key", max_requests=budget, retries=0, opener=opener)
            # Keep this transport fixture portable on Python3.9 without optional official decoder.
            client.segment_image = lambda path, prompt, **kw: client._request("POST", "responses", {"model": SAM_MODEL})
            clients.append(client)
            return client
        result = self.run_pilot(factory)
        self.assertEqual(1, len(clients))
        self.assertEqual(1, result["usage"]["request_attempts"])
        self.assertEqual(401, result["admission_stop_code"])
        self.assertEqual(3, result["metrics"][0]["metrics"]["missing_predictions"])
        self.assertNotIn("PRIVATE_PROVIDER_TEXT", json.dumps(result))

    def test_plan_media_locator_and_implementation_tamper_prevent_any_client(self):
        self.freeze()
        plan_path = self.output / "plan.json"
        original_bytes = plan_path.read_bytes()
        value = json.loads(original_bytes)
        value["cases"][0]["frames"][0]["padded_box"]["x1"] += 1
        benchmark._save(plan_path, value)
        with self.assertRaises(ValidationError): self.run_pilot(lambda _: self.fail("No admission"))
        plan_path.write_bytes(original_bytes)
        with patch.object(padding, "_implementation", return_value={}):
            with self.assertRaises(ValidationError): self.run_pilot(lambda _: self.fail("No admission"))
        original_case = value["cases"][0]
        path = Path(original_case["original_pair_file"])
        path.write_bytes(path.read_bytes() + b"changed")
        with self.assertRaises(ValidationError): self.run_pilot(lambda _: self.fail("No admission"))

    def test_source_frame_tamper_prevents_any_client_and_marker(self):
        plan = self.freeze()
        frame = plan["cases"][0]["envelope"]["contract"]["cases"][0]["frames"][0]
        path = self.root / frame["file"]
        path.write_bytes(path.read_bytes() + b"corrupted source")
        with self.assertRaises(ValidationError): self.run_pilot(lambda _: self.fail("No source-tampered admission"))
        self.assertFalse((self.output / "run-started.json").exists())

    def test_all_real_http_admission_stop_statuses_preserve_missing_denominators(self):
        for status in (401, 402, 403, 429):
            with self.subTest(status=status):
                self.output = self.root / "artifacts/local" / ("stop-" + str(status))
                self.freeze()
                count = []
                def factory(budget):
                    client = Client(budget)
                    def fail(*args, **kwargs):
                        count.append(status)
                        client.ledger.append({"model": SAM_MODEL, "status": "http_error", "http_status": status,
                                              "error_code": "authored_string_code"})
                        raise MetaAPIError("authored", status=status, code="authored_string_code")
                    client.segment_image = fail
                    return client
                result = self.run_pilot(factory)
                self.assertEqual([status], count)
                self.assertEqual(status, result["admission_stop_code"])
                self.assertEqual(3, result["metrics"][0]["metrics"]["missing_predictions"])
                self.assertEqual(0, result["metrics"][0]["metrics"]["unknown_results"])

    def test_budget_retries_reused_client_and_nonempty_ledgers_do_not_spend(self):
        self.freeze()
        client = Client(1)
        client.retries = 1
        result = self.run_pilot(lambda _: client)
        self.assertFalse(client.calls)
        self.assertEqual(0, result["usage"]["request_attempts"])
        self.assertEqual(3, result["metrics"][0]["metrics"]["missing_predictions"])

    def test_private_provenance_masks_and_provider_fields_never_exported(self):
        self.freeze()
        result = self.run_pilot()
        text = json.dumps(result)
        for private in ("inputs/local", "artifacts/local", "PRIVATE_AUTHORIZATION", "PRIVATE_PROVIDER_ID",
                        "raster_base64", "encoded_payload", "PRIVATE_PROVIDER_TEXT", "/Users/private"):
            self.assertNotIn(private, text)
        self.assertTrue(all(frame.get("inference_origin") == "injected_client_not_verified_live"
                            for case in result["results"] for frame in case["frames"]))

    def test_saved_response_summary_or_usage_tamper_invalidates_replay(self):
        self.freeze()
        self.run_pilot()
        summary = self.output / "summary.json"
        value = json.loads(summary.read_text())
        value["usage"]["estimated_usd"] = 999
        benchmark._save(summary, value)
        with self.assertRaises(ValidationError): padding.summarize(self.output, root=self.root)
        summary.unlink()
        state_path = self.output / "run-state.json"
        state = json.loads(state_path.read_text())
        first = next(iter(state["attempts"].values()))
        first["usage"]["estimated_usd"] = 999
        benchmark._save(state_path, state)
        with self.assertRaises(ValidationError): padding.summarize(self.output, root=self.root)

    def test_saved_response_change_or_deletion_invalidates_replay(self):
        self.freeze()
        self.run_pilot()
        response = next((self.output / "responses").rglob("*.json"))
        response.write_bytes(response.read_bytes() + b" ")
        with self.assertRaises(ValidationError): padding.summarize(self.output, root=self.root)
        response.unlink()
        with self.assertRaises((ValidationError, OSError)): padding.summarize(self.output, root=self.root)

    def test_ambiguous_identity_or_multiple_candidates_never_receive_padded_requests(self):
        for mode in ("identity", "multiple"):
            def transform(value, paths):
                for frame in value["parsed"]["frames"]:
                    target = frame["objects"][0]
                    if mode == "identity": target["identity_ambiguous"] = True
                    else: target["candidates"].append(deepcopy(target["candidates"][0]))
            location = original(self.root, mode, transform)
            output = self.root / "artifacts/local" / (mode + "-padding")
            plan = padding.freeze([location], output, root=self.root)
            self.assertEqual(0, plan["planned_requests"])
            result = padding.run(lambda _: self.fail("No admission for ambiguous locator"), output, root=self.root)
            self.assertEqual(0, result["usage"]["request_attempts"])

    def test_unofficial_or_wrong_source_result_is_charged_but_never_scored(self):
        self.freeze()
        count = []
        def factory(budget):
            client = Client(budget)
            def transform(result, path):
                count.append(path)
                if len(count) == 1:
                    result["mask_decoding"] = "skipped_explicitly"
                elif len(count) == 2:
                    result["source"]["sha256"] = "0" * 64
            client.sam_transform = transform
            return client
        result = self.run_pilot(factory)
        self.assertEqual(6, result["usage"]["request_attempts"])
        self.assertEqual("unscored", result["results"][0]["status"])
        self.assertEqual(1, result["metrics"][0]["metrics"]["missing_predictions"])
        self.assertTrue(all(row["status"] == "error" for row in result["results"][0]["frames"]))

    def test_known_uncertain_charge_flag_cannot_be_cleared_in_saved_state(self):
        self.freeze()
        def factory(budget):
            client = Client(budget)
            def fail(*args, **kwargs):
                client.ledger.append({"model": SAM_MODEL, "status": "network_error"})
                raise MetaAPIError("authored", code="network_error")
            client.segment_image = fail
            return client
        self.run_pilot(factory)
        (self.output / "summary.json").unlink()
        path = self.output / "run-state.json"
        state = json.loads(path.read_text())
        next(iter(state["attempts"].values()))["usage"]["unreported_charge_possible"] = False
        benchmark._save(path, state)
        with self.assertRaises(ValidationError): padding.summarize(self.output, root=self.root)

    def test_missing_token_field_cannot_bypass_known_failed_ledger_counts_or_uncertainty(self):
        self.freeze()
        def factory(budget):
            client = Client(budget)
            def fail(*args, **kwargs):
                client.ledger.append({"model": SAM_MODEL, "status": "network_error"})
                raise MetaAPIError("authored", code="network_error")
            client.segment_image = fail
            return client
        self.run_pilot(factory)
        (self.output / "summary.json").unlink()
        path = self.output / "run-state.json"
        state = json.loads(path.read_text())
        for attempt in state["attempts"].values():
            usage = attempt["usage"]
            usage.pop("input_tokens")
            usage.update(request_attempts=0, estimated_usd=0.0, unreported_charge_possible=False)
            self.assertEqual("network_error", usage["ledger"][0]["status"])
        benchmark._save(path, state)
        with self.assertRaises(ValidationError): padding.summarize(self.output, root=self.root)

    def test_missing_ledger_is_rejected_even_if_reported_fields_look_valid(self):
        self.freeze()
        self.run_pilot()
        (self.output / "summary.json").unlink()
        path = self.output / "run-state.json"
        state = json.loads(path.read_text())
        next(iter(state["attempts"].values()))["usage"].pop("ledger")
        benchmark._save(path, state)
        with self.assertRaises(ValidationError): padding.summarize(self.output, root=self.root)

    def test_explicit_incomplete_accounting_keeps_known_ledger_attempts_cost_and_uncertainty(self):
        self.freeze()
        def factory(budget):
            client = Client(budget)
            client.usage_summary = lambda: {"request_attempts": 0}  # Malformed injected reporter, not provider evidence.
            return client
        result = self.run_pilot(factory)
        self.assertEqual(6, result["usage"]["request_attempts"])
        self.assertAlmostEqual(.015, result["usage"]["estimated_usd"])
        self.assertTrue(result["usage"]["accounting_incomplete"])
        self.assertTrue(result["usage"]["unreported_charge_possible"])
        for case in result["results"]:
            for frame in case["frames"]:
                self.assertIsNone(frame["usage"]["estimated_usd"])
                self.assertEqual(.0025, frame["usage"]["known_estimated_usd"])
                self.assertTrue(frame["usage"]["accounting_incomplete"])
                self.assertTrue(frame["usage"]["unreported_charge_possible"])

    def test_interrupted_running_without_ledger_snapshot_is_unknown_accounting_not_zero(self):
        self.freeze()
        self.run_pilot()
        (self.output / "summary.json").unlink()
        path = self.output / "run-state.json"
        state = json.loads(path.read_text())
        state["status"] = "running"
        first = next(iter(state["attempts"].values()))
        first.update(status="running", usage=None)
        benchmark._save(path, state)
        result = padding.summarize(self.output, root=self.root)
        self.assertEqual(5, result["usage"]["request_attempts"])
        self.assertTrue(result["usage"]["request_attempts_incomplete"])
        self.assertTrue(result["usage"]["accounting_incomplete"])
        self.assertTrue(result["usage"]["unreported_charge_possible"])
        unknown = [frame for case in result["results"] for frame in case["frames"]
                   if frame["usage"]["request_attempts"] is None]
        self.assertEqual(1, len(unknown))
        self.assertIsNone(unknown[0]["usage"]["estimated_usd"])
        self.assertEqual(1, result["metrics"][0]["metrics"]["missing_predictions"])

    def test_paid_parse_failure_preserves_private_raw_response_with_source_and_hash_only_publicly(self):
        self.freeze()
        provider = {"model": SAM_MODEL, "status": "completed", "id": "PRIVATE_PROVIDER_ID",
                    "output": [{"type": "message", "content": [{"type": "output_text", "text": "PRIVATE_PROVIDER_BODY"}]}]}
        def factory(budget):
            client = Client(budget)
            def fail(*args, **kwargs):
                client.ledger.append({"model": SAM_MODEL, "status": "ok"})
                raise MetaAPIError("authored parse failure", code="sam_parse_failed", response=provider)
            client.segment_image = fail
            return client
        result = self.run_pilot(factory)
        receipts = list((self.output / "error-responses").rglob("*.json"))
        self.assertEqual(6, len(receipts))
        self.assertEqual(6, result["usage"]["request_attempts"])
        self.assertAlmostEqual(.015, result["usage"]["estimated_usd"])
        self.assertEqual(3, result["metrics"][0]["metrics"]["missing_predictions"])
        self.assertEqual(0, result["metrics"][0]["metrics"]["unknown_results"])
        self.assertTrue(all(case["status"] == "unscored" for case in result["results"]))
        hashes = {frame["source_sha256"] for case in result["results"] for frame in case["frames"]}
        for path in receipts:
            receipt = json.loads(path.read_text())
            self.assertEqual(provider, receipt["response"])
            self.assertIn(receipt["source_sha256"], hashes)
            self.assertEqual(benchmark.manifest_hash(provider), receipt["provider_response_sha256"])
            self.assertTrue(receipt["provider_response_preserved"])
        for secret in ("PRIVATE_PROVIDER_ID", "PRIVATE_PROVIDER_BODY", "inputs/local", "artifacts/local"):
            self.assertNotIn(secret, json.dumps(result))
        self.assertTrue(all(frame["provider_response_preserved"] for case in result["results"] for frame in case["frames"]))
        receipts[0].write_bytes(receipts[0].read_bytes() + b"changed")
        with self.assertRaises(ValidationError): padding.summarize(self.output, root=self.root)

    def test_excessive_failed_provider_body_keeps_bounded_digest_receipt(self):
        plan = self.freeze()
        case = plan["cases"][0]
        row = case["frames"][0]
        provider = {"status": "completed", "output_text": "x" * benchmark.MAX_JSON_BYTES}
        record = padding._retain_error_response(self.output, case, row, provider)
        self.assertFalse(record["provider_response_preserved"])
        receipt_path = padding._error_response_path(self.output, case, row)
        self.assertLess(receipt_path.stat().st_size, 1024)
        receipt = json.loads(receipt_path.read_text())
        self.assertEqual(benchmark.manifest_hash(provider), receipt["provider_response_sha256"])
        self.assertNotIn("response", receipt)
        self.assertEqual("local_size_limit", receipt["omission_reason"])


if __name__ == "__main__":
    unittest.main()
