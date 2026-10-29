"""Authored source/transport/policy controls; no paid model-quality tests."""
from copy import deepcopy
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

from proofloop import benchmark, concept_verified_experiment as concept, image_gripper_experiment as experiment
from proofloop.core import ValidationError
from proofloop.meta import MetaAPIError, MetaClient, SAM_MODEL, parse_sam_boxes
from tests import test_concept_verified_experiment as fixtures
from tests.test_anchor_padding_experiment import original


class GripperClient(fixtures.Client):
    def __init__(self, budget=1, *, near=False, empty=False, sam_error=None, invalid=False):
        super().__init__(budget, sam_error=sam_error)
        self.near, self.empty, self.invalid = near, empty, invalid

    def segment_image(self, *args, **kwargs):
        value = super().segment_image(*args, **kwargs)
        text = value["text"]
        if not self.near:
            text = text.replace("x1=10;y1=10;x2=14;y2=14", "x1=60;y1=10;x2=64;y2=14")
        if self.empty:
            text = ""
        value["text"] = value["response"]["output_text"] = text
        value["objects"] = fixtures.decoded(parse_sam_boxes(text))
        if self.invalid:
            value["objects"][0]["confidence"] = .99
        return value


class ImageGripperExperimentTests(unittest.TestCase):
    def setUp(self):
        fixtures.ConceptVerifiedExperimentTests.setUp(self)
        concept.freeze([self.pilot], self.output, root=self.root)
        concept.run(fixtures.Client, self.output, root=self.root)
        self.parent = self.output
        self.output = self.root / "artifacts/local/gripper-experiment"

    def freeze(self, **kwargs):
        return experiment.freeze(self.parent, self.output, root=self.root, **kwargs)

    def run_trial(self, factory=GripperClient):
        return experiment.run(factory, self.output, root=self.root)

    def test_plan_keeps_every_parent_case_view_and_labels_before_inference(self):
        plan = self.freeze()
        public = experiment.public_plan(self.output, root=self.root)
        self.assertEqual(3, len(plan["cases"]))
        self.assertEqual(6, plan["planned_requests"])
        self.assertEqual(18, plan["max_requests"])
        self.assertEqual("robot gripper", public["concept"])
        self.assertIn("image_gripper_experiment.py", plan["implementation_sha256"])
        self.assertIn("image_gripper_veto.py", plan["implementation_sha256"])
        before = experiment.summarize(self.output, root=self.root)
        self.assertEqual(6, before["geometry_diagnostics"]["missing_or_failed_gripper_diagnostics"])
        self.assertEqual(3, before["metrics"][0]["metrics"]["missing_predictions"])
        self.assertFalse((self.output / "run-started.json").exists())

    def test_all_nine_cases_eighteen_original_views_are_attempted_once_without_spark(self):
        second = original(self.root, "second", transform=fixtures.locator_text)
        third = original(self.root, "third", transform=fixtures.locator_text)
        self.parent = self.root / "artifacts/local/nine-parent"
        concept.freeze([self.pilot, second, third], self.parent, root=self.root)
        concept.run(fixtures.Client, self.parent, root=self.root)
        plan = self.freeze()
        clients = []
        def factory(budget):
            self.assertEqual(1, budget)
            clients.append(GripperClient(budget))
            return clients[-1]
        result = self.run_trial(factory)
        self.assertEqual("completed", result["status"])
        self.assertEqual(18, len(clients))
        self.assertEqual(18, result["usage"]["sam_requests"])
        self.assertEqual(0, result["usage"]["spark_requests"])
        self.assertEqual(9, len(result["results"]))
        self.assertEqual(18, result["geometry_diagnostics"]["total_original_views"])
        for client in clients:
            kind, path, prompt, options = client.calls[0]
            self.assertEqual("sam", kind)
            self.assertEqual("robot gripper", prompt)
            self.assertTrue(options["decode_masks"])
            self.assertIn("inputs/local", str(path))
        self.assertEqual([case["parent_prediction"] for case in plan["cases"]], [case["prediction"] for case in result["results"]])
        self.assertEqual(result, experiment.summarize(self.output, root=self.root))
        with self.assertRaises(ValidationError): self.run_trial()

    def test_near_mask_only_removes_parent_support_and_far_preserves_existing_decision(self):
        self.freeze()
        count = []
        def factory(budget):
            client = GripperClient(budget, near=not count)
            count.append(client)
            return client
        result = self.run_trial(factory)
        self.assertEqual("unknown", result["results"][0]["prediction"])
        self.assertEqual("near_gripper_proposal_withholds_parent", result["results"][0]["reason"])
        self.assertTrue(all(row["prediction"] == "pass" for row in result["results"][1:]))
        self.assertEqual(1, result["geometry_diagnostics"]["near_warning_views"])

    def test_parent_unknown_stays_unknown_while_failed_added_diagnostic_is_separate(self):
        self.parent = self.root / "artifacts/local/unknown-parent"
        concept.freeze([self.pilot], self.parent, root=self.root)
        concept.run(lambda budget: fixtures.Client(budget, score=.79), self.parent, root=self.root)
        self.freeze()
        clients = []
        def factory(budget):
            client = GripperClient(budget, sam_error=MetaAPIError("Authored lost response", code="network_error") if not clients else None)
            clients.append(client)
            return client
        result = self.run_trial(factory)
        self.assertEqual("incomplete", result["status"])
        self.assertEqual(1, result["geometry_diagnostics"]["missing_or_failed_gripper_diagnostics"])
        self.assertEqual(["unknown"] * 3, [row["prediction"] for row in result["results"]])
        self.assertTrue(result["usage"]["unreported_charge_possible"])
        self.assertEqual(0, result["metrics"][0]["metrics"]["missing_predictions"])

    def test_failed_added_diagnostic_cannot_preserve_parent_positive(self):
        self.freeze()
        clients = []
        def factory(budget):
            client = GripperClient(budget, sam_error=MetaAPIError("Authored lost response", code="network_error") if not clients else None)
            clients.append(client)
            return client
        result = self.run_trial(factory)
        self.assertEqual("unscored", result["results"][0]["status"])
        self.assertIsNone(result["results"][0]["prediction"])
        self.assertEqual(1, result["metrics"][0]["metrics"]["missing_predictions"])
        self.assertEqual(["pass", "pass"], [row["prediction"] for row in result["results"][1:]])

    def test_zero_gripper_detections_are_unavailable_not_clear_or_verified_absence(self):
        self.freeze()
        result = self.run_trial(lambda budget: GripperClient(budget, empty=True))
        self.assertEqual("completed", result["status"])
        self.assertEqual(6, result["geometry_diagnostics"]["geometry_unavailable_views"])
        self.assertEqual(0, result["geometry_diagnostics"]["available_no_near_warning_views"])
        self.assertTrue(all(row["prediction"] is None for row in result["results"]))

    def test_diagnostic_view_labels_are_separate_from_task_positive_retention(self):
        _, parent_plan = concept._load(self.parent, self.root)
        first = parent_plan["cases"][0]["frames"][0]["source_sha256"]
        second = parent_plan["cases"][0]["frames"][1]["source_sha256"]
        self.freeze(frame_annotations={first: "near_gripper_proposal", second: "no_near_proposal_observed"})
        clients = []
        def factory(budget):
            client = GripperClient(budget, near=not clients)
            clients.append(client)
            return client
        result = self.run_trial(factory)
        references = {row["label"]: row for row in result["geometry_diagnostics"]["reference_categories"]}
        self.assertEqual((1, 1), (references["near_gripper_proposal"]["matched"], references["no_near_proposal_observed"]["matched"]))
        self.assertEqual(6, result["geometry_diagnostics"]["total_original_views"])
        self.assertEqual(3, result["metrics"][0]["metrics"]["eligible_cases"])

    def test_every_auth_billing_and_rate_status_stops_global_admissions(self):
        for code in (401, 402, 403, 429):
            with self.subTest(code=code):
                self.output = self.root / "artifacts/local" / ("stop-" + str(code))
                self.freeze()
                clients = []
                def factory(budget):
                    clients.append(GripperClient(budget, sam_error=MetaAPIError("private", code="authored_http", status=code)))
                    return clients[-1]
                result = self.run_trial(factory)
                self.assertEqual(1, len(clients))
                self.assertEqual(code, result["admission_stop_code"])
                self.assertEqual(6, result["geometry_diagnostics"]["missing_or_failed_gripper_diagnostics"])
                self.assertTrue(all(row["prediction"] is None for row in result["results"]))

    def test_actual_http_string_error_code_preserves_bound_failed_response(self):
        self.freeze()
        def factory(budget):
            def opener(*args, **kwargs):
                raise HTTPError("https://api.meta.ai/v1/responses", 402, "authored", {},
                    io.BytesIO(b'{"error":{"code":"billing_issue","message":"PRIVATE_PROVIDER_TEXT"}}'))
            client = MetaClient("authored-not-a-key", max_requests=budget, retries=0, opener=opener)
            client.segment_image = lambda *args, **kwargs: client._request("POST", "responses", {"model": SAM_MODEL})
            return client
        result = self.run_trial(factory)
        self.assertEqual(402, result["admission_stop_code"])
        self.assertTrue(result["results"][0]["frames"][0]["provider_response_preserved"])
        self.assertNotIn("PRIVATE_PROVIDER_TEXT", json.dumps(result))

    def test_parent_source_summary_raw_result_or_code_tamper_prevents_dispatch(self):
        plan = self.freeze()
        path = self.parent / "summary.json"
        original_bytes = path.read_bytes()
        path.write_bytes(original_bytes + b" ")
        with self.assertRaises(ValidationError): self.run_trial(lambda _: self.fail("No dispatch"))
        path.write_bytes(original_bytes)
        with patch.object(experiment, "_implementation", return_value={}):
            with self.assertRaises(ValidationError): self.run_trial(lambda _: self.fail("No dispatch"))
        target = Path(plan["cases"][0]["frames"][0]["target_file"])
        target.write_bytes(target.read_bytes() + b"altered")
        with self.assertRaises(ValidationError): self.run_trial(lambda _: self.fail("No dispatch"))
        self.assertFalse((self.output / "run-started.json").exists())

    def test_wrong_budget_retries_or_dirty_clients_are_not_dispatched(self):
        self.freeze()
        clients = []
        def factory(budget):
            client = GripperClient(budget)
            client.retries = 1
            clients.append(client)
            return client
        result = self.run_trial(factory)
        self.assertTrue(all(not client.calls for client in clients))
        self.assertEqual(0, result["usage"]["request_attempts"])
        self.assertTrue(all(row["prediction"] is None for row in result["results"]))

    def test_failed_decoding_completed_response_cannot_become_network_ledger(self):
        self.freeze()
        clients = []
        def factory(budget):
            client = GripperClient(budget, invalid=not clients)
            clients.append(client)
            return client
        result = self.run_trial(factory)
        self.assertEqual(6, result["usage"]["request_attempts"])
        self.assertIsNone(result["results"][0]["prediction"])
        (self.output / "summary.json").unlink()
        path = self.output / "run-state.json"
        state = json.loads(path.read_text())
        first = next(iter(state["attempts"].values()))
        first["usage"] = parent_accounting([{"model": SAM_MODEL, "endpoint": "responses", "attempt": 1, "status": "network_error"}])
        benchmark._save(path, state)
        with self.assertRaises(ValidationError): experiment.summarize(self.output, root=self.root)

    def test_failed_network_attempt_cannot_erase_entire_ledger_for_known_zero(self):
        self.freeze()
        clients = []
        def factory(budget):
            client = GripperClient(budget, sam_error=MetaAPIError("private", code="network_error") if not clients else None)
            clients.append(client)
            return client
        self.run_trial(factory)
        (self.output / "summary.json").unlink()
        path = self.output / "run-state.json"
        state = json.loads(path.read_text())
        next(iter(state["attempts"].values()))["usage"] = parent_accounting([])
        benchmark._save(path, state)
        with self.assertRaises(ValidationError): experiment.summarize(self.output, root=self.root)

    def test_unplanned_response_or_forged_saved_summary_is_rejected(self):
        self.freeze()
        self.run_trial()
        path = self.output / "summary.json"
        value = json.loads(path.read_text())
        value["usage"]["estimated_usd"] = 0
        benchmark._save(path, value)
        with self.assertRaises(ValidationError): experiment.summarize(self.output, root=self.root)
        path.unlink()
        benchmark._save(self.output / "parts" / "unplanned.json", {})
        with self.assertRaises(ValidationError): experiment.summarize(self.output, root=self.root)

    def test_public_projection_excludes_private_payload_and_authorization(self):
        self.freeze()
        result = self.run_trial()
        text = json.dumps(result)
        for forbidden in ("PRIVATE_PROVIDER_TEXT", "PRIVATE_AUTHORIZATION", "raster_base64", "encoded_payload",
                          "inputs/local", "artifacts/local", "output_text", "authored-not-a-key"):
            self.assertNotIn(forbidden, text)
        self.assertIn("not task-positive retention", result["geometry_diagnostics"]["note"])

    def test_cannot_freeze_under_cap_or_with_unbound_geometry_annotations(self):
        for options in ({"max_requests": 5}, {"max_requests": 19},
                        {"frame_annotations": {"0" * 64: "near_gripper_proposal"}},
                        {"frame_annotations": {"0" * 64: "pass"}}):
            with self.subTest(options=options):
                with self.assertRaises(ValidationError): self.freeze(**options)
        with self.assertRaises(ValidationError):
            experiment.freeze(self.parent, self.parent / "child", root=self.root)

    def test_successful_raw_usage_cannot_be_changed_while_preserving_ledger(self):
        self.freeze()
        self.run_trial()
        (self.output / "summary.json").unlink()
        path = self.output / "run-state.json"
        state = json.loads(path.read_text())
        key = next(iter(state["attempts"]))
        case_id = key.split("/")[1]
        part = self.output / "parts" / case_id / "gripper-0.json"
        value = json.loads(part.read_text())
        value["response"]["usage"] = {"input_tokens": 10000, "output_tokens": 10000}
        benchmark._save(part, value)
        state["attempts"][key]["response_sha256"] = benchmark._file_hash(part)
        benchmark._save(path, state)
        with self.assertRaises(ValidationError): experiment.summarize(self.output, root=self.root)

    def test_source_image_tamper_is_caught_before_any_client(self):
        plan = self.freeze()
        frame = plan["cases"][0]["envelope"]["contract"]["cases"][0]["frames"][0]
        (self.root / frame["file"]).write_bytes(b"changed pixels")
        with self.assertRaises(ValidationError): self.run_trial(lambda _: self.fail("No source-tampered admission"))
        self.assertFalse((self.output / "run-started.json").exists())

    def test_reused_client_cannot_dispatch_a_second_gripper_request(self):
        self.freeze()
        client = GripperClient()
        result = self.run_trial(lambda _: client)
        self.assertEqual(1, len(client.calls))
        self.assertEqual(1, result["usage"]["request_attempts"])
        self.assertTrue(all(row["prediction"] is None for row in result["results"]))

    def test_nonzero_sam_image_frame_is_unscored_before_geometry_comparison(self):
        self.freeze()
        clients = []
        def factory(budget):
            client = GripperClient(budget)
            if not clients:
                actual = client.segment_image
                def invalid(*args, **kwargs):
                    result = actual(*args, **kwargs)
                    text = result["text"].replace("<0f>", "<1f>")
                    result["text"] = result["response"]["output_text"] = text
                    result["objects"] = fixtures.decoded(parse_sam_boxes(text))
                    return result
                client.segment_image = invalid
            clients.append(client)
            return client
        result = self.run_trial(factory)
        self.assertEqual("error", result["results"][0]["frames"][0]["status"])
        self.assertEqual("unscored", result["results"][0]["status"])
        self.assertEqual(1, result["geometry_diagnostics"]["missing_or_failed_gripper_diagnostics"])
        self.assertEqual(["pass", "pass"], [row["prediction"] for row in result["results"][1:]])

    def test_multiple_task_checks_are_rejected_instead_of_replicating_aggregate_status(self):
        _, parent_plan = concept._load(self.parent, self.root)
        envelope = deepcopy(parent_plan["cases"][0]["envelope"])
        contract = envelope["contract"]
        extra = deepcopy(contract["protocol"]["checks"][0])
        extra["id"] += "-other"
        contract["protocol"]["checks"].append(extra)
        for case in contract["cases"]:
            case["annotation"]["checks"][extra["id"]] = case["annotation"]["status"]
        approved = benchmark.approve_manifest(contract, benchmark.manifest_hash(contract), confirm_unique_targets=True)
        with self.assertRaises(ValidationError): experiment._one_check(approved)

    def test_projection_never_promotes_unknown_or_unscored_parent(self):
        far = [{"status": "completed", "warning": {"veto": False, "geometry_available": True}}] * 2
        near = [{"status": "completed", "warning": {"veto": True, "geometry_available": True}}] + far[:1]
        missing = [{"status": "error", "warning": {"veto": False, "geometry_available": False}}] + far[:1]
        for frames in (far, near, missing):
            self.assertEqual("unknown", experiment._project("completed", "unknown", frames)["prediction"])
            self.assertIsNone(experiment._project("unscored", None, frames)["prediction"])
        self.assertEqual("unknown", experiment._project("completed", "fail", near)["prediction"])
        self.assertEqual("fail", experiment._project("completed", "fail", far)["prediction"])
        self.assertIsNone(experiment._project("completed", "fail", missing)["prediction"])


def parent_accounting(ledger):
    return concept.accounting._ledger_usage(ledger)


if __name__ == "__main__":
    unittest.main()
