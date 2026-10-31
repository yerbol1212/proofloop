"""Offline all-case selection, stage ordering and accounting regression tests."""
from copy import deepcopy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from proofloop import benchmark, crop_verification as verifier
from proofloop.core import ValidationError
from proofloop.meta import MetaAPIError, MetaClient, SAM_MODEL, SPARK_MODEL
from tests import test_concept_verified_experiment as concept
from tests import test_crop_verification as semantic
from tests import test_image_gripper_experiment as gripper
from tests import test_probe_context_crops as parent_fixtures
from tools import probe_context_crops as probe
from tools import verify_context_crops as experiment


class Client(gripper.GripperClient):
    def __init__(self, budget=1, *, stage_error=None, no_tools=False, contact=False, omit_usage=False, empty=False):
        super().__init__(budget, empty=empty)
        self.stage_error, self.no_tools, self.contact, self.omit_usage = stage_error, no_tools, contact, omit_usage

    def spark_json(self, prompt, schema, **options):
        self.calls.append(("spark", prompt, schema, options))
        is_locator = "locator" in options["name"]
        if self.stage_error == ("locator" if is_locator else "final"):
            self.ledger.append({"model": SPARK_MODEL, "endpoint": "responses", "attempt": 1, "status": "network_error"})
            raise MetaAPIError("PRIVATE_TRANSPORT_TEXT", code="network_error")
        parsed = semantic.located("not_visible", "zero") if self.no_tools and is_locator else (
            semantic.located() if is_locator else semantic.supported("zero" if self.no_tools else "one"))
        if self.contact and not is_locator:
            parsed["frames"][0]["manipulator_relation"] = "held_or_contacting"
        result = semantic.spark(parsed, options["image_paths"])
        usage = deepcopy(result["response"]["usage"])
        entry = {"model": SPARK_MODEL, "endpoint": "responses", "attempt": 1, "status": "ok", "usage": usage}
        if self.omit_usage:
            result["response"].pop("usage")
            entry.pop("usage")
        self.ledger.append(entry)
        return result


class VerifyContextCropsTests(unittest.TestCase):
    def setUp(self):
        parent_fixtures.ContextCropProbeTests.setUp(self)
        self.crop_parent = self.root / "artifacts/local/completed-crop-parent"
        probe.freeze(self.parent, self.crop_parent, root=self.root, max_requests=6)
        probe.run(concept.Client, self.crop_parent, root=self.root)
        self.output = self.root / "artifacts/local/full-source-verification"
        self.commands = []
        def render(args, **options):
            if args[-1] == "-version":
                options["stdout"].write(b"authored-ffmpeg-fixture\n")
                return
            return semantic.CropVerificationTests.render_all(self, args, **options)
        patch("proofloop.crop_verification.subprocess.run", side_effect=render).start()

    def freeze(self, **kwargs):
        return experiment.freeze(self.crop_parent, self.output, root=self.root, **kwargs)

    def run_trial(self, factory=Client):
        return experiment.run(factory, self.output, root=self.root)

    def state(self):
        return json.loads((self.output / "run-state.json").read_text())

    def change_state(self, action):
        (self.output / "summary.json").unlink(missing_ok=True)
        state = self.state()
        action(state)
        benchmark._save(self.output / "run-state.json", state)

    def test_freeze_retains_complete_parent_cases_and_caps_before_any_calls(self):
        plan = self.freeze(max_requests=12)
        self.assertEqual(3, len(plan["cases"]))
        self.assertEqual(3, plan["eligible_pairs"])
        self.assertEqual({"spark": 6, "sam": 6, "total": 12}, plan["request_caps"])
        self.assertIn("proofloop/gripper_geometry.py", plan["implementation_sha256"])
        public = experiment.public_plan(self.output, root=self.root)
        self.assertNotIn("/Users/", json.dumps(public))
        result = experiment.summarize(self.output, root=self.root)
        self.assertEqual("not_started", result["status"])
        self.assertEqual(3, result["metrics"]["missing_predictions"])
        self.assertFalse(result["gate_passed"])
        self.assertFalse((self.output / "run-started.json").exists())

    def test_full_three_stage_flow_one_fresh_client_per_request_and_exact_replay(self):
        self.freeze(max_requests=12)
        clients = []
        def factory(budget):
            self.assertEqual(1, budget)
            clients.append(Client(budget))
            return clients[-1]
        result = self.run_trial(factory)
        self.assertEqual("completed", result["status"])
        self.assertEqual((12, 6, 6), tuple(result["usage"][key] for key in ("request_attempts", "sam_requests", "spark_requests")))
        self.assertEqual(0, result["metrics"]["missing_predictions"])
        self.assertEqual(12, len(clients))
        self.assertEqual(["spark", "sam", "sam", "spark"] * 3, [client.calls[0][0] for client in clients])
        for i, client in enumerate(clients):
            self.assertEqual(1, len(client.ledger))
            self.assertEqual(0, client.retries)
            if i % 4 == 3:
                self.assertEqual(8, len(client.calls[0][3]["image_paths"]))
        self.assertEqual(result, experiment.summarize(self.output, root=self.root))
        self.assertFalse(result["changes_production_acceptance"])
        for text in ("PRIVATE_MODEL_BODY", "PRIVATE_PROVIDER_TEXT", "PRIVATE_TRANSPORT_TEXT", "raster_base64", "inputs/local", "artifacts/local"):
            self.assertNotIn(text, json.dumps(result))
        with self.assertRaises(ValidationError):
            self.run_trial(lambda _: self.fail("No automatic second run"))

    def test_zero_visible_tools_is_explicit_and_makes_no_sam_calls(self):
        self.freeze()
        result = self.run_trial(lambda budget: Client(budget, no_tools=True))
        self.assertEqual("completed", result["status"])
        self.assertEqual((0, 6), (result["usage"]["sam_requests"], result["usage"]["spark_requests"]))
        self.assertEqual(0, result["metrics"]["missing_predictions"])

    def test_completed_contact_withholds_but_failed_review_stays_unscored(self):
        self.freeze()
        result = self.run_trial(lambda budget: Client(budget, contact=True))
        self.assertEqual(["unknown"] * 3, [row["prediction"] for row in result["results"]])
        self.assertEqual(0, result["metrics"]["missing_predictions"])
        self.output = self.root / "artifacts/local/failed-final"
        self.freeze()
        result = self.run_trial(lambda budget: Client(budget, stage_error="final"))
        self.assertEqual([None] * 3, [row["prediction"] for row in result["results"]])
        self.assertEqual(3, result["metrics"]["missing_predictions"])
        self.assertEqual("incomplete", result["status"])

    def test_completed_empty_gripper_results_are_unknown_not_inference_failures(self):
        self.freeze()
        result = self.run_trial(lambda budget: Client(budget, empty=True))
        self.assertEqual("completed", result["status"])
        self.assertEqual(["unknown"] * 3, [row["prediction"] for row in result["results"]])
        self.assertEqual(0, result["metrics"]["missing_predictions"])

    def test_locator_failure_cannot_trigger_sam_final_or_fabricated_unknown(self):
        self.freeze()
        result = self.run_trial(lambda budget: Client(budget, stage_error="locator"))
        self.assertEqual(3, result["usage"]["spark_requests"])
        self.assertEqual(0, result["usage"]["sam_requests"])
        self.assertTrue(all(row["prediction"] is None for row in result["results"]))

    def test_missing_reported_tokens_never_passes_continuation_gate(self):
        self.freeze()
        result = self.run_trial(lambda budget: Client(budget, omit_usage=True))
        self.assertTrue(result["usage"]["unreported_charge_possible"])
        self.assertFalse(result["gate_passed"])
        with patch.object(experiment, "_metrics", return_value={**result["metrics"], "correct": 3,
                "correct_acceptances": 1, "reference_pass_cases": 1}):
            (self.output / "summary.json").unlink()
            self.assertFalse(experiment.summarize(self.output, root=self.root)["gate_passed"])

    def test_insufficient_budget_source_or_parent_drift_blocks_before_dispatch(self):
        with self.assertRaises(ValidationError):
            self.freeze(max_requests=11)
        self.output = self.root / "artifacts/local/protected-run"
        self.freeze()
        path = self.crop_parent / "summary.json"
        path.write_bytes(path.read_bytes() + b" ")
        with self.assertRaises(ValidationError):
            self.run_trial(lambda _: self.fail("No changed-parent dispatch"))
        self.assertFalse((self.output / "run-started.json").exists())

    def test_raw_response_or_ledger_relabeling_is_rejected(self):
        self.freeze()
        self.run_trial()
        self.change_state(lambda state: state["attempts"]["spark/final-0"]["usage"].update(ledger=[]))
        with self.assertRaises(ValidationError):
            experiment.summarize(self.output, root=self.root)

    def test_extra_raw_response_cannot_be_hidden(self):
        self.freeze()
        self.run_trial()
        path = self.output / "parts/unplanned.json"
        path.write_text("{}")
        with self.assertRaises(ValidationError):
            experiment.summarize(self.output, root=self.root)

    def test_unrecorded_final_preparation_is_rejected(self):
        self.freeze()
        self.run_trial()
        path = self.output / "final-pairs/unplanned/image.png"
        path.parent.mkdir(parents=True)
        path.write_bytes(b"authored-unplanned-image")
        with self.assertRaises(ValidationError):
            experiment.summarize(self.output, root=self.root)

    def test_auth_billing_and_rate_errors_stop_further_stage_admission(self):
        for status in (401, 402, 403, 429):
            self.output = self.root / ("artifacts/local/stop-" + str(status))
            self.freeze()
            class HTTPClient(Client):
                def spark_json(self, *args, **kwargs):
                    response = {"error": {"code": "authored_http", "message": "PRIVATE_HTTP_BODY"}}
                    self.ledger.append({"model": SPARK_MODEL, "endpoint": "responses", "attempt": 1,
                        "status": "http_error", "http_status": status, "error_code": "authored_http"})
                    raise MetaAPIError("Authored HTTP error", code="authored_http", status=status, response=response)
            result = self.run_trial(HTTPClient)
            self.assertEqual(status, result["admission_stop_code"])
            self.assertEqual(1, result["usage"]["request_attempts"])
            self.assertEqual(0, result["usage"]["sam_requests"])
            self.assertEqual(3, result["metrics"]["missing_predictions"])
            self.assertFalse(result["gate_passed"])
            self.assertNotIn("PRIVATE_HTTP_BODY", json.dumps(result))


if __name__ == "__main__":
    unittest.main()
