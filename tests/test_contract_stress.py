"""Authored saved-response fixtures; no real recordings, network or API usage."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from proofloop.benchmark import comparison_metrics, evaluate_guided, evaluate_spark, manifest_hash
from proofloop.contract_stress import (GUIDED_SHA256, _load_control, _ledger_records, contract_stress_summary,
                                      prepare_control, run_controls)
from proofloop.core import ValidationError
from proofloop.guided import GUIDED_VERSION, guided_segment_pair
from proofloop.meta import MetaAPIError, MetaClient
from tests.test_benchmark import approved, contract, spark_result
from tests.test_guided import FakeClient, jpeg_header


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def fixture(root):
    value = contract()
    source = root / value["sources"][0]["file"]
    source.parent.mkdir(parents=True)
    source.write_bytes(b"authored-video-container-fixture")
    value["sources"][0]["sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    for case in value["cases"]:
        case["annotation"].update(status="pass", checks={"glass-dispense": "pass"})
        for frame in case["frames"]:
            path = root / frame["file"]
            path.write_bytes(jpeg_header(suffix=path.name.encode()))
            frame["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = approved(value)
    folder = root / "artifacts/local/original-fixture"
    raw, responses = {"guided": {}, "spark": {}}, {"guided": [], "spark": []}
    for case in value["cases"]:
        cid = case["id"]
        pair = guided_segment_pair(FakeClient(), [root / frame["file"] for frame in case["frames"]], value["concepts"])
        pair["locator"].update(usage={"input_tokens": 100, "output_tokens": 200}, latency_seconds=.1)
        objects = [item for by_object in pair["results"].values() for item in by_object.values()]
        for item in objects:
            item.update(usage={"images_processed": 1}, latency_seconds=.1)
        spark = spark_result(case, "pass")
        spark.update(usage={"input_tokens": 100, "output_tokens": 150}, latency_seconds=.1)
        raw["guided"][cid], raw["spark"][cid] = pair["results"], spark
        responses["guided"].extend([pair["locator"]] + objects)
        responses["spark"].append(spark)
        save(folder / "guided" / f"{cid}.json", pair)
        save(folder / "spark" / f"{cid}.json", spark)
    report = {"contract_hash": manifest["contract_hash"], "status": "completed", "inference_origin": "authored_fixture",
              "conditions": {}}
    for name, evaluator in (("guided", evaluate_guided), ("spark", evaluate_spark)):
        cases = evaluator(manifest, raw[name], **({"observation_version": GUIDED_VERSION} if name == "guided" else {}))
        cost_client = NewSpark(0)
        cost_client.ledger = _ledger_records(responses[name])
        report["conditions"][name] = {"status": "completed", "cases": cases, "metrics": comparison_metrics(manifest, cases),
            "usage": cost_client.usage_summary(), **({"observation_version": GUIDED_VERSION} if name == "guided" else {})}
        save(folder / name / "results.json", raw[name])
    save(folder / "approved-manifest.json", manifest)
    save(folder / "report.json", report)
    return folder


class NewSpark:
    retries = 0
    usage_summary = MetaClient.usage_summary

    def __init__(self, budget, *, status="fail", error=False):
        self.max_requests, self.ledger, self.status, self.error = budget, [], status, error
        self.calls = []

    def spark_json(self, prompt, schema, **kwargs):
        self.calls.append((prompt, kwargs))
        if self.error:
            raise MetaAPIError("Authored failed call fixture")
        paths = kwargs["image_paths"]
        case = {"frames": [{"sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in paths]}
        value = spark_result(case, self.status)
        value.update(usage={"input_tokens": 100, "output_tokens": 100}, latency_seconds=.1)
        self.ledger.extend(_ledger_records([value]))
        return value


class ContractSensitivityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.folder = fixture(self.root)
        self.plan = prepare_control(self.folder, "initial", "staging", review_reason="Authored positive fixture is in dispense.", root=self.root)
        self.output = self.root / "artifacts/local/opposite-control"

    def run_control(self, client=None):
        self.client = client or NewSpark(1)
        return run_controls([self.plan], self.output, lambda budget: self.client, root=self.root, inference_origin="authored_fixture")

    def test_same_perception_new_contract_regrounds_with_new_spec_and_fails(self):
        control = _load_control(self.plan, self.root)
        self.assertEqual("pass", control["original_guided"]["status"])
        self.assertEqual("fail", control["new_guided"][0]["status"])
        old = control["original_guided"]["grounding"]["observations"]
        new = control["new_guided"][0]["grounding"]["observations"]
        self.assertNotEqual(old[0]["spec_hash"], new[0]["spec_hash"])
        self.assertNotEqual(old[0]["evidence_id"], new[0]["evidence_id"])
        self.assertEqual(GUIDED_SHA256, self.plan["guided_implementation_sha256"])
        self.assertEqual("assistant_under_user_authorization", self.plan["changed_manifest"]["approval"]["origin"])

    def test_fresh_spark_gets_changed_requirement_not_old_verdict(self):
        value = self.run_control()
        self.assertEqual(1, len(self.client.calls))
        prompt, kwargs = self.client.calls[0]
        self.assertIn('"region_id": "staging"', prompt)
        self.assertIn("must be wholly inside the calibrated staging", prompt)
        self.assertNotIn("Authored positive fixture is in dispense", prompt)
        self.assertEqual(2048, kwargs["max_output_tokens"])
        changed = value["controls"][0]["changed_contract"]
        self.assertEqual("fail", changed["spark"]["status"])
        self.assertEqual("replayed_authored_fixture", changed["guided"]["inference_origin"])
        self.assertEqual("authored_fixture", changed["spark"]["inference_origin"])

    def test_original_cost_stays_visible_when_guided_incremental_cost_is_zero(self):
        value = self.run_control()
        costs = value["controls"][0]["costs"]
        self.assertEqual(3, costs["original_guided_perception"]["request_count"])
        self.assertGreater(costs["original_guided_perception"]["estimated_usd"], 0)
        self.assertEqual(0, costs["incremental_guided"]["request_count"])
        self.assertEqual(1, costs["incremental_spark"]["request_count"])
        self.assertEqual(0, value["new_independent_source_count"])
        self.assertFalse(value["natural_execution_failure_verified"])
        self.assertFalse(value["physical_action_execution_verified"])

    def test_all_preflights_and_freeze_happen_before_first_call(self):
        plan2 = prepare_control(self.folder, "final", "staging", review_reason="Second authored positive fixture.", root=self.root)
        client = NewSpark(2)
        original_call = client.spark_json
        def check_freeze(*args, **kwargs):
            frozen = json.loads((self.output / "frozen-controls.json").read_text())
            self.assertEqual(2, len(frozen["plans"]))
            return original_call(*args, **kwargs)
        client.spark_json = check_freeze
        value = run_controls([self.plan, plan2], self.output, lambda budget: client, root=self.root, inference_origin="authored_fixture")
        self.assertEqual(2, value["incremental_spark_request_count"])
        self.assertEqual(2, len(client.calls))

    def test_changed_source_bytes_stop_before_client_creation(self):
        source = self.root / self.plan["changed_manifest"]["contract"]["cases"][0]["frames"][0]["file"]
        source.write_bytes(b"changed")
        calls = []
        with self.assertRaises(ValidationError):
            run_controls([self.plan], self.output, lambda budget: calls.append(budget), root=self.root)
        self.assertEqual([], calls)

    def test_altered_changed_contract_identity_or_geometry_rejected(self):
        for mutation in (lambda p: p["changed_manifest"]["contract"]["concepts"].update(glass="spoon"),
                         lambda p: p["changed_manifest"]["contract"]["cases"][0]["regions"].update(staging=[.5, 0, 1, 1])):
            value = deepcopy(self.plan)
            mutation(value)
            with self.assertRaises(ValidationError):
                _load_control(value, self.root)

    def test_altered_masks_and_locator_are_bound_to_frozen_files(self):
        path = self.folder / "guided/initial.json"
        raw = json.loads(path.read_text())
        raw["locator"]["parsed"]["frames"][0]["objects"][0]["candidates"][0]["x1"] += 1
        save(path, raw)
        with self.assertRaises(ValidationError):
            _load_control(self.plan, self.root)

    def test_original_region_cannot_be_used_as_negative_control(self):
        with self.assertRaises(ValidationError):
            prepare_control(self.folder, "initial", "dispense", review_reason="Invalid no-change control", root=self.root)

    def test_exact_budget_zero_retries_and_no_overwrite_are_enforced(self):
        bad = NewSpark(2)
        with self.assertRaises(ValidationError):
            self.run_control(bad)
        self.assertEqual([], bad.calls)
        self.assertTrue((self.output / "frozen-controls.json").exists())
        with self.assertRaises(ValidationError):
            self.run_control(NewSpark(1))

    def test_failed_call_is_saved_as_error_not_fake_unknown(self):
        with self.assertRaises(MetaAPIError):
            self.run_control(NewSpark(1, error=True))
        report = json.loads((self.output / "report.json").read_text())
        self.assertEqual("error", report["status"])
        self.assertEqual([], report["completed_case_ids"])
        self.assertFalse((self.output / "initial-comparison.json").exists())

    def test_saved_changed_verdict_and_cost_tampering_are_rejected(self):
        self.run_control()
        path = self.output / "initial-comparison.json"
        original = json.loads(path.read_text())
        altered = deepcopy(original)
        altered["conditions"]["spark"]["cases"][0]["status"] = "pass"
        save(path, altered)
        with self.assertRaises(ValidationError):
            contract_stress_summary(self.output, root=self.root)
        save(path, original)
        path = self.output / "report.json"
        altered = json.loads(path.read_text())
        altered["usage"]["estimated_usd"] = 0
        save(path, altered)
        with self.assertRaises(ValidationError):
            contract_stress_summary(self.output, root=self.root)

    def test_public_summary_omits_paths_raw_pixels_and_provider_reasons(self):
        value = self.run_control()
        text = json.dumps(value)
        for private in ("inputs/local", "artifacts/local", "raster_base64", "authored-sam-shape", "Authored baseline test fixture"):
            self.assertNotIn(private, text)
        self.assertEqual(value, contract_stress_summary(self.output, root=self.root))

    def test_public_replay_rejects_duplicate_controls_and_boolean_request_counts(self):
        self.run_control()
        frozen_path, report_path = self.output / "frozen-controls.json", self.output / "report.json"
        frozen, report = json.loads(frozen_path.read_text()), json.loads(report_path.read_text())
        changed = deepcopy(frozen)
        changed["plans"].append(deepcopy(changed["plans"][0]))
        changed["plan_hash"] = manifest_hash({"plans": changed["plans"]})
        altered_report = deepcopy(report)
        altered_report.update(plan_hash=changed["plan_hash"], request_cap=2, completed_case_ids=["initial", "initial"])
        save(frozen_path, changed)
        save(report_path, altered_report)
        with self.assertRaises(ValidationError):
            contract_stress_summary(self.output, root=self.root)
        save(frozen_path, frozen)
        for mutation in (lambda value: value.update(request_cap=True),
                         lambda value: value["usage"].update(request_attempts=True)):
            altered_report = deepcopy(report)
            mutation(altered_report)
            save(report_path, altered_report)
            with self.assertRaises(ValidationError):
                contract_stress_summary(self.output, root=self.root)


if __name__ == "__main__":
    unittest.main()
