"""Authored transport fixtures; no real media, model calls or quality claims."""
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

from proofloop import benchmark, prospective
from proofloop.core import ValidationError
from proofloop.meta import MetaAPIError, MetaClient, SPARK_MODEL
from tests.test_benchmark import approved, contract, media_fixture
from tests.test_guided import FakeClient as GuidedFake, jpeg_header


class Client(GuidedFake):
    """Run the real benchmark/adapters against authored result structures."""
    def __init__(self, budget, error=None):
        super().__init__()
        self.max_requests = budget
        self.error = error

    def spark_json(self, prompt, schema, **kwargs):
        if self.error is not None:
            self.ledger.append({"model": SPARK_MODEL, "status": "error"})
            raise self.error
        if kwargs.get("name") != "proofloop_recorded_baseline":
            value = super().spark_json(prompt, schema, **kwargs)
            self.ledger[-1]["status"] = "ok"
            return value
        self.calls.append(("spark", prompt, schema, kwargs))
        self.ledger.append({"model": SPARK_MODEL, "status": "ok"})
        return {"sources": [{"sha256": hashlib.sha256(Path(p).read_bytes()).hexdigest()}
                            for p in kwargs["image_paths"]],
                "response": {"model": SPARK_MODEL, "status": "completed", "id": "PRIVATE_PROVIDER_ID"},
                "parsed": {"checks": [{"check_id": "glass-dispense", "status": "pass",
                                        "reason": "PRIVATE_PROVIDER_TEXT /Users/private/raw"}]}}

    def segment_image(self, *args, **kwargs):
        result = super().segment_image(*args, **kwargs)
        self.ledger[-1]["status"] = "ok"
        return result

    def usage_summary(self):
        result = MetaClient.usage_summary(self)
        result.update(private_path="/Users/private/ledger", provider_id="PRIVATE_PROVIDER_ID")
        return result


class ProspectiveTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        value = media_fixture(self.root, approved())
        value = deepcopy(value["contract"])
        for case in value["cases"]:
            for frame in case["frames"]:
                path = self.root / frame["file"]
                path.write_bytes(jpeg_header(suffix=case["id"].encode() + str(frame["source_timestamp_ms"]).encode()))
                frame["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        self.envelope = benchmark.approve_manifest(
            value, benchmark.manifest_hash(value), confirm_unique_targets=True,
            review_origin="assistant_under_user_authorization", authorization_reference="PRIVATE_AUTHORIZATION")
        self.output = self.root / "artifacts/local/prospective"

    def freeze(self, **kwargs):
        return prospective.freeze(self.envelope, self.output, root=self.root, **kwargs)

    def run_pilot(self, factory=Client):
        return prospective.run(factory, self.output, root=self.root)

    def test_freeze_binds_sources_cases_code_budgets_and_copies_provenance(self):
        record = self.freeze()
        self.assertEqual({"spark": 2, "guided": 6}, record["planned_requests"])
        self.assertEqual(8, record["total_planned_requests"])
        self.assertEqual(set(prospective.CODE_FILES), set(record["implementation_sha256"]))
        for case in record["cases"]:
            part = json.loads((self.output / "frozen-cases" / (case["case_id"] + ".json")).read_text())
            self.assertEqual(self.envelope["approval"], part["approval"])
            self.assertEqual(1, len(part["contract"]["cases"]))
        text = json.dumps(prospective.public_freeze(self.output, root=self.root))
        self.assertNotIn("PRIVATE_AUTHORIZATION", text)
        self.assertNotIn("inputs/local", text)
        self.assertIn("assistant_under_user_authorization", text)
        self.assertFalse(json.loads(text)["human_reviewed_exact_contract"])
        self.assertEqual(self.envelope["contract"]["requirements"], json.loads(text)["requirements"])
        self.assertEqual(self.envelope["contract"]["concepts"], json.loads(text)["concepts"])
        initial = prospective.summarize(self.output, root=self.root)
        for result in initial["condition_results"].values():
            self.assertEqual(2, result["metrics"]["missing_predictions"])
            self.assertEqual(0, result["metrics"]["unknown_results"])

    def test_cannot_overwrite_freeze_or_exceed_cap(self):
        for budget in (0, 7, 25, True):
            with self.assertRaises(ValidationError):
                self.freeze(max_requests=budget)
        self.freeze()
        with self.assertRaises(ValidationError):
            self.freeze()

    def test_media_tamper_is_rejected_before_client_creation(self):
        self.freeze()
        path = self.root / self.envelope["contract"]["cases"][0]["frames"][0]["file"]
        path.write_bytes(path.read_bytes() + b"tampered")
        with self.assertRaises(ValidationError):
            self.run_pilot(lambda _: self.fail("No client may be created"))
        self.assertFalse((self.output / "run-started.json").exists())

    def test_case_approval_or_code_tamper_rejected(self):
        self.freeze()
        path = self.output / "frozen-cases/initial.json"
        part = json.loads(path.read_text())
        part["approval"]["authorization_reference"] = "different"
        benchmark._save(path, part)
        with self.assertRaises(ValidationError):
            self.run_pilot(lambda _: self.fail("No client may be created"))
        benchmark._save(path, prospective._single(self.envelope, self.envelope["contract"]["cases"][0]))
        with patch.object(prospective, "_code_hashes", return_value={}):
            with self.assertRaises(ValidationError):
                self.run_pilot(lambda _: self.fail("No client may be created"))

    def test_complete_real_harness_replays_all_sidecars_and_is_one_shot(self):
        self.freeze()
        clients = []
        def factory(budget):
            client = Client(budget)
            clients.append(client)
            return client
        result = self.run_pilot(factory)
        self.assertEqual("completed", result["status"])
        self.assertEqual([1, 3, 1, 3], [client.max_requests for client in clients])
        for condition in result["condition_results"].values():
            self.assertEqual(2, condition["metrics"]["eligible_cases"])
            self.assertEqual(0, condition["metrics"]["missing_predictions"])
            self.assertTrue(all(row["inference_sidecar_replayed"] for row in condition["cases"]))
            self.assertTrue(all(row["inference_origin"] == "injected_client_not_verified_live" for row in condition["cases"]))
        with self.assertRaises(ValidationError):
            self.run_pilot(lambda _: self.fail("Never resume a started pilot"))
        self.assertEqual(result, prospective.summarize(self.output, root=self.root))

    def test_failed_guided_does_not_erase_baseline_or_next_case(self):
        self.freeze()
        count = 0
        def factory(budget):
            nonlocal count
            count += 1
            return Client(budget, MetaAPIError("PRIVATE_RAW_FAILURE", code=500) if count == 2 else None)
        result = self.run_pilot(factory)
        self.assertEqual(4, count)
        self.assertEqual("incomplete", result["status"])
        spark, guided = [result["condition_results"][name] for name in ("spark", "guided")]
        self.assertEqual(0, spark["metrics"]["missing_predictions"])
        self.assertEqual(1, guided["metrics"]["missing_predictions"])
        self.assertEqual(0, guided["metrics"]["unknown_results"])
        self.assertEqual(2, guided["metrics"]["eligible_cases"])
        self.assertTrue(guided["unreported_charge_possible"])
        self.assertEqual("error", guided["cases"][0]["status"])
        self.assertIsNone(guided["cases"][0]["prediction"])
        self.assertNotIn("PRIVATE_RAW_FAILURE", json.dumps(result))

    def test_auth_billing_and_rate_failures_stop_new_admissions(self):
        for code in (401, 402, 403, 429):
            with self.subTest(code=code):
                self.output = self.root / ("artifacts/local/stop" + str(code))
                self.freeze()
                calls = []
                def factory(budget):
                    calls.append(budget)
                    return Client(budget, MetaAPIError("private", code=code))
                result = self.run_pilot(factory)
                self.assertEqual([1], calls)
                self.assertEqual(code, result["admission_stop_code"])
                self.assertEqual(2, result["condition_results"]["spark"]["metrics"]["missing_predictions"])
                self.assertEqual(2, result["condition_results"]["guided"]["metrics"]["missing_predictions"])
                self.assertTrue(all(row["status"] == "not_attempted" for row in result["condition_results"]["guided"]["cases"]))

    def test_actual_meta_http_errors_stop_from_ledger_with_string_provider_codes(self):
        for status, code in ((401, "invalid_api_key"), (402, "insufficient_credits"),
                             (403, "permission_denied"), (429, "rate_limit_exceeded")):
            with self.subTest(status=status):
                self.output = self.root / ("artifacts/local/real-http-shape-" + str(status))
                self.freeze()
                clients, calls = [], []
                def opener(request, **kwargs):
                    calls.append("authored HTTP failure; no network request")
                    body = io.BytesIO(json.dumps({"error": {"code": code, "message": "PRIVATE_PROVIDER_TEXT"}}).encode())
                    raise HTTPError("https://api.meta.ai/v1/responses", status, "authored", {}, body)
                def factory(budget):
                    client = MetaClient("authored-test-key", max_requests=budget, retries=0, opener=opener)
                    clients.append(client)
                    return client
                result = self.run_pilot(factory)
                self.assertEqual(1, len(clients))
                self.assertEqual(1, len(calls))
                self.assertEqual(status, clients[0].ledger[0]["http_status"])
                self.assertEqual(code, clients[0].ledger[0]["error_code"])
                self.assertEqual(status, result["admission_stop_code"])
                self.assertEqual(1, result["condition_results"]["spark"]["request_attempts"])
                self.assertEqual(2, result["condition_results"]["spark"]["metrics"]["missing_predictions"])
                self.assertTrue(all(row["status"] == "not_attempted" for row in result["condition_results"]["guided"]["cases"]))
                self.assertNotIn("PRIVATE_PROVIDER_TEXT", json.dumps(result))

    def test_uncaught_meta_status_stops_admissions_even_with_string_code(self):
        self.freeze()
        count = []
        def factory(budget):
            count.append(budget)
            raise MetaAPIError("authored", status=403, code="permission_denied")
        result = self.run_pilot(factory)
        self.assertEqual([1], count)
        self.assertEqual(403, result["admission_stop_code"])

    def test_invalid_client_budget_retries_or_existing_ledger_never_calls_transport(self):
        for name in ("budget", "retries", "ledger"):
            with self.subTest(name=name):
                self.output = self.root / ("artifacts/local/bad-" + name)
                self.freeze(conditions=("spark",))
                clients = []
                def factory(budget):
                    client = Client(budget)
                    if name == "budget": client.max_requests += 1
                    if name == "retries": client.retries = 1
                    if name == "ledger": client.ledger.append({"status": "ok"})
                    clients.append(client)
                    return client
                result = self.run_pilot(factory)
                self.assertTrue(all(not client.calls for client in clients))
                self.assertEqual(0, result["condition_results"]["spark"]["request_attempts"])
                self.assertEqual(2, result["condition_results"]["spark"]["metrics"]["missing_predictions"])

    def test_same_client_cannot_be_reused_even_if_factory_resets_ledger(self):
        self.freeze(conditions=("spark",))
        client = Client(1)
        def factory(_):
            client.ledger.clear()
            return client
        result = self.run_pilot(factory)
        self.assertEqual(1, len(client.calls))
        self.assertEqual(1, result["condition_results"]["spark"]["metrics"]["missing_predictions"])

    def test_public_summary_never_exports_private_authorization_paths_ids_masks_or_provider_text(self):
        self.freeze()
        result = self.run_pilot()
        text = json.dumps(result)
        for secret in ("PRIVATE_AUTHORIZATION", "PRIVATE_PROVIDER_ID", "PRIVATE_PROVIDER_TEXT", "/Users/private",
                       "raster_base64", "encoded_payload", "inputs/local", "artifacts/local"):
            self.assertNotIn(secret, text)

    def test_changed_or_missing_report_sidecar_rejected(self):
        self.freeze(conditions=("spark",))
        self.run_pilot()
        path = self.output / "parts/initial/spark/spark/results.json"
        original = path.read_bytes()
        path.write_bytes(original + b" ")
        with self.assertRaises(ValidationError):
            prospective.summarize(self.output, root=self.root)
        path.unlink()
        with self.assertRaises(ValidationError):
            prospective.summarize(self.output, root=self.root)

    def test_resealed_invalid_decision_still_rejected_by_sidecar_replay(self):
        self.freeze(conditions=("spark",))
        self.run_pilot()
        destination = self.output / "parts/initial/spark"
        path = destination / "report.json"
        report = json.loads(path.read_text())
        report["conditions"]["spark"]["cases"][0]["status"] = "fail"
        benchmark._save(path, report)
        state_path = self.output / "run-state.json"
        state = json.loads(state_path.read_text())
        state["parts"]["initial/spark"]["artifact_sha256"]["report.json"] = benchmark._file_hash(path)
        benchmark._save(state_path, state)
        with self.assertRaises(ValidationError):
            prospective.summarize(self.output, root=self.root)

    def test_illustrative_case_stays_visible_but_unscored(self):
        value = deepcopy(self.envelope["contract"])
        value["cases"][0].update(role="illustrative", annotation=None)
        self.envelope = approved(value)
        self.freeze(conditions=("spark",))
        result = self.run_pilot()
        condition = result["condition_results"]["spark"]
        self.assertFalse(condition["cases"][0]["scored"])
        self.assertIsNone(condition["cases"][0]["truth"])
        self.assertEqual(1, condition["metrics"]["eligible_cases"])
        self.assertEqual(1, condition["metrics"]["illustrative_cases_excluded"])

    def test_source_change_between_parts_stops_later_paid_admissions(self):
        self.freeze(conditions=("spark",))
        original = benchmark.run_benchmark
        count = []
        def changed(*args, **kwargs):
            result = original(*args, **kwargs)
            path = self.root / self.envelope["contract"]["cases"][1]["frames"][0]["file"]
            path.write_bytes(path.read_bytes() + b"changed")
            return result
        with patch.object(benchmark, "run_benchmark", side_effect=changed):
            with self.assertRaises(ValidationError):
                self.run_pilot(lambda budget: count.append(budget) or Client(budget))
        self.assertEqual([1], count)
        self.assertTrue((self.output / "parts/initial/spark/report.json").exists())

    def test_interrupted_admission_stays_missing_and_uncertain_without_resume(self):
        self.freeze(conditions=("spark",))
        with patch.object(benchmark, "run_benchmark", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                self.run_pilot()
        result = prospective.summarize(self.output, root=self.root)
        condition = result["condition_results"]["spark"]
        self.assertEqual("running", result["status"])
        self.assertEqual(2, condition["metrics"]["missing_predictions"])
        self.assertEqual(0, condition["metrics"]["unknown_results"])
        self.assertTrue(condition["unreported_charge_possible"])
        with self.assertRaises(ValidationError):
            self.run_pilot()

    def test_unplanned_report_and_private_ledger_cost_tamper_are_rejected(self):
        self.freeze(conditions=("spark",))
        self.run_pilot()
        extra = self.output / "parts/unplanned/spark/report.json"
        benchmark._save(extra, {})
        with self.assertRaises(ValidationError):
            prospective.summarize(self.output, root=self.root)
        extra.unlink()
        destination = self.output / "parts/initial/spark"
        report_path = destination / "report.json"
        report = json.loads(report_path.read_text())
        report["conditions"]["spark"]["usage"]["estimated_usd"] = 99
        benchmark._save(report_path, report)
        state_path = self.output / "run-state.json"
        state = json.loads(state_path.read_text())
        state["parts"]["initial/spark"]["usage"]["estimated_usd"] = 99
        state["parts"]["initial/spark"]["artifact_sha256"]["report.json"] = benchmark._file_hash(report_path)
        benchmark._save(state_path, state)
        with self.assertRaises(ValidationError):
            prospective.summarize(self.output, root=self.root)


if __name__ == "__main__":
    unittest.main()
