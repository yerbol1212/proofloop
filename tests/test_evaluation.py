"""Authored diagnostic fixtures; no network, licensed media, or live inference."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from proofloop.benchmark import comparison_metrics, evaluate_sam, evaluate_spark, validate_report
from proofloop.core import ValidationError
from proofloop.evaluation import evaluation_summary, main
from tests.test_benchmark import approved, contract, digest, result, spark_result


def fixture(*, truths=("fail", "fail", "pass", "pass", "unknown", "unknown"),
            sam=("pass", "unknown", "pass", "unknown", "pass", "unknown"),
            spark=("fail", "fail", "pass", "pass", "unknown", "unknown")):
    value = contract()
    template = value["cases"][0]
    value["cases"] = []
    for index, truth in enumerate(truths):
        case = deepcopy(template)
        case.update(id=f"case-{index}",
                    frames=[{"file": f"inputs/local/authored/case-{index}-{time}.jpg",
                             "sha256": digest(f"case-{index}-{time}"), "source_timestamp_ms": time}
                            for time in (500, 1500)],
                    annotation={**template["annotation"], "status": truth, "checks": {"glass-dispense": truth}})
        value["cases"].append(case)
    envelope = approved(value)
    raw = {"sam": {}, "spark": {}}
    for index, case in enumerate(envelope["contract"]["cases"]):
        cid = case["id"]
        raw["sam"][cid] = {frame["sha256"]: {"glass": result(frame["sha256"],
                             region="staging" if sam[index] == "fail" else "dispense", empty=sam[index] == "unknown")}
                           for frame in case["frames"]}
        raw["spark"][cid] = spark_result(case, spark[index])
    report = {"contract_hash": envelope["contract_hash"], "status": "completed",
              "inference_origin": "authored_fixture", "evaluation_role": "development_reprocessing", "conditions": {}}
    for name, evaluator in (("sam", evaluate_sam), ("spark", evaluate_spark)):
        cases = evaluator(envelope, raw[name], inference_origin="authored_fixture")
        report["conditions"][name] = {"status": "completed", "cases": cases,
                                      "metrics": comparison_metrics(envelope, cases)}
    return envelope, report, raw


class EvaluationDiagnosticTests(unittest.TestCase):
    def test_known_fail_and_unknown_acceptance_have_distinct_denominators(self):
        envelope, report, raw = fixture()
        value = evaluation_summary(envelope, report, raw_results=raw)
        metric = value["conditions"]["sam"]["case_metrics"]
        self.assertEqual(2, metric["known_fail_count"])
        self.assertEqual(1, metric["known_fail_accepted"])
        self.assertEqual(.5, metric["known_fail_acceptance_rate"])
        self.assertEqual(2, metric["unknown_truth_count"])
        self.assertEqual(1, metric["unknown_truth_accepted"])
        self.assertEqual(.5, metric["unknown_truth_acceptance_rate"])
        self.assertEqual(3, metric["accepted_decisions"])
        self.assertEqual(2, metric["accepted_decisions_not_supported_by_label"])
        self.assertEqual(.666667, metric["accepted_decision_label_error_rate"])
        self.assertEqual(2, value["report_integrity"]["inference_sidecars_replayed"])

    def test_selective_accuracy_excludes_unknown_truth_and_keeps_coverage(self):
        envelope, report, _ = fixture()
        metric = evaluation_summary(envelope, report)["conditions"]["sam"]["case_metrics"]
        self.assertEqual(4, metric["known_truth_count"])
        self.assertEqual(2, metric["known_truth_decided"])
        self.assertEqual(.5, metric["known_truth_coverage"])
        self.assertEqual(.5, metric["known_truth_selective_accuracy"])
        self.assertEqual(1, metric["unknown_truth_abstained"])
        self.assertEqual(1, metric["unknown_truth_decided"])
        self.assertEqual(.5, metric["abstention_rate"])

    def test_all_abstaining_has_zero_coverage_and_unavailable_selective_accuracy(self):
        envelope, report, _ = fixture(sam=("unknown",) * 6)
        metric = evaluation_summary(envelope, report)["conditions"]["sam"]["case_metrics"]
        self.assertEqual(0, metric["known_truth_coverage"])
        self.assertIsNone(metric["known_truth_selective_accuracy"])
        self.assertIsNone(metric["accepted_decision_label_error_rate"])
        self.assertEqual(0, metric["correct_acceptance_rate"])
        self.assertEqual(2, metric["correct"])

    def test_empty_denominators_are_none_not_a_perfect_score(self):
        envelope, report, _ = fixture(truths=("pass",), sam=("pass",), spark=("pass",))
        metric = evaluation_summary(envelope, report)["conditions"]["sam"]["case_metrics"]
        self.assertIsNone(metric["known_fail_acceptance_rate"])
        self.assertIsNone(metric["unknown_truth_acceptance_rate"])
        self.assertEqual(1, metric["known_truth_selective_accuracy"])

    def test_paired_decisions_compare_exact_same_cases(self):
        envelope, report, _ = fixture()
        value = evaluation_summary(envelope, report)
        pair = value["pairwise"][0]
        self.assertEqual(("sam", "spark"), (pair["left"], pair["right"]))
        self.assertEqual(6, pair["cases"]["compared"])
        self.assertEqual(2, pair["cases"]["both_correct"])
        self.assertEqual(4, pair["cases"]["only_right_correct"])
        self.assertEqual(0, pair["cases"]["only_left_correct"])
        self.assertEqual(4, pair["cases"]["disagree"])
        self.assertEqual(["case-0", "case-1", "case-3", "case-4"],
                         [row["case_id"] for row in pair["cases"]["disagreements"]])

    def test_check_disagreement_remains_visible_when_case_aggregates_agree(self):
        value = contract()
        value["cases"] = value["cases"][:1]
        value["protocol"]["checks"].append({"id": "glass-visible", "kind": "visible", "object_id": "glass",
                                             "region_id": "staging", "expected": True})
        case = value["cases"][0]
        case["annotation"]["checks"]["glass-visible"] = "fail"
        envelope = approved(value)
        frame_raw = {frame["sha256"]: {"glass": result(frame["sha256"], "staging")} for frame in case["frames"]}
        sam = evaluate_sam(envelope, {case["id"]: frame_raw})
        spark = deepcopy(sam)
        spark[0].pop("grounding")
        for check in spark[0]["checks"]:
            check["status"] = "fail"
        report = {"contract_hash": envelope["contract_hash"], "status": "completed", "conditions": {
            name: {"status": "completed", "cases": cases, "metrics": comparison_metrics(envelope, cases)}
            for name, cases in (("sam", sam), ("spark", spark))}}
        pair = evaluation_summary(envelope, report)["pairwise"][0]
        self.assertEqual(0, pair["cases"]["disagree"])
        self.assertEqual(1, pair["checks"]["disagree"])
        self.assertEqual("glass-visible", pair["checks"]["disagreements"][0]["check_id"])

    def test_byte_identical_source_aliases_do_not_inflate_group_count(self):
        envelope, report, _ = fixture()
        value = deepcopy(envelope["contract"])
        alias = {**value["sources"][0], "id": "alias", "file": "inputs/local/authored/alias.mp4"}
        value["sources"].append(alias)
        value["cases"][-1]["source_id"] = "alias"
        envelope = approved(value)
        report["contract_hash"] = envelope["contract_hash"]
        value = evaluation_summary(envelope, report)
        self.assertEqual(2, value["source_id_count"])
        self.assertEqual(1, value["source_file_group_count"])
        self.assertEqual(6, value["source_groups"][0]["case_count"])
        self.assertEqual(["alias", "authored-source"], value["source_groups"][0]["source_ids"])
        self.assertEqual(6, value["conditions"]["sam"]["by_source"][0]["case_metrics"]["eligible"])

    def test_illustrative_is_excluded_from_scores_pairs_and_source_groups(self):
        envelope, report, _ = fixture()
        value = deepcopy(envelope["contract"])
        value["cases"][-1].update(role="illustrative", annotation=None)
        envelope = approved(value)
        report["contract_hash"] = envelope["contract_hash"]
        for condition in report["conditions"].values():
            condition["metrics"] = comparison_metrics(envelope, condition["cases"])
        value = evaluation_summary(envelope, report)
        self.assertEqual(1, value["illustrative_cases_excluded"])
        self.assertEqual(5, value["benchmark_case_count"])
        self.assertEqual(5, value["conditions"]["sam"]["case_metrics"]["eligible"])
        self.assertEqual(5, value["pairwise"][0]["checks"]["compared"])

    def test_unfinished_conditions_are_excluded_not_fabricated_unknowns(self):
        envelope, report, _ = fixture()
        report["status"] = "error"
        report["conditions"]["sam"] = {"status": "error"}
        value = evaluation_summary(envelope, report)
        self.assertEqual({"sam": "error"}, value["excluded_conditions"])
        self.assertEqual(["spark"], list(value["conditions"]))
        self.assertEqual([], value["pairwise"])

    def test_all_incomplete_report_has_no_scores(self):
        envelope, report, _ = fixture()
        report.update(status="running", conditions={"spark": {"status": "running"}})
        value = evaluation_summary(envelope, report)
        self.assertEqual({}, value["conditions"])
        self.assertEqual([], value["pairwise"])
        self.assertEqual({"spark": "running"}, value["excluded_conditions"])
        self.assertFalse(value["report_integrity"]["metrics_recomputed"])

    def test_only_illustrative_cases_produce_unavailable_metrics(self):
        envelope, report, _ = fixture()
        value = deepcopy(envelope["contract"])
        for case in value["cases"]:
            case.update(role="illustrative", annotation=None)
        envelope = approved(value)
        report["contract_hash"] = envelope["contract_hash"]
        for condition in report["conditions"].values():
            condition["metrics"] = comparison_metrics(envelope, condition["cases"])
        value = evaluation_summary(envelope, report)
        self.assertEqual(0, value["benchmark_case_count"])
        self.assertEqual(0, value["source_file_group_count"])
        self.assertEqual(0, value["pairwise"][0]["cases"]["compared"])
        self.assertIsNone(value["conditions"]["sam"]["case_metrics"]["accuracy"])

    def test_report_totals_and_grounded_decisions_must_validate_before_scoring(self):
        envelope, report, _ = fixture()
        report["conditions"]["sam"]["metrics"]["accuracy"] = 1
        with self.assertRaises(ValidationError):
            evaluation_summary(envelope, report)
        envelope, report, _ = fixture()
        report["conditions"]["sam"]["cases"][0]["checks"][0]["status"] = "fail"
        report["conditions"]["sam"]["cases"][0]["status"] = "fail"
        report["conditions"]["sam"]["metrics"] = comparison_metrics(envelope, report["conditions"]["sam"]["cases"])
        with self.assertRaises(ValidationError):
            evaluation_summary(envelope, report)

    def test_shared_validator_receives_private_sidecars(self):
        envelope, report, raw = fixture()
        with patch("proofloop.evaluation.validate_report", wraps=validate_report) as validate:
            evaluation_summary(envelope, report, raw_results=raw)
        validate.assert_called_once_with(envelope, report, raw_results=raw)
        raw["spark"]["case-0"]["parsed"]["checks"][0]["status"] = "pass"
        with self.assertRaises(ValidationError):
            evaluation_summary(envelope, report, raw_results=raw)

    def test_comparison_id_is_order_stable_but_changes_when_decisions_change(self):
        envelope, report, _ = fixture()
        expected = evaluation_summary(envelope, report)["comparison_id"]
        changed = deepcopy(report)
        changed["conditions"] = dict(reversed(list(changed["conditions"].items())))
        for condition in changed["conditions"].values():
            condition["cases"].reverse()
        changed["provider_secret"] = "should not affect public diagnostic identity"
        self.assertEqual(expected, evaluation_summary(envelope, changed)["comparison_id"])
        _, different, _ = fixture(spark=("pass", "fail", "pass", "pass", "unknown", "unknown"))
        self.assertNotEqual(expected, evaluation_summary(envelope, different)["comparison_id"])

    def test_export_contains_no_raw_inputs_paths_reasons_or_provider_bodies(self):
        envelope, report, raw = fixture()
        report["private"] = {"api_key": "unit-test-only-secret"}
        report["conditions"]["spark"]["cases"][0]["checks"][0]["reason"] = "private-reason-do-not-export"
        text = json.dumps(evaluation_summary(envelope, report, raw_results=raw))
        for forbidden in ("inputs/local", "artifacts/local", "raster_base64", "private-reason-do-not-export",
                          "unit-test-only-secret", "authored-sam-shape"):
            self.assertNotIn(forbidden, text)
        self.assertFalse(json.loads(text)["physical_action_execution_verified"])

    def test_cli_writes_new_file_without_overwriting_existing_result(self):
        envelope, report, _ = fixture()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path, report_path, output = root / "manifest.json", root / "report.json", root / "diagnostic.json"
            manifest_path.write_text(json.dumps(envelope))
            report_path.write_text(json.dumps(report))
            self.assertEqual(0, main([str(manifest_path), str(report_path), "--output", str(output)]))
            original = output.read_bytes()
            with patch("sys.stderr"):
                self.assertEqual(2, main([str(manifest_path), str(report_path), "--output", str(output)]))
            self.assertEqual(original, output.read_bytes())

    def test_cli_rejects_invalid_report_without_exporting_private_details(self):
        envelope, report, _ = fixture()
        report["conditions"]["spark"]["metrics"]["accuracy"] = 10
        report["provider_body"] = "private-test-secret"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path, report_path, output = root / "manifest.json", root / "report.json", root / "out.json"
            manifest_path.write_text(json.dumps(envelope))
            report_path.write_text(json.dumps(report))
            with patch("sys.stderr") as stderr:
                self.assertEqual(2, main([str(manifest_path), str(report_path), "--output", str(output)]))
            self.assertFalse(output.exists())
            self.assertNotIn("private-test-secret", str(stderr.write.call_args_list))


if __name__ == "__main__":
    unittest.main()
