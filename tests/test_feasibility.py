"""Integration assertions for the explicitly synthetic software diagnostic."""

import json
from pathlib import Path
import tempfile
import unittest

from proofloop.feasibility import main, run_diagnostic
from proofloop.fixtures import diagnostic_cases


class SyntheticFeasibilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = run_diagnostic()
        cls.cases = {case["case_id"]: case for case in cls.report["cases"]}

    def test_all_authored_cases_and_planner_meet_expectations(self):
        failures = [case for case in self.report["cases"] if not case["assertion_passed"]]
        self.assertEqual([], failures)
        self.assertTrue(self.report["planner"]["assertion_passed"], self.report["planner"])

    def test_stale_attempt_cannot_pass_before_new_observation(self):
        stages = self.cases["repair-fresh-evidence"]["stages"]
        self.assertEqual(["fail", "unknown", "pass", "pass"], [stage["actual"] for stage in stages])
        self.assertEqual("unknown", self.cases["new-attempt-stale-facts"]["actual"])
        self.assertEqual("unknown", self.cases["repair-partial-refresh"]["actual"])

    def test_unknown_conditions_never_pass(self):
        unsupported = [case for case in self.report["cases"]
                       if case["family"] in {"insufficient_evidence", "freshness"}]
        self.assertTrue(unsupported)
        self.assertTrue(all(case["actual"] == "unknown" for case in unsupported), unsupported)

    def test_requirement_edit_uses_a_distinct_specification(self):
        edited = self.cases["edited-requirement"]
        self.assertEqual("pass", edited["actual"])
        self.assertNotEqual(self.report["baseline_protocol_hash"], edited["protocol_hash"])

    def test_reports_keep_synthetic_provenance_and_no_model_claim(self):
        self.assertEqual(0, self.report["api_requests"])
        self.assertFalse(self.report["is_model_benchmark"])
        self.assertFalse(self.report["is_physical_benchmark"])
        self.assertEqual(len(diagnostic_cases()), self.report["summary"]["episodes"])
        for case in self.report["cases"]:
            self.assertEqual("synthetic_oracle", case["evidence_origin"])
            for evidence_id, source in case["evidence_sources"].items():
                self.assertTrue(evidence_id.startswith("fixture://proofloop/"))
                self.assertEqual("synthetic_oracle", source["origin"])
                self.assertFalse(source["physical_recording"])

    def test_cli_exports_raw_json_and_readable_report(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(0, main(["--output", directory]))
            report = json.loads((Path(directory) / "oracle-report.json").read_text())
            self.assertTrue(report["summary"]["all_assertions_passed"])
            self.assertIn("zero API calls", (Path(directory) / "oracle-report.md").read_text())


if __name__ == "__main__":
    unittest.main()
