"""Read-only research projection on authored evidence, not accuracy claims."""
from copy import deepcopy
import json
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from proofloop.contract_stress import prepare_control, run_controls
from proofloop.core import ValidationError
from proofloop.research_lab import ResearchLab, _same
from tests.test_contract_stress import NewSpark, fixture, save


class ResearchLabTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.lab = ResearchLab(self.root)
        self.public = self.root / "artifacts/feasibility/contract-stress-20261028.json"
        self.semantic_public = self.root / "artifacts/feasibility/semantic-candidate-challenge-20261028.json"

    def control(self):
        self.original = fixture(self.root)
        plan = prepare_control(self.original, "initial", "staging", review_reason="Authored fixture positive in dispense.", root=self.root)
        self.run = self.root / "artifacts/local/contract-stress-20261028-01"
        summary = run_controls([plan], self.run, lambda budget: NewSpark(budget), root=self.root,
                               inference_origin="authored_fixture")
        save(self.public, summary)
        return summary

    def test_empty_clone_shows_unavailable_without_any_provider_calls(self):
        with patch("proofloop.meta.MetaClient.spark_json", side_effect=AssertionError("No inference on lab open")):
            value = self.lab.snapshot()
        self.assertFalse(value["available"])
        self.assertEqual(0, value["paid_requests_on_open"])
        self.assertTrue(value["read_only"])
        self.assertEqual("unavailable", value["semantic"]["status"])
        self.assertEqual("unavailable", value["contract_stress"]["status"])
        self.assertEqual("unavailable", value["completion_v2"]["status"])
        self.assertEqual("unavailable", value["distractors_v2"]["status"])

    def test_trial_source_images_are_allowlisted_rehashed_and_invalidated_on_refresh(self):
        folder = self.root / "artifacts/local/semantic-v2-distractors-20261028-01"
        prep = folder / "prepared.json"
        frames = []
        for index in range(2):
            source = folder / f"source-{index}.jpg"
            crop = folder / f"crop-{index}.jpg"
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_bytes(b"explicit authored source fixture " + bytes([index]))
            crop.write_bytes(b"explicit authored crop fixture " + bytes([index]))
            frames.append({"frame_index": index, "source_file": str(source), "crop_file": str(crop),
                "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "crop_sha256": hashlib.sha256(crop.read_bytes()).hexdigest(),
                "width": 100, "height": 80, "candidate_box": {"x1": 10, "y1": 10, "x2": 30, "y2": 30}})
        save(prep, {"frames": frames})
        save(folder / "freeze.json", {"cases": [{"case_id": "fixture", "prepared_file": str(prep)}]})
        public = {"cases": [{"case_id": "fixture"}], "metrics": {"completed": 1}}
        public_file = self.root / "artifacts/feasibility/semantic-v2-distractors-20261028.json"
        save(public_file, public)
        with patch("proofloop.distractor_media.validate_prepared_entries", return_value={}), \
                patch("proofloop.semantic_trial.public_trial_summary", return_value=public):
            state = self.lab.snapshot()
            result = state["distractors_v2"]
            self.assertTrue(result["available"])
            image = result["summary"]["cases"][0]["images"][0]
            token = image["source_url"].rsplit("/", 1)[1]
            path = self.lab.media_path(token)
            self.assertEqual(Path(frames[0]["source_file"]).resolve(), path)
            self.assertNotIn(str(self.root), json.dumps(state))
            original = path.read_bytes()
            path.write_bytes(original + b"changed")
            with self.assertRaises(KeyError):
                self.lab.media_path(token)
            path.write_bytes(original)
            changed = deepcopy(public)
            changed["metrics"]["completed"] = 2
            save(public_file, changed)
            self.assertEqual("invalid", self.lab.snapshot()["distractors_v2"]["status"])
            with self.assertRaises(KeyError):
                self.lab.media_path(token)
        for invalid in ("../.env.local", "0" * 40, "data:image/png,private"):
            with self.assertRaises(KeyError):
                self.lab.media_path(invalid)

    def test_crop_confirmation_retains_all_supported_parent_cases_and_exact_public_totals(self):
        directory = self.root / "artifacts/local/semantic-crop-confirmation-20261028-01"
        parent_dir = self.root / "artifacts/local/semantic-v2-distractors-20261028-01"
        public_file = self.root / "artifacts/feasibility/semantic-crop-confirmation-20261028.json"
        parent_public = self.root / "artifacts/feasibility/semantic-v2-distractors-20261028.json"
        parent = {"cases": [{"case_id": "positive", "attempt_status": "completed",
                             "challenge": {"challenge": {"decision": "supported"}}},
                            {"case_id": "withheld", "attempt_status": "completed",
                             "challenge": {"challenge": {"decision": "unknown"}}}]}
        public = {"metrics": {"completed": 1}, "cases": [{"case_id": "positive"}]}
        plan = {"cases": [{"case_id": "positive", "global_report_file": str(parent_dir / "positive.json")}]}
        save(parent_public, parent)
        save(public_file, public)
        save(directory / "freeze.json", plan)
        with patch.object(self.lab, "_trial_controls", return_value=(parent, {})), \
                patch("proofloop.semantic_local.public_confirmation_summary", return_value=public):
            self.assertEqual("validated", self.lab.snapshot()["crop_confirmation"]["status"])
            changed = deepcopy(public)
            changed["metrics"]["completed"] = 2
            save(public_file, changed)
            self.assertEqual("invalid", self.lab.snapshot()["crop_confirmation"]["status"])
            save(public_file, public)
            plan["cases"] = []
            save(directory / "freeze.json", plan)
            self.assertEqual("invalid", self.lab.snapshot()["crop_confirmation"]["status"])
            plan["cases"] = [{"case_id": "positive", "global_report_file": str(self.root / ".env.local")}]
            save(directory / "freeze.json", plan)
            self.assertEqual("invalid", self.lab.snapshot()["crop_confirmation"]["status"])

    def test_controls_replay_same_evidence_under_opposite_requirement(self):
        expected = self.control()
        value = self.lab.snapshot()
        self.assertTrue(value["available"])
        self.assertTrue(value["contract_stress"]["available"])
        self.assertEqual(expected, value["contract_stress"]["summary"])
        control = value["contract_stress"]["summary"]["controls"][0]
        self.assertEqual("pass", control["original_contract"]["guided"]["status"])
        self.assertEqual("fail", control["changed_contract"]["guided"]["status"])
        self.assertEqual(0, control["costs"]["incremental_guided"]["request_count"])
        self.assertGreater(control["costs"]["original_guided_perception"]["estimated_usd"], 0)

    def test_public_metric_mutation_withholds_experiment(self):
        value = self.control()
        value["incremental_estimated_usd"] = 0
        save(self.public, value)
        result = self.lab.snapshot()
        self.assertEqual("invalid", result["contract_stress"]["status"])
        self.assertNotIn("summary", result["contract_stress"])

    def test_extra_public_field_does_not_enter_projection(self):
        value = self.control()
        value["authorization"] = "private-provider-value"
        save(self.public, value)
        result = self.lab.snapshot()
        self.assertEqual("invalid", result["contract_stress"]["status"])
        self.assertNotIn("private-provider-value", json.dumps(result))

    def test_private_frame_mutation_is_rechecked_on_every_request(self):
        self.control()
        self.assertTrue(self.lab.snapshot()["contract_stress"]["available"])
        manifest = json.loads((self.original / "approved-manifest.json").read_text())
        frame = self.root / manifest["contract"]["cases"][0]["frames"][0]["file"]
        original = frame.read_bytes()
        frame.write_bytes(original + b"changed")
        self.assertEqual("invalid", self.lab.snapshot()["contract_stress"]["status"])
        frame.write_bytes(original)
        self.assertTrue(self.lab.snapshot()["contract_stress"]["available"])

    def test_changed_private_baseline_cannot_be_treated_as_original_reference(self):
        self.control()
        path = self.original / "spark/initial.json"
        value = json.loads(path.read_text())
        value["parsed"]["checks"][0]["status"] = "fail"
        save(path, value)
        self.assertEqual("invalid", self.lab.snapshot()["contract_stress"]["status"])

    def test_missing_private_record_never_falls_back_to_public_only(self):
        self.control()
        (self.run / "initial-spark.json").unlink()
        section = self.lab.snapshot()["contract_stress"]
        self.assertEqual("unavailable", section["status"])
        self.assertNotIn("summary", section)

    def test_malformed_public_json_isolated_from_other_section(self):
        self.control()
        self.semantic_public.write_text('{"duplicate": 1, "duplicate": 2}')
        value = self.lab.snapshot()
        self.assertTrue(value["contract_stress"]["available"])
        self.assertEqual("invalid", value["semantic"]["status"])
        self.assertNotIn("summary", value["semantic"])

    def test_record_symlink_outside_fixed_directory_is_not_followed(self):
        self.public.parent.mkdir(parents=True)
        private = self.root / "private.json"
        private.write_text('{"secret":"do-not-show"}')
        self.public.symlink_to(private)
        result = self.lab.snapshot()
        self.assertEqual("invalid", result["contract_stress"]["status"])
        self.assertNotIn("do-not-show", json.dumps(result))

    def test_oversized_record_is_rejected_before_replay(self):
        self.public.parent.mkdir(parents=True)
        self.public.write_text(" " * 100)
        with patch("proofloop.research_lab.MAX_SUMMARY_BYTES", 50), patch(
                "proofloop.research_lab.contract_stress_summary", side_effect=AssertionError("Oversized input must stop")):
            self.assertEqual("invalid", self.lab.snapshot()["contract_stress"]["status"])

    def test_semantic_requires_complete_private_replay_not_public_shape_alone(self):
        save(self.semantic_public, {"experiment": "authored semantic boundary fixture"})
        with patch("proofloop.research_lab._replay_semantic",
                   side_effect=ValidationError("private file /Users/private missing")):
            result = self.lab.snapshot()
        self.assertEqual("invalid", result["semantic"]["status"])
        self.assertNotIn("/Users", json.dumps(result))

    def test_semantic_public_and_private_projection_must_match_exactly(self):
        public = {"experiment": "authored aggregate composition fixture", "scored": 3, "unscored": 2}
        save(self.semantic_public, public)
        with patch("proofloop.research_lab._replay_semantic", return_value=deepcopy(public)) as replay:
            result = self.lab.snapshot()
            self.assertTrue(result["semantic"]["available"])
            replay.assert_called_once_with(self.root.resolve())
            public["scored"] = 4
            save(self.semantic_public, public)
            self.assertEqual("invalid", self.lab.snapshot()["semantic"]["status"])

    def test_semantic_private_strings_withheld_even_when_summaries_match(self):
        public = {"note": "Bearer private-authorization"}
        save(self.semantic_public, public)
        with patch("proofloop.research_lab._replay_semantic", return_value=public):
            result = self.lab.snapshot()
        self.assertEqual("invalid", result["semantic"]["status"])
        self.assertNotIn("private-authorization", json.dumps(result))

    def test_output_is_detached_from_future_snapshot_mutation(self):
        self.control()
        value = self.lab.snapshot()
        value["contract_stress"]["summary"]["controls"][0]["case_id"] = "changed"
        self.assertEqual("initial", self.lab.snapshot()["contract_stress"]["summary"]["controls"][0]["case_id"])

    def test_boolean_and_numeric_fields_are_not_equivalent(self):
        self.assertFalse(_same({"count": True}, {"count": 1}))


if __name__ == "__main__":
    unittest.main()
