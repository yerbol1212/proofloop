"""Authored orchestration fixtures; no real media or provider inference."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from proofloop import gripper_experiment as experiment
from proofloop.core import ValidationError
from tests import test_contact_experiment as fixtures


class GripperExperimentTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.old_dir, self.fresh_dir, self.previous_dir, self.contact_dir, self.output = [
            self.root / "artifacts/local" / name for name in ("old-source", "fresh-source", "previous", "contact", "grippers")]
        for directory in (self.old_dir, self.fresh_dir, self.previous_dir, self.contact_dir):
            directory.mkdir(parents=True)
        self.old_path, self.fresh_path = self.old_dir / "freeze.json", self.fresh_dir / "freeze.json"
        self.old_plan = fixtures.ContactExperimentTests.source_plan(self, self.old_dir, "old", 2)
        self.fresh_plan = fixtures.ContactExperimentTests.source_plan(self, self.fresh_dir, "fresh", 3)
        fixtures.dump(self.old_path, self.old_plan)
        fixtures.dump(self.fresh_path, self.fresh_plan)
        self.clips = self.old_plan["clips"] + self.fresh_plan["clips"]
        self.previous_path = self.previous_dir / "freeze.json"
        fixtures.dump(self.previous_path, {"authored_prior_freeze": True})
        for entry in self.clips:
            old = entry["clip_id"].startswith("old")
            directory = self.previous_dir if old else self.contact_dir
            fixtures.dump(directory / f"{entry['clip_id']}-video.json", self.raw(entry))
            for version in ("v2", "v3"):
                directory = self.previous_dir if old and version == "v2" else self.contact_dir
                for cp in entry["checkpoints"]:
                    checked = fixtures.ContactExperimentTests.checked(self, entry, cp["frame_index"])
                    checked["fixture_guard_version"] = version
                    fixtures.dump(directory / f"{entry['clip_id']}-{version}-{cp['frame_index']}.json", checked)
        self.historical = {"experiment_version": experiment.baseline.VERSION, "new_provider_attempts": 50,
            "failed_operation_count": 1, "estimated_incremental_usd": .431506,
            "accounting": {"uncertain": 1}, "cohorts": {cohort: {
                "methods": {name: {"authored_exact_published_method": name} for name in experiment.baseline.METHODS},
                "evidence": [{"retained_failure": "network_error", "status": "unscored"}]}
                for cohort in experiment.COHORTS}}
        self.contact_path = self.contact_dir / "freeze.json"
        fixtures.dump(self.contact_dir / "public-result.json", self.historical)
        self.contact_plan = {"freeze_hash": "a" * 64,
            "previous_frozen_file": self.previous_path.relative_to(self.root).as_posix(),
            "old_source_frozen_file": self.old_path.relative_to(self.root).as_posix(),
            "fresh_frozen_file": self.fresh_path.relative_to(self.root).as_posix(),
            "input_records": [experiment.masks._file_record(self.root, path, "authored_baseline_input")
                for directory in (self.old_dir, self.fresh_dir, self.previous_dir)
                for path in directory.rglob("*") if path.is_file()]}
        fixtures.dump(self.contact_path, self.contact_plan)
        self.annotation_path = self.root / "artifacts/local/gripper-seed-review.json"
        self.annotations = {"review_origin": "assistant_under_user_authorization", "reviewed_before_inference": True,
            "clips": [{"clip_id": entry["clip_id"], "seeds": [
                {"object_id": "left_gripper", "track_id": 3, "frame_index": 0,
                 "box": {"x1": 0, "y1": 20, "x2": 9, "y2": 50}},
                {"object_id": "right_gripper", "track_id": 4, "frame_index": 0,
                 "box": {"x1": 80, "y1": 20, "x2": 99, "y2": 50}}]} for entry in self.clips]}
        fixtures.dump(self.annotation_path, self.annotations)
        self.baseline_replay = self.override(experiment.baseline, "replay",
            side_effect=lambda *a, **k: deepcopy(self.historical))
        self.prepare = self.override(experiment, "prepare_clip", side_effect=self.fake_prepare)
        self.override(experiment, "verify_prepared", return_value=None)

    def override(self, owner, name, **kwargs):
        handle = patch.object(owner, name, **kwargs)
        result = handle.start()
        self.addCleanup(handle.stop)
        return result

    def entry(self):
        return deepcopy(self.fresh_plan["clips"][0])

    def raw(self, entry, *, status="completed"):
        return {"status": status, "prepared": deepcopy(entry["prepared"]), "attempts": [{"status": "ok"}],
            "frames": [{"frame_index": index, "timestamp_ms": entry["frames"][index]["timestamp_ms"], "candidates": [
                {"track_id": 1, "object_id": "remote", "box": [.1, .1, .5, .5], "confidence": .95,
                 "mask_supported": True}]} for index in range(3)]}

    def fake_prepare(self, path, seeds, **kwargs):
        entry = next(row for row in self.clips if row["prepared"]["clip_file"] == path)
        prepared = {"clip_file": path, "source": deepcopy(entry["prepared"]["source"]), "seeds": deepcopy(seeds),
            "author_review": kwargs, "fixture_preparation": True}
        prepared["prepared_hash"] = experiment.original._digest(prepared)
        return prepared

    def assessment(self, entry=None, relations=None):
        entry = entry or self.entry()
        relations = relations or ["separated"] * 3
        return {"geometry_version": experiment.geometry.VERSION, "config": deepcopy(experiment.geometry.CONFIG),
            "frames": [{"frame_index": index, "source_sha256": entry["prepared"]["source"]["sha256"],
                "target_track_id": 1, "relation": relation, "separation_supported": relation == "separated",
                "comparisons": []} for index, relation in enumerate(relations)]}

    def freeze(self):
        return experiment.freeze(self.root, self.contact_path, self.annotation_path, self.output)

    def test_prospective_freeze_binds_baseline_all_inputs_geometry_and_four_seed_reviews(self):
        plan = self.freeze()
        self.assertFalse(plan["inference_started"])
        self.assertFalse((self.output / "run-started.json").exists())
        self.assertEqual(4, len(plan["gripper_preparations"]))
        self.assertEqual(4, self.prepare.call_count)
        self.assertEqual(self.contact_plan["freeze_hash"], plan["contact_freeze_hash"])
        self.assertEqual(experiment.original._digest(self.historical), plan["contact_public_sha256"])
        self.assertEqual(experiment.geometry.CONFIG, plan["geometry_config"])
        self.assertIn("gripper_geometry.py", plan["implementation_sha256"])
        captured = {row["file"] for row in plan["input_records"]}
        for path in (self.annotation_path, self.contact_path, self.contact_dir / "public-result.json"):
            self.assertIn(path.relative_to(self.root).as_posix(), captured)
        self.assertEqual(0o600, (self.output / "freeze.json").stat().st_mode & 0o777)

    def test_unpublished_or_changed_contact_result_is_rejected_before_new_preparation(self):
        fixtures.dump(self.contact_dir / "public-result.json", {"changed": True})
        with self.assertRaisesRegex(ValidationError, "published"):
            self.freeze()
        self.prepare.assert_not_called()

    def test_review_requires_exact_four_ids_two_initial_grippers_and_no_extra_fields(self):
        original = deepcopy(self.annotations)
        mutations = (
            lambda v: v.update(reviewed_before_inference=False),
            lambda v: v.update(review_origin="unreviewed"),
            lambda v: v["clips"].pop(),
            lambda v: v["clips"][0].update(clip_id="different_source"),
            lambda v: v["clips"][0]["seeds"][0].update(track_id=1),
            lambda v: v["clips"][0]["seeds"][0].update(frame_index=1),
            lambda v: v["clips"][0]["seeds"][0].update(expected="clear"),
        )
        for index, mutate in enumerate(mutations):
            value = deepcopy(original)
            mutate(value)
            fixtures.dump(self.annotation_path, value)
            with self.subTest(index=index), self.assertRaises(ValidationError):
                self.freeze()

    def test_gripper_preparation_must_keep_original_clip_and_full_source_timeline(self):
        actual = self.fake_prepare
        def altered(*args, **kwargs):
            result = actual(*args, **kwargs)
            result["source"]["sha256"] = "0" * 64
            return result
        self.prepare.side_effect = altered
        with self.assertRaisesRegex(ValidationError, "unchanged"):
            self.freeze()

    def test_rehashed_code_configuration_and_geometry_mutations_fail(self):
        plan = self.freeze()
        for field, value in (("config", {"max_requests": 5}), ("geometry_config", {}), ("geometry_version", "other"),
                ("implementation_sha256", {}), ("methods", []), ("inference_started", True)):
            changed = deepcopy(plan)
            changed[field] = value
            changed["freeze_hash"] = experiment.original._digest({k: v for k, v in changed.items() if k != "freeze_hash"})
            with self.subTest(field=field), self.assertRaises(ValidationError):
                experiment.verify_plan(self.root, changed, files=False)

    def test_changed_seed_review_preparation_and_baseline_inputs_are_detected(self):
        plan = self.freeze()
        fixtures.dump(self.annotation_path, {"changed": True})
        with self.assertRaisesRegex(ValidationError, "review"):
            experiment.verify_plan(self.root, plan)
        fixtures.dump(self.annotation_path, self.annotations)
        changed = deepcopy(plan)
        changed["gripper_preparations"][self.clips[0]["clip_id"]]["seeds"][0]["frame_index"] = 1
        changed["freeze_hash"] = experiment.original._digest({k: v for k, v in changed.items() if k != "freeze_hash"})
        with self.assertRaisesRegex(ValidationError, "source"):
            experiment.verify_plan(self.root, changed, files=False)
        Path(self.clips[0]["frames"][0]["file"]).write_bytes(b"changed source")
        with self.assertRaises(ValidationError):
            experiment.verify_plan(self.root, plan)

    def test_outputs_cannot_be_nested_under_any_immutable_source_or_baseline(self):
        for directory in (self.old_dir, self.fresh_dir, self.contact_dir):
            with self.subTest(directory=directory), self.assertRaises(ValidationError):
                experiment.freeze(self.root, self.contact_path, self.annotation_path, directory / "responses")

    def test_public_plan_excludes_private_inventory_paths_and_preserves_exact_seed_geometry(self):
        self.freeze()
        public = experiment.public_plan(self.root, self.output / "freeze.json")
        encoded = json.dumps(public)
        for forbidden in (str(self.root), "clip_file", "annotations_file", "input_records", "author_review"):
            self.assertNotIn(forbidden, encoded)
        self.assertEqual(set(experiment.COHORTS), set(public["cohorts"]))
        self.assertEqual(self.annotations["clips"][0]["seeds"],
            public["cohorts"][experiment.COHORTS[0]]["sources"][0]["initial_gripper_seeds"])
        self.assertFalse(public["changes_live_placement_acceptance"])

    def test_veto_only_reduces_eligibility_and_never_changes_source_candidate_or_score(self):
        entry = self.entry()
        reports = {index: fixtures.ContactExperimentTests.checked(self, entry, index) for index in range(3)}
        runtime = experiment.masks._runtime(entry, self.raw(entry), reports)
        original = deepcopy(runtime)
        result, changes = experiment._veto(entry, runtime,
            self.assessment(entry, ["separated", "near_or_contact", "unavailable"]))
        self.assertEqual(original, runtime)
        self.assertEqual(original[0], result[0])
        self.assertEqual(2, len(changes))
        for index in (1, 2):
            before, after = original[index]["candidates"][0], result[index]["candidates"][0]
            self.assertFalse(after["fully_visible"])
            self.assertEqual("unknown", after["semantic"]["decision"])
            self.assertNotEqual(before["semantic"]["evidence_sha256"], after["semantic"]["evidence_sha256"])
            for key in ("track_id", "object_id", "box", "confidence", "mask_supported", "identity_ambiguous"):
                self.assertEqual(before[key], after[key])
            self.assertEqual(original[index]["frame_sha256"], result[index]["frame_sha256"])

    def test_geometry_separation_cannot_upgrade_existing_semantic_or_visibility_failure(self):
        entry = self.entry()
        reports = {index: fixtures.ContactExperimentTests.checked(self, entry, index) for index in range(3)}
        reports[0]["evaluation"].update(fully_visible=False, decision="unknown")
        reports[1]["evaluation"].update(identity_ambiguous=True, decision="contradicted")
        runtime = experiment.masks._runtime(entry, self.raw(entry), reports)
        self.assertEqual(runtime, experiment._veto(entry, runtime, self.assessment(entry))[0])

    def test_geometry_source_target_frame_inventory_or_relation_mismatch_is_rejected(self):
        entry = self.entry()
        runtime = experiment.masks._runtime(entry, self.raw(entry), {})
        changes = (
            lambda a: a["frames"].pop(),
            lambda a: a["frames"].append(deepcopy(a["frames"][0])),
            lambda a: a["frames"][0].update(source_sha256="0" * 64),
            lambda a: a["frames"][0].update(target_track_id=3),
            lambda a: a["frames"][0].update(relation="far_enough_probably"),
            lambda a: a["frames"][0].update(separation_supported=False),
        )
        for index, mutate in enumerate(changes):
            value = self.assessment(entry)
            mutate(value)
            with self.subTest(index=index), self.assertRaises(ValidationError):
                experiment._veto(entry, runtime, value)

    def test_current_policy_retains_dense_clear_views_but_contact_resets_confirmation(self):
        entry = self.entry()
        reports = {index: fixtures.ContactExperimentTests.checked(self, entry, index) for index in range(3)}
        self.override(experiment.masks, "_filter", side_effect=lambda runtime, *a: (runtime, {}))
        clear = self.assessment(entry)
        greedy, _, _ = experiment._policy_predictions(entry, self.raw(entry), reports,
            experiment.original.evaluate_tracking, {}, clear, geometry_completed=True)
        current, _, _ = experiment._policy_predictions(entry, self.raw(entry), reports,
            experiment.baseline.current.evaluate_tracking, {}, clear, geometry_completed=True)
        self.assertEqual(["unknown", "supported", "unknown"], [row["status"] for row in greedy])
        self.assertEqual(["unknown", "supported", "supported"], [row["status"] for row in current])
        contact = self.assessment(entry, ["separated", "near_or_contact", "separated"])
        rows, policy, _ = experiment._policy_predictions(entry, self.raw(entry), reports,
            experiment.baseline.current.evaluate_tracking, {}, contact, geometry_completed=True)
        self.assertEqual(["unknown"] * 3, [row["status"] for row in rows])
        self.assertTrue(policy["frames"][1]["confirmation_history_reset"])

    def test_failed_gripper_provider_or_original_guard_stays_unscored(self):
        entry = self.entry()
        reports = {index: fixtures.ContactExperimentTests.checked(self, entry, index) for index in range(3)}
        self.override(experiment.masks, "_filter", side_effect=lambda runtime, *a: (runtime, {}))
        predictions, _, _ = experiment._policy_predictions(entry, self.raw(entry), reports,
            experiment.baseline.current.evaluate_tracking, {}, self.assessment(entry), geometry_completed=False)
        self.assertTrue(all(row["status"] == "unscored" for row in predictions))
        reports[1].update(status="unscored", response=None, evaluation=None)
        predictions, _, _ = experiment._policy_predictions(entry, self.raw(entry), reports,
            experiment.baseline.current.evaluate_tracking, {}, self.assessment(entry), geometry_completed=True)
        self.assertEqual("unscored", predictions[1]["status"])

    def test_candidate_diagnostic_catches_contact_without_hiding_semantic_abstentions(self):
        entry = self.entry()
        reports = {index: fixtures.ContactExperimentTests.checked(self, entry, index) for index in range(3)}
        reports[2]["evaluation"].update(decision="unknown")
        predictions = experiment._candidate_predictions(entry, self.raw(entry), reports,
            self.assessment(entry, ["separated", "near_or_contact", "separated"]), geometry_completed=True)
        self.assertEqual(["supported", "unknown", "unknown"], [row["status"] for row in predictions])

    def mock_run(self, *, attempts=4, fail=False):
        self.freeze()
        self.override(experiment, "load_env_file", return_value=None)
        clients = self.override(experiment.original.RetainingClient, "from_env", return_value=object())
        def video(client, prepared):
            entry = next(row for row in self.clips if row["prepared"]["clip_file"] == prepared["clip_file"])
            result = self.raw(entry, status="unscored" if fail else "completed")
            result["prepared"] = deepcopy(prepared)
            return result
        self.override(experiment, "segment_video", side_effect=video)
        self.override(experiment, "public_tracking_summary", return_value={"status": "unscored" if fail else "completed",
            "cost": {"estimated_usd": None if fail else .01, "unreported_charge_possible": fail}})
        def compare(raw, new, **kwargs):
            entry = next(row for row in self.clips if row["prepared"] == raw["prepared"])
            return self.assessment(entry, ["unavailable"] * 3 if fail else None)
        self.override(experiment.geometry, "compare", side_effect=compare)
        self.override(experiment.distractors, "resolve_distractor_conflicts", return_value={})
        self.override(experiment.masks, "_filter", side_effect=lambda runtime, *a: (runtime, {}))
        journal = self.override(experiment, "AttemptJournal")
        journal.return_value.summary.return_value = {"total_reserved_attempts": attempts}
        return experiment.run(self.root, self.output / "freeze.json"), clients

    def test_mocked_freeze_run_replay_adds_only_four_sam_calls_and_keeps_full_baseline(self):
        result, clients = self.mock_run()
        self.assertEqual(4, clients.call_count)
        self.assertEqual((4, .04, 0), (result["new_provider_attempts"], result["estimated_incremental_usd"], result["new_Spark_requests"]))
        self.assertEqual(self.historical, result["historical_contact_result"])
        self.assertEqual(.431506, result["historical_estimated_incremental_usd"])
        self.assertEqual(1, result["historical_contact_result"]["accounting"]["uncertain"])
        for cohort in experiment.COHORTS:
            self.assertEqual(2, result["cohorts"][cohort]["new_provider_attempts"])
            self.assertEqual(set(experiment.baseline.METHODS) | set(experiment.METHODS), set(result["cohorts"][cohort]["methods"]))
            self.assertEqual(self.historical["cohorts"][cohort]["methods"], result["cohorts"][cohort]["historical_methods"])
        for _, kwargs in clients.call_args_list:
            self.assertEqual((1, 0), (kwargs["max_requests"], kwargs["retries"]))
        self.assertEqual(result, experiment.replay(self.root, self.output / "freeze.json"))
        self.assertNotIn(str(self.root), json.dumps(result))
        self.assertFalse(result["physical_action_authorized"])

    def test_failed_new_calls_retain_unscored_and_unknown_cost_without_retries(self):
        result, clients = self.mock_run(fail=True)
        self.assertEqual((4, 4), (clients.call_count, result["new_failed_operation_count"]))
        self.assertTrue(result["unreported_new_charge_possible"])
        for cohort in result["cohorts"].values():
            for method in experiment.METHODS:
                self.assertEqual(6, cohort["methods"][method]["metrics"]["labels_unscored"])

    def test_second_run_is_rejected_before_any_provider_client(self):
        _, clients = self.mock_run()
        with self.assertRaises(FileExistsError):
            experiment.run(self.root, self.output / "freeze.json")
        self.assertEqual(4, clients.call_count)

    def test_journal_count_or_response_preparation_mismatch_cannot_publish(self):
        with self.assertRaisesRegex(ValidationError, "journal"):
            self.mock_run(attempts=3)

    def test_replay_rejects_new_report_for_different_initial_review(self):
        self.mock_run()
        name = self.clips[0]["clip_id"]
        report_path = self.output / f"{name}-grippers.json"
        result = json.loads(report_path.read_text())
        result["prepared"]["prepared_hash"] = "0" * 64
        fixtures.dump(report_path, result)
        with self.assertRaisesRegex(ValidationError, "preparation"):
            experiment.replay(self.root, self.output / "freeze.json")


if __name__ == "__main__":
    unittest.main()
