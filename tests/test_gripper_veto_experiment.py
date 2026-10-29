"""Authored cached-retest fixtures; no real model, video or accuracy evidence."""
from copy import deepcopy
import json
from unittest.mock import patch
import unittest

from proofloop import gripper_veto_experiment as experiment
from proofloop.core import ValidationError
from tests import test_gripper_experiment as prepared
from tests import test_contact_experiment as fixtures


class GripperVetoExperimentTests(unittest.TestCase):
    override = prepared.GripperExperimentTests.override
    entry = prepared.GripperExperimentTests.entry
    raw = prepared.GripperExperimentTests.raw
    fake_prepare = prepared.GripperExperimentTests.fake_prepare
    assessment = prepared.GripperExperimentTests.assessment

    def setUp(self):
        prepared.GripperExperimentTests.setUp(self)
        self._set_exact_historical_scores()
        strict_result, self.provider_clients = prepared.GripperExperimentTests.mock_run(self)
        self.strict_result = strict_result
        self.strict_path = self.output / "freeze.json"
        self.output = self.root / "artifacts/local/negative-veto"
        self.provider_clients.reset_mock()
        self.transport = self.override(experiment.strict, "segment_video", side_effect=AssertionError("Cached retest cannot infer"))
        self.warning_call = self.override(experiment.warning, "compare", side_effect=self.warning_compare)

    def freeze(self):
        # The source fixture's mock_run invokes this before strict_path exists.
        if not hasattr(self, "strict_path"):
            return prepared.GripperExperimentTests.freeze(self)
        return experiment.freeze(self.root, self.strict_path, self.output)

    def _set_exact_historical_scores(self):
        for cohort, source in zip(experiment.COHORTS, (self.old_plan, self.fresh_plan)):
            references = [row for entry in source["clips"] for row in experiment.original._reference(entry)]
            predictions = {name: [] for name in experiment.strict.baseline.METHODS}
            for entry in source["clips"]:
                raw = self.raw(entry)
                reports = {index: fixtures.ContactExperimentTests.checked(self, entry, index) for index in range(3)}
                for cp in entry["checkpoints"]:
                    index = cp["frame_index"]
                    predictions["sam_presence"].append(experiment.strict.baseline.previous._prediction(
                        entry, raw, {}, index, presence_only=True))
                    for version in ("v2", "v3"):
                        predictions[f"visibility_{version}"].append(experiment.strict.baseline.previous._prediction(
                            entry, raw, reports[index], index, visibility_only=True))
                        predictions[f"candidate_{version}"].append(experiment.strict.baseline.previous._prediction(
                            entry, raw, reports[index], index))
                with patch.object(experiment.strict.masks, "_filter", side_effect=lambda runtime, *a: (runtime, {})):
                    for label, evaluator in (("greedy", experiment.original.evaluate_tracking),
                                             ("current", experiment.strict.baseline.current.evaluate_tracking)):
                        rows, _, _ = experiment.strict.baseline._policy_predictions(entry, raw, reports, evaluator, {})
                        for version in ("v2", "v3"):
                            predictions[f"{version}_{label}"].extend(rows)
            self.historical["cohorts"][cohort]["methods"] = {
                name: experiment.strict.baseline.score_frames(references, predictions[name], iou_threshold=.3)
                for name in predictions}
        fixtures.dump(self.contact_dir / "public-result.json", self.historical)

    def warnings(self, entry=None, flags=None, *, available=True):
        entry = entry or self.entry()
        flags = flags or [False] * 3
        return {"method_version": experiment.warning.VERSION, "config": deepcopy(experiment.warning.CONFIG),
            "source_sha256": entry["prepared"]["source"]["sha256"], "frames": [
                {"frame_index": index, "timestamp_ms": entry["frames"][index]["timestamp_ms"],
                 "source_sha256": entry["prepared"]["source"]["sha256"], "target_track_id": 1,
                 "warning": "near_gripper_proposal" if flag else "no_near_proposal_observed" if available else "geometry_unavailable",
                 "veto": flag, "additional_geometry_complete": available, "comparisons": [],
                 "positive_support_can_be_granted": False, "full_visibility_verified": False}
                for index, flag in enumerate(flags)]}

    def warning_compare(self, raw, grippers, **kwargs):
        entry = next(row for row in self.clips if row["prepared"] == raw["prepared"])
        return self.warnings(entry)

    def reports(self, entry=None):
        entry = entry or self.entry()
        return {index: fixtures.ContactExperimentTests.checked(self, entry, index) for index in range(3)}

    def runtime(self, entry=None, reports=None):
        entry = entry or self.entry()
        return experiment.strict.masks._runtime(entry, self.raw(entry), reports or self.reports(entry))

    def test_prospective_receipt_declares_post_inspection_zero_call_warning_only_protocol(self):
        plan = self.freeze()
        self.assertEqual(experiment.EVALUATION_ROLE, plan["evaluation_role"])
        self.assertTrue(plan["selected_after_inspecting_strict_failure"])
        self.assertFalse(plan["derivation_started"])
        self.assertEqual(0, plan["added_provider_requests"])
        self.assertEqual("warning_only_not_qualification", plan["config"]["trust_mode"])
        self.assertIn("gripper_veto.py", plan["implementation_sha256"])
        self.assertEqual(experiment.original._digest(self.strict_result), plan["strict_public_sha256"])
        self.assertFalse((self.output / "derivation-started.json").exists())
        self.provider_clients.assert_not_called()
        self.transport.assert_not_called()
        self.warning_call.assert_not_called()

    def test_receipt_configuration_or_code_changes_cannot_be_rehashed_into_validity(self):
        plan = self.freeze()
        for field, value in (("added_provider_requests", 1), ("config", {}), ("implementation_sha256", {}),
                ("warning_config", {}), ("evaluation_role", "held_out"), ("selected_after_inspecting_strict_failure", False),
                ("derivation_started", True)):
            changed = deepcopy(plan)
            changed[field] = value
            changed["freeze_hash"] = experiment.original._digest({k: v for k, v in changed.items() if k != "freeze_hash"})
            with self.subTest(field=field), self.assertRaises(ValidationError):
                experiment.verify_plan(self.root, changed)

    def test_changed_strict_public_or_retained_raw_bytes_fail_before_derivation(self):
        plan = self.freeze()
        target = self.strict_path.parent / f"{self.clips[0]['clip_id']}-grippers.json"
        target.write_text('{"changed":true}\n')
        with self.assertRaises(ValidationError):
            experiment.verify_plan(self.root, plan)

    def test_public_reference_is_byte_hashed_and_must_match_canonical_result(self):
        published = self.root / "artifacts/feasibility/gripper-results.json"
        published.parent.mkdir(parents=True)
        published.write_bytes((self.strict_path.parent / "public-result.json").read_bytes())
        plan = experiment.freeze(self.root, self.strict_path, self.output, baseline_public=published)
        self.assertEqual("artifacts/feasibility/gripper-results.json", plan["strict_public_reference"])
        self.assertEqual(fixtures.sha(published), plan["strict_public_file_sha256"])
        published.write_text('{"changed":true}\n')
        with self.assertRaises(ValidationError):
            experiment.verify_plan(self.root, plan)

    def test_private_or_wrong_public_reference_is_rejected(self):
        for value in (self.contact_dir / "public-result.json", self.root / "artifacts/feasibility/wrong.json"):
            if not value.exists():
                value.parent.mkdir(parents=True, exist_ok=True)
                value.write_text("{}\n")
            with self.subTest(value=value), self.assertRaises(ValidationError):
                experiment.freeze(self.root, self.strict_path, self.output, baseline_public=value)

    def test_public_plan_omits_private_paths_inventory_and_provider_payloads(self):
        self.freeze()
        public = experiment.public_plan(self.root, self.output / "freeze.json")
        encoded = json.dumps(public)
        for forbidden in (str(self.root), "input_records", "clip_file", "author_review", "output_text"):
            self.assertNotIn(forbidden, encoded)
        self.assertEqual(0, public["added_provider_requests"])
        self.assertFalse(public["changes_live_placement_acceptance"])
        self.warning_call.assert_not_called()

    def test_far_or_unavailable_geometry_never_changes_any_baseline_runtime_field(self):
        runtime = self.runtime()
        for available in (True, False):
            with self.subTest(available=available):
                modified, _, changes = experiment._runtime_veto(self.entry(), runtime, self.warnings(available=available))
                self.assertEqual(runtime, modified)
                self.assertEqual([], changes)

    def test_negative_risk_only_changes_visibility_semantic_decision_and_derived_evidence_hash(self):
        entry, runtime = self.entry(), self.runtime()
        initial = deepcopy(runtime)
        modified, _, changes = experiment._runtime_veto(entry, runtime, self.warnings(entry, [False, True, False]))
        self.assertEqual(initial, runtime)
        self.assertEqual(initial[0], modified[0])
        self.assertEqual(initial[2], modified[2])
        before, after = initial[1]["candidates"][0], modified[1]["candidates"][0]
        self.assertFalse(after["fully_visible"])
        self.assertEqual("unknown", after["semantic"]["decision"])
        self.assertNotEqual(before["semantic"]["evidence_sha256"], after["semantic"]["evidence_sha256"])
        for field in ("track_id", "object_id", "box", "confidence", "mask_supported", "identity_ambiguous"):
            self.assertEqual(before[field], after[field])
        self.assertFalse(changes[0]["physical_occlusion_asserted"])

    def test_source_frame_target_or_warning_semantics_mismatch_is_rejected(self):
        mutations = (
            lambda v: v["frames"].pop(),
            lambda v: v["frames"].append(deepcopy(v["frames"][0])),
            lambda v: v["frames"][0].update(source_sha256="0" * 64),
            lambda v: v["frames"][0].update(target_track_id=3),
            lambda v: v["frames"][0].update(warning="separated"),
            lambda v: v["frames"][0].update(veto=True),
        )
        for index, mutate in enumerate(mutations):
            assessed = self.warnings()
            mutate(assessed)
            with self.subTest(index=index), self.assertRaises(ValidationError):
                experiment._runtime_veto(self.entry(), self.runtime(), assessed)

    def test_negative_risk_resets_history_and_does_not_support_next_clear_frame_prematurely(self):
        entry = self.entry()
        rows, original, policy, _ = experiment._policies(entry, self.raw(entry), self.reports(entry),
            experiment.strict.baseline.current.evaluate_tracking, {}, self.warnings(entry, [False, True, False]))
        self.assertEqual(["unknown", "supported", "supported"], [row["status"] for row in original])
        self.assertEqual(["unknown"] * 3, [row["status"] for row in rows])
        self.assertTrue(policy["frames"][1]["confirmation_history_reset"])
        self.assertEqual(0, policy["metrics"]["supported_frames"])

    def test_far_and_unavailable_geometry_cannot_upgrade_semantic_unknown_or_provider_failure(self):
        entry, reports = self.entry(), self.reports()
        reports[1]["evaluation"].update(fully_visible=False, decision="unknown")
        reports[2].update(status="unscored", response=None, evaluation=None)
        for available in (True, False):
            rows, original, _, _ = experiment._policies(entry, self.raw(entry), reports,
                experiment.strict.baseline.current.evaluate_tracking, {}, self.warnings(entry, available=available))
            self.assertEqual(original, rows)
            self.assertEqual(["unknown", "unknown", "unscored"], [row["status"] for row in rows])

    def test_subset_suppression_updates_actual_policy_rows_status_and_metrics_together(self):
        entry, runtime = self.entry(), self.runtime()
        evaluator = experiment.strict.baseline.current.evaluate_tracking
        policy = experiment.strict.baseline._policy(entry, runtime, evaluator)
        base = deepcopy(policy)
        base["frames"][-1]["status"] = "unknown"
        result = experiment._intersect_policy(policy, base, runtime,
            experiment._warning_rows(entry, runtime, self.warnings(entry)))
        self.assertEqual("unknown", result["status"])
        self.assertEqual("unknown", result["frames"][-1]["status"])
        self.assertEqual(1, result["metrics"]["supported_frames"])
        self.assertEqual(2, result["metrics"]["unknown_frames"])
        self.assertEqual(1, result["baseline_subset_suppressed_frames"])
        self.assertIn("unchanged_baseline_positive_subset_required", result["frames"][-1]["gap_codes"])

    def test_real_greedy_retiming_after_negative_gap_cannot_create_new_support(self):
        entry = self.entry()
        entry["frames"].append({**deepcopy(entry["frames"][-1]), "frame_index": 3, "sha256": "9" * 64})
        entry["checkpoints"].append({**deepcopy(entry["checkpoints"][-1]), "frame_index": 3})
        for frame, timestamp in zip(entry["frames"], (0, 400, 600, 1000)):
            frame["timestamp_ms"] = timestamp
        raw = self.raw(entry)
        raw["frames"].append({**deepcopy(raw["frames"][-1]), "frame_index": 3, "timestamp_ms": 1000})
        reports = {index: fixtures.ContactExperimentTests.checked(self, entry, index) for index in range(4)}
        runtime = experiment.strict.masks._runtime(entry, raw, reports)
        assessed = self.warnings(entry, [True, False, False, False])
        candidate_runtime, _, _ = experiment._runtime_veto(entry, runtime, assessed)
        unchanged = experiment.strict.baseline._policy(entry, runtime, experiment.original.evaluate_tracking)
        shifted = experiment.strict.baseline._policy(entry, candidate_runtime, experiment.original.evaluate_tracking)
        self.assertEqual("unknown", unchanged["frames"][-1]["status"])
        self.assertEqual("supported", shifted["frames"][-1]["status"])
        rows, _, policy, _ = experiment._policies(entry, raw, reports, experiment.original.evaluate_tracking, {}, assessed)
        self.assertEqual("unknown", rows[-1]["status"])
        self.assertTrue(policy["frames"][-1]["baseline_subset_suppressed"])
        self.assertEqual(0, policy["metrics"]["supported_frames"])
        self.assertEqual(4, policy["metrics"]["unknown_frames"])

    def test_supported_current_chain_rejects_risk_reused_pixels_or_changed_binding(self):
        entry, runtime = self.entry(), self.runtime()
        policy = experiment.strict.baseline._policy(entry, runtime, experiment.strict.baseline.current.evaluate_tracking)
        row = policy["frames"][-1]
        warnings = experiment._warning_rows(entry, runtime, self.warnings(entry))
        experiment._validate_current_chain(row, runtime, warnings, target_id=entry["contract"]["target_id"])
        risky = deepcopy(warnings)
        risky[0]["veto"] = True
        with self.assertRaisesRegex(ValidationError, "risky"):
            experiment._validate_current_chain(row, runtime, risky, target_id=entry["contract"]["target_id"])
        for key, value in (("semantic_evidence_sha256", "0" * 64), ("timestamp_ms", 999), ("track_id", 7)):
            changed = deepcopy(row)
            changed["confirmation_evidence"][0][key] = value
            with self.subTest(key=key), self.assertRaises(ValidationError):
                experiment._validate_current_chain(changed, runtime, warnings, target_id=entry["contract"]["target_id"])

    def test_intersection_rejects_one_missing_policy_and_accepts_only_both_missing(self):
        entry, runtime = self.entry(), self.runtime()
        policy = experiment.strict.baseline._policy(entry, runtime, experiment.strict.baseline.current.evaluate_tracking)
        warnings = experiment._warning_rows(entry, runtime, self.warnings(entry))
        self.assertIsNone(experiment._intersect_policy(None, None, runtime, warnings))
        for candidate, unchanged in ((policy, None), (None, policy)):
            with self.subTest(candidate_missing=candidate is None), self.assertRaisesRegex(ValidationError, "both"):
                experiment._intersect_policy(candidate, unchanged, runtime, warnings)

    def test_current_chain_permits_nullable_unassessed_intermediate_observation(self):
        entry, runtime = self.entry(), self.runtime()
        runtime[1]["candidates"][0].update(fully_visible=None, identity_ambiguous=None, semantic=None)
        policy = experiment.strict.baseline._policy(entry, runtime, experiment.strict.baseline.current.evaluate_tracking)
        row = policy["frames"][-1]
        self.assertEqual([0, 2], [item["frame_index"] for item in row["confirmation_evidence"]])
        self.assertEqual("supported", row["status"])
        warnings = experiment._warning_rows(entry, runtime, self.warnings(entry))
        experiment._validate_current_chain(row, runtime, warnings, target_id=entry["contract"]["target_id"])

    def test_current_chain_rejects_risk_on_intermediate_nonproof_frame(self):
        entry, runtime = self.entry(), self.runtime()
        runtime[1]["candidates"][0].update(fully_visible=None, identity_ambiguous=None, semantic=None)
        policy = experiment.strict.baseline._policy(entry, runtime, experiment.strict.baseline.current.evaluate_tracking)
        row = policy["frames"][-1]
        self.assertEqual([0, 2], [item["frame_index"] for item in row["confirmation_evidence"]])
        warnings = experiment._warning_rows(entry, runtime, self.warnings(entry, [False, True, False]))
        with self.assertRaisesRegex(ValidationError, "intervening"):
            experiment._validate_current_chain(row, runtime, warnings, target_id=entry["contract"]["target_id"])

    def test_current_chain_rejects_intermediate_original_hard_gates_and_track_change_back(self):
        entry, initial = self.entry(), self.runtime()
        initial[1]["candidates"][0].update(fully_visible=None, identity_ambiguous=None, semantic=None)
        policy = experiment.strict.baseline._policy(entry, initial, experiment.strict.baseline.current.evaluate_tracking)
        row = policy["frames"][-1]
        warnings = experiment._warning_rows(entry, initial, self.warnings(entry))
        semantic_gap = deepcopy(self.runtime()[1]["candidates"][0]["semantic"])
        semantic_gap["decision"] = "unknown"
        mutations = (
            lambda f: f.update(candidates=[]),
            lambda f: f["candidates"][0].update(confidence=None),
            lambda f: f["candidates"][0].update(confidence=.79),
            lambda f: f["candidates"][0].update(mask_supported=False),
            lambda f: f["candidates"][0].update(fully_visible=False),
            lambda f: f["candidates"][0].update(identity_ambiguous=True),
            lambda f: f["candidates"][0].update(track_id=7),
            lambda f: f["candidates"][0].update(semantic=semantic_gap),
            lambda f: f["candidates"].append({**deepcopy(f["candidates"][0]), "track_id": 7}),
            lambda f: f["candidates"][0].update(box=[0, .1, .5, .5]),
            lambda f: f.update(reviewed_distractors=[{"track_id": 2, "box": [.2, .2, .6, .6], "origin": "runtime_sam_track"}]),
        )
        for index, mutate in enumerate(mutations):
            runtime = deepcopy(initial)
            mutate(runtime[1])
            with self.subTest(index=index), self.assertRaisesRegex(ValidationError, "intervening"):
                experiment._validate_current_chain(row, runtime, warnings, target_id=entry["contract"]["target_id"])

    def test_current_chain_binds_published_track_target_and_source_to_all_proofs(self):
        entry, runtime = self.entry(), self.runtime()
        row = experiment.strict.baseline._policy(entry, runtime,
            experiment.strict.baseline.current.evaluate_tracking)["frames"][-1]
        warnings = experiment._warning_rows(entry, runtime, self.warnings(entry))
        for field, value in (("track_id", 7), ("source_sha256", "0" * 64), ("frame_sha256", "1" * 64),
                ("timestamp_ms", 1100), ("semantic_evidence_sha256", "2" * 64), ("box", [.2, .2, .5, .5])):
            changed = deepcopy(row)
            changed[field] = value
            with self.subTest(field=field), self.assertRaises(ValidationError):
                experiment._validate_current_chain(changed, runtime, warnings, target_id=entry["contract"]["target_id"])
        with self.assertRaises(ValidationError):
            experiment._validate_current_chain(row, runtime, warnings, target_id="different_target")

    def test_exact_box_guard_evidence_or_source_difference_blocks_new_support(self):
        entry, runtime = self.entry(), self.runtime()
        policy = experiment.strict.baseline._policy(entry, runtime, experiment.strict.baseline.current.evaluate_tracking)
        warnings = experiment._warning_rows(entry, runtime, self.warnings(entry))
        for field, value in (("box", [.2, .2, .5, .5]), ("semantic_evidence_sha256", "0" * 64),
                             ("frame_sha256", "1" * 64), ("timestamp_ms", 1100)):
            changed = deepcopy(policy)
            changed["frames"][-1][field] = value
            result = experiment._intersect_policy(changed, policy, runtime, warnings)
            with self.subTest(field=field):
                self.assertEqual("unknown", result["frames"][-1]["status"])

    def test_candidate_warning_only_removes_existing_support_and_preserves_unscored(self):
        entry, reports = self.entry(), self.reports()
        reports[2].update(status="unscored", response=None, evaluation=None)
        rows = experiment._candidate_predictions(entry, self.raw(entry), reports, self.warnings(entry, [False, True, True]))
        self.assertEqual(["supported", "unknown", "unscored"], [row["status"] for row in rows])

    def test_mocked_run_replay_is_zero_api_and_retains_historical_failures_costs_and_all_methods(self):
        self.freeze()
        result = experiment.run(self.root, self.output / "freeze.json")
        self.assertEqual((0, 0.0), (result["added_provider_requests"], result["estimated_incremental_usd"]))
        self.assertEqual(self.strict_result["contact_public_sha256"], result["historical_contact_public_sha256"])
        self.assertEqual({"contact_trial": 1, "strict_gripper_trial": 0}, result["historical_failed_operation_counts"])
        self.assertEqual(.431506, result["historical_estimated_incremental_usd"]["contact_trial"])
        self.assertEqual(1, result["historical_accounting"]["contact_trial"]["uncertain"])
        for cohort in experiment.COHORTS:
            self.assertEqual(self.strict_result["cohorts"][cohort]["methods"], result["cohorts"][cohort]["historical_methods"])
            self.assertEqual(set(self.strict_result["cohorts"][cohort]["methods"]) | set(experiment.METHODS),
                set(result["cohorts"][cohort]["methods"]))
        self.assertNotIn("historical_contact_result", result)
        self.assertNotIn("historical_strict_gripper_result", result)
        self.assertEqual(result, experiment.replay(self.root, self.output / "freeze.json"))
        self.assertNotIn(str(self.root), json.dumps(result))
        self.provider_clients.assert_not_called()
        self.transport.assert_not_called()

    def test_full_replay_rejects_recomputed_contact_baseline_score_mismatch(self):
        self.freeze()
        self.strict_result["historical_contact_result"]["cohorts"][experiment.COHORTS[0]]["methods"]["v2_current"] = {}
        # Exercise the independent unchanged-score comparison after receipt validation.
        actual = experiment.verify_plan
        def verified(root, plan):
            values = list(actual(root, plan))
            values[1] = deepcopy(self.strict_result)
            return tuple(values)
        with patch.object(experiment, "verify_plan", side_effect=verified), self.assertRaisesRegex(ValidationError, "baseline scores"):
            experiment.run(self.root, self.output / "freeze.json")

    def test_derivation_requires_exclusive_zero_call_marker(self):
        self.freeze()
        with self.assertRaises(ValidationError):
            experiment.replay(self.root, self.output / "freeze.json")
        experiment.run(self.root, self.output / "freeze.json")
        with self.assertRaises(FileExistsError):
            experiment.run(self.root, self.output / "freeze.json")
        self.provider_clients.assert_not_called()


if __name__ == "__main__":
    unittest.main()
