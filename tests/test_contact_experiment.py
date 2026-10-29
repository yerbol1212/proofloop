"""Authored orchestration fixtures; never provider or video-accuracy evidence."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from proofloop import contact_experiment as experiment
from proofloop.core import ValidationError
from proofloop.meta import MetaClient, SPARK_MODEL
from tests.test_visibility_guard import clear as native_clear, contract


def dump(path, value):
    Path(path).write_text(json.dumps(value, sort_keys=True) + "\n")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class ContactExperimentTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.old_dir, self.previous_dir, self.fresh_dir, self.output = [
            self.root / "artifacts/local" / name for name in ("old-source", "previous", "fresh-source", "contact")]
        for directory in (self.old_dir, self.previous_dir, self.fresh_dir):
            directory.mkdir(parents=True)
        self.old_path, self.previous_path, self.fresh_path = [
            directory / "freeze.json" for directory in (self.old_dir, self.previous_dir, self.fresh_dir)]
        self.old_plan = self.source_plan(self.old_dir, "old", 2)
        self.fresh_plan = self.source_plan(self.fresh_dir, "fresh", 3)
        dump(self.old_path, self.old_plan)
        dump(self.fresh_path, self.fresh_plan)
        references = [row for entry in self.old_plan["clips"] for row in experiment.original._reference(entry)]
        predictions = [{"clip_id": row["clip_id"], "frame_index": row["frame_index"],
                        "status": "unknown", "box": None} for row in references]
        score = experiment.score_frames(references, predictions, iou_threshold=.3)
        self.previous_result = {"authored_previous_result": True, "cohorts": {"new_recordings_same_families": {
            "methods": {name: deepcopy(score) for name in
                        ("visibility_v2", "candidate_guard_v2", "mask_policy_v2", "sam_presence")}}}}
        dump(self.previous_dir / "public-result.json", self.previous_result)
        self.previous_plan = {"freeze_hash": "a" * 64,
            "fresh_frozen_file": self.old_path.relative_to(self.root).as_posix(),
            "input_records": experiment.previous._fresh_records(self.root, self.old_path, self.old_plan)}
        dump(self.previous_path, self.previous_plan)
        self.source_verification = self.override(experiment.original, "verify_plan", return_value=None)
        self.previous_replay = self.override(experiment.previous, "replay",
            side_effect=lambda *a, **k: deepcopy(self.previous_result))

    def override(self, owner, name, **kwargs):
        handle = patch.object(owner, name, **kwargs)
        mocked = handle.start()
        self.addCleanup(handle.stop)
        return mocked

    def source_plan(self, directory, prefix, episode):
        annotation = directory / "annotations.json"
        dump(annotation, {"role": "authored_scoring_only"})
        clips = []
        for number, family in enumerate(("lerobot/aloha_static_screw_driver", "lerobot/aloha_static_battery")):
            name = f"{prefix}-{number}"
            original, prepared = directory / f"{name}-original.mp4", directory / f"{name}-prepared.mp4"
            original.write_bytes(f"authored original {name}".encode())
            prepared.write_bytes(f"authored derivative {name}".encode())
            frames = []
            for index in range(3):
                path = directory / f"{name}-{index}.jpg"
                path.write_bytes(f"authored source image {name} {index}".encode())
                frames.append({"frame_index": index, "timestamp_ms": (0, 600, 1000)[index],
                               "file": str(path), "sha256": sha(path)})
            clips.append({"clip_id": name, "source_attribution": {"dataset": family, "episode": episode,
                    "source_role": "authored_test_fixture"}, "original_file": str(original),
                "original_sha256": sha(original), "contract": contract(),
                "prepared": {"clip_file": str(prepared), "prepared_hash": sha(prepared),
                    "source": {"sha256": sha(prepared), "width": 100, "height": 100}},
                "frames": frames, "seeds": [{"track_id": 1}, {"track_id": 2}],
                "derivation": {"mode": "authored_fixture"}, "checkpoints": [
                    {"frame_index": index, "expected": "supported", "target_box": [.1, .1, .5, .5],
                     "distractor_boxes": [], "condition": "clear"} for index in range(3)]})
        return {"freeze_hash": hashlib.sha256(prefix.encode()).hexdigest(),
                "annotations_file": str(annotation), "clips": clips}

    def freeze(self):
        return experiment.freeze(self.root, self.previous_path, self.fresh_path, self.output)

    def entry(self):
        return deepcopy(self.fresh_plan["clips"][0])

    def raw(self, entry=None, *, candidates=True):
        entry = entry or self.entry()
        return {"status": "completed", "prepared": deepcopy(entry["prepared"]), "attempts": [{"status": "ok"}],
            "frames": [{"frame_index": index, "timestamp_ms": entry["frames"][index]["timestamp_ms"], "candidates": [
                {"track_id": 1, "object_id": "remote", "box": [.1, .1, .5, .5], "confidence": .95,
                 "mask_supported": True}] if candidates else []} for index in range(3)]}

    def checked(self, entry=None, index=0):
        entry = entry or self.entry()
        prepared = {"contract": deepcopy(entry["contract"]), "source_sha256": entry["prepared"]["source"]["sha256"],
            "frame_index": index, "track_id": 1, "frame_sha256": entry["frames"][index]["sha256"],
            "source_file": entry["frames"][index]["file"], "crop_file": "/private/never-public/crop.jpg",
            "enlargement_file": "/private/never-public/enlarged.png", "crop_sha256": "c" * 64,
            "enlargement_sha256": "d" * 64, "prepared_hash": "e" * 64, "native_preparation_hash": "f" * 64,
            "candidate_box": {"x1": 10, "y1": 10, "x2": 49, "y2": 49}}
        parsed = native_clear()
        parsed["boundary_relations"] = {side: {"relation": "clear_background_gap", "neighbor": "gripper",
            "observation": "Direct dark background gap."} for side in experiment.contact.SIDES}
        text = json.dumps(parsed)
        raw_response = {"model": SPARK_MODEL, "status": "completed", "usage": {"input_tokens": 91, "output_tokens": 83},
            "output": [{"type": "message", "content": [{"type": "output_text", "text": text}]}]}
        ledger = {"endpoint": "responses", "attempt": 1, "model": SPARK_MODEL,
                  "status": "ok", "usage": raw_response["usage"]}
        client = MetaClient("authored-offline-key", max_requests=1, retries=0)
        client.ledger = [ledger]
        return {"method_version": experiment.contact.VERSION, "status": "completed", "prepared": prepared,
            "response": {"text": text, "parsed": parsed, "response": raw_response,
                "sources": [{"sha256": prepared[key]} for key in ("frame_sha256", "crop_sha256", "enlargement_sha256")]},
            "evaluation": experiment.contact.evaluate(parsed, entry["contract"]), "error": None,
            "usage": client.usage_summary()}

    def test_prospective_freeze_hashes_every_implementation_and_keeps_source_cohorts(self):
        plan = self.freeze()
        self.assertFalse(plan["inference_started"])
        self.assertFalse((self.output / "run-started.json").exists())
        self.assertEqual(self.previous_plan["freeze_hash"], plan["previous_freeze_hash"])
        self.assertEqual(self.old_plan["freeze_hash"], plan["old_source_freeze_hash"])
        self.assertEqual(self.fresh_plan["freeze_hash"], plan["fresh_source_freeze_hash"])
        self.assertEqual(experiment._implementations(), plan["implementation_sha256"])
        self.assertEqual(0o600, (self.output / "freeze.json").stat().st_mode & 0o777)

    def test_configuration_and_code_mutations_fail_even_with_rehashed_receipt(self):
        plan = self.freeze()
        changes = {"config": {"min_spacing_ms": 0}, "methods": [], "implementation_sha256": {},
            "contact_config": {}, "native_config": {}, "current_policy_version": "other", "inference_started": True}
        for field, value in changes.items():
            changed = deepcopy(plan)
            changed[field] = value
            changed["freeze_hash"] = experiment.original._digest({k: v for k, v in changed.items() if k != "freeze_hash"})
            with self.subTest(field=field), self.assertRaises(ValidationError):
                experiment.verify_plan(self.root, changed, files=False)

    def test_changed_fresh_source_bytes_and_historical_evidence_are_rejected(self):
        plan = self.freeze()
        frame = Path(self.entry()["frames"][0]["file"])
        original_bytes = frame.read_bytes()
        frame.write_bytes(b"changed")
        with self.assertRaises(ValidationError):
            experiment.verify_plan(self.root, plan)
        frame.write_bytes(original_bytes)
        self.previous_result["authored_previous_result"] = False
        with self.assertRaisesRegex(ValidationError, "Historical"):
            experiment.verify_plan(self.root, plan)

    def test_episode_contract_source_byte_and_identifier_reuse_are_rejected(self):
        baseline = deepcopy(self.fresh_plan)
        mutations = (
            lambda e: e["source_attribution"].update(episode=2),
            lambda e: e["source_attribution"].update(episode=4),
            lambda e: e.update(original_sha256=self.old_plan["clips"][0]["original_sha256"]),
            lambda e: e["prepared"]["source"].update(sha256=self.old_plan["clips"][0]["prepared"]["source"]["sha256"]),
            lambda e: e["contract"].update(concept="different target"),
            lambda e: e.update(clip_id=self.old_plan["clips"][0]["clip_id"]),
        )
        for index, mutate in enumerate(mutations):
            changed = deepcopy(baseline)
            mutate(changed["clips"][0])
            dump(self.fresh_path, changed)
            with self.subTest(index=index), self.assertRaises(ValidationError):
                self.freeze()

    def test_previous_public_result_must_match_replay_at_admission(self):
        dump(self.previous_dir / "public-result.json", {"changed": True})
        with self.assertRaisesRegex(ValidationError, "published"):
            self.freeze()

    def test_output_cannot_overwrite_historical_or_source_only_directories(self):
        for directory in (self.previous_dir, self.fresh_dir):
            with self.subTest(directory=directory), self.assertRaises(ValidationError):
                experiment.freeze(self.root, self.previous_path, self.fresh_path, directory / "responses")

    def test_public_plan_excludes_private_source_paths_and_retains_frozen_references(self):
        self.freeze()
        public = experiment.public_plan(self.root, self.output / "freeze.json")
        encoded = json.dumps(public)
        for forbidden in (str(self.root), "clip_file", "annotations_file", "input_records"):
            self.assertNotIn(forbidden, encoded)
        self.assertEqual(set(experiment.COHORTS), set(public["cohorts"]))
        self.assertEqual(6, len(public["cohorts"][experiment.COHORTS[1]]["references"]))
        self.assertFalse(public["changes_live_placement_acceptance"])

    def test_missing_candidate_never_prepares_or_creates_provider_client(self):
        client, module = Mock(side_effect=AssertionError("No API without a candidate")), Mock()
        result = experiment.previous._assess(client, self.entry(), self.raw(candidates=False), 0, module, self.output)
        self.assertEqual(experiment.previous._no_candidate(), result)
        client.assert_not_called()
        module.prepare.assert_not_called()
        module.challenge.assert_not_called()

    def test_assessment_uses_actual_candidate_never_reference_geometry(self):
        entry, module = self.entry(), Mock()
        entry["checkpoints"][0]["target_box"] = [.7, .7, .9, .9]
        module.prepare.return_value = {"prepared": True}
        experiment.previous._assess(lambda: "offline-client", entry, self.raw(entry), 0, module, self.output)
        args, kwargs = module.prepare.call_args
        self.assertEqual(entry["frames"][0]["file"], args[0])
        self.assertEqual({"x1": 10, "y1": 10, "x2": 49, "y2": 49}, args[1])
        self.assertEqual(entry["contract"], args[2])
        self.assertEqual(entry["prepared"]["source"]["sha256"], kwargs["source_sha256"])
        module.challenge.assert_called_once_with("offline-client", module.prepare.return_value)

    def test_contact_candidate_source_track_frame_and_contract_binding(self):
        self.override(experiment.contact, "verify", return_value=None)
        checked = self.checked()
        changes = {"candidate_box": {"x1": 11, "y1": 10, "x2": 49, "y2": 49}, "source_sha256": "0" * 64,
            "frame_sha256": "1" * 64, "source_file": "/private/other.jpg", "track_id": 2, "frame_index": 1,
            "contract": {"target_id": "other"}}
        for field, value in changes.items():
            changed = deepcopy(checked)
            changed["prepared"][field] = value
            with self.subTest(field=field), self.assertRaises(ValidationError):
                experiment._verify_contact(changed, self.entry(), self.raw(), 0)

    def test_contact_replay_requires_registered_order_of_all_three_images(self):
        self.override(experiment.contact, "verify", return_value=None)
        valid = self.checked()
        self.assertEqual(1, experiment._verify_contact(valid, self.entry(), self.raw(), 0)[1])
        for order in ((1, 0, 2), (0, 2, 1), (0, 1), (0, 1, 2, 2)):
            checked = deepcopy(valid)
            checked["response"]["sources"] = [valid["response"]["sources"][i] for i in order]
            with self.subTest(order=order), self.assertRaises(ValidationError):
                experiment._verify_contact(checked, self.entry(), self.raw(), 0)

    def test_safe_contact_summary_excludes_paths_provider_payload_and_free_text(self):
        self.override(experiment.contact, "verify", return_value=None)
        summary, count, cost = experiment._verify_contact(self.checked(), self.entry(), self.raw(), 0)
        encoded = json.dumps(summary)
        self.assertNotIn(str(self.root), encoded)
        self.assertNotIn("/private/", encoded)
        self.assertNotIn("Direct dark background gap", encoded)
        self.assertNotIn("output_text", encoded)
        self.assertEqual(1, count)
        self.assertGreater(cost, 0)
        self.assertFalse(summary["physical_action_authorized"])

    def test_missing_candidate_requires_exact_zero_attempt_record(self):
        self.assertEqual((None, 0, 0.0), experiment._verify_contact(experiment.previous._no_candidate(),
            self.entry(), self.raw(candidates=False), 0))
        with self.assertRaises(ValidationError):
            experiment._verify_contact(self.checked(), self.entry(), self.raw(candidates=False), 0)

    def test_network_none_usage_and_incomplete_reported_usage_remain_unscored(self):
        self.override(experiment.contact, "verify", return_value=None)
        for status, response, tokens in (("network_error", None, None), ("invalid_response",
                {"model": SPARK_MODEL, "status": "incomplete", "usage": {"input_tokens": 7, "output_tokens": 4096}},
                {"input_tokens": 7, "output_tokens": 4096})):
            checked = self.checked()
            ledger = {"endpoint": "responses", "attempt": 1, "model": SPARK_MODEL, "status": status}
            if tokens is not None:
                ledger["usage"] = tokens
            client = MetaClient("offline-no-provider", max_requests=1, retries=0)
            client.ledger = [ledger]
            checked.update(status="unscored", response=None, evaluation=None,
                error={"code": status, "response": response}, usage=client.usage_summary())
            with self.subTest(status=status):
                summary, count, cost = experiment._verify_contact(checked, self.entry(), self.raw(), 0)
                self.assertEqual("unscored", summary["status"])
                self.assertIsNone(summary["classification"])
                self.assertEqual((1, client.usage_summary()["estimated_usd"]), (count, cost))

    def test_retained_response_usage_mutation_fails_even_when_unscored(self):
        self.override(experiment.contact, "verify", return_value=None)
        checked = self.checked()
        checked["usage"]["output_tokens"] += 1
        with self.assertRaises(ValidationError):
            experiment._verify_contact(checked, self.entry(), self.raw(), 0)

    def test_current_policy_keeps_clear_dense_view_without_relaxing_spacing(self):
        entry, raw = self.entry(), self.raw()
        reports = {i: self.checked(entry, i) for i in range(3)}
        self.override(experiment.masks, "_filter", side_effect=lambda runtime, *a: (runtime, {}))
        old, _, _ = experiment._policy_predictions(entry, raw, reports, experiment.original.evaluate_tracking, {})
        fresh, policy, _ = experiment._policy_predictions(entry, raw, reports, experiment.current.evaluate_tracking, {})
        self.assertEqual(["unknown", "supported", "unknown"], [row["status"] for row in old])
        self.assertEqual(["unknown", "supported", "supported"], [row["status"] for row in fresh])
        self.assertEqual([0, 2], [row["frame_index"] for row in policy["frames"][2]["confirmation_evidence"]])
        self.assertFalse(any(row["physical_action_authorized"] for row in policy["frames"]))

    def test_explicit_occlusion_resets_history_and_failed_guard_is_unscored(self):
        entry, raw = self.entry(), self.raw()
        reports = {i: self.checked(entry, i) for i in range(3)}
        self.override(experiment.masks, "_filter", side_effect=lambda runtime, *a: (runtime, {}))
        reports[1]["evaluation"].update(fully_visible=False, decision="unknown")
        rows, policy, _ = experiment._policy_predictions(entry, raw, reports, experiment.current.evaluate_tracking, {})
        self.assertEqual(["unknown"] * 3, [row["status"] for row in rows])
        self.assertTrue(policy["frames"][1]["confirmation_history_reset"])
        reports[1].update(status="unscored", evaluation=None)
        rows, _, _ = experiment._policy_predictions(entry, raw, reports, experiment.current.evaluate_tracking, {})
        self.assertEqual("unscored", rows[1]["status"])

    def mock_run(self, *, reserved_attempts=2):
        self.freeze()
        for entry in self.old_plan["clips"]:
            dump(self.previous_dir / f"{entry['clip_id']}-video.json", self.raw(entry, candidates=False))
            for cp in entry["checkpoints"]:
                dump(self.previous_dir / f"{entry['clip_id']}-v2-{cp['frame_index']}.json", experiment.previous._no_candidate())
        self.override(experiment, "load_env_file", return_value=None)
        client = self.override(experiment.original.RetainingClient, "from_env", return_value=object())
        def video(_client, prepared):
            entry = next(row for row in self.fresh_plan["clips"] if row["prepared"] == prepared)
            return self.raw(entry, candidates=False)
        self.override(experiment, "segment_video", side_effect=video)
        self.override(experiment, "public_tracking_summary", return_value={"status": "completed",
            "cost": {"estimated_usd": .01}, "authored_safe_fixture": True})
        self.override(experiment.geometry, "resolve_distractor_conflicts", return_value={})
        self.override(experiment.masks, "_filter", side_effect=lambda runtime, *a: (runtime, {}))
        journal = self.override(experiment, "AttemptJournal")
        journal.return_value.summary.return_value = {"total_reserved_attempts": reserved_attempts}
        return experiment.run(self.root, self.output / "freeze.json"), client

    def test_mocked_end_to_end_separates_costs_cohorts_and_all_nine_methods(self):
        result, client = self.mock_run()
        self.assertEqual((2, .02), (result["new_provider_attempts"], result["estimated_incremental_usd"]))
        self.assertEqual(2, client.call_count)
        self.assertEqual(0, result["cohorts"][experiment.COHORTS[0]]["new_provider_attempts"])
        self.assertEqual(2, result["cohorts"][experiment.COHORTS[1]]["new_provider_attempts"])
        for cohort in result["cohorts"].values():
            self.assertEqual(6, cohort["labeled_frames"])
            self.assertEqual(set(experiment.METHODS), set(cohort["methods"]))
        self.assertEqual(result, experiment.replay(self.root, self.output / "freeze.json"))
        self.assertNotIn(str(self.root), json.dumps(result))
        self.assertFalse(result["physical_action_authorized"])
        self.assertFalse(result["changes_live_placement_acceptance"])
        for _, kwargs in client.call_args_list:
            self.assertEqual((1, 0), (kwargs["max_requests"], kwargs["retries"]))

    def test_crossed_two_by_two_policies_use_correct_guard_and_evaluator(self):
        self.mock_run()
        calls, versions = [], {}
        actual = experiment._policy_predictions
        verify_contact, verify_native = experiment._verify_contact, experiment.previous._verify_guard
        def contact_record(checked, *args):
            versions[id(checked)] = "v3"
            return verify_contact(checked, *args)
        def native_record(checked, *args):
            versions[id(checked)] = "v2"
            return verify_native(checked, *args)
        def capture(entry, raw, reports, evaluator, resolved):
            calls.append((entry["clip_id"], evaluator, tuple(sorted(reports)),
                          tuple(versions[id(reports[index])] for index in sorted(reports))))
            return actual(entry, raw, reports, evaluator, resolved)
        with patch.object(experiment, "_policy_predictions", side_effect=capture), \
                patch.object(experiment, "_verify_contact", side_effect=contact_record), \
                patch.object(experiment.previous, "_verify_guard", side_effect=native_record):
            experiment.replay(self.root, self.output / "freeze.json")
        self.assertEqual(16, len(calls))
        for start in range(0, len(calls), 4):
            self.assertEqual([experiment.original.evaluate_tracking, experiment.current.evaluate_tracking] * 2,
                             [row[1] for row in calls[start:start + 4]])
            self.assertTrue(all(row[2] == (0, 1, 2) for row in calls[start:start + 4]))
            self.assertEqual([("v2",) * 3] * 2 + [("v3",) * 3] * 2,
                             [row[3] for row in calls[start:start + 4]])

    def test_admission_receipt_is_exclusive_before_any_second_run_request(self):
        _, client = self.mock_run()
        first_calls = client.call_count
        with self.assertRaises(FileExistsError):
            experiment.run(self.root, self.output / "freeze.json")
        self.assertEqual(first_calls, client.call_count)

    def test_journal_count_mismatch_cannot_be_published(self):
        with self.assertRaisesRegex(ValidationError, "journal"):
            self.mock_run(reserved_attempts=1)

    def test_previous_baseline_scores_are_exact_and_not_silently_recomputed(self):
        result, _ = self.mock_run()
        old = result["cohorts"][experiment.COHORTS[0]]["methods"]
        prior = self.previous_result["cohorts"]["new_recordings_same_families"]["methods"]
        for current, previous in (("visibility_v2", "visibility_v2"), ("candidate_v2", "candidate_guard_v2"),
                                  ("v2_greedy", "mask_policy_v2"), ("sam_presence", "sam_presence")):
            self.assertEqual(prior[previous], old[current])
        changed = deepcopy(self.previous_result)
        changed["cohorts"]["new_recordings_same_families"]["methods"]["mask_policy_v2"]["metrics"]["coverage"] = .5
        self.previous_result = changed
        dump(self.previous_dir / "public-result.json", changed)
        # Allow only the outer historical digest check, to exercise exact score-preservation validation.
        with patch.object(experiment, "verify_plan", return_value=(self.old_plan, self.fresh_plan)):
            plan = json.loads((self.output / "freeze.json").read_text())
            plan["previous_public_sha256"] = experiment.original._digest(changed)
            dump(self.output / "freeze.json", plan)
            with self.assertRaisesRegex(ValidationError, "baseline"):
                experiment.replay(self.root, self.output / "freeze.json")


if __name__ == "__main__":
    unittest.main()
