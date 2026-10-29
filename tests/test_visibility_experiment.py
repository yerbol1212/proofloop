"""Authored prospective-comparison fixtures; no media extraction or provider calls."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from proofloop.core import ValidationError
from proofloop import visibility_experiment as experiment
from proofloop.meta import MetaClient


def dump(path, value):
    Path(path).write_text(json.dumps(value, sort_keys=True) + "\n")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class VisibilityExperimentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.old_dir = self.root / "artifacts/local/authored-old"
        self.fresh_dir = self.root / "artifacts/local/authored-fresh"
        self.output = self.root / "artifacts/local/authored-visibility"
        self.old_dir.mkdir(parents=True)
        self.fresh_dir.mkdir(parents=True)
        self.old_path, self.fresh_path = self.old_dir / "freeze.json", self.fresh_dir / "freeze.json"
        self.old_plan = self._source_plan(self.old_dir, "old", 1)
        self.fresh_plan = self._source_plan(self.fresh_dir, "fresh", 2)
        dump(self.old_path, self.old_plan)
        dump(self.fresh_path, self.fresh_plan)
        self.old_result = {"authored_previous_result": True}
        self._patch("proofloop.visibility_experiment.original.verify_plan", return_value=None)
        self.previous_replay = self._patch("proofloop.visibility_experiment.original.replay",
            side_effect=lambda *a, **k: deepcopy(self.old_result))
        self._patch("proofloop.visibility_experiment.masks._validated_base", side_effect=lambda *a, **k:
            ({"plan": deepcopy(self.old_plan), "records": []}, deepcopy(self.old_result)))

    def _patch(self, name, **kwargs):
        handle = patch(name, **kwargs)
        result = handle.start()
        self.addCleanup(handle.stop)
        return result

    def _source_plan(self, directory, prefix, episode):
        annotation = directory / "annotations.json"
        dump(annotation, {"authored_scoring_fixture": True})
        entries = []
        for number, family in enumerate(("lerobot/aloha_static_screw_driver", "lerobot/aloha_static_battery")):
            name = f"{prefix}-{number}"
            original, clip = directory / f"{name}-original.mp4", directory / f"{name}-clip.mp4"
            original.write_bytes(f"authored original {name}".encode())
            clip.write_bytes(f"authored clip {name}".encode())
            frames = []
            for index in range(3):
                frame = directory / f"{name}-{index}.jpg"
                frame.write_bytes(f"authored image {name} {index}".encode())
                frames.append({"frame_index": index, "timestamp_ms": index * 500,
                    "file": str(frame), "sha256": sha(frame)})
            entries.append({"clip_id": name, "source_attribution": {"dataset": family,
                    "episode": episode, "source_role": "authored_test_fixture"},
                "original_file": str(original), "original_sha256": sha(original),
                "prepared": {"clip_file": str(clip), "prepared_hash": sha(clip),
                    "source": {"sha256": sha(clip), "width": 100, "height": 100}},
                "contract": {"target_id": "target", "concept": "authored target",
                    "attributes": {"shape": "Distinct shape"}, "exclusions": {"neighbor": "Distinct neighbor"}},
                "derivation": {"mode": "authored_test_fixture"}, "frames": frames,
                "seeds": [{"track_id": 1}, {"track_id": 2}],
                "checkpoints": [{"frame_index": index, "expected": "supported", "target_box": [.1, .1, .5, .5],
                    "distractor_boxes": [], "condition": "clear"} for index in range(3)]})
        return {"freeze_hash": hashlib.sha256(prefix.encode()).hexdigest(),
            "annotations_file": str(annotation), "clips": entries}

    def freeze(self):
        return experiment.freeze(self.root, self.old_path, self.fresh_path, self.output)

    def persist_fresh(self):
        dump(self.fresh_path, self.fresh_plan)

    def entry(self):
        return deepcopy(self.fresh_plan["clips"][0])

    def raw(self, entry=None, candidates=True):
        entry = entry or self.entry()
        return {"status": "completed", "prepared": deepcopy(entry["prepared"]), "attempts": [{"status": "ok"}],
            "frames": [{"frame_index": i, "timestamp_ms": i * 500, "candidates": [
                {"track_id": 1, "object_id": "target", "box": [.1, .1, .5, .5],
                 "confidence": .95, "mask_supported": True}] if candidates else []} for i in range(3)]}

    def checked(self, entry=None, index=0):
        entry = entry or self.entry()
        prepared = {"contract": deepcopy(entry["contract"]), "source_sha256": entry["prepared"]["source"]["sha256"],
            "frame_index": index, "track_id": 1, "frame_sha256": entry["frames"][index]["sha256"],
            "source_file": entry["frames"][index]["file"], "crop_file": "/private/not-public/crop.jpg",
            "crop_sha256": "e" * 64, "prepared_hash": "f" * 64,
            "candidate_box": {"x1": 10, "y1": 10, "x2": 49, "y2": 49}}
        parsed = {"candidate_identity": "target", "fully_visible": True, "identity_ambiguous": False,
            "attributes": {"shape": "present"}, "exclusions": {"neighbor": "absent"}}
        raw_response = {"model": experiment.SPARK_MODEL, "status": "completed", "output": [
            {"type": "message", "content": [{"type": "output_text", "text": json.dumps(parsed)}]}]}
        return {"method_version": experiment.old_guard.VERSION, "status": "completed", "prepared": prepared,
            "response": {"parsed": parsed, "response": raw_response, "source": {"sha256": "e" * 64}},
            "evaluation": experiment.old_guard.evaluate(parsed, entry["contract"]), "error": None,
            "usage": {"request_attempts": 1, "estimated_usd": .01}}

    def test_freeze_is_prospective_and_preserves_separate_source_hashes(self):
        plan = self.freeze()
        self.assertFalse(plan["inference_started"])
        self.assertEqual(self.old_plan["freeze_hash"], plan["original_freeze_hash"])
        self.assertEqual(self.fresh_plan["freeze_hash"], plan["fresh_source_preparation_hash"])
        self.assertFalse((self.output / "run-started.json").exists())
        self.assertEqual(0o600, (self.output / "freeze.json").stat().st_mode & 0o777)
        self.assertEqual(0, self.previous_replay.call_count)

    def test_mutated_configuration_is_rejected_even_with_rehashed_receipt(self):
        plan = self.freeze()
        plan["config"]["min_confidence"] = .1
        plan["freeze_hash"] = experiment.original._digest({k: v for k, v in plan.items() if k != "freeze_hash"})
        with self.assertRaises(ValidationError):
            experiment.verify_plan(self.root, plan, files=False)

    def test_implementation_or_receipt_mutation_is_rejected(self):
        plan = self.freeze()
        for field, value in (("implementation_sha256", {}), ("freeze_hash", "0" * 64),
                             ("inference_started", True), ("new_guard_config", {})):
            changed = deepcopy(plan)
            changed[field] = value
            with self.subTest(field=field), self.assertRaises(ValidationError):
                experiment.verify_plan(self.root, changed, files=False)

    def test_changed_source_bytes_are_rejected_before_run(self):
        plan = self.freeze()
        Path(self.fresh_plan["clips"][0]["frames"][0]["file"]).write_bytes(b"changed source")
        with self.assertRaises(ValidationError):
            experiment.verify_plan(self.root, plan)

    def test_reused_episode_bytes_contract_or_identifier_cannot_enter_new_cohort(self):
        original = deepcopy(self.fresh_plan)
        mutations = (
            lambda e: e["source_attribution"].update(episode=1),
            lambda e: e.update(original_sha256=self.old_plan["clips"][0]["original_sha256"]),
            lambda e: e["prepared"]["source"].update(sha256=self.old_plan["clips"][0]["prepared"]["source"]["sha256"]),
            lambda e: e["contract"].update(concept="changed visual target"),
            lambda e: e.update(clip_id=self.old_plan["clips"][0]["clip_id"]),
        )
        for index, mutate in enumerate(mutations):
            self.fresh_plan = deepcopy(original)
            mutate(self.fresh_plan["clips"][0])
            self.persist_fresh()
            with self.subTest(index=index), self.assertRaises(ValidationError):
                self.freeze()

    def test_output_cannot_be_nested_inside_either_source_directory(self):
        for directory in (self.old_dir, self.fresh_dir):
            with self.subTest(directory=directory), self.assertRaises(ValidationError):
                experiment.freeze(self.root, self.old_path, self.fresh_path, directory / "responses")

    def test_public_plan_contains_only_safe_source_projection_and_separate_cohorts(self):
        self.freeze()
        public = experiment.public_plan(self.root, self.output / "freeze.json")
        text = json.dumps(public)
        self.assertNotIn(str(self.root), text)
        self.assertNotIn("annotations_file", text)
        self.assertNotIn("clip_file", text)
        self.assertEqual(set(experiment.COHORTS), set(public["cohorts"]))
        old_ids = {x["clip_id"] for x in public["cohorts"][experiment.COHORTS[0]]["sources"]}
        fresh_ids = {x["clip_id"] for x in public["cohorts"][experiment.COHORTS[1]]["sources"]}
        self.assertFalse(old_ids & fresh_ids)
        self.assertFalse(public["changes_live_placement_acceptance"])

    def test_missing_candidate_does_not_create_client_prepare_or_call_model(self):
        client = Mock(side_effect=AssertionError("No API when tracker has no candidate"))
        module = Mock()
        result = experiment._assess(client, self.entry(), self.raw(candidates=False), 0, module, self.output)
        self.assertEqual(experiment._no_candidate(), result)
        client.assert_not_called()
        module.prepare.assert_not_called()
        module.challenge.assert_not_called()

    def test_assess_passes_exact_runtime_candidate_and_source_not_reference_box(self):
        entry, raw, module = self.entry(), self.raw(), Mock()
        entry["checkpoints"][0]["target_box"] = [.7, .7, .9, .9]
        module.prepare.return_value = {"authored_preparation": True}
        client = Mock(return_value="authored_client")
        experiment._assess(client, entry, raw, 0, module, self.output)
        args, kwargs = module.prepare.call_args
        self.assertEqual(entry["frames"][0]["file"], args[0])
        self.assertEqual({"x1": 10, "y1": 10, "x2": 49, "y2": 49}, args[1])
        self.assertEqual(entry["contract"], args[2])
        self.assertEqual(entry["prepared"]["source"]["sha256"], kwargs["source_sha256"])
        module.challenge.assert_called_once_with("authored_client", module.prepare.return_value)

    def test_guard_replay_rejects_changed_candidate_source_track_frame_or_contract(self):
        entry, raw, checked = self.entry(), self.raw(), self.checked()
        self._patch("proofloop.visibility_experiment.old_guard.verify", return_value=None)
        self._patch("proofloop.visibility_experiment.original._verified_usage", return_value=None)
        changes = {"candidate_box": {"x1": 11, "y1": 10, "x2": 49, "y2": 49},
            "source_sha256": "0" * 64, "frame_sha256": "1" * 64, "source_file": "/private/other.jpg",
            "track_id": 2, "frame_index": 1, "contract": {"target_id": "different"}}
        for key, value in changes.items():
            changed = deepcopy(checked)
            changed["prepared"][key] = value
            with self.subTest(key=key), self.assertRaises(ValidationError):
                experiment._verify_guard(changed, entry, raw, 0, experiment.old_guard)

    def test_guard_summary_omits_private_paths_and_provider_response_payload(self):
        self._patch("proofloop.visibility_experiment.old_guard.verify", return_value=None)
        self._patch("proofloop.visibility_experiment.original._verified_usage", return_value=None)
        summary, attempts, cost = experiment._verify_guard(self.checked(), self.entry(), self.raw(), 0, experiment.old_guard)
        self.assertNotIn("/private/", json.dumps(summary))
        self.assertNotIn("response", summary)
        self.assertEqual(1, attempts)
        self.assertEqual(.01, cost)

    def test_missing_candidate_requires_exact_no_attempt_record_at_replay(self):
        summary, attempts, cost = experiment._verify_guard(experiment._no_candidate(), self.entry(),
            self.raw(candidates=False), 0, experiment.old_guard)
        self.assertIsNone(summary)
        self.assertEqual((0, 0), (attempts, cost))
        with self.assertRaises(ValidationError):
            experiment._verify_guard(self.checked(), self.entry(), self.raw(candidates=False), 0, experiment.old_guard)

    def test_provider_failure_is_unscored_and_not_successful_unknown(self):
        entry, raw, checked = self.entry(), self.raw(), self.checked()
        checked.update(status="unscored", evaluation=None, error={"code": "incomplete_response", "response": None})
        for options in ({}, {"visibility_only": True}):
            self.assertEqual("unscored", experiment._prediction(entry, raw, checked, 0, **options)["status"])
        raw["status"] = "unscored"
        self.assertEqual("unscored", experiment._prediction(entry, raw, {}, 0, presence_only=True)["status"])
        self.assertEqual("unknown", experiment._prediction(entry, self.raw(candidates=False), experiment._no_candidate(), 0)["status"])

    def test_visibility_and_semantic_identity_are_distinct_diagnostics(self):
        checked, entry, raw = self.checked(), self.entry(), self.raw()
        checked["evaluation"].update(fully_visible=True, decision="contradicted", identity_ambiguous=True)
        self.assertEqual("supported", experiment._prediction(entry, raw, checked, 0, visibility_only=True)["status"])
        self.assertEqual("unknown", experiment._prediction(entry, raw, checked, 0)["status"])
        checked["evaluation"].update(fully_visible=False, decision="supported")
        self.assertEqual("unknown", experiment._prediction(entry, raw, checked, 0, visibility_only=True)["status"])

    def test_two_distinct_views_and_spacing_are_preserved_in_real_policy(self):
        entry, raw = self.entry(), self.raw()
        reports = {i: self.checked(entry, i) for i in range(3)}
        self._patch("proofloop.visibility_experiment.geometry.resolve_distractor_conflicts", return_value={})
        self._patch("proofloop.visibility_experiment.masks._filter", side_effect=lambda runtime, *a: (runtime, {}))
        predictions, policy, _ = experiment._policy_predictions(entry, raw, reports)
        self.assertEqual(["unknown", "supported", "supported"], [p["status"] for p in predictions])
        self.assertTrue(all(not row["physical_action_authorized"] for row in policy["frames"]))
        raw["frames"][1]["timestamp_ms"] = 200
        predictions, _, _ = experiment._policy_predictions(entry, raw, reports)
        self.assertEqual("unknown", predictions[1]["status"])
        self.assertEqual("supported", predictions[2]["status"])

    def test_occlusion_resets_confirmation_and_failed_guard_remains_unscored_in_policy(self):
        entry, raw = self.entry(), self.raw()
        reports = {i: self.checked(entry, i) for i in range(3)}
        reports[1]["evaluation"].update(fully_visible=False, decision="unknown")
        self._patch("proofloop.visibility_experiment.geometry.resolve_distractor_conflicts", return_value={})
        self._patch("proofloop.visibility_experiment.masks._filter", side_effect=lambda runtime, *a: (runtime, {}))
        predictions, _, _ = experiment._policy_predictions(entry, raw, reports)
        self.assertEqual(["unknown"] * 3, [p["status"] for p in predictions])
        reports[1].update(status="unscored", evaluation=None)
        predictions, _, _ = experiment._policy_predictions(entry, raw, reports)
        self.assertEqual("unscored", predictions[1]["status"])

    def test_network_and_incomplete_failures_replay_with_their_actual_retained_usage(self):
        self._patch("proofloop.visibility_experiment.old_guard.verify", return_value=None)
        examples = (("network_error", None, None), ("invalid_response",
            {"status": "incomplete", "model": experiment.SPARK_MODEL,
             "usage": {"input_tokens": 10, "output_tokens": 4096}}, {"input_tokens": 10, "output_tokens": 4096}))
        for status, raw_response, tokens in examples:
            checked = self.checked()
            ledger = {"endpoint": "responses", "attempt": 1, "model": experiment.SPARK_MODEL, "status": status}
            if tokens is not None:
                ledger["usage"] = tokens
            client = MetaClient("offline-no-provider", max_requests=1, retries=0)
            client.ledger = [ledger]
            checked.update(status="unscored", response=None, evaluation=None,
                error={"code": "incomplete_response" if tokens else "network_error", "response": raw_response},
                usage=client.usage_summary())
            with self.subTest(status=status):
                summary, attempts, cost = experiment._verify_guard(checked, self.entry(), self.raw(), 0, experiment.old_guard)
                self.assertEqual("unscored", summary["status"])
                self.assertIsNone(summary["classification"])
                self.assertEqual(1, attempts)
                self.assertEqual(client.usage_summary()["estimated_usd"], cost)

    def test_usage_ledger_mutation_is_rejected_even_for_an_unscored_response(self):
        self._patch("proofloop.visibility_experiment.old_guard.verify", return_value=None)
        checked = self.checked()
        checked.update(status="unscored", response=None, evaluation=None,
            error={"code": "network_error", "response": None}, usage={"requests": [], "request_attempts": 0})
        with self.assertRaises(ValidationError):
            experiment._verify_guard(checked, self.entry(), self.raw(), 0, experiment.old_guard)

    def test_actual_candidate_requires_a_guard_classification_record(self):
        with self.assertRaises(ValidationError):
            experiment._verify_guard({}, self.entry(), self.raw(), 0, experiment.old_guard)
        with self.assertRaises(ValidationError):
            experiment._prediction(self.entry(), self.raw(), {}, 0)

    def test_full_frame_and_crop_must_keep_their_registered_source_order(self):
        self._patch("proofloop.visibility_experiment.guard.verify", return_value=None)
        self._patch("proofloop.visibility_experiment.original._verified_usage", return_value=None)
        checked = self.checked()
        checked["method_version"] = experiment.guard.VERSION
        checked["response"].pop("source")
        checked["response"]["sources"] = [{"sha256": checked["prepared"]["crop_sha256"]},
            {"sha256": checked["prepared"]["frame_sha256"]}]
        self._patch("proofloop.visibility_experiment.guard.evaluate", return_value=checked["evaluation"])
        with self.assertRaisesRegex(ValidationError, "order"):
            experiment._verify_guard(checked, self.entry(), self.raw(), 0, experiment.guard)

    def test_guard_missing_from_disk_is_rejected_by_full_replay(self):
        self.mock_run()
        name = self.fresh_plan["clips"][0]["clip_id"]
        (self.output / f"{name}-v2-0.json").unlink()
        with self.assertRaises(ValidationError):
            experiment.replay(self.root, self.output / "freeze.json")

    def test_visibility_flag_counts_do_not_silently_apply_localization_iou(self):
        references = experiment.original._reference(self.entry())
        predictions = [{"clip_id": r["clip_id"], "frame_index": r["frame_index"],
            "status": "supported", "box": [.7, .7, .9, .9]} for r in references]
        counts = experiment.visibility_counts(references, predictions)
        self.assertEqual(3, counts["reference_clear_count"])
        self.assertEqual(3, counts["clear_called_clear"])
        localized = experiment.score_frames(references, predictions)
        self.assertEqual(0, localized["metrics"]["positives_retained"])

    def test_visibility_confusion_retains_failed_calls_in_each_reference_denominator(self):
        references = experiment.original._reference(self.entry())
        for reference in references[1:]:
            reference.update(expected="unknown", target_box=None, condition="partial_occlusion")
        predictions = [{"clip_id": r["clip_id"], "frame_index": r["frame_index"],
            "status": "unscored", "box": None} for r in references]
        predictions[1].update(status="supported", box=[.1, .1, .5, .5])
        counts = experiment.visibility_counts(references, predictions)
        self.assertEqual(1, counts["reference_clear_count"])
        self.assertEqual(1, counts["clear_unscored"])
        self.assertEqual(2, counts["reference_unclear_count"])
        self.assertEqual(1, counts["unclear_called_clear"])
        self.assertEqual(1, counts["unclear_unscored"])
        self.assertEqual(0, counts["clear_withheld"] + counts["unclear_withheld"])

    def test_visibility_counts_reject_duplicate_and_incomplete_prediction_inventories(self):
        references = experiment.original._reference(self.entry())
        predictions = [{"clip_id": r["clip_id"], "frame_index": r["frame_index"],
            "status": "unknown", "box": None} for r in references]
        for rows in (predictions[:-1], predictions + [predictions[0]]):
            with self.subTest(length=len(rows)), self.assertRaises(ValidationError):
                experiment.visibility_counts(references, rows)

    def mock_run(self, reserved_attempts=2):
        self.freeze()
        for entry in self.old_plan["clips"]:
            dump(self.old_dir / f"{entry['clip_id']}-video.json", self.raw(entry, candidates=False))
            for cp in entry["checkpoints"]:
                dump(self.old_dir / f"{entry['clip_id']}-guard-{cp['frame_index']}.json", experiment._no_candidate())
        self._patch("proofloop.visibility_experiment.load_env_file", return_value=None)
        client = self._patch("proofloop.visibility_experiment.original.RetainingClient.from_env", return_value=object())
        def video(_client, prepared):
            entry = next(e for e in self.fresh_plan["clips"] if e["prepared"] == prepared)
            return self.raw(entry, candidates=False)
        self._patch("proofloop.visibility_experiment.segment_video", side_effect=video)
        self._patch("proofloop.visibility_experiment.public_tracking_summary", return_value={"status": "completed",
            "cost": {"estimated_usd": .01}, "safe_tracking_fixture": True})
        self._patch("proofloop.visibility_experiment.geometry.resolve_distractor_conflicts", return_value={})
        self._patch("proofloop.visibility_experiment.masks._filter", side_effect=lambda runtime, *a: (runtime, {}))
        journal = self._patch("proofloop.visibility_experiment.AttemptJournal")
        journal.return_value.summary.return_value = {"total_reserved_attempts": reserved_attempts}
        return experiment.run(self.root, self.output / "freeze.json"), client

    def test_mocked_end_to_end_run_replay_keeps_cohorts_denominators_and_cost_separate(self):
        result, client = self.mock_run()
        self.assertEqual(2, client.call_count)  # New videos only; no guard calls for missing candidates.
        self.assertEqual(2, result["new_provider_attempts"])
        self.assertEqual(.02, result["estimated_incremental_usd"])
        self.assertEqual(0, result["cohorts"][experiment.COHORTS[0]]["new_provider_attempts"])
        self.assertEqual(2, result["cohorts"][experiment.COHORTS[1]]["new_provider_attempts"])
        for cohort in result["cohorts"].values():
            self.assertEqual(6, cohort["labeled_frames"])
            self.assertEqual(set(experiment.METHODS), set(cohort["methods"]))
            for method in cohort["methods"].values():
                self.assertEqual(0, method["metrics"]["coverage"])
        self.assertEqual(result, experiment.replay(self.root, self.output / "freeze.json"))
        self.assertNotIn(str(self.root), json.dumps(result))
        self.assertFalse(result["changes_live_placement_acceptance"])

    def test_journal_mismatch_is_rejected_in_full_replay(self):
        with self.assertRaisesRegex(ValidationError, "journal"):
            self.mock_run(reserved_attempts=1)

    def test_run_receipt_is_exclusive_and_second_run_cannot_make_requests(self):
        self.mock_run()
        with self.assertRaises(FileExistsError):
            experiment.run(self.root, self.output / "freeze.json")


if __name__ == "__main__":
    unittest.main()
