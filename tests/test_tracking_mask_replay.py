"""Authored cached-replay fixtures, real pure policy, no API calls or media tools."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from proofloop.core import ValidationError
from proofloop import tracking_mask_replay as replay
from proofloop.tracking_evaluation import score_frames
from proofloop.tracking_policy import evaluate_tracking


def dump(path, value):
    Path(path).write_text(json.dumps(value, sort_keys=True) + "\n")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class CachedMaskReplayTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.base_dir = self.root / "artifacts/local/authored-v1"
        self.base_dir.mkdir(parents=True)
        self.output = self.root / "artifacts/local/authored-v2"
        self.frozen = self.base_dir / "freeze.json"
        self.mode = "disjoint"
        self.hidden = False
        self.target_box = [.1, .1, .5, .5]
        self.distractor_box = [.4, .4, .8, .8]
        self.source_hash = "a" * 64
        self.prepared_hash = "b" * 64
        original_file, clip_file = self.base_dir / "original.mp4", self.base_dir / "clip.mp4"
        original_file.write_bytes(b"authored original bytes, not actual video")
        clip_file.write_bytes(b"authored clip bytes, not actual video")
        self.entry = {"clip_id": "authored", "original_file": str(original_file), "original_sha256": sha(original_file),
            "prepared": {"clip_file": str(clip_file), "prepared_hash": self.prepared_hash,
                "source": {"sha256": self.source_hash, "width": 100, "height": 100}},
            "contract": {"target_id": "target"}, "seeds": [{"track_id": 1}, {"track_id": 2}],
            "frames": [], "checkpoints": [{"frame_index": index} for index in range(3)]}
        self.raw = {"status": "completed", "prepared": deepcopy(self.entry["prepared"]), "frames": []}
        for index in range(3):
            frame_file = self.base_dir / f"frame-{index}.jpg"
            frame_file.write_bytes(b"authored source jpeg bytes " + str(index).encode())
            self.entry["frames"].append({"frame_index": index, "timestamp_ms": index * 500,
                "file": str(frame_file), "sha256": sha(frame_file)})
            self.raw["frames"].append({"frame_index": index, "timestamp_ms": index * 500,
                "candidates": [{"track_id": 1, "object_id": "target", "box": self.target_box,
                    "confidence": .95, "mask_supported": True, "mask": {"decoded": True, "raster_sha256": "e" * 64}},
                    {"track_id": 2, "object_id": "distractor", "box": self.distractor_box,
                    "confidence": .95, "mask_supported": True, "mask": {"decoded": True, "raster_sha256": "f" * 64}}]})
        self.base_plan = {"freeze_hash": "c" * 64, "implementation_sha256": {"authored-original.py": "d" * 64},
            "annotations_file": str(self.base_dir / "annotations.json"), "clips": [self.entry],
            "config": {"min_confidence": .8, "min_stable_views": 2, "min_spacing_ms": 500,
                "localization_iou_threshold": .3, "localization_sensitivity_thresholds": [.2, .3, .5]}}
        dump(self.base_dir / "annotations.json", {"authored_reference_fixture": True})
        dump(self.frozen, self.base_plan)
        dump(self.base_dir / "authored-video.json", self.raw)
        self.checks = {}
        for index in range(3):
            crop = self.base_dir / f"crop-{index}.jpg"
            crop.write_bytes(b"authored crop bytes " + str(index).encode())
            checked = {"status": "completed", "prepared": {"source_file": self.entry["frames"][index]["file"],
                "crop_file": str(crop), "frame_sha256": self.entry["frames"][index]["sha256"]},
                "evaluation": {"fully_visible": True, "identity_ambiguous": False, "decision": "supported"}}
            self.checks[index] = checked
            dump(self.base_dir / f"authored-guard-{index}.json", checked)
        for number in range(4):
            dump(self.base_dir / f"authored-pair-{number}.json", {"authored_fixture": True})
        (self.base_dir / "attempts.sqlite3").write_bytes(b"authored journal fixture; no provider calls")
        self.base_result = self._base_result()
        dump(self.base_dir / "public-result.json", self.base_result)
        dump(self.base_dir / "run-started.json", {"freeze_hash": self.base_plan["freeze_hash"]})
        self.original_replay = self._patch("proofloop.tracking_mask_replay.original.replay",
            side_effect=lambda *args, **kwargs: deepcopy(self.base_result))
        self.resolver = self._patch("proofloop.tracking_mask_replay.geometry.resolve_distractor_conflicts",
            side_effect=self._resolved)
        self._patch("proofloop.tracking_mask_replay.original.run", side_effect=AssertionError("No original inference run permitted"))
        self._patch("proofloop.tracking_mask_replay.original.load_env_file", side_effect=AssertionError("No API key access permitted"))

    def _patch(self, target, **kwargs):
        handle = patch(target, **kwargs)
        value = handle.start()
        self.addCleanup(handle.stop)
        return value

    def _base_result(self):
        runtime = []
        for index, raw_frame in enumerate(self.raw["frames"]):
            check = self.checks[index]
            semantic = {"target_id": "target", "track_id": 1, "frame_index": index,
                "source_sha256": self.source_hash, "frame_sha256": self.entry["frames"][index]["sha256"],
                "candidate_box": self.target_box, "decision": check["evaluation"]["decision"],
                "evidence_sha256": replay._digest(check)}
            runtime.append({"frame_index": index, "timestamp_ms": index * 500, "source_sha256": self.source_hash,
                "frame_sha256": self.entry["frames"][index]["sha256"],
                "candidates": [{"track_id": 1, "object_id": "target", "box": self.target_box,
                    "confidence": .95, "mask_supported": True, "fully_visible": check["evaluation"]["fully_visible"],
                    "identity_ambiguous": False, "semantic": semantic}],
                "reviewed_distractors": [{"track_id": 2, "box": self.distractor_box, "origin": "runtime_sam_track"}]})
        policy = evaluate_tracking(runtime, target_id="target", seed={"target_id": "target", "track_id": 1,
            "frame_index": 0, "source_sha256": self.source_hash, "frame_sha256": self.entry["frames"][0]["sha256"],
            "review_sha256": self.prepared_hash, "reviewed_distractor_track_ids": [2]},
            min_confidence=.8, min_stable_views=2, min_spacing_ms=500)
        references = [{"clip_id": "authored", "frame_index": index, "expected": "supported",
            "target_box": self.target_box, "distractor_boxes": [self.distractor_box], "condition": "clear"} for index in range(3)]
        predicted = [{"clip_id": "authored", "frame_index": frame["frame_index"], "status": frame["status"],
            "box": frame["box"] if frame["status"] == "supported" else None} for frame in policy["frames"]]
        scored = score_frames(references, predicted)
        return {"freeze_hash": self.base_plan["freeze_hash"], "implementation_sha256": self.base_plan["implementation_sha256"],
            "source_videos": 1, "references": references, "methods": {name: deepcopy(scored) for name in replay.original.METHODS},
            "evidence": [{"clip_id": "authored", "policy": policy, "source_attribution": {"kind": "authored_test_fixture"}}],
            "accounting": {"total_reserved_attempts": 0}, "estimated_usd": 0}

    def _resolved(self, raw, target_track_id=1):
        frames = []
        for index in range(3):
            disjoint = self.mode in ("disjoint", "invalid_hash", "conflicting_metadata")
            unavailable = self.mode == "unavailable"
            intersection = None if unavailable else 0 if disjoint else 5
            comparison = {"distractor_track_id": 2, "bbox_overlap": True,
                "mask_relation": "unavailable" if unavailable else "disjoint" if disjoint else "overlap",
                "intersection_pixels": intersection, "target_foreground_pixels": None if unavailable else 10,
                "distractor_foreground_pixels": None if unavailable else 20,
                "mask_iou": None if unavailable else 0 if disjoint else .2,
                "target_raster_sha256": None if unavailable else "e" * 64,
                "distractor_raster_sha256": None if unavailable else "f" * 64,
                "target_mapped_mask_sha256": None if unavailable else "e" * 64,
                "distractor_mapped_mask_sha256": None if unavailable else "f" * 64}
            if self.mode == "invalid_hash":
                comparison["target_mapped_mask_sha256"] = "/private/secret-mask"
            frames.append({"frame_index": index, "source_sha256": self.source_hash,
                "target_track_id": 1, "target_available": True, "comparisons": [comparison],
                "conflicting_track_ids": [] if disjoint and self.mode != "conflicting_metadata" else [2]})
        return {"method_version": replay.geometry.GEOMETRY_VERSION, "source_sha256": self.source_hash,
            "prepared_hash": self.prepared_hash, "status": "completed", "frames": [] if self.mode == "missing" else frames,
            "mapping": "box_local_mask_nearest_source_pixel_center_inclusive_xyxy"}

    def freeze(self):
        return replay.freeze_replay(self.root, self.frozen, self.output)

    def run_retest(self):
        if not (self.output / "freeze.json").exists():
            self.freeze()
        return replay.run(self.root, self.output / "freeze.json")

    def test_prospective_receipt_is_saved_before_any_corrected_geometry_is_computed(self):
        plan = self.freeze()
        self.assertEqual(0, self.resolver.call_count)
        self.assertEqual(1, self.original_replay.call_count)
        self.assertTrue((self.output / "freeze.json").is_file())
        self.assertFalse(plan["derivation_started"])
        self.assertEqual(0, plan["added_provider_requests"])
        self.assertEqual(replay._digest(self.base_result), plan["original_public_sha256"])
        self.assertEqual(0o600, (self.output / "freeze.json").stat().st_mode & 0o777)

    def test_disjoint_masks_resolve_a_false_box_veto_with_real_unchanged_policy(self):
        before_bytes = (self.base_dir / "public-result.json").read_bytes()
        result = self.run_retest()
        self.assertEqual(3, result["overlap_veto_frames_v1"])
        self.assertEqual(0, result["overlap_veto_frames_v2"])
        self.assertEqual(3, result["validated_disjoint_bbox_conflicts_removed"])
        self.assertEqual(0, result["methods"]["bbox_conflict_v1"]["metrics"]["positives_retained"])
        self.assertEqual(2, result["methods"]["mask_disjoint_v2"]["metrics"]["positives_retained"])
        self.assertEqual(0, result["added_provider_requests"])
        self.assertEqual(0, result["estimated_incremental_provider_usd"])
        self.assertEqual(before_bytes, (self.base_dir / "public-result.json").read_bytes())
        self.assertEqual(replay.EVALUATION_ROLE, result["evaluation_role"])
        private = json.loads((self.output / "private-result.json").read_text())
        for old, new in zip(private["clips"][0]["original_runtime"], private["clips"][0]["corrected_runtime"]):
            self.assertEqual({key: value for key, value in old.items() if key != "reviewed_distractors"},
                {key: value for key, value in new.items() if key != "reviewed_distractors"})
            self.assertEqual([], new["reviewed_distractors"])
        resolver_raw = self.resolver.call_args.args[0]
        self.assertEqual(self.raw, resolver_raw)
        self.assertNotIn("checkpoints", resolver_raw)
        self.assertNotIn("references", resolver_raw)

    def test_actual_overlap_unavailable_masks_and_missing_metadata_keep_the_original_risk(self):
        for mode in ("overlap", "unavailable", "missing", "conflicting_metadata"):
            self.mode = mode
            self.output = self.root / ("artifacts/local/" + mode)
            result = self.run_retest()
            self.assertEqual(3, result["overlap_veto_frames_v2"], mode)
            self.assertEqual(0, result["validated_disjoint_bbox_conflicts_removed"], mode)
            self.assertEqual(0, result["methods"]["mask_disjoint_v2"]["metrics"]["positives_retained"], mode)

    def test_unvalidated_mask_hash_never_removes_risk_or_leaks_private_strings(self):
        self.mode = "invalid_hash"
        result = self.run_retest()
        self.assertEqual(3, result["bbox_conflicts_with_unvalidated_metadata_retained"])
        self.assertEqual(0, result["overlap_veto_reduction"])
        self.assertNotIn("secret-mask", json.dumps(result))

    def test_visibility_guard_can_still_withhold_all_support_after_geometry_is_fixed(self):
        for index, checked in self.checks.items():
            checked["evaluation"].update(fully_visible=False, decision="unknown")
            dump(self.base_dir / f"authored-guard-{index}.json", checked)
        self.base_result = self._base_result()
        dump(self.base_dir / "public-result.json", self.base_result)
        result = self.run_retest()
        self.assertEqual(3, result["overlap_veto_reduction"])
        self.assertEqual(0, result["methods"]["mask_disjoint_v2"]["metrics"]["positives_retained"])
        self.assertTrue(all("full_visibility_not_supported" in frame["gap_codes"]
            for frame in result["evidence"][0]["corrected_policy"]["frames"]))

    def test_snapshot_detects_raw_record_change_during_original_replay(self):
        path = self.base_dir / "authored-video.json"
        def changed(*args, **kwargs):
            path.write_bytes(path.read_bytes() + b"\n")
            return deepcopy(self.base_result)
        self.original_replay.side_effect = changed
        with self.assertRaises(ValidationError):
            self.freeze()
        self.assertFalse(self.output.exists())
        self.assertEqual(0, self.resolver.call_count)

    def test_frozen_input_hashes_reject_later_changes_before_mask_derivation(self):
        self.freeze()
        for path in (self.base_dir / "authored-video.json", self.base_dir / "authored-guard-0.json",
                self.base_dir / "frame-0.jpg", self.base_dir / "crop-0.jpg", self.base_dir / "attempts.sqlite3"):
            saved = path.read_bytes()
            path.write_bytes(saved + b" ")
            with self.assertRaises(ValidationError):
                replay.replay(self.root, self.output / "freeze.json")
            path.write_bytes(saved)
        self.assertEqual(0, self.resolver.call_count)

    def test_post_geometry_input_change_prevents_any_public_result(self):
        original_resolve = self._resolved
        def changed(*args, **kwargs):
            result = original_resolve(*args, **kwargs)
            path = self.base_dir / "authored-guard-1.json"
            path.write_bytes(path.read_bytes() + b"\n")
            return result
        self.resolver.side_effect = changed
        with self.assertRaises(ValidationError):
            self.run_retest()
        self.assertFalse((self.output / "public-result.json").exists())

    def test_reconstruction_must_match_the_original_validated_policy(self):
        self.base_result["evidence"][0]["policy"]["frames"][0]["identity_basis"] = "authored_seed"
        dump(self.base_dir / "public-result.json", self.base_result)
        with self.assertRaises(ValidationError):
            self.run_retest()
        self.assertEqual(0, self.resolver.call_count)

    def test_geometry_requires_the_exact_clip_and_preparation_binding(self):
        def wrong(*args, **kwargs):
            result = self._resolved(*args, **kwargs)
            result["source_sha256"] = "d" * 64
            return result
        self.resolver.side_effect = wrong
        with self.assertRaises(ValidationError):
            self.run_retest()

    def test_v2_code_config_and_receipt_tampering_are_rejected(self):
        self.freeze()
        path = self.output / "freeze.json"
        original_plan = json.loads(path.read_text())
        for mutation in (lambda value: value["config"].update(added_provider_requests=1),
                lambda value: value.update(original_public_sha256="f" * 64),
                lambda value: value["implementation_sha256"].update(extra_py="f" * 64)):
            changed = deepcopy(original_plan)
            mutation(changed)
            dump(path, changed)
            with self.assertRaises(ValidationError):
                replay.replay(self.root, path)
        dump(path, original_plan)
        with patch("proofloop.tracking_mask_replay._implementations", return_value={}):
            with self.assertRaises(ValidationError):
                replay.replay(self.root, path)

    def test_new_files_are_exclusive_and_read_only_replay_reproduces_the_result(self):
        result = self.run_retest()
        self.assertEqual(result, replay.replay(self.root, self.output / "freeze.json"))
        with self.assertRaises(FileExistsError):
            replay.run(self.root, self.output / "freeze.json")
        with self.assertRaises(FileExistsError):
            self.freeze()
        for name in ("freeze.json", "run-started.json", "private-result.json", "public-result.json"):
            self.assertEqual(0o600, (self.output / name).stat().st_mode & 0o777)

    def test_public_projection_omits_private_paths_and_raw_model_or_mask_payloads(self):
        result = self.run_retest()
        serialized = json.dumps(result)
        for value in (str(self.root), "artifacts/local", "original_frozen_file", "encoded_payload", "raster_base64"):
            self.assertNotIn(value, serialized)
        self.assertFalse(result["physical_action_authorized"])
        self.assertFalse(result["changes_live_placement_acceptance"])

    def test_original_validation_failure_cannot_be_bypassed_or_overwrite_v1(self):
        self.original_replay.side_effect = ValidationError("Original cache failed validation")
        before = self.frozen.read_bytes()
        with self.assertRaises(ValidationError):
            self.freeze()
        self.assertEqual(before, self.frozen.read_bytes())
        self.assertFalse(self.output.exists())
        with self.assertRaises(ValidationError):
            replay.freeze_replay(self.root, self.frozen, self.base_dir)


if __name__ == "__main__":
    unittest.main()
