"""Authored masks exercise a fixed geometric margin, never model accuracy."""
import base64
from copy import deepcopy
import hashlib
import json
import random
import unittest
from unittest.mock import patch

from proofloop import gripper_geometry as guard
from proofloop import tracking_geometry as geometry
from proofloop.core import ValidationError
from proofloop.meta import MetaAPIError


def record(identifier, x, y, *, frame=0, confidence=.95):
    return {"object_id": identifier, "frame": frame, "confidence": confidence,
        "box": {"x1": x, "y1": y, "x2": x, "y2": y, "width": 32, "height": 32},
        "mask": {"width": 1, "height": 1, "authored_pixels": [1]}}


def report(grippers=False):
    source = {"sha256": "a" * 64, "width": 32, "height": 32, "time_base": "1/1000", "frame_count": 1,
        "frames": [{"frame_index": 0, "pts": 0, "timestamp_ms": 0}]}
    return {"prepared": {"source": source, "prepared_hash": ("c" if grippers else "b") * 64,
        "requested_frame_indices": [0], "seeds": [{"track_id": i} for i in ((3, 4) if grippers else (1, 2))]},
        "status": "completed", "objects": [record(3, 14, 10), record(4, 20, 20)] if grippers else [record(1, 10, 10)],
        "private_source_path": "/Users/private/clip.mp4", "private_response": "secret-provider-payload"}


def summary(value):
    grouped = {f["frame_index"]: [] for f in value["prepared"]["source"]["frames"]}
    for item in value["objects"]:
        grouped[item["frame"]].append({"track_id": item["object_id"], "confidence": item["confidence"],
            "mask_supported": item.get("mask_supported", True)})
    return {"source": value["prepared"]["source"], "seeds": value["prepared"]["seeds"],
        "status": value["status"], "frames": [{"frame_index": index, "candidates": candidates}
            for index, candidates in grouped.items()] if value["status"] == "completed" else []}


def decode(records):
    values = []
    for item in records:
        result = deepcopy(item)
        raw = bytes(result["mask"]["authored_pixels"])
        result["mask"].update(raster_base64=base64.b64encode(raw).decode(),
            raster_sha256=hashlib.sha256(raw).hexdigest(), foreground_pixels=sum(raw))
        values.append(result)
    return values


class GripperGeometryTests(unittest.TestCase):
    def compare(self, target=None, grippers=None, *, decoder=decode, **kwargs):
        target = report() if target is None else target
        grippers = report(True) if grippers is None else grippers
        with patch.object(guard, "public_tracking_summary", side_effect=summary) as verify:
            with patch.object(geometry, "decode_sam_masks", side_effect=decoder):
                result = guard.compare(target, grippers, **kwargs)
        self.assertEqual(2, verify.call_count)
        return result

    def test_fixed_margin_four_pixel_separation_requires_both_grippers(self):
        value = self.compare()
        row = value["frames"][0]
        self.assertEqual(3, value["config"]["source_pixel_margin"])
        self.assertEqual("separated", row["relation"])
        self.assertTrue(row["separation_supported"])
        self.assertEqual([4, 4], [c["min_chebyshev_distance_capped4"] for c in row["comparisons"]])
        self.assertTrue(all(c["distance_is_lower_bound"] for c in row["comparisons"]))

    def test_three_pixels_is_near_even_with_all_high_confidence_masks(self):
        grippers = report(True)
        grippers["objects"][0] = record(3, 13, 10)
        row = self.compare(grippers=grippers)["frames"][0]
        self.assertEqual("near_or_contact", row["relation"])
        self.assertFalse(row["separation_supported"])
        self.assertEqual(3, row["comparisons"][0]["min_chebyshev_distance_capped4"])
        self.assertFalse(row["comparisons"][0]["distance_is_lower_bound"])
        self.assertFalse(row["physical_contact_observed"])
        self.assertFalse(row["physical_occlusion_observed"])

    def test_shared_foreground_and_diagonal_chebyshev_distance(self):
        for x, y, expected in ((10, 10, 0), (11, 11, 1), (12, 12, 2), (13, 13, 3), (14, 14, 4)):
            grippers = report(True)
            grippers["objects"][0] = record(3, x, y)
            with self.subTest(x=x, y=y):
                row = self.compare(grippers=grippers)["frames"][0]
                self.assertEqual(expected, row["comparisons"][0]["min_chebyshev_distance_capped4"])
                self.assertEqual(expected > 3, row["separation_supported"])

    def test_overlapping_boxes_with_real_four_pixel_foreground_gap_can_separate(self):
        target, grippers = report(), report(True)
        for item, pixels in ((target["objects"][0], [1, 0, 0, 0, 0]), (grippers["objects"][0], [0, 0, 0, 0, 1])):
            item["box"].update(x1=10, x2=14, y1=10, y2=10)
            item["mask"].update(width=5, height=1, authored_pixels=pixels)
        row = self.compare(target, grippers)["frames"][0]
        self.assertTrue(row["separation_supported"])
        self.assertEqual(4, row["comparisons"][0]["min_chebyshev_distance_capped4"])

    def test_inclusive_box_edge_is_real_zero_distance_foreground(self):
        target, grippers = report(), report(True)
        target["objects"][0]["box"].update(x1=9, x2=10)
        target["objects"][0]["mask"].update(width=2, authored_pixels=[0, 1])
        grippers["objects"][0] = record(3, 10, 10)
        row = self.compare(target, grippers)["frames"][0]
        self.assertEqual(0, row["comparisons"][0]["min_chebyshev_distance_capped4"])

    def test_bitset_margin_matches_independent_brute_force_pixel_distances(self):
        randomizer = random.Random(293)
        for attempt in range(24):
            target, grippers = report(), report(True)
            foregrounds = []
            for item in (target["objects"][0], grippers["objects"][0]):
                foreground = {(randomizer.randrange(24), randomizer.randrange(24)) for _ in range(5)}
                foregrounds.append(foreground)
                item["box"].update(x1=0, y1=0, x2=23, y2=23)
                item["mask"].update(width=24, height=24,
                    authored_pixels=[int((x, y) in foreground) for y in range(24) for x in range(24)])
            expected = min(max(abs(ax - bx), abs(ay - by)) for ax, ay in foregrounds[0] for bx, by in foregrounds[1])
            with self.subTest(attempt=attempt):
                observed = self.compare(target, grippers)["frames"][0]["comparisons"][0]["min_chebyshev_distance_capped4"]
                self.assertEqual(min(4, expected), observed)

    def test_source_frames_are_compared_only_at_the_same_exact_index_and_pts(self):
        target, grippers = report(), report(True)
        for value in (target, grippers):
            value["prepared"]["source"].update(frame_count=2,
                frames=[{"frame_index": 0, "pts": 0, "timestamp_ms": 0},
                        {"frame_index": 1, "pts": 200, "timestamp_ms": 200}])
            value["prepared"]["requested_frame_indices"] = [0, 1]
        target["objects"].append(record(1, 10, 10, frame=1))
        grippers["objects"].extend([record(3, 13, 10, frame=1), record(4, 20, 20, frame=1)])
        rows = self.compare(target, grippers)["frames"]
        self.assertEqual(["separated", "near_or_contact"], [row["relation"] for row in rows])
        self.assertEqual([0, 200], [row["pts"] for row in rows])

    def test_missing_gripper_never_means_verified_absence(self):
        grippers = report(True)
        grippers["objects"] = grippers["objects"][:1]
        row = self.compare(grippers=grippers)["frames"][0]
        self.assertEqual("unavailable", row["relation"])
        self.assertFalse(row["separation_supported"])
        self.assertIsNone(row["comparisons"][1]["min_chebyshev_distance_capped4"])
        self.assertIn("gripper_track_missing_not_verified_absent", row["comparisons"][1]["reason_codes"])

    def test_one_near_and_one_missing_still_withholds_without_claiming_complete_evidence(self):
        grippers = report(True)
        grippers["objects"] = [record(3, 12, 10)]
        row = self.compare(grippers=grippers)["frames"][0]
        self.assertEqual("near_or_contact", row["relation"])
        self.assertFalse(row["separation_supported"])
        self.assertEqual("unavailable", row["comparisons"][1]["relation"])

    def test_missing_target_or_failed_provider_is_unavailable(self):
        for failed_role in ("target_missing", "target_failed", "grippers_failed"):
            target, grippers = report(), report(True)
            if failed_role == "target_missing":
                target["objects"] = []
            elif failed_role == "target_failed":
                target.update(status="unscored", objects=[])
            else:
                grippers.update(status="unscored", objects=[])
            with self.subTest(failed_role=failed_role):
                row = self.compare(target, grippers)["frames"][0]
                self.assertEqual("unavailable", row["relation"])
                self.assertFalse(row["separation_supported"])

    def test_low_unreported_boolean_or_invalid_confidence_never_clears(self):
        for role in ("target", "gripper"):
            for confidence in (None, .7999, True, float("inf"), -1):
                target, grippers = report(), report(True)
                (target if role == "target" else grippers)["objects"][0]["confidence"] = confidence
                with self.subTest(role=role, confidence=confidence):
                    # Reports containing NaN/Infinity are not accepted by the
                    # real source replay; use a non-finite-free hash projection
                    # to exercise the local confidence boundary independently.
                    with patch.object(guard, "_digest", return_value="d" * 64):
                        row = self.compare(target, grippers)["frames"][0]
                    self.assertEqual("unavailable", row["relation"])
                    self.assertFalse(row["separation_supported"])

    def test_exact_confidence_threshold_remains_eligible(self):
        grippers = report(True)
        grippers["objects"][0]["confidence"] = .8
        self.assertEqual("separated", self.compare(grippers=grippers)["frames"][0]["relation"])

    def test_empty_phantom_or_unconfirmed_mask_cannot_establish_separation(self):
        for role in ("target", "gripper"):
            for mode in ("empty", "unconfirmed", "corrupt"):
                target, grippers = report(), report(True)
                item = (target if role == "target" else grippers)["objects"][0]
                if mode == "empty":
                    item["mask"]["authored_pixels"] = [0]
                elif mode == "unconfirmed":
                    item["mask_supported"] = False
                else:
                    item["mask"]["authored_pixels"] = [2]
                with self.subTest(role=role, mode=mode):
                    row = self.compare(target, grippers)["frames"][0]
                    self.assertEqual("unavailable", row["relation"])
                    self.assertFalse(row["separation_supported"])

    def test_invalid_official_decoder_or_raster_hash_cannot_clear(self):
        def bad_hash(records):
            result = decode(records)
            result[0]["mask"]["raster_sha256"] = "f" * 64
            return result
        def missing_parser(records):
            raise MetaAPIError("authored unavailable parser")
        for decoder in (bad_hash, missing_parser):
            with self.subTest(decoder=decoder):
                self.assertEqual("unavailable", self.compare(decoder=decoder)["frames"][0]["relation"])

    def test_source_sha_dimensions_pts_timebase_or_requested_indices_mismatch_raises(self):
        for mode in ("sha", "width", "height", "pts", "timebase", "requested"):
            grippers = report(True)
            source = grippers["prepared"]["source"]
            if mode == "sha":
                source["sha256"] = "f" * 64
            elif mode in {"width", "height"}:
                source[mode] += 1
            elif mode == "pts":
                source["frames"][0]["pts"] = 1
            elif mode == "timebase":
                source["time_base"] = "1/500"
            else:
                grippers["prepared"]["requested_frame_indices"] = []
            with self.subTest(mode=mode), self.assertRaises(ValidationError):
                self.compare(grippers=grippers)

    def test_unreviewed_invalid_repeated_or_target_gripper_ids_rejected(self):
        for kwargs in ({"target_track_id": True}, {"target_track_id": 5}, {"gripper_track_ids": (3, 5)},
                {"gripper_track_ids": (3, 3)}, {"gripper_track_ids": (1, 3)}, {"gripper_track_ids": ()},
                {"gripper_track_ids": (True, 4)}, {"gripper_track_ids": (-1, 4)}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValidationError):
                self.compare(**kwargs)

    def test_report_replay_failure_precedes_mask_decode(self):
        with patch.object(guard, "public_tracking_summary", side_effect=ValidationError("invalid binding")):
            with patch.object(geometry, "decode_sam_masks") as decoder:
                with self.assertRaises(ValidationError):
                    guard.compare(report(), report(True))
        decoder.assert_not_called()

    def test_duplicate_raw_frame_track_records_rejected(self):
        grippers = report(True)
        grippers["objects"].append(deepcopy(grippers["objects"][0]))
        with self.assertRaises(ValidationError):
            self.compare(grippers=grippers)

    def test_pixel_and_time_budgets_withhold_instead_of_clearing(self):
        with patch.object(guard, "MAX_SOURCE_MAPPED_PIXELS", 1):
            self.assertEqual("unavailable", self.compare()["frames"][0]["relation"])
        with patch.object(guard.time, "monotonic", side_effect=[0] + [100] * 100):
            self.assertEqual("unavailable", self.compare()["frames"][0]["relation"])

    def test_public_result_hashes_both_raw_reports_rasters_and_source_placements(self):
        value = self.compare()
        comparison = value["frames"][0]["comparisons"][0]
        for field in ("target_report_sha256", "gripper_report_sha256", "source_frame_map_sha256"):
            self.assertEqual(64, len(value[field]))
        for field in ("target_raster_sha256", "target_mapped_mask_sha256", "gripper_raster_sha256", "gripper_mapped_mask_sha256"):
            self.assertEqual(64, len(comparison[field]))
        grippers = report(True)
        grippers["objects"][0]["box"].update(x1=15, x2=15)
        moved = self.compare(grippers=grippers)["frames"][0]["comparisons"][0]
        self.assertEqual(comparison["gripper_raster_sha256"], moved["gripper_raster_sha256"])
        self.assertNotEqual(comparison["gripper_mapped_mask_sha256"], moved["gripper_mapped_mask_sha256"])

    def test_safe_public_allowlist_preserves_inputs_and_no_action_authorization(self):
        target, grippers = report(), report(True)
        expected = deepcopy([target, grippers])
        value = self.compare(target, grippers)
        self.assertEqual(expected, [target, grippers])
        serialized = json.dumps(value, allow_nan=False)
        for private in ("/Users/", "secret-provider-payload", "raster_base64", "authored_pixels", "encoded_payload"):
            self.assertNotIn(private, serialized)
        self.assertFalse(value["physical_action_authorized"])
        self.assertFalse(value["identity_independently_verified"])
        self.assertFalse(value["full_visibility_independently_verified"])


if __name__ == "__main__":
    unittest.main()
