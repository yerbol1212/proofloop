"""Authored raster geometry fixtures; model/report replay is an explicit boundary."""
import base64
from copy import deepcopy
import hashlib
import json
import unittest
from unittest.mock import patch

from proofloop.core import ValidationError
from proofloop.meta import MetaAPIError
from proofloop.tracking_geometry import resolve_distractor_conflicts


def record(track=1, *, frame=0, box=None, pixels=(1, 0, 1, 0), size=(2, 2)):
    return {"frame": frame, "object_id": track,
            "box": box or {"x1": 1, "y1": 2, "x2": 4, "y2": 5, "width": 16, "height": 16},
            "mask": {"authored_pixels": list(pixels), "width": size[0], "height": size[1]}}


def decoded(objects):
    result = []
    for item in objects:
        mask = item["mask"]
        raw = bytes(mask["authored_pixels"])
        result.append({**deepcopy(item), "mask": {**deepcopy(mask),
            "raster_base64": base64.b64encode(raw).decode(), "raster_sha256": hashlib.sha256(raw).hexdigest(),
            "foreground_pixels": sum(raw)}})
    return result


def report():
    return {"prepared": {"prepared_hash": "b" * 64}, "objects": [record(), record(2, pixels=(0, 1, 0, 1))],
            "private_source_path": "/private/never-public", "private_model_response": "secret-provider-content"}


def summary_for(value):
    grouped = {}
    for item in value["objects"]:
        grouped.setdefault(item["frame"], []).append({"track_id": item["object_id"]})
    return {"source": {"sha256": "a" * 64, "width": 16, "height": 16},
            "seeds": [{"track_id": 1}, {"track_id": 2}], "status": "completed",
            "frames": [{"frame_index": index, "candidates": candidates} for index, candidates in sorted(grouped.items())]}


class TrackingGeometryTests(unittest.TestCase):
    def resolve(self, value=None, *, summary=None, decoder=None, **kwargs):
        value = value if value is not None else report()
        expected = summary if summary is not None else summary_for(value)
        with patch("proofloop.tracking_geometry.public_tracking_summary", return_value=expected) as verify:
            with patch("proofloop.tracking_geometry.decode_sam_masks", side_effect=decoder or decoded):
                result = resolve_distractor_conflicts(value, **kwargs)
        verify.assert_called_once_with(value)
        return result

    def test_overlapping_boxes_with_disjoint_real_foreground_remove_only_geometry_conflict(self):
        result = self.resolve()
        frame, comparison = result["frames"][0], result["frames"][0]["comparisons"][0]
        self.assertTrue(comparison["bbox_overlap"])
        self.assertEqual(("disjoint", 0, 0), (comparison["mask_relation"], comparison["intersection_pixels"], comparison["mask_iou"]))
        self.assertEqual((8, 8), (comparison["target_foreground_pixels"], comparison["distractor_foreground_pixels"]))
        self.assertEqual([], frame["conflicting_track_ids"])
        self.assertEqual("a" * 64, frame["source_sha256"])
        self.assertEqual("b" * 64, result["prepared_hash"])
        self.assertFalse(result["identity_independently_verified"])
        self.assertFalse(result["physical_action_authorized"])

    def test_actual_foreground_intersection_keeps_conflict(self):
        value = report()
        value["objects"][1]["mask"]["authored_pixels"] = [1, 0, 1, 0]
        row = self.resolve(value)["frames"][0]
        comparison = row["comparisons"][0]
        self.assertEqual([2], row["conflicting_track_ids"])
        self.assertEqual(("overlap", 8, 1), (comparison["mask_relation"], comparison["intersection_pixels"], comparison["mask_iou"]))

    def test_offset_boxes_intersect_in_source_coordinates_not_local_raster_indices(self):
        value = report()
        value["objects"][1]["box"]["x1"] = 2
        value["objects"][1]["box"]["x2"] = 5
        value["objects"][1]["mask"]["authored_pixels"] = [1, 0, 1, 0]
        comparison = self.resolve(value)["frames"][0]["comparisons"][0]
        self.assertEqual(4, comparison["intersection_pixels"])
        self.assertEqual(.333333, comparison["mask_iou"])

    def test_box_local_resizing_uses_nearest_source_cell_center_with_inclusive_edges(self):
        value = report()
        for item in value["objects"]:
            item["box"].update(x1=1, y1=1, x2=3, y2=3)
        value["objects"][0]["mask"]["authored_pixels"] = [1, 0, 0, 0]
        value["objects"][1]["mask"]["authored_pixels"] = [0, 0, 0, 1]
        comparison = self.resolve(value)["frames"][0]["comparisons"][0]
        self.assertEqual((1, 4, 0), (comparison["target_foreground_pixels"], comparison["distractor_foreground_pixels"], comparison["intersection_pixels"]))

    def test_nonoverlapping_source_boxes_do_not_conflict_even_with_identical_local_masks(self):
        value = report()
        value["objects"][1]["box"].update(x1=7, x2=10)
        value["objects"][1]["mask"] = deepcopy(value["objects"][0]["mask"])
        frame = self.resolve(value)["frames"][0]
        self.assertFalse(frame["comparisons"][0]["bbox_overlap"])
        self.assertEqual("disjoint", frame["comparisons"][0]["mask_relation"])
        self.assertEqual([], frame["conflicting_track_ids"])

    def test_inclusive_touching_edge_can_have_actual_shared_foreground(self):
        value = report()
        value["objects"][0]["mask"]["authored_pixels"] = [1] * 4
        value["objects"][1]["mask"]["authored_pixels"] = [1] * 4
        value["objects"][1]["box"].update(x1=4, x2=7)
        row = self.resolve(value)["frames"][0]
        self.assertEqual(4, row["comparisons"][0]["intersection_pixels"])
        self.assertEqual([2], row["conflicting_track_ids"])

    def test_decoder_missing_or_invalid_preserves_overlapping_box_risk(self):
        for error in (MetaAPIError("parser missing", code="sam_parser_unavailable"), ValueError("invalid"), KeyError("mask")):
            with self.subTest(error=type(error).__name__):
                def decoder(_):
                    raise error
                row = self.resolve(decoder=decoder)["frames"][0]
                self.assertEqual([2], row["conflicting_track_ids"])
                self.assertEqual("unavailable", row["comparisons"][0]["mask_relation"])
                self.assertIsNone(row["comparisons"][0]["intersection_pixels"])

    def test_empty_foreground_is_unavailable_not_disjoint(self):
        value = report()
        value["objects"][1]["mask"]["authored_pixels"] = [0] * 4
        row = self.resolve(value)["frames"][0]
        self.assertEqual([2], row["conflicting_track_ids"])
        self.assertEqual("unavailable", row["comparisons"][0]["mask_relation"])

    def test_foreground_lost_during_downsampling_cannot_remove_a_conflict(self):
        value = report()
        for item in value["objects"]:
            item["box"].update(x1=1, y1=1, x2=1, y2=1)
        value["objects"][0]["mask"]["authored_pixels"] = [1, 0, 0, 0]
        row = self.resolve(value)["frames"][0]
        self.assertEqual("unavailable", row["comparisons"][0]["mask_relation"])
        self.assertEqual([2], row["conflicting_track_ids"])

    def test_tampered_decoded_raster_hash_keeps_risk(self):
        def decoder(objects):
            result = decoded(objects)
            result[0]["mask"]["raster_sha256"] = "f" * 64
            return result
        row = self.resolve(decoder=decoder)["frames"][0]
        self.assertEqual([2], row["conflicting_track_ids"])
        self.assertEqual("unavailable", row["comparisons"][0]["mask_relation"])

    def test_resized_or_rotated_source_dimensions_cannot_reuse_a_mask(self):
        for dimensions in ((32, 16), (16, 32), (16, 15)):
            value = report()
            value["objects"][0]["box"].update(width=dimensions[0], height=dimensions[1])
            row = self.resolve(value)["frames"][0]
            self.assertEqual([2], row["conflicting_track_ids"])
            self.assertEqual("unavailable", row["comparisons"][0]["mask_relation"])
            self.assertIsNone(row["comparisons"][0]["bbox_overlap"])

    def test_missing_raw_mask_record_cannot_downgrade_present_candidate_risk(self):
        value = report()
        summary = summary_for(value)
        value["objects"] = value["objects"][:1]
        row = self.resolve(value, summary=summary)["frames"][0]
        self.assertEqual([2], row["conflicting_track_ids"])
        self.assertEqual("unavailable", row["comparisons"][0]["mask_relation"])

    def test_missing_target_is_explicit_and_keeps_unresolved_present_distractors(self):
        value = report()
        value["objects"] = value["objects"][1:]
        row = self.resolve(value)["frames"][0]
        self.assertFalse(row["target_available"])
        self.assertEqual([2], row["conflicting_track_ids"])
        self.assertIsNone(row["comparisons"][0]["bbox_overlap"])

    def test_no_distractor_is_not_an_invented_comparison(self):
        value = report()
        value["objects"] = value["objects"][:1]
        row = self.resolve(value)["frames"][0]
        self.assertEqual([], row["comparisons"])
        self.assertEqual([], row["conflicting_track_ids"])

    def test_duplicate_frame_track_records_are_rejected(self):
        value = report()
        value["objects"].append(deepcopy(value["objects"][0]))
        with self.assertRaises(ValidationError):
            self.resolve(value)

    def test_invalid_report_binding_raises_before_mask_processing(self):
        with patch("proofloop.tracking_geometry.public_tracking_summary", side_effect=ValidationError("source changed")):
            with patch("proofloop.tracking_geometry.decode_sam_masks") as decoder:
                with self.assertRaises(ValidationError):
                    resolve_distractor_conflicts(report())
        decoder.assert_not_called()

    def test_unreviewed_or_boolean_target_ids_are_rejected(self):
        for identifier in (True, -1, "1", 2 ** 31, 3):
            with self.subTest(identifier=identifier), self.assertRaises(ValidationError):
                self.resolve(target_track_id=identifier)

    def test_pixel_budget_fallback_does_not_remove_overlapping_conflicts(self):
        with patch("proofloop.tracking_geometry.MAX_SOURCE_MAPPED_PIXELS", 1):
            row = self.resolve()["frames"][0]
        self.assertEqual([2], row["conflicting_track_ids"])
        self.assertEqual("unavailable", row["comparisons"][0]["mask_relation"])

    def test_mask_hashes_bind_geometry_placement_and_each_source_frame(self):
        value = report()
        first = self.resolve(value)["frames"][0]["comparisons"][0]
        for item in value["objects"]:
            item["frame"] = 5
            item["box"].update(x1=2, x2=5)
        result = self.resolve(value)
        moved = result["frames"][0]["comparisons"][0]
        self.assertEqual(first["target_raster_sha256"], moved["target_raster_sha256"])
        self.assertNotEqual(first["target_mapped_mask_sha256"], moved["target_mapped_mask_sha256"])
        self.assertEqual(5, result["frames"][0]["frame_index"])

    def test_public_projection_contains_no_raw_masks_paths_images_or_model_text(self):
        value = report()
        original = deepcopy(value)
        result = self.resolve(value)
        self.assertEqual(original, value)
        self.assertEqual(result, json.loads(json.dumps(result, allow_nan=False)))
        serialized = json.dumps(result)
        for secret in ("/private/", "secret-provider-content", "raster_base64", "authored_pixels", "encoded_payload"):
            self.assertNotIn(secret, serialized)


if __name__ == "__main__":
    unittest.main()
