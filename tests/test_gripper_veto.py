"""Authored negative-only warning contracts; no model accuracy claims."""
from copy import deepcopy
import json
import unittest
from unittest.mock import patch

from proofloop import gripper_veto as veto
from proofloop import tracking_geometry as geometry
from proofloop.core import ValidationError
from tests.test_gripper_geometry import decode, record, report, summary


class GripperVetoTests(unittest.TestCase):
    def compare(self, target=None, grippers=None, *, decoder=decode, **kwargs):
        target = report() if target is None else target
        grippers = report(True) if grippers is None else grippers
        with patch.object(veto, "public_tracking_summary", side_effect=summary):
            with patch.object(geometry, "decode_sam_masks", side_effect=decoder):
                return veto.compare(target, grippers, **kwargs)

    def test_valid_weak_near_proposal_can_only_add_a_negative_warning(self):
        grippers = report(True)
        grippers["objects"][0] = record(3, 11, 10, confidence=.43)
        value = self.compare(grippers=grippers)
        row = value["frames"][0]
        self.assertTrue(row["veto"])
        self.assertEqual("near_gripper_proposal", row["warning"])
        self.assertEqual(1, row["comparisons"][0]["min_chebyshev_distance_capped4"])
        self.assertEqual(.43, row["comparisons"][0]["gripper_confidence"])
        self.assertFalse(value["positive_support_can_be_granted"])
        self.assertFalse(value["acceptance_confidence_threshold_changed"])

    def test_three_pixel_margin_is_unchanged_and_four_never_qualifies_clear(self):
        for distance in range(5):
            grippers = report(True)
            grippers["objects"][0] = record(3, 10 + distance, 10, confidence=.01)
            with self.subTest(distance=distance):
                row = self.compare(grippers=grippers)["frames"][0]
                self.assertEqual(distance <= 3, row["veto"])
                self.assertFalse(row["positive_support_can_be_granted"])
                self.assertFalse(row["separation_verified"])
                self.assertFalse(row["full_visibility_verified"])

    def test_all_far_masks_report_no_near_observation_without_clearance(self):
        value = self.compare()
        row = value["frames"][0]
        self.assertEqual("no_near_proposal_observed", row["warning"])
        self.assertFalse(row["veto"])
        self.assertTrue(row["additional_geometry_complete"])
        for key in ("positive_support_can_be_granted", "separation_verified", "absence_verified", "full_visibility_verified"):
            self.assertFalse(row[key])
        self.assertNotIn("status", row)
        self.assertNotIn("fully_visible", row)

    def test_low_far_confidence_does_not_become_positive_evidence_or_a_forced_veto(self):
        grippers = report(True)
        for item in grippers["objects"]:
            item["confidence"] = .12
        row = self.compare(grippers=grippers)["frames"][0]
        self.assertFalse(row["veto"])
        self.assertEqual("no_near_proposal_observed", row["warning"])
        self.assertFalse(row["positive_support_can_be_granted"])

    def test_unreported_confidence_is_retained_as_missing_without_positive_claims(self):
        grippers = report(True)
        grippers["objects"][0] = record(3, 11, 10, confidence=None)
        row = self.compare(grippers=grippers)["frames"][0]
        self.assertTrue(row["veto"])
        self.assertIsNone(row["comparisons"][0]["gripper_confidence"])
        self.assertFalse(row["full_visibility_verified"])

    def test_missing_proposals_are_geometry_unavailable_never_verified_absence(self):
        grippers = report(True)
        grippers["objects"] = []
        row = self.compare(grippers=grippers)["frames"][0]
        self.assertEqual("geometry_unavailable", row["warning"])
        self.assertFalse(row["additional_geometry_complete"])
        self.assertFalse(row["veto"])
        self.assertFalse(row["absence_verified"])

    def test_known_near_and_one_missing_still_warns_and_discloses_incompleteness(self):
        grippers = report(True)
        grippers["objects"] = [record(3, 10, 10, confidence=.2)]
        row = self.compare(grippers=grippers)["frames"][0]
        self.assertTrue(row["veto"])
        self.assertFalse(row["additional_geometry_complete"])
        self.assertEqual("geometry_unavailable", row["comparisons"][1]["warning"])

    def test_empty_invalid_or_unconfirmed_masks_do_not_fabricate_near_warnings(self):
        for mode in ("empty", "invalid", "unconfirmed"):
            grippers = report(True)
            grippers["objects"][0] = record(3, 10, 10, confidence=.4)
            if mode in {"empty", "invalid"}:
                grippers["objects"][0]["mask"]["authored_pixels"] = [0 if mode == "empty" else 2]
            else:
                grippers["objects"][0]["mask_supported"] = False
            with self.subTest(mode=mode):
                row = self.compare(grippers=grippers)["frames"][0]
                self.assertFalse(row["veto"])
                self.assertFalse(row["additional_geometry_complete"])
                self.assertEqual("geometry_unavailable", row["warning"])

    def test_failed_provider_or_missing_target_has_no_geometry_outcome_to_promote(self):
        for mode in ("failed_grippers", "failed_target", "missing_target"):
            target, grippers = report(), report(True)
            if mode == "failed_grippers":
                grippers.update(status="unscored", objects=[])
            else:
                target["objects"] = []
                if mode == "failed_target":
                    target["status"] = "unscored"
            with self.subTest(mode=mode):
                row = self.compare(target, grippers)["frames"][0]
                self.assertFalse(row["veto"])
                self.assertFalse(row["positive_support_can_be_granted"])
                self.assertEqual("geometry_unavailable", row["warning"])

    def test_official_raster_hash_and_source_mask_placement_are_required(self):
        def wrong_hash(records):
            result = decode(records)
            result[0]["mask"]["raster_sha256"] = "f" * 64
            return result
        row = self.compare(decoder=wrong_hash)["frames"][0]
        self.assertFalse(row["veto"])
        self.assertFalse(row["additional_geometry_complete"])
        target, grippers = report(), report(True)
        grippers["objects"][0] = record(3, 11, 10, confidence=.3)
        first = self.compare(target, grippers)["frames"][0]["comparisons"][0]
        grippers["objects"][0]["box"].update(x1=12, x2=12)
        second = self.compare(target, grippers)["frames"][0]["comparisons"][0]
        self.assertEqual(first["gripper_raster_sha256"], second["gripper_raster_sha256"])
        self.assertNotEqual(first["gripper_mapped_mask_sha256"], second["gripper_mapped_mask_sha256"])

    def test_mismatched_clip_dimensions_pts_and_requested_frames_raise(self):
        for mode in ("sha", "width", "pts", "requested"):
            grippers = report(True)
            if mode == "sha":
                grippers["prepared"]["source"]["sha256"] = "f" * 64
            elif mode == "width":
                grippers["prepared"]["source"]["width"] = 33
            elif mode == "pts":
                grippers["prepared"]["source"]["frames"][0]["pts"] = 1
            else:
                grippers["prepared"]["requested_frame_indices"] = []
            with self.subTest(mode=mode), self.assertRaises(ValidationError):
                self.compare(grippers=grippers)

    def test_unreviewed_repeated_or_wrong_type_track_ids_raise(self):
        for kwargs in ({"target_track_id": True}, {"target_track_id": 5}, {"gripper_track_ids": (3, 5)},
                {"gripper_track_ids": (3, 3)}, {"gripper_track_ids": (1, 3)}, {"gripper_track_ids": ()}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValidationError):
                self.compare(**kwargs)

    def test_report_validation_precedes_any_decoded_mask_use(self):
        with patch.object(veto, "public_tracking_summary", side_effect=ValidationError("invalid report")):
            with patch.object(geometry, "decode_sam_masks") as decoder:
                with self.assertRaises(ValidationError):
                    veto.compare(report(), report(True))
        decoder.assert_not_called()

    def test_pixel_and_time_budget_exhaustion_produces_no_positive_qualification(self):
        with patch.object(veto, "MAX_SOURCE_MAPPED_PIXELS", 1):
            row = self.compare()["frames"][0]
            self.assertEqual("geometry_unavailable", row["warning"])
            self.assertFalse(row["positive_support_can_be_granted"])
        with patch.object(veto.time, "monotonic", side_effect=[0] + [100] * 100):
            row = self.compare()["frames"][0]
            self.assertFalse(row["additional_geometry_complete"])
            self.assertFalse(row["veto"])

    def test_safe_public_projection_does_not_mutate_private_evidence(self):
        target, grippers = report(), report(True)
        expected = deepcopy([target, grippers])
        value = self.compare(target, grippers)
        self.assertEqual(expected, [target, grippers])
        encoded = json.dumps(value, allow_nan=False)
        for private in ("/Users/", "secret-provider-payload", "raster_base64", "authored_pixels", "encoded_payload"):
            self.assertNotIn(private, encoded)
        self.assertFalse(value["changes_live_placement_acceptance"])
        self.assertFalse(value["physical_action_authorized"])


if __name__ == "__main__":
    unittest.main()
