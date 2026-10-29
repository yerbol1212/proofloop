"""Authored negative-only mask controls; no paid model calls or accuracy claims."""
from copy import deepcopy
import unittest
from unittest.mock import patch

from proofloop import image_gripper_veto as module
from proofloop.core import ValidationError
from proofloop.meta import SAM_MODEL, parse_sam_boxes
from tests.test_mask_semantics import decoded


SOURCE = "a" * 64


def result(boxes=((10, 10, 14, 14),), *, score=.95, ids=None):
    ids = ids or list(range(1, len(boxes) + 1))
    text = ""
    for identifier, (x1, y1, x2, y2) in zip(ids, boxes):
        confidence = f";c={score}" if score is not None else ""
        text += (f"<0f>{identifier}<|box;x1={x1};y1={y1};x2={x2};y2={y2};w=100;h=80{confidence}|>"
                 f"<|mask;x=0;y=0{confidence};data=5,5,!!!!!(QO(0lu8?|>")
    return {"source": {"sha256": SOURCE}, "response": {"model": SAM_MODEL, "status": "completed", "output_text": text},
            "text": text, "objects": decoded(parse_sam_boxes(text)), "mask_decoding": "official_meta_sam_parser_0.1.0"}


class ImageGripperVetoTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(patch.stopall)
        patch("proofloop.mask_semantics.decode_sam_masks", side_effect=decoded).start()
        patch("proofloop.tracking_geometry.decode_sam_masks", side_effect=decoded).start()
        self.target = result()

    def assess(self, gripper=None, **kwargs):
        return module.assess_frame(self.target, gripper if gripper is not None else result(((40, 40, 44, 44),)),
                                   source_sha256=SOURCE, width=100, height=80, target_object_id="cloth", **kwargs)

    def test_adjacent_gripper_masks_add_negative_warning(self):
        value = self.assess(result(((15, 10, 19, 14),)))
        self.assertTrue(value["veto"])
        self.assertTrue(value["geometry_available"])
        self.assertEqual("near_gripper_proposal", value["warning"])
        self.assertEqual(1, value["comparisons"][0]["min_chebyshev_distance_capped4"])

    def test_fixed_three_pixel_margin_includes_three_but_not_four(self):
        for x, veto in ((17, True), (18, False)):
            value = self.assess(result(((x, 10, x + 4, 14),)))
            self.assertEqual(veto, value["veto"])
            self.assertEqual(x - 14, value["comparisons"][0]["min_chebyshev_distance_capped4"])

    def test_clear_far_warning_never_grants_positive_support_or_separation(self):
        value = self.assess()
        self.assertEqual("no_near_proposal_observed", value["warning"])
        self.assertFalse(value["veto"])
        self.assertTrue(value["geometry_available"])
        self.assertEqual(4, value["comparisons"][0]["min_chebyshev_distance_capped4"])
        self.assertTrue(value["comparisons"][0]["distance_is_lower_bound"])
        for key in ("positive_support_can_be_granted", "separation_verified", "absence_verified",
                    "full_visibility_verified", "identity_independently_verified", "physical_contact_observed",
                    "physical_occlusion_observed", "acceptance_confidence_threshold_changed", "changes_production_acceptance"):
            self.assertFalse(value[key])

    def test_missing_gripper_detections_remain_unavailable_not_clear(self):
        value = self.assess(result(()))
        self.assertEqual("geometry_unavailable", value["warning"])
        self.assertFalse(value["geometry_available"])
        self.assertIn("gripper_missing_not_verified_absent", value["reason_codes"])

    def test_valid_weak_gripper_can_only_add_negative_risk(self):
        value = self.assess(result(((15, 10, 19, 14),), score=.01))
        self.assertTrue(value["veto"])
        self.assertEqual(.01, value["comparisons"][0]["gripper_confidence"])
        self.assertEqual(.95, value["comparisons"][0]["target_confidence"])
        self.assertFalse(value["positive_support_can_be_granted"])

    def test_missing_gripper_score_is_not_invented(self):
        value = self.assess(result(((15, 10, 19, 14),), score=None))
        self.assertTrue(value["veto"])
        self.assertIsNone(value["comparisons"][0]["gripper_confidence"])

    def test_multiple_grippers_any_near_preserves_all_instance_ids(self):
        value = self.assess(result(((40, 40, 44, 44), (15, 10, 19, 14)), ids=[7, 9]))
        self.assertTrue(value["veto"])
        self.assertEqual([7, 9], [row["gripper_sam_object_id"] for row in value["comparisons"]])

    def test_duplicate_gripper_instance_ids_do_not_become_clear(self):
        gripper = result(((40, 40, 44, 44), (15, 10, 19, 14)), ids=[7, 9])
        gripper["text"] = gripper["text"].replace("<0f>9", "<0f>7")
        gripper["response"]["output_text"] = gripper["text"]
        gripper["objects"][1]["object_id"] = 7
        value = self.assess(gripper)
        self.assertEqual("geometry_unavailable", value["warning"])
        self.assertIn("gripper_raw_mask_or_source_invalid", value["reason_codes"])

    def test_missing_or_multiple_targets_cannot_pick_preferred_target(self):
        for boxes in ((), ((10, 10, 14, 14), (50, 50, 54, 54))):
            self.target = result(boxes)
            value = self.assess()
            self.assertEqual("geometry_unavailable", value["warning"])
            self.assertIn("target_candidate_count_not_one", value["reason_codes"])

    def test_forged_mask_copy_raw_text_or_source_never_becomes_clear(self):
        mutations = (lambda r: r["objects"][0].update(confidence=.99),
                     lambda r: r.update(text="forged"),
                     lambda r: r["source"].update(sha256="b" * 64),
                     lambda r: r["response"].update(model="other"),
                     lambda r: r["objects"][0]["mask"].update(raster_sha256="0" * 64))
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                gripper = result()
                mutate(gripper)
                value = self.assess(gripper)
                self.assertEqual("geometry_unavailable", value["warning"])
                self.assertFalse(value["geometry_available"])

    def test_offsource_and_wrong_source_dimensions_never_create_projection(self):
        for text in (
            result()["text"].replace("x1=10", "x1=99").replace("x2=14", "x2=103"),
            result()["text"].replace("w=100", "w=101")):
            gripper = result()
            gripper["text"] = text
            try:
                gripper["objects"] = decoded(parse_sam_boxes(text))
            except ValueError:
                pass  # Retain a contradictory copy to exercise raw-parser rejection.
            gripper["response"]["output_text"] = text
            value = self.assess(gripper)
            self.assertEqual("geometry_unavailable", value["warning"])
            self.assertEqual([], value["comparisons"])

    def test_nonzero_frame_marker_cannot_be_used_as_still_image_evidence(self):
        gripper = result()
        text = gripper["text"].replace("<0f>", "<1f>")
        gripper.update(text=text, objects=decoded(parse_sam_boxes(text)))
        gripper["response"]["output_text"] = text
        value = self.assess(gripper)
        self.assertEqual("geometry_unavailable", value["warning"])
        self.assertEqual([], value["comparisons"])
        self.assertFalse(value["geometry_available"])

    def test_geometry_exception_text_cannot_escape_closed_reason_enum(self):
        with patch("proofloop.image_gripper_veto.tracking_geometry._mapped",
                   side_effect=ValueError("target_private_provider_payload")):
            value = self.assess()
        self.assertEqual(["mask_geometry_unavailable"], value["reason_codes"])
        self.assertNotIn("private_provider_payload", str(value))
        self.assertTrue(set(value["reason_codes"]).issubset(module.REASON_CODES))

    def test_bad_source_dimensions_or_semantic_identifier_rejected(self):
        for extra in ({"width": True}, {"width": 1281}, {"height": 0}, {"target_object_id": "../gripper"},
                      {"source_sha256": "not-a-hash"}):
            options = dict(source_sha256=SOURCE, width=100, height=80, target_object_id="cloth")
            options.update(extra)
            with self.assertRaises(ValidationError):
                module.assess_frame(self.target, result(), **options)

    def test_resource_limit_withholds_geometry_without_fabricating_far_distance(self):
        with patch.object(module, "MAX_SOURCE_MAPPED_PIXELS", 40):
            value = self.assess()
        self.assertEqual("geometry_unavailable", value["warning"])
        self.assertEqual([], value["comparisons"])
        self.assertIn("geometry_pixel_limit", value["reason_codes"])

    def test_partial_near_warning_survives_later_resource_limit(self):
        with patch.object(module, "MAX_SOURCE_MAPPED_PIXELS", 60):
            value = self.assess(result(((15, 10, 19, 14), (40, 40, 44, 44))))
        self.assertTrue(value["veto"])
        self.assertFalse(value["geometry_available"])
        self.assertEqual("near_gripper_proposal", value["warning"])

    def test_time_limit_is_unavailable_not_far(self):
        with patch("proofloop.image_gripper_veto.time.monotonic", side_effect=[0, 31]):
            value = self.assess()
        self.assertEqual("geometry_unavailable", value["warning"])
        self.assertIn("geometry_time_limit", value["reason_codes"])

    def test_summary_has_replay_receipts_without_raw_provider_text(self):
        gripper = result()
        gripper["response"]["private_marker"] = "do-not-export"
        value = self.assess(gripper)
        self.assertEqual(module.benchmark.manifest_hash(self.target), value["target_raw_result_sha256"])
        self.assertEqual(module.benchmark.manifest_hash(gripper), value["gripper_raw_result_sha256"])
        self.assertNotIn("do-not-export", str(value))
        receipt = value.pop("receipt_sha256")
        self.assertEqual(module.benchmark.manifest_hash(value), receipt)

    def test_nonfinite_or_excessive_wrapper_is_unavailable(self):
        gripper = result()
        gripper["response"]["malformed"] = float("nan")
        self.assertEqual("geometry_unavailable", self.assess(gripper)["warning"])
        with patch.object(module, "MAX_RESULT_BYTES", 10):
            self.assertEqual("geometry_unavailable", self.assess()["warning"])

    def test_inputs_and_target_score_remain_unchanged(self):
        target, gripper = deepcopy(self.target), result(((15, 10, 19, 14),), score=.1)
        original_gripper = deepcopy(gripper)
        self.assess(gripper)
        self.assertEqual(target, self.target)
        self.assertEqual(original_gripper, gripper)


if __name__ == "__main__":
    unittest.main()
