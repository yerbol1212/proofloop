"""Frozen synthetic score fixtures; no inference or model-quality assertions."""
from copy import deepcopy
import json
import unittest

from proofloop.core import ValidationError
from proofloop.tracking_evaluation import METRIC_VERSION, score_frames


TARGET = [.1, .1, .3, .3]
DISTRACTOR = [.7, .7, .9, .9]


def reference(clip="clip-a", frame=0, *, expected="supported", target=None, condition="clear"):
    return {"clip_id": clip, "frame_index": frame, "expected": expected,
            "target_box": deepcopy(TARGET) if target is None and expected == "supported" else deepcopy(target),
            "distractor_boxes": [deepcopy(DISTRACTOR)], "condition": condition}


def prediction(clip="clip-a", frame=0, *, status="supported", box=None):
    return {"clip_id": clip, "frame_index": frame, "status": status,
            "box": deepcopy(TARGET) if box is None and status == "supported" else deepcopy(box)}


class TrackingEvaluationTests(unittest.TestCase):
    def test_correct_positive_requires_label_and_localization(self):
        result = score_frames([reference()], [prediction()])
        metrics, frame = result["metrics"], result["frames"][0]
        self.assertEqual(METRIC_VERSION, result["metric_version"])
        self.assertEqual((1, 1, 0), (metrics["labels_total"], metrics["labels_completed"], metrics["labels_unscored"]))
        self.assertEqual((1, 1, 1), (metrics["positive_count"], metrics["positives_retained"], metrics["supported_correct"]))
        self.assertEqual((1, 1, 1), (metrics["coverage"], metrics["selective_accuracy"], metrics["decision_agreement"]))
        self.assertEqual((1, 0), (frame["target_iou"], frame["distractor_alignment"]))
        self.assertEqual("correct_support", frame["classification"])
        self.assertFalse(result["identity_switches_measured"])
        self.assertFalse(result["physical_action_authorized"])

    def test_high_overlap_with_a_distractor_does_not_earn_target_credit(self):
        result = score_frames([reference()], [prediction(box=DISTRACTOR)])
        metrics, frame = result["metrics"], result["frames"][0]
        self.assertEqual((0, 1), (frame["target_iou"], frame["distractor_alignment"]))
        self.assertEqual("wrong_localization_on_positive", frame["classification"])
        self.assertEqual(1, metrics["wrong_localization_on_positive"])
        self.assertEqual(1, metrics["supported_distractor_alignment_at_threshold"])
        self.assertEqual((0, 0, 1), (metrics["selective_accuracy"], metrics["decision_agreement"], metrics["decision_label_agreement"]))

    def test_foreign_clip_target_box_is_scored_against_its_exact_own_reference(self):
        refs = [reference("clip-a"), reference("clip-b", target=DISTRACTOR)]
        preds = [prediction("clip-b", box=TARGET), prediction("clip-a", box=DISTRACTOR)]
        result = score_frames(refs, preds)
        self.assertEqual(0, result["metrics"]["positives_retained"])
        self.assertEqual(2, result["metrics"]["wrong_localization_on_positive"])
        self.assertTrue(all(row["target_iou"] == 0 for row in result["frames"]))

    def test_support_on_unknown_is_unsupported_even_with_target_overlap(self):
        ref = reference(expected="unknown", target=TARGET, condition="partial_occlusion")
        result = score_frames([ref], [prediction()])
        self.assertEqual(1, result["frames"][0]["target_iou"])
        self.assertEqual("unsupported_support_on_unknown", result["frames"][0]["classification"])
        self.assertEqual(1, result["metrics"]["unsupported_support_on_unknown"])
        self.assertEqual(0, result["metrics"]["supported_correct"])
        self.assertEqual(0, result["metrics"]["selective_accuracy"])
        self.assertIsNone(result["metrics"]["positive_retention"])

    def test_hidden_reference_does_not_invent_an_iou(self):
        ref = reference(expected="unknown", condition="hidden_or_unidentifiable")
        result = score_frames([ref], [prediction()])
        self.assertIsNone(result["frames"][0]["target_iou"])
        self.assertIsNone(result["frames"][0]["target_localization_at_threshold"])
        self.assertEqual(1, result["metrics"]["unsupported_support_on_unknown"])

    def test_all_unknown_has_zero_support_coverage_despite_high_label_agreement(self):
        refs = [reference(frame=i, expected="unknown" if i < 7 else "supported") for i in range(8)]
        preds = [prediction(frame=i, status="unknown") for i in range(8)]
        result = score_frames(refs, preds)
        metrics = result["metrics"]
        self.assertEqual(0, metrics["coverage"])
        self.assertEqual(.875, metrics["decision_agreement"])
        self.assertEqual(.875, metrics["decision_label_agreement"])
        self.assertEqual(1, metrics["abstentions_on_positive"])
        self.assertEqual(7, metrics["unknowns_withheld"])
        self.assertIsNone(metrics["selective_accuracy"])

    def test_unscored_is_neither_unknown_agreement_nor_positive_abstention(self):
        refs = [reference(frame=0, expected="unknown"), reference(frame=1)]
        preds = [prediction(frame=i, status="unscored") for i in range(2)]
        metrics = score_frames(refs, preds)["metrics"]
        self.assertEqual((2, 0, 2), (metrics["labels_total"], metrics["labels_completed"], metrics["labels_unscored"]))
        self.assertEqual((0, 0, 0), (metrics["unknowns_withheld"], metrics["abstentions_on_positive"], metrics["decision_agreement_count"]))
        self.assertEqual((1, 1), (metrics["positive_unscored"], metrics["unknown_unscored"]))
        self.assertEqual(0, metrics["decision_agreement"])
        self.assertEqual(0, metrics["coverage"])
        self.assertIsNone(metrics["selective_accuracy"])
        self.assertIsNone(metrics["completed_decision_agreement"])

    def test_missing_completion_remains_in_all_frame_denominators(self):
        refs = [reference(frame=i) for i in range(2)]
        preds = [prediction(), prediction(frame=1, status="unscored")]
        metrics = score_frames(refs, preds)["metrics"]
        self.assertEqual(.5, metrics["coverage"])
        self.assertEqual(1, metrics["completed_support_coverage"])
        self.assertEqual(.5, metrics["positive_retention"])
        self.assertEqual(.5, metrics["decision_agreement"])
        self.assertEqual(1, metrics["selective_accuracy"])

    def test_unknown_diagnostic_box_gets_no_support_credit(self):
        result = score_frames([reference()], [prediction(status="unknown", box=TARGET)])
        frame, metrics = result["frames"][0], result["metrics"]
        self.assertEqual(1, frame["target_iou"])
        self.assertEqual("abstention_on_positive", frame["classification"])
        self.assertEqual(0, metrics["supported_count"])
        self.assertEqual(0, metrics["positives_retained"])

    def test_frozen_threshold_is_inclusive_and_not_retuned(self):
        ref = reference(target=[0, 0, .5, .5])
        pred = prediction(box=[0, 0, .5, .25])
        accepted = score_frames([ref], [pred], iou_threshold=.5)
        refused = score_frames([ref], [pred], iou_threshold=.50001)
        self.assertEqual(.5, accepted["frames"][0]["target_iou"])
        self.assertEqual(1, accepted["metrics"]["positives_retained"])
        self.assertEqual(0, refused["metrics"]["positives_retained"])
        self.assertEqual(.50001, refused["iou_threshold"])

    def test_exclusive_touching_edges_have_zero_intersection(self):
        result = score_frames([reference(target=[0, 0, .5, 1])], [prediction(box=[.5, 0, 1, 1])])
        self.assertEqual(0, result["frames"][0]["target_iou"])
        self.assertEqual(1, result["metrics"]["wrong_localization_on_positive"])

    def test_no_distractors_is_unavailable_alignment_not_zero_evidence(self):
        ref = reference()
        ref["distractor_boxes"] = []
        frame = score_frames([ref], [prediction()])["frames"][0]
        self.assertIsNone(frame["distractor_alignment"])
        self.assertIsNone(frame["distractor_alignment_at_threshold"])

    def test_multiple_distractors_use_maximum_overlap_only(self):
        ref = reference()
        ref["distractor_boxes"] = [[0, 0, .1, .1], DISTRACTOR, [.95, .95, 1, 1]]
        frame = score_frames([ref], [prediction(box=DISTRACTOR)])["frames"][0]
        self.assertEqual(1, frame["distractor_alignment"])
        self.assertFalse(score_frames([ref], [prediction(box=DISTRACTOR)])["identity_switches_measured"])

    def test_two_clips_eight_checkpoints_have_explicit_group_denominators(self):
        refs, preds = [], []
        for clip in ("screwdriver-e01", "remote-e01"):
            for i in range(8):
                unknown = i in (2, 3, 4)
                condition = "partial_occlusion" if unknown else "reexposed" if i > 4 else "clear"
                refs.append(reference(clip, i * 5, expected="unknown" if unknown else "supported",
                                      target=TARGET, condition=condition))
                preds.append(prediction(clip, i * 5, status="unknown" if unknown else "supported"))
        result = score_frames(refs, list(reversed(preds)))
        self.assertEqual(16, result["metrics"]["labels_total"])
        self.assertEqual((10, 6), (result["metrics"]["positive_count"], result["metrics"]["unknown_count"]))
        self.assertEqual(1, result["metrics"]["decision_agreement"])
        self.assertEqual(.625, result["metrics"]["coverage"])
        self.assertEqual([8, 8], [group["metrics"]["labels_total"] for group in result["by_clip"]])
        self.assertEqual(16, sum(group["metrics"]["labels_total"] for group in result["by_condition"]))

    def test_prediction_order_does_not_change_results_and_inputs_are_not_mutated(self):
        refs = [reference(frame=i) for i in range(2)]
        preds = [prediction(frame=i) for i in range(2)]
        original = deepcopy((refs, preds))
        result = score_frames(refs, preds)
        self.assertEqual(result, score_frames(list(reversed(refs)), list(reversed(preds))))
        self.assertEqual(original, (refs, preds))
        result["frames"][0]["prediction_box"][0] = .01
        self.assertEqual(original, (refs, preds))

    def test_duplicate_reference_or_prediction_key_is_rejected(self):
        for refs, preds in (([reference(), reference()], [prediction()]),
                            ([reference()], [prediction(), prediction()])):
            with self.subTest(refs=refs, preds=preds), self.assertRaises(ValidationError):
                score_frames(refs, preds)

    def test_exact_clip_and_frame_coverage_is_required(self):
        variants = [[prediction(frame=1)], [prediction("foreign-clip")], [],
                    [prediction(), prediction(frame=1)]]
        for preds in variants:
            with self.subTest(preds=preds), self.assertRaises(ValidationError):
                score_frames([reference()], preds)

    def test_reference_and_prediction_rows_are_closed_allowlists(self):
        for field in ("path", "image", "provider_response", "annotation_hint", "secret"):
            ref, pred = reference(), prediction()
            ref[field] = "/private/raw-provider"
            with self.subTest(field=field, row="reference"), self.assertRaises(ValidationError):
                score_frames([ref], [prediction()])
            pred[field] = "/private/raw-provider"
            with self.subTest(field=field, row="prediction"), self.assertRaises(ValidationError):
                score_frames([reference()], [pred])
        for key in tuple(reference()):
            ref = reference()
            del ref[key]
            with self.subTest(missing=key), self.assertRaises(ValidationError):
                score_frames([ref], [prediction()])

    def test_identifiers_and_frame_indices_cannot_contain_paths_or_boolean_numbers(self):
        for identifier in ("", "/private/video.mp4", "../video", "https://provider/secret", "x" * 129, None, True):
            ref, pred = reference(), prediction()
            ref["clip_id"] = pred["clip_id"] = identifier
            with self.subTest(identifier=identifier), self.assertRaises(ValidationError):
                score_frames([ref], [pred])
        for index in (True, -1, 1.0, "0", 2 ** 31, None):
            ref, pred = reference(), prediction()
            ref["frame_index"] = pred["frame_index"] = index
            with self.subTest(index=index), self.assertRaises(ValidationError):
                score_frames([ref], [pred])

    def test_invalid_geometry_is_rejected_at_all_three_box_locations(self):
        bad_boxes = [[], [.1, .2, .3], [.1, .2, .3, .4, .5], (.1, .1, .3, .3),
                     [0, 0, 0, 1], [.3, .1, .1, .3], [-.1, 0, .3, .3], [0, 0, 1.1, 1],
                     [False, 0, .3, .3], [0, 0, float("nan"), .3], [0, 0, float("inf"), .3],
                     [0, 0, 10 ** 1000, .3], [0, 0, 1e-300, 1e-300], ["0", 0, .3, .3], {}]
        for box in bad_boxes:
            for field in ("target", "prediction", "distractor"):
                ref, pred = reference(), prediction()
                if field == "target":
                    ref["target_box"] = box
                elif field == "prediction":
                    pred["box"] = box
                else:
                    ref["distractor_boxes"] = [box]
                with self.subTest(box=box, field=field), self.assertRaises(ValidationError):
                    score_frames([ref], [pred])

    def test_supported_box_is_required_and_unscored_cannot_invent_a_box(self):
        ref, pred = reference(), prediction()
        ref["target_box"] = None
        with self.assertRaises(ValidationError):
            score_frames([ref], [pred])
        pred["box"] = None
        with self.assertRaises(ValidationError):
            score_frames([reference()], [pred])
        with self.assertRaises(ValidationError):
            score_frames([reference()], [prediction(status="unscored", box=TARGET)])

    def test_hidden_condition_cannot_claim_supported_or_supply_invented_target_box(self):
        for ref in (reference(condition="hidden_or_unidentifiable"),
                    reference(expected="unknown", target=TARGET, condition="hidden_or_unidentifiable")):
            with self.subTest(ref=ref), self.assertRaises(ValidationError):
                score_frames([ref], [prediction()])

    def test_invalid_status_condition_threshold_and_collection_types_are_rejected(self):
        for value in (True, None, "pass", [], "unscored"):
            ref = reference()
            ref["expected"] = value
            with self.subTest(expected=value), self.assertRaises(ValidationError):
                score_frames([ref], [prediction()])
        for value in (True, None, "pass", []):
            pred = prediction()
            pred["status"] = value
            with self.subTest(status=value), self.assertRaises(ValidationError):
                score_frames([reference()], [pred])
        for value in ("motion", None, True, []):
            ref = reference()
            ref["condition"] = value
            with self.subTest(condition=value), self.assertRaises(ValidationError):
                score_frames([ref], [prediction()])
        for threshold in (0, -.1, 1.1, True, None, "0.3", float("nan"), float("inf"), 10 ** 1000):
            with self.subTest(threshold=threshold), self.assertRaises(ValidationError):
                score_frames([reference()], [prediction()], iou_threshold=threshold)
        for references, predictions in (([], []), (None, []), ((reference(),), [prediction()]),
                                        ([reference()], None), ([reference()], (prediction(),))):
            with self.subTest(references=references, predictions=predictions), self.assertRaises(ValidationError):
                score_frames(references, predictions)

    def test_result_is_finite_json_and_never_echoes_source_or_model_content(self):
        result = score_frames([reference()], [prediction()])
        encoded = json.dumps(result, allow_nan=False)
        self.assertEqual(result, json.loads(encoded))
        for field in ("path", "image", "provider", "prompt", "response"):
            self.assertNotIn(field + '":', encoded)
        self.assertEqual("normalized_xyxy_exclusive", result["box_coordinates"])


if __name__ == "__main__":
    unittest.main()
