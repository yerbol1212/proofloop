"""Authored policy fixtures, not measurements of model/video accuracy."""
from copy import deepcopy
import json
import math
import unittest

from proofloop.core import ValidationError
from proofloop.confirmation_policy import CONFIRMATION_POLICY_VERSION, evaluate_tracking
from proofloop.tracking_policy import evaluate_tracking as old_evaluate


CLIP = "a" * 64


def candidate(*, track_id=1, **kwargs):
    return {"track_id": track_id, "object_id": "target", "box": [.1, .1, .3, .3],
            "confidence": .95, "mask_supported": True, "fully_visible": True,
            "identity_ambiguous": False, "semantic": None, **kwargs}


def frame(index, *, timestamp=None, item=True, assess=True, **kwargs):
    result = {"frame_index": index, "timestamp_ms": index * 500 if timestamp is None else timestamp,
              "source_sha256": CLIP, "frame_sha256": f"{index + 1:064x}",
              "candidates": [candidate()] if item is True else [] if item is False else [item],
              "reviewed_distractors": [], **kwargs}
    if assess:
        for chosen in result["candidates"]:
            if chosen["fully_visible"] is True and chosen["identity_ambiguous"] is False:
                attach_semantics(result, chosen)
    return result


def attach_semantics(value, chosen=None, decision="supported"):
    chosen = value["candidates"][0] if chosen is None else chosen
    chosen["semantic"] = {"target_id": "target", "track_id": chosen["track_id"],
        "frame_index": value["frame_index"], "source_sha256": value["source_sha256"],
        "frame_sha256": value["frame_sha256"], "candidate_box": list(chosen["box"]),
        "decision": decision, "evidence_sha256": f"{10000 + value['frame_index']:064x}"}
    return value


def seed(first, distractors=None):
    return {"target_id": "target", "track_id": 1, "frame_index": first["frame_index"],
            "source_sha256": first["source_sha256"], "frame_sha256": first["frame_sha256"],
            "review_sha256": "b" * 64, "reviewed_distractor_track_ids": distractors or []}


def run(frames, *, use_seed=True, **kwargs):
    return evaluate_tracking(frames, target_id="target", seed=seed(frames[0]) if use_seed else None, **kwargs)


class ConfirmationPolicyTests(unittest.TestCase):
    def test_dense_sampling_does_not_displace_earlier_qualifying_view(self):
        frames = [frame(0, timestamp=0), frame(1, timestamp=600), frame(2, timestamp=1000)]
        before = old_evaluate(frames, target_id="target", seed=seed(frames[0]))
        self.assertEqual(["unknown", "supported", "unknown"], [row["status"] for row in before["frames"]])
        result = run(frames)
        self.assertEqual(["unknown", "supported", "supported"], [row["status"] for row in result["frames"]])
        row = result["frames"][-1]
        self.assertEqual([0, 2], [item["frame_index"] for item in row["confirmation_evidence"]])
        self.assertTrue(row["confirmation_ends_at_current"])
        self.assertNotIn("source_spacing_insufficient", row["gap_codes"])

    def test_current_view_not_previous_support_must_be_clear(self):
        result = run([frame(0), frame(1), frame(2, item=candidate(fully_visible=False))])
        row = result["frames"][-1]
        self.assertEqual("unknown", result["status"])
        self.assertEqual("evidence_gap", row["lifecycle"])
        self.assertIn("full_visibility_not_supported", row["gap_codes"])
        self.assertEqual([], row["confirmation_evidence"])
        self.assertTrue(row["confirmation_history_reset"])

    def test_insufficient_spacing_still_unknown(self):
        result = run([frame(0, timestamp=0), frame(1, timestamp=400)])
        self.assertEqual("unknown", result["status"])
        self.assertEqual(1, result["frames"][-1]["confirmation_views"])
        self.assertIn("source_spacing_insufficient", result["frames"][-1]["gap_codes"])
        self.assertEqual([1], [x["frame_index"] for x in result["frames"][-1]["confirmation_evidence"]])

    def test_spacing_equality_qualifies(self):
        self.assertEqual("supported", run([frame(0, timestamp=0), frame(1, timestamp=500)])["status"])
        self.assertEqual("unknown", run([frame(0, timestamp=0), frame(1, timestamp=499)])["status"])

    def test_dense_prefix_can_later_form_chain_using_earliest_view(self):
        frames = [frame(i, timestamp=t) for i, t in enumerate([0, 100, 200, 400, 500, 600, 700, 900, 1000])]
        result = run(frames, min_stable_views=3)
        self.assertEqual("supported", result["status"])
        self.assertEqual([0, 4, 8], [v["frame_index"] for v in result["frames"][-1]["confirmation_evidence"]])
        self.assertEqual(1, result["metrics"]["supported_frames"])

    def test_stable_counts_two_through_eight_keep_exact_minimum(self):
        for stable in range(2, 9):
            with self.subTest(stable=stable):
                result = run([frame(i) for i in range(stable)], min_stable_views=stable)
                self.assertEqual(["unknown"] * (stable - 1) + ["supported"],
                                 [r["status"] for r in result["frames"]])
                self.assertEqual(stable, result["frames"][-1]["confirmation_views"])
                self.assertEqual(list(range(stable)),
                                 [v["frame_index"] for v in result["frames"][-1]["confirmation_evidence"]])

    def test_stable_three_dense_final_uses_earlier_chain(self):
        frames = [frame(i, timestamp=t) for i, t in enumerate([0, 500, 1100, 1500])]
        result = run(frames, min_stable_views=3)
        self.assertEqual(["unknown", "unknown", "supported", "supported"], [r["status"] for r in result["frames"]])
        self.assertEqual([0, 1, 3], [r["frame_index"] for r in result["frames"][-1]["confirmation_evidence"]])

    def test_zero_spacing_still_requires_distinct_source_views(self):
        frames = [frame(0, timestamp=0), frame(1, timestamp=1)]
        self.assertEqual("supported", run(frames, min_spacing_ms=0)["status"])
        frames[1]["frame_sha256"] = frames[0]["frame_sha256"]
        attach_semantics(frames[1])
        self.assertEqual("unknown", run(frames, min_spacing_ms=0)["status"])

    def test_missing_frame_resets_and_two_new_semantic_views_reacquire(self):
        result = run([frame(0), frame(1), frame(2, item=False), frame(3), frame(4)])
        self.assertEqual(["unknown", "supported", "unknown", "unknown", "supported"],
                         [row["status"] for row in result["frames"]])
        self.assertEqual([3, 4], [v["frame_index"] for v in result["frames"][-1]["confirmation_evidence"]])
        self.assertEqual(1, result["metrics"]["reacquisition_events"])
        self.assertEqual(1, result["metrics"]["hard_gap_events"])

    def test_all_legacy_hard_gates_reset_confirmation_history(self):
        for update, code in (({"confidence": .7999}, "low_confidence"), ({"confidence": None}, "score_unreported"),
                             ({"mask_supported": False}, "mask_unsupported"),
                             ({"box": [0, .1, .3, .3]}, "frame_boundary_unconfirmed"),
                             ({"fully_visible": False}, "full_visibility_not_supported"),
                             ({"identity_ambiguous": True}, "identity_ambiguous")):
            with self.subTest(update=update):
                frames = [frame(0), frame(1), frame(2, item=candidate(**update)), frame(3)]
                result = run(frames)
                self.assertIn(code, result["frames"][2]["gap_codes"])
                self.assertEqual(0, result["frames"][2]["confirmation_history_size"])
                self.assertEqual("unknown", result["status"])
                self.assertEqual([3], [v["frame_index"] for v in result["frames"][-1]["confirmation_evidence"]])

    def test_semantic_uncertainty_and_contradiction_are_hard_gaps(self):
        for decision in ("unknown", "contradicted"):
            frames = [frame(0), frame(1), attach_semantics(frame(2), decision=decision), frame(3)]
            with self.subTest(decision=decision):
                result = run(frames)
                self.assertEqual("unknown", result["status"])
                self.assertIn("semantic_identity_" + decision, result["frames"][2]["gap_codes"])
                self.assertTrue(result["frames"][2]["confirmation_history_reset"])

    def test_reviewed_distractor_overlap_resets_but_touching_does_not(self):
        for box, expected in (([.2, .2, .4, .4], "unknown"), ([.3, .1, .5, .3], "supported")):
            frames = [frame(0), frame(1, reviewed_distractors=[{"track_id": 2, "box": box, "origin": "runtime_sam_track"}])]
            result = evaluate_tracking(frames, target_id="target", seed=seed(frames[0], [2]))
            self.assertEqual(expected, result["status"])

    def test_track_change_resets_but_changed_semantic_view_starts_new_chain(self):
        frames = [frame(0), frame(1), frame(2, item=candidate(track_id=8)), frame(3, item=candidate(track_id=8))]
        result = run(frames)
        self.assertEqual(["unknown", "supported", "unknown", "supported"], [r["status"] for r in result["frames"]])
        self.assertEqual(1, result["frames"][2]["confirmation_views"])
        self.assertTrue(result["frames"][2]["confirmation_history_reset"])
        self.assertEqual(1, result["metrics"]["track_identifier_changes"])
        self.assertEqual(1, result["metrics"]["hard_gap_events"])
        self.assertEqual(1, result["metrics"]["reacquisition_events"])
        self.assertEqual([2, 3], [r["frame_index"] for r in result["frames"][-1]["confirmation_evidence"]])

    def test_unassessed_visibility_remains_unknown_without_erasing_history(self):
        frames = [frame(0), frame(1, item=candidate(fully_visible=None, identity_ambiguous=None)), frame(2)]
        result = run(frames)
        self.assertEqual(["unknown", "unknown", "supported"], [r["status"] for r in result["frames"]])
        row = result["frames"][1]
        self.assertEqual(1, row["confirmation_views"])
        self.assertFalse(row["confirmation_ends_at_current"])
        self.assertFalse(row["confirmation_history_reset"])

    def test_semantic_null_never_supplies_current_or_seed_proof(self):
        for use_seed in (True, False):
            frames = [frame(0, assess=False), frame(1, assess=False)]
            result = run(frames, use_seed=use_seed)
            with self.subTest(seed=use_seed):
                self.assertEqual("unknown", result["status"])
                self.assertEqual(0, result["frames"][-1]["confirmation_history_size"])
                self.assertTrue(all("current_semantic_evidence_required" in r["gap_codes"] for r in result["frames"]))

    def test_semantic_null_after_support_preserves_history_but_current_unknown(self):
        result = run([frame(0), frame(1), frame(2, assess=False), frame(3)])
        self.assertEqual(["unknown", "supported", "unknown", "supported"], [r["status"] for r in result["frames"]])
        self.assertEqual(2, result["frames"][2]["confirmation_views"])
        self.assertEqual(2, result["frames"][2]["confirmation_history_size"])
        self.assertFalse(result["frames"][2]["confirmation_ends_at_current"])
        self.assertEqual(0, result["metrics"]["hard_gap_frames"])

    def test_unseeded_semantic_chain_is_allowed_but_not_reacquisition_event(self):
        result = run([frame(0), frame(1)], use_seed=False)
        self.assertEqual("supported", result["status"])
        self.assertEqual("runtime_semantic", result["frames"][-1]["identity_basis"])
        self.assertEqual(0, result["metrics"]["reacquisition_events"])

    def test_repeated_current_jpeg_is_unknown_without_reset_or_count(self):
        repeated = frame(2, frame_sha256=frame(0)["frame_sha256"])
        result = run([frame(0), frame(1), repeated, frame(3)])
        row = result["frames"][2]
        self.assertEqual("unknown", row["status"])
        self.assertIn("source_view_reused", row["gap_codes"])
        self.assertFalse(row["confirmation_history_reset"])
        self.assertFalse(row["confirmation_ends_at_current"])
        self.assertEqual(2, row["confirmation_history_size"])
        self.assertEqual("supported", result["status"])

    def test_dense_canonical_jpeg_cannot_reenter_after_spacing(self):
        first, dense = frame(0, timestamp=0), frame(1, timestamp=100)
        repeated = frame(2, timestamp=600, frame_sha256=dense["frame_sha256"])
        result = run([first, dense, repeated])
        self.assertEqual("unknown", result["status"])
        self.assertIn("source_view_reused", result["frames"][-1]["gap_codes"])

    def test_same_jpeg_can_begin_fresh_chain_after_hard_gap_but_not_twice(self):
        frames = [frame(0), frame(1), frame(2, item=False),
                  frame(3, frame_sha256=frame(0)["frame_sha256"]),
                  frame(4, frame_sha256=frame(0)["frame_sha256"]), frame(5)]
        result = run(frames)
        self.assertEqual("unknown", result["frames"][4]["status"])
        self.assertEqual("supported", result["status"])
        self.assertEqual([3, 5], [r["frame_index"] for r in result["frames"][-1]["confirmation_evidence"]])

    def test_multiple_candidates_require_positive_one_and_explicit_exclusion_of_others(self):
        ambiguous = frame(1, candidates=[candidate(), candidate(track_id=2)], assess=False)
        self.assertEqual("unknown", run([frame(0), ambiguous])["status"])
        attach_semantics(ambiguous, ambiguous["candidates"][0])
        attach_semantics(ambiguous, ambiguous["candidates"][1], "contradicted")
        self.assertEqual("supported", run([frame(0), ambiguous])["status"])

    def test_no_gap_or_tracking_claims_authorize_physical_action(self):
        result = run([frame(0), frame(1), frame(2, item=False)])
        self.assertFalse(result["changes_live_placement_acceptance"])
        for row in result["frames"]:
            self.assertFalse(row["physical_action_authorized"])
            self.assertFalse(row["physical_occlusion_observed"])
            self.assertFalse(row["identity_switch_observed"])

    def test_legacy_argument_validation_and_bounds_are_preserved(self):
        for change in ({"min_confidence": True}, {"min_confidence": -1}, {"min_confidence": 1.1},
                       {"min_confidence": math.nan}, {"min_stable_views": 1}, {"min_stable_views": 9},
                       {"min_stable_views": True}, {"min_spacing_ms": -1}, {"min_spacing_ms": 60001},
                       {"min_spacing_ms": False}):
            with self.subTest(change=change), self.assertRaises(ValidationError):
                run([frame(0), frame(1)], **change)
        for value in (None, [], {}, [frame(0)] * 4097):
            with self.subTest(value_type=type(value)), self.assertRaises(ValidationError):
                evaluate_tracking(value, target_id="target")
        with self.assertRaises(ValidationError):
            evaluate_tracking([frame(0)], target_id="../target")
        self.assertEqual("supported", run([frame(0), frame(1)], min_confidence=0)["status"])

    def test_closed_input_contract_rejects_reference_labels_at_every_level(self):
        for level in ("frame", "candidate", "semantic", "seed"):
            frames = [frame(0), frame(1)]
            review = seed(frames[0])
            place = {"frame": frames[0], "candidate": frames[0]["candidates"][0],
                     "semantic": frames[0]["candidates"][0]["semantic"], "seed": review}[level]
            place["ground_truth_box"] = [.1, .1, .3, .3]
            with self.subTest(level=level), self.assertRaises(ValidationError):
                evaluate_tracking(frames, target_id="target", seed=review)

    def test_semantic_binding_is_validated_against_exact_current_pixels(self):
        for change in ({"target_id": "other"}, {"track_id": 7}, {"track_id": True}, {"frame_index": 22},
                       {"source_sha256": "c" * 64}, {"frame_sha256": "d" * 64},
                       {"candidate_box": [.2, .2, .4, .4]}, {"evidence_sha256": "invalid"}, {"decision": "clear"}):
            frames = [frame(0), frame(1)]
            frames[1]["candidates"][0]["semantic"].update(change)
            with self.subTest(change=change), self.assertRaises(ValidationError):
                run(frames)

    def test_supported_semantics_cannot_override_visibility_or_ambiguity(self):
        for change in ({"fully_visible": None}, {"fully_visible": False},
                       {"identity_ambiguous": True}, {"identity_ambiguous": None}):
            frames = [frame(0), frame(1)]
            frames[1]["candidates"][0].update(change)
            with self.subTest(change=change), self.assertRaises(ValidationError):
                run(frames)

    def test_seed_and_runtime_distractor_review_binding_are_validated(self):
        for change in ({"target_id": "other"}, {"track_id": True}, {"frame_index": 9},
                       {"source_sha256": "c" * 64}, {"frame_sha256": "d" * 64}, {"review_sha256": "bad"},
                       {"reviewed_distractor_track_ids": [1]}, {"reviewed_distractor_track_ids": [2, 2]}):
            frames = [frame(0), frame(1)]
            with self.subTest(change=change), self.assertRaises(ValidationError):
                evaluate_tracking(frames, target_id="target", seed={**seed(frames[0]), **change})
        for origin in ("runtime_sam_track", "ground_truth"):
            frames = [frame(0), frame(1, reviewed_distractors=[{"track_id": 2, "box": [.5, .5, .7, .7], "origin": origin}])]
            with self.subTest(origin=origin), self.assertRaises(ValidationError):
                run(frames)

    def test_candidate_numbers_boxes_and_flags_are_validated(self):
        updates = [{"confidence": value} for value in (True, -1, 1.1, math.inf, math.nan, "0.9")]
        updates += [{"box": value} for value in ([0, 0, 0, 1], [.3, .1, .2, .4], [0, 0, 2, 1], [False, .1, .3, .3])]
        updates += [{"track_id": True}, {"mask_supported": "true"}, {"fully_visible": "true"}, {"object_id": "other"}]
        for update in updates:
            with self.subTest(update=update), self.assertRaises(ValidationError):
                run([frame(0), frame(1, item=candidate(**update))])
        with self.assertRaises(ValidationError):
            run([frame(0), frame(1, candidates=[candidate(), candidate()])])

    def test_frame_order_source_and_nullable_jpeg_binding_are_validated(self):
        for second in (frame(0), frame(1, timestamp=0), frame(1, source_sha256="c" * 64)):
            with self.subTest(second=second), self.assertRaises(ValidationError):
                run([frame(0), second])
        frames = [frame(0), frame(1, frame_sha256=None)]
        with self.assertRaises(ValidationError):
            run(frames)
        frames[1]["candidates"][0]["semantic"] = None
        self.assertEqual("unknown", run(frames)["status"])

    def test_gap_event_and_reacquisition_counts_remain_episode_based(self):
        result = run([frame(0), frame(1, item=False), frame(2, item=False), frame(3), frame(4),
                      frame(5), frame(6, item=False)])
        self.assertEqual(3, result["metrics"]["hard_gap_frames"])
        self.assertEqual(2, result["metrics"]["hard_gap_events"])
        self.assertEqual(1, result["metrics"]["reacquisition_events"])

    def test_largest_input_is_bounded_and_every_support_contains_current_frame(self):
        frames = [frame(i, timestamp=i * 100) for i in range(4096)]
        result = run(frames, min_stable_views=8)
        self.assertEqual(4096, result["metrics"]["frames"])
        self.assertEqual("supported", result["status"])
        for row in result["frames"]:
            if row["status"] == "supported":
                evidence = row["confirmation_evidence"]
                self.assertEqual(row["frame_index"], evidence[-1]["frame_index"])
                self.assertEqual(8, len(evidence))
                self.assertEqual(8, len({v["frame_sha256"] for v in evidence}))
                self.assertTrue(all(b["timestamp_ms"] - a["timestamp_ms"] >= 500 for a, b in zip(evidence, evidence[1:])))

    def test_determinism_policy_hash_json_serialization_and_immutability(self):
        frames = [frame(0), frame(1), frame(2)]
        before = deepcopy(frames)
        first = run(frames)
        self.assertEqual(CONFIRMATION_POLICY_VERSION, first["method_version"])
        self.assertEqual(first, run(frames))
        self.assertEqual(before, frames)
        self.assertEqual(first, json.loads(json.dumps(first, allow_nan=False)))
        self.assertRegex(first["policy_hash"], r"^[0-9a-f]{64}$")
        self.assertNotEqual(first["policy_hash"], run(frames, min_spacing_ms=600)["policy_hash"])
        self.assertNotEqual(first["policy_hash"], run(frames, min_confidence=.9)["policy_hash"])
        self.assertEqual(first["metrics"]["frames"], first["metrics"]["supported_frames"] + first["metrics"]["unknown_frames"])


if __name__ == "__main__":
    unittest.main()
