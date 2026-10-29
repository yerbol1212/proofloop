"""Authored tracking-state fixtures; these establish no video/model accuracy."""
from copy import deepcopy
import json
import math
import unittest

from proofloop.core import ValidationError
from proofloop.tracking_policy import TRACKING_POLICY_VERSION, evaluate_tracking


CLIP = "a" * 64


def candidate(*, track_id=1, semantic=None, **kwargs):
    return {"track_id": track_id, "object_id": "target", "box": [.1, .1, .3, .3],
            "confidence": .95, "mask_supported": True, "fully_visible": True,
            "identity_ambiguous": False, "semantic": semantic, **kwargs}


def frame(index, *, item=True, timestamp=None, **kwargs):
    return {"frame_index": index, "timestamp_ms": index * 500 if timestamp is None else timestamp,
            "source_sha256": CLIP, "frame_sha256": f"{index + 1:064x}",
            "candidates": [candidate()] if item is True else [] if item is False else [item],
            "reviewed_distractors": [], **kwargs}


def semantic(frame, decision="supported", *, index=0):
    item = frame["candidates"][index]
    item["semantic"] = {"target_id": "target", "track_id": item["track_id"],
        "frame_index": frame["frame_index"], "source_sha256": frame["source_sha256"],
        "frame_sha256": frame["frame_sha256"], "candidate_box": list(item["box"]),
        "decision": decision, "evidence_sha256": f"{1000 + frame['frame_index']:064x}"}
    return frame


def seed(first=None, *, distractors=None):
    first = frame(0) if first is None else first
    return {"target_id": "target", "track_id": 1, "frame_index": first["frame_index"],
            "source_sha256": first["source_sha256"], "frame_sha256": first["frame_sha256"],
            "review_sha256": "b" * 64, "reviewed_distractor_track_ids": distractors or []}


def run(frames, *, use_seed=True, **kwargs):
    return evaluate_tracking(frames, target_id="target", seed=seed(frames[0]) if use_seed else None, **kwargs)


class TrackingPolicyTests(unittest.TestCase):
    def test_initial_authored_seed_requires_two_distinct_supported_views(self):
        result = run([frame(0), frame(1)])
        self.assertEqual(TRACKING_POLICY_VERSION, result["method_version"])
        self.assertEqual(["unknown", "supported"], [row["status"] for row in result["frames"]])
        self.assertEqual(["acquiring", "tracking_supported"], [row["lifecycle"] for row in result["frames"]])
        self.assertEqual("authored_seed", result["frames"][1]["identity_basis"])
        self.assertEqual(0, result["metrics"]["reacquisition_events"])
        self.assertFalse(result["changes_live_placement_acceptance"])
        self.assertFalse(any(row["physical_action_authorized"] for row in result["frames"]))

    def test_seeded_masks_without_runtime_visibility_remain_unknown(self):
        frames = [frame(i, item=candidate(fully_visible=None, identity_ambiguous=None)) for i in range(3)]
        result = run(frames)
        self.assertEqual(["unknown"] * 3, [row["status"] for row in result["frames"]])
        self.assertEqual(0, result["metrics"]["hard_gap_events"])
        self.assertTrue(all("visibility_unconfirmed" in row["gap_codes"] for row in result["frames"]))
        self.assertFalse(any(row["physical_occlusion_observed"] for row in result["frames"]))

    def test_dense_unassessed_frames_preserve_but_do_not_count_confirmation_views(self):
        frames = [semantic(frame(0)), frame(1, item=candidate(fully_visible=None, identity_ambiguous=None)),
                  semantic(frame(2))]
        result = run(frames)
        self.assertEqual(["unknown", "unknown", "supported"], [row["status"] for row in result["frames"]])
        self.assertEqual([1, 1, 2], [row["confirmation_views"] for row in result["frames"]])
        self.assertEqual("tracking_unverified", result["frames"][1]["lifecycle"])

    def test_missing_track_does_not_establish_occlusion_or_absence(self):
        result = run([frame(0), frame(1), frame(2, item=False)])
        row = result["frames"][-1]
        self.assertEqual("unknown", row["status"])
        self.assertEqual("evidence_gap", row["lifecycle"])
        self.assertEqual(["candidate_missing"], row["gap_codes"])
        self.assertIsNone(row["box"])
        self.assertFalse(row["physical_occlusion_observed"])
        self.assertFalse(row["identity_switch_observed"])
        self.assertTrue(row["semantic_reacquisition_required"])

    def test_same_sam_identifier_cannot_restore_identity_after_missing_frame(self):
        result = run([frame(0), frame(1), frame(2, item=False), frame(3), frame(4)])
        self.assertEqual(["unknown", "supported", "unknown", "unknown", "unknown"],
                         [row["status"] for row in result["frames"]])
        self.assertTrue(all("semantic_reacquisition_required" in row["gap_codes"] for row in result["frames"][3:]))
        self.assertEqual(0, result["frames"][-1]["confirmation_views"])

    def test_after_gap_two_bound_semantic_views_restore_support_across_dense_nulls(self):
        frames = [frame(0), frame(1), frame(2, item=False), semantic(frame(3)),
                  frame(4, item=candidate(fully_visible=None, identity_ambiguous=None)), semantic(frame(5))]
        result = run(frames)
        self.assertEqual([0, 1, 1, 2], [row["confirmation_views"] for row in result["frames"][2:]])
        self.assertEqual("unknown", result["frames"][4]["status"])
        self.assertEqual("supported", result["frames"][5]["status"])
        self.assertEqual("runtime_semantic", result["frames"][5]["identity_basis"])
        self.assertEqual(1, result["metrics"]["reacquisition_events"])

    def test_reacquisition_cannot_count_the_same_jpeg_twice(self):
        duplicate = frame(3, frame_sha256=frame(2)["frame_sha256"])
        result = run([frame(0), frame(1, item=False), semantic(frame(2)), semantic(duplicate)])
        self.assertEqual("unknown", result["status"])
        self.assertEqual(1, result["frames"][-1]["confirmation_views"])
        self.assertIn("source_view_reused", result["frames"][-1]["gap_codes"])

    def test_reacquisition_enforces_frozen_source_spacing(self):
        frames = [frame(0), frame(1, item=False), semantic(frame(2)), semantic(frame(3, timestamp=1100)), semantic(frame(4))]
        result = run(frames)
        self.assertIn("source_spacing_insufficient", result["frames"][3]["gap_codes"])
        self.assertEqual("unknown", result["frames"][3]["status"])
        self.assertEqual("supported", result["frames"][4]["status"])
        strict = run([frame(0), frame(1)], min_spacing_ms=1000)
        self.assertEqual("unknown", strict["status"])

    def test_unseeded_identity_requires_semantics_even_with_confident_masks(self):
        no_semantics = run([frame(0), frame(1)], use_seed=False)
        self.assertEqual("unknown", no_semantics["status"])
        confirmed = run([semantic(frame(0)), semantic(frame(1))], use_seed=False)
        self.assertEqual("supported", confirmed["status"])
        self.assertEqual("runtime_semantic", confirmed["frames"][-1]["identity_basis"])
        self.assertEqual(0, confirmed["metrics"]["reacquisition_events"])

    def test_low_missing_score_mask_failure_partial_visibility_and_ambiguity_invalidate(self):
        for change, gap in (({"confidence": .79}, "low_confidence"), ({"confidence": None}, "score_unreported"),
                            ({"mask_supported": False}, "mask_unsupported"),
                            ({"fully_visible": False}, "full_visibility_not_supported"),
                            ({"identity_ambiguous": True}, "identity_ambiguous")):
            result = run([frame(0), frame(1), frame(2, item=candidate(**change)), frame(3)])
            with self.subTest(change=change):
                self.assertEqual("unknown", result["frames"][2]["status"])
                self.assertIn(gap, result["frames"][2]["gap_codes"])
                self.assertEqual(0, result["frames"][2]["confirmation_views"])
                self.assertIn("semantic_reacquisition_required", result["frames"][3]["gap_codes"])

    def test_runtime_distractor_overlap_invalidates_even_semantically_supported_track(self):
        overlap = semantic(frame(2, reviewed_distractors=[{"track_id": 2, "box": [.2, .2, .4, .4], "origin": "runtime_sam_track"}]))
        frames = [frame(0), frame(1), overlap, frame(3)]
        result = evaluate_tracking(frames, target_id="target", seed=seed(frames[0], distractors=[2]))
        self.assertEqual("unknown", result["frames"][2]["status"])
        self.assertIn("reviewed_distractor_overlap", result["frames"][2]["gap_codes"])
        self.assertTrue(result["frames"][2]["reviewed_distractor_overlap"])
        self.assertFalse(result["frames"][2]["physical_occlusion_observed"])
        self.assertEqual("unknown", result["frames"][3]["status"])

    def test_touching_distractor_edges_are_not_positive_area_overlap(self):
        frames = [frame(i, reviewed_distractors=[{"track_id": 2, "box": [.3, .1, .5, .3], "origin": "runtime_sam_track"}])
                  for i in range(2)]
        result = evaluate_tracking(frames, target_id="target", seed=seed(frames[0], distractors=[2]))
        self.assertEqual("supported", result["status"])

    def test_target_at_image_boundary_stays_unknown_without_claiming_partial_occlusion(self):
        for box in ([0, .1, .3, .3], [.1, 0, .3, .3], [.1, .1, 1, .3], [.1, .1, .3, 1]):
            with self.subTest(box=box):
                result = run([frame(0), frame(1, item=candidate(box=box))])
                self.assertEqual("unknown", result["status"])
                self.assertIn("frame_boundary_unconfirmed", result["frames"][-1]["gap_codes"])
                self.assertFalse(result["frames"][-1]["physical_occlusion_observed"])

    def test_track_identifier_change_is_not_reported_as_a_proven_identity_switch(self):
        result = run([frame(0), frame(1), frame(2, item=candidate(track_id=8)), frame(3, item=candidate(track_id=8))])
        self.assertEqual(1, result["metrics"]["track_identifier_changes"])
        self.assertTrue(result["frames"][2]["track_identifier_changed"])
        self.assertFalse(result["frames"][2]["identity_switch_observed"])
        self.assertEqual("unknown", result["status"])

    def test_new_identifier_can_be_reacquired_only_after_two_semantic_views(self):
        frames = [frame(0), frame(1), semantic(frame(2, item=candidate(track_id=8))),
                  semantic(frame(3, item=candidate(track_id=8)))]
        result = run(frames)
        self.assertEqual("unknown", result["frames"][2]["status"])
        self.assertEqual(1, result["frames"][2]["confirmation_views"])
        self.assertEqual("supported", result["status"])
        self.assertEqual(1, result["metrics"]["reacquisition_events"])

    def test_semantic_conflict_or_uncertainty_resets_prior_support(self):
        for decision in ("unknown", "contradicted"):
            result = run([frame(0), frame(1), semantic(frame(2), decision), frame(3)])
            with self.subTest(decision=decision):
                self.assertEqual("unknown", result["frames"][2]["status"])
                self.assertIn("semantic_identity_" + decision, result["frames"][2]["gap_codes"])
                self.assertEqual("unknown", result["status"])

    def test_multiple_target_candidates_need_all_other_candidates_semantically_excluded(self):
        ambiguous = frame(1, candidates=[candidate(), candidate(track_id=7)])
        result = run([frame(0), ambiguous])
        self.assertEqual("unknown", result["status"])
        self.assertIn("multiple_target_candidates", result["frames"][-1]["gap_codes"])
        clear = frame(1, candidates=[candidate(), candidate(track_id=7)])
        semantic(clear, index=0)
        semantic(clear, "contradicted", index=1)
        self.assertEqual("supported", run([frame(0), clear])["status"])

    def test_semantic_records_must_bind_exact_clip_jpeg_frame_track_target_and_box(self):
        updates = ({"source_sha256": "c" * 64}, {"frame_sha256": "d" * 64}, {"frame_index": 9},
                   {"track_id": 99}, {"track_id": True}, {"target_id": "another"},
                   {"candidate_box": [.2, .2, .4, .4]}, {"evidence_sha256": "short"}, {"decision": "pass"})
        for change in updates:
            frames = [semantic(frame(0)), semantic(frame(1))]
            frames[1]["candidates"][0]["semantic"].update(change)
            with self.subTest(change=change), self.assertRaises(ValidationError):
                run(frames)

    def test_semantic_support_cannot_override_unassessed_or_conflicting_visibility_flags(self):
        for change in ({"fully_visible": False}, {"fully_visible": None}, {"identity_ambiguous": True},
                       {"identity_ambiguous": None}):
            frames = [frame(0), semantic(frame(1, item=candidate(**change)))]
            with self.subTest(change=change), self.assertRaises(ValidationError):
                run(frames)

    def test_reviewed_seed_must_bind_the_exact_initial_frame(self):
        for change in ({"source_sha256": "c" * 64}, {"frame_sha256": "d" * 64}, {"frame_index": 1},
                       {"target_id": "another"}, {"review_sha256": "short"}, {"track_id": True}):
            frames = [frame(0), frame(1)]
            bad = {**seed(frames[0]), **change}
            with self.subTest(change=change), self.assertRaises(ValidationError):
                evaluate_tracking(frames, target_id="target", seed=bad)

    def test_only_initially_reviewed_runtime_distractor_tracks_are_allowed(self):
        for distractor in ({"track_id": 2, "box": [.5, .5, .7, .7], "origin": "ground_truth"},
                           {"track_id": 3, "box": [.5, .5, .7, .7], "origin": "runtime_sam_track"}):
            frames = [frame(0), frame(1, reviewed_distractors=[distractor])]
            with self.subTest(distractor=distractor), self.assertRaises(ValidationError):
                evaluate_tracking(frames, target_id="target", seed=seed(frames[0], distractors=[2]))
        with self.assertRaises(ValidationError):
            evaluate_tracking([frame(0)], target_id="target", seed=seed(distractors=[1]))

    def test_ground_truth_labels_and_arbitrary_provider_fields_are_rejected(self):
        for position in ("frame", "candidate", "semantic", "seed"):
            frames = [semantic(frame(0)), semantic(frame(1))]
            review = seed(frames[0])
            place = {"frame": frames[0], "candidate": frames[0]["candidates"][0],
                     "semantic": frames[0]["candidates"][0]["semantic"], "seed": review}[position]
            place["ground_truth_box"] = [.1, .1, .3, .3]
            with self.subTest(position=position), self.assertRaises(ValidationError):
                evaluate_tracking(frames, target_id="target", seed=review)

    def test_ordering_duplicate_indices_and_clip_changes_are_rejected(self):
        for second in (frame(0), frame(1, timestamp=0), frame(1, source_sha256="c" * 64)):
            with self.subTest(second=second), self.assertRaises(ValidationError):
                run([frame(0), second])

    def test_unbound_source_pixels_never_count_as_a_new_confirmation(self):
        frames = [frame(0), frame(1, frame_sha256=None)]
        result = run(frames)
        self.assertEqual("unknown", result["status"])
        self.assertIn("source_pixels_unbound", result["frames"][-1]["gap_codes"])
        self.assertEqual(1, result["frames"][-1]["confirmation_views"])

    def test_duplicate_tracks_invalid_numbers_and_boxes_fail_closed(self):
        malformed = [{"confidence": value} for value in (True, -1, 1.1, math.inf, math.nan, "0.9")]
        malformed += [{"box": value} for value in ([0, 0, 0, 1], [.2, .2, .1, .4], [0, 0, 2, 1], [False, .1, .3, .3])]
        malformed += [{"track_id": True}, {"mask_supported": "true"}, {"fully_visible": "unknown"}, {"object_id": "other"}]
        for change in malformed:
            with self.subTest(change=change), self.assertRaises(ValidationError):
                run([frame(0), frame(1, item=candidate(**change))])
        with self.assertRaises(ValidationError):
            run([frame(0), frame(1, candidates=[candidate(), candidate()])])

    def test_config_bounds_and_input_count_are_explicit(self):
        for change in ({"min_confidence": True}, {"min_confidence": 1.1}, {"min_stable_views": 1},
                       {"min_stable_views": 9}, {"min_spacing_ms": -1}, {"min_spacing_ms": True}):
            with self.subTest(change=change), self.assertRaises(ValidationError):
                run([frame(0), frame(1)], **change)
        with self.assertRaises(ValidationError):
            evaluate_tracking([], target_id="target")
        with self.assertRaises(ValidationError):
            evaluate_tracking([frame(0)], target_id="../private")

    def test_gap_events_count_episodes_separately_from_missing_frames(self):
        result = run([frame(0), frame(1, item=False), frame(2, item=False),
                      semantic(frame(3)), semantic(frame(4)), frame(5, item=False)])
        self.assertEqual(3, result["metrics"]["hard_gap_frames"])
        self.assertEqual(2, result["metrics"]["hard_gap_events"])
        self.assertEqual(1, result["metrics"]["reacquisition_events"])

    def test_determinism_policy_hash_serializability_and_input_immutability(self):
        frames = [semantic(frame(0)), semantic(frame(1))]
        before = deepcopy(frames)
        first, second = run(frames), run(frames)
        self.assertEqual(first, second)
        self.assertEqual(before, frames)
        self.assertRegex(first["policy_hash"], r"^[0-9a-f]{64}$")
        self.assertNotEqual(first["policy_hash"], run(frames, min_confidence=.9)["policy_hash"])
        self.assertEqual(first, json.loads(json.dumps(first, allow_nan=False)))
        self.assertEqual(first["metrics"]["frames"], first["metrics"]["supported_frames"] + first["metrics"]["unknown_frames"])
        self.assertTrue(all(row["status"] in ("supported", "unknown") for row in first["frames"]))
        self.assertNotIn("accuracy", json.dumps(first))


if __name__ == "__main__":
    unittest.main()
