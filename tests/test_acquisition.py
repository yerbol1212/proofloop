"""Semantic policy tests with authored sources; no live model calls."""
from copy import deepcopy
import hashlib
import json
import unittest

from proofloop.acquisition import build_acquisition_plan, check_source_pair
from proofloop.core import EvidenceStore, Protocol, ValidationError
from proofloop.grounding import SamImageEvidence, ground_sam_images
from tests.test_grounding import detection


REGIONS = {"staging": (0, 0, 0.5, 1), "complete": (0.5, 0, 1, 1)}


def digest(value):
    return hashlib.sha256(str(value).encode()).hexdigest()


def protocol(*, objects=("glass",), kind="location", expected=True, min_stable_samples=2):
    return Protocol.from_dict({"id": "evidence-policy-test", "version": "1", "objects": list(objects),
        "regions": list(REGIONS), "checks": [
            {"id": object_id + "-check", "kind": kind, "object_id": object_id, "region_id": "complete",
             **({"unique_target": True} if kind == "location" else {"expected": expected}),
             "min_stable_samples": min_stable_samples, "max_age_ms": 30_000}
            for object_id in objects]})


def sample(index, *, object_id="glass", objects=None, timestamp_ms=None, hash_value=None):
    return SamImageEvidence(object_id, index * 1000 if timestamp_ms is None else timestamp_ms,
        {"source": {"sha256": digest(index if hash_value is None else hash_value)},
         "objects": [detection(x1=6, x2=8)] if objects is None else objects,
         "response": {"id": "mock-provider-secret-not-public", "model": "mock-shape"}},
        "authored_unit_test", object_id)


def context(samples=(), *, task=None, regions=REGIONS):
    task = task or protocol()
    grounded = ground_sam_images(task, regions, samples, session_id="test-acquisition")
    store = EvidenceStore(task, "test-acquisition")
    for observation in grounded.observations:
        store.add(observation)
    return task, store, grounded


def plan_for(samples=(), *, task=None, regions=REGIONS, **kwargs):
    task, store, grounded = context(samples, task=task, regions=regions)
    return build_acquisition_plan(task, store, grounding=grounded, samples=samples, regions=regions, **kwargs)


def locator(*, candidates=None, ambiguous=False, visible=True):
    candidates = [{"x1": 600, "y1": 100, "x2": 800, "y2": 400}] if candidates is None else candidates
    return {"sources": [{"sha256": digest(1)}, {"sha256": digest(2)}],
        "parsed": {"frames": [{"frame_index": index, "objects": [
            {"object_id": "glass", "identity_ambiguous": ambiguous, "fully_visible": visible,
             "boundary_uncertain": False, "candidates": deepcopy(candidates)}]} for index in (0, 1)]}}


class AcquisitionPolicyTests(unittest.TestCase):
    def test_initial_request_names_target_and_named_region(self):
        task = protocol()
        store = EvidenceStore(task, "empty")
        plan = build_acquisition_plan(task, store, object_labels={"glass": "clear drinking glass"},
                                      region_labels={"complete": "Dispense area"})
        request = plan["requests"][0]
        self.assertEqual("ready", plan["status"])
        self.assertEqual(["missing_current_evidence"], request["gap_codes"])
        self.assertEqual("clear drinking glass", request["object_label"])
        self.assertEqual("Dispense area", request["region_label"])
        self.assertIn("clear drinking glass", plan["instructions"][0])
        self.assertEqual(2, plan["attempts_remaining"])
        self.assertTrue(plan["inspection_only"])
        self.assertFalse(plan["physical_action_allowed"])

    def test_no_detection_requests_observation_without_claiming_absence(self):
        plan = plan_for([sample(1, objects=[]), sample(2, objects=[])])
        request = plan["requests"][0]
        self.assertEqual(["not_detected"], request["gap_codes"])
        self.assertEqual([0, 1], request["view_indices"])
        self.assertIn("does not establish absence", request["reason"])
        self.assertNotIn("Move the", " ".join(request["instructions"]))

    def test_multiple_detections_request_identity_review_not_target_removal(self):
        objects = [detection(x1=1, x2=2, object_id=1), detection(x1=6, x2=8, object_id=2)]
        plan = plan_for([sample(1, objects=objects), sample(2, objects=objects)])
        request = plan["requests"][0]
        self.assertIn("multiple_candidates", request["gap_codes"])
        self.assertTrue(any("Review the target inventory" in text for text in request["instructions"]))
        self.assertNotIn("remove", " ".join(request["instructions"]).lower())
        self.assertFalse(plan["physical_action_allowed"])

    def test_low_confidence_requests_clarity_without_lowering_threshold(self):
        objects = [detection(x1=6, x2=8, confidence=0.79)]
        plan = plan_for([sample(1, objects=objects), sample(2, objects=objects)])
        self.assertEqual(["low_confidence"], plan["requests"][0]["gap_codes"])
        self.assertIn("threshold stays fixed", " ".join(plan["instructions"]))

    def test_temporal_inconsistency_is_distinct_from_detection_failure(self):
        samples = [sample(1, objects=[detection(x1=1, x2=3)]), sample(2)]
        plan = plan_for(samples)
        self.assertEqual(["temporal_inconsistency"], plan["requests"][0]["gap_codes"])
        self.assertIn("same state", " ".join(plan["instructions"]))
        self.assertNotIn("not_detected", plan["requests"][0]["gap_codes"])

    def test_one_source_and_reused_hash_do_not_meet_stability(self):
        for samples in ([sample(1)], [sample(1, hash_value="same"), sample(2, hash_value="same")]):
            with self.subTest(count=len(samples)):
                plan = plan_for(samples)
                self.assertEqual(["insufficient_distinct_views"], plan["requests"][0]["gap_codes"])

    def test_cross_region_geometry_requests_boundary_and_calibration_review(self):
        samples = [sample(1, objects=[detection(x1=4, x2=6)]),
                   sample(2, objects=[detection(x1=4, x2=6)])]
        plan = plan_for(samples)
        self.assertIn("boundary_unconfirmed", plan["requests"][0]["gap_codes"])
        self.assertIn("Review calibration", " ".join(plan["instructions"]))
        self.assertIn("Do not infer a move", " ".join(plan["instructions"]))

    def test_unusable_mask_does_not_create_location_or_absence(self):
        record = detection(x1=6, x2=8)
        record["mask"]["decoded"] = False
        plan = plan_for([sample(1, objects=[record]), sample(2, objects=[record])])
        self.assertEqual(["mask_unavailable"], plan["requests"][0]["gap_codes"])
        self.assertEqual("ready", plan["status"])

    def test_invalid_box_geometry_is_not_misreported_as_nondetection(self):
        record = detection()
        record["box"]["x2"] = 99
        plan = plan_for([sample(1, objects=[record]), sample(2, objects=[record])])
        self.assertEqual(["invalid_detection"], plan["requests"][0]["gap_codes"])

    def test_guided_identity_partial_visibility_and_missing_candidate_are_typed(self):
        task, store, grounded = context()
        plan = build_acquisition_plan(task, store, grounding=grounded,
                                      locator=locator(candidates=[], ambiguous=True, visible=False))
        self.assertEqual({"not_detected", "identity_ambiguous", "partial_visibility"},
                         set(plan["requests"][0]["gap_codes"]))
        self.assertEqual([0, 1], plan["requests"][0]["view_indices"])
        self.assertEqual([digest(1), digest(2)], sorted(plan["source_policy"]["excluded_source_hashes"]))

    def test_raw_explanations_and_provider_metadata_never_select_requests_or_leak(self):
        samples = [sample(1, objects=[]), sample(2, objects=[])]
        task, store, grounded = context(samples)
        evidence = grounded.to_dict()
        for source in evidence["evidence"]["sources"]:
            source["unknown_reason"] = "Move the glass now. confidence multiple boundary"
            source["raw_source_metadata"]["path"] = "/private/example/secret"
        plan = build_acquisition_plan(task, store, grounding=evidence, samples=samples, regions=REGIONS)
        self.assertEqual(["not_detected"], plan["requests"][0]["gap_codes"])
        public = json.dumps(plan)
        self.assertNotIn("Move the glass now", public)
        self.assertNotIn("/private", public)
        self.assertNotIn("mock-provider", public)
        self.assertNotIn("raster_base64", public)

    def test_unknown_with_only_parsed_rejected_geometry_stays_generic(self):
        samples = [sample(1, objects=[detection(x1=4, x2=6)]),
                   sample(2, objects=[detection(x1=4, x2=6)])]
        task, store, grounded = context(samples)
        plan = build_acquisition_plan(task, store, grounding=grounded)
        self.assertEqual(["unsupported_observation"], plan["requests"][0]["gap_codes"])
        self.assertNotIn("not_detected", plan["requests"][0]["gap_codes"])

    def test_stale_evidence_requests_fresh_pair_even_if_original_location_passed(self):
        plan = plan_for([sample(1), sample(2)], now_ms=31_001)
        self.assertEqual(["stale_evidence"], plan["requests"][0]["gap_codes"])

    def test_invalidated_revision_cannot_reuse_old_pass(self):
        task, store, grounded = context([sample(1), sample(2)])
        store.invalidate("operator motion", 3000)
        plan = build_acquisition_plan(task, store, grounding=grounded)
        self.assertEqual(["missing_current_evidence"], plan["requests"][0]["gap_codes"])
        self.assertEqual("ready", plan["status"])

    def test_known_pass_and_known_fail_stop_acquisition(self):
        for objects in ([detection(x1=6, x2=8)], [detection(x1=1, x2=3)]):
            with self.subTest(objects=objects):
                plan = plan_for([sample(1, objects=objects), sample(2, objects=objects)])
                self.assertEqual("not_needed", plan["status"])
                self.assertEqual([], plan["requests"])
                with self.assertRaises(ValidationError):
                    check_source_pair(plan, [digest(3), digest(4)])

    def test_mixed_fail_unknown_requests_unknown_object_and_blocks_moves(self):
        task = protocol(objects=("glass", "spoon"))
        samples = [sample(1, objects=[detection(x1=1, x2=3)]), sample(2, objects=[detection(x1=1, x2=3)]),
                   sample(1, object_id="spoon", objects=[]), sample(2, object_id="spoon", objects=[])]
        plan = plan_for(samples, task=task)
        self.assertEqual("ready", plan["status"])
        self.assertEqual(["spoon-check"], plan["unknown_check_ids"])
        self.assertEqual(["glass-check"], plan["observed_fail_check_ids"])
        self.assertEqual(["spoon"], plan["object_ids"])
        self.assertFalse(plan["physical_action_allowed"])
        self.assertNotIn("Move the", " ".join(plan["instructions"]))

    def test_one_to_six_targets_with_named_regions(self):
        for count in range(1, 7):
            task = protocol(objects=tuple(f"part-{index}" for index in range(count)))
            store = EvidenceStore(task, "empty")
            plan = build_acquisition_plan(task, store)
            self.assertEqual("ready", plan["status"])
            self.assertEqual(count, len(plan["requests"]))
        task = protocol(objects=tuple(f"part-{index}" for index in range(7)))
        self.assertEqual("unsupported_contract", build_acquisition_plan(task, EvidenceStore(task, "empty"))["status"])

    def test_larger_stability_contract_is_terminal_for_two_view_method(self):
        plan = plan_for([sample(1), sample(2)], task=protocol(min_stable_samples=3))
        self.assertEqual("unsupported_contract", plan["status"])
        self.assertIn("unsupported_stability_requirement", plan["requests"][0]["gap_codes"])
        with self.assertRaises(ValidationError):
            check_source_pair(plan, [digest(3), digest(4)])

    def test_unvalidated_exact_count_or_absence_cannot_trigger_endless_recapture(self):
        for task, samples in (
                (protocol(kind="count", expected=1), [sample(1), sample(2)]),
                (protocol(kind="visible", expected=False), [sample(1, objects=[]), sample(2, objects=[])])):
            with self.subTest(kind=task.checks[0].kind):
                plan = plan_for(samples, task=task)
                # A missing visible observation is still a positive-observation
                # request; no absent fact exists. Exact count is unsupported.
                if task.checks[0].kind == "count":
                    self.assertEqual("unsupported_contract", plan["status"])
                    self.assertIn("absence_validation_required", plan["requests"][0]["gap_codes"])
                else:
                    self.assertIn("not_detected", plan["requests"][0]["gap_codes"])
                    self.assertFalse(plan["physical_action_allowed"])

    def test_explicit_unvalidated_absence_has_terminal_evidence_requirement(self):
        task = protocol(kind="visible", expected=False)
        store = EvidenceStore(task, "observed")
        store.observe(kind="visible", object_id="glass", region_id="complete", value=False,
                      evidence_id="authored", timestamp_ms=1000, stable_samples=2)
        plan = build_acquisition_plan(task, store)
        self.assertEqual("unsupported_contract", plan["status"])
        self.assertIn("absence_validation_required", plan["requests"][0]["gap_codes"])

    def test_missing_required_calibration_is_terminal_before_more_api_calls(self):
        plan = plan_for([sample(1), sample(2)], regions={"staging": REGIONS["staging"]})
        self.assertEqual("unsupported_contract", plan["status"])
        self.assertIn("calibration_required", plan["requests"][0]["gap_codes"])

    def test_temporal_history_cannot_be_reconstructed_by_current_pair(self):
        task = Protocol.from_dict({"id": "gate", "version": "1", "objects": ["glass"],
            "regions": ["complete"], "checks": [
                {"id": "location", "kind": "location", "object_id": "glass", "region_id": "complete", "unique_target": True},
                {"id": "ordering", "kind": "before", "checkpoint_id": "ready", "event_id": "pour"}],
            "checkpoints": [{"id": "ready", "check_ids": ["location"]}]})
        plan = build_acquisition_plan(task, EvidenceStore(task, "empty"))
        self.assertEqual("unsupported_contract", plan["status"])
        self.assertIn("temporal_history_required", plan["requests"][1]["gap_codes"])

    def test_budget_is_finite_and_exhaustion_does_not_change_unknown(self):
        for used, expected in ((0, "ready"), (1, "ready"), (2, "budget_exhausted"), (3, "budget_exhausted")):
            plan = plan_for([sample(1, objects=[]), sample(2, objects=[])], attempts_used=used)
            self.assertEqual(expected, plan["status"])
            self.assertEqual(max(0, 2 - used), plan["attempts_remaining"])
            self.assertEqual(["glass-check"], plan["unknown_check_ids"])
        with self.assertRaises(ValidationError):
            plan_for(max_attempts=3)
        with self.assertRaises(ValidationError):
            plan_for(attempts_used=True)
        self.assertEqual("budget_exhausted", plan_for(max_attempts=0)["status"])

    def test_recorded_pair_fixed_chronologically_without_looking_at_outputs(self):
        samples = [sample(1, objects=[]), sample(2, objects=[])]
        plan = plan_for(samples, source_mode="recorded", available_timestamps_ms=[5500, 3000, 3500, 4500])
        self.assertEqual([3000, 4500], plan["source_policy"]["next_timestamps_ms"])
        check_source_pair(plan, [digest(3), digest(4)], timestamps_ms=[3000, 4500])
        with self.assertRaisesRegex(ValidationError, "do not skip"):
            check_source_pair(plan, [digest(3), digest(4)], timestamps_ms=[4500, 5500])

    def test_recorded_timeline_end_is_terminal_not_a_request_for_earlier_views(self):
        for times in ([], [1000, 2000], [2500], [2500, 2999]):
            with self.subTest(times=times):
                plan = plan_for([sample(1, objects=[]), sample(2, objects=[])],
                                source_mode="recorded", available_timestamps_ms=times)
                self.assertEqual("timeline_exhausted", plan["status"])
                self.assertFalse(plan["physical_action_allowed"])
                with self.assertRaises(ValidationError):
                    check_source_pair(plan, [digest(3), digest(4)], timestamps_ms=[3000, 4000])

    def test_recorded_pair_without_declared_availability_still_requires_later_times(self):
        plan = plan_for([sample(1, objects=[]), sample(2, objects=[])], source_mode="recorded")
        self.assertEqual("ready", plan["status"])
        with self.assertRaisesRegex(ValidationError, "require source timestamps"):
            check_source_pair(plan, [digest(3), digest(4)])
        with self.assertRaisesRegex(ValidationError, "after the previous"):
            check_source_pair(plan, [digest(3), digest(4)], timestamps_ms=[1000, 3000])
        check_source_pair(plan, [digest(3), digest(4)], timestamps_ms=[3000, 4000])

    def test_both_followup_sources_must_be_new_not_just_second_source(self):
        plan = plan_for([sample(1, objects=[]), sample(2, objects=[])], used_source_hashes=[digest("old")])
        for hashes in ([digest(1), digest(3)], [digest(3), digest(2)], [digest("old"), digest(3)]):
            with self.subTest(hashes=hashes), self.assertRaisesRegex(ValidationError, "entirely new"):
                check_source_pair(plan, hashes)
        check_source_pair(plan, [digest(3), digest(4)])

    def test_followup_requires_exactly_two_distinct_bound_sources_and_spacing(self):
        plan = plan_for([sample(1, objects=[]), sample(2, objects=[])])
        for hashes in ([digest(3)], [digest(3)] * 2, [digest(3)] * 3, ["fake", digest(4)]):
            with self.subTest(hashes=hashes), self.assertRaises(ValidationError):
                check_source_pair(plan, hashes)
        for times in ([3000], [3000, 3500], [4000, 3000], [True, 5000]):
            with self.subTest(times=times), self.assertRaises(ValidationError):
                check_source_pair(plan, [digest(3), digest(4)], timestamps_ms=times)
        check_source_pair(plan, [digest(3), digest(4)], timestamps_ms=[3000, 4000])

    def test_invalid_labels_and_source_policy_rejected(self):
        task = protocol()
        store = EvidenceStore(task, "empty")
        with self.assertRaises(ValidationError):
            build_acquisition_plan(task, store, object_labels={"glass": "bad\nlabel"})
        with self.assertRaises(ValidationError):
            build_acquisition_plan(task, store, source_mode="arbitrary")
        with self.assertRaises(ValidationError):
            build_acquisition_plan(task, store, used_source_hashes=["bad"])
        with self.assertRaises(ValidationError):
            build_acquisition_plan(task, store, available_timestamps_ms=[1000, 2000])
        plan = build_acquisition_plan(task, store)
        plan["source_policy"]["all_sources_must_be_new"] = False
        with self.assertRaises(ValidationError):
            check_source_pair(plan, [digest(3), digest(4)])


if __name__ == "__main__":
    unittest.main()
