"""Authored negative-witness logic tests; no model requests or physical proof."""
from copy import deepcopy
import hashlib
import unittest
from unittest.mock import patch

from proofloop import benchmark, containment_witness as module, crop_evidence, crop_verification as verifier
from proofloop.core import ValidationError
from tests import test_crop_verification as components


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


class ContainmentWitnessTests(unittest.TestCase):
    def setUp(self):
        self.fixture = components.CropVerificationTests("runTest")
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.configure()

    def configure(self, change=None):
        f = self.fixture
        source = deepcopy(f.envelope["contract"])
        source["protocol"]["checks"][0]["region_id"] = "staging"
        source["requirements"] = "The unique glass must be wholly within STAGING. Boundary-crossing evidence remains UNKNOWN."
        if change:
            change(source)
        f.envelope = benchmark.approve_manifest(source, benchmark.manifest_hash(source), confirm_unique_targets=True)
        self.suffix = getattr(self, "suffix", 0) + 1
        f.source_items = f.make_items("witness-source-" + str(self.suffix))
        f.pair = verifier.prepare_pair(f.envelope, f.source_items, f.root / f"artifacts/local/witness-pair-{self.suffix}", root=f.root)
        self.final = f.final(suffix="witness-final-" + str(self.suffix))

    def evaluation(self, change=None, complete=False):
        parsed = components.supported()
        if not complete:
            for row in parsed["frames"]:
                row["target_extent_covered"] = "contradicted"
        if change:
            change(parsed)
        return verifier.evaluate(self.final, components.spark(parsed, verifier.final_images(self.final)), root=self.fixture.root)

    def summary(self, evaluation):
        f = self.fixture
        report = verifier.derive_task_report(evaluation, root=f.root)
        case = {"case_id": report["case_id"], "contract_hash": f.envelope["contract_hash"],
                "source_sha256s": [row["source_sha256"] for row in f.pair["frames"]]}
        row = {"case_id": report["case_id"], "status": "completed", "prediction": report["overall"],
               "assessment": verifier.public_evaluation(evaluation, root=f.root)}
        cases, rows = [case], [row]
        for index in range(1, 9):
            identifier = "retained-unscored-" + str(index)
            cases.append({"case_id": identifier, "contract_hash": digest(identifier + "-contract"),
                          "source_sha256s": [digest(identifier + "-view0"), digest(identifier + "-view1")]})
            rows.append({"case_id": identifier, "status": "unscored", "prediction": None, "assessment": None})
        return {"plan_sha256": digest("authored-prior-plan"), "gate_passed": False, "status": "incomplete", "cases": cases, "results": rows}

    def assess(self, evaluation=None):
        evaluation = evaluation or self.evaluation()
        declaration = module.declare_control(self.summary(evaluation))
        return module.assess(evaluation, declaration, root=self.fixture.root)

    def test_declaration_preserves_all_nine_including_missing_and_replays_nothing(self):
        summary = self.summary(self.evaluation())
        with patch.object(verifier, "derive_task_report", side_effect=AssertionError("Declaration must not replay")):
            declaration = module.declare_control(summary)
        self.assertEqual(9, len(declaration["cases"]))
        self.assertEqual(8, sum(case["original_core_verdict"] is None for case in declaration["cases"]))
        self.assertEqual(0, declaration["zero_model_calls"])
        self.assertFalse(declaration["original_gate_passed"])
        self.assertEqual("all_projected_target_material_within_goal", declaration["conditional_predicate"])
        self.assertEqual("separately_declared_hypothesis_not_inferred_from_original_task", declaration["condition_source"])
        self.assertIs(False, declaration["original_task_violation_asserted"])
        self.assertIn("containment_witness.py", declaration["implementation_sha256"])

    def test_partial_material_can_support_only_typed_conditional_counterexample(self):
        evaluation = self.evaluation()
        original = verifier.derive_task_report(evaluation, root=self.fixture.root)
        certificate = self.assess(evaluation)
        self.assertEqual("supported_counterexample", certificate["status"])
        self.assertEqual("unknown", certificate["original_core_verdict"])
        self.assertEqual(original, verifier.derive_task_report(evaluation, root=self.fixture.root))
        self.assertEqual("dispense", certificate["witness_material_region_id"])
        self.assertEqual("staging", certificate["goal_region_id"])
        self.assertIsNone(certificate["whole_target_region_id"])
        self.assertFalse(certificate["repair_origin_available"])
        self.assertTrue(certificate["never_grants_pass"])
        self.assertTrue(certificate["never_changes_core_verdict"])
        self.assertTrue(certificate["boundary_crossing_unknown_policy_unchanged"])
        self.assertTrue(certificate["no_current_authority"])
        self.assertFalse(certificate["physical_truth_verified"])
        self.assertEqual("all_projected_target_material_within_goal", certificate["conditional_predicate"])
        self.assertEqual("separately_declared_hypothesis_not_inferred_from_original_task", certificate["condition_source"])
        self.assertIs(False, certificate["original_task_violation_asserted"])
        self.assertIn("fallible", certificate["membership_assumption"])
        self.assertEqual((2, 500, .91), (certificate["stable_source_count"], certificate["oldest_source_timestamp_ms"], certificate["minimum_reported_target_score"]))

    def test_complete_baseline_verdict_is_preserved_without_new_certificate(self):
        certificate = self.assess(self.evaluation(complete=True))
        self.assertEqual("withheld", certificate["status"])
        self.assertEqual("fail", certificate["original_core_verdict"])
        self.assertIsNone(certificate["claim"])
        self.assertIn("not_exclusively_incomplete_target_extent", certificate["withheld_reasons"])

    def test_any_other_semantic_or_contact_uncertainty_prevents_certificate(self):
        for field, value in (("identity", "uncertain"), ("mask_target_only", "contradicted"),
                             ("full_visibility", "uncertain"), ("gripper_mask_coverage", "uncertain"),
                             ("manipulator_relation", "held_or_contacting"), ("full_scene_match_count", "multiple"),
                             ("target_extent_covered", "uncertain")):
            with self.subTest(field=field):
                evaluation = self.evaluation(lambda parsed, key=field, item=value: parsed["frames"][0].update({key: item}))
                self.assertEqual("withheld", self.assess(evaluation)["status"])

    def test_near_gripper_negative_warning_cannot_be_discarded(self):
        f = self.fixture
        bound = f.bind()
        self.final = f.final(bound, f.grippers(bound, box=(304, 200, 310, 205)), suffix="near-final")
        certificate = self.assess()
        self.assertEqual("withheld", certificate["status"])
        self.assertEqual("unknown", certificate["original_core_verdict"])

    def test_check_confidence_is_not_lowered_to_recover_witness(self):
        self.configure(lambda source: source["protocol"]["checks"][0].update(min_confidence=.95))
        certificate = self.assess()
        self.assertEqual("withheld", certificate["status"])
        self.assertIn("reported_target_score_below_unchanged_threshold", certificate["withheld_reasons"])

    def test_original_earliest_support_age_is_binding(self):
        def change(source):
            source["sources"][0]["duration_ms"] = 40000
            source["cases"][0]["evaluation_timestamp_ms"] = 31000
        self.configure(change)
        certificate = self.assess()
        self.assertEqual("withheld", certificate["status"])
        self.assertIn("stale_or_future_source_evidence", certificate["withheld_reasons"])
        self.assertEqual(500, certificate["oldest_source_timestamp_ms"])

    def test_larger_original_stability_requirement_is_preserved(self):
        self.configure(lambda source: source["protocol"]["checks"][0].update(min_stable_samples=3))
        certificate = self.assess()
        self.assertEqual("withheld", certificate["status"])
        self.assertIn("insufficient_distinct_stable_sources", certificate["withheld_reasons"])

    def test_regions_touching_only_at_boundary_are_conservatively_withheld(self):
        self.configure(lambda source: source["cases"][0]["regions"].update(staging=[.5, 0, 1, 1]))
        certificate = self.assess()
        self.assertEqual("withheld", certificate["status"])
        self.assertIn("witness_region_not_strictly_disjoint_from_goal", certificate["withheld_reasons"])

    def test_witness_material_inside_goal_cannot_grant_anything(self):
        self.configure(lambda source: source["protocol"]["checks"][0].update(region_id="dispense"))
        certificate = self.assess()
        self.assertEqual("withheld", certificate["status"])
        self.assertIsNone(certificate["claim"])

    def test_missing_reordered_or_duplicate_inventory_is_not_a_new_denominator(self):
        summary = self.summary(self.evaluation())
        for kind in ("missing", "reordered", "duplicate"):
            changed = deepcopy(summary)
            if kind == "missing":
                changed["cases"].pop(); changed["results"].pop()
            elif kind == "reordered":
                changed["cases"].reverse()
            else:
                changed["cases"][1] = deepcopy(changed["cases"][0])
            with self.assertRaises(ValidationError):
                module.declare_control(changed)

    def test_reusing_original_source_across_different_cases_is_rejected(self):
        summary = self.summary(self.evaluation())
        summary["cases"][1]["source_sha256s"][0] = summary["cases"][0]["source_sha256s"][0]
        with self.assertRaises(ValidationError):
            module.declare_control(summary)
        evaluation = self.evaluation()
        declaration = module.declare_control(self.summary(evaluation))
        declaration["cases"][1]["source_sha256s"][0] = declaration["cases"][0]["source_sha256s"][0]
        declaration["declaration_sha256"] = benchmark.manifest_hash({k: v for k, v in declaration.items() if k != "declaration_sha256"})
        with self.assertRaises(ValidationError):
            module.assess(evaluation, declaration, root=self.fixture.root)

    def test_duplicate_or_reordered_physical_views_cannot_create_stability(self):
        evaluation = self.evaluation()
        declaration = module.declare_control(self.summary(evaluation))
        for kind in ("duplicate", "reordered"):
            changed = deepcopy(evaluation)
            if kind == "duplicate":
                changed["prepared_final"]["pair"]["frames"][1]["source_sha256"] = changed["prepared_final"]["pair"]["frames"][0]["source_sha256"]
            else:
                changed["frames"].reverse()
            with self.assertRaises(ValidationError):
                module.assess(changed, declaration, root=self.fixture.root)

    def test_changed_contract_policy_evaluation_or_original_verdict_cannot_be_substituted(self):
        evaluation = self.evaluation()
        declaration = module.declare_control(self.summary(evaluation))
        for change in (lambda d: d.update(policy="override_core_fail"),
                       lambda d: d.update(conditional_predicate="core_location_predicate"),
                       lambda d: d.update(condition_source="inferred_from_original_task"),
                       lambda d: d.update(original_task_violation_asserted=True),
                       lambda d: d.update(original_task_violation_asserted=0),
                       lambda d: d["cases"][0].update(contract_hash="0" * 64),
                       lambda d: d["cases"][0].update(evaluation_sha256="0" * 64),
                       lambda d: d["cases"][0].update(original_core_verdict="fail")):
            altered = deepcopy(declaration)
            change(altered)
            altered["declaration_sha256"] = benchmark.manifest_hash({k: v for k, v in altered.items() if k != "declaration_sha256"})
            with self.assertRaises(ValidationError):
                module.assess(evaluation, altered, root=self.fixture.root)

    def test_public_certificate_has_no_raw_provider_body_paths_or_encoded_masks(self):
        import json
        encoded = json.dumps(self.assess())
        for forbidden in ("PRIVATE_MODEL_BODY", "artifacts/local", "inputs/local", "raster_base64", "encoded_payload", "raw_result", "reported_model"):
            self.assertNotIn(forbidden, encoded)
        self.assertNotIn('"value": "dispense"', encoded)
        self.assertNotIn('"overall"', encoded)


if __name__ == "__main__":
    unittest.main()
