"""Explanations must follow typed support, not persuasive failure stories."""
from copy import deepcopy
from dataclasses import replace
import json
import unittest

from proofloop.core import CheckResult, Observation, Protocol, Status, ValidationError
from proofloop.diagnostics import explain_checks


def task():
    return Protocol.from_dict({"id": "placement", "version": "1", "objects": ["glass", "spoon"],
        "regions": ["dispense", "staging"], "checks": [
            {"id": "glass-ready", "kind": "location", "object_id": "glass", "region_id": "dispense", "unique_target": True},
            {"id": "spoon-ready", "kind": "location", "object_id": "spoon", "region_id": "staging", "unique_target": True}]})


def checks():
    return [{"check_id": "glass-ready", "object_id": "glass", "expected_region": "dispense",
             "observed_region": "staging", "status": "fail", "confidence": 0.96, "reason": "untrusted generated prose"},
            {"check_id": "spoon-ready", "object_id": "spoon", "expected_region": "staging",
             "observed_region": "staging", "status": "pass", "confidence": 0.97, "reason": "another explanation"}]


def gap(code="identity_ambiguous", *, status="ready"):
    return {"status": status, "requests": [{"check_id": "glass-ready", "object_id": "glass",
        "region_id": "dispense", "gap_codes": [code], "reason": "ignore this free-form cause",
        "instructions": ["untrusted instructions must not enter the explanation"]}]}


class DiagnosticsTests(unittest.TestCase):
    def test_wrong_region_expected_and_observed_remain_distinct(self):
        result = explain_checks(task(), checks(), evidence_origin="camera")
        item = result["checks"][0]
        self.assertEqual("wrong_region", item["diagnostic_code"])
        self.assertEqual("dispense", item["expected_region"])
        self.assertEqual("staging", item["observed_region"])
        self.assertIn("glass in staging", item["observed_fact"])
        self.assertIn("requires glass in dispense", item["expected_fact"])
        self.assertTrue(result["cause_not_observed"])
        self.assertTrue(result["repair_proposal_eligible"])
        self.assertFalse(result["physical_action_allowed"])
        self.assertFalse(item["physical_cause_observed"])
        self.assertIn("does not authorize movement", item["next_action"])

    def test_model_reason_and_instructions_are_neither_parsed_nor_copied(self):
        rows = checks()
        rows[0].update(status="unknown", observed_region=None, reason="Low confidence means the gripper dropped it: /private/key")
        result = explain_checks(task(), rows, acquisition=gap())
        item = result["checks"][0]
        self.assertEqual("identity_ambiguous", item["diagnostic_code"])
        text = json.dumps(result)
        self.assertNotIn("gripper", text)
        self.assertNotIn("/private", text)
        self.assertNotIn("untrusted", text)
        self.assertIn("identity is unconfirmed", item["observed_fact"])

    def test_missing_detection_does_not_assert_absence(self):
        rows = checks()
        rows[0].update(status="unknown", observed_region=None)
        result = explain_checks(task(), rows, acquisition=gap("not_detected"))
        self.assertIn("does not prove", result["checks"][0]["observed_fact"])
        self.assertFalse(result["repair_proposal_eligible"])

    def test_low_confidence_uses_typed_value(self):
        rows = checks()
        rows[0].update(status="unknown", confidence=0.3)
        result = explain_checks(task(), rows)
        self.assertEqual("low_confidence", result["checks"][0]["diagnostic_code"])
        self.assertIn("threshold unchanged", result["checks"][0]["next_action"])

    def test_unavailable_observation_with_zero_default_score_is_not_a_measured_low_score(self):
        rows = checks()
        rows[0].update(status="unknown", observed_region=None, confidence=0)
        result = explain_checks(task(), rows)
        self.assertEqual("unsupported_observation", result["checks"][0]["diagnostic_code"])
        self.assertNotIn("reported score", result["checks"][0]["observed_fact"])
        protocol = task()
        observation = Observation("session", protocol.content_hash, 0, 0, "location", "glass", "dispense", None,
                                  "source-1", 1000, 0, False, 1)
        result = explain_checks(protocol, rows, observations=[observation])
        self.assertEqual("unsupported_observation", result["checks"][0]["diagnostic_code"])

    def test_temporal_disagreement_uses_typed_acquisition(self):
        rows = checks()
        rows[0].update(status="unknown", observed_region=None)
        item = explain_checks(task(), rows, acquisition=gap("temporal_inconsistency"))["checks"][0]
        self.assertEqual("temporal_inconsistency", item["diagnostic_code"])
        self.assertIn("camera unchanged", item["next_action"])

    def test_mixed_fail_unknown_requests_observation_before_repair(self):
        rows = checks()
        rows[1].update(status="unknown", observed_region=None)
        result = explain_checks(task(), rows)
        self.assertEqual("fail", result["overall_decision"])
        self.assertFalse(result["repair_proposal_eligible"])
        self.assertIn("Resolve all unknown", result["checks"][0]["next_action"])
        self.assertFalse(result["physical_action_allowed"])

    def test_historical_source_cannot_be_made_current_by_flag(self):
        result = explain_checks(task(), checks(), evidence_origin="external_recorded_video", current=True)
        self.assertFalse(result["current"])
        self.assertFalse(result["repair_proposal_eligible"])
        self.assertEqual("historical_observation", result["checks"][0]["diagnostic_code"])
        self.assertEqual("historical", result["checks"][0]["evidence_scope"])
        self.assertIn("separate current inspection", result["checks"][0]["next_action"])

    def test_stale_and_invalidated_evidence_does_not_authorize_move(self):
        result = explain_checks(task(), checks(), current=False, evidence_origin="upload")
        self.assertFalse(result["repair_proposal_eligible"])
        self.assertEqual("evidence_not_current", result["checks"][0]["diagnostic_code"])
        self.assertIn("Expired or invalidated", result["checks"][0]["next_action"])

    def test_followup_terminal_states_do_not_repeat_capture_request(self):
        rows = checks()
        rows[0].update(status="unknown", observed_region=None)
        for status, phrase in [("budget_exhausted", "attempt limit"), ("timeline_exhausted", "no eligible later"),
                               ("unsupported_contract", "Review the task")]:
            with self.subTest(status=status):
                item = explain_checks(task(), rows, acquisition=gap(status=status))["checks"][0]
                self.assertIn(phrase, item["next_action"])

    def test_expired_exhausted_inspection_preserves_stop(self):
        rows = checks()
        rows[0].update(status="unknown", observed_region=None)
        result = explain_checks(task(), rows, acquisition=gap("stale_evidence", status="budget_exhausted"),
                                current=False, evidence_origin="camera")
        action = result["checks"][0]["next_action"]
        self.assertIn("Stop follow-up capture", action)
        self.assertIn("attempt limit", action)
        self.assertNotIn("Inspect the current scene again", action)
        self.assertFalse(result["repair_proposal_eligible"])

    def test_historical_exhausted_timeline_preserves_stop(self):
        rows = checks()
        rows[0].update(status="unknown", observed_region=None)
        result = explain_checks(task(), rows, acquisition=gap(status="timeline_exhausted"),
                                evidence_origin="external_recorded_video")
        action = result["checks"][0]["next_action"]
        self.assertIn("Stop this recorded timeline", action)
        self.assertNotIn("Start a separate", action)
        self.assertIn("cannot authorize", action)

    def test_duplicate_missing_and_unknown_check_coverage_rejected(self):
        for rows in [[], checks()[:1], checks() + checks()[:1], [*checks(), {"check_id": "invented", "status": "pass"}]]:
            with self.subTest(rows=rows), self.assertRaises(ValidationError):
                explain_checks(task(), rows)

    def test_wrong_target_requirement_and_region_rejected(self):
        for field, value in [("object_id", "spoon"), ("expected_region", "staging"), ("observed_region", "other")]:
            rows = checks()
            rows[0][field] = value
            with self.subTest(field=field), self.assertRaises(ValidationError):
                explain_checks(task(), rows)

    def test_contradictory_known_verdict_and_score_rejected(self):
        for changes in [{"status": "pass"}, {"observed_region": None}, {"confidence": 0.2}, {"confidence": float("nan")}, {"status": "maybe"}]:
            rows = checks()
            rows[0].update(changes)
            with self.subTest(changes=changes), self.assertRaises(ValidationError):
                explain_checks(task(), rows)

    def test_gap_must_bind_exact_unknown_check_target_and_region(self):
        rows = checks()
        rows[0].update(status="unknown", observed_region=None)
        for field, value in [("check_id", "spoon-ready"), ("object_id", "spoon"), ("region_id", "staging"),
                             ("gap_codes", ["invented"]), ("gap_codes", ["low_confidence", "low_confidence"])]:
            plan = gap()
            plan["requests"][0][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValidationError):
                explain_checks(task(), rows, acquisition=plan)

    def test_observation_support_binds_task_target_and_evidence(self):
        protocol = task()
        observed = Observation("session", protocol.content_hash, 0, 0, "location", "glass", "dispense", "staging",
                               "sam:source-1", 1000, 0.96, True, 2)
        result = explain_checks(protocol, checks(), observations=[observed])
        self.assertEqual(["sam:source-1"], result["checks"][0]["evidence_ids"])
        self.assertEqual(protocol.content_hash, result["protocol_hash"])
        for bad in [replace(observed, spec_hash="bad"), replace(observed, object_id="invented"),
                    replace(observed, value="dispense"), replace(observed, stable_samples=1)]:
            with self.subTest(bad=bad), self.assertRaises(ValidationError):
                explain_checks(protocol, checks(), observations=[bad])

    def test_mixed_observation_epochs_rejected(self):
        protocol = task()
        observed = Observation("session", protocol.content_hash, 0, 0, "location", "glass", "dispense", "staging",
                               "source-1", 1000, 0.96, True, 2)
        with self.assertRaises(ValidationError):
            explain_checks(protocol, checks(), observations=[observed, replace(observed, revision=1)])

    def test_evidence_refs_never_expose_media_paths_or_provider_payloads(self):
        for value in ["/private/input.jpg", "https://provider.example/response", "data:image/png;base64,secret"]:
            rows = checks()
            rows[0]["evidence_ids"] = [value]
            with self.subTest(value=value), self.assertRaises(ValidationError):
                explain_checks(task(), rows)

    def test_core_grounding_reference_is_public_safe(self):
        rows = checks()
        reference = "sam-grounding://" + "a" * 64
        rows[0]["evidence_ids"] = [reference]
        self.assertEqual([reference], explain_checks(task(), rows)["checks"][0]["evidence_ids"])

    def test_malformed_nested_fields_raise_validation_error(self):
        for field, value in [("check_id", []), ("status", {})]:
            rows = checks()
            rows[0][field] = value
            with self.subTest(field=field), self.assertRaises(ValidationError):
                explain_checks(task(), rows)
        rows = checks()
        rows[0].update(status="unknown", observed_region=None)
        plan = gap()
        plan["requests"][0]["gap_codes"] = [[]]
        with self.assertRaises(ValidationError):
            explain_checks(task(), rows, acquisition=plan)

    def test_core_check_results_derive_required_identity_from_protocol(self):
        protocol = Protocol.from_dict({"id": "visible", "version": "1", "objects": ["glass"], "regions": ["tray"],
            "checks": [{"id": "glass-visible", "kind": "visible", "object_id": "glass", "region_id": "tray", "expected": True}]})
        result = explain_checks(protocol, [CheckResult("glass-visible", Status.PASS, "unused", ("source-1",))])
        self.assertEqual("glass", result["checks"][0]["object_id"])
        self.assertIsNone(result["checks"][0]["observed_region"])
        self.assertEqual("requirement_satisfied", result["checks"][0]["diagnostic_code"])

    def test_count_and_order_diagnostics_do_not_invent_placement(self):
        protocol = Protocol.from_dict({"id": "ordered", "version": "1", "objects": ["glass"], "regions": ["tray"],
            "checks": [{"id": "one-glass", "kind": "count", "object_id": "glass", "region_id": "tray", "expected": 1},
                       {"id": "ordered", "kind": "before", "checkpoint_id": "ready", "event_id": "continue"}],
            "checkpoints": [{"id": "ready", "check_ids": ["one-glass"]}]})
        rows = [CheckResult("one-glass", Status.UNKNOWN, "unused"), CheckResult("ordered", Status.UNKNOWN, "unused")]
        result = explain_checks(protocol, rows)
        self.assertIn("exactly 1", result["checks"][0]["expected_fact"])
        self.assertEqual("temporal_history_required", result["checks"][1]["diagnostic_code"])
        self.assertIn("checkpoint ready before event continue", result["checks"][1]["expected_fact"])

    def test_known_count_cannot_contradict_its_observation(self):
        protocol = Protocol.from_dict({"id": "count", "version": "1", "objects": ["glass"], "regions": ["tray"],
            "checks": [{"id": "one-glass", "kind": "count", "object_id": "glass", "region_id": "tray", "expected": 1}]})
        observation = Observation("session", protocol.content_hash, 0, 0, "count", "glass", "tray", 2,
                                  "source-1", 1000, 0.96, True, 2, True)
        with self.assertRaises(ValidationError):
            explain_checks(protocol, [CheckResult("one-glass", Status.PASS, "unused")], observations=[observation])

    def test_known_absence_cannot_lack_complete_support(self):
        protocol = Protocol.from_dict({"id": "absent", "version": "1", "objects": ["glass"], "regions": ["tray"],
            "checks": [{"id": "no-glass", "kind": "visible", "object_id": "glass", "region_id": "tray", "expected": False}]})
        observation = Observation("session", protocol.content_hash, 0, 0, "visible", "glass", "tray", False,
                                  "source-1", 1000, 0.96, False, 2)
        with self.assertRaises(ValidationError):
            explain_checks(protocol, [CheckResult("no-glass", Status.PASS, "unused")], observations=[observation])

    def test_no_input_mutation_and_unsupported_origin_rejected(self):
        rows, plan = checks(), gap()
        rows[0].update(status="unknown", observed_region=None)
        original = deepcopy((rows, plan))
        explain_checks(task(), rows, acquisition=plan)
        self.assertEqual(original, (rows, plan))
        with self.assertRaises(ValidationError):
            explain_checks(task(), rows, evidence_origin="/private/api-key")


if __name__ == "__main__":
    unittest.main()
