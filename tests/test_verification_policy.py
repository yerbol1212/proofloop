"""Authored policy fixtures, not live inference or physical-world validation."""
from copy import deepcopy
import json
import math
from pathlib import Path
import tempfile
import unittest

from proofloop.compiler import authored_coffee_protocol
from proofloop.core import Protocol, ValidationError
from proofloop.verification_policy import (
    POLICY_VERSION, describe_capabilities, inspection_binding,
    public_inspection_result, validate_binding,
)


def state_fixture():
    envelope = authored_coffee_protocol()
    return {
        "session_id": "interactive-policy-fixture", "revision": 4,
        "active_task": {"task_hash": "a" * 64, "protocol": envelope["protocol"],
                        "protocol_hash": envelope["content_hash"], "reviewed": True,
                        "review_origin": "local_operator"},
        "phase": "verified", "status": "pass", "busy": False, "error": None,
        "source_kind": "upload", "stale": False, "evidence_age_ms": 1000,
        "inference_origin": "injected_client_not_verified_live",
        "images": [{"id": "source-one"}, {"id": "source-two"}],
        "checks": [{"check_id": "glass-under-spout", "object_id": "glass",
                    "expected_region": "dispense", "observed_region": "dispense",
                    "status": "pass", "evidence_ids": ["sam-grounding://" + "b" * 64]}],
        "acquisition": None,
        "usage": {"request_attempts": 2, "estimated_usd": 0.005},
    }


def pending_state():
    state = state_fixture()
    state.update(phase="awaiting_capture", status="unknown", error=None)
    state["checks"][0].update(status="unknown", observed_region=None, evidence_ids=[])
    return state


class VerificationPolicyTests(unittest.TestCase):
    def test_capabilities_distinguish_core_vocabulary_from_supported_live_bridge(self):
        result = describe_capabilities()
        self.assertEqual(POLICY_VERSION, result["policy_version"])
        self.assertEqual(["count", "visible", "location", "before"], result["core_predicates"])
        self.assertEqual([{"kind": "location", "requires_unique_target_contract": True}], result["live_predicates"])
        self.assertFalse(result["binding_is_authorization"])
        self.assertEqual(2, result["evidence_input"]["exact_views"])
        self.assertEqual(2, result["acquisition"]["maximum_follow_up_submissions"])
        self.assertFalse(result["result"]["physical_action_authorized"])
        self.assertEqual(result, json.loads(json.dumps(result, allow_nan=False)))
        result["limitations"].clear()
        self.assertTrue(describe_capabilities()["limitations"])

    def test_binding_is_closed_and_contains_no_capture_secret(self):
        state = state_fixture()
        state.update(capture_token="secret", csrf="private")
        result = inspection_binding(state)
        self.assertEqual({"session_id", "task_hash", "protocol_hash", "revision"}, set(result))
        self.assertEqual("interactive-policy-fixture", result["session_id"])
        self.assertNotIn("secret", json.dumps(result))
        result["revision"] = 99
        self.assertEqual(4, state["revision"])

    def test_published_engine_limits_match_the_real_input_validator(self):
        from proofloop.interactive import FRESHNESS_MS, MAX_PIXELS, MAX_TASK_OBJECTS
        from proofloop.meta import MAX_IMAGE_BYTES
        descriptor = describe_capabilities()
        self.assertEqual(MAX_TASK_OBJECTS, descriptor["live_contract"]["max_objects"])
        self.assertEqual(FRESHNESS_MS, descriptor["evidence_input"]["freshness_window_ms"])
        self.assertEqual(MAX_PIXELS, descriptor["evidence_input"]["max_image_pixels"])
        self.assertEqual(MAX_IMAGE_BYTES, descriptor["evidence_input"]["max_image_bytes"])

    def test_binding_requires_reviewed_exact_supported_task(self):
        for key, value in (("reviewed", False), ("reviewed", "true"), ("review_origin", "invented"),
                           ("task_hash", None), ("task_hash", "short"), ("protocol_hash", "b" * 64)):
            state = state_fixture()
            state["active_task"][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValidationError):
                inspection_binding(state)
        state = state_fixture()
        state["active_task"]["protocol"]["checks"][0]["region_id"] = "staging"
        with self.assertRaises(ValidationError):
            inspection_binding(state)

    def test_binding_rejects_malformed_state_identifiers(self):
        for key, value in (("session_id", "../private"), ("session_id", ""), ("session_id", None),
                           ("session_id", "x" * 161), ("revision", True), ("revision", -1), ("revision", "4")):
            state = state_fixture()
            state[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValidationError):
                inspection_binding(state)
        for state in (None, [], {}, {"active_task": None}, {"active_task": {"protocol": None}}):
            with self.subTest(state=state), self.assertRaises(ValidationError):
                inspection_binding(state)

    def test_binding_rejects_core_predicates_not_implemented_by_live_job_bridge(self):
        alternatives = [
            {"id": "visible-glass", "kind": "visible", "object_id": "glass", "region_id": "dispense", "expected": True},
            {"id": "count-glass", "kind": "count", "object_id": "glass", "region_id": "dispense", "expected": 1},
            dict(state_fixture()["active_task"]["protocol"]["checks"][0], min_stable_samples=3),
        ]
        for check in alternatives:
            state = state_fixture()
            state["active_task"]["protocol"]["checks"] = [check]
            state["active_task"]["protocol_hash"] = Protocol.from_dict(state["active_task"]["protocol"]).content_hash
            with self.subTest(check=check), self.assertRaises(ValidationError):
                inspection_binding(state)

    def test_dispatch_accepts_only_current_awaiting_capture_binding(self):
        state = pending_state()
        binding = inspection_binding(state)
        self.assertEqual(binding, validate_binding(binding, state))
        for key, value in (("busy", True), ("busy", None), ("error", "private failure"),
                           ("phase", "verified"), ("phase", "checking"), ("phase", "needs_evidence")):
            changed = deepcopy(state)
            changed[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValidationError):
                validate_binding(binding, changed)
        # Completion validation checks identity and does not accidentally demand
        # that a finished job still have a valid capture token.
        self.assertEqual(binding, validate_binding(binding, state_fixture(), require_ready=False))
        with self.assertRaises(ValidationError):
            validate_binding(binding, state, require_ready="false")

    def test_task_session_protocol_or_revision_change_refuses_dispatch_and_completion(self):
        state = pending_state()
        binding = inspection_binding(state)
        changed_states = []
        for key, value in (("session_id", "interactive-another-session"), ("revision", 5)):
            changed_states.append(dict(state, **{key: value}))
        changed = deepcopy(state)
        changed["active_task"]["task_hash"] = "c" * 64
        changed_states.append(changed)
        changed = deepcopy(state)
        changed["active_task"]["protocol"]["checks"][0]["region_id"] = "staging"
        changed["active_task"]["protocol_hash"] = Protocol.from_dict(changed["active_task"]["protocol"]).content_hash
        changed_states.append(changed)
        for changed in changed_states:
            for ready in (True, False):
                with self.subTest(changed=changed, ready=ready), self.assertRaises(ValidationError):
                    validate_binding(binding, changed, require_ready=ready)
            with self.assertRaises(ValidationError):
                public_inspection_result(binding, changed)

    def test_unknown_or_extra_binding_fields_are_not_silently_accepted(self):
        state = pending_state()
        binding = inspection_binding(state)
        bad = [None, [], {}, dict(binding, capture_token="secret"), dict(binding, revision=True),
               dict(binding, task_hash="A" * 64), dict(binding, protocol_hash="short")]
        for key in binding:
            bad.append({name: value for name, value in binding.items() if name != key})
        for candidate in bad:
            with self.subTest(candidate=candidate), self.assertRaises(ValidationError):
                validate_binding(candidate, state)

    def test_stopped_acquisition_cannot_be_bypassed_by_job_dispatch(self):
        for status in ("budget_exhausted", "unsupported_contract", "timeline_exhausted", "invented", []):
            state = pending_state()
            state["acquisition"] = {"status": status}
            with self.subTest(status=status), self.assertRaises(ValidationError):
                validate_binding(inspection_binding(state), state)

    def test_public_pass_is_a_snapshot_outcome_without_physical_authority(self):
        state = state_fixture()
        result = public_inspection_result(inspection_binding(state), state)
        self.assertEqual("pass", result["status"])
        self.assertTrue(result["completed"])
        self.assertTrue(result["current_at_snapshot"])
        self.assertEqual("supported_completion", result["reason_code"])
        self.assertEqual([], result["evidence_requests"])
        self.assertFalse(result["physical_action_authorized"])
        self.assertFalse(result["physical_action_verified"])
        self.assertEqual("injected_client_not_verified_live", result["inference_origin"])
        self.assertFalse(result["freshness"]["capture_time_independently_verified"])
        self.assertEqual(result, json.loads(json.dumps(result, allow_nan=False)))
        self.assertEqual("pass", state["status"])

    def test_public_fail_requires_consistent_observed_region_and_aggregate(self):
        state = state_fixture()
        state.update(status="fail", phase="mistake")
        state["checks"][0].update(status="fail", observed_region="staging")
        result = public_inspection_result(inspection_binding(state), state)
        self.assertEqual("fail", result["status"])
        self.assertEqual("staging", result["checks"][0]["observed_region"])
        self.assertFalse(result["physical_action_authorized"])

    def test_unknown_preserves_typed_object_specific_missing_evidence(self):
        state = state_fixture()
        state.update(phase="unknown", status="unknown")
        state["checks"][0].update(status="unknown", observed_region=None, evidence_ids=[])
        state["acquisition"] = {"status": "ready", "attempts_used": 1, "attempts_remaining": 1,
            "max_attempts": 2, "requests": [{"check_id": "glass-under-spout", "object_id": "glass",
            "region_id": "dispense", "gap_codes": ["identity_ambiguous", "partial_visibility"]}]}
        result = public_inspection_result(inspection_binding(state), state)
        self.assertEqual("unknown", result["status"])
        self.assertTrue(result["current_at_snapshot"])
        self.assertEqual("specific_evidence_required", result["reason_code"])
        request = result["evidence_requests"][0]
        self.assertEqual("glass", request["object_id"])
        self.assertEqual(["identity_ambiguous", "partial_visibility"], request["gap_codes"])
        self.assertTrue(result["acquisition"]["follow_up_allowed"])

    def test_missing_evidence_requests_cannot_inject_unknown_codes_or_another_object(self):
        for update in ({"gap_codes": ["private-error-message"]}, {"object_id": "another-target"},
                       {"region_id": "staging"}, {"gap_codes": [None]}, {"check_id": []}):
            state = state_fixture()
            state.update(phase="unknown", status="unknown")
            state["checks"][0].update(status="unknown", observed_region=None, evidence_ids=[])
            row = {"check_id": "glass-under-spout", "object_id": "glass", "region_id": "dispense",
                   "gap_codes": ["partial_visibility"], **update}
            state["acquisition"] = {"requests": [row]}
            with self.subTest(update=update):
                result = public_inspection_result(inspection_binding(state), state)
                self.assertEqual(["missing_current_evidence"], result["evidence_requests"][0]["gap_codes"])

    def test_expired_or_historical_success_is_exposed_as_unknown(self):
        for key, value in (("source_kind", "recorded"), ("source_kind", "none"), ("source_kind", []),
                           ("stale", True), ("stale", None), ("evidence_age_ms", 30_001),
                           ("evidence_age_ms", -1), ("evidence_age_ms", True), ("evidence_age_ms", None)):
            state = state_fixture()
            state[key] = value
            with self.subTest(key=key, value=value):
                result = public_inspection_result(inspection_binding(state), state)
                self.assertEqual("unknown", result["status"])
                self.assertFalse(result["current_at_snapshot"])
                self.assertIsNone(result["checks"][0]["observed_region"])
                self.assertEqual([], result["checks"][0]["evidence_ids"])

    def test_each_reviewed_check_can_have_a_stricter_freshness_window(self):
        state = state_fixture()
        state["active_task"]["protocol"]["checks"][0]["max_age_ms"] = 250
        state["active_task"]["protocol_hash"] = Protocol.from_dict(state["active_task"]["protocol"]).content_hash
        result = public_inspection_result(inspection_binding(state), state)
        self.assertEqual("unknown", result["status"])
        self.assertEqual(250, result["freshness"]["window_ms"])

    def test_busy_error_unfinished_and_malformed_phases_never_keep_pass(self):
        for key, value in (("busy", True), ("error", "secret API details"), ("phase", "checking"),
                           ("phase", "error"), ("phase", []), ("phase", "awaiting_capture")):
            state = state_fixture()
            state[key] = value
            with self.subTest(key=key, value=value):
                result = public_inspection_result(inspection_binding(state), state)
                self.assertEqual("unknown", result["status"])
                self.assertFalse(result["current_at_snapshot"])

    def test_forged_aggregate_or_phase_cannot_override_check_status(self):
        for update in ({"status": "fail"}, {"phase": "mistake"}, {"status": "unknown"}):
            state = dict(state_fixture(), **update)
            result = public_inspection_result(inspection_binding(state), state)
            self.assertEqual("unknown", result["status"])
            self.assertEqual("inconsistent_completion_verdict", result["reason_code"])

    def test_missing_duplicate_or_unsupported_images_cannot_keep_pass(self):
        for images in (None, [], [{"id": "one"}], [{"id": "same"}] * 2,
                       [{"id": "../private"}, {"id": "two"}], [None, {"id": "two"}]):
            state = dict(state_fixture(), images=images)
            with self.subTest(images=images):
                result = public_inspection_result(inspection_binding(state), state)
                self.assertEqual("unknown", result["status"])

    def test_exact_check_coverage_and_observation_consistency_are_required(self):
        checks = state_fixture()["checks"]
        candidates = [None, [], checks * 2, [None]]
        for update in ({"check_id": "invented"}, {"check_id": []}, {"object_id": "spoon"},
                       {"expected_region": "staging"}, {"status": "invented"}, {"status": "fail"},
                       {"observed_region": "staging"}, {"observed_region": None},
                       {"evidence_ids": []}, {"evidence_ids": ["/private/key"]},
                       {"evidence_ids": [None]}, {"evidence_ids": ["one", "one"]}):
            candidates.append([{**checks[0], **update}])
        for candidate in candidates:
            state = dict(state_fixture(), checks=candidate)
            with self.subTest(checks=candidate):
                result = public_inspection_result(inspection_binding(state), state)
                self.assertEqual("unknown", result["status"])
                self.assertEqual("unsupported_completion_evidence", result["reason_code"])

    def test_raw_provider_data_paths_tokens_and_freeform_text_are_not_public(self):
        state = state_fixture()
        state.update(capture_token="secret-capture-token", csrf="private-csrf", error=None,
                     provider_response={"secret-api-key": "never"}, source_note="/private/path")
        state["active_task"]["authorization_reference"] = "private-authorization"
        state["checks"][0].update(reason="private-provider-details", grounding_reason="/private/grounding.json")
        state["images"][0].update(url="file:///private/photo.jpg", path="/private/photo.jpg")
        state["usage"]["requests"] = [{"raw": "secret-ledger-response"}]
        result = public_inspection_result(inspection_binding(state), state)
        serialized = json.dumps(result)
        for secret in ("secret-capture-token", "private-csrf", "secret-api-key", "/private/", "private-authorization",
                       "private-provider-details", "secret-ledger-response"):
            self.assertNotIn(secret, serialized)
        self.assertEqual({"request_attempts": 2, "estimated_usd": 0.005}, result["usage"])

    def test_usage_disallows_boolean_negative_or_nonfinite_amounts(self):
        for value in (math.inf, math.nan, -1, True, "0.01"):
            state = state_fixture()
            state["usage"] = {"request_attempts": True, "input_tokens": -1, "estimated_usd": value}
            with self.subTest(value=value):
                result = public_inspection_result(inspection_binding(state), state)
                self.assertIsNone(result["usage"])
                json.dumps(result, allow_nan=False)

    def test_stopped_or_malformed_acquisition_preserves_stop_without_raw_reason(self):
        for update in ({"status": "budget_exhausted", "attempts_used": 2, "attempts_remaining": 0},
                       {"status": "unsupported_contract"}, {"status": "timeline_exhausted"},
                       {"status": []}, {"attempts_used": True}, {"attempts_remaining": 99}):
            state = state_fixture()
            state["acquisition"] = {"status": "ready", "attempts_used": 1, "attempts_remaining": 1,
                "max_attempts": 2, "stop_reason": "private message", **update}
            with self.subTest(update=update):
                result = public_inspection_result(inspection_binding(state), state)
                self.assertFalse(result["acquisition"]["follow_up_allowed"])
                self.assertNotIn("private message", json.dumps(result))

    def test_unknown_inference_origin_is_not_reclassified_as_live(self):
        state = state_fixture()
        state["inference_origin"] = "private-provider-name"
        result = public_inspection_result(inspection_binding(state), state)
        self.assertEqual("unspecified", result["inference_origin"])
        self.assertNotIn("private-provider-name", json.dumps(result))

    def test_real_engine_snapshot_projects_authored_injected_result_and_binds_completion(self):
        # Actual engine lifecycle, but authored pixels and injected model output:
        # this establishes interoperability, never real-world model accuracy.
        from proofloop.interactive import InteractiveEngine
        from tests.test_interactive import FakeClient, FakeCompileClient, REGIONS, image
        with tempfile.TemporaryDirectory() as directory:
            client = FakeClient()
            compiler = FakeCompileClient(placements={"fork": "left", "spoon": "right"})
            engine = InteractiveEngine(Path(directory), client_factory=lambda: client,
                                       compile_client_factory=lambda: compiler)
            draft = engine.draft_task({"requirements": "Fork left and spoon right.",
                                       "objects": ["fork", "spoon"], "confirm_unique_targets": True})["task_draft"]
            engine.activate_task({"draft_id": draft["draft_id"], "draft_hash": draft["draft_hash"],
                                  "confirm_requirements": True, "confirm_unique_targets": True})
            pending = engine.new_inspection()
            binding = validate_binding(inspection_binding(pending), pending)
            completed = engine.inspect_pair({"capture_token": pending["capture_token"], "source": "upload",
                "images": [image(91), image(92)], "regions": deepcopy(REGIONS),
                "confirm_regions": True, "confirm_unique_targets": True,
                "confirm_fresh_capture": True, "confirm_pair_spacing": True})
            result = public_inspection_result(binding, completed)
            self.assertEqual("pass", result["status"])
            self.assertEqual(4, result["usage"]["request_attempts"])
            self.assertEqual("injected_client_not_verified_live", result["inference_origin"])
            self.assertTrue(all(row["evidence_ids"] for row in result["checks"]))
            self.assertNotIn(pending["capture_token"], json.dumps(result))
            engine.new_inspection()
            with self.assertRaises(ValidationError):
                public_inspection_result(binding, engine.snapshot())


if __name__ == "__main__":
    unittest.main()
