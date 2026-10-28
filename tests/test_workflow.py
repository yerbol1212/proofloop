"""Authored workflow fixtures establish gate semantics, not model performance."""
from copy import deepcopy
import unittest

from proofloop.compiler import authored_coffee_protocol
from proofloop.core import ValidationError
from proofloop.workflow import DECLARATION_OUTCOME, validate_continuation, workflow_status


def ready_state():
    envelope = authored_coffee_protocol()
    return {"active_task": {"protocol": envelope["protocol"], "protocol_hash": envelope["content_hash"],
            "task_hash": "a" * 64, "reviewed": True, "review_origin": "local_operator", "human_reviewed": True},
            "revision": 4, "status": "pass", "phase": "verified", "stale": False, "error": None,
            "busy": False, "source_kind": "upload", "images": [{"id": "one"}, {"id": "two"}],
            "checks": [{"check_id": "glass-under-spout", "object_id": "glass", "expected_region": "dispense", "status": "pass"}]}


def valid_replay(state):
    return {"valid": True, "task_hash": state["active_task"]["task_hash"],
            "protocol_hash": state["active_task"]["protocol_hash"], "revision": state["revision"]}


def confirmation(state):
    return {"task_hash": state["active_task"]["task_hash"], "revision": state["revision"],
            "confirm_operator_continuation": True}


class WorkflowTests(unittest.TestCase):
    def test_ready_requires_bound_saved_replay_and_separate_declaration(self):
        state = ready_state()
        unchecked = workflow_status(state)
        self.assertTrue(unchecked["can_check"])
        self.assertFalse(unchecked["can_continue"])
        checked = workflow_status(state, valid_replay(state))
        self.assertTrue(checked["can_continue"])
        self.assertEqual("ready", checked["phase"])
        self.assertEqual(["pass", "pass", "pass", "unknown"], [gate["status"] for gate in checked["gate_checks"]])
        record = validate_continuation(state, confirmation(state), valid_replay(state))
        self.assertEqual(DECLARATION_OUTCOME, record["outcome"])
        self.assertTrue(record["human_declared"])
        self.assertFalse(record["physical_action_verified"])

    def test_unknown_fail_expired_historical_busy_error_and_missing_pair_block(self):
        for key, value in (("stale", True), ("source_kind", "recorded"), ("source_kind", "none"),
                           ("busy", True), ("error", "Upstream inspection failed"), ("images", [{"id": "only"}]),
                           ("status", "unknown"), ("phase", "awaiting_capture")):
            state = ready_state()
            state[key] = value
            with self.subTest(key=key, value=value):
                self.assertFalse(workflow_status(state, valid_replay(state))["can_continue"])
                with self.assertRaises(ValidationError):
                    validate_continuation(state, confirmation(state), valid_replay(state))
        for verdict in ("unknown", "fail"):
            state = ready_state()
            state["checks"][0]["status"] = verdict
            # Even a forged aggregate PASS cannot override individual checks.
            with self.subTest(verdict=verdict), self.assertRaises(ValidationError):
                validate_continuation(state, confirmation(state), valid_replay(state))

    def test_unreviewed_or_changed_protocol_cannot_use_old_pass(self):
        for field, value in (("reviewed", False), ("review_origin", "invented"), ("task_hash", None),
                             ("task_hash", "short"), ("protocol_hash", "b" * 64)):
            state = ready_state()
            state["active_task"][field] = value
            self.assertFalse(workflow_status(state, valid_replay(state))["can_continue"])
        state = ready_state()
        state["active_task"]["protocol"]["checks"][0]["region_id"] = "staging"
        self.assertFalse(workflow_status(state, valid_replay(state))["can_continue"])

    def test_saved_media_failure_or_wrong_task_and_revision_replay_block(self):
        for field, value in (("valid", False), ("valid", "true"), ("task_hash", "b" * 64),
                             ("protocol_hash", "b" * 64), ("revision", 3), ("revision", True)):
            state = ready_state()
            replay = dict(valid_replay(state), **{field: value})
            with self.subTest(field=field, value=value), self.assertRaises(ValidationError):
                validate_continuation(state, confirmation(state), replay)
        state = ready_state()
        replay = valid_replay(state)
        state["active_task"]["task_hash"] = "b" * 64
        self.assertFalse(workflow_status(state, replay)["can_continue"])

    def test_every_required_check_must_match_exact_protocol_once(self):
        for checks in ([], [{"check_id": "invented", "status": "pass"}],
                       [{"check_id": ["invalid"], "status": "pass"}],
                       [dict(ready_state()["checks"][0], expected_region="staging")],
                       [dict(ready_state()["checks"][0], object_id="cup")],
                       ready_state()["checks"] * 2):
            state = ready_state()
            state["checks"] = checks
            self.assertFalse(workflow_status(state, valid_replay(state))["can_continue"])

    def test_submission_must_match_current_exact_task_revision_and_true_confirmation(self):
        state = ready_state()
        for field, value in (("task_hash", "b" * 64), ("revision", 3), ("revision", True),
                             ("confirm_operator_continuation", False), ("confirm_operator_continuation", "true"),
                             ("source_kind", "upload")):
            with self.subTest(field=field), self.assertRaises(ValidationError):
                validate_continuation(state, dict(confirmation(state), **{field: value}), valid_replay(state))

    def test_assistant_declaration_requires_private_authority_and_is_not_human_observation(self):
        state = ready_state()
        payload = dict(confirmation(state), review_origin="assistant_under_user_authorization")
        with self.assertRaises(ValidationError):
            validate_continuation(state, payload, valid_replay(state))
        payload["authorization_reference"] = "Explicit user authorization, private only"
        record = validate_continuation(state, payload, valid_replay(state))
        self.assertFalse(record["human_declared"])
        state["workflow_continuation"] = dict(record, arbitrary_secret="must-not-leak")
        public = workflow_status(state, valid_replay(state))
        self.assertEqual("declared", public["phase"])
        self.assertFalse(public["can_continue"])
        self.assertNotIn("authorization_reference", public["declaration"])
        self.assertNotIn("arbitrary_secret", public["declaration"])
        self.assertFalse(public["declaration"]["human_declared"])
        with self.assertRaisesRegex(ValidationError, "already declared"):
            validate_continuation(state, confirmation(state), valid_replay(state))

    def test_past_declaration_does_not_approve_changed_task_or_revision(self):
        state = ready_state()
        state["workflow_continuation"] = validate_continuation(state, confirmation(state), valid_replay(state))
        for field in ("revision", "task_hash"):
            changed = deepcopy(state)
            if field == "revision":
                changed["revision"] += 1
            else:
                changed["active_task"]["task_hash"] = "b" * 64
            result = workflow_status(changed, valid_replay(changed))
            self.assertIsNone(result["declaration"])
            self.assertTrue(result["can_continue"])
        state["stale"] = True
        result = workflow_status(state, valid_replay(state))
        self.assertEqual("blocked", result["phase"])
        self.assertFalse(result["can_continue"])
        self.assertEqual(DECLARATION_OUTCOME, result["declaration"]["outcome"])

    def test_assistant_task_review_is_labelled_separately_from_local_operator(self):
        state = ready_state()
        state["active_task"].update(review_origin="assistant_under_user_authorization", human_reviewed=False)
        result = workflow_status(state, valid_replay(state))
        self.assertTrue(result["can_continue"])
        self.assertIn("not a local operator review", result["gate_checks"][0]["reason"])


if __name__ == "__main__":
    unittest.main()
