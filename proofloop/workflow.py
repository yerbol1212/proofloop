"""An authored placement preparation gate and a separate operator declaration.

This module neither observes continuation nor authorizes a machine action. A
saved replay proves consistency with recorded observations, not physical safety,
capture time, hidden contents, or the successful completion of a procedure.
"""
from __future__ import annotations

from copy import deepcopy
import re
from typing import Any, Mapping

from .core import Protocol, ValidationError

_HASH = re.compile(r"[0-9a-f]{64}")
CLAIM_LIMIT = "A reviewed visible placement checkpoint followed by an operator declaration. Continuation is not an observed physical action, dispensing permission, or proof of procedure success."
DECLARATION_OUTCOME = "operator_declared_continuation_not_observed"


def _task(state: Mapping[str, Any]):
    task = state.get("active_task")
    if not isinstance(task, dict):
        return None, None
    try:
        protocol = Protocol.from_dict(task["protocol"])
        valid = (task.get("reviewed") is True and task.get("review_origin") in
                 ("local_operator", "assistant_under_user_authorization") and
                 isinstance(task.get("task_hash"), str) and _HASH.fullmatch(task["task_hash"]) and
                 task.get("protocol_hash") == protocol.content_hash and protocol.checks and
                 all(check.kind == "location" and check.unique_target for check in protocol.checks))
        return (task, protocol) if valid else (task, None)
    except (ValidationError, ValueError, TypeError, KeyError, AttributeError):
        return task, None


def _placement_gate(state: Mapping[str, Any], protocol: Protocol | None):
    if protocol is None:
        return "unknown", "Review and activate the exact placement contract first."
    if state.get("busy") is not False:
        return "unknown", "Wait for the current inspection to finish."
    if state.get("error"):
        return "unknown", "Resolve the inspection error and obtain new supported evidence."
    if state.get("source_kind") == "recorded":
        return "unknown", "The reference is historical. Inspect two new views of the current scene."
    if state.get("source_kind") not in ("camera", "upload"):
        return "unknown", "Inspect two fresh views under this reviewed task."
    if state.get("stale") is not False:
        return "unknown", "The placement evidence expired. Inspect two new views before continuing."
    checks = state.get("checks")
    expected = {check.id: check for check in protocol.checks}
    if (not isinstance(checks, list) or len(checks) != len(expected) or
            any(not isinstance(check, dict) or not isinstance(check.get("check_id"), str) or check["check_id"] not in expected or
                check.get("object_id") != expected[check["check_id"]].object_id or
                check.get("expected_region") != expected[check["check_id"]].region_id for check in checks) or
            len({check["check_id"] for check in checks}) != len(expected)):
        return "unknown", "The current checks do not cover the exact reviewed placement contract."
    if any(check.get("status") == "fail" for check in checks):
        return "fail", "A required placement fails. Repair it and inspect two fresh views."
    if any(check.get("status") != "pass" for check in checks):
        return "unknown", "A required placement is unconfirmed. Supply its requested missing evidence."
    images = state.get("images")
    if not isinstance(images, list) or len(images) != 2:
        return "unknown", "The current placement must be supported by two distinct source views."
    if state.get("status") != "pass" or state.get("phase") != "verified":
        return "unknown", "The latest inspection does not support every required placement."
    if type(state.get("revision")) is not int or state["revision"] < 0:
        return "unknown", "The inspection revision is unavailable. Start a new inspection."
    return "pass", "Every required placement passes in the latest current pair. Capture timing remains operator-declared."


def _declaration(state: Mapping[str, Any], task: Mapping[str, Any] | None):
    value = state.get("workflow_continuation")
    if (not task or not isinstance(value, dict) or value.get("outcome") != DECLARATION_OUTCOME or
            value.get("task_hash") != task.get("task_hash") or
            value.get("protocol_hash") != task.get("protocol_hash") or
            type(value.get("revision")) is not int or value["revision"] != state.get("revision")):
        return None
    # Authorization text and arbitrary persisted fields never enter the browser.
    fields = ("outcome", "task_hash", "protocol_hash", "revision", "review_origin", "human_declared",
              "declared_at_ms", "record_id", "record_hash")
    return {**{key: deepcopy(value[key]) for key in fields if key in value},
            "record_hash": value.get("receipt_hash", value.get("record_hash")), "physical_action_verified": False}


def workflow_status(state: Mapping[str, Any], evidence_replay: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Derive preparation gates solely from server state and a bound replay result.

    A replay result must bind the exact task hash, protocol hash and revision.
    The engine is responsible for rehashing media and comparing every replayed
    check against current state before issuing ``valid=True``. It must repeat
    that work when a declaration is submitted, even if a prior UI check passed.
    """
    task, protocol = _task(state)
    review_reason = ("Exact task reviewed by the local operator." if task and task.get("review_origin") == "local_operator" else
                     "Exact task reviewed by the assistant under user authorization; this is not a local operator review.") if protocol else "Review and activate the exact placement contract first."
    placement_status, placement_reason = _placement_gate(state, protocol)
    bound = (isinstance(evidence_replay, Mapping) and protocol is not None and
             evidence_replay.get("task_hash") == task.get("task_hash") and
             evidence_replay.get("protocol_hash") == task.get("protocol_hash") and
             type(evidence_replay.get("revision")) is int and evidence_replay["revision"] == state.get("revision"))
    replay_ok = bool(bound and evidence_replay.get("valid") is True)
    replay_status = "pass" if replay_ok else "fail" if bound and evidence_replay.get("valid") is False else "unknown"
    replay_reason = ("Saved source bytes, masks and decisions replayed for this exact task and revision." if replay_ok else
                     "Saved evidence replay failed. Inspect new evidence before continuing." if replay_status == "fail" else
                     "Check the saved preparation evidence for this exact task and revision.")
    declaration = _declaration(state, task)
    ready = protocol is not None and placement_status == "pass" and replay_ok
    can_check = protocol is not None and placement_status == "pass"
    can_continue = bool(ready and declaration is None)
    blocked_reason = (review_reason if protocol is None else placement_reason if placement_status != "pass" else
                      replay_reason if not replay_ok else None)
    gates = [
        {"id": "task_review", "title": "Review the authored placement contract", "status": "pass" if protocol else "unknown", "reason": review_reason},
        {"id": "current_placements", "title": "Verify the current preparation", "status": placement_status, "reason": placement_reason},
        {"id": "saved_evidence_replay", "title": "Replay the saved placement evidence", "status": replay_status, "reason": replay_reason},
        {"id": "operator_continuation", "title": "Record an explicit continuation declaration", "status": "pass" if declaration else "unknown",
         "reason": "Continuation was declared; no physical continuation was observed." if declaration else "A separate operator declaration is required after the preparation checks pass."},
    ]
    return {"schema_version": "1.0", "title": "Placement preparation checkpoint", "task_hash": task.get("task_hash") if task else None,
            "protocol_hash": task.get("protocol_hash") if task else None, "revision": state.get("revision"),
            "phase": "declared" if ready and declaration else "ready" if ready else "blocked", "gate_checks": gates,
            "can_check": bool(can_check), "can_continue": can_continue, "blocked_reason": blocked_reason,
            "declaration": declaration, "claim_limit": CLAIM_LIMIT}


def validate_continuation(state: Mapping[str, Any], payload: Any,
                          evidence_replay: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Return a declaration record only after freshly replayed preparation passes."""
    allowed = {"task_hash", "revision", "confirm_operator_continuation", "review_origin", "authorization_reference"}
    if not isinstance(payload, dict) or set(payload) - allowed:
        raise ValidationError("Supply the exact task, revision and explicit continuation confirmation")
    task = state.get("active_task") or {}
    if (not isinstance(task, dict) or payload.get("task_hash") != task.get("task_hash") or
            type(payload.get("revision")) is not int or payload["revision"] != state.get("revision")):
        raise ValidationError("The task or inspection changed. Review the current preparation again")
    if payload.get("confirm_operator_continuation") is not True:
        raise ValidationError("Explicitly confirm the continuation declaration")
    origin = payload.get("review_origin", "local_operator")
    reference = payload.get("authorization_reference")
    if (origin not in ("local_operator", "assistant_under_user_authorization") or
            origin == "assistant_under_user_authorization" and
            (not isinstance(reference, str) or not reference.strip() or len(reference) > 4096)):
        raise ValidationError("Assistant continuation requires a direct user authorization reference")
    status = workflow_status(state, evidence_replay)
    if not status["can_continue"]:
        raise ValidationError(status["blocked_reason"] or "Continuation is already declared for this task and revision")
    result = {"schema_version": "1.0", "outcome": DECLARATION_OUTCOME, "task_hash": task["task_hash"],
              "protocol_hash": task["protocol_hash"], "revision": state["revision"], "review_origin": origin,
              "human_declared": origin == "local_operator", "physical_action_verified": False,
              "claim_limit": CLAIM_LIMIT}
    if origin == "assistant_under_user_authorization":
        result["authorization_reference"] = reference.strip()
    return result
