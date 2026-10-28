"""A narrow, deterministic contract between an inspection job and its caller.

Bindings are optimistic concurrency checks, not signatures or credentials. A
persisted result describes its completion snapshot; it never grants continuing
permission to move an object or operate a device. The engine remains responsible
for validating image bytes, calibration, one-use capture tokens and model output.
This module performs no I/O, inference, action dispatch or wall-clock reads.
"""
from __future__ import annotations

from collections.abc import Mapping
import math
import re
from typing import Any

from .core import Protocol, ValidationError


POLICY_VERSION = "placement-job-policy-v1"
_BINDING_FIELDS = {"session_id", "task_hash", "protocol_hash", "revision"}
_HASH = re.compile(r"[0-9a-f]{64}")
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:\-]{0,159}")
_EVIDENCE = re.compile(r"(?:sam-grounding://[0-9a-f]{64}|fixture://proofloop/[A-Za-z0-9_-]{1,80}/[A-Za-z0-9_-]{1,80}|[A-Za-z0-9][A-Za-z0-9_.:\-]{0,159})")
_GAPS = frozenset({
    "missing_current_evidence", "stale_evidence", "not_detected", "multiple_candidates",
    "identity_ambiguous", "partial_visibility", "low_confidence", "boundary_unconfirmed",
    "invalid_detection", "mask_unavailable", "temporal_inconsistency", "insufficient_distinct_views",
    "calibration_required", "unsupported_observation", "unsupported_stability_requirement",
    "absence_validation_required", "temporal_history_required",
})
_ACQUISITION_STATES = frozenset({"ready", "not_needed", "budget_exhausted", "unsupported_contract", "timeline_exhausted"})
_TERMINAL_PHASES = frozenset({"verified", "mistake", "unknown", "error"})
_FRESHNESS_MS = 30_000
_LIMITS = (
    "The verdict describes visible placement in the supplied completion snapshot; persisted results are historical records.",
    "A binding detects a changed session, task, protocol or revision. It is not authentication, a signature or proof of origin.",
    "Capture timing is client-reported or operator-declared, not independently verified.",
    "Prompted SAM masks do not independently establish Spark-assigned target identity; reported scores are not calibrated correctness probabilities.",
    "A passing state does not prove that a proposed physical action occurred and never authorizes one.",
)


def describe_capabilities() -> dict[str, Any]:
    """Return a closed, JSON-serializable descriptor of implemented boundaries.

    Core predicate support is deliberately separate from the currently exposed
    live bridge. An agent must not infer live count, absence or temporal-history
    support from the deterministic core's broader vocabulary.
    """
    return {
        "schema_version": "1.0", "policy_version": POLICY_VERSION,
        "binding_fields": sorted(_BINDING_FIELDS), "binding_is_authorization": False,
        "core_predicates": ["count", "visible", "location", "before"],
        "live_predicates": [{"kind": "location", "requires_unique_target_contract": True}],
        "live_contract": {"review_required": True, "min_objects": 1, "max_objects": 6,
                          "exact_regions": 2, "one_check_per_object": True},
        "evidence_input": {
            "exact_views": 2, "distinct_source_bytes_required": True,
            "previously_inspected_bytes_rejected": True, "same_image_dimensions_required": True,
            "minimum_spacing_ms": 1000, "freshness_window_ms": _FRESHNESS_MS,
            "sources": ["camera", "upload"], "media_types": ["image/jpeg", "image/png"],
            "max_image_bytes": 5 * 1024 * 1024, "max_image_pixels": 16_000_000,
            "guided_mode": {"media_types": ["image/jpeg"], "max_side_pixels": 1280,
                            "exif_allowed": False, "independent_identity_proof": False},
            "reviewed_regions_and_unique_targets_required": True,
            "capture_token": "one-use engine-issued value bound to the current revision",
        },
        "time_basis": {"camera": "client_reported_capture_time",
                       "upload": "server_received_time_capture_age_operator_declared",
                       "capture_time_independently_verified": False},
        "acquisition": {"maximum_follow_up_submissions": 2, "automatic_retry": False,
                        "unknown_requests_specific_evidence": True,
                        "follow_up_limit_cannot_be_reset_by_starting_another_inspection": True},
        "result": {"statuses": ["pass", "fail", "unknown"],
                   "scope": "completion_snapshot_only", "new_revision_invalidates_binding": True,
                   "physical_action_authorized": False, "physical_action_verified": False},
        "limitations": list(_LIMITS),
    }


def _task_protocol(state: Any) -> tuple[Mapping[str, Any], Protocol]:
    if not isinstance(state, Mapping):
        raise ValidationError("Inspection state must be an object")
    task = state.get("active_task")
    if not isinstance(task, Mapping):
        raise ValidationError("An exact reviewed task is required for a bound inspection")
    try:
        protocol = Protocol.from_dict(task.get("protocol"))
    except (ValidationError, TypeError, KeyError, AttributeError) as error:
        raise ValidationError("The active task protocol is invalid") from error
    if (task.get("reviewed") is not True or task.get("review_origin") not in
            ("local_operator", "assistant_under_user_authorization")):
        raise ValidationError("Review and activate the exact task before creating an inspection job")
    if (not isinstance(task.get("task_hash"), str) or not _HASH.fullmatch(task["task_hash"])
            or task.get("protocol_hash") != protocol.content_hash):
        raise ValidationError("The active task and protocol hashes are invalid")
    if (not 1 <= len(protocol.objects) <= 6 or len(protocol.regions) != 2
            or len(protocol.checks) != len(protocol.objects)
            or {check.object_id for check in protocol.checks} != set(protocol.objects)
            or protocol.steps or protocol.checkpoints
            or any(check.kind != "location" or check.unique_target is not True
                   or check.min_stable_samples != 2 for check in protocol.checks)):
        raise ValidationError("Bound live jobs support only one two-view unique-target location check per object")
    return task, protocol


def _binding(value: Any) -> dict[str, Any]:
    if (not isinstance(value, Mapping) or set(value) != _BINDING_FIELDS
            or not isinstance(value.get("session_id"), str) or not _ID.fullmatch(value["session_id"])
            or any(not isinstance(value.get(field), str) or not _HASH.fullmatch(value[field])
                   for field in ("task_hash", "protocol_hash"))
            or type(value.get("revision")) is not int or value["revision"] < 0):
        raise ValidationError("Supply the exact session, task hash, protocol hash and integer revision binding")
    return {key: value[key] for key in sorted(_BINDING_FIELDS)}


def inspection_binding(state: Mapping[str, Any]) -> dict[str, Any]:
    """Identify an exact reviewed task revision without returning capture secrets."""
    task, _ = _task_protocol(state)
    return _binding({"session_id": state.get("session_id"), "task_hash": task["task_hash"],
                     "protocol_hash": task["protocol_hash"], "revision": state.get("revision")})


def validate_binding(expected: Any, state: Mapping[str, Any], *, require_ready: bool = True) -> dict[str, Any]:
    """Reject stale dispatch or completion before spending calls or publishing results.

    ``require_ready=False`` checks identity only and is useful after the worker
    returns. ``inspect_pair`` retains the revision created by ``new_inspection``;
    a completion from another revision is rejected rather than relabeled.
    Callers must hold the engine lock when using this as a dispatch check.
    """
    if type(require_ready) is not bool:
        raise ValidationError("require_ready must be a boolean")
    binding = _binding(expected)
    if binding != inspection_binding(state):
        raise ValidationError("The inspection session, task, protocol or revision changed")
    if require_ready:
        if state.get("busy") is not False:
            raise ValidationError("Wait for the current engine operation before dispatching an inspection")
        if state.get("error") is not None or state.get("phase") != "awaiting_capture":
            raise ValidationError("Start a new inspection and obtain its current capture request before dispatch")
        acquisition = state.get("acquisition")
        if acquisition is not None and (not isinstance(acquisition, Mapping)
                                       or acquisition.get("status") not in ("ready", "not_needed")):
            raise ValidationError("The evidence acquisition policy does not allow another inspection")
    return binding


def _checks(state: Mapping[str, Any], protocol: Protocol) -> tuple[list[dict[str, Any]], bool]:
    raw = state.get("checks")
    expected = {check.id: check for check in protocol.checks}
    valid = isinstance(raw, list) and len(raw) == len(expected)
    rows: dict[str, Any] = {}
    if valid:
        for row in raw:
            if (not isinstance(row, Mapping) or not isinstance(row.get("check_id"), str)
                    or row["check_id"] not in expected or row["check_id"] in rows):
                valid = False
                break
            definition = expected[row["check_id"]]
            ids = row.get("evidence_ids", [])
            observed = row.get("observed_region")
            status = row.get("status")
            if (row.get("object_id") != definition.object_id
                    or row.get("expected_region") != definition.region_id
                    or status not in ("pass", "fail", "unknown")
                    or observed is not None and observed not in protocol.regions
                    or not isinstance(ids, list) or len(ids) != len(set(str(item) for item in ids))
                    or any(not isinstance(item, str) or not _EVIDENCE.fullmatch(item) for item in ids)
                    or status != "unknown" and (not ids or observed is None
                        or (status == "pass") != (observed == definition.region_id))):
                valid = False
                break
            rows[row["check_id"]] = row
    result = []
    for check in protocol.checks:
        row = rows.get(check.id) if valid else None
        result.append({"check_id": check.id, "object_id": check.object_id,
                       "expected_region": check.region_id,
                       "observed_region": row.get("observed_region") if row else None,
                       "status": row["status"] if row else "unknown",
                       "evidence_ids": list(row.get("evidence_ids", [])) if row else []})
    return result, valid


def _requests(state: Mapping[str, Any], protocol: Protocol,
              checks: list[dict[str, Any]], current: bool) -> list[dict[str, Any]]:
    expected = {check.id: check for check in protocol.checks}
    acquisition = state.get("acquisition")
    typed: dict[str, list[str]] = {}
    if isinstance(acquisition, Mapping) and isinstance(acquisition.get("requests"), list):
        for row in acquisition["requests"]:
            if not isinstance(row, Mapping) or not isinstance(row.get("check_id"), str):
                continue
            definition = expected.get(row["check_id"])
            codes = row.get("gap_codes")
            if (definition is not None and row.get("object_id") == definition.object_id
                    and row.get("region_id") == definition.region_id and isinstance(codes, list)
                    and all(isinstance(code, str) and code in _GAPS for code in codes)):
                typed[row["check_id"]] = list(dict.fromkeys(codes))
    result = []
    for row in checks:
        if row["status"] != "unknown":
            continue
        codes = typed.get(row["check_id"], []) if current else []
        if not codes:
            codes = ["stale_evidence" if state.get("stale") is True else "missing_current_evidence"]
        result.append({"check_id": row["check_id"], "object_id": row["object_id"],
                       "expected_region": row["expected_region"], "gap_codes": codes,
                       "required_views": 2, "minimum_spacing_ms": 1000,
                       "physical_action_authorized": False})
    return result


def _acquisition(state: Mapping[str, Any]) -> dict[str, Any] | None:
    raw = state.get("acquisition")
    if raw is None:
        return None
    if (not isinstance(raw, Mapping) or not isinstance(raw.get("status"), str)
            or raw["status"] not in _ACQUISITION_STATES
            or any(type(raw.get(key)) is not int for key in ("attempts_used", "attempts_remaining", "max_attempts"))
            or not 0 <= raw["attempts_used"] <= raw["max_attempts"] <= 2
            or raw["attempts_remaining"] != raw["max_attempts"] - raw["attempts_used"]):
        return {"status": "invalid", "follow_up_allowed": False}
    return {"status": raw["status"], "attempts_used": raw["attempts_used"],
            "attempts_remaining": raw["attempts_remaining"], "max_attempts": raw["max_attempts"],
            "follow_up_allowed": raw["status"] == "ready" and raw["attempts_remaining"] > 0}


def _usage(state: Mapping[str, Any]) -> dict[str, Any] | None:
    raw = state.get("usage")
    if not isinstance(raw, Mapping):
        return None
    result = {key: raw[key] for key in ("request_attempts", "input_tokens", "output_tokens", "sam_images",
                                      "spark_requests_with_unreported_token_usage",
                                      "unsuccessful_spark_requests_with_reported_token_usage")
              if type(raw.get(key)) is int and raw[key] >= 0}
    if type(raw.get("estimated_usd")) in (int, float) and math.isfinite(raw["estimated_usd"]) and raw["estimated_usd"] >= 0:
        result["estimated_usd"] = raw["estimated_usd"]
    return result or None


def public_inspection_result(expected: Any, state: Mapping[str, Any]) -> dict[str, Any]:
    """Project one correctly bound completion snapshot through a closed allowlist.

    Malformed or contradictory evidence becomes UNKNOWN. A mismatched job
    binding is an error, so a result can never be attributed to a newer task.
    ``current_at_snapshot`` is time-scoped, not a statement about a later poll.
    No raw provider response, free-form error, filesystem path, request ledger,
    capture token, CSRF token or continuation authority is returned.
    """
    binding = validate_binding(expected, state, require_ready=False)
    _, protocol = _task_protocol(state)
    checks, checks_valid = _checks(state, protocol)
    phase = state.get("phase")
    completed = isinstance(phase, str) and phase in _TERMINAL_PHASES and state.get("busy") is False
    source = state.get("source_kind")
    age = state.get("evidence_age_ms")
    images = state.get("images")
    pair_valid = (isinstance(images, list) and len(images) == 2
                  and all(isinstance(item, Mapping) and isinstance(item.get("id"), str)
                          and _ID.fullmatch(item["id"]) for item in images)
                  and images[0]["id"] != images[1]["id"])
    window = min(_FRESHNESS_MS, *(check.max_age_ms for check in protocol.checks))
    current = (completed and phase != "error" and state.get("error") is None
               and source in ("camera", "upload") and state.get("stale") is False
               and type(age) is int and 0 <= age <= window and pair_valid and checks_valid)
    reason = "supported_completion"
    if not completed:
        reason = "inspection_not_complete"
    elif phase == "error" or state.get("error") is not None:
        reason = "inspection_failed"
    elif source not in ("camera", "upload"):
        reason = "current_capture_required"
    elif state.get("stale") is not False or type(age) is not int or not 0 <= age <= window:
        reason = "current_evidence_expired_or_unavailable"
    elif not pair_valid or not checks_valid:
        reason = "unsupported_completion_evidence"
    aggregate = ("fail" if any(row["status"] == "fail" for row in checks) else
                 "pass" if all(row["status"] == "pass" for row in checks) else "unknown")
    expected_phase = {"pass": "verified", "fail": "mistake", "unknown": "unknown"}[aggregate]
    if current and (state.get("status") != aggregate or phase != expected_phase):
        current, reason = False, "inconsistent_completion_verdict"
    if not current:
        for row in checks:
            row.update(status="unknown", observed_region=None, evidence_ids=[])
        aggregate = "unknown"
    elif aggregate == "unknown":
        reason = "specific_evidence_required"
    elif aggregate == "fail" and any(row["status"] == "unknown" for row in checks):
        reason = "failure_with_unconfirmed_requirements"
    basis = ("client_reported_capture_time" if source == "camera" else
             "server_received_time_capture_age_operator_declared" if source == "upload" else None)
    origin = state.get("inference_origin")
    if origin not in ("live_meta_api", "injected_client_not_verified_live", "saved_live_meta", "authored_fixture"):
        origin = "unspecified"
    return {
        "schema_version": "1.0", "policy_version": POLICY_VERSION, "binding": binding,
        "status": aggregate, "completed": completed, "reason_code": reason,
        "current_at_snapshot": bool(current), "source_kind": source if source in ("camera", "upload", "recorded", "none") else "unspecified",
        "freshness": {"age_ms": age if type(age) is int and age >= 0 else None,
                      "window_ms": window, "basis": basis,
                      "capture_time_independently_verified": False},
        "inference_origin": origin, "checks": checks,
        "evidence_requests": _requests(state, protocol, checks, bool(current)),
        "acquisition": _acquisition(state), "usage": _usage(state),
        "physical_action_authorized": False, "physical_action_verified": False,
        "limitations": list(_LIMITS),
    }
