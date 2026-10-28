"""Deterministic, source-scoped explanations of already computed checks.

This presentation bridge does not run perception, infer physical causes, or
authorize action. It never classifies a generated reason string. Callers retain
responsibility for evaluating checks and declaring whether their evidence is
current; historical sources always remain historical even if current=True.
"""
from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from typing import Any

from .core import CheckResult, Observation, Protocol, Status, ValidationError


_LABEL = re.compile(r"^[\w .:()\-]{1,160}$", re.UNICODE)
_EVIDENCE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:\-]{0,159}$")
_LOCAL_REFERENCE = re.compile(r"^(?:sam-grounding://[0-9a-f]{64}|fixture://proofloop/[A-Za-z0-9_-]{1,80}/[A-Za-z0-9_-]{1,80})$")
_HISTORICAL = {"recorded", "recorded_reference", "historical", "historical_recording",
               "external_recorded_video", "owned_phone", "saved_live_meta"}
_ORIGINS = _HISTORICAL | {"live", "camera", "upload", "none", "unspecified", "authored_fixture"}
_GAPS = {
    "missing_current_evidence": "There is no supported observation for this check.",
    "stale_evidence": "The supporting observation has expired.",
    "not_detected": "A target detection is unsupported; this does not prove the target is absent.",
    "multiple_candidates": "More than one candidate prevents support for the declared unique target.",
    "identity_ambiguous": "The target's identity is unconfirmed.",
    "partial_visibility": "The entire target's visibility is unconfirmed.",
    "low_confidence": "A reported score is below the reviewed threshold.",
    "boundary_unconfirmed": "The target's complete placement inside a reviewed region is unconfirmed.",
    "invalid_detection": "Usable detection geometry is unavailable.",
    "mask_unavailable": "A usable decoded mask is unavailable.",
    "temporal_inconsistency": "The supplied views support different target states.",
    "insufficient_distinct_views": "The required distinct, stable views are missing.",
    "calibration_required": "The required region needs reviewed image calibration.",
    "unsupported_observation": "The available observation does not support this check.",
    "unsupported_stability_requirement": "The inspection method cannot meet the required view count.",
    "absence_validation_required": "Complete scene coverage or validated absence is unproven.",
    "temporal_history_required": "The required ordered checkpoint and event history is missing.",
}
_ACTIONS = {
    "missing_current_evidence": "Capture two new, clear views of the target and reviewed boundaries.",
    "stale_evidence": "Inspect the current arrangement with two new views.",
    "not_detected": "Show the entire target in focus with the reviewed boundaries visible in both new views.",
    "multiple_candidates": "Review target identity and uniqueness before another inspection.",
    "identity_ambiguous": "Show distinguishing target features in both new views; review the target concept if ambiguity remains.",
    "partial_visibility": "Show the entire target, including its edges, with hands clear in both new views.",
    "low_confidence": "Use focused, well-lit views of the entire target; keep the reviewed threshold unchanged.",
    "boundary_unconfirmed": "Review image calibration and show the target edges and region boundaries clearly.",
    "invalid_detection": "Capture clear views of the target and reviewed boundaries; unsupported geometry cannot certify placement.",
    "mask_unavailable": "Capture two clear views; stop if usable mask support remains unavailable after the follow-up limit.",
    "temporal_inconsistency": "Keep the arrangement and camera unchanged between two new views.",
    "insufficient_distinct_views": "Capture two entirely new views at least one second apart with the arrangement unchanged.",
    "calibration_required": "Review and approve the required region against the current image before inspecting again.",
    "unsupported_observation": "Obtain two clear, supported views before considering a repair.",
    "unsupported_stability_requirement": "Review the task or choose a separately validated evidence method.",
    "absence_validation_required": "Use a method that validates complete scene coverage, or review the task requirement.",
    "temporal_history_required": "Supply a complete ordered event and checkpoint history; current images cannot establish an earlier action.",
}
_CAUSAL_LIMIT = (
    "This explains the recorded check, not the physical cause. Target identity is not independently "
    "proved by a prompted mask, and a passing state does not prove that a proposed action occurred."
)


def _label(value: Any, name: str, *, nullable: bool = False) -> str | None:
    if nullable and value is None:
        return None
    if not isinstance(value, str) or not _LABEL.fullmatch(value) or not value.strip():
        raise ValidationError(f"Diagnostic {name} must be a concise single-line identifier")
    return value


def _evidence_ids(values: Any) -> list[str]:
    if not isinstance(values, (list, tuple)):
        raise ValidationError("Diagnostic evidence IDs must be a list")
    if any(not isinstance(value, str) or not (_EVIDENCE_ID.fullmatch(value) or _LOCAL_REFERENCE.fullmatch(value))
           for value in values):
        raise ValidationError("Diagnostic evidence IDs must be public-safe identifiers")
    return list(dict.fromkeys(values))


def explain_checks(protocol: Protocol, checks: Sequence[Mapping[str, Any] | CheckResult], *,
                   observations: Sequence[Observation] | None = None,
                   acquisition: Mapping[str, Any] | None = None,
                   evidence_origin: str = "unspecified", current: bool = True) -> dict[str, Any]:
    """Describe exact protocol check coverage using typed fields only.

    Public placement DTOs and core CheckResults are accepted. Optional validated
    Observations add provenance and independently cross-check the presentation
    fields; they do not replace checker evaluation. Unknown/malformed coverage,
    mismatched target/region, and inconsistent known verdicts fail closed.
    ``repair_proposal_eligible`` only permits considering the existing bounded
    planner; ``physical_action_allowed`` is always False.
    """
    if not isinstance(protocol, Protocol) or type(current) is not bool:
        raise ValidationError("Diagnostics need a validated protocol and explicit current flag")
    if not isinstance(evidence_origin, str) or evidence_origin not in _ORIGINS:
        raise ValidationError("Unsupported diagnostic evidence origin")
    if not isinstance(checks, (list, tuple)):
        raise ValidationError("Diagnostic checks must be a list")
    definitions = {check.id: check for check in protocol.checks}
    for check in protocol.checks:
        _label(check.id, "check ID")
        _label(check.object_id, "object ID", nullable=True)
        _label(check.region_id, "region ID", nullable=True)
        _label(check.checkpoint_id, "checkpoint ID", nullable=True)
        _label(check.event_id, "event ID", nullable=True)
    rows: dict[str, Mapping[str, Any]] = {}
    for item in checks:
        row = item.to_dict() if isinstance(item, CheckResult) else item
        if not isinstance(row, Mapping):
            raise ValidationError("Diagnostic checks must be objects")
        check_id = row.get("check_id")
        if not isinstance(check_id, str) or check_id not in definitions or check_id in rows:
            raise ValidationError("Diagnostic check IDs must cover the protocol exactly once")
        check = definitions[check_id]
        if (("object_id" in row and row["object_id"] != check.object_id)
                or ("expected_region" in row and row["expected_region"] != check.region_id)
                or ("region_id" in row and row["region_id"] != check.region_id)):
            raise ValidationError("Diagnostic check target or requirement differs from its protocol")
        if not isinstance(row.get("status"), str) or row["status"] not in {status.value for status in Status}:
            raise ValidationError("Diagnostic checks need a supported status")
        observed = row.get("observed_region")
        if check.kind == "location" and observed is not None and observed not in protocol.regions:
            raise ValidationError("Diagnostic observed region is outside the reviewed task")
        confidence = row.get("confidence")
        if confidence is not None and (type(confidence) not in (float, int)
                or not math.isfinite(confidence) or not 0 <= confidence <= 1):
            raise ValidationError("Diagnostic confidence must be finite and within [0, 1]")
        _evidence_ids(row.get("evidence_ids", []))
        rows[check_id] = row
    if set(rows) != set(definitions):
        raise ValidationError("Diagnostic checks must cover the protocol exactly once")

    selected: dict[str, Observation] = {}
    if observations is not None:
        if not isinstance(observations, (list, tuple)) or any(not isinstance(item, Observation) for item in observations):
            raise ValidationError("Diagnostic observations must be validated Observation values")
        epochs = {(item.session_id, item.attempt_id, item.revision) for item in observations}
        if len(epochs) > 1:
            raise ValidationError("Diagnostic observations cannot mix sessions, attempts or revisions")
        for item in observations:
            if (item.spec_hash != protocol.content_hash or item.object_id not in protocol.objects
                    or item.region_id not in protocol.regions
                    or item.kind == "location" and item.value is not None and item.value not in protocol.regions):
                raise ValidationError("Diagnostic observation differs from the reviewed task")
            _evidence_ids([item.evidence_id])
        for check in protocol.checks:
            matching = [item for item in observations if item.kind == check.kind
                        and item.object_id == check.object_id and item.region_id == check.region_id]
            if matching:
                selected[check.id] = max(matching, key=lambda item: item.timestamp_ms)

    requests: dict[str, Mapping[str, Any]] = {}
    acquisition_status = None
    if acquisition is not None:
        if not isinstance(acquisition, Mapping) or not isinstance(acquisition.get("requests"), list):
            raise ValidationError("Diagnostic acquisition must contain typed requests")
        acquisition_status = acquisition.get("status")
        if not isinstance(acquisition_status, str) or acquisition_status not in {"ready", "not_needed", "budget_exhausted", "timeline_exhausted", "unsupported_contract"}:
            raise ValidationError("Diagnostic acquisition status is invalid")
        for request in acquisition["requests"]:
            if not isinstance(request, Mapping):
                raise ValidationError("Diagnostic evidence requests must be objects")
            check_id = request.get("check_id")
            if (not isinstance(check_id, str) or check_id not in rows or check_id in requests or rows[check_id]["status"] != "unknown"
                    or request.get("object_id") != definitions[check_id].object_id
                    or request.get("region_id") != definitions[check_id].region_id):
                raise ValidationError("Diagnostic evidence request does not bind its unknown check")
            codes = request.get("gap_codes")
            if (not isinstance(codes, list) or not codes or any(not isinstance(code, str) or code not in _GAPS for code in codes)
                    or len(codes) != len(set(codes))):
                raise ValidationError("Diagnostic requests require supported typed evidence gaps")
            requests[check_id] = request

    historical = evidence_origin in _HISTORICAL
    is_current = current and not historical
    has_unknown = any(row["status"] == "unknown" for row in rows.values())
    has_fail = any(row["status"] == "fail" for row in rows.values())
    repair_eligible = bool(is_current and not has_unknown and has_fail
                           and all(definitions[key].kind == "location" for key, row in rows.items() if row["status"] == "fail"))
    output = []
    for check in protocol.checks:
        row, observation = rows[check.id], selected.get(check.id)
        decision = row["status"]
        observed = row.get("observed_region") if check.kind == "location" else None
        confidence = row.get("confidence")
        if observation is not None:
            if (check.kind == "location" and "observed_region" in row and observed != observation.value
                    or confidence is not None and confidence != observation.confidence):
                raise ValidationError("Diagnostic presentation differs from its observation")
            observed = observation.value if check.kind == "location" else None
            confidence = observation.confidence
        if check.kind == "location" and decision != "unknown":
            if observed is None or (decision == "pass") != (observed == check.region_id):
                raise ValidationError("Diagnostic known placement contradicts its observed region")
        if decision != "unknown" and (confidence is not None and confidence < check.min_confidence
                or observation is not None and (observation.value is None
                                                or observation.stable_samples < check.min_stable_samples)):
            raise ValidationError("Diagnostic known decision has insufficient observation support")
        if observation is not None and decision != "unknown" and check.kind in ("count", "visible"):
            if (decision == "pass") != (observation.value == check.expected):
                raise ValidationError("Diagnostic known decision contradicts its observed value")
            complete = observation.fully_visible and observation.absence_validated
            if (check.kind == "count" and not complete and observation.value <= check.expected
                    or check.kind == "visible" and observation.value is False and not complete):
                raise ValidationError("Diagnostic known decision lacks complete absence or count support")

        if check.kind == "location":
            expected_fact = f"The reviewed task requires {check.object_id} in {check.region_id}."
        elif check.kind == "visible":
            expected_fact = f"The reviewed task requires {check.object_id} visibility to be {str(check.expected).lower()} in {check.region_id}."
        elif check.kind == "count":
            expected_fact = f"The reviewed task requires exactly {check.expected} {check.object_id} in {check.region_id}."
        else:
            expected_fact = f"The reviewed task requires checkpoint {check.checkpoint_id} before event {check.event_id}."

        codes = list(requests[check.id]["gap_codes"]) if check.id in requests else []
        if decision == "unknown" and not codes:
            if check.kind == "before":
                codes = ["temporal_history_required"]
            elif (confidence is not None and confidence < check.min_confidence
                  and (observed is not None or observation is not None and observation.value is not None)):
                codes = ["low_confidence"]
            elif observation is not None and observation.value is not None and observation.stable_samples < check.min_stable_samples:
                codes = ["insufficient_distinct_views"]
            else:
                codes = ["unsupported_observation"]
        scope = "historical" if historical else "current" if is_current else "not_current"
        if decision == "unknown":
            observed_fact = " ".join(_GAPS[code] for code in codes)
            code = codes[0]
            next_action = " ".join(dict.fromkeys(_ACTIONS[item] for item in codes))
        elif check.kind == "location":
            code = "requirement_satisfied" if decision == "pass" else "wrong_region"
            prefix = "Recorded observations" if historical else "The supplied observations"
            observed_fact = f"{prefix} support {check.object_id} in {observed}."
            next_action = ("No repair is requested by this check. Use the separate preparation gate before continuing."
                           if decision == "pass" else
                           "Resolve all unknown checks before considering a physical repair." if has_unknown else
                           "Review the separate bounded repair proposal, then obtain entirely new evidence. This explanation does not authorize movement.")
        else:
            code = "requirement_satisfied" if decision == "pass" else "requirement_contradicted"
            observed_fact = ("The recorded checker supports this reviewed requirement." if decision == "pass"
                             else "The recorded checker contradicts this reviewed requirement.")
            next_action = "Review the check and use a separately supported action or evidence method."
        terminal = acquisition_status in {"budget_exhausted", "timeline_exhausted", "unsupported_contract"}
        if acquisition_status == "budget_exhausted":
            next_action = "Stop follow-up capture: the attempt limit is reached. Review the task or evidence method."
        elif acquisition_status == "timeline_exhausted":
            next_action = "Stop this recorded timeline: no eligible later source pair remains."
        elif acquisition_status == "unsupported_contract":
            next_action = "Review the task or use an evidence method that supports its requirements."
        if historical:
            next_action = (next_action + " This historical record cannot authorize a current move."
                           if terminal else
                           "Start a separate current inspection. Historical evidence cannot authorize a current move.")
            code = "historical_observation"
        elif not is_current:
            next_action = (next_action + " Expired or invalidated evidence cannot authorize a move."
                           if terminal else
                           "Inspect the current scene again. Expired or invalidated evidence cannot authorize a move.")
            code = "evidence_not_current"
        ids = _evidence_ids(row.get("evidence_ids", []))
        if observation is not None and observation.evidence_id not in ids:
            ids.append(observation.evidence_id)
        output.append({"check_id": check.id, "object_id": check.object_id,
                       "expected_region": check.region_id, "observed_region": observed,
                       "decision": decision, "diagnostic_code": code, "gap_codes": codes,
                       "observed_fact": observed_fact, "expected_fact": expected_fact,
                       "support_limit": _CAUSAL_LIMIT, "next_action": next_action,
                       "evidence_ids": ids, "evidence_scope": scope,
                       "physical_cause_observed": False})
    return {"schema_version": 1, "protocol_hash": protocol.content_hash,
            "evidence_origin": evidence_origin, "current": is_current,
            "overall_decision": "fail" if has_fail else "unknown" if has_unknown else "pass",
            "cause_not_observed": True, "physical_action_allowed": False,
            "repair_proposal_eligible": repair_eligible, "checks": output,
            "claim_limit": _CAUSAL_LIMIT}
