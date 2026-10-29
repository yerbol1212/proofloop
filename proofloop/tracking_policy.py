"""Experimental, deterministic tracking diagnostics; no live placement changes.

The caller verifies clip bytes, extracted JPEG bytes and runtime model response
bindings before calling this pure evaluator. ``source_sha256`` identifies the
clip; ``frame_sha256`` identifies the exact extracted semantic input. Neither a
SAM track identifier nor its repeated mask independently establishes identity.

Only initial seed review is authored identity evidence. Distractor rectangles
must come from runtime tracks whose identifiers were reviewed in that seed;
later reference/ground-truth boxes are never accepted by this input contract.
An evidence gap never establishes its physical cause, including occlusion.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
import hashlib
import json
import math
import re
from typing import Any

from .core import ValidationError


TRACKING_POLICY_VERSION = "tracking_reacquisition_v1"
_HASH = re.compile(r"[0-9a-f]{64}")
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}")
_FRAME_FIELDS = {"frame_index", "timestamp_ms", "source_sha256", "frame_sha256", "candidates", "reviewed_distractors"}
_CANDIDATE_FIELDS = {"track_id", "object_id", "box", "confidence", "mask_supported", "fully_visible", "identity_ambiguous", "semantic"}
_SEMANTIC_FIELDS = {"target_id", "track_id", "frame_index", "source_sha256", "frame_sha256", "candidate_box", "decision", "evidence_sha256"}
_SEED_FIELDS = {"target_id", "track_id", "frame_index", "source_sha256", "frame_sha256", "review_sha256", "reviewed_distractor_track_ids"}
_LIMITATIONS = (
    "Experimental tracking support is not placement PASS and does not change the live placement checker.",
    "Initial target and distractor identities are authored seed contracts. Runtime track identifiers do not independently prove object identity.",
    "An evidence gap does not establish physical occlusion, absence, an identity switch or its cause.",
    "Reacquisition requires distinct source-bound semantic views. Same-model agreement remains correlated and is not independent identity proof.",
    "The caller must verify source bytes, frame extraction and model response bindings. This pure policy validates metadata consistency, not API origin.",
    "Later ground-truth boxes and visibility labels belong only in separate scoring; runtime distractor boxes must be propagated model observations.",
    "Reported confidence thresholds and overlap rules are authored development settings, not calibrated reliability guarantees.",
)


def _closed(value: Any, fields: set[str], name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValidationError(f"Tracking {name} must contain only its exact supported fields")
    return value


def _integer(value: Any, name: str, *, minimum: int = 0, maximum: int = 2 ** 53) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValidationError(f"Tracking {name} must be a bounded integer")
    return value


def _hash(value: Any, name: str, *, nullable: bool = False) -> str | None:
    if nullable and value is None:
        return None
    if not isinstance(value, str) or not _HASH.fullmatch(value):
        raise ValidationError(f"Tracking {name} must be an exact SHA256")
    return value


def _number(value: Any, name: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValidationError(f"Tracking {name} must be finite in [0,1]")
    return float(value)


def _box(value: Any) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValidationError("Tracking boxes require four normalized coordinates")
    result = [_number(item, "box coordinate") for item in value]
    if not result[0] < result[2] or not result[1] < result[3]:
        raise ValidationError("Tracking boxes must have positive normalized area")
    return result


def _overlap(first: Sequence[float], second: Sequence[float]) -> bool:
    return (max(first[0], second[0]) < min(first[2], second[2])
            and max(first[1], second[1]) < min(first[3], second[3]))


def _semantic(raw: Any, candidate: Mapping[str, Any], frame: Mapping[str, Any], target_id: str):
    if raw is None:
        return None
    value = _closed(raw, _SEMANTIC_FIELDS, "semantic assessment")
    if (value["target_id"] != target_id or value["track_id"] != candidate["track_id"]
            or type(value["track_id"]) is not int or type(value["frame_index"]) is not int
            or value["frame_index"] != frame["frame_index"]
            or value["source_sha256"] != frame["source_sha256"]
            or value["frame_sha256"] != frame["frame_sha256"]
            or _box(value["candidate_box"]) != candidate["box"]
            or value["decision"] not in ("supported", "unknown", "contradicted")):
        raise ValidationError("Tracking semantic assessment is not bound to this exact clip, JPEG, frame and candidate")
    _hash(value["frame_sha256"], "semantic JPEG hash")
    _hash(value["evidence_sha256"], "semantic evidence hash")
    if value["decision"] == "supported" and (candidate["fully_visible"] is not True
                                                or candidate["identity_ambiguous"] is not False):
        raise ValidationError("Supported tracking semantics require explicit runtime visibility and unambiguous identity")
    return deepcopy(dict(value))


def _frames(raw: Any, target_id: str) -> list[dict[str, Any]]:
    if not isinstance(raw, (list, tuple)) or not 1 <= len(raw) <= 4096:
        raise ValidationError("Supply one to 4096 ordered tracking frames")
    result = []
    previous_index = previous_time = -1
    source = None
    for item in raw:
        frame = dict(_closed(item, _FRAME_FIELDS, "frame"))
        index = _integer(frame["frame_index"], "frame index")
        timestamp = _integer(frame["timestamp_ms"], "source timestamp")
        if index <= previous_index or timestamp <= previous_time:
            raise ValidationError("Tracking source frames and timestamps must strictly increase")
        previous_index, previous_time = index, timestamp
        _hash(frame["source_sha256"], "clip hash")
        _hash(frame["frame_sha256"], "JPEG hash", nullable=True)
        if source is not None and source != frame["source_sha256"]:
            raise ValidationError("One tracking evaluation must stay on one exact clip")
        source = frame["source_sha256"]
        candidates, distractors = frame["candidates"], frame["reviewed_distractors"]
        if (not isinstance(candidates, list) or len(candidates) > 32
                or not isinstance(distractors, list) or len(distractors) > 32):
            raise ValidationError("Tracking frames support at most 32 candidates and 32 reviewed distractor tracks")
        normalized = []
        seen = set()
        for raw_candidate in candidates:
            candidate = dict(_closed(raw_candidate, _CANDIDATE_FIELDS, "candidate"))
            track_id = _integer(candidate["track_id"], "track identifier")
            if track_id in seen or candidate["object_id"] != target_id:
                raise ValidationError("Tracking candidate IDs must be distinct and name the exact target")
            seen.add(track_id)
            candidate["box"] = _box(candidate["box"])
            if candidate["confidence"] is not None:
                candidate["confidence"] = _number(candidate["confidence"], "reported confidence")
            if type(candidate["mask_supported"]) is not bool or any(
                candidate[key] is not None and type(candidate[key]) is not bool
                for key in ("fully_visible", "identity_ambiguous")
            ):
                raise ValidationError("Tracking visibility/identity flags must be runtime booleans or unassessed null")
            candidate["semantic"] = _semantic(candidate["semantic"], candidate, frame, target_id)
            normalized.append(candidate)
        frame["candidates"] = normalized
        normalized_distractors, seen = [], set()
        for raw_distractor in distractors:
            distractor = dict(_closed(raw_distractor, {"track_id", "box", "origin"}, "runtime distractor"))
            identifier = _integer(distractor["track_id"], "distractor track identifier")
            if identifier in seen or distractor["origin"] != "runtime_sam_track":
                raise ValidationError("Distractor rectangles must identify distinct runtime SAM tracks")
            seen.add(identifier)
            distractor["box"] = _box(distractor["box"])
            normalized_distractors.append(distractor)
        frame["reviewed_distractors"] = normalized_distractors
        result.append(frame)
    return result


def _seed(raw: Any, first: Mapping[str, Any], target_id: str):
    if raw is None:
        return None
    seed = dict(_closed(raw, _SEED_FIELDS, "authored seed review"))
    _integer(seed["track_id"], "seed track identifier")
    _integer(seed["frame_index"], "seed frame index")
    _hash(seed["review_sha256"], "seed review hash")
    _hash(seed["frame_sha256"], "seed JPEG hash")
    if (seed["target_id"] != target_id or seed["source_sha256"] != first["source_sha256"]
            or seed["frame_index"] != first["frame_index"] or seed["frame_sha256"] != first["frame_sha256"]):
        raise ValidationError("The authored seed must bind the first exact clip frame and JPEG")
    identifiers = seed["reviewed_distractor_track_ids"]
    if (not isinstance(identifiers, list) or len(identifiers) > 32
            or any(type(value) is not int or value < 0 or value > 2 ** 53 for value in identifiers)
            or len(set(identifiers)) != len(identifiers) or seed["track_id"] in identifiers):
        raise ValidationError("Reviewed distractor IDs must be distinct from the seed target")
    return deepcopy(seed)


def evaluate_tracking(frames: Sequence[Mapping[str, Any]], *, target_id: str,
                      seed: Mapping[str, Any] | None = None, min_confidence: float = 0.8,
                      min_stable_views: int = 2, min_spacing_ms: int = 500) -> dict[str, Any]:
    """Evaluate tracked mask support without inferring unseen physical causes.

    Dense frames with unassessed visibility remain UNKNOWN but preserve a
    candidate continuity chain. Missing/weak masks, overlap with runtime reviewed
    distractors, an explicit semantic conflict, or a changed track identifier
    invalidate that chain. After a gap, only two or more distinct, adequately
    spaced, source-bound semantic confirmations can restore support.

    Frame timestamps are relative source-media time, not live capture freshness.
    Per-frame reference labels or boxes are intentionally absent from the schema.
    """
    if not isinstance(target_id, str) or not _ID.fullmatch(target_id):
        raise ValidationError("Tracking target must be a concise exact object identifier")
    threshold = _number(min_confidence, "minimum confidence")
    stable = _integer(min_stable_views, "stable view count", minimum=2, maximum=8)
    spacing = _integer(min_spacing_ms, "minimum source spacing", maximum=60_000)
    frames = _frames(frames, target_id)
    seed = _seed(seed, frames[0], target_id)
    reviewed_ids = set(seed["reviewed_distractor_track_ids"]) if seed else set()
    if any(item["track_id"] not in reviewed_ids for frame in frames for item in frame["reviewed_distractors"]):
        raise ValidationError("Runtime distractor tracks must have an authored initial seed review")
    last_track = seed["track_id"] if seed else None
    identity_basis = "authored_seed" if seed else "unconfirmed"
    semantic_required = seed is None
    confirmations: list[tuple[int, int, str]] = []
    confirmation_track = None
    had_gap = False
    previous_hard = False
    metrics = {"frames": len(frames), "supported_frames": 0, "unknown_frames": 0,
               "hard_gap_frames": 0, "hard_gap_events": 0, "reacquisition_events": 0,
               "track_identifier_changes": 0}
    rows = []
    for frame in frames:
        candidates = frame["candidates"]
        gaps, hard, selected = [], False, None
        if len(candidates) == 1:
            selected = candidates[0]
        elif not candidates:
            gaps.append("candidate_missing")
            hard = True
        else:
            supported = [item for item in candidates if item["semantic"] and item["semantic"]["decision"] == "supported"]
            excluded = [item for item in candidates if item["semantic"] and item["semantic"]["decision"] == "contradicted"]
            if len(supported) == 1 and len(excluded) == len(candidates) - 1:
                selected = supported[0]
            else:
                gaps.append("multiple_target_candidates")
                hard = True
        identifier_changed = bool(selected is not None and last_track is not None and selected["track_id"] != last_track)
        if identifier_changed:
            metrics["track_identifier_changes"] += 1
            gaps.append("track_identifier_changed")
            confirmations = []
            confirmation_track = None
            semantic_required, had_gap, identity_basis = True, True, "unconfirmed"
        overlaps = False
        semantic = selected["semantic"] if selected else None
        if selected:
            last_track = selected["track_id"]
            if selected["confidence"] is None:
                gaps.append("score_unreported")
                hard = True
            elif selected["confidence"] < threshold:
                gaps.append("low_confidence")
                hard = True
            if not selected["mask_supported"]:
                gaps.append("mask_unsupported")
                hard = True
            if selected["box"][0] == 0 or selected["box"][1] == 0 or selected["box"][2] == 1 or selected["box"][3] == 1:
                gaps.append("frame_boundary_unconfirmed")
                hard = True
            overlaps = any(_overlap(selected["box"], item["box"]) for item in frame["reviewed_distractors"])
            if overlaps:
                gaps.append("reviewed_distractor_overlap")
                hard = True
            if selected["fully_visible"] is False:
                gaps.append("full_visibility_not_supported")
                hard = True
            if selected["identity_ambiguous"] is True:
                gaps.append("identity_ambiguous")
                hard = True
            if semantic and semantic["decision"] != "supported":
                gaps.append("semantic_identity_" + semantic["decision"])
                hard = True
        status = "unknown"
        lifecycle = "evidence_gap" if hard else "reacquiring" if semantic_required else "tracking_unverified"
        if hard:
            confirmations = []
            confirmation_track = None
            semantic_required, had_gap, identity_basis = True, True, "unconfirmed"
            metrics["hard_gap_frames"] += 1
        elif selected:
            if selected["fully_visible"] is not True:
                gaps.append("visibility_unconfirmed")
            if selected["identity_ambiguous"] is not False:
                gaps.append("identity_assessment_missing")
            semantic_ok = semantic is not None and semantic["decision"] == "supported"
            if semantic_required and not semantic_ok:
                gaps.append("semantic_reacquisition_required")
            eligible = (selected["fully_visible"] is True and selected["identity_ambiguous"] is False
                        and (not semantic_required or semantic_ok))
            if eligible:
                if frame["frame_sha256"] is None:
                    gaps.append("source_pixels_unbound")
                elif confirmation_track is not None and confirmation_track != selected["track_id"]:
                    raise AssertionError("Tracking confirmation chain crossed identifiers")
                else:
                    confirmation_track = selected["track_id"]
                    duplicate = any(item[2] == frame["frame_sha256"] for item in confirmations)
                    close = bool(confirmations and frame["timestamp_ms"] - confirmations[-1][1] < spacing)
                    if duplicate:
                        gaps.append("source_view_reused")
                    elif close:
                        gaps.append("source_spacing_insufficient")
                    else:
                        confirmations.append((frame["frame_index"], frame["timestamp_ms"], frame["frame_sha256"]))
                        confirmations = confirmations[-stable:]
                        if len(confirmations) >= stable:
                            status, lifecycle = "supported", "tracking_supported"
                            if semantic_required and had_gap:
                                metrics["reacquisition_events"] += 1
                            if semantic_required or semantic_ok:
                                identity_basis = "runtime_semantic"
                            semantic_required = False
                        else:
                            gaps.append("distinct_stable_views_required")
                            lifecycle = "reacquiring" if semantic_required else "acquiring"
        if hard and not previous_hard or identifier_changed:
            metrics["hard_gap_events"] += 1
        previous_hard = hard
        metrics[status + "_frames"] += 1
        rows.append({
            "frame_index": frame["frame_index"], "timestamp_ms": frame["timestamp_ms"],
            "source_sha256": frame["source_sha256"], "frame_sha256": frame["frame_sha256"],
            "status": status, "lifecycle": lifecycle, "gap_codes": list(dict.fromkeys(gaps)),
            "track_id": selected["track_id"] if selected else None,
            "box": deepcopy(selected["box"]) if selected else None,
            "reported_confidence": selected["confidence"] if selected else None,
            "semantic_decision": semantic["decision"] if semantic else "not_run",
            "semantic_evidence_sha256": semantic["evidence_sha256"] if semantic else None,
            "confirmation_views": len(confirmations), "identity_basis": identity_basis,
            "semantic_reacquisition_required": semantic_required,
            "track_identifier_changed": identifier_changed, "reviewed_distractor_overlap": overlaps,
            "physical_occlusion_observed": False, "identity_switch_observed": False,
            "physical_action_authorized": False,
        })
    config = {"method_version": TRACKING_POLICY_VERSION, "target_id": target_id, "seed": seed,
              "min_confidence": threshold, "min_stable_views": stable, "min_spacing_ms": spacing}
    policy_hash = hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    return {"schema_version": "1.0", "method_version": TRACKING_POLICY_VERSION,
            "target_id": target_id, "source_sha256": frames[0]["source_sha256"],
            "policy": config, "policy_hash": policy_hash,
            "status": rows[-1]["status"], "frames": rows, "metrics": metrics,
            "changes_live_placement_acceptance": False, "limitations": list(_LIMITATIONS)}
