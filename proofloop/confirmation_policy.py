"""Experimental confirmation chains that end at the current source view.

This is a separate development policy; the frozen v1 evaluator is unchanged.
It uses v1's closed input validation and hard gates, then removes an accidental
dependence on the most recently sampled confirmation. A dense clear sample
cannot displace an earlier, adequately spaced source view.

Every counted view needs its own exact bound, positive runtime semantics. An
authored seed reviews identities and distractor identifiers; it does not replace
current visibility evidence. The earliest eligible occurrence of a JPEG is
canonical within a continuity segment. Later occurrences remain UNKNOWN and
cannot contribute fresh confirmation evidence.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
import hashlib
import json
from typing import Any

from .core import ValidationError
from . import tracking_policy as legacy


CONFIRMATION_POLICY_VERSION = "current_view_confirmation_v2"
TRACKING_POLICY_VERSION = CONFIRMATION_POLICY_VERSION


def _evidence(frame: Mapping[str, Any], selected: Mapping[str, Any]) -> dict[str, Any]:
    return {"frame_index": frame["frame_index"], "timestamp_ms": frame["timestamp_ms"],
            "source_sha256": frame["source_sha256"], "frame_sha256": frame["frame_sha256"],
            "track_id": selected["track_id"],
            "semantic_evidence_sha256": selected["semantic"]["evidence_sha256"]}


def evaluate_tracking(frames: Sequence[Mapping[str, Any]], *, target_id: str,
                      seed: Mapping[str, Any] | None = None, min_confidence: float = 0.8,
                      min_stable_views: int = 2, min_spacing_ms: int = 500) -> dict[str, Any]:
    """Keep all v1 hard gates and require a spaced chain ending at this view.

    A bounded dynamic program stores the earliest-ending chain at each length
    up to ``min_stable_views``. All eligible JPEG hashes in a segment are unique,
    so an earlier-ending chain is always at least as useful as a later one for
    extending to a future, unique view. Work is O(frames * stable_views), with
    at most 4096 canonical hashes and eight chains of at most eight records.

    The confidence and spacing argument validation matches v1; experiments must
    freeze their values. Missing assessments remain UNKNOWN without a reset.
    Explicit occlusion, identity conflicts and every other v1 hard gate reset
    all confirmation history. A changed identifier resets history, but its
    otherwise valid semantic view can start the new chain.
    """
    if not isinstance(target_id, str) or not legacy._ID.fullmatch(target_id):
        raise ValidationError("Tracking target must be a concise exact object identifier")
    threshold = legacy._number(min_confidence, "minimum confidence")
    stable = legacy._integer(min_stable_views, "stable view count", minimum=2, maximum=8)
    spacing = legacy._integer(min_spacing_ms, "minimum source spacing", maximum=60_000)
    normalized = legacy._frames(frames, target_id)
    seed = legacy._seed(seed, normalized[0], target_id)
    reviewed_ids = set(seed["reviewed_distractor_track_ids"]) if seed else set()
    if any(item["track_id"] not in reviewed_ids
           for frame in normalized for item in frame["reviewed_distractors"]):
        raise ValidationError("Runtime distractor tracks must have an authored initial seed review")

    last_track = seed["track_id"] if seed else None
    identity_basis = "authored_seed" if seed else "unconfirmed"
    semantic_required = seed is None
    had_gap = previous_hard = False
    chains: dict[int, list[dict[str, Any]]] = {}
    seen_hashes: set[str] = set()
    metrics = {"frames": len(normalized), "supported_frames": 0, "unknown_frames": 0,
               "hard_gap_frames": 0, "hard_gap_events": 0, "reacquisition_events": 0,
               "track_identifier_changes": 0}
    rows = []
    for frame in normalized:
        candidates = frame["candidates"]
        gaps, hard, selected = [], False, None
        if len(candidates) == 1:
            selected = candidates[0]
        elif not candidates:
            gaps.append("candidate_missing")
            hard = True
        else:
            supported = [item for item in candidates if item["semantic"]
                         and item["semantic"]["decision"] == "supported"]
            excluded = [item for item in candidates if item["semantic"]
                        and item["semantic"]["decision"] == "contradicted"]
            if len(supported) == 1 and len(excluded) == len(candidates) - 1:
                selected = supported[0]
            else:
                gaps.append("multiple_target_candidates")
                hard = True

        changed = bool(selected is not None and last_track is not None
                       and selected["track_id"] != last_track)
        if changed:
            metrics["track_identifier_changes"] += 1
            gaps.append("track_identifier_changed")
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
            if (selected["box"][0] == 0 or selected["box"][1] == 0
                    or selected["box"][2] == 1 or selected["box"][3] == 1):
                gaps.append("frame_boundary_unconfirmed")
                hard = True
            overlaps = any(legacy._overlap(selected["box"], item["box"])
                           for item in frame["reviewed_distractors"])
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

        if hard or changed:
            chains.clear()
            seen_hashes.clear()
            semantic_required, had_gap, identity_basis = True, True, "unconfirmed"
        status = "unknown"
        lifecycle = "evidence_gap" if hard else "reacquiring" if semantic_required else "tracking_unverified"
        confirmation = chains[max(chains)] if chains else []
        current_included = False
        if hard:
            metrics["hard_gap_frames"] += 1
        elif selected:
            if selected["fully_visible"] is not True:
                gaps.append("visibility_unconfirmed")
            if selected["identity_ambiguous"] is not False:
                gaps.append("identity_assessment_missing")
            semantic_ok = semantic is not None and semantic["decision"] == "supported"
            if not semantic_ok:
                gaps.append("current_semantic_evidence_required")
                if semantic_required:
                    gaps.append("semantic_reacquisition_required")
            eligible = (selected["fully_visible"] is True and selected["identity_ambiguous"] is False
                        and semantic_ok)
            if eligible:
                digest = frame["frame_sha256"]
                if digest is None:
                    # Positive semantics already reject this via legacy validation;
                    # keep the source binding gate explicit in this evaluator.
                    gaps.append("source_pixels_unbound")
                elif digest in seen_hashes:
                    gaps.append("source_view_reused")
                else:
                    current = _evidence(frame, selected)
                    prior = []
                    for length in range(stable - 1, 0, -1):
                        chain = chains.get(length)
                        if chain and frame["timestamp_ms"] - chain[-1]["timestamp_ms"] >= spacing:
                            prior = chain
                            break
                    confirmation = prior + [current]
                    current_included = True
                    seen_hashes.add(digest)
                    # Frames arrive in source order. The first chain at a length
                    # is its earliest-ending representative and never displaced.
                    chains.setdefault(len(confirmation), confirmation)
                    if len(confirmation) >= stable:
                        status, lifecycle = "supported", "tracking_supported"
                        if semantic_required and had_gap:
                            metrics["reacquisition_events"] += 1
                        identity_basis, semantic_required = "runtime_semantic", False
                    else:
                        if len(confirmation) == 1 and len(seen_hashes) > 1:
                            gaps.append("source_spacing_insufficient")
                        gaps.append("distinct_stable_views_required")
                        lifecycle = "reacquiring" if semantic_required else "acquiring"

        if (hard and not previous_hard) or changed:
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
            "confirmation_views": len(confirmation), "identity_basis": identity_basis,
            "semantic_reacquisition_required": semantic_required,
            "track_identifier_changed": changed, "reviewed_distractor_overlap": overlaps,
            "physical_occlusion_observed": False, "identity_switch_observed": False,
            "physical_action_authorized": False,
            "confirmation_evidence": deepcopy(confirmation),
            "confirmation_ends_at_current": current_included,
            "confirmation_history_size": len(seen_hashes),
            "confirmation_history_reset": hard or changed,
        })

    config = {"method_version": CONFIRMATION_POLICY_VERSION, "target_id": target_id, "seed": seed,
              "min_confidence": threshold, "min_stable_views": stable, "min_spacing_ms": spacing,
              "current_semantics_required": True, "all_confirmation_semantics_required": True,
              "duplicate_policy": "canonical_earliest_eligible_jpeg_per_continuity_segment",
              "chain_selection": "earliest_end_dynamic_program_ending_at_current"}
    digest = hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    limitations = list(legacy._LIMITATIONS) + [
        "Every confirmation, including the current view, requires positive source-bound runtime semantics; authored seeds do not replace current evidence.",
        "Repeated JPEGs remain UNKNOWN and never count as fresh views within a continuity segment; clear unchanged scenes can therefore be conservatively withheld.",
        "Support always ends at the current qualified source view. Old support never authorizes an unassessed, ambiguous or occluded current frame.",
    ]
    return {"schema_version": "1.0", "method_version": CONFIRMATION_POLICY_VERSION,
            "target_id": target_id, "source_sha256": normalized[0]["source_sha256"],
            "policy": config, "policy_hash": digest,
            "status": rows[-1]["status"], "frames": rows, "metrics": metrics,
            "changes_live_placement_acceptance": False, "limitations": limitations}
