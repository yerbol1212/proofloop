"""Additional negative-only warnings from source-bound gripper proposal masks.

This exposed-development adapter does not qualify clear visibility. Low-score
but valid reviewed gripper proposals may raise an additional near-mask warning;
far, missing, weak or unavailable proposals never authorize baseline support.
The strict frozen geometry module and its confidence threshold are unchanged.
"""
from __future__ import annotations

from copy import deepcopy
import time
from typing import Any, Mapping

from .core import ValidationError
from .meta import MetaAPIError
from . import gripper_geometry as strict
from . import tracking_geometry as geometry
from .video_tracking import public_tracking_summary

VERSION = "negative_only_reviewed_gripper_proposal_veto_v1"
MAX_SOURCE_MAPPED_PIXELS = 32_000_000
MAX_GEOMETRY_SECONDS = 30.0
CONFIG = {"source_pixel_margin": 3, "distance_metric": "chebyshev_foreground_pixel_centers", "distance_cap": 4,
    "confidence_usage": "reported_only_for_negative_proposal_risk_never_positive_qualification",
    "acceptance_confidence_threshold_changed": False, "positive_support_can_be_granted": False,
    "mapping": "box_local_mask_nearest_source_pixel_center_inclusive_xyxy",
    "max_source_mapped_pixels": MAX_SOURCE_MAPPED_PIXELS, "max_geometry_seconds": MAX_GEOMETRY_SECONDS,
    "evaluation_role": "separate_exposed_cached_development_negative_only_adapter"}


def compare(target_report: Mapping[str, Any], gripper_report: Mapping[str, Any], *,
            target_track_id: int = 1, gripper_track_ids=(3, 4)) -> dict[str, Any]:
    if (type(target_track_id) is not int or not 0 <= target_track_id <= 2 ** 31 - 1
            or not isinstance(gripper_track_ids, (list, tuple)) or not 1 <= len(gripper_track_ids) <= 6
            or any(type(i) is not int or not 0 <= i <= 2 ** 31 - 1 for i in gripper_track_ids)
            or len(set(gripper_track_ids)) != len(gripper_track_ids) or target_track_id in gripper_track_ids):
        raise ValidationError("Review distinct bounded target and gripper proposal identifiers")
    target, grippers = public_tracking_summary(target_report), public_tracking_summary(gripper_report)
    source = target_report["prepared"]["source"]
    if (source != gripper_report["prepared"]["source"] or target["source"] != grippers["source"]
            or target_report["prepared"]["requested_frame_indices"] != gripper_report["prepared"]["requested_frame_indices"]):
        raise ValidationError("Gripper warnings require the exact same clip, dimensions, source frame map and PTS")
    if (target_track_id not in {s["track_id"] for s in target["seeds"]}
            or not set(gripper_track_ids).issubset({s["track_id"] for s in grippers["seeds"]})):
        raise ValidationError("Every warning proposal needs its reviewed source-bound seed")
    target_records, gripper_records = strict._records(target_report), strict._records(gripper_report)
    target_frames = {f["frame_index"]: f for f in target["frames"]}
    gripper_frames = {f["frame_index"]: f for f in grippers["frames"]}
    if len(target_frames) != len(target["frames"]) or len(gripper_frames) != len(grippers["frames"]):
        raise ValidationError("Warning source frame identifiers must be unique")
    deadline = time.monotonic() + MAX_GEOMETRY_SECONDS
    mapped_pixels, frames = 0, []

    def resolve(frame, record_map, index, identifier, completed):
        nonlocal mapped_pixels
        candidates = [c for c in (frame or {}).get("candidates", []) if c["track_id"] == identifier]
        if len(candidates) > 1:
            raise ValidationError("Warning candidate identifiers must be unique")
        candidate = candidates[0] if candidates else None
        result = {"confidence": strict._confidence(candidate), "mapped": None, "rows": None, "reason": None}
        if not completed:
            result["reason"] = "provider_operation_unscored"
        elif candidate is None:
            result["reason"] = "proposal_missing_not_verified_absent"
        elif candidate.get("mask_supported") is not True:
            result["reason"] = "proposal_foreground_unconfirmed"
        else:
            record = record_map.get((index, identifier))
            try:
                if record is None:
                    raise ValueError("proposal_mask_record_missing")
                box = geometry._box(record, source["width"], source["height"])
                area = (box["x2"] - box["x1"] + 1) * (box["y2"] - box["y1"] + 1)
                if mapped_pixels + area > MAX_SOURCE_MAPPED_PIXELS or time.monotonic() > deadline:
                    raise ValueError("geometry_resource_limit")
                mapped_pixels += area
                mapped = geometry._mapped(record, source["width"], source["height"])
                result.update(mapped=mapped, rows=strict._rows(mapped, deadline))
            except (MetaAPIError, ValidationError, ValueError, TypeError, KeyError, OverflowError):
                result.update(mapped=None, rows=None, reason="mask_unavailable_or_geometry_resource_limit")
        return result

    for source_frame in source["frames"]:
        index = source_frame["frame_index"]
        a = resolve(target_frames.get(index), target_records, index, target_track_id, target["status"] == "completed")
        comparisons = []
        for identifier in gripper_track_ids:
            b = resolve(gripper_frames.get(index), gripper_records, index, identifier, grippers["status"] == "completed")
            distance, warning = None, "geometry_unavailable"
            reasons = [role + "_" + item["reason"] for role, item in (("target", a), ("gripper", b)) if item["reason"]]
            if a["rows"] is not None and b["rows"] is not None:
                try:
                    distance = strict._distance_capped(a["rows"], b["rows"], deadline)
                    warning = "near_gripper_proposal" if distance <= CONFIG["source_pixel_margin"] else "no_near_proposal_observed"
                except ValueError:
                    reasons.append("geometry_time_limit")
            hashes = {}
            for role, item in (("target", a), ("gripper", b)):
                for field in ("raster_sha256", "mapped_mask_sha256", "foreground_pixels"):
                    hashes[role + "_" + field] = item["mapped"][field] if item["mapped"] else None
            comparisons.append({"gripper_track_id": identifier, "warning": warning,
                "near_proposal": warning == "near_gripper_proposal", "mask_comparison_available": distance is not None,
                "target_confidence": a["confidence"], "gripper_confidence": b["confidence"],
                "min_chebyshev_distance_capped4": distance, "distance_is_lower_bound": distance == CONFIG["distance_cap"],
                "reason_codes": reasons, **hashes})
        veto = any(c["near_proposal"] for c in comparisons)
        complete = all(c["mask_comparison_available"] for c in comparisons)
        warning = "near_gripper_proposal" if veto else "no_near_proposal_observed" if complete else "geometry_unavailable"
        frames.append({"frame_index": index, "pts": source_frame["pts"], "timestamp_ms": source_frame["timestamp_ms"],
            "source_sha256": source["sha256"], "target_track_id": target_track_id, "warning": warning, "veto": veto,
            "additional_geometry_complete": complete, "comparisons": comparisons,
            "positive_support_can_be_granted": False, "separation_verified": False, "absence_verified": False,
            "full_visibility_verified": False, "physical_contact_observed": False, "physical_occlusion_observed": False,
            "physical_action_authorized": False})
    return {"method_version": VERSION, "config": deepcopy(CONFIG), "source_sha256": source["sha256"],
        "source_frame_map_sha256": strict._digest(source["frames"]), "width": source["width"], "height": source["height"],
        "time_base": source["time_base"], "target_prepared_hash": target_report["prepared"]["prepared_hash"],
        "gripper_prepared_hash": gripper_report["prepared"]["prepared_hash"],
        "target_report_sha256": strict._digest(target_report), "gripper_report_sha256": strict._digest(gripper_report),
        "target_track_id": target_track_id, "gripper_track_ids": list(gripper_track_ids), "frames": frames,
        "mapped_source_pixels": mapped_pixels, "veto_frames": sum(f["veto"] for f in frames),
        "positive_support_can_be_granted": False, "acceptance_confidence_threshold_changed": False,
        "identity_independently_verified": False, "full_visibility_independently_verified": False,
        "physical_action_authorized": False, "changes_live_placement_acceptance": False,
        "limitations": [
            "This is a separately labeled exposed cached development adapter, not a new held-out validation.",
            "Valid reviewed proposal masks can only add a negative warning, including when their confidence is weak.",
            "A near mask can be wrong and over-withhold; the warning does not establish physical contact or occlusion.",
            "Far, missing, weak or unavailable proposals never verify absence, separation, full visibility or identity.",
            "Existing baseline confidence, visibility, source binding and UNKNOWN/unscored outcomes remain required.",
            "Final support must be a subset of the corresponding unchanged baseline's supported frame/box decisions.",
            "Distance4 is a lower bound meaning strictly beyond3 source pixels, not an exact larger distance.",
        ]}
