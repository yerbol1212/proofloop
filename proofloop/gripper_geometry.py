"""Source-bound foreground separation from independently prompted gripper tracks.

The fixed three-source-pixel Chebyshev margin was selected before new gripper
masks. A close or unavailable mask comparison withholds support, without
claiming physical contact or occlusion. Separation can only supplement an
existing clear semantic assessment; it never establishes identity/visibility.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
import time
from typing import Any, Mapping

from .core import ValidationError
from .meta import MetaAPIError
from . import tracking_geometry as geometry
from .video_tracking import public_tracking_summary

VERSION = "source_gripper_foreground_separation_v1"
MAX_SOURCE_MAPPED_PIXELS = 32_000_000
MAX_GEOMETRY_SECONDS = 30.0
CONFIG = {"source_pixel_margin": 3, "distance_metric": "chebyshev_foreground_pixel_centers",
    "distance_cap": 4, "minimum_confidence": 0.8, "requires_all_reviewed_grippers": True,
    "mapping": "box_local_mask_nearest_source_pixel_center_inclusive_xyxy",
    "max_source_mapped_pixels": MAX_SOURCE_MAPPED_PIXELS, "max_geometry_seconds": MAX_GEOMETRY_SECONDS}


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _records(report):
    result = {(record["frame"], record["object_id"]): record for record in report["objects"]}
    if len(result) != len(report["objects"]):
        raise ValidationError("Gripper geometry requires unique exact frame/track records")
    return result


def _rows(mapped, deadline):
    box, width, pixels = mapped["box"], mapped["width"], mapped["pixels"]
    result = {}
    for y in range(box["y1"], box["y2"] + 1):
        if time.monotonic() > deadline:
            raise ValueError("geometry_time_limit")
        offset = (y - box["y1"]) * width
        bits = 0
        for x in range(width):
            if pixels[offset + x]:
                bits |= 1 << (box["x1"] + x)
        if bits:
            result[y] = bits
    return result


def _distance_capped(first, second, deadline):
    """Exact min Chebyshev distance for0..3;4means strictly greater than3."""
    for radius in range(CONFIG["source_pixel_margin"] + 1):
        for y, gripper_bits in second.items():
            if time.monotonic() > deadline:
                raise ValueError("geometry_time_limit")
            target_bits = 0
            for yy in range(y - radius, y + radius + 1):
                target_bits |= first.get(yy, 0)
            dilated = target_bits
            for shift in range(1, radius + 1):
                dilated |= (target_bits << shift) | (target_bits >> shift)
            if dilated & gripper_bits:
                return radius
    return CONFIG["distance_cap"]


def _confidence(candidate):
    value = candidate.get("confidence") if candidate else None
    return value if type(value) in {int, float} and math.isfinite(value) and 0 <= value <= 1 else None


def compare(target_report: Mapping[str, Any], gripper_report: Mapping[str, Any], *,
            target_track_id: int = 1, gripper_track_ids=(3, 4)) -> dict[str, Any]:
    if (type(target_track_id) is not int or not 0 <= target_track_id <= 2 ** 31 - 1
            or not isinstance(gripper_track_ids, (list, tuple)) or not 1 <= len(gripper_track_ids) <= 6
            or any(type(i) is not int or not 0 <= i <= 2 ** 31 - 1 for i in gripper_track_ids)
            or len(set(gripper_track_ids)) != len(gripper_track_ids) or target_track_id in gripper_track_ids):
        raise ValidationError("Review distinct bounded target and gripper track identifiers")
    target, grippers = public_tracking_summary(target_report), public_tracking_summary(gripper_report)
    source = target_report["prepared"]["source"]
    if (source != gripper_report["prepared"]["source"] or target["source"] != grippers["source"]
            or target_report["prepared"]["requested_frame_indices"] != gripper_report["prepared"]["requested_frame_indices"]):
        raise ValidationError("Gripper comparison must use the same exact clip, dimensions, source frame map and PTS")
    if (target_track_id not in {s["track_id"] for s in target["seeds"]}
            or not set(gripper_track_ids).issubset({s["track_id"] for s in grippers["seeds"]})):
        raise ValidationError("Target and every gripper ID require their own reviewed source-bound seed")
    target_records, gripper_records = _records(target_report), _records(gripper_report)
    target_frames = {f["frame_index"]: f for f in target["frames"]}
    gripper_frames = {f["frame_index"]: f for f in grippers["frames"]}
    if len(target_frames) != len(target["frames"]) or len(gripper_frames) != len(grippers["frames"]):
        raise ValidationError("Gripper comparison source frame identifiers must be unique")
    deadline = time.monotonic() + MAX_GEOMETRY_SECONDS
    mapped_pixels, rows = 0, []
    def resolve(frame, record_map, index, track_id, completed):
        nonlocal mapped_pixels
        candidates = [c for c in (frame or {}).get("candidates", []) if c["track_id"] == track_id]
        if len(candidates) > 1:
            raise ValidationError("Gripper comparison candidate identifiers must be unique")
        candidate = candidates[0] if candidates else None
        confidence = _confidence(candidate)
        result = {"confidence": confidence, "mapped": None, "rows": None, "reason": None}
        if not completed:
            result["reason"] = "provider_operation_unscored"
        elif candidate is None:
            result["reason"] = "track_missing_not_verified_absent"
        elif confidence is None or confidence < CONFIG["minimum_confidence"]:
            result["reason"] = "track_confidence_unreported_or_below_threshold"
        elif candidate.get("mask_supported") is not True:
            result["reason"] = "track_foreground_unconfirmed"
        else:
            record = record_map.get((index, track_id))
            try:
                if record is None:
                    raise ValueError("track_mask_record_missing")
                box = geometry._box(record, source["width"], source["height"])
                area = (box["x2"] - box["x1"] + 1) * (box["y2"] - box["y1"] + 1)
                if mapped_pixels + area > MAX_SOURCE_MAPPED_PIXELS:
                    raise ValueError("geometry_pixel_limit")
                if time.monotonic() > deadline:
                    raise ValueError("geometry_time_limit")
                mapped_pixels += area
                mapped = geometry._mapped(record, source["width"], source["height"])
                result.update(mapped=mapped, rows=_rows(mapped, deadline))
            except (MetaAPIError, ValidationError, ValueError, TypeError, KeyError, OverflowError):
                result.update(mapped=None, rows=None, reason="mask_unavailable_or_geometry_resource_limit")
        return result
    for source_frame in source["frames"]:
        index = source_frame["frame_index"]
        a = resolve(target_frames.get(index), target_records, index, target_track_id, target["status"] == "completed")
        comparisons = []
        for identifier in gripper_track_ids:
            b = resolve(gripper_frames.get(index), gripper_records, index, identifier, grippers["status"] == "completed")
            distance, relation, reasons = None, "unavailable", []
            if a["reason"]:
                reasons.append("target_" + a["reason"])
            if b["reason"]:
                reasons.append("gripper_" + b["reason"])
            if a["rows"] is not None and b["rows"] is not None:
                try:
                    distance = _distance_capped(a["rows"], b["rows"], deadline)
                    relation = "separated" if distance > CONFIG["source_pixel_margin"] else "near_or_contact"
                except ValueError:
                    reasons.append("geometry_time_limit")
            hashes = {}
            for role, item in (("target", a), ("gripper", b)):
                for field in ("raster_sha256", "mapped_mask_sha256", "foreground_pixels"):
                    hashes[role + "_" + field] = item["mapped"][field] if item["mapped"] else None
            comparisons.append({"gripper_track_id": identifier, "relation": relation,
                "target_confidence": a["confidence"], "gripper_confidence": b["confidence"],
                "min_chebyshev_distance_capped4": distance,
                "distance_is_lower_bound": distance == CONFIG["distance_cap"],
                "reason_codes": reasons, **hashes})
        relation = ("separated" if all(c["relation"] == "separated" for c in comparisons) else "near_or_contact"
                    if any(c["relation"] == "near_or_contact" for c in comparisons) else "unavailable")
        rows.append({"frame_index": index, "pts": source_frame["pts"], "timestamp_ms": source_frame["timestamp_ms"],
            "source_sha256": source["sha256"], "target_track_id": target_track_id, "relation": relation,
            "separation_supported": relation == "separated", "comparisons": comparisons,
            "physical_contact_observed": False, "physical_occlusion_observed": False, "physical_action_authorized": False})
    return {"method_version": VERSION, "config": deepcopy(CONFIG), "source_sha256": source["sha256"],
        "source_frame_map_sha256": _digest(source["frames"]), "width": source["width"], "height": source["height"],
        "time_base": source["time_base"], "target_prepared_hash": target_report["prepared"]["prepared_hash"],
        "gripper_prepared_hash": gripper_report["prepared"]["prepared_hash"],
        "target_report_sha256": _digest(target_report), "gripper_report_sha256": _digest(gripper_report),
        "target_track_id": target_track_id, "gripper_track_ids": list(gripper_track_ids), "frames": rows,
        "mapped_source_pixels": mapped_pixels, "separated_frames": sum(f["separation_supported"] for f in rows),
        "identity_independently_verified": False, "full_visibility_independently_verified": False,
        "physical_action_authorized": False, "changes_live_placement_acceptance": False,
        "limitations": [
            "Three source pixels is a fixed experimental margin, not a calibrated probability or physical distance.",
            "Near foreground does not prove contact or occlusion; separation cannot prove visibility or identity.",
            "Masks can omit hidden geometry or follow an incorrect object. Independent prompts still use the same SAM model.",
            "Missing, low-confidence, empty or invalid gripper masks mean unavailable, never verified absence.",
            "Only separation from every reviewed gripper can supplement an existing clear semantic assessment.",
            "Distance4 is a lower bound meaning strictly beyond3 source pixels, not an exact larger distance.",
        ]}
