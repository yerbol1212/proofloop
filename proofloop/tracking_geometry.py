"""Mask geometry for a separately labeled cached tracking development retest.

The original video report is replay-verified first. Foreground pixels come only
from its actual SAM masks, never later annotations. Disjoint masks can remove a
box-only conflict; they prove neither semantic identity nor full visibility.
"""
from __future__ import annotations

import base64
import hashlib
import json
from typing import Any, Mapping

from .core import ValidationError
from .meta import MetaAPIError, decode_sam_masks
from .video_tracking import public_tracking_summary


GEOMETRY_VERSION = "source_mask_conflicts_v1"
MAX_SOURCE_MAPPED_PIXELS = 32_000_000
_LIMITS = [
    "Source-bound runtime mask geometry only; later reference boxes and labels are not inputs.",
    "Disjoint foreground can resolve overlapping bounding boxes but does not prove separate physical identities.",
    "A mask does not independently establish full visibility, physical occlusion, identity switching or successful execution.",
    "Missing, invalid, empty or excessive masks retain any bounding-box conflict; no absence is inferred.",
    "This cached development retest makes no new model calls and does not alter the frozen original policy or live placement acceptance.",
]


def _box(record: Mapping[str, Any], width: int, height: int) -> dict[str, int]:
    box = record.get("box")
    if (type(box) is not dict or any(type(box.get(key)) is not int for key in ("x1", "y1", "x2", "y2", "width", "height"))
            or (box["width"], box["height"]) != (width, height)
            or not 0 <= box["x1"] <= box["x2"] < width or not 0 <= box["y1"] <= box["y2"] < height):
        raise ValidationError("Tracking mask geometry must match its exact source image")
    return dict(box)


def _overlap(left: dict[str, int], right: dict[str, int]) -> bool:
    return max(left["x1"], right["x1"]) <= min(left["x2"], right["x2"]) and max(left["y1"], right["y1"]) <= min(left["y2"], right["y2"])


def _mapped(record: Mapping[str, Any], width: int, height: int) -> dict[str, Any]:
    box = _box(record, width, height)
    decoded = decode_sam_masks([dict(record)])[0]["mask"]
    raw = base64.b64decode(decoded["raster_base64"], validate=True)
    mw, mh = decoded["width"], decoded["height"]
    if (type(mw) is not int or type(mh) is not int or mw < 1 or mh < 1 or len(raw) != mw * mh
            or any(pixel not in (0, 1) for pixel in raw) or not any(raw)
            or hashlib.sha256(raw).hexdigest() != decoded["raster_sha256"]
            or decoded["foreground_pixels"] != sum(raw)):
        raise ValidationError("Decoded tracking foreground failed its exact raster binding")
    bw, bh = box["x2"] - box["x1"] + 1, box["y2"] - box["y1"] + 1
    # Same box-local extent as grounding.py. Nearest source-cell-center sampling
    # is integer-only: floor((source_offset + 0.5) * mask_extent / box_extent).
    pixels = bytes(raw[((2 * y + 1) * mh // (2 * bh)) * mw + (2 * x + 1) * mw // (2 * bw)]
                   for y in range(bh) for x in range(bw))
    if not any(pixels):
        raise ValidationError("Source-resampled tracking mask has no supported foreground")
    binding = json.dumps(box, sort_keys=True, separators=(",", ":")).encode() + b"\n" + pixels
    return {"box": box, "width": bw, "pixels": pixels, "foreground_pixels": sum(pixels),
            "raster_sha256": decoded["raster_sha256"], "mapped_mask_sha256": hashlib.sha256(binding).hexdigest()}


def _intersection(first: Mapping[str, Any], second: Mapping[str, Any]) -> int:
    a, b = first["box"], second["box"]
    return sum(bool(first["pixels"][(y - a["y1"]) * first["width"] + x - a["x1"]]
                    and second["pixels"][(y - b["y1"]) * second["width"] + x - b["x1"]])
               for y in range(max(a["y1"], b["y1"]), min(a["y2"], b["y2"]) + 1)
               for x in range(max(a["x1"], b["x1"]), min(a["x2"], b["x2"]) + 1))


def resolve_distractor_conflicts(video_report: Mapping[str, Any], target_track_id: int = 1) -> dict[str, Any]:
    """Return source-bound frame comparisons after full original-report replay.

    Mask pixels map through inclusive source boxes without rotating or resizing
    the image. Box-local mask resampling is disclosed above. A public mask hash
    alone cannot substitute for the encoded private response and official decode.
    """
    if type(target_track_id) is not int or not 0 <= target_track_id <= 2 ** 31 - 1:
        raise ValidationError("Tracking geometry target ID must be a bounded integer")
    summary = public_tracking_summary(video_report)
    source, prepared = summary["source"], video_report["prepared"]
    if target_track_id not in {seed["track_id"] for seed in summary["seeds"]}:
        raise ValidationError("Tracking geometry target must have an exact reviewed seed")
    records = {(record["frame"], record["object_id"]): record for record in video_report["objects"]}
    if len(records) != len(video_report["objects"]):
        raise ValidationError("Tracking geometry requires unique exact frame and track records")
    frames, mapped_pixels = [], 0
    for frame in summary["frames"]:
        index = frame["frame_index"]
        candidates = {candidate["track_id"]: candidate for candidate in frame["candidates"]}
        boxes, masks = {}, {}
        for track_id in candidates:
            record = records.get((index, track_id))
            try:
                if record is None:
                    raise ValidationError("Tracking record is unavailable")
                box = _box(record, source["width"], source["height"])
                boxes[track_id] = box
                area = (box["x2"] - box["x1"] + 1) * (box["y2"] - box["y1"] + 1)
                if mapped_pixels + area > MAX_SOURCE_MAPPED_PIXELS:
                    raise ValidationError("Tracking source-mask mapping budget exceeded")
                mapped_pixels += area
                masks[track_id] = _mapped(record, source["width"], source["height"])
            except (MetaAPIError, ValidationError, ValueError, TypeError, KeyError, OverflowError):
                masks[track_id] = None
        comparisons, conflicts = [], []
        for track_id in sorted(set(candidates) - {target_track_id}):
            a, b = masks.get(target_track_id), masks.get(track_id)
            overlap = _overlap(boxes[target_track_id], boxes[track_id]) if target_track_id in boxes and track_id in boxes else None
            intersection = _intersection(a, b) if a is not None and b is not None else None
            relation = "unavailable" if intersection is None else "overlap" if intersection else "disjoint"
            if overlap is not False and relation != "disjoint":
                conflicts.append(track_id)
            union = a["foreground_pixels"] + b["foreground_pixels"] - intersection if intersection is not None else None
            comparisons.append({"distractor_track_id": track_id, "bbox_overlap": overlap,
                "mask_relation": relation, "intersection_pixels": intersection,
                "target_foreground_pixels": a["foreground_pixels"] if a else None,
                "distractor_foreground_pixels": b["foreground_pixels"] if b else None,
                "mask_iou": round(intersection / union, 6) if union else None,
                "target_raster_sha256": a["raster_sha256"] if a else None,
                "distractor_raster_sha256": b["raster_sha256"] if b else None,
                "target_mapped_mask_sha256": a["mapped_mask_sha256"] if a else None,
                "distractor_mapped_mask_sha256": b["mapped_mask_sha256"] if b else None})
        frames.append({"frame_index": index, "source_sha256": source["sha256"],
            "target_track_id": target_track_id, "target_available": target_track_id in candidates,
            "comparisons": comparisons, "conflicting_track_ids": conflicts,
            "physical_action_authorized": False})
    return {"method_version": GEOMETRY_VERSION, "source_sha256": source["sha256"],
        "prepared_hash": prepared["prepared_hash"], "status": summary["status"], "frames": frames,
        "mapping": "box_local_mask_nearest_source_pixel_center_inclusive_xyxy",
        "mapped_source_pixels": mapped_pixels, "max_mapped_source_pixels": MAX_SOURCE_MAPPED_PIXELS,
        "identity_independently_verified": False, "physical_action_authorized": False,
        "changes_live_placement_acceptance": False, "limitations": list(_LIMITS)}
