"""Conservative SAM-image-to-checker bridge; no model calls or absence claims.

Callers supply a human-approved protocol, fixed-camera normalized rectangles,
and explicitly labeled evidence from ``MetaClient.segment_image``.  The bridge
does not infer whether supplied records came from a live API or a test fixture.
Decoded detections can support positive visibility or a count lower bound.
They cannot establish a complete view, a validated absence, or an exact count.
"""

from __future__ import annotations

import base64
import binascii
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
import re
from typing import Any, Iterable, Mapping, Sequence

from proofloop.core import Check, Observation, Protocol, ValidationError


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_MAX_MASK_PIXELS = 16_000_000


@dataclass(frozen=True)
class SamImageEvidence:
    """One concept on one image, with caller-declared provenance preserved."""

    object_id: str
    timestamp_ms: int
    result: Mapping[str, Any]
    evidence_origin: str
    concept: str = ""


@dataclass(frozen=True)
class GroundingResult:
    observations: tuple[Observation, ...]
    evidence: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"observations": [observation.to_dict() for observation in self.observations],
                "evidence": deepcopy(dict(self.evidence))}


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValidationError(f"{label} must be an integer >= {minimum}")
    return value


def _number(value: Any, label: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValidationError(f"{label} must be a finite number")
    return float(value)


def _rectangles(protocol: Protocol, regions: Mapping[str, Sequence[float]]) -> dict[str, tuple[float, ...]]:
    if not isinstance(regions, Mapping):
        raise ValidationError("regions must map IDs to normalized rectangles")
    result: dict[str, tuple[float, ...]] = {}
    for region_id, raw in regions.items():
        if region_id not in protocol.regions or not isinstance(raw, (list, tuple)) or len(raw) != 4:
            raise ValidationError("rectangle must identify a protocol region and contain four coordinates")
        rectangle = tuple(_number(value, "region coordinate") for value in raw)
        x1, y1, x2, y2 = rectangle
        if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
            raise ValidationError("region rectangle must have positive area inside [0,1]")
        result[region_id] = rectangle
    values = list(result.values())
    for index, first in enumerate(values):
        for second in values[index + 1:]:
            if _overlap(first, second):
                raise ValidationError("approved grounding rectangles must not overlap")
    return result


def _overlap(first: Sequence[float], second: Sequence[float]) -> bool:
    return max(first[0], second[0]) < min(first[2], second[2]) and max(first[1], second[1]) < min(first[3], second[3])


def _inside(inner: Sequence[float], outer: Sequence[float]) -> bool:
    return outer[0] <= inner[0] and outer[1] <= inner[1] and inner[2] <= outer[2] and inner[3] <= outer[3]


def _detection(record: Any, regions: Mapping[str, tuple[float, ...]]) -> dict[str, Any]:
    """Validate the adapter's parsed geometry and decoded binary-raster shape."""

    if not isinstance(record, Mapping) or type(record.get("frame")) is not int or record["frame"] != 0:
        raise ValidationError("image detections need frame zero")
    object_id = _integer(record.get("object_id"), "SAM object ID")
    confidence = _number(record.get("confidence"), "reported SAM confidence")
    if not 0 <= confidence <= 1:
        raise ValidationError("reported SAM confidence is outside [0,1]")
    box, mask = record.get("box"), record.get("mask")
    if not isinstance(box, Mapping) or not isinstance(mask, Mapping):
        raise ValidationError("detection needs a box and decoded mask")
    width = _integer(box.get("width"), "source width", minimum=1)
    height = _integer(box.get("height"), "source height", minimum=1)
    x1, y1, x2, y2 = (_integer(box.get(key), key) for key in ("x1", "y1", "x2", "y2"))
    if not (x1 <= x2 < width and y1 <= y2 < height):
        raise ValidationError("detection box is outside its source image")
    normalized_box = (x1 / width, y1 / height, (x2 + 1) / width, (y2 + 1) / height)
    if mask.get("decoded") is not True or mask.get("raster_format") != "base64_row_major_uint8_0_or_1":
        raise ValidationError("decoded official-parser raster is required")
    mask_width = _integer(mask.get("width"), "mask width", minimum=1)
    mask_height = _integer(mask.get("height"), "mask height", minimum=1)
    # SAM masks are box-local rasters. Their origin is the returned box's
    # top-left, not the source image's origin; the official parser exposes
    # those box bounds independently of the raster dimensions. Map pixel-cell
    # edges through the box extent, including when the raster is resampled.
    if mask_width * mask_height > _MAX_MASK_PIXELS:
        raise ValidationError("decoded raster exceeds the grounding pixel limit")
    try:
        raw = mask.get("raster_base64")
        if not isinstance(raw, str) or len(raw) > 4 * ((_MAX_MASK_PIXELS + 2) // 3):
            raise ValidationError("missing or excessive decoded raster")
        raster = base64.b64decode(raw, validate=True)
    except (binascii.Error, ValueError):
        raise ValidationError("decoded raster is not valid base64") from None
    if len(raster) != mask_width * mask_height or any(pixel not in (0, 1) for pixel in raster):
        raise ValidationError("decoded raster is not the claimed binary shape")
    foreground_pixels = raster.count(1)
    if not foreground_pixels:
        raise ValidationError("decoded mask has no foreground")
    if type(mask.get("foreground_pixels")) is not int or mask["foreground_pixels"] != foreground_pixels:
        raise ValidationError("decoded foreground metadata does not match the raster")
    if mask.get("raster_sha256") != hashlib.sha256(raster).hexdigest():
        raise ValidationError("decoded raster digest does not match its metadata")
    min_x, min_y, max_x, max_y = mask_width, mask_height, -1, -1
    for index, pixel in enumerate(raster):
        if pixel:
            pixel_y, pixel_x = divmod(index, mask_width)
            min_x, min_y = min(min_x, pixel_x), min(min_y, pixel_y)
            max_x, max_y = max(max_x, pixel_x), max(max_y, pixel_y)
    box_width, box_height = x2 - x1 + 1, y2 - y1 + 1
    normalized_foreground = (
        (x1 + min_x / mask_width * box_width) / width,
        (y1 + min_y / mask_height * box_height) / height,
        (x1 + (max_x + 1) / mask_width * box_width) / width,
        (y1 + (max_y + 1) / mask_height * box_height) / height,
    )
    containing = [region_id for region_id, rectangle in regions.items()
                  if _inside(normalized_box, rectangle) and _inside(normalized_foreground, rectangle)]
    if len(containing) != 1:
        raise ValidationError("box and foreground must be wholly inside one approved region")
    return {"sam_object_id": object_id, "region_id": containing[0], "confidence": confidence,
            "box": normalized_box, "foreground": normalized_foreground}


def _sample(sample: SamImageEvidence, protocol: Protocol, rectangles: Mapping[str, tuple[float, ...]],
            index: int) -> dict[str, Any]:
    if sample.object_id not in protocol.objects:
        raise ValidationError("sample identifies an unknown protocol object")
    _integer(sample.timestamp_ms, "source timestamp")
    if not isinstance(sample.evidence_origin, str) or not sample.evidence_origin.strip():
        raise ValidationError("caller-declared evidence_origin is required")
    if not isinstance(sample.concept, str) or not isinstance(sample.result, Mapping):
        raise ValidationError("sample needs a concept string and segment_image result")
    source = sample.result.get("source")
    if not isinstance(source, Mapping) or not isinstance(source.get("sha256"), str) or not _SHA256.fullmatch(source["sha256"]):
        raise ValidationError("source metadata needs the exact image SHA-256 digest")
    response = sample.result.get("response", {})
    parsed: dict[str, Any] = {
        "sample_index": index,
        "object_id": sample.object_id,
        "timestamp_ms": sample.timestamp_ms,
        "evidence_origin": sample.evidence_origin,
        "concept": sample.concept,
        "raw_source_metadata": deepcopy(dict(source)),
        "source_sha256": source["sha256"],
        "response_id": response.get("id") if isinstance(response, Mapping) else None,
        "reported_model": response.get("model") if isinstance(response, Mapping) else None,
        "mask_decoding": sample.result.get("mask_decoding"),
        "detections": [],
        "unknown_reason": None,
    }
    objects = sample.result.get("objects")
    if not isinstance(objects, list) or not objects:
        parsed["unknown_reason"] = "Zero or unavailable detections do not prove absence"
        return parsed
    try:
        detections = [_detection(record, rectangles) for record in objects]
        ids = [detection["sam_object_id"] for detection in detections]
        if len(ids) != len(set(ids)):
            raise ValidationError("duplicate SAM instance IDs are ambiguous")
        for pos, first in enumerate(detections):
            for second in detections[pos + 1:]:
                if _overlap(first["box"], second["box"]):
                    raise ValidationError("overlapping same-concept detections cannot establish distinct instances")
        parsed["detections"] = detections
    except (ValidationError, TypeError, KeyError) as error:
        parsed["unknown_reason"] = str(error)
    return parsed


def _check_value(check: Check, sample: Mapping[str, Any]) -> tuple[int | bool | None, float, str | None]:
    if sample["unknown_reason"]:
        return None, 0.0, sample["unknown_reason"]
    detections = sample["detections"]
    if any(detection["confidence"] < check.min_confidence for detection in detections):
        return None, 0.0, "Reported detection confidence is below the approved threshold"
    relevant = [detection for detection in detections if detection["region_id"] == check.region_id]
    if not relevant:
        return None, 0.0, "No supported foreground in this region; absence is unvalidated"
    return (True if check.kind == "visible" else len(relevant)), min(item["confidence"] for item in relevant), None


def ground_sam_images(protocol: Protocol, regions: Mapping[str, Sequence[float]],
                      samples: Iterable[SamImageEvidence], *, session_id: str,
                      attempt_id: int = 0, revision: int = 0) -> GroundingResult:
    """Ground final per-check facts from explicitly labeled SAM image results.

    Repeated source hashes never increase stability or refresh source age.
    The latest consistent run must contain at least two distinct hashes; the
    protocol's larger stability requirement remains binding. Observation time
    is the oldest required supporting image, so its age cannot hide stale
    support. Return values are sorted by timestamp for ``EvidenceStore.add``.
    Invalid/missing perception yields unknown; malformed calibration or source
    identity raises ``ValidationError`` instead of silently accepting it.
    """

    if not isinstance(protocol, Protocol):
        raise ValidationError("a human-approved Protocol instance is required")
    rectangles = _rectangles(protocol, regions)
    raw_samples = list(samples)
    if any(not isinstance(sample, SamImageEvidence) for sample in raw_samples):
        raise ValidationError("samples must be SamImageEvidence instances")
    parsed = [_sample(sample, protocol, rectangles, index) for index, sample in enumerate(raw_samples)]
    parsed.sort(key=lambda item: (item["timestamp_ms"], item["sample_index"]))
    latest_time = max((sample["timestamp_ms"] for sample in parsed), default=0)
    first_seen: dict[tuple[str, str], int] = {}
    for sample in parsed:
        key = (sample["object_id"], sample["source_sha256"])
        first_seen.setdefault(key, sample["timestamp_ms"])
        sample["original_source_timestamp_ms"] = first_seen[key]
    observations: list[Observation] = []
    outcomes: list[dict[str, Any]] = []
    evidence_records: dict[str, Any] = {}
    for check in protocol.checks:
        if check.kind == "before":
            continue  # Image segmentation cannot establish an authored temporal event.
        relevant = [sample for sample in parsed if sample["object_id"] == check.object_id]
        value: int | bool | None = None
        confidence, stable_samples, timestamp = 0.0, 1, latest_time
        reason = "No source images were supplied for this object"
        supporting: list[dict[str, Any]] = []
        if check.region_id not in rectangles:
            reason = "The required region has no approved rectangle calibration"
        elif relevant:
            last = relevant[-1]
            value, confidence, reason = _check_value(check, last)
            if value is not None:
                candidates: dict[str, dict[str, Any]] = {}
                for sample in reversed(relevant):
                    prior_value, _, prior_reason = _check_value(check, sample)
                    if prior_reason or type(prior_value) is not type(value) or prior_value != value:
                        break
                    source_time = sample["original_source_timestamp_ms"]
                    if latest_time - source_time <= check.max_age_ms:
                        candidates[sample["source_sha256"]] = sample
                ordered = sorted(candidates.values(), key=lambda sample: sample["original_source_timestamp_ms"], reverse=True)
                supporting = ordered[:max(2, check.min_stable_samples)]
                if not supporting:
                    value, confidence = None, 0.0
                    reason = "Repeated or stale source images cannot refresh current evidence"
                else:
                    stable_samples = len(supporting)
                    timestamp = min(sample["original_source_timestamp_ms"] for sample in supporting)
                    confidence = min(_check_value(check, sample)[1] for sample in supporting)
                    reason = ("Positive visibility supported by decoded detections" if check.kind == "visible"
                              else "Detected-instance lower bound; exact count and absence remain unvalidated")
                    if stable_samples < max(2, check.min_stable_samples):
                        reason = "Insufficient distinct source hashes for the approved stability requirement"
        if not supporting and relevant:
            supporting = [relevant[-1]]
        manifest = {
            "check_id": check.id,
            "spec_hash": protocol.content_hash,
            "session_id": session_id,
            "attempt_id": attempt_id,
            "revision": revision,
            "source_indices": [sample["sample_index"] for sample in supporting],
            "source_hashes": [sample["source_sha256"] for sample in supporting],
            "response_ids": [sample["response_id"] for sample in supporting],
            "grounded_value": value,
            "grounded_confidence": confidence,
            "stable_samples": stable_samples,
            "timestamp_ms": timestamp,
            "reason": reason,
        }
        digest = hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        evidence_id = f"sam-grounding://{digest}"
        observation = Observation(session_id=session_id, spec_hash=protocol.content_hash,
                                  attempt_id=attempt_id, revision=revision, kind=check.kind,
                                  object_id=check.object_id, region_id=check.region_id, value=value,
                                  evidence_id=evidence_id, timestamp_ms=timestamp,
                                  confidence=confidence, stable_samples=stable_samples,
                                  fully_visible=False, absence_validated=False)
        observations.append(observation)
        evidence_records[evidence_id] = manifest
        outcomes.append({"check_id": check.id, "reason": reason, "evidence_id": evidence_id})
    observations.sort(key=lambda item: (item.timestamp_ms, item.object_id, item.kind, item.region_id))
    return GroundingResult(tuple(observations), {
        "bridge_version": "1.0", "sources": parsed, "observations": evidence_records,
        "check_grounding": outcomes,
        "limits": ["Origins are caller supplied and preserved; no live-inference status is inferred.",
                   "Confidence is a reported model score, not a correctness probability.",
                   "No region is automatically fully visible and no absence is automatically validated.",
                   "Camera calibration and concept-to-object identity are caller approved."],
    })
