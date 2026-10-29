"""Typed, finite evidence acquisition for UNKNOWN checks, without model calls.

The policy inspects source-bound observations, validated locator fields and SAM
geometry. Human-readable checker/grounder reason strings never select a branch.
Requests concern observation only; this module cannot authorize a physical move,
validate absence, change a threshold, or search a recording until a task passes.
"""
from __future__ import annotations

from collections import defaultdict
import math
import re
from typing import Any, Iterable, Mapping, Sequence

from .core import Check, EvidenceStore, Protocol, Status, ValidationError, evaluate
from .grounding import GroundingResult, SamImageEvidence


_HASH = re.compile(r"^[0-9a-f]{64}$")
_MAX_FOLLOW_UPS = 2
_SPACING_MS = 1_000

_MESSAGES = {
    "missing_current_evidence": (
        "There is no supported observation in this task revision.",
        "Capture the entire target and the approved region boundaries in both new views."),
    "stale_evidence": (
        "The supporting observation has expired.",
        "Capture the current arrangement in two new views within the approved freshness window."),
    "not_detected": (
        "A source view has no supported target detection; this does not establish absence.",
        "Show the entire target in focus and keep its approved region boundaries visible in both new views."),
    "multiple_candidates": (
        "More than one candidate matches a declared unique target.",
        "Review the target inventory and identity. If uniqueness cannot be supported, revise the task contract before inspecting again."),
    "identity_ambiguous": (
        "The locator could not support the target's identity.",
        "Capture distinguishing target features in both views. Review the target concept if its identity remains ambiguous."),
    "partial_visibility": (
        "The locator could not support the entire target's visibility.",
        "Frame the entire target, including its edges, with hands clear and the region boundaries visible."),
    "low_confidence": (
        "A reported detection score is below the approved threshold.",
        "Use a focused, well-lit view of the entire target and region boundaries. The approved threshold stays fixed."),
    "boundary_unconfirmed": (
        "The supplied target geometry is not wholly supported inside one approved region.",
        "Review calibration rectangles against the image and capture the target edges and region boundaries clearly. Do not infer a move from this UNKNOWN result."),
    "invalid_detection": (
        "The source detection geometry is unavailable or invalid.",
        "Capture two clear views of the entire target and region boundaries. Unsupported geometry cannot certify placement."),
    "mask_unavailable": (
        "A usable decoded target mask is unavailable.",
        "Capture two clear views of the entire target. Stop if mask support remains unavailable after the follow-up limit."),
    "temporal_inconsistency": (
        "The latest source views support different target states.",
        "Keep the arrangement and camera unchanged between the two new views. Both views must support the same state."),
    "insufficient_distinct_views": (
        "There are too few distinct, consistent source images for the approved stability requirement.",
        "Capture two entirely new views at least one second apart with the arrangement unchanged."),
    "calibration_required": (
        "The required region has no approved image calibration.",
        "Review and approve that region against the current view before another inspection. A missing calibration cannot be repaired by taking more images."),
    "unsupported_observation": (
        "The available source evidence does not support this target observation.",
        "Capture the entire target and region boundaries clearly in both new views. Unsupported evidence does not prove absence."),
    "unsupported_stability_requirement": (
        "This check requires more than the supported two-view inspection can supply.",
        "Review the task or use a separately validated acquisition method that meets its stability requirement."),
    "absence_validation_required": (
        "This predicate requires supported complete visibility or validated absence, which this image bridge does not establish.",
        "Use an acquisition method that independently validates full scene coverage or revise the task with human review. Additional detections alone cannot prove absence or an exact count."),
    "temporal_history_required": (
        "A pair of current images cannot establish the missing ordered event history.",
        "Use a separately recorded, complete event and checkpoint history. Do not infer an earlier event from a current image."),
}


def _nonnegative(value: Any, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValidationError(f"{name} must be a nonnegative integer")
    return value


def _label(value: Any, fallback: str) -> str:
    if value is None:
        return fallback
    if (not isinstance(value, str) or not value.strip() or len(value) > 160
            or any(ord(character) < 32 for character in value)):
        raise ValidationError("Acquisition labels must be concise single-line strings")
    return value


def _evidence(value: GroundingResult | Mapping[str, Any] | None) -> Mapping[str, Any]:
    if isinstance(value, GroundingResult):
        return value.evidence
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValidationError("grounding must be a GroundingResult or saved grounding mapping")
    return value.get("evidence", value)


def _rectangles(regions: Mapping[str, Sequence[float]] | None, protocol: Protocol) -> dict[str, tuple[float, ...]]:
    if regions is None:
        return {}
    if not isinstance(regions, Mapping):
        raise ValidationError("regions must map approved region IDs to rectangles")
    result = {}
    for key, value in regions.items():
        if (key not in protocol.regions or not isinstance(value, (list, tuple)) or len(value) != 4
                or any(type(number) not in (int, float) or not math.isfinite(number) for number in value)):
            raise ValidationError("Acquisition regions require approved normalized rectangles")
        x1, y1, x2, y2 = value
        if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
            raise ValidationError("Acquisition regions require approved normalized rectangles")
        result[key] = tuple(value)
    return result


def _box(record: Mapping[str, Any]) -> tuple[float, ...] | None:
    box = record.get("box")
    if (not isinstance(box, Mapping)
            or any(type(box.get(key)) is not int for key in ("width", "height", "x1", "y1", "x2", "y2"))):
        return None
    width, height = box["width"], box["height"]
    if not (width > 0 and height > 0 and 0 <= box["x1"] <= box["x2"] < width
            and 0 <= box["y1"] <= box["y2"] < height):
        return None
    return (box["x1"] / width, box["y1"] / height,
            (box["x2"] + 1) / width, (box["y2"] + 1) / height)


def _inside(box: Sequence[float], region: Sequence[float]) -> bool:
    return region[0] <= box[0] and region[1] <= box[1] and box[2] <= region[2] and box[3] <= region[3]


def _source_hash(value: Any) -> str | None:
    return value if isinstance(value, str) and _HASH.fullmatch(value) else None


def _sources(grounding: Mapping[str, Any], samples: Sequence[SamImageEvidence], locator: Mapping[str, Any] | None):
    """Internal evidence-only records; no provider payload enters the DTO."""
    records = []
    for source in grounding.get("sources", []):
        if isinstance(source, Mapping):
            records.append(source)
    # Grounding sources retain accepted per-source regions and confidence. Raw
    # samples supply rejected geometry without parsing their explanatory text.
    sample_sources = []
    for sample in samples:
        source = sample.result.get("source", {})
        digest = _source_hash(source.get("sha256")) if isinstance(source, Mapping) else None
        if digest is None:
            raise ValidationError("Acquisition SAM samples require bound source hashes")
        sample_sources.append({"object_id": sample.object_id, "source_sha256": digest,
                               "timestamp_ms": sample.timestamp_ms, "result": sample.result})
    frame_hashes = []
    if isinstance(locator, Mapping):
        for source in locator.get("sources", []):
            if isinstance(source, Mapping) and _source_hash(source.get("sha256")):
                frame_hashes.append(source["sha256"])
    if not frame_hashes:
        candidates = sorted(records + sample_sources, key=lambda item: item.get("timestamp_ms", 0))
        frame_hashes = list(dict.fromkeys(item.get("source_sha256") for item in candidates
                                        if _source_hash(item.get("source_sha256"))))
    return records, sample_sources, frame_hashes[-2:]


def _diagnose(check: Check, observation, sources, raw_sources, locator, rectangles, frame_hashes):
    causes: dict[str, set[int]] = defaultdict(set)

    def add(code, source_hash=None, view_index=None):
        if view_index is not None:
            causes[code].add(view_index)
        elif source_hash in frame_hashes:
            causes[code].add(frame_hashes.index(source_hash))
        else:
            causes[code]  # Preserve a typed cause even when the view is unavailable.

    if observation is None:
        add("missing_current_evidence")
        return causes
    if check.kind == "before":
        add("temporal_history_required")
        return causes
    if check.min_stable_samples > 2:
        add("unsupported_stability_requirement")
    if observation.confidence < check.min_confidence and observation.value is not None:
        add("low_confidence")
    if check.kind in ("count", "visible"):
        needs_absence = check.kind == "count" or observation.value is False
        if needs_absence and not (observation.fully_visible and observation.absence_validated):
            add("absence_validation_required")

    located = locator.get("parsed", {}).get("frames", []) if isinstance(locator, Mapping) else []
    for frame in located:
        if not isinstance(frame, Mapping):
            continue
        view_index = frame.get("frame_index")
        if type(view_index) is not int or view_index not in (0, 1):
            continue
        for target in frame.get("objects", []):
            if not isinstance(target, Mapping) or target.get("object_id") != check.object_id:
                continue
            candidates = target.get("candidates")
            if isinstance(candidates, list):
                if not candidates:
                    add("not_detected", view_index=view_index)
                elif len(candidates) > 1:
                    add("multiple_candidates", view_index=view_index)
            if target.get("identity_ambiguous") is True:
                add("identity_ambiguous", view_index=view_index)
            if target.get("fully_visible") is False:
                add("partial_visibility", view_index=view_index)

    relevant_raw = [source for source in raw_sources if source["object_id"] == check.object_id]
    for source in relevant_raw:
        objects = source["result"].get("objects")
        digest = source["source_sha256"]
        if not isinstance(objects, list):
            add("invalid_detection", digest)
            continue
        if not objects:
            add("not_detected", digest)
            continue
        if check.kind == "location" and len(objects) > 1:
            add("multiple_candidates", digest)
        seen_ids = []
        boxes = []
        for item in objects:
            if not isinstance(item, Mapping):
                add("invalid_detection", digest)
                continue
            seen_ids.append(item.get("object_id"))
            box = _box(item)
            if box is None:
                add("invalid_detection", digest)
            else:
                boxes.append(box)
                if rectangles and sum(_inside(box, rectangle) for rectangle in rectangles.values()) != 1:
                    add("boundary_unconfirmed", digest)
            mask = item.get("mask")
            if not isinstance(mask, Mapping) or mask.get("decoded") is not True:
                add("mask_unavailable", digest)
            confidence = item.get("confidence")
            if type(confidence) in (int, float) and math.isfinite(confidence) and confidence < check.min_confidence:
                add("low_confidence", digest)
        if len(seen_ids) != len(set(str(value) for value in seen_ids)):
            add("multiple_candidates", digest)
        for index, first in enumerate(boxes):
            if any(max(first[0], other[0]) < min(first[2], other[2])
                   and max(first[1], other[1]) < min(first[3], other[3]) for other in boxes[index + 1:]):
                add("multiple_candidates", digest)

    relevant = sorted((source for source in sources if source.get("object_id") == check.object_id),
                      key=lambda item: item.get("timestamp_ms", 0))
    per_source_values = []
    for source in relevant:
        detections = source.get("detections", [])
        if (check.kind == "location" and len(detections) == 1
                and detections[0].get("confidence", 0) >= check.min_confidence):
            per_source_values.append((source.get("source_sha256"), detections[0].get("region_id")))
        elif not detections and not relevant_raw and not located:
            # Empty parsed detections also cover rejected geometry. They cannot
            # specifically establish zero detections without raw SAM evidence.
            add("unsupported_observation", source.get("source_sha256"))
    if observation.stable_samples < check.min_stable_samples:
        distinct_values = []
        seen = set()
        for digest, value in reversed(per_source_values):
            if digest not in seen:
                seen.add(digest)
                distinct_values.append(value)
            if len(distinct_values) == 2:
                break
        if len(distinct_values) == 2 and distinct_values[0] != distinct_values[1]:
            add("temporal_inconsistency")
        elif observation.value is not None:
            add("insufficient_distinct_views")
    if not causes:
        add("unsupported_observation")
    return causes


def build_acquisition_plan(protocol: Protocol, store: EvidenceStore, *,
                           grounding: GroundingResult | Mapping[str, Any] | None = None,
                           locator: Mapping[str, Any] | None = None,
                           samples: Sequence[SamImageEvidence] = (),
                           regions: Mapping[str, Sequence[float]] | None = None,
                           attempts_used: int = 0, max_attempts: int = _MAX_FOLLOW_UPS,
                           used_source_hashes: Iterable[str] = (), source_mode: str = "live",
                           available_timestamps_ms: Sequence[int] | None = None,
                           now_ms: int | None = None,
                           source_pair_expired: bool = False,
                           region_labels: Mapping[str, str] | None = None,
                           object_labels: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Return a public-safe request for UNKNOWN checks with a finite budget.

    ``attempts_used`` counts submitted follow-ups, including API errors. The
    caller persists this count through unresolved UNKNOWN inspections, consumes
    it before dispatch, and resets only for a completed decision or a genuinely
    new task/session. This pure function cannot enforce the caller's lifecycle.

    A recorded source may provide every remaining candidate timestamp. The next
    chronological admissible pair is fixed without looking at model outcomes;
    insufficient later timestamps is terminal, not permission to reuse frames.
    ``source_mode='recorded'`` is timeline analysis, never physical repair proof.
    The current-scene caller can supply ``source_pair_expired=True`` when its
    whole-pair window expired even though the latest observation alone is recent.
    This supplies typed stale-evidence requests without changing source times.
    """
    if not isinstance(protocol, Protocol) or not isinstance(store, EvidenceStore):
        raise ValidationError("Acquisition requires an approved protocol and EvidenceStore")
    attempts_used = _nonnegative(attempts_used, "attempts_used")
    max_attempts = _nonnegative(max_attempts, "max_attempts")
    if max_attempts > _MAX_FOLLOW_UPS:
        raise ValidationError("Acquisition supports at most two follow-up submissions")
    if source_mode not in ("live", "recorded"):
        raise ValidationError("source_mode must be live or recorded")
    if type(source_pair_expired) is not bool or source_pair_expired and source_mode != "live":
        raise ValidationError("Only current live evidence can declare its source pair expired")
    at = store.current_time_ms if now_ms is None else _nonnegative(now_ms, "now_ms")
    verdicts = evaluate(protocol, store, now_ms=at)
    unknown_ids = [result.check_id for result in verdicts if source_pair_expired or result.status == Status.UNKNOWN]
    failed_ids = [result.check_id for result in verdicts if not source_pair_expired and result.status == Status.FAIL]
    evidence = _evidence(grounding)
    if not isinstance(evidence, Mapping):
        raise ValidationError("Saved grounding evidence must be a mapping")
    if not isinstance(samples, (list, tuple)) or any(not isinstance(sample, SamImageEvidence) for sample in samples):
        raise ValidationError("Acquisition samples must be SamImageEvidence values")
    rectangles = _rectangles(regions, protocol)
    sources, raw_sources, frame_hashes = _sources(evidence, samples, locator)
    exclusions = set()
    for digest in used_source_hashes:
        if not _source_hash(digest):
            raise ValidationError("Used source hashes must be exact SHA-256 values")
        exclusions.add(digest)
    for source in sources + raw_sources:
        digest = _source_hash(source.get("source_sha256"))
        if digest:
            exclusions.add(digest)
    exclusions.update(frame_hashes)
    policy = {"required_views": 2, "min_spacing_ms": _SPACING_MS,
              "all_sources_must_be_new": True,
              "after_timestamp_ms": at if source_mode == "recorded" else None,
              "next_timestamps_ms": None, "excluded_source_hashes": sorted(exclusions)}
    next_times = None
    if available_timestamps_ms is not None:
        if source_mode != "recorded" or not isinstance(available_timestamps_ms, (list, tuple)):
            raise ValidationError("Available timestamps describe a recorded source timeline")
        times = sorted(set(_nonnegative(value, "available timestamp") for value in available_timestamps_ms))
        later = [value for value in times if value > at]
        if later:
            second = next((value for value in later[1:] if value - later[0] >= _SPACING_MS), None)
            if second is not None:
                next_times = [later[0], second]
        policy["next_timestamps_ms"] = next_times
    requests = []
    unsupported = not 1 <= len(protocol.objects) <= 6
    for check in protocol.checks:
        if check.id not in unknown_ids:
            continue
        matching = [item for item in store.observations if item.kind == check.kind
                    and item.object_id == check.object_id and item.region_id == check.region_id
                    and item.attempt_id == store.attempt_id and item.revision == store.revision
                    and item.timestamp_ms <= at]
        observation = matching[-1] if matching else None
        if check.kind == "before":
            causes = {"temporal_history_required": set()}
        elif source_pair_expired or observation is not None and at - observation.timestamp_ms > check.max_age_ms:
            causes = {"stale_evidence": set()}
        else:
            causes = _diagnose(check, observation, sources, raw_sources, locator, rectangles, frame_hashes)
        if check.min_stable_samples > 2:
            causes["unsupported_stability_requirement"] = set()
        if check.kind != "before" and regions is not None and check.region_id not in rectangles:
            causes["calibration_required"] = set()
        unsupported = unsupported or bool(set(causes) & {
            "unsupported_stability_requirement", "absence_validation_required", "temporal_history_required",
            "calibration_required"})
        codes = list(causes)
        object_label = _label((object_labels or {}).get(check.object_id), check.object_id or "workflow event")
        region_label = _label((region_labels or {}).get(check.region_id), check.region_id or "required checkpoint")
        requests.append({"check_id": check.id, "object_id": check.object_id, "object_label": object_label,
                         "region_id": check.region_id, "region_label": region_label,
                         "gap_codes": codes,
                         "view_indices": sorted({index for indices in causes.values() for index in indices}),
                         "reason": " ".join(_MESSAGES[code][0] for code in codes),
                         "instructions": list(dict.fromkeys(
                             _MESSAGES[code][1].replace("the entire target", f"the entire {object_label}")
                             .replace("target edges", f"{object_label} edges")
                             .replace("target features", f"{object_label} features")
                             for code in codes))})
    status, stop_reason = "ready", None
    if not unknown_ids:
        status = "not_needed"
    elif unsupported:
        status, stop_reason = "unsupported_contract", "The current two-view acquisition method cannot establish one or more required predicates. Review the task or acquisition method."
    elif attempts_used >= max_attempts:
        status, stop_reason = "budget_exhausted", "The follow-up limit is reached. Keep the result unverified and review the task or evidence method; do not keep sampling until it passes."
    elif source_mode == "recorded" and available_timestamps_ms is not None and next_times is None:
        status, stop_reason = "timeline_exhausted", "This recording has no remaining chronological pair with the required spacing. Stop this timeline analysis; a new live inspection is a separate session."
    object_ids = list(dict.fromkeys(request["object_id"] for request in requests if request["object_id"] is not None))
    instructions = []
    for request in requests:
        for instruction in request["instructions"]:
            text = request["object_label"] + ": " + instruction
            if text not in instructions:
                instructions.append(text)
    if status == "ready":
        instructions.append("Keep all declared task targets and their approved region boundaries visible in both views, focusing on the listed missing evidence.")
        instructions.append("Use two entirely new source images at least one second apart. Keep the arrangement unchanged between views; this request does not authorize moving a target.")
    return {"schema_version": 1, "status": status,
            "title": {"ready": "Capture the specific missing evidence", "not_needed": "No additional evidence requested",
                      "budget_exhausted": "Evidence follow-up limit reached", "timeline_exhausted": "Recorded evidence is exhausted",
                      "unsupported_contract": "Review the evidence method"}[status],
            "reason": status, "instructions": instructions, "object_ids": object_ids,
            "inspection_only": True, "physical_action_allowed": False,
            "attempts_used": attempts_used, "max_attempts": max_attempts,
            "attempts_remaining": max(0, max_attempts - attempts_used),
            "unknown_check_ids": unknown_ids, "observed_fail_check_ids": failed_ids,
            "requests": requests, "source_policy": policy, "stop_reason": stop_reason}


def check_source_pair(plan: Mapping[str, Any], source_hashes: Sequence[str], *,
                      timestamps_ms: Sequence[int] | None = None) -> None:
    """Reject unsupported/reused/out-of-order follow-up pairs before model calls.

    Capture timing and the timestamp-to-byte mapping remain caller declared;
    these checks do not authenticate a camera or prove that motion occurred.
    """
    if not isinstance(plan, Mapping) or plan.get("status") != "ready":
        raise ValidationError("The evidence acquisition plan does not allow another follow-up")
    if (not isinstance(source_hashes, (list, tuple)) or len(source_hashes) != 2
            or any(not _source_hash(value) for value in source_hashes)
            or source_hashes[0] == source_hashes[1]):
        raise ValidationError("A follow-up needs exactly two distinct SHA-256-bound source images")
    policy = plan.get("source_policy", {})
    if not isinstance(policy, Mapping) or policy.get("required_views") != 2 or policy.get("all_sources_must_be_new") is not True:
        raise ValidationError("Acquisition pair policy is invalid")
    if set(source_hashes) & set(policy.get("excluded_source_hashes", [])):
        raise ValidationError("Both follow-up source images must be entirely new; old views cannot refresh evidence")
    after = policy.get("after_timestamp_ms")
    if timestamps_ms is None:
        if after is not None:
            raise ValidationError("Recorded follow-ups require source timestamps")
        return
    if not isinstance(timestamps_ms, (list, tuple)) or len(timestamps_ms) != 2:
        raise ValidationError("A follow-up requires exactly two source timestamps")
    first, second = (_nonnegative(value, "source timestamp") for value in timestamps_ms)
    if second - first < policy.get("min_spacing_ms", _SPACING_MS):
        raise ValidationError("Follow-up views must be at least one second apart in source order")
    if after is not None and first <= after:
        raise ValidationError("Both recorded follow-up views must occur after the previous evidence")
    chosen = policy.get("next_timestamps_ms")
    if chosen is not None and list(timestamps_ms) != chosen:
        raise ValidationError("Use the next chronological evidence pair; do not skip forward until the task passes")
