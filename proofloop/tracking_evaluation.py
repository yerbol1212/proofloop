"""Deterministic scoring of frozen temporal annotations, with no inference or I/O.

UNKNOWN is an authored insufficiency-of-evidence label. Unscored is a missing
method result and never earns UNKNOWN agreement. Bounding-box overlap measures
localization against annotations; neither a track ID nor IoU proves identity.
"""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import math
import re
from typing import Any

from .core import ValidationError


SCHEMA_VERSION = "1.0"
METRIC_VERSION = "tracking-frame-score-v1"
_CLIP = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:\-]{0,127}\Z")
_REFERENCE_FIELDS = {"clip_id", "frame_index", "expected", "target_box", "distractor_boxes", "condition"}
_PREDICTION_FIELDS = {"clip_id", "frame_index", "status", "box"}
_CONDITIONS = {"clear", "partial_occlusion", "reexposed", "hidden_or_unidentifiable"}
_LIMITS = [
    "Descriptive scores on frozen selected frames, not a population reliability estimate.",
    "Frames from a clip are correlated; separate clip IDs do not prove independent physical recordings.",
    "UNKNOWN labels require withheld support. Unscored results are missing completions, not abstentions.",
    "Coverage uses all labeled frames. Selective accuracy uses only supported predictions; zero denominators are null.",
    "Correct support requires a positive label and target-box IoU at the frozen threshold; overlap does not independently prove identity.",
    "Distractor alignment is only maximum box overlap with an authored distractor, not proof of an identity switch.",
    "Selected-frame evidence does not establish visibility or motion between sampled frames and grants no physical-action authority.",
]


def _fraction(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 6) if denominator else None


def _box(value: Any, *, optional: bool) -> list[float] | None:
    if value is None and optional:
        return None
    if (type(value) is not list or len(value) != 4
            or any(type(coordinate) not in (int, float) or not 0 <= coordinate <= 1 for coordinate in value)):
        raise ValidationError("Tracking boxes require four finite normalized numeric coordinates")
    x1, y1, x2, y2 = map(float, value)
    if (not 0 <= x1 < x2 <= 1 or not 0 <= y1 < y2 <= 1
            or (x2 - x1) * (y2 - y1) <= 0):
        raise ValidationError("Tracking boxes require positive exclusive area inside the unit image")
    return [x1, y1, x2, y2]


def _key(row: Any, fields: set[str]) -> tuple[str, int]:
    if type(row) is not dict or set(row) != fields:
        raise ValidationError("Tracking rows must contain exactly their supported fields")
    clip, frame = row["clip_id"], row["frame_index"]
    if type(clip) is not str or _CLIP.fullmatch(clip) is None:
        raise ValidationError("Tracking clip IDs must be bounded simple identifiers")
    if type(frame) is not int or not 0 <= frame <= 2 ** 31 - 1:
        raise ValidationError("Tracking frame indices must be bounded nonnegative integers")
    return clip, frame


def _references(rows: Any) -> dict[tuple[str, int], dict[str, Any]]:
    if type(rows) is not list or not 1 <= len(rows) <= 10_000:
        raise ValidationError("Supply 1 to 10000 frozen tracking reference rows")
    result = {}
    for row in rows:
        key = _key(row, _REFERENCE_FIELDS)
        if key in result:
            raise ValidationError("Tracking reference keys must be unique")
        if type(row["expected"]) is not str or row["expected"] not in ("supported", "unknown"):
            raise ValidationError("Tracking expected labels must be supported or unknown")
        if type(row["condition"]) is not str or row["condition"] not in _CONDITIONS:
            raise ValidationError("Tracking reference condition is unsupported")
        target = _box(row["target_box"], optional=row["expected"] == "unknown")
        if row["condition"] == "hidden_or_unidentifiable" and (row["expected"] != "unknown" or target is not None):
            raise ValidationError("Hidden or unidentifiable references require UNKNOWN without an invented target box")
        distractors = row["distractor_boxes"]
        if type(distractors) is not list or len(distractors) > 64:
            raise ValidationError("Tracking reference distractors must be a bounded list of boxes")
        result[key] = {"clip_id": key[0], "frame_index": key[1], "expected": row["expected"],
                       "condition": row["condition"], "target_box": target,
                       "distractor_boxes": [_box(box, optional=False) for box in distractors]}
    return result


def _predictions(rows: Any) -> dict[tuple[str, int], dict[str, Any]]:
    if type(rows) is not list or not 1 <= len(rows) <= 10_000:
        raise ValidationError("Supply one tracking prediction row per frozen reference")
    result = {}
    for row in rows:
        key = _key(row, _PREDICTION_FIELDS)
        if key in result:
            raise ValidationError("Tracking prediction keys must be unique")
        status = row["status"]
        if type(status) is not str or status not in ("supported", "unknown", "unscored"):
            raise ValidationError("Tracking status must be supported, unknown or unscored")
        box = _box(row["box"], optional=status != "supported")
        if status == "unscored" and box is not None:
            raise ValidationError("Unscored tracking frames cannot contain a fabricated prediction box")
        result[key] = {"clip_id": key[0], "frame_index": key[1], "status": status, "box": box}
    return result


def _iou(left: list[float] | None, right: list[float] | None) -> float | None:
    if left is None or right is None:
        return None
    intersection = max(0.0, min(left[2], right[2]) - max(left[0], right[0])) * max(
        0.0, min(left[3], right[3]) - max(left[1], right[1]))
    union = ((left[2] - left[0]) * (left[3] - left[1])
             + (right[2] - right[0]) * (right[3] - right[1]) - intersection)
    return min(1.0, max(0.0, intersection / union))


def _metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    completed = sum(row["status"] != "unscored" for row in rows)
    positives = sum(row["expected"] == "supported" for row in rows)
    unknowns = total - positives
    retained = sum(row["classification"] == "correct_support" for row in rows)
    supported = sum(row["status"] == "supported" for row in rows)
    unknown_withheld = sum(row["classification"] == "correct_unknown" for row in rows)
    decision_correct = retained + unknown_withheld
    label_correct = sum(row["label_agreement"] is True for row in rows)
    return {
        "labels_total": total, "labels_completed": completed, "labels_unscored": total - completed,
        "positive_count": positives, "positives_retained": retained,
        "positive_retention": _fraction(retained, positives),
        "positive_unscored": sum(row["expected"] == "supported" and row["status"] == "unscored" for row in rows),
        "unknown_count": unknowns,
        "unknown_unscored": sum(row["expected"] == "unknown" and row["status"] == "unscored" for row in rows),
        "unknowns_withheld": unknown_withheld,
        "unsupported_support_on_unknown": sum(row["classification"] == "unsupported_support_on_unknown" for row in rows),
        "unknown_support_rate": _fraction(sum(row["classification"] == "unsupported_support_on_unknown" for row in rows), unknowns),
        "wrong_localization_on_positive": sum(row["classification"] == "wrong_localization_on_positive" for row in rows),
        "abstentions_on_positive": sum(row["classification"] == "abstention_on_positive" for row in rows),
        "supported_count": supported, "supported_correct": retained,
        "selective_accuracy": _fraction(retained, supported),
        "coverage": _fraction(supported, total), "coverage_denominator": "all_labeled_frames",
        "completed_support_coverage": _fraction(supported, completed),
        "completion_rate": _fraction(completed, total),
        "decision_agreement_count": decision_correct,
        "decision_agreement": _fraction(decision_correct, total),
        "decision_agreement_denominator": "all_labeled_frames_including_unscored",
        "decision_label_agreement_count": label_correct,
        "decision_label_agreement": _fraction(label_correct, total),
        "completed_decision_agreement": _fraction(decision_correct, completed),
        "supported_distractor_alignment_at_threshold": sum(row["status"] == "supported" and row["distractor_alignment_at_threshold"] is True for row in rows),
    }


def score_frames(references: list[dict[str, Any]], predictions: list[dict[str, Any]], *,
                 iou_threshold: float = .3) -> dict[str, Any]:
    """Score exactly matching clip/frame keys against pre-inference annotations.

    All rates expose their count denominators. A supported result earns retained
    support only on a positive label with target overlap at the frozen threshold.
    UNKNOWN predictions may retain their diagnostic box but receive no support
    credit. Unscored rows receive no agreement or abstention credit. Inputs are
    copied and never passed to a model, written to disk or changed in place.
    """
    if type(iou_threshold) not in (int, float) or not 0 < iou_threshold <= 1 or not math.isfinite(iou_threshold):
        raise ValidationError("Freeze a finite tracking IoU threshold greater than zero and at most one")
    truth, output = _references(references), _predictions(predictions)
    if set(truth) != set(output):
        raise ValidationError("Tracking predictions must cover exactly the frozen clip and frame keys")
    frames = []
    for key, reference in sorted(truth.items()):
        prediction = output[key]
        status, expected = prediction["status"], reference["expected"]
        target_iou = _iou(prediction["box"], reference["target_box"])
        overlaps = [_iou(prediction["box"], box) for box in reference["distractor_boxes"]]
        alignment = max((value for value in overlaps if value is not None), default=None)
        localized = target_iou is not None and target_iou >= iou_threshold
        classification = ("unscored" if status == "unscored" else
                          "correct_unknown" if status == "unknown" and expected == "unknown" else
                          "abstention_on_positive" if status == "unknown" else
                          "unsupported_support_on_unknown" if expected == "unknown" else
                          "correct_support" if localized else "wrong_localization_on_positive")
        frames.append({
            "clip_id": key[0], "frame_index": key[1], "condition": reference["condition"],
            "expected": expected, "status": status, "prediction_box": deepcopy(prediction["box"]),
            "target_iou": round(target_iou, 6) if target_iou is not None else None,
            "target_localization_at_threshold": bool(localized) if target_iou is not None else None,
            "distractor_alignment": round(alignment, 6) if alignment is not None else None,
            "distractor_alignment_at_threshold": bool(alignment >= iou_threshold) if alignment is not None else None,
            "completed": status != "unscored",
            "label_agreement": (status == expected) if status != "unscored" else None,
            "decision_agreement": classification in ("correct_support", "correct_unknown") if status != "unscored" else None,
            "classification": classification,
            "physical_action_authorized": False,
        })
    clips: dict[str, list[dict[str, Any]]] = defaultdict(list)
    conditions: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in frames:
        clips[row["clip_id"]].append(row)
        conditions[row["condition"]].append(row)
    return {
        "schema_version": SCHEMA_VERSION, "metric_version": METRIC_VERSION,
        "iou_threshold": float(iou_threshold), "box_coordinates": "normalized_xyxy_exclusive",
        "metrics": _metrics(frames), "frames": frames,
        "by_clip": [{"clip_id": clip, "metrics": _metrics(rows)} for clip, rows in sorted(clips.items())],
        "by_condition": [{"condition": condition, "metrics": _metrics(rows)} for condition, rows in sorted(conditions.items())],
        "physical_action_authorized": False, "physical_action_verified": False,
        "identity_switches_measured": False, "limitations": list(_LIMITS),
    }
