"""Zero-new-inference retest of exposed cached SAM tracking observations.

Only one policy input changes: a runtime distractor whose source-bound decoded
masks are demonstrably disjoint may cease to impose a box-only overlap veto.
The frozen v1 policy, semantic judgments, visibility flags, thresholds, timing,
seeds, annotations and original scores remain intact. This is a development
retest after inspecting failures, not an independent reliability benchmark.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Any, Mapping

from .core import ValidationError
from . import tracking_experiment as original
from . import tracking_geometry as geometry
from .tracking_evaluation import score_frames
from .tracking_policy import evaluate_tracking


VERSION = "cached_tracking_mask_disjoint_retest_v2"
EVALUATION_ROLE = "exposed_cached_output_development_retest"
CONFIG = {"target_track_id": 1, "geometry_method": "source_mask_conflicts_v1",
    "change": "remove_only_validated_source_mask_disjoint_distractor_veto",
    "fallback": "retain_runtime_distractor_when_mask_or_metadata_is_unavailable",
    "semantic_judgments_unchanged": True, "visibility_flags_unchanged": True,
    "thresholds_and_stability_unchanged": True, "reference_labels_unchanged": True,
    "added_provider_requests": 0}
FILES = ("tracking_mask_replay.py", "tracking_geometry.py")
MAX_JSON_BYTES = 128 * 1024 * 1024
MAX_INPUT_BYTES = 512 * 1024 * 1024
MAX_INPUT_FILES = 1024
_HASH = re.compile(r"[0-9a-f]{64}\Z")
_COMPARISON_FIELDS = {"distractor_track_id", "bbox_overlap", "mask_relation", "intersection_pixels",
    "target_foreground_pixels", "distractor_foreground_pixels", "mask_iou", "target_raster_sha256",
    "distractor_raster_sha256", "target_mapped_mask_sha256", "distractor_mapped_mask_sha256"}
_MASK_HASH_FIELDS = ("target_raster_sha256", "distractor_raster_sha256",
                     "target_mapped_mask_sha256", "distractor_mapped_mask_sha256")
_LIMITS = [
    "Exposed cached-output development retest after inspecting v1 failures; no new scenes, semantic judgments or independent accuracy evidence.",
    "Disjoint SAM foreground removes a box-only geometry conflict. It does not independently prove semantic identity, full visibility, physical occlusion or execution.",
    "Missing, invalid or unvalidated mask metadata retains every applicable original bounding-box risk.",
    "The original semantic policy and its visibility, confidence, stability, timing and seed requirements are unchanged; removing a geometry veto need not improve supported localization.",
    "Author annotations are used only for scoring. They do not enter runtime distractor selection or mask geometry.",
    "Local source/code hashes detect changes relative to this receipt; they are not signatures, trusted timestamps or proof of API origin.",
    "No provider request is made by this retest, and live placement acceptance and physical-action authority are unchanged.",
]


def _digest(value: Any) -> str:
    return original._digest(value)


def _local(root: Path, value: str | Path) -> Path:
    return original._local(root, value)


def _file_record(root: Path, path: Path, role: str) -> dict[str, Any]:
    path = _local(root, path)
    if not path.is_file() or not 0 <= path.stat().st_size <= MAX_INPUT_BYTES:
        raise ValidationError("Cached retest input is unavailable or excessive")
    digest, size = hashlib.sha256(), 0
    with path.open("rb") as stream:
        while True:
            part = stream.read(1024 * 1024)
            if not part:
                break
            size += len(part)
            if size > MAX_INPUT_BYTES:
                raise ValidationError("Cached retest input exceeds its byte limit")
            digest.update(part)
    return {"file": path.relative_to(root).as_posix(), "sha256": digest.hexdigest(), "bytes": size, "role": role}


def _read_json_bytes(data: bytes) -> Any:
    def unique(rows):
        value = {}
        for key, item in rows:
            if key in value:
                raise ValueError
            value[key] = item
        return value
    try:
        result = json.loads(data.decode("utf-8"), object_pairs_hook=unique,
            parse_constant=lambda _: (_ for _ in ()).throw(ValueError()))
        _digest(result)  # Reject non-finite numeric overflow as well as NaN.
        return result
    except (ValueError, UnicodeError, TypeError, RecursionError):
        raise ValidationError("Cached retest JSON is invalid") from None


def _snapshot(root: Path, frozen_file: Path) -> dict[str, Any]:
    """Capture JSON once; hash all media/crops and the persistent request journal."""
    records, captured = {}, {}

    def add(path, role, *, as_json=False):
        path = _local(root, path)
        key = path.relative_to(root).as_posix()
        if key in records:
            return captured.get(key)
        if len(records) >= MAX_INPUT_FILES:
            raise ValidationError("Cached retest has too many input records")
        record = _file_record(root, path, role)
        if as_json:
            if record["bytes"] > MAX_JSON_BYTES:
                raise ValidationError("Cached retest JSON is excessive")
            with path.open("rb") as stream:
                data = stream.read(MAX_JSON_BYTES + 1)
            if len(data) > MAX_JSON_BYTES or hashlib.sha256(data).hexdigest() != record["sha256"]:
                raise ValidationError("Cached retest JSON changed while being captured")
            captured[key] = _read_json_bytes(data)
        records[key] = record
        return captured.get(key)

    frozen = add(frozen_file, "original_freeze", as_json=True)
    add(frozen["annotations_file"], "original_annotations", as_json=True)
    output = frozen_file.parent
    for entry in frozen["clips"]:
        name = entry["clip_id"]
        add(entry["original_file"], "original_video")
        add(entry["prepared"]["clip_file"], "tracking_clip")
        for frame in entry["frames"]:
            add(frame["file"], "decoded_source_jpeg")
        add(output / f"{name}-video.json", "tracking_response", as_json=True)
        for checkpoint in entry["checkpoints"]:
            checked = add(output / f"{name}-guard-{checkpoint['frame_index']}.json", "semantic_response", as_json=True)
            if isinstance(checked, dict) and isinstance(checked.get("prepared"), dict):
                for key, role in (("source_file", "semantic_source_jpeg"), ("crop_file", "semantic_crop_jpeg")):
                    if key in checked["prepared"]:
                        add(checked["prepared"][key], role)
        for pair in range(4):
            add(output / f"{name}-pair-{pair}.json", "still_pair_response", as_json=True)
    add(output / "attempts.sqlite3", "persistent_original_request_journal")
    for name, role in (("run-started.json", "original_admission_marker"), ("public-result.json", "original_public_result")):
        if (output / name).is_file():
            add(output / name, role, as_json=True)
    return {"plan": frozen, "records": [records[key] for key in sorted(records)], "json": captured}


def _unchanged(root: Path, records: list[dict[str, Any]]) -> None:
    for record in records:
        if _file_record(root, root / record["file"], record["role"]) != record:
            raise ValidationError("Cached retest input changed during validation or derivation")


def _implementations() -> dict[str, str]:
    return {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in FILES}


def _validated_base(root: Path, frozen_file: Path, *, expected_records=None):
    snapshot = _snapshot(root, frozen_file)
    if expected_records is not None and snapshot["records"] != expected_records:
        raise ValidationError("Cached original records differ from the prospective replay receipt")
    result = original.replay(root, frozen_file)
    _unchanged(root, snapshot["records"])
    if (result["freeze_hash"] != snapshot["plan"]["freeze_hash"]
            or result["implementation_sha256"] != snapshot["plan"]["implementation_sha256"]):
        raise ValidationError("Original replay does not identify the captured frozen experiment")
    saved_public = snapshot["json"].get((frozen_file.parent / "public-result.json").relative_to(root).as_posix())
    if saved_public is not None and _digest(saved_public) != _digest(result):
        raise ValidationError("Original published result differs from its validated replay")
    return snapshot, result


def freeze_replay(root: str | Path, frozen: str | Path, outputdir: str | Path) -> dict[str, Any]:
    """Write a new private prospective receipt before computing corrected results."""
    root = Path(root).resolve()
    frozen_file, output = _local(root, frozen), _local(root, outputdir)
    if output == frozen_file.parent or output.is_relative_to(frozen_file.parent):
        raise ValidationError("Keep the cached retest in a new directory outside the original experiment")
    snapshot, base = _validated_base(root, frozen_file)
    plan = {"replay_version": VERSION, "evaluation_role": EVALUATION_ROLE,
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "derivation_started": False,
        "original_frozen_file": frozen_file.relative_to(root).as_posix(),
        "original_freeze_hash": base["freeze_hash"], "original_public_sha256": _digest(base),
        "original_implementation_sha256": deepcopy(base["implementation_sha256"]),
        "implementation_sha256": _implementations(), "config": deepcopy(CONFIG), "config_sha256": _digest(CONFIG),
        "input_records": snapshot["records"], "added_provider_requests": 0}
    plan["replay_plan_hash"] = _digest(plan)
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    original._write(output / "freeze.json", plan)
    return plan


def _plan(root: Path, planfile: Path) -> dict[str, Any]:
    with planfile.open("rb") as stream:
        data = stream.read(MAX_JSON_BYTES + 1)
    if len(data) > MAX_JSON_BYTES:
        raise ValidationError("Cached replay receipt is excessive")
    plan = _read_json_bytes(data)
    if (plan["replay_version"] != VERSION or plan["evaluation_role"] != EVALUATION_ROLE
            or plan["derivation_started"] is not False or plan["added_provider_requests"] != 0
            or plan["config"] != CONFIG or plan["config_sha256"] != _digest(CONFIG)
            or plan["implementation_sha256"] != _implementations()
            or plan["replay_plan_hash"] != _digest({key: value for key, value in plan.items() if key != "replay_plan_hash"})):
        raise ValidationError("Cached replay code, configuration or prospective receipt changed")
    return plan


def _runtime(entry: Mapping[str, Any], raw_track: Mapping[str, Any], guard_reports: Mapping[int, Any]):
    source = entry["prepared"]["source"]
    runtime = []
    for tracked in raw_track["frames"]:
        index = tracked["frame_index"]
        candidate = next((item for item in tracked["candidates"] if item["track_id"] == 1), None)
        checked = guard_reports.get(index)
        semantic, visible, ambiguous = None, None, None
        if checked and checked["status"] == "completed":
            prepared, evaluation = checked["prepared"], checked["evaluation"]
            visible, ambiguous = evaluation["fully_visible"], evaluation["identity_ambiguous"]
            semantic = {"target_id": entry["contract"]["target_id"], "track_id": 1, "frame_index": index,
                "source_sha256": source["sha256"], "frame_sha256": prepared["frame_sha256"],
                "candidate_box": candidate["box"], "decision": evaluation["decision"], "evidence_sha256": _digest(checked)}
        runtime.append({"frame_index": index, "timestamp_ms": tracked["timestamp_ms"], "source_sha256": source["sha256"],
            "frame_sha256": entry["frames"][index]["sha256"],
            "candidates": [{key: candidate[key] for key in ("track_id", "object_id", "box", "confidence", "mask_supported")}
                | {"fully_visible": visible, "identity_ambiguous": ambiguous, "semantic": semantic}] if candidate else [],
            "reviewed_distractors": [{"track_id": item["track_id"], "box": item["box"], "origin": "runtime_sam_track"}
                for item in tracked["candidates"] if item["track_id"] != 1]})
    return runtime


def _policy(entry, runtime, base_config):
    if not runtime:
        return None
    source = entry["prepared"]["source"]
    return evaluate_tracking(runtime, target_id=entry["contract"]["target_id"], seed={
        "target_id": entry["contract"]["target_id"], "track_id": 1, "frame_index": 0,
        "source_sha256": source["sha256"], "frame_sha256": entry["frames"][0]["sha256"],
        "review_sha256": entry["prepared"]["prepared_hash"],
        "reviewed_distractor_track_ids": [seed["track_id"] for seed in entry["seeds"] if seed["track_id"] != 1]},
        **{key: base_config[key] for key in ("min_confidence", "min_stable_views", "min_spacing_ms")})


def _bbox_overlap(first, second):
    return max(first[0], second[0]) < min(first[2], second[2]) and max(first[1], second[1]) < min(first[3], second[3])


def _comparison(value: Any, distractor_id: int, bbox_overlap: bool) -> dict[str, Any] | None:
    """Validate all metadata needed to remove a risk; malformed facts never do."""
    try:
        if (type(value) is not dict or set(value) != _COMPARISON_FIELDS
                or type(value["distractor_track_id"]) is not int or value["distractor_track_id"] != distractor_id
                or type(value["bbox_overlap"]) is not bool or value["bbox_overlap"] != bbox_overlap
                or value["mask_relation"] not in ("disjoint", "overlap", "unavailable")):
            return None
        for field in ("intersection_pixels", "target_foreground_pixels", "distractor_foreground_pixels"):
            if value[field] is not None and (type(value[field]) is not int or not 0 <= value[field] <= geometry.MAX_SOURCE_MAPPED_PIXELS):
                return None
        for field in _MASK_HASH_FIELDS:
            if value[field] is not None and (type(value[field]) is not str or _HASH.fullmatch(value[field]) is None):
                return None
        ratio = value["mask_iou"]
        if ratio is not None and (type(ratio) not in (int, float) or not math.isfinite(ratio) or not 0 <= ratio <= 1):
            return None
        if value["mask_relation"] in ("disjoint", "overlap"):
            first, second, count = value["target_foreground_pixels"], value["distractor_foreground_pixels"], value["intersection_pixels"]
            if (type(first) is not int or type(second) is not int or first < 1 or second < 1
                    or type(count) is not int or not 0 <= count <= min(first, second)
                    or any(value[field] is None for field in _MASK_HASH_FIELDS)
                    or ratio != round(count / (first + second - count), 6)
                    or (count == 0) != (value["mask_relation"] == "disjoint")):
                return None
        elif value["intersection_pixels"] is not None or ratio is not None:
            return None
        return deepcopy(value)
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return None


def _filter(runtime, resolved, entry, raw_track):
    source = entry["prepared"]["source"]
    if (not isinstance(resolved, Mapping) or resolved.get("method_version") != geometry.GEOMETRY_VERSION
            or resolved.get("source_sha256") != source["sha256"]
            or resolved.get("prepared_hash") != entry["prepared"]["prepared_hash"]
            or resolved.get("mapping") != "box_local_mask_nearest_source_pixel_center_inclusive_xyxy"):
        raise ValidationError("Mask comparison is not bound to the frozen source and mapping")
    geometry_frames = resolved.get("frames")
    if type(geometry_frames) is not list:
        raise ValidationError("Mask comparisons require source-indexed frames")
    indexed = {}
    for row in geometry_frames:
        if type(row) is not dict or type(row.get("frame_index")) is not int or row["frame_index"] in indexed:
            raise ValidationError("Mask comparison source frame is invalid or duplicated")
        indexed[row["frame_index"]] = row
    if set(indexed) - {row["frame_index"] for row in runtime}:
        raise ValidationError("Mask comparisons contain unrelated source frames")
    raw_by_frame = {frame["frame_index"]: frame for frame in raw_track["frames"]}
    corrected, audits = deepcopy(runtime), []
    for row in corrected:
        index = row["frame_index"]
        frame = indexed.get(index)
        candidate = row["candidates"][0] if len(row["candidates"]) == 1 else None
        frame_valid = bool(isinstance(frame, dict) and frame.get("source_sha256") == source["sha256"]
            and type(frame.get("target_track_id")) is int and frame["target_track_id"] == 1
            and type(frame.get("target_available")) is bool and frame["target_available"] == bool(candidate)
            and type(frame.get("comparisons")) is list and type(frame.get("conflicting_track_ids")) is list
            and all(type(identifier) is int for identifier in frame["conflicting_track_ids"]))
        comparisons = {}
        if frame_valid:
            for comparison in frame["comparisons"]:
                identifier = comparison.get("distractor_track_id") if isinstance(comparison, dict) else None
                if type(identifier) is not int or identifier in comparisons:
                    frame_valid = False
                    break
                comparisons[identifier] = comparison
        retained, audited = [], []
        raw_masks = {item["track_id"]: item.get("mask", {}) for item in raw_by_frame[index]["candidates"]}
        for distractor in row["reviewed_distractors"]:
            identifier = distractor["track_id"]
            overlap = bool(candidate and _bbox_overlap(candidate["box"], distractor["box"]))
            checked = _comparison(comparisons.get(identifier), identifier, overlap) if frame_valid and candidate else None
            if checked and checked["mask_relation"] == "disjoint":
                target_mask, distractor_mask = raw_masks.get(1, {}), raw_masks.get(identifier, {})
                if (not isinstance(target_mask, dict) or not isinstance(distractor_mask, dict)
                        or target_mask.get("decoded") is not True or distractor_mask.get("decoded") is not True
                        or target_mask.get("raster_sha256") != checked["target_raster_sha256"]
                        or distractor_mask.get("raster_sha256") != checked["distractor_raster_sha256"]):
                    checked = None
            remove = bool(overlap and checked and checked["mask_relation"] == "disjoint"
                and identifier not in frame["conflicting_track_ids"])
            if not remove:
                retained.append(distractor)
            audited.append({"distractor_track_id": identifier, "bbox_overlap": overlap,
                "metadata_valid": checked is not None, "removed_disjoint_veto": remove,
                "retained_bbox_risk": overlap and not remove,
                "comparison": checked,
                "fallback_reason": None if checked is not None else "mask_or_comparison_metadata_unavailable"})
        row["reviewed_distractors"] = retained
        audits.append({"frame_index": index, "source_sha256": source["sha256"], "target_track_id": 1,
            "geometry_frame_valid": frame_valid, "comparisons": audited,
            "retained_distractor_track_ids": [item["track_id"] for item in retained],
            "removed_disjoint_track_ids": [item["distractor_track_id"] for item in audited if item["removed_disjoint_veto"]]})
    return corrected, {"method_version": geometry.GEOMETRY_VERSION, "source_sha256": source["sha256"],
        "prepared_hash": entry["prepared"]["prepared_hash"],
        "mapping": "box_local_mask_nearest_source_pixel_center_inclusive_xyxy", "frames": audits}


def _events(policy):
    if policy is None:
        return {"hard_gap_events": 0, "reacquisition_events": 0, "track_identifier_changes": 0}
    return {key: policy["metrics"][key] for key in ("hard_gap_events", "reacquisition_events", "track_identifier_changes")}


def _overlap_frames(policy):
    return sum("reviewed_distractor_overlap" in frame["gap_codes"] for frame in policy["frames"]) if policy else 0


def _derive(root: Path, plan: Mapping[str, Any]):
    frozen_file = _local(root, plan["original_frozen_file"])
    snapshot, base = _validated_base(root, frozen_file, expected_records=plan["input_records"])
    if (_digest(base) != plan["original_public_sha256"] or base["freeze_hash"] != plan["original_freeze_hash"]
            or base["implementation_sha256"] != plan["original_implementation_sha256"]):
        raise ValidationError("Original comparison differs from the prospective replay receipt")
    base_plan, output = snapshot["plan"], frozen_file.parent
    by_clip = {item["clip_id"]: item for item in base["evidence"]}
    predictions, evidence, private = [], [], []
    v1_events, v2_events = Counter(), Counter()
    v1_overlap = v2_overlap = removed = remaining = invalid = 0
    for entry in base_plan["clips"]:
        name = entry["clip_id"]
        raw = snapshot["json"][(output / f"{name}-video.json").relative_to(root).as_posix()]
        checks = {row["frame_index"]: snapshot["json"][(output / f"{name}-guard-{row['frame_index']}.json").relative_to(root).as_posix()]
            for row in entry["checkpoints"]}
        runtime = _runtime(entry, raw, checks)
        before = _policy(entry, runtime, base_plan["config"])
        if before != by_clip[name]["policy"]:
            raise ValidationError("Reconstructed original runtime differs from its validated v1 policy")
        resolved = geometry.resolve_distractor_conflicts(raw, target_track_id=1)
        corrected, audit = _filter(runtime, resolved, entry, raw)
        after = _policy(entry, corrected, base_plan["config"])
        v1_events.update(_events(before))
        v2_events.update(_events(after))
        v1_overlap += _overlap_frames(before)
        v2_overlap += _overlap_frames(after)
        for frame in audit["frames"]:
            for comparison in frame["comparisons"]:
                removed += comparison["removed_disjoint_veto"]
                remaining += comparison["retained_bbox_risk"]
                invalid += comparison["bbox_overlap"] and not comparison["metadata_valid"]
        after_by_frame = {row["frame_index"]: row for row in after["frames"]} if after else {}
        for checkpoint in entry["checkpoints"]:
            index = checkpoint["frame_index"]
            row = after_by_frame.get(index)
            status = "unscored" if not row or checks[index]["status"] == "unscored" else row["status"]
            predictions.append({"clip_id": name, "frame_index": index, "status": status,
                "box": row["box"] if row and status == "supported" else None})
        evidence.append({"clip_id": name, "source_attribution": deepcopy(by_clip[name]["source_attribution"]),
            "tracking_clip_sha256": entry["prepared"]["source"]["sha256"],
            "original_policy_sha256": _digest(before), "corrected_policy": after,
            "source_mask_geometry": audit, "event_counts_v1": _events(before), "event_counts_v2": _events(after),
            "overlap_veto_frames_v1": _overlap_frames(before), "overlap_veto_frames_v2": _overlap_frames(after)})
        private.append({"clip_id": name, "original_runtime": runtime, "corrected_runtime": corrected})
    references = deepcopy(base["references"])
    scored = score_frames(references, predictions, iou_threshold=base_plan["config"]["localization_iou_threshold"])
    original_score = deepcopy(base["methods"]["semantic_reacquisition"])
    result = {"replay_version": VERSION, "replay_plan_hash": plan["replay_plan_hash"], "evaluation_role": EVALUATION_ROLE,
        "original_freeze_hash": base["freeze_hash"], "original_public_sha256": _digest(base),
        "original_implementation_sha256": deepcopy(base["implementation_sha256"]),
        "implementation_sha256": deepcopy(plan["implementation_sha256"]), "config": deepcopy(CONFIG),
        "config_sha256": plan["config_sha256"], "unchanged_policy_config": {key: base_plan["config"][key]
            for key in ("min_confidence", "min_stable_views", "min_spacing_ms", "localization_iou_threshold")},
        "source_videos": base["source_videos"], "labeled_frames": len(references), "references": references,
        "methods": {"bbox_conflict_v1": original_score, "mask_disjoint_v2": scored},
        "unmodified_baseline_metrics": {name: deepcopy(base["methods"][name]["metrics"]) for name in original.METHODS[:3]},
        "event_counts_v1": dict(v1_events), "event_counts_v2": dict(v2_events),
        "overlap_veto_frames_v1": v1_overlap, "overlap_veto_frames_v2": v2_overlap,
        "overlap_veto_reduction": v1_overlap - v2_overlap,
        "validated_disjoint_bbox_conflicts_removed": removed,
        "bbox_conflicts_retained": remaining, "bbox_conflicts_with_unvalidated_metadata_retained": invalid,
        "localization_sensitivity": {str(threshold): score_frames(references, predictions, iou_threshold=threshold)["metrics"]
            for threshold in base_plan["config"]["localization_sensitivity_thresholds"]},
        "evidence": evidence, "original_accounting": deepcopy(base["accounting"]),
        "original_estimated_usd": base["estimated_usd"], "added_provider_requests": 0,
        "estimated_incremental_provider_usd": 0.0, "changes_live_placement_acceptance": False,
        "physical_action_authorized": False, "limitations": list(_LIMITS)}
    _unchanged(root, snapshot["records"])
    if plan["implementation_sha256"] != _implementations():
        raise ValidationError("Cached replay implementation changed during derivation")
    return result, private


def replay(root: str | Path, planfile: str | Path) -> dict[str, Any]:
    """Validate and recompute the cached retest without writing or making API calls."""
    root = Path(root).resolve()
    path = _local(root, planfile)
    plan = _plan(root, path)
    return _derive(root, plan)[0]


def run(root: str | Path, planfile: str | Path) -> dict[str, Any]:
    """Derive only after the prospective receipt exists; write new files exclusively."""
    root = Path(root).resolve()
    path = _local(root, planfile)
    plan = _plan(root, path)
    original._write(path.parent / "run-started.json", {"replay_plan_hash": plan["replay_plan_hash"], "added_provider_requests": 0})
    public, private = _derive(root, plan)
    original._write(path.parent / "private-result.json", {"replay_plan_hash": plan["replay_plan_hash"],
        "clips": private, "public_result_sha256": _digest(public)})
    original._write(path.parent / "public-result.json", public)
    return public


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--frozen", required=True, type=Path, help="Original validated v1 freeze.json")
    parser.add_argument("--output", required=True, type=Path, help="New private retest directory outside the original experiment")
    args = parser.parse_args(argv)
    plan = freeze_replay(args.root, args.frozen, args.output)
    result = run(args.root, Path(args.output) / "freeze.json")
    print(json.dumps({"replay_plan_hash": plan["replay_plan_hash"], "added_provider_requests": 0,
        "overlap_veto_reduction": result["overlap_veto_reduction"],
        "v1_positive_retention": result["methods"]["bbox_conflict_v1"]["metrics"]["positive_retention"],
        "v2_positive_retention": result["methods"]["mask_disjoint_v2"]["metrics"]["positive_retention"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
