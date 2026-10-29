"""Prospective visibility comparison on exposed and new licensed recordings.

Reference labels are scoring-only. Both assessors see the same actual tracker
candidate; v2 additionally sees its full source frame. Failed calls remain
unscored, and historical tracking results are never overwritten.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .core import ValidationError
from .inference_journal import AttemptJournal
from .meta import SPARK_MODEL, load_env_file, output_text
from . import tracking_experiment as original
from . import tracking_guard as old_guard
from . import tracking_mask_replay as masks
from . import tracking_geometry as geometry
from . import visibility_guard as guard
from .tracking_evaluation import score_frames
from .video_tracking import segment_video, public_tracking_summary

VERSION = "frozen_full_frame_visibility_comparison_v1"
METHODS = ("sam_presence", "visibility_v1", "visibility_v2", "candidate_guard_v1",
           "candidate_guard_v2", "mask_policy_v1", "mask_policy_v2")
COHORTS = ("exposed_development", "new_recordings_same_families")
CONFIG = {"max_requests": 50, "estimated_allocation_usd": 3.0, "automatic_retry": False,
    "max_output_tokens": 4096, "reasoning_effort": "low", "min_confidence": 0.8,
    "min_stable_views": 2, "min_spacing_ms": 500, "localization_iou_threshold": 0.3,
    "localization_sensitivity_thresholds": [0.2, 0.3, 0.5],
    "candidate_origin": "actual_sam_video_track_only",
    "new_assessor_inputs": "current_full_frame_then_registered_candidate_crop_no_reference",
    "selection": "eight_frozen_checkpoints_per_clip_no_response_based_selection"}
FILES = tuple(sorted(set(original.FILES + masks.FILES + ("visibility_guard.py", "visibility_experiment.py"))))
LIMITATIONS = [
    "Exposed development and new recordings are reported separately; dataset families, camera setup and tasks remain exposed.",
    "Reference visibility labels and boxes are assistant-authored before new inference, not official dataset annotations.",
    "Both assessors use the same manually initialized SAM video candidates. V2 adds full-frame context and changes its rubric, so this does not isolate a prompt-only improvement.",
    "Visibility alone is not target identity. Candidate support and the stricter two-view policy are scored separately.",
    "Same-model semantic judgments and SAM masks do not independently prove identity, full disappearance recovery, physical action or continuous visibility.",
    "The new-source remote's final boundary is uncertain under a nearby gripper; it is labeled UNKNOWN rather than counted as clear re-exposure.",
    "Missing candidates are explicit UNKNOWN; missing/failed provider classifications are unscored, never successful abstentions.",
    "No thresholds or labels are changed after inference, no automatic retries occur, and live placement acceptance remains unchanged.",
    "Reported costs are estimates; request-count admission is not a provider-enforced dollar limit.",
]


def _implementations():
    return {name: original._sha(Path(__file__).with_name(name)) for name in FILES}


def _file(root: Path, path: str | Path, role: str):
    return masks._file_record(root, original._local(root, path), role)


def _fresh_records(root: Path, frozen: Path, plan: Mapping[str, Any]):
    records = [_file(root, frozen, "fresh_source_preparation"),
               _file(root, plan["annotations_file"], "fresh_scoring_annotations")]
    for entry in plan["clips"]:
        records.extend([_file(root, entry["original_file"], "fresh_original_video"),
                        _file(root, entry["prepared"]["clip_file"], "fresh_tracking_clip")])
        records.extend(_file(root, row["file"], "fresh_source_jpeg") for row in entry["frames"])
    return records


def freeze(root: str | Path, original_frozen: str | Path, fresh_frozen: str | Path,
           output: str | Path) -> dict[str, Any]:
    root = Path(root).resolve()
    old_path, fresh_path = original._local(root, original_frozen), original._local(root, fresh_frozen)
    snapshot, old_result = masks._validated_base(root, old_path)
    fresh_plan = original._read(fresh_path)
    original.verify_plan(root, fresh_plan)
    old_sources = {entry["source_attribution"]["dataset"]: entry for entry in snapshot["plan"]["clips"]}
    if (len(fresh_plan["clips"]) != 2 or len(old_sources) != 2
            or {e["source_attribution"]["dataset"] for e in fresh_plan["clips"]} != set(old_sources)):
        raise ValidationError("Compare exactly two new source recordings from the same declared families")
    for entry in fresh_plan["clips"]:
        previous = old_sources[entry["source_attribution"]["dataset"]]
        if (entry["source_attribution"]["episode"] in {0, previous["source_attribution"]["episode"]}
                or entry["original_sha256"] == previous["original_sha256"]
                or entry["prepared"]["source"]["sha256"] == previous["prepared"]["source"]["sha256"]
                or entry["contract"] != previous["contract"]):
            raise ValidationError("New source recordings need new bytes and the unchanged visual contract")
    if set(e["clip_id"] for e in fresh_plan["clips"]) & set(e["clip_id"] for e in snapshot["plan"]["clips"]):
        raise ValidationError("Cohort clip identifiers must differ")
    destination = original._local(root, output)
    if (destination in (old_path.parent, fresh_path.parent) or destination.is_relative_to(old_path.parent)
            or destination.is_relative_to(fresh_path.parent)):
        raise ValidationError("Keep visibility responses outside historical/source-only directories")
    plan = {"experiment_version": VERSION, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "inference_started": False, "config": deepcopy(CONFIG), "methods": list(METHODS),
        "implementation_sha256": _implementations(), "new_guard_config": deepcopy(guard.CONFIG),
        "old_guard_config": deepcopy(old_guard.CONFIG),
        "original_frozen_file": old_path.relative_to(root).as_posix(),
        "original_freeze_hash": snapshot["plan"]["freeze_hash"],
        "original_public_sha256": original._digest(old_result),
        "fresh_frozen_file": fresh_path.relative_to(root).as_posix(),
        "fresh_source_preparation_hash": fresh_plan["freeze_hash"],
        "input_records": snapshot["records"] + _fresh_records(root, fresh_path, fresh_plan),
        "limitations": list(LIMITATIONS)}
    plan["freeze_hash"] = original._digest(plan)
    destination.mkdir(parents=True, exist_ok=False, mode=0o700)
    original._write(destination / "freeze.json", plan)
    return plan


def verify_plan(root: Path, plan: Mapping[str, Any], *, files=True):
    if (plan["experiment_version"] != VERSION or plan["inference_started"] is not False
            or plan["config"] != CONFIG or plan["methods"] != list(METHODS)
            or plan["new_guard_config"] != guard.CONFIG or plan["old_guard_config"] != old_guard.CONFIG
            or plan["implementation_sha256"] != _implementations() or plan["limitations"] != LIMITATIONS
            or plan["freeze_hash"] != original._digest({k: v for k, v in plan.items() if k != "freeze_hash"})):
        raise ValidationError("Visibility experiment code, configuration or prospective receipt changed")
    old = original._read(original._local(root, plan["original_frozen_file"]))
    fresh = original._read(original._local(root, plan["fresh_frozen_file"]))
    if old["freeze_hash"] != plan["original_freeze_hash"] or fresh["freeze_hash"] != plan["fresh_source_preparation_hash"]:
        raise ValidationError("Visibility source preparation differs from the receipt")
    if files:
        masks._unchanged(root, plan["input_records"])
        original.verify_plan(root, old)
        original.verify_plan(root, fresh)
        if original._digest(original.replay(root, original._local(root, plan["original_frozen_file"]))) != plan["original_public_sha256"]:
            raise ValidationError("Historical comparison changed")
    return old, fresh


def public_plan(root: str | Path, frozen: str | Path):
    root = Path(root).resolve()
    plan = original._read(original._local(root, frozen))
    old, fresh = verify_plan(root, plan)
    return {"experiment_version": VERSION, "freeze_hash": plan["freeze_hash"], "inference_started": False,
        "config": deepcopy(CONFIG), "methods": list(METHODS), "implementation_sha256": plan["implementation_sha256"],
        "original_freeze_hash": plan["original_freeze_hash"],
        "fresh_source_preparation_hash": plan["fresh_source_preparation_hash"],
        "input_inventory_sha256": original._digest(plan["input_records"]),
        "cohorts": {name: {"source_videos": 2, "labeled_frames": 16,
            "sources": [{"clip_id": entry["clip_id"], "source_attribution": entry["source_attribution"],
                         "original_sha256": entry["original_sha256"], "derivation": entry["derivation"]}
                        for entry in source["clips"]],
            "references": [r for entry in source["clips"] for r in original._reference(entry)]}
            for name, source in zip(COHORTS, (old, fresh))},
        "limitations": list(LIMITATIONS), "changes_live_placement_acceptance": False}


def _no_candidate():
    return {"status": "not_run_no_tracker_candidate", "usage": {"request_attempts": 0, "estimated_usd": 0.0}}


def _assess(client, entry, raw, index, module, destination):
    frame = next((row for row in raw["frames"] if row["frame_index"] == index), None)
    candidate = original._candidate(frame) if frame else None
    if candidate is None:
        return _no_candidate()
    source = entry["prepared"]["source"]
    prepared = module.prepare(entry["frames"][index]["file"],
        original._pixels(candidate["box"], source["width"], source["height"]),
        entry["contract"], destination, source_sha256=source["sha256"], frame_index=index, track_id=1)
    return module.challenge(client(), prepared)


def run(root: str | Path, frozen: str | Path):
    root = Path(root).resolve()
    path = original._local(root, frozen)
    plan = original._read(path)
    old, fresh = verify_plan(root, plan)
    output = path.parent
    original._write(output / "run-started.json", {"freeze_hash": plan["freeze_hash"],
        "started_at_utc": datetime.now(timezone.utc).isoformat(), "automatic_retry": False})
    load_env_file(root / ".env.local")
    journal = AttemptJournal(output / "attempts.sqlite3", request_limit=CONFIG["max_requests"])
    def client(label, *, timeout=90):
        return original.RetainingClient.from_env(journal=journal,
            operation_id=original._digest([plan["freeze_hash"], label])[:32],
            guard=lambda: verify_plan(root, plan, files=False), max_requests=1, retries=0, timeout=timeout)
    for cohort, sources in zip(COHORTS, (old, fresh)):
        for entry in sources["clips"]:
            name = entry["clip_id"]
            if cohort == COHORTS[0]:
                raw = original._read(original._local(root, plan["original_frozen_file"]).parent / f"{name}-video.json")
            else:
                raw = segment_video(client(name + "-video", timeout=180), entry["prepared"])
                original._write(output / f"{name}-video.json", raw)
            for cp in entry["checkpoints"]:
                index = cp["frame_index"]
                if cohort == COHORTS[1]:
                    label = f"{name}-v1-{index}"
                    checked = _assess(lambda: client(label), entry, raw, index, old_guard, output / label)
                    original._write(output / f"{label}.json", checked)
                label = f"{name}-v2-{index}"
                checked = _assess(lambda: client(label), entry, raw, index, guard, output / label)
                original._write(output / f"{label}.json", checked)
    result = replay(root, path)
    original._write(output / "public-result.json", result)
    return result


def _verify_guard(checked, entry, raw, index, module):
    frame = next((f for f in raw["frames"] if f["frame_index"] == index), None)
    candidate = original._candidate(frame) if frame else None
    if candidate is None:
        if checked != _no_candidate():
            raise ValidationError("Guard classification requires its actual video candidate")
        return None, 0, 0.0
    if checked.get("status") not in ("completed", "unscored"):
        raise ValidationError("Actual candidates require completed or retained unscored assessments")
    prepared = checked["prepared"]
    module.verify(prepared)
    source = entry["prepared"]["source"]
    if (prepared["contract"] != entry["contract"] or prepared["source_sha256"] != source["sha256"]
            or prepared["frame_index"] != index or prepared["track_id"] != 1
            or prepared["frame_sha256"] != entry["frames"][index]["sha256"]
            or prepared["source_file"] != entry["frames"][index]["file"]
            or prepared["candidate_box"] != original._pixels(candidate["box"], source["width"], source["height"])):
        raise ValidationError("Visibility evidence differs from the source-bound actual tracked candidate")
    response = checked.get("response")
    raw_response = response["response"] if response is not None else checked["error"]["response"]
    original._verified_usage(checked, [raw_response], limit=1)
    if checked["status"] == "completed":
        parsed = masks._read_json_bytes(output_text(raw_response).encode())
        if (raw_response.get("status") != "completed" or raw_response.get("model") != SPARK_MODEL
                or parsed != response["parsed"] or module.evaluate(parsed, entry["contract"]) != checked["evaluation"]
                or checked.get("error") is not None or checked.get("method_version") != module.VERSION):
            raise ValidationError("Visibility classification differs from its retained provider response")
        expected = [prepared["frame_sha256"], prepared["crop_sha256"]]
        if module is guard:
            if response.get("source") is not None or [s.get("sha256") for s in response.get("sources", [])] != expected:
                raise ValidationError("Full-frame/crop order differs from the registered source images")
        elif response.get("source", {}).get("sha256") != prepared["crop_sha256"]:
            raise ValidationError("Old crop response differs from the registered source image")
    elif checked.get("evaluation") is not None or checked.get("error") is None:
        raise ValidationError("Unscored guard cannot contain a classification")
    summary = guard.public_summary(checked) if module is guard else {
        "method_version": old_guard.VERSION, "status": checked["status"],
        "prepared_hash": prepared["prepared_hash"], "source_sha256": prepared["source_sha256"],
        "frame_sha256": prepared["frame_sha256"], "crop_sha256": prepared["crop_sha256"],
        "frame_index": index, "track_id": 1, "candidate_box": prepared["candidate_box"],
        "classification": deepcopy(checked["evaluation"])}
    return summary, checked["usage"]["request_attempts"], checked["usage"]["estimated_usd"]


def _prediction(entry, raw, checked, index, *, visibility_only=False, presence_only=False):
    frame = next((f for f in raw["frames"] if f["frame_index"] == index), None)
    candidate = original._candidate(frame) if frame else None
    status, box = "unknown", None
    if raw["status"] != "completed" or (not presence_only and checked.get("status") == "unscored"):
        status = "unscored"
    elif candidate:
        if presence_only:
            supported = (candidate["confidence"] is not None and candidate["confidence"] >= CONFIG["min_confidence"]
                         and candidate["mask_supported"])
        else:
            if checked.get("status") != "completed":
                raise ValidationError("Candidate classification is missing")
            assessment = checked["evaluation"]
            supported = assessment["fully_visible"] if visibility_only else assessment["decision"] == "supported"
        if supported:
            status, box = "supported", candidate["box"]
    return {"clip_id": entry["clip_id"], "frame_index": index, "status": status, "box": box}


def visibility_counts(references, predictions):
    """Direct visibility flags, separately from the localization-IoU scores."""
    reference_by_key = {(r["clip_id"], r["frame_index"]): r for r in references}
    predicted_by_key = {(r["clip_id"], r["frame_index"]): r for r in predictions}
    # Reuse the closed scorer for inventory/status validation, not its IoU labels.
    score_frames(references, predictions, iou_threshold=CONFIG["localization_iou_threshold"])
    clear = [key for key, row in reference_by_key.items() if row["expected"] == "supported"]
    unclear = [key for key, row in reference_by_key.items() if row["expected"] == "unknown"]
    def count(keys, status):
        return sum(predicted_by_key[key]["status"] == status for key in keys)
    return {"basis": "visibility_flag_only_not_identity_or_localization_accuracy",
        "reference_clear_count": len(clear), "clear_called_clear": count(clear, "supported"),
        "clear_withheld": count(clear, "unknown"), "clear_unscored": count(clear, "unscored"),
        "reference_unclear_count": len(unclear), "unclear_called_clear": count(unclear, "supported"),
        "unclear_withheld": count(unclear, "unknown"), "unclear_unscored": count(unclear, "unscored")}


def _policy_predictions(entry, raw, reports):
    runtime = masks._runtime(entry, raw, reports)
    resolved = geometry.resolve_distractor_conflicts(raw, target_track_id=1)
    corrected, audit = masks._filter(runtime, resolved, entry, raw)
    policy = masks._policy(entry, corrected, CONFIG)
    by_index = {f["frame_index"]: f for f in policy["frames"]} if policy else {}
    predictions = []
    for cp in entry["checkpoints"]:
        index = cp["frame_index"]
        row = by_index.get(index)
        status = "unscored" if raw["status"] != "completed" or not row or reports[index]["status"] == "unscored" else row["status"]
        predictions.append({"clip_id": entry["clip_id"], "frame_index": index, "status": status,
                            "box": row["box"] if row and status == "supported" else None})
    return predictions, policy, audit


def replay(root: str | Path, frozen: str | Path):
    root = Path(root).resolve()
    path = original._local(root, frozen)
    plan = original._read(path)
    old, fresh = verify_plan(root, plan)
    output = path.parent
    marker = original._read(output / "run-started.json")
    if marker.get("freeze_hash") != plan["freeze_hash"] or marker.get("automatic_retry") is not False:
        raise ValidationError("Visibility run lacks its matching one-execution receipt")
    new_attempts, new_cost, failed_operations = 0, 0.0, 0
    cohorts = {}
    for cohort, sources in zip(COHORTS, (old, fresh)):
        references = [r for entry in sources["clips"] for r in original._reference(entry)]
        predictions = {method: [] for method in METHODS}
        evidence = []
        cohort_attempts, cohort_cost = 0, 0.0
        for entry in sources["clips"]:
            name = entry["clip_id"]
            parent = original._local(root, plan["original_frozen_file"]).parent if cohort == COHORTS[0] else output
            raw = original._read(parent / f"{name}-video.json")
            if raw.get("prepared") != entry["prepared"]:
                raise ValidationError("Tracking response is not bound to the frozen video preparation")
            tracking = public_tracking_summary(raw)
            if cohort == COHORTS[1]:
                cohort_attempts += len(raw["attempts"])
                cohort_cost += tracking["cost"]["estimated_usd"] or 0.0
                failed_operations += raw["status"] != "completed"
            reports, summaries = {}, {}
            for version, module in (("v1", old_guard), ("v2", guard)):
                reports[version], summaries[version] = {}, []
                for cp in entry["checkpoints"]:
                    index = cp["frame_index"]
                    source_file = (parent / f"{name}-guard-{index}.json") if cohort == COHORTS[0] and version == "v1" else output / f"{name}-{version}-{index}.json"
                    checked = original._read(source_file)
                    reports[version][index] = checked
                    summary, attempts, cost = _verify_guard(checked, entry, raw, index, module)
                    summaries[version].append(summary if summary is not None else {"frame_index": index, **_no_candidate()})
                    if cohort == COHORTS[1] or version == "v2":
                        cohort_attempts += attempts
                        cohort_cost += cost
                        failed_operations += checked["status"] == "unscored"
                    predictions[f"visibility_{version}"].append(_prediction(entry, raw, checked, index, visibility_only=True))
                    predictions[f"candidate_guard_{version}"].append(_prediction(entry, raw, checked, index))
                scored_policy, policy, audit = _policy_predictions(entry, raw, reports[version])
                predictions[f"mask_policy_{version}"].extend(scored_policy)
                summaries[version + "_policy"] = policy
                if version == "v1":
                    summaries["geometry_audit"] = audit
            for cp in entry["checkpoints"]:
                predictions["sam_presence"].append(_prediction(entry, raw, {}, cp["frame_index"], presence_only=True))
            evidence.append({"clip_id": name, "source_attribution": entry["source_attribution"],
                "original_sha256": entry["original_sha256"], "derivation": entry["derivation"],
                "tracking": tracking, "assessments": summaries})
        methods = {method: score_frames(references, predictions[method], iou_threshold=CONFIG["localization_iou_threshold"])
                   for method in METHODS}
        cohorts[cohort] = {"source_videos": 2, "sampled_source_frames": sum(len(e["frames"]) for e in sources["clips"]),
            "labeled_frames": len(references), "references": references, "methods": methods,
            "visibility_only_diagnostics": {version: visibility_counts(references, predictions[f"visibility_{version}"])
                                            for version in ("v1", "v2")},
            "localization_sensitivity": {str(t): {m: score_frames(references, predictions[m], iou_threshold=t)["metrics"]
                for m in METHODS} for t in CONFIG["localization_sensitivity_thresholds"]},
            "evidence": evidence, "new_provider_attempts": cohort_attempts, "estimated_incremental_usd": round(cohort_cost, 6)}
        new_attempts += cohort_attempts
        new_cost += cohort_cost
    accounting = AttemptJournal(output / "attempts.sqlite3", request_limit=CONFIG["max_requests"]).summary()
    if accounting["total_reserved_attempts"] != new_attempts:
        raise ValidationError("Visibility request journal and retained calls disagree")
    masks._unchanged(root, plan["input_records"])
    if plan["implementation_sha256"] != _implementations():
        raise ValidationError("Visibility implementation changed during replay")
    return {"experiment_version": VERSION, "freeze_hash": plan["freeze_hash"], "config": deepcopy(CONFIG),
        "implementation_sha256": plan["implementation_sha256"], "original_freeze_hash": plan["original_freeze_hash"],
        "original_public_sha256": plan["original_public_sha256"],
        "fresh_source_preparation_hash": plan["fresh_source_preparation_hash"],
        "annotation_origin": "assistant_visual_review_frozen_before_inference_not_dataset_ground_truth",
        "cohorts": cohorts, "new_provider_attempts": new_attempts, "estimated_incremental_usd": round(new_cost, 6),
        "failed_operation_count": failed_operations, "accounting": accounting,
        "changes_live_placement_acceptance": False, "physical_action_authorized": False,
        "limitations": list(LIMITATIONS)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "run", "replay", "public-plan"))
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--original-frozen", type=Path)
    parser.add_argument("--fresh-frozen", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--frozen", type=Path)
    args = parser.parse_args(argv)
    if args.action == "freeze":
        result = freeze(args.root, args.original_frozen, args.fresh_frozen, args.output)
        print(result["freeze_hash"])
    elif args.action == "public-plan":
        result = public_plan(args.root, args.frozen)
        original._write(args.output, result)
        print(result["freeze_hash"])
    else:
        result = run(args.root, args.frozen) if args.action == "run" else replay(args.root, args.frozen)
        print({"freeze_hash": result["freeze_hash"], "new_provider_attempts": result["new_provider_attempts"],
               "estimated_incremental_usd": result["estimated_incremental_usd"]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
