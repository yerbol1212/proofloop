"""Prospective gripper-mask ablation over an immutable contact experiment.

Four additional SAM requests track two initially reviewed grippers. The target
tracks, Spark judgments, scoring references, thresholds and historical failures
remain unchanged. Geometric separation can remove no semantic hard gate; a
nearby, missing or unresolvable gripper withholds support without asserting a
physical cause.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .core import ValidationError
from .inference_journal import AttemptJournal
from .meta import load_env_file
from . import contact_experiment as baseline
from . import gripper_geometry as geometry
from . import tracking_experiment as original
from . import tracking_geometry as distractors
from . import tracking_mask_replay as masks
from .video_tracking import prepare_clip, verify_prepared, segment_video, public_tracking_summary


VERSION = "frozen_reviewed_gripper_geometry_ablation_v1"
COHORTS = baseline.COHORTS
METHODS = ("v2_geometry_greedy", "v2_geometry_current", "v3_geometry_greedy", "v3_geometry_current")
CONFIG = {"max_requests": 4, "estimated_allocation_usd": 3.0, "automatic_retry": False,
    "new_provider": "SAM_video_only_no_new_Spark", "seed_frame_index": 0,
    "gripper_track_ids": [3, 4], "target_track_id": 1, "min_confidence": .8,
    "min_stable_views": 2, "min_spacing_ms": 500, "localization_iou_threshold": .3,
    "localization_sensitivity_thresholds": [.2, .3, .5],
    "separation_rule": "only_all_reviewed_grippers_separated_can_qualify",
    "near_contact_means": "unknown_visibility_not_asserted_physical_occlusion",
    "guard_inputs": "unchanged_retained_v2_v3_no_new_semantics_or_reference_boxes"}
FILES = tuple(sorted(set(baseline.FILES + ("gripper_geometry.py", "gripper_experiment.py"))))
LIMITATIONS = [
    "All four recordings were exposed in the completed contact experiment; this is a development ablation, not a new unseen cohort.",
    "The two additional gripper seeds per clip are assistant-reviewed frame-zero boxes. Later labels and boxes never enter inference or the geometry veto.",
    "SAM target and gripper masks can make correlated mistakes. Mask separation is not independent proof of identity, complete visibility or full-disappearance recovery.",
    "The fixed source-pixel separation margin is conservative. Near contact or unavailable evidence means UNKNOWN, not verified physical occlusion.",
    "The geometric veto only removes eligibility. Original confidence, mask, semantic identity, visibility, distinct-frame, count and spacing gates remain required.",
    "All original provider failures, incomplete calls, baselines and costs are retained. No historical call is retried or reclassified as successful abstention.",
    "Only four new SAM video requests are admitted, with no automatic retries or new Spark calls. Estimated dollars are advisory, not a provider-enforced cap.",
    "This experiment changes neither live placement acceptance nor physical-action authorization and observes only sampled source views.",
]


def _implementations():
    return {name: original._sha(Path(__file__).with_name(name)) for name in FILES}


def _sources(root: Path, plan: Mapping[str, Any]):
    return tuple(original._read(original._local(root, plan[key]))
                 for key in ("old_source_frozen_file", "fresh_frozen_file"))


def _annotation(value, clips):
    if (type(value) is not dict or set(value) != {"review_origin", "reviewed_before_inference", "clips"}
            or value["review_origin"] not in {"assistant_under_user_authorization", "local_operator"}
            or value["reviewed_before_inference"] is not True or type(value["clips"]) is not list
            or len(value["clips"]) != 4 or len(clips) != 4):
        raise ValidationError("Review exactly two initial gripper seeds on each of four frozen clips")
    by_id = {}
    for row in value["clips"]:
        if (type(row) is not dict or set(row) != {"clip_id", "seeds"} or type(row["clip_id"]) is not str
                or row["clip_id"] in by_id or type(row["seeds"]) is not list or len(row["seeds"]) != 2):
            raise ValidationError("Each gripper review must identify one exact frozen clip")
        if (any(type(seed) is not dict or set(seed) != {"object_id", "track_id", "frame_index", "box"}
                or type(seed["track_id"]) is not int or type(seed["frame_index"]) is not int
                or seed["frame_index"] != 0 for seed in row["seeds"])
                or {seed["track_id"] for seed in row["seeds"]} != {3, 4}):
            raise ValidationError("Gripper seeds must use only distinct track IDs 3 and 4 on frame zero")
        by_id[row["clip_id"]] = deepcopy(row["seeds"])
    if set(by_id) != {entry["clip_id"] for entry in clips}:
        raise ValidationError("Gripper reviews differ from the four baseline source clips")
    return by_id


def _prepared(entry, seeds, origin):
    prepared = prepare_clip(entry["prepared"]["clip_file"], seeds, review_origin=origin,
        review_note="Both frame-zero robot gripper boxes reviewed before this separate geometry inference.",
        propagation_direction="both")
    if prepared["source"] != entry["prepared"]["source"]:
        raise ValidationError("Gripper tracking must submit the unchanged baseline clip and exact source timeline")
    return prepared


def freeze(root: str | Path, contact_frozen: str | Path, annotations: str | Path, output: str | Path):
    root = Path(root).resolve()
    historical_path = original._local(root, contact_frozen)
    historical_plan = original._read(historical_path)
    records = baseline._snapshot_previous(root, historical_path, historical_plan)
    historical_public = baseline.replay(root, historical_path)
    masks._unchanged(root, records)
    if original._digest(original._read(historical_path.parent / "public-result.json")) != original._digest(historical_public):
        raise ValidationError("Completed contact baseline must match its published retained replay")
    sources = _sources(root, historical_plan)
    clips = [entry for source in sources for entry in source["clips"]]
    annotation_path = original._local(root, annotations)
    reviewed = original._read(annotation_path)
    by_id = _annotation(reviewed, clips)
    preparations = {entry["clip_id"]: _prepared(entry, by_id[entry["clip_id"]], reviewed["review_origin"])
                    for entry in clips}
    records_by_path = {record["file"]: record for record in records}
    annotation_record = masks._file_record(root, annotation_path, "initial_gripper_seed_review")
    records_by_path[annotation_record["file"]] = annotation_record
    if len(records_by_path) > masks.MAX_INPUT_FILES:
        raise ValidationError("Gripper baseline inventory exceeds its bounded file limit")
    destination = original._local(root, output)
    protected = [historical_path.parent, *(original._local(root, historical_plan[key]).parent
                 for key in ("old_source_frozen_file", "fresh_frozen_file"))]
    if any(destination == directory or destination.is_relative_to(directory) for directory in protected):
        raise ValidationError("New gripper outputs must remain outside immutable baseline/source directories")
    plan = {"experiment_version": VERSION, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "inference_started": False, "config": deepcopy(CONFIG), "methods": list(METHODS),
        "implementation_sha256": _implementations(), "geometry_config": deepcopy(geometry.CONFIG),
        "geometry_version": geometry.VERSION, "contact_frozen_file": historical_path.relative_to(root).as_posix(),
        "contact_freeze_hash": historical_plan["freeze_hash"],
        "contact_public_sha256": original._digest(historical_public),
        "annotations_file": annotation_path.relative_to(root).as_posix(), "annotations_sha256": original._sha(annotation_path),
        "input_records": [records_by_path[key] for key in sorted(records_by_path)],
        "gripper_preparations": preparations, "limitations": list(LIMITATIONS)}
    plan["freeze_hash"] = original._digest(plan)
    destination.mkdir(parents=True, exist_ok=False, mode=0o700)
    original._write(destination / "freeze.json", plan)
    return plan


def verify_plan(root: Path, plan: Mapping[str, Any], *, files=True):
    if (plan["experiment_version"] != VERSION or plan["inference_started"] is not False
            or plan["config"] != CONFIG or plan["methods"] != list(METHODS) or plan["limitations"] != LIMITATIONS
            or plan["implementation_sha256"] != _implementations() or plan["geometry_config"] != geometry.CONFIG
            or plan["geometry_version"] != geometry.VERSION
            or plan["freeze_hash"] != original._digest({k: v for k, v in plan.items() if k != "freeze_hash"})):
        raise ValidationError("Gripper experiment code, configuration or prospective receipt changed")
    historical_path = original._local(root, plan["contact_frozen_file"])
    historical_plan = original._read(historical_path)
    if historical_plan["freeze_hash"] != plan["contact_freeze_hash"]:
        raise ValidationError("Gripper baseline differs from its frozen contact experiment")
    sources = _sources(root, historical_plan)
    clips = [entry for source in sources for entry in source["clips"]]
    annotation_path = original._local(root, plan["annotations_file"])
    if original._sha(annotation_path) != plan["annotations_sha256"]:
        raise ValidationError("Initial gripper seed review changed")
    reviewed = original._read(annotation_path)
    by_id = _annotation(reviewed, clips)
    if set(plan["gripper_preparations"]) != set(by_id):
        raise ValidationError("Gripper preparation inventory differs from the reviewed baseline clips")
    for entry in clips:
        prepared = plan["gripper_preparations"][entry["clip_id"]]
        clean_seeds = [{key: seed[key] for key in ("object_id", "track_id", "frame_index", "box")}
                       for seed in prepared["seeds"]]
        if (clean_seeds != sorted(by_id[entry["clip_id"]], key=lambda seed: (seed["frame_index"], seed["track_id"]))
                or prepared["clip_file"] != entry["prepared"]["clip_file"]
                or prepared["source"] != entry["prepared"]["source"]):
            raise ValidationError("Gripper seeds or source differ from the exact reviewed initial clip")
        if files:
            verify_prepared(prepared)
    if files:
        masks._unchanged(root, plan["input_records"])
        public = baseline.replay(root, historical_path)
        if (original._digest(public) != plan["contact_public_sha256"]
                or original._digest(original._read(historical_path.parent / "public-result.json")) != plan["contact_public_sha256"]):
            raise ValidationError("Historical contact public result changed")
        masks._unchanged(root, plan["input_records"])
    return historical_plan, sources


def public_plan(root: str | Path, frozen: str | Path):
    root = Path(root).resolve()
    plan = original._read(original._local(root, frozen))
    _, sources = verify_plan(root, plan)
    return {"experiment_version": VERSION, "freeze_hash": plan["freeze_hash"], "inference_started": False,
        "config": deepcopy(CONFIG), "methods": list(METHODS), "implementation_sha256": plan["implementation_sha256"],
        "contact_freeze_hash": plan["contact_freeze_hash"], "contact_public_sha256": plan["contact_public_sha256"],
        "geometry_config": deepcopy(geometry.CONFIG), "geometry_version": geometry.VERSION,
        "initial_review_sha256": plan["annotations_sha256"], "input_inventory_sha256": original._digest(plan["input_records"]),
        "cohorts": {cohort: {"references": [row for entry in source["clips"] for row in original._reference(entry)],
            "sources": [{"clip_id": entry["clip_id"], "source_attribution": entry["source_attribution"],
                "source_sha256": entry["prepared"]["source"]["sha256"],
                "gripper_prepared_hash": plan["gripper_preparations"][entry["clip_id"]]["prepared_hash"],
                "initial_gripper_seeds": [{key: seed[key] for key in ("object_id", "track_id", "frame_index", "box")}
                    for seed in plan["gripper_preparations"][entry["clip_id"]]["seeds"]]}
                for entry in source["clips"]]} for cohort, source in zip(COHORTS, sources)},
        "limitations": list(LIMITATIONS), "changes_live_placement_acceptance": False}


def run(root: str | Path, frozen: str | Path):
    root = Path(root).resolve()
    path = original._local(root, frozen)
    plan = original._read(path)
    _, sources = verify_plan(root, plan)
    output = path.parent
    original._write(output / "run-started.json", {"freeze_hash": plan["freeze_hash"],
        "started_at_utc": datetime.now(timezone.utc).isoformat(), "automatic_retry": False})
    load_env_file(root / ".env.local")
    journal = AttemptJournal(output / "attempts.sqlite3", request_limit=CONFIG["max_requests"])
    for source in sources:
        for entry in source["clips"]:
            name = entry["clip_id"]
            client = original.RetainingClient.from_env(journal=journal,
                operation_id=original._digest([plan["freeze_hash"], name, "grippers"] )[:32],
                guard=lambda: verify_plan(root, plan, files=False), max_requests=1, retries=0, timeout=180)
            raw = segment_video(client, plan["gripper_preparations"][name])
            original._write(output / f"{name}-grippers.json", raw)
    result = replay(root, path)
    original._write(output / "public-result.json", result)
    return result


def _locations(root, historical_path, historical_plan, cohort):
    target_directory = (original._local(root, historical_plan["previous_frozen_file"]).parent
                        if cohort == COHORTS[0] else historical_path.parent)
    return target_directory


def _veto(entry, runtime, assessment):
    """Only reduce eligibility; never infer a cause or substitute target pixels."""
    source = entry["prepared"]["source"]
    rows = assessment.get("frames")
    if type(rows) is not list:
        raise ValidationError("Gripper geometry lacks its explicit frame inventory")
    by_index = {}
    for row in rows:
        if (type(row) is not dict or type(row.get("frame_index")) is not int
                or row["frame_index"] in by_index or row.get("source_sha256") != source["sha256"]
                or row.get("target_track_id") != 1 or row.get("relation") not in {"separated", "near_or_contact", "unavailable"}
                or row.get("separation_supported") is not (row["relation"] == "separated")):
            raise ValidationError("Gripper geometry differs from its source/frame/target binding")
        by_index[row["frame_index"]] = row
    if set(by_index) != {row["frame_index"] for row in runtime}:
        raise ValidationError("Gripper geometry and target runtime frame inventories differ")
    result = deepcopy(runtime)
    changes = []
    for frame in result:
        relation = by_index[frame["frame_index"]]["relation"]
        for candidate in frame["candidates"]:
            if relation != "separated":
                before = {key: deepcopy(candidate[key]) for key in ("fully_visible", "semantic")}
                candidate["fully_visible"] = False
                if candidate["semantic"] is not None:
                    candidate["semantic"]["decision"] = "unknown"
                    candidate["semantic"]["evidence_sha256"] = original._digest({
                        "original_guard_evidence_sha256": before["semantic"]["evidence_sha256"],
                        "geometry_veto": by_index[frame["frame_index"]]})
                changes.append({"frame_index": frame["frame_index"], "relation": relation,
                    "previous_visibility": before["fully_visible"], "eligibility_withheld": True})
    return result, changes


def _policy_predictions(entry, raw, reports, evaluator, resolved, assessment, *, geometry_completed):
    runtime = masks._runtime(entry, raw, reports)
    corrected, distractor_audit = masks._filter(runtime, resolved, entry, raw)
    vetoed, geometry_changes = _veto(entry, corrected, assessment)
    policy = baseline._policy(entry, vetoed, evaluator)
    indexed = {row["frame_index"]: row for row in policy["frames"]} if policy else {}
    predictions = []
    for cp in entry["checkpoints"]:
        index = cp["frame_index"]
        row = indexed.get(index)
        status = ("unscored" if not geometry_completed or raw["status"] != "completed" or row is None
                  or reports[index]["status"] == "unscored" else row["status"])
        predictions.append({"clip_id": entry["clip_id"], "frame_index": index, "status": status,
                            "box": row["box"] if row and status == "supported" else None})
    return predictions, policy, {"distractor_geometry": distractor_audit, "gripper_vetoes": geometry_changes}


def _candidate_predictions(entry, raw, reports, assessment, *, geometry_completed):
    rows = {row["frame_index"]: row for row in assessment["frames"]}
    predictions = []
    for checkpoint in entry["checkpoints"]:
        index = checkpoint["frame_index"]
        prediction = baseline.previous._prediction(entry, raw, reports[index], index)
        if not geometry_completed:
            prediction.update(status="unscored", box=None)
        elif prediction["status"] == "supported" and rows[index]["relation"] != "separated":
            prediction.update(status="unknown", box=None)
        predictions.append(prediction)
    return predictions


def replay(root: str | Path, frozen: str | Path):
    root = Path(root).resolve()
    path = original._local(root, frozen)
    plan = original._read(path)
    historical_plan, sources = verify_plan(root, plan)
    historical_path = original._local(root, plan["contact_frozen_file"])
    historical_public = original._read(historical_path.parent / "public-result.json")
    marker = original._read(path.parent / "run-started.json")
    if marker.get("freeze_hash") != plan["freeze_hash"] or marker.get("automatic_retry") is not False:
        raise ValidationError("Gripper run lacks matching exclusive admission")
    cohorts, attempts, cost, failed, uncertain_cost = {}, 0, 0.0, 0, False
    for cohort, source in zip(COHORTS, sources):
        predictions = {method: [] for method in METHODS}
        candidates = {version: [] for version in ("v2", "v3")}
        references = [row for entry in source["clips"] for row in original._reference(entry)]
        evidence, cohort_attempts, cohort_cost = [], 0, 0.0
        for entry in source["clips"]:
            name = entry["clip_id"]
            target_directory = _locations(root, historical_path, historical_plan, cohort)
            raw = original._read(target_directory / f"{name}-video.json")
            new = original._read(path.parent / f"{name}-grippers.json")
            if raw.get("prepared") != entry["prepared"] or new.get("prepared") != plan["gripper_preparations"][name]:
                raise ValidationError("Target or gripper response differs from its exact frozen preparation")
            tracking = public_tracking_summary(new)
            assessed = geometry.compare(raw, new, target_track_id=1, gripper_track_ids=(3, 4))
            resolved = distractors.resolve_distractor_conflicts(raw, target_track_id=1)
            cohort_attempts += len(new["attempts"])
            charge = tracking["cost"]["estimated_usd"]
            cohort_cost += charge or 0.0
            uncertain_cost = uncertain_cost or charge is None or tracking["cost"].get("unreported_charge_possible", False)
            failed += new["status"] != "completed"
            policies = {}
            for version in ("v2", "v3"):
                report_directory = target_directory if cohort == COHORTS[0] and version == "v2" else historical_path.parent
                reports = {cp["frame_index"]: original._read(report_directory / f"{name}-{version}-{cp['frame_index']}.json")
                           for cp in entry["checkpoints"]}
                # Full source/image/response verification is already required by the immutable baseline replay.
                for label, evaluator in (("greedy", original.evaluate_tracking), ("current", baseline.current.evaluate_tracking)):
                    rows, policy, audit = _policy_predictions(entry, raw, reports, evaluator, resolved, assessed,
                        geometry_completed=new["status"] == "completed")
                    method = f"{version}_geometry_{label}"
                    predictions[method].extend(rows)
                    policies[method] = {"policy": policy, "audit": audit}
                candidates[version].extend(_candidate_predictions(entry, raw, reports, assessed,
                    geometry_completed=new["status"] == "completed"))
            evidence.append({"clip_id": name, "source_attribution": deepcopy(entry["source_attribution"]),
                "tracking_source_sha256": entry["prepared"]["source"]["sha256"], "gripper_tracking": tracking,
                "gripper_geometry": assessed, "policies": policies})
        old_cohort = historical_public["cohorts"][cohort]
        methods = {method: baseline.score_frames(references, predictions[method], iou_threshold=.3) for method in METHODS}
        cohorts[cohort] = {"source_videos": len(source["clips"]), "references": references, "labeled_frames": len(references),
            "historical_methods": deepcopy(old_cohort["methods"]),
            "methods": {**deepcopy(old_cohort["methods"]), **methods},
            "candidate_geometry_diagnostics": {version: {
                "localization": baseline.score_frames(references, candidates[version], iou_threshold=.3),
                "visibility_counts": baseline.previous.visibility_counts(references, candidates[version])}
                for version in ("v2", "v3")},
            "localization_sensitivity": {str(threshold): {method: baseline.score_frames(
                references, predictions[method], iou_threshold=threshold)["metrics"] for method in METHODS}
                for threshold in CONFIG["localization_sensitivity_thresholds"]},
            "evidence": evidence, "new_provider_attempts": cohort_attempts,
            "estimated_incremental_usd": round(cohort_cost, 6)}
        attempts += cohort_attempts
        cost += cohort_cost
    accounting = AttemptJournal(path.parent / "attempts.sqlite3", request_limit=4).summary()
    if accounting["total_reserved_attempts"] != attempts:
        raise ValidationError("Gripper attempt journal differs from retained new video calls")
    masks._unchanged(root, plan["input_records"])
    if _implementations() != plan["implementation_sha256"]:
        raise ValidationError("Gripper experiment implementation changed during replay")
    return {"experiment_version": VERSION, "freeze_hash": plan["freeze_hash"], "config": deepcopy(CONFIG),
        "implementation_sha256": plan["implementation_sha256"], "geometry_config": deepcopy(geometry.CONFIG),
        "contact_freeze_hash": plan["contact_freeze_hash"], "contact_public_sha256": plan["contact_public_sha256"],
        "historical_contact_result": deepcopy(historical_public), "cohorts": cohorts,
        "new_provider_attempts": attempts, "estimated_incremental_usd": round(cost, 6),
        "unreported_new_charge_possible": bool(uncertain_cost), "new_failed_operation_count": failed,
        "historical_estimated_incremental_usd": historical_public["estimated_incremental_usd"],
        "accounting": accounting, "new_Spark_requests": 0,
        "changes_live_placement_acceptance": False, "physical_action_authorized": False,
        "limitations": list(LIMITATIONS)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "public-plan", "run", "replay"))
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--contact-frozen", type=Path)
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--frozen", type=Path)
    args = parser.parse_args(argv)
    if args.action == "freeze":
        result = freeze(args.root, args.contact_frozen, args.annotations, args.output)
        print(result["freeze_hash"])
    elif args.action == "public-plan":
        result = public_plan(args.root, args.frozen)
        original._write(args.output, result)
        print(result["freeze_hash"])
    else:
        result = run(args.root, args.frozen) if args.action == "run" else replay(args.root, args.frozen)
        print({key: result[key] for key in ("freeze_hash", "new_provider_attempts", "estimated_incremental_usd")})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
