"""Frozen boundary-contact and confirmation-selection ablations on real videos.

Historical outputs remain immutable. The two guard versions and two policies
are crossed so a temporal gain cannot conceal a contact classification error.
References never enter inference or policy inputs.
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
from . import confirmation_policy as current
from . import contact_guard as contact
from . import tracking_experiment as original
from . import tracking_geometry as geometry
from . import tracking_mask_replay as masks
from . import visibility_experiment as previous
from . import visibility_guard as native
from .tracking_evaluation import score_frames
from .video_tracking import segment_video, public_tracking_summary

VERSION = "frozen_boundary_contact_confirmation_ablation_v1"
COHORTS = ("exposed_episode_2", "new_episode_3")
METHODS = ("sam_presence", "visibility_v2", "visibility_v3", "candidate_v2", "candidate_v3",
           "v2_greedy", "v2_current", "v3_greedy", "v3_current")
CONFIG = {"max_requests": 50, "estimated_allocation_usd": 3.0, "automatic_retry": False,
    "reasoning_effort": "low", "max_output_tokens": 4096, "min_confidence": 0.8,
    "min_stable_views": 2, "min_spacing_ms": 500, "localization_iou_threshold": 0.3,
    "localization_sensitivity_thresholds": [0.2, 0.3, 0.5],
    "candidate_origin": "same_actual_sam_video_candidates_for_both_guards",
    "ablation": "two_guards_crossed_with_two_policies_no_reference_labels_in_runtime",
    "sampling": "eight_preselected_checkpoints_per_clip_before_new_inference"}
FILES = tuple(sorted(set(previous.FILES + ("contact_guard.py", "confirmation_policy.py", "contact_experiment.py"))))
LIMITATIONS = [
    "Episode 2 is exposed development; episode 3 is a new recording in the same exposed tasks, cameras and dataset families.",
    "Initial target/distractor seeds and scoring references are assistant-authored visual reviews, not official visibility ground truth.",
    "White-on-white near-contact labels are annotation-sensitive. Unresolved boundary ownership is UNKNOWN, not proof of physical occlusion.",
    "V3 changes the rubric and adds an exact nearest-neighbor enlargement of the same crop pixels; it adds no source detail and does not isolate either intervention.",
    "Current-view confirmation changes selection of qualifying history, not the frozen spacing, count, confidence or hard evidence gates. Every counted view also requires positive source-bound semantics.",
    "The first clear source view cannot satisfy a two-view requirement. No future frame or label is used to support it retrospectively.",
    "Same-model assessments and mask geometry do not independently prove identity, full-occlusion recovery, continuous visibility or physical action.",
    "Incomplete calls are unscored, no automatic retries occur, and historical outputs, reference labels and live placement acceptance are unchanged.",
    "Estimated allocation is advisory; persistent request-count admission is not a provider-enforced dollar cap.",
]


def _implementations():
    return {name: original._sha(Path(__file__).with_name(name)) for name in FILES}


def _snapshot_previous(root, path, plan):
    records = {r["file"]: deepcopy(r) for r in plan["input_records"]}
    def add(file, role):
        record = masks._file_record(root, original._local(root, file), role)
        if record["file"] not in records:
            records[record["file"]] = record
    add(path, "previous_visibility_freeze")
    for file in path.parent.rglob("*"):
        if file.is_file() and file.suffix in {".json", ".jpg", ".png", ".sqlite3"}:
            add(file, "previous_visibility_retained_evidence")
    if len(records) > masks.MAX_INPUT_FILES:
        raise ValidationError("Historical visibility evidence exceeds bounded input inventory")
    return [records[key] for key in sorted(records)]


def freeze(root: str | Path, previous_frozen: str | Path, fresh_frozen: str | Path, output: str | Path):
    root = Path(root).resolve()
    prior_path, fresh_path = original._local(root, previous_frozen), original._local(root, fresh_frozen)
    prior_plan = original._read(prior_path)
    records = _snapshot_previous(root, prior_path, prior_plan)
    prior_result = previous.replay(root, prior_path)
    masks._unchanged(root, records)
    if original._digest(original._read(prior_path.parent / "public-result.json")) != original._digest(prior_result):
        raise ValidationError("Previous published visibility result differs from replay")
    old_source_path = original._local(root, prior_plan["fresh_frozen_file"])
    old_source, fresh_source = original._read(old_source_path), original._read(fresh_path)
    original.verify_plan(root, old_source)
    original.verify_plan(root, fresh_source)
    families = {e["source_attribution"]["dataset"]: e for e in old_source["clips"]}
    if len(families) != 2 or {e["source_attribution"]["dataset"] for e in fresh_source["clips"]} != set(families):
        raise ValidationError("Freeze two new recordings in the declared source families")
    for entry in fresh_source["clips"]:
        old = families[entry["source_attribution"]["dataset"]]
        if (entry["source_attribution"]["episode"] != 3 or old["source_attribution"]["episode"] != 2
                or entry["contract"] != old["contract"] or entry["original_sha256"] == old["original_sha256"]
                or entry["prepared"]["source"]["sha256"] == old["prepared"]["source"]["sha256"]):
            raise ValidationError("Compare episode 2 with new episode 3 under the same visual contracts")
    if {e["clip_id"] for e in old_source["clips"]} & {e["clip_id"] for e in fresh_source["clips"]}:
        raise ValidationError("Old and new cohort clip identifiers must differ")
    destination = original._local(root, output)
    if any(destination == p.parent or destination.is_relative_to(p.parent) for p in (prior_path, fresh_path)):
        raise ValidationError("Keep new results outside historical and source-only directories")
    plan = {"experiment_version": VERSION, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "inference_started": False, "config": deepcopy(CONFIG), "methods": list(METHODS),
        "implementation_sha256": _implementations(), "contact_config": deepcopy(contact.CONFIG),
        "native_config": deepcopy(native.CONFIG), "current_policy_version": current.CONFIRMATION_POLICY_VERSION,
        "previous_frozen_file": prior_path.relative_to(root).as_posix(),
        "previous_freeze_hash": prior_plan["freeze_hash"], "previous_public_sha256": original._digest(prior_result),
        "old_source_frozen_file": old_source_path.relative_to(root).as_posix(),
        "old_source_freeze_hash": old_source["freeze_hash"],
        "fresh_frozen_file": fresh_path.relative_to(root).as_posix(), "fresh_source_freeze_hash": fresh_source["freeze_hash"],
        "input_records": records + previous._fresh_records(root, fresh_path, fresh_source), "limitations": list(LIMITATIONS)}
    plan["freeze_hash"] = original._digest(plan)
    destination.mkdir(parents=True, exist_ok=False, mode=0o700)
    original._write(destination / "freeze.json", plan)
    return plan


def verify_plan(root: Path, plan: Mapping[str, Any], *, files=True):
    if (plan["experiment_version"] != VERSION or plan["inference_started"] is not False
            or plan["config"] != CONFIG or plan["methods"] != list(METHODS) or plan["limitations"] != LIMITATIONS
            or plan["implementation_sha256"] != _implementations() or plan["contact_config"] != contact.CONFIG
            or plan["native_config"] != native.CONFIG or plan["current_policy_version"] != current.CONFIRMATION_POLICY_VERSION
            or plan["freeze_hash"] != original._digest({k: v for k, v in plan.items() if k != "freeze_hash"})):
        raise ValidationError("Contact experiment implementation, settings or receipt changed")
    prior_path = original._local(root, plan["previous_frozen_file"])
    prior = original._read(prior_path)
    old = original._read(original._local(root, plan["old_source_frozen_file"]))
    fresh = original._read(original._local(root, plan["fresh_frozen_file"]))
    if (prior["freeze_hash"] != plan["previous_freeze_hash"] or old["freeze_hash"] != plan["old_source_freeze_hash"]
            or fresh["freeze_hash"] != plan["fresh_source_freeze_hash"]):
        raise ValidationError("Contact source preparation differs from the frozen receipt")
    if files:
        masks._unchanged(root, plan["input_records"])
        original.verify_plan(root, old)
        original.verify_plan(root, fresh)
        # Validate the previous run at freeze/admission/replay, never infer again.
        if original._digest(previous.replay(root, prior_path)) != plan["previous_public_sha256"]:
            raise ValidationError("Historical visibility comparison changed")
        masks._unchanged(root, plan["input_records"])
    return old, fresh


def public_plan(root: str | Path, frozen: str | Path):
    root = Path(root).resolve()
    plan = original._read(original._local(root, frozen))
    old, fresh = verify_plan(root, plan)
    return {"experiment_version": VERSION, "freeze_hash": plan["freeze_hash"], "inference_started": False,
        "config": deepcopy(CONFIG), "methods": list(METHODS), "implementation_sha256": plan["implementation_sha256"],
        "previous_freeze_hash": plan["previous_freeze_hash"], "previous_public_sha256": plan["previous_public_sha256"],
        "input_inventory_sha256": original._digest(plan["input_records"]),
        "cohorts": {cohort: {"source_videos": len(source["clips"]), "references": [r for e in source["clips"] for r in original._reference(e)],
            "sources": [{"clip_id": e["clip_id"], "source_attribution": e["source_attribution"],
                         "original_sha256": e["original_sha256"], "derivation": e["derivation"]} for e in source["clips"]]}
            for cohort, source in zip(COHORTS, (old, fresh))},
        "limitations": list(LIMITATIONS), "changes_live_placement_acceptance": False}


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
    def client(label, timeout=90):
        return original.RetainingClient.from_env(journal=journal,
            operation_id=original._digest([plan["freeze_hash"], label])[:32],
            guard=lambda: verify_plan(root, plan, files=False), max_requests=1, retries=0, timeout=timeout)
    historical = original._local(root, plan["previous_frozen_file"]).parent
    for cohort, source in zip(COHORTS, (old, fresh)):
        for entry in source["clips"]:
            name = entry["clip_id"]
            if cohort == COHORTS[0]:
                raw = original._read(historical / f"{name}-video.json")
            else:
                raw = segment_video(client(name + "-video", 180), entry["prepared"])
                original._write(output / f"{name}-video.json", raw)
            for cp in entry["checkpoints"]:
                index = cp["frame_index"]
                modules = (("v3", contact),) if cohort == COHORTS[0] else (("v2", native), ("v3", contact))
                for version, module in modules:
                    label = f"{name}-{version}-{index}"
                    checked = previous._assess(lambda: client(label), entry, raw, index, module, output / label)
                    original._write(output / f"{label}.json", checked)
    result = replay(root, path)
    original._write(output / "public-result.json", result)
    return result


def _verify_contact(checked, entry, raw, index):
    tracked = next((f for f in raw["frames"] if f["frame_index"] == index), None)
    candidate = original._candidate(tracked) if tracked else None
    if candidate is None:
        if checked != previous._no_candidate():
            raise ValidationError("Contact evidence requires its actual SAM candidate")
        return None, 0, 0.0
    prepared = checked["prepared"]
    source = entry["prepared"]["source"]
    if (prepared["contract"] != entry["contract"] or prepared["source_sha256"] != source["sha256"]
            or prepared["frame_index"] != index or prepared["track_id"] != 1
            or prepared["frame_sha256"] != entry["frames"][index]["sha256"]
            or prepared["source_file"] != entry["frames"][index]["file"]
            or prepared["candidate_box"] != original._pixels(candidate["box"], source["width"], source["height"])):
        raise ValidationError("Contact assessment differs from the exact runtime candidate/frame")
    summary = contact.public_summary(checked)
    response = checked.get("response")
    raw_response = response["response"] if response is not None else checked["error"]["response"]
    original._verified_usage(checked, [raw_response], limit=1)
    return summary, checked["usage"]["request_attempts"], checked["usage"]["estimated_usd"]


def _policy(entry, runtime, evaluator):
    if not runtime:
        return None
    source = entry["prepared"]["source"]
    return evaluator(runtime, target_id=entry["contract"]["target_id"], seed={
        "target_id": entry["contract"]["target_id"], "track_id": 1, "frame_index": 0,
        "source_sha256": source["sha256"], "frame_sha256": entry["frames"][0]["sha256"],
        "review_sha256": entry["prepared"]["prepared_hash"],
        "reviewed_distractor_track_ids": [s["track_id"] for s in entry["seeds"] if s["track_id"] != 1]},
        **{k: CONFIG[k] for k in ("min_confidence", "min_stable_views", "min_spacing_ms")})


def _policy_predictions(entry, raw, reports, evaluator, resolved):
    runtime = masks._runtime(entry, raw, reports)
    corrected, audit = masks._filter(runtime, resolved, entry, raw)
    policy = _policy(entry, corrected, evaluator)
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
        raise ValidationError("Contact run lacks its matching exclusive admission receipt")
    historical = original._local(root, plan["previous_frozen_file"]).parent
    old_public = original._read(historical / "public-result.json")
    if original._digest(old_public) != plan["previous_public_sha256"]:
        raise ValidationError("Previous public result changed")
    cohorts, attempts, estimated, failed = {}, 0, 0.0, 0
    for cohort, source in zip(COHORTS, (old, fresh)):
        references = [r for entry in source["clips"] for r in original._reference(entry)]
        predictions = {method: [] for method in METHODS}
        evidence, count, cost = [], 0, 0.0
        for entry in source["clips"]:
            name = entry["clip_id"]
            directory = historical if cohort == COHORTS[0] else output
            raw = original._read(directory / f"{name}-video.json")
            if raw.get("prepared") != entry["prepared"]:
                raise ValidationError("Contact video response differs from its frozen preparation")
            tracking = public_tracking_summary(raw)
            resolved = geometry.resolve_distractor_conflicts(raw, target_track_id=1)
            if cohort == COHORTS[1]:
                count += len(raw["attempts"])
                cost += tracking["cost"]["estimated_usd"] or 0.0
                failed += raw["status"] != "completed"
            assessments = {}
            for version, module in (("v2", native), ("v3", contact)):
                reports, summaries = {}, []
                for cp in entry["checkpoints"]:
                    index = cp["frame_index"]
                    report_dir = historical if cohort == COHORTS[0] and version == "v2" else output
                    checked = original._read(report_dir / f"{name}-{version}-{index}.json")
                    reports[index] = checked
                    summary, used, charge = (_verify_contact(checked, entry, raw, index) if version == "v3"
                        else previous._verify_guard(checked, entry, raw, index, native))
                    summaries.append(summary if summary is not None else {"frame_index": index, **previous._no_candidate()})
                    if cohort == COHORTS[1] or version == "v3":
                        count += used
                        cost += charge
                        failed += checked["status"] == "unscored"
                    predictions[f"visibility_{version}"].append(previous._prediction(entry, raw, checked, index, visibility_only=True))
                    predictions[f"candidate_{version}"].append(previous._prediction(entry, raw, checked, index))
                assessments[version] = summaries
                for label, evaluator in (("greedy", original.evaluate_tracking), ("current", current.evaluate_tracking)):
                    rows, policy, audit = _policy_predictions(entry, raw, reports, evaluator, resolved)
                    predictions[f"{version}_{label}"].extend(rows)
                    assessments[f"{version}_{label}_policy"] = policy
                    if version == "v2" and label == "greedy":
                        assessments["geometry_audit"] = audit
            for cp in entry["checkpoints"]:
                predictions["sam_presence"].append(previous._prediction(entry, raw, {}, cp["frame_index"], presence_only=True))
            evidence.append({"clip_id": name, "source_attribution": entry["source_attribution"],
                "original_sha256": entry["original_sha256"], "derivation": entry["derivation"],
                "tracking": tracking, "assessments": assessments})
        methods = {name: score_frames(references, predictions[name], iou_threshold=CONFIG["localization_iou_threshold"])
                   for name in METHODS}
        if cohort == COHORTS[0]:
            expected = old_public["cohorts"]["new_recordings_same_families"]
            for key, previous_key in (("visibility_v2", "visibility_v2"), ("candidate_v2", "candidate_guard_v2"),
                                      ("v2_greedy", "mask_policy_v2"), ("sam_presence", "sam_presence")):
                if methods[key] != expected["methods"][previous_key]:
                    raise ValidationError("Reused baseline differs from the historical published score")
        cohorts[cohort] = {"source_videos": len(source["clips"]), "sampled_source_frames": sum(len(e["frames"]) for e in source["clips"]),
            "labeled_frames": len(references), "references": references, "methods": methods,
            "visibility_only_diagnostics": {version: previous.visibility_counts(references, predictions[f"visibility_{version}"])
                                            for version in ("v2", "v3")},
            "localization_sensitivity": {str(t): {name: score_frames(references, predictions[name], iou_threshold=t)["metrics"]
                for name in METHODS} for t in CONFIG["localization_sensitivity_thresholds"]},
            "evidence": evidence, "new_provider_attempts": count, "estimated_incremental_usd": round(cost, 6)}
        attempts += count
        estimated += cost
    accounting = AttemptJournal(output / "attempts.sqlite3", request_limit=CONFIG["max_requests"]).summary()
    if accounting["total_reserved_attempts"] != attempts:
        raise ValidationError("Contact attempt journal differs from retained provider calls")
    masks._unchanged(root, plan["input_records"])
    if _implementations() != plan["implementation_sha256"]:
        raise ValidationError("Contact implementation changed during replay")
    return {"experiment_version": VERSION, "freeze_hash": plan["freeze_hash"], "config": deepcopy(CONFIG),
        "implementation_sha256": plan["implementation_sha256"], "previous_freeze_hash": plan["previous_freeze_hash"],
        "previous_public_sha256": plan["previous_public_sha256"], "cohorts": cohorts,
        "new_provider_attempts": attempts, "estimated_incremental_usd": round(estimated, 6),
        "failed_operation_count": failed, "accounting": accounting,
        "annotation_origin": "assistant_visual_review_frozen_before_inference_not_dataset_ground_truth",
        "changes_live_placement_acceptance": False, "physical_action_authorized": False, "limitations": list(LIMITATIONS)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "public-plan", "run", "replay"))
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--previous-frozen", type=Path)
    parser.add_argument("--fresh-frozen", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--frozen", type=Path)
    args = parser.parse_args(argv)
    if args.action == "freeze":
        result = freeze(args.root, args.previous_frozen, args.fresh_frozen, args.output)
        print(result["freeze_hash"])
    elif args.action == "public-plan":
        result = public_plan(args.root, args.frozen)
        original._write(args.output, result)
        print(result["freeze_hash"])
    else:
        result = run(args.root, args.frozen) if args.action == "run" else replay(args.root, args.frozen)
        print({k: result[k] for k in ("freeze_hash", "new_provider_attempts", "estimated_incremental_usd")})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
