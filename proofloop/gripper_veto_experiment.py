"""Post-inspection, zero-inference development retest of negative gripper risk.

This separate receipt preserves the failed strict gripper experiment. The new
adapter can only remove existing support. A far or unavailable proposal never
qualifies a target or supplies missing semantic/visibility evidence.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .core import ValidationError
from . import gripper_experiment as strict
from . import gripper_veto as warning
from . import tracking_experiment as original
from . import tracking_mask_replay as masks


VERSION = "cached_negative_gripper_veto_development_retest_v1"
EVALUATION_ROLE = "post_inspection_exposed_cached_development_retest"
TRUST_MODE = "warning_only_not_qualification"
COHORTS = strict.COHORTS
METHODS = ("v2_veto_greedy", "v2_veto_current", "v3_veto_greedy", "v3_veto_current")
CONFIG = {"added_provider_requests": 0, "automatic_retry": False, "trust_mode": TRUST_MODE,
    "target_track_id": 1, "gripper_track_ids": [3, 4], "source_pixel_margin": 3,
    "acceptance_confidence_threshold_changed": False, "guard_judgments_unchanged": True,
    "reference_labels_unchanged": True, "min_stable_views": 2, "min_spacing_ms": 500,
    "positive_support_rule": "subset_of_same_guard_same_policy_baseline_exact_source_frame_box_evidence",
    "negative_risk_resets_confirmation": True,
    "unavailable_geometry": "unexamined_no_positive_evidence_no_baseline_upgrade",
    "localization_iou_threshold": .3, "localization_sensitivity_thresholds": [.2, .3, .5]}
FILES = tuple(sorted(set(strict.FILES + ("gripper_veto.py", "gripper_veto_experiment.py"))))
LIMITATIONS = [
    "This approach was selected after inspecting the failed strict gripper gate and the same four exposed recordings. It is a cached development retest, not held-out evidence.",
    "Valid decoded gripper proposals may add negative near-mask risk regardless of their reported confidence. They never supply positive qualification or lower the target acceptance threshold.",
    "No-near-proposal, missing, weak or unavailable geometry does not establish absence, separation, complete visibility, identity or physical action.",
    "A near proposal can be wrong and over-withhold. A veto is uncertainty, not verified physical contact or occlusion.",
    "Every supported source frame/box remains in the corresponding original guard/policy's supported set; geometry cannot repair original UNKNOWN or unscored judgments.",
    "Risk resets confirmation history. Recomputed current-view chains must use only distinct, source-bound, positively assessed, risk-free views at the original spacing.",
    "Original failures, uncertain billing, classifications, annotations, methods and strict-gate losses remain preserved in their canonical public files and hash-bound score projections.",
    "No API call or automatic retry occurs. Live placement acceptance and physical-action authorization are unchanged.",
]


def _implementations():
    return {name: original._sha(Path(__file__).with_name(name)) for name in FILES}


def _public_file(root, value):
    candidate = Path(value)
    candidate = candidate if candidate.is_absolute() else root / candidate
    resolved = candidate.resolve()
    if (candidate.is_symlink() or not resolved.is_relative_to(root / "artifacts/feasibility")
            or resolved.suffix != ".json" or not resolved.is_file()
            or not 0 < resolved.stat().st_size <= masks.MAX_JSON_BYTES):
        raise ValidationError("Public strict-result reference must be a bounded public artifact JSON")
    return resolved


def freeze(root: str | Path, strict_frozen: str | Path, output: str | Path, *, baseline_public: str | Path | None = None):
    root = Path(root).resolve()
    source_path = original._local(root, strict_frozen)
    source_plan = original._read(source_path)
    records = strict.baseline._snapshot_previous(root, source_path, source_plan)
    public = strict.replay(root, source_path)
    masks._unchanged(root, records)
    source_public = source_path.parent / "public-result.json"
    if original._digest(original._read(source_public)) != original._digest(public):
        raise ValidationError("Strict gripper result must match its canonical published replay")
    reference = "sha256:" + original._sha(source_public)
    if baseline_public is not None:
        linked = _public_file(root, baseline_public)
        if (original._sha(linked) != original._sha(source_public)
                or original._digest(original._read(linked)) != original._digest(public)):
            raise ValidationError("Public strict-result reference must identify the unchanged canonical result")
        reference = linked.relative_to(root).as_posix()
    records_by_path = {record["file"]: record for record in records}
    if len(records_by_path) > masks.MAX_INPUT_FILES:
        raise ValidationError("Cached gripper veto input inventory exceeds its file bound")
    destination = original._local(root, output)
    contact_path = original._local(root, source_plan["contact_frozen_file"])
    contact_plan = original._read(contact_path)
    protected = [source_path.parent, contact_path.parent, *(original._local(root, contact_plan[key]).parent
                 for key in ("old_source_frozen_file", "fresh_frozen_file"))]
    if any(destination == directory or destination.is_relative_to(directory) for directory in protected):
        raise ValidationError("Keep cached veto results outside original experiment directories")
    plan = {"experiment_version": VERSION, "evaluation_role": EVALUATION_ROLE,
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "derivation_started": False,
        "selected_after_inspecting_strict_failure": True, "added_provider_requests": 0,
        "config": deepcopy(CONFIG), "methods": list(METHODS), "implementation_sha256": _implementations(),
        "warning_config": deepcopy(warning.CONFIG), "warning_version": warning.VERSION,
        "strict_frozen_file": source_path.relative_to(root).as_posix(), "strict_freeze_hash": source_plan["freeze_hash"],
        "strict_public_sha256": original._digest(public), "strict_public_file_sha256": original._sha(source_public),
        "strict_public_reference": reference, "input_records": [records_by_path[key] for key in sorted(records_by_path)],
        "limitations": list(LIMITATIONS)}
    plan["freeze_hash"] = original._digest(plan)
    destination.mkdir(parents=True, exist_ok=False, mode=0o700)
    original._write(destination / "freeze.json", plan)
    return plan


def verify_plan(root: Path, plan: Mapping[str, Any]):
    if (plan["experiment_version"] != VERSION or plan["evaluation_role"] != EVALUATION_ROLE
            or plan["derivation_started"] is not False or plan["selected_after_inspecting_strict_failure"] is not True
            or plan["added_provider_requests"] != 0 or plan["config"] != CONFIG or plan["methods"] != list(METHODS)
            or plan["implementation_sha256"] != _implementations() or plan["warning_config"] != warning.CONFIG
            or plan["warning_version"] != warning.VERSION or plan["limitations"] != LIMITATIONS
            or plan["freeze_hash"] != original._digest({k: v for k, v in plan.items() if k != "freeze_hash"})):
        raise ValidationError("Cached veto implementation, zero-call protocol or receipt changed")
    path = original._local(root, plan["strict_frozen_file"])
    source_plan = original._read(path)
    if source_plan["freeze_hash"] != plan["strict_freeze_hash"]:
        raise ValidationError("Strict source receipt changed")
    masks._unchanged(root, plan["input_records"])
    public = strict.replay(root, path)
    if (original._digest(public) != plan["strict_public_sha256"]
            or original._digest(original._read(path.parent / "public-result.json")) != plan["strict_public_sha256"]
            or original._sha(path.parent / "public-result.json") != plan["strict_public_file_sha256"]):
        raise ValidationError("Canonical strict gripper result changed")
    if (not plan["strict_public_reference"].startswith("sha256:") and
            original._sha(_public_file(root, plan["strict_public_reference"])) != plan["strict_public_file_sha256"]):
        raise ValidationError("Published strict gripper reference changed")
    if (plan["strict_public_reference"].startswith("sha256:") and
            plan["strict_public_reference"] != "sha256:" + plan["strict_public_file_sha256"]):
        raise ValidationError("Strict content-addressed public reference changed")
    masks._unchanged(root, plan["input_records"])
    contact_path = original._local(root, source_plan["contact_frozen_file"])
    contact_plan = original._read(contact_path)
    sources = strict._sources(root, contact_plan)
    return source_plan, public, contact_path, contact_plan, sources


def public_plan(root: str | Path, frozen: str | Path):
    root = Path(root).resolve()
    plan = original._read(original._local(root, frozen))
    _, _, _, _, sources = verify_plan(root, plan)
    return {"experiment_version": VERSION, "evaluation_role": EVALUATION_ROLE, "freeze_hash": plan["freeze_hash"],
        "derivation_started": False, "selected_after_inspecting_strict_failure": True, "added_provider_requests": 0,
        "trust_mode": TRUST_MODE, "config": deepcopy(CONFIG), "methods": list(METHODS),
        "implementation_sha256": plan["implementation_sha256"], "warning_config": deepcopy(warning.CONFIG),
        "strict_freeze_hash": plan["strict_freeze_hash"], "strict_public_sha256": plan["strict_public_sha256"],
        "strict_public_file_sha256": plan["strict_public_file_sha256"], "strict_public_reference": plan["strict_public_reference"],
        "input_inventory_sha256": original._digest(plan["input_records"]),
        "cohorts": {cohort: {"sources": [{"clip_id": entry["clip_id"], "source_attribution": entry["source_attribution"],
            "source_sha256": entry["prepared"]["source"]["sha256"]} for entry in source["clips"]],
            "references": [row for entry in source["clips"] for row in original._reference(entry)]}
            for cohort, source in zip(COHORTS, sources)}, "limitations": list(LIMITATIONS),
        "changes_live_placement_acceptance": False, "physical_action_authorized": False}


def _warning_rows(entry, runtime, assessed):
    indexed = {}
    runtime_by_index = {frame["frame_index"]: frame for frame in runtime}
    if assessed.get("source_sha256") != entry["prepared"]["source"]["sha256"]:
        raise ValidationError("Warning source differs from its exact submitted clip")
    for row in assessed.get("frames", []):
        if (type(row) is not dict or type(row.get("frame_index")) is not int or row["frame_index"] in indexed
                or row.get("source_sha256") != entry["prepared"]["source"]["sha256"]
                or type(row.get("target_track_id")) is not int or row["target_track_id"] != 1
                or row["frame_index"] not in runtime_by_index
                or row.get("timestamp_ms") != runtime_by_index[row["frame_index"]]["timestamp_ms"]
                or row.get("warning") not in {"near_gripper_proposal", "no_near_proposal_observed", "geometry_unavailable"}
                or type(row.get("veto")) is not bool or type(row.get("additional_geometry_complete")) is not bool
                or row["veto"] is not (row["warning"] == "near_gripper_proposal")):
            raise ValidationError("Warning differs from its exact source frame, target or negative-only semantics")
        indexed[row["frame_index"]] = row
    if set(indexed) != {frame["frame_index"] for frame in runtime}:
        raise ValidationError("Warning and baseline runtime frame inventories differ")
    return indexed


def _runtime_veto(entry, runtime, assessed):
    warnings = _warning_rows(entry, runtime, assessed)
    result = deepcopy(runtime)
    changes = []
    for frame in result:
        current = warnings[frame["frame_index"]]
        if not current["veto"]:
            continue
        for selected in frame["candidates"]:
            selected["fully_visible"] = False
            if selected["semantic"] is not None:
                semantic = selected["semantic"]
                semantic["evidence_sha256"] = original._digest({"original_guard_sha256": semantic["evidence_sha256"],
                                                               "negative_warning": current})
                semantic["decision"] = "unknown"
            changes.append({"frame_index": frame["frame_index"], "warning": current["warning"],
                            "additional_risk_withheld": True, "physical_occlusion_asserted": False})
    return result, warnings, changes


def _validate_current_chain(row, runtime, warnings, *, target_id):
    """Require the published supported current chain to remain pixel-bound and risk-free."""
    if "confirmation_evidence" not in row:
        return  # The preserved legacy greedy evaluator does not expose chain records.
    chain = row["confirmation_evidence"]
    legacy = strict.baseline.current.legacy
    normalized = legacy._frames(runtime, target_id)
    indexed = {frame["frame_index"]: frame for frame in normalized}
    if (len(chain) < CONFIG["min_stable_views"] or not row.get("confirmation_ends_at_current")
            or chain[-1]["frame_index"] != row["frame_index"]
            or chain[-1]["timestamp_ms"] != row["timestamp_ms"]
            or chain[-1]["frame_sha256"] != row["frame_sha256"]
            or chain[-1]["semantic_evidence_sha256"] != row["semantic_evidence_sha256"]):
        raise ValidationError("Supported current view lacks its complete current-ending chain")
    # Validate the complete interval, not just the selected proof endpoints.
    # Nullable unassessed observations preserve continuity; original hard gates
    # and any selected identifier change invalidate a chain crossing that view.
    for frame in normalized:
        if not chain[0]["frame_index"] <= frame["frame_index"] <= row["frame_index"]:
            continue
        candidates = frame["candidates"]
        selected = candidates[0] if len(candidates) == 1 else None
        if len(candidates) > 1:
            supported = [item for item in candidates if item["semantic"] and item["semantic"]["decision"] == "supported"]
            excluded = [item for item in candidates if item["semantic"] and item["semantic"]["decision"] == "contradicted"]
            if len(supported) == 1 and len(excluded) == len(candidates) - 1:
                selected = supported[0]
        risk = warnings.get(frame["frame_index"])
        semantic = selected["semantic"] if selected else None
        if (risk is None or risk["veto"] or frame["source_sha256"] != row["source_sha256"]
                or selected is None or selected["track_id"] != row["track_id"]
                or frame["frame_index"] == row["frame_index"] and selected["box"] != row["box"]
                or selected["confidence"] is None or selected["confidence"] < strict.baseline.CONFIG["min_confidence"]
                or not selected["mask_supported"]
                or selected["box"][0] == 0 or selected["box"][1] == 0
                or selected["box"][2] == 1 or selected["box"][3] == 1
                or any(legacy._overlap(selected["box"], item["box"]) for item in frame["reviewed_distractors"])
                or selected["fully_visible"] is False or selected["identity_ambiguous"] is True
                or semantic is not None and semantic["decision"] != "supported"):
            raise ValidationError("Supported confirmation chain crosses an intervening hard gate, identifier change or risky view")
    seen, prior_time = set(), None
    for proof in chain:
        frame = indexed.get(proof["frame_index"])
        selected = frame["candidates"][0] if frame and len(frame["candidates"]) == 1 else None
        semantic = selected.get("semantic") if selected else None
        if (frame is None or selected is None or warnings[proof["frame_index"]]["veto"]
                or selected["fully_visible"] is not True or selected["identity_ambiguous"] is not False
                or selected["mask_supported"] is not True or selected["confidence"] is None
                or selected["confidence"] < strict.baseline.CONFIG["min_confidence"]
                or semantic is None or semantic["decision"] != "supported"
                or semantic["target_id"] != target_id or selected["object_id"] != target_id
                or proof["source_sha256"] != row["source_sha256"] or proof["track_id"] != row["track_id"]
                or proof["source_sha256"] != frame["source_sha256"] or proof["frame_sha256"] != frame["frame_sha256"]
                or proof["timestamp_ms"] != frame["timestamp_ms"] or proof["track_id"] != selected["track_id"]
                or proof["semantic_evidence_sha256"] != semantic["evidence_sha256"]
                or proof["frame_sha256"] in seen or prior_time is not None
                and proof["timestamp_ms"] - prior_time < CONFIG["min_spacing_ms"]):
            raise ValidationError("Supported confirmation chain contains unbound, repeated, unspaced or risky evidence")
        seen.add(proof["frame_sha256"])
        prior_time = proof["timestamp_ms"]


def _intersect_policy(candidate, unchanged, runtime, warnings):
    """Intersect every full policy row, updating the actual published status/metrics."""
    if candidate is None and unchanged is None:
        return None
    if candidate is None or unchanged is None:
        raise ValidationError("Veto support requires both the candidate and unchanged baseline policy")
    if (candidate["target_id"] != unchanged["target_id"]
            or candidate["source_sha256"] != unchanged["source_sha256"]):
        raise ValidationError("Veto and unchanged policies must bind the same target and source")
    output = deepcopy(candidate)
    base = {row["frame_index"]: row for row in unchanged["frames"]}
    if set(base) != {row["frame_index"] for row in output["frames"]}:
        raise ValidationError("Veto and unchanged policies differ in source frame inventory")
    suppressed = 0
    for row in output["frames"]:
        original_row = base[row["frame_index"]]
        bound = all(row.get(field) == original_row.get(field) for field in (
            "timestamp_ms", "source_sha256", "frame_sha256", "track_id", "box", "semantic_evidence_sha256"))
        if row["status"] == "supported" and (original_row["status"] != "supported" or not bound):
            row.update(status="unknown", lifecycle="baseline_subset_withheld", baseline_subset_suppressed=True)
            row["gap_codes"] = list(dict.fromkeys([*row["gap_codes"], "unchanged_baseline_positive_subset_required"]))
            suppressed += 1
        if row["status"] == "supported":
            _validate_current_chain(row, runtime, warnings, target_id=output["target_id"])
    output["metrics"]["supported_frames"] = sum(row["status"] == "supported" for row in output["frames"])
    output["metrics"]["unknown_frames"] = len(output["frames"]) - output["metrics"]["supported_frames"]
    # Additional subset withholding may suppress the underlying first recovery.
    # Count only published support after a hard evidence gap, preserving the raw
    # evaluator's separate diagnostics under a clearly named field.
    output["underlying_policy_reacquisition_events"] = candidate["metrics"]["reacquisition_events"]
    awaiting_recovery, recoveries = False, 0
    for row in output["frames"]:
        if row.get("confirmation_history_reset") or row["lifecycle"] == "evidence_gap" or row.get("track_identifier_changed"):
            awaiting_recovery = True
        if row["status"] == "supported" and awaiting_recovery:
            recoveries += 1
            awaiting_recovery = False
    output["metrics"]["reacquisition_events"] = recoveries
    output["status"] = output["frames"][-1]["status"]
    output["underlying_policy_hash"] = candidate["policy_hash"]
    output["policy"] = {**deepcopy(candidate["policy"]), "negative_warning_adapter": warning.VERSION,
        "trust_mode": TRUST_MODE, "unchanged_positive_subset_policy_hash": unchanged["policy_hash"]}
    output["policy_hash"] = original._digest(output["policy"])
    output["trust_mode"] = TRUST_MODE
    output["baseline_subset_suppressed_frames"] = suppressed
    output["positive_support_subset_verified"] = True
    return output


def _policies(entry, raw, reports, evaluator, resolved, assessed):
    runtime = strict.masks._runtime(entry, raw, reports)
    corrected, audit = strict.masks._filter(runtime, resolved, entry, raw)
    base = strict.baseline._policy(entry, corrected, evaluator)
    vetoed, warnings, changes = _runtime_veto(entry, corrected, assessed)
    candidate = strict.baseline._policy(entry, vetoed, evaluator)
    checked = _intersect_policy(candidate, base, vetoed, warnings)
    indexed = {row["frame_index"]: row for row in checked["frames"]} if checked else {}
    original_rows = {row["frame_index"]: row for row in base["frames"]} if base else {}
    predictions, unchanged = [], []
    for cp in entry["checkpoints"]:
        index = cp["frame_index"]
        for selected, output in ((indexed.get(index), predictions), (original_rows.get(index), unchanged)):
            status = ("unscored" if raw["status"] != "completed" or selected is None
                      or reports[index]["status"] == "unscored" else selected["status"])
            output.append({"clip_id": entry["clip_id"], "frame_index": index, "status": status,
                           "box": selected["box"] if selected and status == "supported" else None})
    return predictions, unchanged, checked, {"negative_warnings_applied": changes, "distractor_geometry": audit}


def _candidate_predictions(entry, raw, reports, assessed):
    runtime = strict.masks._runtime(entry, raw, reports)
    warnings = _warning_rows(entry, runtime, assessed)
    rows = []
    for cp in entry["checkpoints"]:
        index = cp["frame_index"]
        prediction = strict.baseline.previous._prediction(entry, raw, reports[index], index)
        if prediction["status"] == "supported" and warnings[index]["veto"]:
            prediction.update(status="unknown", box=None)
        rows.append(prediction)
    return rows


def run(root: str | Path, frozen: str | Path):
    root = Path(root).resolve()
    path = original._local(root, frozen)
    plan = original._read(path)
    verify_plan(root, plan)
    original._write(path.parent / "derivation-started.json", {"freeze_hash": plan["freeze_hash"],
        "started_at_utc": datetime.now(timezone.utc).isoformat(), "added_provider_requests": 0})
    result = replay(root, path)
    original._write(path.parent / "public-result.json", result)
    return result


def replay(root: str | Path, frozen: str | Path):
    root = Path(root).resolve()
    path = original._local(root, frozen)
    plan = original._read(path)
    strict_plan, historical, contact_path, contact_plan, sources = verify_plan(root, plan)
    marker = original._read(path.parent / "derivation-started.json")
    if marker.get("freeze_hash") != plan["freeze_hash"] or marker.get("added_provider_requests") != 0:
        raise ValidationError("Cached veto derivation requires its matching zero-call admission receipt")
    original_contact = historical["historical_contact_result"]
    strict_path = original._local(root, plan["strict_frozen_file"])
    cohorts = {}
    for cohort, source in zip(COHORTS, sources):
        references = [row for entry in source["clips"] for row in original._reference(entry)]
        predictions, preserved = {name: [] for name in METHODS}, {name: [] for name in METHODS}
        candidate_diagnostics, evidence = {version: [] for version in ("v2", "v3")}, []
        for entry in source["clips"]:
            name = entry["clip_id"]
            target_directory = strict._locations(root, contact_path, contact_plan, cohort)
            raw = original._read(target_directory / f"{name}-video.json")
            grippers = original._read(strict_path.parent / f"{name}-grippers.json")
            if raw.get("prepared") != entry["prepared"] or grippers.get("prepared") != strict_plan["gripper_preparations"][name]:
                raise ValidationError("Cached veto target/gripper responses differ from retained preparations")
            assessed = warning.compare(raw, grippers, target_track_id=1, gripper_track_ids=(3, 4))
            resolved = strict.distractors.resolve_distractor_conflicts(raw, target_track_id=1)
            policies = {}
            for version in ("v2", "v3"):
                directory = target_directory if cohort == COHORTS[0] and version == "v2" else contact_path.parent
                reports = {cp["frame_index"]: original._read(directory / f"{name}-{version}-{cp['frame_index']}.json")
                           for cp in entry["checkpoints"]}
                for label, evaluator in (("greedy", original.evaluate_tracking), ("current", strict.baseline.current.evaluate_tracking)):
                    method = f"{version}_veto_{label}"
                    rows, unchanged, policy, audit = _policies(entry, raw, reports, evaluator, resolved, assessed)
                    predictions[method].extend(rows)
                    preserved[method].extend(unchanged)
                    policies[method] = {"policy": policy, "audit": audit}
                candidate_diagnostics[version].extend(_candidate_predictions(entry, raw, reports, assessed))
            evidence.append({"clip_id": name, "source_attribution": entry["source_attribution"],
                "source_sha256": entry["prepared"]["source"]["sha256"], "warning_only_geometry": assessed,
                "policies": policies, "trust_mode": TRUST_MODE})
        new_methods = {name: strict.baseline.score_frames(references, predictions[name], iou_threshold=.3) for name in METHODS}
        for version in ("v2", "v3"):
            for label in ("greedy", "current"):
                name = f"{version}_veto_{label}"
                recomputed = strict.baseline.score_frames(references, preserved[name], iou_threshold=.3)
                if recomputed != original_contact["cohorts"][cohort]["methods"][f"{version}_{label}"]:
                    raise ValidationError("Unchanged contact baseline scores differ from their published method")
        cohorts[cohort] = {"labeled_frames": len(references), "references": references,
            "historical_methods": deepcopy(historical["cohorts"][cohort]["methods"]),
            "methods": {**deepcopy(historical["cohorts"][cohort]["methods"]), **new_methods},
            "candidate_veto_diagnostics": {version: {
                "localization": strict.baseline.score_frames(references, candidate_diagnostics[version], iou_threshold=.3),
                "visibility_counts": strict.baseline.previous.visibility_counts(references, candidate_diagnostics[version])}
                for version in ("v2", "v3")},
            "localization_sensitivity": {str(threshold): {method: strict.baseline.score_frames(
                references, predictions[method], iou_threshold=threshold)["metrics"] for method in METHODS}
                for threshold in CONFIG["localization_sensitivity_thresholds"]}, "evidence": evidence}
    masks._unchanged(root, plan["input_records"])
    if plan["implementation_sha256"] != _implementations():
        raise ValidationError("Cached veto code changed during derivation")
    return {"experiment_version": VERSION, "evaluation_role": EVALUATION_ROLE, "freeze_hash": plan["freeze_hash"],
        "selected_after_inspecting_strict_failure": True, "trust_mode": TRUST_MODE,
        "config": deepcopy(CONFIG), "implementation_sha256": plan["implementation_sha256"],
        "warning_config": deepcopy(warning.CONFIG), "strict_freeze_hash": plan["strict_freeze_hash"],
        "strict_public_sha256": plan["strict_public_sha256"], "strict_public_file_sha256": plan["strict_public_file_sha256"],
        "strict_public_reference": plan["strict_public_reference"],
        "historical_contact_public_sha256": historical["contact_public_sha256"],
        "historical_provider_attempts": {"contact_trial": original_contact["new_provider_attempts"],
            "strict_gripper_trial": historical["new_provider_attempts"]},
        "historical_estimated_incremental_usd": {"contact_trial": original_contact["estimated_incremental_usd"],
            "strict_gripper_trial": historical["estimated_incremental_usd"]},
        "historical_failed_operation_counts": {"contact_trial": original_contact["failed_operation_count"],
            "strict_gripper_trial": historical["new_failed_operation_count"]},
        "historical_accounting": {"contact_trial": deepcopy(original_contact["accounting"]),
            "strict_gripper_trial": deepcopy(historical["accounting"])},
        "added_provider_requests": 0, "estimated_incremental_usd": 0.0, "cohorts": cohorts,
        "changes_live_placement_acceptance": False, "physical_action_authorized": False,
        "limitations": list(LIMITATIONS)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "public-plan", "run", "replay"))
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--strict-frozen", type=Path)
    parser.add_argument("--baseline-public", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--frozen", type=Path)
    args = parser.parse_args(argv)
    if args.action == "freeze":
        result = freeze(args.root, args.strict_frozen, args.output, baseline_public=args.baseline_public)
        print(result["freeze_hash"])
    elif args.action == "public-plan":
        result = public_plan(args.root, args.frozen)
        original._write(args.output, result)
        print(result["freeze_hash"])
    else:
        result = run(args.root, args.frozen) if args.action == "run" else replay(args.root, args.frozen)
        print({key: result[key] for key in ("freeze_hash", "added_provider_requests", "estimated_incremental_usd")})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
