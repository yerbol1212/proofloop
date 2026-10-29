"""Portable, zero-inference replay of the published tracking policy comparison.

The bundle contains cached, source-bound perception measurements. It replays
policies and scoring, not video decoding, SAM mask distances or API provenance.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
from typing import Any

from .core import ValidationError
from . import confirmation_policy, tracking_policy
from .tracking_evaluation import score_frames
from . import gripper_veto_experiment as veto

VERSION = "portable_tracking_policy_replay_v1"
BUNDLE_SHA256 = "6c140a3bacd7bab49839cdadf5c633f8be18762b039da7a76759ed7a2e379d0f"
MAX_BUNDLE_BYTES = 2 * 1024 * 1024
COHORTS = ("exposed_episode_2", "new_episode_3")
METHODS = ("v2_greedy", "v2_current", "v2_veto_current")
_HASH = re.compile(r"[0-9a-f]{64}\Z")
_CONFIG = {"min_confidence": .8, "min_stable_views": 2, "min_spacing_ms": 500,
           "warning_margin": 3, "iou_thresholds": [.2, .3, .5]}
LIMITS = [
    "Cached development policy/score replay; no new model calls or held-out accuracy evidence.",
    "Perception judgments, mask availability and proximity measurements are supplied cached inputs, not independently regenerated here.",
    "Original video pixels, encoded provider masks and arbitrary provider text are not included.",
    "Hashes check consistency against the packaged receipt, not trusted capture time, identity or API origin.",
    "Results do not change live placement acceptance or authorize physical action.",
]


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def default_bundle() -> Path:
    return Path(__file__).with_name("data") / "contact_replay_v1.json"


def load_bundle(path: Path | str | None = None, *, expected_sha256: str | None = None):
    path = Path(path) if path is not None else default_bundle()
    expected = expected_sha256 if expected_sha256 is not None else BUNDLE_SHA256
    if not isinstance(expected, str) or not _HASH.fullmatch(expected):
        raise ValidationError("Supply the exact trusted bundle SHA-256")
    if path.is_symlink() or not path.is_file() or not 0 < path.stat().st_size <= MAX_BUNDLE_BYTES:
        raise ValidationError("Replay bundle must be a bounded regular JSON file")
    payload = path.read_bytes()
    if len(payload) > MAX_BUNDLE_BYTES or hashlib.sha256(payload).hexdigest() != expected:
        raise ValidationError("Replay bundle differs from the trusted receipt")
    def unique(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate field")
            result[key] = value
        return result
    try:
        bundle = json.loads(payload.decode("utf-8"), object_pairs_hook=unique,
                            parse_constant=lambda _: (_ for _ in ()).throw(ValueError()))
        _digest(bundle)
    except (ValueError, TypeError, UnicodeError, RecursionError):
        raise ValidationError("Replay bundle JSON is invalid") from None
    fields = {"schema_version", "fixture_id", "scope", "evaluation_role", "canonical_result_sha256",
              "implementation_sha256", "config", "clips", "expected_methods"}
    if (type(bundle) is not dict or set(bundle) != fields or bundle.get("schema_version") != "1.0"
            or bundle.get("fixture_id") != "latest-tracking"
            or bundle.get("scope") != "cached_policy_and_score_replay_only"
            or bundle.get("evaluation_role") != "post_inspection_exposed_cached_development_retest"
            or bundle.get("config") != _CONFIG or type(bundle.get("clips")) is not list
            or len(bundle["clips"]) != 4 or set(bundle.get("expected_methods", {})) != set(COHORTS)
            or not isinstance(bundle.get("canonical_result_sha256"), str)
            or not _HASH.fullmatch(bundle["canonical_result_sha256"])
            or type(bundle.get("implementation_sha256")) is not dict
            or set(bundle["implementation_sha256"]) != set(veto.FILES)):
        raise ValidationError("Replay bundle has an unsupported schema or scope")
    for name, digest in bundle.get("implementation_sha256", {}).items():
        if not re.fullmatch(r"[a-z_]+\.py", name) or not _HASH.fullmatch(digest):
            raise ValidationError("Invalid implementation receipt")
        implementation = Path(__file__).with_name(name)
        if not implementation.is_file() or hashlib.sha256(implementation.read_bytes()).hexdigest() != digest:
            raise ValidationError("Frozen policy dependency differs from the replay receipt")
    if not bundle.get("implementation_sha256"):
        raise ValidationError("Replay requires its frozen implementation inventory")
    return bundle


def _validate_warnings(clip):
    runtime = {row["frame_index"]: row for row in clip["runtime"]}
    warnings = clip["warning_measurements"]
    if len(runtime) != len(clip["runtime"]) or len(warnings) != len(runtime):
        raise ValidationError("Replay measurements require one row per exact source frame")
    seen = set()
    for row in warnings:
        index = row.get("frame_index")
        frame = runtime.get(index)
        if (type(index) is not int or index in seen or frame is None
                or row.get("source_sha256") != clip["source_sha256"]
                or frame["source_sha256"] != clip["source_sha256"]
                or row.get("timestamp_ms") != frame["timestamp_ms"]
                or row.get("target_track_id") != 1 or type(row.get("comparisons")) is not list
                or len(row["comparisons"]) != 2):
            raise ValidationError("Replay warning differs from its exact source frame")
        seen.add(index)
        near, available, ids = False, True, set()
        for comparison in row["comparisons"]:
            identifier = comparison.get("gripper_track_id")
            distance = comparison.get("min_chebyshev_distance_capped4")
            if identifier not in (3, 4) or identifier in ids:
                raise ValidationError("Replay warning requires both reviewed gripper identifiers")
            ids.add(identifier)
            if distance is not None and (type(distance) is not int or not 0 <= distance <= 4):
                raise ValidationError("Replay warning distance is invalid")
            present = distance is not None
            if comparison.get("mask_comparison_available") is not present:
                raise ValidationError("Replay mask availability contradicts its distance")
            if present and any(not isinstance(comparison.get(role + "_mapped_mask_sha256"), str)
                               or not _HASH.fullmatch(comparison[role + "_mapped_mask_sha256"])
                               for role in ("target", "gripper")):
                raise ValidationError("Cached distance lacks its source-mapped mask receipts")
            is_near = present and distance <= _CONFIG["warning_margin"]
            if comparison.get("near_proposal") is not is_near:
                raise ValidationError("Replay near warning contradicts the fixed margin")
            near |= is_near
            available &= present
        expected = "near_gripper_proposal" if near else "no_near_proposal_observed" if available else "geometry_unavailable"
        if (row.get("veto") is not near or row.get("warning") != expected
                or row.get("additional_geometry_complete") is not available):
            raise ValidationError("Replay warning cannot create positive geometry evidence")


def replay(path: Path | str | None = None, *, expected_sha256: str | None = None):
    bundle = load_bundle(path, expected_sha256=expected_sha256)
    references = {name: [] for name in COHORTS}
    predictions = {name: {method: [] for method in METHODS} for name in COHORTS}
    policies, seen = [], set()
    for clip in bundle["clips"]:
        if (type(clip) is not dict or set(clip) != {"cohort", "clip_id", "target_id", "source_attribution",
                "source_sha256", "seed", "runtime", "warning_measurements", "references", "unscored_indices"}
                or not isinstance(clip["source_sha256"], str) or not _HASH.fullmatch(clip["source_sha256"])
                or type(clip["unscored_indices"]) is not list
                or any(type(i) is not int for i in clip["unscored_indices"])):
            raise ValidationError("Replay source has unsupported fields or provenance")
        key = (clip["cohort"], clip["clip_id"])
        if key[0] not in COHORTS or key in seen or not 1 <= len(clip["runtime"]) <= 120:
            raise ValidationError("Replay source inventory is invalid")
        seen.add(key)
        _validate_warnings(clip)
        arguments = {"target_id": clip["target_id"], "seed": clip["seed"],
                     **{name: _CONFIG[name] for name in ("min_confidence", "min_stable_views", "min_spacing_ms")}}
        current = confirmation_policy.evaluate_tracking(clip["runtime"], **arguments)
        greedy = tracking_policy.evaluate_tracking(clip["runtime"], **arguments)
        entry = {"prepared": {"source": {"sha256": clip["source_sha256"]}}}
        assessed = {"source_sha256": clip["source_sha256"], "frames": clip["warning_measurements"]}
        changed, warnings, _ = veto._runtime_veto(entry, clip["runtime"], assessed)
        warned = confirmation_policy.evaluate_tracking(changed, **arguments)
        warned = veto._intersect_policy(warned, current, changed, warnings)
        for method, policy in zip(METHODS, (greedy, current, warned)):
            indexed = {row["frame_index"]: row for row in policy["frames"]}
            for reference in clip["references"]:
                index = reference["frame_index"]
                if reference["clip_id"] != clip["clip_id"] or index not in indexed:
                    raise ValidationError("Replay reference differs from its scoring-only source")
                row = indexed[index]
                status = "unscored" if index in clip["unscored_indices"] else row["status"]
                predictions[key[0]][method].append({"clip_id": key[1], "frame_index": index,
                    "status": status, "box": row["box"] if status == "supported" else None})
        references[key[0]].extend(deepcopy(clip["references"]))
        policies.append({"cohort": key[0], "clip_id": key[1], "source_attribution": clip["source_attribution"],
                         "policy": warned, "warning_measurements": deepcopy(clip["warning_measurements"]),
                         "unscored_indices": list(clip["unscored_indices"])})
    cohorts = {}
    for cohort in COHORTS:
        methods = {name: score_frames(references[cohort], predictions[cohort][name], iou_threshold=.3) for name in METHODS}
        if methods != bundle["expected_methods"][cohort]:
            raise ValidationError("Portable policy replay differs from the published comparison")
        cohorts[cohort] = {"methods": methods, "localization_sensitivity": {
            str(threshold): {name: score_frames(references[cohort], predictions[cohort][name], iou_threshold=threshold)["metrics"]
                             for name in METHODS} for threshold in _CONFIG["iou_thresholds"]}}
    return {"schema_version": "1.0", "replay_version": VERSION, "fixture_id": "latest-tracking",
            "mode": "cached_demo", "scope": bundle["scope"], "evaluation_role": bundle["evaluation_role"],
            "bundle_sha256": expected_sha256 or BUNDLE_SHA256, "canonical_result_sha256": bundle["canonical_result_sha256"],
            "provider_requests": 0, "estimated_incremental_usd": 0.0, "cohorts": cohorts, "policies": policies,
            "physical_action_authorized": False, "changes_live_placement_acceptance": False, "limitations": list(LIMITS)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--sha256", help="Trusted expected digest for an explicitly supplied bundle")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        report = replay(args.bundle, expected_sha256=args.sha256)
        if args.output:
            with args.output.open("x") as stream:
                json.dump(report, stream, sort_keys=True, indent=2, allow_nan=False)
                stream.write("\n")
        print(json.dumps({"replay_version": VERSION, "provider_requests": 0, "scope": report["scope"],
            "cohorts": {name: {method: {key: value["metrics"][key] for key in (
                "positives_retained", "positive_count", "positive_unscored", "unsupported_support_on_unknown", "unknown_count")}
                for method, value in cohort["methods"].items()} for name, cohort in report["cohorts"].items()}}, sort_keys=True))
    except (OSError, ValueError, KeyError, TypeError):
        print("Replay rejected: the bundle, frozen policy or expected comparison is invalid.")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
