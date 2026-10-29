"""One prospective low-reasoning repair of an exposed incomplete baseline pair.

Original tracking trials, labels and results remain unchanged. This run repeats
the exact original two JPEGs and concept using the unchanged guided adapter;
only the new bounded client's omitted Spark reasoning default becomes ``low``.
An exclusive run marker and separate persistent three-attempt journal prevent
automatic or accidental re-execution. Results concern completion and candidate
diagnostics on exposed inputs, not accuracy or independent generalization.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping

from .bounded_spark import BOUNDED_SPARK_VERSION, BoundedSparkClient
from .core import ValidationError
from .inference_journal import AttemptJournal
from .meta import MetaClient, SPARK_MODEL, load_env_file
from . import tracking_experiment as original


VERSION = "tracking_baseline_completion_repair_v1"
CLIP_ID = "screwdriver"
PAIR_INDEX = 1
FRAME_INDICES = [8, 13]
CONFIG = {"bounded_spark_version": BOUNDED_SPARK_VERSION, "reasoning_effort": "low",
          "max_output_tokens": 4096, "max_requests": 3, "automatic_retry": False,
          "maximum_run_executions": 1}
_FILES = ("tracking_completion.py", "bounded_spark.py")
_LIMITS = [
    "A separate prospective completion repair of one exposed incomplete pair; the original experiment is preserved.",
    "The exact original frames and concept are reused. Completion and candidate support do not establish accuracy or generalization.",
    "The Spark locator and box-prompted SAM share an identity assumption; SAM does not independently verify the target.",
    "An exclusive local marker and journal bound request attempts, not provider billing or execution against a hostile local operator.",
    "Saved provider responses and hashes support local replay consistency, not cryptographic proof of API origin or physical action.",
]


class CompletionClient(BoundedSparkClient, original.RetainingClient):
    """Low omitted reasoning plus original partial-response retention and journal."""


def _implementations():
    return {name: original._sha(Path(__file__).with_name(name)) for name in _FILES}


def _failure(saved: Mapping[str, Any]) -> dict[str, Any]:
    """Retain only safe facts about the one already observed incomplete call."""
    try:
        response = saved["error"]["response"]
        if (saved["status"] != "unscored" or saved["result"] is not None
                or saved["error"]["code"] != "incomplete_response"
                or response["status"] != "incomplete" or response["model"] != SPARK_MODEL
                or response["max_output_tokens"] != 4096
                or len(saved["retained_responses"]) != 1
                or saved["retained_responses"][0]["response"] != response
                or saved["usage"]["request_attempts"] != 1):
            raise ValueError
        return {"status": "unscored", "error_code": "incomplete_response",
                "response_status": "incomplete", "model": SPARK_MODEL, "max_output_tokens": 4096,
                "reasoning_effort_reported_by_provider": response.get("reasoning", {}).get("effort"),
                "usage": _public_usage(saved["usage"])}
    except (KeyError, TypeError, ValueError, AttributeError):
        raise ValidationError("Completion repair requires the unchanged single incomplete 4096-token baseline call") from None


def _public_usage(value: Mapping[str, Any]) -> dict[str, Any]:
    # Caller validates the complete private ledger with original._verify_pair.
    keys = ("request_attempts", "input_tokens", "output_tokens", "sam_images", "estimated_usd",
            "spark_requests_with_unreported_token_usage", "unsuccessful_spark_requests_with_reported_token_usage")
    return {key: value[key] for key in keys if key in value}


def _selected(root: Path, plan: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, str]]:
    try:
        entries = [entry for entry in plan["clips"] if entry["clip_id"] == CLIP_ID]
        if len(entries) != 1:
            raise ValueError
        entry = entries[0]
        selected = entry["checkpoints"][PAIR_INDEX * 2:PAIR_INDEX * 2 + 2]
        if [item["frame_index"] for item in selected] != FRAME_INDICES:
            raise ValueError
        frames = [{"frame_index": index, "timestamp_ms": entry["frames"][index]["timestamp_ms"],
                   "file": str(original._local(root, entry["frames"][index]["file"])),
                   "sha256": entry["frames"][index]["sha256"]} for index in FRAME_INDICES]
        concepts = {entry["contract"]["target_id"]: entry["contract"]["concept"]}
        return frames, concepts
    except (KeyError, TypeError, IndexError, ValueError):
        raise ValidationError("Completion repair must preserve the original screwdriver pair at frames 8 and 13") from None


def freeze_repair(root: str | Path, original_frozen: str | Path, output_dir: str | Path) -> dict[str, Any]:
    root = Path(root).resolve()
    frozen = original._local(root, original_frozen)
    parent_plan = original._read(frozen)
    parent_result = original.replay(root, frozen)
    if parent_result.get("failed_operation_count") != 1:
        raise ValidationError("Completion repair is scoped to the one incomplete original operation")
    frames, concepts = _selected(root, parent_plan)
    pair_file = frozen.parent / f"{CLIP_ID}-pair-{PAIR_INDEX}.json"
    saved = original._read(pair_file)
    original._verify_pair(saved, [frame["file"] for frame in frames], concepts)
    failure = _failure(saved)
    plan = {"repair_version": VERSION, "evaluation_role": "prospective_completion_repair_of_exposed_pair",
            "created_at_utc": datetime.now(timezone.utc).isoformat(), "inference_started": False,
            "config": deepcopy(CONFIG), "implementation_sha256": _implementations(),
            "original_frozen_file": str(frozen), "original_frozen_sha256": original._sha(frozen),
            "original_freeze_hash": parent_plan["freeze_hash"],
            "original_implementation_sha256": deepcopy(parent_plan["implementation_sha256"]),
            "original_public_result_hash": original._digest(parent_result),
            "original_pair_file": str(pair_file), "original_pair_sha256": original._sha(pair_file),
            "original_failure": failure, "clip_id": CLIP_ID, "pair_index": PAIR_INDEX,
            "source_frames": frames, "concepts": concepts}
    plan["freeze_hash"] = original._digest(plan)
    output = original._local(root, output_dir)
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    original._write(output / "freeze.json", plan)
    return plan


def verify_repair(root: str | Path, plan: Mapping[str, Any], *, replay_original: bool = True) -> None:
    root = Path(root).resolve()
    expected = {"repair_version", "evaluation_role", "created_at_utc", "inference_started", "config",
                "implementation_sha256", "original_frozen_file", "original_frozen_sha256", "original_freeze_hash",
                "original_implementation_sha256", "original_public_result_hash", "original_pair_file",
                "original_pair_sha256", "original_failure", "clip_id", "pair_index", "source_frames", "concepts", "freeze_hash"}
    try:
        if (not isinstance(plan, Mapping) or set(plan) != expected or plan["repair_version"] != VERSION
                or plan["evaluation_role"] != "prospective_completion_repair_of_exposed_pair"
                or plan["inference_started"] is not False or plan["config"] != CONFIG
                or plan["implementation_sha256"] != _implementations()
                or plan["freeze_hash"] != original._digest({key: value for key, value in plan.items() if key != "freeze_hash"})
                or plan["clip_id"] != CLIP_ID or type(plan["pair_index"]) is not int or plan["pair_index"] != PAIR_INDEX):
            raise ValueError
        frozen = original._local(root, plan["original_frozen_file"])
        pair_file = original._local(root, plan["original_pair_file"])
        if (original._sha(frozen) != plan["original_frozen_sha256"]
                or pair_file != frozen.parent / f"{CLIP_ID}-pair-{PAIR_INDEX}.json"
                or original._sha(pair_file) != plan["original_pair_sha256"]):
            raise ValueError
        parent_plan = original._read(frozen)
        original.verify_plan(root, parent_plan, files=False)
        if (parent_plan["freeze_hash"] != plan["original_freeze_hash"]
                or parent_plan["implementation_sha256"] != plan["original_implementation_sha256"]):
            raise ValueError
        frames, concepts = _selected(root, parent_plan)
        if frames != plan["source_frames"] or concepts != plan["concepts"]:
            raise ValueError
        if any(original._sha(frame["file"]) != frame["sha256"] for frame in frames):
            raise ValueError
        saved = original._read(pair_file)
        original._verify_pair(saved, [frame["file"] for frame in frames], concepts)
        if _failure(saved) != plan["original_failure"]:
            raise ValueError
        if replay_original:
            result = original.replay(root, frozen)
            if result.get("failed_operation_count") != 1 or original._digest(result) != plan["original_public_result_hash"]:
                raise ValueError
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        raise ValidationError("Completion repair source, original failure, implementation or frozen configuration changed") from None


def _marker(plan: Mapping[str, Any]) -> dict[str, Any]:
    return {"repair_version": VERSION, "freeze_hash": plan["freeze_hash"],
            "max_requests": 3, "automatic_retry": False, "maximum_run_executions": 1}


def run_repair(root: str | Path, frozen_file: str | Path) -> dict[str, Any]:
    root = Path(root).resolve()
    frozen = original._local(root, frozen_file)
    plan = original._read(frozen)
    verify_repair(root, plan)
    output = frozen.parent
    original._write(output / "run-started.json", _marker(plan))
    load_env_file(root / ".env.local")
    journal = AttemptJournal(output / "attempts.sqlite3", request_limit=3)
    client = CompletionClient.from_env(journal=journal, operation_id=plan["freeze_hash"][:32],
        guard=lambda: verify_repair(root, plan, replay_original=False), max_requests=3, retries=0, timeout=90)
    saved = original._pair(client, [frame["file"] for frame in plan["source_frames"]], plan["concepts"])
    record = {"repair_freeze_hash": plan["freeze_hash"], "report": saved, "report_hash": original._digest(saved)}
    original._write(output / "pair-result.json", record)
    result = replay_repair(root, frozen)
    original._write(output / "public-result.json", result)
    return result


def _classifications(plan: Mapping[str, Any], saved: Mapping[str, Any]) -> list[dict[str, Any]]:
    if saved["status"] != "completed":
        return []
    result = saved["result"]
    rows = []
    for source, located in zip(plan["source_frames"], result["locator"]["parsed"]["frames"]):
        target = located["objects"][0]
        eligible = not target["identity_ambiguous"] and target["fully_visible"] and len(target["candidates"]) == 1
        locator_box = (original._normalized(original.normalized_box_to_pixels(target["candidates"][0], located["width"], located["height"]),
                                           located["width"], located["height"]) if eligible else None)
        records = result["results"][source["sha256"]]
        record = records.get(target["object_id"])
        objects = record["objects"] if record else []
        supported = bool(eligible and len(objects) == 1 and objects[0]["confidence"] is not None
                         and objects[0]["confidence"] >= original.CONFIG["min_confidence"]
                         and objects[0]["mask"]["foreground_pixels"] > 0)
        rows.append({"frame_index": source["frame_index"], "source_frame_sha256": source["sha256"],
                     "locator_support": "supported" if eligible else "unknown",
                     "locator_box": locator_box, "fully_visible": target["fully_visible"],
                     "identity_ambiguous": target["identity_ambiguous"], "candidate_count": len(target["candidates"]),
                     "guided_support": "supported" if supported else "unknown",
                     "guided_box": original._normalized(objects[0]["box"], located["width"], located["height"]) if supported else None,
                     "placement_pass": False, "accuracy_scored": False})
    return rows


def replay_repair(root: str | Path, frozen_file: str | Path) -> dict[str, Any]:
    root = Path(root).resolve()
    frozen = original._local(root, frozen_file)
    plan = original._read(frozen)
    verify_repair(root, plan)
    output = frozen.parent
    if original._read(output / "run-started.json") != _marker(plan):
        raise ValidationError("Completion repair marker differs from the frozen one-run admission")
    record = original._read(output / "pair-result.json")
    if (set(record) != {"repair_freeze_hash", "report", "report_hash"}
            or record["repair_freeze_hash"] != plan["freeze_hash"]
            or record["report_hash"] != original._digest(record["report"])):
        raise ValidationError("Completion repair report is not bound to this exact freeze")
    saved = record["report"]
    original._verify_pair(saved, [frame["file"] for frame in plan["source_frames"]], plan["concepts"])
    journal_path = output / "attempts.sqlite3"
    if not journal_path.is_file() or journal_path.is_symlink():
        raise ValidationError("Completion repair requires its persistent attempt journal")
    accounting = AttemptJournal(journal_path, request_limit=3).summary(plan["freeze_hash"][:32])
    if (accounting["total_reserved_attempts"] != saved["usage"]["request_attempts"]
            or accounting["operation_attempts"] != saved["usage"]["request_attempts"]
            or accounting["outcomes"]["reserved"] != 0):
        raise ValidationError("Completion repair retained responses and durable attempts disagree")
    return {"repair_version": VERSION, "evaluation_role": plan["evaluation_role"],
            "freeze_hash": plan["freeze_hash"], "original_freeze_hash": plan["original_freeze_hash"],
            "original_pair_sha256": plan["original_pair_sha256"], "config": deepcopy(CONFIG),
            "implementation_sha256": deepcopy(plan["implementation_sha256"]),
            "clip_id": CLIP_ID, "frame_indices": list(FRAME_INDICES),
            "source_frame_sha256": [frame["sha256"] for frame in plan["source_frames"]],
            "original_failure": deepcopy(plan["original_failure"]),
            "repair_status": saved["status"], "error_code": saved["error"].get("code") if saved["error"] else None,
            "usage": _public_usage(saved["usage"]), "accounting": accounting,
            "classifications": _classifications(plan, saved),
            "accuracy_scored": False, "changes_original_experiment": False,
            "changes_live_placement_acceptance": False, "limitations": list(_LIMITS)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "run", "replay"))
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--original-frozen", type=Path)
    parser.add_argument("--frozen", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.action == "freeze":
        if args.original_frozen is None or args.output is None:
            parser.error("freeze requires --original-frozen and --output")
        result = freeze_repair(args.root, args.original_frozen, args.output)
        public = {"freeze_hash": result["freeze_hash"], "inference_started": False, "max_requests": 3}
    else:
        if args.frozen is None:
            parser.error("run/replay requires --frozen")
        result = run_repair(args.root, args.frozen) if args.action == "run" else replay_repair(args.root, args.frozen)
        public = {"freeze_hash": result["freeze_hash"], "repair_status": result["repair_status"], "usage": result["usage"]}
    print(json.dumps(public))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
