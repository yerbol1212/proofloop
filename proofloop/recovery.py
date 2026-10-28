"""Private, replayable recovery receipts; no claim that a physical move occurred.

Hashes detect changes relative to a receipt. They are not signatures, trusted
timestamps, proof of API origin, or independent proof of declared capture time.
The verifier replays saved observations at their recorded evaluation time.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Mapping

from .core import EvidenceStore, Observation, Protocol, ValidationError, aggregate_status, evaluate
from .grounding import SamImageEvidence, ground_sam_images
from .session import _bound_result, _repair_proposal

ROOT = Path(__file__).resolve().parents[1]
OUTCOMES = {"awaiting_fresh_evidence", "fresh_goal_verified_after_requested_repair",
            "fresh_goal_verified_after_inspection", "still_failed", "needs_evidence", "error", "cancelled_task_change"}


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def task_contract(task: Mapping[str, Any], regions: Mapping[str, Any] | None = None) -> dict[str, Any]:
    fields = ("protocol", "protocol_hash", "task_hash", "concepts", "perception_mode", "perception_version")
    result = {key: deepcopy(task[key]) for key in fields if key in task}
    if regions is not None:
        result["regions"] = deepcopy(dict(regions))
    return result


def _relative(root: Path, path: Path) -> str:
    root, path = root.resolve(), path.resolve()
    if not any(directory in path.parents for directory in ((root / "artifacts/local"), (root / "inputs/local"))):
        raise ValidationError("Receipt sources must stay in private project directories")
    return path.relative_to(root).as_posix()


def _reference(root: Path, path: Path) -> dict[str, Any]:
    return {"file": _relative(root, path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def evidence_bundle(root: Path, task: Mapping[str, Any], regions: Mapping[str, Any], *,
                    source_kind: str, inference_origin: str, revision: int, timestamps: list[int],
                    evaluation_now_ms: int, timestamp_basis: str, image_paths: list[Path],
                    result_paths: list[dict[str, Path]], guided_path: Path | None = None) -> dict[str, Any]:
    if len(image_paths) != 2 or len(timestamps) != 2 or len(result_paths) != 2:
        raise ValidationError("Receipt evidence requires two source views")
    contract = task_contract(task, regions)
    frames = [{**_reference(root, path), "timestamp_ms": timestamps[index],
               "results": {key: _reference(root, value) for key, value in result_paths[index].items()}}
              for index, path in enumerate(image_paths)]
    bundle = {"contract": contract, "contract_hash": digest(contract), "source_kind": source_kind,
            "inference_origin": inference_origin, "revision": revision, "frames": frames,
            "evaluation_now_ms": evaluation_now_ms, "timestamp_basis": timestamp_basis}
    if task.get("perception_mode") == "guided":
        if guided_path is None:
            raise ValidationError("Guided recovery requires its saved source-bound locator")
        bundle["guided"] = _reference(root, guided_path)
    return bundle


def _read_reference(root: Path, reference: Any, *, json_file: bool = False):
    if not isinstance(reference, dict) or set(reference) - {"file", "sha256", "timestamp_ms", "results"}:
        raise ValidationError("Receipt file reference is malformed")
    relative = reference.get("file")
    if not isinstance(relative, str) or Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise ValidationError("Receipt reference must be a private project-relative file")
    path = root / relative
    _relative(root, path)
    if not path.is_file() or path.stat().st_size > 64 * 1024 * 1024:
        raise ValidationError("Receipt referenced file is unavailable or excessive")
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != reference.get("sha256"):
        raise ValidationError("Receipt referenced content changed")
    return json.loads(data) if json_file else data


def replay_evidence(bundle: Any, root: Path) -> dict[str, Any]:
    if not isinstance(bundle, dict) or bundle.get("contract_hash") != digest(bundle.get("contract")):
        raise ValidationError("Receipt evidence contract changed")
    contract = bundle["contract"]
    protocol = Protocol.from_dict(contract["protocol"])
    if protocol.content_hash != contract.get("protocol_hash"):
        raise ValidationError("Receipt protocol hash does not match the executable checks")
    if (set(contract["concepts"]) != set(protocol.objects)
            or set(contract["regions"]) != set(protocol.regions)):
        raise ValidationError("Receipt concepts or calibration do not match its protocol")
    frames = bundle.get("frames")
    if not isinstance(frames, list) or len(frames) != 2 or frames[0]["sha256"] == frames[1]["sha256"]:
        raise ValidationError("Receipt replay requires two distinct source views")
    times = [frame.get("timestamp_ms") for frame in frames]
    if any(type(value) is not int for value in times) or times[0] < 0 or times[1] - times[0] < 1000:
        raise ValidationError("Receipt source timestamps do not support the declared spacing")
    if type(bundle.get("evaluation_now_ms")) is not int or bundle["evaluation_now_ms"] < times[-1]:
        raise ValidationError("Receipt evaluation time precedes its evidence")
    if type(bundle.get("revision")) is not int or bundle["revision"] < 0:
        raise ValidationError("Receipt evidence revision is invalid")
    samples = []
    # Use the same bounded header parser as the upload boundary; no new decoder
    # dependency, fabricated masks or network calls enter an offline replay.
    from .interactive import _image_dimensions
    guided = None
    located = None
    if contract.get("perception_mode") == "guided":
        from .guided import _locator_bound
        guided = _read_reference(root, bundle["guided"], json_file=True)
        if guided.get("provenance", {}).get("version") != contract.get("perception_version"):
            raise ValidationError("Receipt locator version differs from its reviewed task")
        metadata = []
        for index, frame in enumerate(frames):
            data = _read_reference(root, frame)
            width, height = _image_dimensions(data, "image/jpeg")
            metadata.append({"frame_index": index, "width": width, "height": height, "sha256": frame["sha256"]})
        located = _locator_bound(guided["locator"], metadata, contract["concepts"])
    for frame in frames:
        data = _read_reference(root, frame)
        mime = "image/png" if data.startswith(b"\x89PNG") else "image/jpeg"
        dimensions = _image_dimensions(data, mime)
        if not isinstance(frame.get("results"), dict) or set(frame["results"]) - set(protocol.objects):
            raise ValidationError("Receipt results identify an unexpected target")
        for object_id, reference in frame["results"].items():
            result = _read_reference(root, reference, json_file=True)
            _bound_result(result, frame["sha256"])
            if any((item.get("box", {}).get("width"), item.get("box", {}).get("height")) != dimensions
                   for item in result.get("objects", [])):
                raise ValidationError("Receipt mask geometry differs from its exact source image")
            if contract.get("perception_mode") == "guided":
                provenance = result.get("guided_provenance", {})
                index = frames.index(frame)
                target = next(item for item in located[index]["objects"] if item["object_id"] == object_id)
                if (provenance.get("version") != contract.get("perception_version")
                        or target["identity_ambiguous"] or not target["fully_visible"] or len(target["candidates"]) != 1
                        or digest(guided["results"].get(frame["sha256"], {}).get(object_id)) != digest(result)):
                    raise ValidationError("Receipt guided method differs from its reviewed version")
            samples.append(SamImageEvidence(object_id, frame["timestamp_ms"], result,
                           "receipt_replay_" + bundle["source_kind"], contract["concepts"][object_id]))
    store = EvidenceStore(protocol, "recovery-offline-replay")
    store.revision = bundle["revision"]
    grounded = ground_sam_images(protocol, contract["regions"], samples, session_id=store.session_id, revision=store.revision)
    for observation in grounded.observations:
        store.add(observation)
    verdicts = evaluate(protocol, store, now_ms=bundle["evaluation_now_ms"])
    return {"status": aggregate_status(verdicts).value, "checks": [item.to_dict() for item in verdicts],
            "observations": [item.to_dict() for item in grounded.observations],
            "source_hashes": [frame["sha256"] for frame in frames]}


def seal_receipt(record: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(dict(record))
    result.pop("receipt_hash", None)
    result["receipt_hash"] = digest(result)
    return result


def save_receipt(path: Path, record: Mapping[str, Any]) -> dict[str, Any]:
    result = seal_receipt(record)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return result


def _statuses(checks: Any, protocol: Protocol) -> dict[str, str]:
    expected = {check.id: check for check in protocol.checks}
    if (not isinstance(checks, list) or len(checks) != len(expected)
            or any(not isinstance(item, dict) or item.get("check_id") not in expected
                   or item.get("status") not in ("pass", "fail", "unknown") for item in checks)):
        raise ValidationError("Receipt checks must cover the exact protocol once")
    result = {item["check_id"]: item["status"] for item in checks}
    if len(result) != len(expected):
        raise ValidationError("Receipt checks cannot contain duplicate IDs")
    for item in checks:
        if "object_id" in item and item["object_id"] != expected[item["check_id"]].object_id:
            raise ValidationError("Receipt check is bound to a different object")
    return result


def _replay_plan(value, protocol, before):
    plan = _repair_proposal(protocol, tuple(Observation.from_dict(item) for item in before["observations"]),
                            {"stage_id": "recovery", "status": before["status"], "checks": before["checks"]})
    if plan.get("request_inspection") or plan["plan"]["action_ids"] != value["plan_action_ids"]:
        raise ValidationError("Receipt proposed actions differ from the bounded repair planner")
    names = {f"move-{obj}-{source}-to-{target}": (obj, source, target)
             for obj in protocol.objects for source in protocol.regions for target in protocol.regions if source != target}
    steps = [f"Move the {names[action][0]} from {names[action][1].upper()} to {names[action][2].upper()}."
             for action in plan["plan"]["action_ids"] if action in names]
    steps.append("Keep every target visible and hands clear, then capture two fresh views at least one second apart.")
    if value.get("plan_steps") != steps or value.get("planner_version") != "declared_location_dijkstra_v1":
        raise ValidationError("Receipt repair presentation differs from its declared planner")


def verify_recovery_receipt(receipt: Path | Mapping[str, Any], *, root: Path = ROOT, _depth: int = 0) -> dict[str, Any]:
    """Recompute a private historical receipt; return a safe result, never paths."""
    value, valid, reason = {}, False, "Receipt verification failed; no verified recovery claim is available."
    try:
        root = Path(root).resolve()
        if _depth > 32:
            raise ValidationError("Receipt parent chain exceeds its verification budget")
        if isinstance(receipt, Mapping):
            value = deepcopy(dict(receipt))
        else:
            path = Path(receipt)
            _relative(root, path)
            if path.stat().st_size > 128 * 1024 * 1024:
                raise ValidationError("Receipt is excessive")
            value = json.loads(path.read_text())
        if (not isinstance(value, dict) or value.get("receipt_hash") != seal_receipt(value)["receipt_hash"]
                or value.get("schema_version") != "1.0" or value.get("kind") not in ("repair", "inspection")
                or value.get("outcome") not in OUTCOMES or value.get("physical_action_execution_verified") is not False):
            raise ValidationError("Receipt structure or hash changed")
        if digest(value["contract"]) != value["contract_hash"]:
            raise ValidationError("Receipt task contract changed")
        protocol = Protocol.from_dict(value["contract"]["protocol"])
        if protocol.content_hash != value["contract"].get("protocol_hash"):
            raise ValidationError("Receipt task protocol changed")
        if (any(type(value.get(key)) is not int or value[key] < 0 for key in ("before_revision", "requested_revision"))
                or value["requested_revision"] <= value["before_revision"]
                or value.get("after_revision") is not None and
                (type(value["after_revision"]) is not int or value["after_revision"] < value["requested_revision"])):
            raise ValidationError("Receipt revisions do not link a request to later evidence")
        before_statuses = _statuses(value["before_checks"], protocol)
        aggregate = lambda statuses: "fail" if "fail" in statuses.values() else "unknown" if "unknown" in statuses.values() else "pass"
        if aggregate(before_statuses) != value.get("before_status"):
            raise ValidationError("Receipt earlier aggregate differs from its checks")
        if value.get("after_status") is not None:
            if aggregate(_statuses(value["after_checks"], protocol)) != value["after_status"]:
                raise ValidationError("Receipt later aggregate differs from its checks")
        elif value.get("after_checks") or value.get("after") or value.get("after_revision") is not None:
            raise ValidationError("Receipt has unlinked later evidence")
        parent_ref = value.get("previous_receipt")
        if parent_ref is not None:
            parent = _read_reference(root, parent_ref, json_file=True)
            parent_verification = verify_recovery_receipt(parent, root=root, _depth=_depth + 1)
            if (not parent_verification["valid"] or parent["receipt_hash"] != value.get("previous_receipt_hash")
                    or any(parent.get(key) != value.get(key) for key in
                           ("receipt_id", "kind", "contract", "contract_hash", "before", "before_status", "before_checks",
                            "before_revision", "plan_action_ids", "plan_steps", "planner_version"))
                    or value["requested_revision"] < parent["requested_revision"]
                    or parent["outcome"].startswith("fresh_goal_verified")):
                raise ValidationError("Receipt request parent does not match the linked recovery")
        elif value.get("previous_receipt_hash") is not None or value["outcome"] != "awaiting_fresh_evidence":
            raise ValidationError("Completed recovery needs its immutable request parent")
        before = replay_evidence(value["before"], root) if value.get("before") else None
        after = replay_evidence(value["after"], root) if value.get("after") else None
        if value.get("after_status") in ("pass", "fail") and after is None:
            raise ValidationError("A conclusive later result needs actual saved evidence")
        if value["outcome"] == "awaiting_fresh_evidence" and (after or value.get("after_status") is not None):
            raise ValidationError("An awaiting request cannot already assert a later result")
        for name, computed in (("before", before), ("after", after)):
            if computed:
                binding = {key: item for key, item in value[name]["contract"].items() if key != "regions"}
                if (digest(binding) != value["contract_hash"]
                        or computed["status"] != value[name + "_status"]
                        or _statuses(computed["checks"], protocol) != _statuses(value[name + "_checks"], protocol)
                        or value[name]["revision"] != value[name + "_revision"]):
                    raise ValidationError("Receipt replay differs from the reported result")
        if value["kind"] == "repair" and (not before or before["status"] != "fail"
                or not any(action.startswith("move-") for action in value.get("plan_action_ids", []))):
            raise ValidationError("Physical repair request requires a supported earlier failure and move proposal")
        if value["kind"] == "repair":
            _replay_plan(value, protocol, before)
        elif value.get("plan_action_ids"):
            raise ValidationError("An inspection request cannot assert physical repair actions")
        if after:
            if (value["after"]["revision"] < value["requested_revision"]
                    or value["requested_revision"] <= value["before_revision"]
                    or before and set(before["source_hashes"]) & set(after["source_hashes"])):
                raise ValidationError("Recovery used stale, reused or unlinked evidence")
        expected = ("fresh_goal_verified_after_requested_repair" if value["kind"] == "repair"
                    else "fresh_goal_verified_after_inspection")
        if value["outcome"].startswith("fresh_goal_verified") and (not after or after["status"] != "pass" or value["outcome"] != expected):
            raise ValidationError("Receipt has no fresh passing evidence for its claimed outcome")
        if value["outcome"] == "still_failed" and (not after or after["status"] != "fail"):
            raise ValidationError("Receipt failure outcome does not match its evidence")
        if value["outcome"] == "needs_evidence" and (not after or after["status"] != "unknown"):
            raise ValidationError("Receipt uncertainty outcome does not match its evidence")
        valid = True
        reason = ("Saved sources, mask rasters, task contract and historical check results replay consistently. Capture timing and physical execution remain unproved."
                  if after else "The request record is consistent. No fresh passing recovery outcome is established by this request alone.")
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        pass
    if not isinstance(value, dict):
        value = {}
    receipt_hash = value.get("receipt_hash")
    receipt_hash = receipt_hash if isinstance(receipt_hash, str) and re.fullmatch(r"[0-9a-f]{64}", receipt_hash) else None
    return {"valid": valid, "scope": "completed" if value.get("after") else "requested",
            "receipt_hash": receipt_hash, "outcome": value.get("outcome") if valid else "error",
            "reason": reason, "physical_action_execution_verified": False}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    result = verify_recovery_receipt(args.receipt, root=args.root)
    print(json.dumps(result, indent=2))
    return 0 if result["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
