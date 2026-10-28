"""Approve and evaluate one bounded, recorded phone placement/correction session.

Source times and stage boundaries come from a reviewed capture manifest. They
are human annotations, not actions detected by SAM. Plans never update measured
facts. All raw media, approvals, responses and traces remain local.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
from typing import Any, Mapping

from .core import EvidenceStore, Protocol, RepairAction, ValidationError, aggregate_status, evaluate, plan_repair
from .grounding import SamImageEvidence, ground_sam_images
from .meta import MAX_IMAGE_BYTES, SAM_MODEL, MetaAPIError, MetaClient, load_env_file

ROOT = Path(__file__).resolve().parents[1]
STAGES = ("initial_correct", "wrong_region", "occluded", "corrected")
ORIGINS = ("owned_phone", "synthetic_test_fixture", "synthetic_fixture", "imagegen_synthetic")
MAX_REQUESTS = 16
MAX_JSON_BYTES = 1024 * 1024
REVIEW_ORIGINS = ("local_operator", "assistant_under_user_authorization")


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def contract_hash(contract: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical(contract)).hexdigest()


def _digest(value: Any) -> None:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValidationError("Expected an exact source SHA-256 digest")


def _seconds(value: Any) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValidationError("Source seconds must be finite and nonnegative")
    return float(value)


def _validate_contract(contract: Mapping[str, Any]) -> Protocol:
    if not isinstance(contract, Mapping) or set(contract) != {
        "schema_version", "capture_manifest_file", "capture", "protocol", "regions", "concepts"
    } or contract["schema_version"] != "1.0":
        raise ValidationError("Unsupported session contract")
    if not isinstance(contract["capture_manifest_file"], str) or not contract["capture_manifest_file"]:
        raise ValidationError("Capture manifest path is required")
    protocol = Protocol.from_dict(contract["protocol"])
    if (len(protocol.objects) != 2 or len(protocol.regions) != 2 or len(protocol.checks) != 2
            or protocol.steps or protocol.checkpoints
            or {check.object_id for check in protocol.checks} != set(protocol.objects)
            or any(check.kind != "location" or check.unique_target is not True for check in protocol.checks)):
        raise ValidationError("This session supports two explicit unique-target placement checks only")
    concepts = contract["concepts"]
    if (not isinstance(concepts, Mapping) or set(concepts) != set(protocol.objects)
            or any(not isinstance(value, str) or not value.strip() or len(value) > 128 for value in concepts.values())
            or len(set(concepts.values())) != len(concepts)):
        raise ValidationError("Approve one distinct SAM concept for each target object")
    # Exercise the shared rectangle validator without manufacturing observations.
    if set(contract["regions"]) != set(protocol.regions):
        raise ValidationError("Every region needs an approved calibration rectangle")
    ground_sam_images(protocol, contract["regions"], [], session_id="contract-validation")
    capture = contract["capture"]
    if (not isinstance(capture, Mapping) or capture.get("evidence_origin") not in ORIGINS
            or capture.get("offsets_confirmed") is not True):
        raise ValidationError("Reviewed capture offsets and declared evidence origin are required")
    source = capture.get("source")
    if not isinstance(source, Mapping):
        raise ValidationError("Capture source metadata is required")
    _digest(source.get("sha256"))
    duration = _seconds(source.get("duration_seconds"))
    if not 0 < duration <= 120:
        raise ValidationError("Use one source clip of at most 120 seconds")
    stages = capture.get("stages")
    if not isinstance(stages, list) or len(stages) != len(STAGES):
        raise ValidationError("The session requires four recorded stages")
    seen_hashes: set[str] = set()
    previous_end = 0.0
    for stage_id, stage in zip(STAGES, stages):
        if not isinstance(stage, Mapping) or stage.get("id") != stage_id:
            raise ValidationError("Capture stages must be in the declared session order")
        start, end = _seconds(stage.get("start_seconds")), _seconds(stage.get("end_seconds"))
        if not previous_end <= start < end <= duration:
            raise ValidationError("Stage windows must be ordered, disjoint and inside the clip")
        frames = stage.get("frames")
        if not isinstance(frames, list) or len(frames) != 2:
            raise ValidationError("Select exactly two stable frames per stage")
        times = []
        for frame in frames:
            if not isinstance(frame, Mapping):
                raise ValidationError("Frame metadata is required")
            _digest(frame.get("sha256"))
            if frame["sha256"] in seen_hashes:
                raise ValidationError("A prior or repeated frame cannot support a new stage")
            seen_hashes.add(frame["sha256"])
            timestamp = frame.get("source_timestamp_ms")
            if type(timestamp) is not int or not start * 1000 <= timestamp < end * 1000:
                raise ValidationError("Actual frame PTS must lie inside its declared stage")
            if not isinstance(frame.get("file"), str) or Path(frame["file"]).name != frame["file"]:
                raise ValidationError("Frame files must be relative filenames")
            times.append(timestamp)
        if times[1] - times[0] < 1000:
            raise ValidationError("Stable samples must be at least one source second apart")
        previous_end = end
    return protocol


def session_contract(capture_manifest: dict[str, Any], protocol: Protocol, regions: dict[str, Any],
                     concepts: dict[str, str], *, manifest_file: str) -> dict[str, Any]:
    contract = {"schema_version": "1.0", "capture_manifest_file": manifest_file,
                "capture": deepcopy(capture_manifest), "protocol": protocol.to_dict(),
                "regions": deepcopy(regions), "concepts": deepcopy(concepts)}
    _validate_contract(contract)
    return contract


def _review_provenance(approval: Mapping[str, Any]) -> dict[str, Any]:
    """Validate declared reviewer identity without implying a human signed an assistant review."""
    origin = approval.get("origin")
    if origin not in REVIEW_ORIGINS:
        raise ValidationError("Use an explicit supported contract reviewer origin")
    human_reviewed = origin == "local_operator"
    if approval.get("human_reviewed_exact_contract") is not human_reviewed:
        raise ValidationError("Exact-contract human review must agree with the declared reviewer origin")
    provenance = {"origin": origin, "human_reviewed_exact_contract": human_reviewed}
    reference = approval.get("authorization_reference")
    if origin == "assistant_under_user_authorization":
        if not isinstance(reference, str) or not reference.strip() or len(reference) > 2000:
            raise ValidationError("Assistant review requires a nonempty direct user instruction reference")
        provenance["authorization_reference"] = reference
    return provenance


def approve_contract(contract: dict[str, Any], expected_hash: str, *, confirm_unique_targets: bool,
                     review_origin: str = "local_operator",
                     authorization_reference: str | None = None) -> dict[str, Any]:
    _validate_contract(contract)
    if confirm_unique_targets is not True or expected_hash != contract_hash(contract):
        raise ValidationError("Review the exact contract hash and explicitly confirm unique target tools")
    provenance = _review_provenance({"origin": review_origin,
                                    "human_reviewed_exact_contract": review_origin == "local_operator",
                                    "authorization_reference": authorization_reference})
    return {"contract": deepcopy(contract), "contract_hash": expected_hash,
            "approval": {"approved": True, "unique_targets_confirmed": True, **provenance}}


def _approved(envelope: Mapping[str, Any]) -> Protocol:
    if not isinstance(envelope, Mapping) or set(envelope) != {"contract", "contract_hash", "approval"}:
        raise ValidationError("A reviewed session approval is required")
    protocol = _validate_contract(envelope["contract"])
    approval = envelope["approval"]
    if (not isinstance(approval, Mapping) or approval.get("approved") is not True
            or approval.get("unique_targets_confirmed") is not True
            or envelope["contract_hash"] != contract_hash(envelope["contract"])):
        raise ValidationError("Session approval is missing or stale")
    _review_provenance(approval)
    return protocol


def _snapshot(store: EvidenceStore, *, event: str, stage_id: str, now_ms: int) -> dict[str, Any]:
    store.advance_time(now_ms)
    checks = evaluate(store.protocol, store, now_ms=now_ms)
    return {"event": event, "stage_id": stage_id, "source_timestamp_ms": now_ms,
            "attempt_id": store.attempt_id, "revision": store.revision,
            "status": aggregate_status(checks).value, "checks": [item.to_dict() for item in checks]}


def _repair_proposal(protocol: Protocol, observations: tuple[Any, ...], snapshot: dict[str, Any]) -> dict[str, Any]:
    known = {}
    goals = {check.object_id: check.region_id for check in protocol.checks}
    for check in protocol.checks:
        result = next(item for item in snapshot["checks"] if item["check_id"] == check.id)
        observation = next((item for item in observations if item.object_id == check.object_id), None)
        if result["status"] == "unknown" or observation is None or observation.value not in protocol.regions:
            return {"event": "repair_proposal", "stage_id": snapshot["stage_id"], "hypothetical": True,
                    "request_inspection": True, "reason": "Observe every target location before planning a move"}
        known[check.object_id] = observation.value
    actions = []
    for object_id in protocol.objects:
        for source in protocol.regions:
            for target in protocol.regions:
                if source != target:
                    actions.append(RepairAction.from_dict({"id": f"move-{object_id}-{source}-to-{target}",
                        "kind": "physical", "preconditions": {object_id: source},
                        "effects": {object_id: target}, "cost": 1.0}))
        actions.append(RepairAction.from_dict({"id": f"inspect-{object_id}", "kind": "inspect",
                                              "reveals": [object_id], "cost": 0.25}))
    plan = plan_repair(known, known, actions, goals, max_states=512)
    return {"event": "repair_proposal", "stage_id": snapshot["stage_id"], "hypothetical": True,
            "request_inspection": plan is None, "plan": plan.to_dict() if plan else None,
            "action_model": "Declared move cost=1, subsequent inspection cost=0.25; physical obstacles not modeled",
            "execution_verified": False}


def _bound_result(result: Any, source_hash: str) -> None:
    if (not isinstance(result, Mapping) or not isinstance(result.get("source"), Mapping)
            or result["source"].get("sha256") != source_hash):
        raise ValidationError("Result source does not match its exact extracted frame")
    response = result.get("response")
    if (not isinstance(response, Mapping) or response.get("status") != "completed"
            or response.get("model") != SAM_MODEL
            or result.get("mask_decoding") != "official_meta_sam_parser_0.1.0"):
        raise ValidationError("Use completed SAM results with official mask decoding")


def evaluate_session(envelope: Mapping[str, Any], results: Mapping[str, Any], *,
                     inference_origin: str = "provided_results_not_verified_live") -> dict[str, Any]:
    """Replay stage-isolated observations; no API calls or predicted facts written."""
    protocol = _approved(envelope)
    contract = envelope["contract"]
    capture = contract["capture"]
    if not isinstance(results, Mapping) or set(results) - set(STAGES):
        raise ValidationError("Unexpected session result stage")
    store = EvidenceStore(protocol, "recorded-" + envelope["contract_hash"][:16])
    trace, stage_statuses, previous_end = [], {}, 0
    for index, stage in enumerate(capture["stages"]):
        stage_id = stage["id"]
        if index:
            if stage_id == "corrected":
                store.start_attempt(previous_end)
            store.invalidate("Human-declared movement, covering or correction boundary", previous_end)
            trace.append(_snapshot(store, event="invalidation", stage_id=stage_id, now_ms=previous_end))
        by_frame = results.get(stage_id, {})
        if not isinstance(by_frame, Mapping) or set(by_frame) - {frame["sha256"] for frame in stage["frames"]}:
            raise ValidationError("Unexpected source frame in session results")
        samples = []
        for frame in stage["frames"]:
            by_object = by_frame.get(frame["sha256"], {})
            if not isinstance(by_object, Mapping) or set(by_object) - set(protocol.objects):
                raise ValidationError("Unexpected target object in session results")
            for object_id, result in by_object.items():
                _bound_result(result, frame["sha256"])
                samples.append(SamImageEvidence(object_id, frame["source_timestamp_ms"], result,
                                               capture["evidence_origin"], contract["concepts"][object_id]))
        grounded = ground_sam_images(protocol, contract["regions"], samples, session_id=store.session_id,
                                     attempt_id=store.attempt_id, revision=store.revision)
        # Missing-object observations carry latest supplied time (or zero). Keep
        # that explicit unknown inside the current stage, never backdate it.
        applied = []
        for observation in grounded.observations:
            if observation.timestamp_ms < store.current_time_ms:
                observation = replace(observation, value=None, timestamp_ms=store.current_time_ms, confidence=0.0)
            store.add(observation)
            applied.append(observation)
        end = round(stage["end_seconds"] * 1000)
        snapshot = _snapshot(store, event="stage", stage_id=stage_id, now_ms=end)
        snapshot["observations"] = [item.to_dict() for item in applied]
        snapshot["frame_hashes"] = [frame["sha256"] for frame in stage["frames"]]
        trace.append(snapshot)
        stage_statuses[stage_id] = snapshot["status"]
        if stage_id == "wrong_region":
            trace.append(_repair_proposal(protocol, tuple(applied), snapshot))
        previous_end = end
    expected = dict(zip(STAGES, ("pass", "fail", "unknown", "pass")))
    gates = {f"{stage_id}_{status}": stage_statuses[stage_id] == status for stage_id, status in expected.items()}
    gates["invalidation_snapshots_unknown"] = all(item["status"] == "unknown" for item in trace if item["event"] == "invalidation")
    return {"schema_version": "1.0", "experiment": "recorded_placement_correction_session",
            "contract_hash": envelope["contract_hash"], "protocol_hash": protocol.content_hash,
            "approval_provenance": _review_provenance(envelope["approval"]),
            "inference_origin": inference_origin, "evidence_origin": capture["evidence_origin"],
            "stage_statuses": stage_statuses, "gates": gates, "trace": trace,
            "recorded_observation_loop_supported": all(gates.values()),
            "physical_action_execution_verified": False,
            "limits": ["Source ownership, unique targets, calibration and action boundaries are declared assumptions; reviewer provenance is explicit.",
                       "Local approval records an assertion, not independent proof that a person or assistant performed the stated review.",
                       "Location results depend on the approved unique-target scene contract; no exact count or absence is proved.",
                       "This evaluates selected frames of a recorded clip, not continuous tracking or an interactive repair experiment.",
                       "A plan is hypothetical; its predicted effects never create real observations.",
                       "The recorded person did not necessarily see or follow the planner's later proposal.",
                       "Passing this small scripted case is not a held-out accuracy or industrial reliability result."]}


def _inside(path: Path, directory: Path) -> Path:
    resolved, allowed = path.resolve(), directory.resolve()
    if resolved == allowed or allowed not in resolved.parents:
        raise ValidationError("Use a file or run directory inside the required private project directory")
    return resolved


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_capture_files(envelope: Mapping[str, Any], *, project_root: Path = ROOT) -> dict[str, Path]:
    _approved(envelope)
    contract = envelope["contract"]
    phone = project_root / "inputs/local/phone"
    manifest_file = _inside(project_root / contract["capture_manifest_file"], phone)
    if _read(manifest_file) != contract["capture"]:
        raise ValidationError("Capture manifest differs from the approved contract")
    capture, files = contract["capture"], {}
    source = _inside(manifest_file.parent / capture["source"]["file"], phone)
    if _file_hash(source) != capture["source"]["sha256"]:
        raise ValidationError("Original source clip changed after approval")
    for stage in capture["stages"]:
        for frame in stage["frames"]:
            path = _inside(manifest_file.parent / frame["file"], phone)
            if not 0 < path.stat().st_size <= MAX_IMAGE_BYTES or _file_hash(path) != frame["sha256"]:
                raise ValidationError("Extracted image changed or exceeds the image size limit")
            files[frame["sha256"]] = path
    return files


def _save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if path.exists():
        raise ValidationError("Refusing to replace an existing local artifact")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def _read(path: Path) -> Any:
    if path.stat().st_size > MAX_JSON_BYTES:
        raise ValidationError("Contract or manifest JSON exceeds its local size limit")
    return json.loads(path.read_text(encoding="utf-8"))


def run_session(client: MetaClient, envelope: Mapping[str, Any], output: Path, *,
                project_root: Path = ROOT, inference_origin: str = "caller_provided_client_not_verified_live") -> dict[str, Any]:
    """At most 16 new SAM calls; stop on failure, never reuse responses implicitly."""
    protocol = _approved(envelope)
    files = verify_capture_files(envelope, project_root=project_root)
    destination = _inside(output, project_root / "artifacts/local")
    if destination.exists() or client.max_requests > MAX_REQUESTS or client.retries != 0 or client.ledger:
        raise ValidationError("Use a new private run directory and a fresh capped client with zero retries")
    destination.mkdir(parents=True, mode=0o700)
    _save(destination / "approval.json", envelope)
    results: dict[str, Any] = {}
    failures = []
    for stage in envelope["contract"]["capture"]["stages"]:
        results[stage["id"]] = {}
        for frame_index, frame in enumerate(stage["frames"]):
            by_object = results[stage["id"]][frame["sha256"]] = {}
            for object_index, object_id in enumerate(protocol.objects):
                try:
                    if _file_hash(files[frame["sha256"]]) != frame["sha256"]:
                        raise ValidationError("Selected frame changed during inference")
                    result = client.segment_image(files[frame["sha256"]], envelope["contract"]["concepts"][object_id], decode_masks=True)
                    _bound_result(result, frame["sha256"])
                    by_object[object_id] = result
                    _save(destination / f"{stage['id']}-{frame_index}-{object_index}.json", result)
                except (MetaAPIError, ValueError, OSError) as error:
                    failures.append({"stage_id": stage["id"], "frame_index": frame_index,
                                     "object_id": object_id, "error_type": type(error).__name__})
                    if isinstance(error, MetaAPIError) and error.response is not None:
                        _save(destination / "failed-response.json", error.response)
                    break
            if failures:
                break
        if failures:
            break
    report = evaluate_session(envelope, results, inference_origin=inference_origin)
    try:
        verify_capture_files(envelope, project_root=project_root)
        report["gates"]["source_files_still_match"] = True
    except (ValueError, OSError, KeyError):
        failures.append({"error_type": "CaptureBindingChanged", "reason": "Original clip, manifest or frame changed during inference"})
        report["gates"]["source_files_still_match"] = False
    report["gates"]["no_inference_or_storage_failures"] = not failures
    report["recorded_observation_loop_supported"] = all(report["gates"].values())
    report.update(failures=failures, usage=client.usage_summary(), new_requests=len(client.ledger))
    _save(destination / "report.json", report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    draft = commands.add_parser("draft", help="Bind a protocol draft to reviewed clip frames and calibration")
    draft.add_argument("--capture", type=Path, required=True)
    draft.add_argument("--protocol", type=Path, required=True)
    draft.add_argument("--regions", type=Path, required=True)
    draft.add_argument("--concepts", type=Path, required=True)
    draft.add_argument("--output", type=Path, required=True)
    approve = commands.add_parser("approve", help="Record the declared reviewer's review of the exact contract")
    approve.add_argument("--draft", type=Path, required=True)
    approve.add_argument("--hash", required=True)
    approve.add_argument("--confirm-unique-targets", action="store_true")
    approve.add_argument("--review-origin", choices=REVIEW_ORIGINS, default="local_operator")
    approve.add_argument("--authorization-reference", help="Direct user instruction reference required for assistant review")
    approve.add_argument("--output", type=Path, required=True)
    run = commands.add_parser("run", help="Evaluate an approved recorded clip with real SAM image calls")
    run.add_argument("--approval", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--env-file", type=Path, default=ROOT / ".env.local")
    args = parser.parse_args(argv)
    try:
        destination = _inside(args.output, ROOT / "artifacts/local")
        if args.command == "draft":
            manifest_file = _inside(args.capture, ROOT / "inputs/local/phone")
            proposal = _read(args.protocol)
            protocol = Protocol.from_dict(proposal.get("protocol", proposal))
            contract = session_contract(_read(manifest_file), protocol, _read(args.regions), _read(args.concepts),
                                        manifest_file=str(manifest_file.relative_to(ROOT)))
            value = {"approved": False, "contract_hash": contract_hash(contract), "contract": contract}
            _save(destination, value)
            print(f"Unapproved contract hash: {value['contract_hash']}")
        elif args.command == "approve":
            value = _read(args.draft)
            envelope = approve_contract(value["contract"], args.hash, confirm_unique_targets=args.confirm_unique_targets,
                                        review_origin=args.review_origin, authorization_reference=args.authorization_reference)
            verify_capture_files(envelope)
            _save(destination, envelope)
            print("Saved approval of the exact protocol, calibration, concepts, unique targets and source frames.")
        else:
            envelope = _read(args.approval)
            verify_capture_files(envelope)
            if destination.exists():
                raise ValidationError("Choose a new private run directory")
            load_env_file(args.env_file)
            client = MetaClient.from_env(max_requests=MAX_REQUESTS, retries=0)
            report = run_session(client, envelope, destination, inference_origin="live_meta_api_on_recorded_frames")
            print(json.dumps({"stage_statuses": report["stage_statuses"], "new_requests": report["new_requests"],
                              "failures": report["failures"], "estimated_usd": report["usage"]["estimated_usd"]}))
            return 0 if report["recorded_observation_loop_supported"] else 1
    except (MetaAPIError, ValueError, OSError, KeyError, TypeError) as error:
        print(f"Session stopped: {type(error).__name__}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
