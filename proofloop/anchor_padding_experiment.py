"""Exposed-development SAM anchor ablation; never a live acceptance fallback.

Reuses previously measured Spark locators and labels. New SAM requests use only
a deterministic padded box on the original image. This is post-outcome tuning,
not a fresh-source accuracy result or independent semantic verification.
"""
from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
import time
from typing import Any, Callable, Mapping, Sequence

from . import benchmark, guided, prospective
from .core import ValidationError
from .meta import MetaAPIError, MetaClient

VERSION = "cached_locator_image_margin_sam_development_v1"
STOP_HTTP = (401, 402, 403, 429)
PADDING_RULE = "ceil(max(source_width,source_height)/20) pixels symmetrically, clamped inclusive bounds"


def _implementation() -> dict[str, str]:
    folder = Path(__file__).resolve().parent
    names = prospective.CODE_FILES + ("capture.py", "anchor_padding_experiment.py")
    return {name: benchmark._file_hash(folder / name) for name in names}


def padded_box(box: Mapping[str, int], width: int, height: int) -> dict[str, int]:
    """Fixed ceil(5% of larger image dimension), inclusive pixels, clipped."""
    if (type(width) is not int or type(height) is not int or not 1 <= width <= 1280 or not 1 <= height <= 1280
            or not isinstance(box, Mapping) or set(box) != {"x1", "y1", "x2", "y2"}
            or any(type(value) is not int for value in box.values())
            or not 0 <= box["x1"] <= box["x2"] < width or not 0 <= box["y1"] <= box["y2"] < height):
        raise ValidationError("Padding requires a bounded source-pixel box")
    margin = (max(width, height) + 19) // 20
    return {"x1": max(0, box["x1"] - margin), "y1": max(0, box["y1"] - margin),
            "x2": min(width - 1, box["x2"] + margin), "y2": min(height - 1, box["y2"] + margin)}


def _frames(envelope: Mapping[str, Any], locator: Mapping[str, Any], root: Path) -> list[dict[str, Any]]:
    paths = benchmark.verify_media(envelope, project_root=root)
    case, = envelope["contract"]["cases"]
    images = [paths[frame["sha256"]] for frame in case["frames"]]
    _, frames = guided._frames(images)
    parsed = guided._locator_bound(locator, frames, envelope["contract"]["concepts"])
    result = []
    for frame, located in zip(frames, parsed):
        for target in located["objects"]:
            candidates = target["candidates"]
            reason = ("identity_ambiguous" if target["identity_ambiguous"] else
                      "not_fully_visible" if not target["fully_visible"] else
                      "no_candidate" if not candidates else "multiple_candidates" if len(candidates) != 1 else None)
            original = None if reason else guided.normalized_box_to_pixels(candidates[0], frame["width"], frame["height"])
            result.append({"frame_index": frame["frame_index"], "object_id": target["object_id"],
                           "source_sha256": frame["sha256"], "width": frame["width"], "height": frame["height"],
                           "eligible": reason is None, "withheld_reason": reason, "original_box": original,
                           "padded_box": padded_box(original, frame["width"], frame["height"]) if original else None})
    return result


def freeze(pilot_dirs: Sequence[Path], out_dir: Path, *, root: Path = benchmark.ROOT,
           max_requests: int = 18) -> dict[str, Any]:
    """Snapshot all original plans/locators before any padded SAM inference."""
    if (not isinstance(pilot_dirs, (list, tuple)) or not 1 <= len(pilot_dirs) <= 3
            or type(max_requests) is not int or not 1 <= max_requests <= 18):
        raise ValidationError("Use one to three original pilots and a cap of 1-18 SAM attempts")
    output = benchmark._local_path(out_dir, root, output=True)
    if output.exists():
        raise ValidationError("Padding freeze needs a new private directory")
    cases, pilots, seen = [], [], set()
    for location in pilot_dirs:
        original_dir, frozen, singles = prospective._load(location, root)
        prior = prospective.summarize(location, root=root)  # Replays original private responses/accounting.
        if prior["status"] != "completed" or "guided" not in frozen["conditions"]:
            raise ValidationError("Padding ablation requires a fully completed original guided pilot")
        pilots.append({"directory": str(original_dir), "freeze_sha256": frozen["freeze_sha256"],
                       "envelope": deepcopy(frozen["envelope"])})
        for record in frozen["cases"]:
            identifier = record["case_id"]
            if identifier in seen:
                raise ValidationError("Original pilots cannot repeat a case")
            seen.add(identifier)
            pair_path = original_dir / "parts" / identifier / "guided" / "guided" / (identifier + ".json")
            pair = prospective._read(pair_path)
            locator = pair["locator"]
            envelope = singles[identifier]
            rows = _frames(envelope, locator, root)
            cases.append({"case_id": identifier, "pilot_index": len(pilots) - 1, "envelope": envelope,
                          "original_pair_file": str(pair_path), "original_pair_sha256": benchmark._file_hash(pair_path),
                          "locator": deepcopy(locator), "locator_sha256": benchmark.manifest_hash(locator), "frames": rows})
    planned = sum(row["eligible"] for case in cases for row in case["frames"])
    if len(cases) > 9 or planned > max_requests:
        raise ValidationError("Padding plan exceeds its case or paid-attempt cap")
    body = {"schema_version": VERSION, "method": VERSION, "evaluation_role": "post_outcome_exposed_development",
            "padding_rule": PADDING_RULE,
            "implementation_sha256": _implementation(), "pilots": pilots, "cases": cases,
            "max_requests": max_requests, "planned_requests": planned,
            "created_unix_ms": int(time.time() * 1000)}
    body["plan_sha256"] = benchmark.manifest_hash(body)
    output.mkdir(parents=True, mode=0o700)
    benchmark._save(output / "plan.json", body)
    return deepcopy(body)


def _load(out_dir: Path, root: Path) -> tuple[Path, dict[str, Any]]:
    output = benchmark._local_path(out_dir, root, output=True)
    plan = prospective._read(output / "plan.json")
    if (not isinstance(plan, dict) or plan.get("schema_version") != VERSION
            or plan.get("method") != VERSION or plan.get("evaluation_role") != "post_outcome_exposed_development"
            or plan.get("padding_rule") != PADDING_RULE
            or plan.get("plan_sha256") != benchmark.manifest_hash({k: v for k, v in plan.items() if k != "plan_sha256"})
            or plan.get("implementation_sha256") != _implementation()):
        raise ValidationError("Padding plan or its implementation changed")
    count, seen = 0, set()
    for pilot in plan["pilots"]:
        _, original, _ = prospective._load(Path(pilot["directory"]), root)
        if original["freeze_sha256"] != pilot["freeze_sha256"] or original["envelope"] != pilot["envelope"]:
            raise ValidationError("Original pilot approval changed")
    expected = [case for pilot in plan["pilots"] for case in pilot["envelope"]["contract"]["cases"]]
    if [case["case_id"] for case in plan["cases"]] != [case["id"] for case in expected]:
        raise ValidationError("Padding case inventory changed")
    for case in plan["cases"]:
        if case["case_id"] in seen:
            raise ValidationError("Padding plan repeats a case")
        seen.add(case["case_id"])
        pilot = plan["pilots"][case["pilot_index"]]
        single = prospective._single(pilot["envelope"], next(c for c in pilot["envelope"]["contract"]["cases"] if c["id"] == case["case_id"]))
        pair_path = benchmark._local_path(case["original_pair_file"], root, output=True)
        if case["envelope"] != single or benchmark._file_hash(pair_path) != case["original_pair_sha256"]:
            raise ValidationError("Original case or locator pair changed")
        pair = prospective._read(pair_path)
        if pair["locator"] != case["locator"] or benchmark.manifest_hash(case["locator"]) != case["locator_sha256"]:
            raise ValidationError("Cached locator changed")
        if case["frames"] != _frames(single, case["locator"], root):
            raise ValidationError("Padding anchors or eligibility changed")
        count += sum(row["eligible"] for row in case["frames"])
    if (type(plan["max_requests"]) is not int or not 1 <= plan["max_requests"] <= 18
            or not 1 <= len(plan["cases"]) <= 9 or count != plan["planned_requests"] or count > plan["max_requests"]):
        raise ValidationError("Padding plan request allowance changed")
    return output, plan


def public_plan(out_dir: Path, *, root: Path = benchmark.ROOT) -> dict[str, Any]:
    _, plan = _load(out_dir, root)
    return {"schema_version": VERSION, "method": VERSION, "evaluation_role": plan["evaluation_role"],
            "plan_sha256": plan["plan_sha256"], "implementation_sha256": plan["implementation_sha256"],
            "padding_rule": plan["padding_rule"], "planned_requests": plan["planned_requests"],
            "max_requests": plan["max_requests"], "created_unix_ms": plan["created_unix_ms"],
            "prior_freeze_hashes": [pilot["freeze_sha256"] for pilot in plan["pilots"]],
            "cases": [{"case_id": case["case_id"], "original_contract_hash": case["envelope"]["contract_hash"],
                       "locator_sha256": case["locator_sha256"], "original_pair_sha256": case["original_pair_sha256"],
                       "source_sha256": [source["sha256"] for source in case["envelope"]["contract"]["sources"]],
                       "truth": case["envelope"]["contract"]["cases"][0]["annotation"]["status"]
                                if case["envelope"]["contract"]["cases"][0]["annotation"] else None,
                       "frames": deepcopy(case["frames"])} for case in plan["cases"]],
            "limits": ["Post-outcome exposed development ablation using cached Spark locators; no fresh-source accuracy claim.",
                       "Identity/visibility gates, labels, region calibration and confidence/stability thresholds are unchanged.",
                       "Padded SAM masks do not independently prove semantic identity; larger anchors can include distractors.",
                       "No live acceptance promotion, implicit fallback or verified physical correction is provided."]}


def _key(case: Mapping[str, Any], row: Mapping[str, Any]) -> str:
    return case["case_id"] + "/" + str(row["frame_index"]) + "/" + row["object_id"]


def _response_path(output: Path, case: Mapping[str, Any], row: Mapping[str, Any]) -> Path:
    return output / "responses" / case["case_id"] / (str(row["frame_index"]) + "-" + row["object_id"] + ".json")


def _error_response_path(output: Path, case: Mapping[str, Any], row: Mapping[str, Any]) -> Path:
    return output / "error-responses" / case["case_id"] / (str(row["frame_index"]) + "-" + row["object_id"] + ".json")


def _retain_error_response(output: Path, case: Mapping[str, Any], row: Mapping[str, Any], response: Mapping[str, Any]) -> dict[str, Any]:
    """Keep failed charged provider output privately; never turn it into masks."""
    receipt = {"source_sha256": row["source_sha256"], "provider_response_sha256": benchmark.manifest_hash(response),
               "provider_response_preserved": True, "response": deepcopy(response)}
    encoded = json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False).encode()
    # Match the existing private reader bound. Excessive provider bodies retain
    # an explicit digest/size receipt, with no claim that the full body survived.
    if len(encoded) + 1 > benchmark.MAX_JSON_BYTES:
        receipt = {"source_sha256": row["source_sha256"], "provider_response_sha256": receipt["provider_response_sha256"],
                   "provider_response_preserved": False, "omission_reason": "local_size_limit", "serialized_bytes": len(encoded) + 1}
    path = _error_response_path(output, case, row)
    benchmark._save(path, receipt)
    return {"error_response_sha256": benchmark._file_hash(path),
            "provider_response_sha256": receipt["provider_response_sha256"],
            "provider_response_preserved": receipt["provider_response_preserved"]}


def _ledger_usage(ledger: Any, *, incomplete: bool = False) -> dict[str, Any]:
    if (not isinstance(ledger, list) or len(ledger) > 1 or any(not isinstance(value, Mapping) for value in ledger)):
        raise ValidationError("Padding accounting needs its bounded private attempt ledger")
    raw = MetaClient.usage_summary(type("Ledger", (), {"ledger": ledger})())
    usage = prospective._safe_usage(raw, ledger)
    known = usage["estimated_usd"]
    if incomplete:
        usage.update(estimated_usd=None, unreported_charge_possible=True)
    return {**usage, "known_estimated_usd": known, "accounting_incomplete": incomplete, "ledger": deepcopy(ledger)}


def _capture_usage(client: Any) -> dict[str, Any]:
    ledger = deepcopy(client.ledger)
    try:
        prospective._usage(client)  # Reject missing or contradictory client totals.
    except Exception:
        return _ledger_usage(ledger, incomplete=True)
    return _ledger_usage(ledger)


def _validated_usage(value: Any, status: str) -> dict[str, Any] | None:
    if value is None and status == "running":
        return None  # Process died before a ledger snapshot: attempts/cost are unknown.
    if not isinstance(value, Mapping) or type(value.get("accounting_incomplete")) is not bool:
        raise ValidationError("Finished padding attempts require explicit ledger accounting")
    expected = _ledger_usage(value.get("ledger"), incomplete=value["accounting_incomplete"])
    if (set(value) != set(expected) or value != expected
            or any(type(value[name]) is not type(item) for name, item in expected.items())):
        raise ValidationError("Padding accounting differs from its private attempt ledger")
    return expected


def _prompt(row: Mapping[str, Any]) -> str:
    box = row["padded_box"]
    return (f"<0f>1<|box;x1={box['x1']};y1={box['y1']};x2={box['x2']};y2={box['y2']};"
            f"w={row['width']};h={row['height']}|>")


def run(client_factory: Callable[[int], Any], out_dir: Path, *, root: Path = benchmark.ROOT) -> dict[str, Any]:
    output, plan = _load(out_dir, root)
    try:
        with (output / "run-started.json").open("x", encoding="utf-8") as stream:
            json.dump({"plan_sha256": plan["plan_sha256"]}, stream)
        (output / "run-started.json").chmod(0o600)
    except FileExistsError:
        raise ValidationError("Padding experiment already started; automatic resume is disabled") from None
    if (output / "run-state.json").exists() or (output / "responses").exists():
        raise ValidationError("Padding experiment outputs already exist")
    state = {"plan_sha256": plan["plan_sha256"], "status": "running", "attempts": {}, "admission_stop_code": None}
    benchmark._save(output / "run-state.json", state)
    seen = []
    for case in plan["cases"]:
        for row in case["frames"]:
            if not row["eligible"] or state["admission_stop_code"] is not None:
                continue
            _load(out_dir, root)  # Recheck every source, approval, locator and implementation before dispatch.
            key = _key(case, row)
            entry = {"status": "running", "usage": None}
            state["attempts"][key] = entry
            benchmark._save(output / "run-state.json", state)
            client, admitted = None, False
            started = time.monotonic()
            try:
                client = client_factory(1)
                prospective._fresh(client, 1, seen)
                admitted = True
                entry["inference_origin"] = "live_meta_adapter" if isinstance(client, MetaClient) else "injected_client_not_verified_live"
                paths = benchmark.verify_media(case["envelope"], project_root=root)
                result = client.segment_image(paths[row["source_sha256"]], _prompt(row), decode_masks=True)
                destination = _response_path(output, case, row)
                benchmark._save(destination, result)  # Preserve invalid charged responses for private diagnosis.
                entry["response_sha256"] = benchmark._file_hash(destination)
                guided._sam_bound(result, {"sha256": row["source_sha256"], "width": row["width"], "height": row["height"]})
                # Validate decoded raster/confidence through the unchanged bridge
                # before marking this charged response usable. One frame is not
                # treated as a two-view prediction; the final replay uses both.
                benchmark.evaluate_sam(case["envelope"], {case["case_id"]: {
                    row["source_sha256"]: {row["object_id"]: result}}})
                benchmark.verify_media(case["envelope"], project_root=root)
                entry["status"] = "completed"
            except Exception as error:
                entry.update(status="error", error_type=type(error).__name__ if type(error).__name__ in prospective._ERROR_TYPES else "condition_error")
                if isinstance(error, MetaAPIError) and isinstance(error.response, Mapping):
                    try:
                        entry.update(_retain_error_response(output, case, row, error.response))
                    except (OSError, ValueError, TypeError):
                        entry["error_response_save_failed"] = True
                status = getattr(error, "status", None)
                if type(status) is int and status in STOP_HTTP:
                    state["admission_stop_code"] = status
            finally:
                if admitted:
                    for attempt in client.ledger:
                        status = attempt.get("http_status")
                        if type(status) is int and status in STOP_HTTP:
                            state["admission_stop_code"] = status
                            break
                    entry["usage"] = _capture_usage(client)
                else:
                    entry["usage"] = _ledger_usage([])
                entry["wall_seconds"] = round(time.monotonic() - started, 3)
                benchmark._save(output / "run-state.json", state)
    state["status"] = "completed" if (len(state["attempts"]) == plan["planned_requests"]
        and all(entry["status"] == "completed" for entry in state["attempts"].values())) else "incomplete"
    benchmark._save(output / "run-state.json", state)
    result = summarize(out_dir, root=root)
    benchmark._save(output / "summary.json", result)
    return result


def summarize(out_dir: Path, *, root: Path = benchmark.ROOT) -> dict[str, Any]:
    output, plan = _load(out_dir, root)
    state_file = output / "run-state.json"
    state = prospective._read(state_file) if state_file.exists() else {
        "plan_sha256": plan["plan_sha256"], "status": "not_started", "attempts": {}, "admission_stop_code": None}
    if state.get("plan_sha256") != plan["plan_sha256"] or state.get("status") not in ("not_started", "running", "completed", "incomplete"):
        raise ValidationError("Padding run state does not match its plan")
    if state.get("admission_stop_code") is not None and (type(state["admission_stop_code"]) is not int or state["admission_stop_code"] not in STOP_HTTP):
        raise ValidationError("Padding admission stop code is invalid")
    expected = {_key(case, row) for case in plan["cases"] for row in case["frames"] if row["eligible"]}
    if not isinstance(state.get("attempts"), Mapping) or set(state["attempts"]) - expected:
        raise ValidationError("Padding run contains an unplanned attempt")
    if state["status"] == "completed" and (set(state["attempts"]) != expected
            or any(entry.get("status") != "completed" for entry in state["attempts"].values())):
        raise ValidationError("Completed padding state omits planned evidence")
    expected_paths = {_response_path(output, case, row) for case in plan["cases"] for row in case["frames"] if row["eligible"]}
    if set((output / "responses").rglob("*.json")) - expected_paths:
        raise ValidationError("Padding experiment contains an unplanned response")
    expected_errors = {_error_response_path(output, case, row) for case in plan["cases"] for row in case["frames"] if row["eligible"]}
    if set((output / "error-responses").rglob("*.json")) - expected_errors:
        raise ValidationError("Padding experiment contains an unplanned error response")
    public = public_plan(out_dir, root=root)
    predictions, case_rows = {}, []
    totals = {"request_attempts": 0, "reported_input_tokens": 0, "reported_output_tokens": 0,
              "estimated_usd": 0.0, "unreported_charge_possible": False, "accounting_incomplete": False,
              "request_attempts_incomplete": False, "wall_seconds": 0.0}
    for case in plan["cases"]:
        raw, rows, complete = {}, [], True
        for row in case["frames"]:
            frame = deepcopy(row)
            frame.update(status="withheld" if not row["eligible"] else "not_attempted", scores=[])
            entry = state["attempts"].get(_key(case, row))
            if row["eligible"] and not entry:
                complete = False
            if entry:
                if entry.get("status") not in ("running", "completed", "error"):
                    raise ValidationError("Padding attempt has an invalid completion status")
                frame.update(status=entry["status"], inference_origin=entry.get("inference_origin") if entry.get("inference_origin")
                             in ("live_meta_adapter", "injected_client_not_verified_live") else "unverified")
                path = _response_path(output, case, row)
                if "response_sha256" in entry:
                    if benchmark._file_hash(path) != entry["response_sha256"]:
                        raise ValidationError("Saved padded SAM response changed")
                    result = prospective._read(path)
                    if entry["status"] == "completed":
                        guided._sam_bound(result, {"sha256": row["source_sha256"], "width": row["width"], "height": row["height"]})
                        raw.setdefault(row["source_sha256"], {})[row["object_id"]] = result
                        frame["scores"] = [score if type(score := obj.get("confidence")) in (int, float)
                                           and math.isfinite(score) and 0 <= score <= 1 else None
                                           for obj in result["objects"]]
                elif entry["status"] == "completed":
                    raise ValidationError("Completed padded SAM response is missing")
                elif path.exists() and entry["status"] != "running":
                    raise ValidationError("A finished padding response lacks its content hash")
                error_path = _error_response_path(output, case, row)
                if "error_response_sha256" in entry:
                    if entry["status"] != "error" or benchmark._file_hash(error_path) != entry["error_response_sha256"]:
                        raise ValidationError("Saved failed provider response changed")
                    receipt = prospective._read(error_path)
                    if (receipt.get("source_sha256") != row["source_sha256"]
                            or receipt.get("provider_response_sha256") != entry.get("provider_response_sha256")
                            or receipt.get("provider_response_preserved") is not entry.get("provider_response_preserved")):
                        raise ValidationError("Failed provider response lost its original source binding")
                    if receipt["provider_response_preserved"]:
                        if benchmark.manifest_hash(receipt.get("response")) != receipt["provider_response_sha256"]:
                            raise ValidationError("Failed provider response body differs from its digest")
                    elif receipt.get("omission_reason") != "local_size_limit" or type(receipt.get("serialized_bytes")) is not int or receipt["serialized_bytes"] <= benchmark.MAX_JSON_BYTES:
                        raise ValidationError("Failed provider response omission lacks a bounded receipt")
                    frame.update(error_response_sha256=entry["error_response_sha256"],
                                 provider_response_sha256=entry["provider_response_sha256"],
                                 provider_response_preserved=entry["provider_response_preserved"])
                elif error_path.exists():
                    raise ValidationError("Failed provider response lacks a bound content hash")
                if entry.get("error_response_save_failed") is True:
                    frame["error_response_save_failed"] = True
                if entry["status"] == "error":
                    frame["error_type"] = entry.get("error_type") if entry.get("error_type") in prospective._ERROR_TYPES else "condition_error"
                complete &= entry["status"] == "completed"
                usage = _validated_usage(entry.get("usage"), entry["status"])
                if usage is None:
                    attempts, cost, known_cost = None, None, 0
                    uncertain, incomplete = True, True
                    totals["request_attempts_incomplete"] = True
                else:
                    attempts, cost, known_cost = usage["request_attempts"], usage["estimated_usd"], usage["known_estimated_usd"]
                    uncertain, incomplete = usage["unreported_charge_possible"], usage["accounting_incomplete"]
                    totals["reported_input_tokens"] += usage["input_tokens"]
                    totals["reported_output_tokens"] += usage["output_tokens"]
                totals["request_attempts"] += attempts or 0
                wall = benchmark._safe_number(entry.get("wall_seconds"))
                totals["estimated_usd"] += known_cost
                totals["wall_seconds"] += wall or 0
                totals["accounting_incomplete"] |= incomplete or wall is None
                totals["unreported_charge_possible"] |= uncertain or entry["status"] == "running"
                frame["usage"] = {"request_attempts": attempts, "estimated_usd": cost, "known_estimated_usd": known_cost,
                                  "unreported_charge_possible": uncertain, "accounting_incomplete": incomplete}
                frame["wall_seconds"] = wall
            rows.append(frame)
        envelope = case["envelope"]
        case_row = {"case_id": case["case_id"], "status": "completed" if complete else "unscored",
                    "missing_prediction": not complete, "prediction": None, "frames": rows}
        if complete:
            value, = benchmark.evaluate_sam(envelope, {case["case_id"]: raw}, inference_origin="cached_locator_padded_anchor_development")
            benchmark._replay_grounding(benchmark._approved(envelope), envelope["contract"]["cases"][0], value)
            predictions.setdefault(case["pilot_index"], []).append(value)
            case_row.update(prediction=value["status"], checks=[{"check_id": item["check_id"], "status": item["status"],
                           **({"gap_reason": prospective._GAPS.get(item.get("reason"), "grounding_gap")} if item["status"] == "unknown" else {})}
                           for item in value["checks"]])
        case_rows.append(case_row)
    if totals["request_attempts"] > plan["max_requests"]:
        raise ValidationError("Padding experiment exceeded its global request cap")
    for name in ("estimated_usd", "wall_seconds"):
        totals[name] = round(totals[name], 6)
    public.update(status=state["status"], results=case_rows, usage=totals, admission_stop_code=state.get("admission_stop_code"),
        metrics=[{"prior_freeze_sha256": pilot["freeze_sha256"],
                  "metrics": benchmark.comparison_metrics(pilot["envelope"], predictions.get(index, []))}
                 for index, pilot in enumerate(plan["pilots"])],
        accounting_note="Original Spark locator/other baseline costs are reused, not newly incurred; SAM cost estimates can exclude ambiguous charges.")
    if (output / "summary.json").exists() and prospective._read(output / "summary.json") != public:
        raise ValidationError("Saved padded-anchor scores or accounting differ from replay")
    return public
