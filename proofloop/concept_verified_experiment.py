"""One-shot, exposed-development concept SAM plus mask-semantic experiment.

This never changes live acceptance. Every original case and all original gates
remain visible; failed charged work is missing evidence, not a fake UNKNOWN.
"""
from __future__ import annotations

from copy import deepcopy
from importlib import metadata
import json
import math
from pathlib import Path
import time
from typing import Any, Callable, Mapping, Sequence

from . import anchor_padding_experiment as accounting
from . import benchmark, guided, prospective, prospective_audit
from .core import ValidationError
from .meta import MetaAPIError, MetaClient, SAM_MODEL, SPARK_MODEL
from . import meta

VERSION = "concept_sam_mask_semantics_exposed_development_v1"
STOP_HTTP = (401, 402, 403, 429)


def _semantics():
    from . import mask_semantics
    return mask_semantics


def _implementation() -> dict[str, str]:
    names = tuple(dict.fromkeys(_semantics().CODE_FILES + (
        "capture.py", "anchor_padding_experiment.py", "prospective_audit.py",
        "mask_semantics.py", "concept_verified_experiment.py")))
    folder = Path(__file__).resolve().parent
    return {name: benchmark._file_hash(folder / name) for name in names}


def _dependencies() -> dict[str, Any]:
    try:
        parser = metadata.version("meta-sam-parser")
    except metadata.PackageNotFoundError:
        parser = None
    from . import semantic
    extractor = semantic._extractor_path()
    return {"meta_sam_parser_distribution": parser,
            "ffmpeg_version": _semantics()._version(extractor),
            "sam_model": SAM_MODEL, "spark_model": SPARK_MODEL,
            "spark_reasoning_effort": "low", "spark_max_output_tokens": 4096}


def _preflight(plan: Mapping[str, Any]) -> None:
    """Resolve local dependencies before any charged request can be admitted."""
    meta._sam_decoder()
    if _dependencies() != plan["dependencies"]:
        raise ValidationError("Local parser or image extractor changed after declaration")


def _rows(envelope: Mapping[str, Any], locator: Mapping[str, Any], root: Path) -> list[dict[str, Any]]:
    rows = accounting._frames(envelope, locator, root)
    if len(envelope["contract"]["concepts"]) != 1 or len(rows) != 2:
        raise ValidationError("This experiment requires one unchanged target and two original views")
    prospective_audit._bound_text(locator, [row["source_sha256"] for row in rows],
                                  guided.locator_schema(list(envelope["contract"]["concepts"])))
    return [{key: deepcopy(value) for key, value in row.items() if key != "padded_box"} for row in rows]


def freeze(pilot_dirs: Sequence[Path], out_dir: Path, *, root: Path = benchmark.ROOT,
           max_requests: int = 27) -> dict[str, Any]:
    """Bind original completed pilots, approvals, locators, media and code before inference."""
    if (not isinstance(pilot_dirs, (list, tuple)) or not 1 <= len(pilot_dirs) <= 3
            or type(max_requests) is not int or not 1 <= max_requests <= 27):
        raise ValidationError("Use one to three original pilots and an explicit cap of 1-27")
    output = benchmark._local_path(out_dir, root, output=True)
    if output.exists():
        raise ValidationError("Concept experiment requires a fresh private directory")
    cases, pilots, seen = [], [], set()
    for location in pilot_dirs:
        directory, original, singles = prospective._load(location, root)
        prior = prospective.summarize(location, root=root)
        if prior["status"] != "completed" or "guided" not in original["conditions"]:
            raise ValidationError("Only fully completed original guided pilots may be reused")
        pilots.append({"directory": str(directory), "freeze_sha256": original["freeze_sha256"],
                       "envelope": deepcopy(original["envelope"])})
        for record in original["cases"]:
            identifier = record["case_id"]
            if identifier in seen:
                raise ValidationError("Original cases cannot repeat")
            seen.add(identifier)
            pair_file = directory / "parts" / identifier / "guided" / "guided" / (identifier + ".json")
            pair = prospective_audit._read(pair_file)
            locator = pair["locator"]
            envelope = singles[identifier]
            cases.append({"case_id": identifier, "pilot_index": len(pilots) - 1,
                          "envelope": envelope, "original_pair_file": str(pair_file),
                          "original_pair_sha256": benchmark._file_hash(pair_file),
                          "locator": deepcopy(locator), "locator_sha256": benchmark.manifest_hash(locator),
                          "frames": _rows(envelope, locator, root)})
    count = len(cases)
    if not 1 <= count <= 9 or count * 3 > max_requests:
        raise ValidationError("All original cases and diagnostic frames must fit the request cap")
    plan = {"schema_version": VERSION, "method": VERSION,
            "evaluation_role": "post_outcome_exposed_development", "cases": cases, "pilots": pilots,
            "implementation_sha256": _implementation(), "dependencies": _dependencies(),
            "request_caps": {"sam": count * 2, "spark": count, "total": max_requests},
            "sam_prompt": "unchanged_positive_target_concept_on_each_original_full_image",
            "stage_order": "all_original_sam_frames_then_eligible_pair_challenges",
            "created_unix_ms": int(time.time() * 1000)}
    plan["plan_sha256"] = benchmark.manifest_hash(plan)
    output.mkdir(parents=True, mode=0o700)
    benchmark._save(output / "plan.json", plan)
    return deepcopy(plan)


def _load(out_dir: Path, root: Path) -> tuple[Path, dict[str, Any]]:
    output = benchmark._local_path(out_dir, root, output=True)
    plan = prospective_audit._read(output / "plan.json")
    if (not isinstance(plan, dict) or plan.get("schema_version") != VERSION
            or plan.get("method") != VERSION or plan.get("evaluation_role") != "post_outcome_exposed_development"
            or plan.get("plan_sha256") != benchmark.manifest_hash({k: v for k, v in plan.items() if k != "plan_sha256"})
            or plan.get("implementation_sha256") != _implementation() or plan.get("dependencies") != _dependencies()
            or plan.get("sam_prompt") != "unchanged_positive_target_concept_on_each_original_full_image"
            or plan.get("stage_order") != "all_original_sam_frames_then_eligible_pair_challenges"):
        raise ValidationError("Concept experiment plan, code or dependencies changed")
    expected = []
    for index, pilot in enumerate(plan["pilots"]):
        _, original, _ = prospective._load(Path(pilot["directory"]), root)
        if original["freeze_sha256"] != pilot["freeze_sha256"] or original["envelope"] != pilot["envelope"]:
            raise ValidationError("Original pilot approval changed")
        expected.extend((index, case) for case in pilot["envelope"]["contract"]["cases"])
    if (not 1 <= len(plan["pilots"]) <= 3 or not 1 <= len(expected) <= 9
            or len({case["id"] for _, case in expected}) != len(expected)
            or len(plan["cases"]) != len(expected)):
        raise ValidationError("Concept experiment inventory changed")
    for record, (index, case) in zip(plan["cases"], expected):
        single = prospective._single(plan["pilots"][index]["envelope"], case)
        pair_file = benchmark._local_path(record["original_pair_file"], root, output=True)
        if (record["case_id"] != case["id"] or record["pilot_index"] != index or record["envelope"] != single
                or benchmark._file_hash(pair_file) != record["original_pair_sha256"]):
            raise ValidationError("Original case or private source response changed")
        pair = prospective_audit._read(pair_file)
        if (record["locator"] != pair["locator"]
                or record["locator_sha256"] != benchmark.manifest_hash(record["locator"])
                or record["frames"] != _rows(single, record["locator"], root)):
            raise ValidationError("Original locator, frame geometry or veto changed")
    caps = plan.get("request_caps")
    count = len(expected)
    if (not isinstance(caps, dict) or set(caps) != {"sam", "spark", "total"}
            or any(type(value) is not int for value in caps.values())
            or caps["sam"] != count * 2 or caps["spark"] != count or not count * 3 <= caps["total"] <= 27):
        raise ValidationError("Concept experiment request cap changed")
    return output, plan


def public_plan(out_dir: Path, *, root: Path = benchmark.ROOT) -> dict[str, Any]:
    _, plan = _load(out_dir, root)
    return {"schema_version": VERSION, "method": VERSION, "evaluation_role": plan["evaluation_role"],
            "plan_sha256": plan["plan_sha256"], "implementation_sha256": plan["implementation_sha256"],
            "dependencies": plan["dependencies"], "request_caps": plan["request_caps"],
            "sam_prompt": plan["sam_prompt"], "stage_order": plan["stage_order"],
            "created_unix_ms": plan["created_unix_ms"],
            "prior_freeze_hashes": [pilot["freeze_sha256"] for pilot in plan["pilots"]],
            "cases": [{"case_id": record["case_id"], "original_contract_hash": record["envelope"]["contract_hash"],
                       "original_pair_sha256": record["original_pair_sha256"],
                       "locator_sha256": record["locator_sha256"],
                       "protocol": benchmark._approved(record["envelope"]).to_dict(),
                       "concepts": deepcopy(record["envelope"]["contract"]["concepts"]),
                       "regions": deepcopy(record["envelope"]["contract"]["cases"][0]["regions"]),
                       "truth": record["envelope"]["contract"]["cases"][0]["annotation"]["status"],
                       "frames": deepcopy(record["frames"]),
                       "source_sha256": [source["sha256"] for source in record["envelope"]["contract"]["sources"]]}
                      for record in plan["cases"]],
            "physical_action_execution_verified": False,
            "limits": ["Post-outcome exposed development on every original pilot case; not fresh-source evaluation.",
                       "Original labels, source images, regions, confidence 0.8 and two-view checks remain unchanged.",
                       "Cached identity/full-visibility/single-candidate vetoes cannot be overridden.",
                       "All original frames receive diagnostic concept SAM; only eligible pairs may receive one semantic challenge.",
                       "A mask-aware Spark assessment is a model judgment, not independently verified identity.",
                       "No production promotion, hidden fallback or physical repair is provided."]}


def _attempt_key(case: Mapping[str, Any], index: int | None = None) -> str:
    return ("spark/" + case["case_id"]) if index is None else ("sam/" + case["case_id"] + "/" + str(index))


def _part_path(output: Path, case: Mapping[str, Any], index: int | None = None) -> Path:
    return output / "parts" / case["case_id"] / ("challenge.json" if index is None else "sam-" + str(index) + ".json")


def _raw_text(response: Mapping[str, Any]) -> str:
    text = meta.output_text(dict(response))
    if "output_text" in response:
        other = meta.output_text({key: value for key, value in response.items() if key != "output_text"})
        if other and other != text:
            raise ValidationError("Provider output has contradictory text lanes")
    if type(text) is not str or len(text.encode()) > benchmark.MAX_JSON_BYTES:
        raise ValidationError("Provider text exceeds the private replay limit")
    return text


def _replay_sam(result: Mapping[str, Any], case: Mapping[str, Any], row: Mapping[str, Any]) -> None:
    """Reparse retained provider text and use Meta's official mask decoder again."""
    benchmark._bound_result(result, row["source_sha256"])
    response = result.get("response")
    if (not isinstance(response, Mapping) or response.get("status") != "completed"
            or response.get("model") != SAM_MODEL or result.get("mask_decoding") != "official_meta_sam_parser_0.1.0"):
        raise ValidationError("Concept SAM requires a completed official-decoded provider response")
    try:
        text = _raw_text(response)
        decoded = meta.decode_sam_masks(meta.parse_sam_boxes(text))
    except (ValueError, TypeError, MetaAPIError) as error:
        raise ValidationError("Concept SAM provider text cannot be officially replayed") from error
    if result.get("text") != text or decoded != result.get("objects"):
        raise ValidationError("Decoded SAM objects differ from the retained provider text")
    for obj in decoded:
        box = obj["box"]
        if obj["frame"] != 0 or (box["width"], box["height"]) != (row["width"], row["height"]):
            raise ValidationError("Concept SAM object does not match the source frame")
    benchmark.evaluate_sam(case["envelope"], {case["case_id"]: {
        row["source_sha256"]: {row["object_id"]: result}}})


def _capture(client: Any, model: str, status: str) -> dict[str, Any]:
    value = accounting._capture_usage(client)
    if status == "error" and not value["ledger"]:
        # The high-level dispatch began but no durable child accounting
        # survived. Zero retained entries cannot establish zero paid work.
        value = accounting._ledger_usage([], incomplete=True)
    return _usage(value, model, status)


def _usage(value: Any, model: str, status: str) -> dict[str, Any] | None:
    usage = accounting._validated_usage(value, status)
    if usage is None:
        return None
    ledger = usage["ledger"]
    if any(entry.get("model") != model for entry in ledger):
        raise ValidationError("Attempt ledger uses a different model")
    if status == "completed" and (len(ledger) != 1 or ledger[0].get("status") != "ok"):
        raise ValidationError("A completed result requires one successful paid-attempt record")
    return usage


def _bind_response_usage(result: Mapping[str, Any], entry: Mapping[str, Any], model: str) -> None:
    response = result.get("response")
    if not isinstance(response, Mapping):
        raise ValidationError("Retained provider result has no response body")
    _bind_provider_body(response, entry, model)


def _bind_provider_body(response: Mapping[str, Any], entry: Mapping[str, Any], model: str) -> None:
    """Bind failed local parses too: a retained HTTP success cannot become a network loss."""
    _bind_provider_accounting({"model": response.get("model"), "status": response.get("status"),
                              "has_error": bool(response.get("error")),
                              "usage_sha256": benchmark.manifest_hash(meta._reported_usage(response.get("usage")))}, entry, model)


def _bind_provider_accounting(response: Mapping[str, Any], entry: Mapping[str, Any], model: str) -> None:
    usage = _usage(entry.get("usage"), model, entry["status"])
    if usage is None or len(usage["ledger"]) != 1:
        raise ValidationError("Retained provider body has no unique attempt ledger")
    ledger, = usage["ledger"]
    if response.get("usage_sha256") != benchmark.manifest_hash(meta._reported_usage(ledger.get("usage"))):
        raise ValidationError("Provider response and attempt token usage differ")
    if response.get("model") not in (None, model):
        raise ValidationError("Retained provider model differs from the admitted attempt")
    status = ledger.get("status")
    completed = response.get("status") == "completed" and response.get("has_error") is False
    if status == "ok":
        if not completed or response.get("model") != model:
            raise ValidationError("Successful attempt ledger contradicts retained provider completion")
    elif status == "invalid_response":
        if completed:
            raise ValidationError("Rejected provider completion contradicts retained successful response")
    elif status == "http_error":
        if completed or type(ledger.get("http_status")) is not int:
            raise ValidationError("HTTP failure ledger contradicts its retained response")
    else:
        raise ValidationError("A transport-loss ledger cannot contain a retained provider response")


def _retain_failure(output: Path, key: str, response: Mapping[str, Any]) -> dict[str, Any]:
    receipt = {"plan_bound_attempt": key, "provider_response_sha256": benchmark.manifest_hash(response),
               "provider_response_preserved": True, "response": deepcopy(response),
               "provider_accounting": {"model": response.get("model"), "status": response.get("status"),
                                       "has_error": bool(response.get("error")),
                                       "usage_sha256": benchmark.manifest_hash(meta._reported_usage(response.get("usage")))}}
    encoded = json.dumps(receipt, sort_keys=True, allow_nan=False).encode()
    if len(encoded) > benchmark.MAX_JSON_BYTES - 256:
        receipt = {"plan_bound_attempt": key, "provider_response_sha256": receipt["provider_response_sha256"],
                   "provider_response_preserved": False, "omission_reason": "local_size_limit", "serialized_bytes": len(encoded),
                   "provider_accounting": receipt["provider_accounting"]}
    path = output / "error-responses" / (key + ".json")
    benchmark._save(path, receipt)
    return {"error_response_sha256": benchmark._file_hash(path),
            "provider_response_sha256": receipt["provider_response_sha256"],
            "provider_response_preserved": receipt["provider_response_preserved"]}


def _validate_failure(output: Path, key: str, entry: Mapping[str, Any]) -> None:
    path = output / "error-responses" / (key + ".json")
    if "error_response_sha256" not in entry:
        if path.exists():
            raise ValidationError("Failed provider response lacks its bound receipt")
        return
    if entry["status"] != "error" or benchmark._file_hash(path) != entry["error_response_sha256"]:
        raise ValidationError("Failed provider response changed")
    receipt = prospective_audit._read(path)
    if (receipt.get("plan_bound_attempt") != key
            or receipt.get("provider_response_sha256") != entry.get("provider_response_sha256")
            or receipt.get("provider_response_preserved") is not entry.get("provider_response_preserved")):
        raise ValidationError("Failed provider response binding changed")
    if receipt["provider_response_preserved"]:
        if benchmark.manifest_hash(receipt.get("response")) != receipt["provider_response_sha256"]:
            raise ValidationError("Failed provider body contradicts its digest")
        _bind_provider_body(receipt["response"], entry, SAM_MODEL if key.startswith("sam/") else SPARK_MODEL)
    elif (receipt.get("omission_reason") != "local_size_limit" or type(receipt.get("serialized_bytes")) is not int
          or receipt["serialized_bytes"] <= benchmark.MAX_JSON_BYTES - 256):
        raise ValidationError("Failed provider omission has no bounded receipt")
    _bind_provider_accounting(receipt.get("provider_accounting", {}), entry,
                              SAM_MODEL if key.startswith("sam/") else SPARK_MODEL)


def _attempt(factory, output, plan, case, index, seen, state, root, action):
    _load(output, root)
    key, model = _attempt_key(case, index), SAM_MODEL if index is not None else SPARK_MODEL
    entry = {"status": "running", "usage": None, "dispatch_started": False}
    state["attempts"][key] = entry
    benchmark._save(output / "run-state.json", state)
    client, admitted, result = None, False, None
    started = time.monotonic()
    try:
        client = factory(1)
        prospective._fresh(client, 1, seen)
        admitted = True
        entry["inference_origin"] = "live_meta_adapter" if isinstance(client, MetaClient) else "injected_client_not_verified_live"
        entry["dispatch_started"] = True
        benchmark._save(output / "run-state.json", state)
        result = action(client)
        path = _part_path(output, case, index)
        benchmark._save(path, result)
        entry["response_sha256"] = benchmark._file_hash(path)
        if index is not None:
            _replay_sam(result, case, case["frames"][index])
        else:
            _semantics().summarize(result, root=root)
            if result.get("status") != "completed":
                # Preserve the semantic agent's complete private error report;
                # its failed generation is never a scoring UNKNOWN.
                raise ValidationError("Mask semantic challenge did not complete")
        entry["status"] = "completed"
    except Exception as error:
        entry.update(status="error", error_type=type(error).__name__ if type(error).__name__ in prospective._ERROR_TYPES else "condition_error")
        response = getattr(error, "response", None)
        if isinstance(response, Mapping):
            try:
                entry.update(_retain_failure(output, key, response))
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
            entry["usage"] = _capture(client, model, entry["status"])
        else:
            entry["usage"] = accounting._ledger_usage([])
        entry["wall_seconds"] = round(time.monotonic() - started, 3)
        benchmark._save(output / "run-state.json", state)
    return result if entry["status"] == "completed" else None


def run(client_factory: Callable[[int], Any], out_dir: Path, *, root: Path = benchmark.ROOT) -> dict[str, Any]:
    """Run diagnostic SAM first, then at most one eligible semantic challenge per pair."""
    output, plan = _load(out_dir, root)
    _preflight(plan)
    try:
        with (output / "run-started.json").open("x", encoding="utf-8") as stream:
            json.dump({"plan_sha256": plan["plan_sha256"]}, stream)
        (output / "run-started.json").chmod(0o600)
    except FileExistsError:
        raise ValidationError("Concept experiment already started; automatic resume is disabled") from None
    if any((output / name).exists() for name in ("parts", "pairs", "run-state.json", "error-responses")):
        raise ValidationError("Concept experiment output already exists")
    state = {"plan_sha256": plan["plan_sha256"], "status": "running", "attempts": {},
             "pairs": {}, "admission_stop_code": None}
    benchmark._save(output / "run-state.json", state)
    seen, sam_results = [], {}
    for case in plan["cases"]:
        by_frame = sam_results[case["case_id"]] = {}
        paths = benchmark.verify_media(case["envelope"], project_root=root)
        for index, row in enumerate(case["frames"]):
            if state["admission_stop_code"] is not None:
                break
            prompt = case["envelope"]["contract"]["concepts"][row["object_id"]]
            value = _attempt(client_factory, output, plan, case, index, seen, state, root,
                lambda client, path=paths[row["source_sha256"]], concept=prompt:
                    client.segment_image(path, concept, decode_masks=True))
            if value is not None:
                by_frame[row["source_sha256"]] = {row["object_id"]: value}
    for case in plan["cases"]:
        if state["admission_stop_code"] is not None:
            break
        if any(state["attempts"].get(_attempt_key(case, i), {}).get("status") != "completed" for i in range(2)):
            continue
        _load(output, root)
        try:
            prepared = _semantics().prepare_pair(case["envelope"], sam_results[case["case_id"]], case["locator"],
                                                output / "pairs" / case["case_id"], root=root)
        except Exception as error:
            # Source/code/approval drift still stops the whole experiment;
            # ordinary local rendering failures isolate only this pair.
            _load(output, root)
            receipt = {"case_id": case["case_id"], "plan_sha256": plan["plan_sha256"],
                       "original_contract_hash": case["envelope"]["contract_hash"],
                       "sam_response_sha256": [state["attempts"][_attempt_key(case, i)]["response_sha256"] for i in range(2)],
                       "error_type": type(error).__name__ if type(error).__name__ in prospective._ERROR_TYPES else "condition_error"}
            error_path = output / "parts" / case["case_id"] / "preparation-error.json"
            benchmark._save(error_path, receipt)
            state["pairs"][case["case_id"]] = {"status": "preparation_error", "error_sha256": benchmark._file_hash(error_path)}
            benchmark._save(output / "run-state.json", state)
            continue
        path = output / "parts" / case["case_id"] / "prepared.json"
        benchmark._save(path, prepared)
        state["pairs"][case["case_id"]] = {"prepared_sha256": benchmark._file_hash(path), "status": prepared["status"]}
        benchmark._save(output / "run-state.json", state)
        if prepared["status"] == "eligible":
            _attempt(client_factory, output, plan, case, None, seen, state, root,
                     lambda client, value=prepared: _semantics().challenge(client, value, root=root))
    state["status"] = "completed" if state["admission_stop_code"] is None and all(
        state["attempts"].get(_attempt_key(case, i), {}).get("status") == "completed"
        for case in plan["cases"] for i in range(2)) and all(
        case["case_id"] in state["pairs"] and (state["pairs"][case["case_id"]]["status"] == "withheld"
        or state["attempts"].get(_attempt_key(case), {}).get("status") == "completed") for case in plan["cases"]) else "incomplete"
    benchmark._save(output / "run-state.json", state)
    result = summarize(output, root=root)
    benchmark._save(output / "summary.json", result)
    return result


def _public_attempt(output, case, index, entry, totals):
    if entry is None:
        return {"status": "not_attempted"}
    status = entry.get("status")
    if status not in ("running", "completed", "error") or type(entry.get("dispatch_started")) is not bool:
        raise ValidationError("Concept attempt completion status is invalid")
    model = SAM_MODEL if index is not None else SPARK_MODEL
    usage = _usage(entry.get("usage"), model, status)
    if (not entry["dispatch_started"] and (status == "completed" or "response_sha256" in entry
            or "error_response_sha256" in entry or usage is not None and usage["ledger"])):
        raise ValidationError("Pre-dispatch failure cannot contain provider evidence or paid attempts")
    if entry["dispatch_started"] and usage is not None and not usage["ledger"]:
        if not usage["accounting_incomplete"]:
            raise ValidationError("A dispatched outcome without retained attempts cannot claim known zero accounting")
        totals["request_attempts_incomplete"] = True
    public = {"status": status, "inference_origin": entry.get("inference_origin") if entry.get("inference_origin")
              in ("live_meta_adapter", "injected_client_not_verified_live") else "unverified"}
    if usage is None:
        totals["request_attempts_incomplete"] = totals["accounting_incomplete"] = totals["unreported_charge_possible"] = True
        public["usage"] = {"request_attempts": None, "estimated_usd": None, "unreported_charge_possible": True}
    else:
        totals["request_attempts"] += usage["request_attempts"]
        totals["sam_requests" if index is not None else "spark_requests"] += usage["request_attempts"]
        totals["reported_input_tokens"] += usage["input_tokens"]
        totals["reported_output_tokens"] += usage["output_tokens"]
        totals["estimated_usd"] += usage["known_estimated_usd"]
        totals["accounting_incomplete"] |= usage["accounting_incomplete"]
        totals["unreported_charge_possible"] |= usage["unreported_charge_possible"] or status == "running"
        public["usage"] = {name: usage[name] for name in ("request_attempts", "estimated_usd", "known_estimated_usd",
                                                         "accounting_incomplete", "unreported_charge_possible")}
    wall = benchmark._safe_number(entry.get("wall_seconds"))
    totals["wall_seconds"] += wall or 0
    totals["accounting_incomplete"] |= wall is None
    public["wall_seconds"] = wall
    _validate_failure(output, _attempt_key(case, index), entry)
    for name in ("error_response_sha256", "provider_response_sha256", "provider_response_preserved", "error_response_save_failed"):
        if name in entry:
            public[name] = entry[name]
    if status == "error":
        public["error_type"] = entry.get("error_type") if entry.get("error_type") in prospective._ERROR_TYPES else "condition_error"
    return public


def summarize(out_dir: Path, *, root: Path = benchmark.ROOT) -> dict[str, Any]:
    """Replay exact sources, responses and ledgers; return only public-safe fields."""
    output, plan = _load(out_dir, root)
    path = output / "run-state.json"
    state = prospective_audit._read(path) if path.exists() else {"plan_sha256": plan["plan_sha256"],
        "status": "not_started", "attempts": {}, "pairs": {}, "admission_stop_code": None}
    expected = {_attempt_key(case, index) for case in plan["cases"] for index in (0, 1, None)}
    if (state.get("plan_sha256") != plan["plan_sha256"] or state.get("status") not in ("not_started", "running", "completed", "incomplete")
            or not isinstance(state.get("attempts"), Mapping) or set(state["attempts"]) - expected
            or not isinstance(state.get("pairs"), Mapping)
            or set(state["pairs"]) - {case["case_id"] for case in plan["cases"]}
            or state.get("admission_stop_code") is not None and (type(state["admission_stop_code"]) is not int
                or state["admission_stop_code"] not in STOP_HTTP)):
        raise ValidationError("Concept run state contradicts the frozen plan")
    allowed_parts = {_part_path(output, case, index) for case in plan["cases"] for index in (0, 1, None)}
    allowed_parts.update(output / "parts" / case["case_id"] / name for case in plan["cases"]
                         for name in ("prepared.json", "preparation-error.json"))
    if set((output / "parts").rglob("*.json")) - allowed_parts:
        raise ValidationError("Concept experiment has an unplanned response file")
    if set((output / "error-responses").rglob("*.json")) - {
            output / "error-responses" / (key + ".json") for key in expected}:
        raise ValidationError("Concept experiment has an unplanned failed response")
    totals = {"request_attempts": 0, "sam_requests": 0, "spark_requests": 0, "reported_input_tokens": 0,
              "reported_output_tokens": 0, "estimated_usd": 0.0, "wall_seconds": 0.0,
              "request_attempts_incomplete": False, "accounting_incomplete": False, "unreported_charge_possible": False}
    rows, predictions = [], {}
    for case in plan["cases"]:
        raw, frames, complete = {}, [], True
        for index, row in enumerate(case["frames"]):
            entry = state["attempts"].get(_attempt_key(case, index))
            frame = {**deepcopy(row), **_public_attempt(output, case, index, entry, totals), "scores": []}
            part = _part_path(output, case, index)
            if entry and "response_sha256" in entry:
                if benchmark._file_hash(part) != entry["response_sha256"]:
                    raise ValidationError("Concept SAM response changed")
                value = prospective_audit._read(part)
                if entry["status"] in ("completed", "error"):
                    # Parsing/decoding failure leaves the provider response
                    # unscored but does not erase its successful paid request.
                    _bind_response_usage(value, entry, SAM_MODEL)
                if entry["status"] == "completed":
                    _replay_sam(value, case, row)
                    raw[row["source_sha256"]] = {row["object_id"]: value}
                    frame["scores"] = [obj["confidence"] for obj in value["objects"]]
                    frame["object_count"] = len(value["objects"])
            elif entry and entry["status"] == "completed" or part.exists() and (not entry or entry["status"] != "running"):
                raise ValidationError("Concept SAM response lacks completed content binding")
            complete &= entry is not None and entry["status"] == "completed"
            frames.append(frame)
        row = {"case_id": case["case_id"], "status": "unscored", "prediction": None,
               "missing_prediction": True, "frames": frames, "semantic_challenge": {"status": "not_attempted"}}
        prepared_state = state["pairs"].get(case["case_id"])
        projected = None
        if prepared_state and prepared_state.get("status") == "preparation_error":
            if not complete or state["attempts"].get(_attempt_key(case)) is not None:
                raise ValidationError("Failed preparation cannot contain semantic inference")
            error_path = output / "parts" / case["case_id"] / "preparation-error.json"
            if benchmark._file_hash(error_path) != prepared_state.get("error_sha256"):
                raise ValidationError("Local preparation error receipt changed")
            error_receipt = prospective_audit._read(error_path)
            expected_error = {"case_id": case["case_id"], "plan_sha256": plan["plan_sha256"],
                              "original_contract_hash": case["envelope"]["contract_hash"],
                              "sam_response_sha256": [state["attempts"][_attempt_key(case, i)]["response_sha256"] for i in range(2)],
                              "error_type": error_receipt.get("error_type")}
            if error_receipt != expected_error or error_receipt["error_type"] not in prospective._ERROR_TYPES | {"condition_error"}:
                raise ValidationError("Local preparation failure lost its original evidence binding")
            row["preparation_error"] = {"error_type": error_receipt["error_type"], "receipt_sha256": prepared_state["error_sha256"]}
            prepared_state = None
        if prepared_state:
            if not complete:
                raise ValidationError("A prepared semantic pair lacks completed source SAM results")
            prepared_path = output / "parts" / case["case_id"] / "prepared.json"
            if benchmark._file_hash(prepared_path) != prepared_state.get("prepared_sha256"):
                raise ValidationError("Prepared semantic evidence changed")
            prepared = prospective_audit._read(prepared_path)
            _semantics().validate_prepared(prepared, root=root)
            if prepared.get("status") != prepared_state.get("status"):
                raise ValidationError("Prepared semantic eligibility changed")
            # Bind the independent helper back to the exact frozen case and
            # fresh SAM map; its own validator also rechecks generated assets.
            if (prepared.get("envelope") != case["envelope"] or prepared.get("sam_results") != raw
                    or prepared.get("cached_locator") != case["locator"]):
                raise ValidationError("Prepared semantic pair uses different approval or SAM evidence")
            challenge_entry = state["attempts"].get(_attempt_key(case))
            if prepared["status"] == "withheld":
                if challenge_entry is not None:
                    raise ValidationError("A cached or mask veto cannot admit a semantic challenge")
                projected = {}
                row["semantic_challenge"] = {"status": "withheld", "frames": [
                    {"frame_index": frame["frame_index"], "reasons": deepcopy(frame["withheld_reasons"])}
                    for frame in prepared["frames"]]}
            elif prepared["status"] == "eligible":
                public_attempt = _public_attempt(output, case, None, challenge_entry, totals)
                row["semantic_challenge"] = public_attempt
                challenge_path = _part_path(output, case)
                if challenge_entry and "response_sha256" in challenge_entry:
                    if benchmark._file_hash(challenge_path) != challenge_entry["response_sha256"]:
                        raise ValidationError("Mask semantic response changed")
                    report = prospective_audit._read(challenge_path)
                    if report.get("prepared") != prepared:
                        raise ValidationError("Mask semantic report uses a different prepared pair")
                    semantic = _semantics().summarize(report, root=root)
                    row["semantic_challenge"]["assessment"] = semantic
                    if (report["usage"]["ledger"] != challenge_entry["usage"]["ledger"]
                            or report["usage"]["accounting_incomplete"] != challenge_entry["usage"]["accounting_incomplete"]):
                        raise ValidationError("Semantic report and outer attempt accounting differ")
                    if challenge_entry["status"] == "completed":
                        if report.get("status") != "completed":
                            raise ValidationError("Finished semantic assessment is incomplete")
                        _bind_response_usage(report["challenger"], challenge_entry, SPARK_MODEL)
                        projected = _semantics().supported_results(report, root=root)
                elif challenge_entry and challenge_entry["status"] == "completed":
                    raise ValidationError("Finished semantic challenge has no retained response")
            else:
                raise ValidationError("Prepared semantic status is invalid")
        elif state["attempts"].get(_attempt_key(case)) is not None:
            raise ValidationError("A semantic attempt has no bound prepared evidence")
        if projected is not None:
            value, = benchmark.evaluate_sam(case["envelope"], {case["case_id"]: projected},
                                           inference_origin="concept_mask_semantics_exposed_development")
            benchmark._replay_grounding(benchmark._approved(case["envelope"]),
                                       case["envelope"]["contract"]["cases"][0], value)
            predictions.setdefault(case["pilot_index"], []).append(value)
            row.update(status="completed", missing_prediction=False, prediction=value["status"],
                       checks=[{"check_id": check["check_id"], "status": check["status"],
                               **({"gap_reason": prospective._GAPS.get(check.get("reason"), "grounding_gap")}
                                  if check["status"] == "unknown" else {})} for check in value["checks"]])
        rows.append(row)
    if state["status"] == "completed" and any(row["missing_prediction"] for row in rows):
        raise ValidationError("Completed concept state has missing predictions")
    if (totals["request_attempts"] > plan["request_caps"]["total"]
            or totals["sam_requests"] > plan["request_caps"]["sam"] or totals["spark_requests"] > plan["request_caps"]["spark"]):
        raise ValidationError("Concept experiment exceeded its frozen request cap")
    for name in ("estimated_usd", "wall_seconds"):
        totals[name] = round(totals[name], 6)
    result = {**public_plan(out_dir, root=root), "status": state["status"], "results": rows, "usage": totals,
              "admission_stop_code": state["admission_stop_code"],
              "metrics": [{"prior_freeze_sha256": pilot["freeze_sha256"],
                           "metrics": benchmark.comparison_metrics(pilot["envelope"], predictions.get(index, []))}
                          for index, pilot in enumerate(plan["pilots"])],
              "accounting_note": "Original locators/baselines are reused. New attempt costs are ledger estimates, not a provider invoice."}
    if (output / "summary.json").exists() and prospective_audit._read(output / "summary.json") != result:
        raise ValidationError("Saved concept result differs from evidence replay")
    return result
