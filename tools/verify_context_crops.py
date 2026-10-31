"""One-shot exposed-development verification of the frozen crop proposals.

No production integration, provider-response rewriting, retries or resume.
Transport lives here; source/geometry/semantic validation lives in the additive
crop_verification component. Every original pair stays in the denominator.
"""
from copy import deepcopy
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from proofloop import benchmark, concept_verified_experiment as accounting
from proofloop import image_gripper_experiment as inventory, prospective, prospective_audit
from proofloop import crop_verification as verifier
from proofloop.core import ValidationError
from proofloop.meta import MetaClient, SAM_MODEL, SPARK_MODEL
from tools import probe_context_crops as crop_probe

VERSION = "crop_full_source_contact_verification_development_v1"
LIMITS = [
    "All original exposed cases and source views remain; this is not new-source accuracy.",
    "Only pairs with two eligible frozen crop projections enter new inference.",
    "Each eligible pair permits one two-view Spark manipulator locator, at most two spatial SAM requests, and one eight-image Spark mask/contact review.",
    "No manual manipulator anchors, concept retries, favorable substitution or hidden fallback.",
    "Derived target geometry is not a provider response on the full original scene.",
    "Contact, uncertain semantics, incomplete mask coverage or unavailable required evidence cannot authorize a task result.",
    "A far gripper mask is not proof of separation; semantic assessment is fallible and not independent human truth.",
    "Completed missing or unsupported evidence yields UNKNOWN; failed required inference remains unscored.",
    "No production promotion, physical execution or calibrated safety guarantee.",
]


def _implementation():
    root = Path(__file__).resolve().parents[1]
    values = {**crop_probe._implementation(),
              **{"proofloop/" + key: value for key, value in verifier.code_hashes().items()},
              "proofloop/crop_verification.py": benchmark._file_hash(root / "proofloop/crop_verification.py"),
              "tools/verify_context_crops.py": benchmark._file_hash(Path(__file__).resolve())}
    return values


def _prepared_path(output, case):
    return output / "frozen-pairs" / case["case_id"] / "prepared.json"


def _stage_case(number, stage):
    return {"case_id": stage + "-" + str(number)}


def _key(number, stage, index=None):
    return accounting._attempt_key(_stage_case(number, stage), index)


def _path(output, key):
    return output / "parts" / (key + ".json")


def _directory_receipt(directory):
    return inventory._parent_inventory(directory) if any(directory.rglob("*")) else {}


def _parent(parent_dir, root):
    directory, plan = crop_probe._load(parent_dir, root)
    summary = crop_probe.summarize(directory, root=root)
    if summary["status"] != "completed" or summary["usage"]["accounting_incomplete"]:
        raise ValidationError("Verification needs a complete crop parent with complete accounting")
    _, original = accounting._load(Path(plan["parent_dir"]), root)
    if len(original["cases"]) != len(summary["pairs"]):
        raise ValidationError("Original pair inventory changed")
    return directory, plan, summary, original


def freeze(parent_dir, out_dir, *, root=benchmark.ROOT, max_requests=16):
    root = Path(root)
    output = benchmark._local_path(out_dir, root, output=True)
    directory, original, summary, concept = _parent(parent_dir, root)
    if (output.exists() or output == directory or directory in output.parents
            or type(max_requests) is not int or not 1 <= max_requests <= 36):
        raise ValidationError("Use a fresh sibling output and explicit allowance from one to36")
    eligible = sum(pair["proposal_pair_eligible"] is True for pair in summary["pairs"])
    if not eligible or 4 * eligible > max_requests:
        raise ValidationError("All eligible pairs must fit the explicit four-request-per-pair ceiling")
    output.mkdir(parents=True, mode=0o700)
    cases = []
    for case, pair in zip(concept["cases"], summary["pairs"]):
        if case["case_id"] != pair["case_id"]:
            raise ValidationError("Parent pair ordering changed")
        row = {"case_id": pair["case_id"], "envelope": deepcopy(case["envelope"]),
               "reference": pair["reference"], "eligible": pair["proposal_pair_eligible"] is True,
               "view_indices": list(pair["view_indices"]), "preparation_inventory": None}
        if row["eligible"]:
            items = [{"crop_prepared": prospective_audit._read(crop_probe._prepared_file(directory, i)),
                      "crop_projection": prospective_audit._read(crop_probe._projection_path(directory, i))}
                     for i in row["view_indices"]]
            prepared = verifier.prepare_pair(row["envelope"], items, _prepared_path(output, row).parent, root=root)
            benchmark._save(_prepared_path(output, row), prepared)
            row["preparation_inventory"] = inventory._parent_inventory(_prepared_path(output, row).parent)
        cases.append(row)
    plan = {"version": VERSION, "parent_dir": str(directory), "parent_plan_sha256": original["plan_sha256"],
            "parent_inventory": inventory._parent_inventory(directory),
            "parent_summary_sha256": benchmark.manifest_hash(summary), "cases": cases,
            "max_requests": max_requests, "eligible_pairs": eligible,
            "request_caps": {"spark": eligible * 2, "sam": eligible * 2, "total": max_requests},
            "implementation_sha256": _implementation(), "dependencies": accounting._dependencies(),
            "created_unix_ms": int(time.time() * 1000), "limits": list(LIMITS)}
    plan["plan_sha256"] = benchmark.manifest_hash(plan)
    benchmark._save(output / "plan.json", plan)
    _load(output, root)
    return deepcopy(plan)


def _load(out_dir, root):
    root = Path(root)
    output = benchmark._local_path(out_dir, root, output=True)
    plan = prospective_audit._read(output / "plan.json")
    if (plan.get("version") != VERSION or plan.get("limits") != LIMITS
            or plan.get("plan_sha256") != benchmark.manifest_hash({k: v for k, v in plan.items() if k != "plan_sha256"})
            or plan.get("implementation_sha256") != _implementation()
            or plan.get("dependencies") != accounting._dependencies()):
        raise ValidationError("Verification declaration, implementation or dependency changed")
    directory, parent, summary, original = _parent(plan["parent_dir"], root)
    if (plan["parent_plan_sha256"] != parent["plan_sha256"]
            or plan["parent_inventory"] != inventory._parent_inventory(directory)
            or plan["parent_summary_sha256"] != benchmark.manifest_hash(summary)
            or len(plan["cases"]) != len(original["cases"])):
        raise ValidationError("Verification parent inventory changed")
    expected_prepared = set()
    for row, case, pair in zip(plan["cases"], original["cases"], summary["pairs"]):
        if (row["case_id"] != case["case_id"] or row["case_id"] != pair["case_id"]
                or row["envelope"] != case["envelope"] or row["reference"] != pair["reference"]
                or row["eligible"] is not (pair["proposal_pair_eligible"] is True)
                or row["view_indices"] != pair["view_indices"]):
            raise ValidationError("Verification lost or relabeled an original case")
        if row["eligible"]:
            path = _prepared_path(output, row)
            if inventory._parent_inventory(path.parent) != row["preparation_inventory"]:
                raise ValidationError("Frozen verification preparation changed")
            verifier.validate_pair(prospective_audit._read(path), root=root)
            expected_prepared.add(path)
        elif row["preparation_inventory"] is not None:
            raise ValidationError("An ineligible case cannot receive a new preparation")
    if set((output / "frozen-pairs").rglob("prepared.json")) != expected_prepared:
        raise ValidationError("Verification has a missing or additional selected pair")
    count = sum(row["eligible"] for row in plan["cases"])
    if (type(plan["max_requests"]) is not int or not 4 * count <= plan["max_requests"] <= 36
            or plan["eligible_pairs"] != count or plan["request_caps"] != {
                "spark": count * 2, "sam": count * 2, "total": plan["max_requests"]}):
        raise ValidationError("Verification request allowance changed")
    return output, plan


def public_plan(out_dir, *, root=benchmark.ROOT):
    _, plan = _load(out_dir, root)
    return {key: deepcopy(plan[key]) for key in ("version", "plan_sha256", "parent_plan_sha256",
        "parent_summary_sha256", "max_requests", "eligible_pairs", "request_caps", "implementation_sha256",
        "dependencies", "created_unix_ms", "limits")} | {
        "parent_inventory_sha256": benchmark.manifest_hash(plan["parent_inventory"]),
        "cases": [{"case_id": row["case_id"], "reference": row["reference"], "eligible": row["eligible"],
                   "contract_hash": row["envelope"]["contract_hash"], "view_indices": row["view_indices"],
                   "source_sha256s": [frame["sha256"] for frame in row["envelope"]["contract"]["cases"][0]["frames"]],
                   "preparation_inventory_sha256": benchmark.manifest_hash(row["preparation_inventory"])}
                  for row in plan["cases"]], "production_promotion_allowed": False}


def _attempt(factory, output, number, stage, index, seen, state, action, validate):
    key = _key(number, stage, index)
    if key in state["attempts"]:
        raise ValidationError("A verification stage cannot be repeated")
    entry = {"status": "running", "dispatch_started": False, "usage": None}
    state["attempts"][key] = entry
    benchmark._save(output / "run-state.json", state)
    client, admitted, value = None, False, None
    started = time.monotonic()
    model = SAM_MODEL if index is not None else SPARK_MODEL
    try:
        client = factory(1)
        prospective._fresh(client, 1, seen)
        admitted = True
        entry.update(dispatch_started=True, inference_origin="live_meta_adapter" if isinstance(client, MetaClient)
                     else "injected_client_not_verified_live")
        benchmark._save(output / "run-state.json", state)
        value = action(client)
        benchmark._save(_path(output, key), value)
        entry["response_sha256"] = benchmark._file_hash(_path(output, key))
        validate(value)
        entry["status"] = "completed"
    except Exception as error:
        entry.update(status="error", error_type=type(error).__name__ if type(error).__name__ in prospective._ERROR_TYPES else "condition_error")
        response = getattr(error, "response", None)
        if response is None and isinstance(value, dict):
            response = value.get("response")
        if isinstance(response, dict):
            try:
                entry.update(accounting._retain_failure(output, key, response))
            except (OSError, ValueError, TypeError):
                entry["error_response_save_failed"] = True
        status = getattr(error, "status", None)
        if type(status) is int and status in accounting.STOP_HTTP:
            state["admission_stop_code"] = status
    finally:
        entry["usage"] = accounting._capture(client, model, entry["status"]) if admitted else accounting.accounting._ledger_usage([])
        if admitted:
            for item in client.ledger:
                if type(item.get("http_status")) is int and item["http_status"] in accounting.STOP_HTTP:
                    state["admission_stop_code"] = item["http_status"]
        entry["wall_seconds"] = round(time.monotonic() - started, 3)
        benchmark._save(output / "run-state.json", state)
    return value if entry["status"] == "completed" else None


def run(factory, out_dir, *, root=benchmark.ROOT):
    output, plan = _load(out_dir, root)
    accounting._preflight(plan)
    try:
        with (output / "run-started.json").open("x") as marker:
            json.dump({"plan_sha256": plan["plan_sha256"]}, marker)
    except FileExistsError:
        raise ValidationError("Verification already started; no resume or automatic retries") from None
    if any((output / name).exists() for name in ("parts", "final-pairs", "run-state.json", "error-responses")):
        raise ValidationError("Verification output already contains run artifacts")
    state = {"plan_sha256": plan["plan_sha256"], "status": "running", "attempts": {}, "pairs": {}, "admission_stop_code": None}
    benchmark._save(output / "run-state.json", state)
    seen = []
    for number, case in enumerate(plan["cases"]):
        if not case["eligible"] or state["admission_stop_code"] is not None:
            continue
        _load(output, root)
        prepared = prospective_audit._read(_prepared_path(output, case))
        located = _attempt(factory, output, number, "locator", None, seen, state,
            lambda client: client.spark_json(verifier.locator_prompt(prepared), verifier.locator_schema(),
                name="proofloop_crop_manipulator_locator_v1", image_paths=verifier.locator_images(prepared),
                max_output_tokens=4096, reasoning_effort="low"),
            lambda value: verifier.bind_locator(prepared, value, root=root))
        if located is None or state["admission_stop_code"] is not None:
            continue
        bound = verifier.bind_locator(prepared, located, root=root)
        grippers, failed = {}, False
        for request in verifier.spatial_prompts(prepared, bound, root=root):
            if not request["request_required"]:
                continue
            _load(output, root)
            index = request["frame_index"]
            value = _attempt(factory, output, number, "gripper", index, seen, state,
                lambda client, req=request: client.segment_image(req["source_file"], req["prompt"], decode_masks=True),
                lambda result, req=request: verifier.validate_gripper_result(prepared, bound, req["frame_index"], result, root=root))
            if value is None:
                failed = True
            else:
                grippers[index] = value
            if state["admission_stop_code"] is not None:
                break
        if failed or state["admission_stop_code"] is not None:
            continue
        _load(output, root)
        try:
            final = verifier.prepare_final(prepared, bound, grippers, output / "final-pairs" / case["case_id"], root=root)
            verifier.validate_final_stage(final, prepared, bound, grippers, root=root)
            path = output / "final-pairs" / case["case_id"] / "prepared.json"
            benchmark._save(path, final)
            state["pairs"][case["case_id"]] = {"status": "prepared", "inventory": inventory._parent_inventory(path.parent)}
            benchmark._save(output / "run-state.json", state)
        except Exception as error:
            _load(output, root)
            directory = output / "final-pairs" / case["case_id"]
            state["pairs"][case["case_id"]] = {"status": "preparation_error", "inventory": _directory_receipt(directory),
                "error_type": type(error).__name__ if type(error).__name__ in prospective._ERROR_TYPES else "condition_error"}
            benchmark._save(output / "run-state.json", state)
            continue
        _attempt(factory, output, number, "final", None, seen, state,
            lambda client: client.spark_json(verifier.final_prompt(final), verifier.final_schema(),
                name="proofloop_crop_mask_contact_review_v1", image_paths=verifier.final_images(final),
                max_output_tokens=4096, reasoning_effort="low"),
            lambda value: verifier.evaluate(final, value, root=root))
    result = summarize(output, root=root)
    state["status"] = "completed" if result["metrics"]["missing_predictions"] == 0 and state["admission_stop_code"] is None else "incomplete"
    benchmark._save(output / "run-state.json", state)
    result = summarize(output, root=root)
    benchmark._save(output / "summary.json", result)
    return result


def _metrics(rows):
    scored = [row for row in rows if row["prediction"] is not None]
    positives = [row for row in rows if row["reference"] == "pass"]
    negatives = [row for row in rows if row["reference"] != "pass"]
    decided = [row for row in scored if row["prediction"] != "unknown"]
    return {"total_cases": len(rows), "scored_cases": len(scored), "missing_predictions": len(rows) - len(scored),
            "reference_pass_cases": len(positives), "correct_acceptances": sum(row["prediction"] == "pass" for row in positives),
            "reference_nonpass_cases": len(negatives), "incorrect_passes": sum(row["prediction"] == "pass" for row in negatives),
            "correct": sum(row["reference"] == row["prediction"] for row in scored),
            "decided_cases": len(decided), "correct_decided_cases": sum(row["reference"] == row["prediction"] for row in decided),
            "unknown_cases": sum(row["prediction"] == "unknown" for row in scored)}


def summarize(out_dir, *, root=benchmark.ROOT):
    output, plan = _load(out_dir, root)
    path = output / "run-state.json"
    state = prospective_audit._read(path) if path.exists() else {"plan_sha256": plan["plan_sha256"], "status": "not_started", "attempts": {}, "pairs": {}, "admission_stop_code": None}
    expected = {_key(n, stage, index) for n, row in enumerate(plan["cases"]) if row["eligible"]
                for stage, index in (("locator", None), ("gripper", 0), ("gripper", 1), ("final", None))}
    if (state.get("plan_sha256") != plan["plan_sha256"] or state.get("status") not in {"not_started", "running", "completed", "incomplete"}
            or not isinstance(state.get("attempts"), dict) or set(state["attempts"]) - expected
            or not isinstance(state.get("pairs"), dict) or set(state["pairs"]) - {row["case_id"] for row in plan["cases"] if row["eligible"]}
            or state.get("admission_stop_code") is not None and (type(state["admission_stop_code"]) is not int or state["admission_stop_code"] not in accounting.STOP_HTTP)):
        raise ValidationError("Verification run contradicts its immutable selection")
    for folder in ("parts", "error-responses"):
        allowed = {output / folder / (key + ".json") for key in expected}
        if set((output / folder).rglob("*.json")) - allowed:
            raise ValidationError("Verification contains an unplanned raw response")
    expected_final_files = set()
    for case_id, receipt in state["pairs"].items():
        if not isinstance(receipt, dict) or receipt.get("status") not in {"prepared", "preparation_error"}:
            raise ValidationError("Final preparation status is invalid")
        directory = output / "final-pairs" / case_id
        if _directory_receipt(directory) != receipt.get("inventory"):
            raise ValidationError("Final preparation inventory differs from its run receipt")
        expected_final_files.update(directory / name for name in receipt["inventory"])
    if {p for p in (output / "final-pairs").rglob("*") if p.is_file()} != expected_final_files:
        raise ValidationError("Verification contains an unrecorded final preparation")
    totals = {"request_attempts": 0, "sam_requests": 0, "spark_requests": 0, "reported_input_tokens": 0,
              "reported_output_tokens": 0, "estimated_usd": 0.0, "wall_seconds": 0.0,
              "request_attempts_incomplete": False, "accounting_incomplete": False, "unreported_charge_possible": False}
    rows = []
    for number, case in enumerate(plan["cases"]):
        row = {"case_id": case["case_id"], "reference": case["reference"], "selected_by_parent_geometry": case["eligible"],
               "prediction": None, "status": "not_started" if state["status"] == "not_started" else "unscored", "attempts": {}}
        if not case["eligible"]:
            if state["status"] != "not_started":
                row.update(prediction="unknown", status="withheld_original_crop_gate")
            rows.append(row)
            continue
        prepared = prospective_audit._read(_prepared_path(output, case))
        raw = {}
        for stage, index in (("locator", None), ("gripper", 0), ("gripper", 1), ("final", None)):
            key = _key(number, stage, index)
            entry = state["attempts"].get(key)
            label = stage + ("-" + str(index) if index is not None else "")
            row["attempts"][label] = accounting._public_attempt(output, _stage_case(number, stage), index, entry, totals)
            file = _path(output, key)
            if entry and "response_sha256" in entry:
                if benchmark._file_hash(file) != entry["response_sha256"]:
                    raise ValidationError("Verification raw response changed")
                result = prospective_audit._read(file)
                accounting._bind_response_usage(result, entry, SAM_MODEL if index is not None else SPARK_MODEL)
                if entry["status"] == "completed":
                    raw[label] = result
            elif entry and entry["status"] == "completed" or file.exists() and (not entry or entry["status"] != "running"):
                raise ValidationError("Verification raw response lacks its receipt")
        if "locator" in raw:
            bound = verifier.bind_locator(prepared, raw["locator"], root=root)
            requests = verifier.spatial_prompts(prepared, bound, root=root)
            required = {req["frame_index"] for req in requests if req["request_required"]}
            for index in (0, 1):
                key = _key(number, "gripper", index)
                if index not in required and key in state["attempts"]:
                    raise ValidationError("Unrequested manipulator inference was admitted")
                if "gripper-" + str(index) in raw:
                    verifier.validate_gripper_result(prepared, bound, index, raw["gripper-" + str(index)], root=root)
            receipt = state["pairs"].get(case["case_id"])
            if receipt and receipt["status"] == "prepared":
                final_path = output / "final-pairs" / case["case_id"] / "prepared.json"
                if inventory._parent_inventory(final_path.parent) != receipt["inventory"]:
                    raise ValidationError("Final verification preparation changed")
                final = prospective_audit._read(final_path)
                verifier.validate_final_stage(final, prepared, bound,
                    {i: raw["gripper-" + str(i)] for i in required if "gripper-" + str(i) in raw}, root=root)
                if not required.issubset({i for i in (0, 1) if "gripper-" + str(i) in raw}):
                    raise ValidationError("Final review bypassed a required manipulator result")
                if "final" in raw:
                    evaluation = verifier.evaluate(final, raw["final"], root=root)
                    report = verifier.derive_task_report(evaluation, root=root)
                    row.update(status="completed", prediction=report["overall"],
                               assessment=verifier.public_evaluation(evaluation, root=root), task_report=report)
            elif "final" in raw:
                raise ValidationError("Final review lacks its source-bound preparation")
        elif any(key != "locator" for key in raw):
            raise ValidationError("Later verification stage bypassed its completed locator")
        rows.append(row)
    if (totals["request_attempts"] > plan["max_requests"] or totals["sam_requests"] > plan["request_caps"]["sam"]
            or totals["spark_requests"] > plan["request_caps"]["spark"]):
        raise ValidationError("Verification exceeded its declared request caps")
    for key in ("estimated_usd", "wall_seconds"):
        totals[key] = round(totals[key], 6)
    metrics = _metrics(rows)
    if state["status"] == "completed" and (metrics["missing_predictions"] or state["admission_stop_code"] is not None):
        raise ValidationError("Incomplete verification cannot claim completion")
    result = {**public_plan(out_dir, root=root), "status": state["status"], "results": rows, "metrics": metrics,
              "usage": totals, "admission_stop_code": state["admission_stop_code"],
              "gate_passed": state["status"] == "completed" and not totals["accounting_incomplete"]
                  and not totals["request_attempts_incomplete"] and not totals["unreported_charge_possible"]
                  and metrics["missing_predictions"] == 0 and metrics["correct"] == len(rows)
                  and metrics["correct_acceptances"] == metrics["reference_pass_cases"] and metrics["reference_pass_cases"] > 0,
              "changes_production_acceptance": False, "physical_action_execution_verified": False}
    # The unchanged grounding bridge exposes coordinate tuples in memory.
    # Public records use JSON arrays, so replay must compare that exact wire
    # representation rather than tuple/list implementation details.
    result = json.loads(json.dumps(result, allow_nan=False))
    if (output / "summary.json").exists() and prospective_audit._read(output / "summary.json") != result:
        raise ValidationError("Verification summary differs from retained evidence")
    return result
