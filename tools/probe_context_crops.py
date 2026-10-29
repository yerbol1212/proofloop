"""One-shot crop-SAM proposal feasibility; no semantic or placement acceptance.

Freeze and publish public_plan before run. The caller supplies independent fresh
one-request clients sharing a persistent request allowance. No retries/resume.
"""
from copy import deepcopy
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from proofloop import benchmark, concept_verified_experiment as parent, crop_evidence as crop
from proofloop import image_gripper_experiment as inventory, prospective, prospective_audit
from proofloop.core import ValidationError
from proofloop.meta import MetaClient, SAM_MODEL

VERSION = "context_crop_sam_proposal_feasibility_v1"
SCOPE = "all_original_views_exposed_development_proposals_only"
LIMITS = ["Proposal eligibility is not task correctness, positive acceptance, identity or full visibility.",
          "All original cases and views remain; missing original proposals select no crop.",
          "One unchanged concept-SAM request per available crop; no retries, semantic calls, fallback or best-of selection.",
          "Provider confidence remains unchanged at the 0.8 gate, with two distinct source views per eligible pair.",
          "Full-source semantic, visibility, gripper and whole-region verification is still absent from this probe.",
          "A favorable exposed-development probe does not qualify production promotion or fresh-source accuracy."]


def _implementation():
    folder = Path(__file__).resolve().parents[1]
    names = {**parent._implementation(), **crop.code_hashes(),
             "image_gripper_experiment.py": benchmark._file_hash(folder / "proofloop/image_gripper_experiment.py")}
    return {**{"proofloop/" + key: value for key, value in names.items()},
            "tools/probe_context_crops.py": benchmark._file_hash(Path(__file__).resolve())}


def _prepared_file(output, index):
    return output / "frozen-frames" / ("view-" + str(index)) / "prepared.json"


def _case(index):
    return {"case_id": "view-" + str(index)}


def _path(output, index):
    return output / "parts" / ("view-" + str(index) + ".json")


def _projection_path(output, index):
    return output / "projections" / ("view-" + str(index) + ".json")


def _frame_inventory(original):
    return [(case, index) for case in original["cases"] for index in range(2)]


def freeze(parent_dir, out_dir, *, root=benchmark.ROOT, max_requests=18):
    """Bind the completed parent and prepare every original view before inference."""
    root = Path(root)
    output = benchmark._local_path(out_dir, root, output=True)
    directory, original = parent._load(parent_dir, root)
    summary = parent.summarize(directory, root=root)
    if output.exists() or output == directory or directory in output.parents or summary["status"] != "completed":
        raise ValidationError("Crop probe needs a fresh sibling directory and completed original concept trial")
    if type(max_requests) is not int or not 1 <= max_requests <= 18:
        raise ValidationError("Crop probe allows an explicit cap of one to eighteen requests")
    output.mkdir(parents=True, mode=0o700)
    frames = []
    for index, (case, frame_index) in enumerate(_frame_inventory(original)):
        target_file = parent._part_path(directory, case, frame_index)
        target = prospective_audit._read(target_file)
        prepared = crop.prepare_frame(case["envelope"], frame_index, target, case["locator"],
            _prepared_file(output, index).parent, root=root)
        frames.append({"view_index": index, "case_id": case["case_id"], "pair_frame_index": frame_index,
                       "target_response_sha256": benchmark._file_hash(target_file),
                       "prepared_file_sha256": benchmark._file_hash(_prepared_file(output, index)),
                       "prepared_sha256": prepared["prepared_sha256"],
                       "details": deepcopy(prepared["details"]), "crop_sha256": prepared["crop_sha256"]})
    planned = sum(row["details"]["crop_available"] for row in frames)
    if planned > max_requests:
        raise ValidationError("Every available original crop must fit the explicit request allowance")
    plan = {"version": VERSION, "scope": SCOPE, "parent_dir": str(directory),
            "parent_plan_sha256": original["plan_sha256"], "parent_inventory": inventory._parent_inventory(directory),
            "parent_summary_sha256": benchmark.manifest_hash(summary), "frames": frames,
            "cases": [{"case_id": case["case_id"], "contract_hash": case["envelope"]["contract_hash"],
                       "reference": case["envelope"]["contract"]["cases"][0]["annotation"]["status"],
                       "concepts": deepcopy(case["envelope"]["contract"]["concepts"])} for case in original["cases"]],
            "planned_requests": planned, "max_requests": max_requests,
            "implementation_sha256": _implementation(), "dependencies": parent._dependencies(),
            "created_unix_ms": int(time.time() * 1000), "limits": list(LIMITS)}
    plan["plan_sha256"] = benchmark.manifest_hash(plan)
    benchmark._save(output / "plan.json", plan)
    return deepcopy(plan)


def _load(out_dir, root):
    root = Path(root)
    output = benchmark._local_path(out_dir, root, output=True)
    plan = prospective_audit._read(output / "plan.json")
    if (plan.get("version") != VERSION or plan.get("scope") != SCOPE or plan.get("limits") != LIMITS
            or plan.get("implementation_sha256") != _implementation() or plan.get("dependencies") != parent._dependencies()
            or plan.get("plan_sha256") != benchmark.manifest_hash({k: v for k, v in plan.items() if k != "plan_sha256"})):
        raise ValidationError("Crop probe plan, implementation or dependency changed")
    directory, original = parent._load(Path(plan["parent_dir"]), root)
    if (original["plan_sha256"] != plan["parent_plan_sha256"]
            or inventory._parent_inventory(directory) != plan["parent_inventory"]
            or benchmark.manifest_hash(prospective_audit._read(directory / "summary.json")) != plan["parent_summary_sha256"]):
        raise ValidationError("Crop probe parent inventory or completed results changed")
    expected = _frame_inventory(original)
    cases = [{"case_id": case["case_id"], "contract_hash": case["envelope"]["contract_hash"],
              "reference": case["envelope"]["contract"]["cases"][0]["annotation"]["status"],
              "concepts": deepcopy(case["envelope"]["contract"]["concepts"])} for case in original["cases"]]
    if plan["cases"] != cases or len(plan["frames"]) != len(expected):
        raise ValidationError("Crop probe lost an original case or source view")
    prepared_files = {_prepared_file(output, i) for i in range(len(expected))}
    if set((output / "frozen-frames").rglob("*.json")) != prepared_files:
        raise ValidationError("Crop probe has a missing or unplanned immutable preparation")
    for index, (row, (case, frame_index)) in enumerate(zip(plan["frames"], expected)):
        prepared_file = _prepared_file(output, index)
        prepared = prospective_audit._read(prepared_file)
        crop.validate_prepared(prepared, root=root)
        target_file = parent._part_path(directory, case, frame_index)
        if (prepared["envelope"] != case["envelope"] or prepared["cached_locator"] != case["locator"]
                or prepared["cached_target_result"] != prospective_audit._read(target_file)
                or row != {"view_index": index, "case_id": case["case_id"], "pair_frame_index": frame_index,
                           "target_response_sha256": benchmark._file_hash(target_file),
                           "prepared_file_sha256": benchmark._file_hash(prepared_file),
                           "prepared_sha256": prepared["prepared_sha256"], "details": prepared["details"],
                           "crop_sha256": prepared["crop_sha256"]}):
            raise ValidationError("Crop probe source, parent response or immutable crop preparation changed")
    count = sum(row["details"]["crop_available"] for row in plan["frames"])
    if (type(plan["max_requests"]) is not int or not count <= plan["max_requests"] <= 18
            or type(plan["planned_requests"]) is not int or plan["planned_requests"] != count):
        raise ValidationError("Crop probe fixed allowance or available-proposal inventory changed")
    return output, plan


def public_plan(out_dir, *, root=benchmark.ROOT):
    _, plan = _load(out_dir, root)
    return {key: deepcopy(plan[key]) for key in ("version", "scope", "plan_sha256", "parent_plan_sha256",
        "parent_summary_sha256", "cases", "planned_requests", "max_requests", "implementation_sha256", "dependencies",
        "created_unix_ms", "limits")} | {"crop_rule": crop.RULE, "transform": crop.TRANSFORM,
        "parent_inventory_sha256": benchmark.manifest_hash(plan["parent_inventory"]),
        "frames": [{**{key: deepcopy(row[key]) for key in ("view_index", "case_id", "pair_frame_index", "target_response_sha256",
                    "prepared_file_sha256", "prepared_sha256", "crop_sha256")},
                    "details": {key: deepcopy(value) for key, value in row["details"].items() if key != "source_file"}}
                   for row in plan["frames"]], "task_metrics": None, "placement_acceptance": False}


def run(factory, out_dir, *, root=benchmark.ROOT):
    output, plan = _load(out_dir, root)
    parent._preflight(plan)
    try:
        with (output / "run-started.json").open("x") as stream:
            json.dump({"plan_sha256": plan["plan_sha256"]}, stream)
    except FileExistsError:
        raise ValidationError("Crop probe already started; no automatic resume") from None
    if any((output / name).exists() for name in ("parts", "projections", "run-state.json", "error-responses")):
        raise ValidationError("Crop probe run artifacts already exist")
    state = {"plan_sha256": plan["plan_sha256"], "status": "running", "attempts": {}, "admission_stop_code": None}
    benchmark._save(output / "run-state.json", state)
    seen = []
    for index, row in enumerate(plan["frames"]):
        if not row["details"]["crop_available"]:
            continue
        if state["admission_stop_code"] is not None:
            break
        _load(output, root)
        prepared = prospective_audit._read(_prepared_file(output, index))
        entry = {"status": "running", "dispatch_started": False, "usage": None}
        state["attempts"][str(index)] = entry
        benchmark._save(output / "run-state.json", state)
        client, admitted, started = None, False, time.monotonic()
        try:
            client = factory(1)
            prospective._fresh(client, 1, seen)
            admitted = True
            entry.update(dispatch_started=True, inference_origin="live_meta_adapter" if isinstance(client, MetaClient) else "injected_client_not_verified_live")
            benchmark._save(output / "run-state.json", state)
            result = client.segment_image(prepared["crop_file"], row["details"]["concept"], decode_masks=True)
            benchmark._save(_path(output, index), result)
            entry["response_sha256"] = benchmark._file_hash(_path(output, index))
            projection = crop.project_result(prepared, result, root=root)
            benchmark._save(_projection_path(output, index), projection)
            entry["projection_sha256"] = benchmark._file_hash(_projection_path(output, index))
            entry["status"] = "completed"
        except Exception as error:
            entry.update(status="error", error_type=type(error).__name__ if type(error).__name__ in prospective._ERROR_TYPES else "condition_error")
            response = getattr(error, "response", None)
            if isinstance(response, dict):
                try:
                    entry.update(parent._retain_failure(output, "sam/view-" + str(index) + "/0", response))
                except (OSError, ValueError, TypeError):
                    entry["error_response_save_failed"] = True
            if type(getattr(error, "status", None)) is int and error.status in parent.STOP_HTTP:
                state["admission_stop_code"] = error.status
        finally:
            entry["usage"] = parent._capture(client, SAM_MODEL, entry["status"]) if admitted else parent.accounting._ledger_usage([])
            if admitted:
                for attempt in client.ledger:
                    if type(attempt.get("http_status")) is int and attempt["http_status"] in parent.STOP_HTTP:
                        state["admission_stop_code"] = attempt["http_status"]
            entry["wall_seconds"] = round(time.monotonic() - started, 3)
            benchmark._save(output / "run-state.json", state)
    state["status"] = "completed" if len(state["attempts"]) == plan["planned_requests"] and all(
        entry["status"] == "completed" for entry in state["attempts"].values()) else "incomplete"
    benchmark._save(output / "run-state.json", state)
    result = summarize(output, root=root)
    benchmark._save(output / "summary.json", result)
    return result


def summarize(out_dir, *, root=benchmark.ROOT):
    output, plan = _load(out_dir, root)
    path = output / "run-state.json"
    state = prospective_audit._read(path) if path.exists() else {"plan_sha256": plan["plan_sha256"], "status": "not_started", "attempts": {}, "admission_stop_code": None}
    expected = {str(i) for i, row in enumerate(plan["frames"]) if row["details"]["crop_available"]}
    if (state.get("plan_sha256") != plan["plan_sha256"] or state.get("status") not in {"not_started", "running", "completed", "incomplete"}
            or not isinstance(state.get("attempts"), dict) or set(state["attempts"]) - expected
            or state.get("admission_stop_code") is not None and (type(state["admission_stop_code"]) is not int or state["admission_stop_code"] not in parent.STOP_HTTP)
            or state["status"] == "completed" and (set(state["attempts"]) != expected or any(e["status"] != "completed" for e in state["attempts"].values()))):
        raise ValidationError("Crop probe state contradicts its original-view declaration")
    for folder, allowed in (("parts", {_path(output, int(i)) for i in expected}),
                            ("projections", {_projection_path(output, int(i)) for i in expected}),
                            ("error-responses", {output / "error-responses/sam" / ("view-" + i) / "0.json" for i in expected})):
        if set((output / folder).rglob("*.json")) - allowed:
            raise ValidationError("Crop probe contains an unplanned response, projection or failure")
    totals = {"request_attempts": 0, "sam_requests": 0, "spark_requests": 0, "reported_input_tokens": 0, "reported_output_tokens": 0,
              "estimated_usd": 0.0, "wall_seconds": 0.0, "request_attempts_incomplete": False, "accounting_incomplete": False, "unreported_charge_possible": False}
    views = []
    for index, row in enumerate(plan["frames"]):
        entry = state["attempts"].get(str(index))
        view = {"view_index": index, "case_id": row["case_id"], "pair_frame_index": row["pair_frame_index"],
                "source_sha256": row["details"]["source_sha256"], "crop_sha256": row["crop_sha256"],
                "scores": [], "proposal_eligible": None, "withheld_reasons": []}
        if not row["details"]["crop_available"]:
            view.update(status="not_requested_no_unique_original_proposal", proposal_eligible=False,
                        withheld_reasons=["original_sam_candidate_count_not_one"])
        else:
            view.update(parent._public_attempt(output, _case(index), 0, entry, totals))
            raw_file, projection_file = _path(output, index), _projection_path(output, index)
            if entry and "response_sha256" in entry:
                if benchmark._file_hash(raw_file) != entry["response_sha256"]:
                    raise ValidationError("Crop SAM raw response changed")
                result = prospective_audit._read(raw_file)
                if entry["status"] in {"completed", "error"}:
                    parent._bind_response_usage(result, entry, SAM_MODEL)
                if entry["status"] == "completed":
                    if benchmark._file_hash(projection_file) != entry.get("projection_sha256"):
                        raise ValidationError("Crop SAM projection lacks its exact saved receipt")
                    prepared = prospective_audit._read(_prepared_file(output, index))
                    projection = prospective_audit._read(projection_file)
                    if projection["actual_crop_response"] != result:
                        raise ValidationError("Crop projection uses a different retained raw response")
                    crop.validate_projection(prepared, projection, root=root)
                    view.update(proposal_eligible=projection["status"] == "eligible_geometry",
                                scores=[record["confidence"] for record in projection["derived_full_source_geometry"]],
                                object_count=len(projection["derived_full_source_geometry"]),
                                withheld_reasons=projection["withheld_reasons"],
                                projection_sha256=projection["projection_sha256"], raw_response_sha256=entry["response_sha256"])
            elif entry and entry["status"] == "completed" or raw_file.exists() and (not entry or entry["status"] != "running"):
                raise ValidationError("Crop SAM response lacks its completed content receipt")
            if projection_file.exists() and (not entry or "projection_sha256" not in entry):
                raise ValidationError("Crop projection lacks its attempt receipt")
        views.append(view)
    if totals["request_attempts"] > plan["max_requests"] or totals["request_attempts"] > plan["planned_requests"]:
        raise ValidationError("Crop probe exceeded its declared SAM request allowance")
    for key in ("estimated_usd", "wall_seconds"):
        totals[key] = round(totals[key], 6)
    pairs = []
    for case in plan["cases"]:
        pair = [view for view in views if view["case_id"] == case["case_id"]]
        if len(pair) != 2:
            raise ValidationError("Crop probe lost an original two-view case")
        eligibility = None if any(view["proposal_eligible"] is None for view in pair) else all(view["proposal_eligible"] for view in pair)
        pairs.append({"case_id": case["case_id"], "reference": case["reference"], "proposal_pair_eligible": eligibility,
                      "placement_accepted": False, "view_indices": [view["view_index"] for view in pair]})
    positives = [pair for pair in pairs if pair["reference"] == "pass"]
    result = {**public_plan(out_dir, root=root), "status": state["status"], "views": views, "pairs": pairs, "usage": totals,
              "admission_stop_code": state["admission_stop_code"], "task_metrics": None, "placement_acceptance": False,
              "proposal_feasibility": {"total_original_views": len(views), "total_original_pairs": len(pairs),
                 "no_unique_original_proposal_views": sum(not row["details"]["crop_available"] for row in plan["frames"]),
                 "completed_crop_views": sum(view["status"] == "completed" for view in views),
                 "eligible_proposal_pairs": sum(pair["proposal_pair_eligible"] is True for pair in pairs),
                 "unscored_crop_views": sum(row["details"]["crop_available"] and view["status"] != "completed" for row, view in zip(plan["frames"], views)),
                 "reference_pass_pairs": len(positives), "reference_pass_proposal_pairs_eligible": sum(pair["proposal_pair_eligible"] is True for pair in positives),
                 "all_reference_pass_pairs_proposal_eligible": bool(positives) and all(pair["proposal_pair_eligible"] is True for pair in positives)}}
    if (output / "summary.json").exists() and prospective_audit._read(output / "summary.json") != result:
        raise ValidationError("Crop probe summary differs from retained evidence/accounting replay")
    return result
