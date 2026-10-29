"""One-shot image-gripper warning follow-up to the exposed concept experiment.

No target inference, judgment or original task label is changed. Independently
prompted gripper proposals can only withhold a parent decision. A far mask is
not positive evidence of identity, complete visibility, absence or safety.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import time
from typing import Any, Callable, Mapping

from . import benchmark, concept_verified_experiment as parent, prospective, prospective_audit
from .core import ValidationError
from .meta import MetaAPIError, MetaClient, SAM_MODEL

VERSION = "source_image_gripper_negative_warning_development_v1"
CONCEPT = "robot gripper"
ANNOTATIONS = {"near_gripper_proposal", "no_near_proposal_observed"}


def _veto():
    from . import image_gripper_veto
    return image_gripper_veto


def _implementation() -> dict[str, str]:
    names = tuple(dict.fromkeys(tuple(parent._implementation()) + _veto().CODE_FILES + ("image_gripper_experiment.py",)))
    folder = Path(__file__).resolve().parent
    return {name: benchmark._file_hash(folder / name) for name in names}


def _parent_inventory(directory: Path) -> dict[str, str]:
    files = sorted(path for path in directory.rglob("*") if path.is_file())
    if not files or len(files) > 256 or sum(path.stat().st_size for path in files) > 128 * 1024 * 1024:
        raise ValidationError("Parent experiment artifact inventory is unavailable or excessive")
    if any(path.is_symlink() for path in files):
        raise ValidationError("Parent experiment artifacts must be regular retained files")
    return {str(path.relative_to(directory)): benchmark._file_hash(path) for path in files}


def _one_check(envelope: Mapping[str, Any]) -> None:
    if len(benchmark._approved(envelope).checks) != 1:
        raise ValidationError("This additive comparison requires one unchanged task check per source case")


def freeze(parent_dir: Path, out_dir: Path, *, root: Path = benchmark.ROOT, max_requests: int = 18,
           frame_annotations: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Freeze every original frame and parent response before any new gripper calls."""
    output = benchmark._local_path(out_dir, root, output=True)
    if output.exists():
        raise ValidationError("Image gripper study requires a fresh private directory")
    directory, original = parent._load(parent_dir, root)
    if output == directory or directory in output.parents:
        raise ValidationError("The follow-up output cannot be inside its frozen parent experiment")
    summary = parent.summarize(directory, root=root)
    if summary["status"] != "completed":
        raise ValidationError("Image gripper follow-up requires its complete frozen parent experiment")
    cases = []
    for record, result in zip(original["cases"], summary["results"]):
        _one_check(record["envelope"])
        if record["case_id"] != result["case_id"]:
            raise ValidationError("Parent case order changed")
        frames = []
        for index, row in enumerate(record["frames"]):
            path = parent._part_path(directory, record, index)
            if result["frames"][index]["status"] != "completed":
                raise ValidationError("Every original target proposal must remain complete")
            frames.append({**deepcopy(row), "target_file": str(path), "target_sha256": benchmark._file_hash(path)})
        cases.append({"case_id": record["case_id"], "pilot_index": record["pilot_index"],
                      "envelope": deepcopy(record["envelope"]), "frames": frames,
                      "parent_status": result["status"], "parent_prediction": result["prediction"],
                      "parent_result_sha256": benchmark.manifest_hash(result)})
    count = sum(len(case["frames"]) for case in cases)
    if type(max_requests) is not int or not 1 <= count <= max_requests <= 18 or not 1 <= len(cases) <= 9:
        raise ValidationError("All original cases and images must fit an explicit maximum of 18 requests")
    hashes = [row["source_sha256"] for case in cases for row in case["frames"]]
    labels = dict(frame_annotations or {})
    if len(set(hashes)) != len(hashes) or set(labels) - set(hashes) or any(label not in ANNOTATIONS for label in labels.values()):
        raise ValidationError("Geometry references must identify distinct original views and supported diagnostic categories")
    plan = {"schema_version": VERSION, "method": VERSION, "concept": CONCEPT,
            "evaluation_role": "post_outcome_exposed_development_negative_only_followup",
            "parent_dir": str(directory), "parent_plan_sha256": original["plan_sha256"],
            "parent_summary_sha256": benchmark.manifest_hash(summary), "parent_inventory": _parent_inventory(directory),
            "cases": cases, "pilots": deepcopy(original["pilots"]),
            "max_requests": max_requests, "planned_requests": count, "frame_annotations": labels,
            "implementation_sha256": _implementation(), "dependencies": parent._dependencies(),
            "geometry_config": deepcopy(_veto().CONFIG), "created_unix_ms": int(time.time() * 1000)}
    plan["plan_sha256"] = benchmark.manifest_hash(plan)
    output.mkdir(parents=True, mode=0o700)
    benchmark._save(output / "plan.json", plan)
    return deepcopy(plan)


def _load(out_dir: Path, root: Path):
    output = benchmark._local_path(out_dir, root, output=True)
    plan = prospective_audit._read(output / "plan.json")
    if (not isinstance(plan, dict) or plan.get("schema_version") != VERSION or plan.get("method") != VERSION
            or plan.get("concept") != CONCEPT or plan.get("evaluation_role") != "post_outcome_exposed_development_negative_only_followup"
            or plan.get("plan_sha256") != benchmark.manifest_hash({key: value for key, value in plan.items() if key != "plan_sha256"})
            or plan.get("implementation_sha256") != _implementation() or plan.get("dependencies") != parent._dependencies()
            or plan.get("geometry_config") != _veto().CONFIG):
        raise ValidationError("Image gripper plan, implementation or dependencies changed")
    directory, original = parent._load(Path(plan["parent_dir"]), root)
    if (original["plan_sha256"] != plan["parent_plan_sha256"]
            or plan["parent_inventory"] != _parent_inventory(directory) or plan["pilots"] != original["pilots"]):
        raise ValidationError("Frozen parent plan, response or artifact inventory changed")
    summary = prospective_audit._read(directory / "summary.json")
    if summary.get("status") != "completed" or benchmark.manifest_hash(summary) != plan["parent_summary_sha256"]:
        raise ValidationError("Parent summary changed")
    if len(plan["cases"]) != len(original["cases"]) or len(plan["cases"]) != len(summary["results"]):
        raise ValidationError("Image gripper case denominator changed")
    hashes = []
    for record, prior, result in zip(plan["cases"], original["cases"], summary["results"]):
        _one_check(record["envelope"])
        if (record["case_id"] != prior["case_id"] or record["case_id"] != result["case_id"]
                or record["envelope"] != prior["envelope"] or record["pilot_index"] != prior["pilot_index"]
                or record["parent_status"] != result["status"] or record["parent_prediction"] != result["prediction"]
                or record["parent_result_sha256"] != benchmark.manifest_hash(result) or len(record["frames"]) != 2):
            raise ValidationError("Parent task, label, source or decision changed")
        for index, (row, prior_row) in enumerate(zip(record["frames"], prior["frames"])):
            target_path = parent._part_path(directory, prior, index)
            if row != {**prior_row, "target_file": str(target_path), "target_sha256": benchmark._file_hash(target_path)}:
                raise ValidationError("Original target result or source view changed")
            hashes.append(row["source_sha256"])
    labels = plan.get("frame_annotations")
    if (not isinstance(labels, dict) or len(set(hashes)) != len(hashes) or set(labels) - set(hashes)
            or any(label not in ANNOTATIONS for label in labels.values())
            or type(plan["max_requests"]) is not int or not 1 <= len(hashes) <= plan["max_requests"] <= 18
            or plan["planned_requests"] != len(hashes)):
        raise ValidationError("Image gripper request/reference denominator changed")
    return output, plan


def public_plan(out_dir: Path, *, root: Path = benchmark.ROOT) -> dict[str, Any]:
    _, plan = _load(out_dir, root)
    return {"schema_version": VERSION, "method": VERSION, "concept": CONCEPT,
            "evaluation_role": plan["evaluation_role"], "plan_sha256": plan["plan_sha256"],
            "parent_plan_sha256": plan["parent_plan_sha256"], "parent_summary_sha256": plan["parent_summary_sha256"],
            "parent_inventory_sha256": benchmark.manifest_hash(plan["parent_inventory"]),
            "parent_artifact_count": len(plan["parent_inventory"]), "geometry_config": plan["geometry_config"],
            "implementation_sha256": plan["implementation_sha256"], "dependencies": plan["dependencies"],
            "planned_requests": plan["planned_requests"], "max_requests": plan["max_requests"],
            "created_unix_ms": plan["created_unix_ms"], "frame_annotations": deepcopy(plan["frame_annotations"]),
            "cases": [{"case_id": case["case_id"], "original_contract_hash": case["envelope"]["contract_hash"],
                       "truth": case["envelope"]["contract"]["cases"][0]["annotation"]["status"],
                       "parent_status": case["parent_status"], "parent_prediction": case["parent_prediction"],
                       "parent_result_sha256": case["parent_result_sha256"],
                       "frames": [{key: deepcopy(value) for key, value in row.items() if key != "target_file"}
                                  for row in case["frames"]]} for case in plan["cases"]],
            "limits": ["Post-outcome development reusing exposed source media, original target inference and judgments.",
                       "The added SAM concept is robot gripper on every original full image; no Spark calls or retries.",
                       "Near proposals can withhold parent support; far proposals never establish full visibility or identity.",
                       "UNKNOWN stays UNKNOWN; missing added diagnostics remain explicit and cannot qualify parent positives.",
                       "Geometry reference labels and view denominators are separate from task success and positive retention.",
                       "No production promotion, physical contact claim, safety assurance or observed repair is provided."]}


def _key(case, index):
    return "sam/" + case["case_id"] + "/" + str(index)


def _path(output, case, index):
    return output / "parts" / case["case_id"] / ("gripper-" + str(index) + ".json")


def _replay(result, row):
    records = parent._semantics()._sam(result, row["source_sha256"], row["width"], row["height"])
    if any(type(record.get("frame")) is not int or record["frame"] != 0 for record in records):
        raise ValidationError("A completed gripper image response requires only integer source frame zero")


def run(client_factory: Callable[[int], Any], out_dir: Path, *, root: Path = benchmark.ROOT) -> dict[str, Any]:
    output, plan = _load(out_dir, root)
    parent._preflight(plan)
    try:
        with (output / "run-started.json").open("x", encoding="utf-8") as stream:
            json.dump({"plan_sha256": plan["plan_sha256"]}, stream)
        (output / "run-started.json").chmod(0o600)
    except FileExistsError:
        raise ValidationError("Image gripper study already started; automatic resume is disabled") from None
    if any((output / name).exists() for name in ("parts", "run-state.json", "error-responses")):
        raise ValidationError("Image gripper inference output already exists")
    state = {"plan_sha256": plan["plan_sha256"], "status": "running", "attempts": {}, "admission_stop_code": None}
    benchmark._save(output / "run-state.json", state)
    seen = []
    for case in plan["cases"]:
        paths = benchmark.verify_media(case["envelope"], project_root=root)
        for index, row in enumerate(case["frames"]):
            if state["admission_stop_code"] is not None:
                break
            _load(output, root)
            key = _key(case, index)
            entry = {"status": "running", "dispatch_started": False, "usage": None}
            state["attempts"][key] = entry
            benchmark._save(output / "run-state.json", state)
            client, admitted = None, False
            started = time.monotonic()
            try:
                client = client_factory(1)
                prospective._fresh(client, 1, seen)
                admitted = True
                entry.update(dispatch_started=True, inference_origin="live_meta_adapter" if isinstance(client, MetaClient)
                             else "injected_client_not_verified_live")
                benchmark._save(output / "run-state.json", state)
                result = client.segment_image(paths[row["source_sha256"]], CONCEPT, decode_masks=True)
                path = _path(output, case, index)
                benchmark._save(path, result)
                entry["response_sha256"] = benchmark._file_hash(path)
                _replay(result, row)
                entry["status"] = "completed"
            except Exception as error:
                entry.update(status="error", error_type=type(error).__name__ if type(error).__name__ in prospective._ERROR_TYPES else "condition_error")
                if isinstance(getattr(error, "response", None), Mapping):
                    try:
                        entry.update(parent._retain_failure(output, key, error.response))
                    except (OSError, ValueError, TypeError):
                        entry["error_response_save_failed"] = True
                status = getattr(error, "status", None)
                if type(status) is int and status in parent.STOP_HTTP:
                    state["admission_stop_code"] = status
            finally:
                if admitted:
                    for attempt in client.ledger:
                        status = attempt.get("http_status")
                        if type(status) is int and status in parent.STOP_HTTP:
                            state["admission_stop_code"] = status
                    entry["usage"] = parent._capture(client, SAM_MODEL, entry["status"])
                else:
                    entry["usage"] = parent.accounting._ledger_usage([])
                entry["wall_seconds"] = round(time.monotonic() - started, 3)
                benchmark._save(output / "run-state.json", state)
    state["status"] = "completed" if (len(state["attempts"]) == plan["planned_requests"]
        and all(entry["status"] == "completed" for entry in state["attempts"].values())) else "incomplete"
    benchmark._save(output / "run-state.json", state)
    result = summarize(output, root=root)
    benchmark._save(output / "summary.json", result)
    return result


def _project(parent_status: str, prediction: str | None, frames: list[Mapping[str, Any]]) -> dict[str, Any]:
    """A negative-only policy; incomplete extra diagnostics cannot qualify positives."""
    if parent_status != "completed" or prediction not in benchmark.STATUSES:
        return {"status": "unscored", "prediction": None, "missing_prediction": True, "reason": "parent_unscored"}
    if prediction == "unknown":
        return {"status": "completed", "prediction": "unknown", "missing_prediction": False, "reason": "parent_unknown_unchanged"}
    if any(frame.get("warning", {}).get("veto") is True for frame in frames):
        return {"status": "completed", "prediction": "unknown", "missing_prediction": False, "reason": "near_gripper_proposal_withholds_parent"}
    if any(frame.get("status") != "completed" or frame.get("warning", {}).get("geometry_available") is not True for frame in frames):
        return {"status": "unscored", "prediction": None, "missing_prediction": True, "reason": "added_geometry_unavailable"}
    return {"status": "completed", "prediction": prediction, "missing_prediction": False, "reason": "parent_decision_preserved_without_near_warning"}


def _metrics(plan, rows):
    output = []
    for index, pilot in enumerate(plan["pilots"]):
        values = []
        for case, row in zip(plan["cases"], rows):
            if case["pilot_index"] != index or row["missing_prediction"]:
                continue
            values.append({"case_id": case["case_id"], "status": row["prediction"],
                           "checks": [{"check_id": check.id, "status": row["prediction"]}
                                      for check in benchmark._approved(case["envelope"]).checks]})
        output.append({"prior_freeze_sha256": pilot["freeze_sha256"],
                       "metrics": benchmark.comparison_metrics(pilot["envelope"], values)})
    return output


def summarize(out_dir: Path, *, root: Path = benchmark.ROOT) -> dict[str, Any]:
    output, plan = _load(out_dir, root)
    path = output / "run-state.json"
    state = prospective_audit._read(path) if path.exists() else {"plan_sha256": plan["plan_sha256"],
        "status": "not_started", "attempts": {}, "admission_stop_code": None}
    expected = {_key(case, index) for case in plan["cases"] for index in range(2)}
    if (state.get("plan_sha256") != plan["plan_sha256"] or state.get("status") not in ("not_started", "running", "completed", "incomplete")
            or not isinstance(state.get("attempts"), Mapping) or set(state["attempts"]) - expected
            or state.get("admission_stop_code") is not None and (type(state["admission_stop_code"]) is not int
                or state["admission_stop_code"] not in parent.STOP_HTTP)):
        raise ValidationError("Image gripper state contradicts its plan")
    if state["status"] == "completed" and (set(state["attempts"]) != expected
            or any(entry.get("status") != "completed" for entry in state["attempts"].values())):
        raise ValidationError("Completed gripper experiment omits planned diagnostics")
    if set((output / "parts").rglob("*.json")) - {_path(output, case, index) for case in plan["cases"] for index in range(2)}:
        raise ValidationError("Unplanned gripper response exists")
    if set((output / "error-responses").rglob("*.json")) - {output / "error-responses" / (key + ".json") for key in expected}:
        raise ValidationError("Unplanned failed gripper response exists")
    totals = {"request_attempts": 0, "sam_requests": 0, "spark_requests": 0, "reported_input_tokens": 0,
              "reported_output_tokens": 0, "estimated_usd": 0.0, "wall_seconds": 0.0,
              "request_attempts_incomplete": False, "accounting_incomplete": False, "unreported_charge_possible": False}
    results, views = [], []
    for case in plan["cases"]:
        frames = []
        for index, row in enumerate(case["frames"]):
            entry = state["attempts"].get(_key(case, index))
            frame = {"frame_index": index, "source_sha256": row["source_sha256"],
                     "width": row["width"], "height": row["height"],
                     "target_response_sha256": row["target_sha256"],
                     **parent._public_attempt(output, case, index, entry, totals), "scores": [],
                     "warning": {"warning": "geometry_unavailable", "veto": False, "geometry_available": False,
                                 "reason_codes": ["gripper_diagnostic_unscored"], "positive_support_can_be_granted": False}}
            response_path = _path(output, case, index)
            if entry and "response_sha256" in entry:
                if benchmark._file_hash(response_path) != entry["response_sha256"]:
                    raise ValidationError("Gripper response changed")
                result = prospective_audit._read(response_path)
                if entry["status"] in ("completed", "error"):
                    parent._bind_response_usage(result, entry, SAM_MODEL)
                if entry["status"] == "completed":
                    _replay(result, row)
                    target = prospective_audit._read(Path(row["target_file"]))
                    frame["scores"] = [record["confidence"] for record in result["objects"]]
                    frame["object_count"] = len(result["objects"])
                    frame["warning"] = _veto().assess_frame(target, result, source_sha256=row["source_sha256"],
                        width=row["width"], height=row["height"], target_object_id=row["object_id"])
            elif entry and entry["status"] == "completed" or response_path.exists() and (not entry or entry["status"] != "running"):
                raise ValidationError("Gripper response lacks completed content binding")
            label = plan["frame_annotations"].get(row["source_sha256"])
            frame["geometry_reference"] = label
            frame["diagnostic_match"] = (frame["warning"]["warning"] == label
                                          if label is not None and frame["status"] == "completed"
                                          and (frame["warning"].get("geometry_available") is True
                                               or frame["warning"].get("veto") is True) else None)
            frames.append(frame)
            views.append(frame)
        results.append({"case_id": case["case_id"], "parent_prediction": case["parent_prediction"],
                        "frames": frames, **_project(case["parent_status"], case["parent_prediction"], frames)})
    if totals["request_attempts"] > plan["max_requests"]:
        raise ValidationError("Image gripper request cap exceeded")
    for name in ("estimated_usd", "wall_seconds"):
        totals[name] = round(totals[name], 6)
    diagnostics = {"total_original_views": plan["planned_requests"],
                   "completed_gripper_diagnostics": sum(view["status"] == "completed" for view in views),
                   "missing_or_failed_gripper_diagnostics": sum(view["status"] != "completed" for view in views),
                   "near_warning_views": sum(view["warning"].get("veto") is True for view in views),
                   "available_no_near_warning_views": sum(view["warning"]["warning"] == "no_near_proposal_observed" for view in views),
                   "geometry_unavailable_views": sum(view["warning"].get("geometry_available") is not True for view in views),
                   "reference_categories": [{"label": label, "total": sum(view["geometry_reference"] == label for view in views),
                                              "matched": sum(view["geometry_reference"] == label and view["diagnostic_match"] is True for view in views),
                                              "unscored": sum(view["geometry_reference"] == label and view["diagnostic_match"] is None for view in views)}
                                             for label in sorted(ANNOTATIONS)],
                   "note": "A far/no-warning diagnostic is not task-positive retention or evidence of full visibility."}
    result = {**public_plan(out_dir, root=root), "status": state["status"], "usage": totals,
              "results": results, "geometry_diagnostics": diagnostics, "metrics": _metrics(plan, results),
              "admission_stop_code": state["admission_stop_code"],
              "accounting_note": "Only new gripper calls are charged here; original target SAM and Spark judgments are reused unchanged."}
    if (output / "summary.json").exists() and prospective_audit._read(output / "summary.json") != result:
        raise ValidationError("Saved gripper summary differs from replayed diagnostics/accounting")
    return result
