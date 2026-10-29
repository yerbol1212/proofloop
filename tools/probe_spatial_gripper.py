"""Last bounded manual-box gripper diagnostic; no task or production verdicts.

Import freeze/public_plan/run/summarize from a local research driver. The caller
supplies fresh one-request clients with a shared persistent allowance of six.
"""
from copy import deepcopy
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from proofloop import benchmark, concept_verified_experiment as parent, guided
from proofloop import image_gripper_experiment as prior, image_gripper_veto, prospective, prospective_audit
from proofloop.capture import _jpeg_dimensions
from proofloop.core import ValidationError
from proofloop.meta import MetaClient, SAM_MODEL

VERSION = "manual_spatial_gripper_six_view_diagnostic_v1"
ORIGIN = "assistant_authored_manual_gripper_boxes"


def _implementation():
    root = Path(__file__).resolve().parents[1]
    return {**{"proofloop/" + name: value for name, value in prior._implementation().items()},
            "tools/probe_spatial_gripper.py": benchmark._file_hash(root / "tools/probe_spatial_gripper.py")}


def _entries(annotation, original, directory, root):
    if (not isinstance(annotation, dict) or annotation.get("annotation_origin") != ORIGIN
            or annotation.get("coordinate_system") != "original_source_pixels_inclusive_xyxy"
            or annotation.get("annotation_sha256") != benchmark.manifest_hash({k: v for k, v in annotation.items() if k != "annotation_sha256"})
            or not isinstance(annotation.get("frames"), list) or not 1 <= len(annotation["frames"]) <= 6):
        raise ValidationError("Use the frozen assistant-authored source-pixel box declaration")
    sources = {row["source_sha256"]: (case, index, row) for case in original["cases"] for index, row in enumerate(case["frames"])}
    result, seen = [], set()
    for declared in annotation["frames"]:
        digest = declared.get("source_sha256")
        if digest not in sources or digest in seen:
            raise ValidationError("Spatial diagnostics need unique original parent images")
        seen.add(digest)
        case, index, row = sources[digest]
        source_frame = case["envelope"]["contract"]["cases"][0]["frames"][index]
        decoded_index = source_frame.get("decoded_frame_index")
        source = benchmark.verify_media(case["envelope"], project_root=root)[digest]
        box, width, height = declared.get("box"), row["width"], row["height"]
        if (not isinstance(box, dict) or set(box) != {"x1", "y1", "x2", "y2"}
                or any(type(value) is not int for value in box.values())
                or not 0 <= box["x1"] <= box["x2"] < width or not 0 <= box["y1"] <= box["y2"] < height
                or declared.get("width") != width or declared.get("height") != height
                or type(declared.get("pair_frame_index")) is not int or declared["pair_frame_index"] != index
                or type(declared.get("decoded_frame_index")) is not int
                or declared["decoded_frame_index"] < 0
                or decoded_index is not None and (type(decoded_index) is not int
                    or decoded_index != declared["decoded_frame_index"])
                or declared.get("source_id") != case["envelope"]["contract"]["cases"][0]["source_id"]
                or benchmark._local_path(declared.get("file"), root) != source
                or _jpeg_dimensions(source.read_bytes()) != (width, height)):
            raise ValidationError("Manual box and declaration must match the exact original JPEG coordinates")
        target = parent._part_path(directory, case, index)
        result.append({"case_id": case["case_id"], "source_sha256": digest, "width": width, "height": height,
                       "source_file": str(source), "target_file": str(target), "target_sha256": benchmark._file_hash(target),
                       "object_id": row["object_id"], "frame_index": index,
                       "source_id": declared["source_id"], "decoded_frame_index": declared["decoded_frame_index"],
                       "decoded_frame_index_origin": "parent_source_frame" if decoded_index is not None else "manual_annotation_only_not_parent_verified",
                       "box": deepcopy(box), "prompt": parent.accounting._prompt({"padded_box": box, "width": width, "height": height})})
    return result


def freeze(entries, outdir, *, root=benchmark.ROOT, max_requests=6, parent_dir=None):
    output = benchmark._local_path(outdir, root, output=True)
    directory, original = parent._load(parent_dir or Path(root) / "artifacts/local/judge-20261030/concept-verified-development", root)
    summary = parent.summarize(directory, root=root)
    if output.exists() or output == directory or directory in output.parents or summary["status"] != "completed":
        raise ValidationError("Use a fresh sibling private directory and completed parent")
    frames = _entries(entries, original, directory, root)
    if type(max_requests) is not int or not len(frames) <= max_requests <= 6:
        raise ValidationError("Every manual source view must fit the maximum of six requests")
    plan = {"version": VERSION, "annotation": deepcopy(entries), "entries": frames, "parent_dir": str(directory),
            "parent_plan_sha256": original["plan_sha256"], "parent_inventory": prior._parent_inventory(directory),
            "parent_summary_sha256": benchmark.manifest_hash(summary), "max_requests": max_requests,
            "implementation_sha256": _implementation(), "dependencies": parent._dependencies()}
    plan["plan_sha256"] = benchmark.manifest_hash(plan)
    output.mkdir(parents=True, mode=0o700)
    benchmark._save(output / "plan.json", plan)
    return deepcopy(plan)


def _load(outdir, root):
    output = benchmark._local_path(outdir, root, output=True)
    plan = prospective_audit._read(output / "plan.json")
    if (plan.get("version") != VERSION or plan.get("implementation_sha256") != _implementation()
            or plan.get("dependencies") != parent._dependencies()
            or plan.get("plan_sha256") != benchmark.manifest_hash({k: v for k, v in plan.items() if k != "plan_sha256"})):
        raise ValidationError("Spatial diagnostic plan, code or dependency changed")
    directory, original = parent._load(Path(plan["parent_dir"]), root)
    if (original["plan_sha256"] != plan["parent_plan_sha256"] or prior._parent_inventory(directory) != plan["parent_inventory"]
            or benchmark.manifest_hash(prospective_audit._read(directory / "summary.json")) != plan["parent_summary_sha256"]
            or plan["entries"] != _entries(plan["annotation"], original, directory, root)
            or type(plan["max_requests"]) is not int or not len(plan["entries"]) <= plan["max_requests"] <= 6):
        raise ValidationError("Spatial diagnostic parent, box, source or budget changed")
    return output, plan


def public_plan(outdir, *, root=benchmark.ROOT):
    _, plan = _load(outdir, root)
    return {"version": VERSION, "plan_sha256": plan["plan_sha256"], "parent_plan_sha256": plan["parent_plan_sha256"],
            "parent_summary_sha256": plan["parent_summary_sha256"], "annotation_sha256": plan["annotation"]["annotation_sha256"],
            "annotation_origin": ORIGIN, "planned_requests": len(plan["entries"]), "max_requests": plan["max_requests"],
            "implementation_sha256": plan["implementation_sha256"], "dependencies": plan["dependencies"],
            "entries": [{key: deepcopy(value) for key, value in row.items() if key not in {"source_file", "target_file"}}
                        for row in plan["entries"]],
            "scope": "exposed_development_manual_anchor_six_view_diagnostic_only",
            "limits": ["Manual gripper boxes are assistant annotations, not automatic localization or ground-truth masks.",
                       "Only a spatial object token enters SAM; no semantic text, Spark inference or production promotion.",
                       "These six views do not improve or replace the original nine task results or positive-retention denominators.",
                       "Near proposals are negative warnings; far masks do not establish visibility, safety or physical separation."]}


def _case(index):
    return {"case_id": "view-" + str(index)}


def _path(output, index):
    return output / "parts" / ("view-" + str(index) + ".json")


def _replay(result, row):
    prior._replay(result, row)
    guided._sam_bound(result, {"sha256": row["source_sha256"], "width": row["width"], "height": row["height"]})


def run(factory, outdir, *, root=benchmark.ROOT):
    output, plan = _load(outdir, root)
    parent._preflight(plan)
    try:
        with (output / "run-started.json").open("x") as stream:
            json.dump({"plan_sha256": plan["plan_sha256"]}, stream)
    except FileExistsError:
        raise ValidationError("Spatial diagnostic already started; no automatic resume") from None
    if any((output / name).exists() for name in ("parts", "run-state.json", "error-responses")):
        raise ValidationError("Spatial diagnostic output already exists")
    state = {"plan_sha256": plan["plan_sha256"], "status": "running", "attempts": {}, "admission_stop_code": None}
    seen = []
    for index, row in enumerate(plan["entries"]):
        if state["admission_stop_code"] is not None:
            break
        _load(output, root)
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
            result = client.segment_image(row["source_file"], row["prompt"], decode_masks=True)
            benchmark._save(_path(output, index), result)
            entry["response_sha256"] = benchmark._file_hash(_path(output, index))
            _replay(result, row)
            entry["status"] = "completed"
        except Exception as error:
            entry.update(status="error", error_type=type(error).__name__ if type(error).__name__ in prospective._ERROR_TYPES else "condition_error")
            if isinstance(getattr(error, "response", None), dict):
                try:
                    entry.update(parent._retain_failure(output, "sam/view-" + str(index) + "/0", error.response))
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
    state["status"] = "completed" if len(state["attempts"]) == len(plan["entries"]) and all(
        entry["status"] == "completed" for entry in state["attempts"].values()) else "incomplete"
    benchmark._save(output / "run-state.json", state)
    result = summarize(output, root=root)
    benchmark._save(output / "summary.json", result)
    return result


def summarize(outdir, *, root=benchmark.ROOT):
    output, plan = _load(outdir, root)
    path = output / "run-state.json"
    state = prospective_audit._read(path) if path.exists() else {"plan_sha256": plan["plan_sha256"], "status": "not_started", "attempts": {}, "admission_stop_code": None}
    expected = {str(index) for index in range(len(plan["entries"]))}
    if (state.get("plan_sha256") != plan["plan_sha256"] or state.get("status") not in {"not_started", "running", "completed", "incomplete"}
            or not isinstance(state.get("attempts"), dict) or set(state["attempts"]) - expected
            or state.get("admission_stop_code") is not None and state["admission_stop_code"] not in parent.STOP_HTTP
            or state["status"] == "completed" and (set(state["attempts"]) != expected or any(v["status"] != "completed" for v in state["attempts"].values()))):
        raise ValidationError("Spatial diagnostic state contradicts the six-view plan")
    if set((output / "parts").glob("*.json")) - {_path(output, index) for index in range(len(plan["entries"]))}:
        raise ValidationError("Unplanned spatial result exists")
    if set((output / "error-responses").rglob("*.json")) - {
            output / "error-responses" / "sam" / ("view-" + str(index)) / "0.json" for index in range(len(plan["entries"]))}:
        raise ValidationError("Unplanned failed spatial response exists")
    totals = {"request_attempts": 0, "sam_requests": 0, "spark_requests": 0, "reported_input_tokens": 0, "reported_output_tokens": 0,
              "estimated_usd": 0.0, "wall_seconds": 0.0, "request_attempts_incomplete": False, "accounting_incomplete": False, "unreported_charge_possible": False}
    views = []
    for index, row in enumerate(plan["entries"]):
        entry = state["attempts"].get(str(index))
        view = {"view_index": index, "source_sha256": row["source_sha256"], "scores": [],
                **parent._public_attempt(output, _case(index), 0, entry, totals), "diagnostic": None}
        path = _path(output, index)
        if entry and "response_sha256" in entry:
            if benchmark._file_hash(path) != entry["response_sha256"]:
                raise ValidationError("Spatial raw response changed")
            result = prospective_audit._read(path)
            if entry["status"] in {"completed", "error"}:
                parent._bind_response_usage(result, entry, SAM_MODEL)
            if entry["status"] == "completed":
                _replay(result, row)
                view.update(scores=[record["confidence"] for record in result["objects"]], object_count=len(result["objects"]))
                view["diagnostic"] = image_gripper_veto.assess_frame(prospective_audit._read(Path(row["target_file"])), result,
                    source_sha256=row["source_sha256"], width=row["width"], height=row["height"], target_object_id=row["object_id"])
        elif entry and entry["status"] == "completed" or path.exists() and (not entry or entry["status"] != "running"):
            raise ValidationError("Spatial response lacks its completed content receipt")
        views.append(view)
    if totals["request_attempts"] > plan["max_requests"]:
        raise ValidationError("Spatial six-request cap exceeded")
    for key in ("estimated_usd", "wall_seconds"):
        totals[key] = round(totals[key], 6)
    result = {**public_plan(outdir, root=root), "status": state["status"], "views": views, "usage": totals,
              "admission_stop_code": state["admission_stop_code"], "task_metrics": None,
              "completed_views": sum(view["status"] == "completed" for view in views),
              "missing_or_failed_views": sum(view["status"] != "completed" for view in views)}
    if (output / "summary.json").exists() and prospective_audit._read(output / "summary.json") != result:
        raise ValidationError("Spatial summary differs from raw response/accounting replay")
    return result
