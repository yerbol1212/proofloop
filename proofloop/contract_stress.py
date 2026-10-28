"""Paired opposite-region contract controls on already exposed recorded images.

Saved task-independent guided perception is replayed, then re-grounded under a
new specification. Spark receives fresh calls for the changed requirements.
These authored counterfactual contracts are not new physical failure episodes.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Callable, Mapping, Sequence

from .benchmark import (ROOT, _approved, _file_hash, _local_path, _read, _save, approve_manifest,
                        baseline_prompt, baseline_schema, comparison_metrics, evaluate_guided,
                        evaluate_spark, manifest_hash, validate_report, verify_media)
from .core import ValidationError
from .followup import _estimate
from .guided import GUIDED_VERSION, guided_segment_pair
from .meta import MetaAPIError

KIND = "authored_opposite_region_contract_control"
GUIDED_SHA256 = "8f77c12f32911d28c3a17bf53d47f2bf30ac021bea0d68a20d6c1dd0487a3a23"
REVIEW_REFERENCE = "User authorized additional online-video evaluation and adoption of project practices."


class _SavedClient:
    retries, max_requests = 0, 3

    def __init__(self, pair: Mapping[str, Any], object_id: str):
        self.pair, self.object_id, self.ledger = pair, object_id, []

    def spark_json(self, *_args, **_kwargs):
        return deepcopy(self.pair["locator"])

    def segment_image(self, path, _prompt, **_kwargs):
        result = self.pair["results"].get(_file_hash(Path(path)), {})
        if set(result) != {self.object_id}:
            raise ValidationError("Saved guided target coverage differs from its locator")
        return deepcopy(result[self.object_id])


def _changed_contract(original: Mapping[str, Any], case_id: str, region: str, reason: str) -> dict[str, Any]:
    protocol = _approved(original)
    if (len(protocol.objects) != 1 or len(protocol.checks) != 1 or len(protocol.regions) != 2
            or protocol.checks[0].kind != "location" or region not in protocol.regions
            or region == protocol.checks[0].region_id):
        raise ValidationError("Control needs a single location check and the other calibrated region")
    case = next((case for case in original["contract"]["cases"] if case["id"] == case_id), None)
    if (case is None or case["role"] != "benchmark" or case["annotation"]["status"] != "pass"
            or not isinstance(reason, str) or not reason.strip() or len(reason) > 2000):
        raise ValidationError("Select a reviewed positive case and explain its opposite-region FAIL label")
    value = deepcopy(original["contract"])
    value["dataset_id"] = "opposite-region-" + case_id
    value["protocol"]["version"] = "authored-opposite-region-control-1"
    value["protocol"]["checks"][0]["region_id"] = region
    value["sources"] = [source for source in value["sources"] if source["id"] == case["source_id"]]
    selected = deepcopy(case)
    selected["annotation"] = {"status": "fail", "checks": {protocol.checks[0].id: "fail"},
                              "reason": reason, "origin": "source_frames_reviewed_before_inference"}
    value["cases"] = [selected]
    value["requirements"] = (
        f"For this authored contract-sensitivity control, the sole {value['concepts'][protocol.objects[0]]} "
        f"must be wholly inside the calibrated {region} image region in BOTH source frames. "
        "This declared requirement deliberately differs from the original demonstration goal. "
        "A positively supported unique target wholly in the other calibrated region FAILS this check. "
        "Missing, multiple, ambiguous, partial, stale or inconsistent evidence remains UNKNOWN. "
        "Evaluate only this declared placement contract; do not infer success from the recording's original goal. "
        "This is not a natural execution error, evidence of a new physical failure, or verification of a complete procedure."
    )
    return value


def prepare_control(original_run: Path, case_id: str, region: str, *, review_reason: str,
                    root: Path = ROOT) -> dict[str, Any]:
    """Freeze the selected existing files and a reviewed changed specification."""
    root = Path(root).resolve()
    folder = _local_path(original_run, root, output=True)
    original = _read(folder / "approved-manifest.json")
    changed = _changed_contract(original, case_id, region, review_reason)
    relative = str(folder.relative_to(root))
    files = ["approved-manifest.json", "report.json", "guided/results.json", "spark/results.json",
             f"guided/{case_id}.json", f"spark/{case_id}.json"]
    return {"schema_version": "1.0", "kind": KIND, "evaluation_role": "development_reprocessing",
            "original_run": relative, "case_id": case_id, "required_region": region, "review_reason": review_reason,
            "original_contract_hash": original["contract_hash"],
            "input_hashes": {name: _file_hash(folder / name) for name in files},
            "guided_implementation_sha256": GUIDED_SHA256,
            "changed_manifest": approve_manifest(changed, manifest_hash(changed), confirm_unique_targets=True,
                review_origin="assistant_under_user_authorization", authorization_reference=REVIEW_REFERENCE)}


def _ledger_records(responses: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [{"status": "ok", "model": item["response"]["model"], "usage": item.get("usage"),
             "latency_seconds": item.get("latency_seconds")} for item in responses]


def _case_cost(folder: Path, manifest: Mapping[str, Any], report: Mapping[str, Any], name: str, selected: str) -> dict[str, Any]:
    ledger = report["conditions"][name].get("usage", {}).get("requests")
    if not isinstance(ledger, list):
        raise ValidationError("Original per-request cost ledger is required")
    cursor, selected_entries = 0, None
    for case in manifest["contract"]["cases"]:
        raw = _read(folder / name / f"{case['id']}.json")
        responses = [raw] if name == "spark" else [raw["locator"]] + [
            result for frame in raw["frames"] for result in raw["results"][frame["sha256"]].values()]
        expected = _ledger_records(responses)
        actual = ledger[cursor:cursor + len(expected)]
        if len(actual) != len(expected) or any(any(entry.get(key) != item[key] for key in item)
                                               for entry, item in zip(actual, expected)):
            raise ValidationError("Original cost ledger differs from saved per-case responses")
        if case["id"] == selected:
            selected_entries = actual
        cursor += len(expected)
    if (cursor != len(ledger) or selected_entries is None
            or _estimate(ledger) != report["conditions"][name]["usage"].get("estimated_usd")):
        raise ValidationError("Original request totals or cost estimate changed")
    return {"request_count": len(selected_entries), "estimated_usd": _estimate(selected_entries)}


def _load_control(plan: Mapping[str, Any], root: Path) -> dict[str, Any]:
    if (not isinstance(plan, Mapping) or plan.get("schema_version") != "1.0" or plan.get("kind") != KIND
            or plan.get("evaluation_role") != "development_reprocessing"
            or plan.get("guided_implementation_sha256") != GUIDED_SHA256
            or _file_hash(Path(__file__).with_name("guided.py")) != GUIDED_SHA256):
        raise ValidationError("Contract control method differs from the frozen guided implementation")
    folder = _local_path(plan.get("original_run"), root, output=True)
    case_id = plan.get("case_id")
    original = _read(folder / "approved-manifest.json")
    changed = _changed_contract(original, case_id, plan.get("required_region"), plan.get("review_reason"))
    expected_files = {"approved-manifest.json", "report.json", "guided/results.json", "spark/results.json",
                      f"guided/{case_id}.json", f"spark/{case_id}.json"}
    if (plan.get("original_contract_hash") != original["contract_hash"]
            or not isinstance(plan.get("input_hashes"), Mapping) or set(plan["input_hashes"]) != expected_files
            or any(_file_hash(folder / name) != digest for name, digest in plan["input_hashes"].items())
            or plan.get("changed_manifest", {}).get("contract") != changed):
        raise ValidationError("Frozen original files or authored changed contract differ")
    new_protocol = _approved(plan["changed_manifest"])
    if (plan["changed_manifest"]["approval"].get("origin") != "assistant_under_user_authorization"
            or plan["changed_manifest"]["approval"].get("authorization_reference") != REVIEW_REFERENCE):
        raise ValidationError("Contract control must retain assistant review provenance")
    report = _read(folder / "report.json")
    if (report.get("status") != "completed" or not {"guided", "spark"} <= set(report.get("conditions", {}))
            or report["conditions"]["guided"].get("observation_version") != GUIDED_VERSION
            or report.get("inference_origin") not in ("live_meta_api", "authored_fixture")):
        raise ValidationError("Original comparison needs completed guided-v2 and Spark observations")
    paths = verify_media(original, project_root=root)
    sidecars = {name: _read(folder / name / "results.json") for name, condition in report["conditions"].items()
                if condition["status"] == "completed"}
    validate_report(original, report, raw_results=sidecars)
    case = changed["cases"][0]
    pair = _read(folder / "guided" / f"{case_id}.json")
    if pair.get("results") != sidecars["guided"][case_id]:
        raise ValidationError("Saved selected pair differs from the evaluated original sidecar")
    object_id = new_protocol.objects[0]
    images = [paths[frame["sha256"]] for frame in case["frames"]]
    replayed_pair = guided_segment_pair(_SavedClient(pair, object_id), images, changed["concepts"])
    if replayed_pair != pair:
        raise ValidationError("Saved guided locator, masks and provenance do not replay identically")
    original_guided = next(item for item in report["conditions"]["guided"]["cases"] if item["case_id"] == case_id)
    original_spark = next(item for item in report["conditions"]["spark"]["cases"] if item["case_id"] == case_id)
    original_spark_raw = _read(folder / "spark" / f"{case_id}.json")
    if (original_guided["status"] != "pass" or original_spark["status"] != "pass"
            or original_spark_raw != sidecars["spark"][case_id]):
        raise ValidationError("Original reference must be a bound positive result in both conditions")
    origin = "replayed_saved_live_meta_observations" if report["inference_origin"] == "live_meta_api" else "replayed_authored_fixture"
    new_guided = evaluate_guided(plan["changed_manifest"], {case_id: replayed_pair["results"]},
                                 observation_version=GUIDED_VERSION, inference_origin=origin)
    if any(item["spec_hash"] != new_protocol.content_hash for item in new_guided[0]["grounding"]["observations"]):
        raise ValidationError("Changed evaluation reused an observation from a different specification")
    return {"manifest": plan["changed_manifest"], "original": original, "case": case, "paths": images,
            "original_guided": original_guided, "original_spark": original_spark,
            "new_guided": new_guided, "guided_raw": {case_id: replayed_pair["results"]},
            "guided_origin": origin, "original_origin": report["inference_origin"],
            "original_guided_cost": _case_cost(folder, original, report, "guided", case_id),
            "original_spark_cost": _case_cost(folder, original, report, "spark", case_id)}


def _report_for(control: Mapping[str, Any], spark_raw: Mapping[str, Any], origin: str) -> dict[str, Any]:
    manifest = control["manifest"]
    spark_cases = evaluate_spark(manifest, spark_raw, inference_origin=origin)
    return {"status": "completed", "contract_hash": manifest["contract_hash"],
            "evaluation_role": "development_reprocessing", "inference_origin": "mixed_replay_and_new_inference",
            "conditions": {
                "guided": {"status": "completed", "observation_version": GUIDED_VERSION,
                           "cases": control["new_guided"], "metrics": comparison_metrics(manifest, control["new_guided"])},
                "spark": {"status": "completed", "cases": spark_cases, "metrics": comparison_metrics(manifest, spark_cases)}}}


def run_controls(plans: Sequence[Mapping[str, Any]], output: Path, client_factory: Callable[[int], Any], *,
                 root: Path = ROOT, inference_origin: str = "live_meta_api") -> dict[str, Any]:
    """Freeze all controls before at most two fresh Spark requests; never retry."""
    root = Path(root).resolve()
    if (not isinstance(plans, (list, tuple)) or not 1 <= len(plans) <= 2
            or any(not isinstance(plan, Mapping) or not isinstance(plan.get("case_id"), str) for plan in plans)
            or inference_origin not in ("live_meta_api", "authored_fixture")
            or len({plan.get("case_id") for plan in plans}) != len(plans)):
        raise ValidationError("Use one or two distinct reviewed controls and an explicit inference origin")
    controls = [_load_control(plan, root) for plan in plans]  # All preflight before any paid request.
    output = _local_path(output, root, output=True)
    if output.exists() and any(output.iterdir()):
        raise ValidationError("Contract controls require a new empty run folder")
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    frozen = {"plans": deepcopy(list(plans)), "plan_hash": manifest_hash({"plans": list(plans)})}
    _save(output / "frozen-controls.json", frozen)
    client = client_factory(len(plans))
    if client.retries != 0 or client.max_requests != len(plans) or client.ledger:
        raise ValidationError("Contract controls require an unused exact Spark budget and zero retries")
    report = {"schema_version": "1.0", "kind": KIND, "status": "running", "plan_hash": frozen["plan_hash"],
              "new_spark_inference_origin": inference_origin, "completed_case_ids": [], "request_cap": len(plans)}
    _save(output / "report.json", report)
    try:
        for plan, control in zip(plans, controls):
            case_id = plan["case_id"]
            verify_media(control["manifest"], project_root=root)
            result = client.spark_json(baseline_prompt(control["manifest"], control["case"]),
                baseline_schema(_approved(control["manifest"])), name="proofloop_opposite_region_control",
                image_paths=control["paths"], max_output_tokens=2048)
            _save(output / f"{case_id}-spark.json", result)
            verify_media(control["manifest"], project_root=root)
            benchmark = _report_for(control, {case_id: result}, inference_origin)
            validate_report(control["manifest"], benchmark,
                            raw_results={"guided": control["guided_raw"], "spark": {case_id: result}})
            _save(output / f"{case_id}-comparison.json", benchmark)
            report["completed_case_ids"].append(case_id)
            report["usage"] = client.usage_summary()
            _save(output / "report.json", report)
            if report["usage"].get("estimated_usd", 1) > .10:
                raise ValidationError("Estimated control spending exceeded the stopping threshold")
        report["status"] = "completed"
    except (MetaAPIError, ValidationError, ValueError, OSError, TypeError) as error:
        report.update(status="error", error_type=type(error).__name__)
        report["usage"] = client.usage_summary()
        _save(output / "report.json", report)
        raise
    _save(output / "report.json", report)
    return contract_stress_summary(output, root=root)


def contract_stress_summary(run_directory: Path, *, root: Path = ROOT) -> dict[str, Any]:
    """Rebuild a public-safe DTO from the actual frozen inputs and saved results."""
    root = Path(root).resolve()
    folder = _local_path(run_directory, root, output=True)
    frozen, report = _read(folder / "frozen-controls.json"), _read(folder / "report.json")
    plans = frozen.get("plans")
    if (not isinstance(plans, list) or not 1 <= len(plans) <= 2
            or any(not isinstance(plan, Mapping) or not isinstance(plan.get("case_id"), str) for plan in plans)
            or len({plan["case_id"] for plan in plans}) != len(plans)
            or frozen.get("plan_hash") != manifest_hash({"plans": plans})
            or report.get("plan_hash") != frozen["plan_hash"] or report.get("kind") != KIND
            or report.get("status") != "completed" or type(report.get("request_cap")) is not int
            or report.get("request_cap") != len(plans)
            or report.get("completed_case_ids") != [plan["case_id"] for plan in plans]
            or report.get("new_spark_inference_origin") not in ("live_meta_api", "authored_fixture")):
        raise ValidationError("Contract control report does not bind a completed frozen schedule")
    output, new_raw = [], []
    for plan in plans:
        control = _load_control(plan, root)
        case_id = plan["case_id"]
        spark = _read(folder / f"{case_id}-spark.json")
        benchmark = _read(folder / f"{case_id}-comparison.json")
        recomputed = _report_for(control, {case_id: spark}, report["new_spark_inference_origin"])
        validate_report(control["manifest"], benchmark,
                        raw_results={"guided": control["guided_raw"], "spark": {case_id: spark}})
        if manifest_hash(benchmark) != manifest_hash(recomputed):
            raise ValidationError("Saved changed-contract comparison differs from replay")
        new_raw.append(spark)
        original_protocol = _approved(control["original"])
        changed_protocol = _approved(control["manifest"])
        safe_case = lambda case: {"status": case["status"], "checks": [
            {"check_id": item["check_id"], "status": item["status"]} for item in case["checks"]]}
        source = next(source for source in control["original"]["contract"]["sources"] if source["id"] == control["case"]["source_id"])
        output.append({"case_id": case_id, "source": {
                       **{key: deepcopy(source[key]) for key in ("id", "publisher", "page_url", "media_url", "sha256")},
                       "license": {key: source["license"][key] for key in ("name", "url", "attribution")}},
            "source_frames": [{"sha256": frame["sha256"], "source_timestamp_ms": frame["source_timestamp_ms"]}
                              for frame in control["case"]["frames"]],
            "regions": deepcopy(control["case"]["regions"]), "concepts": deepcopy(control["manifest"]["contract"]["concepts"]),
            "original_contract": {"contract_hash": control["original"]["contract_hash"], "protocol_hash": original_protocol.content_hash,
                "required_region": original_protocol.checks[0].region_id, "expected": "pass",
                "guided": safe_case(control["original_guided"]), "spark": safe_case(control["original_spark"]),
                "inference_origin": control["original_origin"]},
            "changed_contract": {"contract_hash": control["manifest"]["contract_hash"], "protocol_hash": changed_protocol.content_hash,
                "required_region": changed_protocol.checks[0].region_id, "expected": "fail",
                "guided": {**safe_case(benchmark["conditions"]["guided"]["cases"][0]), "inference_origin": control["guided_origin"]},
                "spark": {**safe_case(benchmark["conditions"]["spark"]["cases"][0]),
                          "inference_origin": report["new_spark_inference_origin"]}},
            "costs": {"original_guided_perception": control["original_guided_cost"],
                      "original_spark_reference": control["original_spark_cost"],
                      "incremental_guided": {"request_count": 0, "estimated_usd": 0},
                      "incremental_spark": {"request_count": 1, "estimated_usd": _estimate(_ledger_records([spark]))}}})
    ledger = _ledger_records(new_raw)
    usage = report.get("usage", {})
    if (not isinstance(usage, Mapping) or type(usage.get("request_attempts")) is not int
            or usage.get("request_attempts") != len(ledger) or not isinstance(usage.get("requests"), list)
            or len(usage["requests"]) != len(ledger)
            or any(any(actual.get(key) != expected[key] for key in expected)
                   for actual, expected in zip(usage["requests"], ledger))
            or usage.get("estimated_usd") != _estimate(ledger)):
        raise ValidationError("New Spark request count or cost ledger differs from saved responses")
    return {"schema_version": "1.0", "experiment": KIND, "evaluation_role": "development_reprocessing",
            "plan_hash": frozen["plan_hash"], "status": "completed", "case_count": len(output),
            "new_independent_source_count": 0, "incremental_spark_request_count": len(ledger),
            "incremental_guided_request_count": 0, "incremental_estimated_usd": _estimate(ledger),
            "controls": output, "physical_action_execution_verified": False,
            "natural_execution_failure_verified": False,
            "limits": ["Same already exposed video evidence with an authored opposite-region requirement; not new video accuracy.",
                       "Guided observations are saved Meta perception replayed and re-grounded under the changed specification, not fresh inference.",
                       "Original perception cost remains explicit; zero incremental guided requests does not mean free perception.",
                       "Spark changed-contract verdicts use fresh calls; original-contract verdicts are saved references.",
                       "Hashes check local consistency, not trusted review timing or independent model identity."]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        value = contract_stress_summary(args.run_directory)
        data = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
        if args.output:
            with args.output.open("x", encoding="utf-8") as stream:
                stream.write(data)
        else:
            print(data, end="")
        return 0
    except (ValidationError, ValueError, OSError, MetaAPIError) as error:
        print(f"Contract stress stopped: {type(error).__name__}; check frozen local inputs.", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
