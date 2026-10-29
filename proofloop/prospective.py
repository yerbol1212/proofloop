"""One-shot, predeclared recorded-frame pilot around the existing benchmark.

This adds failure isolation and a replayable denominator, not another perception
method. Caller-reviewed labels are not an independent audit or a physical loop.
All raw responses, approvals, media paths and authorization remain private.
"""
from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
import time
from types import SimpleNamespace
from typing import Any, Callable, Mapping, Sequence

from . import benchmark
from .core import ValidationError
from .meta import MetaClient

VERSION = "prospective_recorded_pilot_v1"
CODE_FILES = ("benchmark.py", "guided.py", "meta.py", "grounding.py", "core.py", "session.py",
              "bounded_spark.py", "inference_journal.py", "prospective.py")
_STOP_CODES = (401, 402, 403, 429)
_ERROR_TYPES = {"MetaAPIError", "ValidationError", "ValueError", "TypeError", "OSError", "RuntimeError"}
_GAPS = {"No current evidence for this attempt and revision": "missing_current_evidence",
         "Evidence is stale": "stale_evidence",
         "Observation is unavailable or ambiguous": "unavailable_or_ambiguous",
         "Observation confidence is below the approved threshold": "low_confidence",
         "Insufficient repeated stable observations": "insufficient_stable_observations",
         "Location requires an approved unique-target scene contract": "unique_target_unconfirmed",
         "No detection does not prove absence": "absence_unproven"}
_USAGE_COUNTS = ("request_attempts", "input_tokens", "output_tokens", "sam_images",
                 "spark_requests_with_unreported_token_usage",
                 "unsuccessful_spark_requests_with_reported_token_usage")


def _hash(value: Any) -> str:
    return benchmark.manifest_hash(value)


def _code_hashes() -> dict[str, str]:
    folder = Path(__file__).resolve().parent
    return {name: benchmark._file_hash(folder / name) for name in CODE_FILES}


def _read(path: Path) -> Any:
    if path.is_symlink():
        raise ValidationError("Prospective records must be regular private JSON files")
    return benchmark._read(path)


def _directory(out_dir: Path, root: Path) -> Path:
    return benchmark._local_path(out_dir, root, output=True)


def _single(envelope: Mapping[str, Any], case: Mapping[str, Any]) -> dict[str, Any]:
    contract = deepcopy(envelope["contract"])
    contract["cases"] = [deepcopy(case)]
    contract["sources"] = [source for source in contract["sources"] if source["id"] == case["source_id"]]
    approval = envelope["approval"]
    return benchmark.approve_manifest(
        contract, _hash(contract), confirm_unique_targets=approval["unique_targets_confirmed"],
        review_origin=approval["origin"], authorization_reference=approval.get("authorization_reference"))


def freeze(envelope: Mapping[str, Any], out_dir: Path, *, root: Path = benchmark.ROOT,
           conditions: Sequence[str] = ("spark", "guided"), max_requests: int = 24) -> dict[str, Any]:
    """Validate and persist every case/condition before any client can be created.

    Publish ``public_freeze`` before calling ``run`` to establish a visible
    declaration. Its hash alone is not a trusted timestamp or independent review.
    """
    benchmark.validate_manifest(envelope, project_root=root)
    if type(max_requests) is not int or not 1 <= max_requests <= 24:
        raise ValidationError("Prospective pilots allow an explicit cap of 1-24 requests")
    required = benchmark.request_requirements(envelope, conditions)
    if sum(required.values()) > max_requests:
        raise ValidationError("Predeclared conditions exceed the explicit request cap")
    output = _directory(Path(out_dir), root)
    if output.exists() and any(output.iterdir()):
        raise ValidationError("Freeze requires a new empty private output directory")
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    records = []
    for case in envelope["contract"]["cases"]:
        part = _single(envelope, case)
        record = {"case_id": case["id"], "envelope_sha256": _hash(part),
                  "contract_hash": part["contract_hash"],
                  "planned_requests": benchmark.request_requirements(part, conditions)}
        records.append(record)
        benchmark._save(output / "frozen-cases" / (case["id"] + ".json"), part)
    body = {"schema_version": VERSION, "scope": "caller_authorized_curated_new_source_pilot",
            "envelope": deepcopy(envelope), "conditions": list(conditions), "max_requests": max_requests,
            "planned_requests": required, "total_planned_requests": sum(required.values()),
            "implementation_sha256": _code_hashes(), "cases": records,
            "method_versions": {"spark": "benchmark_spark_same_pair_v1",
                                "guided": "normalized_boxes_v2"},
            "created_unix_ms": int(time.time() * 1000)}
    body["freeze_sha256"] = _hash(body)
    benchmark._save(output / "freeze.json", body)
    return deepcopy(body)


def _load(out_dir: Path, root: Path) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    output = _directory(Path(out_dir), root)
    frozen = _read(output / "freeze.json")
    if not isinstance(frozen, dict) or frozen.get("schema_version") != VERSION:
        raise ValidationError("Unsupported prospective freeze")
    body = {key: value for key, value in frozen.items() if key != "freeze_sha256"}
    if frozen.get("freeze_sha256") != _hash(body) or frozen.get("implementation_sha256") != _code_hashes():
        raise ValidationError("Prospective freeze or implementation changed")
    envelope = frozen["envelope"]
    benchmark.validate_manifest(envelope, project_root=root)
    conditions = frozen["conditions"]
    required = benchmark.request_requirements(envelope, conditions)
    if (type(frozen["max_requests"]) is not int or not 1 <= frozen["max_requests"] <= 24
            or frozen["planned_requests"] != required
            or frozen["total_planned_requests"] != sum(required.values())
            or sum(required.values()) > frozen["max_requests"]):
        raise ValidationError("Prospective request plan changed")
    expected = []
    parts = {}
    for case in envelope["contract"]["cases"]:
        single = _single(envelope, case)
        stored = _read(output / "frozen-cases" / (case["id"] + ".json"))
        if stored != single:
            raise ValidationError("A derived case approval changed")
        record = {"case_id": case["id"], "envelope_sha256": _hash(single),
                  "contract_hash": single["contract_hash"],
                  "planned_requests": benchmark.request_requirements(single, conditions)}
        expected.append(record)
        parts[case["id"]] = single
    if frozen["cases"] != expected:
        raise ValidationError("Prospective case coverage or hashes changed")
    return output, frozen, parts


def public_freeze(out_dir: Path, *, root: Path = benchmark.ROOT) -> dict[str, Any]:
    """Allowlisted preregistration suitable for publication before inference."""
    _, frozen, _ = _load(out_dir, root)
    envelope = frozen["envelope"]
    empty = {"contract_hash": envelope["contract_hash"], "status": "running",
             "conditions": {name: {"status": "running"} for name in frozen["conditions"]}}
    public = benchmark.public_summary(envelope, empty)
    cases = []
    for case in envelope["contract"]["cases"]:
        annotation = case["annotation"]
        cases.append({"case_id": case["id"], "source_id": case["source_id"], "role": case["role"],
                      "regions": deepcopy(case["regions"]),
                      "evaluation_timestamp_ms": case["evaluation_timestamp_ms"],
                      "frames": [{"sha256": frame["sha256"],
                                  "source_timestamp_ms": frame["source_timestamp_ms"]} for frame in case["frames"]],
                      "annotation": ({"status": annotation["status"], "checks": deepcopy(annotation["checks"]),
                                      "origin": annotation["origin"]} if annotation else None)})
    return {"schema_version": VERSION, "scope": frozen["scope"], "freeze_sha256": frozen["freeze_sha256"],
            "contract_hash": envelope["contract_hash"], "dataset_id": public["dataset_id"],
            "protocol": benchmark._approved(envelope).to_dict(),
            "requirements": envelope["contract"]["requirements"],
            "concepts": deepcopy(envelope["contract"]["concepts"]),
            "sources": public["sources"], "cases": cases,
            "conditions": list(frozen["conditions"]), "planned_requests": frozen["planned_requests"],
            "total_planned_requests": frozen["total_planned_requests"], "max_requests": frozen["max_requests"],
            "implementation_sha256": frozen["implementation_sha256"], "method_versions": frozen["method_versions"],
            "created_unix_ms": frozen["created_unix_ms"],
            "approval_origin": envelope["approval"]["origin"],
            "human_reviewed_exact_contract": envelope["approval"]["human_reviewed_exact_contract"],
            "physical_action_execution_verified": False,
            "limits": ["Caller-authorized curated recorded-source pilot, not an independent audit or physical correction loop.",
                       "The declaration hash is not a trusted timestamp or proof of no earlier model exposure.",
                       "All predeclared cases remain denominators; missing/error predictions are not UNKNOWN."]}


def _safe_usage(raw: Any, ledger: Any) -> dict[str, Any]:
    if (not isinstance(raw, Mapping) or not isinstance(ledger, list)
            or any(not isinstance(entry, Mapping) for entry in ledger)):
        raise ValidationError("Client accounting is unavailable")
    recomputed = MetaClient.usage_summary(SimpleNamespace(ledger=ledger))
    if any(raw.get(name) != recomputed[name] for name in _USAGE_COUNTS + ("estimated_usd",)):
        raise ValidationError("Request/token/cost totals contradict the private attempt ledger")
    counts = {name: value if type(value := raw.get(name)) is int and value >= 0 else None for name in _USAGE_COUNTS}
    counts["request_attempts"] = len(ledger)
    cost = raw.get("estimated_usd")
    counts["estimated_usd"] = cost if type(cost) in (int, float) and math.isfinite(cost) and cost >= 0 else None
    counts["failed_or_unresolved_attempts"] = sum(entry.get("status") != "ok" for entry in ledger)
    counts["unreported_charge_possible"] = bool(counts["failed_or_unresolved_attempts"]
                                                or counts["spark_requests_with_unreported_token_usage"]
                                                or counts["estimated_usd"] is None)
    return counts


def _usage(client: Any) -> dict[str, Any]:
    """Keep accounting without exposing ledger URLs, provider IDs or error text."""
    return _safe_usage(client.usage_summary(), getattr(client, "ledger", []))


def _fresh(client: Any, planned: int, seen: list[Any]) -> None:
    if (type(getattr(client, "max_requests", None)) is not int or client.max_requests != planned
            or getattr(client, "retries", None) != 0 or getattr(client, "ledger", None) != []
            or any(client is prior for prior in seen)):
        raise ValidationError("Each part requires a fresh exact-budget client, zero retries and an empty ledger")
    seen.append(client)


def run(client_factory: Callable[[int], Any], out_dir: Path, *, root: Path = benchmark.ROOT) -> dict[str, Any]:
    """Run each (case, method) once; never resume, overwrite, or auto-retry."""
    output, frozen, singles = _load(out_dir, root)
    marker = output / "run-started.json"
    try:
        with marker.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps({"freeze_sha256": frozen["freeze_sha256"],
                                     "started_unix_ms": int(time.time() * 1000)}))
        marker.chmod(0o600)
    except FileExistsError:
        raise ValidationError("This prospective pilot has already started; automatic resume is disabled") from None
    if (output / "parts").exists() or (output / "run-state.json").exists():
        raise ValidationError("Prospective run output already exists")
    state: dict[str, Any] = {"freeze_sha256": frozen["freeze_sha256"], "status": "running", "parts": {},
                             "admission_stop_code": None}
    benchmark._save(output / "run-state.json", state)
    seen: list[Any] = []
    for record in frozen["cases"]:
        for condition in frozen["conditions"]:
            if state["admission_stop_code"] is not None:
                break
            _load(out_dir, root)  # Recheck source/code/all approvals before the next paid admission.
            identifier = record["case_id"]
            key = identifier + "/" + condition
            planned = record["planned_requests"][condition]
            destination = output / "parts" / identifier / condition
            client = None
            admitted = False
            started = time.monotonic()
            entry: dict[str, Any] = {"status": "running", "planned_requests": planned}
            state["parts"][key] = entry
            benchmark._save(output / "run-state.json", state)
            try:
                client = client_factory(planned)
                _fresh(client, planned, seen)
                admitted = True
                entry["inference_origin"] = ("live_meta_adapter" if isinstance(client, MetaClient)
                                             else "injected_client_not_verified_live")
                report = benchmark.run_benchmark(lambda budget: client, singles[identifier], destination,
                                                  max_requests=planned, conditions=(condition,), project_root=root,
                                                  evaluation_role="new_source_holdback")
                sub = report["conditions"][condition]
                entry["status"] = sub["status"]
                if sub["status"] == "error":
                    entry["error_type"] = sub.get("error_type") if sub.get("error_type") in _ERROR_TYPES else "condition_error"
                code = sub.get("error_code")
                if type(code) is int and code in _STOP_CODES:
                    state["admission_stop_code"] = code
            except Exception as error:
                entry.update(status="error", error_type=type(error).__name__ if type(error).__name__ in _ERROR_TYPES else "condition_error")
                code = getattr(error, "code", None)
                if type(code) is int and code in _STOP_CODES:
                    state["admission_stop_code"] = code
                status = getattr(error, "status", None)
                if type(status) is int and status in _STOP_CODES:
                    state["admission_stop_code"] = status
            finally:
                # The benchmark intentionally catches MetaAPIError and saves its
                # string provider code. Actual HTTP status survives in the local
                # attempt ledger, independently of that provider-specific code.
                if client is not None and admitted:
                    for attempt in getattr(client, "ledger", []):
                        status = attempt.get("http_status") if isinstance(attempt, Mapping) else None
                        if type(status) is int and status in _STOP_CODES:
                            state["admission_stop_code"] = status
                            break
                if client is not None and admitted:
                    try:
                        entry["usage"] = _usage(client)
                    except Exception:
                        entry["usage"] = {"request_attempts": len(getattr(client, "ledger", [])),
                                          "estimated_usd": None, "unreported_charge_possible": True}
                else:
                    entry["usage"] = {"request_attempts": 0, "estimated_usd": 0, "unreported_charge_possible": False}
                entry["wall_seconds"] = round(time.monotonic() - started, 3)
                entry["artifact_sha256"] = {str(path.relative_to(destination)): benchmark._file_hash(path)
                                            for path in sorted(destination.rglob("*.json"))} if destination.exists() else {}
                benchmark._save(output / "run-state.json", state)
    state["status"] = "completed" if all(value["status"] == "completed" for value in state["parts"].values()) and len(state["parts"]) == len(frozen["cases"]) * len(frozen["conditions"]) else "incomplete"
    benchmark._save(output / "run-state.json", state)
    return summarize(out_dir, root=root)


def summarize(out_dir: Path, *, root: Path = benchmark.ROOT) -> dict[str, Any]:
    """Replay every saved part against exact media, approvals and private responses."""
    output, frozen, singles = _load(out_dir, root)
    public = public_freeze(out_dir, root=root)
    state_path = output / "run-state.json"
    state = _read(state_path) if state_path.exists() else {"status": "not_started", "parts": {}, "admission_stop_code": None}
    if state.get("freeze_sha256", frozen["freeze_sha256"]) != frozen["freeze_sha256"]:
        raise ValidationError("Run state belongs to a different prospective freeze")
    expected_keys = {record["case_id"] + "/" + condition for record in frozen["cases"] for condition in frozen["conditions"]}
    if not isinstance(state.get("parts"), Mapping) or set(state["parts"]) - expected_keys:
        raise ValidationError("Run state contains unplanned parts")
    if state.get("status") not in ("not_started", "running", "completed", "incomplete"):
        raise ValidationError("Run state has an invalid completion state")
    if state.get("admission_stop_code") is not None and (type(state["admission_stop_code"]) is not int or state["admission_stop_code"] not in _STOP_CODES):
        raise ValidationError("Run state has an invalid admission stop code")
    expected_reports = {output / "parts" / record["case_id"] / name / "report.json"
                        for record in frozen["cases"] for name in frozen["conditions"]}
    if set((output / "parts").rglob("report.json")) - expected_reports:
        raise ValidationError("The pilot contains an unplanned part report")
    conditions = {}
    for condition in frozen["conditions"]:
        predictions = []
        records = []
        attempts = tokens_in = tokens_out = 0
        total_cost = total_wall = 0.0
        unknown_cost = unknown_tokens = ambiguous = False
        for case_record in frozen["cases"]:
            identifier = case_record["case_id"]
            key = identifier + "/" + condition
            entry = state["parts"].get(key, {})
            destination = output / "parts" / identifier / condition
            report_path = destination / "report.json"
            annotation = singles[identifier]["contract"]["cases"][0]["annotation"]
            record = {"case_id": identifier, "truth": annotation["status"] if annotation else None,
                      "scored": annotation is not None, "status": "not_attempted", "prediction": None,
                      "missing_prediction": True, "planned_requests": case_record["planned_requests"][condition]}
            interrupted_accounting = bool(entry and entry.get("status") == "running" and "usage" not in entry)
            usage = entry.get("usage", {"request_attempts": 0, "estimated_usd": None,
                                        "unreported_charge_possible": True} if interrupted_accounting else {})
            if entry:
                if entry.get("status") not in ("running", "completed", "error"):
                    raise ValidationError("A part has an invalid completion state")
                if (not isinstance(usage, Mapping) or type(usage.get("request_attempts")) is not int
                        or not 0 <= usage["request_attempts"] <= case_record["planned_requests"][condition]):
                    raise ValidationError("A part has invalid request accounting")
                if any(usage.get(name) is not None and (type(usage[name]) is not int or usage[name] < 0)
                       for name in _USAGE_COUNTS):
                    raise ValidationError("A part has invalid token accounting")
                if usage.get("estimated_usd") is not None and benchmark._safe_number(usage["estimated_usd"]) is None:
                    raise ValidationError("A part has invalid cost accounting")
                public_usage = {name: benchmark._safe_number(usage.get(name)) for name in _USAGE_COUNTS}
                public_usage.update(estimated_usd=benchmark._safe_number(usage.get("estimated_usd")),
                                    failed_or_unresolved_attempts=benchmark._safe_number(usage.get("failed_or_unresolved_attempts")),
                                    unreported_charge_possible=usage.get("unreported_charge_possible") is True)
                origin = entry.get("inference_origin")
                record.update(status=entry["status"],
                              inference_origin=origin if origin in ("live_meta_adapter", "injected_client_not_verified_live") else "unverified",
                              usage=public_usage, wall_seconds=benchmark._safe_number(entry.get("wall_seconds")))
                record["request_accounting_incomplete"] = interrupted_accounting
                if "error_type" in entry:
                    record["error_type"] = entry["error_type"] if entry["error_type"] in _ERROR_TYPES else "condition_error"
                recorded = entry.get("artifact_sha256", {})
                actual = {str(path.relative_to(destination)): benchmark._file_hash(path)
                          for path in sorted(destination.rglob("*.json"))} if destination.exists() else {}
                if entry["status"] != "running" and recorded != actual:
                    raise ValidationError("A sealed private part report or response changed")
            if report_path.exists():
                if not entry:
                    raise ValidationError("A part report has no recorded admission")
                report = _read(report_path)
                if set(report.get("conditions", {})) != {condition}:
                    raise ValidationError("A part report contains an unplanned method")
                saved_approval = _read(destination / "approved-manifest.json")
                if saved_approval != singles[identifier]:
                    raise ValidationError("A part report changed its approved case")
                raw_path = destination / condition / "results.json"
                sub = report["conditions"][condition]
                if not raw_path.exists() and sub["status"] != "running":
                    raise ValidationError("A finished report requires its private inference sidecar")
                verification = benchmark.validate_report(singles[identifier], report,
                                                         raw_results={condition: _read(raw_path)} if raw_path.exists() else None)
                if sub.get("usage") is not None:
                    saved_usage = sub["usage"]
                    replayed_usage = _safe_usage(saved_usage, saved_usage.get("requests"))
                    if entry.get("status") != "running" and usage != replayed_usage:
                        raise ValidationError("Part accounting differs from its private request ledger")
                if entry.get("status") == "completed" and sub["status"] != "completed":
                    raise ValidationError("Run state and part completion disagree")
                if sub["status"] == "completed":
                    prediction = sub["cases"][0]
                    predictions.append(prediction)
                    record.update(status="completed", prediction=prediction["status"], missing_prediction=False,
                                  checks=[{"check_id": item["check_id"], "status": item["status"],
                                           **({"gap_reason": _GAPS.get(item.get("reason"), "grounding_gap")
                                               if condition != "spark" else "model_unknown"}
                                              if item["status"] == "unknown" else {})} for item in prediction["checks"]],
                                  inference_sidecar_replayed=verification["inference_sidecars_replayed"] == 1)
            elif entry.get("status") == "completed":
                raise ValidationError("A completed part report is missing")
            attempts += usage.get("request_attempts", 0)
            for name in ("input_tokens", "output_tokens"):
                if entry and usage.get(name) is None:
                    unknown_tokens = True
            tokens_in += usage.get("input_tokens") or 0
            tokens_out += usage.get("output_tokens") or 0
            cost = usage.get("estimated_usd")
            if entry and cost is None:
                unknown_cost = True
            total_cost += cost or 0
            wall = benchmark._safe_number(entry.get("wall_seconds"))
            if entry and entry.get("wall_seconds") is not None and wall is None:
                raise ValidationError("A part has invalid elapsed-time accounting")
            total_wall += wall or 0
            ambiguous |= bool(usage.get("unreported_charge_possible")) or bool(entry and entry["status"] == "running")
            records.append(record)
        conditions[condition] = {"metrics": benchmark.comparison_metrics(frozen["envelope"], predictions),
                                 "cases": records, "request_attempts": attempts,
                                 "request_accounting_incomplete": any(row.get("request_accounting_incomplete") for row in records),
                                 "reported_input_tokens": tokens_in, "reported_output_tokens": tokens_out,
                                 "token_usage_incomplete": unknown_tokens, "estimated_usd": round(total_cost, 6),
                                 "cost_estimate_incomplete": unknown_cost, "unreported_charge_possible": ambiguous,
                                 "wall_seconds": round(total_wall, 3)}
    public.update(status=state["status"], condition_results=conditions,
                  admission_stop_code=state.get("admission_stop_code"),
                  accounting_note="Costs are estimates, not billing hard stops; failures can have unreported charges.")
    return public
