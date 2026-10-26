"""Run the deterministic, synthetic oracle diagnostic without making API calls.

Usage from the project directory::

    python -m proofloop.feasibility --output artifacts/feasibility

This is a software diagnostic.  A successful result does not establish physical
perception accuracy, access to Meta models, or end-to-end product feasibility.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from proofloop.core import EvidenceStore, Protocol, RepairAction, evaluate, plan_repair
from proofloop.fixtures import DiagnosticCase, EVIDENCE_ORIGIN, diagnostic_cases, protocol_fixture


def _status(results: list[dict[str, Any]]) -> str:
    states = {result["status"] for result in results}
    if "fail" in states:
        return "fail"
    if not results or "unknown" in states:
        return "unknown"
    return "pass"


def _snapshot(protocol: Protocol, store: EvidenceStore, *, label: str,
              expected: str, now_ms: int) -> dict[str, Any]:
    checks = [result.to_dict() for result in evaluate(protocol, store, now_ms=now_ms)]
    actual = _status(checks)
    return {
        "label": label,
        "now_ms": now_ms,
        "expected": expected,
        "actual": actual,
        "assertion_passed": expected == actual,
        "checks": checks,
    }


def _run_case(case: DiagnosticCase) -> dict[str, Any]:
    protocol = Protocol.from_dict(case.protocol)
    store = EvidenceStore(protocol, session_id=f"synthetic:{case.case_id}")
    stages: list[dict[str, Any]] = []
    sources: dict[str, dict[str, Any]] = {}
    runtime_inputs: list[dict[str, Any]] = []
    latest_time = 0
    for index, event in enumerate(case.events):
        arguments = dict(event.arguments)
        if event.kind == "expect":
            latest_time = max(latest_time, arguments["now_ms"])
            stages.append(_snapshot(protocol, store, label=arguments["label"],
                                    expected=arguments["status"], now_ms=arguments["now_ms"]))
            continue
        latest_time = max(latest_time, arguments.get("timestamp_ms", 0))
        if event.kind == "observe":
            store.observe(**arguments)
            sources[arguments["evidence_id"]] = {
                "origin": EVIDENCE_ORIGIN,
                "object_id": arguments["object_id"],
                "region_id": arguments["region_id"],
                "timestamp_ms": arguments["timestamp_ms"],
                "media_type": "symbolic_fact",
                "physical_recording": False,
            }
        elif event.kind == "invalidate":
            store.invalidate(**arguments)
        elif event.kind == "start_attempt":
            store.start_attempt(**arguments)
        elif event.kind in {"record_checkpoint", "record_event"}:
            evidence_id = f"fixture://proofloop/{case.case_id}/{event.kind}-{index}"
            arguments.setdefault("evidence_id", evidence_id)
            getattr(store, event.kind)(**arguments)
            sources[evidence_id] = {
                "origin": EVIDENCE_ORIGIN,
                "timestamp_ms": arguments["timestamp_ms"],
                "media_type": "symbolic_event",
                "physical_recording": False,
            }
        else:
            raise ValueError(f"Unknown fixture event kind: {event.kind}")
        runtime_inputs.append({"kind": event.kind, "arguments": arguments})
    stages.append(_snapshot(protocol, store, label="final", expected=case.expected_status,
                            now_ms=latest_time + 100))
    return {
        "case_id": case.case_id,
        "family": case.family,
        "description": case.description,
        "evidence_origin": EVIDENCE_ORIGIN,
        "protocol_hash": protocol.content_hash,
        "protocol_version": case.protocol["version"],
        "expected": case.expected_status,
        "actual": stages[-1]["actual"],
        "assertion_passed": all(stage["assertion_passed"] for stage in stages),
        "stages": stages,
        "runtime_input_events": runtime_inputs,
        "evidence_sources": sources,
    }


def _planner_diagnostic() -> dict[str, Any]:
    action_data = [
        {
            "id": "move-screwdriver-left",
            "kind": "physical",
            "preconditions": {"screwdriver_region": "right"},
            "effects": {"screwdriver_region": "left"},
            "cost": 1,
        },
        {
            "id": "inspect-screwdriver-region",
            "kind": "inspect",
            "preconditions": {},
            "effects": {},
            "reveals": ["screwdriver_region"],
            "cost": 1,
        },
    ]
    initial_world = {"screwdriver_region": "right"}
    known = dict(initial_world)
    plan = plan_repair(initial_world=initial_world, known=known,
                       actions=[RepairAction.from_dict(action) for action in action_data],
                       goal={"screwdriver_region": "left"}, max_states=64)
    expected_actions = ["move-screwdriver-left", "inspect-screwdriver-region"]
    expected_cost = 2
    actual_actions = list(plan.action_ids) if plan else None
    actual_cost = plan.total_cost if plan else None
    return {
        "diagnostic_id": "repair-requires-reobservation",
        "evidence_origin": "synthetic_action_model",
        "initial_world": initial_world,
        "initial_known": known,
        "goal": {"screwdriver_region": "left"},
        "authored_actions": action_data,
        "expected_action_ids": expected_actions,
        "actual_action_ids": actual_actions,
        "expected_cost": expected_cost,
        "actual_cost": actual_cost,
        "assertion_passed": actual_actions == expected_actions and actual_cost == expected_cost,
        "plan": plan.to_dict() if plan else None,
        "claim_limit": "A shortest path in this authored finite model; physical repair still needs fresh observations.",
        "minimality_basis": "The sole world-changing move and a later inspection are both required; each costs one.",
    }


def run_diagnostic() -> dict[str, Any]:
    """Return public-safe raw results with model and camera claims excluded."""

    cases: list[dict[str, Any]] = []
    for case in diagnostic_cases():
        try:
            cases.append(_run_case(case))
        except Exception as exc:
            # A broken fixture or runtime is a failed diagnostic, never a skip.
            cases.append({
                "case_id": case.case_id,
                "family": case.family,
                "description": case.description,
                "evidence_origin": EVIDENCE_ORIGIN,
                "expected": case.expected_status,
                "actual": "error",
                "assertion_passed": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "stages": [],
                "runtime_input_events": [],
                "evidence_sources": {},
            })
    try:
        planner = _planner_diagnostic()
    except Exception as exc:
        planner = {"diagnostic_id": "repair-requires-reobservation", "assertion_passed": False,
                   "error_type": type(exc).__name__, "error": str(exc)}
    counts = Counter(case["actual"] for case in cases)
    expected_counts = Counter(case["expected"] for case in cases)
    stages = [stage for case in cases for stage in case["stages"]]
    passed = sum(case["assertion_passed"] for case in cases)
    checks = [check for stage in stages for check in stage["checks"]]
    return {
        "report_schema_version": "1.0",
        "title": "ProofLoop synthetic oracle diagnostic",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "evidence_origin": EVIDENCE_ORIGIN,
        "is_model_benchmark": False,
        "is_physical_benchmark": False,
        "api_requests": 0,
        "api_cost_usd": 0,
        "scope": "Deterministic checker, freshness, editable specification and bounded repair-planner integration.",
        "limitations": [
            "All observed facts and expected answers are authored synthetic fixtures.",
            "No image, video, phone recording, segmentation result or Meta model inference is evaluated here.",
            "Fixture assertion success establishes software behavior under supplied facts, not perception accuracy.",
            "The alternative-order fixture tests independent fact arrival; it does not infer real physical dependencies.",
            "Real Meta access and the camera-to-observation-to-correction loop require separate feasibility checks.",
        ],
        "summary": {
            "episodes": len(cases),
            "episodes_matching_all_expectations": passed,
            "episodes_with_mismatch_or_error": len(cases) - passed,
            "expected_final_outcomes": dict(sorted(expected_counts.items())),
            "actual_final_outcomes": dict(sorted(counts.items())),
            "stage_assertions": len(stages),
            "stage_assertions_passed": sum(stage["assertion_passed"] for stage in stages),
            "check_snapshots": len(checks),
            "check_status_counts": dict(sorted(Counter(check["status"] for check in checks).items())),
            "planner_assertions": 1,
            "planner_assertions_passed": int(planner["assertion_passed"]),
            "all_assertions_passed": passed == len(cases) and bool(planner["assertion_passed"]),
        },
        "baseline_protocol_hash": Protocol.from_dict(protocol_fixture()).content_hash,
        "cases": cases,
        "planner": planner,
    }


def render_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# ProofLoop synthetic oracle diagnostic", "",
        "This diagnostic supplies authored symbolic facts. It makes **zero API calls** and contains "
        "**no camera or model benchmark**. It checks software behavior under the supplied facts.", "",
        f"Generated (UTC): {report['generated_at_utc']}", "",
        f"Episode assertions: **{summary['episodes_matching_all_expectations']}/{summary['episodes']}**. "
        f"Stage assertions: **{summary['stage_assertions_passed']}/{summary['stage_assertions']}**. "
        f"Planner assertions: **{summary['planner_assertions_passed']}/{summary['planner_assertions']}**.", "",
        "| Synthetic episode | Expected | Actual | All stage assertions |", "| --- | --- | --- | --- |",
    ]
    for case in report["cases"]:
        lines.append(f"| {case['case_id']} | {case['expected']} | {case['actual']} | "
                     f"{'pass' if case['assertion_passed'] else 'FAIL'} |")
    lines += ["", "## Repair model", "", report["planner"].get("claim_limit", "Planner diagnostic failed."), "",
              f"Expected actions: `{report['planner'].get('expected_action_ids')}`; "
              f"returned actions: `{report['planner'].get('actual_action_ids')}`.", "",
              "## Interpretation limits", ""]
    lines.extend(f"- {limitation}" for limitation in report["limitations"])
    lines += ["", "The adjacent JSON contains each stage's expected/actual status, check reasons, evidence IDs, "
              "synthetic origins, specification hashes, raw counts and the authored repair action model.", ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/feasibility"),
                        help="Directory for public-safe oracle-report.json and oracle-report.md")
    args = parser.parse_args(argv)
    report = run_diagnostic()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "oracle-report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output / "oracle-report.md").write_text(render_markdown(report), encoding="utf-8")
    summary = report["summary"]
    print(f"Synthetic oracle diagnostic: {summary['episodes_matching_all_expectations']}/{summary['episodes']} "
          f"episodes; planner {summary['planner_assertions_passed']}/1; API requests 0.")
    print(f"Report: {args.output / 'oracle-report.md'}")
    return 0 if summary["all_assertions_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
