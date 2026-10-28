"""Paired, source-grouped diagnostics for validated recorded-video decisions.

This adds no inference and never tunes confidence thresholds after seeing labels.
UNKNOWN ground truth is an evidence-insufficiency label, not a known task failure.
"""
from __future__ import annotations

import argparse
import hashlib
from itertools import combinations
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

from .benchmark import STATUSES, _read, validate_report
from .core import ValidationError

SCHEMA_VERSION = "1.0"


def _fraction(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 6) if denominator else None


def _metrics(rows: Sequence[Mapping[str, Any]], condition: str) -> dict[str, Any]:
    """All denominators are explicit; an empty denominator is unavailable."""
    confusion = {truth: {prediction: 0 for prediction in STATUSES} for truth in STATUSES}
    for row in rows:
        confusion[row["expected"]][row["predictions"][condition]] += 1
    totals = {truth: sum(confusion[truth].values()) for truth in STATUSES}
    eligible = len(rows)
    known = totals["pass"] + totals["fail"]
    known_decided = sum(confusion[truth][prediction] for truth in ("pass", "fail")
                        for prediction in ("pass", "fail"))
    known_correct = confusion["pass"]["pass"] + confusion["fail"]["fail"]
    accepted = sum(confusion[truth]["pass"] for truth in STATUSES)
    known_fail_accepted = confusion["fail"]["pass"]
    unknown_accepted = confusion["unknown"]["pass"]
    unknown_decided = confusion["unknown"]["pass"] + confusion["unknown"]["fail"]
    correct = known_correct + confusion["unknown"]["unknown"]
    abstained = sum(confusion[truth]["unknown"] for truth in STATUSES)
    return {
        "eligible": eligible, "ground_truth_counts": totals, "correct": correct,
        "accuracy": _fraction(correct, eligible), "confusion": confusion,
        "known_truth_count": known, "known_truth_decided": known_decided,
        "known_truth_abstained": known - known_decided,
        "known_truth_coverage": _fraction(known_decided, known),
        "known_truth_decided_correct": known_correct,
        "known_truth_selective_accuracy": _fraction(known_correct, known_decided),
        "known_fail_count": totals["fail"], "known_fail_accepted": known_fail_accepted,
        "known_fail_acceptance_rate": _fraction(known_fail_accepted, totals["fail"]),
        "unknown_truth_count": totals["unknown"],
        "unknown_truth_abstained": confusion["unknown"]["unknown"],
        "unknown_truth_decided": unknown_decided,
        "unknown_truth_accepted": unknown_accepted,
        "unknown_truth_acceptance_rate": _fraction(unknown_accepted, totals["unknown"]),
        "accepted_decisions": accepted,
        "accepted_decisions_not_supported_by_label": known_fail_accepted + unknown_accepted,
        "accepted_decision_label_error_rate": _fraction(known_fail_accepted + unknown_accepted, accepted),
        "abstained": abstained, "abstention_rate": _fraction(abstained, eligible),
        "correct_acceptances": confusion["pass"]["pass"],
        "correct_acceptance_rate": _fraction(confusion["pass"]["pass"], totals["pass"]),
    }


def _paired(rows: Sequence[Mapping[str, Any]], left: str, right: str) -> dict[str, Any]:
    both_correct = left_only = right_only = neither_correct = 0
    disagreements = []
    for row in rows:
        left_status, right_status = row["predictions"][left], row["predictions"][right]
        left_correct, right_correct = left_status == row["expected"], right_status == row["expected"]
        both_correct += left_correct and right_correct
        left_only += left_correct and not right_correct
        right_only += right_correct and not left_correct
        neither_correct += not left_correct and not right_correct
        if left_status != right_status:
            disagreements.append({key: row[key] for key in ("case_id", "source_sha256", "expected")}
                                 | ({"check_id": row["check_id"]} if "check_id" in row else {})
                                 | {"left_status": left_status, "right_status": right_status,
                                    "left_matches_label": left_correct, "right_matches_label": right_correct})
    return {"compared": len(rows), "agree": len(rows) - len(disagreements),
            "disagree": len(disagreements), "both_correct": both_correct,
            "only_left_correct": left_only, "only_right_correct": right_only,
            "neither_correct": neither_correct, "disagreements": disagreements}


def evaluation_summary(envelope: Mapping[str, Any], report: Mapping[str, Any], *,
                       raw_results: Mapping[str, Mapping[str, Any]] | None = None) -> dict[str, Any]:
    """Return an allowlisted diagnostic DTO after the shared report replay.

    Completed conditions are compared on the exact same approved cases/checks.
    Running/error conditions are listed as excluded, never fabricated UNKNOWNs.
    Source groups collapse byte-identical encoded videos, including ID aliases.
    Different file hashes do not establish independent recordings or captures.
    """
    integrity = validate_report(envelope, report, raw_results=raw_results)
    contract = envelope["contract"]
    completed = sorted(name for name, value in report["conditions"].items() if value["status"] == "completed")
    sources = {source["id"]: source for source in contract["sources"]}
    cases = sorted((case for case in contract["cases"] if case["role"] == "benchmark"), key=lambda case: case["id"])
    predicted = {name: {case["case_id"]: case for case in report["conditions"][name]["cases"]}
                 for name in completed}
    case_rows, check_rows = [], []
    groups: dict[str, dict[str, Any]] = {}
    for case in cases:
        source_hash = sources[case["source_id"]]["sha256"]
        group = groups.setdefault(source_hash, {"source_sha256": source_hash, "source_ids": [], "case_ids": []})
        if case["source_id"] not in group["source_ids"]:
            group["source_ids"].append(case["source_id"])
        group["case_ids"].append(case["id"])
        base = {"case_id": case["id"], "source_sha256": source_hash}
        case_rows.append({**base, "expected": case["annotation"]["status"],
                          "predictions": {name: predicted[name][case["id"]]["status"] for name in completed}})
        for check_id, expected in sorted(case["annotation"]["checks"].items()):
            check_rows.append({**base, "check_id": check_id, "expected": expected,
                               "predictions": {name: next(item["status"] for item in predicted[name][case["id"]]["checks"]
                                                          if item["check_id"] == check_id) for name in completed}})
    source_groups = []
    for source_hash, group in sorted(groups.items()):
        source_groups.append({**group, "source_ids": sorted(group["source_ids"]), "case_count": len(group["case_ids"])})
    conditions = {}
    for name in completed:
        conditions[name] = {
            "case_metrics": _metrics(case_rows, name), "check_metrics": _metrics(check_rows, name),
            "by_source": [{"source_sha256": group["source_sha256"],
                           "case_metrics": _metrics([row for row in case_rows if row["source_sha256"] == group["source_sha256"]], name),
                           "check_metrics": _metrics([row for row in check_rows if row["source_sha256"] == group["source_sha256"]], name)}
                          for group in source_groups],
        }
    # Hash only allowlisted experimental decisions. It identifies the comparison,
    # not a unique inference run, a trusted timestamp, or provider authenticity.
    projection = {"schema_version": SCHEMA_VERSION, "contract_hash": envelope["contract_hash"],
                  "evaluation_role": report.get("evaluation_role"),
                  "conditions": {name: {"status": value["status"],
                                        **({"observation_version": value["observation_version"]}
                                           if "observation_version" in value else {})}
                                 for name, value in sorted(report["conditions"].items())},
                  "case_decisions": case_rows, "check_decisions": check_rows}
    comparison_id = hashlib.sha256(json.dumps(projection, sort_keys=True, separators=(",", ":"),
                                              allow_nan=False).encode()).hexdigest()
    return {
        "schema_version": SCHEMA_VERSION, "comparison_id": comparison_id,
        "contract_hash": envelope["contract_hash"], "dataset_id": contract["dataset_id"],
        "evaluation_role": report.get("evaluation_role"), "report_integrity": integrity,
        "benchmark_case_count": len(cases),
        "illustrative_cases_excluded": len(contract["cases"]) - len(cases),
        "source_id_count": sum(len(group["source_ids"]) for group in source_groups),
        "source_file_group_count": len(source_groups), "source_groups": source_groups,
        "conditions": conditions,
        "excluded_conditions": {name: value["status"] for name, value in sorted(report["conditions"].items())
                                if value["status"] != "completed"},
        "pairwise": [{"left": left, "right": right, "cases": _paired(case_rows, left, right),
                      "checks": _paired(check_rows, left, right)} for left, right in combinations(completed, 2)],
        "physical_action_execution_verified": False,
        "limits": [
            "Descriptive selected-case results only; no population reliability, significance, or confidence-interval claim.",
            "Cases/checks from a source are correlated; distinct encoded source hashes do not prove independent captures.",
            "Known-truth coverage and selective accuracy exclude UNKNOWN ground truth. UNKNOWN labels are reported separately.",
            "FAIL-to-PASS uses known FAIL labels as its denominator; UNKNOWN-to-PASS is a separate unsupported acceptance.",
            "Selective accuracy can rise by abstaining. Compare it with coverage and correct acceptance on identical cases.",
            "A deterministic comparison ID binds decisions, not a unique inference run or trusted provenance.",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("report", type=Path)
    parser.add_argument("--output", type=Path, help="New diagnostic JSON file; existing files are not overwritten")
    args = parser.parse_args(argv)
    try:
        value = evaluation_summary(_read(args.manifest), _read(args.report))
        data = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
        if args.output:
            with args.output.open("x", encoding="utf-8") as stream:
                stream.write(data)
        else:
            print(data, end="")
        return 0
    except (ValidationError, ValueError, OSError) as error:
        print(f"Evaluation diagnostics stopped: {type(error).__name__}; check approved inputs and output.", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
