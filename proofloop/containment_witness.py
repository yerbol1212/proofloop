"""Zero-model, certificate-only control; never changes a core task verdict.

Nonempty target material outside a goal conditionally refutes whole containment.
It cannot locate the entire object elsewhere or override UNKNOWN policies.
"""
from copy import deepcopy
from pathlib import Path

from . import benchmark, crop_verification as verifier
from .core import ValidationError
from .grounding import _detection, _rectangles

VERSION = "conditional_goal_containment_counterexample_control_v1"
POLICY = "diagnostic_only_whole_projected_containment_no_core_override"
ASSUMPTION = ("The nonempty foreground includes positively identified target material. Existing model identity/purity "
              "classifications are fallible support, not proof of pixel membership or physical truth.")
CLAIM = "Some model-supported target material lies outside the goal region."
HYPOTHESIS = {"conditional_predicate": "all_projected_target_material_within_goal",
              "condition_source": "separately_declared_hypothesis_not_inferred_from_original_task",
              "original_task_violation_asserted": False}


def code_hashes():
    return {**verifier.code_hashes(), "containment_witness.py": benchmark._file_hash(Path(__file__).resolve())}


def _inventory(summary):
    cases, rows = summary.get("cases"), summary.get("results")
    if (not isinstance(cases, list) or not isinstance(rows, list) or len(cases) != 9 or len(rows) != 9
            or [case.get("case_id") for case in cases] != [row.get("case_id") for row in rows]
            or len({case.get("case_id") for case in cases}) != 9):
        raise ValidationError("The control must preserve all nine original cases in their exact declared order")
    records, seen_sources = [], set()
    for case, row in zip(cases, rows):
        sources = case.get("source_sha256s")
        if not isinstance(sources, list) or len(sources) != 2 or len(set(sources)) != 2:
            raise ValidationError("Each original case requires its two distinct original source hashes")
        if seen_sources.intersection(sources):
            raise ValidationError("The nine-case control requires eighteen globally distinct original sources")
        seen_sources.update(sources)
        for digest in sources + [case.get("contract_hash")]:
            benchmark._digest(digest)
        prediction = row.get("prediction")
        assessment = row.get("assessment")
        if prediction not in (None, "unknown", "pass", "fail") or assessment is not None and not isinstance(assessment, dict):
            raise ValidationError("Original verdict or assessment binding is invalid")
        evaluation_hash = assessment.get("evaluation_sha256") if assessment else None
        if evaluation_hash is not None:
            benchmark._digest(evaluation_hash)
        records.append({"case_id": case["case_id"], "contract_hash": case["contract_hash"],
                        "source_sha256s": deepcopy(sources), "original_core_verdict": prediction,
                        "evaluation_sha256": evaluation_hash, "original_result_row_sha256": benchmark.manifest_hash(row)})
    return records


def declare_control(original_all_case_summary):
    """Only declare hashes/inventory; do not replay a cached evaluation here."""
    declaration = {"version": VERSION, "policy": POLICY, "zero_model_calls": 0, **HYPOTHESIS,
        "original_summary_sha256": benchmark.manifest_hash(original_all_case_summary),
        "original_study_plan_sha256": original_all_case_summary.get("plan_sha256"),
        "original_gate_passed": original_all_case_summary.get("gate_passed"),
        "cases": _inventory(original_all_case_summary), "implementation_sha256": code_hashes(),
        "membership_assumption": ASSUMPTION, "never_grants_pass": True, "never_changes_core_verdict": True}
    benchmark._digest(declaration["original_study_plan_sha256"])
    declaration["declaration_sha256"] = benchmark.manifest_hash(declaration)
    return declaration


def assess(evaluation, declaration, *, root=benchmark.ROOT):
    """Return a public-safe conditional witness, or explain why none is supported."""
    if (declaration.get("version") != VERSION or declaration.get("policy") != POLICY
            or declaration.get("conditional_predicate") != HYPOTHESIS["conditional_predicate"]
            or declaration.get("condition_source") != HYPOTHESIS["condition_source"]
            or declaration.get("original_task_violation_asserted") is not False
            or type(declaration.get("zero_model_calls")) is not int or declaration["zero_model_calls"] != 0
            or declaration.get("never_grants_pass") is not True or declaration.get("never_changes_core_verdict") is not True
            or declaration.get("membership_assumption") != ASSUMPTION or declaration.get("implementation_sha256") != code_hashes()
            or declaration.get("declaration_sha256") != benchmark.manifest_hash({k: v for k, v in declaration.items() if k != "declaration_sha256"})
            or len(declaration.get("cases", [])) != 9 or len({row["case_id"] for row in declaration["cases"]}) != 9
            or any(len(row.get("source_sha256s", [])) != 2 for row in declaration["cases"])
            or len({digest for row in declaration["cases"] for digest in row["source_sha256s"]}) != 18):
        raise ValidationError("The separately published diagnostic declaration changed")
    report = verifier.derive_task_report(evaluation, root=root)
    pair = evaluation["prepared_final"]["pair"]
    envelope, frames = pair["envelope"], pair["frames"]
    case, = envelope["contract"]["cases"]
    declared = next((row for row in declaration["cases"] if row["case_id"] == case["id"]), None)
    if (declared is None or declared["evaluation_sha256"] != evaluation["evaluation_sha256"]
            or declared["contract_hash"] != envelope["contract_hash"]
            or declared["source_sha256s"] != [row["source_sha256"] for row in frames]
            or declared["original_core_verdict"] != report["overall"]):
        raise ValidationError("Cached evidence differs from the exact declared case, contract, sources or original verdict")
    protocol = benchmark._approved(envelope)
    reasons, witnesses = [], []
    check = protocol.checks[0] if len(protocol.checks) == 1 else None
    if check is None or check.kind != "location" or check.unique_target is not True:
        reasons.append("requires_one_approved_unique_target_location_check")
    if any(row["withheld_reasons"] != ["target_extent_covered_contradicted"] for row in evaluation["frames"]):
        reasons.append("not_exclusively_incomplete_target_extent")
    oldest = min(row["source_timestamp_ms"] for row in case["frames"])
    now = case["evaluation_timestamp_ms"]
    if check is not None:
        if len({row["source_sha256"] for row in frames}) < max(2, check.min_stable_samples):
            reasons.append("insufficient_distinct_stable_sources")
        if now - oldest > check.max_age_ms or any(row["source_timestamp_ms"] > now for row in case["frames"]):
            reasons.append("stale_or_future_source_evidence")
    if not reasons:
        rectangles = _rectangles(protocol, case["regions"])
        for source in frames:
            try:
                witness = _detection(source["target_record"], rectangles)
            except (ValidationError, KeyError, TypeError):
                reasons.append("material_not_wholly_within_one_reviewed_region")
                break
            region, goal = witness["region_id"], rectangles[check.region_id]
            other = rectangles[region]
            if region == check.region_id or not (other[2] < goal[0] or goal[2] < other[0] or other[3] < goal[1] or goal[3] < other[1]):
                reasons.append("witness_region_not_strictly_disjoint_from_goal")
            if witness["confidence"] < max(.8, check.min_confidence):
                reasons.append("reported_target_score_below_unchanged_threshold")
            witnesses.append({"source_sha256": source["source_sha256"], "projection_sha256": source["projection_sha256"],
                "witness_material_region_id": region, "reported_confidence": witness["confidence"],
                "returned_box": list(witness["box"]), "foreground_extent": list(witness["foreground"]),
                "raster_sha256": source["target_record"]["mask"]["raster_sha256"]})
        if len(witnesses) == 2 and witnesses[0]["witness_material_region_id"] != witnesses[1]["witness_material_region_id"]:
            reasons.append("witness_material_region_inconsistent_between_sources")
    supported = not reasons and len(witnesses) == 2
    certificate = {"version": VERSION, "kind": "goal_containment_counterexample", "policy": POLICY, **HYPOTHESIS,
        "declaration_sha256": declaration["declaration_sha256"], "case_id": case["id"],
        "status": "supported_counterexample" if supported else "withheld", "claim": CLAIM if supported else None,
        "membership_assumption": ASSUMPTION, "original_core_verdict": report["overall"],
        "whole_target_region_id": None, "repair_origin_available": False, "never_grants_pass": True,
        "never_changes_core_verdict": True, "no_current_authority": True, "physical_truth_verified": False,
        "boundary_crossing_unknown_policy_unchanged": True, "goal_region_id": check.region_id if check else None,
        "witness_material_region_id": witnesses[0]["witness_material_region_id"] if supported else None,
        "stable_source_count": len(witnesses) if supported else 0, "oldest_source_timestamp_ms": oldest,
        "minimum_reported_target_score": min(row["reported_confidence"] for row in witnesses) if supported else None,
        "witnesses": witnesses, "withheld_reasons": list(dict.fromkeys(reasons)), "model_requests": 0,
        "evaluation_sha256": evaluation["evaluation_sha256"]}
    certificate["certificate_sha256"] = benchmark.manifest_hash(certificate)
    return certificate
