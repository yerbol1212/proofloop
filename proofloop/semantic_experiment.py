"""Read-only replay of the fixed 2026-10-28 semantic feasibility experiment.

Failed API attempts stay unscored. Public classifications, inputs, intermediate
locator anchors and costs must match their locally retained records.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .core import ValidationError
from .guided import GUIDED_VERSION, _frames, _locator_bound, normalized_box_to_pixels
from .meta import MetaClient, SPARK_MODEL
from .semantic import (SEMANTIC_VERSION, _hash, challenge_prompt, challenge_schema,
                       public_challenge_summary, verify_prepared)

PUBLIC_REPORT = "artifacts/feasibility/semantic-candidate-challenge-20261028.json"
_BASE = "artifacts/local/semantic-review-20261028-"
_CASE_IDS = ("candy-e00-early-unsupported", "candy-e00-visible-collection",
             "coffee-e04-dispense", "coffee-e05-staging")
_LIMITS = [
    "Selected development cases on previously exposed sources and previously generated locator anchors; not a held-out comparison.",
    "Same Spark model assigned and challenged candidates; errors can remain correlated and agreement is not independent identity proof.",
    "No location PASS or guided-v2 acceptance changes are produced by this diagnostic.",
    "The early-candy 2048-cap call and its explicit 4096-cap development rerun returned incomplete; both costs are included and neither is scored. The other three preselected positive controls received one 4096-cap call each.",
    "Duplicate matching objects, printed adversarial instructions and broader contradictory attributes lack real-media evaluation in this batch.",
    "The experiment compares scoped semantic candidate support, not historical placement accuracy or physical task success.",
    "The first two rejected responses retain usage ledgers and the observed failure classification; their raw provider response bodies were not retained.",
]


def _path(root: Path, value: str | Path) -> Path:
    path = (root / value).resolve()
    try:
        path.relative_to(root)
    except ValueError:
        raise ValidationError("Semantic experiment evidence must stay inside the project") from None
    if not path.is_file():
        raise ValidationError("Required local semantic experiment evidence is unavailable")
    return path


def _read(root: Path, value: str | Path) -> dict[str, Any]:
    try:
        data = json.loads(_path(root, value).read_text())
    except (OSError, UnicodeError, ValueError):
        raise ValidationError("Invalid local semantic experiment JSON") from None
    if not isinstance(data, dict):
        raise ValidationError("Semantic experiment records require JSON objects")
    return data


def _freeze(root: Path, suffix: str) -> dict[str, Any]:
    frozen = _read(root, _BASE + suffix + "/freeze-approved.json")
    if frozen.get("freeze_hash") != _hash({key: value for key, value in frozen.items() if key != "freeze_hash"}):
        raise ValidationError("Semantic experiment freeze changed")
    if (frozen.get("method_version") != SEMANTIC_VERSION
            or frozen.get("evaluation_role") != "development_reprocessing"
            or frozen.get("automatic_retries") != 0
            or [entry.get("case_id") for entry in frozen.get("cases", [])] != list(_CASE_IDS)):
        raise ValidationError("Semantic experiment reviewed scope changed")
    return frozen


def _reviewed_candidate(root: Path, entry: Mapping[str, Any]) -> dict[str, Any]:
    prepared = _read(root, entry["prepared_file"])
    verify_prepared(prepared)
    if (prepared["prepared_hash"] != entry["prepared_hash"]
            or entry.get("reviewed_source_and_crops") is not True
            or entry.get("review_origin") != "assistant_visual_review_before_semantic_inference"
            or entry.get("prompt_sha256") != hashlib.sha256(challenge_prompt(prepared).encode()).hexdigest()
            or entry.get("schema_sha256") != _hash(challenge_schema(prepared["contract"]))):
        raise ValidationError("Semantic candidate differs from the prospectively reviewed source/crop contract")
    guided = _read(root, entry["prior_guided_artifact"])
    if guided.get("provenance", {}).get("version") != GUIDED_VERSION:
        raise ValidationError("Original semantic candidate must come from the frozen guided v2 method")
    sources, frames = _frames([_path(root, frame["source_file"]) for frame in prepared["frames"]])
    if guided.get("frames") != frames or prepared["guided_locator_hash"] != _hash(guided["locator"]):
        raise ValidationError("Semantic original locator/source binding changed")
    object_ids = [item["object_id"] for item in guided["locator"]["parsed"]["frames"][0]["objects"]]
    located = _locator_bound(guided["locator"], frames, {key: "original reviewed target" for key in object_ids})
    for frame, prior, source in zip(prepared["frames"], located, frames):
        target = next(item for item in prior["objects"] if item["object_id"] == prepared["contract"]["object_id"])
        if (len(target["candidates"]) != 1
                or frame["candidate_box"] != normalized_box_to_pixels(target["candidates"][0], source["width"], source["height"])
                or frame["original_candidate_count"] != len(target["candidates"])
                or frame["original_locator"] != {key: target[key] for key in (
                    "identity_ambiguous", "fully_visible", "boundary_uncertain")}):
            raise ValidationError("Semantic candidate anchor/flags differ from the actual original locator")
        _path(root, frame["crop_file"])
    return prepared


def _failed_usage(saved: Any) -> dict[str, Any]:
    if not isinstance(saved, Mapping) or not isinstance(saved.get("requests"), list) or len(saved["requests"]) != 1:
        raise ValidationError("Failed semantic attempt requires its one-request ledger")
    ledger = saved["requests"]
    entry = ledger[0]
    if (not isinstance(entry, Mapping) or entry.get("model") != SPARK_MODEL or entry.get("endpoint") != "responses"
            or entry.get("attempt") != 1 or entry.get("status") != "invalid_response"
            or not isinstance(entry.get("usage"), Mapping)
            or any(type(entry["usage"].get(key)) is not int or entry["usage"][key] < 0
                   for key in ("input_tokens", "output_tokens"))):
        raise ValidationError("Failed semantic attempt ledger has unsupported provenance")
    replay = MetaClient("offline-semantic-failure-ledger", max_requests=1, retries=0)
    replay.ledger = deepcopy(ledger)
    recomputed = replay.usage_summary()
    if recomputed != saved:
        raise ValidationError("Failed semantic attempt cost totals changed")
    return {key: recomputed[key] for key in ("request_attempts", "input_tokens", "output_tokens", "estimated_usd")}


def build_public_experiment(project_root: str | Path) -> dict[str, Any]:
    """Regenerate the complete allowlisted summary from the fixed local run."""
    root = Path(project_root).resolve()
    first, second, frozen = (_freeze(root, suffix) for suffix in ("01", "02", "03"))
    if (second.get("parent_freeze_hash") != first["freeze_hash"]
            or frozen.get("parent_freeze_hash") != second["freeze_hash"]
            or second.get("prior_module_sha256") != first["module_sha256"]
            or frozen.get("prior_module_sha256") != second["module_sha256"]
            or first["cases"] != second["cases"] or first["cases"] != frozen["cases"]
            or frozen.get("max_total_attempts_including_prior_incomplete") != 5
            or frozen.get("estimated_allocation_usd") != 0.25):
        raise ValidationError("Semantic prospective freeze chain changed")
    if hashlib.sha256(_path(root, "proofloop/semantic.py").read_bytes()).hexdigest() != frozen["module_sha256"]:
        raise ValidationError("Semantic diagnostic module differs from its last recorded execution freeze")
    if hashlib.sha256(_path(root, "proofloop/guided.py").read_bytes()).hexdigest() != frozen["guided_v2_module_sha256"]:
        raise ValidationError("Historical guided v2 source changed")
    cases, unavailable, attempts = [], [], []
    for index, (suffix, method, cap) in enumerate((("01", first, 2048), ("02", second, 4096))):
        record = _read(root, _BASE + suffix + "/stopped-error.json")
        if record.get("case_id") != _CASE_IDS[0] or record.get("exception_class") != "MetaAPIError":
            raise ValidationError("Original failed semantic attempt changed")
        usage = _failed_usage(record["usage"])
        if usage["output_tokens"] != cap:
            raise ValidationError("Original incomplete semantic attempt token cap changed")
        attempts.append({"attempt_index": index + 1, "case_id": _CASE_IDS[0], "output_token_cap": cap,
                         "status": "incomplete", "scored": False, "method_module_sha256": method["module_sha256"],
                         "raw_response_retained": False,
                         "failure_evidence": "observed_classification_and_usage_ledger_only",
                         "usage": usage})
    for index, entry in enumerate(frozen["cases"]):
        prepared = _reviewed_candidate(root, entry)
        metadata = {key: deepcopy(entry[key]) for key in (
            "case_id", "source_family", "source_id", "source_timestamps_ms", "expected_identity")}
        if metadata["expected_identity"] != ("unknown" if index == 0 else "supported"):
            raise ValidationError("Semantic candidate annotation differs from the reviewed fixed experiment")
        if index == 0:
            unavailable.append(dict(metadata, status="incomplete", attempt_indices=[1, 2],
                                    raw_response_retained=False,
                                    failure_evidence="observed_classification_and_usage_ledger_only"))
            continue
        directory = _BASE + "03/" + entry["case_id"] + "/"
        attempt = _read(root, directory + "attempt.json")
        if (attempt.get("attempt_index") != index + 2 or attempt.get("case_id") != entry["case_id"]
                or attempt.get("method_module_sha256") != frozen["module_sha256"]
                or attempt.get("output_token_cap") != 4096):
            raise ValidationError("Remaining semantic attempt differs from its frozen request scope")
        if attempt.get("status") == "completed" and attempt.get("scored") is True:
            report = _read(root, directory + "live-challenge.json")
            if report["prepared"] != prepared:
                raise ValidationError("Completed semantic request used a different prepared candidate")
            summary = public_challenge_summary(report)
            if attempt["usage"] != report["usage"]:
                raise ValidationError("Semantic attempt ledger differs from its recorded completed response")
            usage = summary["usage"]
            cases.append(dict(metadata, challenge=summary))
        elif attempt.get("status") == "incomplete" and attempt.get("scored") is False:
            usage = _failed_usage(attempt["usage"])
            response = attempt.get("response")
            if (attempt.get("error_code") != "incomplete_response" or not isinstance(response, Mapping)
                    or response.get("model") != SPARK_MODEL or response.get("status") != "incomplete"
                    or response.get("usage") != attempt["usage"]["requests"][0]["usage"]):
                raise ValidationError("Unscored semantic failure does not match its retained API response")
            unavailable.append(dict(metadata, status="incomplete", attempt_indices=[index + 2],
                                    raw_response_retained=True,
                                    failure_evidence="retained_response_and_usage_ledger"))
        else:
            raise ValidationError("Unsupported semantic attempt outcome; no classification can be invented")
        attempts.append({"attempt_index": index + 2, "case_id": entry["case_id"], "output_token_cap": 4096,
                         "status": attempt["status"], "scored": attempt["scored"],
                         "raw_response_retained": True,
                         "failure_evidence": ("not_applicable_completed_response" if attempt["scored"]
                                              else "retained_response_and_usage_ledger"),
                         "method_module_sha256": frozen["module_sha256"], "usage": usage})
    timing = _read(root, _BASE + "03/batch-timing.json")
    cost = round(sum(attempt["usage"]["estimated_usd"] for attempt in attempts), 6)
    seconds = timing.get("completed_batch_wall_seconds")
    if (timing.get("estimated_total_usd") != cost or type(seconds) not in (int, float) or not 0 <= seconds <= 1000):
        raise ValidationError("Semantic total cost/timing annotation changed")
    negative = [case for case in cases if case["expected_identity"] == "unknown"]
    positive = [case for case in cases if case["expected_identity"] == "supported"]
    return {"schema_version": 1, "experiment": "source_bound_semantic_candidate_challenge_20261028",
            "method_version": SEMANTIC_VERSION, "freeze_hash": frozen["freeze_hash"],
            "parent_freeze_hash": frozen["parent_freeze_hash"], "method_module_sha256": frozen["module_sha256"],
            "guided_v2_module_sha256_unchanged": frozen["guided_v2_module_sha256"],
            "evaluation_role": "development_reprocessing", "cases": cases, "unavailable_cases": unavailable,
            "attempts": attempts, "metrics": {"requested_cases": 4, "completed_cases": len(cases),
                "source_recordings": len({entry["source_id"] for entry in frozen["cases"]}),
                "expected_identity_matches": sum(case["expected_identity"] == case["challenge"]["decision"] for case in cases),
                "negative_candidate_pairs": 1, "scored_negative_pairs": len(negative),
                "negative_pairs_withheld": sum(case["challenge"]["decision"] == "unknown" for case in negative),
                "positive_candidate_pairs": 3, "scored_positive_pairs": len(positive),
                "positive_pairs_supported": sum(case["challenge"]["decision"] == "supported" for case in positive)},
            "cost_summary": {"actual_requests": len(attempts), "completed_scored_requests": len(cases),
                "unscored_incomplete_requests": sum(attempt["status"] == "incomplete" for attempt in attempts),
                "estimated_usd": cost, "completed_batch_wall_seconds": seconds, "max_total_attempts": 5,
                "automatic_retries": 0, "explicit_development_reruns": 1, "estimated_allocation_usd": 0.25,
                "estimate_is_billing_guarantee": False},
            "source_attribution": {
                "candy": "https://huggingface.co/datasets/lerobot/aloha_static_candy/blob/7a19ab84a6b7f376d699c7f12c28535725fd0593/README.md",
                "coffee": "https://huggingface.co/datasets/lerobot/aloha_static_coffee/blob/1c52c4c794b6f93d931489263cf15e4331600bfc/README.md",
                "license_basis": "Pinned dataset cards explicitly declare MIT; authored crop derivations remain private."},
            "limitations": deepcopy(_LIMITS)}


def replay_public_experiment(project_root: str | Path) -> dict[str, Any]:
    """Return the fixed safe DTO only after the public aggregate exactly replays."""
    root = Path(project_root).resolve()
    try:
        rebuilt = build_public_experiment(root)
        if _read(root, PUBLIC_REPORT) != rebuilt:
            raise ValidationError("Public semantic experiment differs from its replayed sources, decisions or costs")
        return rebuilt
    except (KeyError, TypeError, ValueError, StopIteration):
        raise ValidationError("Semantic experiment cannot be replayed from its local evidence") from None
