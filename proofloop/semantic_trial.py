"""Frozen, bounded semantic-v2 trials; saved-response replay makes no API calls.

Labels and source times are reviewed declarations, not trusted capture metadata.
Raw responses stay local. An incomplete request is unscored, never UNKNOWN.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Callable, Mapping, Sequence
from urllib.parse import urlsplit

from .core import ValidationError
from .semantic import _hash
from . import semantic_v2 as method

EXPERIMENT = "bounded_semantic_candidate_v2_trial"
ALLOCATION_USD = 0.5
_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
_ENTRY_KEYS = {"case_id", "expected_identity", "source_family", "source_id", "source_timestamps_ms",
               "anchor_origin", "source_attribution", "prepared_file", "prepared_hash"}
_ATTRIBUTION_KEYS = {"page_url", "license", "license_url", "title", "revision", "dataset_id", "source_url"}
_LIMITS = [
    "Development experiment; reviewed labels and source timestamps are self-reported declarations.",
    "Cases sharing recordings, source frames or models are correlated, not independent physical trials.",
    "Authored candidates test the challenger and do not measure automatic locator accuracy.",
    "Unscored API failures are not UNKNOWN predictions; rates use explicit completed-case denominators.",
    "Saved source/code hashes detect changes locally; they do not independently prove API origin or capture time.",
    "Allocation is an estimate, not a provider-enforced billing limit; unreported charges remain unknown.",
    "Semantic support never grants placement PASS, physical execution proof or changes guided-v2 acceptance.",
]


def _path(value: str | Path, root: Path, *, output: bool = False) -> Path:
    if not isinstance(value, (str, Path)):
        raise ValidationError("Trial path must be local")
    path = (root / value).resolve()
    boundary = root / "artifacts" / "local" if output else root
    if not path.is_relative_to(boundary) or path == boundary:
        raise ValidationError("Trial path is outside its local evidence boundary")
    return path


def _read(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
        if not isinstance(value, dict):
            raise ValueError()
        return value
    except (OSError, ValueError):
        raise ValidationError("Trial evidence file is missing or invalid") from None


def _write(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as target:
        target.write(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")
    path.chmod(0o600)


def _implementations() -> dict[str, str]:
    return {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ("semantic_trial.py", "semantic_v2.py", "semantic.py", "meta.py", "candidate_review.py")
            if Path(__file__).with_name(name).is_file()}


def _entries(entries: Sequence[Mapping[str, Any]], root: Path) -> list[dict[str, Any]]:
    if not isinstance(entries, (list, tuple)) or not 1 <= len(entries) <= 8:
        raise ValidationError("Freeze one to eight reviewed semantic cases")
    frozen, ids, preparations = [], set(), set()
    for entry in entries:
        if not isinstance(entry, Mapping) or set(entry) != _ENTRY_KEYS:
            raise ValidationError("Trial case metadata must use the exact reviewed fields")
        for key in ("case_id", "source_family", "source_id"):
            if not isinstance(entry[key], str) or not _ID.fullmatch(entry[key]):
                raise ValidationError("Trial case and source IDs must be concise safe identifiers")
        if entry["case_id"] in {"freeze", "run-started"}:
            raise ValidationError("Trial case ID is reserved")
        times, attribution = entry["source_timestamps_ms"], entry["source_attribution"]
        if (entry["expected_identity"] not in ("supported", "unknown")
                or entry["anchor_origin"] not in ("saved_spark_locator", "authored_candidate")
                or not isinstance(times, list) or len(times) != 2
                or any(type(value) is not int or value < 0 for value in times) or times[0] >= times[1]
                or not isinstance(attribution, Mapping) or not {"page_url", "license"} <= set(attribution)
                or set(attribution) - _ATTRIBUTION_KEYS
                or any(not isinstance(value, str) or not 1 <= len(value) <= 1200
                       or any(ord(char) < 32 for char in value) for value in attribution.values())):
            raise ValidationError("Trial labels, origin, timestamps or source attribution are invalid")
        for key, value in attribution.items():
            if key.endswith("url"):
                parsed = urlsplit(value)
                if parsed.scheme != "https" or not parsed.netloc or parsed.username or parsed.password:
                    raise ValidationError("Source attribution needs public HTTPS URLs")
        prepared_path = _path(entry["prepared_file"], root)
        prepared = _read(prepared_path)
        method.verify_prepared_v2(prepared)
        if (not isinstance(entry["prepared_hash"], str)
                or prepared["prepared_hash"] != entry["prepared_hash"]
                or prepared.get("anchor_origin", "saved_spark_locator") != entry["anchor_origin"]
                or entry["case_id"] in ids or entry["prepared_hash"] in preparations):
            raise ValidationError("Trial anchor origin, preparation binding or uniqueness changed")
        ids.add(entry["case_id"])
        preparations.add(entry["prepared_hash"])
        saved = deepcopy(dict(entry))
        saved["prepared_file"] = str(prepared_path.relative_to(root))
        saved["prepared_file_sha256"] = hashlib.sha256(prepared_path.read_bytes()).hexdigest()
        saved["request_binding"] = method.request_binding(prepared)
        frozen.append(saved)
    return frozen


def freeze_trial(entries: Sequence[Mapping[str, Any]], out_dir: str | Path, *, phase: str,
                 root: str | Path) -> dict[str, Any]:
    """Review before inference; refuse overwriting an earlier trial or preparation."""
    root = Path(root).resolve()
    if not isinstance(phase, str) or not _ID.fullmatch(phase):
        raise ValidationError("Trial phase must be a concise safe identifier")
    cases = _entries(entries, root)
    plan = {"schema_version": 1, "experiment": EXPERIMENT, "phase": phase,
            "method_version": method.SEMANTIC_V2_VERSION, "method_config": deepcopy(method.METHOD_CONFIG),
            "max_requests": len(cases), "estimated_allocation_usd": ALLOCATION_USD,
            "implementation_sha256": _implementations(), "cases": cases,
            "frozen_at_utc": datetime.now(timezone.utc).isoformat()}
    plan["freeze_hash"] = _hash(plan)
    folder = _path(out_dir, root, output=True)
    folder.mkdir(parents=True, exist_ok=False, mode=0o700)
    _write(folder / "freeze.json", plan)
    return plan


def _verify(run_dir: str | Path, root: Path) -> tuple[Path, dict[str, Any]]:
    folder = _path(run_dir, root, output=True)
    plan = _read(folder / "freeze.json")
    if (plan.get("schema_version") != 1 or plan.get("experiment") != EXPERIMENT
            or plan.get("method_version") != method.SEMANTIC_V2_VERSION
            or plan.get("method_config") != method.METHOD_CONFIG
            or plan.get("implementation_sha256") != _implementations()
            or plan.get("estimated_allocation_usd") != ALLOCATION_USD
            or plan.get("freeze_hash") != _hash({key: value for key, value in plan.items() if key != "freeze_hash"})):
        raise ValidationError("Frozen semantic trial method, code or configuration changed")
    entries = [{key: value for key, value in entry.items() if key in _ENTRY_KEYS} for entry in plan["cases"]]
    if plan["cases"] != _entries(entries, root) or plan["max_requests"] != len(entries):
        raise ValidationError("Frozen semantic trial sources, crops, requests or labels changed")
    return folder, plan


def run_trial(run_dir: str | Path, client_factory: Callable[[], Any], *, root: str | Path) -> dict[str, Any]:
    """Attempt each case once with a fresh one-request client; never auto-resume."""
    root = Path(root).resolve()
    folder, plan = _verify(run_dir, root)
    if any(path.name != "freeze.json" for path in folder.iterdir()):
        raise ValidationError("A trial cannot be rerun or overwrite an existing attempt")
    _write(folder / "run-started.json", {"freeze_hash": plan["freeze_hash"],
                                       "started_at_utc": datetime.now(timezone.utc).isoformat()})
    spent = 0.0
    for entry in plan["cases"]:
        _verify(run_dir, root)  # Rehash source/crops/code immediately before every dispatch.
        if spent >= ALLOCATION_USD:
            break
        client = client_factory()
        if client.retries != 0 or client.max_requests != 1 or client.ledger:
            raise ValidationError("Trial factory must provide a fresh one-request zero-retry client")
        report = method.challenge_candidate_pair_v2(client, _read(_path(entry["prepared_file"], root)))
        _write(folder / f"{entry['case_id']}.json", report)  # Retain raw failure before any export.
        spent += method.public_challenge_summary_v2(report)["usage"]["estimated_usd"]
    return public_trial_summary(run_dir, root=root)


def public_trial_summary(run_dir: str | Path, *, root: str | Path) -> dict[str, Any]:
    """Recompute completion, identity and cost metrics exclusively from local evidence."""
    root = Path(root).resolve()
    folder, plan = _verify(run_dir, root)
    cases, summaries = [], []
    allowed_files = {"freeze.json", "run-started.json"} | {f"{entry['case_id']}.json" for entry in plan["cases"]}
    if any(path.name not in allowed_files for path in folder.iterdir()):
        raise ValidationError("Trial directory contains an unreviewed attempt")
    for entry in plan["cases"]:
        case = {key: deepcopy(entry[key]) for key in _ENTRY_KEYS - {"prepared_file", "prepared_hash"}}
        report_file = folder / f"{entry['case_id']}.json"
        summary = None
        if report_file.is_file():
            report = _read(report_file)
            if (report.get("prepared") != _read(_path(entry["prepared_file"], root))
                    or report.get("request_binding") != entry["request_binding"]):
                raise ValidationError("Retained semantic attempt differs from its frozen case")
            summary = method.public_challenge_summary_v2(report)
            summaries.append(summary)
        completed = summary is not None and summary["status"] == "completed"
        case.update(attempt_status=summary["status"] if summary else "not_attempted",
                    matches_expected=(summary["challenge"]["decision"] == entry["expected_identity"])
                        if completed else None, challenge=summary)
        cases.append(case)
    def ratio(numerator, denominator):
        return numerator / denominator if denominator else None
    completed = [case for case in cases if case["attempt_status"] == "completed"]
    positives = [case for case in cases if case["expected_identity"] == "supported"]
    negatives = [case for case in cases if case["expected_identity"] == "unknown"]
    positive_done = [case for case in positives if case["attempt_status"] == "completed"]
    negative_done = [case for case in negatives if case["attempt_status"] == "completed"]
    retained = sum(case["matches_expected"] is True for case in positive_done)
    unsupported = sum(case["matches_expected"] is False for case in negative_done)
    correct = sum(case["matches_expected"] is True for case in completed)
    metrics = {"case_count": len(cases), "attempted": len(summaries), "completed": len(completed),
               "unscored": sum(case["attempt_status"] == "unscored" for case in cases),
               "not_attempted": len(cases) - len(summaries),
               "completion_rate": ratio(len(completed), len(summaries)), "scored": len(completed),
               "correct": correct, "scored_accuracy": ratio(correct, len(completed)),
               "known_positive": {"count": len(positives), "completed": len(positive_done),
                                  "supported": retained, "retention_rate": ratio(retained, len(positive_done))},
               "known_negative": {"count": len(negatives), "completed": len(negative_done),
                                  "unsupported_accepted": unsupported,
                                  "unsupported_acceptance_rate": ratio(unsupported, len(negative_done))}}
    costs = {key: sum(summary["usage"][key] for summary in summaries)
             for key in ("request_attempts", "input_tokens", "output_tokens", "estimated_usd")}
    costs["estimated_usd"] = round(costs["estimated_usd"], 6)
    costs["estimated_allocation_usd"] = ALLOCATION_USD
    return {"schema_version": 1, "experiment": EXPERIMENT, "method_version": method.SEMANTIC_V2_VERSION,
            "phase": plan["phase"], "evaluation_role": "development_reprocessing",
            "freeze_hash": plan["freeze_hash"], "method_config": deepcopy(method.METHOD_CONFIG),
            "cases": cases, "metrics": metrics, "cost_summary": costs,
            "limitations": list(_LIMITS), "changes_guided_v2_acceptance": False}
