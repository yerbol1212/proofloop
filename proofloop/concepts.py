"""Bounded SAM prompt commissioning on two explicitly supplied development views.

This diagnostic compares candidate text prompts on the same two images. It
does not establish accuracy, independent trials, object identity, uniqueness,
absence, freshness, or a winning prompt. An operator must review the masks and
freeze the chosen object-to-prompt mapping in a separate task contract.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

from .core import ValidationError
from .grounding import _detection
from .meta import MetaAPIError, MetaClient, SAM_MODEL, _image_data, load_env_file
from .session import _bound_result

ROOT = Path(__file__).resolve().parents[1]
LOCAL_ROOT = ROOT / "artifacts/local"
MAX_CANDIDATES = 4
MAX_CONCEPT_CHARACTERS = 100


def _concept(value: Any) -> str:
    if (not isinstance(value, str) or not value.strip() or len(value) > MAX_CONCEPT_CHARACTERS
            or not value.isprintable()):
        raise ValidationError("SAM prompts must contain 1 to 100 printable characters")
    return value.strip()


def validate_concept_mapping(objects: Sequence[str], concepts: Any = None) -> dict[str, str]:
    """Validate an explicit reviewed mapping; never choose prompts from results."""
    if concepts is None:
        return {value: value.replace("_", " ").replace("-", " ") for value in objects}
    if not isinstance(concepts, Mapping) or set(concepts) != set(objects):
        raise ValidationError("SAM prompt mapping must identify every task object exactly once")
    return {value: _concept(concepts[value]) for value in objects}


def validate_candidates(candidates: Any) -> list[str]:
    if not isinstance(candidates, (list, tuple)) or not 1 <= len(candidates) <= MAX_CANDIDATES:
        raise ValidationError("Supply 1 to 4 explicit candidate SAM prompts")
    result = [_concept(value) for value in candidates]
    if len({value.casefold() for value in result}) != len(result):
        raise ValidationError("Candidate SAM prompts must be distinct")
    return result


def _destination(output: Path, local_root: Path) -> Path:
    output, local_root = Path(output).resolve(), Path(local_root).resolve()
    if local_root not in output.parents:
        raise ValidationError("Commissioning outputs must use a new directory inside artifacts/local")
    if output.exists():
        raise ValidationError("Commissioning output already exists; use a new run directory")
    return output


def _save(path: Path, value: Any) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def _images(images: Sequence[Path]) -> tuple[list[Path], list[dict[str, Any]]]:
    if not isinstance(images, (list, tuple)) or len(images) != 2:
        raise ValidationError("Commissioning requires exactly two source images")
    paths = [Path(value).resolve() for value in images]
    sources = [_image_data(path)[1] for path in paths]
    if sources[0]["sha256"] == sources[1]["sha256"]:
        raise ValidationError("Commissioning requires two distinct image hashes")
    return paths, sources


def run_commissioning(client: MetaClient, images: Sequence[Path], candidates: Sequence[str], output: Path, *,
                      input_origin: str, inference_origin: str = "caller_provided_client_not_verified_live",
                      local_root: Path = LOCAL_ROOT) -> dict[str, Any]:
    """Issue exactly two calls per candidate at most; stop on the first failure.

    No response cache is silently reused. Candidate order is caller supplied and
    no candidate is approved or ranked by this diagnostic. Source bytes are
    checked before each call and after the run.
    """
    candidates = validate_candidates(candidates)
    paths, sources = _images(images)
    if input_origin not in ("licensed_online", "owned_phone", "synthetic", "operator_declared_other"):
        raise ValidationError("Declare a supported commissioning input origin")
    budget = 2 * len(candidates)
    if client.max_requests != budget or client.retries != 0 or client.ledger:
        raise ValidationError("Commissioning requires a fresh exact two-per-candidate request budget and zero retries")
    destination = _destination(output, local_root)
    destination.mkdir(parents=True, mode=0o700)
    report: dict[str, Any] = {
        "schema_version": "1.0", "experiment": "sam_concept_commissioning", "commissioning_only": True,
        "inference_origin": inference_origin, "input_origin": input_origin,
        "input_origin_basis": "caller_declaration_not_independently_verified",
        "expected_model": SAM_MODEL, "sources": sources, "max_requests": budget,
        "attempted_requests": 0, "completed_requests": 0, "candidates": [], "failures": [],
        "operator_review_required": True, "selected_concept": None, "accuracy_established": False,
        "held_out_benchmark": False, "physical_repair_verified": False,
        "limits": ["The same two development views are reused across candidate prompts, not independent trials.",
                   "More detections or higher reported confidence do not establish correct identity or accuracy.",
                   "Zero detections do not establish absence; detections do not establish uniqueness or freshness.",
                   "Inspect returned masks and freeze any selected prompt in a separately reviewed task contract.",
                   "Changing a task's prompt mapping invalidates its previous evidence; no winner is automatically approved."],
    }
    _save(destination / "plan.json", {"candidates": candidates, "sources": sources,
        "input_origin": input_origin, "inference_origin": inference_origin, "max_requests": budget,
        "selection_policy": "operator_review_only_no_automatic_winner"})
    stop = False
    for candidate_index, candidate in enumerate(candidates):
        summary = {"concept": candidate, "views": [], "completed_views": 0}
        report["candidates"].append(summary)
        for image_index, path in enumerate(paths):
            try:
                if hashlib.sha256(path.read_bytes()).hexdigest() != sources[image_index]["sha256"]:
                    raise ValidationError("Commissioning source changed during the run")
                if report["attempted_requests"] >= budget:
                    raise ValidationError("Commissioning request budget exhausted")
                report["attempted_requests"] += 1
                result = client.segment_image(path, candidate, decode_masks=True)
                _bound_result(result, sources[image_index]["sha256"])
                objects = result.get("objects")
                if not isinstance(objects, list):
                    raise ValidationError("Commissioning result requires a detection list")
                detections = [_detection(value, {"image": (0.0, 0.0, 1.0, 1.0)}) for value in objects]
                # Preserve exact decoded masks privately for operator inspection;
                # the summary exposes neither response IDs nor mask payloads.
                _save(destination / f"candidate-{candidate_index}-view-{image_index}.json", result)
                summary["views"].append({"image_index": image_index,
                    "source_sha256": sources[image_index]["sha256"], "detection_count": len(detections),
                    "reported_confidences": [value["confidence"] for value in detections],
                    "normalized_boxes": [list(value["box"]) for value in detections],
                    "decoded_mask_count": len(detections), "all_returned_masks_validated": True})
                summary["completed_views"] += 1
                report["completed_requests"] += 1
            except (MetaAPIError, ValueError, OSError, TypeError, KeyError, AttributeError) as error:
                report["failures"].append({"candidate_index": candidate_index, "image_index": image_index,
                                           "error_type": type(error).__name__})
                if isinstance(error, MetaAPIError) and error.response is not None:
                    _save(destination / "failed-response.json", error.response)
                stop = True
                break
        if stop:
            break
    try:
        if any(hashlib.sha256(path.read_bytes()).hexdigest() != sources[index]["sha256"] for index, path in enumerate(paths)):
            report["failures"].append({"error_type": "SourceChangedAfterRun"})
    except OSError:
        report["failures"].append({"error_type": "SourceUnavailableAfterRun"})
    report["overall"] = "commissioning_complete_requires_operator_review" if not report["failures"] else "commissioning_incomplete"
    report["usage"] = client.usage_summary()
    _save(destination / "report.json", report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", nargs=2, type=Path, required=True)
    parser.add_argument("--candidate", action="append", required=True, help="Explicit short SAM prompt; repeat 1 to 4 times")
    parser.add_argument("--max-requests", type=int, required=True, help="Explicit exact cap: two times the number of candidates")
    parser.add_argument("--input-origin", required=True,
                        choices=("licensed_online", "owned_phone", "synthetic", "operator_declared_other"))
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env.local")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    parser.add_argument("--output", type=Path, default=LOCAL_ROOT / f"concepts-{stamp}")
    args = parser.parse_args(argv)
    try:
        candidates = validate_candidates(args.candidate)
        if args.max_requests != 2 * len(candidates):
            raise ValidationError("The explicit request budget must equal two times the number of candidates")
        _destination(args.output, LOCAL_ROOT)
        _images(args.images)
        load_env_file(args.env_file)
        client = MetaClient.from_env(max_requests=args.max_requests, retries=0, timeout=30)
        report = run_commissioning(client, args.images, candidates, args.output,
                                  input_origin=args.input_origin, inference_origin="live_meta_api")
    except (MetaAPIError, ValueError, OSError) as error:
        print(f"Concept commissioning stopped: {type(error).__name__}", file=sys.stderr)
        return 2
    print(json.dumps({key: report[key] for key in ("overall", "attempted_requests", "completed_requests", "operator_review_required")}, indent=2))
    print(f"Private commissioning report: {args.output / 'report.json'}")
    return 0 if not report["failures"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
