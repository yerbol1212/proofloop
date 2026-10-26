"""Bounded SAM integration experiment on explicitly generated tabletop images.

This is not a camera benchmark.  A successful run can establish positive
visibility on two distinct generated image hashes, plus conservative behavior
for a single wrong-region or occluded image.  No repair is physically executed.
Raw provider responses and the report stay inside ignored artifacts/local.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping

from .core import EvidenceStore, Protocol, evaluate
from .grounding import SamImageEvidence, ground_sam_images
from .meta import MetaAPIError, MetaClient, SAM_MODEL, load_env_file


SCENES = ("correct.png", "correct-variant.png", "wrong-region.png", "occluded.png")
CONCEPTS = ("screwdriver", "pliers")
MAX_SAM_CALLS = 8
MAX_REQUESTS = 12
MAX_CACHE_BYTES = 96 * 1024 * 1024
ROOT = Path(__file__).resolve().parents[1]
LOCAL_ROOT = ROOT / "artifacts" / "local"


def _protocol() -> Protocol:
    return Protocol.from_dict({
        "id": "synthetic-kit-visibility", "version": "1", "objects": list(CONCEPTS),
        "regions": ["left", "right"], "checks": [
            {"id": "screwdriver-left", "kind": "visible", "object_id": "screwdriver",
             "region_id": "left", "expected": True, "min_stable_samples": 2},
            {"id": "pliers-right", "kind": "visible", "object_id": "pliers",
             "region_id": "right", "expected": True, "min_stable_samples": 2},
        ],
    })


def _entries(manifest: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    images = manifest.get("images")
    if not isinstance(images, list) or len(images) != len(SCENES):
        raise ValueError("Expected exactly four generated images in the manifest")
    entries = {item.get("file"): item for item in images if isinstance(item, Mapping)}
    if set(entries) != set(SCENES):
        raise ValueError("Manifest must identify each expected generated scene exactly once")
    if manifest.get("physical_measurement") is not False:
        raise ValueError("This experiment requires explicitly nonphysical generated inputs")
    for item in entries.values():
        digest = item.get("sha256")
        if (not isinstance(digest, str) or len(digest) != 64
                or any(char not in "0123456789abcdef" for char in digest)):
            raise ValueError("Manifest source digest is malformed")
        if item.get("evidence_origin") != "imagegen_synthetic":
            raise ValueError("Generated image provenance is required")
    return entries


def _result_checks(results: Mapping[str, Any], entries: Mapping[str, Mapping[str, Any]]) -> None:
    if set(results) - set(SCENES):
        raise ValueError("Unexpected result scene")
    for scene, by_concept in results.items():
        if not isinstance(by_concept, Mapping) or set(by_concept) - set(CONCEPTS):
            raise ValueError("Unexpected result concept")
        for result in by_concept.values():
            if not isinstance(result, Mapping):
                raise ValueError("Result must be a segment_image result")
            source = result.get("source")
            if not isinstance(source, Mapping) or source.get("sha256") != entries[scene]["sha256"]:
                raise ValueError("SAM result does not identify the exact manifest source")


def _evaluate(protocol: Protocol, manifest: Mapping[str, Any], entries: Mapping[str, Mapping[str, Any]],
              results: Mapping[str, Any], scene_names: tuple[str, ...], session: str) -> dict[str, Any]:
    samples = []
    for index, scene in enumerate(scene_names):
        for concept, result in results.get(scene, {}).items():
            samples.append(SamImageEvidence(concept, index * 1000, result,
                                           entries[scene]["evidence_origin"], concept))
    grounded = ground_sam_images(protocol, manifest["regions"], samples, session_id=session)
    store = EvidenceStore(protocol, session)
    for observation in grounded.observations:
        store.add(observation)
    check_results = {item.check_id: item.to_dict() for item in evaluate(protocol, store)}
    observations = {item.object_id: {"value": item.value, "stable_samples": item.stable_samples,
                                   "fully_visible": item.fully_visible,
                                   "absence_validated": item.absence_validated}
                    for item in grounded.observations}
    source_summaries = []
    for source in grounded.evidence["sources"]:
        source_summaries.append({
            "object_id": source["object_id"], "source_sha256": source["source_sha256"],
            "input_origin": source["evidence_origin"], "reported_model": source["reported_model"],
            "reported_detection_count": len(results[scene_names[source["timestamp_ms"] // 1000]]
                                            [source["object_id"]].get("objects", [])),
            "supported_detection_count": len(source["detections"]),
            "regions": [item["region_id"] for item in source["detections"]],
            "reported_confidences": [item["confidence"] for item in source["detections"]],
            "mask_foreground_regions": [dict(region_id=item["region_id"],
                                              normalized_foreground=list(item["foreground"]))
                                        for item in source["detections"]],
            "unknown_reason": source["unknown_reason"],
        })
    return {"checks": check_results, "observations": observations, "sources": source_summaries}


def evaluate_scene_results(manifest: Mapping[str, Any], results: Mapping[str, Any], *,
                           inference_origin: str = "provided_results_not_verified_live") -> dict[str, Any]:
    """Evaluate actual adapter outputs without inferring live status or absence.

    Missing calls produce unknown checks. Caller-declared inference provenance
    is preserved; passing mock results does not establish model performance.
    No mask payload, provider response text, or response ID enters this summary.
    """
    entries = _entries(manifest)
    _result_checks(results, entries)
    protocol = _protocol()
    scenes = {scene: _evaluate(protocol, manifest, entries, results, (scene,), scene)
              for scene in SCENES}
    positive = _evaluate(protocol, manifest, entries, results,
                         ("correct.png", "correct-variant.png"), "two-generated-correct-views")
    wrong_sources = scenes["wrong-region.png"]["sources"]
    wrong_right = any(source["object_id"] == "screwdriver" and source["regions"] == ["right"]
                      and all(confidence >= 0.8 for confidence in source["reported_confidences"])
                      for source in wrong_sources)
    occluded = scenes["occluded.png"]
    occluded_unknown = (occluded["checks"]["screwdriver-left"]["status"] == "unknown"
                        and occluded["observations"]["screwdriver"]["value"] is None
                        and not occluded["observations"]["screwdriver"]["absence_validated"])
    completed = sum(len(by_concept) for by_concept in results.values())
    gates = {
        "all_eight_sam_results_available": completed == MAX_SAM_CALLS,
        "positive_visibility_two_distinct_generated_sources": all(
            item["status"] == "pass" for item in positive["checks"].values()),
        "wrong_region_positive_screwdriver_right_supported": wrong_right,
        "wrong_region_left_absence_not_claimed": (
            scenes["wrong-region.png"]["checks"]["screwdriver-left"]["status"] == "unknown"
            and not scenes["wrong-region.png"]["observations"]["screwdriver"]["absence_validated"]),
        "occluded_screwdriver_remains_unknown": occluded_unknown,
    }
    return {
        "schema_version": "1.0", "experiment": "generated_scene_sam_integration",
        "inference_origin": inference_origin, "physical_measurement": False,
        "physical_repair_executed": False, "held_out_benchmark": False,
        "expected_model": SAM_MODEL, "completed_sam_results": completed,
        "protocol": protocol.to_dict(), "protocol_hash": protocol.content_hash,
        "gates": gates, "scenes": scenes, "positive_generated_pair": positive,
        "overall": "synthetic_integration_gates_passed" if all(gates.values()) else "incomplete",
        "limits": [
            "Generated shared-reference edits are not independent physical trials.",
            "Wrong-region and occluded images are single observations; stability stays unproven.",
            "Zero detections never establish physical absence or complete visibility.",
            "Reported model confidence is not a calibrated correctness probability.",
            "Rectangles and concept-to-object mappings are manually specified for these generated views.",
            "No physical repair, phone-camera performance, or industrial reliability is established.",
        ],
    }


def _local_destination(output: Path) -> Path:
    destination = output.resolve()
    local_root = LOCAL_ROOT.resolve()
    if destination == local_root or local_root not in destination.parents:
        raise ValueError("Use a run directory strictly inside this project's artifacts/local")
    return destination


def _save(path: Path, value: Any) -> None:
    """Atomic private output; a partially written cache is never reused."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
    temporary.replace(path)


def run_scenes(client: MetaClient, manifest_path: Path, output: Path, *, resume: bool = False,
               inference_origin: str = "caller_provided_client_not_verified_live") -> dict[str, Any]:
    """Make at most eight new segment calls; never retry an ambiguous request.

    The CLI supplies a twelve-attempt client cap. Explicit resume only accepts
    source-, concept-, model-, and provenance-matching completed response caches.
    A resumed report labels reused responses; it never calls them fresh requests.
    """
    destination = _local_destination(output)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = _entries(manifest)
    for scene, entry in entries.items():
        image = manifest_path.parent / scene
        if hashlib.sha256(image.read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError("Generated source image differs from its manifest hash")
    if destination.exists() and not resume:
        raise ValueError("Run directory exists; choose a new output or explicitly --resume")
    destination.mkdir(parents=True, exist_ok=True, mode=0o700)
    if getattr(client, "retries", 0) != 0:
        raise ValueError("Scene experiments require zero client retries")
    if getattr(client, "max_requests", MAX_REQUESTS) > MAX_REQUESTS:
        raise ValueError("Client request cap must be at most twelve")
    results: dict[str, Any] = {}
    reused, attempted, failures = [], 0, []
    manifest_hash = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    for scene in SCENES:
        results[scene] = {}
        for concept in CONCEPTS:
            cache_path = destination / f"{Path(scene).stem}-{concept}.json"
            cache_metadata = {"schema_version": "1.0", "scene": scene, "concept": concept,
                              "source_sha256": entries[scene]["sha256"], "model": SAM_MODEL,
                              "manifest_sha256": manifest_hash, "inference_origin": inference_origin}
            if resume and cache_path.exists():
                if cache_path.stat().st_size > MAX_CACHE_BYTES:
                    raise ValueError("Cached response exceeds the private cache size limit")
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
                if cached.get("metadata") != cache_metadata:
                    raise ValueError("Cache metadata differs; refusing paid re-request or invalid reuse")
                result = cached.get("result")
                _result_checks({scene: {concept: result}}, entries)
                if (result.get("response", {}).get("status") != "completed"
                        or result.get("response", {}).get("model") != SAM_MODEL
                        or result.get("mask_decoding") != "official_meta_sam_parser_0.1.0"):
                    raise ValueError("Only completed official-decoded SAM responses can be resumed")
                results[scene][concept] = result
                reused.append({"scene": scene, "concept": concept})
                continue
            if attempted >= MAX_SAM_CALLS:
                raise ValueError("Eight-image SAM inference limit reached")
            attempted += 1
            try:
                result = client.segment_image(manifest_path.parent / scene, concept, decode_masks=True)
                _result_checks({scene: {concept: result}}, entries)
                _save(cache_path, {"metadata": cache_metadata, "result": result})
                results[scene][concept] = result
            except (MetaAPIError, ValueError, OSError) as error:
                # Do not expose provider text, private paths, or exception bodies.
                failures.append({"scene": scene, "concept": concept, "error_type": type(error).__name__})
                if isinstance(error, MetaAPIError) and error.response is not None:
                    _save(destination / f"{Path(scene).stem}-{concept}-failed-response.json", error.response)
                # Stop rather than multiply charges after an API/decode/file failure.
                break
        if failures:
            break
    report = evaluate_scene_results(manifest, results, inference_origin=inference_origin)
    report.update(mode="explicit_resume" if resume else "new_requests", new_sam_attempts=attempted,
                  reused_completed_responses=reused, failures=failures,
                  usage_this_invocation=client.usage_summary())
    _save(destination / "report.json", report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "inputs/synthetic/manifest.json")
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env.local")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    parser.add_argument("--output", type=Path, default=LOCAL_ROOT / f"live-scenes-{stamp}")
    parser.add_argument("--resume", action="store_true", help="Explicitly reuse completed matching responses")
    args = parser.parse_args(argv)
    try:
        _local_destination(args.output)
        load_env_file(args.env_file)
        client = MetaClient.from_env(max_requests=MAX_REQUESTS, retries=0)
        report = run_scenes(client, args.manifest, args.output, resume=args.resume,
                            inference_origin="live_meta_api")
    except (MetaAPIError, ValueError, OSError) as error:
        print(f"Scene experiment stopped: {type(error).__name__}", file=sys.stderr)
        return 2
    print(json.dumps({"overall": report["overall"], "gates": report["gates"],
                      "new_sam_attempts": report["new_sam_attempts"],
                      "reused_completed_responses": len(report["reused_completed_responses"]),
                      "failures": report["failures"]}, indent=2))
    print(f"Private report: {args.output / 'report.json'}")
    return 0 if report["overall"] == "synthetic_integration_gates_passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
