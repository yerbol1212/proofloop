"""Live access gate, deliberately separate from the offline checker tests."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

from .meta import MetaAPIError, MetaClient, SAM_MODEL, SPARK_MODEL, load_env_file


SMOKE_SCHEMA = {
    "type": "object", "properties": {
        "project": {"type": "string", "enum": ["ProofLoop"]},
        "requires_fresh_evidence": {"type": "boolean"},
        "result_when_occluded": {"type": "string", "enum": ["unknown"]},
    }, "required": ["project", "requires_fresh_evidence", "result_when_occluded"],
    "additionalProperties": False,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify real Meta model access; never substitutes cached inference")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--models-only", action="store_true", help="Free access check without inference")
    group.add_argument("--spark-only", action="store_true", help="List models and run one tiny structured Spark call")
    parser.add_argument("--image", type=Path, help="Owned local image to send to SAM; remains ignored/local")
    parser.add_argument("--concept", default="screwdriver", help="One short concrete noun phrase")
    parser.add_argument("--env-file", type=Path, default=Path(".env.local"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/local"))
    args = parser.parse_args(argv)
    if not args.models_only and not args.spark_only and args.image is None:
        parser.error("Supply --image PATH for the full gate, or select --models-only / --spark-only")
    # Refuse alternate output destinations: sanitized traces may still contain
    # owned model output, which is private until reviewed for publication.
    root = Path(__file__).resolve().parents[1]
    destination = args.output.resolve()
    local_root = root / "artifacts" / "local"
    if destination != local_root and local_root not in destination.parents:
        parser.error("Output must remain inside this project's artifacts/local directory")
    load_env_file(args.env_file)
    try:
        client = MetaClient.from_env(max_requests=4, retries=0)
    except (MetaAPIError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 2
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    run_dir = destination / f"smoke-{run_id}"
    run_dir.mkdir(parents=True, exist_ok=False)
    report = {"run_id": run_id, "mode": "live_api", "models": {}, "gates": {},
              "source_note": "A synthetic image tests transport only; physical feasibility requires independently recorded owned scenes."}
    status = 0
    try:
        models = client.list_models()
        report["models"] = {"spark_listed": SPARK_MODEL in models, "sam_listed": SAM_MODEL in models,
                            "model_count": len(models)}
        report["gates"]["model_list"] = "pass"
        if not args.models_only:
            spark = client.spark_json(
                "Project: ProofLoop. Return requires_fresh_evidence=true. "
                "If a critical area is occluded, result_when_occluded must be unknown.",
                SMOKE_SCHEMA, name="proofloop_smoke", max_output_tokens=512,
            )
            if spark["parsed"]["requires_fresh_evidence"] is not True:
                raise MetaAPIError("Spark smoke semantics failed", code="smoke_semantics_failed")
            (run_dir / "spark.json").write_text(json.dumps(spark, indent=2), encoding="utf-8")
            report["gates"]["spark_structured_output"] = "pass"
            report["spark"] = {"model": SPARK_MODEL, "latency_seconds": spark["latency_seconds"],
                               "usage": spark["usage"], "parsed": spark["parsed"]}
        if args.image is not None and not args.models_only and not args.spark_only:
            sam = client.segment_image(args.image, args.concept)
            (run_dir / "sam.json").write_text(json.dumps(sam, indent=2), encoding="utf-8")
            report["gates"]["sam_transport_and_parse"] = "pass"
            report["gates"]["sam_nonempty_grounding"] = "pass" if sam["objects"] else "unproven_zero_detections"
            report["gates"]["sam_official_mask_decode"] = "pass" if sam["objects"] and all(
                obj["mask"]["decoded"] and obj["mask"]["foreground_pixels"] > 0 for obj in sam["objects"]
            ) else "unproven_no_foreground_masks"
            if report["gates"]["sam_official_mask_decode"] != "pass":
                status = 1
            report["sam"] = {"model": SAM_MODEL, "latency_seconds": sam["latency_seconds"],
                             "objects": len(sam["objects"]), "source": sam["source"]}
    except (MetaAPIError, ValueError, OSError) as error:
        # OSError file paths can include private identifiers; only expose class.
        message = str(error) if isinstance(error, MetaAPIError) else type(error).__name__
        report["error"] = {"type": type(error).__name__, "message": message}
        if isinstance(error, MetaAPIError) and error.response is not None:
            (run_dir / "failed_response.json").write_text(json.dumps(error.response, indent=2), encoding="utf-8")
        report["gates"]["remaining_live_checks"] = "blocked"
        status = 1
    finally:
        report["usage"] = client.usage_summary()
        report["overall"] = "requested_gates_passed" if not status else "incomplete"
        (run_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Local trace: {run_dir}")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
