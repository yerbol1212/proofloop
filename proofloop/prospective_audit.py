"""Offline consistency audit of retained Spark text in a frozen pilot.

This supplements the frozen study without changing its code or results. It
replays existing reports, then requires completed, source-bound provider text
to match every direct Spark result and guided locator's parsed JSON. A receipt
contains counts and hashes only; it does not authenticate a provider or label.
No credential loading, network transport or inference is performed.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping
import hashlib
import json
import math
from pathlib import Path
import sys

from . import benchmark, guided, prospective
from .core import ValidationError
from .meta import MetaAPIError, SPARK_MODEL, output_text, validate_json

VERSION = "prospective_raw_text_audit_v1"
MAX_RECORDS = 512
MAX_TOTAL_BYTES = 256 * 1024 * 1024


def _json(text):
    def unique(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate")
            result[key] = value
        return result

    def nonfinite(_value):
        raise ValueError("nonfinite")

    def finite_float(value):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("nonfinite exponent")
        return number

    try:
        return json.loads(text, object_pairs_hook=unique, parse_constant=nonfinite,
                          parse_float=finite_float)
    except (ValueError, TypeError, UnicodeError, RecursionError):
        raise ValidationError("Audit evidence requires finite JSON without duplicate fields") from None


def _read(path):
    if path.is_symlink() or not path.is_file() or not 0 < path.stat().st_size <= benchmark.MAX_JSON_BYTES:
        raise ValidationError("Audit evidence requires bounded regular JSON files")
    try:
        return _json(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError):
        raise ValidationError("Audit evidence could not be read") from None


def _canonical(value):
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (ValueError, TypeError, RecursionError):
        raise ValidationError("Audit evidence is not finite JSON") from None


def _bound_text(wrapper, source_hashes, schema):
    if (not isinstance(wrapper, Mapping) or type(source_hashes) is not list or len(source_hashes) != 2
            or len(set(source_hashes)) != 2 or any(not benchmark._SHA.fullmatch(item) for item in source_hashes)):
        raise ValidationError("Audit requires two distinct frozen source images")
    response = wrapper.get("response")
    sources = wrapper.get("sources")
    if (not isinstance(response, dict) or response.get("status") != "completed"
            or response.get("model") != SPARK_MODEL or wrapper.get("source") is not None
            or type(sources) is not list
            or [item.get("sha256") if isinstance(item, Mapping) else None for item in sources] != source_hashes):
        raise ValidationError("Audit requires completed Spark evidence bound to the exact ordered source pair")
    try:
        text = output_text(response)
        if type(text) is not str or not text.strip() or len(text.encode("utf-8")) > benchmark.MAX_JSON_BYTES:
            raise ValueError("empty or excessive")
        # A convenience text field cannot conceal contradictory message lanes.
        if "output_text" in response:
            messages = output_text({key: value for key, value in response.items() if key != "output_text"})
            if messages and messages != text:
                raise ValueError("conflicting text lanes")
        parsed = _json(text)
        validate_json(parsed, schema)
        if (not isinstance(parsed, dict) or _canonical(parsed) != _canonical(wrapper.get("parsed"))
                or "text" in wrapper and wrapper["text"] != text):
            raise ValueError("parsed or wrapper text mismatch")
    except (MetaAPIError, ValueError, TypeError, KeyError, AttributeError):
        raise ValidationError("Retained Spark text does not match its validated parsed result") from None
    return {"provider_response_sha256": benchmark.manifest_hash(response),
            "provider_text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "parsed_sha256": benchmark.manifest_hash(parsed),
            "source_pair_sha256": benchmark.manifest_hash(source_hashes)}


def audit(out_dir: Path, *, root: Path = benchmark.ROOT):
    """Return an allowlisted receipt only for a fully completed, rebound study."""
    output = prospective._directory(Path(out_dir), Path(root))
    files = sorted(output.rglob("*.json"))
    if (not files or len(files) > MAX_RECORDS
            or sum(path.stat().st_size for path in files) > MAX_TOTAL_BYTES):
        raise ValidationError("Audit evidence inventory is missing or excessive")
    records = {path: _read(path) for path in files}
    output, frozen, singles = prospective._load(out_dir, root)
    summary = prospective.summarize(out_dir, root=root)
    if (summary["status"] != "completed" or not frozen["conditions"]
            or set(frozen["conditions"]) - {"spark", "guided"}):
        raise ValidationError("Audit requires completed direct Spark and/or guided pilot parts")
    fingerprints = []
    counts = {"spark": 0, "guided": 0}
    source_images = set()
    for case in frozen["envelope"]["contract"]["cases"]:
        identifier = case["id"]
        expected = [frame["sha256"] for frame in case["frames"]]
        source_images.update(expected)
        envelope = singles[identifier]
        protocol = benchmark._approved(envelope)
        for condition in frozen["conditions"]:
            row = next((row for row in summary["condition_results"][condition]["cases"]
                        if row["case_id"] == identifier), None)
            if (row is None or row["status"] != "completed" or row["missing_prediction"]
                    or row.get("inference_sidecar_replayed") is not True):
                raise ValidationError("Every audit part requires a completed replayed inference sidecar")
            destination = output / "parts" / identifier / condition
            raw_path = destination / condition / "results.json"
            raw = records.get(raw_path)
            if not isinstance(raw, Mapping) or set(raw) != {identifier}:
                raise ValidationError("Audit sidecar must contain exactly its frozen case")
            if condition == "spark":
                wrapper = raw[identifier]
                schema = benchmark.baseline_schema(protocol)
            else:
                pair = records.get(destination / "guided" / (identifier + ".json"))
                paths = [Path(root) / frame["file"] for frame in case["frames"]]
                _, frames = guided._frames(paths)
                if not isinstance(pair, Mapping) or pair.get("frames") != frames:
                    raise ValidationError("Guided audit requires its exact frozen source geometry")
                wrapper = pair.get("locator")
                guided._locator_bound(wrapper, frames, envelope["contract"]["concepts"])
                schema = guided.locator_schema(list(protocol.objects))
            fingerprint = _bound_text(wrapper, expected, schema)
            fingerprint.update(part_sha256=benchmark.manifest_hash([identifier, condition]),
                               raw_sidecar_sha256=benchmark._file_hash(raw_path))
            fingerprints.append(fingerprint)
            counts[condition] += 1
    if not fingerprints:
        raise ValidationError("Audit requires at least one completed source-bound Spark response")
    receipt = {"schema_version": VERSION, "scope": "retained_record_consistency_only", "status": "passed",
               "new_model_requests": 0, "freeze_sha256": frozen["freeze_sha256"],
               "summary_sha256": benchmark.manifest_hash(summary),
               "auditor_sha256": benchmark._file_hash(Path(__file__)),
               "case_count": len(frozen["cases"]),
               "source_recordings_count": len(frozen["envelope"]["contract"]["sources"]),
               "source_images_count": len(source_images), "completed_parts_checked": len(fingerprints),
               "direct_spark_texts_checked": counts["spark"], "guided_locator_texts_checked": counts["guided"],
               "provider_text_parsed_matches": len(fingerprints),
               "source_pair_bindings_checked": len(fingerprints),
               "strict_json_records_checked": len(records),
               "records_sha256": benchmark.manifest_hash(fingerprints)}
    receipt["receipt_sha256"] = benchmark.manifest_hash(receipt)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="Existing private prospective pilot folder")
    args = parser.parse_args(argv)
    try:
        print(json.dumps(audit(args.output), sort_keys=True, indent=2))
        return 0
    except (ValidationError, OSError, ValueError, TypeError, KeyError, AttributeError):
        print("Prospective raw-text audit failed; inspect the frozen private evidence", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
