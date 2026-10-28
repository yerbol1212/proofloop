"""Local source binding for the fixed, manually anchored distractor review.

No network or model calls. Hashes establish consistency with an exposed author
review, not trusted capture timestamps, automatic localization or API origin.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tempfile
from typing import Any, Mapping, Sequence

from .capture import _jpeg_dimensions
from .core import ValidationError


REVIEW_DIRECTORY = "artifacts/local/semantic-distractor-review-20261028"
SOURCE_DIRECTORY = "inputs/local/semantic-distractors-20261028"
FROZEN_REVIEW_SHA256 = "cba501f6226700d744fa4044671439c79c12feb1272311aabb8909816654ec5e"
MAX_VIDEO_BYTES = 40 * 1024 * 1024
MAX_RECORD_BYTES = 2 * 1024 * 1024
_HASH = re.compile(r"^[0-9a-f]{64}$")


def _same(first: Any, second: Any) -> bool:
    return json.dumps(first, sort_keys=True, allow_nan=False) == json.dumps(
        second, sort_keys=True, allow_nan=False)


def _file(root: Path, value: str | Path, directory: str, limit: int) -> Path:
    if not isinstance(value, (str, Path)):
        raise ValidationError("Distractor evidence path is invalid")
    path = (root / value).resolve()
    allowed = (root / directory).resolve()
    try:
        allowed.relative_to(root)
        path.relative_to(allowed)
    except ValueError:
        raise ValidationError("Distractor evidence is outside its fixed directory") from None
    if not path.is_file() or not 0 < path.stat().st_size <= limit:
        raise ValidationError("Distractor evidence is missing or exceeds its size bound")
    return path


def _digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def _hashed_file(root: Path, value: str | Path, digest: str, directory: str, limit: int) -> Path:
    if not isinstance(digest, str) or not _HASH.fullmatch(digest):
        raise ValidationError("Distractor evidence hash is invalid")
    path = _file(root, value, directory, limit)
    if _digest(path) != digest:
        raise ValidationError("Distractor source or reviewed evidence changed")
    return path


def _read_json(path: Path) -> dict[str, Any]:
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValidationError("Duplicate distractor record field")
            result[key] = value
        return result

    def invalid(_value):
        raise ValidationError("Distractor records require finite values")

    try:
        data = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs, parse_constant=invalid)
    except (ValueError, UnicodeError):
        raise ValidationError("Distractor record is not valid JSON") from None
    if not isinstance(data, dict):
        raise ValidationError("Distractor record must be an object")
    return data


def _box(value: Any, width: int, height: int) -> dict[str, int]:
    if (not isinstance(value, Mapping) or set(value) != {"x1", "y1", "x2", "y2"}
            or any(type(item) is not int for item in value.values())
            or not 0 <= value["x1"] < value["x2"] < width
            or not 0 <= value["y1"] < value["y2"] < height):
        raise ValidationError("Distractor candidate box is outside reviewed source pixels")
    return dict(value)


def _reextract(root: Path, review: Mapping[str, Any], sources: Mapping[str, Mapping[str, Any]]) -> None:
    executable = root / ".venv/bin/ffmpeg"
    try:
        version = subprocess.run([str(executable), "-version"], capture_output=True, check=True,
                                 timeout=5).stdout.decode("utf-8", "replace").splitlines()[0]
        if version != review["extractor"]:
            raise ValidationError("Distractor source extraction requires the reviewed FFmpeg version")
        seen = set()
        with tempfile.TemporaryDirectory(prefix="proofloop-distractor-replay-") as temporary:
            for case in review["cases"]:
                source = sources[case["source_id"]]
                for frame in case["frames"]:
                    key = (case["source_id"], frame["source_timestamp_ms"])
                    if key in seen:
                        continue
                    seen.add(key)
                    target = Path(temporary) / f"frame-{len(seen)}.jpg"
                    args = [str(executable), "-hide_banner", "-loglevel", "error", "-nostdin", "-ss",
                            str(frame["source_timestamp_ms"] / 1000), "-i", source["source_path"],
                            "-frames:v", "1", "-q:v", "2", "-pix_fmt", "yuvj444p", "-threads", "1",
                            "-map_metadata", "-1", str(target)]
                    subprocess.run(args, capture_output=True, check=True, timeout=20)
                    if _digest(target) != frame["sha256"]:
                        raise ValidationError("Distractor frame does not replay from its pinned video")
    except (OSError, subprocess.SubprocessError, IndexError):
        raise ValidationError("Distractor video-frame replay failed") from None


def verify_source_review(root: str | Path, *, reextract: bool = False) -> dict[str, Any]:
    """Verify fixed review, selected video/cards and exact reviewed source bytes.

    Set reextract=True before first inference to also replay four video decodes.
    The default performs no subprocesses and is suitable for read-only lab load.
    """
    root = Path(root).resolve()
    record = _file(root, f"{REVIEW_DIRECTORY}/frozen-review-manifest.json", REVIEW_DIRECTORY, MAX_RECORD_BYTES)
    sidecar = _file(root, f"{REVIEW_DIRECTORY}/frozen-review-manifest.sha256", REVIEW_DIRECTORY, 100)
    if sidecar.read_text(encoding="ascii").strip() != FROZEN_REVIEW_SHA256 or _digest(record) != FROZEN_REVIEW_SHA256:
        raise ValidationError("Distractor review differs from the frozen author review")
    review = _read_json(record)
    if (review.get("schema_version") != 1 or review.get("inference_started") is not False
            or review.get("evaluation_role") != "authored_anchor_stress_test"
            or review.get("case_count") != 6 or review.get("unique_source_video_count") != 2
            or not isinstance(review.get("sources"), list) or len(review["sources"]) != 5
            or not isinstance(review.get("cases"), list) or len(review["cases"]) != 6):
        raise ValidationError("Distractor review scope changed")
    selected = {}
    for source in review["sources"]:
        if source.get("selected_for_inference") is not True:
            continue
        identifier = source.get("source_id")
        if not isinstance(identifier, str) or identifier in selected:
            raise ValidationError("Distractor source IDs must be unique")
        video = _hashed_file(root, source["source_path"], source["sha256"], SOURCE_DIRECTORY, MAX_VIDEO_BYTES)
        if type(source.get("bytes")) is not int or video.stat().st_size != source["bytes"]:
            raise ValidationError("Distractor source byte count changed")
        card = _hashed_file(root, source["dataset_card_path"], source["dataset_card_sha256"],
                            SOURCE_DIRECTORY, MAX_RECORD_BYTES)
        info = _hashed_file(root, source["info_path"], source["info_sha256"], SOURCE_DIRECTORY, MAX_RECORD_BYTES)
        if (source.get("license") not in {"mit", "apache-2.0"}
                or not re.search(r"^license: " + re.escape(source["license"]) + r"$", card.read_text(), re.MULTILINE)):
            raise ValidationError("Distractor dataset card license changed")
        metadata = _read_json(info)
        shape = metadata.get("features", {}).get("observation.images.cam_high", {}).get("shape")
        if shape != [source.get("height"), source.get("width"), 3] or shape != [480, 640, 3]:
            raise ValidationError("Distractor source geometry differs from pinned dataset metadata")
        selected[identifier] = source
    if len(selected) != 2:
        raise ValidationError("Distractor review requires exactly two selected recordings")
    seen = set()
    for case in review["cases"]:
        if (case.get("case_id") in seen or case.get("source_id") not in selected
                or case.get("anchor_origin") != "manually_authored_after_visual_review"
                or case.get("automatic_localization") is not False
                or case.get("label_origin") != "author_visual_review_before_any_semantic_inference"
                or case.get("expected_identity") not in {"target", "excluded"}
                or case.get("expected_diagnostic_decision") != (
                    "supported" if case.get("expected_identity") == "target" else "unknown")):
            raise ValidationError("Distractor case provenance or expected identity changed")
        seen.add(case["case_id"])
        frames = case.get("frames")
        if (not isinstance(frames, list) or len(frames) != 2
                or [frame.get("frame_index") for frame in frames] != [0, 1]
                or [frame.get("source_timestamp_ms") for frame in frames] != [0, 1000]):
            raise ValidationError("Distractor pair must retain the two reviewed source times")
        for frame in frames:
            image = _hashed_file(root, frame["image_path"], frame["sha256"], REVIEW_DIRECTORY, MAX_RECORD_BYTES)
            if (frame.get("width"), frame.get("height")) != (640, 480) or _jpeg_dimensions(image.read_bytes()) != (640, 480):
                raise ValidationError("Distractor source frame geometry changed")
            if frame.get("expected_candidate_relation") != case["expected_identity"]:
                raise ValidationError("Distractor frame identity differs from its reviewed pair")
            box = _box(frame["candidate_box"], 640, 480)
            crop = _hashed_file(root, frame["candidate_review_crop_path"], frame["candidate_review_crop_sha256"],
                                REVIEW_DIRECTORY, MAX_RECORD_BYTES)
            if _jpeg_dimensions(crop.read_bytes()) != (box["x2"] - box["x1"] + 1, box["y2"] - box["y1"] + 1):
                raise ValidationError("Distractor review crop differs from the authored rectangle")
        if frames[0]["sha256"] == frames[1]["sha256"]:
            raise ValidationError("Distractor review requires distinct source frame bytes")
    if reextract:
        _reextract(root, review, selected)
    return deepcopy(review)


def source_attribution(review: Mapping[str, Any], source_id: str) -> dict[str, Any]:
    """Return only public dataset attribution; never local paths or raw records."""
    matches = [source for source in review.get("sources", [])
               if source.get("source_id") == source_id and source.get("selected_for_inference") is True]
    if len(matches) != 1:
        raise ValidationError("Distractor attribution requires one reviewed source")
    fields = ("source_id", "dataset", "revision", "remote_path", "url", "license", "sha256", "bytes",
              "width", "height", "duration_seconds", "dataset_card_url", "dataset_card_sha256", "info_sha256")
    return {key: deepcopy(matches[0][key]) for key in fields}


def validate_prepared_entries(root: str | Path, entries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Bind runner entries to the fixed review without accepting new labels.

    Each entry supplies case_id, source_id, expected_identity (the diagnostic
    decision: supported/unknown), anchor_origin, source_timestamps_ms and
    prepared_file.
    The prepared contract may add method/version/budget fields but must retain
    every reviewed concept/attribute/exclusion. Crops are verified separately by
    the semantic adapter because its context rule differs from review QA crops.
    """
    root = Path(root).resolve()
    review = verify_source_review(root)
    cases = {case["case_id"]: case for case in review["cases"]}
    if (not isinstance(entries, (list, tuple)) or len(entries) != len(cases)
            or [entry.get("case_id") for entry in entries] != list(cases)):
        raise ValidationError("Distractor run must preserve the six ordered reviewed cases")
    for entry in entries:
        case = cases[entry["case_id"]]
        if (entry.get("source_id") != case["source_id"]
                or entry.get("expected_identity") != case["expected_diagnostic_decision"]
                or entry.get("anchor_origin") != "authored_candidate"
                or not _same(entry.get("source_timestamps_ms"),
                             [frame["source_timestamp_ms"] for frame in case["frames"]])):
            raise ValidationError("Distractor run source or identity label differs from the author review")
        path = _file(root, entry.get("prepared_file"), "artifacts/local", MAX_RECORD_BYTES)
        prepared = _read_json(path)
        contract = prepared.get("contract", {})
        if any(not _same(contract.get(key), value) for key, value in case["contract"].items()):
            raise ValidationError("Distractor semantic contract differs from its reviewed visual requirements")
        frames = prepared.get("frames")
        if not isinstance(frames, list) or len(frames) != 2:
            raise ValidationError("Distractor preparation needs the exact reviewed frame pair")
        for actual, expected in zip(frames, case["frames"]):
            for key in ("frame_index", "sha256", "width", "height", "candidate_box"):
                if not _same(actual.get(key), expected[key]):
                    raise ValidationError("Distractor preparation changed the reviewed source anchor")
            source = _hashed_file(root, actual.get("source_file"), expected["sha256"], REVIEW_DIRECTORY, MAX_RECORD_BYTES)
            if source != Path(expected["image_path"]).resolve():
                raise ValidationError("Distractor preparation uses an unreviewed source path")
    return review
