"""Private phone-video preflight and timestamped frame extraction.

This module makes no inference calls. Stage labels, visibility, and ownership
are declared assumptions with explicit reviewer provenance; an unedited video
is evidence, not automatic truth.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
PHONE_ROOT = ROOT / "inputs" / "local" / "phone"
MAX_FRAME_BYTES = 5 * 1024 * 1024
MAX_VIDEO_SECONDS = 600
MAX_STAGES = 32
MIN_SAMPLE_GAP_SECONDS = 1.0
TIMESTAMP_BASIS = "decoded video PTS normalized to first video frame; not wall-clock time"

DEFAULT_PLAN = {
    "schema_version": "1.0",
    "evidence_origin": "owned_phone",
    "origin_verification": "caller_declared_not_independently_verified",
    "ground_truth_origin": "human_declared_not_model_inferred",
    "offsets_confirmed": False,
    "instructions": "Watch the original clip and adjust every stage boundary and sample offset before confirming.",
    "stages": [
        {"id": "initial_correct", "declared_label": "fork left; spoon right; hands out; fully uncovered",
         "start_seconds": 0, "end_seconds": 8, "sample_seconds": [3, 5]},
        {"id": "wrong_region", "declared_label": "fork right; spoon right; hands out; fully uncovered",
         "start_seconds": 13, "end_seconds": 21, "sample_seconds": [16, 18]},
        {"id": "occluded", "declared_label": "whole tray covered by opaque board; contents not visible",
         "start_seconds": 26, "end_seconds": 34, "sample_seconds": [29, 31]},
        {"id": "corrected", "declared_label": "fresh correction: fork left; spoon right; hands out; uncovered",
         "start_seconds": 39, "end_seconds": 47, "sample_seconds": [42, 44]},
    ],
}


class CaptureError(ValueError):
    """An unusable capture, plan, or local decoder; no model calls were made."""


def default_plan() -> dict[str, Any]:
    return deepcopy(DEFAULT_PLAN)


def _inside_phone(path: Path, *, label: str) -> Path:
    resolved = path.resolve()
    phone_root = PHONE_ROOT.resolve()
    if resolved == phone_root or phone_root not in resolved.parents:
        raise CaptureError(f"Keep {label} strictly inside inputs/local/phone/ (ignored by Git)")
    return resolved


def _tool(name: str) -> str:
    location = shutil.which(name)
    if not location:
        # A project-local runtime can provide tools without changing the user's
        # global PATH.
        candidate = Path(sys.executable).parent / name
        if candidate.is_file() and os.access(candidate, os.X_OK):
            location = str(candidate)
    if not location:
        raise CaptureError(f"{name} is unavailable. Provide ffmpeg and ffprobe on PATH or beside this Python runtime.")
    return location


def _run(command: list[str], *, timeout: int) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(command, check=True, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise CaptureError(f"{Path(command[0]).name} exceeded its local decoding time limit") from exc
    except subprocess.CalledProcessError as exc:
        # stderr can contain arbitrary private metadata; never echo it to a public log.
        raise CaptureError(f"{Path(command[0]).name} failed (exit {exc.returncode}); inspect the original locally") from exc
    except OSError as exc:
        raise CaptureError(f"Could not start {Path(command[0]).name}") from exc


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise CaptureError(f"{name} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise CaptureError(f"{name} must be a finite number") from exc
    if not math.isfinite(number):
        raise CaptureError(f"{name} must be a finite number")
    return number


def _rate(value: Any) -> float | None:
    if not isinstance(value, str):
        return None
    try:
        top, bottom = value.split("/")
        rate = float(top) / float(bottom)
        return rate if math.isfinite(rate) and rate > 0 else None
    except (ValueError, ZeroDivisionError):
        return None


def probe_media(source: Path) -> dict[str, Any]:
    """Read the first video stream, including phone display rotation, locally."""
    source = _inside_phone(Path(source), label="original media")
    if source.suffix.lower() not in {".mov", ".mp4"} or not source.is_file():
        raise CaptureError("Use an existing original .MOV or .mp4 inside inputs/local/phone/")
    result = _run([
        _tool("ffprobe"), "-v", "error", "-select_streams", "v:0", "-show_entries",
        "stream=index,codec_name,width,height,avg_frame_rate,duration,start_time,sample_aspect_ratio:"
        "stream_tags=rotate:stream_side_data=rotation:format=duration", "-of", "json", str(source),
    ], timeout=30)
    try:
        value = json.loads(result.stdout)
        stream = value["streams"][0]
        width, height = int(stream["width"]), int(stream["height"])
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        raise CaptureError("ffprobe did not return a usable video stream") from exc
    if not (1 <= width <= 16384 and 1 <= height <= 16384):
        raise CaptureError("Video dimensions are outside the supported range")
    duration = _number(stream.get("duration", value.get("format", {}).get("duration")), "video duration")
    if not (0 < duration <= MAX_VIDEO_SECONDS):
        raise CaptureError("Use a clip between 0 and 600 seconds long")
    aspect = stream.get("sample_aspect_ratio", "1:1")
    if aspect not in {"1:1", "0:1", "N/A", None}:
        raise CaptureError("Use a square-pixel phone video so displayed region coordinates stay consistent")
    rotation_value = stream.get("tags", {}).get("rotate", 0)
    for side_data in stream.get("side_data_list", []):
        if "rotation" in side_data:
            rotation_value = side_data["rotation"]
            break
    rotation = _number(rotation_value, "display rotation") % 360
    if rotation not in {0, 90, 180, 270}:
        raise CaptureError("Use video with a 0, 90, 180, or 270 degree display rotation")
    display_width, display_height = (height, width) if rotation in {90, 270} else (width, height)
    return {
        "file": source.name, "sha256": _sha256(source), "bytes": source.stat().st_size,
        "duration_seconds": duration, "encoded_width": width, "encoded_height": height,
        "display_rotation_degrees": int(rotation), "display_width": display_width,
        "display_height": display_height, "average_frame_rate": _rate(stream.get("avg_frame_rate")),
        "timestamp_basis": TIMESTAMP_BASIS,
    }


def validate_plan(plan: Mapping[str, Any], duration_seconds: float) -> list[dict[str, Any]]:
    """Validate declared windows without treating their labels as observations."""
    if not isinstance(plan, Mapping) or plan.get("schema_version") != "1.0":
        raise CaptureError("Capture plan must use schema_version 1.0")
    if plan.get("evidence_origin") not in {"owned_phone", "synthetic_test_fixture"}:
        raise CaptureError("Declare owned_phone or synthetic_test_fixture provenance explicitly")
    annotation_origin = plan.get("ground_truth_origin")
    if annotation_origin not in {"human_declared_not_model_inferred", "user_task_with_assistant_reviewed_timestamps"}:
        raise CaptureError("Declare stage annotation provenance explicitly; timeline labels are not model ground truth")
    if annotation_origin == "user_task_with_assistant_reviewed_timestamps":
        review = plan.get("annotation_review")
        if (not isinstance(review, Mapping) or set(review) != {
                "review_origin", "authorization_reference", "human_reviewed_offsets"}
                or review.get("review_origin") != "assistant_under_user_authorization"
                or review.get("human_reviewed_offsets") is not False):
            raise CaptureError("Assistant-reviewed timestamps need explicit authority and no claim of human offset review")
        reference = review.get("authorization_reference")
        if not isinstance(reference, str) or not reference.strip() or len(reference) > 2000:
            raise CaptureError("Assistant timestamp review requires a nonempty direct user instruction reference")
    elif "annotation_review" in plan:
        raise CaptureError("Assistant annotation review must use its explicit annotation provenance mode")
    stages = plan.get("stages")
    if not isinstance(stages, list) or not 1 <= len(stages) <= MAX_STAGES:
        raise CaptureError("Plan must contain between 1 and 32 stages")
    duration = _number(duration_seconds, "video duration")
    validated = []
    ids: set[str] = set()
    all_offsets: set[float] = set()
    previous_end = 0.0
    for stage in stages:
        if not isinstance(stage, Mapping):
            raise CaptureError("Each stage must be an object")
        identifier = stage.get("id")
        if not isinstance(identifier, str) or not re.fullmatch(r"[a-z][a-z0-9_\-]{0,47}", identifier):
            raise CaptureError("Stage IDs must be short lowercase identifiers")
        if identifier in ids:
            raise CaptureError("Stage IDs must be unique")
        ids.add(identifier)
        label = stage.get("declared_label")
        if not isinstance(label, str) or not 1 <= len(label) <= 500:
            raise CaptureError("Each stage needs a short human declared_label")
        start = _number(stage.get("start_seconds"), "stage start")
        end = _number(stage.get("end_seconds"), "stage end")
        if not (0 <= start < end <= duration + 0.001):
            raise CaptureError("Stage windows must fit inside the video duration")
        if start < previous_end:
            raise CaptureError("Stage windows must be chronological and nonoverlapping")
        previous_end = end
        requested = stage.get("sample_seconds")
        if not isinstance(requested, list) or not 2 <= len(requested) <= 8:
            raise CaptureError("Each stable checkpoint needs two to eight original frame offsets")
        offsets = [_number(item, "sample offset") for item in requested]
        for index, offset in enumerate(offsets):
            if not (start <= offset < end and offset < duration):
                raise CaptureError("Sample offsets must lie inside their declared stable window")
            if offset in all_offsets:
                raise CaptureError("Duplicate source timestamps cannot establish independent stable samples")
            all_offsets.add(offset)
            if index and offset - offsets[index - 1] < MIN_SAMPLE_GAP_SECONDS:
                raise CaptureError("Frame offsets within a checkpoint must increase by at least one second")
        validated.append({"id": identifier, "declared_label": label,
                          "start_seconds": start, "end_seconds": end, "sample_seconds": offsets})
    return validated


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _jpeg_dimensions(data: bytes) -> tuple[int, int]:
    """Read JPEG dimensions without an optional image dependency."""
    if not data.startswith(b"\xff\xd8"):
        raise CaptureError("Decoder output is not a JPEG")
    cursor = 2
    while cursor + 4 <= len(data):
        if data[cursor] != 0xFF:
            raise CaptureError("Decoder output has a malformed JPEG header")
        while cursor < len(data) and data[cursor] == 0xFF:
            cursor += 1
        if cursor >= len(data):
            break
        marker = data[cursor]
        cursor += 1
        if marker in {0x01, 0xD8} or 0xD0 <= marker <= 0xD7:
            continue
        if marker in {0xD9, 0xDA} or cursor + 2 > len(data):
            break
        size = struct.unpack(">H", data[cursor:cursor + 2])[0]
        if size < 2 or cursor + size > len(data):
            raise CaptureError("Decoder output has a truncated JPEG header")
        if marker in {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}:
            if size < 7:
                break
            height, width = struct.unpack(">HH", data[cursor + 3:cursor + 7])
            if 1 <= width <= 1280 and 1 <= height <= 1280:
                return width, height
            raise CaptureError("Extracted JPEG dimensions exceed the 1280 pixel limit")
        cursor += size
    raise CaptureError("Decoder output has no usable JPEG dimensions")


def _extract_frame(source: Path, requested: float, end: float, target: Path) -> dict[str, Any]:
    filters = (
        f"setpts=PTS-STARTPTS,select=gte(t\\,{requested:.6f}),showinfo,"
        "scale=w=min(1280\\,iw):h=min(1280\\,ih):force_original_aspect_ratio=decrease,setsar=1"
    )
    result = _run([
        _tool("ffmpeg"), "-hide_banner", "-nostdin", "-loglevel", "info", "-i", str(source),
        "-map", "0:v:0", "-an", "-sn", "-dn", "-map_metadata", "-1", "-vf", filters,
        "-frames:v", "1", "-fps_mode", "passthrough", "-q:v", "2", "-f", "image2", "-n", str(target),
    ], timeout=120)
    matches = re.findall(r"\bn:\s*\d+\s+pts:\s*-?\d+\s+pts_time:\s*([-+\deE.]+)", result.stderr)
    if not matches or not target.is_file():
        raise CaptureError("Decoder did not return a JPEG and an actual decoded frame timestamp")
    actual = _number(matches[0], "decoded frame timestamp")
    if actual + 0.000001 < requested or actual >= end:
        raise CaptureError("Actual decoded frame falls outside the requested stable checkpoint")
    data = target.read_bytes()
    if not data or len(data) > MAX_FRAME_BYTES:
        raise CaptureError("Extracted JPEG exceeds the hosted model's 5 MiB input limit")
    width, height = _jpeg_dimensions(data)
    target.chmod(0o600)
    return {
        "file": target.name, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data),
        "width": width, "height": height, "requested_seconds": requested,
        "source_timestamp_seconds": actual, "source_timestamp_ms": round(actual * 1000),
        "timestamp_basis": TIMESTAMP_BASIS,
    }


def _save(path: Path, value: Any) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def extract_capture(source: Path, plan: Mapping[str, Any], output: Path, *,
                    confirmed_offsets: bool = False) -> dict[str, Any]:
    """Extract original frames after the declared reviewer confirms the clip plan.

    No stage transition, ownership, visibility or successful correction is
    inferred here. Original decoded PTS, not requested seek times, are retained.
    """
    if confirmed_offsets is not True:
        raise CaptureError("Watch the original, edit the plan, then explicitly --confirm-offsets")
    source = _inside_phone(Path(source), label="original media")
    destination = _inside_phone(Path(output), label="extracted frames")
    if destination.exists():
        raise CaptureError("Output already exists; choose a fresh run directory so evidence is never overwritten")
    info = probe_media(source)
    stages = validate_plan(plan, info["duration_seconds"])
    _tool("ffmpeg")
    destination.mkdir(parents=True, exist_ok=False, mode=0o700)
    manifest = {
        "schema_version": "1.0", "experiment": "phone_capture_preflight",
        "evidence_origin": plan["evidence_origin"],
        "origin_verification": "caller_declared_not_independently_verified",
        "ground_truth_origin": plan["ground_truth_origin"], "offsets_confirmed": True,
        "inference_performed": False, "physical_repair_verified": False,
        "source": dict(info, file=os.path.relpath(source, destination)),
        "extraction": {"decoder": "ffmpeg", "autorotate": True, "max_dimension": 1280,
                       "format": "jpeg", "quality": 2, "timestamps": TIMESTAMP_BASIS,
                       "no_frame_interpolation": True},
        "stages": [], "status": "incomplete",
        "limits": [
            "Stage labels and boundaries are reviewer annotations, not model observations.",
            "PTS are media-relative and do not prove wall-clock capture time.",
            "Duplicate image hashes cannot establish distinct stable visual evidence.",
            "Region rectangles must be approved on the displayed extracted image before grounding.",
        ],
    }
    if "annotation_review" in plan:
        manifest["annotation_review"] = deepcopy(plan["annotation_review"])
    try:
        seen_timestamps: set[float] = set()
        for stage in stages:
            frames = []
            for index, offset in enumerate(stage["sample_seconds"], 1):
                target = destination / f"{stage['id']}-{index:02d}.jpg"
                frame = _extract_frame(source, offset, stage["end_seconds"], target)
                actual = frame["source_timestamp_seconds"]
                if actual in seen_timestamps:
                    raise CaptureError("Decoder returned duplicate actual source timestamps")
                if frames and actual - frames[-1]["source_timestamp_seconds"] < MIN_SAMPLE_GAP_SECONDS - 0.001:
                    raise CaptureError("Actual decoded checkpoint frames are less than one second apart")
                seen_timestamps.add(actual)
                frames.append(frame)
            manifest["stages"].append(dict(stage, frames=frames,
                                          distinct_frame_hashes=len({item["sha256"] for item in frames}) == len(frames)))
        # An in-place changed source must not be certified under the preflight digest.
        if _sha256(source) != info["sha256"]:
            raise CaptureError("Original video changed during extraction; discard this run and retry")
        manifest["status"] = "extracted_pending_visual_review"
        _save(destination / "manifest.json", manifest)
        return manifest
    except (CaptureError, OSError):
        # Partial frames stay private and clearly incomplete; never masquerade as a completed run.
        if not (destination / "manifest.incomplete.json").exists():
            _save(destination / "manifest.incomplete.json", manifest)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan_parser = commands.add_parser("plan", help="Write the editable human annotation template")
    plan_parser.add_argument("--output", type=Path, default=PHONE_ROOT / "capture-plan.json")
    preflight_parser = commands.add_parser("preflight", help="Inspect an original MOV/MP4 locally")
    preflight_parser.add_argument("source", type=Path)
    extract_parser = commands.add_parser("extract", help="Extract timestamped JPEGs from confirmed stable windows")
    extract_parser.add_argument("source", type=Path)
    extract_parser.add_argument("--plan", type=Path, required=True)
    extract_parser.add_argument("--output", type=Path, required=True)
    extract_parser.add_argument("--confirm-offsets", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command == "plan":
            destination = _inside_phone(args.output, label="capture plans")
            destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            _save(destination, default_plan())
            print(f"Editable plan: {destination}\nWatch the clip and correct the offsets before extraction.")
        elif args.command == "preflight":
            print(json.dumps(probe_media(args.source), indent=2, allow_nan=False))
        else:
            plan_path = _inside_phone(args.plan, label="capture plans")
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
            manifest = extract_capture(args.source, plan, args.output, confirmed_offsets=args.confirm_offsets)
            print(json.dumps({"status": manifest["status"], "stages": len(manifest["stages"]),
                              "frames": sum(len(stage["frames"]) for stage in manifest["stages"]),
                              "manifest": str(args.output.resolve() / "manifest.json"),
                              "model_requests": 0}, indent=2))
    except (CaptureError, OSError, json.JSONDecodeError) as exc:
        print(f"Capture stopped: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
