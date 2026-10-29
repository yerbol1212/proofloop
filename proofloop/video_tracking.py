"""Source-bound, reviewed-seed SAM video experiment; never an identity proof.

Wire and rates checked against official Meta documentation on 2026-10-29:
https://dev.meta.ai/docs/sam/segmenting
https://dev.meta.ai/docs/sam/reading-segmentation
https://dev.meta.ai/docs/pricing-rate-limits

Non-streaming SAM video is explicitly supported. This adapter sends one bounded
clip in one request through the frozen Meta transport. Missing records become
UNKNOWN observations; they never establish absence. Track IDs are seed-relative
and must not establish identity across full occlusion or exit/re-entry.
"""
from __future__ import annotations

import base64
from copy import deepcopy
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any, Mapping, Sequence

from .core import ValidationError
from .meta import (MetaAPIError, MetaClient, SAM_MODEL, _sam_decoder,
                   decode_sam_masks, output_text, parse_sam_boxes)


TRACKING_VERSION = "sam_video_tracking_v1"
MAX_CLIP_BYTES = 32 * 1024 * 1024
MAX_CLIP_MS = 12_000
MAX_CLIP_FRAMES = 120
MAX_SEEDS = 6
MAX_TOTAL_MASK_PIXELS = 32_000_000
_HASH = re.compile(r"[0-9a-f]{64}\Z")
_ID = re.compile(r"[A-Za-z][A-Za-z0-9_-]{0,63}\Z")
_FRAME = re.compile(r"<(\d+)f>")
_SAFE_ERROR_CODES = {"incomplete_response", "network_error", "sam_parse_failed", "sam_invalid_video_frame",
    "sam_invalid_object_id", "sam_invalid_video_geometry", "sam_raster_limit", "invalid_sam_mask", "invalid_sam_raster",
    "model_refusal", "invalid_output", "invalid_json", "meta_api_error", "video_adapter_error"}
_PREPARED_KEYS = {"method_version", "clip_file", "source", "seeds", "anchor_origin", "author_review",
    "propagation_direction", "requested_frame_indices", "prompt", "probe_version", "prepared_hash"}
_LIMITS = [
    "Reviewed seed boxes are authored inputs, not automatic target localization or independent semantic identity proof.",
    "A track ID may not establish identity across full occlusion, exit/re-entry, or long clips.",
    "Missing frame/object records mean UNKNOWN, never verified absence; returned scores are not calibrated correctness probabilities.",
    "The source SHA identifies the submitted clip bytes; frame indices and PTS identify decoded frames within that clip.",
    "Reported costs use published video-frame rates and the 50-frame minimum, not the image-segmentation estimate; provider billing remains authoritative.",
    "This experiment does not change live guided-v2 acceptance, verify physical execution or authorize actions.",
]


def _hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _integer(value: Any, *, minimum: int = 0, maximum: int = 2 ** 63 - 1) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValidationError("Video tracking integer is invalid")
    return value


def _probe_path(value: str | Path | None) -> Path:
    default = Path(__file__).resolve().parents[1] / ".venv/bin/ffprobe"
    candidate = value if value is not None else default if default.is_file() else shutil.which("ffprobe")
    if candidate is None or not Path(candidate).is_file():
        raise ValidationError("A local ffprobe executable is required before tracking")
    return Path(candidate).resolve()


def _probe(clip_path: Path, ffprobe_path: str | Path | None = None) -> tuple[dict[str, Any], str]:
    executable = _probe_path(ffprobe_path)
    try:
        version_result = subprocess.run([str(executable), "-version"], stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, timeout=10, check=True)
        version = version_result.stdout.decode("utf-8").splitlines()[0]
        if not version.startswith("ffprobe version ") or len(version) > 512:
            raise ValueError
        result = subprocess.run([str(executable), "-v", "error", "-select_streams", "v", "-show_streams", "-show_frames",
            "-show_entries", "stream=index,codec_type,codec_name,width,height,time_base,duration,nb_frames:stream_tags=rotate:stream_side_data=rotation:format=format_name,duration:frame=media_type,width,height,pts,best_effort_timestamp",
            "-of", "json", str(clip_path)], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=30, check=True)
        if len(result.stdout) > 8 * 1024 * 1024:
            raise ValueError
        metadata = json.loads(result.stdout)
        streams = metadata["streams"]
        if type(streams) is not list or len(streams) != 1:
            raise ValueError
        stream = streams[0]
        if stream.get("codec_type") != "video" or stream.get("codec_name") not in {"h264", "hevc", "mpeg4"}:
            raise ValueError
        width = _integer(stream["width"], minimum=16, maximum=1280)
        height = _integer(stream["height"], minimum=16, maximum=1280)
        if width * height > 1_638_400:
            raise ValueError
        if Fraction(str(stream.get("tags", {}).get("rotate", "0"))) != 0:
            raise ValueError
        if any(Fraction(str(item.get("rotation", "0"))) != 0 for item in stream.get("side_data_list", [])):
            raise ValueError
        time_base = Fraction(stream["time_base"])
        if not 0 < time_base <= 1 or time_base.denominator > 10 ** 9:
            raise ValueError
        format_data = metadata["format"]
        if not set(format_data["format_name"].split(",")) & {"mov", "mp4"}:
            raise ValueError
        durations = [Fraction(str(value)) for value in (format_data.get("duration"), stream.get("duration")) if value is not None]
        if not durations or any(not 0 < duration <= Fraction(MAX_CLIP_MS, 1000) for duration in durations):
            raise ValueError
        duration = max(durations) * 1000
        duration_ms = (duration.numerator + duration.denominator - 1) // duration.denominator
        frames = metadata["frames"]
        if type(frames) is not list or not 1 <= len(frames) <= MAX_CLIP_FRAMES:
            raise ValueError
        if stream.get("nb_frames") not in (None, "N/A") and int(stream["nb_frames"]) != len(frames):
            raise ValueError
        frame_map = []
        first_pts = None
        previous_pts = None
        for index, frame in enumerate(frames):
            if frame.get("media_type") != "video" or (frame.get("width"), frame.get("height")) != (width, height):
                raise ValueError
            raw_pts = frame.get("best_effort_timestamp", frame.get("pts"))
            if type(raw_pts) not in (str, int) or type(raw_pts) is str and re.fullmatch(r"-?[0-9]{1,19}", raw_pts) is None:
                raise ValueError
            pts = int(raw_pts)
            if not -(2 ** 63) < pts < 2 ** 63 or previous_pts is not None and pts <= previous_pts:
                raise ValueError
            if first_pts is None:
                first_pts = pts
            timestamp = (pts - first_pts) * time_base * 1000
            timestamp_ms = timestamp.numerator // timestamp.denominator
            if timestamp_ms >= duration_ms + 2:
                raise ValueError
            frame_map.append({"frame_index": index, "pts": pts, "timestamp_ms": timestamp_ms})
            previous_pts = pts
        return {"width": width, "height": height, "codec": stream["codec_name"], "container": "mp4_or_mov",
            "duration_ms": duration_ms, "frame_count": len(frame_map), "time_base": str(time_base),
            "first_pts": first_pts, "frames": frame_map}, version
    except (OSError, subprocess.SubprocessError, UnicodeError, ValueError, TypeError, KeyError, ZeroDivisionError, IndexError):
        raise ValidationError("Video probe failed the bounded upright MP4/MOV frame contract") from None


def _clip_bytes(path: Path) -> tuple[bytes, str]:
    if path.is_symlink() or not path.is_file() or not 1 <= path.stat().st_size <= MAX_CLIP_BYTES:
        raise ValidationError("Tracking requires a regular clip of at most 32 MiB")
    with path.open("rb") as stream:
        data = stream.read(MAX_CLIP_BYTES + 1)
    if not 12 <= len(data) <= MAX_CLIP_BYTES or data[4:8] != b"ftyp":
        raise ValidationError("Tracking requires an MP4 or MOV media signature")
    return data, hashlib.sha256(data).hexdigest()


def _seeds(value: Any, source: Mapping[str, Any]) -> list[dict[str, Any]]:
    if type(value) is not list or not 1 <= len(value) <= MAX_SEEDS:
        raise ValidationError("Review between one and six video seed objects")
    result = []
    track_ids, object_ids = set(), set()
    for seed in value:
        if type(seed) is not dict or set(seed) != {"object_id", "track_id", "frame_index", "box"}:
            raise ValidationError("Each reviewed seed needs an object ID, track ID, frame index and pixel box")
        object_id, track_id, index, box = seed["object_id"], seed["track_id"], seed["frame_index"], seed["box"]
        if type(object_id) is not str or _ID.fullmatch(object_id) is None or object_id in object_ids:
            raise ValidationError("Video seed object IDs must be distinct safe identifiers")
        _integer(track_id, maximum=2 ** 31 - 1)
        _integer(index, maximum=source["frame_count"] - 1)
        if track_id in track_ids:
            raise ValidationError("Each reviewed object needs one distinct SAM track ID")
        if type(box) is not dict or set(box) != {"x1", "y1", "x2", "y2"} or any(type(coordinate) is not int for coordinate in box.values()):
            raise ValidationError("Review an inclusive integer pixel seed box")
        if not 0 <= box["x1"] < box["x2"] < source["width"] or not 0 <= box["y1"] < box["y2"] < source["height"]:
            raise ValidationError("Reviewed seed box must be inside its source frame")
        result.append({**deepcopy(seed), "source_sha256": source["sha256"],
            "pts": source["frames"][index]["pts"], "timestamp_ms": source["frames"][index]["timestamp_ms"]})
        track_ids.add(track_id)
        object_ids.add(object_id)
    return sorted(result, key=lambda item: (item["frame_index"], item["track_id"]))


def _prompt(seeds: Sequence[Mapping[str, Any]], source: Mapping[str, Any]) -> str:
    output = []
    for index in sorted({seed["frame_index"] for seed in seeds}):
        records = []
        for seed in seeds:
            if seed["frame_index"] == index:
                box = seed["box"]
                records.append(str(seed["track_id"]) + "<|box;" + ";".join(key + "=" + str(box[key]) for key in ("x1", "y1", "x2", "y2"))
                    + ";w=" + str(source["width"]) + ";h=" + str(source["height"]) + "|>")
        output.append("<" + str(index) + "f>" + ",".join(records))
    return "".join(output)


def prepare_clip(clip_path: str | Path, seeds: list[dict[str, Any]], *, review_note: str,
                 review_origin: str = "assistant_under_user_authorization", propagation_direction: str = "both",
                 ffprobe_path: str | Path | None = None) -> dict[str, Any]:
    """Freeze already reviewed pixel seeds and an exact decoded source map; no API call.

    The caller must review the source frame and author each seed box before
    invoking this function. This records that declaration; it does not verify a
    person's identity, manufacture a model locator, or review later frames.
    """
    path = Path(clip_path).absolute()
    data, sha256 = _clip_bytes(path)
    source, probe_version = _probe(path, ffprobe_path)
    source = {"sha256": sha256, "bytes": len(data), "mime": "video/mp4" if path.suffix.lower() != ".mov" else "video/quicktime", **source}
    reviewed = _seeds(seeds, source)
    if (review_origin not in ("local_operator", "assistant_under_user_authorization") or type(review_note) is not str
            or not 1 <= len(review_note.strip()) <= 1500 or any(ord(character) < 32 for character in review_note)):
        raise ValidationError("Record the reviewed seed origin and a concise source review note")
    earliest = min(seed["frame_index"] for seed in reviewed)
    if propagation_direction not in ("both", "forward", "backward") or propagation_direction == "backward" and earliest == 0:
        raise ValidationError("Use a documented tracking direction with a valid earliest seed frame")
    requested = list(range(source["frame_count"])) if propagation_direction == "both" else list(range(earliest, source["frame_count"])) if propagation_direction == "forward" else list(range(earliest + 1))
    prepared = {"method_version": TRACKING_VERSION, "clip_file": str(path), "source": source, "seeds": reviewed,
        "anchor_origin": "authored_reviewed_video_boxes", "author_review": {"origin": review_origin,
            "source_frame_and_boxes_reviewed": True, "note": review_note}, "propagation_direction": propagation_direction,
        "requested_frame_indices": requested, "prompt": _prompt(reviewed, source), "probe_version": probe_version}
    prepared["prepared_hash"] = _hash(prepared)
    return prepared


def verify_prepared(prepared: Mapping[str, Any], *, ffprobe_path: str | Path | None = None) -> None:
    try:
        if (not isinstance(prepared, Mapping) or set(prepared) != _PREPARED_KEYS or prepared["method_version"] != TRACKING_VERSION
                or type(prepared["prepared_hash"]) is not str or _HASH.fullmatch(prepared["prepared_hash"]) is None
                or _hash({key: value for key, value in prepared.items() if key != "prepared_hash"}) != prepared["prepared_hash"]
                or prepared["anchor_origin"] != "authored_reviewed_video_boxes"):
            raise ValueError
        review = prepared["author_review"]
        if type(review) is not dict or set(review) != {"origin", "source_frame_and_boxes_reviewed", "note"} or review["source_frame_and_boxes_reviewed"] is not True:
            raise ValueError
        seeds = [{key: deepcopy(seed[key]) for key in ("object_id", "track_id", "frame_index", "box")} for seed in prepared["seeds"]]
        reproduced = prepare_clip(prepared["clip_file"], seeds, review_note=review["note"], review_origin=review["origin"],
            propagation_direction=prepared["propagation_direction"], ffprobe_path=ffprobe_path)
        if reproduced != dict(prepared):
            raise ValueError
    except (ValidationError, OSError, ValueError, TypeError, KeyError):
        raise ValidationError("Reviewed video preparation changed or failed source replay") from None


def request_binding(prepared: Mapping[str, Any]) -> dict[str, Any]:
    return {"prepared_hash": prepared["prepared_hash"], "source_sha256": prepared["source"]["sha256"],
        "prompt_sha256": hashlib.sha256(prepared["prompt"].encode()).hexdigest(),
        "model": SAM_MODEL, "stream": False, "store": False,
        "metadata": {"include_confidence": "true", "mask_encoding": "one_bit",
            "propagation_direction": prepared["propagation_direction"]}}


def video_accounting(prepared: Mapping[str, Any], usage: Any, *, completed: bool) -> dict[str, Any]:
    usage = usage if isinstance(usage, Mapping) else {}
    reported = usage.get("video_frames_processed")
    valid = type(reported) is int and 50 <= reported <= 15_000
    planned = max(50, len(prepared["requested_frame_indices"]))
    billed = reported if valid else planned if completed else None
    return {"request_attempts": 1, "video_requests": 1, "automatic_retries": 0,
        "requested_tracking_frames": len(prepared["requested_frame_indices"]), "minimum_billed_frames_per_request": 50,
        "reported_billed_frames": reported if valid else None,
        "reported_video_frames_invalid": reported is not None and not valid,
        "estimated_billable_frames": billed, "usd_per_1000_video_frames": 0.20,
        "pre_request_estimated_usd": round(planned * 0.20 / 1000, 6),
        "estimated_usd": round(billed * 0.20 / 1000, 6) if billed is not None else None,
        "estimate_basis": "reported_video_frames_processed" if valid else "requested_tracking_range_with_documented_minimum" if completed else "unreported_failed_or_ambiguous_attempt",
        "provider_billing_verified": False, "unreported_charge_possible": not completed}


def _frame_records(prepared: Mapping[str, Any], text: str, *, decode_masks: bool) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    markers = _FRAME.findall(text)
    if len(markers) != len(set(int(value) for value in markers)):
        raise MetaAPIError("SAM video repeated a source frame marker", code="sam_invalid_video_frame")
    try:
        objects = parse_sam_boxes(text)
    except ValueError:
        raise MetaAPIError("SAM video output failed geometry parsing", code="sam_parse_failed") from None
    by_track = {seed["track_id"]: seed for seed in prepared["seeds"]}
    source, requested = prepared["source"], set(prepared["requested_frame_indices"])
    total_area = 0
    frame_candidates = {index: [] for index in requested}
    for record in objects:
        if record["frame"] not in requested:
            raise MetaAPIError("SAM video returned an unrequested source frame", code="sam_invalid_video_frame")
        if record["object_id"] not in by_track:
            raise MetaAPIError("SAM video returned an unreviewed track ID", code="sam_invalid_object_id")
        box, mask = record["box"], record["mask"]
        if (box["width"], box["height"]) != (source["width"], source["height"]) or mask["width"] > source["width"] or mask["height"] > source["height"]:
            raise MetaAPIError("SAM video geometry differs from its source clip", code="sam_invalid_video_geometry")
        area = mask["width"] * mask["height"]
        total_area += area
        if total_area > MAX_TOTAL_MASK_PIXELS:
            raise MetaAPIError("SAM video masks exceed the total decode budget", code="sam_raster_limit")
        mask_support = False
        mask_summary = {"decoded": False, "width": mask["width"], "height": mask["height"],
            "encoding": mask["encoding"], "foreground_pixels": None, "raster_sha256": None}
        if decode_masks:
            # Decode one record at a time so a clip does not amplify the
            # official parser's batch allocation. Retain only safe mask facts;
            # the complete encoded mask remains in the private response.
            decoded = decode_sam_masks([record])[0]["mask"]
            mask_support = decoded["foreground_pixels"] > 0
            mask_summary.update(decoded=True, foreground_pixels=decoded["foreground_pixels"], raster_sha256=decoded["raster_sha256"])
        candidate = {"track_id": record["object_id"], "object_id": by_track[record["object_id"]]["object_id"],
            "box": [box["x1"] / source["width"], box["y1"] / source["height"],
                (box["x2"] + 1) / source["width"], (box["y2"] + 1) / source["height"]],
            "confidence": record["confidence"], "mask_supported": mask_support, "mask": mask_summary,
            "identity_independently_verified": False}
        frame_candidates[record["frame"]].append(candidate)
    frames = []
    for frame in source["frames"]:
        candidates = sorted(frame_candidates.get(frame["frame_index"], []), key=lambda item: item["track_id"])
        frames.append({"source_sha256": source["sha256"], "source_kind": "video_clip_frame",
            **deepcopy(frame), "time_base": source["time_base"], "width": source["width"], "height": source["height"],
            "tracking_requested": frame["frame_index"] in requested, "candidates": candidates,
            "missing_track_ids": sorted(set(by_track) - {candidate["track_id"] for candidate in candidates}),
            "observation_status": "recorded_candidates" if candidates else "unknown",
            "unknown_reason": None if candidates else "no_track_record" if frame["frame_index"] in requested else "outside_requested_tracking_range",
            "absence_verified": False})
    return sorted(objects, key=lambda record: (record["frame"], record["object_id"])), frames


def segment_video(client: MetaClient, prepared: Mapping[str, Any], *, decode_masks: bool = True,
                  ffprobe_path: str | Path | None = None) -> dict[str, Any]:
    """Exactly one completed or unscored SAM video attempt; no fallback or retry."""
    verify_prepared(prepared, ffprobe_path=ffprobe_path)
    if type(client.max_requests) is not int or client.max_requests != 1 or type(client.retries) is not int or client.retries != 0 or client.ledger:
        raise ValidationError("Use a fresh Meta client with exactly one request and no retries")
    if type(decode_masks) is not bool:
        raise ValidationError("Choose mask decoding explicitly")
    if decode_masks:
        _sam_decoder()  # The dependency is checked before paid inference.
    data, sha256 = _clip_bytes(Path(prepared["clip_file"]))
    if sha256 != prepared["source"]["sha256"]:
        raise ValidationError("Video bytes changed before model dispatch")
    binding = request_binding(prepared)
    report = {"method_version": TRACKING_VERSION, "status": "unscored", "prepared": deepcopy(dict(prepared)),
        "request_binding": binding, "response": None, "text": None, "objects": [], "frames": [],
        "mask_decoding": "official_meta_sam_parser_0.1.0" if decode_masks else "skipped_explicitly",
        "usage": {}, "cost": None, "error": None, "changes_guided_v2_acceptance": False}
    try:
        result = client._request("POST", "responses", {"model": SAM_MODEL,
            "input": [{"type": "message", "role": "user", "content": [
                {"type": "input_text", "text": prepared["prompt"]},
                {"type": "input_video", "video_url": "data:" + prepared["source"]["mime"] + ";base64," + base64.b64encode(data).decode("ascii")},
            ]}], "stream": False, "store": False, "metadata": deepcopy(binding["metadata"])})
        report["response"] = result
        report["usage"] = deepcopy(result.get("usage", {})) if isinstance(result.get("usage", {}), dict) else {}
        text = output_text(result)
        report["text"] = text
        objects, frames = _frame_records(prepared, text, decode_masks=decode_masks)
        report.update(status="completed", objects=objects, frames=frames)
    except MetaAPIError as error:
        if report["response"] is None:
            report["response"] = error.response
        if isinstance(report["response"], Mapping) and isinstance(report["response"].get("usage"), dict):
            report["usage"] = deepcopy(report["response"]["usage"])
        elif client.ledger and isinstance(client.ledger[-1].get("usage"), dict):
            report["usage"] = deepcopy(client.ledger[-1]["usage"])
        report["error"] = {"code": error.code if error.code in _SAFE_ERROR_CODES else "meta_api_error",
            "http_status": error.status if type(error.status) is int and 100 <= error.status <= 599 else None}
    except Exception:
        # An unexpected optional-decoder/adapter exception must not lose an
        # already billable private response or become an empty detection set.
        # The private raw response and request ledger remain inspectable.
        report.update(status="unscored", objects=[], frames=[], error={"code": "video_adapter_error", "http_status": None})
        if isinstance(report["response"], Mapping) and isinstance(report["response"].get("usage"), dict):
            report["usage"] = deepcopy(report["response"]["usage"])
        elif client.ledger and isinstance(client.ledger[-1].get("usage"), dict):
            report["usage"] = deepcopy(client.ledger[-1]["usage"])
    report["cost"] = video_accounting(prepared, report["usage"], completed=report["status"] == "completed")
    report["attempts"] = deepcopy(client.ledger)
    return report


def public_tracking_summary(report: Mapping[str, Any], *, ffprobe_path: str | Path | None = None) -> dict[str, Any]:
    """Rebind response geometry and decode mask facts before a public projection."""
    try:
        prepared = report["prepared"]
        verify_prepared(prepared, ffprobe_path=ffprobe_path)
        if report["method_version"] != TRACKING_VERSION or report["request_binding"] != request_binding(prepared) or report["changes_guided_v2_acceptance"] is not False:
            raise ValueError
        attempts = report["attempts"]
        if (type(attempts) is not list or len(attempts) != 1 or type(attempts[0]) is not dict
                or attempts[0].get("endpoint") != "responses" or attempts[0].get("model") != SAM_MODEL
                or attempts[0].get("attempt") != 1 or attempts[0].get("status") not in
                    {"ok", "http_error", "network_error", "invalid_json", "invalid_response"}
                or attempts[0].get("usage", {}) != report["usage"]):
            raise ValueError
        decoding = report["mask_decoding"] == "official_meta_sam_parser_0.1.0"
        if not decoding and report["mask_decoding"] != "skipped_explicitly":
            raise ValueError
        if report["status"] == "completed":
            if (report["response"].get("status") != "completed" or report["response"].get("error")
                    or report["error"] is not None or attempts[0]["status"] != "ok"):
                raise ValueError
            text = output_text(report["response"])
            objects, frames = _frame_records(prepared, text, decode_masks=decoding)
            if text != report["text"] or objects != report["objects"] or frames != report["frames"]:
                raise ValueError
        elif report["status"] == "unscored" and report["error"] is not None and report["frames"] == [] and report["objects"] == []:
            frames = []
            if (type(report["error"]) is not dict or set(report["error"]) != {"code", "http_status"}
                    or type(report["error"]["code"]) is not str or report["error"]["code"] not in _SAFE_ERROR_CODES
                    or report["error"]["http_status"] is not None and (type(report["error"]["http_status"]) is not int or not 100 <= report["error"]["http_status"] <= 599)):
                raise ValueError
        else:
            raise ValueError
        response_usage = report["response"].get("usage", {}) if isinstance(report["response"], Mapping) else report["usage"]
        if not isinstance(response_usage, Mapping):
            response_usage = {}
        if report["usage"] != response_usage or report["cost"] != video_accounting(prepared, report["usage"], completed=report["status"] == "completed"):
            raise ValueError
        source = {key: deepcopy(prepared["source"][key]) for key in ("sha256", "bytes", "mime", "width", "height", "codec", "duration_ms", "frame_count", "time_base", "first_pts")}
        usage = {key: report["usage"][key] for key in ("video_frames_processed", "input_tokens", "output_tokens", "total_tokens")
            if type(report["usage"].get(key)) is int and 0 <= report["usage"][key] < 2 ** 63}
        return {"method_version": TRACKING_VERSION, "status": report["status"], "request_binding": request_binding(prepared),
            "source": source, "seeds": deepcopy(prepared["seeds"]), "anchor_origin": prepared["anchor_origin"],
            "review_origin": prepared["author_review"]["origin"], "seed_review_declared": True,
            "frames": deepcopy(frames), "mask_decoding": report["mask_decoding"], "usage": usage,
            "cost": deepcopy(report["cost"]), "error": deepcopy(report["error"]), "limitations": list(_LIMITS),
            "changes_guided_v2_acceptance": False, "physical_action_authorized": False}
    except (ValidationError, MetaAPIError, ValueError, TypeError, KeyError, AttributeError):
        raise ValidationError("Video tracking report failed source-bound replay") from None
