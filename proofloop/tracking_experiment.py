"""Frozen real-video development trials; reference labels never enter inference.

Four diagnostic comparisons share exact selected frames: independent two-view
Spark localization, existing guided-v2 still refinement, seeded SAM video
presence, and experimental source-bound semantic reacquisition. These are not
four interchangeable placement verifiers; initialization and costs differ.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
from typing import Any, Mapping

from .core import ValidationError
from .guided import (GUIDED_VERSION, guided_segment_pair, normalized_box_to_pixels, _frames,
                     _locator_bound, _sam_bound)
from .inference_journal import AttemptJournal, JournaledMetaClient
from .meta import MetaAPIError, MetaClient, SAM_MODEL, SPARK_MODEL, decode_sam_masks, load_env_file, output_text, parse_sam_boxes
from . import tracking_guard as guard
from .tracking_evaluation import score_frames
from .tracking_policy import evaluate_tracking
from .video_tracking import prepare_clip, segment_video, public_tracking_summary, verify_prepared

VERSION = "frozen_occlusion_tracking_comparison_v1"
METHODS = ("spark_two_view_locator", "guided_v2_stills", "sam_video_presence", "semantic_reacquisition")
CONFIG = {"min_confidence": 0.8, "min_stable_views": 2, "min_spacing_ms": 500,
          "localization_iou_threshold": 0.3, "localization_sensitivity_thresholds": [0.2, 0.3, 0.5],
          "max_requests": 42, "estimated_allocation_usd": 3.0, "automatic_retry": False,
          "semantic_checkpoints": "only preselected annotated frames with an actual tracker candidate",
          "baseline": "unchanged guided-v2 request settings; its Spark locator is the shared Spark baseline"}
FILES = ("tracking_experiment.py", "video_tracking.py", "tracking_policy.py", "tracking_guard.py",
         "tracking_evaluation.py", "guided.py", "meta.py", "semantic.py", "inference_journal.py")


class RetainingClient(JournaledMetaClient):
    """Retain completed partial baseline calls when a later pair request fails."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.retained_responses = []

    def _request(self, *args, **kwargs):
        try:
            response = super()._request(*args, **kwargs)
        except MetaAPIError as error:
            self.retained_responses.append({"status": "error", "response": deepcopy(error.response)})
            raise
        self.retained_responses.append({"status": "received", "response": deepcopy(response)})
        return response


def _digest(value):
    return guard.digest(value)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _read(path):
    def unique(rows):
        value = {}
        for key, item in rows:
            if key in value:
                raise ValidationError("Duplicate experiment JSON key")
            value[key] = item
        return value
    try:
        return json.loads(Path(path).read_text(), object_pairs_hook=unique,
                          parse_constant=lambda _: (_ for _ in ()).throw(ValueError()))
    except (ValueError, OSError, UnicodeError):
        raise ValidationError("Experiment JSON is unavailable or invalid") from None


def _write(path, value):
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w") as output:
        output.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
        output.flush()
        os.fsync(output.fileno())


def _local(root, value):
    path = (root / value).resolve()
    if not any(path.is_relative_to(root / prefix) for prefix in ("inputs/local", "artifacts/local")):
        raise ValidationError("Experiment inputs must remain in private local storage")
    return path


def _implementations():
    return {name: _sha(Path(__file__).with_name(name)) for name in FILES}


def _pixels(normalized, width, height):
    return {"x1": round(normalized[0] * width), "y1": round(normalized[1] * height),
            "x2": min(width - 1, round(normalized[2] * width) - 1),
            "y2": min(height - 1, round(normalized[3] * height) - 1)}


def _normalized(box, width, height):
    return [box["x1"] / width, box["y1"] / height, (box["x2"] + 1) / width, (box["y2"] + 1) / height]


def _reference(entry):
    return [{"clip_id": entry["clip_id"], **{key: row[key] for key in
             ("frame_index", "expected", "target_box", "distractor_boxes", "condition")}}
            for row in entry["checkpoints"]]


def _attribution(value):
    fields = {"dataset", "revision", "episode", "remote_path", "url", "license", "license_url", "source_role"}
    if (type(value) is not dict or set(value) != fields
            or not isinstance(value["dataset"], str) or not re.fullmatch(r"lerobot/[A-Za-z0-9_-]{1,80}", value["dataset"])
            or not isinstance(value["revision"], str) or not re.fullmatch(r"[0-9a-f]{40}", value["revision"])
            or type(value["episode"]) is not int or not 0 <= value["episode"] <= 999999
            or value["remote_path"] != f"videos/chunk-000/observation.images.cam_high/episode_{value['episode']:06d}.mp4"
            or value["url"] != f"https://huggingface.co/datasets/{value['dataset']}/resolve/{value['revision']}/{value['remote_path']}"
            or value["license_url"] != f"https://huggingface.co/datasets/{value['dataset']}/blob/{value['revision']}/README.md"
            or value["license"] not in ("mit", "apache-2.0")
            or value["source_role"] != "new_source_episode_same_exposed_dataset_family"):
        raise ValidationError("Tracking attribution must be a closed pinned licensed-source record")
    return deepcopy(value)


def _derivation(entry):
    """Replay native frame selection; a license cannot be attached to another clip."""
    tool_root = Path(__file__).resolve().parents[1] / ".venv/bin"
    probe = subprocess.run([str(tool_root / "ffprobe"), "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate,width,height:frame=best_effort_timestamp_time", "-show_frames",
        "-of", "json", entry["original_file"]], capture_output=True, check=True, timeout=30)
    native = json.loads(probe.stdout)
    source = entry["prepared"]["source"]
    stream, frames = native["streams"][0], native["frames"]
    if (stream["r_frame_rate"] != "50/1" or (stream["width"], stream["height"]) != (source["width"], source["height"])
            or len(frames) != source["frame_count"] * 10
            or any(abs(float(frame["best_effort_timestamp_time"]) - i / 50) > 0.000001 for i, frame in enumerate(frames))
            or any(frame["timestamp_ms"] != frame["frame_index"] * 200 for frame in source["frames"])):
        raise ValidationError("Original and derivative source frame timelines do not match the fixed selection")
    def decoded(path, selection=False):
        command = [str(tool_root / "ffmpeg"), "-v", "error", "-nostdin", "-i", path, "-map", "0:v:0"]
        if selection:
            command += ["-vf", r"select=not(mod(n\,10))", "-fps_mode", "passthrough"]
        command += ["-pix_fmt", "yuv420p", "-f", "framemd5", "-"]
        result = subprocess.run(command, capture_output=True, check=True, timeout=30)
        return [line.split(",")[-1].strip() for line in result.stdout.decode().splitlines() if line and not line.startswith("#")]
    first, second = decoded(entry["original_file"], True), decoded(entry["prepared"]["clip_file"])
    if first != second or len(first) != source["frame_count"]:
        raise ValidationError("Tracking clip does not preserve the selected original decoded pixels")
    return {"original_sha256": entry["original_sha256"], "tracking_clip_sha256": source["sha256"],
            "native_fps": 50, "selected_native_frame_stride": 10, "tracking_fps": 5,
            "verified_frames": len(first), "decoded_frame_md5_list_sha256": _digest(first),
            "pixel_transform": "lossless_h264_native_yuv420p_no_crop_no_resize"}


def _reextract(entry):
    executable = Path(__file__).resolve().parents[1] / ".venv/bin/ffmpeg"
    with tempfile.TemporaryDirectory(prefix="proofloop-track-frames-") as temporary:
        subprocess.run([str(executable), "-hide_banner", "-loglevel", "error", "-nostdin", "-i",
            entry["prepared"]["clip_file"], "-map", "0:v:0", "-q:v", "2", "-pix_fmt", "yuvj444p",
            "-threads", "1", "-map_metadata", "-1", str(Path(temporary) / "frame-%03d.jpg")],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True, timeout=30)
        for frame in entry["frames"]:
            if _sha(Path(temporary) / f"frame-{frame['frame_index'] + 1:03d}.jpg") != frame["sha256"]:
                raise ValidationError("Selected source frames do not replay from their submitted clip")


def freeze(root: str | Path, annotations: str | Path, output_dir: str | Path):
    root = Path(root).resolve()
    annotations = _local(root, annotations)
    authored = _read(annotations)
    if (set(authored) != {"review_origin", "reviewed_before_inference", "clips"}
            or authored["review_origin"] != "assistant_under_user_authorization"
            or authored["reviewed_before_inference"] is not True or len(authored["clips"]) != 2):
        raise ValidationError("Freeze exactly two explicitly reviewed source clips before inference")
    entries = []
    for row in authored["clips"]:
        if set(row) != {"clip_id", "source_attribution", "original_file", "original_sha256", "clip_file",
                       "seeds", "contract", "frames_directory", "checkpoints"}:
            raise ValidationError("Experiment clip has unsupported fields")
        guard._contract(row["contract"])
        _attribution(row["source_attribution"])
        if not isinstance(row["clip_id"], str) or not re.fullmatch(r"[a-z][a-z0-9_-]{0,63}", row["clip_id"]):
            raise ValidationError("Tracking clip identifiers must be simple names")
        if (any(s["frame_index"] != 0 for s in row["seeds"])
                or sum(s["track_id"] == 1 and s["object_id"] == row["contract"]["target_id"] for s in row["seeds"]) != 1):
            raise ValidationError("Review target track one and distractors only on the first frame")
        original = _local(root, row["original_file"])
        if _sha(original) != row["original_sha256"]:
            raise ValidationError("Original licensed source changed")
        prepared = prepare_clip(_local(root, row["clip_file"]), row["seeds"],
            review_note="Initial target and nearby distractor reviewed from the first source frame; later labels are scoring only.",
            propagation_direction="forward")
        directory = _local(root, row["frames_directory"])
        frames = [{**frame, "file": str(directory / f"frame-{frame['frame_index'] + 1:03d}.jpg"),
                   "sha256": _sha(directory / f"frame-{frame['frame_index'] + 1:03d}.jpg")}
                  for frame in prepared["source"]["frames"]]
        if (len(row["checkpoints"]) != 8 or any(c["frame_index"] >= len(frames) for c in row["checkpoints"])
                or [c["frame_index"] for c in row["checkpoints"]] != sorted(c["frame_index"] for c in row["checkpoints"])):
            raise ValidationError("Freeze eight ordered checkpoints per clip")
        entry = {**deepcopy(row), "original_file": str(original), "prepared": prepared, "frames": frames}
        entry["derivation"] = _derivation(entry)
        # Validate annotations through the scorer, without using them to predict.
        references = _reference(entry)
        score_frames(references, [{"clip_id": r["clip_id"], "frame_index": r["frame_index"],
                                   "status": "unscored", "box": None} for r in references])
        _reextract(entry)
        entries.append(entry)
    if len({e["clip_id"] for e in entries}) != 2:
        raise ValidationError("Clip identifiers must be distinct")
    plan = {"experiment_version": VERSION, "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "evaluation_role": "new_episodes_in_exposed_dataset_families_development",
            "inference_started": False, "config": deepcopy(CONFIG), "methods": list(METHODS),
            "guard_config": deepcopy(guard.CONFIG), "implementation_sha256": _implementations(),
            "annotations_file": str(annotations), "annotations_sha256": _sha(annotations), "clips": entries}
    plan["freeze_hash"] = _digest(plan)
    output = _local(root, output_dir)
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    _write(output / "freeze.json", plan)
    return plan


def verify_plan(root, plan, *, files=True):
    if (plan["experiment_version"] != VERSION or plan["config"] != CONFIG or plan["methods"] != list(METHODS)
            or plan["guard_config"] != guard.CONFIG or plan["implementation_sha256"] != _implementations()
            or plan["freeze_hash"] != _digest({k: v for k, v in plan.items() if k != "freeze_hash"})
            or _sha(_local(root, plan["annotations_file"])) != plan["annotations_sha256"]):
        raise ValidationError("Frozen experiment code, configuration or annotations changed")
    authored = _read(plan["annotations_file"])
    if len(authored["clips"]) != len(plan["clips"]):
        raise ValidationError("Frozen clips differ from their pre-inference annotations")
    for declared, entry in zip(authored["clips"], plan["clips"]):
        for key, value in declared.items():
            expected = str(_local(root, value)) if key == "original_file" else value
            if entry.get(key) != expected:
                raise ValidationError("Frozen labels or contracts differ from the retained author review")
        _attribution(entry["source_attribution"])
        if entry["prepared"]["clip_file"] != str(_local(root, declared["clip_file"])):
            raise ValidationError("Frozen preparation names a different submitted clip")
        source_frames = entry["prepared"]["source"]["frames"]
        if len(entry["frames"]) != len(source_frames):
            raise ValidationError("Frozen JPEG inventory differs from the submitted source frame map")
        for index, (frame, source_frame) in enumerate(zip(entry["frames"], source_frames)):
            if ({k: frame[k] for k in ("frame_index", "pts", "timestamp_ms")} != source_frame
                    or frame["file"] != str(_local(root, declared["frames_directory"]) / f"frame-{index + 1:03d}.jpg")):
                raise ValidationError("Frozen JPEG frame indices, source times or paths changed")
    marker = Path(plan["annotations_file"]).parent / "run-started.json"
    if marker.exists() and _read(marker).get("freeze_hash") != plan["freeze_hash"]:
        raise ValidationError("Run admission marker differs from the frozen experiment")
    if files:
        for entry in plan["clips"]:
            verify_prepared(entry["prepared"])
            if _sha(_local(root, entry["original_file"])) != entry["original_sha256"]:
                raise ValidationError("Original source changed after freeze")
            if _derivation(entry) != entry["derivation"]:
                raise ValidationError("Original-to-tracking derivation changed after freeze")
            _reextract(entry)
            for frame in entry["frames"]:
                if _sha(_local(root, frame["file"])) != frame["sha256"]:
                    raise ValidationError("Experimental frame changed after freeze")


def _pair(client, paths, concepts):
    try:
        result = guided_segment_pair(client, paths, concepts)
        return {"status": "completed", "result": result, "error": None, "usage": client.usage_summary(),
                "retained_responses": deepcopy(getattr(client, "retained_responses", []))}
    except MetaAPIError as error:
        return {"status": "unscored", "result": None, "error": {"code": error.code,
                "http_status": error.status, "response": error.response}, "usage": client.usage_summary(),
                "retained_responses": deepcopy(getattr(client, "retained_responses", []))}
    except (ValidationError, OSError, ValueError, TypeError, KeyError):
        return {"status": "unscored", "result": None, "error": {"code": "local_validation_failed", "http_status": None},
                "usage": client.usage_summary(), "retained_responses": deepcopy(getattr(client, "retained_responses", []))}


def run(root: str | Path, frozen_file: str | Path):
    root = Path(root).resolve()
    frozen_file = _local(root, frozen_file)
    plan = _read(frozen_file)
    verify_plan(root, plan)
    output = frozen_file.parent
    if Path(plan["annotations_file"]).parent != output:
        raise ValidationError("Keep the frozen review and run marker in the same experiment directory")
    _write(output / "run-started.json", {"freeze_hash": plan["freeze_hash"], "automatic_retry": False,
        "started_at_utc": datetime.now(timezone.utc).isoformat()})
    load_env_file(root / ".env.local")
    journal = AttemptJournal(output / "attempts.sqlite3", request_limit=CONFIG["max_requests"])
    def client(label, limit, timeout=90):
        return RetainingClient.from_env(journal=journal, operation_id=_digest([plan["freeze_hash"], label])[:32],
            guard=lambda: verify_plan(root, plan, files=False), max_requests=limit, retries=0, timeout=timeout)
    for entry in plan["clips"]:
        name = entry["clip_id"]
        track = segment_video(client(name + "-video", 1, 180), entry["prepared"])
        _write(output / (name + "-video.json"), track)
        by_frame = {f["frame_index"]: f for f in track["frames"]}
        for checkpoint in entry["checkpoints"]:
            index = checkpoint["frame_index"]
            frame = by_frame.get(index)
            candidate = next((c for c in frame["candidates"] if c["track_id"] == 1), None) if frame else None
            if candidate:
                width, height = entry["prepared"]["source"]["width"], entry["prepared"]["source"]["height"]
                prepared = guard.prepare(entry["frames"][index]["file"], _pixels(candidate["box"], width, height),
                    entry["contract"], output / f"{name}-guard-{index}",
                    source_sha256=entry["prepared"]["source"]["sha256"], frame_index=index, track_id=1)
                assessment = guard.challenge(client(f"{name}-guard-{index}", 1), prepared)
            else:
                assessment = {"status": "not_run_no_tracker_candidate", "usage": {"request_attempts": 0, "estimated_usd": 0}}
            _write(output / f"{name}-guard-{index}.json", assessment)
        indices = [c["frame_index"] for c in entry["checkpoints"]]
        for pair in range(4):
            paths = [entry["frames"][index]["file"] for index in indices[2 * pair:2 * pair + 2]]
            report = _pair(client(f"{name}-pair-{pair}", 3), paths, {entry["contract"]["target_id"]: entry["contract"]["concept"]})
            _write(output / f"{name}-pair-{pair}.json", report)
    result = replay(root, frozen_file)
    _write(output / "public-result.json", result)
    return result


def _candidate(frame):
    return next((c for c in frame["candidates"] if c["track_id"] == 1), None)


def _verified_usage(saved, responses, *, limit):
    usage = saved["usage"]
    ledger = usage.get("requests")
    if (not isinstance(ledger, list) or not 1 <= len(ledger) <= limit or len(ledger) != len(responses)):
        raise ValidationError("Experimental request accounting lacks its exact retained attempts")
    for attempt, raw in zip(ledger, responses):
        if (attempt.get("endpoint") != "responses" or attempt.get("attempt") != 1
                or attempt.get("model") not in (SAM_MODEL, SPARK_MODEL)
                or attempt.get("status") not in ("ok", "invalid_response", "network_error", "invalid_json", "http_error")):
            raise ValidationError("Experimental accounting contains an unsupported request attempt")
        raw_usage = raw.get("usage", {}) if isinstance(raw, dict) else {}
        if not isinstance(raw_usage, dict):
            raw_usage = {}
        if attempt.get("usage", {}) != raw_usage:
            raise ValidationError("Reported usage differs from its retained provider response")
        if attempt["status"] == "ok" and (not isinstance(raw, dict) or raw.get("status") != "completed"
                                           or raw.get("model") != attempt["model"]):
            raise ValidationError("A successful attempt lacks its retained completed response")
    adapter = MetaClient("offline-replay-no-provider", max_requests=limit, retries=0)
    adapter.ledger = deepcopy(ledger)
    if adapter.usage_summary() != usage:
        raise ValidationError("Experimental cost summary differs from its retained attempts")
    return usage


def _verify_pair(saved, paths, concepts):
    _verified_usage(saved, [r["response"] for r in saved["retained_responses"]], limit=3)
    if saved["status"] != "completed":
        if saved["status"] != "unscored" or saved["error"] is None or saved["result"] is not None:
            raise ValidationError("Invalid incomplete pair record")
        return
    result = saved["result"]
    _, frames = _frames(paths)
    located = _locator_bound(result["locator"], frames, concepts)
    if (result["frames"] != frames or set(result["results"]) != {f["sha256"] for f in frames}
            or result["request_limit"] != 3
            or saved["retained_responses"][0]["response"] != result["locator"]["response"]):
        raise ValidationError("Two-view baseline coverage differs from its exact request")
    if json.loads(output_text(result["locator"]["response"])) != result["locator"]["parsed"]:
        raise ValidationError("Locator classification differs from retained response")
    retained_index = 1
    for frame, localization in zip(frames, located):
        target = localization["objects"][0]
        eligible = not target["identity_ambiguous"] and target["fully_visible"] and len(target["candidates"]) == 1
        records = result["results"][frame["sha256"]]
        if set(records) != ({target["object_id"]} if eligible else set()):
            raise ValidationError("Guided still records disagree with eligible locator candidates")
        for record in records.values():
            _sam_bound(record, frame)
            box = normalized_box_to_pixels(target["candidates"][0], frame["width"], frame["height"])
            prompt = (f"<0f>1<|box;x1={box['x1']};y1={box['y1']};x2={box['x2']};y2={box['y2']};"
                      f"w={frame['width']};h={frame['height']}|>")
            provenance = record["guided_provenance"]
            if (provenance["version"] != GUIDED_VERSION or provenance["locator_model"] != SPARK_MODEL
                    or provenance["coordinate_system"] != "normalized_0_1000"
                    or provenance["locator_normalized_box"] != target["candidates"][0]
                    or provenance["locator_box"] != box or provenance["sam_prompt"] != prompt
                    or provenance["boundary_uncertain"] != target["boundary_uncertain"]
                    or saved["retained_responses"][retained_index]["response"] != record["response"]):
                raise ValidationError("Guided SAM record differs from its original locator prompt")
            retained_index += 1
            if decode_sam_masks(parse_sam_boxes(output_text(record["response"]))) != record["objects"]:
                raise ValidationError("Guided still records do not replay")
    if retained_index != len(saved["retained_responses"]):
        raise ValidationError("Guided pair has unrelated retained responses")


def replay(root: str | Path, frozen_file: str | Path):
    root = Path(root).resolve()
    frozen_file = _local(root, frozen_file)
    plan, output = _read(frozen_file), frozen_file.parent
    verify_plan(root, plan)
    references = [r for e in plan["clips"] for r in _reference(e)]
    predictions = {method: [] for method in METHODS}
    evidence = []
    estimated_usd = 0.0
    failed_requests = 0
    observed_attempts = 0
    for entry in plan["clips"]:
        name, source = entry["clip_id"], entry["prepared"]["source"]
        raw_track = _read(output / f"{name}-video.json")
        if raw_track.get("prepared") != entry["prepared"]:
            raise ValidationError("Video response is not bound to this frozen source preparation")
        summary = public_tracking_summary(raw_track)
        estimated_usd += summary["cost"]["estimated_usd"] or 0
        failed_requests += raw_track["status"] != "completed"
        observed_attempts += len(raw_track["attempts"])
        guard_reports = {c["frame_index"]: _read(output / f"{name}-guard-{c['frame_index']}.json") for c in entry["checkpoints"]}
        runtime = []
        for tracked in raw_track["frames"]:
            index = tracked["frame_index"]
            candidate, checked = _candidate(tracked), guard_reports.get(index)
            semantic, visible, ambiguous = None, None, None
            if checked and checked["status"] in ("completed", "unscored"):
                prepared = checked["prepared"]
                guard.verify(prepared)
                if (not candidate or prepared["contract"] != entry["contract"]
                        or prepared["source_sha256"] != source["sha256"] or prepared["frame_index"] != index
                        or prepared["track_id"] != 1 or prepared["frame_sha256"] != entry["frames"][index]["sha256"]
                        or prepared["candidate_box"] != _pixels(candidate["box"], source["width"], source["height"])):
                    raise ValidationError("Semantic assessment is not bound to the actual tracked candidate")
                if checked["status"] == "completed":
                    response = checked["response"]
                    parsed = json.loads(output_text(response["response"]))
                    if (parsed != response["parsed"] or guard.evaluate(parsed, entry["contract"]) != checked["evaluation"]
                            or response["source"]["sha256"] != prepared["crop_sha256"]):
                        raise ValidationError("Semantic assessment does not replay from its response and exact crop")
                    evaluation = checked["evaluation"]
                    visible, ambiguous = evaluation["fully_visible"], evaluation["identity_ambiguous"]
                    semantic = {"target_id": entry["contract"]["target_id"], "track_id": 1, "frame_index": index,
                        "source_sha256": source["sha256"], "frame_sha256": prepared["frame_sha256"],
                        "candidate_box": candidate["box"], "decision": evaluation["decision"], "evidence_sha256": _digest(checked)}
                raw_guard = checked["response"]["response"] if checked["response"] is not None else checked["error"]["response"]
                _verified_usage(checked, [raw_guard], limit=1)
                observed_attempts += checked["usage"]["request_attempts"]
                failed_requests += checked["status"] == "unscored"
                estimated_usd += checked["usage"]["estimated_usd"]
            runtime.append({"frame_index": index, "timestamp_ms": tracked["timestamp_ms"], "source_sha256": source["sha256"],
                "frame_sha256": entry["frames"][index]["sha256"],
                "candidates": [{k: candidate[k] for k in ("track_id", "object_id", "box", "confidence", "mask_supported")}
                               | {"fully_visible": visible, "identity_ambiguous": ambiguous, "semantic": semantic}] if candidate else [],
                "reviewed_distractors": [{"track_id": c["track_id"], "box": c["box"], "origin": "runtime_sam_track"}
                                        for c in tracked["candidates"] if c["track_id"] != 1]})
        policy = evaluate_tracking(runtime, target_id=entry["contract"]["target_id"], seed={
            "target_id": entry["contract"]["target_id"], "track_id": 1, "frame_index": 0,
            "source_sha256": source["sha256"], "frame_sha256": entry["frames"][0]["sha256"],
            "review_sha256": entry["prepared"]["prepared_hash"],
            "reviewed_distractor_track_ids": [s["track_id"] for s in entry["seeds"] if s["track_id"] != 1]},
            **{k: CONFIG[k] for k in ("min_confidence", "min_stable_views", "min_spacing_ms")}) if runtime else None
        tracked_by_index = {f["frame_index"]: f for f in raw_track["frames"]}
        policy_by_index = {f["frame_index"]: f for f in policy["frames"]} if policy else {}
        for cp in entry["checkpoints"]:
            index = cp["frame_index"]
            frame = tracked_by_index.get(index)
            candidate = _candidate(frame) if frame else None
            known = bool(candidate and candidate["confidence"] is not None and candidate["confidence"] >= CONFIG["min_confidence"] and candidate["mask_supported"])
            predictions["sam_video_presence"].append({"clip_id": name, "frame_index": index,
                "status": "unscored" if not frame else "supported" if known else "unknown", "box": candidate["box"] if known else None})
            p = policy_by_index.get(index)
            status = "unscored" if not p or guard_reports[index]["status"] == "unscored" else p["status"]
            predictions["semantic_reacquisition"].append({"clip_id": name, "frame_index": index,
                "status": status, "box": p["box"] if p and status == "supported" else None})
        checkpoints = entry["checkpoints"]
        for pair in range(4):
            selected = checkpoints[pair * 2:pair * 2 + 2]
            paths = [entry["frames"][c["frame_index"]]["file"] for c in selected]
            saved = _read(output / f"{name}-pair-{pair}.json")
            _verify_pair(saved, paths, {entry["contract"]["target_id"]: entry["contract"]["concept"]})
            observed_attempts += saved["usage"]["request_attempts"]
            estimated_usd += saved["usage"]["estimated_usd"]
            failed_requests += saved["status"] == "unscored"
            for offset, cp in enumerate(selected):
                for method in METHODS[:2]:
                    status, box = "unscored", None
                    if saved["status"] == "completed":
                        result = saved["result"]
                        located = result["locator"]["parsed"]["frames"][offset]["objects"][0]
                        eligible = not located["identity_ambiguous"] and located["fully_visible"] and len(located["candidates"]) == 1
                        status = "unknown"
                        if eligible and method == "spark_two_view_locator":
                            box = _normalized(normalized_box_to_pixels(located["candidates"][0], source["width"], source["height"]), source["width"], source["height"])
                            status = "supported"
                        elif eligible and method == "guided_v2_stills":
                            record = result["results"][entry["frames"][cp["frame_index"]]["sha256"]].get(entry["contract"]["target_id"])
                            objects = record["objects"] if record else []
                            if len(objects) == 1 and objects[0]["confidence"] is not None and objects[0]["confidence"] >= CONFIG["min_confidence"] and objects[0]["mask"]["foreground_pixels"] > 0:
                                status = "supported"
                                box = _normalized(objects[0]["box"], source["width"], source["height"])
                    predictions[method].append({"clip_id": name, "frame_index": cp["frame_index"], "status": status, "box": box})
        evidence.append({"clip_id": name, "source_attribution": entry["source_attribution"],
            "tracking_clip_sha256": source["sha256"], "original_sha256": entry["original_sha256"],
            "derivation": deepcopy(entry["derivation"]),
            "initialization": "manually_reviewed_target_and_distractor_boxes_on_frame_zero_only",
            "tracking": summary, "policy": policy,
            "checkpoints": [{"frame_index": c["frame_index"], "timestamp_ms": entry["frames"][c["frame_index"]]["timestamp_ms"],
                "source_frame_sha256": entry["frames"][c["frame_index"]]["sha256"], "expected": c["expected"],
                "condition": c["condition"], "guard_status": guard_reports[c["frame_index"]]["status"]} for c in checkpoints]})
    accounting = AttemptJournal(output / "attempts.sqlite3", request_limit=CONFIG["max_requests"]).summary()
    if accounting["total_reserved_attempts"] != observed_attempts:
        raise ValidationError("Persistent attempt journal and retained experiment requests disagree")
    return {"experiment_version": VERSION, "freeze_hash": plan["freeze_hash"], "config": deepcopy(CONFIG),
        "implementation_sha256": plan["implementation_sha256"], "evaluation_role": plan["evaluation_role"],
        "annotation_origin": "assistant_visual_review_frozen_before_inference_not_dataset_ground_truth",
        "source_videos": 2, "labeled_frames": len(references), "references": references,
        "methods": {method: score_frames(references, predictions[method], iou_threshold=CONFIG["localization_iou_threshold"]) for method in METHODS},
        "localization_sensitivity": {str(t): {method: score_frames(references, predictions[method], iou_threshold=t)["metrics"] for method in METHODS}
                                     for t in CONFIG["localization_sensitivity_thresholds"]},
        "evidence": evidence, "accounting": accounting, "estimated_usd": round(estimated_usd, 6),
        "failed_operation_count": failed_requests, "changes_live_placement_acceptance": False,
        "limitations": ["New source episodes from exposed dataset families; a small development cohort, not an independent generalization benchmark.",
            "These clips show partial occlusion and re-exposure; full disappearance and true identity switches are not established.",
            "Tracking uses authored initial seeds; still baselines localize automatically. Initialization and compute differ.",
            "Four methods score candidate visibility/localization at selected frames, not placement PASS or continuous procedural correctness.",
            "The stronger semantic policy uses extra Spark calls. Report coverage beside unsupported support; abstention alone is not success.",
            "Only sampled source frames are observed; unobserved intervals and hidden state remain unverified.",
            "Reported costs are estimates, not provider-enforced spending limits. Same-model identity judgments remain correlated."]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "run", "replay"))
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--frozen", type=Path)
    args = parser.parse_args(argv)
    if args.action == "freeze":
        result = freeze(args.root, args.annotations, args.output)
        print(json.dumps({"freeze_hash": result["freeze_hash"], "inference_started": False}))
    else:
        result = run(args.root, args.frozen) if args.action == "run" else replay(args.root, args.frozen)
        print(json.dumps({"freeze_hash": result["freeze_hash"], "labeled_frames": result["labeled_frames"],
                          "estimated_usd": result["estimated_usd"], "request_attempts": result["accounting"]["total_reserved_attempts"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
