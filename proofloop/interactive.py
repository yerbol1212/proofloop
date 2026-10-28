"""Local interactive placement checks: proposals never substitute for fresh SAM evidence.

A prior owned recording is an explicitly historical starting point. Every
inspection has a one-use revision-bound token and two new images. Upload time
and spacing are operator declarations; camera timestamps are client-reported.
Neither mechanism is cryptographic proof of capture time or scene uniqueness.
"""
from __future__ import annotations

import base64
import binascii
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import struct
import threading
import time
import zlib
from typing import Any

from .core import EvidenceStore, Protocol, ValidationError, aggregate_status, evaluate
from .grounding import SamImageEvidence, ground_sam_images
from .meta import MAX_IMAGE_BYTES, MetaAPIError, MetaClient, load_env_file
from .session import _approved, _bound_result, _repair_proposal, evaluate_session, verify_capture_files

ROOT = Path(__file__).resolve().parents[1]
TOKEN_LIFETIME_MS = 20 * 60 * 1000
FRESHNESS_MS = 30000
DEFAULT_REGIONS = {"left": [0.02, 0.08, 0.48, 0.94], "right": [0.52, 0.08, 0.98, 0.94]}
MAX_PIXELS = 16_000_000
UPRIGHT_IMAGE_MESSAGE = "JPEG orientation metadata requires rotation. Export an upright PNG or JPEG with orientation removed, or use camera capture."


def _now_ms() -> int:
    return round(time.time() * 1000)


def _read_json(path: Path) -> Any:
    if path.stat().st_size > 32 * 1024 * 1024:
        raise ValidationError("Local evidence record exceeds its size limit")
    return json.loads(path.read_text(encoding="utf-8"))


def _image_dimensions(data: bytes, mime: str) -> tuple[int, int]:
    """Inspect bounded PNG/JPEG headers; full image decoding remains the provider's job."""
    if mime == "image/png":
        if not data.startswith(b"\x89PNG\r\n\x1a\n"):
            raise ValidationError("Image signature does not match its declared type")
        offset, dimensions, has_pixels, ended = 8, None, False, False
        while offset + 12 <= len(data):
            size = struct.unpack(">I", data[offset:offset + 4])[0]
            kind = data[offset + 4:offset + 8]
            if size > MAX_IMAGE_BYTES or offset + size + 12 > len(data):
                raise ValidationError("PNG image is truncated or excessive")
            chunk = data[offset + 8:offset + 8 + size]
            crc = struct.unpack(">I", data[offset + 8 + size:offset + 12 + size])[0]
            if zlib.crc32(kind + chunk) & 0xffffffff != crc:
                raise ValidationError("PNG image checksum failed")
            if offset == 8:
                if kind != b"IHDR" or size != 13:
                    raise ValidationError("PNG image header is invalid")
                dimensions = struct.unpack(">II", chunk[:8])
            if kind == b"IDAT" and size:
                has_pixels = True
            if kind == b"eXIf":
                raise ValidationError("PNG orientation metadata is not supported. Export an upright PNG without orientation metadata, or use camera capture.")
            offset += size + 12
            if kind == b"IEND":
                ended = size == 0 and offset == len(data)
                break
        if dimensions is None or not ended or not has_pixels:
            raise ValidationError("PNG image is incomplete")
    else:
        if not data.startswith(b"\xff\xd8\xff") or not data.endswith(b"\xff\xd9"):
            raise ValidationError("JPEG image is incomplete")
        offset, dimensions = 2, None
        while offset + 3 < len(data):
            if data[offset] != 255:
                raise ValidationError("JPEG image marker is invalid")
            while offset < len(data) and data[offset] == 255:
                offset += 1
            if offset >= len(data):
                break
            marker = data[offset]
            offset += 1
            if marker in (0xD8, 0xD9) or 0xD0 <= marker <= 0xD7:
                continue
            if marker == 0xDA:
                break
            if offset + 2 > len(data):
                break
            size = struct.unpack(">H", data[offset:offset + 2])[0]
            if size < 2 or offset + size > len(data):
                raise ValidationError("JPEG image segment is truncated")
            if marker == 0xE1:
                _validate_exif_orientation(data[offset + 2:offset + size])
            if marker in (0xC0, 0xC1, 0xC2):
                if size < 8:
                    raise ValidationError("JPEG image dimensions are missing")
                height, width = struct.unpack(">HH", data[offset + 3:offset + 7])
                dimensions = (width, height)
            offset += size
        if dimensions is None:
            raise ValidationError("Use a baseline or progressive JPEG image")
    width, height = dimensions
    if min(width, height) < 16 or width * height > MAX_PIXELS:
        raise ValidationError("Image dimensions must be at least 16 pixels and at most 16 megapixels")
    return width, height


def _validate_exif_orientation(segment: bytes) -> None:
    if not segment.startswith(b"Exif\0\0"):
        return
    tiff = segment[6:]
    if len(tiff) < 8 or tiff[:2] not in (b"II", b"MM"):
        raise ValidationError("JPEG orientation metadata is invalid")
    endian = "<" if tiff[:2] == b"II" else ">"
    if struct.unpack(endian + "H", tiff[2:4])[0] != 42:
        raise ValidationError("JPEG orientation metadata is invalid")
    offset = struct.unpack(endian + "I", tiff[4:8])[0]
    if offset < 8 or offset + 2 > len(tiff):
        raise ValidationError("JPEG orientation metadata is invalid")
    count = struct.unpack(endian + "H", tiff[offset:offset + 2])[0]
    if offset + 2 + count * 12 > len(tiff):
        raise ValidationError("JPEG orientation metadata is invalid")
    for index in range(count):
        entry = tiff[offset + 2 + index * 12:offset + 14 + index * 12]
        tag, kind, length = struct.unpack(endian + "HHI", entry[:8])
        if tag == 0x0112:
            if kind != 3 or length != 1:
                raise ValidationError("JPEG orientation metadata is invalid")
            if struct.unpack(endian + "H", entry[8:10])[0] != 1:
                raise ValidationError(UPRIGHT_IMAGE_MESSAGE)


def _decode_image(item: Any) -> tuple[bytes, str, int, int]:
    if not isinstance(item, dict) or not isinstance(item.get("data_url"), str):
        raise ValidationError("Supply two encoded PNG or JPEG images")
    value = item["data_url"]
    if len(value) > 4 * ((MAX_IMAGE_BYTES + 2) // 3) + 64:
        raise ValidationError("Each encoded image must be at most 5 MiB")
    match = re.fullmatch(r"data:(image/(?:jpeg|png));base64,([A-Za-z0-9+/=]+)", value)
    if not match:
        raise ValidationError("Supply PNG or JPEG images as base64 data URLs")
    try:
        data = base64.b64decode(match[2], validate=True)
    except (ValueError, binascii.Error):
        raise ValidationError("Image encoding is invalid") from None
    if not data or len(data) > MAX_IMAGE_BYTES:
        raise ValidationError("Each image must contain between one byte and 5 MiB")
    width, height = _image_dimensions(data, match[1])
    return data, match[1], width, height


class InteractiveEngine:
    """Thread-safe local engine. Network work runs outside the state lock.

    ``client_factory`` is a zero-argument injection point for offline tests. The
    default creates one fresh four-request Meta client with no automatic retry.
    No injected or cached result is relabeled as a verified live API response.
    """

    def __init__(self, root: Path = ROOT, seed_report: Path | None = None, client_factory=None):
        self.root = Path(root).resolve()
        self._lock = threading.RLock()
        self._client_factory = client_factory
        self._media: dict[str, Path] = {}
        self._media_hashes: dict[str, str] = {}
        self._seen_hashes: set[str] = set()
        self._token: str | None = None
        self._token_issued_ms: int | None = None
        self._token_revision: int | None = None
        self._protocol = Protocol.from_dict({"id": "interactive-cutlery", "version": "1",
            "objects": ["fork", "spoon"], "regions": ["left", "right"],
            "checks": [{"id": "fork-left", "kind": "location", "object_id": "fork", "region_id": "left",
                        "unique_target": True, "max_age_ms": FRESHNESS_MS},
                       {"id": "spoon-right", "kind": "location", "object_id": "spoon", "region_id": "right",
                        "unique_target": True, "max_age_ms": FRESHNESS_MS}]})
        self._concepts = {"fork": "fork", "spoon": "spoon"}
        session_id = "interactive-" + secrets.token_hex(12)
        self._directory = self.root / "artifacts/local" / session_id
        self._directory.mkdir(parents=True, mode=0o700)
        self._state: dict[str, Any] = {
            "session_id": session_id, "phase": "needs_evidence", "status": "unknown", "revision": 0,
            "source_kind": "none", "images": [], "checks": [], "proposal": None,
            "regions": deepcopy(DEFAULT_REGIONS), "capture_token": None, "trace": [],
            "busy": False, "error": None, "verified_at_ms": None, "evidence_received_at_ms": None,
            "evidence_observed_at_ms": None, "evidence_age_basis": None,
            "source_note": "Approve the regions and unique targets, then inspect two fresh views.",
            "inference_origin": None,
            "usage": None,
        }
        # Exact old bytes remain blocked after a local server restart. This is
        # an accidental-reuse guard, not a claim of replay-proof capture.
        for capture_file in (self.root / "artifacts/local").glob("interactive-*/inspection-*/capture.json"):
            try:
                hashes = _read_json(capture_file).get("source_sha256", [])
                if isinstance(hashes, list):
                    self._seen_hashes.update(value for value in hashes if isinstance(value, str)
                                             and re.fullmatch(r"[0-9a-f]{64}", value))
            except (OSError, ValueError, TypeError, AttributeError):
                pass
        self._unknown_checks("No evidence has been inspected in this session")
        candidate = Path(seed_report) if seed_report is not None else self.root / "artifacts/local/phone-session-20261028-01/report.json"
        if candidate.is_file():
            try:
                self._load_seed(candidate)
            except (OSError, ValueError, TypeError, KeyError):
                self._state["error"] = "The recorded reference could not be verified. Capture a fresh inspection instead."
                self._trace("seed_unavailable", "Recorded reference could not be verified", "unknown")
        self._persist()

    def _write(self, path: Path, value: Any) -> None:
        # All writes are private; no raw source or model response is public.
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")

    def _persist(self) -> None:
        self._write(self._directory / "state.json", self._state)

    def _trace(self, event: str, label: str, status: str) -> None:
        self._state["trace"].append({"id": secrets.token_hex(6), "event": event, "label": label,
                                      "status": status, "revision": self._state["revision"]})
        self._state["trace"] = self._state["trace"][-100:]

    def _unknown_checks(self, reason: str) -> None:
        self._state["checks"] = [{"check_id": check.id, "object_id": check.object_id,
            "status": "unknown", "expected_region": check.region_id, "observed_region": None,
            "reason": reason, "confidence": None} for check in self._protocol.checks]

    def _public_checks(self, snapshot: dict[str, Any], observations) -> list[dict[str, Any]]:
        observations = {item.object_id: item for item in observations}
        results = {item["check_id"]: item for item in snapshot["checks"]}
        return [{"check_id": check.id, "object_id": check.object_id,
                 "status": results[check.id]["status"], "expected_region": check.region_id,
                 "observed_region": observations[check.object_id].value,
                 "reason": results[check.id]["reason"],
                 "confidence": observations[check.object_id].confidence} for check in self._protocol.checks]

    def _register_media(self, path: Path, width: int, height: int, objects=()) -> dict[str, Any]:
        media_id = secrets.token_hex(16)
        self._media[media_id] = path
        self._media_hashes[media_id] = hashlib.sha256(path.read_bytes()).hexdigest()
        boxes = []
        for object_id, result in objects:
            for detection in result.get("objects", []):
                box = detection.get("box", {})
                if all(type(box.get(key)) is int for key in ("x1", "y1", "x2", "y2", "width", "height")):
                    if (box["width"], box["height"]) == (width, height):
                        x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
                        if 0 <= x1 <= x2 < width and 0 <= y1 <= y2 < height:
                            boxes.append({"object_id": object_id, "x1": x1 / width, "y1": y1 / height,
                                          "x2": (x2 + 1) / width, "y2": (y2 + 1) / height})
        return {"id": media_id, "url": "/api/media/" + media_id, "width": width, "height": height, "boxes": boxes}

    def _proposal(self, observations, snapshot: dict[str, Any]) -> dict[str, Any] | None:
        if snapshot["status"] != "fail":
            return None
        planned = _repair_proposal(self._protocol, tuple(observations), dict(snapshot, stage_id="interactive"))
        if planned.get("request_inspection"):
            return {"title": "Inspect the targets before planning a move", "steps": ["Capture two clear views of both targets."],
                    "hypothetical": True}
        steps = []
        for action in planned["plan"]["action_ids"]:
            if action.startswith("move-"):
                _, obj, source, _, target = action.split("-")
                steps.append(f"Move the {obj} from {source.upper()} to {target.upper()}.")
        steps.append("Keep both targets visible and hands clear, then capture two fresh views at least one second apart.")
        return {"title": "Return each target to its approved region", "steps": steps, "hypothetical": True,
                "action_ids": planned["plan"]["action_ids"], "execution_verified": False}

    def _load_seed(self, report_path: Path) -> None:
        allowed = (self.root / "artifacts/local").resolve()
        if allowed not in report_path.resolve().parents:
            raise ValidationError("Use a recorded reference inside the private artifact directory")
        report = _read_json(report_path)
        if (report.get("inference_origin") != "live_meta_api_on_recorded_frames"
                or report.get("failures") or report.get("evidence_origin") != "owned_phone"):
            raise ValidationError("Recorded reference has no completed owned-phone live result")
        envelope = _read_json(report_path.parent / "approval.json")
        protocol = _approved(envelope)
        if (set(protocol.objects) != {"fork", "spoon"} or set(protocol.regions) != {"left", "right"}
                or {check.object_id: check.region_id for check in protocol.checks} != {"fork": "left", "spoon": "right"}):
            raise ValidationError("The reference must use the approved fork and spoon task")
        files = verify_capture_files(envelope, project_root=self.root)
        contract = envelope["contract"]
        stage = next(item for item in contract["capture"]["stages"] if item["id"] == "wrong_region")
        results = {"wrong_region": {}}
        for index, frame in enumerate(stage["frames"]):
            by_object = results["wrong_region"][frame["sha256"]] = {}
            for object_index, object_id in enumerate(protocol.objects):
                result = _read_json(report_path.parent / f"wrong_region-{index}-{object_index}.json")
                _bound_result(result, frame["sha256"])
                by_object[object_id] = result
        recomputed = evaluate_session(envelope, results)
        wrong = next(item for item in recomputed["trace"] if item["event"] == "stage" and item["stage_id"] == "wrong_region")
        original = next(item for item in report["trace"] if item["event"] == "stage" and item["stage_id"] == "wrong_region")
        if wrong != original or wrong["status"] != "fail" or report["contract_hash"] != envelope["contract_hash"]:
            raise ValidationError("Recorded reference no longer matches the bound inference")
        self._protocol, self._concepts = protocol, dict(contract["concepts"])
        from .core import Observation
        observations = tuple(Observation.from_dict(item) for item in wrong["observations"])
        images = [self._register_media(files[frame["sha256"]], frame["width"], frame["height"],
                  results["wrong_region"][frame["sha256"]].items()) for frame in stage["frames"]]
        self._seen_hashes.update(files)
        self._state.update(phase="mistake", status="fail", source_kind="recorded", images=images,
            regions=deepcopy(contract["regions"]), checks=self._public_checks(wrong, observations),
            proposal=self._proposal(observations, wrong),
            source_note="Historical recorded evidence: the fork was on RIGHT. Start repair before capturing the current scene.",
            inference_origin="replayed_hash_bound_live_meta_api_on_recorded_frames")
        self._trace("recorded_mistake", "Recorded fork on RIGHT; required LEFT", "fail")

    def _api_ready(self) -> bool:
        if os.environ.get("META_API_KEY", "").strip():
            return True
        try:
            for line in (self.root / ".env.local").read_text(encoding="utf-8").splitlines():
                key, separator, value = line.partition("=")
                if separator and key.strip() == "META_API_KEY" and value.strip().strip("\"'"):
                    return True
        except (OSError, UnicodeError):
            pass
        return False

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            state = deepcopy(self._state)
            evidence_time = state["evidence_observed_at_ms"] if state["source_kind"] == "camera" else state["evidence_received_at_ms"]
            age = None if evidence_time is None else max(0, _now_ms() - evidence_time)
            state.update(api_ready=self._api_ready(), evidence_age_ms=age,
                         stale=state["source_kind"] == "recorded" or age is not None and age > FRESHNESS_MS,
                         pending_token_issued_at_ms=self._token_issued_ms)
            if state["stale"] and state["phase"] == "verified":
                state["phase"], state["status"] = "unknown", "unknown"
                state["source_note"] = "The earlier capture passed, but is now older than 30 seconds. Inspect again after any movement."
                for check in state["checks"]:
                    check.update(status="unknown", reason="The earlier captured evidence is older than 30 seconds")
            return state

    def _invalidate(self, event: str) -> dict[str, Any]:
        with self._lock:
            if self._state["busy"]:
                raise ValidationError("An inspection is already running")
            self._state["revision"] += 1
            self._state.update(phase="awaiting_capture", status="unknown", error=None, proposal=None,
                               verified_at_ms=None, usage=None,
                               source_note="Prior images are reference only. Current evidence was invalidated; capture two new views.")
            self._unknown_checks("Previous evidence was invalidated; awaiting two fresh views")
            self._token = secrets.token_urlsafe(32)
            self._token_issued_ms, self._token_revision = _now_ms(), self._state["revision"]
            self._state["capture_token"] = self._token
            self._trace(event, "Evidence invalidated; a new capture is required", "unknown")
            self._persist()
            return self.snapshot()

    def begin_repair(self) -> dict[str, Any]:
        return self._invalidate("repair_started")

    def new_inspection(self) -> dict[str, Any]:
        return self._invalidate("new_inspection")

    def media_path(self, media_id: str) -> Path:
        with self._lock:
            if not isinstance(media_id, str) or not re.fullmatch(r"[0-9a-f]{32}", media_id) or media_id not in self._media:
                raise ValidationError("Unknown local evidence image")
            path = self._media[media_id]
            resolved = path.resolve()
            allowed = (self._directory.resolve(), (self.root / "inputs/local/phone").resolve())
            if (not path.is_file() or not any(directory in resolved.parents for directory in allowed)
                    or hashlib.sha256(path.read_bytes()).hexdigest() != self._media_hashes[media_id]):
                raise ValidationError("Local evidence image is unavailable")
            return path

    def _validate_pair(self, payload: Any):
        if not isinstance(payload, dict):
            raise ValidationError("An inspection payload is required")
        supplied = payload.get("capture_token")
        if (not self._token or not isinstance(supplied, str) or not secrets.compare_digest(supplied, self._token)
                or self._token_revision != self._state["revision"] or self._token_issued_ms is None
                or not 0 <= _now_ms() - self._token_issued_ms <= TOKEN_LIFETIME_MS):
            raise ValidationError("Start a new inspection to obtain a current capture token")
        if any(payload.get(name) is not True for name in (
            "confirm_regions", "confirm_unique_targets", "confirm_fresh_capture", "confirm_pair_spacing")):
            raise ValidationError("Confirm regions, unique targets, fresh captures and one-second spacing")
        source = payload.get("source")
        if source not in ("camera", "upload"):
            raise ValidationError("Select camera capture or fresh photo upload")
        regions = payload.get("regions")
        if not isinstance(regions, dict) or set(regions) != {"left", "right"}:
            raise ValidationError("Approve both nonoverlapping region rectangles")
        ground_sam_images(self._protocol, regions, [], session_id=self._state["session_id"])
        items = payload.get("images")
        if not isinstance(items, list) or len(items) != 2:
            raise ValidationError("Supply exactly two fresh images")
        decoded = [_decode_image(item) for item in items]
        hashes = [hashlib.sha256(item[0]).hexdigest() for item in decoded]
        if len(set(hashes)) != 2 or any(value in self._seen_hashes for value in hashes):
            raise ValidationError("Repeated or previously inspected images cannot verify a new correction")
        if decoded[0][2:] != decoded[1][2:]:
            raise ValidationError("Both images must have the same dimensions and camera orientation")
        if source == "camera":
            times = [item.get("captured_at_ms") for item in items]
            if (any(type(value) is not int for value in times) or times[0] < self._token_issued_ms
                    or not 1000 <= times[1] - times[0] <= FRESHNESS_MS
                    or times[1] > _now_ms() + 5000 or _now_ms() - times[0] > FRESHNESS_MS):
                raise ValidationError("Camera captures must follow this inspection and be one second apart and recent")
            basis = "client_reported_camera_epoch_ms_not_independently_verified"
        else:
            times = [0, 1000]
            basis = "operator_declared_freshness_and_spacing; logical_order_only_not_measured_capture_time"
        return source, deepcopy(regions), decoded, hashes, times, basis

    def inspect_pair(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            if self._state["busy"]:
                raise ValidationError("An inspection is already running")
            source, regions, decoded, hashes, times, basis = self._validate_pair(payload)
            revision = self._state["revision"]
            issued_at = self._token_issued_ms
            self._token = None
            self._state["capture_token"] = None
            self._seen_hashes.update(hashes)
            self._state.update(phase="checking", status="unknown", busy=True, error=None, source_kind=source,
                               proposal=None, verified_at_ms=None, regions=regions, images=[], usage=None,
                               source_note="Inspecting two new views with Meta SAM. The correction is not yet verified.",
                               inference_origin="live_meta_api" if self._client_factory is None else "injected_client_not_verified_live",
                               evidence_received_at_ms=_now_ms(),
                               evidence_observed_at_ms=times[0] if source == "camera" else None,
                               evidence_age_basis="client_reported_capture_time" if source == "camera" else "server_received_time_capture_age_operator_declared")
            self._unknown_checks("Fresh image inspection is in progress")
            run_directory = self._directory / ("inspection-" + str(revision))
            paths = []
            try:
                run_directory.mkdir(mode=0o700)
                for index, (data, mime, width, height) in enumerate(decoded):
                    path = run_directory / f"view-{index}.{'png' if mime == 'image/png' else 'jpg'}"
                    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                    with os.fdopen(descriptor, "wb") as stream:
                        stream.write(data)
                    paths.append(path)
                    self._state["images"].append(self._register_media(path, width, height))
                self._write(run_directory / "capture.json", {"source": source, "revision": revision,
                    "regions": regions, "source_sha256": hashes, "timestamp_basis": basis,
                    "source_timestamps_ms": times, "token_issued_at_ms": issued_at,
                    "operator_confirmations": {key: True for key in ("regions", "unique_targets", "fresh_capture", "pair_spacing")}})
                self._trace("inspection_started", "Two fresh views submitted; up to four SAM calls", "unknown")
                self._persist()
            except OSError:
                self._state.update(phase="error", status="unknown", busy=False,
                                   error="Local evidence could not be saved. Start a new inspection.")
                return self.snapshot()
        client, results, grounded = None, [], None
        try:
            if self._client_factory is None:
                load_env_file(self.root / ".env.local")
                client = MetaClient.from_env(max_requests=4, retries=0, timeout=25)
            else:
                client = self._client_factory()
            if client.max_requests > 4 or client.retries != 0 or client.ledger:
                raise ValidationError("Use a fresh bounded client with no retries")
            samples = []
            for index, path in enumerate(paths):
                frame_results = []
                for object_id in self._protocol.objects:
                    result = client.segment_image(path, self._concepts[object_id], decode_masks=True)
                    _bound_result(result, hashes[index])
                    if any((item.get("box", {}).get("width"), item.get("box", {}).get("height")) != decoded[index][2:]
                           for item in result.get("objects", [])):
                        raise ValidationError("SAM geometry does not match the exact supplied image")
                    self._write(run_directory / f"result-{index}-{object_id}.json", result)
                    samples.append(SamImageEvidence(object_id, times[index], result,
                                   "operator_declared_owned_" + source, self._concepts[object_id]))
                    frame_results.append((object_id, result))
                results.append(frame_results)
            if any(hashlib.sha256(path.read_bytes()).hexdigest() != hashes[index] for index, path in enumerate(paths)):
                raise ValidationError("Supplied image changed during inspection")
            store = EvidenceStore(self._protocol, self._state["session_id"])
            store.revision = revision
            grounded = ground_sam_images(self._protocol, regions, samples, session_id=store.session_id, revision=revision)
            for observation in grounded.observations:
                store.add(observation)
            now = _now_ms() if source == "camera" else 1000
            verdicts = evaluate(self._protocol, store, now_ms=now)
            outcome = {"status": aggregate_status(verdicts).value, "checks": [item.to_dict() for item in verdicts]}
            self._write(run_directory / "evaluation.json", {"timestamp_basis": basis, "grounding": grounded.to_dict(),
                         "evaluation": outcome, "usage": client.usage_summary()})
            with self._lock:
                if self._state["revision"] != revision:
                    raise ValidationError("Inspection revision changed; results cannot update this session")
                status = outcome["status"]
                self._state.update(status=status, phase={"pass": "verified", "fail": "mistake", "unknown": "unknown"}[status],
                    checks=self._public_checks(outcome, grounded.observations),
                    proposal=self._proposal(grounded.observations, outcome),
                    verified_at_ms=_now_ms() if status == "pass" else None,
                    source_note=("Captured placement passed. Inspect again after any movement. " if status == "pass" else
                                 "Fresh evidence supports a misplaced target. " if status == "fail" else
                                 "The views do not support every target location. Reveal both targets and inspect again. ") +
                                ("Camera capture time is client-reported." if source == "camera" else
                                 "Photo freshness and one-second spacing were declared by the operator; capture time was not measured."))
                self._state["images"] = [self._register_media(path, decoded[index][2], decoded[index][3], results[index])
                                         for index, path in enumerate(paths)]
                self._trace("inspection_complete", {"pass": "Fresh captured placement verified", "fail": "Mistake observed in fresh captures",
                                                   "unknown": "Insufficient visual evidence; no pass issued"}[status], status)
                self._state["usage"] = client.usage_summary()
        except (MetaAPIError, OSError, ValueError, TypeError, KeyError, AttributeError) as error:
            with self._lock:
                self._state.update(phase="error", status="unknown", verified_at_ms=None, proposal=None,
                    error="The Meta inspection could not complete. No correction was verified. Start a new inspection and capture new views.",
                    source_note="Inspection failed; no correction was verified. Start a new inspection with two new views.",
                    usage=client.usage_summary() if client is not None else None)
                self._unknown_checks("Inspection failed; no supported current result")
                self._trace("inspection_failed", "Inspection stopped without a verified result", "unknown")
                try:
                    self._write(run_directory / "failure.json", {"error_type": type(error).__name__,
                                 "response": error.response if isinstance(error, MetaAPIError) else None,
                                 "usage": client.usage_summary() if client is not None else None})
                except OSError:
                    pass
        finally:
            with self._lock:
                self._state["busy"] = False
                try:
                    self._persist()
                except OSError:
                    self._state.update(phase="error", status="unknown", verified_at_ms=None,
                                       error="The inspection state could not be saved; inspect again.")
        return self.snapshot()
