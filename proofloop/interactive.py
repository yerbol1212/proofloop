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

from .acquisition import build_acquisition_plan, check_source_pair
from .compiler import COFFEE_PLACEMENT_PRESET, CompilerError, authored_coffee_protocol, compile_protocol
from .concepts import validate_concept_mapping
from .core import EvidenceStore, Protocol, ValidationError, aggregate_status, evaluate
from .diagnostics import explain_checks
from .grounding import SamImageEvidence, ground_sam_images
from .guided import GUIDED_VERSION, guided_segment_pair
from .meta import MAX_IMAGE_BYTES, MetaAPIError, MetaClient, load_env_file
from .recovery import digest, evidence_bundle, replay_evidence, save_receipt, task_contract, verify_recovery_receipt
from .session import _approved, _bound_result, _repair_proposal, evaluate_session, verify_capture_files
from .workflow import validate_continuation, workflow_status

ROOT = Path(__file__).resolve().parents[1]
TOKEN_LIFETIME_MS = 20 * 60 * 1000
FRESHNESS_MS = 30000
DEFAULT_REGIONS = {"left": [0.02, 0.08, 0.48, 0.94], "right": [0.52, 0.08, 0.98, 0.94]}
MAX_PIXELS = 16_000_000
MAX_TASK_OBJECTS = 6
PERCEPTION_NOTES = {
    "concept": "SAM detects reviewed text concepts; reported confidence is not a calibrated correctness probability.",
    "guided": "Spark assigns object identity from the two images; SAM refines its prompted boundaries and does not independently verify identity. Both views must support full target visibility; ambiguous identity or unsupported visibility stays unknown.",
}
UPRIGHT_IMAGE_MESSAGE = "JPEG orientation metadata requires rotation. Export an upright PNG or JPEG with orientation removed, or use camera capture."


def _perception_details(mode: str) -> dict[str, Any]:
    details = {"perception_mode": mode, "perception_note": PERCEPTION_NOTES[mode]}
    if mode == "guided":
        details.update(perception_version=GUIDED_VERSION,
                       perception_note=f"Guided method version: {GUIDED_VERSION}. " + details["perception_note"])
    return details


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
    default creates one fresh bounded Meta client with no automatic retry.
    No injected or cached result is relabeled as a verified live API response.
    """

    def __init__(self, root: Path = ROOT, seed_report: Path | None = None, client_factory=None,
                 compile_client_factory=None):
        self.root = Path(root).resolve()
        self._lock = threading.RLock()
        self._client_factory = client_factory
        self._compile_client_factory = compile_client_factory
        self._draft_record: dict[str, Any] | None = None
        self._evidence_record: dict[str, Any] | None = None
        self._workflow_replay = None
        self._reset_acquisition()
        self._recovery_record: dict[str, Any] | None = None
        self._recovery_path: Path | None = None
        self._recovery_sequence = 0
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
            "busy": False, "busy_operation": None, "error": None, "verified_at_ms": None, "evidence_received_at_ms": None,
            "evidence_observed_at_ms": None, "evidence_age_basis": None,
            "source_note": "Approve the regions and unique targets, then inspect two fresh views.",
            "inference_origin": None,
            "usage": None,
            "perception_provenance": None,
            "active_task": self._task_metadata("Fork and spoon placement", "Place the fork in LEFT and the spoon in RIGHT.",
                                               "authored_default", reviewed=False),
            "task_draft": None, "draft_error": None,
            "recovery": None,
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

    def _task_metadata(self, title: str, requirements: str, origin: str, *, reviewed: bool,
                       review_origin: str | None = None, model: str | None = None,
                       perception_mode: str = "concept") -> dict[str, Any]:
        return {"title": title, "objects": list(self._protocol.objects), "regions": list(self._protocol.regions), "requirements": requirements,
                "protocol_hash": self._protocol.content_hash, "protocol": self._protocol.to_dict(),
                "concepts": dict(self._concepts), "origin": origin, "reviewed": reviewed,
                "human_reviewed": reviewed and review_origin == "local_operator", "review_origin": review_origin,
                "model": model, "task_hash": None, **_perception_details(perception_mode)}

    @staticmethod
    def _validate_task_input(payload: Any) -> tuple[str, str, list[str], dict[str, str], str, list[str], str]:
        if not isinstance(payload, dict):
            raise ValidationError("Supply task requirements and an explicit object inventory")
        allowed = {"requirements", "objects", "title", "confirm_unique_targets", "concepts", "perception_mode", "regions", "draft_source"}
        if set(payload) - allowed:
            raise ValidationError("Task draft contains unsupported fields")
        text, objects = payload.get("requirements"), payload.get("objects")
        if not isinstance(text, str) or not text.strip() or len(text.encode("utf-8")) > 8192:
            raise ValidationError("Task requirements must contain 1 to 8192 UTF-8 bytes")
        if (not isinstance(objects, list) or not 1 <= len(objects) <= MAX_TASK_OBJECTS
                or any(not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", value) for value in objects)
                or len(set(objects)) != len(objects)):
            raise ValidationError("Supply 1 to 6 distinct object IDs using letters, digits, underscores, dots or hyphens")
        regions = payload.get("regions", ["left", "right"])
        if (not isinstance(regions, list) or len(regions) != 2
                or any(not isinstance(value, str) or not re.fullmatch(r"[a-z0-9][a-z0-9_.-]{0,63}", value) for value in regions)
                or len(set(regions)) != 2):
            raise ValidationError("Supply exactly two distinct lowercase region IDs using letters, digits, underscores, dots or hyphens")
        title = payload.get("title", "Custom placement task")
        if not isinstance(title, str) or not title.strip() or len(title) > 120:
            raise ValidationError("Task title must contain 1 to 120 characters")
        if payload.get("confirm_unique_targets") is not True:
            raise ValidationError("Confirm one visually distinct instance of every target before drafting")
        if "concepts" in payload and payload["concepts"] is None:
            raise ValidationError("SAM prompt mapping must identify every task object exactly once")
        concepts = validate_concept_mapping(objects, payload.get("concepts"))
        mode = payload.get("perception_mode", "concept")
        if not isinstance(mode, str) or mode not in PERCEPTION_NOTES:
            raise ValidationError("Select concept or guided perception mode before drafting")
        if mode == "guided" and any("." in value for value in objects):
            raise ValidationError("Guided perception requires object IDs without dots")
        source = payload.get("draft_source", "spark")
        if source not in ("spark", "authored_coffee_preset"):
            raise ValidationError("Select Spark compilation or the authored coffee placement preset")
        if source == "authored_coffee_preset":
            supplied = {"title": title.strip(), "requirements": text.strip(), "objects": objects,
                        "regions": regions, "concepts": concepts, "perception_mode": mode}
            if supplied != COFFEE_PLACEMENT_PRESET:
                raise ValidationError("The authored coffee preset is fixed. Use Spark to draft changed requirements")
        return title.strip(), text.strip(), list(objects), concepts, mode, list(regions), source

    def draft_task(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Compile a proposal without changing the active task or its evidence.

        Uniqueness is an intended operator-declared scene contract, never an
        inference from a detector. A separate exact-draft review activates it.
        """
        with self._lock:
            if self._state["busy"]:
                raise ValidationError("An inspection is already running")
            title, requirements, objects, concepts, mode, regions, source = self._validate_task_input(payload)
            self._state.update(busy=True, busy_operation="task_draft", draft_error=None, task_draft=None)
            self._draft_record = None
            draft_id = secrets.token_hex(16)
        client = None
        try:
            self._persist()
            if source == "authored_coffee_preset":
                envelope = authored_coffee_protocol()
            else:
                if self._compile_client_factory is None:
                    load_env_file(self.root / ".env.local")
                    client = MetaClient.from_env(max_requests=1, retries=0, timeout=45)
                else:
                    client = self._compile_client_factory()
                if client.max_requests != 1 or client.retries != 0 or client.ledger:
                    raise ValidationError("Use a fresh one-request compiler client with no retries")
                envelope = compile_protocol(client, requirements, objects=objects, regions=regions,
                                            predicate_kind="location", unique_objects=objects)
            raw = deepcopy(envelope["protocol"])
            # A repeated semantic task is still a new reviewed specification;
            # evidence cannot silently migrate between edit/review versions.
            raw["version"] = "draft-" + draft_id[:16]
            protocol = Protocol.from_dict(raw)
            if (set(protocol.objects) != set(objects) or list(protocol.regions) != regions
                    or not protocol.checks or any(check.kind != "location" or not check.unique_target for check in protocol.checks)):
                raise CompilerError("The task must contain only declared-unique placement checks in the two named regions")
            origin = ("authored_frozen_coffee_preset" if source == "authored_coffee_preset" else
                      "live_meta_api" if self._compile_client_factory is None else "injected_compiler_not_verified_live")
            draft = {"draft_id": draft_id, "title": title, "requirements": requirements, "objects": objects, "regions": regions,
                     "protocol": protocol.to_dict(), "protocol_hash": protocol.content_hash,
                     "concepts": concepts,
                     "origin": origin, "model": envelope["model"], "requires_human_approval": True,
                     "approved": False, "claim_limit": envelope["claim_limit"],
                     **_perception_details(mode)}
            if source == "authored_coffee_preset":
                draft.update(preset_id="coffee-placement-v1", preset_protocol_hash=envelope["content_hash"],
                             suggested_regions=envelope["suggested_regions"],
                             calibration_note="Starting rectangles match the fixed coffee development videos. Review and adjust both named regions against your actual view before inspection.")
            draft["draft_hash"] = hashlib.sha256(json.dumps(draft, sort_keys=True, separators=(",", ":"),
                                                              allow_nan=False).encode()).hexdigest()
            record = {"draft": draft, "compiler_envelope": envelope, "usage": client.usage_summary() if client is not None else
                      {"request_attempts": 0, "estimated_usd": 0, "estimate_origin": "authored_preset_no_inference"},
                      "created_at_ms": _now_ms(), "unique_target_contract_origin": "caller_declared_before_compilation"}
            self._write(self._directory / ("task-draft-" + draft_id + ".json"), record)
            with self._lock:
                self._draft_record = record
                self._state["task_draft"] = draft
                self._trace("task_drafted", "Authored coffee preset awaits review; active task unchanged" if source == "authored_coffee_preset" else
                            "Spark task draft awaits review; active task unchanged", self._state["status"])
        except (MetaAPIError, OSError, ValueError, TypeError, KeyError, AttributeError) as error:
            with self._lock:
                message = ("Spark could not produce a supported placement draft. Use explicit named-region requirements; "
                           "count, absence, temporal and quality requirements are not supported here.") if isinstance(error, CompilerError) else (
                           "The Spark task draft could not complete. The active task and evidence are unchanged. Try again.")
                self._state["draft_error"] = message
                try:
                    self._write(self._directory / ("task-draft-failure-" + draft_id + ".json"),
                                {"error_type": type(error).__name__,
                                 "response": error.response if isinstance(error, MetaAPIError) else None,
                                 "usage": client.usage_summary() if client is not None else None})
                except OSError:
                    pass
        finally:
            with self._lock:
                self._state.update(busy=False, busy_operation=None)
                try:
                    self._persist()
                except OSError:
                    self._state["draft_error"] = "The task draft could not be saved. Retry before activating it."
                    self._state["task_draft"], self._draft_record = None, None
        return self.snapshot()

    def activate_task(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Freeze an exact reviewed draft and discard all prior observations."""
        with self._lock:
            if self._state["busy"]:
                raise ValidationError("An inspection is already running")
            if (not isinstance(payload, dict) or set(payload) - {"draft_id", "draft_hash", "confirm_requirements",
                    "confirm_unique_targets", "review_origin", "authorization_reference"}):
                raise ValidationError("Supply the exact draft ID, hash and review confirmations")
            draft = self._state["task_draft"]
            if (not draft or self._draft_record is None or payload.get("draft_id") != draft["draft_id"]
                    or payload.get("draft_hash") != draft["draft_hash"]):
                raise ValidationError("Review the current task draft before activating it")
            if draft["perception_mode"] == "guided" and draft.get("perception_version") != GUIDED_VERSION:
                raise ValidationError("Guided perception changed. Request and review a new task draft before activation")
            if digest({key: value for key, value in draft.items() if key != "draft_hash"}) != draft["draft_hash"]:
                raise ValidationError("The task draft changed. Request and review a new exact draft before activation")
            if payload.get("confirm_requirements") is not True or payload.get("confirm_unique_targets") is not True:
                raise ValidationError("Confirm the exact requirements and every unique target before activation")
            origin = payload.get("review_origin", "local_operator")
            reference = payload.get("authorization_reference")
            if (origin not in ("local_operator", "assistant_under_user_authorization")
                    or origin == "assistant_under_user_authorization" and
                    (not isinstance(reference, str) or not reference.strip() or len(reference) > 4096)):
                raise ValidationError("Assistant review requires a direct user authorization reference")
            protocol = Protocol.from_dict(draft["protocol"])
            if list(protocol.regions) != draft["regions"] or protocol.content_hash != draft["protocol_hash"]:
                raise ValidationError("The reviewed region inventory does not match its executable protocol")
            approval = {"draft_id": draft["draft_id"], "draft_hash": draft["draft_hash"],
                        "protocol_hash": protocol.content_hash, "review_origin": origin,
                        "concepts": dict(draft["concepts"]),
                        "regions": list(draft["regions"]),
                        "perception_mode": draft["perception_mode"],
                        "human_reviewed_exact_contract": origin == "local_operator", "reviewed_at_ms": _now_ms(),
                        "confirm_requirements": True, "confirm_unique_targets": True}
            if origin == "assistant_under_user_authorization":
                approval["authorization_reference"] = reference.strip()
            if draft["perception_mode"] == "guided":
                approval["perception_version"] = draft["perception_version"]
            # Persist approval before changing the in-memory executable task.
            self._write(self._directory / ("task-approval-" + draft["draft_id"] + ".json"), approval)
            self._cancel_recovery()
            changed_regions = set(self._protocol.regions) != set(protocol.regions)
            self._protocol, self._concepts = protocol, dict(draft["concepts"])
            metadata = self._task_metadata(draft["title"], draft["requirements"], draft["origin"],
                                           reviewed=True, review_origin=origin, model=draft["model"],
                                           perception_mode=draft["perception_mode"])
            metadata["activated_from_draft_id"] = draft["draft_id"]
            metadata["task_hash"] = draft["draft_hash"]
            if draft.get("preset_id"):
                metadata.update(preset_id=draft["preset_id"], preset_protocol_hash=draft["preset_protocol_hash"],
                                calibration_note=draft["calibration_note"])
                self._state["regions"] = deepcopy(draft["suggested_regions"])
            elif changed_regions:
                # New names start with a new layout; old label-to-rectangle
                # associations never silently carry across named contracts.
                self._state["regions"] = {region: deepcopy(rectangle) for region, rectangle in
                                          zip(protocol.regions, DEFAULT_REGIONS.values())}
            self._state.update(active_task=metadata, task_draft=None, draft_error=None, images=[], source_kind="none",
                               inference_origin=None, evidence_received_at_ms=None, evidence_observed_at_ms=None,
                               evidence_age_basis=None, perception_provenance=None)
            self._evidence_record = None
            self._reset_acquisition()
            self._draft_record = None
            self._invalidate("task_activated")
            self._state["source_note"] = "Reviewed task activated. Prior evidence belongs to a different task; capture two new views."
            self._persist()
            return self.snapshot()

    def _write(self, path: Path, value: Any) -> None:
        # All writes are private; no raw source or model response is public.
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")

    def _persist(self) -> None:
        self._write(self._directory / "state.json", self._state)

    def _recovery_summary(self, verification=None) -> dict[str, Any] | None:
        record = self._recovery_record
        if record is None:
            return None
        keys = ("receipt_id", "kind", "outcome", "before_revision", "requested_revision", "after_revision",
                "before_status", "after_status", "plan_steps", "receipt_hash")
        summary = {key: deepcopy(record.get(key)) for key in keys}
        summary.update(task_hash=record["contract"].get("task_hash"), protocol_hash=record["contract"]["protocol_hash"],
                       physical_action_execution_verified=False,
                       verification_note="Fresh goal evidence can verify the requested state; physical execution and capture timing are not proved.")
        for stage in ("before", "after"):
            summary[stage + "_checks"] = [{key: str(check[key])[:256] for key in
                ("check_id", "object_id", "status", "reason") if key in check} for check in record.get(stage + "_checks", [])][:64]
        if verification is not None:
            summary["verification"] = verification
            if not verification["valid"]:
                summary["outcome"] = "error"
                summary["verification_note"] = verification["reason"]
        return summary

    def _save_recovery(self, record: dict[str, Any]) -> None:
        previous = self._recovery_record
        record = deepcopy(record)
        record.pop("receipt_hash", None)
        parent = previous if previous and previous["receipt_id"] == record["receipt_id"] else None
        record["previous_receipt_hash"] = parent["receipt_hash"] if parent else None
        record["previous_receipt"] = ({"file": self._recovery_path.relative_to(self.root).as_posix(),
                                       "sha256": hashlib.sha256(self._recovery_path.read_bytes()).hexdigest()}
                                      if parent else None)
        path = self._directory / f"recovery-{record['receipt_id']}-{self._recovery_sequence:04d}.json"
        self._recovery_sequence += 1
        saved = save_receipt(path, record)
        self._recovery_record, self._recovery_path = saved, path
        self._state["recovery"] = self._recovery_summary(verify_recovery_receipt(path, root=self.root))

    def _before_recovery_evidence(self) -> dict[str, Any] | None:
        if self._evidence_record is None:
            return None
        bundle = deepcopy(self._evidence_record)
        if self._state["source_kind"] == "camera":
            bundle["evaluation_now_ms"] = _now_ms()
        elif self._state["source_kind"] == "upload":
            bundle["evaluation_now_ms"] = 1000 + max(0, _now_ms() - self._state["evidence_received_at_ms"])
        return bundle

    def _start_recovery(self, before: dict[str, Any]) -> None:
        proposal = before.get("proposal") or {}
        physical = before["status"] == "fail" and any(value.startswith("move-") for value in proposal.get("action_ids", []))
        existing = self._recovery_record
        if (not physical and existing and existing["kind"] == "repair"
                and existing["outcome"] in ("awaiting_fresh_evidence", "needs_evidence", "error")):
            record = deepcopy(existing)
            record.update(outcome="awaiting_fresh_evidence", requested_revision=self._state["revision"],
                          after=None, after_status=None, after_checks=[], after_revision=None)
        else:
            contract = task_contract(before["active_task"])
            record = {"schema_version": "1.0", "receipt_id": secrets.token_hex(12),
                "kind": "repair" if physical else "inspection", "outcome": "awaiting_fresh_evidence",
                "contract": contract, "contract_hash": digest(contract),
                "before": self._before_recovery_evidence(), "before_status": before["status"],
                "before_checks": deepcopy(before["checks"]), "before_revision": before["revision"],
                "requested_revision": self._state["revision"], "after_revision": None,
                "after": None, "after_status": None, "after_checks": [],
                "plan_action_ids": list(proposal.get("action_ids", [])) if physical else [],
                "planner_version": "declared_location_dijkstra_v1" if physical else None,
                "plan_steps": list(proposal.get("steps", []))[:32] if physical else
                    list((before.get("inspection_request") or {}).get("instructions", []))[:32],
                "physical_action_execution_verified": False,
                "limits": ["Source timing, uniqueness, calibration and reviewer identity are declared assumptions.",
                           "Hashes detect changes relative to this receipt; they are not signatures, trusted timestamps or proof of API origin.",
                           "Recorded references describe historical scenes; two pictures do not prove the requested motion was executed.",
                           "A passing outcome establishes the captured goal state after a request, not physical action recognition."]}
        self._save_recovery(record)

    def _finish_recovery(self, outcome: str) -> None:
        if (self._recovery_record is None or self._recovery_record["outcome"] == "cancelled_task_change"
                or self._recovery_record["outcome"].startswith("fresh_goal_verified")):
            return
        record = deepcopy(self._recovery_record)
        if digest(task_contract(self._state["active_task"])) != record["contract_hash"]:
            self._cancel_recovery()
            return
        record.update(after=deepcopy(self._evidence_record), after_status=self._state["status"],
                      after_checks=deepcopy(self._state["checks"]), after_revision=self._state["revision"])
        record["outcome"] = ("fresh_goal_verified_after_requested_repair" if record["kind"] == "repair" else
                             "fresh_goal_verified_after_inspection") if outcome == "pass" else {
                                 "fail": "still_failed", "unknown": "needs_evidence", "error": "error"}[outcome]
        self._save_recovery(record)

    def _complete_recovery_safely(self, outcome: str) -> None:
        try:
            self._finish_recovery(outcome)
        except (OSError, ValueError, TypeError, KeyError, AttributeError):
            # A receipt error must not rewrite a separately supported image
            # verdict; it only removes the recovery claim.
            summary = self._recovery_summary()
            if summary is not None:
                summary.update(outcome="error", verification_note="Recovery receipt could not be saved or replayed; no verified recovery claim is available.")
                self._state["recovery"] = summary

    def _cancel_recovery(self) -> None:
        if self._recovery_record is not None and self._recovery_record["outcome"] in (
                "awaiting_fresh_evidence", "needs_evidence", "still_failed", "error"):
            record = deepcopy(self._recovery_record)
            record["outcome"] = "cancelled_task_change"
            self._save_recovery(record)

    def recovery_receipt(self) -> dict[str, Any] | None:
        with self._lock:
            verification = self.verify_recovery()
            return deepcopy(self._recovery_summary(verification))

    def verify_recovery(self) -> dict[str, Any]:
        with self._lock:
            if self._recovery_path is None:
                return {"valid": False, "scope": "requested", "receipt_hash": None, "outcome": None,
                        "reason": "No recovery receipt has been recorded.", "physical_action_execution_verified": False}
            result = verify_recovery_receipt(self._recovery_path, root=self.root)
            self._state["recovery"] = self._recovery_summary(result)
            return result

    def _trace(self, event: str, label: str, status: str) -> None:
        self._state["trace"].append({"id": secrets.token_hex(6), "event": event, "label": label,
                                      "status": status, "revision": self._state["revision"]})
        self._state["trace"] = self._state["trace"][-100:]

    def _unknown_checks(self, reason: str) -> None:
        self._state["checks"] = [{"check_id": check.id, "object_id": check.object_id,
            "status": "unknown", "expected_region": check.region_id, "observed_region": None,
            "reason": reason, "grounding_reason": None, "confidence": None} for check in self._protocol.checks]

    def _public_checks(self, snapshot: dict[str, Any], observations, grounding=None) -> list[dict[str, Any]]:
        observations = {item.object_id: item for item in observations}
        results = {item["check_id"]: item for item in snapshot["checks"]}
        reasons = {} if grounding is None else {item["check_id"]: item["reason"]
                    for item in grounding.evidence.get("check_grounding", [])}
        return [{"check_id": check.id, "object_id": check.object_id,
                 "status": results[check.id]["status"], "expected_region": check.region_id,
                 "observed_region": observations[check.object_id].value,
                 "reason": results[check.id]["reason"],
                 "grounding_reason": reasons.get(check.id),
                 "evidence_ids": list(results[check.id].get("evidence_ids", [])),
                 "confidence": observations[check.object_id].confidence} for check in self._protocol.checks]

    def _reset_acquisition(self) -> None:
        self._acquisition_context = None
        self._acquisition_plan = None
        self._acquisition_attempts = 0
        self._acquisition_pending = False

    def _current_acquisition(self, state: dict[str, Any]) -> dict[str, Any] | None:
        if state["busy"] and state.get("busy_operation") == "inspection":
            return None
        if self._acquisition_context is None:
            return deepcopy(self._acquisition_plan)
        context = self._acquisition_context
        elapsed = max(0, _now_ms() - context["received_at_ms"])
        # Uploaded spacing remains operator-declared. Advancing logical time
        # by server elapsed time makes expiration explicit without inventing a
        # measured capture timestamp.
        now = context["now_ms"] + elapsed
        plan = build_acquisition_plan(self._protocol, context["store"],
            grounding=context["grounding"], locator=context["locator"],
            samples=context["samples"], regions=context["regions"],
            attempts_used=self._acquisition_attempts, max_attempts=2,
            used_source_hashes=tuple(self._seen_hashes), now_ms=now,
            object_labels=self._concepts)
        if plan["status"] == "not_needed":
            return None
        return plan

    def _inspection_request(self, state: dict[str, Any]) -> dict[str, Any] | None:
        """Present structured acquisition gaps; no explanation-string matching."""
        if state["busy"] and state.get("busy_operation") == "inspection":
            return None
        plan = state.get("acquisition")
        if plan:
            requests = plan["requests"]
            instructions = list(plan["instructions"])
            common = "Capture two entirely new views at least one second apart, with the arrangement unchanged between views."
            instructions.append(common)
            if plan.get("stop_reason"):
                instructions = [plan["stop_reason"], "Keep the result unverified. Review the task or capture setup before beginning a different reviewed task."]
            codes = {code for request in requests for code in request["gap_codes"]}
            # Compatibility reason IDs are chosen from typed diagnostic codes,
            # never by searching generated explanation text.
            reason = "insufficient_visual_evidence"
            if codes & {"multiple_candidates", "identity_ambiguous", "target_identity_ambiguous"}:
                reason = "target_identity_ambiguous"
                instructions.insert(0, "Confirm exactly one visually distinct instance of each requested target. If this is untrue, revise the task contract.")
            elif codes & {"low_confidence", "confidence_below_threshold"}:
                reason = "confidence_below_threshold"
            elif codes & {"temporal_inconsistency", "insufficient_stable_samples", "stable_pair_required", "repeated_source", "insufficient_distinct_views"}:
                reason = "stable_pair_required"
            elif codes & {"boundary_unsupported", "region_boundary", "calibration_unconfirmed", "boundary_unconfirmed", "calibration_required"}:
                reason = "region_support_unconfirmed"
            elif codes & {"stale_evidence", "evidence_expired"}:
                reason = "evidence_expired"
            terminal = plan["status"] != "ready"
            return {"title": "Evidence acquisition stopped" if terminal else "Capture the specific missing evidence",
                "instructions": instructions, "object_ids": list(dict.fromkeys(item["object_id"] for item in requests)),
                "reason": reason, "status": plan["status"], "attempts_used": plan["attempts_used"],
                "attempts_remaining": plan["attempts_remaining"], "max_attempts": plan["max_attempts"],
                "requests": requests, "stop_reason": plan.get("stop_reason")}
        objects = list(dict.fromkeys(check.object_id for check in self._protocol.checks))
        common = "Capture two new views at least one second apart, with the arrangement unchanged between views."
        regions = " and ".join(region.upper() for region in self._protocol.regions)
        if state["phase"] in ("needs_evidence", "awaiting_capture"):
            title, reason = "Capture the current arrangement", "fresh_capture_required"
            instructions = ["Show every required target and both named areas in the current camera view.",
                            f"Review {regions} rectangles against this view and confirm one distinct instance of each target.", common]
        elif state["phase"] == "error":
            title, reason = "Inspection did not complete", "inspection_incomplete"
            instructions = ["No physical correction was verified. Check the capture setup and API availability before requesting new evidence.", common]
        elif state["stale"] and state["source_kind"] != "recorded":
            title, reason = "Refresh the expired evidence", "evidence_expired"
            instructions = ["The earlier views have expired. Show every target and review both named regions again.", common]
        elif state["source_kind"] == "recorded":
            title, reason = "Inspect the current scene", "historical_reference"
            instructions = ["The displayed result is historical and cannot certify the current scene.", common]
        else:
            return None
        return {"title": title, "instructions": instructions, "object_ids": objects, "reason": reason}

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
            return {"title": "Inspect the targets before planning a move", "steps": ["Capture two clear views of every target."],
                    "hypothetical": True}
        steps = []
        action_names = {f"move-{obj}-{source}-to-{target}": (obj, source, target)
                        for obj in self._protocol.objects for source in self._protocol.regions
                        for target in self._protocol.regions if source != target}
        for action in planned["plan"]["action_ids"]:
            if action in action_names:
                obj, source, target = action_names[action]
                steps.append(f"Move the {obj} from {source.upper()} to {target.upper()}.")
        steps.append("Keep every target visible and hands clear, then capture two fresh views at least one second apart.")
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
        approval = envelope["approval"]
        self._state["active_task"] = self._task_metadata("Fork and spoon placement",
            "Place the fork in LEFT and the spoon in RIGHT.", "replayed_hash_bound_live_meta_api_on_recorded_frames",
            reviewed=True, review_origin=approval["origin"])
        self._state["active_task"]["task_hash"] = envelope["contract_hash"]
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
        self._evidence_record = evidence_bundle(self.root, self._state["active_task"], contract["regions"],
            source_kind="recorded", inference_origin=self._state["inference_origin"], revision=self._state["revision"],
            timestamps=[frame["source_timestamp_ms"] for frame in stage["frames"]],
            evaluation_now_ms=round(stage["end_seconds"] * 1000), timestamp_basis="recorded_source_pts",
            image_paths=[files[frame["sha256"]] for frame in stage["frames"]],
            result_paths=[{obj: report_path.parent / f"wrong_region-{index}-{object_index}.json"
                           for object_index, obj in enumerate(protocol.objects)} for index in range(2)])
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
            if state["stale"] and state["source_kind"] != "recorded" and state["phase"] in ("verified", "mistake", "unknown"):
                state["phase"], state["status"] = "unknown", "unknown"
                state["proposal"] = None
                state["source_note"] = "The earlier captured evidence is now older than 30 seconds. Inspect the current scene again."
                for check in state["checks"]:
                    check.update(status="unknown", reason="The earlier captured evidence is older than 30 seconds")
            state["acquisition"] = self._current_acquisition(state)
            state["diagnostic"] = explain_checks(self._protocol, state["checks"],
                acquisition=state["acquisition"], evidence_origin=state["source_kind"],
                current=state["source_kind"] in ("camera", "upload") and not state["stale"]
                        and state["phase"] in ("verified", "mistake", "unknown"))
            state["inspection_request"] = self._inspection_request(state)
            state["workflow"] = workflow_status(state, self._workflow_replay)
            state.pop("workflow_continuation", None)
            return state

    def _check_workflow_binding(self, payload: Any, state: dict[str, Any]) -> None:
        if (not isinstance(payload, dict) or set(payload) != {"task_hash", "revision"}
                or payload.get("task_hash") != state["active_task"].get("task_hash")
                or type(payload.get("revision")) is not int or payload["revision"] != state["revision"]):
            raise ValidationError("The task or inspection changed. Review the current preparation again")

    def _replay_workflow_evidence(self, state: dict[str, Any]) -> dict[str, Any]:
        binding = {"task_hash": state["active_task"].get("task_hash"),
                   "protocol_hash": state["active_task"]["protocol_hash"], "revision": state["revision"]}
        proof = {**binding, "valid": False}
        try:
            bundle = deepcopy(self._evidence_record)
            if (not bundle or bundle["contract"].get("task_hash") != binding["task_hash"]
                    or bundle["contract"]["protocol_hash"] != binding["protocol_hash"]
                    or bundle["revision"] != binding["revision"]):
                return proof
            bundle["evaluation_now_ms"] = (_now_ms() if state["source_kind"] == "camera" else
                bundle["evaluation_now_ms"] + max(0, _now_ms() - state["evidence_received_at_ms"]))
            replayed = replay_evidence(bundle, self.root)
            recorded = {item["check_id"]: item["status"] for item in state["checks"]}
            replay_statuses = {item["check_id"]: item["status"] for item in replayed["checks"]}
            proof["valid"] = (replayed["status"] == "pass" and state["status"] == "pass"
                              and recorded == replay_statuses)
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            pass
        return proof

    def check_workflow(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            state = self.snapshot()
            self._check_workflow_binding(payload, state)
            if not workflow_status(state)["can_check"]:
                raise ValidationError("Preparation is blocked. Obtain current passing evidence for the reviewed task")
            self._workflow_replay = self._replay_workflow_evidence(state)
            return self.snapshot()

    def continue_workflow(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            state = self.snapshot()
            # Every declaration rehashes and replays the saved sources, even
            # after a successful UI check. A cached button state grants nothing.
            self._workflow_replay = self._replay_workflow_evidence(state)
            state = self.snapshot()
            record = validate_continuation({**state, "workflow_continuation": self._state.get("workflow_continuation")},
                                           payload, self._workflow_replay)
            record.update(record_id=secrets.token_hex(12), declared_at_ms=_now_ms(),
                          placement_evidence=deepcopy(self._evidence_record))
            record = save_receipt(self._directory / ("workflow-" + record["record_id"] + ".json"), record)
            self._state["workflow_continuation"] = record
            self._trace("continuation_declared", "Continuation declared; physical action remains unobserved", "pass")
            self._persist()
            return self.snapshot()

    def _invalidate(self, event: str) -> dict[str, Any]:
        with self._lock:
            if self._state["busy"]:
                raise ValidationError("An inspection is already running")
            plan = self._current_acquisition(self._state)
            if plan and plan["status"] not in ("ready", "not_needed"):
                raise ValidationError("Evidence acquisition stopped. Review the task or capture setup before continuing")
            if plan and plan["status"] == "ready":
                self._acquisition_plan = plan
                self._acquisition_pending = True
            self._workflow_replay = None
            self._state.pop("workflow_continuation", None)
            self._acquisition_context = None
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
        with self._lock:
            before = self.snapshot()
            self._invalidate("repair_started")
            self._start_recovery(before)
            self._persist()
            return self.snapshot()

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
        if (self._state["active_task"].get("perception_mode", "concept") == "guided"
                and self._state["active_task"].get("perception_version") != GUIDED_VERSION):
            raise ValidationError("Guided perception changed. Request and review a new task draft before inspection")
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
        if not isinstance(regions, dict) or set(regions) != set(self._protocol.regions):
            raise ValidationError("Approve both nonoverlapping rectangles using the active task's exact region IDs")
        ground_sam_images(self._protocol, regions, [], session_id=self._state["session_id"])
        items = payload.get("images")
        if not isinstance(items, list) or len(items) != 2:
            raise ValidationError("Supply exactly two fresh images")
        decoded = [_decode_image(item) for item in items]
        if self._state["active_task"].get("perception_mode", "concept") == "guided" and any(
                mime != "image/jpeg" or max(width, height) > 1280 or b"Exif\0\0" in data
                for data, mime, width, height in decoded):
            raise ValidationError("Guided perception requires upright JPEG images without EXIF, up to 1280 pixels per side")
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
            plan = self._current_acquisition(self._state)
            if plan and plan["status"] not in ("ready", "not_needed"):
                raise ValidationError("Evidence acquisition stopped. Review the task or capture setup before continuing")
            source, regions, decoded, hashes, times, basis = self._validate_pair(payload)
            if plan and plan["status"] == "ready":
                check_source_pair(plan, hashes, timestamps_ms=times)
            if self._acquisition_pending:
                self._acquisition_attempts += 1
                self._acquisition_pending = False
                self._acquisition_plan["attempts_used"] = self._acquisition_attempts
                self._acquisition_plan["attempts_remaining"] = max(0, 2 - self._acquisition_attempts)
                if self._acquisition_attempts >= 2:
                    self._acquisition_plan.update(status="budget_exhausted", stop_reason="The two-follow-up limit is reached. Keep the result unverified and review the task or capture setup.")
            self._acquisition_context = None
            revision = self._state["revision"]
            mode = self._state["active_task"].get("perception_mode", "concept")
            issued_at = self._token_issued_ms
            self._token = None
            self._state["capture_token"] = None
            self._seen_hashes.update(hashes)
            self._evidence_record = None
            self._state.update(phase="checking", status="unknown", busy=True, busy_operation="inspection", error=None, source_kind=source,
                               proposal=None, verified_at_ms=None, regions=regions, images=[], usage=None, perception_provenance=None,
                               source_note=("Inspecting two new views with task-independent Spark localization and Meta SAM. "
                                            "The correction is not yet verified." if mode == "guided" else
                                            "Inspecting two new views with Meta SAM. The correction is not yet verified."),
                               inference_origin="live_meta_api" if self._client_factory is None else "injected_client_not_verified_live",
                               evidence_received_at_ms=_now_ms(),
                               evidence_observed_at_ms=times[0] if source == "camera" else None,
                               evidence_age_basis="client_reported_capture_time" if source == "camera" else "server_received_time_capture_age_operator_declared")
            self._unknown_checks("Fresh image inspection is in progress")
            run_directory = self._directory / ("inspection-" + str(revision))
            request_budget = 2 * len(self._protocol.objects) + (1 if mode == "guided" else 0)
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
                    "protocol_hash": self._protocol.content_hash,
                    "task_hash": self._state["active_task"]["task_hash"], "concepts": dict(self._concepts),
                    "perception_mode": mode,
                    **({"perception_version": self._state["active_task"]["perception_version"]} if mode == "guided" else {}),
                    "regions": regions, "source_sha256": hashes, "timestamp_basis": basis,
                    "source_timestamps_ms": times, "token_issued_at_ms": issued_at,
                    "operator_confirmations": {key: True for key in ("regions", "unique_targets", "fresh_capture", "pair_spacing")}})
                call_label = f"one Spark and up to {request_budget - 1} SAM calls" if mode == "guided" else f"at most {request_budget} SAM calls"
                self._trace("inspection_started", f"Two fresh views submitted; {call_label}", "unknown")
                self._persist()
            except OSError:
                self._state.update(phase="error", status="unknown", busy=False, busy_operation=None,
                                   error="Local evidence could not be saved. Start a new inspection.")
                return self.snapshot()
        client, results, grounded = None, [], None
        result_paths = [{}, {}]
        try:
            if self._client_factory is None:
                load_env_file(self.root / ".env.local")
                client = MetaClient.from_env(max_requests=request_budget, retries=0, timeout=45 if mode == "guided" else 25)
            else:
                client = self._client_factory()
            if client.max_requests != request_budget or client.retries != 0 or client.ledger:
                raise ValidationError("Use a fresh bounded client with no retries")
            samples = []
            guided = guided_segment_pair(client, paths, self._concepts) if mode == "guided" else None
            if guided is not None:
                if guided.get("provenance", {}).get("version") != self._state["active_task"]["perception_version"]:
                    raise ValidationError("Guided result does not match the reviewed perception version")
                self._write(run_directory / "guided.json", guided)
            for index, path in enumerate(paths):
                frame_results = []
                for object_id in self._protocol.objects:
                    if guided is not None:
                        result = guided["results"].get(hashes[index], {}).get(object_id)
                        if result is None:
                            continue  # Missing anchors remain missing evidence, never fabricated SAM completion.
                    else:
                        result = client.segment_image(path, self._concepts[object_id], decode_masks=True)
                    _bound_result(result, hashes[index])
                    if any((item.get("box", {}).get("width"), item.get("box", {}).get("height")) != decoded[index][2:]
                           for item in result.get("objects", [])):
                        raise ValidationError("SAM geometry does not match the exact supplied image")
                    result_path = run_directory / f"result-{index}-{object_id}.json"
                    self._write(result_path, result)
                    result_paths[index][object_id] = result_path
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
                    checks=self._public_checks(outcome, grounded.observations, grounded),
                    proposal=self._proposal(grounded.observations, outcome),
                    verified_at_ms=_now_ms() if status == "pass" else None,
                    source_note=("Captured placement passed. Inspect again after any movement. " if status == "pass" else
                                 "Fresh evidence supports a misplaced target. " if status == "fail" else
                                 "The views do not support every target location. Review the specific evidence requests and follow-up limit. ") +
                                ("Camera capture time is client-reported." if source == "camera" else
                                 "Photo freshness and one-second spacing were declared by the operator; capture time was not measured."))
                if mode == "guided":
                    self._state["source_note"] += " " + PERCEPTION_NOTES["guided"]
                    self._state["perception_provenance"] = {
                        "method": "spark_box_guided_sam", "task_independent_locator": True,
                        "version": guided["provenance"]["version"],
                        "identity_basis": "Spark-assigned; SAM refines boundaries without independently verifying identity",
                        "missing_or_ambiguous_anchors": len(guided["unavailable"]),
                    }
                if any(check["status"] == "unknown" for check in outcome["checks"]):
                    self._acquisition_context = {"store": store, "grounding": grounded.to_dict(),
                        "samples": samples, "regions": regions, "locator": guided.get("locator") if guided else None,
                        "now_ms": now, "received_at_ms": _now_ms()}
                    self._acquisition_plan = None
                    # A known misplaced item cannot authorize motion while
                    # another required target remains unconfirmed.
                    self._state["proposal"] = None
                else:
                    self._reset_acquisition()
                self._state["images"] = [self._register_media(path, decoded[index][2], decoded[index][3], results[index])
                                         for index, path in enumerate(paths)]
                self._evidence_record = evidence_bundle(self.root, self._state["active_task"], regions,
                    source_kind=source, inference_origin=self._state["inference_origin"], revision=revision,
                    timestamps=times, evaluation_now_ms=now, timestamp_basis=basis,
                    image_paths=paths, result_paths=result_paths,
                    guided_path=run_directory / "guided.json" if mode == "guided" else None)
                self._trace("inspection_complete", {"pass": "Fresh captured placement verified", "fail": "Mistake observed in fresh captures",
                                                   "unknown": "Insufficient visual evidence; no pass issued"}[status], status)
                self._state["usage"] = client.usage_summary()
                self._complete_recovery_safely(status)
        except (MetaAPIError, OSError, ValueError, TypeError, KeyError, AttributeError) as error:
            with self._lock:
                if self._acquisition_plan is None:
                    self._acquisition_context = {"store": EvidenceStore(self._protocol, self._state["session_id"]),
                        "grounding": None, "samples": [], "regions": regions, "locator": None,
                        "now_ms": 1000, "received_at_ms": _now_ms()}
                self._state.update(phase="error", status="unknown", verified_at_ms=None, proposal=None,
                    error="The Meta inspection could not complete. No correction was verified. Start a new inspection and capture new views.",
                    source_note="Inspection failed; no correction was verified. Start a new inspection with two new views.",
                    usage=client.usage_summary() if client is not None else None)
                self._unknown_checks("Inspection failed; no supported current result")
                self._complete_recovery_safely("error")
                self._trace("inspection_failed", "Inspection stopped without a verified result", "unknown")
                try:
                    self._write(run_directory / "failure.json", {"error_type": type(error).__name__,
                                 "response": error.response if isinstance(error, MetaAPIError) else None,
                                 "usage": client.usage_summary() if client is not None else None})
                except OSError:
                    pass
        finally:
            with self._lock:
                self._state.update(busy=False, busy_operation=None)
                try:
                    self._persist()
                except OSError:
                    self._state.update(phase="error", status="unknown", verified_at_ms=None,
                                       error="The inspection state could not be saved; inspect again.")
        return self.snapshot()
