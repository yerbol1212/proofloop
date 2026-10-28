"""Durable asynchronous bridge to the existing reviewed inspection engine.

An operation is dispatched at most once by this service. A lost/expired worker
claim is terminal and is never reassigned. Stored verdicts are completion-time
records, never permission to perform a physical action or evidence of a later
scene. Tests may inject an engine/callback; those results retain their origin.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import secrets
import tempfile
import threading
import time
from typing import Any, Callable

from .core import ValidationError
from .jobs import (JobConflict, JobError, JobIntegrityError, JobLeaseError,
                   JobStore, JobValidationError, MAX_LEASE_MS)
from .meta import MetaAPIError
from .verification_policy import (POLICY_VERSION, describe_capabilities,
                                  public_inspection_result, validate_binding)

_RESULT_LIMIT = 512 * 1024
_INNER_FIELDS = {"capture_token", "source", "images", "regions", "confirm_regions",
                 "confirm_unique_targets", "confirm_fresh_capture", "confirm_pair_spacing"}
_TOP_FIELDS = {"schema_version", "policy_version", "binding", "status", "completed", "reason_code",
               "current_at_snapshot", "source_kind", "freshness", "inference_origin", "checks",
               "evidence_requests", "acquisition", "usage", "physical_action_authorized",
               "physical_action_verified", "limitations"}
_REASONS = {"supported_completion", "inspection_not_complete", "inspection_failed", "current_capture_required",
            "current_evidence_expired_or_unavailable", "unsupported_completion_evidence",
            "inconsistent_completion_verdict", "specific_evidence_required", "failure_with_unconfirmed_requirements"}
_GAPS = {"missing_current_evidence", "stale_evidence", "not_detected", "multiple_candidates",
         "identity_ambiguous", "partial_visibility", "low_confidence", "boundary_unconfirmed",
         "invalid_detection", "mask_unavailable", "temporal_inconsistency", "insufficient_distinct_views",
         "calibration_required", "unsupported_observation", "unsupported_stability_requirement",
         "absence_validation_required", "temporal_history_required"}
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:\-]{0,159}\Z")
_EVIDENCE = re.compile(r"(?:sam-grounding://[0-9a-f]{64}|fixture://proofloop/[A-Za-z0-9_-]{1,80}/[A-Za-z0-9_-]{1,80}|[A-Za-z0-9][A-Za-z0-9_.:\-]{0,159})\Z")
_USAGE = {"request_attempts", "input_tokens", "output_tokens", "sam_images",
          "spark_requests_with_unreported_token_usage", "unsuccessful_spark_requests_with_reported_token_usage"}


def _now_ms() -> int:
    return time.time_ns() // 1_000_000


def _encoded(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def _bounded_id(value: Any) -> bool:
    return type(value) is str and _ID.fullmatch(value) is not None


def _public_result(value: Any, job: dict[str, Any]) -> dict[str, Any]:
    """Validate a persisted policy DTO again before exposing any of its bytes."""
    try:
        if (type(value) is not dict or set(value) != _TOP_FIELDS
                or value["schema_version"] != "1.0" or value["policy_version"] != POLICY_VERSION
                or value["binding"] != job["binding"] or value["status"] != job["result"]["status"]
                or value["binding"]["revision"] != job["result"]["revision"]
                or value["status"] not in ("pass", "fail", "unknown")
                or value["completed"] is not True or value["reason_code"] not in _REASONS
                or type(value["current_at_snapshot"]) is not bool
                or value["source_kind"] not in ("camera", "upload", "recorded", "none", "unspecified")
                or value["inference_origin"] not in ("live_meta_api", "injected_client_not_verified_live", "saved_live_meta", "authored_fixture", "unspecified")
                or value["physical_action_authorized"] is not False or value["physical_action_verified"] is not False
                or value["limitations"] != describe_capabilities()["limitations"]):
            raise ValueError
        freshness = value["freshness"]
        if (type(freshness) is not dict or set(freshness) != {"age_ms", "window_ms", "basis", "capture_time_independently_verified"}
                or freshness["age_ms"] is not None and (type(freshness["age_ms"]) is not int or freshness["age_ms"] < 0)
                or type(freshness["window_ms"]) is not int or not 0 <= freshness["window_ms"] <= 30_000
                or freshness["basis"] not in (None, "client_reported_capture_time", "server_received_time_capture_age_operator_declared")
                or freshness["capture_time_independently_verified"] is not False):
            raise ValueError
        checks = value["checks"]
        if type(checks) is not list or not 1 <= len(checks) <= 6:
            raise ValueError
        seen = set()
        for row in checks:
            if (type(row) is not dict or set(row) != {"check_id", "object_id", "expected_region", "observed_region", "status", "evidence_ids"}
                    or any(not _bounded_id(row[key]) for key in ("check_id", "object_id", "expected_region"))
                    or row["observed_region"] is not None and not _bounded_id(row["observed_region"])
                    or row["status"] not in ("pass", "fail", "unknown") or row["check_id"] in seen
                    or type(row["evidence_ids"]) is not list or len(row["evidence_ids"]) > 32
                    or any(type(item) is not str or _EVIDENCE.fullmatch(item) is None for item in row["evidence_ids"])):
                raise ValueError
            seen.add(row["check_id"])
        requests = value["evidence_requests"]
        if type(requests) is not list or len(requests) > len(checks):
            raise ValueError
        by_id = {row["check_id"]: row for row in checks}
        for row in requests:
            if (type(row) is not dict or set(row) != {"check_id", "object_id", "expected_region", "gap_codes", "required_views", "minimum_spacing_ms", "physical_action_authorized"}
                    or row["check_id"] not in by_id
                    or any(row[key] != by_id[row["check_id"]][key] for key in ("object_id", "expected_region"))
                    or type(row["gap_codes"]) is not list or not row["gap_codes"]
                    or any(type(code) is not str or code not in _GAPS for code in row["gap_codes"])
                    or type(row["required_views"]) is not int or row["required_views"] != 2
                    or type(row["minimum_spacing_ms"]) is not int or row["minimum_spacing_ms"] != 1000
                    or row["physical_action_authorized"] is not False):
                raise ValueError
        acquisition = value["acquisition"]
        if acquisition is not None:
            if type(acquisition) is not dict or type(acquisition.get("follow_up_allowed")) is not bool:
                raise ValueError
            if acquisition.get("status") == "invalid":
                if acquisition != {"status": "invalid", "follow_up_allowed": False}:
                    raise ValueError
            elif (set(acquisition) != {"status", "attempts_used", "attempts_remaining", "max_attempts", "follow_up_allowed"}
                    or acquisition["status"] not in ("ready", "not_needed", "budget_exhausted", "unsupported_contract", "timeline_exhausted")
                    or any(type(acquisition[key]) is not int for key in ("attempts_used", "attempts_remaining", "max_attempts"))
                    or not 0 <= acquisition["attempts_used"] <= acquisition["max_attempts"] <= 2
                    or acquisition["attempts_remaining"] != acquisition["max_attempts"] - acquisition["attempts_used"]):
                raise ValueError
        usage = value["usage"]
        if usage is not None and (type(usage) is not dict or set(usage) - (_USAGE | {"estimated_usd"})
                or any(type(amount) is not int or amount < 0 for key, amount in usage.items() if key != "estimated_usd")
                or "estimated_usd" in usage and (type(usage["estimated_usd"]) not in (int, float)
                    or not math.isfinite(usage["estimated_usd"]) or usage["estimated_usd"] < 0)):
            raise ValueError
        return deepcopy(value)
    except (ValueError, KeyError, TypeError, AttributeError):
        raise JobIntegrityError("Stored inspection result is invalid") from None


class InspectionJobService:
    """One worker per service; SQLite claims fence additional service processes.

    ``inspect_callback`` receives (inspection, binding, job_id, guard). Production
    callbacks must invoke guard before every provider request. The default calls
    the engine directly with its inside-lock expected-binding check. Admission
    validates images with the engine but does not consume the capture token.
    """

    def __init__(self, engine: Any, *, store: JobStore | None = None,
                 db_path: str | Path | None = None, artifact_root: str | Path | None = None,
                 inspect_callback: Callable[..., dict[str, Any]] | None = None,
                 clock_ms: Callable[[], int] | None = None, lease_ms: int = 30_000,
                 poll_interval: float = .1):
        if type(lease_ms) is not int or not 20 <= lease_ms <= MAX_LEASE_MS:
            raise JobValidationError("Service lease duration is invalid")
        if type(poll_interval) not in (int, float) or not math.isfinite(poll_interval) or not .001 <= poll_interval <= 10:
            raise JobValidationError("Service poll interval is invalid")
        self.engine = engine
        directory = Path(engine.root) / "artifacts/local/job-runtime"
        self.store = store if store is not None else JobStore(db_path or directory / "jobs.sqlite3")
        self.artifact_root = Path(artifact_root or directory / "results")
        if self.artifact_root.is_symlink() or self.artifact_root.exists() and not self.artifact_root.is_dir():
            raise JobValidationError("Job result directory must be private and regular")
        self.artifact_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.artifact_root, 0o700)
        self._inspect_callback = inspect_callback
        self._clock_ms = clock_ms or _now_ms
        self._lease_ms = lease_ms
        self._poll_interval = float(poll_interval)
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._lifecycle_lock = threading.RLock()
        self._journal_lock = threading.RLock()
        self._thread: threading.Thread | None = None
        self._worker_id = "worker-" + secrets.token_hex(12)
        self._worker_error: str | None = None

    @staticmethod
    def _payload(payload: Any) -> dict[str, Any]:
        if type(payload) is not dict or set(payload) != {"binding", "inspection"}:
            raise ValidationError("Supply the exact inspection job binding and inspection payload")
        inspection = payload["inspection"]
        if type(inspection) is not dict or set(inspection) != _INNER_FIELDS:
            raise ValidationError("Supply only the supported inspection capture fields")
        images = inspection.get("images")
        if (type(images) is not list or len(images) != 2 or any(type(item) is not dict
                or set(item) - {"data_url", "captured_at_ms"} for item in images)):
            raise ValidationError("Supply exactly two supported inspection images")
        # Take a private copy before returning to a caller or starting a worker.
        try:
            return json.loads(json.dumps(payload, allow_nan=False))
        except (ValueError, TypeError, RecursionError):
            raise ValidationError("Inspection job payload must be finite JSON") from None

    def submit_inspection(self, payload: dict[str, Any], *, idempotency_key: str) -> dict[str, Any]:
        request = self._payload(payload)
        with self._lifecycle_lock:
            if self._stop.is_set():
                raise JobConflict("The inspection worker is closed")
        lookup = getattr(self.store, "lookup_submission", None)
        if lookup is not None:
            with self._journal_lock:
                previous = lookup("inspection", request, idempotency_key=idempotency_key, binding=request["binding"])
            if previous is not None:
                return self.get(previous["id"])
        with self.engine._lock:
            validate_binding(request["binding"], self.engine.snapshot())
            self.engine._validate_pair(request["inspection"])
            with self._journal_lock:
                job = self.store.submit("inspection", request, idempotency_key=idempotency_key,
                                        binding=request["binding"], now_ms=self._clock_ms())
        self._wake.set()
        return self.get(job["id"])

    def capabilities(self) -> dict[str, Any]:
        return {**describe_capabilities(), "enabled": True, "jobs": {
            "asynchronous": True, "automatic_reexecution": False,
            "idempotency": "exact binding and inspection payload",
            "cancellation": "queued_before_dispatch_only", "lease_ms": self._lease_ms,
            "states": ["queued", "running", "succeeded", "failed", "interrupted", "cancelled"],
            "stored_result_scope": "historical_completion_snapshot",
            "worker_available": (not self._stop.is_set() and self._worker_error is None
                                 and self._thread is not None and self._thread.is_alive()),
            "worker_error": self._worker_error,
        }}

    def get(self, job_id: str) -> dict[str, Any]:
        with self._journal_lock:
            integrity = self.store.verify(job_id)
            if not integrity["valid"]:
                raise JobIntegrityError("Stored job verification failed")
            job = self.store.get(job_id)
        normalized = None
        result_valid: bool | None = None
        if job["state"] == "succeeded":
            try:
                path = self._result_path(job_id)
                if path.is_symlink() or not path.is_file() or path.stat().st_size > _RESULT_LIMIT:
                    raise ValueError
                data = path.read_bytes()
                if hashlib.sha256(data).hexdigest() != job["result"].get("result_sha256"):
                    raise ValueError
                normalized = _public_result(json.loads(data), job)
                result_valid = True
            except (OSError, ValueError, KeyError, TypeError, JobIntegrityError):
                result_valid = False
                # Retain the historical operation state but expose no verdict,
                # including the small summary, without its complete artifact.
                job["result"] = None
        return {**job, "inspection_result": normalized,
                "integrity": {"journal_valid": True, "result_valid": result_valid},
                "result_scope": "historical_completion_snapshot", "physical_action_authorized": False}

    def events(self, job_id: str, *, after_sequence: int = 0, limit: int = 200) -> list[dict[str, Any]]:
        with self._journal_lock:
            return self.store.events(job_id, after_sequence=after_sequence, limit=limit)

    def cancel(self, job_id: str) -> dict[str, Any]:
        with self._journal_lock:
            self.store.cancel(job_id, now_ms=self._clock_ms())
        return self.get(job_id)

    def start(self) -> "InspectionJobService":
        with self._lifecycle_lock:
            if self._stop.is_set():
                raise JobConflict("The inspection worker is closed")
            if self._thread is None:
                with self._journal_lock:
                    self.store.recover_expired(now_ms=self._clock_ms())
                self._thread = threading.Thread(target=self._run, name="proofloop-inspection-worker", daemon=True)
                self._thread.start()
        return self

    def close(self, *, timeout: float = 5) -> None:
        if type(timeout) not in (int, float) or not math.isfinite(timeout) or not 0 <= timeout <= 60:
            raise JobValidationError("Worker close timeout is invalid")
        self._stop.set()
        self._wake.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout)

    def _result_path(self, job_id: str) -> Path:
        if type(job_id) is not str or re.fullmatch(r"[0-9a-f]{32}", job_id) is None:
            raise JobValidationError("Job identifier is invalid")
        return self.artifact_root / (job_id + ".json")

    def _save_result(self, job_id: str, value: dict[str, Any]) -> str:
        data = _encoded(value)
        if len(data) > _RESULT_LIMIT:
            raise JobIntegrityError("Inspection result is excessive")
        final = self._result_path(job_id)
        descriptor, temporary = tempfile.mkstemp(prefix=".result-", dir=self.artifact_root)
        try:
            os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, final)
            directory = os.open(self.artifact_root, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
        return hashlib.sha256(data).hexdigest()

    def _fail(self, job: dict[str, Any], code: str) -> None:
        try:
            with self._journal_lock:
                self.store.fail(job["id"], job["lease"]["claim_token"], code, now_ms=self._clock_ms())
        except JobLeaseError:
            with self._journal_lock:
                self.store.recover_expired(now_ms=self._clock_ms())

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                # Read the clock after acquiring the local journal lock. A
                # timestamp captured before a competing submit's commit can
                # otherwise look older than that job despite a correct clock.
                with self._journal_lock:
                    self.store.recover_expired(now_ms=self._clock_ms())
                    job = self.store.claim_next(self._worker_id, now_ms=self._clock_ms(), lease_ms=self._lease_ms)
                if job is not None:
                    self._execute(job)
                    continue
            except JobConflict:
                # A different service process may commit a newer timestamp
                # between obtaining our clock value and acquiring SQLite's
                # write lock. Retry queue inspection, never external inference.
                pass
            except (JobError, OSError, ValueError, TypeError):
                # A corrupt queue is not skipped or silently rewritten. Stop
                # dispatch and expose one fixed diagnostic, never exception text.
                self._worker_error = "worker_journal_unavailable"
                return
            self._wake.wait(self._poll_interval)
            self._wake.clear()

    def _execute(self, job: dict[str, Any]) -> None:
        token = job["lease"]["claim_token"]
        binding = job["binding"]
        heartbeat_done = threading.Event()
        lease_lost = threading.Event()
        claim_lock = threading.RLock()

        def guard() -> None:
            if self._stop.is_set() or lease_lost.is_set():
                raise JobLeaseError("The inspection dispatch is unavailable")
            with self.engine._lock:
                validate_binding(binding, self.engine.snapshot(), require_ready=False)
                with claim_lock:
                    with self._journal_lock:
                        self.store.heartbeat(job["id"], token, now_ms=self._clock_ms(), lease_ms=self._lease_ms)

        def heartbeat() -> None:
            while not heartbeat_done.wait(self._lease_ms / 3000):
                try:
                    with claim_lock:
                        with self._journal_lock:
                            self.store.heartbeat(job["id"], token, now_ms=self._clock_ms(), lease_ms=self._lease_ms)
                except (JobError, OSError, ValueError):
                    lease_lost.set()
                    return

        heartbeat_thread = threading.Thread(target=heartbeat, name="proofloop-job-heartbeat", daemon=True)
        heartbeat_thread.start()
        try:
            with self.engine._lock:
                if self._stop.is_set():
                    raise JobLeaseError("The inspection dispatch is unavailable")
                validate_binding(binding, self.engine.snapshot())
                self.engine._validate_pair(job["payload"]["inspection"])
                with claim_lock:
                    with self._journal_lock:
                        self.store.mark_dispatched(job["id"], token, now_ms=self._clock_ms())
            if self._inspect_callback is None:
                state = self.engine.inspect_pair(job["payload"]["inspection"], expected_binding=binding)
            else:
                state = self._inspect_callback(job["payload"]["inspection"], binding, job["id"], guard)
            with self.engine._lock:
                validate_binding(binding, self.engine.snapshot(), require_ready=False)
                validate_binding(binding, state, require_ready=False)
                if state.get("phase") == "error" or state.get("error") is not None:
                    self._fail(job, "model_error")
                    return
                result = public_inspection_result(binding, state)
                if not result["completed"]:
                    self._fail(job, "validation_error")
                    return
                reference = {"status": result["status"], "revision": binding["revision"]}
                # Assert the exact closed public contract before publication.
                _public_result(result, {"binding": binding, "result": reference})
                reference["result_sha256"] = self._save_result(job["id"], result)
                recovery = state.get("recovery")
                if isinstance(recovery, dict) and isinstance(recovery.get("receipt_id"), str) and re.fullmatch(r"[0-9a-f]{24,64}", recovery["receipt_id"]):
                    reference["receipt_id"] = recovery["receipt_id"]
                with claim_lock:
                    if lease_lost.is_set():
                        raise JobLeaseError("The inspection claim expired")
                    with self._journal_lock:
                        self.store.complete(job["id"], token, reference, now_ms=self._clock_ms())
        except JobLeaseError:
            with self._journal_lock:
                self.store.recover_expired(now_ms=self._clock_ms())
        except ValidationError:
            self._fail(job, "stale_binding")
        except MetaAPIError:
            self._fail(job, "provider_error")
        except OSError:
            self._fail(job, "artifact_error")
        except (JobError, ValueError, TypeError, KeyError, AttributeError):
            self._fail(job, "worker_error")
        except Exception:
            self._fail(job, "worker_error")
        finally:
            heartbeat_done.set()
            heartbeat_thread.join(min(1, self._lease_ms / 1000))


JobService = InspectionJobService
