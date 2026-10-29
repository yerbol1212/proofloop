"""Durable verification jobs without automatic replay of external requests.

The SQLite journal is local tamper evidence, not a signature or a trusted clock.
Payloads and lease tokens are private. Public projections never contain provider
responses, exception text, images, credentials, worker identities, or file paths.
An expired lease becomes terminal even if dispatch was not recorded: operators
must make a new request before another model call can occur.
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
import sqlite3
from typing import Any


MAX_PAYLOAD_BYTES = 16 * 1024 * 1024
MAX_LEASE_MS = 300_000
STATES = frozenset({"queued", "running", "succeeded", "failed", "interrupted", "cancelled"})
TERMINAL_STATES = STATES - {"queued", "running"}
ERROR_CODES = frozenset({"invalid_request", "task_changed", "model_error", "validation_error",
    "worker_error", "dispatch_error", "artifact_error", "interrupted_no_dispatch", "outcome_unknown",
    "payload_integrity_error", "stale_binding", "transport_error", "provider_error", "budget_exhausted"})
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_JOB_ID = re.compile(r"[0-9a-f]{32}\Z")
_OPAQUE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,159}\Z")
_GENESIS = "0" * 64


class JobError(Exception):
    """Base exception; messages are static and safe for server-side mapping."""


class JobValidationError(JobError):
    pass


class JobConflict(JobError):
    pass


class JobNotFound(JobError):
    pass


class JobLeaseError(JobConflict):
    pass


class JobIntegrityError(JobError):
    pass


def _canonical(value: Any) -> str:
    """Strict JSON; reject bool-as-integer, non-string keys and nested excess."""
    def visit(item: Any, depth: int) -> None:
        if depth > 64:
            raise JobValidationError("Job JSON nesting is excessive")
        if item is None or type(item) in (str, bool, int):
            return
        if type(item) is float and math.isfinite(item):
            return
        if type(item) is list:
            for child in item:
                visit(child, depth + 1)
            return
        if type(item) is dict and all(type(key) is str for key in item):
            for child in item.values():
                visit(child, depth + 1)
            return
        raise JobValidationError("Job values must be finite JSON")
    visit(value, 0)
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
    except (TypeError, ValueError, RecursionError):
        raise JobValidationError("Job values must be finite JSON") from None


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _time(value: Any) -> int:
    if type(value) is not int or not 0 <= value < 2 ** 63:
        raise JobValidationError("Job time must be a bounded nonnegative integer")
    return value


def _identifier(value: Any, pattern: re.Pattern, label: str) -> str:
    if type(value) is not str or pattern.fullmatch(value) is None:
        raise JobValidationError(label)
    return value


def _binding(value: Any) -> dict[str, Any]:
    if type(value) is not dict or set(value) != {"task_hash", "protocol_hash", "revision", "session_id"}:
        raise JobValidationError("Job binding requires task and protocol hashes, revision and session")
    _identifier(value["task_hash"], _HEX64, "Job task hash is invalid")
    _identifier(value["protocol_hash"], _HEX64, "Job protocol hash is invalid")
    _identifier(value["session_id"], _OPAQUE_ID, "Job session is invalid")
    if type(value["revision"]) is not int or not 0 <= value["revision"] < 2 ** 63:
        raise JobValidationError("Job revision is invalid")
    return deepcopy(value)


def _result(value: Any) -> dict[str, Any]:
    if type(value) is not dict or set(value) - {"revision", "status", "receipt_id", "result_sha256"} or not {"revision", "status"} <= set(value):
        raise JobValidationError("Job result must be a safe application reference")
    if type(value["revision"]) is not int or not 0 <= value["revision"] < 2 ** 63:
        raise JobValidationError("Job result revision is invalid")
    if value["status"] not in ("pass", "fail", "unknown"):
        raise JobValidationError("Job result status is invalid")
    if "receipt_id" in value and (type(value["receipt_id"]) is not str or re.fullmatch(r"[0-9a-f]{24,64}", value["receipt_id"]) is None):
        raise JobValidationError("Job receipt reference is invalid")
    if "result_sha256" in value:
        _identifier(value["result_sha256"], _HEX64, "Job result digest is invalid")
    return deepcopy(value)


class JobStore:
    """One SQLite connection per operation; transactions fence competing workers.

    submit is idempotent for an exact kind/payload/binding match. A reused key
    with changed input raises JobConflict. claim_next is private worker output;
    every other returned job is a redacted public projection.
    """

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        if self.db_path.exists() and (self.db_path.is_symlink() or not self.db_path.is_file()):
            raise JobValidationError("Job database must be a regular private file")
        directory_existed = self.db_path.parent.exists()
        self.db_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        # Only change directories this store creates, never a caller's project.
        if not directory_existed:
            os.chmod(self.db_path.parent, 0o700)
        descriptor = os.open(self.db_path, os.O_CREAT | os.O_RDWR, 0o600)
        os.close(descriptor)
        os.chmod(self.db_path, 0o600)
        connection = self._connect()
        try:
            # DELETE mode keeps concurrent transactions portable across older
            # supported Python builds; no WAL-specific version assumptions.
            connection.execute("PRAGMA journal_mode=DELETE")
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY, kind TEXT NOT NULL, key_hash TEXT NOT NULL UNIQUE,
                    payload_json TEXT NOT NULL, payload_sha256 TEXT NOT NULL, binding_json TEXT NOT NULL,
                    state TEXT NOT NULL, created_ms INTEGER NOT NULL, updated_ms INTEGER NOT NULL,
                    attempt INTEGER NOT NULL DEFAULT 0, dispatched INTEGER NOT NULL DEFAULT 0,
                    worker_id TEXT, claim_token TEXT, lease_expires_ms INTEGER,
                    result_json TEXT, error_code TEXT, event_count INTEGER NOT NULL DEFAULT 0,
                    event_head TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS jobs_queue ON jobs(state, created_ms, id);
                CREATE TABLE IF NOT EXISTS job_events (
                    job_id TEXT NOT NULL REFERENCES jobs(id), sequence INTEGER NOT NULL,
                    event_json TEXT NOT NULL, event_hash TEXT NOT NULL,
                    PRIMARY KEY(job_id, sequence)
                );
                CREATE TRIGGER IF NOT EXISTS job_events_no_update BEFORE UPDATE ON job_events
                BEGIN SELECT RAISE(ABORT, 'Job events are append only'); END;
                CREATE TRIGGER IF NOT EXISTS job_events_no_delete BEFORE DELETE ON job_events
                BEGIN SELECT RAISE(ABORT, 'Job events are append only'); END;
                CREATE TABLE IF NOT EXISTS job_submission_intents (
                    job_id TEXT PRIMARY KEY REFERENCES jobs(id), key_hash TEXT NOT NULL UNIQUE,
                    fingerprint TEXT NOT NULL, binding_json TEXT NOT NULL,
                    payload_sha256 TEXT NOT NULL, capability_sha256 TEXT NOT NULL
                );
                CREATE TRIGGER IF NOT EXISTS job_submission_intents_no_update BEFORE UPDATE ON job_submission_intents
                BEGIN SELECT RAISE(ABORT, 'Job submission intents are immutable'); END;
                CREATE TRIGGER IF NOT EXISTS job_submission_intents_no_delete BEFORE DELETE ON job_submission_intents
                BEGIN SELECT RAISE(ABORT, 'Job submission intents are immutable'); END;
            """)
        finally:
            connection.close()
        self._secure_files()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.db_path), timeout=10, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout=10000")
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA synchronous=FULL")
        return connection

    def _secure_files(self) -> None:
        for suffix in ("", "-journal", "-wal", "-shm"):
            try:
                os.chmod(str(self.db_path) + suffix, 0o600)
            except FileNotFoundError:
                pass

    @staticmethod
    def _public(row: sqlite3.Row | dict) -> dict[str, Any]:
        try:
            binding = _binding(json.loads(row["binding_json"]))
            result = _result(json.loads(row["result_json"])) if row["result_json"] else None
            if (row["kind"] != "inspection" or row["state"] not in STATES
                    or row["error_code"] is not None and row["error_code"] not in ERROR_CODES
                    or type(row["attempt"]) is not int or row["attempt"] not in (0, 1)
                    or row["dispatched"] not in (0, 1)):
                raise ValueError
            _identifier(row["id"], _JOB_ID, "Stored job identifier is invalid")
            _identifier(row["payload_sha256"], _HEX64, "Stored job payload digest is invalid")
            _identifier(row["event_head"], _HEX64, "Stored job event digest is invalid")
            _time(row["created_ms"])
            _time(row["updated_ms"])
            if type(row["event_count"]) is not int or row["event_count"] < 0:
                raise ValueError
            if row["lease_expires_ms"] is not None:
                _time(row["lease_expires_ms"])
            return {"schema_version": 1, "id": row["id"], "kind": row["kind"], "state": row["state"],
                "binding": binding, "payload_sha256": row["payload_sha256"],
                "created_ms": row["created_ms"], "updated_ms": row["updated_ms"],
                "attempt": row["attempt"], "dispatched": bool(row["dispatched"]),
                "lease_expires_ms": row["lease_expires_ms"], "result": result,
                "error_code": row["error_code"], "event_count": row["event_count"], "event_head": row["event_head"]}
        except (ValueError, TypeError, KeyError, RecursionError, JobValidationError):
            raise JobIntegrityError("Stored job projection is invalid") from None

    @staticmethod
    def _read(connection: sqlite3.Connection, job_id: str) -> sqlite3.Row:
        row = connection.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        if row is None:
            raise JobNotFound("Job is unavailable")
        return row

    def _append(self, connection: sqlite3.Connection, job_id: str, event_type: str, now_ms: int) -> dict[str, Any]:
        row = self._read(connection, job_id)
        projection = self._public(row)
        projection.pop("event_count")
        projection.pop("event_head")
        event = {"schema_version": 1, "job_id": job_id, "sequence": row["event_count"] + 1,
            "event_type": event_type, "at_ms": now_ms, "previous_hash": row["event_head"], "job": projection}
        encoded = _canonical(event)
        event_hash = _digest(encoded)
        connection.execute("INSERT INTO job_events VALUES (?, ?, ?, ?)", (job_id, event["sequence"], encoded, event_hash))
        connection.execute("UPDATE jobs SET event_count=?, event_head=? WHERE id=?", (event["sequence"], event_hash, job_id))
        return self._public(self._read(connection, job_id))

    @staticmethod
    def _begin(connection: sqlite3.Connection) -> None:
        connection.execute("BEGIN IMMEDIATE")

    @staticmethod
    def _submission(kind: str, payload: dict[str, Any], idempotency_key: str,
                    binding: dict[str, Any]) -> tuple[str, str, str, str]:
        if kind != "inspection":
            raise JobValidationError("Job kind is unsupported")
        if type(payload) is not dict:
            raise JobValidationError("Job payload must be an object")
        encoded = _canonical(payload)
        if len(encoded.encode("utf-8")) > MAX_PAYLOAD_BYTES:
            raise JobValidationError("Job payload is excessive")
        checked_binding = _binding(binding)
        if (type(idempotency_key) is not str or not 1 <= len(idempotency_key.encode("utf-8")) <= 256
                or any(ord(character) < 33 or ord(character) > 126 for character in idempotency_key)):
            raise JobValidationError("Job idempotency key is invalid")
        key_hash = _digest(idempotency_key)
        payload_hash = _digest(encoded)
        binding_json = _canonical(checked_binding)
        return encoded, payload_hash, key_hash, binding_json

    def lookup_submission(self, kind: str, payload: dict[str, Any], *, idempotency_key: str,
                          binding: dict[str, Any]) -> dict[str, Any] | None:
        """Read an exact existing submission before checking today's engine state.

        Retrying a completed request returns its historical job reference. It
        never rebinds that result to a changed session or authorizes redispatch.
        """
        encoded, _, key_hash, binding_json = self._submission(kind, payload, idempotency_key, binding)
        connection = self._connect()
        try:
            connection.execute("BEGIN")
            row = connection.execute("SELECT * FROM jobs WHERE key_hash=?", (key_hash,)).fetchone()
            if row is None:
                return None
            self._verify_stored(connection, row, check_payload=True)
            if (row["kind"], row["payload_json"], row["binding_json"]) != (kind, encoded, binding_json):
                raise JobConflict("Idempotency key was already used for different input")
            return self._public(row)
        finally:
            connection.close()

    def lookup_intent(self, kind: str, *, idempotency_key: str, binding: dict[str, Any],
                      submission_fingerprint: str) -> dict[str, Any] | None:
        """Recover only the original admitted job without images or redispatch.

        The fingerprint is SHA256 of the original exact UTF-8 HTTP request
        bytes, verified at admission. Possession of the opaque key, binding and
        original fingerprint is a read-only recovery capability. It never
        validates a new payload or admits/claims/retries an external request.
        """
        _, _, key_hash, binding_json = self._submission(kind, {}, idempotency_key, binding)
        _identifier(submission_fingerprint, _HEX64, "Submission fingerprint is invalid")
        connection = self._connect()
        try:
            connection.execute("BEGIN")
            row = connection.execute("SELECT * FROM jobs WHERE key_hash=?", (key_hash,)).fetchone()
            if row is None:
                return None
            self._verify_stored(connection, row, check_payload=True)
            capability = connection.execute("SELECT * FROM job_submission_intents WHERE job_id=?", (row["id"],)).fetchone()
            if (capability is None or row["kind"] != kind or row["binding_json"] != binding_json
                    or not secrets.compare_digest(capability["fingerprint"], submission_fingerprint)):
                raise JobConflict("Submission recovery credentials do not match the original request")
            return self._public(row)
        finally:
            connection.close()

    def submit(self, kind: str, payload: dict[str, Any], *, idempotency_key: str,
               binding: dict[str, Any], now_ms: int,
               submission_fingerprint: str | None = None) -> dict[str, Any]:
        encoded, payload_hash, key_hash, binding_json = self._submission(kind, payload, idempotency_key, binding)
        if submission_fingerprint is not None:
            _identifier(submission_fingerprint, _HEX64, "Submission fingerprint is invalid")
        now_ms = _time(now_ms)
        connection = self._connect()
        try:
            self._begin(connection)
            old = connection.execute("SELECT * FROM jobs WHERE key_hash=?", (key_hash,)).fetchone()
            if old is not None:
                self._verify_stored(connection, old, check_payload=True)
                if (old["kind"], old["payload_json"], old["binding_json"]) != (kind, encoded, binding_json):
                    raise JobConflict("Idempotency key was already used for different input")
                result = self._public(old)
            else:
                job_id = secrets.token_hex(16)
                connection.execute("""INSERT INTO jobs
                    (id,kind,key_hash,payload_json,payload_sha256,binding_json,state,created_ms,updated_ms,event_head)
                    VALUES (?,?,?,?,?,?,'queued',?,?,?)""",
                    (job_id, kind, key_hash, encoded, payload_hash, binding_json, now_ms, now_ms, _GENESIS))
                if submission_fingerprint is not None:
                    capability = {"job_id": job_id, "key_hash": key_hash, "fingerprint": submission_fingerprint,
                                  "binding_json": binding_json, "payload_sha256": payload_hash}
                    connection.execute("""INSERT INTO job_submission_intents
                        (job_id,key_hash,fingerprint,binding_json,payload_sha256,capability_sha256) VALUES (?,?,?,?,?,?)""",
                        (*capability.values(), _digest(_canonical(capability))))
                result = self._append(connection, job_id, "submitted", now_ms)
            connection.commit()
            return result
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()
            self._secure_files()

    def get(self, job_id: str) -> dict[str, Any]:
        _identifier(job_id, _JOB_ID, "Job identifier is invalid")
        connection = self._connect()
        try:
            connection.execute("BEGIN")
            row = self._read(connection, job_id)
            self._verify_stored(connection, row)
            return self._public(row)
        finally:
            connection.close()

    def list_jobs(self, *, limit: int = 20) -> list[dict[str, Any]]:
        if type(limit) is not int or not 1 <= limit <= 100:
            raise JobValidationError("Job listing limit is invalid")
        connection = self._connect()
        try:
            connection.execute("BEGIN")
            rows = connection.execute("SELECT * FROM jobs ORDER BY created_ms DESC,id DESC LIMIT ?", (limit,)).fetchall()
            for row in rows:
                self._verify_stored(connection, row)
            return [self._public(row) for row in rows]
        finally:
            connection.close()

    def claim_next(self, worker_id: str, *, now_ms: int, lease_ms: int = 30_000) -> dict[str, Any] | None:
        _identifier(worker_id, _OPAQUE_ID, "Worker identifier is invalid")
        now_ms = _time(now_ms)
        self._lease_duration(lease_ms)
        expiry_ms = _time(now_ms + lease_ms)
        connection = self._connect()
        try:
            self._begin(connection)
            row = connection.execute("SELECT * FROM jobs WHERE state='queued' ORDER BY created_ms,id LIMIT 1").fetchone()
            if row is None:
                connection.commit()
                return None
            self._verify_stored(connection, row, check_payload=True)
            self._clock(row, now_ms)
            if _digest(row["payload_json"]) != row["payload_sha256"]:
                raise JobIntegrityError("Stored job payload changed")
            try:
                payload = json.loads(row["payload_json"])
                if type(payload) is not dict or _canonical(payload) != row["payload_json"]:
                    raise ValueError
            except (ValueError, TypeError, JobValidationError):
                raise JobIntegrityError("Stored job payload is invalid") from None
            token = secrets.token_hex(32)
            connection.execute("""UPDATE jobs SET state='running',updated_ms=?,attempt=1,
                worker_id=?,claim_token=?,lease_expires_ms=? WHERE id=?""",
                (now_ms, worker_id, token, expiry_ms, row["id"]))
            result = self._append(connection, row["id"], "claimed", now_ms)
            connection.commit()
            result["payload"] = payload
            result["lease"] = {"worker_id": worker_id, "claim_token": token, "lease_expires_ms": expiry_ms}
            return result
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()
            self._secure_files()

    @staticmethod
    def _lease_duration(lease_ms: int) -> None:
        if type(lease_ms) is not int or not 1 <= lease_ms <= MAX_LEASE_MS:
            raise JobValidationError("Job lease duration is invalid")

    @staticmethod
    def _clock(row: sqlite3.Row, now_ms: int) -> None:
        if now_ms < row["updated_ms"]:
            raise JobConflict("Job clock moved backwards")

    def _fenced(self, connection: sqlite3.Connection, job_id: str, token: str, now_ms: int) -> sqlite3.Row:
        _identifier(job_id, _JOB_ID, "Job identifier is invalid")
        _identifier(token, _HEX64, "Job claim token is invalid")
        row = self._read(connection, job_id)
        self._verify_stored(connection, row)
        self._clock(row, now_ms)
        if (row["state"] != "running" or not secrets.compare_digest(row["claim_token"] or "", token)
                or row["lease_expires_ms"] is None or now_ms >= row["lease_expires_ms"]):
            raise JobLeaseError("Job claim is unavailable or expired")
        return row

    def _worker_update(self, job_id: str, token: str, *, now_ms: int, event_type: str,
                       changes: dict[str, Any], dispatch_required: bool = False) -> dict[str, Any]:
        now_ms = _time(now_ms)
        connection = self._connect()
        try:
            self._begin(connection)
            row = self._fenced(connection, job_id, token, now_ms)
            if dispatch_required and not row["dispatched"]:
                raise JobConflict("Job dispatch was not recorded")
            if event_type == "dispatched" and row["dispatched"]:
                raise JobConflict("Job dispatch was already recorded")
            assignments = ["updated_ms=?"] + [key + "=?" for key in changes]
            connection.execute("UPDATE jobs SET " + ",".join(assignments) + " WHERE id=?",
                [now_ms] + list(changes.values()) + [job_id])
            result = self._append(connection, job_id, event_type, now_ms)
            connection.commit()
            return result
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()
            self._secure_files()

    def mark_dispatched(self, job_id: str, claim_token: str, *, now_ms: int) -> dict[str, Any]:
        return self._worker_update(job_id, claim_token, now_ms=now_ms, event_type="dispatched", changes={"dispatched": 1})

    def heartbeat(self, job_id: str, claim_token: str, *, now_ms: int, lease_ms: int = 30_000) -> dict[str, Any]:
        self._lease_duration(lease_ms)
        _time(now_ms)
        expiry_ms = _time(now_ms + lease_ms)
        return self._worker_update(job_id, claim_token, now_ms=now_ms, event_type="heartbeat", changes={"lease_expires_ms": expiry_ms})

    def complete(self, job_id: str, claim_token: str, result: dict[str, Any], *, now_ms: int) -> dict[str, Any]:
        checked = _result(result)
        return self._worker_update(job_id, claim_token, now_ms=now_ms, event_type="succeeded", dispatch_required=True,
            changes={"state": "succeeded", "result_json": _canonical(checked), "claim_token": None,
                "worker_id": None, "lease_expires_ms": None})

    def fail(self, job_id: str, claim_token: str, error_code: str, *, now_ms: int) -> dict[str, Any]:
        if type(error_code) is not str or error_code not in ERROR_CODES - {"interrupted_no_dispatch", "outcome_unknown"}:
            raise JobValidationError("Job error code is unsupported")
        return self._worker_update(job_id, claim_token, now_ms=now_ms, event_type="failed",
            changes={"state": "failed", "error_code": error_code, "claim_token": None,
                "worker_id": None, "lease_expires_ms": None})

    def cancel(self, job_id: str, *, now_ms: int) -> dict[str, Any]:
        _identifier(job_id, _JOB_ID, "Job identifier is invalid")
        now_ms = _time(now_ms)
        connection = self._connect()
        try:
            self._begin(connection)
            row = self._read(connection, job_id)
            self._verify_stored(connection, row)
            self._clock(row, now_ms)
            if row["state"] != "queued":
                raise JobConflict("Only queued jobs can be cancelled")
            connection.execute("UPDATE jobs SET state='cancelled',updated_ms=? WHERE id=?", (now_ms, job_id))
            result = self._append(connection, job_id, "cancelled", now_ms)
            connection.commit()
            return result
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()
            self._secure_files()

    def recover_expired(self, *, now_ms: int) -> list[dict[str, Any]]:
        now_ms = _time(now_ms)
        connection = self._connect()
        try:
            self._begin(connection)
            rows = connection.execute("SELECT * FROM jobs WHERE state='running' AND lease_expires_ms<=? ORDER BY created_ms,id", (now_ms,)).fetchall()
            result = []
            for row in rows:
                self._verify_stored(connection, row)
                self._clock(row, now_ms)
                error = "outcome_unknown" if row["dispatched"] else "interrupted_no_dispatch"
                connection.execute("""UPDATE jobs SET state='interrupted',updated_ms=?,error_code=?,
                    claim_token=NULL,worker_id=NULL,lease_expires_ms=NULL WHERE id=?""", (now_ms, error, row["id"]))
                result.append(self._append(connection, row["id"], "interrupted", now_ms))
            connection.commit()
            return result
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()
            self._secure_files()

    def events(self, job_id: str, *, limit: int = 200, after_sequence: int = 0) -> list[dict[str, Any]]:
        _identifier(job_id, _JOB_ID, "Job identifier is invalid")
        if type(limit) is not int or not 1 <= limit <= 200 or type(after_sequence) is not int or not 0 <= after_sequence < 2 ** 63:
            raise JobValidationError("Job event cursor is invalid")
        connection = self._connect()
        try:
            connection.execute("BEGIN")
            self._verify_stored(connection, self._read(connection, job_id))
            values = connection.execute("SELECT event_json,event_hash FROM job_events WHERE job_id=? AND sequence>? ORDER BY sequence LIMIT ?", (job_id, after_sequence, limit))
            return [{**json.loads(row["event_json"]), "hash": row["event_hash"]} for row in values]
        except (ValueError, TypeError):
            raise JobIntegrityError("Stored job journal is invalid") from None
        finally:
            connection.close()

    def verify(self, job_id: str) -> dict[str, Any]:
        """Recompute payload, event chain, legal transitions and final projection.

        An administrator who rewrites the database and every hash can forge a
        consistent journal. Verification makes no remote origin/time guarantee.
        """
        _identifier(job_id, _JOB_ID, "Job identifier is invalid")
        limits = ["Local hashes are not signatures or trusted timestamps.",
            "A dispatch marker cannot prove a provider received or completed a request.",
            "No interrupted external request is automatically replayed."]
        connection = self._connect()
        try:
            connection.execute("BEGIN")
            row = self._read(connection, job_id)
            count, head = self._verify_stored(connection, row, check_payload=True)
            return {"valid": True, "event_count": count, "event_head": head, "limits": limits}
        except JobNotFound:
            raise
        except (JobIntegrityError, JobValidationError, KeyError, ValueError, TypeError, RecursionError):
            return {"valid": False, "event_count": None, "event_head": None, "limits": limits}
        finally:
            connection.close()

    def _verify_stored(self, connection: sqlite3.Connection, row: sqlite3.Row, *, check_payload: bool = False) -> tuple[int, str]:
        """Validate within the caller's read/write transaction before exposure."""
        try:
            projection = self._public(row)
            capability = connection.execute("SELECT * FROM job_submission_intents WHERE job_id=?", (row["id"],)).fetchone()
            if capability is not None:
                expected_capability = {"job_id": row["id"], "key_hash": row["key_hash"],
                    "fingerprint": _identifier(capability["fingerprint"], _HEX64, "Stored submission fingerprint is invalid"),
                    "binding_json": row["binding_json"], "payload_sha256": row["payload_sha256"]}
                if (any(capability[key] != value for key, value in expected_capability.items())
                        or capability["capability_sha256"] != _digest(_canonical(expected_capability))):
                    raise JobIntegrityError("Stored submission recovery capability changed")
            if check_payload:
                payload = json.loads(row["payload_json"])
                if (type(payload) is not dict or _canonical(payload) != row["payload_json"]
                        or _digest(row["payload_json"]) != row["payload_sha256"]):
                    raise JobIntegrityError("Stored job payload changed")
            previous_hash = _GENESIS
            previous = None
            count = 0
            for stored in connection.execute("SELECT * FROM job_events WHERE job_id=? ORDER BY sequence", (row["id"],)):
                count += 1
                event = json.loads(stored["event_json"])
                if (type(event) is not dict or set(event) != {"schema_version", "job_id", "sequence", "event_type", "at_ms", "previous_hash", "job"}
                        or type(event["schema_version"]) is not int or event["schema_version"] != 1
                        or event["job_id"] != row["id"] or type(event["sequence"]) is not int or event["sequence"] != count
                        or stored["sequence"] != count or event["previous_hash"] != previous_hash
                        or _digest(_canonical(event)) != stored["event_hash"]):
                    raise JobIntegrityError("Stored job journal changed")
                _time(event["at_ms"])
                self._verify_transition(previous, event)
                previous, previous_hash = event, stored["event_hash"]
            expected = deepcopy(projection)
            expected.pop("event_count")
            expected.pop("event_head")
            if count == 0 or count != row["event_count"] or previous_hash != row["event_head"] or previous["job"] != expected:
                raise JobIntegrityError("Stored job projection differs from its journal")
            return count, previous_hash
        except (JobValidationError, KeyError, ValueError, TypeError, RecursionError):
            raise JobIntegrityError("Stored job journal is invalid") from None

    @staticmethod
    def _verify_transition(previous: dict[str, Any] | None, event: dict[str, Any]) -> None:
        current = event["job"]
        if type(current) is not dict or current.get("updated_ms") != event["at_ms"]:
            raise JobIntegrityError("Stored job event projection is invalid")
        required = {"schema_version", "id", "kind", "state", "binding", "payload_sha256", "created_ms", "updated_ms",
            "attempt", "dispatched", "lease_expires_ms", "result", "error_code"}
        if (set(current) != required or current["id"] != event["job_id"] or type(current["schema_version"]) is not int
                or current["schema_version"] != 1 or current["kind"] != "inspection"):
            raise JobIntegrityError("Stored job event projection is invalid")
        _binding(current["binding"])
        _identifier(current["payload_sha256"], _HEX64, "Stored job hash is invalid")
        _time(current["created_ms"])
        _time(current["updated_ms"])
        if current["result"] is not None:
            _result(current["result"])
        if current["error_code"] is not None and current["error_code"] not in ERROR_CODES:
            raise JobIntegrityError("Stored job error code is invalid")
        if type(current["attempt"]) is not int or type(current["dispatched"]) is not bool:
            raise JobIntegrityError("Stored job execution flags are invalid")
        if previous is None:
            if (event["event_type"] != "submitted" or current["state"] != "queued" or current["attempt"] != 0
                    or current["dispatched"] or current["created_ms"] != current["updated_ms"]
                    or current["lease_expires_ms"] is not None or current["result"] is not None or current["error_code"] is not None):
                raise JobIntegrityError("Stored job genesis is invalid")
            return
        old = previous["job"]
        if event["at_ms"] < previous["at_ms"] or any(current[key] != old[key] for key in ("id", "kind", "binding", "payload_sha256", "created_ms", "schema_version")):
            raise JobIntegrityError("Stored job binding or clock changed")
        event_type = event["event_type"]
        allowed = {("queued", "claimed"): "running", ("queued", "cancelled"): "cancelled",
            ("running", "dispatched"): "running", ("running", "heartbeat"): "running",
            ("running", "succeeded"): "succeeded", ("running", "failed"): "failed",
            ("running", "interrupted"): "interrupted"}
        if allowed.get((old["state"], event_type)) != current["state"]:
            raise JobIntegrityError("Stored job transition is invalid")
        if current["attempt"] != (1 if event_type == "claimed" else old["attempt"]):
            raise JobIntegrityError("Stored job attempt changed")
        if current["dispatched"] != (True if event_type == "dispatched" else old["dispatched"]):
            raise JobIntegrityError("Stored job dispatch changed")
        if current["state"] == "running":
            expiry = current["lease_expires_ms"]
            if type(expiry) is not int or not current["updated_ms"] < expiry <= current["updated_ms"] + MAX_LEASE_MS:
                raise JobIntegrityError("Stored job lease is invalid")
            if current["result"] is not None or current["error_code"] is not None:
                raise JobIntegrityError("Running job has terminal fields")
            if event_type == "dispatched" and (old["dispatched"] or expiry != old["lease_expires_ms"]):
                raise JobIntegrityError("Stored job dispatch event is invalid")
        elif current["lease_expires_ms"] is not None:
            raise JobIntegrityError("Terminal job retains a lease")
        if event_type in ("heartbeat", "dispatched", "succeeded", "failed") and event["at_ms"] >= old["lease_expires_ms"]:
            raise JobIntegrityError("Stored job used an expired claim")
        if event_type == "succeeded" and (not old["dispatched"] or current["result"] is None or current["error_code"] is not None):
            raise JobIntegrityError("Stored job completion is invalid")
        if event_type == "failed" and (current["error_code"] not in ERROR_CODES - {"outcome_unknown", "interrupted_no_dispatch"} or current["result"] is not None):
            raise JobIntegrityError("Stored job failure is invalid")
        if event_type == "interrupted" and (event["at_ms"] < old["lease_expires_ms"] or current["result"] is not None
                or current["error_code"] != ("outcome_unknown" if old["dispatched"] else "interrupted_no_dispatch")):
            raise JobIntegrityError("Stored job interruption is invalid")
        if event_type == "cancelled" and (current["result"] is not None or current["error_code"] is not None):
            raise JobIntegrityError("Stored job cancellation is invalid")
