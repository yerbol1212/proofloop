"""Shared paid-call admission and bounded, private operational diagnostics.

The immutable attempt journal remains authoritative for request reservations.
Telemetry is a separate allowlisted projection: no prompts, images, credentials,
paths or provider error text are accepted. Export is local JSON only.
"""
from __future__ import annotations

import argparse
from contextlib import closing, contextmanager
import json
import math
import os
from pathlib import Path
import re
import secrets
import shutil
import sqlite3
import tempfile
import threading
import time
from typing import Any, Callable

from .bounded_spark import BoundedSparkClient
from .inference_journal import AttemptJournal
from .meta import MetaAPIError, SAM_MODEL, SPARK_MODEL, load_env_file


VERSION = "local_runtime_control_v1"
SCHEMA_VERSION = "1.0"
MAX_SPANS = 1024
MAX_PUBLIC_SPANS = 100
MAX_METRIC_ROWS = 2048
MIN_FREE_BYTES = 32 * 1024 * 1024
KINDS = frozenset({"compiler", "inspection", "shadow_tracking", "system"})
STAGES = frozenset({"admission", "queue_wait", "job_validation", "provider_spark", "provider_sam",
                    "provider_other", "verification", "replay", "evidence_request", "artifact_publication", "execution", "worker", "result_read"})
OUTCOMES = frozenset({"ok", "rejected", "uncertain", "error", "cancelled"})
CODES = frozenset({"none", "operation_error", "provider_rejected", "provider_uncertain", "validation_error",
                   "worker_unavailable", "worker_closed", "worker_failed", "journal_unavailable",
                   "storage_unavailable", "storage_capacity_low", "telemetry_unavailable",
                   "request_allowance_exhausted", "stale_binding", "model_error", "artifact_error",
                   "worker_error", "lease_lost", "integrity_error", "verdict_pass", "verdict_fail", "verdict_unknown",
                   "missing_current_evidence", "stale_evidence", "not_detected", "multiple_candidates",
                   "identity_ambiguous", "partial_visibility", "low_confidence", "boundary_unconfirmed",
                   "invalid_detection", "mask_unavailable", "temporal_inconsistency", "insufficient_distinct_views",
                   "calibration_required", "unsupported_observation", "unsupported_stability_requirement",
                   "absence_validation_required", "temporal_history_required"})
_HEX32 = re.compile(r"[0-9a-f]{32}\Z")


class RuntimeUnavailable(RuntimeError):
    def __init__(self, code: str = "operation_error"):
        self.code = code if code in CODES else "operation_error"
        super().__init__("The local runtime is unavailable")


def _identifier(value):
    if type(value) is not str or not _HEX32.fullmatch(value):
        raise ValueError("Use an opaque runtime operation identifier")
    return value


class RuntimeTelemetry:
    """Bounded durable spans and finite-label lifetime counters, with no network."""
    def __init__(self, path: str | Path):
        self.path = Path(path)
        if self.path.is_symlink() or self.path.exists() and not self.path.is_file():
            raise ValueError("Telemetry must use a private regular file")
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        descriptor = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o600)
        os.close(descriptor)
        os.chmod(self.path, 0o600)
        with closing(self._connect()) as db:
            db.execute("PRAGMA journal_mode=DELETE")
            db.executescript("""
                CREATE TABLE IF NOT EXISTS spans (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT, span_id TEXT NOT NULL,
                    operation_id TEXT NOT NULL, operation_kind TEXT NOT NULL, stage TEXT NOT NULL,
                    outcome TEXT NOT NULL, code TEXT NOT NULL, model TEXT NOT NULL,
                    started_unix_ns INTEGER NOT NULL, duration_ns INTEGER NOT NULL);
                CREATE TABLE IF NOT EXISTS counters (
                    operation_kind TEXT NOT NULL, stage TEXT NOT NULL, outcome TEXT NOT NULL,
                    code TEXT NOT NULL, model TEXT NOT NULL, observations INTEGER NOT NULL,
                    total_duration_ns INTEGER NOT NULL,
                    PRIMARY KEY(operation_kind, stage, outcome, code, model));
            """)

    def _connect(self):
        db = sqlite3.connect(str(self.path), timeout=2, isolation_level=None)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA synchronous=FULL")
        db.execute("PRAGMA busy_timeout=2000")
        return db

    def health(self):
        if self.path.is_symlink() or not self.path.is_file():
            raise OSError("Telemetry unavailable")
        with closing(self._connect()) as db:
            if (db.execute("SELECT count(*) FROM counters").fetchone()[0] > MAX_METRIC_ROWS
                    or db.execute("SELECT count(*) FROM spans").fetchone()[0] > MAX_SPANS):
                raise ValueError("Telemetry exceeds its bounded projection")

    @staticmethod
    def _validate_labels(row):
        if (row["operation_kind"] not in KINDS or row["stage"] not in STAGES
                or row["outcome"] not in OUTCOMES or row["code"] not in CODES
                or row["model"] not in {"none", "spark", "sam"}):
            raise ValueError("Stored telemetry labels are invalid")

    def record(self, operation_id, kind, stage, *, outcome="ok", code="none", model="none",
               started_unix_ns=None, duration_ns=0):
        _identifier(operation_id)
        if kind not in KINDS or stage not in STAGES or outcome not in OUTCOMES or model not in {"none", "spark", "sam"}:
            raise ValueError("Use only supported redacted telemetry labels")
        # Arbitrary exception text cannot become either a label or span value.
        code = code if code in CODES else "operation_error"
        started = time.time_ns() if started_unix_ns is None else started_unix_ns
        if (type(started) is not int or not 0 <= started < 2 ** 63
                or type(duration_ns) is not int or not 0 <= duration_ns < 2 ** 63):
            raise ValueError("Telemetry timing must be a nonnegative bounded integer")
        values = (secrets.token_hex(8), operation_id, kind, stage, outcome, code, model, started, duration_ns)
        with closing(self._connect()) as db:
            db.execute("BEGIN IMMEDIATE")
            try:
                if db.execute("SELECT count(*) FROM counters").fetchone()[0] >= MAX_METRIC_ROWS:
                    old = db.execute("SELECT 1 FROM counters WHERE operation_kind=? AND stage=? AND outcome=? AND code=? AND model=?",
                                     (kind, stage, outcome, code, model)).fetchone()
                    if old is None:
                        raise ValueError("Telemetry label capacity reached")
                db.execute("INSERT INTO spans(span_id,operation_id,operation_kind,stage,outcome,code,model,started_unix_ns,duration_ns) VALUES (?,?,?,?,?,?,?,?,?)", values)
                db.execute("""INSERT INTO counters VALUES (?,?,?,?,?,1,?)
                    ON CONFLICT(operation_kind,stage,outcome,code,model) DO UPDATE SET
                    observations=observations+1,total_duration_ns=total_duration_ns+excluded.total_duration_ns""",
                    (kind, stage, outcome, code, model, duration_ns))
                db.execute("DELETE FROM spans WHERE sequence NOT IN (SELECT sequence FROM spans ORDER BY sequence DESC LIMIT ?)", (MAX_SPANS,))
                db.commit()
            except BaseException:
                db.rollback()
                raise

    def snapshot(self, *, span_limit=MAX_PUBLIC_SPANS):
        if type(span_limit) is not int or not 0 <= span_limit <= MAX_PUBLIC_SPANS:
            raise ValueError("Use a bounded runtime trace limit")
        self.health()
        with closing(self._connect()) as db:
            db.execute("BEGIN")
            spans = [dict(row) for row in db.execute("SELECT * FROM spans ORDER BY sequence DESC LIMIT ?", (MAX_SPANS,))]
            counters = [dict(row) for row in db.execute("SELECT * FROM counters ORDER BY operation_kind,stage,outcome,code,model LIMIT ?", (MAX_METRIC_ROWS + 1,))]
        if len(counters) > MAX_METRIC_ROWS:
            raise ValueError("Telemetry label capacity exceeded")
        for span in spans:
            self._validate_labels(span)
            _identifier(span["operation_id"])
            if (type(span["span_id"]) is not str or re.fullmatch(r"[0-9a-f]{16}", span["span_id"]) is None
                    or any(type(span[key]) is not int or not 0 <= span[key] < 2 ** 63
                           for key in ("sequence", "started_unix_ns", "duration_ns"))):
                raise ValueError("Stored telemetry timing or correlation is invalid")
        for row in counters:
            self._validate_labels(row)
            if any(type(row[key]) is not int or not 0 <= row[key] < 2 ** 63
                   for key in ("observations", "total_duration_ns")):
                raise ValueError("Stored telemetry counters are invalid")
        distributions = {}
        for span in spans:
            distributions.setdefault((span["operation_kind"], span["stage"]), []).append(span["duration_ns"])
        metrics = []
        for row in counters:
            samples = sorted(distributions.get((row["operation_kind"], row["stage"]), []))
            def percentile(fraction):
                return round(samples[max(0, math.ceil(len(samples) * fraction) - 1)] / 1_000_000, 3) if samples else None
            metrics.append({**{key: row[key] for key in ("operation_kind", "stage", "outcome", "code", "model", "observations")},
                "total_duration_ms": round(row["total_duration_ns"] / 1_000_000, 3),
                "stage_window_p50_ms": percentile(.5), "stage_window_p95_ms": percentile(.95),
                "stage_window_samples": len(samples)})
        return {"schema_version": SCHEMA_VERSION, "method_version": VERSION, "span_capacity": MAX_SPANS,
                "retained_span_count": len(spans), "metrics_scope": "lifetime_counters_bounded_recent_stage_latency_window",
                "metrics": metrics, "recent_spans": spans[:span_limit], "contains_model_inputs": False,
                "network_export_enabled": False, "physical_action_authorized": False}

    def export_otel(self):
        """Return OpenTelemetry JSON trace fields locally; send nothing externally."""
        spans = self.snapshot()["recent_spans"]
        projected = []
        for span in spans:
            attrs = {"proofloop.operation_kind": span["operation_kind"], "proofloop.stage": span["stage"],
                     "proofloop.outcome": span["outcome"], "proofloop.code": span["code"], "proofloop.model": span["model"]}
            projected.append({"traceId": span["operation_id"], "spanId": span["span_id"],
                "name": "proofloop." + span["stage"], "kind": 1,
                "startTimeUnixNano": str(span["started_unix_ns"]),
                "endTimeUnixNano": str(span["started_unix_ns"] + span["duration_ns"]),
                "attributes": [{"key": key, "value": {"stringValue": value}} for key, value in attrs.items()],
                "status": {"code": 1 if span["outcome"] == "ok" else 2}})
        return {"resourceSpans": [{"resource": {"attributes": [{"key": "service.name", "value": {"stringValue": "proofloop-local"}}]},
                "scopeSpans": [{"scope": {"name": "proofloop.runtime", "version": VERSION}, "spans": projected}]}]}


class _OperationJournal:
    """Preserve a one-call adapter while sharing a larger bounded parent job."""
    def __init__(self, journal, operation_id, *, local_limit, operation_limit):
        self.journal, self.operation_id = journal, operation_id
        self.local_limit, self.operation_limit = local_limit, operation_limit

    def reserve(self, operation_id, endpoint, payload, *, operation_limit):
        if operation_id != self.operation_id or operation_limit != self.local_limit:
            raise ValueError("Runtime client allowance differs from its admitted operation")
        return self.journal.reserve(operation_id, endpoint, payload, operation_limit=self.operation_limit)

    def settle(self, *args, **kwargs):
        return self.journal.settle(*args, **kwargs)


class RuntimeMetaClient(BoundedSparkClient):
    def __init__(self, api_key, *, runtime_control, operation_kind, **kwargs):
        self.runtime_control, self.operation_kind = runtime_control, operation_kind
        super().__init__(api_key, **kwargs)

    def _request(self, method, endpoint, payload=None):
        model = (payload or {}).get("model")
        label = "spark" if model == SPARK_MODEL else "sam" if model == SAM_MODEL else "none"
        stage = "provider_" + label if label != "none" else "provider_other"
        with self.runtime_control.span(self._operation_id, self.operation_kind, stage, model=label):
            return super()._request(method, endpoint, payload)


class RuntimeCompilerClient(RuntimeMetaClient):
    """Concrete production compiler type; injected test factories stay distinct."""
    runtime_origin = "live_meta_api"


class RuntimeControl:
    def __init__(self, journal: AttemptJournal, artifact_root: str | Path, *, env_file: str | Path | None = None,
                 telemetry: RuntimeTelemetry | None = None, minimum_free_bytes=MIN_FREE_BYTES):
        self.journal, self.artifact_root = journal, Path(artifact_root)
        if self.artifact_root.is_symlink() or self.artifact_root.exists() and not self.artifact_root.is_dir():
            raise ValueError("Use a private regular runtime directory")
        self.artifact_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        if type(minimum_free_bytes) is not int or minimum_free_bytes < 0:
            raise ValueError("Use a nonnegative runtime storage reserve")
        self.minimum_free_bytes = minimum_free_bytes
        self.env_file = Path(env_file) if env_file is not None else None
        self.telemetry = telemetry or RuntimeTelemetry(self.artifact_root / "telemetry.sqlite3")
        self._workers = {}
        self._lock = threading.RLock()
        self._telemetry_error = False

    def bind_worker(self, kind: str, probe: Callable[[], dict[str, Any]]):
        if kind not in {"inspection", "shadow_tracking"} or not callable(probe):
            raise ValueError("Register a supported local worker health probe")
        with self._lock:
            if kind in self._workers and self._workers[kind] != probe:
                raise ValueError("A runtime worker probe is already registered")
            self._workers[kind] = probe

    def readiness(self, kind: str | None = None):
        if kind is not None and kind not in KINDS:
            raise ValueError("Use a supported runtime operation kind")
        reasons, checks, remaining = [], {}, None
        try:
            summary = self.journal.summary()
            remaining = summary["remaining_attempts"]
            if type(remaining) is not int or remaining < 0:
                raise ValueError
            checks["journal_available"] = True
            checks["request_allowance_available"] = remaining > 0
            if remaining == 0:
                reasons.append("request_allowance_exhausted")
        except (OSError, sqlite3.Error, ValueError, TypeError, KeyError):
            checks.update(journal_available=False, request_allowance_available=False)
            reasons.append("journal_unavailable")
        try:
            if self.artifact_root.is_symlink() or not self.artifact_root.is_dir():
                raise OSError
            if shutil.disk_usage(self.artifact_root).free < self.minimum_free_bytes:
                checks["storage_available"] = False
                reasons.append("storage_capacity_low")
            else:
                descriptor, probe = tempfile.mkstemp(prefix=".runtime-probe-", dir=self.artifact_root)
                try:
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
                    os.unlink(probe)
                checks["storage_available"] = True
        except (OSError, ValueError):
            checks["storage_available"] = False
            reasons.append("storage_unavailable")
        try:
            if self._telemetry_error:
                raise OSError
            self.telemetry.health()
            checks["telemetry_available"] = True
        except (OSError, sqlite3.Error, ValueError):
            checks["telemetry_available"] = False
            reasons.append("telemetry_unavailable")
        with self._lock:
            workers = list(self._workers.items())
        selected = [(name, probe) for name, probe in workers if kind is None or name == kind]
        if kind in {"inspection", "shadow_tracking"} and not selected:
            checks[kind + "_worker_available"] = False
            reasons.append("worker_unavailable")
        for name, probe in selected:
            try:
                state = probe()
                available = state.get("worker_available") is True
                code = state.get("code", "worker_unavailable")
            except Exception:
                available, code = False, "worker_unavailable"
            checks[name + "_worker_available"] = available
            if not available:
                reasons.append(code if code in {"worker_unavailable", "worker_closed", "worker_failed"} else "worker_unavailable")
        return {"schema_version": SCHEMA_VERSION, "method_version": VERSION, "ready": not reasons,
                "operation_kind": kind or "all_registered", "checks": checks,
                "reason_codes": list(dict.fromkeys(reasons)), "remaining_attempts": remaining,
                "provider_connectivity_tested": False, "physical_action_authorized": False}

    def require_ready(self, kind):
        state = self.readiness(kind)
        if not state["ready"]:
            raise RuntimeUnavailable(state["reason_codes"][0])

    def record(self, operation_id, kind, stage, **kwargs):
        try:
            self.telemetry.record(operation_id, kind, stage, **kwargs)
        except (OSError, sqlite3.Error, ValueError, TypeError):
            # Never replace the result of a possibly charged request with a
            # telemetry exception. Subsequent admissions expose unavailable.
            self._telemetry_error = True

    @contextmanager
    def span(self, operation_id, kind, stage, *, model="none"):
        started, monotonic = time.time_ns(), time.monotonic_ns()
        outcome, code = "ok", "none"
        try:
            yield
        except BaseException as error:
            if isinstance(error, RuntimeUnavailable):
                outcome, code = "rejected", error.code
            elif isinstance(error, MetaAPIError):
                outcome, code = (("uncertain", "provider_uncertain") if error.code == "network_error"
                                 else ("rejected", "provider_rejected"))
            else:
                outcome, code = "error", "operation_error"
            raise
        finally:
            self.record(operation_id, kind, stage, model=model, outcome=outcome, code=code,
                started_unix_ns=started, duration_ns=max(0, time.monotonic_ns() - monotonic))

    def _client(self, cls, operation_id, kind, guard, *, operation_limit, **kwargs):
        _identifier(operation_id)
        local_limit = kwargs.get("max_requests", operation_limit)
        if (type(local_limit) is not int or not 1 <= local_limit <= operation_limit <= 13
                or kwargs.get("retries", 0) != 0):
            raise ValueError("Use bounded runtime requests with no automatic retries")
        self.require_ready(kind)
        if self.env_file is not None:
            load_env_file(self.env_file)
        def admission_guard():
            self.require_ready(kind)
            guard()
        kwargs.update(max_requests=local_limit, retries=0)
        return cls.from_env(runtime_control=self, operation_kind=kind,
            journal=_OperationJournal(self.journal, operation_id, local_limit=local_limit, operation_limit=operation_limit),
            operation_id=operation_id, guard=admission_guard, **kwargs)

    def compiler_client(self):
        return self._client(RuntimeCompilerClient, secrets.token_hex(16), "compiler", lambda: None,
                            operation_limit=1, max_requests=1, retries=0, timeout=45)

    def inspection_client(self, operation_id, guard, **kwargs):
        limit = kwargs.get("max_requests", 13)
        return self._client(RuntimeMetaClient, operation_id, "inspection", guard, operation_limit=limit, **kwargs)

    def tracking_client(self, operation_id, guard, **kwargs):
        kwargs.setdefault("max_requests", 1)
        kwargs.setdefault("timeout", 180)
        return self._client(RuntimeMetaClient, operation_id, "shadow_tracking", guard, operation_limit=10, **kwargs)

    def snapshot(self):
        try:
            telemetry = self.telemetry.snapshot()
        except (OSError, sqlite3.Error, ValueError):
            self._telemetry_error = True
            telemetry = {"schema_version": SCHEMA_VERSION, "available": False, "code": "telemetry_unavailable"}
        return {"schema_version": SCHEMA_VERSION, "method_version": VERSION,
                "readiness": self.readiness(), "telemetry": telemetry,
                "automatic_retry": False, "physical_action_authorized": False}

    def export_otel(self):
        return self.telemetry.export_otel()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Read local redacted runtime telemetry; never contact models")
    parser.add_argument("telemetry", type=Path)
    parser.add_argument("--format", choices=("json", "otel"), default="json")
    args = parser.parse_args(argv)
    if not args.telemetry.is_file() or args.telemetry.is_symlink():
        parser.error("Choose an existing regular local telemetry database")
    telemetry = RuntimeTelemetry(args.telemetry)
    print(json.dumps(telemetry.export_otel() if args.format == "otel" else telemetry.snapshot(), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
