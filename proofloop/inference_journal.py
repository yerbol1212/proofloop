"""Private, persistent attempt admission around the frozen Meta adapter.

Reservations consume allowance even if a process dies before sending bytes.
An uncertain request is never refunded or automatically retried. This enforces
request counts, not provider billing, calibrated model quality or dollar caps.
No credentials, prompts, images or provider message bodies enter this journal.
"""
from __future__ import annotations

from contextlib import closing
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import sqlite3
import time
from typing import Any, Callable

from .meta import MetaAPIError, MetaClient, SAM_MODEL, SPARK_MODEL


class AttemptJournal:
    def __init__(self, path: str | Path, *, request_limit: int = 120):
        if type(request_limit) is not int or not 1 <= request_limit <= 100_000:
            raise ValueError("Use a request allowance from 1 to 100000")
        self.path, self.request_limit = Path(path), request_limit
        if self.path.is_symlink() or self.path.exists() and not self.path.is_file():
            raise ValueError("Attempt journal must be a regular private file")
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        descriptor = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o600)
        os.close(descriptor)
        os.chmod(self.path, 0o600)
        with closing(self._connect()) as db:
            db.execute("PRAGMA journal_mode=DELETE")
            db.execute("CREATE TABLE IF NOT EXISTS configuration (singleton INTEGER PRIMARY KEY CHECK(singleton=1), request_limit INTEGER NOT NULL)")
            db.execute("CREATE TABLE IF NOT EXISTS operation_limits (operation_id TEXT PRIMARY KEY, request_limit INTEGER NOT NULL)")
            db.execute("INSERT OR IGNORE INTO configuration VALUES (1, ?)", (request_limit,))
            if db.execute("SELECT request_limit FROM configuration WHERE singleton=1").fetchone()[0] != request_limit:
                raise ValueError("The persistent journal allowance cannot be changed by reopening it")
            db.execute("""CREATE TABLE IF NOT EXISTS attempts (
                id TEXT PRIMARY KEY, operation_id TEXT NOT NULL, endpoint TEXT NOT NULL,
                model TEXT, request_sha256 TEXT NOT NULL, started_ms INTEGER NOT NULL,
                finished_ms INTEGER, state TEXT NOT NULL, usage_json TEXT NOT NULL)""")
            db.execute("CREATE INDEX IF NOT EXISTS attempts_operation ON attempts(operation_id)")

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(str(self.path), timeout=10, isolation_level=None)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA busy_timeout=10000")
        db.execute("PRAGMA synchronous=FULL")
        return db

    def reserve(self, operation_id: str, endpoint: str, payload: dict[str, Any] | None,
                *, operation_limit: int) -> str:
        if (not isinstance(operation_id, str) or not re.fullmatch(r"[0-9a-f]{32}", operation_id)
                or endpoint not in ("responses", "models") or type(operation_limit) is not int
                or not 1 <= operation_limit <= 13):
            raise ValueError("Invalid bounded provider operation")
        model = (payload or {}).get("model")
        if model not in (None, SPARK_MODEL, SAM_MODEL):
            raise ValueError("Unsupported provider model")
        canonical = json.dumps({"endpoint": endpoint, "payload": payload}, sort_keys=True,
                               separators=(",", ":"), allow_nan=False).encode()
        attempt_id = secrets.token_hex(16)
        db = self._connect()
        try:
            db.execute("BEGIN IMMEDIATE")
            db.execute("INSERT OR IGNORE INTO operation_limits VALUES (?,?)", (operation_id, operation_limit))
            if db.execute("SELECT request_limit FROM operation_limits WHERE operation_id=?", (operation_id,)).fetchone()[0] != operation_limit:
                raise ValueError("The persistent operation allowance cannot be changed")
            total = db.execute("SELECT count(*) FROM attempts").fetchone()[0]
            operation = db.execute("SELECT count(*) FROM attempts WHERE operation_id=?", (operation_id,)).fetchone()[0]
            if total >= self.request_limit or operation >= operation_limit:
                raise MetaAPIError("Persistent request allowance reached", code="persistent_request_limit")
            db.execute("INSERT INTO attempts VALUES (?,?,?,?,?,?,NULL,'reserved','{}')",
                       (attempt_id, operation_id, endpoint, model, hashlib.sha256(canonical).hexdigest(),
                        int(time.time() * 1000)))
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()
        return attempt_id

    @staticmethod
    def _usage(value: Any) -> dict[str, int]:
        # Arbitrary provider strings and nested objects never cross this boundary.
        if not isinstance(value, dict):
            return {}
        return {key: value[key] for key in ("input_tokens", "output_tokens", "total_tokens")
                if type(value.get(key)) is int and 0 <= value[key] < 2 ** 63}

    def settle(self, attempt_id: str, *, state: str, usage: Any = None) -> None:
        if state not in ("ok", "rejected", "uncertain"):
            raise ValueError("Invalid attempt outcome")
        with closing(self._connect()) as db:
            changed = db.execute("UPDATE attempts SET state=?,finished_ms=?,usage_json=? WHERE id=? AND state='reserved'",
                (state, int(time.time() * 1000), json.dumps(self._usage(usage), sort_keys=True), attempt_id)).rowcount
            if changed != 1:
                raise ValueError("Provider attempt is missing or already settled")

    def summary(self, operation_id: str | None = None) -> dict[str, Any]:
        with closing(self._connect()) as db:
            db.execute("BEGIN")  # One coherent snapshot while other workers reserve.
            rows = db.execute("SELECT state,usage_json FROM attempts" +
                              (" WHERE operation_id=?" if operation_id is not None else ""),
                              (operation_id,) if operation_id is not None else ()).fetchall()
            total = db.execute("SELECT count(*) FROM attempts").fetchone()[0]
        counts = {name: sum(row["state"] == name for row in rows)
                  for name in ("reserved", "ok", "rejected", "uncertain")}
        reported = {key: sum(json.loads(row["usage_json"]).get(key, 0) for row in rows)
                    for key in ("input_tokens", "output_tokens", "total_tokens")}
        return {"request_limit": self.request_limit, "total_reserved_attempts": total,
                "remaining_attempts": max(0, self.request_limit - total),
                "operation_attempts": len(rows), "outcomes": counts, "reported_tokens": reported,
                "unsettled_outcome_is_unknown": counts["reserved"] > 0,
                "automatic_retry": False, "hard_dollar_limit": False,
                "allowance_scope": "this_local_journal_all_processes_and_restarts"}


class JournaledMetaClient(MetaClient):
    """Same provider adapter, with durable admission before every HTTP attempt."""
    def __init__(self, api_key: str, *, journal: AttemptJournal, operation_id: str,
                 guard: Callable[[], None], **kwargs: Any):
        if kwargs.get("retries", 0) != 0:
            raise ValueError("Journaled inference never automatically retries")
        super().__init__(api_key, **kwargs)
        self._journal, self._operation_id, self._guard = journal, operation_id, guard

    def _request(self, method: str, endpoint: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        self._guard()  # Lease and binding must still be valid before each call.
        attempt_id = self._journal.reserve(self._operation_id, endpoint, payload,
                                           operation_limit=self.max_requests)
        initial = len(self.ledger)
        try:
            response = super()._request(method, endpoint, payload)
        except BaseException as error:
            entry = self.ledger[-1] if len(self.ledger) > initial else {}
            # A transport timeout or interrupted Python process cannot tell us
            # whether Meta completed and billed the request.
            uncertain = not isinstance(error, MetaAPIError) or error.code == "network_error"
            self._journal.settle(attempt_id, state="uncertain" if uncertain else "rejected",
                                 usage=entry.get("usage"))
            raise
        self._journal.settle(attempt_id, state="ok", usage=response.get("usage"))
        return response
