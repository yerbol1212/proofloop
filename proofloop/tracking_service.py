"""Local durable, bounded tracking diagnostics; never placement authority.

HTTP callers supply opaque asset IDs, not paths or URLs. Each claimed job is
executed once; lease loss is terminal. Production supplies a shared-journal
client factory. Cached and injected callbacks have explicit origins.
"""
from __future__ import annotations

from copy import deepcopy
from contextlib import closing, nullcontext
import hashlib
import json
import math
import os
from pathlib import Path
import re
import secrets
import sqlite3
import subprocess
import tempfile
import threading
import time

from .core import ValidationError
from . import meta
from . import video_tracking as video
from . import visibility_guard as visibility
from . import gripper_veto as warnings
from . import gripper_veto_experiment as veto
from . import tracking_experiment as original
from . import tracking_geometry as geometry
from . import tracking_mask_replay as masks
from . import confirmation_policy as current

VERSION = "shadow_tracking_diagnostic_v1"
MAX_ASSET_BYTES = 8 * 1024 * 1024
MAX_ASSETS = 64
MAX_REQUESTS = 10
_TRACE_GAPS = {
    "candidate_missing":"not_detected", "multiple_target_candidates":"multiple_candidates",
    "score_unreported":"low_confidence", "low_confidence":"low_confidence", "mask_unsupported":"mask_unavailable",
    "frame_boundary_unconfirmed":"boundary_unconfirmed", "reviewed_distractor_overlap":"boundary_unconfirmed",
    "full_visibility_not_supported":"partial_visibility", "visibility_unconfirmed":"partial_visibility",
    "identity_ambiguous":"identity_ambiguous", "identity_assessment_missing":"identity_ambiguous",
    "semantic_identity_unknown":"identity_ambiguous", "semantic_identity_contradicted":"unsupported_observation",
    "semantic_reacquisition_required":"identity_ambiguous", "current_semantic_evidence_required":"identity_ambiguous",
    "source_pixels_unbound":"invalid_detection", "source_view_reused":"insufficient_distinct_views",
    "source_spacing_insufficient":"insufficient_distinct_views", "distinct_stable_views_required":"insufficient_distinct_views",
    "track_identifier_changed":"temporal_inconsistency",
    "unchanged_baseline_positive_subset_required":"unsupported_observation",
}
_ID = re.compile(r"[0-9a-f]{32}\Z")
_HASH = re.compile(r"[0-9a-f]{64}\Z")
_SAFE = re.compile(r"[a-z][a-z0-9_-]{0,95}\Z")
_PROOF = {"frame_index", "timestamp_ms", "source_sha256", "frame_sha256", "track_id", "semantic_evidence_sha256"}
_FRAME = {"frame_index", "timestamp_ms", "source_sha256", "frame_sha256", "status", "track_id", "box",
          "gap_codes", "warning", "confirmation_evidence"}
_RESULT = {"schema_version", "mode", "inference_origin", "status", "frames", "request_attempts",
           "estimated_usd", "unreported_charge_possible", "trust_mode", "physical_action_authorized", "changes_live_placement_acceptance"}


class TrackingConflict(ValidationError):
    pass


class TrackingNotFound(ValidationError):
    pass


def _encoded(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def _digest(value):
    return hashlib.sha256(_encoded(value)).hexdigest()


def _atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor, temporary = tempfile.mkstemp(prefix=".tracking-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(_encoded(value)); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _read(path):
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 8 * 1024 * 1024:
        raise ValidationError("Tracking record is unavailable or excessive")
    def unique(rows):
        result = {}
        for key, value in rows:
            if key in result:
                raise ValueError("duplicate")
            result[key] = value
        return result
    try:
        return json.loads(path.read_text(), object_pairs_hook=unique,
                          parse_constant=lambda _: (_ for _ in ()).throw(ValueError()))
    except (ValueError, OSError, UnicodeError):
        raise ValidationError("Tracking record is invalid") from None


def _integer(value, maximum=2**53):
    return type(value) is int and 0 <= value <= maximum


def public_result(result):
    """Closed path-free DTO, validated again when reloading persisted results."""
    try:
        if (type(result) is not dict or set(result) != _RESULT or result["schema_version"] != VERSION
                or result["mode"] not in {"cached_demo", "real_meta_video"}
                or result["inference_origin"] not in {"cached_development_projection", "injected_callback", "injected_provider", "live_meta_api"}
                or result["status"] not in {"supported", "unknown"} or result["trust_mode"] != "warning_only_not_qualification"
                or result["physical_action_authorized"] is not False or result["changes_live_placement_acceptance"] is not False
                or not _integer(result["request_attempts"], MAX_REQUESTS)
                or type(result["unreported_charge_possible"]) is not bool
                or type(result["frames"]) is not list or not 1 <= len(result["frames"]) <= 480
                or result["estimated_usd"] is not None and (type(result["estimated_usd"]) not in {int, float}
                    or not math.isfinite(result["estimated_usd"]) or result["estimated_usd"] < 0)):
            raise ValueError
        seen = set()
        for row in result["frames"]:
            if (type(row) is not dict or set(row) != _FRAME or not _integer(row["frame_index"], 4095)
                    or not _integer(row["timestamp_ms"]) or type(row["source_sha256"]) is not str
                    or not _HASH.fullmatch(row["source_sha256"])
                    or row["frame_sha256"] is not None and not _HASH.fullmatch(row["frame_sha256"])
                    or row["status"] not in {"supported", "unknown", "unscored"}
                    or row["track_id"] is not None and not _integer(row["track_id"], 2**31-1)
                    or type(row["gap_codes"]) is not list or len(row["gap_codes"]) > 32
                    or any(type(code) is not str or not _SAFE.fullmatch(code) for code in row["gap_codes"])
                    or row["warning"] not in {"near_gripper_proposal", "no_near_proposal_observed", "geometry_unavailable"}
                    or type(row["confirmation_evidence"]) is not list or len(row["confirmation_evidence"]) > 8):
                raise ValueError
            key = row["source_sha256"], row["frame_index"]
            if key in seen:
                raise ValueError
            seen.add(key)
            if row["box"] is not None:
                current.legacy._box(row["box"])
            if row["status"] == "supported" and (row["box"] is None or row["frame_sha256"] is None
                    or row["warning"] == "near_gripper_proposal" or len(row["confirmation_evidence"]) < 2):
                raise ValueError
            proof_hashes, prior_time = set(), None
            for proof in row["confirmation_evidence"]:
                if (type(proof) is not dict or set(proof) != _PROOF or not _integer(proof["frame_index"],4095)
                        or not _integer(proof["timestamp_ms"]) or proof["source_sha256"] != row["source_sha256"]
                        or proof["track_id"] != row["track_id"] or not _HASH.fullmatch(proof["frame_sha256"])
                        or not _HASH.fullmatch(proof["semantic_evidence_sha256"])
                        or proof["frame_sha256"] in proof_hashes or proof["frame_index"] > row["frame_index"]
                        or prior_time is not None and proof["timestamp_ms"] - prior_time < 500):
                    raise ValueError
                proof_hashes.add(proof["frame_sha256"]); prior_time = proof["timestamp_ms"]
            if row["status"] == "supported":
                last = row["confirmation_evidence"][-1]
                if any(last[key] != row[key] for key in ("frame_index","timestamp_ms","frame_sha256","track_id")):
                    raise ValueError
        expected = "supported" if result["frames"][-1]["status"] == "supported" else "unknown"
        if result["status"] != expected or result["mode"] == "cached_demo" and (result["request_attempts"] != 0
                or result["estimated_usd"] != 0 or result["inference_origin"] != "cached_development_projection"):
            raise ValueError
        return deepcopy(result)
    except (KeyError, TypeError, ValueError, AttributeError):
        raise ValidationError("Tracking result violates the diagnostic-only public schema") from None


def result_from_policy(policy, warning_rows, *, mode, origin, request_attempts=0, estimated_usd=0.0,
                       unreported_charge_possible=False, unscored_indices=()):
    warnings_by_index = {row["frame_index"]: row for row in warning_rows}
    frames = [{key: deepcopy(row[key]) for key in ("frame_index", "timestamp_ms", "source_sha256", "frame_sha256",
                "status", "track_id", "box", "gap_codes", "confirmation_evidence")} | {
                "warning": warnings_by_index[row["frame_index"]]["warning"]} for row in policy["frames"]]
    for row in frames:
        if row["frame_index"] in unscored_indices:
            row.update(status="unscored", box=None, confirmation_evidence=[])
    return public_result({"schema_version": VERSION, "mode": mode, "inference_origin": origin,
        "status": "supported" if frames[-1]["status"] == "supported" else "unknown", "frames": frames,
        "request_attempts": request_attempts, "estimated_usd": estimated_usd,
        "unreported_charge_possible": unreported_charge_possible, "trust_mode": "warning_only_not_qualification",
        "physical_action_authorized": False, "changes_live_placement_acceptance": False})


class TrackingService:
    def __init__(self, private_root, *, client_factory=None, cached_callback=None, execute_callback=None,
                 clock_ms=None, lease_ms=30000, inference_origin="injected_callback", runtime_control=None):
        self.root = Path(private_root).resolve()
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.assets = self.root / "assets"; self.assets.mkdir(exist_ok=True, mode=0o700)
        self.db_path = self.root / "tracking.sqlite3"
        if self.db_path.is_symlink() or self.db_path.exists() and not self.db_path.is_file():
            raise ValidationError("Tracking database must be a private regular file")
        self.runtime_control = runtime_control
        self.client_factory = client_factory or (runtime_control.tracking_client if runtime_control is not None else None)
        self.cached_callback, self.callback = cached_callback, execute_callback
        # A caller-provided origin string cannot promote injected execution.
        self.origin = "injected_provider"
        self._trusted_factory = bool(runtime_control is not None and self.client_factory == runtime_control.tracking_client)
        self.clock = clock_ms or (lambda: time.time_ns() // 1000000)
        if not _integer(lease_ms, 300000) or lease_ms < 30:
            raise ValidationError("Tracking lease must be between30 and300000milliseconds")
        self.lease_ms, self.stop, self.worker = lease_ms, threading.Event(), None
        self._worker_failed = False
        with closing(self._db()) as db:
            db.executescript("""PRAGMA journal_mode=DELETE;
                CREATE TABLE IF NOT EXISTS diagnostic_jobs(id TEXT PRIMARY KEY,key_hash TEXT UNIQUE NOT NULL,
                payload TEXT NOT NULL,payload_hash TEXT NOT NULL,state TEXT NOT NULL,created INTEGER NOT NULL,
                updated INTEGER NOT NULL,token TEXT,expires INTEGER,result_hash TEXT,error TEXT);
                CREATE TABLE IF NOT EXISTS diagnostic_events(job_id TEXT,sequence INTEGER,kind TEXT,at INTEGER,
                PRIMARY KEY(job_id,sequence));
                CREATE TABLE IF NOT EXISTS diagnostic_assets(id TEXT PRIMARY KEY,created INTEGER NOT NULL);""")
            # Migrate existing issued assets once; active reservations are also
            # counted, so concurrent processes cannot exceed the same bound.
            for file in self.assets.glob("*.json"):
                if _ID.fullmatch(file.stem):
                    db.execute("INSERT OR IGNORE INTO diagnostic_assets VALUES(?,?)",(file.stem,self.clock()))
        self.db_path.chmod(0o600)
        if runtime_control is not None:
            runtime_control.bind_worker("shadow_tracking",self.readiness)

    def _trace(self, identifier, stage, **kwargs):
        if self.runtime_control is not None:
            self.runtime_control.record(identifier,"shadow_tracking",stage,**kwargs)

    def _span(self, identifier, stage):
        return (self.runtime_control.span(identifier,"shadow_tracking",stage)
                if self.runtime_control is not None else nullcontext())

    def _trace_evidence(self, identifier, result):
        # One observation per distinct normalized reason per job, never raw
        # frame labels or model text. The closed result has at most480frames;
        # this projection emits at most32 finite-label observations.
        reasons = set()
        for frame in result["frames"]:
            if frame["status"] == "unscored":
                reasons.add("model_error")
            elif frame["status"] == "unknown":
                if frame["warning"] == "near_gripper_proposal":
                    reasons.add("boundary_unconfirmed")
                reasons.update(_TRACE_GAPS.get(code,"unsupported_observation") for code in frame["gap_codes"])
        for code in sorted(reasons)[:32]:
            self._trace(identifier,"evidence_request",code=code)

    def _db(self):
        db = sqlite3.connect(str(self.db_path), isolation_level=None, timeout=10)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA synchronous=FULL")
        return db

    def _event(self, db, identifier, kind):
        sequence = db.execute("SELECT count(*) FROM diagnostic_events WHERE job_id=?",(identifier,)).fetchone()[0] + 1
        db.execute("INSERT INTO diagnostic_events VALUES(?,?,?,?)",(identifier,sequence,kind,self.clock()))

    def register_asset(self, data, mime="video/mp4"):
        if type(data) is not bytes or not 12 <= len(data) <= MAX_ASSET_BYTES or mime != "video/mp4" or data[4:8] != b"ftyp":
            raise ValidationError("Upload one MP4 video between12bytes and8MiB")
        identifier = secrets.token_hex(16)
        path = self.assets / (identifier + ".mp4")
        with closing(self._db()) as db:
            db.execute("BEGIN IMMEDIATE")
            if db.execute("SELECT count(*) FROM diagnostic_assets").fetchone()[0] >= MAX_ASSETS:
                raise ValidationError("Tracking asset allowance reached")
            try:
                db.execute("INSERT INTO diagnostic_assets VALUES(?,?)",(identifier,self.clock()))
            except sqlite3.IntegrityError:
                raise ValidationError("Tracking asset identifier collision") from None
            db.commit()
        created = False
        try:
            descriptor = os.open(path, os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
            created = True
            with os.fdopen(descriptor,"wb") as stream:
                stream.write(data); stream.flush(); os.fsync(stream.fileno())
            source, version = video._probe(path)
            metadata = {"asset_id":identifier,"sha256":hashlib.sha256(data).hexdigest(),"bytes":len(data),
                        "source":source,"probe_version":version}
            _atomic(self.assets / (identifier + ".json"),metadata)
            return self.asset(identifier)
        except BaseException:
            if created:
                path.unlink(missing_ok=True)
                (self.assets / (identifier + ".json")).unlink(missing_ok=True)
            with closing(self._db()) as db:
                db.execute("DELETE FROM diagnostic_assets WHERE id=?",(identifier,))
            raise

    def asset(self, identifier):
        path, metadata = self._asset(identifier)
        source = metadata["source"]
        return {"asset_id":identifier,"sha256":metadata["sha256"],"bytes":metadata["bytes"],
            **{key:source[key] for key in ("width","height","frame_count","duration_ms")},
            "frames":[{"frame_index":row["frame_index"],"timestamp_ms":row["timestamp_ms"]} for row in source["frames"]]}

    def _asset(self, identifier):
        if type(identifier) is not str or not _ID.fullmatch(identifier):
            raise ValidationError("Use an issued tracking asset identifier")
        path = self.assets / (identifier + ".mp4")
        metadata = _read(self.assets / (identifier + ".json"))
        if (path.is_symlink() or not path.is_file() or not 12 <= path.stat().st_size <= MAX_ASSET_BYTES
                or metadata["asset_id"] != identifier or metadata["bytes"] != path.stat().st_size
                or metadata["sha256"] != hashlib.sha256(path.read_bytes()).hexdigest()):
            raise ValidationError("Tracking source asset changed")
        return path, metadata

    def _request(self, value):
        if type(value) is not dict or type(value.get("idempotency_key")) is not str or not 1 <= len(value["idempotency_key"]) <= 128:
            raise ValidationError("Use one bounded tracking idempotency key")
        mode = value.get("mode")
        if mode == "cached_demo":
            if set(value) != {"idempotency_key","mode","fixture_id"} or value["fixture_id"] != "latest-tracking":
                raise ValidationError("Requested cached tracking example is unavailable")
        elif mode == "real_meta_video":
            if set(value) != {"idempotency_key","mode","asset_id","reviewed","contract","seeds","checkpoints"} or value["reviewed"] is not True:
                raise ValidationError("Review the exact bounded source, contract and frame-zero seeds")
            path, metadata = self._asset(value["asset_id"])
            contract = visibility._contract(value["contract"])
            source = {"sha256":metadata["sha256"],**metadata["source"]}
            seeds = video._seeds(value["seeds"],source)
            ids = {row["track_id"] for row in seeds}
            if (not 1 <= len(seeds) <= 4 or 1 not in ids or ids - {1,2,3,4}
                    or bool(ids & {3,4}) and not {3,4} <= ids
                    or any(row["frame_index"] != 0 for row in seeds)
                    or next(row for row in seeds if row["track_id"] == 1)["object_id"] != contract["target_id"]):
                raise ValidationError("Use target1, optionaldistractor2, andoptionalpairedgrippers3/4 atframezero")
            indices = value["checkpoints"]
            if (type(indices) is not list or not 2 <= len(indices) <= 8 or any(not _integer(i,source["frame_count"]-1) for i in indices)
                    or indices != sorted(set(indices)) or indices[0] != 0
                    or source["frames"][indices[-1]]["timestamp_ms"] < 500):
                raise ValidationError("Choose2to8orderedsourcecheckpointsincludingzero spanningatleast500ms")
        else:
            raise ValidationError("Select cached_demo or real_meta_video diagnostics")
        admitted = deepcopy({key:item for key,item in value.items() if key != "idempotency_key"})
        if mode == "real_meta_video":
            admitted.update(_asset_sha256=metadata["sha256"], _asset_receipt_sha256=_digest(metadata))
        return admitted

    def submit(self, request):
        started, monotonic = time.time_ns(), time.monotonic_ns()
        if (type(request) is not dict or type(request.get("idempotency_key")) is not str
                or not 1 <= len(request["idempotency_key"]) <= 128):
            raise ValidationError("Use one bounded tracking idempotency key")
        key_hash = _digest(request["idempotency_key"])
        with closing(self._db()) as db:
            db.execute("BEGIN IMMEDIATE")
            existing = db.execute("SELECT id,payload,payload_hash FROM diagnostic_jobs WHERE key_hash=?",(key_hash,)).fetchone()
            if existing:
                retained = json.loads(existing["payload"])
                if existing["payload_hash"] != _digest(retained):
                    raise ValidationError("Stored tracking payload integrity failed")
                public_payload = {key:value for key,value in retained.items() if key not in {"_asset_sha256","_asset_receipt_sha256"}}
                if _digest(public_payload) != _digest({key:value for key,value in request.items() if key != "idempotency_key"}):
                    raise TrackingConflict("Tracking key was already used for different input")
                db.commit(); return self.get(existing["id"])
            if self.stop.is_set() or self._worker_failed:
                raise TrackingConflict("Tracking worker is closed or unavailable")
            payload = self._request(request)
            if payload["mode"] == "cached_demo" and self.cached_callback is None:
                raise ValidationError("Requested cached tracking example is unavailable")
            if payload["mode"] == "real_meta_video":
                if self.callback is None and self.client_factory is None:
                    raise ValidationError("Meta tracking runtime is unavailable")
                if self.runtime_control is not None:
                    self.runtime_control.require_ready("shadow_tracking")
                if self.callback is None and self._trusted_factory:
                    self._preflight()
            encoded = _encoded(payload).decode()
            if db.execute("SELECT count(*) FROM diagnostic_jobs WHERE state IN ('queued','running')").fetchone()[0] >= 32:
                raise ValidationError("Tracking queue allowance reached")
            identifier, now = secrets.token_hex(16), self.clock()
            db.execute("INSERT INTO diagnostic_jobs VALUES(?,?,?,?,?,?,?,?,?,?,?)",(identifier,key_hash,encoded,
                _digest(payload),"queued",now,now,None,None,None,None))
            self._event(db,identifier,"queued"); db.commit()
        self._trace(identifier,"admission",started_unix_ns=started,duration_ns=max(0,time.monotonic_ns()-monotonic))
        return self.get(identifier)

    @staticmethod
    def _preflight():
        """Check the actual local adapter dependencies without model transport."""
        try:
            meta._sam_decoder()
            ffmpeg = Path(visibility._extractor_path())
            ffprobe = video._probe_path(None)
            if any(not path.is_file() or not os.access(path,os.X_OK) for path in (ffmpeg,ffprobe)):
                raise ValueError
        except (meta.MetaAPIError,ImportError,OSError,TypeError,ValueError,SyntaxError):
            raise ValidationError("Tracking local media dependencies are unavailable") from None

    def _recover(self, db):
        rows = db.execute("SELECT id FROM diagnostic_jobs WHERE state='running' AND expires<=?",(self.clock(),)).fetchall()
        for row in rows:
            db.execute("UPDATE diagnostic_jobs SET state='interrupted',error='outcome_unknown',updated=? WHERE id=?",(self.clock(),row["id"]))
            self._event(db,row["id"],"interrupted")

    def get(self, identifier):
        if type(identifier) is not str or not _ID.fullmatch(identifier):
            raise TrackingNotFound("Unknown tracking diagnostic")
        with closing(self._db()) as db:
            db.execute("BEGIN IMMEDIATE"); self._recover(db)
            row = db.execute("SELECT * FROM diagnostic_jobs WHERE id=?",(identifier,)).fetchone(); db.commit()
        if row is None:
            raise TrackingNotFound("Unknown tracking diagnostic")
        if (row["state"] not in {"queued","running","succeeded","failed","interrupted"}
                or row["error"] not in {None,"diagnostic_failed","outcome_unknown"}
                or not _integer(row["created"]) or not _integer(row["updated"])):
            raise ValidationError("Tracking job metadata integrity failed")
        result, valid = None, None
        if row["state"] == "succeeded":
            try:
                result = public_result(_read(self.root / "results" / (identifier + ".json")))
                if _digest(result) != row["result_hash"]:
                    raise ValidationError("changed")
                valid = True
            except (ValidationError,OSError):
                result, valid = None, False
        return {"job_id":identifier,"kind":"tracking_diagnostic","state":row["state"],"created_ms":row["created"],
            "updated_ms":row["updated"],"error_code":"result_integrity_error" if valid is False else row["error"],
            "result":result,"integrity":{"result_valid":valid},"physical_action_authorized":False,
            "changes_live_placement_acceptance":False}

    def events(self, identifier, after_sequence=0):
        self.get(identifier)
        if not _integer(after_sequence):
            raise ValidationError("Use a nonnegative event cursor")
        with closing(self._db()) as db:
            rows = db.execute("SELECT sequence,kind,at FROM diagnostic_events WHERE job_id=? AND sequence>? ORDER BY sequence LIMIT 200",(identifier,after_sequence)).fetchall()
        if any(row["kind"] not in {"queued","running","succeeded","failed","interrupted"}
               or not _integer(row["sequence"]) or not _integer(row["at"]) for row in rows):
            raise ValidationError("Tracking event metadata integrity failed")
        return [{"sequence":row["sequence"],"event":row["kind"],"at_ms":row["at"]} for row in rows]

    def run_once(self):
        with closing(self._db()) as db:
            db.execute("BEGIN IMMEDIATE"); self._recover(db)
            row = db.execute("SELECT * FROM diagnostic_jobs WHERE state='queued' ORDER BY created,id LIMIT 1").fetchone()
            if row is None:
                db.commit(); return False
            token, identifier = secrets.token_hex(32), row["id"]
            db.execute("UPDATE diagnostic_jobs SET state='running',token=?,expires=?,updated=? WHERE id=?",(token,self.clock()+self.lease_ms,self.clock(),identifier))
            self._event(db,identifier,"running"); db.commit()
        execution_started, execution_clock = time.time_ns(), time.monotonic_ns()
        observed_outcome, observed_code = "error", "operation_error"
        self._trace(identifier,"queue_wait",duration_ns=max(0,self.clock()-row["created"])*1_000_000)
        def guard():
            with closing(self._db()) as db:
                active = db.execute("SELECT state,token,expires,payload_hash FROM diagnostic_jobs WHERE id=?",(identifier,)).fetchone()
            if (self.stop.is_set() or active["state"] != "running" or active["token"] != token or active["expires"] <= self.clock()
                    or active["payload_hash"] != row["payload_hash"]):
                raise TrackingConflict("Tracking worker lease is no longer current")
            if payload["mode"] == "real_meta_video":
                _, metadata = self._asset(payload["asset_id"])
                if (metadata["sha256"] != payload["_asset_sha256"]
                        or _digest(metadata) != payload["_asset_receipt_sha256"]):
                    raise TrackingConflict("Admitted tracking asset binding changed")
        ended = threading.Event()
        def heartbeat():
            while not ended.wait(self.lease_ms/3000):
                with closing(self._db()) as db:
                    db.execute("UPDATE diagnostic_jobs SET expires=? WHERE id=? AND state='running' AND token=? AND expires>?",
                        (self.clock()+self.lease_ms,identifier,token,self.clock()))
        heartbeat_thread = threading.Thread(target=heartbeat,daemon=True); heartbeat_thread.start()
        try:
            payload = json.loads(row["payload"])
            if _digest(payload) != row["payload_hash"]:
                raise ValidationError("Stored tracking payload changed")
            guard()
            if payload["mode"] == "cached_demo":
                result = self.cached_callback(payload,identifier,guard)
            else:
                self._asset(payload["asset_id"])
                result = (self.callback or self._execute_meta)(payload,identifier,guard)
                if self.callback is not None:
                    result = deepcopy(result)
                    result["inference_origin"] = "injected_callback"
            with self._span(identifier,"verification"):
                result = public_result(result)
                if result["mode"] != payload["mode"]:
                    raise ValidationError("Tracking result mode differs from its admitted request")
                if payload["mode"] == "real_meta_video" and any(frame["source_sha256"] != payload["_asset_sha256"] for frame in result["frames"]):
                    raise ValidationError("Tracking result source differs from its admitted asset")
            self._trace_evidence(identifier,result)
            with self._span(identifier,"artifact_publication"):
                guard(); _atomic(self.root / "results" / (identifier + ".json"),result); guard()
                with closing(self._db()) as db:
                    db.execute("BEGIN IMMEDIATE")
                    changed = db.execute("UPDATE diagnostic_jobs SET state='succeeded',result_hash=?,updated=? WHERE id=? AND state='running' AND token=? AND expires>?",
                        (_digest(result),self.clock(),identifier,token,self.clock())).rowcount
                    if changed:
                        self._event(db,identifier,"succeeded")
                    db.commit()
            observed_outcome, observed_code = ("ok","none") if changed else ("uncertain","lease_lost")
        except Exception as error:
            observed_outcome = "uncertain" if self.stop.is_set() else "error"
            observed_code = "lease_lost" if self.stop.is_set() else "validation_error" if isinstance(error,ValidationError) else "operation_error"
            with closing(self._db()) as db:
                db.execute("BEGIN IMMEDIATE")
                state, code = ("interrupted","outcome_unknown") if self.stop.is_set() else ("failed","diagnostic_failed")
                changed = db.execute("UPDATE diagnostic_jobs SET state=?,error=?,updated=? WHERE id=? AND state='running' AND token=? AND expires>?",
                    (state,code,self.clock(),identifier,token,self.clock())).rowcount
                if changed:
                    self._event(db,identifier,state)
                self._recover(db); db.commit()
        finally:
            ended.set(); heartbeat_thread.join(timeout=1)
            self._trace(identifier,"execution",outcome=observed_outcome,code=observed_code,
                        started_unix_ns=execution_started,duration_ns=max(0,time.monotonic_ns()-execution_clock))
        return True

    def start(self):
        if self.worker and self.worker.is_alive():
            return
        self.stop.clear()
        self._worker_failed = False
        def work():
            while not self.stop.is_set():
                try:
                    if not self.run_once():
                        self.stop.wait(.1)
                except Exception:
                    self._worker_failed = True
                    return
        self.worker = threading.Thread(target=work,daemon=True); self.worker.start()

    def close(self):
        self.stop.set()
        if self.worker:
            self.worker.join(timeout=2)

    def capabilities(self):
        return {"schema_version":VERSION,"diagnostic_only":True,"max_asset_bytes":MAX_ASSET_BYTES,
            "max_assets":MAX_ASSETS,
            "max_frames":120,"max_duration_ms":12000,"max_checkpoints":8,"max_requests":MAX_REQUESTS,
            "cached_demo_available":self.cached_callback is not None,
            "cached_demo_scope":"one_exposed_remote_new_clip_policy_projection",
            "local_model_requirements":{"ffmpeg_executable":True,"ffprobe_executable":True,
                "official_sam_parser_importable":True,"checked_before_new_production_admission":True},
            "provider_connectivity_tested":False,
            "meta_callback_available":self.client_factory is not None or self.callback is not None,
            "worker_available":bool(self.worker and self.worker.is_alive()),"automatic_retry":False,
            "changes_live_placement_acceptance":False,"physical_action_authorized":False}

    def readiness(self):
        available = bool(not self.stop.is_set() and not self._worker_failed and self.worker and self.worker.is_alive())
        code = "worker_closed" if self.stop.is_set() else "worker_failed" if self._worker_failed else "worker_unavailable"
        return {"worker_available":available,"code":None if available else code}

    def _execute_meta(self, payload, identifier, guard):
        path, metadata = self._asset(payload["asset_id"])
        output = self.root / "runs" / identifier; output.mkdir(parents=True,exist_ok=False,mode=0o700)
        calls, costs, unknown_cost, created_clients = 0, 0.0, False, []
        def client():
            nonlocal calls
            guard(); self._asset(payload["asset_id"])
            if calls >= MAX_REQUESTS:
                raise ValidationError("Tracking request allowance exhausted")
            calls += 1
            adapter = self.client_factory(operation_id=identifier,guard=guard,max_requests=1,retries=0,timeout=180)
            created_clients.append(adapter)
            return adapter
        target_seeds = [row for row in payload["seeds"] if row["track_id"] in {1,2}]
        prepared = video.prepare_clip(path,target_seeds,review_note="Operator reviewed frame-zero tracking seeds.",review_origin="local_operator")
        raw = video.segment_video(client(),prepared); _atomic(output / "target.json",raw)
        summary = video.public_tracking_summary(raw)
        charge = summary["cost"]["estimated_usd"]; costs += charge or 0; unknown_cost |= charge is None
        source = prepared["source"]
        warning_rows = [{"frame_index":row["frame_index"],"timestamp_ms":row["timestamp_ms"],
            "source_sha256":source["sha256"],"target_track_id":1,"warning":"geometry_unavailable",
            "veto":False,"additional_geometry_complete":False} for row in source["frames"]]
        gripper_seeds = [row for row in payload["seeds"] if row["track_id"] in {3,4}]
        if gripper_seeds and raw["status"] == "completed":
            grippers = video.segment_video(client(),video.prepare_clip(path,gripper_seeds,
                review_note="Operator reviewed both frame-zero gripper seeds.",review_origin="local_operator"))
            _atomic(output / "grippers.json",grippers)
            charge = video.public_tracking_summary(grippers)["cost"]["estimated_usd"]
            costs += charge or 0; unknown_cost |= charge is None
            warning_rows = warnings.compare(raw,grippers)["frames"]
        frames = [{"frame_index":row["frame_index"],"timestamp_ms":row["timestamp_ms"],"sha256":None,"file":None} for row in source["frames"]]
        reports = {}
        executable = visibility._extractor_path()
        for index in payload["checkpoints"]:
            guard()
            file = output / f"frame-{index}.jpg"
            subprocess.run([executable,"-hide_banner","-loglevel","error","-nostdin","-i",str(path),"-vf",f"select=eq(n\\,{index})",
                "-frames:v","1","-q:v","2","-pix_fmt","yuvj444p","-threads","1","-map_metadata","-1",str(file)],
                check=True,capture_output=True,timeout=30)
            file.chmod(0o600); frames[index].update(file=str(file),sha256=hashlib.sha256(file.read_bytes()).hexdigest())
            tracked = next((row for row in raw["frames"] if row["frame_index"] == index),None)
            candidate = original._candidate(tracked) if tracked else None
            if candidate is None:
                reports[index] = {"status":"not_run_no_tracker_candidate","usage":{"request_attempts":0,"estimated_usd":0.0}}
            else:
                checked = visibility.prepare(file,original._pixels(candidate["box"],source["width"],source["height"]),
                    payload["contract"],output / f"guard-{index}",source_sha256=source["sha256"],frame_index=index,track_id=1)
                reports[index] = visibility.challenge(client(),checked)
                charge = reports[index]["usage"]["estimated_usd"]; costs += charge or 0
                unknown_cost |= reports[index]["usage"].get("spark_requests_with_unreported_token_usage",0) > 0
            _atomic(output / f"guard-{index}.json",reports[index])
        entry = {"prepared":prepared,"contract":payload["contract"],"frames":frames,"seeds":target_seeds}
        if raw["status"] == "completed":
            runtime = masks._runtime(entry,raw,reports)
            corrected, _ = masks._filter(runtime,geometry.resolve_distractor_conflicts(raw),entry,raw)
        else:
            corrected = [{"frame_index":row["frame_index"],"timestamp_ms":row["timestamp_ms"],
                "source_sha256":source["sha256"],"frame_sha256":frames[row["frame_index"]]["sha256"],
                "candidates":[],"reviewed_distractors":[]} for row in source["frames"]]
        base = veto.strict.baseline._policy(entry,corrected,current.evaluate_tracking)
        changed, indexed, _ = veto._runtime_veto(entry,corrected,{"source_sha256":source["sha256"],"frames":warning_rows})
        proposed = veto.strict.baseline._policy(entry,changed,current.evaluate_tracking)
        policy = veto._intersect_policy(proposed,base,changed,indexed)
        from .runtime_control import RuntimeMetaClient
        trusted = self._trusted_factory and bool(created_clients) and all(
            type(adapter) is RuntimeMetaClient and adapter.runtime_control is self.runtime_control
            and adapter.operation_kind == "shadow_tracking" for adapter in created_clients)
        return result_from_policy(policy,warning_rows,mode="real_meta_video",origin="live_meta_api" if trusted else "injected_provider",
            request_attempts=sum(len(adapter.ledger) for adapter in created_clients),estimated_usd=round(costs,6),unreported_charge_possible=unknown_cost,
            unscored_indices=range(len(source["frames"])) if raw["status"] != "completed" else
                [index for index,report in reports.items() if report["status"] == "unscored"])
