"""Loopback-only HTTP boundary for the interactive ProofLoop workbench.

The server exposes explicit engine methods and a compiled frontend directory.
It never serves the project directory or reads an API key into the browser.
"""

from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re
import secrets
import socket
import sys
from typing import Any
from urllib.parse import parse_qs, unquote, urlsplit

from .core import ValidationError
from .jobs import JobConflict, JobIntegrityError, JobNotFound, JobValidationError
from .verification_policy import describe_capabilities


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WEB_ROOT = ROOT / "web" / "dist"
MAX_JSON_BYTES = 16 * 1024 * 1024
MAX_CONTROL_BYTES = 1024
MAX_TASK_BYTES = 24 * 1024
MAX_MEDIA_BYTES = 10 * 1024 * 1024
SOCKET_TIMEOUT_SECONDS = 20
MIME_TYPES = {
    ".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8", ".json": "application/json; charset=utf-8",
    ".svg": "image/svg+xml", ".png": "image/png", ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg", ".webp": "image/webp", ".ico": "image/x-icon",
    ".woff": "font/woff", ".woff2": "font/woff2", ".txt": "text/plain; charset=utf-8",
}
MEDIA_MIME_TYPES = {suffix: MIME_TYPES[suffix] for suffix in (".png", ".jpg", ".jpeg", ".webp")}
CSP = (
    "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
    "img-src 'self' data: blob:; media-src 'self' blob:; connect-src 'self'; "
    "object-src 'none'; base-uri 'none'; frame-ancestors 'none'; form-action 'self'"
)

# Exact trusted validation messages only. Never reflect arbitrary ValueError
# text, provider responses, supplied filenames, or filesystem paths to HTTP.
TRUSTED_VALIDATION_MESSAGES = frozenset({
    "Evidence acquisition stopped. Review the task or capture setup before continuing",
    "Supply 1 to 6 distinct object IDs using letters, digits, underscores, dots or hyphens",
    "Supply exactly two distinct lowercase region IDs using letters, digits, underscores, dots or hyphens",
    "The authored coffee preset is fixed. Use Spark to draft changed requirements",
    "The task draft changed. Request and review a new exact draft before activation",
    "The reviewed region inventory does not match its executable protocol",
    "Approve both nonoverlapping rectangles using the active task's exact region IDs",
    "Select Spark compilation or the authored coffee placement preset",
    "The task or inspection changed. Review the current preparation again",
    "Preparation is blocked. Obtain current passing evidence for the reviewed task",
    "Supply the exact task, revision and explicit continuation confirmation",
    "Explicitly confirm the continuation declaration",
    "Assistant continuation requires a direct user authorization reference",
    "Continuation is already declared for this task and revision",
    "An inspection is already running",
    "An inspection payload is required",
    "Start a new inspection to obtain a current capture token",
    "Confirm regions, unique targets, fresh captures and one-second spacing",
    "Select camera capture or fresh photo upload",
    "Approve both nonoverlapping region rectangles",
    "Supply exactly two fresh images",
    "Supply two encoded PNG or JPEG images",
    "Each encoded image must be at most 5 MiB",
    "Supply PNG or JPEG images as base64 data URLs",
    "Image encoding is invalid",
    "Each image must contain between one byte and 5 MiB",
    "Image signature does not match its declared type",
    "PNG image is truncated or excessive",
    "PNG image checksum failed",
    "PNG image header is invalid",
    "PNG image is incomplete",
    "JPEG image is incomplete",
    "JPEG image marker is invalid",
    "JPEG image segment is truncated",
    "JPEG image dimensions are missing",
    "JPEG orientation metadata requires rotation. Export an upright PNG or JPEG with orientation removed, or use camera capture.",
    "JPEG orientation metadata is invalid",
    "Use a baseline or progressive JPEG image",
    "Image dimensions must be at least 16 pixels and at most 16 megapixels",
    "Repeated or previously inspected images cannot verify a new correction",
    "Both images must have the same dimensions and camera orientation",
    "Camera captures must follow this inspection and be one second apart and recent",
    "regions must map IDs to normalized rectangles",
    "rectangle must identify a protocol region and contain four coordinates",
    "region coordinate must be a finite number",
    "region rectangle must have positive area inside [0,1]",
    "approved grounding rectangles must not overlap",
    "Supply task requirements and an explicit object inventory",
    "Task draft contains unsupported fields",
    "Task requirements must contain 1 to 8192 UTF-8 bytes",
    "Supply 2 to 6 distinct object IDs using letters, digits, underscores, dots or hyphens",
    "Task title must contain 1 to 120 characters",
    "Confirm one visually distinct instance of every target before drafting",
    "Supply the exact draft ID, hash and review confirmations",
    "Review the current task draft before activating it",
    "Confirm the exact requirements and every unique target before activation",
    "Assistant review requires a direct user authorization reference",
    "SAM prompts must contain 1 to 100 printable characters",
    "SAM prompt mapping must identify every task object exactly once",
    "Select concept or guided perception mode before drafting",
    "Guided perception requires object IDs without dots",
    "Guided perception requires upright JPEG images without EXIF, up to 1280 pixels per side",
    "Guided perception changed. Request and review a new task draft before activation",
    "Guided perception changed. Request and review a new task draft before inspection",
})


class _RequestError(Exception):
    def __init__(self, status: int, code: str, message: str):
        self.status = status
        self.code = code
        self.message = message


def _reject_json_constant(value: str) -> None:
    raise ValueError("Non-finite JSON number")


def _unique_json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON field")
        result[key] = value
    return result


class ProofLoopServer(ThreadingHTTPServer):
    daemon_threads = True
    block_on_close = False

    def get_request(self):
        connection, address = super().get_request()
        connection.settimeout(SOCKET_TIMEOUT_SECONDS)
        return connection, address


def create_server(engine: Any, host: str = "127.0.0.1", port: int = 0,
                  web_root: Path = DEFAULT_WEB_ROOT, *, video_lab: Any = None,
                  research_lab: Any = None, job_service: Any = None,
                  attempt_journal: Any = None) -> ThreadingHTTPServer:
    """Create, without starting, a server that binds only the IPv4 loopback.

    ``engine`` is intentionally injectable so transport tests require no model
    calls. Engine media identifiers are the only route into private artifacts.
    """
    if host != "127.0.0.1":
        raise ValueError("ProofLoop must bind to 127.0.0.1")
    if isinstance(port, bool) or not isinstance(port, int) or not 0 <= port <= 65535:
        raise ValueError("Port must be between 0 and 65535")
    resolved_web_root = Path(web_root).resolve()
    csrf_token = secrets.token_urlsafe(32)
    if video_lab is None:
        from .video_lab import VideoLab
        video_lab = VideoLab(root=ROOT)
    if research_lab is None:
        from .research_lab import ResearchLab
        research_lab = ResearchLab(root=ROOT)

    class Handler(BaseHTTPRequestHandler):
        server_version = "ProofLoop"
        sys_version = ""
        protocol_version = "HTTP/1.0"

        def log_message(self, format, *args):
            # Request URLs, form data, and engine diagnostics remain private.
            pass

        def _response(self, status: int, payload: bytes, content_type: str,
                      *, head: bool = False) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Security-Policy", CSP)
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Permissions-Policy", "camera=(self), microphone=(), geolocation=()")
            self.send_header("Connection", "close")
            self.end_headers()
            if not head:
                self.wfile.write(payload)
            self.close_connection = True

        def _json(self, status: int, value: Any, *, head: bool = False) -> None:
            payload = json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")
            self._response(status, payload, "application/json; charset=utf-8", head=head)

        def _error(self, status: int, code: str, message: str) -> None:
            self._json(status, {"error": {"code": code, "message": message}})

        def send_error(self, code, message=None, explain=None):
            # BaseHTTPRequestHandler's default HTML includes arbitrary parser
            # messages. Keep even malformed/unsupported requests non-reflective.
            self._error(code, "http_error", "The HTTP request could not be accepted.")

        def _origin(self) -> str:
            return f"http://127.0.0.1:{self.server.server_address[1]}"

        def _boundary(self, *, mutation: bool = False) -> None:
            hosts = self.headers.get_all("Host", [])
            expected_host = f"127.0.0.1:{self.server.server_address[1]}"
            if hosts != [expected_host]:
                raise _RequestError(403, "invalid_host", "Open ProofLoop using its local 127.0.0.1 URL.")
            origins = self.headers.get_all("Origin", [])
            if (origins and origins != [self._origin()]) or (mutation and not origins):
                raise _RequestError(403, "invalid_origin", "This request must come from the local ProofLoop page.")
            if mutation:
                tokens = self.headers.get_all("X-ProofLoop-Token", [])
                if len(tokens) != 1 or not secrets.compare_digest(tokens[0], csrf_token):
                    raise _RequestError(403, "invalid_token", "Refresh the local ProofLoop page before continuing.")

        def _path(self) -> str:
            parsed = urlsplit(self.path)
            if parsed.scheme or parsed.netloc or parsed.fragment or not parsed.path.startswith("/"):
                raise _RequestError(400, "invalid_path", "Use a local application path.")
            try:
                path = unquote(parsed.path, encoding="utf-8", errors="strict")
            except UnicodeError:
                raise _RequestError(400, "invalid_path", "Use a valid application path.")
            if "\x00" in path or "\\" in path or any(
                part.startswith(".") for part in path.split("/") if part
            ):
                raise _RequestError(404, "not_found", "This resource is unavailable.")
            return path

        def _read_json(self, limit: int) -> dict[str, Any]:
            if self.headers.get_all("Transfer-Encoding", []):
                raise _RequestError(400, "invalid_body", "Transfer encoding is unsupported.")
            lengths = self.headers.get_all("Content-Length", [])
            if len(lengths) != 1 or not re.fullmatch(r"[0-9]+", lengths[0]):
                raise _RequestError(411, "invalid_length", "Provide one JSON content length.")
            length = int(lengths[0])
            if length > limit:
                raise _RequestError(413, "body_too_large", "The capture exceeds the local upload limit.")
            content_types = self.headers.get_all("Content-Type", [])
            if len(content_types) != 1 or content_types[0].split(";", 1)[0].strip().lower() != "application/json":
                raise _RequestError(415, "invalid_content_type", "Send this request as JSON.")
            try:
                body = self.rfile.read(length)
                if len(body) != length:
                    raise _RequestError(400, "invalid_body", "The JSON request was incomplete.")
                value = json.loads(body.decode("utf-8"), parse_constant=_reject_json_constant,
                                   object_pairs_hook=_unique_json_object)
            except (UnicodeError, ValueError):
                raise _RequestError(400, "invalid_json", "Provide a valid JSON object.")
            except (socket.timeout, TimeoutError):
                raise _RequestError(408, "request_timeout", "The local upload timed out.")
            if not isinstance(value, dict):
                raise _RequestError(400, "invalid_json", "Provide a JSON object.")
            return value

        def _snapshot(self, value: Any) -> dict[str, Any]:
            if not isinstance(value, dict):
                raise RuntimeError("Invalid engine snapshot")
            return {**value, "csrf_token": csrf_token}

        def _media(self, media_id: str, *, head: bool, provider: Any = engine) -> None:
            if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", media_id):
                raise _RequestError(404, "not_found", "This image is unavailable.")
            try:
                candidate = provider.media_path(media_id)
                path = Path(candidate) if candidate is not None else None
                if path is None or path.suffix.lower() not in MEDIA_MIME_TYPES or not path.is_file():
                    raise FileNotFoundError()
                if path.stat().st_size > MAX_MEDIA_BYTES:
                    raise FileNotFoundError()
                payload = path.read_bytes()
                if len(payload) > MAX_MEDIA_BYTES:
                    raise FileNotFoundError()
            except (OSError, ValueError, KeyError, TypeError):
                raise _RequestError(404, "not_found", "This image is unavailable.")
            self._response(200, payload, MEDIA_MIME_TYPES[path.suffix.lower()], head=head)

        def _frontend(self, path: str, *, head: bool) -> None:
            candidate = (resolved_web_root / path.lstrip("/")).resolve()
            if candidate != resolved_web_root and resolved_web_root not in candidate.parents:
                raise _RequestError(404, "not_found", "This resource is unavailable.")
            if path == "/" or (not candidate.is_file() and re.fullmatch(r"/[A-Za-z0-9_/-]+", path)):
                candidate = resolved_web_root / "index.html"
            # Re-resolve the fallback as index.html itself could be a symlink.
            candidate = candidate.resolve()
            if resolved_web_root not in candidate.parents or candidate.suffix.lower() not in MIME_TYPES:
                raise _RequestError(404, "not_found", "This resource is unavailable.")
            try:
                if not candidate.is_file():
                    raise FileNotFoundError()
                payload = candidate.read_bytes()
            except OSError:
                raise _RequestError(404, "not_found", "This resource is unavailable.")
            self._response(200, payload, MIME_TYPES[candidate.suffix.lower()], head=head)

        def _get(self, *, head: bool = False) -> None:
            self._boundary()
            path = self._path()
            job_match = re.fullmatch(r"/api/v1/jobs/([0-9a-f]{32})(/events)?", path)
            if path == "/api/v1/capabilities":
                self._json(200, job_service.capabilities() if job_service is not None else
                           {**describe_capabilities(), "enabled": False}, head=head)
            elif path == "/api/v1/accounting":
                if attempt_journal is None:
                    raise _RequestError(503, "service_unavailable", "Persistent request accounting is unavailable.")
                self._json(200, attempt_journal.summary(), head=head)
            elif job_match:
                if job_service is None:
                    raise _RequestError(503, "service_unavailable", "The durable inspection service is unavailable.")
                if job_match[2]:
                    query = parse_qs(urlsplit(self.path).query, keep_blank_values=True)
                    values = query.get("after_sequence", ["0"])
                    if set(query) - {"after_sequence"} or len(values) != 1 or not re.fullmatch(r"[0-9]{1,9}", values[0]):
                        raise _RequestError(400, "invalid_request", "Use a nonnegative event sequence.")
                    self._json(200, {"events": job_service.events(job_match[1], after_sequence=int(values[0]))}, head=head)
                else:
                    self._json(200, {"job": job_service.get(job_match[1])}, head=head)
            elif path == "/api/health":
                self._json(200, {"ok": True}, head=head)
            elif path in ("/api/state", "/api/config"):
                self._json(200, self._snapshot(engine.snapshot()), head=head)
            elif path == "/api/lab":
                self._json(200, video_lab.snapshot(), head=head)
            elif path == "/api/research-lab":
                self._json(200, research_lab.snapshot(), head=head)
            elif path.startswith("/api/research-lab/media/"):
                self._media(path[len("/api/research-lab/media/"):], head=head, provider=research_lab)
            elif path == "/api/recovery":
                self._json(200, {"recovery": engine.recovery_receipt(),
                                 "verification": engine.verify_recovery()}, head=head)
            elif path.startswith("/api/lab/media/"):
                self._media(path[len("/api/lab/media/"):], head=head, provider=video_lab)
            elif path.startswith("/api/media/"):
                self._media(path[len("/api/media/"):], head=head)
            elif path == "/api" or path.startswith("/api/"):
                raise _RequestError(404, "not_found", "This API endpoint is unavailable.")
            else:
                self._frontend(path, head=head)

        def _post(self) -> None:
            self._boundary(mutation=True)
            path = self._path()
            cancel_match = re.fullmatch(r"/api/v1/jobs/([0-9a-f]{32})/cancel", path)
            if path == "/api/v1/inspections" or cancel_match:
                if job_service is None:
                    raise _RequestError(503, "service_unavailable", "The durable inspection service is unavailable.")
                payload = self._read_json(MAX_JSON_BYTES if not cancel_match else MAX_CONTROL_BYTES)
                if cancel_match:
                    if payload:
                        raise _RequestError(400, "invalid_request", "Cancellation does not accept extra fields.")
                    self._json(200, {"job": job_service.cancel(cancel_match[1])})
                else:
                    if set(payload) != {"idempotency_key", "binding", "inspection"}:
                        raise _RequestError(400, "invalid_request", "Supply an idempotency key, exact binding and inspection.")
                    result = job_service.submit_inspection({key: payload[key] for key in ("binding", "inspection")},
                                                          idempotency_key=payload["idempotency_key"])
                    self._json(202, {"job": result})
                return
            if path == "/api/inspect" and job_service is not None:
                raise _RequestError(409, "durable_inspection_required", "Refresh ProofLoop to submit through the durable inspection service.")
            methods = {"/api/repair": "begin_repair", "/api/new": "new_inspection", "/api/inspect": "inspect_pair",
                       "/api/task/draft": "draft_task", "/api/task/activate": "activate_task",
                       "/api/workflow/check": "check_workflow", "/api/workflow/continue": "continue_workflow"}
            if path not in methods:
                raise _RequestError(404, "not_found", "This API endpoint is unavailable.")
            limit = MAX_JSON_BYTES if path == "/api/inspect" else MAX_TASK_BYTES if path.startswith("/api/task/") else MAX_CONTROL_BYTES
            payload = self._read_json(limit)
            if path in ("/api/repair", "/api/new") and payload:
                raise _RequestError(400, "invalid_request", "This action does not accept extra fields.")
            method = getattr(engine, methods[path])
            result = method(payload) if path not in ("/api/repair", "/api/new") else method()
            self._json(200, self._snapshot(result))

        def _dispatch(self, action) -> None:
            try:
                action()
            except _RequestError as exc:
                self._error(exc.status, exc.code, exc.message)
            except JobNotFound:
                self._error(404, "not_found", "This inspection job is unavailable.")
            except JobConflict:
                self._error(409, "job_conflict", "This request conflicts with an existing job or its current state.")
            except JobValidationError:
                self._error(400, "invalid_request", "Check the job binding and inspection request.")
            except JobIntegrityError:
                self._error(409, "integrity_error", "This job could not pass its integrity checks.")
            except ValueError as exc:
                try:
                    busy = bool(engine.snapshot().get("busy", False))
                except Exception:
                    busy = False
                message = "An inspection is already running." if busy else (
                    str(exc) if isinstance(exc, ValidationError) and str(exc) in TRUSTED_VALIDATION_MESSAGES else
                    "Check the capture confirmations and use a fresh pair from the current session.")
                self._error(409 if busy else 400, "busy" if busy else "invalid_request", message)
            except (BrokenPipeError, ConnectionResetError, socket.timeout):
                self.close_connection = True
            except Exception:
                # No local paths, API exceptions, or model bodies cross HTTP.
                self._error(500, "internal_error", "The local operation could not be completed. Refresh the session.")

        def do_GET(self):
            self._dispatch(self._get)

        def do_HEAD(self):
            self._dispatch(lambda: self._get(head=True))

        def do_POST(self):
            self._dispatch(self._post)

        def do_OPTIONS(self):
            self._error(405, "method_not_allowed", "Use the local ProofLoop page.")

    server = ProofLoopServer((host, port), Handler)
    server.base_url = f"http://127.0.0.1:{server.server_address[1]}"
    return server


def _default_seed(root: Path) -> Path | None:
    candidates = [path for path in (root / "artifacts" / "local").glob("phone-session-*/report.json")
                  if path.is_file()]
    return max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Open ProofLoop's local interactive workbench")
    parser.add_argument("--port", type=int, default=8790)
    parser.add_argument("--seed", type=Path, help="Optional local phone-session report; no report data is published")
    parser.add_argument("--request-limit", type=int, default=120,
                        help="Persistent inspection request allowance; fixed when the local journal is created (default 120)")
    args = parser.parse_args(argv)
    if not (DEFAULT_WEB_ROOT / "index.html").is_file():
        print("Frontend is not built. From web/, run npm install and npm run build, then start this server again.", file=sys.stderr)
        return 2
    from .interactive import InteractiveEngine
    from .inference_journal import AttemptJournal, JournaledMetaClient
    from .job_service import InspectionJobService

    seed = args.seed if args.seed is not None else _default_seed(ROOT)
    if seed is not None and not seed.is_absolute():
        seed = ROOT / seed
    try:
        engine = InteractiveEngine(root=ROOT, seed_report=seed)
        journal = AttemptJournal(ROOT / "artifacts/local/job-runtime/attempts.sqlite3", request_limit=args.request_limit)

        def inspect(inspection, binding, job_id, guard):
            def factory(**kwargs):
                return JournaledMetaClient.from_env(journal=journal, operation_id=job_id, guard=guard, **kwargs)
            return engine.inspect_pair(inspection, expected_binding=binding, runtime_client_factory=factory)

        jobs = InspectionJobService(engine, inspect_callback=inspect)
        server = create_server(engine, port=args.port, job_service=jobs, attempt_journal=journal)
    except (OSError, ValueError):
        print("Could not start the local ProofLoop server. Check the local report and choose an available port.", file=sys.stderr)
        return 2
    print(f"ProofLoop is ready at {server.base_url}", flush=True)
    jobs.start()
    try:
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        jobs.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
