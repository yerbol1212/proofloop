"""Narrow read-only MCP 2025-06-18 stdio adapter for a local ProofLoop server.

Run ``python -m proofloop.mcp_server --port 8790`` from an MCP client. This
process never starts a model request, submits an inspection, reads an image or
performs a physical action. It forwards four fixed GET routes to IPv4 loopback.
The underlying HTTP API owns public projection and journal verification; stored
verdicts remain historical completion snapshots, not permission to act.

Protocol references (not a compatibility certification):
https://modelcontextprotocol.io/specification/2025-06-18/basic/lifecycle
https://modelcontextprotocol.io/specification/2025-06-18/basic/transports
https://modelcontextprotocol.io/specification/2025-06-18/server/tools
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
import re
import sys
from typing import Any, BinaryIO
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener


PROTOCOL_VERSION = "2025-06-18"
MAX_MESSAGE_BYTES = 1024 * 1024
MAX_BACKEND_BYTES = 1024 * 1024
MAX_REQUESTS_PER_SESSION = 10_000
MAX_EVENT_CURSOR = 999_999_999  # Matches the local HTTP API's bounded cursor.
_JOB_ID = re.compile(r"[0-9a-f]{32}\Z")
_ANNOTATIONS = {"readOnlyHint": True, "destructiveHint": False,
    "idempotentHint": True, "openWorldHint": False}
_EMPTY_SCHEMA = {"type": "object", "properties": {}, "additionalProperties": False}
_ID_SCHEMA = {"type": "string", "pattern": "^[0-9a-f]{32}$", "minLength": 32, "maxLength": 32}
_TOOLS = [
    {"name": "proofloop_capabilities", "title": "ProofLoop capabilities",
     "description": "Read implemented verification capabilities and limits. Does not start inference or authorize physical actions.",
     "inputSchema": _EMPTY_SCHEMA, "annotations": _ANNOTATIONS},
    {"name": "proofloop_inspection", "title": "Read a ProofLoop inspection job",
     "description": "Read one known job and its verified public completion record. A stored result is historical; it does not verify the current scene or authorize an action.",
     "inputSchema": {"type": "object", "properties": {"job_id": _ID_SCHEMA},
         "required": ["job_id"], "additionalProperties": False}, "annotations": _ANNOTATIONS},
    {"name": "proofloop_events", "title": "Read ProofLoop job events",
     "description": "Read a bounded page of verified local journal events for one known job. Hashes are local tamper evidence, not signatures or trusted timestamps.",
     "inputSchema": {"type": "object", "properties": {"job_id": _ID_SCHEMA,
         "after_sequence": {"type": "integer", "minimum": 0, "maximum": MAX_EVENT_CURSOR, "default": 0}},
         "required": ["job_id"], "additionalProperties": False}, "annotations": _ANNOTATIONS},
    {"name": "proofloop_accounting", "title": "Read ProofLoop request accounting",
     "description": "Read persistent request allowance and reported token totals. This does not request models, verify provider billing or enforce a dollar limit.",
     "inputSchema": _EMPTY_SCHEMA, "annotations": _ANNOTATIONS},
]


class _InvalidParams(Exception):
    pass


class BackendError(Exception):
    """Fixed safe error code; never contains HTTP bodies or exception text."""
    def __init__(self, code: str = "backend_unavailable"):
        allowed = {"backend_unavailable", "backend_rejected", "not_found", "backend_conflict",
            "invalid_backend_response", "result_too_large"}
        self.code = code if code in allowed else "backend_unavailable"
        super().__init__(self.code)


def _encoded(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=True, allow_nan=False, separators=(",", ":")).encode("utf-8")


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Repeated JSON key")
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError("Non-finite JSON number")


def _check_json(value: Any, depth: int = 0) -> None:
    if depth > 64:
        raise ValueError("Excessive JSON nesting")
    if value is None or type(value) in (str, bool, int):
        return
    if type(value) is float and math.isfinite(value):
        return
    if type(value) is list:
        for child in value:
            _check_json(child, depth + 1)
        return
    if type(value) is dict and all(type(key) is str for key in value):
        for child in value.values():
            _check_json(child, depth + 1)
        return
    raise ValueError("Invalid JSON value")


def _parse(data: bytes) -> Any:
    value = json.loads(data.decode("utf-8"), object_pairs_hook=_unique_object, parse_constant=_reject_constant)
    _check_json(value)
    return value


def _port(value: Any) -> int:
    if type(value) is not int or not 1 <= value <= 65535:
        raise ValueError("Use a loopback port from 1 to 65535")
    return value


def _tool_path(name: Any, arguments: Any) -> str:
    if type(name) is not str or type(arguments) is not dict:
        raise _InvalidParams
    if name in ("proofloop_capabilities", "proofloop_accounting"):
        if arguments:
            raise _InvalidParams
        return "/api/v1/" + ("capabilities" if name == "proofloop_capabilities" else "accounting")
    if name not in ("proofloop_inspection", "proofloop_events"):
        raise _InvalidParams
    allowed = {"job_id", "after_sequence"} if name == "proofloop_events" else {"job_id"}
    if set(arguments) - allowed or "job_id" not in arguments:
        raise _InvalidParams
    job_id = arguments["job_id"]
    if type(job_id) is not str or _JOB_ID.fullmatch(job_id) is None:
        raise _InvalidParams
    path = "/api/v1/jobs/" + job_id
    if name == "proofloop_events":
        sequence = arguments.get("after_sequence", 0)
        if type(sequence) is not int or not 0 <= sequence <= MAX_EVENT_CURSOR:
            raise _InvalidParams
        path += "/events?after_sequence=" + str(sequence)
    return path


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class LoopbackReader:
    """No arbitrary URLs, redirects, environment proxies, cookies or auth keys."""
    def __init__(self, port: int = 8790):
        self.port = _port(port)
        self._base = "http://127.0.0.1:" + str(self.port)
        self._opener = build_opener(ProxyHandler({}), _NoRedirect())

    def read(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        path = _tool_path(name, arguments)
        url = self._base + path
        request = Request(url, method="GET", headers={"Accept": "application/json", "Connection": "close"})
        try:
            with self._opener.open(request, timeout=5) as response:
                if response.geturl() != url or response.getcode() != 200:
                    raise BackendError("backend_rejected")
                content_type = response.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
                if content_type != "application/json" or response.headers.get("Content-Encoding", "identity").lower() != "identity":
                    raise BackendError("invalid_backend_response")
                lengths = response.headers.get_all("Content-Length", [])
                if lengths and (len(lengths) != 1 or re.fullmatch(r"[0-9]{1,10}", lengths[0]) is None):
                    raise BackendError("invalid_backend_response")
                if lengths and int(lengths[0]) > MAX_BACKEND_BYTES:
                    raise BackendError("result_too_large")
                data = response.read(MAX_BACKEND_BYTES + 1)
                if len(data) > MAX_BACKEND_BYTES:
                    raise BackendError("result_too_large")
                if lengths and len(data) != int(lengths[0]):
                    raise BackendError("invalid_backend_response")
                value = _parse(data)
                if type(value) is not dict or "error" in value:
                    raise BackendError("invalid_backend_response")
                if name == "proofloop_inspection" and (set(value) != {"job"} or type(value["job"]) is not dict or value["job"].get("id") != arguments["job_id"]):
                    raise BackendError("invalid_backend_response")
                if name == "proofloop_events" and (set(value) != {"events"} or type(value["events"]) is not list
                        or len(value["events"]) > 200 or any(type(event) is not dict or event.get("job_id") != arguments["job_id"] for event in value["events"])):
                    raise BackendError("invalid_backend_response")
                return value
        except HTTPError as error:
            code = {404: "not_found", 409: "backend_conflict", 503: "backend_unavailable"}.get(error.code, "backend_rejected")
            error.close()
            raise BackendError(code) from None
        except (URLError, OSError, TimeoutError):
            raise BackendError("backend_unavailable") from None
        except (ValueError, TypeError, RecursionError, UnicodeError):
            raise BackendError("invalid_backend_response") from None


def _error(request_id: Any, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}


def _tool_error(code: str) -> dict[str, Any]:
    messages = {
        "backend_unavailable": "The local ProofLoop API is unavailable.",
        "backend_rejected": "The local ProofLoop API rejected this read.",
        "not_found": "This ProofLoop job is unavailable.",
        "backend_conflict": "This ProofLoop record could not pass its current state or integrity checks.",
        "invalid_backend_response": "The local ProofLoop API returned an unsupported public response.",
        "result_too_large": "This ProofLoop read exceeds the MCP message limit; request a smaller event range.",
    }
    safe_code = code if code in messages else "backend_unavailable"
    value = {"error": {"code": safe_code, "message": messages[safe_code]}}
    return {"content": [{"type": "text", "text": _encoded(value).decode()}], "structuredContent": value, "isError": True}


class MCPServer:
    """A bounded synchronous stdio session advertising only four read-only tools."""
    def __init__(self, *, port: int = 8790, reader: Any = None):
        _port(port)
        self.reader = reader if reader is not None else LoopbackReader(port)
        self._phase = "new"
        self._seen_ids: set[tuple[type, Any]] = set()

    @staticmethod
    def _params(value: Any, *, allowed: set[str], required: set[str] | None = None) -> dict[str, Any]:
        if type(value) is not dict or set(value) - (allowed | {"_meta"}) or (required or set()) - set(value):
            raise _InvalidParams
        if "_meta" in value and type(value["_meta"]) is not dict:
            raise _InvalidParams
        return value

    def handle(self, message: Any) -> dict[str, Any] | None:
        if type(message) is not dict:
            return _error(None, -32600, "Use one JSON-RPC request object per line.")
        request_id = message.get("id")
        has_id = "id" in message
        valid_id = (type(request_id) is int and abs(request_id) <= 2 ** 53 - 1 or
            type(request_id) is str and 1 <= len(request_id) <= 128)
        if (message.get("jsonrpc") != "2.0" or type(message.get("method")) is not str
                or set(message) - {"jsonrpc", "id", "method", "params"} or has_id and not valid_id):
            return _error(request_id if has_id and valid_id else None, -32600, "The MCP request envelope is invalid.")
        if not has_id:
            # Notifications never cause HTTP calls and never receive a reply.
            if message["method"] == "notifications/initialized" and self._phase == "negotiated":
                try:
                    self._params(message.get("params", {}), allowed=set())
                    self._phase = "ready"
                except _InvalidParams:
                    pass
            return None
        key = (type(request_id), request_id)
        if key in self._seen_ids:
            return _error(request_id, -32600, "MCP request IDs must be unique within a session.")
        if len(self._seen_ids) >= MAX_REQUESTS_PER_SESSION:
            return _error(request_id, -32000, "The bounded MCP session request limit was reached.")
        self._seen_ids.add(key)
        try:
            method = message["method"]
            params = message.get("params", {})
            if method == "ping":
                self._params(params, allowed=set())
                result = {}
            elif method == "initialize":
                if self._phase != "new":
                    return _error(request_id, -32600, "This MCP session is already initialized.")
                self._params(params, allowed={"protocolVersion", "capabilities", "clientInfo"},
                    required={"protocolVersion", "capabilities", "clientInfo"})
                if params["protocolVersion"] != PROTOCOL_VERSION:
                    return _error(request_id, -32602, "Unsupported MCP protocol version; use 2025-06-18.")
                info = params["clientInfo"]
                if (type(params["capabilities"]) is not dict or type(info) is not dict
                        or set(info) - {"name", "version", "title"} or not {"name", "version"} <= set(info)
                        or any(type(value) is not str or not 1 <= len(value) <= 128
                            or any(ord(character) < 32 for character in value) for value in info.values())):
                    raise _InvalidParams
                self._phase = "negotiated"
                result = {"protocolVersion": PROTOCOL_VERSION, "capabilities": {"tools": {"listChanged": False}},
                    "serverInfo": {"name": "proofloop-readonly", "version": "1.0.0"},
                    "instructions": "Read-only access to the local ProofLoop public API. Stored results describe completion snapshots, not the current physical scene. No tool submits inference or authorizes a physical action."}
            elif method in ("tools/list", "tools/call"):
                if self._phase != "ready":
                    return _error(request_id, -32002, "Complete MCP initialization and its initialized notification first.")
                if method == "tools/list":
                    self._params(params, allowed=set())
                    result = {"tools": deepcopy(_TOOLS)}
                else:
                    self._params(params, allowed={"name", "arguments"}, required={"name"})
                    name, arguments = params["name"], params.get("arguments", {})
                    _tool_path(name, arguments)
                    try:
                        value = self.reader.read(name, deepcopy(arguments))
                        if type(value) is not dict:
                            raise BackendError("invalid_backend_response")
                        _check_json(value)
                        result = {"content": [{"type": "text", "text": _encoded(value).decode("utf-8")}],
                            "structuredContent": value, "isError": False}
                        if len(_encoded(result)) > MAX_MESSAGE_BYTES - 2048:
                            result = _tool_error("result_too_large")
                    except BackendError as error:
                        result = _tool_error(error.code)
                    except Exception:
                        result = _tool_error("backend_unavailable")
            else:
                return _error(request_id, -32601, "This MCP method is unsupported.")
            return {"jsonrpc": "2.0", "id": request_id, "result": result}
        except _InvalidParams:
            return _error(request_id, -32602, "The MCP method parameters or tool arguments are invalid.")
        except Exception:
            return _error(request_id, -32603, "The MCP read could not be completed.")


def serve(stdin: BinaryIO, stdout: BinaryIO, *, port: int = 8790, reader: Any = None) -> int:
    """Read bounded binary UTF-8 lines; write only bounded JSON-RPC lines."""
    server = MCPServer(port=port, reader=reader)
    try:
        while True:
            line = stdin.readline(MAX_MESSAGE_BYTES + 1)
            if not line:
                return 0
            if len(line) > MAX_MESSAGE_BYTES:
                while line and not line.endswith(b"\n"):
                    line = stdin.readline(MAX_MESSAGE_BYTES + 1)
                response = _error(None, -32600, "The MCP input line exceeds the message limit.")
            else:
                try:
                    response = server.handle(_parse(line))
                except (UnicodeError, ValueError, TypeError, RecursionError):
                    response = _error(None, -32700, "The MCP input is not supported UTF-8 JSON.")
            if response is not None:
                encoded = _encoded(response)
                if len(encoded) + 1 > MAX_MESSAGE_BYTES:
                    encoded = _encoded(_error(response.get("id"), -32603, "The MCP response exceeds the message limit."))
                stdout.write(encoded + b"\n")
                stdout.flush()
    except (BrokenPipeError, ConnectionResetError, KeyboardInterrupt):
        return 0


class _StderrParser(argparse.ArgumentParser):
    def _print_message(self, message, file=None):
        if message:
            sys.stderr.write(message)


def main(argv: list[str] | None = None, *, stdin: BinaryIO | None = None, stdout: BinaryIO | None = None) -> int:
    parser = _StderrParser(description="Read-only MCP stdio access to a running local ProofLoop server")
    parser.add_argument("--port", type=int, default=8790, help="Existing ProofLoop IPv4 loopback port (default 8790)")
    args = parser.parse_args(argv)
    try:
        port = _port(args.port)
    except ValueError:
        parser.error("Use a loopback port from 1 to 65535")
    return serve(sys.stdin.buffer if stdin is None else stdin, sys.stdout.buffer if stdout is None else stdout, port=port)


if __name__ == "__main__":
    raise SystemExit(main())
