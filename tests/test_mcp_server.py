"""MCP stdio and loopback GET tests; authored API responses, no model calls."""
from copy import deepcopy
from io import BytesIO, StringIO
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch

from proofloop.mcp_server import (BackendError, LoopbackReader, MAX_MESSAGE_BYTES, MAX_EVENT_CURSOR,
    MCPServer, PROTOCOL_VERSION, main, serve)


JOB_ID = "a" * 32


def request(request_id, method, params=None):
    result = {"jsonrpc": "2.0", "id": request_id, "method": method}
    if params is not None:
        result["params"] = params
    return result


def initialize(request_id=1, version=PROTOCOL_VERSION):
    return request(request_id, "initialize", {"protocolVersion": version, "capabilities": {},
        "clientInfo": {"name": "Authored test client", "version": "1.0"}})


def initialized():
    return {"jsonrpc": "2.0", "method": "notifications/initialized"}


def call(request_id, name, arguments=None):
    params = {"name": name}
    if arguments is not None:
        params["arguments"] = arguments
    return request(request_id, "tools/call", params)


def lines(messages):
    return b"".join(json.dumps(message).encode() + b"\n" for message in messages)


class FakeReader:
    def __init__(self):
        self.calls = []
        self.value = {"authored_fixture": True, "physical_action_authorized": False}
        self.error = None

    def read(self, name, arguments):
        self.calls.append((name, deepcopy(arguments)))
        if self.error is not None:
            raise self.error
        return deepcopy(self.value)


class MCPProtocolTests(unittest.TestCase):
    def setUp(self):
        self.reader = FakeReader()
        self.server = MCPServer(reader=self.reader)

    def ready(self):
        self.assertIn("result", self.server.handle(initialize()))
        self.assertIsNone(self.server.handle(initialized()))

    def test_initialize_pins_version_and_only_advertises_read_tools(self):
        response = self.server.handle(initialize())
        self.assertEqual(1, response["id"])
        self.assertEqual(PROTOCOL_VERSION, response["result"]["protocolVersion"])
        self.assertEqual({"tools": {"listChanged": False}}, response["result"]["capabilities"])
        self.assertIn("not the current physical scene", response["result"]["instructions"])
        self.assertEqual([], self.reader.calls)

    def test_wrong_version_fails_without_echo_and_can_retry_with_a_new_id(self):
        result = self.server.handle(initialize(version="private-unsupported-version"))
        self.assertEqual(-32602, result["error"]["code"])
        self.assertNotIn("private-unsupported-version", json.dumps(result))
        self.assertIn("result", self.server.handle(initialize(request_id=2)))

    def test_ping_is_available_before_and_after_initialization_without_http(self):
        self.assertEqual({}, self.server.handle(request("before", "ping"))["result"])
        self.ready()
        self.assertEqual({}, self.server.handle(request("after", "ping", {}))["result"])
        self.assertEqual([], self.reader.calls)

    def test_tools_require_initialized_notification_and_do_not_run_early(self):
        result = self.server.handle(call(0, "proofloop_capabilities"))
        self.assertEqual(-32002, result["error"]["code"])
        self.server.handle(initialize())
        result = self.server.handle(request(2, "tools/list"))
        self.assertEqual(-32002, result["error"]["code"])
        self.assertEqual([], self.reader.calls)
        self.server.handle(initialized())
        self.assertIn("tools", self.server.handle(request(3, "tools/list"))["result"])

    def test_tool_list_has_four_closed_schemas_and_correct_annotations(self):
        self.ready()
        tools = self.server.handle(request(2, "tools/list"))["result"]["tools"]
        self.assertEqual({"proofloop_capabilities", "proofloop_inspection", "proofloop_events", "proofloop_accounting"},
            {tool["name"] for tool in tools})
        for tool in tools:
            self.assertFalse(tool["inputSchema"]["additionalProperties"])
            self.assertEqual({"readOnlyHint": True, "destructiveHint": False,
                "idempotentHint": True, "openWorldHint": False}, tool["annotations"])
        self.assertEqual([], self.reader.calls)
        tools[0]["annotations"]["readOnlyHint"] = False
        self.assertTrue(self.server.handle(request(3, "tools/list"))["result"]["tools"][0]["annotations"]["readOnlyHint"])

    def test_success_returns_identical_structured_and_serialized_json(self):
        self.ready()
        for number, name, arguments in ((2, "proofloop_capabilities", {}), (3, "proofloop_inspection", {"job_id": JOB_ID}),
                (4, "proofloop_events", {"job_id": JOB_ID, "after_sequence": 3}), (5, "proofloop_accounting", {})):
            response = self.server.handle(call(number, name, arguments))
            result = response["result"]
            self.assertFalse(result["isError"])
            self.assertEqual(self.reader.value, result["structuredContent"])
            self.assertEqual(result["structuredContent"], json.loads(result["content"][0]["text"]))
        self.assertEqual(4, len(self.reader.calls))

    def test_unknown_tools_fields_and_argument_types_never_dispatch_http(self):
        self.ready()
        invalid = [call(2, "proofloop_submit_inspection", {}), call(3, "proofloop_capabilities", {"url": "https://example.com"}),
            call(4, "proofloop_inspection", {"job_id": "../../.env.local"}), call(5, "proofloop_inspection", {}),
            call(6, "proofloop_events", {"job_id": JOB_ID, "after_sequence": True}),
            call(7, "proofloop_events", {"job_id": JOB_ID, "after_sequence": -1}),
            call(8, "proofloop_events", {"job_id": JOB_ID, "after_sequence": MAX_EVENT_CURSOR + 1}),
            call(9, "proofloop_inspection", {"job_id": JOB_ID, "confirm": True}),
            call(10, "proofloop_capabilities", []), call(11, "proofloop_accounting", None),
            request(12, "tools/call", {"name": "proofloop_capabilities", "secret": "private"})]
        # Omitted arguments are valid for no-argument tools; explicit null is not.
        invalid[9]["params"]["arguments"] = None
        for value in invalid:
            result = self.server.handle(value)
            self.assertEqual(-32602, result["error"]["code"], value)
        self.assertEqual([], self.reader.calls)

    def test_unknown_methods_and_params_produce_static_protocol_errors(self):
        self.ready()
        self.assertEqual(-32601, self.server.handle(request(2, "private-method"))["error"]["code"])
        for number, value in enumerate((request(3, "ping", {"secret": "private"}),
                request(4, "tools/list", {"cursor": "unissued-private-cursor"}),
                request(5, "tools/list", {"_meta": []})), start=3):
            response = self.server.handle(value)
            self.assertEqual(-32602, response["error"]["code"])
            self.assertNotIn("private", json.dumps(response))

    def test_valid_meta_is_ignored_and_never_sent_to_http(self):
        self.ready()
        value = call(2, "proofloop_capabilities", {})
        value["params"]["_meta"] = {"progressToken": "private-progress-token"}
        response = self.server.handle(value)
        self.assertFalse(response["result"]["isError"])
        self.assertEqual([("proofloop_capabilities", {})], self.reader.calls)
        self.assertNotIn("private-progress-token", json.dumps(response))

    def test_notifications_never_reply_or_execute_http(self):
        self.ready()
        for value in ({"jsonrpc": "2.0", "method": "unknown/private-method"},
                {"jsonrpc": "2.0", "method": "tools/call", "params": {"name": "proofloop_accounting"}},
                {"jsonrpc": "2.0", "method": "notifications/cancelled", "params": {"requestId": 1}}, initialized()):
            self.assertIsNone(self.server.handle(value))
        self.assertEqual([], self.reader.calls)

    def test_invalid_initialized_notification_does_not_advance_session(self):
        self.server.handle(initialize())
        self.assertIsNone(self.server.handle({**initialized(), "params": {"secret": "ignored"}}))
        self.assertEqual(-32002, self.server.handle(request(2, "tools/list"))["error"]["code"])
        self.server.handle(initialized())
        self.assertIn("result", self.server.handle(request(3, "tools/list")))

    def test_duplicate_request_ids_and_second_initialize_are_rejected(self):
        self.ready()
        self.assertEqual(-32600, self.server.handle(request(1, "ping"))["error"]["code"])
        self.assertEqual(-32600, self.server.handle(initialize(request_id=2))["error"]["code"])
        self.assertEqual([], self.reader.calls)

    def test_invalid_envelopes_and_ids_cannot_execute_http(self):
        for value in ([], {}, {"jsonrpc": "1.0", "id": 1, "method": "ping"}, request(None, "ping"),
                request(True, "ping"), request(1.5, "ping"), request("x" * 129, "ping"),
                {**request(9, "ping"), "url": "https://example.com"}):
            response = self.server.handle(value)
            self.assertEqual(-32600, response["error"]["code"])
        self.assertEqual([], self.reader.calls)

    def test_tool_errors_are_structured_fixed_and_never_echo_exception_text(self):
        self.ready()
        for index, error in enumerate((BackendError("not_found"), BackendError("backend_conflict"),
                RuntimeError("/Users/private/.env.local Authorization secret-key")), start=2):
            self.reader.error = error
            result = self.server.handle(call(index, "proofloop_capabilities"))["result"]
            self.assertTrue(result["isError"])
            self.assertEqual(result["structuredContent"], json.loads(result["content"][0]["text"]))
            self.assertNotIn("/Users/private", json.dumps(result))
            self.assertNotIn("secret-key", json.dumps(result))

    def test_large_or_invalid_tool_result_fails_closed_with_bounded_error(self):
        self.ready()
        self.reader.value = {"large": "x" * MAX_MESSAGE_BYTES}
        result = self.server.handle(call(2, "proofloop_capabilities"))["result"]
        self.assertTrue(result["isError"])
        self.assertEqual("result_too_large", result["structuredContent"]["error"]["code"])
        self.reader.value = {"bad": float("nan")}
        result = self.server.handle(call(3, "proofloop_capabilities"))["result"]
        self.assertTrue(result["isError"])

    def test_stdio_is_json_only_and_recovers_after_bad_utf8_duplicate_keys_and_oversize_line(self):
        source = (b'not-json\n' + b'\xff\n' + b'{"jsonrpc":"2.0","id":1,"id":2,"method":"ping"}\n'
            + b"x" * (MAX_MESSAGE_BYTES + 20) + b"\n" + lines([initialize(3), initialized(), request(4, "ping")]))
        output = BytesIO()
        self.assertEqual(0, serve(BytesIO(source), output, reader=self.reader))
        encoded = output.getvalue().splitlines()
        self.assertEqual(6, len(encoded))
        self.assertTrue(all(len(line) < MAX_MESSAGE_BYTES for line in encoded))
        values = [json.loads(line) for line in encoded]
        self.assertEqual([-32700, -32700, -32700, -32600], [value["error"]["code"] for value in values[:4]])
        self.assertEqual({}, values[-1]["result"])
        self.assertEqual([], self.reader.calls)

    def test_binary_stream_main_can_be_injected_without_touching_stdio(self):
        output = BytesIO()
        self.assertEqual(0, main([], stdin=BytesIO(lines([request(1, "ping")])), stdout=output))
        self.assertEqual({"jsonrpc": "2.0", "id": 1, "result": {}}, json.loads(output.getvalue()))

    def test_invalid_ports_and_host_override_fail_before_reading_messages(self):
        for port in (0, -1, 65536, True, "8790"):
            with self.assertRaises(ValueError):
                MCPServer(port=port)
        for argv in (["--port", "0"], ["--port", "65536"], ["--host", "example.com"]):
            with patch("sys.stderr", new=StringIO()), self.assertRaises(SystemExit):
                main(argv, stdin=BytesIO(), stdout=BytesIO())


class LoopbackAndSubprocessTests(unittest.TestCase):
    def setUp(self):
        self.requests = []
        self.mode = "ok"
        self.redirect_to = None
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                owner.requests.append((self.command, self.path, dict(self.headers)))
                if owner.mode == "redirect":
                    self.send_response(302)
                    self.send_header("Location", owner.redirect_to)
                    self.end_headers()
                    return
                status = 404 if owner.mode == "error" else 200
                if owner.mode == "error":
                    body = b'{"error":"/Users/private/.env.local Authorization secret-provider-key"}'
                elif owner.mode == "bad-json":
                    body = b'{"secret":'
                elif owner.mode == "duplicate":
                    body = b'{"state":"ok","state":"wrong"}'
                elif owner.mode == "oversize":
                    body = b"x" * (MAX_MESSAGE_BYTES + 1)
                elif self.path == "/api/v1/capabilities":
                    body = json.dumps({"enabled": True, "physical_action_authorized": False}).encode()
                elif self.path == "/api/v1/accounting":
                    body = json.dumps({"total_reserved_attempts": 3, "automatic_retry": False, "hard_dollar_limit": False}).encode()
                elif "/events?" in self.path:
                    body = json.dumps({"events": [{"job_id": JOB_ID, "sequence": 1, "event_type": "submitted"}]}).encode()
                else:
                    body = json.dumps({"job": {"id": JOB_ID, "state": "queued", "result": None}}).encode()
                if owner.mode == "wrong-job":
                    body = json.dumps({"job": {"id": "b" * 32}}).encode()
                self.send_response(status)
                self.send_header("Content-Type", "text/html" if owner.mode == "html" else "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                try:
                    self.wfile.write(body)
                except (BrokenPipeError, ConnectionResetError):
                    pass

            def do_POST(self):
                owner.requests.append((self.command, self.path, dict(self.headers)))
                self.send_response(405)
                self.end_headers()

        self.http = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.http.serve_forever, kwargs={"poll_interval": .01}, daemon=True)
        self.thread.start()
        self.addCleanup(self._stop)
        self.port = self.http.server_address[1]
        self.reader = LoopbackReader(self.port)

    def _stop(self):
        self.http.shutdown()
        self.http.server_close()
        self.thread.join(2)

    def test_only_four_fixed_get_routes_are_used_without_auth_or_capture_tokens(self):
        self.reader.read("proofloop_capabilities", {})
        self.reader.read("proofloop_inspection", {"job_id": JOB_ID})
        self.reader.read("proofloop_events", {"job_id": JOB_ID, "after_sequence": 7})
        self.reader.read("proofloop_accounting", {})
        self.assertEqual(["/api/v1/capabilities", "/api/v1/jobs/" + JOB_ID,
            "/api/v1/jobs/" + JOB_ID + "/events?after_sequence=7", "/api/v1/accounting"], [value[1] for value in self.requests])
        self.assertTrue(all(value[0] == "GET" for value in self.requests))
        for _, _, headers in self.requests:
            self.assertEqual("127.0.0.1:" + str(self.port), headers["Host"])
            self.assertFalse({"Authorization", "Cookie", "X-ProofLoop-Token"} & set(headers))

    def test_redirects_are_not_followed_even_to_another_loopback_path(self):
        self.mode = "redirect"
        self.redirect_to = "http://127.0.0.1:" + str(self.port) + "/api/inspect"
        with self.assertRaises(BackendError):
            self.reader.read("proofloop_capabilities", {})
        self.assertEqual(1, len(self.requests))
        self.assertEqual("/api/v1/capabilities", self.requests[0][1])

    def test_environment_proxies_are_disabled_for_loopback_reads(self):
        environment = {"HTTP_PROXY": "http://127.0.0.1:1", "http_proxy": "http://127.0.0.1:1", "NO_PROXY": "", "no_proxy": ""}
        with patch.dict(os.environ, environment):
            reader = LoopbackReader(self.port)
            self.assertTrue(reader.read("proofloop_capabilities", {})["enabled"])
        self.assertEqual(1, len(self.requests))

    def test_backend_failures_are_redacted_and_malformed_public_data_is_rejected(self):
        for mode in ("error", "bad-json", "duplicate", "html", "oversize", "wrong-job"):
            self.mode = mode
            with self.assertRaises(BackendError) as raised:
                self.reader.read("proofloop_inspection", {"job_id": JOB_ID})
            self.assertNotIn("secret-provider-key", str(raised.exception))
            self.assertNotIn("/Users/private", str(raised.exception))

    def test_real_subprocess_initializes_lists_pings_and_reads_every_tool(self):
        messages = [initialize(), initialized(), request(2, "tools/list"), request(3, "ping"),
            call(4, "proofloop_capabilities", {}), call(5, "proofloop_inspection", {"job_id": JOB_ID}),
            call(6, "proofloop_events", {"job_id": JOB_ID}), call(7, "proofloop_accounting", {})]
        result = subprocess.run([sys.executable, "-m", "proofloop.mcp_server", "--port", str(self.port)],
            input=lines(messages), stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=10,
            cwd=Path(__file__).resolve().parents[1], check=False)
        self.assertEqual(0, result.returncode, result.stderr)
        self.assertEqual(b"", result.stderr)
        responses = [json.loads(line) for line in result.stdout.splitlines()]
        self.assertEqual([1, 2, 3, 4, 5, 6, 7], [value["id"] for value in responses])
        self.assertEqual(PROTOCOL_VERSION, responses[0]["result"]["protocolVersion"])
        self.assertEqual(4, len(responses[1]["result"]["tools"]))
        self.assertEqual({}, responses[2]["result"])
        self.assertTrue(all(not value["result"]["isError"] for value in responses[3:]))
        self.assertEqual(4, len(self.requests))
        self.assertTrue(all(value[0] == "GET" for value in self.requests))


if __name__ == "__main__":
    unittest.main()
