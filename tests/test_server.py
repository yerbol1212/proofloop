"""Real loopback HTTP boundary tests with an authored fake engine, no API calls."""

import http.client
import json
from pathlib import Path
import tempfile
import threading
import unittest

from proofloop.server import create_server, MAX_CONTROL_BYTES, MAX_JSON_BYTES, MAX_TASK_BYTES
from proofloop.core import ValidationError


class FakeEngine:
    def __init__(self, media):
        self.media = media
        self.calls = []
        self.busy = False
        self.fail = None

    def snapshot(self):
        return {"phase": "seeded", "busy": self.busy, "capture_token": "test-capture-token"}

    def begin_repair(self):
        return self._call("repair")

    def new_inspection(self):
        return self._call("new")

    def inspect_pair(self, payload):
        return self._call("inspect", payload)

    def draft_task(self, payload):
        return self._call("task_draft", payload)

    def activate_task(self, payload):
        return self._call("task_activate", payload)

    def check_workflow(self, payload):
        return self._call("workflow_check", payload)

    def continue_workflow(self, payload):
        return self._call("workflow_continue", payload)

    def _call(self, name, payload=None):
        if self.fail is not None:
            raise self.fail
        self.calls.append((name, payload))
        return self.snapshot()

    def media_path(self, media_id):
        if media_id != "known-image":
            raise KeyError(media_id)
        return self.media

    def recovery_receipt(self):
        return None

    def verify_recovery(self):
        return {"valid": False, "scope": "requested", "receipt_hash": None, "outcome": None,
                "reason": "No linked recovery is available.", "physical_action_execution_verified": False}


class ServerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.web = self.root / "dist"
        self.web.mkdir()
        (self.web / "index.html").write_text("<html>ProofLoop fixture</html>")
        (self.web / "app.js").write_text("console.log('fixture')")
        (self.web / ".env.local").write_text("PRIVATE=do-not-serve")
        self.media = self.root / "private-frame.jpg"
        self.media.write_bytes(b"authored-test-jpeg")
        self.engine = FakeEngine(self.media)
        self.server = create_server(self.engine, web_root=self.web)
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
        self.thread.start()
        self.host = f"127.0.0.1:{self.server.server_address[1]}"
        self.origin = f"http://{self.host}"
        self.token = self.request("GET", "/api/state")[2]["csrf_token"]

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(2)
        self.temp.cleanup()

    def request(self, method, path, body=None, headers=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.server.server_address[1], timeout=3)
        if isinstance(body, dict):
            body = json.dumps(body).encode()
        all_headers = {"Host": self.host, **(headers or {})}
        connection.request(method, path, body=body, headers=all_headers)
        response = connection.getresponse()
        content = response.read()
        response_headers = dict(response.getheaders())
        value = json.loads(content) if "application/json" in response_headers.get("Content-Type", "") and content else content
        status = response.status
        connection.close()
        return status, response_headers, value

    def mutation_headers(self, **overrides):
        return {"Origin": self.origin, "X-ProofLoop-Token": self.token,
                "Content-Type": "application/json", **overrides}

    def test_state_provides_stable_csrf_and_security_headers(self):
        status, headers, body = self.request("GET", "/api/state")
        self.assertEqual(200, status)
        self.assertEqual(self.token, body["csrf_token"])
        self.assertGreater(len(self.token), 30)
        self.assertEqual("no-store", headers["Cache-Control"])
        self.assertIn("frame-ancestors 'none'", headers["Content-Security-Policy"])
        self.assertNotIn("Access-Control-Allow-Origin", headers)
        self.assertNotIn("Set-Cookie", headers)

    def test_recovery_replay_route_is_read_only_same_origin_and_contains_no_capture_token(self):
        status, headers, body = self.request("GET", "/api/recovery")
        self.assertEqual(200, status)
        self.assertEqual("no-store", headers["Cache-Control"])
        self.assertIsNone(body["recovery"])
        self.assertFalse(body["verification"]["physical_action_execution_verified"])
        self.assertNotIn("capture_token", json.dumps(body))
        self.assertEqual(403, self.request("GET", "/api/recovery", headers={"Origin": "https://other.example"})[0])
        self.assertEqual(404, self.request("POST", "/api/recovery", {}, self.mutation_headers())[0])
        self.assertEqual(b"", self.request("HEAD", "/api/recovery")[2])
        self.assertEqual([], self.engine.calls)

    def test_host_header_prevents_rebinding(self):
        for host in ("localhost:" + str(self.server.server_address[1]), "evil.example", "127.0.0.1"):
            with self.subTest(host=host):
                self.assertEqual(403, self.request("GET", "/api/state", headers={"Host": host})[0])

    def test_foreign_origins_cannot_read_or_mutate(self):
        headers = self.mutation_headers(Origin="https://example.com")
        self.assertEqual(403, self.request("GET", "/api/state", headers=headers)[0])
        self.assertEqual(403, self.request("POST", "/api/repair", {}, headers)[0])
        self.assertEqual([], self.engine.calls)

    def test_mutations_require_same_origin_and_token(self):
        for headers in ({"Content-Type": "application/json"},
                        self.mutation_headers(**{"X-ProofLoop-Token": "wrong"}),
                        {"Origin": self.origin, "Content-Type": "application/json"}):
            with self.subTest(headers=headers):
                self.assertEqual(403, self.request("POST", "/api/repair", {}, headers)[0])
        self.assertEqual([], self.engine.calls)

    def test_valid_actions_delegate_exactly_once(self):
        for path, payload in (("/api/repair", {}), ("/api/new", {}),
                              ("/api/inspect", {"images": ["test", "test2"]})):
            status, _, body = self.request("POST", path, payload, self.mutation_headers())
            self.assertEqual(200, status)
            self.assertEqual(self.token, body["csrf_token"])
        self.assertEqual([("repair", None), ("new", None), ("inspect", {"images": ["test", "test2"]})], self.engine.calls)

    def test_task_routes_preserve_csrf_and_delegate_exact_payloads_once(self):
        draft = {"requirements": "Fork LEFT, spoon RIGHT", "objects": ["fork", "spoon"], "confirm_unique_targets": True}
        activation = {"draft_id": "fixture", "draft_hash": "0" * 64,
                      "confirm_requirements": True, "confirm_unique_targets": True}
        for path, payload in (("/api/task/draft", draft), ("/api/task/activate", activation)):
            self.assertEqual(403, self.request("POST", path, payload, {"Content-Type": "application/json"})[0])
            status, _, state = self.request("POST", path, payload, self.mutation_headers())
            self.assertEqual(200, status)
            self.assertEqual(self.token, state["csrf_token"])
        self.assertEqual([("task_draft", draft), ("task_activate", activation)], self.engine.calls)

    def test_workflow_routes_require_origin_and_csrf_then_delegate_exact_payload_once(self):
        check = {"task_hash": "a" * 64, "revision": 3}
        continuation = {**check, "confirm_operator_continuation": True, "review_origin": "local_operator"}
        for path, payload in (("/api/workflow/check", check), ("/api/workflow/continue", continuation)):
            for headers in ({"Content-Type": "application/json"},
                            self.mutation_headers(Origin="https://other.example"),
                            self.mutation_headers(**{"X-ProofLoop-Token": "wrong"})):
                with self.subTest(path=path, headers=headers):
                    before = list(self.engine.calls)
                    self.assertEqual(403, self.request("POST", path, payload, headers)[0])
                    self.assertEqual(before, self.engine.calls)
            status, headers, state = self.request("POST", path, payload, self.mutation_headers())
            self.assertEqual(200, status)
            self.assertEqual("no-store", headers["Cache-Control"])
            self.assertEqual(self.token, state["csrf_token"])
        self.assertEqual([("workflow_check", check), ("workflow_continue", continuation)], self.engine.calls)

    def test_workflow_routes_are_post_only_and_bound_before_reading_large_payloads(self):
        for path in ("/api/workflow/check", "/api/workflow/continue"):
            with self.subTest(path=path):
                self.assertEqual(404, self.request("GET", path)[0])
                self.assertEqual(404, self.request("HEAD", path)[0])
                status, _, body = self.request("POST", path, b"", self.mutation_headers(
                    **{"Content-Length": str(MAX_CONTROL_BYTES + 1)}))
                self.assertEqual(413, status)
                self.assertEqual("body_too_large", body["error"]["code"])
        self.assertEqual([], self.engine.calls)

    def test_workflow_binding_errors_are_helpful_without_reflecting_untrusted_details(self):
        message = "The task or inspection changed. Review the current preparation again"
        for path in ("/api/workflow/check", "/api/workflow/continue"):
            with self.subTest(path=path):
                self.engine.fail = ValidationError(message)
                status, _, body = self.request("POST", path, {}, self.mutation_headers())
                self.assertEqual(400, status)
                self.assertEqual(message, body["error"]["message"])
                self.engine.fail = ValidationError("/private/evidence API_SECRET provider-body")
                status, _, body = self.request("POST", path, {}, self.mutation_headers())
                self.assertEqual(400, status)
                self.assertNotIn("API_SECRET", json.dumps(body))
                self.assertNotIn("/private/evidence", json.dumps(body))
        self.assertEqual([], self.engine.calls)

    def test_task_routes_have_small_bounded_body_limit(self):
        for path in ("/api/task/draft", "/api/task/activate"):
            self.assertEqual(413, self.request("POST", path, b"", self.mutation_headers(
                **{"Content-Length": str(MAX_TASK_BYTES + 1)}))[0])
        self.assertEqual([], self.engine.calls)

    def test_task_validation_messages_are_safe_and_specific(self):
        self.engine.fail = ValidationError("Review the current task draft before activating it")
        status, _, body = self.request("POST", "/api/task/activate", {}, self.mutation_headers())
        self.assertEqual(400, status)
        self.assertEqual(str(self.engine.fail), body["error"]["message"])

    def test_control_actions_reject_extra_fields(self):
        self.assertEqual(400, self.request("POST", "/api/repair", {"ignored": True}, self.mutation_headers())[0])
        self.assertEqual([], self.engine.calls)

    def test_declared_large_body_rejected_before_upload(self):
        status, _, body = self.request("POST", "/api/inspect", b"", self.mutation_headers(
            **{"Content-Length": str(MAX_JSON_BYTES + 1)}))
        self.assertEqual(413, status)
        self.assertEqual("body_too_large", body["error"]["code"])

    def test_chunked_body_not_accepted(self):
        self.assertEqual(400, self.request("POST", "/api/inspect", b"{}", self.mutation_headers(
            **{"Transfer-Encoding": "chunked"}))[0])

    def test_duplicate_length_or_host_is_rejected(self):
        for repeated in ("Content-Length", "Host"):
            connection = http.client.HTTPConnection("127.0.0.1", self.server.server_address[1], timeout=3)
            connection.putrequest("POST", "/api/repair", skip_host=True)
            for name, value in {"Host": self.host, "Content-Length": "2", **self.mutation_headers()}.items():
                connection.putheader(name, value)
                if name == repeated:
                    connection.putheader(name, value)
            connection.endheaders(b"{}")
            response = connection.getresponse()
            response.read()
            self.assertEqual(403 if repeated == "Host" else 411, response.status)
            connection.close()
        self.assertEqual([], self.engine.calls)

    def test_json_is_object_finite_utf8_and_correct_content_type(self):
        for raw in (b"[]", b"bad", b"{\"value\":NaN}", b"\xff"):
            with self.subTest(raw=raw):
                self.assertEqual(400, self.request("POST", "/api/inspect", raw, self.mutation_headers())[0])
        self.assertEqual(415, self.request("POST", "/api/inspect", b"{}", self.mutation_headers(
            **{"Content-Type": "text/plain"}))[0])
        self.assertEqual([], self.engine.calls)

    def test_api_paths_never_fall_back_to_frontend(self):
        for path in ("/api", "/api/unknown", "/api/media/", "/api/media/../../private-frame.jpg"):
            with self.subTest(path=path):
                status, headers, _ = self.request("GET", path)
                self.assertEqual(404, status)
                self.assertIn("application/json", headers["Content-Type"])

    def test_static_frontend_and_valid_navigation_work(self):
        for path in ("/", "/workbench", "/sessions/current"):
            self.assertEqual(b"<html>ProofLoop fixture</html>", self.request("GET", path)[2])
        status, headers, body = self.request("GET", "/app.js?v=1")
        self.assertEqual(200, status)
        self.assertIn("text/javascript", headers["Content-Type"])
        self.assertEqual(b"console.log('fixture')", body)

    def test_traversal_dotfiles_and_absent_assets_not_served(self):
        for path in ("/../private-frame.jpg", "/%2e%2e/private-frame.jpg", "/.env.local",
                     "/assets/.secret", "/missing.js", "/%00bad", "/x%5c..%5cfile", "/unknown%25route"):
            with self.subTest(path=path):
                self.assertEqual(404, self.request("GET", path)[0])

    def test_symlink_cannot_escape_frontend_root(self):
        (self.web / "leak.jpg").symlink_to(self.media)
        self.assertEqual(404, self.request("GET", "/leak.jpg")[0])

    def test_only_opaque_allowlisted_media_can_be_served(self):
        status, headers, body = self.request("GET", "/api/media/known-image")
        self.assertEqual(200, status)
        self.assertEqual("image/jpeg", headers["Content-Type"])
        self.assertEqual(self.media.read_bytes(), body)
        self.assertEqual("no-store", headers["Cache-Control"])
        self.assertEqual(404, self.request("GET", "/api/media/unknown-image")[0])
        self.assertEqual(404, self.request("GET", "/api/media/private-frame.jpg")[0])

    def test_unknown_exception_and_validation_messages_are_redacted(self):
        for failure, expected_status in ((RuntimeError("/private/key API_SECRET"), 500),
                                         (ValueError("/private/key API_SECRET"), 400),
                                         (ValidationError("/private/key API_SECRET"), 400)):
            self.engine.fail = failure
            status, _, body = self.request("POST", "/api/repair", {}, self.mutation_headers())
            self.assertEqual(expected_status, status)
            self.assertNotIn("API_SECRET", json.dumps(body))
            self.assertNotIn("/private/key", json.dumps(body))

    def test_trusted_typed_capture_errors_are_helpful(self):
        messages = ["Repeated or previously inspected images cannot verify a new correction",
                    "JPEG orientation metadata requires rotation. Export an upright PNG or JPEG with orientation removed, or use camera capture."]
        for message in messages:
            self.engine.fail = ValidationError(message)
            status, _, body = self.request("POST", "/api/inspect", {}, self.mutation_headers())
            self.assertEqual(400, status)
            self.assertEqual(message, body["error"]["message"])
        # Same string in a base exception is not an engine validation contract.
        self.engine.fail = ValueError(message)
        self.assertNotEqual(message, self.request("POST", "/api/inspect", {}, self.mutation_headers())[2]["error"]["message"])

    def test_busy_validation_maps_to_conflict(self):
        self.engine.busy = True
        self.engine.fail = ValueError("already running")
        status, _, body = self.request("POST", "/api/repair", {}, self.mutation_headers())
        self.assertEqual(409, status)
        self.assertEqual("busy", body["error"]["code"])

    def test_state_can_be_polled_while_inspection_handler_waits(self):
        entered = threading.Event()
        release = threading.Event()
        outcome = []

        def slow_inspection(payload):
            self.engine.busy = True
            entered.set()
            release.wait(3)
            self.engine.busy = False
            return self.engine.snapshot()

        self.engine.inspect_pair = slow_inspection
        thread = threading.Thread(target=lambda: outcome.append(self.request(
            "POST", "/api/inspect", {}, self.mutation_headers())))
        thread.start()
        try:
            self.assertTrue(entered.wait(2))
            status, _, state = self.request("GET", "/api/state")
            self.assertEqual(200, status)
            self.assertTrue(state["busy"])
        finally:
            release.set()
            thread.join(3)
        self.assertEqual(200, outcome[0][0])

    def test_head_returns_headers_without_media_or_html_body(self):
        for path in ("/", "/api/media/known-image", "/api/state"):
            status, headers, body = self.request("HEAD", path)
            self.assertEqual(200, status)
            self.assertGreater(int(headers["Content-Length"]), 0)
            self.assertEqual(b"", body)

    def test_foreign_preflight_has_no_cors_permission(self):
        status, headers, _ = self.request("OPTIONS", "/api/inspect", headers={"Origin": "https://example.com"})
        self.assertEqual(405, status)
        self.assertNotIn("Access-Control-Allow-Origin", headers)

    def test_loopback_binding_is_mandatory(self):
        with self.assertRaises(ValueError):
            create_server(self.engine, host="0.0.0.0", web_root=self.web)


if __name__ == "__main__":
    unittest.main()
