"""Real HTTP tests for the durable API with an authored, no-inference service."""
from copy import deepcopy
import http.client
import json
from pathlib import Path
import tempfile
import threading
import unittest

from proofloop.core import ValidationError
from proofloop.jobs import JobConflict, JobNotFound, JobValidationError
from proofloop.server import create_server, MAX_CONTROL_BYTES, MAX_JSON_BYTES
from proofloop.verification_policy import describe_capabilities
from tests.test_server import FakeEngine


JOB_ID = "a" * 32
BINDING = {"session_id": "interactive-fixture", "task_hash": "b" * 64,
    "protocol_hash": "c" * 64, "revision": 4}
PUBLIC_JOB = {"schema_version": 1, "id": JOB_ID, "kind": "inspection", "state": "queued",
    "binding": BINDING, "payload_sha256": "d" * 64, "created_ms": 1000, "updated_ms": 1000,
    "attempt": 0, "dispatched": False, "lease_expires_ms": None, "result": None,
    "error_code": None, "event_count": 1, "event_head": "e" * 64}


class FakeJobService:
    """Return only public objects; capture exact calls without a worker thread."""
    def __init__(self):
        self.calls = []
        self.fail = None

    def _call(self, method, *args, **kwargs):
        if self.fail is not None:
            raise self.fail
        self.calls.append((method, deepcopy(args), deepcopy(kwargs)))

    def capabilities(self):
        self._call("capabilities")
        return {"enabled": True, **describe_capabilities()}

    def submit_inspection(self, payload, *, idempotency_key):
        self._call("submit_inspection", payload, idempotency_key=idempotency_key)
        return deepcopy(PUBLIC_JOB)

    def get(self, job_id):
        self._call("get", job_id)
        return deepcopy(PUBLIC_JOB)

    def events(self, job_id, *, after_sequence=0):
        self._call("events", job_id, after_sequence=after_sequence)
        return [{"schema_version": 1, "job_id": JOB_ID, "sequence": after_sequence + 1,
            "event_type": "submitted", "at_ms": 1000, "previous_hash": "0" * 64,
            "job": {key: value for key, value in deepcopy(PUBLIC_JOB).items() if key not in ("event_count", "event_head")},
            "hash": "e" * 64}]

    def cancel(self, job_id):
        self._call("cancel", job_id)
        return {**deepcopy(PUBLIC_JOB), "state": "cancelled", "event_count": 2}


class ServerJobTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.web = self.root / "dist"
        self.web.mkdir()
        (self.web / "index.html").write_text("<html>Authored job API fixture</html>")
        self.media = self.root / "private.jpg"
        self.media.write_bytes(b"fixture")
        self.engine = FakeEngine(self.media)
        self.service = FakeJobService()
        self._start(self.service)

    def _start(self, service):
        self.server = create_server(self.engine, web_root=self.web, job_service=service)
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
        self.thread.start()
        self.addCleanup(self._stop)
        self.host = "127.0.0.1:" + str(self.server.server_address[1])
        self.origin = "http://" + self.host
        self.csrf = self.request("GET", "/api/state")[2]["csrf_token"]

    def _stop(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(2)

    def request(self, method, path, body=None, headers=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.server.server_address[1], timeout=3)
        if isinstance(body, dict):
            body = json.dumps(body).encode()
        connection.request(method, path, body=body, headers={"Host": self.host, **(headers or {})})
        response = connection.getresponse()
        content = response.read()
        values = dict(response.getheaders())
        result = json.loads(content) if content and "application/json" in values.get("Content-Type", "") else content
        status = response.status
        connection.close()
        return status, values, result

    def mutation_headers(self, **kwargs):
        return {"Origin": self.origin, "X-ProofLoop-Token": self.csrf, "Content-Type": "application/json", **kwargs}

    def body(self):
        return {"idempotency_key": "private-idempotency-key", "binding": deepcopy(BINDING),
            "inspection": {"capture_token": "private-capture-token", "images": ["private-image-1", "private-image-2"]}}

    def test_capabilities_are_read_only_explicit_and_do_not_start_inference(self):
        status, headers, body = self.request("GET", "/api/v1/capabilities")
        self.assertEqual(200, status)
        self.assertTrue(body["enabled"])
        self.assertFalse(body["result"]["physical_action_authorized"])
        self.assertEqual("no-store", headers["Cache-Control"])
        self.assertNotIn("Access-Control-Allow-Origin", headers)
        self.assertNotIn("csrf_token", body)
        self.assertEqual([("capabilities", (), {})], self.service.calls)
        self.assertEqual([], self.engine.calls)

    def test_submission_passes_exact_bound_payload_once_and_returns_accepted_reference(self):
        payload = self.body()
        status, headers, body = self.request("POST", "/api/v1/inspections", payload, self.mutation_headers())
        self.assertEqual(202, status)
        self.assertEqual({"job": PUBLIC_JOB}, body)
        self.assertEqual("no-store", headers["Cache-Control"])
        self.assertEqual([("submit_inspection", ({"binding": BINDING, "inspection": payload["inspection"]},),
            {"idempotency_key": payload["idempotency_key"]})], self.service.calls)
        self.assertEqual([], self.engine.calls)
        serialized = json.dumps(body)
        for secret in ("private-idempotency-key", "private-capture-token", "private-image-1", "csrf_token"):
            self.assertNotIn(secret, serialized)

    def test_poll_and_event_cursor_are_read_only_and_preserve_safe_public_results(self):
        status, headers, body = self.request("GET", "/api/v1/jobs/" + JOB_ID)
        self.assertEqual(200, status)
        self.assertEqual({"job": PUBLIC_JOB}, body)
        self.assertEqual("no-store", headers["Cache-Control"])
        status, _, body = self.request("GET", "/api/v1/jobs/" + JOB_ID + "/events?after_sequence=7")
        self.assertEqual(200, status)
        self.assertEqual(8, body["events"][0]["sequence"])
        self.assertEqual([("get", (JOB_ID,), {}), ("events", (JOB_ID,), {"after_sequence": 7})], self.service.calls)
        self.assertEqual([], self.engine.calls)

    def test_default_cursor_is_zero_and_head_does_not_return_json_body(self):
        status, _, _ = self.request("GET", "/api/v1/jobs/" + JOB_ID + "/events")
        self.assertEqual(200, status)
        self.assertEqual(("events", (JOB_ID,), {"after_sequence": 0}), self.service.calls[-1])
        for path in ("/api/v1/capabilities", "/api/v1/jobs/" + JOB_ID, "/api/v1/jobs/" + JOB_ID + "/events"):
            status, _, body = self.request("HEAD", path)
            self.assertEqual(200, status)
            self.assertEqual(b"", body)
        self.assertEqual([], self.engine.calls)

    def test_cancel_delegates_once_with_empty_control_body(self):
        status, _, body = self.request("POST", "/api/v1/jobs/" + JOB_ID + "/cancel", {}, self.mutation_headers())
        self.assertEqual(200, status)
        self.assertEqual("cancelled", body["job"]["state"])
        self.assertEqual([("cancel", (JOB_ID,), {})], self.service.calls)
        self.assertEqual([], self.engine.calls)

    def test_foreign_origin_and_rebinding_cannot_read_or_submit_jobs(self):
        for path in ("/api/v1/capabilities", "/api/v1/jobs/" + JOB_ID, "/api/v1/jobs/" + JOB_ID + "/events"):
            for headers in ({"Origin": "https://other.example"}, {"Host": "evil.example"}):
                self.assertEqual(403, self.request("GET", path, headers=headers)[0])
        for path, payload in (("/api/v1/inspections", self.body()), ("/api/v1/jobs/" + JOB_ID + "/cancel", {})):
            self.assertEqual(403, self.request("POST", path, payload, self.mutation_headers(Origin="https://other.example"))[0])
            self.assertEqual(403, self.request("POST", path, payload, self.mutation_headers(Host="evil.example"))[0])
        self.assertEqual([], self.service.calls)

    def test_job_mutations_require_current_csrf_and_explicit_origin(self):
        for path, payload in (("/api/v1/inspections", self.body()), ("/api/v1/jobs/" + JOB_ID + "/cancel", {})):
            for headers in ({"Content-Type": "application/json"},
                    {"Origin": self.origin, "Content-Type": "application/json"},
                    self.mutation_headers(**{"X-ProofLoop-Token": "wrong"})):
                self.assertEqual(403, self.request("POST", path, payload, headers)[0])
        self.assertEqual([], self.service.calls)

    def test_invalid_submission_shape_and_cancel_extra_fields_are_rejected_before_service(self):
        body = self.body()
        for payload in ({}, {key: value for key, value in body.items() if key != "binding"},
                {key: value for key, value in body.items() if key != "inspection"},
                {**body, "api_key": "private-key"}):
            self.assertEqual(400, self.request("POST", "/api/v1/inspections", payload, self.mutation_headers())[0])
        self.assertEqual(400, self.request("POST", "/api/v1/jobs/" + JOB_ID + "/cancel",
            {"confirm": True}, self.mutation_headers())[0])
        self.assertEqual([], self.service.calls)

    def test_duplicate_json_keys_cannot_ambiguously_bind_or_name_a_paid_request(self):
        original = json.dumps(self.body())
        cases = [original.replace('"idempotency_key": "private-idempotency-key"',
                    '"idempotency_key":"first", "idempotency_key":"second"'),
            original.replace('"revision": 4', '"revision":4, "revision":5'),
            original.replace('"capture_token": "private-capture-token"',
                    '"capture_token":"first", "capture_token":"second"')]
        for body in cases:
            status, _, result = self.request("POST", "/api/v1/inspections", body.encode(), self.mutation_headers())
            self.assertEqual(400, status)
            self.assertNotIn("first", json.dumps(result))
            self.assertNotIn("second", json.dumps(result))
        self.assertEqual([], self.service.calls)

    def test_invalid_cursor_is_rejected_without_service_dispatch(self):
        for query in ("after_sequence=-1", "after_sequence=true", "after_sequence=1.5", "after_sequence=",
                "after_sequence=1&after_sequence=2", "token=private-key", "after_sequence=9223372036854775808"):
            self.assertEqual(400, self.request("GET", "/api/v1/jobs/" + JOB_ID + "/events?" + query)[0], query)
        self.assertEqual([], self.service.calls)

    def test_malformed_job_ids_and_unimplemented_methods_cannot_hit_frontend(self):
        for path in ("/api/v1/jobs/not-an-id", "/api/v1/jobs/" + "a" * 31,
                "/api/v1/jobs/" + "A" * 32, "/api/v1/jobs/../../.env.local", "/api/v1/inspections",
                "/api/v1/jobs/" + JOB_ID + "/cancel"):
            self.assertEqual(404, self.request("GET", path)[0], path)
        for path in ("/api/v1/capabilities", "/api/v1/jobs/" + JOB_ID, "/api/v1/jobs/" + JOB_ID + "/events"):
            self.assertEqual(404, self.request("POST", path, {}, self.mutation_headers())[0], path)
        self.assertEqual([], self.service.calls)

    def test_bounded_upload_and_small_cancel_limits_apply_before_reading_body(self):
        for path, limit in (("/api/v1/inspections", MAX_JSON_BYTES), ("/api/v1/jobs/" + JOB_ID + "/cancel", MAX_CONTROL_BYTES)):
            status, _, body = self.request("POST", path, b"", self.mutation_headers(**{"Content-Length": str(limit + 1)}))
            self.assertEqual(413, status)
            self.assertEqual("body_too_large", body["error"]["code"])
        self.assertEqual([], self.service.calls)

    def test_typed_errors_map_without_reflecting_exception_or_payload_details(self):
        private = "/Users/private/.env.local Authorization: secret-model-key raw-provider-response"
        errors = [(JobConflict(private), 409), (JobNotFound(private), 404), (JobValidationError(private), 400),
            (ValidationError(private), 400), (RuntimeError(private), 500)]
        for error, expected in errors:
            for method, path, payload in (("POST", "/api/v1/inspections", self.body()),
                    ("GET", "/api/v1/jobs/" + JOB_ID, None),
                    ("GET", "/api/v1/jobs/" + JOB_ID + "/events", None),
                    ("POST", "/api/v1/jobs/" + JOB_ID + "/cancel", {})):
                self.service.fail = error
                status, _, body = self.request(method, path, payload, self.mutation_headers() if method == "POST" else None)
                self.assertEqual(expected, status, (type(error).__name__, path))
                self.assertIn("error", body)
                serialized = json.dumps(body)
                for secret in ("/Users/private", "secret-model-key", "raw-provider-response", "private-image-1", "private-idempotency-key"):
                    self.assertNotIn(secret, serialized)
        self.assertEqual([], self.service.calls)

    def test_disabled_service_reports_capabilities_and_rejects_jobs_without_starting_worker(self):
        # Replace the current server explicitly; create_server(None) must be safe
        # in transport tests and must never create a paid worker as a side effect.
        self._stop()
        self._start(None)
        status, _, body = self.request("GET", "/api/v1/capabilities")
        self.assertEqual(200, status)
        self.assertFalse(body["enabled"])
        for method, path, payload in (("POST", "/api/v1/inspections", self.body()),
                ("GET", "/api/v1/jobs/" + JOB_ID, None),
                ("GET", "/api/v1/jobs/" + JOB_ID + "/events", None),
                ("POST", "/api/v1/jobs/" + JOB_ID + "/cancel", {})):
            status, _, result = self.request(method, path, payload, self.mutation_headers() if method == "POST" else None)
            self.assertEqual(503, status)
            self.assertEqual("service_unavailable", result["error"]["code"])
        self.assertEqual([], self.engine.calls)
        self.assertEqual([], self.service.calls)


if __name__ == "__main__":
    unittest.main()
