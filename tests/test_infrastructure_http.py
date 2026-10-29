import base64
import hashlib
import http.client
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from proofloop.server import create_server
from proofloop.inference_journal import AttemptJournal
from proofloop.runtime_control import RuntimeControl
from proofloop.tracking_service import TrackingService, result_from_policy
from proofloop.replay_bundle import replay
from proofloop.client import ProofLoopClient
from tests.test_server import FakeEngine
from tests.test_server_jobs import FakeJobService, PUBLIC_JOB, BINDING


class RecoverableService(FakeJobService):
    def submit_inspection(self, payload, *, idempotency_key, submission_fingerprint=None):
        self._call("submit", payload, idempotency_key=idempotency_key, submission_fingerprint=submission_fingerprint)
        return PUBLIC_JOB

    def lookup_intent(self, payload):
        self._call("lookup", payload)
        return PUBLIC_JOB


class InfrastructureHttpTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        web = self.root / "web"; web.mkdir()
        (web / "index.html").write_text("<html>Infrastructure fixture</html>")
        self.control = RuntimeControl(AttemptJournal(self.root / "attempts.sqlite3"), self.root / "runtime", minimum_free_bytes=0)
        self.control.bind_worker("inspection", lambda: {"worker_available": True})
        def cached(payload, identifier, guard):
            guard()
            source = next(item for item in replay()["policies"] if item["clip_id"] == "remote_new")
            return result_from_policy(source["policy"], source["warning_measurements"], mode="cached_demo", origin="cached_development_projection")
        self.tracking = TrackingService(self.root / "tracking", cached_callback=cached)
        self.control.bind_worker("shadow_tracking", self.tracking.readiness)
        self.tracking.start()
        self.jobs = RecoverableService()
        self.server = create_server(FakeEngine(self.root / "private.jpg"), web_root=web, job_service=self.jobs,
            runtime_control=self.control, attempt_journal=self.control.journal, tracking_service=self.tracking,
            inference_enabled=False)
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": .01}, daemon=True)
        self.thread.start()
        self.addCleanup(self.stop)
        self.host = "127.0.0.1:" + str(self.server.server_address[1])
        self.origin = "http://" + self.host
        self.token = self.request("GET", "/api/state")[1]["csrf_token"]

    def stop(self):
        self.server.shutdown(); self.server.server_close(); self.thread.join(2); self.tracking.close()

    def request(self, method, path, payload=None, headers=None):
        body = json.dumps(payload).encode() if isinstance(payload, dict) else payload
        connection = http.client.HTTPConnection("127.0.0.1", self.server.server_address[1], timeout=5)
        connection.request(method, path, body=body, headers={"Host": self.host, **(headers or {})})
        response = connection.getresponse()
        result = json.loads(response.read())
        status = response.status; connection.close()
        return status, result

    def headers(self, **extra):
        return {"Origin": self.origin, "X-ProofLoop-Token": self.token, "Content-Type": "application/json", **extra}

    def test_liveness_readiness_and_redacted_operational_exports_are_distinct(self):
        self.assertEqual((200, {"ok": True}), self.request("GET", "/api/health"))
        self.assertEqual(200, self.request("GET", "/api/ready")[0])
        status, result = self.request("GET", "/api/v1/operations")
        self.assertEqual(200, status); self.assertFalse(result["inference_enabled"])
        self.assertNotIn(str(self.root), json.dumps(result))
        self.assertEqual(200, self.request("GET", "/api/v1/operations/otel")[0])
        self.tracking.close()
        self.assertEqual(503, self.request("GET", "/api/ready")[0])
        self.assertFalse(ProofLoopClient(self.server.server_address[1]).readiness()["ready"])
        self.assertEqual(200, self.request("GET", "/api/health")[0])

    def test_cached_diagnostic_post_reload_poll_and_idempotence_make_zero_provider_calls(self):
        payload = {"idempotency_key": "fixture-unique-key", "mode": "cached_demo", "fixture_id": "latest-tracking"}
        with patch("proofloop.meta.MetaClient._request", side_effect=AssertionError("No API")) as transport:
            status, result = self.request("POST", "/api/v1/tracking/jobs", payload, self.headers())
            self.assertEqual(202, status); identifier = result["job"]["job_id"]
            for _ in range(100):
                _, result = self.request("GET", "/api/v1/tracking/jobs/" + identifier)
                if result["job"]["state"] not in {"queued", "running"}: break
                time.sleep(.01)
            self.assertEqual("succeeded", result["job"]["state"])
            diagnostic = result["job"]["result"]
            self.assertEqual(0, diagnostic["request_attempts"])
            self.assertEqual("cached_development_projection", diagnostic["inference_origin"])
            self.assertEqual("unknown", diagnostic["frames"][-1]["status"])
            self.assertFalse(diagnostic["physical_action_authorized"])
            repeated = self.request("POST", "/api/v1/tracking/jobs", payload, self.headers())[1]
            self.assertEqual(identifier, repeated["job"]["job_id"])
            self.assertEqual(200, self.request("GET", "/api/v1/tracking/jobs/" + identifier + "/events")[0])
        transport.assert_not_called()

    def test_offline_service_refuses_live_requests_and_unsafe_asset_uploads(self):
        status, result = self.request("POST", "/api/v1/tracking/jobs", {"mode": "real_meta_video"}, self.headers())
        self.assertEqual(503, status); self.assertEqual("inference_disabled", result["error"]["code"])
        for payload in ({"media_type": "video/mp4", "data": "!bad!"}, {"path": "/private/file"},
                        {"media_type": "video/mp4", "data": base64.b64encode(b"invalid").decode()}):
            self.assertEqual(400, self.request("POST", "/api/v1/tracking/assets", payload, self.headers())[0])

    def test_recovery_body_receipt_is_checked_before_submission_and_lookup_has_no_inspection(self):
        body = json.dumps({"idempotency_key": "opaque", "binding": BINDING, "inspection": {"images": []}}).encode()
        status, _ = self.request("POST", "/api/v1/inspections", body,
            self.headers(**{"X-ProofLoop-Submission-Fingerprint": "0" * 64}))
        self.assertEqual(400, status); self.assertEqual([], self.jobs.calls)
        fingerprint = hashlib.sha256(body).hexdigest()
        self.assertEqual(202, self.request("POST", "/api/v1/inspections", body,
            self.headers(**{"X-ProofLoop-Submission-Fingerprint": fingerprint}))[0])
        self.assertEqual(fingerprint, self.jobs.calls[-1][2]["submission_fingerprint"])
        lookup = {"schema_version": 2, "idempotency_key": "opaque", "binding": BINDING,
                  "submission_fingerprint": fingerprint}
        status, result = self.request("POST", "/api/v1/inspections/lookup", lookup, self.headers())
        self.assertEqual(200, status); self.assertTrue(result["found"])
        self.assertEqual("lookup", self.jobs.calls[-1][0])
        self.assertNotIn("inspection", self.jobs.calls[-1][1][0])

    def test_all_new_mutations_keep_the_existing_origin_and_csrf_boundary(self):
        for path in ("/api/v1/tracking/jobs", "/api/v1/tracking/assets", "/api/v1/inspections/lookup"):
            self.assertEqual(403, self.request("POST", path, {}, {"Content-Type": "application/json"})[0])
        self.assertEqual(403, self.request("GET", "/api/v1/replay", headers={"Origin": "https://foreign.invalid"})[0])

    def test_public_replay_endpoint_recomputes_the_published_policy_scores(self):
        status, report = self.request("GET", "/api/v1/replay")
        self.assertEqual(200, status)
        self.assertEqual(0, report["provider_requests"])
        self.assertEqual(3, report["cohorts"]["exposed_episode_2"]["methods"]["v2_veto_current"]["metrics"]["positives_retained"])

    def test_external_python_client_submits_and_recovers_one_cached_job(self):
        client = ProofLoopClient(self.server.server_address[1])
        first = client.cached_tracking("external-client-intent")
        terminal = client.wait_tracking(first["job_id"], timeout=5)
        self.assertEqual("succeeded", terminal["state"])
        self.assertEqual("cached_demo", terminal["result"]["mode"])
        second = client.cached_tracking("external-client-intent")
        self.assertEqual(first["job_id"], second["job_id"])
        self.assertEqual(0, client.replay()["provider_requests"])


if __name__ == "__main__":
    unittest.main()
