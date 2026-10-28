"""Durable worker integration and failure injection, using authored fake inference."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from proofloop.core import ValidationError
from proofloop.interactive import InteractiveEngine
from proofloop.job_service import InspectionJobService
from proofloop.jobs import JobConflict, JobIntegrityError, JobStore, JobValidationError
from proofloop.meta import MetaAPIError
from proofloop.verification_policy import inspection_binding, validate_binding
from tests.test_interactive import FakeClient, FakeCompileClient, REGIONS, image
from tests.test_verification_policy import pending_state, state_fixture


def inspection_payload():
    return {"capture_token": "private-capture-secret", "source": "upload",
            "images": [{"data_url": "private-image-one"}, {"data_url": "private-image-two"}],
            "regions": {"dispense": [0, 0, .4, 1], "staging": [.6, 0, 1, 1]},
            "confirm_regions": True, "confirm_unique_targets": True,
            "confirm_fresh_capture": True, "confirm_pair_spacing": True}


class FakeEngine:
    """Schema-faithful engine fixture; not a claim about physical verification."""
    def __init__(self, root):
        self.root = Path(root)
        self._lock = threading.RLock()
        self.state = pending_state()
        self.calls = 0
        self.entered = threading.Event()
        self.release = threading.Event()
        self.release.set()
        self.result_status = "pass"
        self.result_error = None

    def snapshot(self):
        with self._lock:
            return deepcopy(self.state)

    def _validate_pair(self, payload):
        if payload["capture_token"] != "private-capture-secret":
            raise ValidationError("Invalid fixture capture token")
        if payload.get("source") != "upload":
            raise ValidationError("Invalid fixture source")

    def inspect_pair(self, payload, *, expected_binding=None):
        with self._lock:
            validate_binding(expected_binding, self.state)
            self._validate_pair(payload)
            self.calls += 1
            self.state.update(phase="checking", busy=True)
            self.entered.set()
        self.release.wait(3)
        with self._lock:
            result = state_fixture()
            for key in ("session_id", "revision", "active_task"):
                result[key] = deepcopy(self.state[key])
            if self.result_status == "fail":
                result.update(status="fail", phase="mistake")
                result["checks"][0].update(status="fail", observed_region="staging")
            elif self.result_status == "unknown":
                result.update(status="unknown", phase="unknown")
                result["checks"][0].update(status="unknown", observed_region=None, evidence_ids=[])
                result["acquisition"] = {"status": "ready", "attempts_used": 0, "attempts_remaining": 2,
                    "max_attempts": 2, "requests": [{"check_id": "glass-under-spout", "object_id": "glass",
                    "region_id": "dispense", "gap_codes": ["identity_ambiguous"]}]}
            if self.result_error:
                result.update(phase="error", status="unknown", error=self.result_error)
            self.state = result
            return self.snapshot()


class JobServiceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.engine = FakeEngine(self.root)
        self.services = []
        self.addCleanup(self.cleanup_services)

    def cleanup_services(self):
        self.engine.release.set()
        for service in self.services:
            service.close(timeout=2)

    def service(self, **kwargs):
        service = InspectionJobService(self.engine, **{"poll_interval": .005, **kwargs})
        self.services.append(service)
        return service

    def request(self):
        return {"binding": inspection_binding(self.engine.snapshot()), "inspection": inspection_payload()}

    def finish(self, service, job, timeout=3):
        until = time.monotonic() + timeout
        while time.monotonic() < until:
            result = service.get(job["id"])
            if result["state"] not in ("queued", "running"):
                return result
            time.sleep(.005)
        self.fail("The authored worker did not reach a terminal state")

    def test_submission_is_durable_and_does_not_dispatch_until_started(self):
        service = self.service()
        self.assertFalse(service.capabilities()["jobs"]["worker_available"])
        job = service.submit_inspection(self.request(), idempotency_key="first")
        self.assertEqual("queued", job["state"])
        self.assertEqual(0, self.engine.calls)
        service.start()
        result = self.finish(service, job)
        self.assertEqual("succeeded", result["state"])
        self.assertEqual("pass", result["inspection_result"]["status"])
        self.assertTrue(result["integrity"]["result_valid"])
        self.assertEqual(1, self.engine.calls)
        self.assertFalse(result["physical_action_authorized"])
        self.assertEqual("historical_completion_snapshot", result["result_scope"])

    def test_completed_retry_returns_same_job_after_engine_binding_changes(self):
        service = self.service().start()
        request = self.request()
        job = service.submit_inspection(request, idempotency_key="retry")
        first = self.finish(service, job)
        with self.engine._lock:
            self.engine.state["revision"] += 1
        second = service.submit_inspection(request, idempotency_key="retry")
        self.assertEqual(first, second)
        self.assertEqual(1, self.engine.calls)

    def test_parallel_same_key_submissions_dispatch_once(self):
        service = self.service()
        request = self.request()
        with ThreadPoolExecutor(max_workers=5) as pool:
            jobs = list(pool.map(lambda _: service.submit_inspection(request, idempotency_key="same"), range(10)))
        self.assertEqual(1, len({job["id"] for job in jobs}))
        service.start()
        self.assertEqual("succeeded", self.finish(service, jobs[0])["state"])
        self.assertEqual(1, self.engine.calls)

    def test_reused_key_with_changed_payload_is_rejected_before_dispatch(self):
        service = self.service()
        request = self.request()
        service.submit_inspection(request, idempotency_key="conflict")
        request["inspection"]["images"][0]["data_url"] = "different"
        with self.assertRaises(JobConflict):
            service.submit_inspection(request, idempotency_key="conflict")
        self.assertEqual(0, self.engine.calls)

    def test_invalid_outer_inner_and_image_fields_never_enter_the_queue(self):
        service = self.service()
        requests = [None, {}, dict(self.request(), extra=True), deepcopy(self.request()), deepcopy(self.request())]
        requests[-2]["inspection"]["physical_action"] = "move"
        requests[-1]["inspection"]["images"][0]["path"] = "/private/secret"
        for request in requests:
            with self.subTest(request=request), self.assertRaises(ValidationError):
                service.submit_inspection(request, idempotency_key="invalid")
        self.assertEqual([], service.store.list_jobs())

    def test_invalid_capture_or_stale_binding_is_not_admitted(self):
        service = self.service()
        request = self.request()
        request["inspection"]["capture_token"] = "wrong"
        with self.assertRaises(ValidationError):
            service.submit_inspection(request, idempotency_key="wrong")
        request = self.request()
        request["binding"]["revision"] += 1
        with self.assertRaises(ValidationError):
            service.submit_inspection(request, idempotency_key="stale")
        self.assertEqual([], service.store.list_jobs())

    def test_payload_is_deep_copied_before_returning_to_caller(self):
        service = self.service()
        request = self.request()
        job = service.submit_inspection(request, idempotency_key="copy")
        request["inspection"]["capture_token"] = "mutated"
        service.start()
        self.assertEqual("succeeded", self.finish(service, job)["state"])
        self.assertEqual(1, self.engine.calls)

    def test_queued_cancellation_remains_terminal_without_inference(self):
        service = self.service()
        job = service.submit_inspection(self.request(), idempotency_key="cancel")
        result = service.cancel(job["id"])
        self.assertEqual("cancelled", result["state"])
        service.start()
        self.assertEqual("cancelled", service.get(job["id"])["state"])
        self.assertEqual(0, self.engine.calls)

    def test_binding_change_before_worker_dispatch_fails_without_call(self):
        service = self.service()
        job = service.submit_inspection(self.request(), idempotency_key="change")
        with self.engine._lock:
            self.engine.state["revision"] += 1
        service.start()
        result = self.finish(service, job)
        self.assertEqual(("failed", "stale_binding", False), (result["state"], result["error_code"], result["dispatched"]))
        self.assertEqual(0, self.engine.calls)

    def test_distinct_keys_for_one_capture_do_not_repeat_inference(self):
        service = self.service()
        request = self.request()
        jobs = [service.submit_inspection(request, idempotency_key=key) for key in ("one", "two")]
        service.start()
        results = [self.finish(service, job) for job in jobs]
        self.assertEqual(["failed", "succeeded"], sorted(job["state"] for job in results))
        self.assertEqual(1, self.engine.calls)

    def test_guard_is_passed_to_callback_and_rechecks_binding_before_another_request(self):
        def callback(payload, binding, job_id, guard):
            guard()
            self.assertRegex(job_id, r"^[0-9a-f]{32}$")
            with self.engine._lock:
                self.engine.state["revision"] += 1
            guard()
            self.fail("A changed binding must stop the second provider dispatch")
        service = self.service(inspect_callback=callback).start()
        job = service.submit_inspection(self.request(), idempotency_key="guard")
        result = self.finish(service, job)
        self.assertEqual(("failed", "stale_binding"), (result["state"], result["error_code"]))
        self.assertEqual(0, self.engine.calls)
        self.assertIsNone(result["inspection_result"])

    def test_completion_from_another_revision_is_rejected(self):
        def callback(payload, binding, job_id, guard):
            guard()
            result = self.engine.inspect_pair(payload, expected_binding=binding)
            result["revision"] += 1
            return result
        service = self.service(inspect_callback=callback).start()
        result = self.finish(service, service.submit_inspection(self.request(), idempotency_key="late"))
        self.assertEqual(("failed", "stale_binding"), (result["state"], result["error_code"]))
        self.assertIsNone(result["inspection_result"])

    def test_unknown_preserves_specific_missing_evidence_and_fail_remains_fail(self):
        for status in ("unknown", "fail"):
            with self.subTest(status=status):
                self.engine = FakeEngine(self.root / status)
                self.engine.result_status = status
                service = self.service().start()
                result = self.finish(service, service.submit_inspection(self.request(), idempotency_key=status))
                normalized = result["inspection_result"]
                self.assertEqual(("succeeded", status), (result["state"], normalized["status"]))
                if status == "unknown":
                    self.assertEqual(["identity_ambiguous"], normalized["evidence_requests"][0]["gap_codes"])
                self.assertFalse(normalized["physical_action_authorized"])

    def test_engine_error_is_failed_job_not_successful_unknown(self):
        self.engine.result_error = "private provider failure containing details"
        service = self.service().start()
        result = self.finish(service, service.submit_inspection(self.request(), idempotency_key="error"))
        self.assertEqual(("failed", "model_error"), (result["state"], result["error_code"]))
        self.assertIsNone(result["inspection_result"])
        self.assertNotIn("private provider", json.dumps(result))

    def test_direct_provider_exception_is_redacted_and_not_retried(self):
        attempts = []
        def callback(payload, binding, job_id, guard):
            guard()
            attempts.append(job_id)
            raise MetaAPIError("private-token-and-provider-body")
        service = self.service(inspect_callback=callback).start()
        result = self.finish(service, service.submit_inspection(self.request(), idempotency_key="provider"))
        self.assertEqual(("failed", "provider_error"), (result["state"], result["error_code"]))
        self.assertEqual(1, len(attempts))
        self.assertNotIn("private-token", json.dumps(result))

    def test_heartbeat_keeps_a_long_inspection_claim_alive(self):
        self.engine.release.clear()
        service = self.service(lease_ms=90).start()
        job = service.submit_inspection(self.request(), idempotency_key="heartbeat")
        self.assertTrue(self.engine.entered.wait(1))
        time.sleep(.22)
        self.assertEqual("running", service.get(job["id"])["state"])
        self.engine.release.set()
        result = self.finish(service, job)
        self.assertEqual("succeeded", result["state"])
        self.assertIn("heartbeat", [event["event_type"] for event in service.events(job["id"])])

    def test_expired_dispatched_lease_stays_interrupted_after_late_completion(self):
        clock = [1000]
        self.engine.release.clear()
        service = self.service(lease_ms=100, clock_ms=lambda: clock[0]).start()
        job = service.submit_inspection(self.request(), idempotency_key="expired")
        self.assertTrue(self.engine.entered.wait(1))
        clock[0] = 1200
        service.store.recover_expired(now_ms=clock[0])
        self.engine.release.set()
        result = self.finish(service, job)
        self.assertEqual(("interrupted", "outcome_unknown"), (result["state"], result["error_code"]))
        time.sleep(.03)
        self.assertEqual("interrupted", service.get(job["id"])["state"])
        self.assertIsNone(result["inspection_result"])
        self.assertEqual(1, self.engine.calls)

    def test_completed_result_is_available_from_new_service_without_new_model_call(self):
        service = self.service().start()
        request = self.request()
        job = service.submit_inspection(request, idempotency_key="restart")
        original = self.finish(service, job)
        service.close()
        restarted = self.service().start()
        self.assertEqual(original["inspection_result"], restarted.get(job["id"])["inspection_result"])
        self.assertEqual(job["id"], restarted.submit_inspection(request, idempotency_key="restart")["id"])
        self.assertEqual(1, self.engine.calls)

    def test_atomic_result_publication_failure_cannot_publish_success(self):
        service = self.service()
        job = service.submit_inspection(self.request(), idempotency_key="disk")
        with patch.object(service, "_save_result", side_effect=OSError("private disk path")):
            service.start()
            result = self.finish(service, job)
        self.assertEqual(("failed", "artifact_error"), (result["state"], result["error_code"]))
        self.assertIsNone(result["inspection_result"])

    def test_tampered_missing_or_symlinked_result_is_not_exposed(self):
        for mutation in ("edit", "missing", "symlink"):
            with self.subTest(mutation=mutation):
                self.engine = FakeEngine(self.root / mutation)
                service = self.service().start()
                result = self.finish(service, service.submit_inspection(self.request(), idempotency_key=mutation))
                path = service.artifact_root / (result["id"] + ".json")
                if mutation == "edit":
                    path.write_text('{"private-provider":"do not expose"}')
                else:
                    data = path.read_bytes()
                    path.unlink()
                    if mutation == "symlink":
                        other = self.root / "private-result-copy.json"
                        other.write_bytes(data)
                        path.symlink_to(other)
                public = service.get(result["id"])
                self.assertFalse(public["integrity"]["result_valid"])
                self.assertIsNone(public["inspection_result"])
                self.assertNotIn("private-provider", json.dumps(public))

    def test_public_projection_and_events_exclude_images_tokens_paths_and_exceptions(self):
        service = self.service().start()
        job = service.submit_inspection(self.request(), idempotency_key="public")
        result = self.finish(service, job)
        text = json.dumps([result, service.events(job["id"]), service.capabilities()])
        for secret in ("private-image", "private-capture-secret", str(self.root), "claim_token", "worker_id", "data_url"):
            self.assertNotIn(secret, text)
        self.assertEqual(0o600, (service.artifact_root / (job["id"] + ".json")).stat().st_mode & 0o777)
        self.assertFalse(list(service.artifact_root.glob(".result-*")))

    def test_start_is_idempotent_close_refuses_new_admission_but_keeps_reads(self):
        service = self.service()
        job = service.submit_inspection(self.request(), idempotency_key="lifecycle")
        self.assertIs(service, service.start())
        self.assertTrue(service.capabilities()["jobs"]["worker_available"])
        self.assertIs(service, service.start())
        self.finish(service, job)
        service.close()
        self.assertFalse(service.capabilities()["jobs"]["worker_available"])
        self.assertEqual("succeeded", service.get(job["id"])["state"])
        with self.assertRaises(JobConflict):
            service.start()
        with self.assertRaises(JobConflict):
            service.submit_inspection(self.request(), idempotency_key="after-close")

    def test_invalid_lifecycle_parameters_are_rejected(self):
        for parameters in ({"lease_ms": True}, {"lease_ms": 1}, {"poll_interval": False}, {"poll_interval": float("nan")}):
            with self.subTest(parameters=parameters), self.assertRaises(JobValidationError):
                self.service(**parameters)
        service = self.service()
        with self.assertRaises(JobValidationError):
            service.close(timeout=-1)

    def test_corrupt_queue_fails_closed_without_dispatch(self):
        service = self.service()
        job = service.submit_inspection(self.request(), idempotency_key="corrupt")
        import sqlite3
        with sqlite3.connect(service.store.db_path) as connection:
            connection.execute("UPDATE jobs SET payload_json='{}' WHERE id=?", (job["id"],))
        service.start()
        until = time.monotonic() + 1
        while service.capabilities()["jobs"]["worker_error"] is None and time.monotonic() < until:
            time.sleep(.005)
        self.assertEqual("worker_journal_unavailable", service.capabilities()["jobs"]["worker_error"])
        self.assertEqual(0, self.engine.calls)
        with self.assertRaises(JobIntegrityError):
            service.get(job["id"])

    def test_actual_interactive_engine_runs_fake_model_through_bound_worker(self):
        fake = FakeClient()
        engine = InteractiveEngine(self.root / "actual-engine", client_factory=lambda: fake,
                                   compile_client_factory=lambda: FakeCompileClient({"fork": "left", "spoon": "right"}))
        draft = engine.draft_task({"requirements": "Put fork in LEFT and spoon in RIGHT.", "objects": ["fork", "spoon"],
                                   "confirm_unique_targets": True})["task_draft"]
        engine.activate_task({"draft_id": draft["draft_id"], "draft_hash": draft["draft_hash"],
                              "confirm_requirements": True, "confirm_unique_targets": True})
        state = engine.new_inspection()
        payload = {"capture_token": state["capture_token"], "source": "upload", "images": [image(31), image(32)],
                   "regions": deepcopy(REGIONS), "confirm_regions": True, "confirm_unique_targets": True,
                   "confirm_fresh_capture": True, "confirm_pair_spacing": True}
        service = InspectionJobService(engine, poll_interval=.005).start()
        self.services.append(service)
        job = service.submit_inspection({"binding": inspection_binding(state), "inspection": payload}, idempotency_key="engine")
        result = self.finish(service, job)
        self.assertEqual("succeeded", result["state"])
        self.assertEqual("pass", result["inspection_result"]["status"])
        self.assertEqual("injected_client_not_verified_live", result["inspection_result"]["inference_origin"])
        self.assertEqual(4, len(fake.ledger))


if __name__ == "__main__":
    unittest.main()
