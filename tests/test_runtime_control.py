"""Authored runtime failures and transport doubles; no paid or live inference."""
from copy import deepcopy
from contextlib import redirect_stdout
from io import BytesIO, StringIO
import json
import os
from pathlib import Path
import sqlite3
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock, patch
from urllib.error import URLError

from proofloop import runtime_control as runtime
from proofloop.inference_journal import AttemptJournal
from proofloop.job_service import InspectionJobService
from proofloop.jobs import JobConflict, JobValidationError
from proofloop.meta import MetaAPIError, SAM_MODEL, SPARK_MODEL
from proofloop.verification_policy import inspection_binding
from tests.test_job_service import FakeEngine, inspection_payload


def response(usage=None):
    return BytesIO(json.dumps({"status": "completed", "output": [], "usage": usage or {"input_tokens": 2, "output_tokens": 3}}).encode())


class RuntimeControlTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.journal = AttemptJournal(self.root / "runtime/attempts.sqlite3", request_limit=20)
        self.control = runtime.RuntimeControl(self.journal, self.root / "runtime")
        self.environment = patch.dict(os.environ, {"META_API_KEY": "authored-test-key"})
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def healthy(self, kind="inspection"):
        self.control.bind_worker(kind, lambda: {"worker_available": True})

    def test_shared_compiler_and_inspection_exhaustion_blocks_before_transport(self):
        journal = AttemptJournal(self.root / "small/attempts.sqlite3", request_limit=2)
        self.control = runtime.RuntimeControl(journal, self.root / "small")
        self.healthy()
        network = Mock(side_effect=lambda *a, **k: response())
        compiler = self.control.compiler_client()
        self.assertIsInstance(compiler, runtime.RuntimeCompilerClient)
        self.assertEqual((1, 0, []), (compiler.max_requests, compiler.retries, compiler.ledger))
        compiler._opener = network
        compiler._request("POST", "responses", {"model": SPARK_MODEL, "input": "private prompt"})
        inspection = self.control.inspection_client("a" * 32, lambda: None, max_requests=1, opener=network)
        inspection._request("POST", "responses", {"model": SAM_MODEL})
        for creator in (self.control.compiler_client, lambda: self.control.inspection_client("b" * 32, lambda: None, max_requests=1, opener=network)):
            with self.assertRaises(runtime.RuntimeUnavailable) as caught:
                creator()
            self.assertEqual("request_allowance_exhausted", caught.exception.code)
        self.assertEqual(2, network.call_count)
        self.assertEqual(2, journal.summary()["total_reserved_attempts"])

    def test_ten_fresh_single_call_tracking_clients_share_fixed_parent_cap(self):
        self.healthy("shadow_tracking")
        self.healthy("inspection")
        network = Mock(side_effect=lambda *a, **k: response())
        for _ in range(10):
            client = self.control.tracking_client("a" * 32, lambda: None, opener=network)
            self.assertEqual((1, 0, []), (client.max_requests, client.retries, client.ledger))
            client._request("POST", "responses", {"model": SAM_MODEL})
        client = self.control.tracking_client("a" * 32, lambda: None, opener=network)
        with self.assertRaises(MetaAPIError) as caught:
            client._request("POST", "responses", {"model": SAM_MODEL})
        self.assertEqual("persistent_request_limit", caught.exception.code)
        self.assertEqual(10, network.call_count)
        self.assertEqual(10, self.journal.summary("a" * 32)["operation_attempts"])
        with self.assertRaises(ValueError):
            self.control.inspection_client("a" * 32, lambda: None, max_requests=1, opener=network)._request("POST", "responses", {"model": SAM_MODEL})

    def test_compiler_is_one_request_per_persistent_operation_after_restart(self):
        compiler = self.control.compiler_client()
        compiler._opener = Mock(side_effect=lambda *a, **k: response())
        compiler._request("POST", "responses", {"model": SPARK_MODEL})
        reopened = AttemptJournal(self.journal.path, request_limit=20)
        with self.assertRaises(MetaAPIError):
            reopened.reserve(compiler._operation_id, "responses", {"model": SPARK_MODEL}, operation_limit=1)
        self.assertEqual(1, compiler._opener.call_count)

    def test_bound_worker_required_and_failure_is_redacted(self):
        self.assertFalse(self.control.readiness("inspection")["ready"])
        self.control.bind_worker("inspection", lambda: {"worker_available": False, "code": "private /Users/path API_KEY"})
        state = self.control.readiness("inspection")
        self.assertEqual(["worker_unavailable"], state["reason_codes"])
        with self.assertRaises(runtime.RuntimeUnavailable):
            self.control.inspection_client("a" * 32, lambda: self.fail("A worker guard ran"), max_requests=1)
        self.assertNotIn("/Users/", json.dumps(state))
        self.assertEqual(0, self.journal.summary()["total_reserved_attempts"])

    def test_compiler_does_not_require_an_asynchronous_worker(self):
        self.assertTrue(self.control.readiness("compiler")["ready"])
        self.assertIsInstance(self.control.compiler_client(), runtime.RuntimeCompilerClient)

    def test_journal_failure_storage_failure_and_capacity_fail_closed(self):
        for owner, name, effect, expected in ((self.journal, "summary", OSError("private path"), "journal_unavailable"),
                (runtime.tempfile, "mkstemp", OSError("private path"), "storage_unavailable")):
            with self.subTest(expected=expected), patch.object(owner, name, side_effect=effect):
                state = self.control.readiness("compiler")
                self.assertFalse(state["ready"])
                self.assertIn(expected, state["reason_codes"])
                with self.assertRaises(runtime.RuntimeUnavailable):
                    self.control.compiler_client()
        with patch.object(runtime.shutil, "disk_usage", return_value=type("Disk", (), {"free": 0})()):
            self.assertIn("storage_capacity_low", self.control.readiness("compiler")["reason_codes"])
        self.assertEqual(0, self.journal.summary()["total_reserved_attempts"])

    def test_guard_checked_again_before_reservation(self):
        self.healthy()
        network = Mock(side_effect=lambda *a, **k: response())
        client = self.control.inspection_client("a" * 32, lambda: None, max_requests=1, opener=network)
        with patch.object(self.control, "readiness", return_value={"ready": False, "reason_codes": ["worker_failed"]}):
            with self.assertRaises(runtime.RuntimeUnavailable):
                client._request("POST", "responses", {"model": SPARK_MODEL})
        network.assert_not_called()
        self.assertEqual(0, self.journal.summary()["total_reserved_attempts"])

    def test_provider_network_uncertainty_retained_without_retry_or_private_trace(self):
        client = self.control.compiler_client()
        client._opener = Mock(side_effect=URLError("secret-provider-body /Users/private"))
        with self.assertRaises(MetaAPIError):
            client._request("POST", "responses", {"model": SPARK_MODEL, "input": "secret-prompt"})
        self.assertEqual(1, client._opener.call_count)
        self.assertEqual(1, self.journal.summary()["outcomes"]["uncertain"])
        span = self.control.snapshot()["telemetry"]["recent_spans"][0]
        self.assertEqual(("uncertain", "provider_uncertain", "compiler", "provider_spark"),
                         (span["outcome"], span["code"], span["operation_kind"], span["stage"]))
        for secret in ("secret-provider-body", "/Users/", "secret-prompt", "authored-test-key"):
            self.assertNotIn(secret, json.dumps(self.control.snapshot()))

    def test_incomplete_response_still_retains_reported_usage_and_rejected_trace(self):
        compiler = self.control.compiler_client()
        compiler._opener = Mock(return_value=BytesIO(json.dumps({"status": "incomplete", "usage": {"input_tokens": 5, "output_tokens": 4096}}).encode()))
        with self.assertRaises(MetaAPIError):
            compiler._request("POST", "responses", {"model": SPARK_MODEL})
        self.assertEqual(4096, self.journal.summary()["reported_tokens"]["output_tokens"])
        self.assertEqual("rejected", self.control.snapshot()["telemetry"]["recent_spans"][0]["outcome"])

    def test_low_reasoning_default_and_explicit_output_cap_remain(self):
        compiler = self.control.compiler_client()
        captured = []
        def opener(request, **kwargs):
            captured.append(json.loads(request.data))
            return BytesIO(json.dumps({"status": "completed", "output": [{"type": "message", "content": [{"type": "output_text", "text": '{"ok":true}'}]}]}).encode())
        compiler._opener = opener
        compiler.spark_json("private", {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"], "additionalProperties": False}, max_output_tokens=128)
        self.assertEqual({"effort": "low"}, captured[0]["reasoning"])
        self.assertEqual(128, captured[0]["max_output_tokens"])

    def test_invalid_operation_limits_retries_and_labels_reject(self):
        self.healthy()
        for kwargs in ({"max_requests": 0}, {"max_requests": 14}, {"max_requests": True}, {"retries": 1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.control.inspection_client("a" * 32, lambda: None, **kwargs)
        for invalid in ("private", "g" * 32, True):
            with self.assertRaises(ValueError):
                self.control.inspection_client(invalid, lambda: None)
        with self.assertRaises(ValueError):
            self.control.telemetry.record("a" * 32, "compiler", "secret-prompt")

    def test_bounded_spans_persist_lifetime_counts_and_valid_local_otel_export(self):
        with patch.object(runtime, "MAX_SPANS", 8):
            for index in range(12):
                self.control.record("a" * 32, "compiler", "verification", duration_ns=(index+1)*1_000_000,
                                    code="secret-provider-body" if index==0 else "none")
            snapshot = self.control.telemetry.snapshot()
            self.assertEqual(8, snapshot["retained_span_count"])
            self.assertEqual(12, sum(row["observations"] for row in snapshot["metrics"]))
            self.assertEqual(12.0, snapshot["metrics"][0]["stage_window_p95_ms"])
            reopened = runtime.RuntimeTelemetry(self.control.telemetry.path)
            self.assertEqual(snapshot, reopened.snapshot())
            exported = reopened.export_otel()
            spans = exported["resourceSpans"][0]["scopeSpans"][0]["spans"]
            self.assertEqual(8, len(spans))
            self.assertTrue(all(len(row["traceId"])==32 and len(row["spanId"])==16 for row in spans))
            self.assertNotIn("secret-provider-body", json.dumps(exported))

    def test_telemetry_failure_does_not_replace_provider_outcome_but_blocks_next_admission(self):
        client = self.control.compiler_client()
        client._opener = Mock(side_effect=lambda *a, **k: response())
        with patch.object(self.control.telemetry, "record", side_effect=OSError("private path")):
            self.assertEqual("completed", client._request("POST", "responses", {"model": SPARK_MODEL})["status"])
        self.assertEqual(1, self.journal.summary()["outcomes"]["ok"])
        self.assertIn("telemetry_unavailable", self.control.readiness("compiler")["reason_codes"])
        with self.assertRaises(runtime.RuntimeUnavailable):
            self.control.compiler_client()

    def test_corrupt_private_telemetry_cannot_inject_secrets_into_public_projection(self):
        self.control.record("a" * 32, "compiler", "verification")
        with sqlite3.connect(self.control.telemetry.path) as db:
            db.execute("UPDATE spans SET code='secret-provider-body /Users/private'")
        projected = self.control.snapshot()
        self.assertFalse(projected["telemetry"]["available"])
        self.assertIn("telemetry_unavailable", projected["readiness"]["reason_codes"])
        self.assertNotIn("secret-provider-body", json.dumps(projected))

    def test_finite_metric_label_capacity_is_enforced_before_unbounded_growth(self):
        with patch.object(runtime, "MAX_METRIC_ROWS", 1):
            self.control.record("a" * 32, "compiler", "verification")
            self.control.record("a" * 32, "compiler", "verification", code="verdict_unknown")
            self.assertIn("telemetry_unavailable", self.control.readiness("compiler")["reason_codes"])
            with sqlite3.connect(self.control.telemetry.path) as db:
                self.assertEqual(1, db.execute("SELECT count(*) FROM counters").fetchone()[0])

    def test_local_cli_exports_structured_or_otel_json_without_model_calls(self):
        self.control.record("a"*32,"compiler","provider_spark",model="spark")
        for chosen in ("json","otel"):
            stream=StringIO()
            with redirect_stdout(stream),patch("proofloop.meta.MetaClient._request",side_effect=AssertionError("CLI cannot call models")):
                self.assertEqual(0,runtime.main([str(self.control.telemetry.path),"--format",chosen]))
            result=json.loads(stream.getvalue())
            self.assertIn("resourceSpans" if chosen=="otel" else "metrics",result)
            self.assertNotIn(str(self.root),stream.getvalue())


class ControlledJobServiceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.engine = FakeEngine(self.root)
        self.journal = AttemptJournal(self.root / "runtime/attempts.sqlite3", request_limit=2)
        self.control = runtime.RuntimeControl(self.journal, self.root / "runtime")
        self.service = InspectionJobService(self.engine, runtime_control=self.control, poll_interval=.005)
        self.addCleanup(self.service.close)

    def request(self):
        return {"binding": inspection_binding(self.engine.snapshot()), "inspection": inspection_payload()}

    def finish(self, job):
        deadline=time.monotonic()+3
        while time.monotonic()<deadline:
            value=self.service.get(job["id"])
            if value["state"] not in ("queued", "running"):
                return value
            time.sleep(.005)
        self.fail("Authored inspection worker did not finish")

    def test_unstarted_dead_and_error_workers_cannot_admit_new_work(self):
        for state in ("unstarted", "error", "dead"):
            if state=="error":self.service._worker_error="private-error /Users/path"
            if state=="dead":
                self.service._worker_error=None
                self.service._thread=threading.Thread(target=lambda:None)
                self.service._thread.start()
                self.service._thread.join()
            with self.subTest(state=state),self.assertRaises(runtime.RuntimeUnavailable):
                self.service.submit_inspection(self.request(),idempotency_key=state)
        self.assertEqual([],self.service.store.list_jobs())
        self.assertEqual(0,self.engine.calls)
        self.assertEqual(0,self.journal.summary()["total_reserved_attempts"])

    def test_live_worker_readiness_and_staged_private_timings(self):
        self.service.start()
        self.assertTrue(self.service.readiness()["ready"])
        job=self.service.submit_inspection(self.request(),idempotency_key="authored")
        self.assertEqual("succeeded",self.finish(job)["state"])
        spans=self.control.snapshot()["telemetry"]["recent_spans"]
        # A successful publication can become visible just before execution's
        # finally span; wait for that authored worker bookkeeping only.
        deadline=time.monotonic()+1
        while not any(row["stage"]=="execution" for row in spans) and time.monotonic()<deadline:
            time.sleep(.005);spans=self.control.snapshot()["telemetry"]["recent_spans"]
        self.assertTrue({"admission","queue_wait","job_validation","verification","artifact_publication","execution"}.issubset({r["stage"] for r in spans}))
        self.assertTrue(all(row["operation_id"]==job["id"] for row in spans))
        self.assertTrue(all(row["duration_ns"]>=0 for row in spans))
        for secret in ("private-capture-secret","private-image-one",str(self.root)):
            self.assertNotIn(secret,json.dumps(self.control.snapshot()))

    def test_exhausted_allowance_blocks_new_job_but_preserves_exact_idempotent_retrieval(self):
        self.service.start();request=self.request()
        job=self.service.submit_inspection(request,idempotency_key="existing")
        result=self.finish(job)
        for index in range(2):self.journal.reserve(f"{index+1:032x}","responses",{"model":SPARK_MODEL},operation_limit=1)
        self.assertFalse(self.service.readiness()["ready"])
        self.assertEqual(result,self.service.submit_inspection(request,idempotency_key="existing"))
        with self.assertRaises(runtime.RuntimeUnavailable):self.service.submit_inspection(self.request(),idempotency_key="new")
        self.assertEqual(1,self.engine.calls)

    def test_unknown_verdict_reports_only_allowlisted_gap_labels(self):
        self.engine.result_status="unknown";self.service.start()
        result=self.finish(self.service.submit_inspection(self.request(),idempotency_key="unknown"))
        self.assertEqual("unknown",result["inspection_result"]["status"])
        metrics=self.control.snapshot()["telemetry"]["metrics"]
        self.assertTrue(any(r["code"]=="verdict_unknown" for r in metrics))
        self.assertTrue(any(r["stage"]=="evidence_request" and r["code"]=="identity_ambiguous" for r in metrics))

    def test_intent_lookup_recovers_verified_historical_result_after_binding_change_and_shutdown(self):
        self.service.start();request=self.request()
        job=self.service.submit_inspection(request,idempotency_key="recover",submission_fingerprint="a"*64)
        finished=self.finish(job)
        with self.engine._lock:self.engine.state["revision"]+=1
        self.service.close()
        intent={"schema_version":2,"idempotency_key":"recover","binding":request["binding"],"submission_fingerprint":"a"*64}
        self.assertEqual(finished,self.service.lookup_intent(intent))
        self.assertEqual(1,self.engine.calls)
        self.assertEqual(0,self.journal.summary()["total_reserved_attempts"])

    def test_intent_lookup_validation_absence_and_fingerprint_conflict_do_not_dispatch(self):
        self.service.start();request=self.request()
        intent={"schema_version":2,"idempotency_key":"absent","binding":request["binding"],"submission_fingerprint":"a"*64}
        self.assertIsNone(self.service.lookup_intent(intent))
        for change in ({"schema_version":True},{"schema_version":1},{"extra":"private"},{"submission_fingerprint":"invalid"}):
            with self.subTest(change=change),self.assertRaises(JobValidationError):
                self.service.lookup_intent({**intent,**change})
        job=self.service.submit_inspection(request,idempotency_key="absent",submission_fingerprint="a"*64)
        self.finish(job)
        with self.assertRaises(JobConflict):self.service.lookup_intent({**intent,"submission_fingerprint":"b"*64})
        self.assertEqual(1,self.engine.calls)

    def test_intent_lookup_rechecks_result_bytes_instead_of_restoring_unverified_verdict(self):
        self.service.start();request=self.request()
        job=self.service.submit_inspection(request,idempotency_key="tampered",submission_fingerprint="a"*64)
        self.finish(job)
        self.service._result_path(job["id"]).write_text('{"private":"corrupt result"}')
        found=self.service.lookup_intent({"schema_version":2,"idempotency_key":"tampered","binding":request["binding"],"submission_fingerprint":"a"*64})
        self.assertFalse(found["integrity"]["result_valid"])
        self.assertIsNone(found["result"])
        self.assertIsNone(found["inspection_result"])
        self.assertEqual(1,self.engine.calls)


if __name__=="__main__":
    unittest.main()
