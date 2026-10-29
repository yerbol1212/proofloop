"""Adversarial admission and client protocol checks; no model transport."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sqlite3
import tempfile
import threading
import unittest
from unittest.mock import patch

from proofloop.client import ClientError, ProofLoopClient
from proofloop import jobs
from proofloop.job_service import InspectionJobService
from tests.test_jobs import BINDING
from tests.test_job_service import FakeEngine, inspection_payload
from proofloop.verification_policy import inspection_binding


class QueueCapacityTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(); self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / "jobs.sqlite3"
        self.stores = [jobs.JobStore(self.path) for _ in range(8)]

    def submit(self, store, key, payload=None):
        return store.submit("inspection", payload or {"image": "authored"}, idempotency_key=key,
                            binding=BINDING, now_ms=10)

    def compete(self, payload):
        barrier = threading.Barrier(len(self.stores))
        def admit(pair):
            index, store = pair
            barrier.wait()
            try:
                return self.submit(store, "distinct-" + str(index), payload)
            except jobs.JobCapacityError:
                return None
        with ThreadPoolExecutor(max_workers=8) as pool:
            return list(pool.map(admit, enumerate(self.stores)))

    def test_distinct_keys_cannot_overfill_pending_queue_across_instances(self):
        with patch.object(jobs, "MAX_PENDING_JOBS", 3):
            accepted = [item for item in self.compete({"image": "authored"}) if item]
            self.assertEqual(3, len(accepted))
            original = accepted[0]
            key = next("distinct-" + str(index) for index, store in enumerate(self.stores)
                       if store.lookup_submission("inspection", {"image": "authored"},
                           idempotency_key="distinct-" + str(index), binding=BINDING) is not None)
            historical = self.stores[0].lookup_submission("inspection", {"image": "authored"},
                idempotency_key=key, binding=BINDING)
            self.assertEqual(historical["id"], self.submit(self.stores[1], key)["id"])
            self.stores[0].cancel(original["id"], now_ms=11)
            self.assertEqual("queued", self.submit(self.stores[-1], "new-slot")["state"])

    def test_retained_byte_cap_counts_terminal_jobs_and_is_atomic(self):
        payload = {"image": "owned-λ" * 20}
        size = len(jobs._canonical(payload).encode("utf-8"))
        with patch.object(jobs, "MAX_RETAINED_PAYLOAD_BYTES", size * 3):
            accepted = [item for item in self.compete(payload) if item]
            self.assertEqual(3, len(accepted))
            for item in accepted:
                self.stores[0].cancel(item["id"], now_ms=11)
            with self.assertRaises(jobs.JobCapacityError):
                self.submit(self.stores[-1], "new-after-cancel", payload)
            with sqlite3.connect(self.path) as db:
                retained = db.execute("SELECT sum(length(CAST(payload_json AS BLOB))) FROM jobs").fetchone()[0]
            self.assertEqual(size * 3, retained)

    def test_exact_original_lookup_has_no_capture_or_readiness_side_effect(self):
        engine = FakeEngine(self.path.parent / "engine")
        runner = InspectionJobService(engine)
        self.addCleanup(runner.close)
        request = {"binding": inspection_binding(engine.snapshot()), "inspection": inspection_payload()}
        original = runner.submit_inspection(request, idempotency_key="owned-original")
        with engine._lock:
            engine.state["revision"] += 1
        self.assertEqual(original["id"], runner.lookup_submission(request, idempotency_key="owned-original")["id"])
        self.assertIsNone(runner.lookup_submission(request, idempotency_key="not-admitted"))
        self.assertEqual(0, engine.calls)
        self.assertEqual(1, len(runner.store.list_jobs()))


class ClientProtocolTests(unittest.TestCase):
    def test_malformed_error_envelopes_never_escape_as_attribute_errors(self):
        for error in (None, "provider body", [], 42, {"code": "unsafe/private/provider"}):
            class Response:
                status = 503
                def read(self, limit): return json.dumps({"error": error}).encode()
                def getheader(self, key, default=""): return "application/json"
            class Connection:
                def __init__(self, *args, **kwargs): pass
                def request(self, *args, **kwargs): pass
                def getresponse(self): return Response()
                def close(self): pass
            with self.subTest(error=error), patch("proofloop.client.http.client.HTTPConnection", Connection):
                with self.assertRaisesRegex(ClientError, "^local_request_rejected$"):
                    ProofLoopClient().operations()


if __name__ == "__main__":
    unittest.main()
