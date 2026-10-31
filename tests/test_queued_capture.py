"""Real queue/engine integration with fake clocks and authored model fixtures."""
from copy import deepcopy
import time
import threading
import unittest
from unittest.mock import patch

from proofloop.job_service import InspectionJobService
from proofloop.core import ValidationError
from proofloop.recovery import verify_recovery_receipt
from proofloop.verification_policy import inspection_binding
from tests.test_interactive import FakeClient, image
from tests import test_upload_freshness as freshness_fixtures


class QueuedCaptureTests(unittest.TestCase):
    def setUp(self):
        self.case = freshness_fixtures.UploadFreshnessTests()
        self.case.setUp()
        self.addCleanup(self.case.doCleanups)
        self.case.approve_task()
        self.engine = self.case.engine
        self.services = []
        self.addCleanup(self.close_services)

    def close_services(self):
        for service in self.services:
            service.close(timeout=2)

    def service(self, **kwargs):
        service = InspectionJobService(self.engine, clock_ms=lambda: self.case.clock_ms,
            poll_interval=.005, **kwargs)
        self.services.append(service)
        return service

    def request(self, index):
        payload = self.case.payload(index)
        return {"binding": inspection_binding(self.engine.snapshot()), "inspection": payload}

    def finish(self, service, job):
        until = time.monotonic() + 3
        while time.monotonic() < until:
            current = service.get(job["id"])
            if current["state"] not in ("queued", "running"):
                return current
            time.sleep(.005)
        self.fail("The authored queue did not reach a terminal result")

    def advance(self, value):
        self.case.clock_ms += value
        self.case.monotonic_ms += value

    def test_expired_upload_queue_rejected_before_model_dispatch(self):
        service = self.service()
        job = service.submit_inspection(self.request(31), idempotency_key="expired-queue")
        self.advance(31_001)
        service.start()
        result = self.finish(service, job)
        self.assertEqual(("failed", "validation_error", False),
            (result["state"], result["error_code"], result["dispatched"]))
        self.assertEqual([], self.case.client.ledger)
        self.assertEqual("unknown", self.engine.snapshot()["status"])
        self.assertEqual({}, self.engine._job_capture_anchors)

    def test_queue_and_inference_share_original_receipt_time(self):
        self.case.delayed_client(15_000)
        service = self.service()
        job = service.submit_inspection(self.request(31), idempotency_key="combined-age")
        self.advance(20_000)
        service.start()
        result = self.finish(service, job)
        self.assertEqual(("succeeded", "unknown"), (result["state"], result["inspection_result"]["status"]))
        state = self.engine.snapshot()
        self.assertEqual((100_000, 35_000), (state["evidence_received_at_ms"], state["evidence_age_ms"]))
        self.assertEqual(36_000, self.engine._evidence_record["evaluation_now_ms"])
        self.assertEqual(4, len(self.case.client.ledger))

    def test_fresh_queue_plus_inference_preserves_passing_clock(self):
        self.case.delayed_client(15_000)
        service = self.service()
        job = service.submit_inspection(self.request(31), idempotency_key="fresh-combined-age")
        self.advance(10_000)
        service.start()
        result = self.finish(service, job)
        self.assertEqual("pass", result["inspection_result"]["status"])
        self.assertEqual(25_000, result["inspection_result"]["freshness"]["age_ms"])
        self.assertEqual(100_000, self.engine.snapshot()["evidence_received_at_ms"])
        self.assertEqual(4, len(self.case.client.ledger))

    def test_idempotent_repeat_does_not_refresh_queued_clock(self):
        service = self.service()
        request = self.request(31)
        job = service.submit_inspection(request, idempotency_key="same-request")
        self.advance(20_000)
        same = service.submit_inspection(request, idempotency_key="same-request")
        self.assertEqual(job["id"], same["id"])
        self.advance(11_001)
        service.start()
        self.assertEqual("validation_error", self.finish(service, job)["error_code"])
        self.assertEqual([], self.case.client.ledger)

    def test_monotonic_queue_age_expires_even_when_wall_time_stands_still(self):
        service = self.service()
        job = service.submit_inspection(self.request(31), idempotency_key="monotonic-queue")
        self.case.monotonic_ms += 31_001
        service.start()
        self.assertEqual("validation_error", self.finish(service, job)["error_code"])
        self.assertEqual([], self.case.client.ledger)

    def test_observed_queue_expiry_cannot_revive_after_wall_rollback(self):
        service = self.service()
        job = service.submit_inspection(self.request(31), idempotency_key="queue-clock-rollback")
        self.case.clock_ms += 31_001
        self.assertEqual("queued", service.get(job["id"])["state"])
        self.case.clock_ms = 110_000
        service.start()
        self.assertEqual("validation_error", self.finish(service, job)["error_code"])
        self.assertEqual([], self.case.client.ledger)

    def test_missing_runtime_queue_clock_is_not_reconstructed_from_job_creation(self):
        service = self.service()
        job = service.submit_inspection(self.request(31), idempotency_key="lost-runtime-anchor")
        self.engine._job_capture_anchors.clear()  # Simulates loss of a process-local clock anchor.
        service.start()
        self.assertEqual("validation_error", self.finish(service, job)["error_code"])
        self.assertEqual([], self.case.client.ledger)

    def test_another_service_for_same_engine_keeps_original_clock(self):
        first = self.service()
        job = first.submit_inspection(self.request(31), idempotency_key="service-handoff")
        self.advance(20_000)
        self.case.delayed_client(15_000)
        second = self.service().start()
        self.assertEqual("unknown", self.finish(second, job)["inspection_result"]["status"])
        self.assertEqual(100_000, self.engine.snapshot()["evidence_received_at_ms"])

    def test_cancelled_jobs_do_not_leak_bounded_admission_anchors(self):
        service = self.service()
        for index in range(40):
            job = service.submit_inspection(self.request(index * 2 + 1), idempotency_key="cancel-" + str(index))
            service.cancel(job["id"])
            self.assertEqual({}, self.engine._job_capture_anchors)
        self.assertEqual([], self.case.client.ledger)

    def test_task_activation_clears_current_recovery_but_keeps_historical_file(self):
        self.case.client = FakeClient(wrong=True)
        self.engine.inspect_pair(self.case.payload(1))
        self.engine.begin_repair()
        self.case.client = FakeClient()
        self.engine.inspect_pair(self.case.payload(3))
        historical_path = self.engine._recovery_path
        historical_bytes = historical_path.read_bytes()
        self.case.approve_task()
        self.assertIsNone(self.engine.snapshot()["recovery"])
        self.assertEqual(historical_bytes, historical_path.read_bytes())
        self.assertTrue(verify_recovery_receipt(historical_path, root=self.case.root)["valid"])
        self.case.client = FakeClient()
        service = self.service().start()
        job = service.submit_inspection(self.request(5), idempotency_key="new-task")
        self.assertNotIn("receipt_id", self.finish(service, job)["result"])

    def test_later_same_task_job_does_not_reference_an_earlier_recovery(self):
        self.case.client = FakeClient(wrong=True)
        self.engine.inspect_pair(self.case.payload(1))
        self.engine.begin_repair()
        self.case.client = FakeClient()
        self.engine.inspect_pair(self.case.payload(3))
        historical = deepcopy(self.engine.snapshot()["recovery"])
        self.case.client = FakeClient()
        service = self.service().start()
        job = service.submit_inspection(self.request(5), idempotency_key="later-same-task")
        result = self.finish(service, job)
        self.assertEqual("pass", result["inspection_result"]["status"])
        self.assertNotIn("receipt_id", result["result"])
        self.assertEqual(historical["receipt_id"], self.engine.snapshot()["recovery"]["receipt_id"])

    def test_matching_current_recovery_is_attached_after_actual_replay(self):
        self.case.client = FakeClient(wrong=True)
        self.engine.inspect_pair(self.case.payload(1))
        self.engine.begin_repair()
        self.case.client = FakeClient()
        service = self.service().start()
        job = service.submit_inspection(self.request(3), idempotency_key="matching-repair")
        result = self.finish(service, job)
        receipt = self.engine.recovery_receipt()
        self.assertTrue(receipt["verification"]["valid"])
        self.assertEqual(receipt["receipt_id"], result["result"]["receipt_id"])
        self.assertEqual(receipt["after_revision"], result["binding"]["revision"])

    def test_missing_receipt_file_cannot_publish_a_cached_recovery_reference(self):
        self.case.client = FakeClient(wrong=True)
        self.engine.inspect_pair(self.case.payload(1))
        self.engine.begin_repair()
        self.case.client = FakeClient()
        service = self.service()
        save_result = service._save_result

        def remove_receipt(job_id, result):
            saved = save_result(job_id, result)
            self.engine._recovery_path.unlink()
            return saved

        with patch.object(service, "_save_result", side_effect=remove_receipt):
            service.start()
            job = service.submit_inspection(self.request(3), idempotency_key="missing-repair-file")
            result = self.finish(service, job)
        self.assertEqual(("succeeded", "pass"), (result["state"], result["inspection_result"]["status"]))
        self.assertNotIn("receipt_id", result["result"])
        self.assertFalse(self.engine.recovery_receipt()["verification"]["valid"])

    def test_running_job_poll_highwater_survives_rollback_and_terminal_cleanup(self):
        entered, release = threading.Event(), threading.Event()
        self.addCleanup(release.set)

        def block_last_call(count):
            if count == 4:
                entered.set()
                if not release.wait(2):
                    raise AssertionError("The fixture call was not released")

        self.case.client = FakeClient(callback=block_last_call)
        service = self.service().start()
        job = service.submit_inspection(self.request(31), idempotency_key="running-highwater")
        self.assertTrue(entered.wait(2))
        self.case.clock_ms = 131_001
        self.assertEqual("running", service.get(job["id"])["state"])
        self.case.clock_ms = 110_000
        release.set()
        result = self.finish(service, job)
        self.assertEqual("unknown", result["inspection_result"]["status"])
        self.assertEqual(("unknown", True, 31_001),
            tuple(self.engine.snapshot()[key] for key in ("status", "stale", "evidence_age_ms")))
        self.assertEqual({}, self.engine._job_capture_anchors)
        self.assertEqual(4, len(self.case.client.ledger))

    def test_a_different_request_key_cannot_refresh_the_same_uploaded_images(self):
        service = self.service()
        request = self.request(31)
        original = service.submit_inspection(request, idempotency_key="original-image-pair")
        self.advance(20_000)
        repeated = service.submit_inspection(request, idempotency_key="different-key-same-images")
        self.assertEqual(100_000, self.engine._job_capture_anchors[repeated["id"]]["received_at_ms"])
        self.advance(11_001)
        service.start()
        self.assertEqual("failed", self.finish(service, original)["state"])
        self.assertEqual("failed", self.finish(service, repeated)["state"])
        self.assertEqual([], self.case.client.ledger)

    def test_one_reused_view_and_changed_calibration_cannot_refresh_known_source_age(self):
        service = self.service()
        original = self.request(31)
        service.submit_inspection(original, idempotency_key="first-sources")
        self.advance(20_000)
        changed = deepcopy(original)
        changed["inspection"]["images"][1] = image(40)
        # Keep the original capture token and revision: only a new source and
        # calibration were supplied. The old first source is still known old.
        changed["inspection"]["regions"]["left"][0] = .06
        job = service.submit_inspection(changed, idempotency_key="changed-region-and-one-view")
        self.assertEqual(100_000, self.engine._job_capture_anchors[job["id"]]["received_at_ms"])
        self.advance(11_001)
        service.start()
        self.assertEqual("failed", self.finish(service, job)["state"])
        self.assertEqual([], self.case.client.ledger)

    def test_cancelling_before_expiry_retires_that_capture_intent(self):
        service = self.service()
        request = self.request(31)
        original = service.submit_inspection(request, idempotency_key="cancel-original")
        self.advance(10_000)
        service.cancel(original["id"])
        self.advance(10_000)
        with self.assertRaisesRegex(ValidationError, "previously inspected"):
            service.submit_inspection(request, idempotency_key="rekey-after-cancel")
        self.assertEqual({}, self.engine._job_capture_anchors)
        self.assertEqual([], self.case.client.ledger)

    def test_reversing_views_does_not_make_the_known_pair_younger(self):
        service = self.service()
        request = self.request(31)
        service.submit_inspection(request, idempotency_key="ordered-pair")
        self.advance(20_000)
        reversed_pair = deepcopy(request)
        reversed_pair["inspection"]["images"].reverse()
        job = service.submit_inspection(reversed_pair, idempotency_key="reversed-pair")
        self.assertEqual(100_000, self.engine._job_capture_anchors[job["id"]]["received_at_ms"])
        self.advance(11_001)
        service.start()
        self.assertEqual("failed", self.finish(service, job)["state"])
        self.assertEqual([], self.case.client.ledger)

    def test_direct_inspection_cannot_refresh_a_known_expired_queued_source(self):
        service = self.service()
        request = self.request(31)
        service.submit_inspection(request, idempotency_key="queue-before-direct")
        self.advance(31_001)
        with self.assertRaisesRegex(ValidationError, "queued capture expired"):
            self.engine.inspect_pair(request["inspection"])
        self.assertEqual([], self.case.client.ledger)

    def test_direct_inspection_of_still_fresh_queued_sources_keeps_their_age(self):
        service = self.service()
        request = self.request(31)
        service.submit_inspection(request, idempotency_key="fresh-queue-before-direct")
        self.advance(10_000)
        result = self.engine.inspect_pair(request["inspection"])
        self.assertEqual(("pass", 10_000, 100_000),
            (result["status"], result["evidence_age_ms"], result["evidence_received_at_ms"]))
        self.assertEqual(4, len(self.case.client.ledger))


if __name__ == "__main__":
    unittest.main()
