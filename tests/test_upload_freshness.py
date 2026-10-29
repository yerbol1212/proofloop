"""Elapsed server time for uploaded evidence, using authored transport fixtures.

No model calls or trusted capture-time claims: uploads retain operator-declared
spacing, while actual server elapsed time consumes their freshness window.
"""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from proofloop.core import Protocol, ValidationError
from proofloop.interactive import InteractiveEngine
from proofloop.recovery import digest, replay_evidence, seal_receipt, verify_recovery_receipt
from tests.test_interactive import FakeClient, FakeCompileClient, REGIONS, image


class UploadFreshnessTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.clock_ms = 100_000
        self.monotonic_ms = 500_000
        clock = patch("proofloop.interactive._now_ms", side_effect=lambda: self.clock_ms)
        clock.start()
        self.addCleanup(clock.stop)
        monotonic = patch("proofloop.interactive._monotonic_ms", side_effect=lambda: self.monotonic_ms)
        monotonic.start()
        self.addCleanup(monotonic.stop)
        self.client = FakeClient()
        self.clients = []

        def factory():
            self.clients.append(self.client)
            return self.client

        self.engine = InteractiveEngine(self.root, client_factory=factory,
            compile_client_factory=lambda: FakeCompileClient({"fork": "left", "spoon": "right"}))

    def payload(self, index):
        state = self.engine.new_inspection()
        return {"capture_token": state["capture_token"], "source": "upload",
            "images": [image(index), image(index + 1)], "regions": deepcopy(REGIONS),
            "confirm_regions": True, "confirm_unique_targets": True,
            "confirm_fresh_capture": True, "confirm_pair_spacing": True}

    def delayed_client(self, delay_ms, **options):
        def finish(count):
            if count == 4:
                self.clock_ms += delay_ms
        self.client = FakeClient(callback=finish, **options)

    def approve_task(self):
        draft = self.engine.draft_task({"requirements": "Fork LEFT and spoon RIGHT",
            "objects": ["fork", "spoon"], "confirm_unique_targets": True})["task_draft"]
        self.engine.activate_task({"draft_id": draft["draft_id"], "draft_hash": draft["draft_hash"],
            "confirm_requirements": True, "confirm_unique_targets": True})

    def binding(self):
        state = self.engine.snapshot()
        return {"task_hash": state["active_task"]["task_hash"], "revision": state["revision"]}

    def test_expired_initial_upload_saves_unknown_and_requests_new_evidence(self):
        self.delayed_client(31_001)
        state = self.engine.inspect_pair(self.payload(1))
        self.assertEqual(("unknown", "unknown", True, 31_001),
            (state["status"], state["phase"], state["stale"], state["evidence_age_ms"]))
        self.assertIsNone(state["verified_at_ms"])
        self.assertIsNone(state["proposal"])
        self.assertEqual(("ready", 0), (state["acquisition"]["status"], state["acquisition"]["attempts_used"]))
        self.assertTrue(all("stale_evidence" in row["gap_codes"] for row in state["acquisition"]["requests"]))
        self.assertEqual(32_001, self.engine._evidence_record["evaluation_now_ms"])
        self.assertEqual("unknown", replay_evidence(self.engine._evidence_record, self.root)["status"])
        saved = json.loads(next(self.engine._directory.glob("inspection-*/evaluation.json")).read_text())
        self.assertEqual("unknown", saved["evaluation"]["status"])
        persisted = json.loads((self.engine._directory / "state.json").read_text())
        self.assertEqual("unknown", persisted["status"])
        self.assertEqual(4, len(self.client.ledger))

    def test_expired_wrong_placement_does_not_offer_a_move(self):
        self.delayed_client(31_001, wrong=True)
        state = self.engine.inspect_pair(self.payload(1))
        self.assertEqual("unknown", state["status"])
        self.assertTrue(all(row["status"] == "unknown" for row in state["checks"]))
        self.assertIsNone(state["proposal"])
        self.assertFalse(state["diagnostic"]["repair_proposal_eligible"])

    def test_expired_repair_completion_keeps_a_valid_needs_evidence_receipt(self):
        self.client = FakeClient(wrong=True)
        self.assertEqual("fail", self.engine.inspect_pair(self.payload(1))["status"])
        self.engine.begin_repair()
        self.delayed_client(31_001)
        state = self.engine.inspect_pair(self.payload(3))
        self.assertEqual("unknown", state["status"])
        self.assertEqual(("needs_evidence", "unknown"),
            (state["recovery"]["outcome"], state["recovery"]["after_status"]))
        self.assertTrue(self.engine.verify_recovery()["valid"])
        self.assertEqual(32_001, self.engine._recovery_record["after"]["evaluation_now_ms"])
        self.assertEqual(8, sum(len(client.ledger) for client in self.clients))

    def test_nonexpired_delayed_upload_replays_without_double_counting_latency(self):
        self.approve_task()
        self.delayed_client(10_000)
        state = self.engine.inspect_pair(self.payload(1))
        self.assertEqual("pass", state["status"])
        self.assertEqual(11_000, self.engine._evidence_record["evaluation_now_ms"])
        self.clock_ms = 125_000  # 25 seconds total, not 10 + 25 seconds.
        checked = self.engine.check_workflow(self.binding())
        self.assertTrue(checked["workflow"]["can_continue"])
        self.assertEqual("pass", checked["status"])
        self.assertEqual(4, len(self.client.ledger))
        self.clock_ms = 130_001
        with self.assertRaises(ValidationError):
            self.engine.check_workflow(self.binding())
        self.assertFalse(self.engine.snapshot()["workflow"]["can_continue"])

    def test_expired_completions_preserve_the_finite_unknown_followup_chain(self):
        for index, expected_used in ((1, 0), (3, 1), (5, 2)):
            with self.subTest(attempt=expected_used):
                self.delayed_client(31_001)
                state = self.engine.inspect_pair(self.payload(index))
                self.assertEqual("unknown", state["status"])
                self.assertEqual(expected_used, state["acquisition"]["attempts_used"])
        self.assertEqual("budget_exhausted", state["acquisition"]["status"])
        with self.assertRaisesRegex(ValidationError, "acquisition stopped"):
            self.engine.new_inspection()
        self.assertEqual(12, sum(len(client.ledger) for client in self.clients))

    def test_oldest_camera_view_expired_with_latest_still_recent_cannot_close_repair(self):
        self.client = FakeClient(wrong=True)
        self.engine.inspect_pair(self.payload(1))
        self.engine.begin_repair()
        payload = self.payload(3)
        payload["source"] = "camera"
        payload["images"][0]["captured_at_ms"] = 100_000
        payload["images"][1]["captured_at_ms"] = 101_500
        self.clock_ms = 102_000
        self.delayed_client(28_001)
        state = self.engine.inspect_pair(payload)
        self.assertEqual(("unknown", True, 30_001),
            (state["status"], state["stale"], state["evidence_age_ms"]))
        self.assertEqual(("needs_evidence", "unknown"),
            (state["recovery"]["outcome"], state["recovery"]["after_status"]))
        self.assertTrue(self.engine.verify_recovery()["valid"])
        self.assertEqual(130_001, self.engine._evidence_record["evaluation_now_ms"])
        self.assertEqual("unknown", replay_evidence(self.engine._evidence_record, self.root)["status"])
        self.assertEqual(("ready", 0),
            (state["acquisition"]["status"], state["acquisition"]["attempts_used"]))
        self.assertTrue(all(row["gap_codes"] == ["stale_evidence"] for row in state["acquisition"]["requests"]))

        # An old-style success assertion at this exact time now fails closed;
        # the replay does not invent a newer capture or an earlier completion.
        asserted_success = deepcopy(self.engine._recovery_record)
        asserted_success.update(outcome="fresh_goal_verified_after_requested_repair", after_status="pass")
        for row in asserted_success["after_checks"]:
            row["status"] = "pass"
        self.assertFalse(verify_recovery_receipt(seal_receipt(asserted_success), root=self.root)["valid"])

        # Historical timelines retain their own reviewed contract window; they
        # do not inherit the workbench's extra 30-second current-pair ceiling.
        # This is another authored contract over the same authored test images.
        historical = deepcopy(self.engine._evidence_record)
        historical["source_kind"] = "recorded"
        for check in historical["contract"]["protocol"]["checks"]:
            check["max_age_ms"] = 60_000
        protocol = Protocol.from_dict(historical["contract"]["protocol"])
        historical["contract"]["protocol_hash"] = protocol.content_hash
        historical["contract_hash"] = digest(historical["contract"])
        self.assertEqual("pass", replay_evidence(historical, self.root)["status"])
        historical["source_kind"] = "camera"
        self.assertEqual("unknown", replay_evidence(historical, self.root)["status"])

    def test_fresh_camera_pair_retains_its_original_epoch_time(self):
        payload = self.payload(1)
        payload["source"] = "camera"
        payload["images"][0]["captured_at_ms"] = 100_000
        payload["images"][1]["captured_at_ms"] = 101_500
        self.clock_ms = 102_000
        state = self.engine.inspect_pair(payload)
        self.assertEqual(("pass", False, 2000),
            (state["status"], state["stale"], state["evidence_age_ms"]))
        self.assertEqual(102_000, self.engine._evidence_record["evaluation_now_ms"])
        self.assertEqual([100_000, 101_500],
            [row["timestamp_ms"] for row in self.engine._evidence_record["frames"]])
        self.assertEqual("pass", replay_evidence(self.engine._evidence_record, self.root)["status"])

    def test_later_expiration_does_not_rewrite_an_actually_fresh_completed_receipt(self):
        self.client = FakeClient(wrong=True)
        self.engine.inspect_pair(self.payload(1))
        self.engine.begin_repair()
        self.delayed_client(10_000)
        completed = self.engine.inspect_pair(self.payload(3))
        self.assertEqual("pass", completed["status"])
        self.assertEqual("fresh_goal_verified_after_requested_repair", completed["recovery"]["outcome"])
        saved_receipt = deepcopy(self.engine._recovery_record)
        self.clock_ms = 130_001
        expired = self.engine.snapshot()
        self.assertEqual(("unknown", True), (expired["status"], expired["stale"]))
        self.assertEqual(saved_receipt, self.engine._recovery_record)
        self.assertTrue(self.engine.verify_recovery()["valid"])
        self.assertEqual("fresh_goal_verified_after_requested_repair", expired["recovery"]["outcome"])

    def test_observed_expiration_cannot_revive_after_wall_clock_rollback(self):
        self.approve_task()
        self.engine.inspect_pair(self.payload(1))
        self.clock_ms = 131_001
        self.assertTrue(self.engine.snapshot()["stale"])
        self.clock_ms = 110_000
        rolled_back = self.engine.snapshot()
        self.assertEqual(("unknown", True, 31_001),
            (rolled_back["status"], rolled_back["stale"], rolled_back["evidence_age_ms"]))
        with self.assertRaises(ValidationError):
            self.engine.check_workflow(self.binding())
        self.assertFalse(self.engine._replay_workflow_evidence(rolled_back)["valid"])
        self.assertEqual(4, len(self.client.ledger))

    def test_preexpiry_wall_rollback_cannot_freeze_upload_age(self):
        self.approve_task()
        self.engine.inspect_pair(self.payload(1))
        self.clock_ms = 120_000
        self.monotonic_ms = 520_000
        self.assertEqual(20_000, self.engine.snapshot()["evidence_age_ms"])
        self.clock_ms = 90_000
        self.monotonic_ms = 531_001
        expired = self.engine.snapshot()
        self.assertEqual(("unknown", True, 31_001),
            (expired["status"], expired["stale"], expired["evidence_age_ms"]))
        with self.assertRaises(ValidationError):
            self.engine.check_workflow(self.binding())
        self.assertEqual(4, len(self.client.ledger))

    def test_camera_age_advances_monotonically_after_wall_rollback(self):
        payload = self.payload(1)
        payload["source"] = "camera"
        payload["images"][0]["captured_at_ms"] = 100_000
        payload["images"][1]["captured_at_ms"] = 101_500
        self.clock_ms = 102_000
        self.assertEqual("pass", self.engine.inspect_pair(payload)["status"])
        self.clock_ms = 99_000
        self.monotonic_ms = 529_001
        expired = self.engine.snapshot()
        self.assertEqual(("unknown", True, 31_001),
            (expired["status"], expired["stale"], expired["evidence_age_ms"]))
        current_time, _ = self.engine._current_evidence_time()
        self.assertEqual(131_001, current_time)
        self.assertFalse(self.engine._replay_workflow_evidence(expired)["valid"])

    def test_missing_runtime_clock_anchor_cannot_reconstruct_current_authority(self):
        self.approve_task()
        self.engine.inspect_pair(self.payload(1))
        self.engine._evidence_received_monotonic_ms = None
        unavailable = self.engine.snapshot()
        self.assertEqual(("unknown", True, None),
            (unavailable["status"], unavailable["stale"], unavailable["evidence_age_ms"]))
        with self.assertRaises(ValidationError):
            self.engine.check_workflow(self.binding())
        self.assertIsNone(self.engine._before_recovery_evidence())
        self.assertEqual(4, len(self.client.ledger))

    def test_slow_persistence_preserves_evaluation_history_but_no_current_permission(self):
        self.approve_task()
        self.client = FakeClient(wrong=True)
        self.engine.inspect_pair(self.payload(1))
        self.engine.begin_repair()
        self.delayed_client(10_000)
        write = self.engine._write

        def slow_evaluation_write(path, value):
            if path.name == "evaluation.json":
                self.clock_ms += 21_001
            return write(path, value)

        with patch.object(self.engine, "_write", side_effect=slow_evaluation_write):
            delivered = self.engine.inspect_pair(self.payload(3))
        self.assertEqual(("unknown", True), (delivered["status"], delivered["stale"]))
        self.assertEqual("fresh_goal_verified_after_requested_repair", delivered["recovery"]["outcome"])
        self.assertEqual(11_000, self.engine._recovery_record["after"]["evaluation_now_ms"])
        self.assertTrue(self.engine.verify_recovery()["valid"])
        with self.assertRaises(ValidationError):
            self.engine.check_workflow(self.binding())
        self.assertFalse(self.engine.snapshot()["workflow"]["can_continue"])
        self.assertEqual(8, sum(len(client.ledger) for client in self.clients))

    def test_monotonic_model_latency_expires_an_upload_even_if_wall_clock_rolls_back(self):
        def finish(count):
            if count == 4:
                self.clock_ms = 90_000
                self.monotonic_ms = 531_001
        self.client = FakeClient(callback=finish)
        state = self.engine.inspect_pair(self.payload(1))
        self.assertEqual(("unknown", True, 31_001),
            (state["status"], state["stale"], state["evidence_age_ms"]))
        self.assertEqual(32_001, self.engine._evidence_record["evaluation_now_ms"])
        self.assertEqual("unknown", replay_evidence(self.engine._evidence_record, self.root)["status"])
        self.assertEqual(4, len(self.client.ledger))

    def test_wall_rollback_does_not_reset_unresolved_acquisition_budget(self):
        for index in (1, 3, 5):
            self.client = FakeClient(covered=True)
            state = self.engine.inspect_pair(self.payload(index))
        self.assertEqual(("budget_exhausted", 2),
            (state["acquisition"]["status"], state["acquisition"]["attempts_used"]))
        self.clock_ms -= 20_000
        self.monotonic_ms += 31_001
        rolled_back = self.engine.snapshot()
        self.assertEqual(("budget_exhausted", 2),
            (rolled_back["acquisition"]["status"], rolled_back["acquisition"]["attempts_used"]))
        self.assertTrue(all(row["gap_codes"] == ["stale_evidence"] for row in rolled_back["acquisition"]["requests"]))
        with self.assertRaisesRegex(ValidationError, "acquisition stopped"):
            self.engine.new_inspection()
        self.assertEqual(12, sum(len(client.ledger) for client in self.clients))


if __name__ == "__main__":
    unittest.main()
