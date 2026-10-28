"""Acquisition lifecycle integration with authored images and injected clients.

These checks count fake calls and inspect public state. They do not measure
model performance, establish owned capture timing, or prove a physical repair.
"""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from proofloop.core import ValidationError
from proofloop.interactive import InteractiveEngine
from tests.test_interactive import FakeClient, FakeCompileClient, REGIONS, image


class MixedClient(FakeClient):
    def __init__(self):
        super().__init__(wrong=True)

    def segment_image(self, path, concept, *, decode_masks):
        result = super().segment_image(path, concept, decode_masks=decode_masks)
        if concept == "spoon":
            result["objects"] = []
        return result


class ClientQueue:
    def __init__(self, *clients):
        self.waiting = list(clients)
        self.used = []

    def __call__(self):
        if not self.waiting:
            raise AssertionError("Unexpected inspection dispatched after its finite budget")
        client = self.waiting.pop(0)
        self.used.append(client)
        return client

    @property
    def call_count(self):
        return sum(len(client.ledger) for client in self.used)


class AcquisitionEngineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.clock = patch("proofloop.interactive._now_ms", return_value=100_000)
        self.clock.start()

    def tearDown(self):
        self.clock.stop()
        self.temp.cleanup()

    def engine(self, *clients):
        self.queue = ClientQueue(*clients)
        self.subject = InteractiveEngine(self.root, client_factory=self.queue)
        return self.subject

    def payload(self, index, *, new_inspection=True):
        state = self.subject.new_inspection() if new_inspection else self.subject.snapshot()
        return {"capture_token": state["capture_token"], "source": "upload",
                "images": [image(index), image(index + 1)], "regions": deepcopy(REGIONS),
                "confirm_regions": True, "confirm_unique_targets": True,
                "confirm_fresh_capture": True, "confirm_pair_spacing": True}

    def inspect(self, index):
        return self.subject.inspect_pair(self.payload(index))

    def test_initial_unknown_allows_only_two_followups_then_blocks_before_calls(self):
        self.engine(FakeClient(covered=True), FakeClient(covered=True), FakeClient(covered=True))
        initial = self.inspect(1)
        self.assertEqual(("unknown", "ready", 0, 2),
            (initial["status"], initial["acquisition"]["status"], initial["acquisition"]["attempts_used"],
             initial["acquisition"]["attempts_remaining"]))
        first = self.inspect(3)
        self.assertEqual(("ready", 1, 1), (first["acquisition"]["status"],
            first["acquisition"]["attempts_used"], first["acquisition"]["attempts_remaining"]))
        second = self.inspect(5)
        self.assertEqual(("unknown", "budget_exhausted", 2, 0), (second["status"],
            second["acquisition"]["status"], second["acquisition"]["attempts_used"],
            second["acquisition"]["attempts_remaining"]))
        self.assertIsNone(second["proposal"])
        self.assertIsNone(second["verified_at_ms"])
        self.assertEqual(12, self.queue.call_count)
        for action in (self.subject.new_inspection, self.subject.begin_repair):
            with self.subTest(action=action.__name__), self.assertRaisesRegex(ValidationError, "acquisition stopped"):
                action()
        with self.assertRaisesRegex(ValidationError, "acquisition stopped"):
            self.subject.inspect_pair({})
        self.assertEqual(12, self.queue.call_count)
        self.assertEqual(3, len(self.queue.used))

    def test_invalid_pair_does_not_spend_followup_or_consume_current_token(self):
        self.engine(FakeClient(covered=True), FakeClient(covered=True))
        self.inspect(1)
        valid = self.payload(3)
        invalid = deepcopy(valid)
        invalid["images"][1] = deepcopy(invalid["images"][0])
        with self.assertRaises(ValidationError):
            self.subject.inspect_pair(invalid)
        state = self.subject.snapshot()
        self.assertEqual((0, 2), (state["acquisition"]["attempts_used"], state["acquisition"]["attempts_remaining"]))
        self.assertEqual(valid["capture_token"], state["capture_token"])
        self.assertEqual(4, self.queue.call_count)
        state = self.subject.inspect_pair(valid)
        self.assertEqual(1, state["acquisition"]["attempts_used"])
        self.assertEqual(8, self.queue.call_count)

    def test_reusing_either_source_is_rejected_without_spending_followup(self):
        self.engine(FakeClient(covered=True), FakeClient(covered=True))
        self.inspect(1)
        valid = self.payload(3)
        for old_index, position in ((1, 0), (2, 1)):
            bad = deepcopy(valid)
            bad["images"][position] = image(old_index)
            with self.subTest(old_index=old_index), self.assertRaisesRegex(ValidationError, "previously inspected"):
                self.subject.inspect_pair(bad)
            self.assertEqual(0, self.subject.snapshot()["acquisition"]["attempts_used"])
            self.assertEqual(4, self.queue.call_count)
        self.assertEqual(1, self.subject.inspect_pair(valid)["acquisition"]["attempts_used"])

    def test_repeated_new_inspection_and_begin_repair_cannot_reset_unresolved_budget(self):
        self.engine(FakeClient(covered=True), FakeClient(covered=True), FakeClient(covered=True))
        self.inspect(1)
        self.inspect(3)
        for action in (self.subject.new_inspection, self.subject.new_inspection, self.subject.begin_repair):
            state = action()
            self.assertEqual((1, 1), (state["acquisition"]["attempts_used"], state["acquisition"]["attempts_remaining"]))
            self.assertEqual({"not_detected"}, {code for request in state["acquisition"]["requests"]
                                                for code in request["gap_codes"]})
        state = self.subject.inspect_pair(self.payload(5, new_inspection=False))
        self.assertEqual("budget_exhausted", state["acquisition"]["status"])
        self.assertEqual(12, self.queue.call_count)

    def test_api_error_consumes_allocated_followup_and_cannot_offer_free_retries(self):
        self.engine(FakeClient(covered=True), FakeClient(fail_on=1), FakeClient(fail_on=1))
        self.inspect(1)
        first = self.inspect(3)
        self.assertEqual(("error", "unknown", "ready", 1),
            (first["phase"], first["status"], first["acquisition"]["status"], first["acquisition"]["attempts_used"]))
        second = self.inspect(5)
        self.assertEqual(("error", "unknown", "budget_exhausted", 2),
            (second["phase"], second["status"], second["acquisition"]["status"], second["acquisition"]["attempts_used"]))
        self.assertEqual(6, self.queue.call_count)
        with self.assertRaisesRegex(ValidationError, "acquisition stopped"):
            self.subject.new_inspection()
        self.assertEqual(6, self.queue.call_count)

    def test_initial_api_error_still_has_finite_followup_chain(self):
        self.engine(FakeClient(fail_on=1), FakeClient(fail_on=1), FakeClient(fail_on=1))
        initial = self.inspect(1)
        self.assertEqual("error", initial["phase"])
        self.assertEqual(0, initial["acquisition"]["attempts_used"])
        self.assertEqual({"missing_current_evidence"},
                         {code for request in initial["acquisition"]["requests"] for code in request["gap_codes"]})
        self.inspect(3)
        final = self.inspect(5)
        self.assertEqual("budget_exhausted", final["acquisition"]["status"])
        self.assertEqual(3, self.queue.call_count)

    def test_pending_capture_preserves_specific_missing_object_and_request(self):
        self.engine(MixedClient())
        state = self.inspect(1)
        self.assertEqual("fail", state["status"])
        self.assertEqual(["spoon"], state["acquisition"]["object_ids"])
        self.assertEqual(["spoon"], state["inspection_request"]["object_ids"])
        before = deepcopy(state["acquisition"]["requests"])
        pending = self.subject.new_inspection()
        self.assertTrue(all(check["status"] == "unknown" for check in pending["checks"]))
        self.assertEqual(before, pending["acquisition"]["requests"])
        self.assertEqual(["spoon"], pending["inspection_request"]["object_ids"])
        self.assertIn("entire spoon", " ".join(pending["inspection_request"]["instructions"]))
        self.assertIn("all declared task targets", " ".join(pending["inspection_request"]["instructions"]))
        self.assertEqual(4, self.queue.call_count)

    def test_mixed_fail_unknown_never_authorizes_a_target_move(self):
        self.engine(MixedClient())
        state = self.inspect(1)
        self.assertEqual({"fork": "fail", "spoon": "unknown"},
                         {check["object_id"]: check["status"] for check in state["checks"]})
        self.assertIsNone(state["proposal"])
        self.assertFalse(state["acquisition"]["physical_action_allowed"])
        self.assertNotIn("Move the fork", " ".join(state["inspection_request"]["instructions"]))
        repair = self.subject.begin_repair()
        self.assertIsNone(repair["proposal"])
        self.assertEqual("inspection", repair["recovery"]["kind"])
        self.assertFalse(repair["recovery"]["physical_action_execution_verified"])
        self.assertEqual(4, self.queue.call_count)

    def test_known_pass_and_fail_end_chain_allowing_later_new_inspection(self):
        for wrong, expected in ((False, "pass"), (True, "fail")):
            with self.subTest(expected=expected):
                # Each subcase needs a separate root because source bytes are
                # intentionally blocked across local engine restarts.
                with tempfile.TemporaryDirectory() as location:
                    self.queue = ClientQueue(FakeClient(covered=True), FakeClient(wrong=wrong), FakeClient(covered=True))
                    self.subject = InteractiveEngine(Path(location), client_factory=self.queue)
                    self.inspect(1)
                    decided = self.inspect(3)
                    self.assertEqual(expected, decided["status"])
                    self.assertIsNone(decided["acquisition"])
                    new = self.inspect(5)
                    self.assertEqual(("ready", 0, 2), (new["acquisition"]["status"],
                        new["acquisition"]["attempts_used"], new["acquisition"]["attempts_remaining"]))
                    self.assertEqual(12, self.queue.call_count)

    def test_second_followup_can_pass_before_limit_is_presented_as_terminal(self):
        self.engine(FakeClient(covered=True), FakeClient(covered=True), FakeClient())
        self.inspect(1)
        self.inspect(3)
        final = self.inspect(5)
        self.assertEqual("pass", final["status"])
        self.assertEqual("verified", final["phase"])
        self.assertIsNone(final["acquisition"])
        self.assertEqual(12, self.queue.call_count)

    def test_reviewed_task_change_resets_budget_and_still_blocks_old_source_bytes(self):
        self.engine(FakeClient(covered=True), FakeClient(covered=True), FakeClient(covered=True), FakeClient(covered=True))
        self.inspect(1)
        self.inspect(3)
        self.inspect(5)
        old = self.subject.snapshot()["active_task"]["protocol_hash"]
        compiler = FakeCompileClient()
        self.subject._compile_client_factory = lambda: compiler
        draft_state = self.subject.draft_task({"title": "Reviewed reversed placement",
            "requirements": "Place fork in RIGHT and spoon in LEFT.", "objects": ["fork", "spoon"],
            "confirm_unique_targets": True})
        draft = draft_state["task_draft"]
        self.assertEqual("budget_exhausted", draft_state["acquisition"]["status"])
        activated = self.subject.activate_task({"draft_id": draft["draft_id"], "draft_hash": draft["draft_hash"],
            "confirm_requirements": True, "confirm_unique_targets": True})
        self.assertNotEqual(old, activated["active_task"]["protocol_hash"])
        self.assertIsNone(activated["acquisition"])
        repeated = self.payload(1, new_inspection=False)
        with self.assertRaisesRegex(ValidationError, "previously inspected"):
            self.subject.inspect_pair(repeated)
        self.assertEqual(12, self.queue.call_count)
        state = self.subject.inspect_pair(self.payload(7, new_inspection=False))
        self.assertEqual(("ready", 0, 2), (state["acquisition"]["status"],
            state["acquisition"]["attempts_used"], state["acquisition"]["attempts_remaining"]))
        self.assertEqual(16, self.queue.call_count)
        self.assertEqual(1, len(compiler.ledger))


if __name__ == "__main__":
    unittest.main()
