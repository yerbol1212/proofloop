"""Offline authored recovery receipts; no physical execution or model claims."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from proofloop.core import ValidationError
from proofloop.interactive import InteractiveEngine
from proofloop.recovery import seal_receipt, verify_recovery_receipt
from tests import test_interactive as interactive_fixtures
from tests.test_interactive import FakeClient, FakeCompileClient, REGIONS, image, picture


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.client = FakeClient(wrong=True)
        self.engine = InteractiveEngine(self.root, client_factory=lambda: self.client)

    def payload(self, index):
        state = self.engine.new_inspection()
        return {"capture_token": state["capture_token"], "source": "upload", "images": [image(index), image(index + 1)],
                "regions": deepcopy(REGIONS), "confirm_regions": True, "confirm_unique_targets": True,
                "confirm_fresh_capture": True, "confirm_pair_spacing": True}

    def requested_repair(self):
        self.engine.inspect_pair(self.payload(1))
        return self.engine.begin_repair()

    def completed_repair(self, **client_options):
        self.requested_repair()
        self.client = FakeClient(**client_options)
        return self.engine.inspect_pair(self.payload(3))

    def test_request_binds_supported_failure_and_plan_without_claiming_execution(self):
        state = self.requested_repair()
        summary = state["recovery"]
        self.assertEqual("repair", summary["kind"])
        self.assertEqual("fail", summary["before_status"])
        self.assertEqual("awaiting_fresh_evidence", summary["outcome"])
        self.assertGreater(summary["requested_revision"], summary["before_revision"])
        self.assertTrue(any("Move the fork" in step for step in summary["plan_steps"]))
        self.assertFalse(summary["physical_action_execution_verified"])
        self.assertTrue(self.engine.verify_recovery()["valid"])
        self.assertEqual("requested", self.engine.verify_recovery()["scope"])
        self.assertEqual("unknown", state["status"])

    def test_fresh_goal_closes_linked_repair_and_replays_offline(self):
        state = self.completed_repair()
        summary = state["recovery"]
        self.assertEqual("pass", state["status"])
        self.assertEqual("fresh_goal_verified_after_requested_repair", summary["outcome"])
        self.assertEqual(("fail", "pass"), (summary["before_status"], summary["after_status"]))
        self.assertFalse(summary["physical_action_execution_verified"])
        verification = verify_recovery_receipt(self.engine._recovery_path, root=self.root)
        self.assertTrue(verification["valid"])
        self.assertEqual("completed", verification["scope"])
        records = sorted(self.engine._directory.glob("recovery-*.json"))
        self.assertEqual(2, len(records))
        first, last = [json.loads(path.read_text()) for path in records]
        self.assertEqual("awaiting_fresh_evidence", first["outcome"])
        self.assertEqual(first["receipt_hash"], last["previous_receipt_hash"])
        self.assertTrue(all(path.stat().st_mode & 0o777 == 0o600 for path in records))

    def test_wrong_repair_stays_failed_and_unknown_cannot_close_recovery(self):
        for options, expected in (({"wrong": True}, "still_failed"), ({"covered": True}, "needs_evidence")):
            self.engine = InteractiveEngine(self.root, client_factory=lambda: self.client)
            self.client = FakeClient(wrong=True)
            # Unique fixture bytes per session respect the restart reuse guard.
            offset = 10 if options.get("covered") else 0
            self.engine.inspect_pair(self.payload(1 + offset))
            self.engine.begin_repair()
            self.client = FakeClient(**options)
            state = self.engine.inspect_pair(self.payload(3 + offset))
            self.assertEqual(expected, state["recovery"]["outcome"])
            self.assertIsNone(state["verified_at_ms"])
            self.assertTrue(self.engine.verify_recovery()["valid"])
            self.assertFalse(state["recovery"]["physical_action_execution_verified"])

    def test_unknown_request_is_inspection_not_physical_repair(self):
        state = self.engine.begin_repair()
        self.assertEqual("inspection", state["recovery"]["kind"])
        self.assertEqual([], self.engine._recovery_record["plan_action_ids"])
        self.client = FakeClient()
        state = self.engine.inspect_pair(self.payload(1))
        self.assertEqual("fresh_goal_verified_after_inspection", state["recovery"]["outcome"])
        self.assertTrue(self.engine.verify_recovery()["valid"])

    def test_task_edit_cancels_pending_repair_without_relabeling_old_sources(self):
        self.requested_repair()
        old_hash = self.engine._recovery_record["contract_hash"]
        self.engine._compile_client_factory = lambda: FakeCompileClient()
        draft = self.engine.draft_task({"requirements": "Fork RIGHT and spoon LEFT", "objects": ["fork", "spoon"],
                                       "confirm_unique_targets": True})["task_draft"]
        state = self.engine.activate_task({"draft_id": draft["draft_id"], "draft_hash": draft["draft_hash"],
                                          "confirm_requirements": True, "confirm_unique_targets": True})
        self.assertEqual("cancelled_task_change", state["recovery"]["outcome"])
        self.assertEqual(old_hash, self.engine._recovery_record["contract_hash"])
        self.assertTrue(self.engine.verify_recovery()["valid"])

    def test_old_views_cannot_close_a_requested_repair(self):
        self.requested_repair()
        self.client = FakeClient()
        with self.assertRaises(ValidationError):
            self.engine.inspect_pair(self.payload(1))
        self.assertEqual([], self.client.ledger)
        self.assertEqual("awaiting_fresh_evidence", self.engine.snapshot()["recovery"]["outcome"])

    def test_media_mask_contract_and_receipt_tampering_fail_safe_replay(self):
        self.completed_repair()
        original = deepcopy(self.engine._recovery_record)
        media = self.root / original["after"]["frames"][0]["file"]
        saved = media.read_bytes()
        media.write_bytes(picture(77))
        self.assertFalse(self.engine.verify_recovery()["valid"])
        self.assertEqual("error", self.engine.snapshot()["recovery"]["outcome"])
        self.assertEqual("pass", self.engine.snapshot()["status"])
        media.write_bytes(saved)
        reference = original["after"]["frames"][0]["results"]["fork"]
        result_path = self.root / reference["file"]
        saved_result = result_path.read_bytes()
        result = json.loads(saved_result)
        result["objects"][0]["mask"]["raster_base64"] = "AAAA"
        result_path.write_text(json.dumps(result))
        self.assertFalse(verify_recovery_receipt(original, root=self.root)["valid"])
        # Even updating the outer file digest and receipt seal cannot make an
        # inconsistent decoded raster satisfy the executable checker.
        resealed = deepcopy(original)
        resealed["after"]["frames"][0]["results"]["fork"]["sha256"] = hashlib.sha256(result_path.read_bytes()).hexdigest()
        self.assertFalse(verify_recovery_receipt(seal_receipt(resealed), root=self.root)["valid"])
        result_path.write_bytes(saved_result)
        changed = deepcopy(original)
        changed["contract"]["protocol"]["checks"][0]["region_id"] = "right"
        self.assertFalse(verify_recovery_receipt(seal_receipt(changed), root=self.root)["valid"])
        changed = deepcopy(original)
        changed["after_status"] = "fail"
        self.assertFalse(verify_recovery_receipt(changed, root=self.root)["valid"])

    def test_rehashed_reused_sources_cannot_create_a_new_recovery_result(self):
        self.completed_repair(wrong=True)
        record = deepcopy(self.engine._recovery_record)
        record["after"]["frames"] = deepcopy(record["before"]["frames"])
        self.assertFalse(verify_recovery_receipt(seal_receipt(record), root=self.root)["valid"])

    def test_safe_export_omits_private_references_and_later_inspection_keeps_closed_receipt(self):
        self.completed_repair()
        summary = self.engine.recovery_receipt()
        serialized = json.dumps(summary)
        for private in (str(self.root), "artifacts/local", "inputs/local", "raster_base64", "private-provider-id", "capture_token", "authorization_reference"):
            self.assertNotIn(private, serialized)
        closed_hash = summary["receipt_hash"]
        self.client = FakeClient(wrong=True)
        state = self.engine.inspect_pair(self.payload(5))
        self.assertEqual("fail", state["status"])
        self.assertEqual(closed_hash, state["recovery"]["receipt_hash"])

    def test_missing_or_malformed_receipt_never_leaks_local_diagnostics(self):
        for value in ({}, {"receipt_hash": str(self.root)}, {"schema_version": "bad"}):
            result = verify_recovery_receipt(value, root=self.root)
            self.assertFalse(result["valid"])
            self.assertNotIn(str(self.root), json.dumps(result))
        self.assertFalse(verify_recovery_receipt(self.root / "artifacts/local/missing.json", root=self.root)["valid"])

    def test_changed_plan_duplicate_checks_and_invalid_revisions_are_rejected_even_when_resealed(self):
        self.requested_repair()
        original = deepcopy(self.engine._recovery_record)
        mutations = [lambda record: record["plan_action_ids"].__setitem__(0, "move-spoon-right-to-left"),
                     lambda record: record["plan_steps"].__setitem__(0, "Move the spoon instead."),
                     lambda record: record["before_checks"].append(deepcopy(record["before_checks"][0])),
                     lambda record: record["before_checks"][0].update(object_id="spoon"),
                     lambda record: record.update(requested_revision=True),
                     lambda record: record.update(before_revision=record["requested_revision"])]
        for mutation in mutations:
            record = deepcopy(original)
            mutation(record)
            self.assertFalse(verify_recovery_receipt(seal_receipt(record), root=self.root)["valid"])

    def test_parent_request_file_is_hashed_and_cannot_be_dropped_or_changed(self):
        self.completed_repair()
        original = deepcopy(self.engine._recovery_record)
        parent = self.root / original["previous_receipt"]["file"]
        saved = parent.read_bytes()
        parent.write_bytes(saved + b" ")
        self.assertFalse(self.engine.verify_recovery()["valid"])
        parent.write_bytes(saved)
        record = deepcopy(original)
        record.update(previous_receipt=None, previous_receipt_hash=None)
        self.assertFalse(verify_recovery_receipt(seal_receipt(record), root=self.root)["valid"])
        record = deepcopy(original)
        record["previous_receipt_hash"] = "0" * 64
        self.assertFalse(verify_recovery_receipt(seal_receipt(record), root=self.root)["valid"])

    def test_guided_receipt_replays_saved_locator_and_keeps_semantic_identity_limit(self):
        case = interactive_fixtures.InteractiveTests("test_guided_runs_one_task_independent_locator_plus_four_actual_sam_calls")
        case.setUp()
        self.addCleanup(case.tearDown)
        case.activate_guided()
        case.guided_client()
        case.engine.inspect_pair(case.guided_payload())
        case.engine.begin_repair()
        case.guided_client()
        state = case.engine.inspect_pair(case.guided_payload(offset=10))
        self.assertEqual("fresh_goal_verified_after_inspection", state["recovery"]["outcome"])
        self.assertTrue(case.engine.verify_recovery()["valid"])
        record = case.engine._recovery_record
        locator_path = case.root / record["after"]["guided"]["file"]
        locator_path.write_bytes(locator_path.read_bytes() + b" ")
        self.assertFalse(case.engine.verify_recovery()["valid"])

    def test_expired_failure_does_not_authorize_a_physical_repair(self):
        with patch("proofloop.interactive._now_ms", return_value=10000):
            self.engine.inspect_pair(self.payload(1))
        with patch("proofloop.interactive._now_ms", return_value=50000):
            before = self.engine.snapshot()
            state = self.engine.begin_repair()
        self.assertEqual("unknown", before["status"])
        self.assertIsNone(before["proposal"])
        self.assertEqual("inspection", state["recovery"]["kind"])
        self.assertTrue(self.engine.verify_recovery()["valid"])

    def test_receipt_storage_failure_does_not_rewrite_independent_goal_verdict(self):
        self.requested_repair()
        self.client = FakeClient()
        with patch("proofloop.interactive.save_receipt", side_effect=OSError("Authored disk failure")):
            state = self.engine.inspect_pair(self.payload(3))
        self.assertEqual("pass", state["status"])
        self.assertEqual("error", state["recovery"]["outcome"])
        self.assertFalse(state["recovery"]["physical_action_execution_verified"])
