"""Prospective completion infrastructure with authored HTTP/JPEG fixtures only."""
from copy import deepcopy
from io import BytesIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from proofloop import tracking_completion as repair
from proofloop import tracking_experiment as original
from proofloop.core import ValidationError
from proofloop.inference_journal import AttemptJournal
from proofloop.meta import MetaAPIError, SPARK_MODEL
from tests.test_guided import jpeg_header


def wire_response(*, completed=False):
    parsed = {"coordinate_system": "normalized_0_1000", "frames": [{"frame_index": index,
        "width": 100, "height": 80, "objects": [{"object_id": "target", "identity_ambiguous": False,
        "fully_visible": False, "boundary_uncertain": False, "candidates": []}]} for index in range(2)]}
    return {"model": SPARK_MODEL, "status": "completed" if completed else "incomplete",
        "reasoning": {"effort": "low" if completed else "high"}, "max_output_tokens": 4096,
        "output": [{"type": "message", "content": [{"type": "output_text", "text": json.dumps(parsed)}]}] if completed else [],
        "usage": {"input_tokens": 17, "output_tokens": 29 if completed else 4096}}


def http_response(value):
    return BytesIO(json.dumps(value).encode())


class TrackingCompletionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.original_dir = self.root / "artifacts/local/original"
        self.original_dir.mkdir(parents=True)
        self.output = self.root / "artifacts/local/repair"
        frames = []
        for index in range(14):
            path = self.original_dir / f"frame-{index}.jpg"
            path.write_bytes(jpeg_header(suffix=f"authored-{index}".encode()))
            frames.append({"file": str(path), "sha256": original._sha(path), "timestamp_ms": index * 200})
        self.parent_plan = {"freeze_hash": "a" * 64, "implementation_sha256": {"guided.py": "b" * 64},
            "clips": [{"clip_id": "screwdriver", "contract": {"target_id": "target", "concept": "light-handled screwdriver"},
                "frames": frames, "checkpoints": [{"frame_index": index, "expected": "MUST_NOT_ENTER_NEW_PROMPT"}
                    for index in (0, 5, 8, 13, 17, 22, 30, 37)]}]}
        self.parent_result = {"failed_operation_count": 1, "ground_truth": "MUST_NOT_ENTER_NEW_PROMPT"}
        self.original_frozen = self.original_dir / "freeze.json"
        original._write(self.original_frozen, self.parent_plan)
        journal = AttemptJournal(self.original_dir / "attempts.sqlite3", request_limit=3)
        client = original.RetainingClient("authored-not-real", journal=journal, operation_id="c" * 32,
            guard=lambda: None, max_requests=3, retries=0,
            opener=lambda *args, **kwargs: http_response(wire_response()))
        self.failed = original._pair(client, [frames[index]["file"] for index in (8, 13)], {"target": "light-handled screwdriver"})
        self.failed_file = self.original_dir / "screwdriver-pair-1.json"
        original._write(self.failed_file, self.failed)
        self.original_pair_bytes = self.failed_file.read_bytes()
        self.replay_override = patch.object(original, "replay", side_effect=lambda *args, **kwargs: deepcopy(self.parent_result))
        self.plan_override = patch.object(original, "verify_plan")
        self.replay_override.start()
        self.plan_override.start()
        self.addCleanup(self.replay_override.stop)
        self.addCleanup(self.plan_override.stop)

    def freeze(self):
        return repair.freeze_repair(self.root, self.original_frozen, self.output)

    def execute_repair(self, *, completed=True):
        calls = []
        def from_env(**kwargs):
            def opener(request, **options):
                calls.append(json.loads(request.data))
                value = wire_response(completed=completed)
                value["reasoning"] = {"effort": "low"}
                return http_response(value)
            return repair.CompletionClient("authored-key-not-real", opener=opener, **kwargs)
        with patch.object(repair.CompletionClient, "from_env", side_effect=from_env):
            result = repair.run_repair(self.root, self.output / "freeze.json")
        return result, calls

    def test_freeze_is_read_only_and_retains_original_failure_without_gt_or_api_dispatch(self):
        with patch.object(repair.CompletionClient, "from_env", side_effect=AssertionError("Freeze attempted inference")):
            plan = self.freeze()
        self.assertEqual("unscored", plan["original_failure"]["status"])
        self.assertEqual("incomplete_response", plan["original_failure"]["error_code"])
        self.assertEqual(4096, plan["original_failure"]["max_output_tokens"])
        self.assertEqual(4096, plan["original_failure"]["usage"]["output_tokens"])
        self.assertEqual("high", plan["original_failure"]["reasoning_effort_reported_by_provider"])
        self.assertEqual([8, 13], [frame["frame_index"] for frame in plan["source_frames"]])
        self.assertEqual({"target": "light-handled screwdriver"}, plan["concepts"])
        self.assertEqual(self.original_pair_bytes, self.failed_file.read_bytes())
        self.assertNotIn("MUST_NOT_ENTER_NEW_PROMPT", json.dumps(plan))
        self.assertFalse((self.output / "run-started.json").exists())
        repair.verify_repair(self.root, plan)

    def test_freeze_refuses_more_than_the_single_exposed_failed_operation(self):
        self.parent_result["failed_operation_count"] = 2
        with self.assertRaises(ValidationError):
            self.freeze()
        self.assertFalse(self.output.exists())

    def test_freeze_refuses_different_pair_frames_or_non_incomplete_original(self):
        changed = deepcopy(self.failed)
        changed["error"]["code"] = "network_error"
        self.failed_file.write_text(json.dumps(changed))
        with self.assertRaises(ValidationError):
            self.freeze()
        self.failed_file.write_bytes(self.original_pair_bytes)
        self.parent_plan["clips"][0]["checkpoints"][2]["frame_index"] = 7
        self.original_frozen.write_text(json.dumps(self.parent_plan))
        with self.assertRaises(ValidationError):
            self.freeze()

    def test_existing_output_directory_cannot_be_reused_for_a_new_freeze(self):
        self.freeze()
        with self.assertRaises(FileExistsError):
            self.freeze()

    def test_rehashed_config_concept_or_selection_tampering_cannot_change_repair(self):
        plan = self.freeze()
        for key, value in (("config", {**repair.CONFIG, "max_requests": 4}),
                           ("concepts", {"target": "another object"}), ("pair_index", 0),
                           ("original_failure", {**plan["original_failure"], "error_code": "other"}),
                           ("implementation_sha256", {"tracking_completion.py": "d" * 64})):
            changed = deepcopy(plan)
            changed[key] = value
            changed["freeze_hash"] = original._digest({name: item for name, item in changed.items() if name != "freeze_hash"})
            with self.subTest(key=key), self.assertRaises(ValidationError):
                repair.verify_repair(self.root, changed, replay_original=False)

    def test_changed_frame_or_original_failure_bytes_block_before_client_creation(self):
        plan = self.freeze()
        for path in (Path(plan["source_frames"][0]["file"]), self.failed_file, self.original_frozen):
            before = path.read_bytes()
            path.write_bytes(before + b"modified")
            with self.subTest(path=path.name), patch.object(repair.CompletionClient, "from_env") as factory:
                with self.assertRaises(ValidationError):
                    repair.run_repair(self.root, self.output / "freeze.json")
                factory.assert_not_called()
                self.assertFalse((self.output / "run-started.json").exists())
            path.write_bytes(before)

    def test_completed_probe_uses_one_low_4096_request_and_preserves_original_failure(self):
        self.freeze()
        result, requests = self.execute_repair()
        self.assertEqual(1, len(requests))
        self.assertEqual({"effort": "low"}, requests[0]["reasoning"])
        self.assertEqual(4096, requests[0]["max_output_tokens"])
        self.assertEqual(2, sum(item["type"] == "input_image" for item in requests[0]["input"][0]["content"]))
        self.assertNotIn("MUST_NOT_ENTER_NEW_PROMPT", json.dumps(requests))
        self.assertEqual("completed", result["repair_status"])
        self.assertEqual(1, result["usage"]["request_attempts"])
        self.assertEqual(29, result["usage"]["output_tokens"])
        self.assertEqual(1, result["accounting"]["total_reserved_attempts"])
        self.assertEqual([8, 13], [row["frame_index"] for row in result["classifications"]])
        self.assertTrue(all(row["locator_support"] == "unknown" for row in result["classifications"]))
        self.assertFalse(result["accuracy_scored"])
        self.assertFalse(result["changes_original_experiment"])
        self.assertEqual(self.original_pair_bytes, self.failed_file.read_bytes())
        self.assertEqual(result, repair.replay_repair(self.root, self.output / "freeze.json"))

    def test_still_incomplete_is_preserved_unscored_with_usage_without_retry(self):
        self.freeze()
        result, requests = self.execute_repair(completed=False)
        self.assertEqual(1, len(requests))
        self.assertEqual("unscored", result["repair_status"])
        self.assertEqual("incomplete_response", result["error_code"])
        self.assertEqual(4096, result["usage"]["output_tokens"])
        self.assertEqual([], result["classifications"])
        self.assertEqual(1, result["accounting"]["outcomes"]["rejected"])
        self.assertEqual(self.original_pair_bytes, self.failed_file.read_bytes())

    def test_exclusive_admission_marker_refuses_reexecution_before_new_client_or_http(self):
        self.freeze()
        self.execute_repair()
        with patch.object(repair.CompletionClient, "from_env") as factory:
            with self.assertRaises(FileExistsError):
                repair.run_repair(self.root, self.output / "freeze.json")
            factory.assert_not_called()

    def test_marker_or_private_result_binding_tampering_cannot_replay(self):
        self.freeze()
        self.execute_repair()
        marker_path = self.output / "run-started.json"
        marker = marker_path.read_bytes()
        marker_path.write_text(json.dumps({"freeze_hash": "d" * 64}))
        with self.assertRaises(ValidationError):
            repair.replay_repair(self.root, self.output / "freeze.json")
        marker_path.write_bytes(marker)
        result_path = self.output / "pair-result.json"
        record = original._read(result_path)
        record["repair_freeze_hash"] = "d" * 64
        result_path.write_text(json.dumps(record))
        with self.assertRaises(ValidationError):
            repair.replay_repair(self.root, self.output / "freeze.json")

    def test_extra_or_unsettled_durable_attempt_cannot_be_hidden_from_replay(self):
        self.freeze()
        self.execute_repair()
        journal = AttemptJournal(self.output / "attempts.sqlite3", request_limit=3)
        journal.reserve("e" * 32, "responses", {"model": SPARK_MODEL}, operation_limit=1)
        with self.assertRaises(ValidationError):
            repair.replay_repair(self.root, self.output / "freeze.json")

    def test_missing_persistent_journal_is_not_recreated_during_replay(self):
        self.freeze()
        self.execute_repair()
        journal = self.output / "attempts.sqlite3"
        journal.unlink()
        with self.assertRaises(ValidationError):
            repair.replay_repair(self.root, self.output / "freeze.json")
        self.assertFalse(journal.exists())

    def test_mro_preserves_failed_response_usage_and_three_attempt_persistent_budget(self):
        journal = AttemptJournal(self.root / "artifacts/local/client.sqlite3", request_limit=3)
        calls = []
        def opener(*args, **kwargs):
            calls.append(1)
            return http_response(wire_response())
        client = repair.CompletionClient("authored-key", journal=journal, operation_id="f" * 32,
            guard=lambda: None, max_requests=3, retries=0, opener=opener)
        for index in range(4):
            with self.assertRaises(MetaAPIError):
                client._request("POST", "responses", {"model": SPARK_MODEL})
        self.assertEqual(3, len(calls))
        self.assertEqual(3, client.usage_summary()["request_attempts"])
        self.assertEqual(3 * 4096, client.usage_summary()["output_tokens"])
        self.assertEqual(3, journal.summary()["total_reserved_attempts"])
        self.assertTrue(all(row["status"] == "error" for row in client.retained_responses))

    def test_invalid_or_changed_original_public_replay_is_refused(self):
        plan = self.freeze()
        self.parent_result["unexpected_change"] = True
        with self.assertRaises(ValidationError):
            repair.verify_repair(self.root, plan)


if __name__ == "__main__":
    unittest.main()
