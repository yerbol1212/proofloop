"""Offline authored manual-box transport controls; no model-quality claims."""
from copy import deepcopy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from proofloop import benchmark, concept_verified_experiment as parent, prospective
from proofloop.core import ValidationError
from proofloop.meta import MetaAPIError, SAM_MODEL
from tests import test_image_gripper_experiment as fixtures
from tests.test_prospective import Client as OriginalClient
from tools import probe_spatial_gripper as probe


class SpatialGripperProbeTests(unittest.TestCase):
    def setUp(self):
        fixtures.ImageGripperExperimentTests.setUp(self)
        _, original = parent._load(self.parent, self.root)
        rows = []
        for case in original["cases"]:
            source_case, = case["envelope"]["contract"]["cases"]
            for index, (row, frame) in enumerate(zip(case["frames"], source_case["frames"])):
                rows.append({"source_id": source_case["source_id"], "decoded_frame_index": len(rows),
                             "pair_frame_index": index, "file": frame["file"], "source_sha256": row["source_sha256"],
                             "width": row["width"], "height": row["height"],
                             "box": {"x1": 60, "y1": 10, "x2": 64, "y2": 14}, "review_note": "Authored fixture"})
        self.annotation = {"annotation_origin": probe.ORIGIN,
                           "coordinate_system": "original_source_pixels_inclusive_xyxy", "frames": rows}
        self.sign()

    def sign(self):
        self.annotation["annotation_sha256"] = benchmark.manifest_hash({k: v for k, v in self.annotation.items() if k != "annotation_sha256"})

    def freeze(self, **options):
        return probe.freeze(self.annotation, self.output, root=self.root, parent_dir=self.parent, **options)

    def run_trial(self, factory=fixtures.GripperClient):
        return probe.run(factory, self.output, root=self.root)

    def test_exact_six_original_sources_manual_boxes_and_only_spatial_tokens(self):
        plan = self.freeze()
        public = probe.public_plan(self.output, root=self.root)
        self.assertEqual(6, public["planned_requests"])
        self.assertIn("tools/probe_spatial_gripper.py", plan["implementation_sha256"])
        clients = []
        def factory(budget):
            self.assertEqual(1, budget)
            clients.append(fixtures.GripperClient(budget))
            return clients[-1]
        result = self.run_trial(factory)
        self.assertEqual(6, len(clients))
        self.assertEqual("completed", result["status"])
        self.assertEqual(6, result["usage"]["sam_requests"])
        self.assertEqual(0, result["usage"]["spark_requests"])
        self.assertIsNone(result["task_metrics"])
        for client in clients:
            _kind, path, prompt, options = client.calls[0]
            self.assertEqual("<0f>1<|box;x1=60;y1=10;x2=64;y2=14;w=100;h=80|>", prompt)
            self.assertNotIn("robot", prompt)
            self.assertTrue(options["decode_masks"])
        self.assertEqual(result, probe.summarize(self.output, root=self.root))
        with self.assertRaises(ValidationError): self.run_trial()
        for forbidden in ("source_file", "target_file", "PRIVATE_PROVIDER_TEXT", "raster_base64", "inputs/local", "artifacts/local"):
            self.assertNotIn(forbidden, json.dumps(result))

    def test_maximum_six_and_unique_source_boxes_are_enforced_before_admission(self):
        with self.assertRaises(ValidationError): self.freeze(max_requests=5)
        with self.assertRaises(ValidationError): self.freeze(max_requests=7)
        self.annotation["frames"].append(deepcopy(self.annotation["frames"][0]))
        self.sign()
        with self.assertRaises(ValidationError): self.freeze()
        self.annotation["frames"].pop()
        self.annotation["frames"][0]["box"]["x2"] = 100
        self.sign()
        with self.assertRaises(ValidationError): self.freeze()

    def test_plan_prompt_box_source_or_dependency_tamper_blocks_the_first_client(self):
        self.freeze()
        path = self.output / "plan.json"
        original_bytes = path.read_bytes()
        plan = json.loads(original_bytes)
        plan["entries"][0]["prompt"] += " robot gripper"
        benchmark._save(path, plan)
        with self.assertRaises(ValidationError): self.run_trial(lambda _: self.fail("No altered prompt admission"))
        path.write_bytes(original_bytes)
        with patch.object(probe, "_implementation", return_value={}):
            with self.assertRaises(ValidationError): self.run_trial(lambda _: self.fail("No changed-code admission"))
        source = Path(plan["entries"][0]["source_file"])
        source.write_bytes(b"altered pixels")
        with self.assertRaises(ValidationError): self.run_trial(lambda _: self.fail("No changed-source admission"))
        self.assertFalse((self.output / "run-started.json").exists())

    def test_http_stop_keeps_remaining_views_missing_and_private_error_body_bound(self):
        self.freeze()
        clients = []
        def factory(budget):
            client = fixtures.GripperClient(budget, sam_error=MetaAPIError("private", code="authored_http", status=429))
            clients.append(client)
            return client
        result = self.run_trial(factory)
        self.assertEqual(1, len(clients))
        self.assertEqual(429, result["admission_stop_code"])
        self.assertEqual(6, result["missing_or_failed_views"])
        self.assertTrue(result["usage"]["unreported_charge_possible"])
        self.assertTrue(all(view["diagnostic"] is None for view in result["views"]))

    def test_failed_completed_response_cannot_be_relabelled_as_network_loss(self):
        self.freeze()
        clients = []
        def factory(budget):
            client = fixtures.GripperClient(budget, invalid=not clients)
            clients.append(client)
            return client
        result = self.run_trial(factory)
        self.assertEqual(6, result["usage"]["request_attempts"])
        self.assertEqual(1, result["missing_or_failed_views"])
        (self.output / "summary.json").unlink()
        path = self.output / "run-state.json"
        state = json.loads(path.read_text())
        state["attempts"]["0"]["usage"] = parent.accounting._ledger_usage([
            {"model": SAM_MODEL, "endpoint": "responses", "attempt": 1, "status": "network_error"}])
        benchmark._save(path, state)
        with self.assertRaises(ValidationError): probe.summarize(self.output, root=self.root)

    def test_known_network_dispatch_cannot_erase_entire_ledger(self):
        self.freeze()
        clients = []
        def factory(budget):
            client = fixtures.GripperClient(budget, sam_error=MetaAPIError("private", code="network_error") if not clients else None)
            clients.append(client)
            return client
        self.run_trial(factory)
        (self.output / "summary.json").unlink()
        path = self.output / "run-state.json"
        state = json.loads(path.read_text())
        state["attempts"]["0"]["usage"] = parent.accounting._ledger_usage([])
        benchmark._save(path, state)
        with self.assertRaises(ValidationError): probe.summarize(self.output, root=self.root)

    def test_rehashed_manual_index_must_equal_original_parent_decoded_index(self):
        _, original, _ = prospective._load(self.pilot, self.root)
        contract = deepcopy(original["envelope"]["contract"])
        frame_index = 0
        for case in contract["cases"]:
            for frame in case["frames"]:
                frame["decoded_frame_index"] = frame_index
                frame_index += 1
        approved = benchmark.approve_manifest(contract, benchmark.manifest_hash(contract), confirm_unique_targets=True)
        indexed_pilot = self.root / "artifacts/local/indexed-pilot"
        prospective.freeze(approved, indexed_pilot, root=self.root, max_requests=12)
        def factory(budget):
            client = OriginalClient(budget)
            client.transform = fixtures.fixtures.locator_text
            return client
        prospective.run(factory, indexed_pilot, root=self.root)
        self.parent = self.root / "artifacts/local/indexed-parent"
        parent.freeze([indexed_pilot], self.parent, root=self.root)
        parent.run(fixtures.fixtures.Client, self.parent, root=self.root)
        self.annotation["frames"][0]["decoded_frame_index"] = 999
        self.sign()
        with self.assertRaises(ValidationError): self.freeze()
        self.annotation["frames"][0]["decoded_frame_index"] = 0
        self.sign()
        plan = self.freeze()
        self.assertEqual("parent_source_frame", plan["entries"][0]["decoded_frame_index_origin"])


if __name__ == "__main__":
    unittest.main()
