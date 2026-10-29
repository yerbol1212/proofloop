"""Offline all-view inventory, bounded transport and raw-evidence integrity tests."""
from copy import deepcopy
import json
from pathlib import Path
import struct
import unittest
from unittest.mock import patch

from proofloop import benchmark, concept_verified_experiment as parent
from proofloop.core import ValidationError
from proofloop.meta import MetaAPIError, SAM_MODEL, parse_sam_boxes
from tests import test_image_gripper_experiment as fixtures
from tests import test_concept_verified_experiment as concept_fixtures
from tests.test_anchor_padding_experiment import original
from tools import probe_context_crops as probe


class ContextCropProbeTests(unittest.TestCase):
    def setUp(self):
        fixtures.ImageGripperExperimentTests.setUp(self)
        self.output = self.root / "artifacts/local/crop-probe"
        def render(_ffmpeg, source, target, rectangle):
            width, height = rectangle["x2"] - rectangle["x1"] + 1, rectangle["y2"] - rectangle["y1"] + 1
            target.write_bytes(b"\x89PNG\r\n\x1a\n" + struct.pack(">I", 13) + b"IHDR" + struct.pack(">II", width, height)
                               + Path(source).read_bytes())
        patch("proofloop.crop_evidence._crop_png", side_effect=render).start()

    def freeze(self, **options):
        return probe.freeze(self.parent, self.output, root=self.root, **options)

    def run_trial(self, factory=concept_fixtures.Client):
        return probe.run(factory, self.output, root=self.root)

    def state(self):
        return json.loads((self.output / "run-state.json").read_text())

    def alter_state(self, mutation):
        (self.output / "summary.json").unlink()
        state = self.state()
        mutation(state)
        benchmark._save(self.output / "run-state.json", state)

    def test_freezes_all_original_pairs_views_and_immutable_preparations_before_requests(self):
        plan = self.freeze()
        self.assertEqual(3, len(plan["cases"]))
        self.assertEqual(6, len(plan["frames"]))
        self.assertEqual(6, plan["planned_requests"])
        self.assertEqual(18, plan["max_requests"])
        self.assertEqual(6, len(list((self.output / "frozen-frames").rglob("prepared.json"))))
        public = probe.public_plan(self.output, root=self.root)
        self.assertIn("tools/probe_context_crops.py", public["implementation_sha256"])
        self.assertIn("proofloop/crop_evidence.py", public["implementation_sha256"])
        result = probe.summarize(self.output, root=self.root)
        self.assertEqual("not_started", result["status"])
        self.assertEqual(6, result["proposal_feasibility"]["unscored_crop_views"])
        self.assertTrue(all(pair["proposal_pair_eligible"] is None for pair in result["pairs"]))
        self.assertFalse((self.output / "run-started.json").exists())

    def test_one_request_per_crop_same_concept_no_spark_no_task_acceptance(self):
        self.freeze()
        clients = []
        def factory(budget):
            self.assertEqual(1, budget)
            clients.append(concept_fixtures.Client(budget))
            return clients[-1]
        result = self.run_trial(factory)
        self.assertEqual("completed", result["status"])
        self.assertEqual(6, len(clients))
        self.assertEqual((6, 0), (result["usage"]["sam_requests"], result["usage"]["spark_requests"]))
        for client in clients:
            kind, path, prompt, options = client.calls[0]
            self.assertEqual("sam", kind)
            self.assertEqual("glass", prompt)
            self.assertTrue(options["decode_masks"])
            self.assertEqual("context.png", Path(path).name)
        self.assertIsNone(result["task_metrics"])
        self.assertFalse(result["placement_acceptance"])
        self.assertTrue(all(pair["proposal_pair_eligible"] for pair in result["pairs"]))
        self.assertTrue(all(not pair["placement_accepted"] for pair in result["pairs"]))
        self.assertEqual(result, probe.summarize(self.output, root=self.root))
        with self.assertRaises(ValidationError):
            self.run_trial(lambda _: self.fail("No replay inference"))
        for forbidden in ("source_file", "target_file", "PRIVATE_PROVIDER_TEXT", "raster_base64", "inputs/local", "artifacts/local"):
            self.assertNotIn(forbidden, json.dumps(result))

    def test_nine_pairs_eighteen_views_fifteen_crops_three_missing_stay_explicit(self):
        second = original(self.root, "second", transform=concept_fixtures.locator_text)
        third = original(self.root, "third", transform=concept_fixtures.locator_text)
        self.parent = self.root / "artifacts/local/nine-parent"
        parent.freeze([self.pilot, second, third], self.parent, root=self.root)
        clients = []
        def original_factory(budget):
            client = concept_fixtures.Client(budget)
            index = len(clients)
            if index in (1, 8, 15):
                actual = client.segment_image
                def empty(*args, **kwargs):
                    value = actual(*args, **kwargs)
                    value.update(text="", objects=[])
                    value["response"]["output_text"] = ""
                    return value
                client.segment_image = empty
            clients.append(client)
            return client
        parent.run(original_factory, self.parent, root=self.root)
        plan = self.freeze(max_requests=15)
        self.assertEqual((9, 18, 15), (len(plan["cases"]), len(plan["frames"]), plan["planned_requests"]))
        clients.clear()
        def factory(budget):
            clients.append(concept_fixtures.Client(budget))
            return clients[-1]
        result = self.run_trial(factory)
        self.assertEqual(15, len(clients))
        self.assertEqual("completed", result["status"])
        self.assertEqual((18, 9, 3, 15), tuple(result["proposal_feasibility"][key] for key in (
            "total_original_views", "total_original_pairs", "no_unique_original_proposal_views", "completed_crop_views")))
        missing = [view for view in result["views"] if view["status"] == "not_requested_no_unique_original_proposal"]
        self.assertEqual([1, 8, 15], [view["view_index"] for view in missing])
        self.assertTrue(all(view["proposal_eligible"] is False for view in missing))
        self.assertEqual(9, len(result["pairs"]))
        self.assertEqual(3, sum(pair["proposal_pair_eligible"] is False for pair in result["pairs"]))

    def test_insufficient_or_excessive_budget_is_rejected_before_any_client(self):
        with self.assertRaises(ValidationError):
            self.freeze(max_requests=5)
        self.output = self.root / "artifacts/local/excess-budget"
        with self.assertRaises(ValidationError):
            self.freeze(max_requests=19)

    def test_parent_inventory_summary_raw_source_and_code_change_prevent_first_dispatch(self):
        plan = self.freeze()
        summary = self.parent / "summary.json"
        before = summary.read_bytes()
        summary.write_bytes(before + b" ")
        with self.assertRaises(ValidationError):
            self.run_trial(lambda _: self.fail("No changed parent admission"))
        summary.write_bytes(before)
        with patch.object(probe, "_implementation", return_value={}):
            with self.assertRaises(ValidationError):
                self.run_trial(lambda _: self.fail("No changed code admission"))
        prepared_file = probe._prepared_file(self.output, 0)
        prepared_file.write_bytes(prepared_file.read_bytes() + b" ")
        with self.assertRaises(ValidationError):
            self.run_trial(lambda _: self.fail("No changed preparation admission"))
        self.assertFalse((self.output / "run-started.json").exists())

    def test_missing_preparation_cannot_drop_a_difficult_view(self):
        self.freeze()
        probe._prepared_file(self.output, 0).unlink()
        with self.assertRaises(ValidationError):
            self.run_trial(lambda _: self.fail("No dropped-view admission"))

    def test_low_score_proposals_withheld_without_becoming_task_unknown_predictions(self):
        self.freeze()
        result = self.run_trial(lambda budget: concept_fixtures.Client(budget, score=.79))
        self.assertEqual(6, result["proposal_feasibility"]["completed_crop_views"])
        self.assertEqual(0, result["proposal_feasibility"]["eligible_proposal_pairs"])
        self.assertTrue(all(view["proposal_eligible"] is False for view in result["views"]))
        self.assertTrue(all("crop_sam_score_below_0_8" in view["withheld_reasons"] for view in result["views"]))
        self.assertIsNone(result["task_metrics"])
        self.assertNotIn("prediction", json.dumps(result))

    def test_multiple_crop_candidates_cannot_choose_the_highest_score(self):
        self.freeze()
        def factory(budget):
            client = concept_fixtures.Client(budget)
            actual = client.segment_image
            def multiple(*args, **kwargs):
                value = actual(*args, **kwargs)
                text = value["text"] + "," + value["text"][4:].replace("1<|box;", "2<|box;", 1)
                value.update(text=text, objects=concept_fixtures.decoded(parse_sam_boxes(text)))
                value["response"]["output_text"] = text
                return value
            client.segment_image = multiple
            return client
        result = self.run_trial(factory)
        self.assertEqual(0, result["proposal_feasibility"]["eligible_proposal_pairs"])
        self.assertTrue(all(view["object_count"] == 2 for view in result["views"]))

    def test_all_auth_billing_rate_statuses_stop_and_keep_remaining_views_unscored(self):
        for code in parent.STOP_HTTP:
            self.output = self.root / "artifacts/local" / ("stop-" + str(code))
            self.freeze()
            clients = []
            def factory(budget):
                client = concept_fixtures.Client(budget, sam_error=MetaAPIError("private", code="authored_http", status=code))
                clients.append(client)
                return client
            result = self.run_trial(factory)
            self.assertEqual(1, len(clients))
            self.assertEqual(code, result["admission_stop_code"])
            self.assertEqual(6, result["proposal_feasibility"]["unscored_crop_views"])
            self.assertTrue(all(view["proposal_eligible"] is None for view in result["views"]))

    def test_wrong_budget_retries_and_dirty_clients_do_not_dispatch(self):
        self.freeze()
        clients = []
        def factory(budget):
            client = concept_fixtures.Client(budget)
            client.retries = 1
            clients.append(client)
            return client
        result = self.run_trial(factory)
        self.assertTrue(all(not client.calls for client in clients))
        self.assertEqual(0, result["usage"]["request_attempts"])
        self.assertEqual(6, result["proposal_feasibility"]["unscored_crop_views"])

    def test_failed_successful_response_cannot_become_unbilled_network_loss(self):
        self.freeze()
        clients = []
        def factory(budget):
            client = concept_fixtures.Client(budget)
            if not clients:
                actual = client.segment_image
                def invalid(*args, **kwargs):
                    value = actual(*args, **kwargs)
                    value["objects"][0]["confidence"] = .99
                    return value
                client.segment_image = invalid
            clients.append(client)
            return client
        result = self.run_trial(factory)
        self.assertEqual(6, result["usage"]["sam_requests"])
        self.assertEqual(1, result["proposal_feasibility"]["unscored_crop_views"])
        self.alter_state(lambda state: state["attempts"]["0"].update(usage=parent.accounting._ledger_usage([
            {"model": SAM_MODEL, "endpoint": "responses", "attempt": 1, "status": "network_error"}])))
        with self.assertRaises(ValidationError):
            probe.summarize(self.output, root=self.root)

    def test_network_dispatch_cannot_erase_ledger_and_claim_known_zero(self):
        self.freeze()
        clients = []
        def factory(budget):
            client = concept_fixtures.Client(budget, sam_error=MetaAPIError("private", code="network_error") if not clients else None)
            clients.append(client)
            return client
        result = self.run_trial(factory)
        self.assertTrue(result["usage"]["unreported_charge_possible"])
        self.alter_state(lambda state: state["attempts"]["0"].update(usage=parent.accounting._ledger_usage([])))
        with self.assertRaises(ValidationError):
            probe.summarize(self.output, root=self.root)

    def test_projection_and_raw_response_must_match_exact_saved_digests(self):
        self.freeze()
        self.run_trial()
        (self.output / "summary.json").unlink()
        path = probe._projection_path(self.output, 0)
        value = json.loads(path.read_text())
        value["derived_full_source_geometry"][0]["confidence"] = .99
        benchmark._save(path, value)
        state = self.state()
        state["attempts"]["0"]["projection_sha256"] = benchmark._file_hash(path)
        benchmark._save(self.output / "run-state.json", state)
        with self.assertRaises(ValidationError):
            probe.summarize(self.output, root=self.root)

    def test_unplanned_response_and_false_completed_state_rejected(self):
        self.freeze()
        self.run_trial()
        self.alter_state(lambda state: state["attempts"].pop("0"))
        with self.assertRaises(ValidationError):
            probe.summarize(self.output, root=self.root)


if __name__ == "__main__":
    unittest.main()
