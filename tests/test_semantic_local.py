"""Authored controls for crop-only adapter/replay; not semantic-quality scores."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from proofloop.core import ValidationError
from proofloop.meta import MetaAPIError
from proofloop.semantic_v2 import challenge_candidate_pair_v2
from proofloop import semantic_local as local
from tests import test_semantic as fixtures
from tests.test_semantic_v2 import ErrorClient


class CropClient(fixtures.Client):
    def spark_json(self, prompt, schema, **kwargs):
        result = super().spark_json(prompt, schema, **kwargs)
        for frame in result["parsed"]["frames"]:
            del frame["full_scene_match_count"]
        result["text"] = json.dumps(result["parsed"])
        result["response"]["output_text"] = result["text"]
        return result


class LocalSemanticTests(unittest.TestCase):
    setUp = fixtures.SemanticChallengeTests.setUp
    crop = fixtures.SemanticChallengeTests.crop

    def global_report(self, mutate=None):
        return challenge_candidate_pair_v2(fixtures.Client(mutate), self.prepared)

    def report(self, client=None):
        global_report = self.global_report()
        report = local.challenge_crop_confirmation(client or CropClient(), global_report)
        return global_report, report

    def entries(self):
        folder = self.root / "artifacts/local/source-global"
        folder.mkdir(parents=True)
        path = folder / "global.json"
        path.write_text(json.dumps(self.global_report()))
        return [{"case_id": "candidate", "global_report_file": str(path.relative_to(self.root))}]

    def test_only_two_crops_low_reasoning_without_global_images_or_count(self):
        client = CropClient()
        global_report, report = self.report(client)
        prompt, schema, options = client.calls[0]
        self.assertEqual(1, len(client.calls))
        self.assertEqual([Path(frame["crop_file"]) for frame in self.prepared["frames"]], options["image_paths"])
        self.assertEqual("low", options["reasoning_effort"])
        self.assertEqual(4096, options["max_output_tokens"])
        self.assertNotIn("full_scene_match_count", json.dumps(schema))
        self.assertNotIn("source_file", prompt)
        self.assertNotIn("original_locator", prompt)
        self.assertIn("candidate_box_in_crop", prompt)
        summary = local.public_confirmation(report, global_report)
        self.assertEqual("supported", summary["final_support"])
        self.assertNotIn("full_scene_match_count", json.dumps(summary))

    def test_candidate_attributes_cannot_transfer_from_larger_device(self):
        prompt = local.confirmation_prompt(self.prepared)
        self.assertIn("Never borrow visual attributes", prompt)
        self.assertIn("Context may contain other objects", prompt)
        for leaked_example in ("battery", "remote", "screwdriver", "expected_identity", "global_decision"):
            self.assertNotIn(leaked_example, prompt)

    def test_withheld_local_candidate_cannot_keep_global_support(self):
        def not_target(result, _):
            for frame in result["parsed"]["frames"]:
                frame["candidate_relation"] = "other"
                frame["required_attributes"][0]["support"] = "contradicted"
        global_report, report = self.report(CropClient(not_target))
        summary = local.public_confirmation(report, global_report)
        self.assertEqual("supported", summary["global_decision"])
        self.assertEqual("unknown", summary["final_support"])
        self.assertIn("candidate_other", summary["local_assessment"]["reasons"])

    def test_global_unknown_rejected_before_new_request(self):
        def unknown(result, _):
            result["parsed"]["frames"][0]["candidate_relation"] = "uncertain"
        client = CropClient()
        with self.assertRaises(ValidationError):
            local.challenge_crop_confirmation(client, self.global_report(unknown))
        self.assertEqual([], client.calls)

    def test_unknown_attribute_visibility_exclusion_or_text_withholds(self):
        mutations = (lambda f: f["required_attributes"][0].update(support="uncertain"),
                     lambda f: f["excluded_matches"][0].update(support="supported"),
                     lambda f: f.update(fully_visible=False), lambda f: f.update(image_text_used_as_identity_evidence=True))
        for mutate in mutations:
            def transform(result, _, mutate=mutate):
                mutate(result["parsed"]["frames"][0])
            global_report, report = self.report(CropClient(transform))
            with self.subTest(mutate=mutate):
                self.assertEqual("unknown", local.public_confirmation(report, global_report)["final_support"])

    def test_incomplete_response_retained_unscored_and_billed(self):
        global_report, report = self.report(ErrorClient())
        summary = local.public_confirmation(report, global_report)
        self.assertEqual("unscored", summary["status"])
        self.assertIsNone(summary["final_support"])
        self.assertIsNone(summary["local_assessment"])
        self.assertEqual(4096, summary["usage"]["output_tokens"])
        self.assertTrue(summary["failure"]["raw_response_retained"])
        self.assertIsNotNone(report["error"]["response"])

    def test_budget_nonfresh_client_rejected_before_call(self):
        for change in (lambda c: setattr(c, "max_requests", 2), lambda c: setattr(c, "retries", 1),
                       lambda c: c.ledger.append({"existing": True})):
            client = CropClient()
            change(client)
            with self.subTest(change=change), self.assertRaises(ValidationError):
                local.challenge_crop_confirmation(client, self.global_report())
            self.assertEqual([], client.calls)

    def test_wrong_model_or_attachment_order_is_retained_unscored(self):
        for change in (lambda r, _: r["sources"].reverse(), lambda r, _: r["response"].update(model="other")):
            global_report, report = self.report(CropClient(change))
            with self.subTest(change=change):
                self.assertEqual("unscored", report["status"])
                self.assertIsNotNone(report["challenger"])
                self.assertEqual("unscored", local.public_confirmation(report, global_report)["status"])

    def test_misordered_attribute_ids_retained_without_fabricated_decision(self):
        def reorder(result, _):
            result["parsed"]["frames"][0]["required_attributes"].reverse()
        global_report, report = self.report(CropClient(reorder))
        self.assertEqual("unscored", report["status"])
        self.assertIsNone(local.public_confirmation(report, global_report)["local_assessment"])

    def test_modified_decisions_bindings_and_usage_rejected(self):
        global_report, report = self.report()
        for change in (lambda r: r["evaluation"].update(decision="unknown"),
                       lambda r: r["binding"].update(global_report_sha256="0" * 64),
                       lambda r: r["usage"].update(output_tokens=0),
                       lambda r: r["config"].update(reasoning_effort="high")):
            changed = deepcopy(report)
            change(changed)
            with self.subTest(change=change), self.assertRaises(ValidationError):
                local.public_confirmation(changed, global_report)

    def test_edited_parsed_classification_cannot_replace_actual_response_text(self):
        global_report, report = self.report()
        report["challenger"]["parsed"]["frames"][0]["candidate_relation"] = "other"
        report["evaluation"] = local._evaluate(report["challenger"]["parsed"], self.prepared["contract"])
        with self.assertRaises(ValidationError):
            local.public_confirmation(report, global_report)

    def test_public_projection_contains_no_raw_paths_or_provider_body(self):
        for client in (CropClient(), ErrorClient()):
            global_report, report = self.report(client)
            encoded = json.dumps(local.public_confirmation(report, global_report))
            for secret in (str(self.root), "private-provider-id", "private-error-provider-id",
                           "should-never-be-public", "source_file", "crop_file"):
                self.assertNotIn(secret, encoded)

    def test_valid_assessment_cannot_be_relabelled_failure(self):
        global_report, report = self.report()
        report.update(status="unscored", evaluation=None, error={"exception_class": "ValidationError",
            "code": "local_validation_failed", "http_status": None, "response": deepcopy(report["challenger"]["response"])})
        with self.assertRaises(ValidationError):
            local.public_confirmation(report, global_report)

    def test_frozen_trial_run_and_replay_make_exactly_one_call(self):
        plan = local.freeze_confirmation(self.entries(), "artifacts/local/confirmation", root=self.root)
        clients = []
        def factory():
            client = CropClient()
            clients.append(client)
            return client
        summary = local.run_confirmation("artifacts/local/confirmation", factory, root=self.root)
        self.assertEqual(1, len(clients))
        self.assertEqual(1, summary["metrics"]["supported"])
        self.assertEqual(plan["freeze_hash"], summary["freeze_hash"])
        with patch.object(local, "challenge_crop_confirmation", side_effect=AssertionError("Replay cannot call API")):
            self.assertEqual(summary, local.public_confirmation_summary("artifacts/local/confirmation", root=self.root))
        with self.assertRaises(ValidationError):
            local.run_confirmation("artifacts/local/confirmation", factory, root=self.root)
        self.assertEqual(1, len(clients))

    def test_unattempted_trial_is_not_unknown(self):
        local.freeze_confirmation(self.entries(), "artifacts/local/confirmation", root=self.root)
        summary = local.public_confirmation_summary("artifacts/local/confirmation", root=self.root)
        self.assertIsNone(summary["cases"][0]["confirmation"])
        self.assertEqual(0, summary["metrics"]["attempted"])
        self.assertEqual(0, summary["metrics"]["withheld"])

    def test_changed_code_global_report_and_new_attempt_are_rejected(self):
        entries = self.entries()
        local.freeze_confirmation(entries, "artifacts/local/confirmation", root=self.root)
        with patch.object(local, "_code_hashes", return_value={}):
            with self.assertRaises(ValidationError):
                local.public_confirmation_summary("artifacts/local/confirmation", root=self.root)
        source = self.root / entries[0]["global_report_file"]
        source.write_text(source.read_text() + "\n")
        with self.assertRaises(ValidationError):
            local.public_confirmation_summary("artifacts/local/confirmation", root=self.root)


if __name__ == "__main__":
    unittest.main()
