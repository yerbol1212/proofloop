"""Offline prompt commissioning contract tests; no model quality or API claims."""
import hashlib
import io
import json
from contextlib import redirect_stderr
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from proofloop.concepts import main, run_commissioning, validate_candidates, validate_concept_mapping
from proofloop.core import ValidationError
from tests.test_interactive import FakeClient, picture


class ConceptTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.local = self.root / "artifacts/local"
        self.images = [self.root / "first.png", self.root / "second.png"]
        for index, path in enumerate(self.images):
            path.write_bytes(picture(index))
        self.client = FakeClient()
        self.output = self.local / "commissioning"

    def tearDown(self):
        self.temp.cleanup()

    def run_probe(self, candidates=None, **kwargs):
        return run_commissioning(self.client, self.images, candidates or ["cup", "glass"], self.output,
                                 input_origin="synthetic", local_root=self.local, **kwargs)

    def test_mapping_is_explicit_complete_and_default_is_only_text_normalization(self):
        self.assertEqual({"coffee-spoon": "coffee spoon", "glass_cup": "glass cup"},
                         validate_concept_mapping(["coffee-spoon", "glass_cup"]))
        mapping = {"glass": "drinking glass", "fork": "dining fork"}
        self.assertEqual(mapping, validate_concept_mapping(["glass", "fork"], mapping))
        for value in ({}, {"glass": "cup"}, {"glass": "cup", "fork": ""},
                      {"glass": "cup", "fork": "fork", "extra": "spoon"}):
            with self.assertRaises(ValidationError):
                validate_concept_mapping(["glass", "fork"], value)

    def test_commissioning_is_not_accuracy_or_automatic_prompt_selection(self):
        report = self.run_probe(inference_origin="authored_mock_no_api_calls")
        self.assertEqual(4, len(self.client.ledger))
        self.assertEqual(["cup", "cup", "glass", "glass"], [row["concept"] for row in self.client.ledger])
        self.assertTrue(all(row["decode_masks"] for row in self.client.ledger))
        self.assertEqual("authored_mock_no_api_calls", report["inference_origin"])
        self.assertTrue(report["commissioning_only"])
        self.assertFalse(report["accuracy_established"])
        self.assertFalse(report["held_out_benchmark"])
        self.assertTrue(report["operator_review_required"])
        self.assertIsNone(report["selected_concept"])
        self.assertEqual(4, report["completed_requests"])
        self.assertEqual([1, 1], [value["detection_count"] for value in report["candidates"][0]["views"]])
        self.assertEqual([hashlib.sha256(path.read_bytes()).hexdigest() for path in self.images],
                         [value["source_sha256"] for value in report["candidates"][0]["views"]])
        serialized = json.dumps(report)
        self.assertNotIn("private-provider-id", serialized)
        self.assertNotIn("raster_base64", serialized)
        self.assertNotIn(str(self.root), serialized)
        for path in self.output.iterdir():
            self.assertEqual(0o600, path.stat().st_mode & 0o777)

    def test_zero_detections_are_reported_without_absence_or_winner(self):
        self.client.covered = True
        report = self.run_probe()
        self.assertEqual([0, 0], [value["detection_count"] for value in report["candidates"][0]["views"]])
        self.assertEqual("commissioning_complete_requires_operator_review", report["overall"])
        self.assertIsNone(report["selected_concept"])
        self.assertFalse(report["accuracy_established"])

    def test_invalid_candidates_and_budgets_stop_before_network(self):
        for values in ([], ["x"] * 5, ["cup", "CUP"], [""], ["x" * 101], ["cup\n"]):
            with self.subTest(values=values), self.assertRaises(ValidationError):
                validate_candidates(values)
        for field, value in (("max_requests", 8), ("retries", 1), ("ledger", [{"used": True}])):
            self.client = FakeClient()
            setattr(self.client, field, value)
            with self.assertRaises(ValidationError):
                self.run_probe()
        self.assertFalse(self.output.exists())

    def test_two_distinct_images_and_private_new_output_are_mandatory(self):
        self.images[1].write_bytes(self.images[0].read_bytes())
        with self.assertRaises(ValidationError):
            self.run_probe()
        self.assertEqual([], self.client.ledger)
        self.images[1].write_bytes(picture(9))
        self.output = self.root / "public-report"
        with self.assertRaises(ValidationError):
            self.run_probe()
        self.assertEqual([], self.client.ledger)
        self.output = self.local / "existing"
        self.output.mkdir(parents=True)
        with self.assertRaises(ValidationError):
            self.run_probe()

    def test_first_api_error_stops_without_retry_and_keeps_response_private(self):
        self.client.fail_on = 2
        report = self.run_probe()
        self.assertEqual(2, len(self.client.ledger))
        self.assertEqual(1, report["completed_requests"])
        self.assertEqual("commissioning_incomplete", report["overall"])
        self.assertEqual("MetaAPIError", report["failures"][0]["error_type"])
        self.assertNotIn("hidden-body", json.dumps(report))
        self.assertEqual({"private": "hidden-body"}, json.loads((self.output / "failed-response.json").read_text()))

    def test_wrong_source_model_or_unvalidated_mask_cannot_count_as_completed(self):
        for field in ("source", "model", "mask"):
            def corrupt(value):
                if field == "source":
                    value["source"]["sha256"] = "0" * 64
                elif field == "model":
                    value["response"]["model"] = "unapproved-model"
                else:
                    value["objects"][0]["mask"]["raster_base64"] = "invalid"
                return value
            self.output = self.local / field
            self.client = FakeClient(transform=corrupt)
            report = self.run_probe()
            self.assertEqual(1, len(self.client.ledger))
            self.assertEqual(0, report["completed_requests"])
            self.assertEqual("commissioning_incomplete", report["overall"])

    def test_source_changed_between_calls_stops_and_is_not_silently_rebound(self):
        self.client.callback = lambda number: self.images[0].write_bytes(picture(99)) if number == 1 else None
        report = self.run_probe()
        self.assertEqual(1, len(self.client.ledger))
        self.assertEqual(0, report["completed_requests"])
        self.assertIn("SourceChangedAfterRun", [value["error_type"] for value in report["failures"]])

    def test_cli_rejects_oversized_explicit_budget_before_loading_credentials(self):
        with patch("proofloop.concepts.load_env_file") as env, patch("proofloop.concepts.MetaClient.from_env") as factory, redirect_stderr(io.StringIO()):
            code = main(["--images", str(self.images[0]), str(self.images[1]), "--candidate", "cup",
                         "--max-requests", "8", "--input-origin", "synthetic"])
        self.assertEqual(2, code)
        env.assert_not_called()
        factory.assert_not_called()


if __name__ == "__main__":
    unittest.main()
