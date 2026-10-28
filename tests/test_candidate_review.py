"""Candidate provenance tests on authored fixtures, not model performance."""
from copy import deepcopy
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from proofloop.candidate_review import prepare_authored_candidate_pair, verify_authored_prepared
from proofloop.core import ValidationError
from proofloop.semantic import _hash
from tests.test_semantic import contract, jpeg


class CandidateReviewTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.paths = [self.root / f"source-{i}.jpg" for i in range(2)]
        for i, path in enumerate(self.paths):
            path.write_bytes(jpeg(suffix=str(i).encode()))
        self.boxes = [{"x1": 10, "y1": 10, "x2": 40, "y2": 40}] * 2

    def prepare(self):
        def crop(_program, _source, target, _box):
            target.write_bytes(jpeg(suffix=b"authored-crop"))
        with patch("proofloop.candidate_review._crop", side_effect=crop), \
                patch("proofloop.semantic._crop", side_effect=crop), \
                patch("proofloop.candidate_review._extractor_path", return_value="authored-fixture-extractor"), \
                patch("proofloop.semantic._extractor_path", return_value="authored-fixture-extractor"), \
                patch("proofloop.candidate_review._extractor_version", return_value="authored-fixture"), \
                patch("proofloop.semantic._extractor_version", return_value="authored-fixture"):
            return prepare_authored_candidate_pair(self.paths, self.boxes, contract(), self.root / "prepared",
                                                    review_note="Authored shape fixture; no model locator.")

    def test_authored_candidates_do_not_fabricate_locator_flags_or_model_results(self):
        value = self.prepare()
        self.assertEqual("authored_candidate", value["anchor_origin"])
        self.assertNotIn("guided_locator_hash", value)
        self.assertTrue(all("original_locator" not in frame and "original_candidate_count" not in frame
                            for frame in value["frames"]))
        self.assertTrue((self.root / "prepared/prepared.json").is_file())

    def test_bad_geometry_fails_before_creating_output_or_extracting(self):
        for field, bad in (("x1", True), ("x2", 100), ("y2", 10)):
            boxes = deepcopy(self.boxes)
            boxes[0][field] = bad
            with patch("proofloop.candidate_review._crop") as crop, self.assertRaises(ValidationError):
                prepare_authored_candidate_pair(self.paths, boxes, contract(), self.root / "bad", review_note="Review")
            crop.assert_not_called()
            self.assertFalse((self.root / "bad").exists())

    def test_forged_locator_or_review_cannot_be_added_even_with_recomputed_digest(self):
        original = self.prepare()
        mutations = [lambda value: value["frames"][0].update(original_locator={"identity_ambiguous": False}),
                     lambda value: value.update(guided_locator_hash="0" * 64),
                     lambda value: value["author_review"].update(origin="human_review"),
                     lambda value: value["author_review"].update(source_views_and_crops_reviewed=False)]
        for change in mutations:
            value = deepcopy(original)
            change(value)
            value["prepared_hash"] = _hash({key: item for key, item in value.items() if key != "prepared_hash"})
            with self.assertRaises(ValidationError):
                verify_authored_prepared(value)

    def test_changed_source_or_crop_bytes_cannot_replay(self):
        value = self.prepare()
        for field in ("source_file", "crop_file"):
            file = Path(value["frames"][0][field])
            original = file.read_bytes()
            file.write_bytes(original + b"changed")
            with patch("proofloop.semantic._extractor_version", return_value="authored-fixture"), \
                    self.assertRaises(ValidationError):
                verify_authored_prepared(value)
            file.write_bytes(original)
