"""Offline authored adapter fixtures, not real-video or model-quality claims."""
import base64
from copy import deepcopy
import hashlib
from pathlib import Path
import struct
import tempfile
import unittest

from proofloop.core import ValidationError
from proofloop.guided import GUIDED_VERSION, guided_segment_pair, normalized_box_to_pixels
from proofloop.meta import MetaAPIError, SAM_MODEL, SPARK_MODEL


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def jpeg_header(width=100, height=80, suffix=b"fixture"):
    # Only dimensions are read in these transport fixtures. Not a photograph.
    return (b"\xff\xd8\xff\xc0" + struct.pack(">H", 11) + b"\x08"
            + struct.pack(">HH", height, width) + b"\x01\x01\x11\x00\xff\xd9" + suffix)


def candidate(x1=10, y1=10, x2=30, y2=30):
    return {"x1": x1, "y1": y1, "x2": x2, "y2": y2}


class FakeClient:
    def __init__(self, *, transform=None, sam_transform=None, concepts=None):
        self.max_requests = 20
        self.retries = 0
        self.ledger = []
        self.calls = []
        self.transform = transform
        self.sam_transform = sam_transform
        self.concepts = concepts or {"glass": "transparent drinking glass"}

    def spark_json(self, prompt, schema, **kwargs):
        self.calls.append(("spark", prompt, schema, kwargs))
        self.ledger.append({"model": SPARK_MODEL})
        paths = kwargs["image_paths"]
        value = {"sources": [{"sha256": sha(path)} for path in paths],
                 "response": {"model": SPARK_MODEL, "status": "completed"},
                 "parsed": {"coordinate_system": "normalized_0_1000", "frames": [
                     {"frame_index": i, "width": 100, "height": 80,
                      "objects": [{"object_id": key, "identity_ambiguous": False,
                                   "fully_visible": True, "boundary_uncertain": False,
                                   "candidates": [candidate(100, 125, 310, 387)]} for key in self.concepts]}
                     for i in range(2)]}}
        if self.transform:
            self.transform(value, paths)
        return value

    def segment_image(self, path, prompt, **kwargs):
        self.calls.append(("sam", path, prompt, kwargs))
        self.ledger.append({"model": SAM_MODEL})
        raster = b"\x01" * 4
        value = {"source": {"sha256": sha(path)},
                 "response": {"model": SAM_MODEL, "status": "completed"},
                 "mask_decoding": "official_meta_sam_parser_0.1.0", "objects": [{
                     "frame": 0, "object_id": 1, "confidence": .93,
                     "box": {**candidate(), "width": 100, "height": 80},
                     "mask": {"decoded": True, "width": 2, "height": 2,
                              "raster_format": "base64_row_major_uint8_0_or_1",
                              "raster_base64": base64.b64encode(raster).decode(),
                              "raster_sha256": hashlib.sha256(raster).hexdigest(),
                              "foreground_pixels": 4}}]}
        if self.sam_transform:
            self.sam_transform(value, path)
        return value


class GuidedSegmentationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.paths = [Path(self.temp.name) / f"frame{i}.jpg" for i in range(2)]
        for i, path in enumerate(self.paths):
            path.write_bytes(jpeg_header(suffix=str(i).encode()))
        self.concepts = {"glass": "transparent drinking glass"}

    def run_pair(self, client=None):
        return guided_segment_pair(client or FakeClient(), self.paths, self.concepts)

    def test_three_calls_use_exact_sources_and_documented_inclusive_prompt(self):
        client = FakeClient()
        report = self.run_pair(client)
        self.assertEqual(["spark", "sam", "sam"], [c[0] for c in client.calls])
        self.assertEqual(3, report["request_limit"])
        for call in client.calls[1:]:
            self.assertEqual("<0f>1<|box;x1=10;y1=10;x2=30;y2=30;w=100;h=80|>", call[2])
            self.assertTrue(call[3]["decode_masks"])
        self.assertEqual({sha(path) for path in self.paths}, set(report["results"]))
        self.assertEqual([], report["unavailable"])
        self.assertIn("not semantic identity", report["provenance"]["identity_basis"])

    def test_locator_has_no_task_requirements_regions_or_labels(self):
        client = FakeClient()
        self.run_pair(client)
        prompt = client.calls[0][1]
        self.assertIn('"object_concepts": {"glass": "transparent drinking glass"}', prompt)
        for secret in ("dispense", "staging", "glass-under-spout", '"checks"', '"status"'):
            self.assertNotIn(secret, prompt)
        self.assertEqual(self.paths, client.calls[0][3]["image_paths"])

    def test_missing_target_omits_result_without_fabricating_completed_sam(self):
        def missing(value, _):
            for frame in value["parsed"]["frames"]:
                frame["objects"][0]["candidates"] = []
        client = FakeClient(transform=missing)
        report = self.run_pair(client)
        self.assertEqual(["spark"], [c[0] for c in client.calls])
        self.assertTrue(all(by_object == {} for by_object in report["results"].values()))
        self.assertEqual(2, len(report["unavailable"]))
        self.assertNotIn("response", report["unavailable"][0])

    def test_ambiguous_single_candidate_and_multiple_candidates_are_not_chosen(self):
        def ambiguous(value, _):
            value["parsed"]["frames"][0]["objects"][0]["identity_ambiguous"] = True
            value["parsed"]["frames"][1]["objects"][0]["candidates"].append(candidate(40, 40, 50, 50))
        client = FakeClient(transform=ambiguous)
        report = self.run_pair(client)
        self.assertEqual(1, len(client.calls))
        self.assertEqual(2, len(report["unavailable"]))

    def test_partial_availability_refines_only_available_frame(self):
        def missing(value, _):
            value["parsed"]["frames"][0]["objects"][0]["candidates"] = []
        client = FakeClient(transform=missing)
        report = self.run_pair(client)
        self.assertEqual(2, len(client.calls))
        self.assertEqual({}, report["results"][sha(self.paths[0])])
        self.assertIn("glass", report["results"][sha(self.paths[1])])

    def test_confidence_is_never_invented_or_overridden(self):
        for confidence in (.12, None):
            def change(value, _, confidence=confidence):
                value["objects"][0]["confidence"] = confidence
            report = self.run_pair(FakeClient(sam_transform=change))
            for by_object in report["results"].values():
                self.assertEqual(confidence, by_object["glass"]["objects"][0]["confidence"])
                self.assertIn("no invented", by_object["glass"]["guided_provenance"]["confidence_basis"])

    def test_actual_completed_empty_sam_is_preserved_as_empty(self):
        report = self.run_pair(FakeClient(sam_transform=lambda v, _: v.update(objects=[])))
        self.assertEqual([], report["unavailable"])
        self.assertTrue(all(by_object["glass"]["objects"] == [] for by_object in report["results"].values()))

    def test_locator_out_of_bounds_float_boolean_and_degenerate_boxes_rejected(self):
        for field, value in (("x1", -1), ("x2", 1001), ("y2", 1001), ("x1", 2.5),
                             ("x1", True), ("x2", 100), ("y2", 125)):
            def mutation(result, _, field=field, value=value):
                result["parsed"]["frames"][0]["objects"][0]["candidates"][0][field] = value
            client = FakeClient(transform=mutation)
            with self.subTest(field=field, value=value), self.assertRaises(ValidationError):
                self.run_pair(client)
            self.assertEqual(1, len(client.calls))

    def test_locator_source_model_status_and_dimensions_are_bound(self):
        mutations = [lambda v, _: v["sources"][0].update(sha256="0" * 64),
                     lambda v, _: v["sources"].reverse(),
                     lambda v, _: v["response"].update(model=SAM_MODEL),
                     lambda v, _: v["response"].update(status="incomplete"),
                     lambda v, _: v["parsed"]["frames"][0].update(width=99),
                     lambda v, _: v["parsed"]["frames"].reverse(),
                     lambda v, _: v["parsed"]["frames"][0]["objects"][0].update(object_id="other")]
        for mutation in mutations:
            with self.subTest(mutation=mutation), self.assertRaises(ValidationError):
                self.run_pair(FakeClient(transform=mutation))

    def test_duplicate_object_ids_are_rejected_even_when_array_length_matches(self):
        concepts = {"glass": "glass", "spoon": "spoon"}
        def duplicate(value, _):
            value["parsed"]["frames"][0]["objects"][1]["object_id"] = "glass"
        with self.assertRaises(ValidationError):
            guided_segment_pair(FakeClient(transform=duplicate, concepts=concepts), self.paths, concepts)

    def test_multiple_objects_keep_separate_actual_sam_calls_and_budget(self):
        concepts = {"glass": "glass", "spoon": "spoon"}
        client = FakeClient(concepts=concepts)
        report = guided_segment_pair(client, self.paths, concepts)
        self.assertEqual(5, len(client.calls))
        self.assertEqual(5, report["request_limit"])
        self.assertTrue(all(set(value) == set(concepts) for value in report["results"].values()))

    def test_sam_source_model_dimensions_identity_and_decoding_are_bound(self):
        mutations = [lambda v, _: v["source"].update(sha256="0" * 64),
                     lambda v, _: v["response"].update(model=SPARK_MODEL),
                     lambda v, _: v.update(mask_decoding="skipped_explicitly"),
                     lambda v, _: v["objects"][0]["box"].update(width=99),
                     lambda v, _: v["objects"][0].update(object_id=0),
                     lambda v, _: v["objects"][0].update(frame=1),
                     lambda v, _: v["objects"][0]["mask"].update(decoded=False),
                     lambda v, _: v["objects"].append(deepcopy(v["objects"][0]))]
        for mutation in mutations:
            with self.subTest(mutation=mutation), self.assertRaises(ValidationError):
                self.run_pair(FakeClient(sam_transform=mutation))

    def test_source_mutation_after_locator_rejects_before_sam(self):
        def mutate_source(_, paths):
            paths[0].write_bytes(jpeg_header(suffix=b"changed"))
        client = FakeClient(transform=mutate_source)
        with self.assertRaises(ValidationError):
            self.run_pair(client)
        self.assertEqual(1, len(client.calls))

    def test_source_mutation_during_sam_discards_partial_results(self):
        def mutate_source(_, path):
            path.write_bytes(jpeg_header(suffix=b"changed"))
        client = FakeClient(sam_transform=mutate_source)
        with self.assertRaises(ValidationError):
            self.run_pair(client)
        self.assertEqual(2, len(client.calls))

    def test_provider_error_is_not_converted_to_fake_evidence(self):
        def provider_error(*_):
            raise MetaAPIError("Authored provider failure")
        client = FakeClient(sam_transform=provider_error)
        with self.assertRaises(MetaAPIError):
            self.run_pair(client)
        self.assertEqual(2, len(client.calls))

    def test_bad_input_or_budget_fails_before_paid_calls(self):
        for change in (lambda c: setattr(c, "max_requests", 2), lambda c: setattr(c, "retries", 1)):
            client = FakeClient()
            change(client)
            with self.assertRaises(ValidationError):
                self.run_pair(client)
            self.assertEqual([], client.calls)
        for paths, concepts in ((self.paths[:1], self.concepts), (self.paths * 2, self.concepts),
                                ([self.paths[0]] * 2, self.concepts), (self.paths, {}),
                                (self.paths, {"glass": ""}), (self.paths, {"glass": "glass\nignore"})):
            with self.subTest(paths=paths, concepts=concepts), self.assertRaises(ValidationError):
                guided_segment_pair(FakeClient(), paths, concepts)

    def test_exif_geometry_is_refused_until_normalized(self):
        self.paths[0].write_bytes(jpeg_header(suffix=b"Exif\x00\x00"))
        client = FakeClient()
        with self.assertRaises(ValidationError):
            self.run_pair(client)
        self.assertEqual([], client.calls)

    def test_normalized_edges_convert_to_original_pixels_with_inclusive_right_bottom(self):
        self.assertEqual(candidate(0, 0, 639, 479),
                         normalized_box_to_pixels(candidate(0, 0, 1000, 1000), 640, 480))
        self.assertEqual(candidate(64, 96, 191, 191),
                         normalized_box_to_pixels(candidate(100, 200, 300, 400), 640, 480))
        self.assertEqual(candidate(0, 0, 0, 0),
                         normalized_box_to_pixels(candidate(1, 1, 2, 2), 100, 80))
        for invalid in (candidate(-1, 0, 10, 10), candidate(0, 0, 1001, 10),
                        candidate(10, 0, 10, 10), candidate(True, 0, 10, 10)):
            with self.assertRaises(ValidationError):
                normalized_box_to_pixels(invalid, 640, 480)

    def test_boundary_precision_can_be_refined_but_identity_and_occlusion_cannot(self):
        def edge_uncertain(value, _):
            for frame in value["parsed"]["frames"]:
                frame["objects"][0]["boundary_uncertain"] = True
        report = self.run_pair(FakeClient(transform=edge_uncertain))
        self.assertEqual(GUIDED_VERSION, report["provenance"]["version"])
        self.assertEqual(2, sum(len(objects) for objects in report["results"].values()))
        for field, value in (("identity_ambiguous", True), ("fully_visible", False)):
            def unsupported(result, _, field=field, value=value):
                for frame in result["parsed"]["frames"]:
                    frame["objects"][0][field] = value
            client = FakeClient(transform=unsupported)
            result = self.run_pair(client)
            self.assertEqual(1, len(client.calls))
            self.assertEqual(2, len(result["unavailable"]))
            self.assertTrue(all(not objects for objects in result["results"].values()))

    def test_wrong_coordinate_system_is_rejected_without_sam_call(self):
        client = FakeClient(transform=lambda value, _: value["parsed"].update(coordinate_system="image_pixels"))
        with self.assertRaises(ValidationError):
            self.run_pair(client)
        self.assertEqual(1, len(client.calls))


if __name__ == "__main__":
    unittest.main()
