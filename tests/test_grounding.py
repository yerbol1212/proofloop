"""Mock adapter-shape tests; these fixtures are not real API inferences."""

import base64
from copy import deepcopy
import hashlib
import unittest

from proofloop.core import EvidenceStore, Protocol, Status, ValidationError, evaluate
from proofloop.grounding import SamImageEvidence, ground_sam_images


REGIONS = {"left": (0, 0, 0.5, 1), "right": (0.5, 0, 1, 1)}


def protocol(kind="visible", expected=True, *, min_stable_samples=2):
    contract = {"unique_target": True} if kind == "location" else {"expected": expected}
    return Protocol.from_dict({"id": "mock-kit", "version": "1", "objects": ["tool"],
                               "regions": ["left", "right"], "checks": [
                                   {"id": "check-tool", "kind": kind, "object_id": "tool", "region_id": "left",
                                    **contract, "min_stable_samples": min_stable_samples}]})


def decoded_mask(width, height, raster=None):
    raster = bytes([1] * (width * height)) if raster is None else bytes(raster)
    return {"width": width, "height": height, "decoded": True,
            "raster_format": "base64_row_major_uint8_0_or_1",
            "raster_base64": base64.b64encode(raster).decode(), "foreground_pixels": sum(raster),
            "raster_sha256": hashlib.sha256(raster).hexdigest()}


def detection(*, x1=1, x2=3, object_id=0, confidence=0.95):
    return {"frame": 0, "object_id": object_id,
            "box": {"x1": x1, "y1": 2, "x2": x2, "y2": 4, "width": 10, "height": 10},
            "confidence": confidence,
            "mask": decoded_mask(x2 - x1 + 1, 3)}


def sample(number, *, objects=None, timestamp=None, image_id=None, origin="mock_adapter_test"):
    image_id = str(number) if image_id is None else image_id
    return SamImageEvidence("tool", number * 100 if timestamp is None else timestamp,
                            {"source": {"sha256": hashlib.sha256(image_id.encode()).hexdigest(),
                                        "mime": "image/png", "bytes": 123, "extra_metadata": "preserved"},
                             "objects": [detection()] if objects is None else objects,
                             "response": {"id": f"mock-{number}", "model": "mock-sam-shape"},
                             "mask_decoding": "official_meta_sam_parser_0.1.0"}, origin, "tool")


def grounded(samples, *, task=None, regions=None):
    task = task or protocol()
    return task, ground_sam_images(task, REGIONS if regions is None else regions, samples,
                                  session_id="mock-session")


def status(task, result):
    store = EvidenceStore(task, "mock-session")
    for observation in result.observations:
        store.add(observation)
    return evaluate(task, store)[0].status


class SamGroundingTests(unittest.TestCase):
    def test_two_distinct_sources_support_positive_visible(self):
        task, result = grounded([sample(1), sample(2)])
        observation = result.observations[0]
        self.assertIs(True, observation.value)
        self.assertEqual(2, observation.stable_samples)
        self.assertFalse(observation.fully_visible)
        self.assertFalse(observation.absence_validated)
        self.assertEqual(Status.PASS, status(task, result))

    def test_box_local_raster_uses_nonzero_source_offset(self):
        _, result = grounded([sample(1), sample(2)])
        detection_result = result.evidence["sources"][0]["detections"][0]
        self.assertEqual((0.1, 0.2, 0.4, 0.5), detection_result["foreground"])
        self.assertEqual(detection_result["box"], detection_result["foreground"])

    def test_resampled_raster_pixel_edges_map_through_box_extent(self):
        record = detection(x1=0, x2=3)
        record["box"].update(y1=2, y2=5)
        record["mask"] = decoded_mask(2, 2, [0, 1, 0, 0])
        task, result = grounded([sample(1, objects=[record]), sample(2, objects=[record])])
        foreground = result.evidence["sources"][0]["detections"][0]["foreground"]
        self.assertEqual((0.2, 0.2, 0.4, 0.4), foreground)
        self.assertEqual(Status.PASS, status(task, result))

    def test_synthetic_mask_with_live_observed_geometry_maps_to_left_region(self):
        # Only dimensions/box metadata match the live synthetic scene. This
        # all-ones test raster is authored here, not copied model output.
        record = detection()
        record["box"] = {"x1": 301, "y1": 159, "x2": 418, "y2": 1044,
                         "width": 1254, "height": 1254}
        record["mask"] = decoded_mask(118, 886)
        task, result = grounded([sample(1, objects=[record]), sample(2, objects=[record])])
        foreground = result.evidence["sources"][0]["detections"][0]["foreground"]
        expected = (301 / 1254, 159 / 1254, 419 / 1254, 1045 / 1254)
        for actual, target in zip(foreground, expected):
            self.assertAlmostEqual(actual, target)
        self.assertEqual(Status.PASS, status(task, result))

    def test_whole_image_box_retains_whole_image_mask_geometry(self):
        record = detection()
        record["box"].update(x1=0, y1=0, x2=9, y2=9)
        raster = bytearray(100)
        raster[21] = 1
        record["mask"] = decoded_mask(10, 10, raster)
        task, result = grounded([sample(1, objects=[record]), sample(2, objects=[record])],
                                regions={"left": (0, 0, 1, 1)})
        foreground = result.evidence["sources"][0]["detections"][0]["foreground"]
        self.assertEqual((0.1, 0.2, 0.2, 0.3), foreground)
        self.assertEqual(Status.PASS, status(task, result))

    def test_identical_hashes_cannot_supply_stability(self):
        task, result = grounded([sample(1, image_id="same"), sample(2, image_id="same")])
        self.assertEqual(1, result.observations[0].stable_samples)
        self.assertEqual(Status.UNKNOWN, status(task, result))

    def test_protocol_larger_stability_requirement_applies(self):
        task, result = grounded([sample(1), sample(2)], task=protocol(min_stable_samples=3))
        self.assertEqual(Status.UNKNOWN, status(task, result))
        task, result = grounded([sample(1), sample(2), sample(3)], task=protocol(min_stable_samples=3))
        self.assertEqual(Status.PASS, status(task, result))

    def test_zero_detections_cannot_become_absence(self):
        task, result = grounded([sample(1, objects=[]), sample(2, objects=[])])
        self.assertIsNone(result.observations[0].value)
        self.assertEqual(Status.UNKNOWN, status(task, result))

    def test_missing_confidence_undecoded_or_crossing_region_is_unknown(self):
        undecoded = detection()
        undecoded["mask"]["decoded"] = False
        for record in (detection(confidence=None), undecoded, detection(x1=3, x2=6)):
            with self.subTest(record=record):
                task, result = grounded([sample(1, objects=[record]), sample(2, objects=[record])])
                self.assertIsNone(result.observations[0].value)
                self.assertEqual(Status.UNKNOWN, status(task, result))

    def test_incomplete_count_is_lower_bound_and_cannot_pass_exact(self):
        task, result = grounded([sample(1), sample(2)], task=protocol("count", 1))
        self.assertEqual(1, result.observations[0].value)
        self.assertEqual(Status.UNKNOWN, status(task, result))

    def test_supported_lower_bound_can_prove_too_many(self):
        records = [detection(x1=0, x2=1, object_id=0), detection(x1=3, x2=4, object_id=1)]
        task, result = grounded([sample(1, objects=records), sample(2, objects=records)],
                                task=protocol("count", 1))
        self.assertEqual(2, result.observations[0].value)
        self.assertEqual(Status.FAIL, status(task, result))

    def test_duplicate_or_overlapping_instances_cannot_inflate_count(self):
        for records in ([detection(), detection(object_id=1)], [detection(), detection()]):
            task, result = grounded([sample(1, objects=records), sample(2, objects=records)],
                                    task=protocol("count", 1))
            self.assertIsNone(result.observations[0].value)
            self.assertEqual(Status.UNKNOWN, status(task, result))

    def test_new_ambiguous_sample_does_not_fall_back_to_old_pass(self):
        task, result = grounded([sample(1), sample(2), sample(3, objects=[])])
        self.assertIsNone(result.observations[0].value)
        self.assertEqual(Status.UNKNOWN, status(task, result))

    def test_stability_resets_after_unknown(self):
        task, result = grounded([sample(1), sample(2), sample(3, objects=[]), sample(4)])
        self.assertEqual(1, result.observations[0].stable_samples)
        self.assertEqual(Status.UNKNOWN, status(task, result))

    def test_duplicate_hash_does_not_refresh_age(self):
        task, result = grounded([sample(1, timestamp=100), sample(2, timestamp=200),
                                sample(3, timestamp=40000, image_id="2")])
        self.assertIsNone(result.observations[0].value)
        self.assertEqual(Status.UNKNOWN, status(task, result))

    def test_oldest_required_support_controls_observation_age(self):
        task, result = grounded([sample(1, timestamp=100), sample(2, timestamp=500)])
        self.assertEqual(100, result.observations[0].timestamp_ms)
        store = EvidenceStore(task, "mock-session")
        store.add(result.observations[0])
        self.assertEqual(Status.UNKNOWN, evaluate(task, store, now_ms=30200)[0].status)

    def test_source_metadata_and_caller_origin_are_preserved(self):
        source = sample(1, origin="explicitly_synthetic_test")
        _, result = grounded([source, sample(2, origin="explicitly_synthetic_test")])
        audit = result.evidence["sources"][0]
        self.assertEqual(source.result["source"], audit["raw_source_metadata"])
        self.assertEqual("explicitly_synthetic_test", audit["evidence_origin"])
        self.assertEqual("mock-sam-shape", audit["reported_model"])
        self.assertNotIn("is_live", audit)

    def test_foreground_inside_region_does_not_override_crossing_box(self):
        bad = detection(x1=3, x2=6)
        # Foreground occupies the left half of the local box, entirely inside
        # the left region. The box still crosses its boundary and is ambiguous.
        bad["mask"] = decoded_mask(4, 3, [1, 0, 0, 0] * 3)
        task, result = grounded([sample(1, objects=[bad]), sample(2, objects=[bad])])
        self.assertEqual(Status.UNKNOWN, status(task, result))

    def test_bad_raster_digest_is_unknown(self):
        bad = detection()
        bad["mask"]["raster_sha256"] = "0" * 64
        task, result = grounded([sample(1, objects=[bad]), sample(2, objects=[bad])])
        self.assertEqual(Status.UNKNOWN, status(task, result))

    def test_raster_dimensions_must_match_decoded_bytes(self):
        bad = detection()
        bad["mask"]["width"] = 5
        task, result = grounded([sample(1, objects=[bad]), sample(2, objects=[bad])])
        self.assertIsNone(result.observations[0].value)
        self.assertEqual(Status.UNKNOWN, status(task, result))

    def test_detections_in_other_region_do_not_prove_absence_here(self):
        records = [detection(x1=6, x2=7)]
        task, result = grounded([sample(1, objects=records), sample(2, objects=records)])
        self.assertIsNone(result.observations[0].value)
        self.assertEqual(Status.UNKNOWN, status(task, result))

    def test_declared_unique_location_uses_positive_evidence_for_pass_and_fail(self):
        for box, value, expected in ((detection(), "left", Status.PASS),
                                     (detection(x1=6, x2=7), "right", Status.FAIL)):
            task, result = grounded([sample(1, objects=[box]), sample(2, objects=[box])],
                                    task=protocol("location"))
            observation = result.observations[0]
            self.assertEqual(value, observation.value)
            self.assertEqual("left", observation.region_id)
            self.assertFalse(observation.fully_visible)
            self.assertFalse(observation.absence_validated)
            self.assertEqual(expected, status(task, result))

    def test_location_multiple_targets_or_missing_target_are_unknown(self):
        two = [detection(), detection(x1=6, x2=7, object_id=1)]
        for records in ([], two, [detection(x1=3, x2=6)], [detection(confidence=0.2)]):
            task, result = grounded([sample(1, objects=records), sample(2, objects=records)],
                                    task=protocol("location"))
            self.assertIsNone(result.observations[0].value)
            self.assertEqual(Status.UNKNOWN, status(task, result))

    def test_location_stability_resets_when_target_region_changes(self):
        right = [detection(x1=6, x2=7)]
        task, result = grounded([sample(1, objects=right), sample(2, objects=right), sample(3)],
                                task=protocol("location"))
        self.assertEqual("left", result.observations[0].value)
        self.assertEqual(1, result.observations[0].stable_samples)
        self.assertEqual(Status.UNKNOWN, status(task, result))
        task, result = grounded([sample(1, objects=right), sample(2, objects=right), sample(3), sample(4)],
                                task=protocol("location"))
        self.assertEqual(Status.PASS, status(task, result))

    def test_location_duplicate_hash_cannot_prove_correction(self):
        task, result = grounded([sample(1, image_id="same"), sample(2, image_id="same")],
                                task=protocol("location"))
        self.assertEqual(Status.UNKNOWN, status(task, result))

    def test_overlapping_calibration_is_rejected(self):
        with self.assertRaises(ValidationError):
            grounded([sample(1)], regions={"left": (0, 0, 0.7, 1), "right": (0.5, 0, 1, 1)})

    def test_missing_object_sources_and_missing_calibration_are_unknown(self):
        for sources, regions in (([], REGIONS), ([sample(1), sample(2)], {"right": REGIONS["right"]})):
            task, result = grounded(sources, regions=regions)
            self.assertIsNone(result.observations[0].value)
            self.assertEqual(Status.UNKNOWN, status(task, result))

    def test_malformed_source_hash_and_missing_origin_are_rejected(self):
        original = sample(1)
        wrong_hash = deepcopy(original.result)
        wrong_hash["source"]["sha256"] = "not-a-digest"
        for bad in (SamImageEvidence("tool", 1, wrong_hash, "mock"),
                    SamImageEvidence("tool", 1, original.result, "")):
            with self.assertRaises(ValidationError):
                grounded([bad])


if __name__ == "__main__":
    unittest.main()
