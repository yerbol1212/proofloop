"""Mock-only experiment tests. These are not genuine SAM model evaluations."""

import base64
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from proofloop.live_scenes import CONCEPTS, SCENES, evaluate_scene_results, run_scenes
from proofloop.meta import MetaAPIError, SAM_MODEL


def manifest():
    return {"physical_measurement": False,
            "regions": {"left": [0, 0, 0.5, 1], "right": [0.5, 0, 1, 1]},
            "images": [{"file": scene, "sha256": hashlib.sha256(scene.encode()).hexdigest(),
                        "evidence_origin": "imagegen_synthetic"} for scene in SCENES]}


def detection(region):
    x1, x2 = (1, 3) if region == "left" else (6, 8)
    raster = bytearray(100)
    for y in range(2, 5):
        for x in range(x1, x2 + 1):
            raster[y * 10 + x] = 1
    return {"frame": 0, "object_id": 0, "confidence": 0.95,
            "box": {"x1": x1, "x2": x2, "y1": 2, "y2": 4, "width": 10, "height": 10},
            "mask": {"width": 10, "height": 10, "decoded": True,
                     "raster_format": "base64_row_major_uint8_0_or_1",
                     "raster_base64": base64.b64encode(raster).decode(),
                     "foreground_pixels": sum(raster), "raster_sha256": hashlib.sha256(raster).hexdigest()}}


def result(scene, concept):
    region = "right" if concept == "pliers" or scene == "wrong-region.png" else "left"
    objects = [] if concept == "screwdriver" and scene == "occluded.png" else [detection(region)]
    return {"source": {"sha256": hashlib.sha256(scene.encode()).hexdigest()}, "objects": objects,
            "response": {"id": "mock-only", "model": SAM_MODEL, "status": "completed"},
            "mask_decoding": "official_meta_sam_parser_0.1.0", "usage": {}, "text": "private-provider-text"}


def results():
    return {scene: {concept: result(scene, concept) for concept in CONCEPTS} for scene in SCENES}


class MockClient:
    """No networking, model inference, or credentials are used in these tests."""

    retries = 0
    max_requests = 12

    def __init__(self, failure=None):
        self.calls = []
        self.failure = failure

    def segment_image(self, image_path, concept, *, decode_masks):
        self.calls.append((image_path.name, concept, decode_masks))
        if self.failure == len(self.calls):
            raise MetaAPIError("Mock failure", code="mock_failure")
        return result(image_path.name, concept)

    def usage_summary(self):
        return {"request_attempts": len(self.calls), "fixture": "mock_only_no_real_requests"}


class MockSceneEvaluationTests(unittest.TestCase):
    def test_positive_pair_requires_two_distinct_sources(self):
        report = evaluate_scene_results(manifest(), results(), inference_origin="mock_adapter_test")
        self.assertTrue(report["gates"]["positive_visibility_two_distinct_generated_sources"])
        self.assertTrue(all(item["status"] == "unknown" for item in report["scenes"]["correct.png"]["checks"].values()))
        altered_manifest, altered_results = manifest(), results()
        correct_hash = altered_manifest["images"][0]["sha256"]
        altered_manifest["images"][1]["sha256"] = correct_hash
        for item in altered_results["correct-variant.png"].values():
            item["source"]["sha256"] = correct_hash
        duplicate = evaluate_scene_results(altered_manifest, altered_results)
        self.assertFalse(duplicate["gates"]["positive_visibility_two_distinct_generated_sources"])

    def test_wrong_region_reports_detection_without_absence_or_false_failure(self):
        report = evaluate_scene_results(manifest(), results())
        wrong = report["scenes"]["wrong-region.png"]
        self.assertTrue(report["gates"]["wrong_region_positive_screwdriver_right_supported"])
        self.assertEqual("unknown", wrong["checks"]["screwdriver-left"]["status"])
        source = next(item for item in wrong["sources"] if item["object_id"] == "screwdriver")
        self.assertEqual(["right"], source["regions"])
        self.assertFalse(wrong["observations"]["screwdriver"]["absence_validated"])

    def test_occlusion_zero_detections_does_not_become_absence(self):
        report = evaluate_scene_results(manifest(), results())
        occluded = report["scenes"]["occluded.png"]
        self.assertIsNone(occluded["observations"]["screwdriver"]["value"])
        self.assertEqual("unknown", occluded["checks"]["screwdriver-left"]["status"])
        self.assertTrue(report["gates"]["occluded_screwdriver_remains_unknown"])

    def test_occlusion_spurious_positive_does_not_meet_abstention_gate(self):
        altered = results()
        altered["occluded.png"]["screwdriver"]["objects"] = [detection("left")]
        report = evaluate_scene_results(manifest(), altered)
        self.assertFalse(report["gates"]["occluded_screwdriver_remains_unknown"])

    def test_partial_results_cannot_claim_completed_feasibility(self):
        report = evaluate_scene_results(manifest(), {})
        self.assertFalse(report["gates"]["all_eight_sam_results_available"])
        self.assertEqual("incomplete", report["overall"])
        self.assertFalse(report["gates"]["wrong_region_positive_screwdriver_right_supported"])

    def test_source_identity_mismatch_is_rejected(self):
        altered = results()
        altered["correct.png"]["screwdriver"]["source"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "exact manifest source"):
            evaluate_scene_results(manifest(), altered)

    def test_reports_omit_masks_private_text_and_physical_claims(self):
        report = evaluate_scene_results(manifest(), results())
        serialized = json.dumps(report)
        self.assertNotIn("raster_base64", serialized)
        self.assertNotIn("private-provider-text", serialized)
        self.assertNotIn("mock-only", serialized)
        self.assertEqual("provided_results_not_verified_live", report["inference_origin"])
        self.assertFalse(report["physical_measurement"])
        self.assertFalse(report["physical_repair_executed"])
        self.assertFalse(report["held_out_benchmark"])


class MockSceneRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.local = self.root / "local"
        self.manifest_path = self.root / "manifest.json"
        self.manifest_path.write_text(json.dumps(manifest()))
        for scene in SCENES:
            (self.root / scene).write_bytes(scene.encode())
        self.output = self.local / "mock-run"
        self.local_patch = patch("proofloop.live_scenes.LOCAL_ROOT", self.local)
        self.local_patch.start()

    def tearDown(self):
        self.local_patch.stop()
        self.temporary.cleanup()

    def run_mock(self, client, **kwargs):
        return run_scenes(client, self.manifest_path, self.output,
                          inference_origin="mock_adapter_test", **kwargs)

    def test_eight_calls_no_implicit_reuse_and_explicit_completed_resume(self):
        client = MockClient()
        report = self.run_mock(client)
        self.assertEqual(8, len(client.calls))
        self.assertTrue(all(decode for _, _, decode in client.calls))
        self.assertEqual(8, report["new_sam_attempts"])
        self.assertEqual("mock_adapter_test", report["inference_origin"])
        with self.assertRaisesRegex(ValueError, "exists"):
            self.run_mock(MockClient())
        resumed_client = MockClient()
        resumed = self.run_mock(resumed_client, resume=True)
        self.assertEqual(0, len(resumed_client.calls))
        self.assertEqual(8, len(resumed["reused_completed_responses"]))
        self.assertEqual("explicit_resume", resumed["mode"])

    def test_failure_stops_requests_and_resume_only_calls_missing(self):
        first = self.run_mock(MockClient(failure=3))
        self.assertEqual(3, first["new_sam_attempts"])
        self.assertEqual(1, len(first["failures"]))
        self.assertEqual("incomplete", first["overall"])
        second_client = MockClient()
        second = self.run_mock(second_client, resume=True)
        self.assertEqual(6, len(second_client.calls))
        self.assertEqual(2, len(second["reused_completed_responses"]))

    def test_cache_metadata_mismatch_refuses_paid_re_request(self):
        self.run_mock(MockClient())
        cached = self.output / "correct-screwdriver.json"
        value = json.loads(cached.read_text())
        value["metadata"]["inference_origin"] = "fabricated_origin"
        cached.write_text(json.dumps(value))
        client = MockClient()
        with self.assertRaisesRegex(ValueError, "Cache metadata differs"):
            self.run_mock(client, resume=True)
        self.assertEqual([], client.calls)

    def test_incomplete_or_undecoded_cached_result_is_refused(self):
        self.run_mock(MockClient())
        cached = self.output / "correct-screwdriver.json"
        value = json.loads(cached.read_text())
        value["result"]["response"]["status"] = "in_progress"
        cached.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, "completed official-decoded"):
            self.run_mock(MockClient(), resume=True)

    def test_source_hash_is_checked_before_paid_calls(self):
        (self.root / "correct.png").write_bytes(b"altered")
        client = MockClient()
        with self.assertRaisesRegex(ValueError, "manifest hash"):
            self.run_mock(client)
        self.assertEqual([], client.calls)

    def test_outputs_outside_local_or_unbounded_client_are_rejected(self):
        client = MockClient()
        with self.assertRaisesRegex(ValueError, "artifacts/local"):
            run_scenes(client, self.manifest_path, self.root / "public-report")
        for setting, value in (("retries", 1), ("max_requests", 13)):
            with self.subTest(setting=setting):
                altered_client = MockClient()
                setattr(altered_client, setting, value)
                with self.assertRaises(ValueError):
                    self.run_mock(altered_client, resume=True)
                self.assertEqual([], altered_client.calls)


if __name__ == "__main__":
    unittest.main()
