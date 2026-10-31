"""Authored multi-stage integrity/geometry/core controls, never accuracy claims."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from proofloop import benchmark, crop_evidence, crop_verification as module
from proofloop.core import ValidationError
from proofloop.meta import SAM_MODEL, SPARK_MODEL, decode_sam_masks, parse_sam_boxes
from tests import test_crop_evidence as fixtures


def spark(parsed, paths):
    text = json.dumps(parsed)
    return {"source": None, "sources": [{"sha256": fixtures.sha(path)} for path in paths], "text": text,
            "parsed": deepcopy(parsed), "response": {"model": SPARK_MODEL, "status": "completed", "output_text": text,
            "private_message": "PRIVATE_MODEL_BODY", "usage": {"input_tokens": 100, "output_tokens": 50}}}


def located(state="visible", count="one"):
    return {"coordinate_system": "normalized_0_1000", "frames": [
        {"frame_index": i, "state": state, "terminal_tool_count": count,
         "boxes": ([{"x1": 600, "y1": 300, "x2": 660, "y2": 500}] if count == "one" else
                   [{"x1": 600, "y1": 300, "x2": 660, "y2": 500}, {"x1": 800, "y1": 300, "x2": 900, "y2": 500}]
                   if count == "two" else [])} for i in range(2)]}


def supported(count="one"):
    return {"frames": [dict({field: "supported" for field in module.FIELDS}, frame_index=i,
        full_scene_match_count="one", manipulator_relation="released", gripper_mask_coverage="supported",
        visible_terminal_tool_count=count) for i in range(2)]}


class CropVerificationTests(unittest.TestCase):
    render = fixtures.CropEvidenceTests.render

    def setUp(self):
        fixtures.CropEvidenceTests.setUp(self)
        patch("proofloop.tracking_geometry.decode_sam_masks", side_effect=fixtures.decoded).start()
        patch("proofloop.crop_verification.subprocess.run", side_effect=self.render_all).start()
        source = deepcopy(self.envelope["contract"])
        source["cases"][0]["regions"]["dispense"] = [0, 0, .5, 1]
        self.envelope = benchmark.approve_manifest(source, benchmark.manifest_hash(source), confirm_unique_targets=True)
        self.source_items = self.make_items()
        self.pair = module.prepare_pair(self.envelope, self.source_items, self.root / "artifacts/local/pair", root=self.root)

    def render_all(self, args, **options):
        if args[-1] == "-version":
            options["stdout"].write(b"fixture-ffmpeg-1\n")
            return subprocess.CompletedProcess(args, 0)
        if "-vf" in args:
            return fixtures.CropEvidenceTests.render(self, args, **options)
        data = options["input"]
        _magic, dims, _max, _pixels = data.split(b"\n", 3)
        width, height = map(int, dims.split())
        Path(args[-1]).write_bytes(b"\x89PNG\r\n\x1a\n" + struct.pack(">I", 13) + b"IHDR" + struct.pack(">II", width, height)
                                  + hashlib.sha256(data).digest())
        return subprocess.CompletedProcess(args, 0)

    def make_items(self, suffix="source"):
        items = []
        for i, path in enumerate(self.paths):
            target = fixtures.sam(fixtures.sha(path))
            prepared = crop_evidence.prepare_frame(self.envelope, i, target, self.locator,
                self.root / f"artifacts/local/{suffix}-{i}", root=self.root)
            response = fixtures.sam(prepared["crop_sha256"], 296, 296, (125, 111, 171, 184), .91)
            projection = crop_evidence.project_result(prepared, response, root=self.root)
            items.append({"crop_prepared": prepared, "crop_projection": projection})
        return items

    def bind(self, value=None):
        return module.bind_locator(self.pair, spark(value or located(), module.locator_images(self.pair)), root=self.root)

    def grippers(self, bound, *, box=(384, 144, 421, 239), confidence=.95, empty=False):
        results = {}
        for request in module.spatial_prompts(self.pair, bound, root=self.root):
            if not request["request_required"]:
                continue
            value = fixtures.sam(request["source_sha256"], 640, 480, box, confidence)
            if len(request["requested_object_ids"]) == 2:
                text = value["text"] + "," + value["text"][4:].replace("1<|box;", "2<|box;", 1)
                value.update(text=text, objects=fixtures.decoded(parse_sam_boxes(text)))
                value["response"]["output_text"] = text
            if empty:
                value.update(text="", objects=[])
                value["response"]["output_text"] = ""
            results[request["frame_index"]] = value
        return results

    def final(self, bound=None, grippers=None, suffix="final"):
        bound = bound or self.bind()
        return module.prepare_final(self.pair, bound, self.grippers(bound) if grippers is None else grippers,
                                    self.root / "artifacts/local" / suffix, root=self.root)

    def assess(self, final=None, parsed=None):
        final = final or self.final()
        return module.evaluate(final, spark(parsed or supported(), module.final_images(final)), root=self.root)

    def test_exact_eight_attachments_and_no_forged_full_scene_provider_response(self):
        final = self.final()
        evaluation = self.assess(final)
        self.assertEqual("supported", evaluation["decision"])
        self.assertEqual(8, len(module.final_images(final)))
        self.assertEqual([path.resolve() for path in self.paths], module.final_images(final)[:2])
        report = module.derive_task_report(evaluation, root=self.root)
        self.assertEqual("pass", report["overall"])
        self.assertEqual(2, report["grounding"]["observations"][0]["stable_samples"])
        sources = report["grounding"]["evidence"]["sources"]
        self.assertTrue(all(source["reported_model"] is None and source["response_id"] is None for source in sources))
        self.assertTrue(all(source["raw_source_metadata"]["derivation"] == "native_crop_integer_source_projection" for source in sources))
        self.assertFalse(report["new_current_authority"])
        self.assertFalse(report["changes_production_acceptance"])
        self.assertFalse(report["physical_action_execution_verified"])

    def test_spatial_prompt_only_actual_original_coordinates_and_ids(self):
        bound = self.bind(located(count="two"))
        requests = module.spatial_prompts(self.pair, bound, root=self.root)
        for request in requests:
            self.assertTrue(request["request_required"])
            self.assertEqual([1, 2], request["requested_object_ids"])
            self.assertTrue(request["prompt"].startswith("<0f>1<|box;x1=384;y1=144;x2=422;y2=239;w=640;h=480|>,2<|box;"))
            self.assertNotIn("gripper", request["prompt"])
            self.assertIn(request["source_file"], [str(p.resolve()) for p in self.paths])
        final = self.final(bound)
        result = self.assess(final, supported(count="two"))
        self.assertEqual("supported", result["decision"])

    def test_locator_state_count_and_boxes_must_match_exactly(self):
        values = []
        value = located()
        value["frames"][0]["terminal_tool_count"] = "two"
        values.append(value)
        value = located()
        value["frames"][0]["state"] = "not_visible"
        values.append(value)
        value = located(count="two")
        value["frames"][0]["boxes"][1] = deepcopy(value["frames"][0]["boxes"][0])
        values.append(value)
        for value in values:
            with self.assertRaises(ValidationError):
                self.bind(value)

    def test_uncertain_or_too_many_terminal_tools_cannot_be_overridden_by_final(self):
        bound = self.bind(located("uncertain", "multiple"))
        self.assertTrue(all(not row["request_required"] for row in module.spatial_prompts(self.pair, bound, root=self.root)))
        final = self.final(bound, {}, suffix="uncertain")
        evaluation = self.assess(final)
        self.assertEqual("withheld", evaluation["decision"])
        self.assertEqual("unknown", module.derive_task_report(evaluation, root=self.root)["overall"])
        self.assertTrue(all("terminal_tool_locator_uncertain" in row["withheld_reasons"] for row in evaluation["frames"]))

    def test_zero_visible_tool_needs_consistent_final_zero_and_empty_mask_is_not_absence(self):
        bound = self.bind(located("not_visible", "zero"))
        final = self.final(bound, {}, suffix="zero")
        self.assertTrue(all(row["gripper_mask_is_empty_visual_aid"] for row in final["frames"]))
        self.assertTrue(all(not row["geometry"]["geometry_available"] for row in final["frames"]))
        evaluation = self.assess(final, supported("zero"))
        self.assertEqual("supported", evaluation["decision"])
        self.assertTrue(any("not verified physical absence" in text for text in evaluation["limits"]))
        self.assertFalse(evaluation["physical_action_execution_verified"])

    def test_completed_missing_gripper_masks_are_unknown_not_success_or_stage_failure(self):
        bound = self.bind()
        results = self.grippers(bound, empty=True)
        module.validate_gripper_result(self.pair, bound, 0, results[0], root=self.root)
        final = self.final(bound, results)
        evaluation = self.assess(final)
        self.assertEqual("completed", evaluation["status"])
        self.assertEqual("withheld", evaluation["decision"])
        self.assertTrue(all("terminal_tool_sam_missing_requested_mask" in row["withheld_reasons"] for row in evaluation["frames"]))
        self.assertEqual("unknown", module.derive_task_report(evaluation, root=self.root)["overall"])

    def test_missing_entire_required_stage_and_incomplete_provider_response_cannot_be_scored(self):
        bound = self.bind()
        results = self.grippers(bound)
        with self.assertRaises(ValidationError):
            self.final(bound, {0: results[0]}, suffix="missing-stage")
        results[0]["response"]["status"] = "incomplete"
        with self.assertRaises(ValidationError):
            module.validate_gripper_result(self.pair, bound, 0, results[0], root=self.root)

    def test_actual_still_image_frame_one_provider_records_are_rejected_end_to_end(self):
        bound = self.bind()
        results = self.grippers(bound)
        raw = deepcopy(results[0])
        text = raw["text"].replace("<0f>", "<1f>")
        raw.update(text=text, objects=fixtures.decoded(parse_sam_boxes(text)))
        raw["response"]["output_text"] = text
        self.assertEqual(1, raw["objects"][0]["frame"])
        with self.assertRaises(ValidationError):
            module.validate_gripper_result(self.pair, bound, 0, raw, root=self.root)
        with self.assertRaises(ValidationError):
            self.final(bound, {0: raw, 1: results[1]}, suffix="frame-one-gripper")
        items = deepcopy(self.source_items)
        projection = items[0]["crop_projection"]
        crop_raw = projection["actual_crop_response"]
        text = crop_raw["text"].replace("<0f>", "<1f>")
        crop_raw.update(text=text, objects=fixtures.decoded(parse_sam_boxes(text)))
        crop_raw["response"]["output_text"] = text
        projection["derived_full_source_geometry"][0]["frame"] = 1
        projection["projection_sha256"] = crop_evidence._hash({k: v for k, v in projection.items() if k != "projection_sha256"})
        with self.assertRaises(ValidationError):
            module.prepare_pair(self.envelope, items, self.root / "artifacts/local/frame-one-target", root=self.root)

    def test_undeclared_gripper_ids_and_full_source_hash_mismatch_rejected(self):
        bound = self.bind()
        results = self.grippers(bound)
        value = deepcopy(results[0])
        text = value["text"].replace("<0f>1", "<0f>3")
        value.update(text=text, objects=fixtures.decoded(parse_sam_boxes(text)))
        value["response"]["output_text"] = text
        with self.assertRaises(ValidationError):
            module.validate_gripper_result(self.pair, bound, 0, value, root=self.root)
        value = deepcopy(results[0])
        value["source"]["sha256"] = self.pair["frames"][0]["crop_sha256"]
        with self.assertRaises(ValidationError):
            module.validate_gripper_result(self.pair, bound, 0, value, root=self.root)

    def test_low_or_missing_gripper_confidence_never_qualifies_positive(self):
        for index, confidence in enumerate((.79, None)):
            bound = self.bind()
            final = self.final(bound, self.grippers(bound, confidence=confidence), suffix=f"low-{index}")
            result = self.assess(final)
            self.assertEqual("withheld", result["decision"])
            self.assertTrue(all("terminal_tool_sam_confidence_unavailable_or_below_0_8" in row["withheld_reasons"] for row in result["frames"]))

    def test_near_geometry_vetoes_claimed_release_even_with_low_gripper_score(self):
        bound = self.bind()
        final = self.final(bound, self.grippers(bound, box=(304, 200, 310, 205), confidence=.1))
        result = self.assess(final)
        self.assertEqual("withheld", result["decision"])
        self.assertTrue(all(row["geometry"]["veto"] for row in result["frames"]))
        self.assertTrue(all("near_terminal_tool_foreground" in row["withheld_reasons"] for row in result["frames"]))
        self.assertTrue(all(not row["geometry"]["positive_support_granted"] for row in result["frames"]))

    def test_far_geometry_cannot_override_contact_or_uncertain_semantics(self):
        final = self.final()
        for field, value in (("manipulator_relation", "held_or_contacting"), ("manipulator_relation", "uncertain"),
                             ("gripper_mask_coverage", "uncertain"), ("mask_target_only", "contradicted"),
                             ("full_visibility", "uncertain"), ("identity", "contradicted"),
                             ("target_extent_covered", "uncertain"), ("full_scene_match_count", "multiple"),
                             ("visible_terminal_tool_count", "two")):
            parsed = supported()
            parsed["frames"][0][field] = value
            evaluation = self.assess(final, parsed)
            self.assertEqual("withheld", evaluation["decision"])
            self.assertEqual("unknown", module.derive_task_report(evaluation, root=self.root)["overall"])

    def test_stage_bindings_cannot_be_replaced_by_different_self_consistent_evidence(self):
        bound = self.bind()
        results = self.grippers(bound)
        final = self.final(bound, results)
        module.validate_final_stage(final, self.pair, bound, results, root=self.root)
        altered = deepcopy(results)
        altered[0]["response"]["private_extra"] = "different stage"
        with self.assertRaises(ValidationError):
            module.validate_final_stage(final, self.pair, bound, altered, root=self.root)

    def test_final_classifications_cannot_replace_raw_response_or_attachment_order(self):
        final = self.final()
        raw = spark(supported(), module.final_images(final))
        raw["parsed"]["frames"][0]["manipulator_relation"] = "held_or_contacting"
        with self.assertRaises(ValidationError):
            module.evaluate(final, raw, root=self.root)
        raw = spark(supported(), module.final_images(final))
        raw["sources"].reverse()
        with self.assertRaises(ValidationError):
            module.evaluate(final, raw, root=self.root)

    def test_duplicate_json_keys_and_incomplete_spark_are_unscored(self):
        final = self.final()
        raw = spark(supported(), module.final_images(final))
        raw["response"]["status"] = "incomplete"
        with self.assertRaises(ValidationError):
            module.evaluate(final, raw, root=self.root)
        raw = spark(supported(), module.final_images(final))
        text = raw["text"].replace('"identity": "supported"', '"identity": "supported", "identity": "contradicted"', 1)
        raw["text"] = raw["response"]["output_text"] = text
        with self.assertRaises(ValidationError):
            module.evaluate(final, raw, root=self.root)

    def test_public_receipt_has_no_paths_masks_provider_text_or_ids(self):
        evaluation = self.assess()
        public = module.public_evaluation(evaluation, root=self.root)
        encoded = json.dumps(public)
        for value in ("inputs/local", "artifacts/local", "raster_base64", "encoded_payload", "PRIVATE_MODEL_BODY", "raw_result", "source_file"):
            self.assertNotIn(value, encoded)
        self.assertIn("evaluation_sha256", public)
        self.assertFalse(public["new_current_authority"])

    def test_prompts_contain_no_task_requirements_regions_labels_or_answer(self):
        final = self.final()
        for prompt in (module.locator_prompt(self.pair), module.final_prompt(final)):
            self.assertNotIn(self.envelope["contract"]["requirements"], prompt)
            self.assertNotIn('"regions"', prompt)
            self.assertNotIn('"annotation"', prompt)
            self.assertNotIn('"expected"', prompt)
            self.assertNotIn('"min_confidence"', prompt)

    def test_source_projection_or_rendered_mask_tamper_cannot_be_rehashed(self):
        final = self.final()
        path = Path(final["frames"][0]["gripper_mask_file"])
        path.write_bytes(path.read_bytes() + b"replacement")
        final["frames"][0]["gripper_mask_sha256"] = fixtures.sha(path)
        final["final_prepared_sha256"] = module._hash({k: v for k, v in final.items() if k != "final_prepared_sha256"})
        with self.assertRaises(ValidationError):
            module.validate_final(final, root=self.root)

    def test_whole_source_region_check_still_rejects_qualified_wrong_placement(self):
        source = deepcopy(self.envelope["contract"])
        source["protocol"]["checks"][0]["region_id"] = "staging"
        self.envelope = benchmark.approve_manifest(source, benchmark.manifest_hash(source), confirm_unique_targets=True)
        self.source_items = self.make_items("other-region")
        self.pair = module.prepare_pair(self.envelope, self.source_items, self.root / "artifacts/local/other-pair", root=self.root)
        report = module.derive_task_report(self.assess(), root=self.root)
        self.assertEqual("fail", report["overall"])
        self.assertEqual(2, report["grounding"]["observations"][0]["stable_samples"])

    def test_repeated_original_source_or_wrong_pair_order_cannot_add_stability(self):
        with self.assertRaises(ValidationError):
            module.prepare_pair(self.envelope, [self.source_items[0], self.source_items[0]], self.root / "artifacts/local/repeated", root=self.root)
        with self.assertRaises(ValidationError):
            module.prepare_pair(self.envelope, self.source_items[::-1], self.root / "artifacts/local/reversed", root=self.root)


class CropVerificationRealCodecTests(unittest.TestCase):
    def test_real_codec_mask_union_and_explicit_derived_grounding(self):
        try:
            import meta_sam_parser  # noqa: F401
        except (ImportError, TypeError, SyntaxError):
            self.skipTest("Optional official SAM parser requires Python >=3.10")
        try:
            executable = module.semantic._extractor_path()
        except ValidationError:
            self.skipTest("Optional FFmpeg is required for the actual codec integration")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = fixtures.contract()
            source["cases"] = source["cases"][:1]
            source["cases"][0]["regions"]["dispense"] = [0, 0, .5, 1]
            folder = root / "inputs/local/authored"
            folder.mkdir(parents=True)
            (folder / "video.mp4").write_text("authored-video")
            paths = []
            for i, frame in enumerate(source["cases"][0]["frames"]):
                path = root / frame["file"]
                subprocess.run([executable, "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
                    "testsrc2=size=640x480:rate=1", "-vf", f"hue=h={30 * i}", "-frames:v", "1", "-threads", "1",
                    "-map_metadata", "-1", str(path)], check=True, capture_output=True, timeout=15)
                frame["sha256"] = fixtures.sha(path)
                paths.append(path)
            envelope = benchmark.approve_manifest(source, benchmark.manifest_hash(source), confirm_unique_targets=True)
            parsed = {"coordinate_system": "normalized_0_1000", "frames": [
                {"frame_index": i, "width": 640, "height": 480, "objects": [{"object_id": "glass",
                 "identity_ambiguous": False, "fully_visible": True, "boundary_uncertain": False,
                 "candidates": [{"x1": 400, "x2": 480, "y1": 390, "y2": 550}]}]} for i in range(2)]}
            old_locator = spark(parsed, paths)
            items = []
            for i, path in enumerate(paths):
                original = fixtures.sam(fixtures.sha(path), decoder=decode_sam_masks)
                prepared = crop_evidence.prepare_frame(envelope, i, original, old_locator, root / f"artifacts/local/crop-{i}", root=root)
                raw = fixtures.sam(prepared["crop_sha256"], 296, 296, (125, 111, 171, 184), .91, decoder=decode_sam_masks)
                items.append({"crop_prepared": prepared, "crop_projection": crop_evidence.project_result(prepared, raw, root=root)})
            pair = module.prepare_pair(envelope, items, root / "artifacts/local/pair", root=root)
            locator = module.bind_locator(pair, spark(located(), paths), root=root)
            results = {i: fixtures.sam(fixtures.sha(path), 640, 480, (384, 144, 421, 239), .95, decoder=decode_sam_masks)
                       for i, path in enumerate(paths)}
            final = module.prepare_final(pair, locator, results, root / "artifacts/local/final", root=root)
            module.validate_final_stage(final, pair, locator, results, root=root)
            for i in range(2):
                actual = subprocess.run([executable, "-hide_banner", "-loglevel", "error", "-i", final["frames"][i]["gripper_mask_file"],
                    "-f", "rawvideo", "-pix_fmt", "gray", "pipe:1"], check=True, capture_output=True, timeout=15).stdout
                expected = bytearray(640 * 480)
                mapped = module.tracking_geometry._mapped(results[i]["objects"][0], 640, 480)
                box = mapped["box"]
                for y in range(box["y1"], box["y2"] + 1):
                    for x in range(box["x1"], box["x2"] + 1):
                        if mapped["pixels"][(y - box["y1"]) * mapped["width"] + x - box["x1"]]:
                            expected[y * 640 + x] = 255
                self.assertEqual(bytes(expected), actual)
            assessment = module.evaluate(final, spark(supported(), module.final_images(final)), root=root)
            report = module.derive_task_report(assessment, root=root)
            self.assertEqual("pass", report["overall"])
            self.assertTrue(all(item["reported_model"] is None for item in report["grounding"]["evidence"]["sources"]))


if __name__ == "__main__":
    unittest.main()
