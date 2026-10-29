"""Authored crop transport/projection controls; no perception accuracy claims."""
from copy import deepcopy
import base64
import hashlib
import json
from pathlib import Path
import re
import struct
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from proofloop import benchmark, crop_evidence as module
from proofloop.core import ValidationError
from proofloop.meta import SAM_MODEL, SPARK_MODEL, parse_sam_boxes, decode_sam_masks
from tests.test_benchmark import contract
from tests.test_guided import jpeg_header


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def decoded(records):
    result = deepcopy(records)
    for record in result:
        mask = record["mask"]
        raw = b"\x01" * (mask["width"] * mask["height"])
        mask.update(decoded=True, raster_format="base64_row_major_uint8_0_or_1",
                    raster_base64=base64.b64encode(raw).decode(), foreground_pixels=len(raw),
                    raster_sha256=hashlib.sha256(raw).hexdigest())
    return result


def sam(source_hash, width=640, height=480, box=(257, 191, 303, 264), confidence=.73, decoder=decoded):
    x1, y1, x2, y2 = box
    c = f";c={confidence}" if confidence is not None else ""
    text = (f"<0f>1<|box;x1={x1};y1={y1};x2={x2};y2={y2};w={width};h={height}{c}|>"
            f"<|mask;x=0;y=0{c};data=5,5,!!!!!(QO(0lu8?|>")
    return {"source": {"sha256": source_hash}, "response": {"status": "completed", "model": SAM_MODEL,
             "output_text": text, "usage": {"images_processed": 1}}, "text": text,
            "objects": decoder(parse_sam_boxes(text)), "mask_decoding": "official_meta_sam_parser_0.1.0"}


class CropEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        source = contract()
        source["cases"] = source["cases"][:1]
        (self.root / "inputs/local/authored").mkdir(parents=True)
        (self.root / "inputs/local/authored/video.mp4").write_text("authored-video")
        self.paths = []
        for i, frame in enumerate(source["cases"][0]["frames"]):
            path = self.root / frame["file"]
            path.write_bytes(jpeg_header(640, 480, str(i).encode()))
            frame["sha256"] = sha(path)
            self.paths.append(path)
        self.envelope = benchmark.approve_manifest(source, benchmark.manifest_hash(source), confirm_unique_targets=True)
        self.target = sam(sha(self.paths[0]))
        parsed = {"coordinate_system": "normalized_0_1000", "frames": [
            {"frame_index": i, "width": 640, "height": 480, "objects": [{"object_id": "glass",
             "identity_ambiguous": False, "fully_visible": True, "boundary_uncertain": False,
             "candidates": [{"x1": 400, "x2": 480, "y1": 390, "y2": 550}]}]} for i in range(2)]}
        text = json.dumps(parsed)
        self.locator = {"source": None, "sources": [{"sha256": sha(path)} for path in self.paths],
                        "response": {"status": "completed", "model": SPARK_MODEL, "output_text": text},
                        "text": text, "parsed": parsed}
        self.addCleanup(patch.stopall)
        patch("proofloop.mask_semantics.decode_sam_masks", side_effect=decoded).start()
        patch("proofloop.semantic._extractor_path", return_value="fixture-ffmpeg").start()
        self.commands = []
        patch("proofloop.crop_evidence.subprocess.run", side_effect=self.render).start()

    def render(self, args, **options):
        self.commands.append((args, options))
        if args[-1] == "-version":
            options["stdout"].write(b"fixture-ffmpeg-1\n")
        else:
            filters = args[args.index("-vf") + 1]
            w, h = map(int, re.search(r"crop=w=(\d+):h=(\d+)", filters).groups())
            data = Path(args[args.index("-i") + 1]).read_bytes()
            header = b"\x89PNG\r\n\x1a\n" + struct.pack(">I", 13) + b"IHDR" + struct.pack(">II", w, h)
            Path(args[-1]).write_bytes(header + hashlib.sha256(data + filters.encode()).digest())
        return subprocess.CompletedProcess(args, 0)

    def prepare(self, suffix="prepared", target=None):
        return module.prepare_frame(self.envelope, 0, self.target if target is None else target, self.locator,
                                    self.root / "artifacts/local" / suffix, root=self.root)

    def result(self, prepared, **changes):
        b = prepared["details"]["crop_rectangle"]
        return sam(prepared["crop_sha256"], b["x2"] - b["x1"] + 1,
                   b["y2"] - b["y1"] + 1, (125, 111, 171, 184), .91, **changes)

    def locator_change(self, mutation):
        mutation(self.locator["parsed"]["frames"][0]["objects"][0])
        text = json.dumps(self.locator["parsed"])
        self.locator["text"] = self.locator["response"]["output_text"] = text

    def test_native_square_is_centered_inclusive_and_integer(self):
        self.assertEqual({"x1": 132, "y1": 80, "x2": 427, "y2": 375},
                         module.crop_geometry(self.target["objects"][0]["box"], 640, 480))

    def test_square_clamps_at_source_boundary_without_resize(self):
        self.assertEqual({"x1": 0, "y1": 0, "x2": 255, "y2": 255},
                         module.crop_geometry({"x1": 0, "y1": 0, "x2": 10, "y2": 20}, 640, 480))

    def test_oversize_square_uses_whole_source_without_distortion(self):
        self.assertEqual({"x1": 0, "y1": 0, "x2": 639, "y2": 479},
                         module.crop_geometry({"x1": 100, "y1": 100, "x2": 300, "y2": 300}, 640, 480))

    def test_small_image_uses_whole_source_without_upsampling(self):
        self.assertEqual({"x1": 0, "y1": 0, "x2": 99, "y2": 79},
                         module.crop_geometry({"x1": 10, "y1": 10, "x2": 30, "y2": 30}, 100, 80))

    def test_invalid_box_and_boolean_frame_are_rejected(self):
        with self.assertRaises(ValidationError):
            module.crop_geometry({"x1": -1, "y1": 0, "x2": 20, "y2": 20}, 640, 480)
        with self.assertRaises(ValidationError):
            module.prepare_frame(self.envelope, True, self.target, self.locator,
                                 self.root / "artifacts/local/invalid", root=self.root)

    def test_low_original_score_can_locate_crop_without_becoming_evidence(self):
        prepared = self.prepare()
        self.assertTrue(prepared["details"]["crop_available"])
        self.assertEqual(.73, prepared["cached_target_result"]["objects"][0]["confidence"])
        self.assertEqual("original_JPEG_decode_to_RGB24_then_integer_crop_to_lossless_PNG_no_resize", prepared["transform"])
        crop_commands = [a for a, _ in self.commands if "-vf" in a]
        self.assertTrue(all("format=rgb24,crop=" in a[a.index("-vf") + 1] for a in crop_commands))
        self.assertTrue(all("scale" not in a[a.index("-vf") + 1] for a in crop_commands))

    def test_missing_proposal_has_no_crop_and_no_projection(self):
        target = deepcopy(self.target)
        target.update(text="", objects=[])
        target["response"]["output_text"] = ""
        prepared = self.prepare(target=target)
        self.assertFalse(prepared["details"]["crop_available"])
        self.assertIsNone(prepared["crop_file"])
        self.assertEqual([], self.commands)
        with self.assertRaises(ValidationError):
            module.project_result(prepared, {}, root=self.root)

    def test_multiple_proposals_cannot_select_preferred_crop(self):
        target = deepcopy(self.target)
        text = target["text"] + target["text"].replace("<0f>1", "<0f>2")
        target.update(text=text, objects=decoded(parse_sam_boxes(text)))
        target["response"]["output_text"] = text
        prepared = self.prepare(target=target)
        self.assertFalse(prepared["details"]["crop_available"])
        self.assertEqual([], self.commands)

    def test_source_and_raw_proposal_copies_are_bound(self):
        for target in (dict(self.target, source={"sha256": "0" * 64}), deepcopy(self.target)):
            if target.get("source", {}).get("sha256") != "0" * 64:
                target["objects"][0]["confidence"] = .99
            with self.assertRaises(ValidationError):
                self.prepare(target=target)

    def test_projection_is_exact_translation_preserving_raw_crop_response(self):
        prepared = self.prepare()
        response = self.result(prepared)
        before = deepcopy(response)
        projected = module.project_result(prepared, response, root=self.root)
        self.assertEqual("eligible_geometry", projected["status"])
        self.assertEqual(before, response)
        self.assertEqual(before, projected["actual_crop_response"])
        record, = projected["derived_full_source_geometry"]
        self.assertEqual({"x1": 257, "y1": 191, "x2": 303, "y2": 264, "width": 640, "height": 480}, record["box"])
        self.assertEqual(.91, record["confidence"])
        self.assertEqual(before["objects"][0]["mask"], record["mask"])
        self.assertEqual(296, projected["actual_crop_response"]["objects"][0]["box"]["width"])
        self.assertNotIn("response", projected)
        self.assertNotIn("source", projected)
        self.assertEqual(1, projected["physical_source_view_count"])
        self.assertFalse(projected["placement_accepted"])
        self.assertTrue(projected["full_scene_semantic_assessment_required"])

    def test_cached_occlusion_ambiguity_and_duplicates_cannot_be_cleared(self):
        for name, change in (("occluded", lambda o: o.update(fully_visible=False)),
                             ("ambiguous", lambda o: o.update(identity_ambiguous=True)),
                             ("multiple", lambda o: o["candidates"].extend(deepcopy(o["candidates"]))),
                             ("missing", lambda o: o.update(candidates=[]))):
            saved = deepcopy(self.locator)
            self.locator_change(change)
            prepared = self.prepare(name)
            self.assertTrue(prepared["details"]["crop_available"])
            projected = module.project_result(prepared, self.result(prepared), root=self.root)
            self.assertEqual("withheld", projected["status"])
            self.assertTrue(any(reason.startswith("cached_") for reason in projected["withheld_reasons"]))
            self.locator = saved

    def test_low_new_score_never_upgraded(self):
        prepared = self.prepare()
        response = sam(prepared["crop_sha256"], 296, 296, (125, 111, 171, 184), .79)
        projected = module.project_result(prepared, response, root=self.root)
        self.assertEqual("withheld", projected["status"])
        self.assertIn("crop_sam_score_below_0_8", projected["withheld_reasons"])
        self.assertEqual(.79, projected["derived_full_source_geometry"][0]["confidence"])

    def test_missing_new_score_never_invented(self):
        prepared = self.prepare()
        response = sam(prepared["crop_sha256"], 296, 296, (125, 111, 171, 184), None)
        projected = module.project_result(prepared, response, root=self.root)
        self.assertIn("crop_sam_score_unavailable", projected["withheld_reasons"])
        self.assertIsNone(projected["derived_full_source_geometry"][0]["confidence"])

    def test_artificial_crop_edge_touch_withholds_even_high_score(self):
        prepared = self.prepare()
        response = sam(prepared["crop_sha256"], 296, 296, (0, 111, 171, 184), .99)
        projected = module.project_result(prepared, response, root=self.root)
        self.assertIn("artificial_crop_edge_touched", projected["withheld_reasons"])
        self.assertEqual(["left"], projected["crop_boundary_receipts"][0]["artificial_edges"])

    def test_original_frame_edge_is_reported_and_still_requires_visibility(self):
        target = sam(sha(self.paths[0]), box=(0, 0, 180, 180))
        prepared = self.prepare(target=target)
        response = sam(prepared["crop_sha256"], 640, 480, (0, 111, 171, 184), .91)
        projected = module.project_result(prepared, response, root=self.root)
        self.assertTrue(prepared["details"]["full_source_crop"])
        self.assertEqual(["left"], projected["crop_boundary_receipts"][0]["touched_edges"])
        self.assertEqual([], projected["crop_boundary_receipts"][0]["artificial_edges"])
        self.assertTrue(projected["full_scene_semantic_assessment_required"])
        self.assertFalse(projected["placement_accepted"])

    def test_zero_and_multiple_crop_results_never_pick_a_winner(self):
        prepared = self.prepare()
        for text in ("", self.result(prepared)["text"] + self.result(prepared)["text"].replace("<0f>1", "<0f>2")):
            response = self.result(prepared)
            response.update(text=text, objects=decoded(parse_sam_boxes(text)))
            response["response"]["output_text"] = text
            projected = module.project_result(prepared, response, root=self.root)
            self.assertIn("crop_sam_candidate_count_not_one", projected["withheld_reasons"])

    def test_crop_response_requires_exact_crop_hash_dimensions_and_completed_model(self):
        prepared = self.prepare()
        responses = [self.result(prepared) for _ in range(4)]
        responses[0]["source"]["sha256"] = prepared["details"]["source_sha256"]
        responses[1] = sam(prepared["crop_sha256"], 640, 480)
        responses[2]["response"]["status"] = "incomplete"
        responses[3]["response"]["model"] = SPARK_MODEL
        for response in responses:
            with self.assertRaises(ValidationError):
                module.project_result(prepared, response, root=self.root)

    def test_crop_bytes_cannot_be_rehashed_to_replace_source_derivation(self):
        prepared = self.prepare()
        crop = Path(prepared["crop_file"])
        crop.write_bytes(crop.read_bytes() + b"replacement")
        prepared["crop_sha256"] = sha(crop)
        prepared["crop_bytes"] = crop.stat().st_size
        prepared["prepared_sha256"] = module._hash({k: v for k, v in prepared.items() if k != "prepared_sha256"})
        with self.assertRaises(ValidationError):
            module.validate_prepared(prepared, root=self.root)

    def test_rehashed_rectangle_change_fails_fixed_crop_rule(self):
        prepared = self.prepare()
        prepared["details"]["crop_rectangle"]["x1"] += 1
        prepared["prepared_sha256"] = module._hash({k: v for k, v in prepared.items() if k != "prepared_sha256"})
        with self.assertRaises(ValidationError):
            module.validate_prepared(prepared, root=self.root)

    def test_changed_code_hash_extractor_and_source_are_rejected(self):
        prepared = self.prepare()
        for key, value in (("implementation_sha256", {}), ("extractor_version", "changed")):
            altered = deepcopy(prepared)
            altered[key] = value
            altered["prepared_sha256"] = module._hash({k: v for k, v in altered.items() if k != "prepared_sha256"})
            with self.assertRaises(ValidationError):
                module.validate_prepared(altered, root=self.root)
        self.paths[0].write_bytes(self.paths[0].read_bytes() + b"changed")
        with self.assertRaises(ValidationError):
            module.validate_prepared(prepared, root=self.root)

    def test_projected_score_or_box_cannot_be_changed_and_rehashed(self):
        prepared = self.prepare()
        projected = module.project_result(prepared, self.result(prepared), root=self.root)
        projected["derived_full_source_geometry"][0]["confidence"] = .99
        projected["projection_sha256"] = module._hash({k: v for k, v in projected.items() if k != "projection_sha256"})
        with self.assertRaises(ValidationError):
            module.validate_projection(prepared, projected, root=self.root)

    def test_existing_output_and_symlink_crop_are_rejected(self):
        prepared = self.prepare()
        with self.assertRaises(ValidationError):
            self.prepare()
        crop = Path(prepared["crop_file"])
        saved = crop.with_suffix(".saved")
        crop.rename(saved)
        crop.symlink_to(saved)
        with self.assertRaises(ValidationError):
            module.validate_prepared(prepared, root=self.root)

    def test_failed_bounded_media_command_cannot_produce_preparation(self):
        with patch("proofloop.crop_evidence._crop_png", side_effect=ValidationError("media failed")):
            with self.assertRaises(ValidationError):
                self.prepare()


class CropEvidenceRealCodecTests(unittest.TestCase):
    def test_official_decode_and_real_ffmpeg_preserve_native_crop_and_exact_projection(self):
        try:
            import meta_sam_parser  # noqa: F401
        except (ImportError, SyntaxError, TypeError):
            self.skipTest("Optional official SAM parser requires Python >=3.10")
        try:
            executable = module.semantic._extractor_path()
        except ValidationError:
            self.skipTest("Optional FFmpeg is required for the actual codec integration")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = contract()
            source["cases"] = source["cases"][:1]
            (root / "inputs/local/authored").mkdir(parents=True)
            (root / "inputs/local/authored/video.mp4").write_text("authored-video")
            paths = []
            for i, frame in enumerate(source["cases"][0]["frames"]):
                path = root / frame["file"]
                subprocess.run([executable, "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
                    "testsrc2=size=640x480:rate=1", "-vf", f"hue=h={30 * i}", "-frames:v", "1", "-threads", "1",
                    "-map_metadata", "-1", str(path)], check=True, capture_output=True, timeout=15)
                frame["sha256"] = sha(path)
                paths.append(path)
            envelope = benchmark.approve_manifest(source, benchmark.manifest_hash(source), confirm_unique_targets=True)
            parsed = {"coordinate_system": "normalized_0_1000", "frames": [
                {"frame_index": i, "width": 640, "height": 480, "objects": [{"object_id": "glass",
                 "identity_ambiguous": False, "fully_visible": True, "boundary_uncertain": False,
                 "candidates": [{"x1": 400, "x2": 480, "y1": 390, "y2": 550}]}]} for i in range(2)]}
            text = json.dumps(parsed)
            locator = {"source": None, "sources": [{"sha256": sha(path)} for path in paths],
                "response": {"status": "completed", "model": SPARK_MODEL, "output_text": text},
                "text": text, "parsed": parsed}
            target = sam(sha(paths[0]), decoder=decode_sam_masks)
            prepared = module.prepare_frame(envelope, 0, target, locator, root / "artifacts/local/prepared", root=root)
            self.assertEqual((296, 296), module._png_dimensions(Path(prepared["crop_file"])))
            response = sam(prepared["crop_sha256"], 296, 296, (125, 111, 171, 184), .91, decoder=decode_sam_masks)
            projected = module.project_result(prepared, response, root=root)
            module.validate_projection(prepared, projected, root=root)
            self.assertEqual(target["objects"][0]["box"], projected["derived_full_source_geometry"][0]["box"])
            decoded_crop = subprocess.run([executable, "-hide_banner", "-loglevel", "error", "-i", prepared["crop_file"],
                "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"], check=True, capture_output=True, timeout=15).stdout
            full = subprocess.run([executable, "-hide_banner", "-loglevel", "error", "-i", str(paths[0]),
                "-vf", "format=rgb24", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"],
                check=True, capture_output=True, timeout=15).stdout
            rectangle = prepared["details"]["crop_rectangle"]
            expected = b"".join(full[(y * 640 + rectangle["x1"]) * 3:(y * 640 + rectangle["x2"] + 1) * 3]
                                for y in range(rectangle["y1"], rectangle["y2"] + 1))
            self.assertEqual(expected, decoded_crop)


if __name__ == "__main__":
    unittest.main()
