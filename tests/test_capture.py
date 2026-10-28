"""Local decoder simulations only; these tests contain no camera or Meta inference."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from proofloop.capture import (
    CaptureError, _extract_frame, _jpeg_dimensions, _tool, default_plan, extract_capture,
    main, probe_media, validate_plan,
)


def jpeg(width=640, height=360, extra=b""):
    # Minimal SOF header for metadata-parser tests, not a displayable camera frame.
    return b"\xff\xd8\xff\xc0" + struct.pack(">H B H H B", 8, 8, height, width, 0) + b"\xff\xd9" + extra


def probe_response(rotation=None, duration="48.0", width=1920, height=1080):
    stream = {"width": width, "height": height, "duration": duration,
              "avg_frame_rate": "30000/1001", "sample_aspect_ratio": "1:1"}
    if rotation is not None:
        stream["side_data_list"] = [{"rotation": rotation}]
    return {"streams": [stream], "format": {"duration": duration}}


def short_plan():
    plan = default_plan()
    plan["evidence_origin"] = "synthetic_test_fixture"
    plan["stages"] = [{"id": "initial_correct", "declared_label": "test annotation only",
                       "start_seconds": 0, "end_seconds": 7, "sample_seconds": [1, 3]}]
    return plan


def assistant_plan():
    plan = short_plan()
    plan["ground_truth_origin"] = "user_task_with_assistant_reviewed_timestamps"
    plan["annotation_review"] = {
        "review_origin": "assistant_under_user_authorization",
        "authorization_reference": "Direct user instruction: check the uploaded clip and proceed with next steps.",
        "human_reviewed_offsets": False,
    }
    return plan


class CapturePlanTests(unittest.TestCase):
    def test_locked_schedule_has_four_stages_and_two_frames_each(self):
        plan = default_plan()
        stages = validate_plan(plan, 48)
        self.assertEqual(["initial_correct", "wrong_region", "occluded", "corrected"],
                         [stage["id"] for stage in stages])
        self.assertEqual([[3, 5], [16, 18], [29, 31], [42, 44]],
                         [stage["sample_seconds"] for stage in stages])
        self.assertFalse(plan["offsets_confirmed"])
        self.assertEqual("fork left; spoon right; hands out; fully uncovered", stages[0]["declared_label"])
        self.assertEqual("fork right; spoon right; hands out; fully uncovered", stages[1]["declared_label"])
        self.assertEqual("human_declared_not_model_inferred", plan["ground_truth_origin"])

    def test_default_plan_returns_an_independent_copy(self):
        altered = default_plan()
        altered["stages"][0]["sample_seconds"][0] = 999
        self.assertEqual(3, default_plan()["stages"][0]["sample_seconds"][0])

    def test_offsets_require_distinct_original_times_and_one_second_gap(self):
        for offsets in ([1, 1], [1, 1.9], [3, 1]):
            with self.subTest(offsets=offsets):
                plan = short_plan()
                plan["stages"][0]["sample_seconds"] = offsets
                with self.assertRaises(CaptureError):
                    validate_plan(plan, 8)

    def test_rejects_offsets_outside_declared_window_or_video(self):
        for offsets in ([-1, 3], [1, 7], [1, 9]):
            with self.subTest(offsets=offsets):
                plan = short_plan()
                plan["stages"][0]["sample_seconds"] = offsets
                with self.assertRaises(CaptureError):
                    validate_plan(plan, 8)
        with self.assertRaises(CaptureError):
            validate_plan(default_plan(), 46)

    def test_rejects_nonfinite_and_boolean_offsets(self):
        for value in (float("nan"), float("inf"), True, None):
            plan = short_plan()
            plan["stages"][0]["sample_seconds"] = [value, 3]
            with self.subTest(value=value), self.assertRaises(CaptureError):
                validate_plan(plan, 8)

    def test_stage_identifiers_and_windows_are_unambiguous(self):
        plan = short_plan()
        plan["stages"].append(deepcopy(plan["stages"][0]))
        with self.assertRaisesRegex(CaptureError, "unique"):
            validate_plan(plan, 8)
        plan["stages"][1]["id"] = "second"
        with self.assertRaisesRegex(CaptureError, "nonoverlapping"):
            validate_plan(plan, 8)
        plan["stages"][0]["id"] = "../escape"
        with self.assertRaisesRegex(CaptureError, "identifiers"):
            validate_plan(plan, 8)

    def test_no_implicit_provenance_or_automatic_ground_truth(self):
        for field, value in (("evidence_origin", "live_verified"),
                             ("ground_truth_origin", "inferred_from_timeline")):
            plan = short_plan()
            plan[field] = value
            with self.assertRaises(CaptureError):
                validate_plan(plan, 8)

    def test_assistant_timestamp_review_has_explicit_nonhuman_provenance(self):
        plan = assistant_plan()
        self.assertEqual([1, 3], validate_plan(plan, 8)[0]["sample_seconds"])
        self.assertIs(False, plan["annotation_review"]["human_reviewed_offsets"])

    def test_assistant_timestamp_review_rejects_missing_or_contradictory_authority(self):
        malformed = []
        for field in ("review_origin", "authorization_reference", "human_reviewed_offsets"):
            plan = assistant_plan()
            del plan["annotation_review"][field]
            malformed.append(plan)
        for field, value in (("review_origin", "local_operator"),
                             ("human_reviewed_offsets", True),
                             ("authorization_reference", " \n "),
                             ("authorization_reference", None),
                             ("authorization_reference", "x" * 2001)):
            plan = assistant_plan()
            plan["annotation_review"][field] = value
            malformed.append(plan)
        plan = assistant_plan()
        plan["ground_truth_origin"] = "human_declared_not_model_inferred"
        malformed.append(plan)
        for index, plan in enumerate(malformed):
            with self.subTest(case=index), self.assertRaises(CaptureError):
                validate_plan(plan, 8)


class CaptureDecoderTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.phone = self.root / "phone"
        self.phone.mkdir()
        self.source = self.phone / "01-correction-loop.MOV"
        self.source.write_bytes(b"explicit synthetic video stand-in; not decodable")
        self.output = self.phone / "run-01"
        self.commands = []
        self.probe = probe_response()
        self.timestamp_shift = 0.033
        self.duplicate_timestamps = False
        self.no_timestamp = False
        self.phone_patch = patch("proofloop.capture.PHONE_ROOT", self.phone)
        self.tool_patch = patch("proofloop.capture.shutil.which", side_effect=lambda name: name)
        self.run_patch = patch("proofloop.capture.subprocess.run", side_effect=self.fake_run)
        self.phone_patch.start()
        self.tool_patch.start()
        self.run_patch.start()

    def tearDown(self):
        self.run_patch.stop()
        self.tool_patch.stop()
        self.phone_patch.stop()
        self.temporary.cleanup()

    def fake_run(self, command, **kwargs):
        self.commands.append((command, kwargs))
        if command[0] == "ffprobe":
            return subprocess.CompletedProcess(command, 0, stdout=json.dumps(self.probe), stderr="")
        # Select offset is part of the ffmpeg filter, not an assumed output PTS.
        filters = command[command.index("-vf") + 1]
        requested = float(filters.split("select=gte(t\\,")[1].split(")")[0])
        actual = 1.033 if self.duplicate_timestamps else requested + self.timestamp_shift
        target = Path(command[-1])
        target.write_bytes(jpeg(extra=target.name.encode()))
        stderr = "no timestamp" if self.no_timestamp else f"[Parsed_showinfo_2] n: 0 pts: 1 pts_time:{actual} fmt:yuv420p"
        return subprocess.CompletedProcess(command, 0, stdout="", stderr=stderr)

    def test_preflight_handles_phone_display_rotation_and_hashes_original(self):
        self.probe = probe_response(rotation=-90, width=1080, height=1920)
        info = probe_media(self.source)
        self.assertEqual((1920, 1080), (info["display_width"], info["display_height"]))
        self.assertEqual(270, info["display_rotation_degrees"])
        self.assertEqual(hashlib.sha256(self.source.read_bytes()).hexdigest(), info["sha256"])
        self.assertAlmostEqual(29.97002997, info["average_frame_rate"])
        self.assertNotIn(str(self.root), json.dumps(info))

    def test_extraction_keeps_actual_pts_relative_paths_and_declared_labels(self):
        manifest = extract_capture(self.source, short_plan(), self.output, confirmed_offsets=True)
        stage = manifest["stages"][0]
        frame = stage["frames"][0]
        self.assertEqual(1.033, frame["source_timestamp_seconds"])
        self.assertEqual(1033, frame["source_timestamp_ms"])
        self.assertEqual(1, frame["requested_seconds"])
        self.assertEqual(640, frame["width"])
        self.assertEqual("../01-correction-loop.MOV", manifest["source"]["file"])
        self.assertEqual("initial_correct-01.jpg", frame["file"])
        self.assertEqual("test annotation only", stage["declared_label"])
        self.assertEqual("synthetic_test_fixture", manifest["evidence_origin"])
        self.assertFalse(manifest["inference_performed"])
        self.assertFalse(manifest["physical_repair_verified"])
        self.assertEqual("human_declared_not_model_inferred", manifest["ground_truth_origin"])
        self.assertNotIn("annotation_review", manifest)
        self.assertEqual("extracted_pending_visual_review", manifest["status"])
        self.assertTrue(stage["distinct_frame_hashes"])
        self.assertEqual(0o600, (self.output / "manifest.json").stat().st_mode & 0o777)
        self.assertNotIn(str(self.root), json.dumps(manifest))
        for command, kwargs in self.commands:
            self.assertNotIn("shell", kwargs)
            if command[0] == "ffmpeg":
                self.assertIn("setpts=PTS-STARTPTS", command[command.index("-vf") + 1])
                self.assertIn("-map_metadata", command)
                self.assertIn("-n", command)

    def test_extraction_preserves_assistant_annotation_review_without_human_claim(self):
        plan = assistant_plan()
        expected_review = deepcopy(plan["annotation_review"])
        manifest = extract_capture(self.source, plan, self.output, confirmed_offsets=True)
        plan["annotation_review"]["authorization_reference"] = "changed later"
        self.assertEqual("user_task_with_assistant_reviewed_timestamps", manifest["ground_truth_origin"])
        self.assertEqual(expected_review, manifest["annotation_review"])
        self.assertIs(False, manifest["annotation_review"]["human_reviewed_offsets"])
        stored = json.loads((self.output / "manifest.json").read_text())
        self.assertEqual(expected_review, stored["annotation_review"])

    def test_unattributed_assistant_timestamp_review_stops_before_output_creation(self):
        plan = assistant_plan()
        del plan["annotation_review"]
        with self.assertRaises(CaptureError):
            extract_capture(self.source, plan, self.output, confirmed_offsets=True)
        self.assertFalse(self.output.exists())

    def test_confirmation_is_required_before_even_decoding(self):
        with self.assertRaisesRegex(CaptureError, "confirm-offsets"):
            extract_capture(self.source, short_plan(), self.output)
        self.assertEqual([], self.commands)
        self.assertFalse(self.output.exists())

    def test_unsafe_paths_and_overwrite_are_rejected(self):
        with self.assertRaisesRegex(CaptureError, "ignored by Git"):
            probe_media(self.root / "outside.MOV")
        with self.assertRaisesRegex(CaptureError, "ignored by Git"):
            extract_capture(self.source, short_plan(), self.root / "public", confirmed_offsets=True)
        self.output.mkdir()
        with self.assertRaisesRegex(CaptureError, "never overwritten"):
            extract_capture(self.source, short_plan(), self.output, confirmed_offsets=True)

    def test_missing_decoder_is_clear_and_does_not_create_output(self):
        with patch("proofloop.capture.shutil.which", return_value=None), \
                patch("proofloop.capture.sys.executable", str(self.root / "missing-python")):
            with self.assertRaisesRegex(CaptureError, "ffprobe is unavailable"):
                extract_capture(self.source, short_plan(), self.output, confirmed_offsets=True)
        self.assertFalse(self.output.exists())

    def test_tool_can_use_verified_project_runtime_without_global_path_changes(self):
        runtime = self.root / "runtime"
        runtime.mkdir()
        tool = runtime / "ffprobe"
        tool.write_text("local fixture executable; never run")
        tool.chmod(0o700)
        with patch("proofloop.capture.shutil.which", return_value=None), \
                patch("proofloop.capture.sys.executable", str(runtime / "python")):
            self.assertEqual(str(tool), _tool("ffprobe"))

    def test_actual_pts_outside_window_does_not_create_completed_manifest(self):
        self.timestamp_shift = 7
        with self.assertRaisesRegex(CaptureError, "outside"):
            extract_capture(self.source, short_plan(), self.output, confirmed_offsets=True)
        self.assertFalse((self.output / "manifest.json").exists())
        partial = json.loads((self.output / "manifest.incomplete.json").read_text())
        self.assertEqual("incomplete", partial["status"])

    def test_actual_pts_are_required_not_guessed_from_offsets(self):
        self.no_timestamp = True
        with self.assertRaisesRegex(CaptureError, "actual decoded frame timestamp"):
            extract_capture(self.source, short_plan(), self.output, confirmed_offsets=True)
        self.assertFalse((self.output / "manifest.json").exists())

    def test_repeated_actual_timestamp_is_rejected(self):
        self.duplicate_timestamps = True
        # The second returned frame is before its requested offset: either guard rejects it.
        with self.assertRaises(CaptureError):
            extract_capture(self.source, short_plan(), self.output, confirmed_offsets=True)
        self.assertFalse((self.output / "manifest.json").exists())

    def test_duplicate_content_is_reported_without_claiming_two_independent_views(self):
        with patch("proofloop.capture._extract_frame") as extract:
            extract.side_effect = [
                {"file": f"frame-{index}.jpg", "sha256": "0" * 64,
                 "source_timestamp_seconds": index, "source_timestamp_ms": index * 1000}
                for index in (1, 3)
            ]
            manifest = extract_capture(self.source, short_plan(), self.output, confirmed_offsets=True)
        self.assertFalse(manifest["stages"][0]["distinct_frame_hashes"])

    def test_source_changed_during_extraction_is_rejected(self):
        original = self.fake_run

        def changed(command, **kwargs):
            result = original(command, **kwargs)
            if command[0] == "ffmpeg":
                self.source.write_bytes(b"changed original")
            return result

        with patch("proofloop.capture.subprocess.run", side_effect=changed):
            with self.assertRaisesRegex(CaptureError, "changed during extraction"):
                extract_capture(self.source, short_plan(), self.output, confirmed_offsets=True)
        self.assertFalse((self.output / "manifest.json").exists())

    def test_preflight_rejects_missing_stream_unbounded_video_and_nonrightangle(self):
        for response in ({"streams": []}, probe_response(duration="nan"),
                         probe_response(duration="601"), probe_response(rotation=30)):
            self.probe = response
            with self.subTest(response=response), self.assertRaises(CaptureError):
                probe_media(self.source)

    def test_jpeg_dimensions_reject_oversize_and_malformed_output(self):
        self.assertEqual((640, 360), _jpeg_dimensions(jpeg()))
        for data in (b"not jpeg", b"\xff\xd8\xff\xc0\x00\xff", jpeg(width=2000)):
            with self.subTest(data=data[:10]), self.assertRaises(CaptureError):
                _jpeg_dimensions(data)


if __name__ == "__main__":
    unittest.main()
