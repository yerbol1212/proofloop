"""Authored SAM video transport/geometry fixtures; no paid inference."""
import base64
from copy import deepcopy
import hashlib
from io import BytesIO
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from urllib.error import HTTPError, URLError

from proofloop.core import ValidationError
from proofloop.meta import MetaAPIError, MetaClient
from proofloop.video_tracking import (MAX_CLIP_FRAMES, TRACKING_VERSION, _hash, _probe,
    prepare_clip, public_tracking_summary, segment_video, verify_prepared, video_accounting)


def probe_metadata(frames=4):
    return {"width": 64, "height": 48, "codec": "h264", "container": "mp4_or_mov",
        "duration_ms": frames * 100, "frame_count": frames, "time_base": "1/10", "first_pts": 0,
        "frames": [{"frame_index": index, "pts": index, "timestamp_ms": index * 100} for index in range(frames)]}


def raw_probe_metadata():
    return {"streams": [{"codec_type": "video", "codec_name": "h264", "width": 64, "height": 48,
        "time_base": "1/10", "duration": "0.4", "nb_frames": "4"}],
        "format": {"format_name": "mov,mp4,m4a,3gp,3g2,mj2", "duration": "0.4"},
        "frames": [{"media_type": "video", "width": 64, "height": 48, "pts": 20 + index,
            "best_effort_timestamp": 20 + index} for index in range(4)]}


def record(frame=0, track_id=1, *, width=64, height=48, confidence="0.9", mask_payload="abc", x=1):
    c = ";c=" + confidence if confidence is not None else ""
    return (f"<{frame}f>{track_id}<|box;x1={x};y1=2;x2={x + 4};y2=6;w={width};h={height}{c}|>"
        f"<|mask;x=0;y=0{c};data=5,5,!{mask_payload}|>")


def lane(text, *, status="completed", usage=None):
    return {"status": status, "output": [{"type": "message", "content": [{"type": "output_text", "text": text}]}],
        "usage": {"video_frames_processed": 50} if usage is None else usage}


class VideoTrackingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        # Deliberately authored byte+probe fixture, not an actual encoded video
        # or a physical scene. The separate real-probe test exercises decoding.
        self.path = self.root / "authored-transport.mp4"
        self.path.write_bytes(b"\x00\x00\x00\x18ftypisom-authored-not-real-video")
        self.probe = patch("proofloop.video_tracking._probe", return_value=(probe_metadata(), "ffprobe version authored-fixture"))
        self.probe.start()
        self.addCleanup(self.probe.stop)
        self.seeds = [{"object_id": "fork", "track_id": 1, "frame_index": 0,
            "box": {"x1": 1, "y1": 2, "x2": 5, "y2": 6}}]
        self.prepared = prepare_clip(self.path, self.seeds, review_note="Authored transport fixture; no physical inference.")
        self.sent = []

    def client(self, value=None, *, error=None):
        def opener(request, **kwargs):
            self.sent.append((request, kwargs))
            if error is not None:
                raise error
            return BytesIO(json.dumps(value if value is not None else lane(record())).encode())
        return MetaClient("authored-key-not-real", max_requests=1, retries=0, opener=opener)

    def segment(self, text=None, *, usage=None, decode_masks=False, prepared=None):
        value = lane(record() if text is None else text, usage=usage)
        return segment_video(self.client(value), self.prepared if prepared is None else prepared, decode_masks=decode_masks)

    def test_preparation_binds_exact_clip_source_pts_and_reviewed_seed(self):
        source = self.prepared["source"]
        self.assertEqual(hashlib.sha256(self.path.read_bytes()).hexdigest(), source["sha256"])
        self.assertEqual((64, 48, 4, "1/10"), (source["width"], source["height"], source["frame_count"], source["time_base"]))
        seed = self.prepared["seeds"][0]
        self.assertEqual(source["sha256"], seed["source_sha256"])
        self.assertEqual((0, 0), (seed["pts"], seed["timestamp_ms"]))
        self.assertEqual("authored_reviewed_video_boxes", self.prepared["anchor_origin"])
        self.assertEqual("<0f>1<|box;x1=1;y1=2;x2=5;y2=6;w=64;h=48|>", self.prepared["prompt"])
        verify_prepared(self.prepared)
        self.assertEqual([], self.sent)

    def test_multiple_reviewed_target_and_distractor_seeds_share_one_prompt(self):
        seeds = self.seeds + [{"object_id": "spoon_distractor", "track_id": 9, "frame_index": 0,
            "box": {"x1": 20, "y1": 3, "x2": 30, "y2": 12}}]
        prepared = prepare_clip(self.path, seeds[::-1], review_note="Reviewed target and distractor authored fixture.")
        self.assertIn(",9<|box;x1=20;y1=3;x2=30;y2=12;w=64;h=48|>", prepared["prompt"])
        text = record(track_id=9, x=20) + "," + record(track_id=1).split("<0f>", 1)[1]
        report = self.segment(text, prepared=prepared)
        self.assertEqual([1, 9], [row["track_id"] for row in report["frames"][0]["candidates"]])
        self.assertEqual(["fork", "spoon_distractor"], [row["object_id"] for row in report["frames"][0]["candidates"]])
        self.assertEqual(1, len(self.sent))

    def test_one_nonstreaming_request_uses_video_metadata_and_exact_original_bytes(self):
        report = self.segment()
        request, kwargs = self.sent[0]
        wire = json.loads(request.data)
        self.assertEqual("POST", request.get_method())
        self.assertEqual("https://api.meta.ai/v1/responses", request.full_url)
        self.assertEqual("sam-3.1", wire["model"])
        self.assertFalse(wire["stream"])
        self.assertFalse(wire["store"])
        self.assertEqual({"include_confidence": "true", "mask_encoding": "one_bit", "propagation_direction": "both"}, wire["metadata"])
        media = wire["input"][0]["content"][1]
        self.assertEqual("input_video", media["type"])
        self.assertEqual(self.path.read_bytes(), base64.b64decode(media["video_url"].split(",", 1)[1]))
        self.assertEqual("completed", report["status"])
        self.assertEqual(1, report["cost"]["request_attempts"])
        self.assertEqual(0.01, report["cost"]["estimated_usd"])
        self.assertNotIn("sam_images", report["cost"])

    def test_unordered_sparse_source_frames_are_sorted_and_missing_is_unknown(self):
        report = self.segment(record(3) + "\n" + record(0))
        self.assertEqual([0, 3], [row["frame"] for row in report["objects"]])
        self.assertEqual([0, 1, 2, 3], [row["frame_index"] for row in report["frames"]])
        self.assertEqual([0, 100, 200, 300], [row["timestamp_ms"] for row in report["frames"]])
        for row in report["frames"][1:3]:
            self.assertEqual([], row["candidates"])
            self.assertEqual("unknown", row["observation_status"])
            self.assertEqual("no_track_record", row["unknown_reason"])
            self.assertEqual([1], row["missing_track_ids"])
            self.assertFalse(row["absence_verified"])
        self.assertFalse(report["frames"][0]["candidates"][0]["identity_independently_verified"])

    def test_empty_completed_response_is_unknown_on_every_requested_frame(self):
        report = self.segment("")
        self.assertEqual("completed", report["status"])
        self.assertTrue(all(frame["observation_status"] == "unknown" for frame in report["frames"]))
        self.assertTrue(all(not frame["absence_verified"] for frame in report["frames"]))

    def test_forward_and_backward_ranges_bind_source_index_not_line_order(self):
        seeds = deepcopy(self.seeds)
        seeds[0]["frame_index"] = 2
        for direction, expected in (("forward", [2, 3]), ("backward", [0, 1, 2])):
            prepared = prepare_clip(self.path, seeds, review_note="Reviewed later authored seed.", propagation_direction=direction)
            self.assertEqual(expected, prepared["requested_frame_indices"])
            self.assertIn("<2f>", prepared["prompt"])
            report = self.segment(record(2), prepared=prepared)
            self.assertEqual(expected, [frame["frame_index"] for frame in report["frames"] if frame["tracking_requested"]])
            self.assertTrue(any(frame["unknown_reason"] == "outside_requested_tracking_range" for frame in report["frames"]))

    def test_invalid_seed_shapes_ids_duplicates_and_bounds_fail_before_network(self):
        variants = [[], self.seeds * 2, [{**self.seeds[0], "track_id": True}], [{**self.seeds[0], "track_id": -1}],
            [{**self.seeds[0], "frame_index": 4}], [{**self.seeds[0], "frame_index": True}],
            [{**self.seeds[0], "object_id": "../private"}], [{**self.seeds[0], "box": {"x1": 1, "y1": 2, "x2": 64, "y2": 6}}],
            [{**self.seeds[0], "box": {"x1": 1.0, "y1": 2, "x2": 5, "y2": 6}}]]
        for seeds in variants:
            with self.assertRaises(ValidationError):
                prepare_clip(self.path, seeds, review_note="Authored invalid test.")
        with self.assertRaises(ValidationError):
            prepare_clip(self.path, self.seeds, review_note="Authored test.", propagation_direction="backward")
        with self.assertRaises(ValidationError):
            prepare_clip(self.path, self.seeds, review_note="private\ntext")
        self.assertEqual([], self.sent)

    def test_modified_source_or_prepared_contract_cannot_reach_dispatch(self):
        for mutation in (lambda value: value.update(prompt="fork"),
                lambda value: value["source"].update(width=63),
                lambda value: value["seeds"][0].update(pts=99),
                lambda value: value["author_review"].update(source_frame_and_boxes_reviewed=False)):
            value = deepcopy(self.prepared)
            mutation(value)
            with self.assertRaises(ValidationError):
                segment_video(self.client(), value, decode_masks=False)
            value["prepared_hash"] = _hash({key: item for key, item in value.items() if key != "prepared_hash"})
            with self.assertRaises(ValidationError):
                segment_video(self.client(), value, decode_masks=False)
        self.path.write_bytes(self.path.read_bytes() + b"changed")
        with self.assertRaises(ValidationError):
            segment_video(self.client(), self.prepared, decode_masks=False)
        self.assertEqual([], self.sent)

    def test_invalid_source_signatures_symlinks_and_missing_files_fail_before_probe(self):
        invalid = self.root / "bad.mp4"
        invalid.write_bytes(b"not-an-mp4")
        symlink = self.root / "link.mp4"
        symlink.symlink_to(self.path)
        for path in (invalid, symlink, self.root / "missing.mp4"):
            with self.assertRaises(ValidationError):
                prepare_clip(path, self.seeds, review_note="Authored invalid source.")

    def test_fresh_one_request_no_retry_client_is_required(self):
        for client in (MetaClient("authored", max_requests=2), MetaClient("authored", max_requests=1, retries=1)):
            with self.assertRaises(ValidationError):
                segment_video(client, self.prepared, decode_masks=False)
        client = self.client()
        self.segment()
        client.ledger.append({"status": "ok"})
        with self.assertRaises(ValidationError):
            segment_video(client, self.prepared, decode_masks=False)

    def test_parser_dependency_is_checked_before_paid_dispatch(self):
        with patch("proofloop.video_tracking._sam_decoder", side_effect=MetaAPIError("unavailable", code="sam_parser_unavailable")):
            with self.assertRaises(MetaAPIError):
                segment_video(self.client(), self.prepared, decode_masks=True)
        self.assertEqual([], self.sent)

    def test_duplicate_frames_and_object_ids_are_unscored_not_missing_predictions(self):
        for text in (record(0) + "\n" + record(0), record(0) + "," + record(0).split("<0f>")[1]):
            report = self.segment(text)
            self.assertEqual("unscored", report["status"])
            self.assertEqual([], report["frames"])
            self.assertEqual([], report["objects"])
            self.assertIsNotNone(report["response"])

    def test_unknown_track_out_of_range_frame_and_dimension_mismatch_are_rejected(self):
        for text, code in ((record(track_id=2), "sam_invalid_object_id"), (record(4), "sam_invalid_video_frame"),
                (record(width=63), "sam_invalid_video_geometry")):
            report = self.segment(text)
            self.assertEqual("unscored", report["status"])
            self.assertEqual(code, report["error"]["code"])
            self.assertEqual([], report["frames"])

    def test_mask_decoding_is_per_record_and_only_binary_support_is_claimed(self):
        calls = []
        def decode(records):
            calls.append(len(records))
            row = deepcopy(records[0])
            row["mask"].update(decoded=True, foreground_pixels=10, raster_sha256="a" * 64,
                raster_base64=base64.b64encode(bytes([1] * 10 + [0] * 15)).decode())
            return [row]
        with patch("proofloop.video_tracking._sam_decoder", return_value=None), patch("proofloop.video_tracking.decode_sam_masks", side_effect=decode):
            report = self.segment(record(0) + "\n" + record(1), decode_masks=True)
            summary = public_tracking_summary(report)
        self.assertEqual([1, 1, 1, 1], calls)
        self.assertTrue(summary["frames"][0]["candidates"][0]["mask_supported"])
        self.assertNotIn("raster_base64", json.dumps(summary))
        self.assertNotIn("encoded_payload", json.dumps(summary))
        self.assertFalse(summary["frames"][0]["candidates"][0]["identity_independently_verified"])

    def test_zero_foreground_and_skipped_masks_do_not_supply_support(self):
        def empty(records):
            row = deepcopy(records[0])
            row["mask"].update(decoded=True, foreground_pixels=0, raster_sha256="a" * 64)
            return [row]
        with patch("proofloop.video_tracking._sam_decoder", return_value=None), patch("proofloop.video_tracking.decode_sam_masks", side_effect=empty):
            report = self.segment(decode_masks=True)
        self.assertFalse(report["frames"][0]["candidates"][0]["mask_supported"])
        self.assertFalse(self.segment()["frames"][0]["candidates"][0]["mask_supported"])

    def test_mask_decode_failure_retains_response_and_usage_as_unscored(self):
        with patch("proofloop.video_tracking._sam_decoder", return_value=None), patch("proofloop.video_tracking.decode_sam_masks",
                side_effect=MetaAPIError("private payload", code="invalid_sam_mask")):
            report = self.segment(decode_masks=True)
        self.assertEqual("unscored", report["status"])
        self.assertEqual("invalid_sam_mask", report["error"]["code"])
        self.assertEqual({"video_frames_processed": 50}, report["usage"])
        self.assertEqual(0.01, report["cost"]["estimated_usd"])

    def test_unexpected_decoder_exception_does_not_lose_the_paid_response(self):
        with patch("proofloop.video_tracking._sam_decoder", return_value=None), patch("proofloop.video_tracking.decode_sam_masks",
                side_effect=IndexError("private optional decoder internals")):
            report = self.segment(decode_masks=True)
        self.assertEqual("unscored", report["status"])
        self.assertEqual("video_adapter_error", report["error"]["code"])
        self.assertIsNotNone(report["response"])
        self.assertEqual({"video_frames_processed": 50}, report["usage"])
        summary = public_tracking_summary(report)
        self.assertEqual([], summary["frames"])
        self.assertNotIn("private optional", json.dumps(summary))

    def test_normalized_box_uses_inclusive_source_edges_and_confidence_is_not_invented(self):
        report = self.segment(record(confidence=None))
        candidate = report["frames"][0]["candidates"][0]
        self.assertEqual([1 / 64, 2 / 48, 6 / 64, 7 / 48], candidate["box"])
        self.assertIsNone(candidate["confidence"])

    def test_http_and_ambiguous_failures_retain_private_diagnostics_without_retry(self):
        body = {"error": {"message": "/Users/private/API_KEY", "code": "invalid_video"}, "usage": {"video_frames_processed": 50}}
        error = HTTPError("https://api.meta.ai/v1/responses", 400, "private", {}, BytesIO(json.dumps(body).encode()))
        report = segment_video(self.client(error=error), self.prepared, decode_masks=False)
        self.assertEqual("unscored", report["status"])
        self.assertEqual(body, report["response"])
        self.assertEqual(1, len(self.sent))
        summary = public_tracking_summary(report)
        self.assertNotIn("API_KEY", json.dumps(summary))
        self.assertNotIn("/Users/private", json.dumps(summary))
        report = segment_video(self.client(error=URLError("private ambiguous network")), self.prepared, decode_masks=False)
        self.assertEqual("network_error", report["error"]["code"])
        self.assertIsNone(report["cost"]["estimated_usd"])
        self.assertTrue(report["cost"]["unreported_charge_possible"])

    def test_incomplete_response_retains_video_usage_and_is_not_empty_detection(self):
        report = segment_video(self.client(lane("", status="incomplete", usage={"video_frames_processed": 50})), self.prepared, decode_masks=False)
        self.assertEqual("unscored", report["status"])
        self.assertEqual("incomplete_response", report["error"]["code"])
        self.assertEqual([], report["frames"])
        self.assertEqual(50, report["cost"]["reported_billed_frames"])

    def test_video_cost_uses_reported_billed_frames_or_documented_50_minimum(self):
        for usage, expected, basis in (({}, 50, "requested_tracking_range_with_documented_minimum"),
                ({"video_frames_processed": 70}, 70, "reported_video_frames_processed")):
            cost = video_accounting(self.prepared, usage, completed=True)
            self.assertEqual(expected, cost["estimated_billable_frames"])
            self.assertEqual(expected * .0002, cost["estimated_usd"])
            self.assertEqual(basis, cost["estimate_basis"])
        cost = video_accounting(self.prepared, {"video_frames_processed": 4}, completed=True)
        self.assertTrue(cost["reported_video_frames_invalid"])
        self.assertEqual(50, cost["estimated_billable_frames"])
        self.assertIsNone(video_accounting(self.prepared, {}, completed=False)["estimated_usd"])

    def test_public_replay_rejects_modified_verdict_geometry_usage_or_cost(self):
        report = self.segment()
        for mutation in (lambda value: value["frames"][0].update(timestamp_ms=99),
                lambda value: value["frames"][0]["candidates"][0].update(object_id="other"),
                lambda value: value["objects"][0]["box"].update(x2=10),
                lambda value: value["usage"].update(video_frames_processed=60),
                lambda value: value["cost"].update(estimated_usd=0),
                lambda value: value["attempts"].append(deepcopy(value["attempts"][0])),
                lambda value: value["attempts"][0].update(model="muse-spark-1.3"),
                lambda value: value["request_binding"].update(source_sha256="a" * 64)):
            changed = deepcopy(report)
            mutation(changed)
            with self.assertRaises(ValidationError):
                public_tracking_summary(changed)

    def test_private_error_code_cannot_be_injected_into_public_failure(self):
        report = self.segment(record(track_id=2))
        report["error"]["code"] = "private_secret_code"
        with self.assertRaises(ValidationError):
            public_tracking_summary(report)

    def test_public_summary_omits_local_source_and_all_provider_text(self):
        report = self.segment()
        report["private_extra"] = {"file": str(self.path), "api_key": "authored-secret"}
        summary = public_tracking_summary(report)
        serialized = json.dumps(summary)
        self.assertEqual(TRACKING_VERSION, summary["method_version"])
        for value in (str(self.root), "authored-secret", "encoded_payload", "raster_base64", "output_text", "private_extra"):
            self.assertNotIn(value, serialized)
        self.assertFalse(summary["physical_action_authorized"])
        self.assertFalse(summary["changes_guided_v2_acceptance"])

    def test_official_decoder_public_one_bit_example_when_available(self):
        try:
            import meta_sam_parser  # noqa: F401
        except (ImportError, TypeError, SyntaxError):
            self.skipTest("Official SAM parser is optional and requires Python >=3.10")
        report = self.segment(record(mask_payload="!!!!(QO(0lu8?"), decode_masks=True)
        self.assertEqual("completed", report["status"])
        self.assertEqual(10, report["frames"][0]["candidates"][0]["mask"]["foreground_pixels"])


class VideoProbeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "clip.mp4"
        self.path.write_bytes(b"\x00\x00\x00\x18ftypisom-authored")

    def probe(self, metadata):
        def run(command, **kwargs):
            return SimpleNamespace(stdout=b"ffprobe version authored-fixture\n" if "-version" in command else json.dumps(metadata).encode())
        with patch("proofloop.video_tracking._probe_path", return_value=Path("/authored/ffprobe")), patch("proofloop.video_tracking.subprocess.run", side_effect=run):
            return _probe(self.path)[0]

    def test_exact_pts_mapping_is_relative_to_first_decoded_pts(self):
        source = self.probe(raw_probe_metadata())
        self.assertEqual(20, source["first_pts"])
        self.assertEqual([20, 21, 22, 23], [row["pts"] for row in source["frames"]])
        self.assertEqual([0, 100, 200, 300], [row["timestamp_ms"] for row in source["frames"]])

    def test_probe_rejects_unsupported_video_format_rotation_dimensions_and_timing(self):
        changes = [lambda value: value["streams"][0].update(codec_name="vp9"),
            lambda value: value["streams"].append(deepcopy(value["streams"][0])),
            lambda value: value["streams"][0].update(width=2000),
            lambda value: value["streams"][0].update(tags={"rotate": "90"}),
            lambda value: value["streams"][0].update(side_data_list=[{"rotation": -90}]),
            lambda value: value["format"].update(duration="12.1"),
            lambda value: value["format"].update(format_name="webm"),
            lambda value: value["streams"][0].update(time_base="0/1"),
            lambda value: value["frames"][1].update(best_effort_timestamp=20),
            lambda value: value["frames"][0].update(width=63),
            lambda value: value["streams"][0].update(nb_frames="99"),
            lambda value: value.update(frames=value["frames"] * (MAX_CLIP_FRAMES + 1))]
        for mutation in changes:
            metadata = raw_probe_metadata()
            mutation(metadata)
            with self.assertRaises(ValidationError):
                self.probe(metadata)

    def test_real_small_authored_mp4_probe_if_ffmpeg_available(self):
        root = Path(__file__).resolve().parents[1]
        ffmpeg = root / ".venv/bin/ffmpeg"
        ffprobe = root / ".venv/bin/ffprobe"
        ffmpeg = ffmpeg if ffmpeg.is_file() else shutil.which("ffmpeg")
        ffprobe = ffprobe if ffprobe.is_file() else shutil.which("ffprobe")
        if not ffmpeg or not ffprobe:
            self.skipTest("Optional ffmpeg/ffprobe tools are unavailable")
        target = Path(self.temp.name) / "real-authored.mp4"
        subprocess.run([str(ffmpeg), "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=64x48:rate=10:duration=0.6",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(target)],
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True, timeout=20)
        source, version = _probe(target, ffprobe)
        self.assertEqual((64, 48, 6), (source["width"], source["height"], source["frame_count"]))
        self.assertEqual([0, 100, 200, 300, 400, 500], [row["timestamp_ms"] for row in source["frames"]])
        self.assertTrue(version.startswith("ffprobe version "))


if __name__ == "__main__":
    unittest.main()
