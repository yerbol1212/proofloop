"""Offline authored provenance fixtures; these do not evaluate model quality."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from proofloop import distractor_media as media
from proofloop.core import ValidationError


def jpeg(width, height, suffix):
    return (b"\xff\xd8\xff\xc0" + struct.pack(">H", 11) + b"\x08" + struct.pack(">HH", height, width)
            + b"\x01\x01\x11\x00\xff\xd9" + suffix)


def digest(value):
    return hashlib.sha256(value).hexdigest()


class DistractorMediaTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.folder = self.root / media.REVIEW_DIRECTORY
        self.folder.mkdir(parents=True)
        self.inputs = self.root / media.SOURCE_DIRECTORY
        self.inputs.mkdir(parents=True)
        self.manifest = self.folder / "frozen-review-manifest.json"
        self.sidecar = self.folder / "frozen-review-manifest.sha256"
        self.pin = patch.object(media, "FROZEN_REVIEW_SHA256", "authored-test-fixture")
        self.pin.start()
        self.addCleanup(self.pin.stop)
        sources, cases = [], []
        self.decode_bytes = {}
        for index in range(2):
            name = f"source-{index}"
            video = self.inputs / f"{name}.mp4"
            video.write_bytes(f"authored-video-transport-{index}".encode())
            card = self.inputs / f"{name}-README.md"
            card.write_text("---\nlicense: mit\n---\nAuthored fixture\n")
            info = self.inputs / f"{name}-info.json"
            info.write_text(json.dumps({"features": {"observation.images.cam_high": {"shape": [480, 640, 3]}}}))
            sources.append({"source_id": name, "dataset": f"authored/{name}", "revision": str(index) * 40,
                            "remote_path": "videos/episode.mp4", "url": f"https://example.org/{name}.mp4",
                            "source_path": str(video), "sha256": digest(video.read_bytes()),
                            "bytes": video.stat().st_size, "width": 640, "height": 480, "duration_seconds": 8.0,
                            "license": "mit", "dataset_card_path": str(card),
                            "dataset_card_sha256": digest(card.read_bytes()),
                            "dataset_card_url": f"https://example.org/{name}/README.md",
                            "info_path": str(info), "info_sha256": digest(info.read_bytes()),
                            "selected_for_inference": True})
            images = []
            for frame_index in range(2):
                image = self.folder / f"{name}-{frame_index}.jpg"
                content = jpeg(640, 480, f"authored-frame-{index}-{frame_index}".encode())
                image.write_bytes(content)
                images.append(image)
                self.decode_bytes[(str(video), float(frame_index))] = content
            for candidate in range(3):
                case_id = f"{name}-{candidate}"
                expected = "target" if candidate == 0 else "excluded"
                frames = []
                for frame_index, image in enumerate(images):
                    crop = self.folder / f"{case_id}-crop-{frame_index}.jpg"
                    crop.write_bytes(jpeg(41, 31, f"authored-crop-{case_id}-{frame_index}".encode()))
                    frames.append({"frame_index": frame_index, "source_timestamp_ms": frame_index * 1000,
                                   "image_path": str(image), "sha256": digest(image.read_bytes()),
                                   "width": 640, "height": 480,
                                   "candidate_box": {"x1": 100, "y1": 100, "x2": 140, "y2": 130},
                                   "candidate_review_crop_path": str(crop),
                                   "candidate_review_crop_sha256": digest(crop.read_bytes()),
                                   "expected_candidate_relation": expected})
                cases.append({"case_id": case_id, "source_id": name,
                              "anchor_origin": "manually_authored_after_visual_review",
                              "automatic_localization": False,
                              "label_origin": "author_visual_review_before_any_semantic_inference",
                              "contract": {"object_id": "target", "positive_concept": "authored white object",
                                           "required_visual_attributes": [{"id": "solid", "description": "solid body"}],
                                           "excluded_concepts": [{"id": "tape", "description": "flat tape"}]},
                              "expected_identity": expected,
                              "expected_diagnostic_decision": "supported" if candidate == 0 else "unknown",
                              "frames": frames})
        sources.extend({"source_id": f"screened-out-{index}", "selected_for_inference": False} for index in range(3))
        self.review = {"schema_version": 1, "inference_started": False,
                       "evaluation_role": "authored_anchor_stress_test", "case_count": 6,
                       "unique_source_video_count": 2, "extractor": "ffmpeg authored fixture",
                       "sources": sources, "cases": cases}
        self.seal()

    def seal(self, *, approve_fixture=True):
        self.manifest.write_text(json.dumps(self.review, indent=2) + "\n")
        checksum = digest(self.manifest.read_bytes())
        self.sidecar.write_text(checksum + "\n")
        if approve_fixture:
            media.FROZEN_REVIEW_SHA256 = checksum

    def entries(self):
        entries = []
        for case in self.review["cases"]:
            prepared = {"contract": dict(deepcopy(case["contract"]), version="authored-fixture-v1"),
                        "frames": [{"source_file": frame["image_path"], **{
                            key: deepcopy(frame[key]) for key in ("frame_index", "sha256", "width", "height", "candidate_box")}}
                                   for frame in case["frames"]]}
            path = self.folder / f"{case['case_id']}-prepared.json"
            path.write_text(json.dumps(prepared))
            entries.append({"case_id": case["case_id"], "source_id": case["source_id"],
                            "expected_identity": case["expected_diagnostic_decision"],
                            "anchor_origin": "authored_candidate", "source_timestamps_ms": [0, 1000],
                            "prepared_file": str(path)})
        return entries

    def test_valid_review_preserves_manual_origin_and_returns_a_copy(self):
        result = media.verify_source_review(self.root)
        self.assertEqual(6, len(result["cases"]))
        self.assertFalse(result["cases"][0]["automatic_localization"])
        result["cases"].clear()
        self.assertEqual(6, len(media.verify_source_review(self.root)["cases"]))

    def test_editing_review_and_sidecar_does_not_change_pinned_author_labels(self):
        self.review["cases"][1]["expected_identity"] = "target"
        self.seal(approve_fixture=False)
        with self.assertRaisesRegex(ValidationError, "frozen author review"):
            media.verify_source_review(self.root)

    def test_modified_selected_video_is_rejected(self):
        Path(self.review["sources"][0]["source_path"]).write_bytes(b"replacement recording")
        with self.assertRaisesRegex(ValidationError, "changed"):
            media.verify_source_review(self.root)

    def test_missing_selected_video_is_rejected(self):
        Path(self.review["sources"][0]["source_path"]).unlink()
        with self.assertRaisesRegex(ValidationError, "missing"):
            media.verify_source_review(self.root)

    def test_modified_card_and_metadata_are_rejected(self):
        for key in ("dataset_card_path", "info_path"):
            with self.subTest(key=key):
                path = Path(self.review["sources"][0][key])
                original = path.read_bytes()
                path.write_bytes(original + b" altered")
                with self.assertRaisesRegex(ValidationError, "changed"):
                    media.verify_source_review(self.root)
                path.write_bytes(original)

    def test_card_license_must_match_reviewed_declaration(self):
        source = self.review["sources"][0]
        source["license"] = "apache-2.0"
        self.seal()
        with self.assertRaisesRegex(ValidationError, "license"):
            media.verify_source_review(self.root)

    def test_metadata_geometry_must_match_source_frames(self):
        source = self.review["sources"][0]
        info = Path(source["info_path"])
        info.write_text(json.dumps({"features": {"observation.images.cam_high": {"shape": [720, 1280, 3]}}}))
        source["info_sha256"] = digest(info.read_bytes())
        self.seal()
        with self.assertRaisesRegex(ValidationError, "geometry"):
            media.verify_source_review(self.root)

    def test_changed_source_frame_is_rejected(self):
        Path(self.review["cases"][0]["frames"][0]["image_path"]).write_bytes(jpeg(640, 480, b"another capture"))
        with self.assertRaisesRegex(ValidationError, "changed"):
            media.verify_source_review(self.root)

    def test_review_crop_must_have_exact_authored_rectangle_dimensions(self):
        frame = self.review["cases"][0]["frames"][0]
        crop = Path(frame["candidate_review_crop_path"])
        crop.write_bytes(jpeg(42, 31, b"one extra source pixel"))
        frame["candidate_review_crop_sha256"] = digest(crop.read_bytes())
        self.seal()
        with self.assertRaisesRegex(ValidationError, "authored rectangle"):
            media.verify_source_review(self.root)

    def test_same_source_hash_cannot_satisfy_two_distinct_reviewed_times(self):
        frames = self.review["cases"][0]["frames"]
        frames[1]["image_path"] = frames[0]["image_path"]
        frames[1]["sha256"] = frames[0]["sha256"]
        self.seal()
        with self.assertRaisesRegex(ValidationError, "distinct source"):
            media.verify_source_review(self.root)

    def test_symlink_cannot_escape_fixed_video_directory(self):
        source = self.review["sources"][0]
        video = Path(source["source_path"])
        outside = self.root / "outside-video.mp4"
        outside.write_bytes(video.read_bytes())
        video.unlink()
        video.symlink_to(outside)
        with self.assertRaisesRegex(ValidationError, "outside its fixed"):
            media.verify_source_review(self.root)

    def test_public_attribution_does_not_include_private_paths_or_extra_fields(self):
        review = media.verify_source_review(self.root)
        review["sources"][0]["private_provider_response"] = {"secret": "never publish"}
        public = media.source_attribution(review, "source-0")
        text = json.dumps(public)
        self.assertNotIn(str(self.root), text)
        self.assertNotIn("private_provider_response", text)
        self.assertNotIn("source_path", text)
        self.assertEqual("mit", public["license"])

    def test_attribution_rejects_screened_out_source(self):
        with self.assertRaisesRegex(ValidationError, "one reviewed source"):
            media.source_attribution(self.review, "screened-out-0")

    def test_prepared_entries_retain_exact_concepts_labels_anchors_and_order(self):
        entries = self.entries()
        self.assertEqual(6, len(media.validate_prepared_entries(self.root, entries)["cases"]))
        mutations = [lambda values: values[0].update(expected_identity="excluded"),
                     lambda values: values[0].update(source_id="source-1"),
                     lambda values: values[0].update(anchor_origin="saved_spark_locator"),
                     lambda values: values[0].update(source_timestamps_ms=[0, 2000]),
                     lambda values: values.reverse(),
                     lambda values: values.pop()]
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                changed = deepcopy(entries)
                mutate(changed)
                with self.assertRaises(ValidationError):
                    media.validate_prepared_entries(self.root, changed)

    def test_prepared_contract_cannot_silently_drop_exclusions(self):
        entries = self.entries()
        path = Path(entries[0]["prepared_file"])
        prepared = json.loads(path.read_text())
        prepared["contract"]["excluded_concepts"] = []
        path.write_text(json.dumps(prepared))
        with self.assertRaisesRegex(ValidationError, "visual requirements"):
            media.validate_prepared_entries(self.root, entries)

    def test_prepared_candidate_cannot_move_to_a_different_object(self):
        entries = self.entries()
        path = Path(entries[0]["prepared_file"])
        prepared = json.loads(path.read_text())
        prepared["frames"][0]["candidate_box"]["x1"] += 20
        path.write_text(json.dumps(prepared))
        with self.assertRaisesRegex(ValidationError, "reviewed source anchor"):
            media.validate_prepared_entries(self.root, entries)

    def test_identical_frame_bytes_at_an_unreviewed_path_are_rejected(self):
        entries = self.entries()
        path = Path(entries[0]["prepared_file"])
        prepared = json.loads(path.read_text())
        replacement = self.folder / "copied-frame.jpg"
        replacement.write_bytes(Path(prepared["frames"][0]["source_file"]).read_bytes())
        prepared["frames"][0]["source_file"] = str(replacement)
        path.write_text(json.dumps(prepared))
        with self.assertRaisesRegex(ValidationError, "unreviewed source path"):
            media.validate_prepared_entries(self.root, entries)

    def test_optional_video_replay_decodes_each_distinct_source_time_once(self):
        def decode(args, **_kwargs):
            if args[-1] == "-version":
                return subprocess.CompletedProcess(args, 0, stdout=b"ffmpeg authored fixture\n")
            source, time = args[args.index("-i") + 1], float(args[args.index("-ss") + 1])
            Path(args[-1]).write_bytes(self.decode_bytes[(source, time)])
            return subprocess.CompletedProcess(args, 0, stdout=b"")

        with patch.object(media.subprocess, "run", side_effect=decode) as runner:
            media.verify_source_review(self.root, reextract=True)
        self.assertEqual(5, runner.call_count)  # Version and four distinct source times, no model calls.

    def test_frame_from_wrong_video_cannot_replay_successfully(self):
        def decode(args, **_kwargs):
            if args[-1] == "-version":
                return subprocess.CompletedProcess(args, 0, stdout=b"ffmpeg authored fixture\n")
            Path(args[-1]).write_bytes(jpeg(640, 480, b"wrong source frame"))
            return subprocess.CompletedProcess(args, 0, stdout=b"")

        with patch.object(media.subprocess, "run", side_effect=decode):
            with self.assertRaisesRegex(ValidationError, "pinned video"):
                media.verify_source_review(self.root, reextract=True)


if __name__ == "__main__":
    unittest.main()
