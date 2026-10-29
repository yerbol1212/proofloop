"""Full authored experiment choreography; no real videos, models or network.

JPEG headers, probe metadata, crops, binary masks and provider responses are
explicit test fixtures. They test source binding and budget enforcement, never
physical perception, tracking quality, calibration or real model reliability.
"""
from copy import deepcopy
import hashlib
from io import BytesIO
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from proofloop.core import ValidationError
from proofloop.inference_journal import AttemptJournal
from proofloop.meta import SAM_MODEL, SPARK_MODEL
from proofloop import tracking_experiment as experiment
from tests.test_guided import jpeg_header


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, sort_keys=True) + "\n")


def response(value):
    return BytesIO(json.dumps(value).encode())


def lane(text, model, *, status="completed"):
    return {"model": model, "status": status, "output": [{"type": "message", "content": [
        {"type": "output_text", "text": text}]}], "usage": {"input_tokens": 10, "output_tokens": 20,
            **({"video_frames_processed": 50} if model == SAM_MODEL else {})}}


def sam_record(index=0, track_id=1):
    return (f"<{index}f>{track_id}<|box;x1=8;y1=6;x2=23;y2=23;w=64;h=48;c=0.95|>"
        "<|mask;x=0;y=0;c=0.95;data=18,16,!authored-transport-mask|>")


class FrozenExperimentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.local = self.root / "artifacts/local/authored-tracking"
        self.local.mkdir(parents=True)
        self.output = self.local / "experiment"
        self.output.mkdir()
        self.annotations = self.output / "annotations.json"
        self.payloads = []
        self.clients = []
        self.baseline_mode = "completed"
        self.still_sam_mode = "completed"
        self.guard_mode = "completed"
        self.video_mode = "completed"
        self.authored_frame_bytes = {}
        self.impl = {name: "f" * 64 for name in experiment.FILES}
        self.contract = {"target_id": "target", "concept": "authored silver fork",
            "attributes": {"fork_shape": "visible fork tines"}, "exclusions": {"spoon": "rounded spoon bowl"}}
        self.reference_box = [.125, .125, .375, .5]
        clips = []
        for number in range(2):
            directory = self.local / ("clip-" + str(number))
            directory.mkdir()
            original = directory / "original.mp4"
            original.write_bytes(b"authored-original-" + str(number).encode())
            clip = directory / "clip.mp4"
            clip.write_bytes(b"\x00\x00\x00\x18ftypisom-authored-clip-" + str(number).encode())
            frame_directory = directory / "frames"
            frame_directory.mkdir()
            for index in range(8):
                frame_file = frame_directory / f"frame-{index + 1:03d}.jpg"
                data = jpeg_header(64, 48, f"authored-clip-{number}-frame-{index}".encode())
                frame_file.write_bytes(data)
                self.authored_frame_bytes[str(frame_file)] = data
            dataset, revision = "lerobot/authored_REFERENCE_ONLY_DO_NOT_SEND", "f" * 40
            remote = f"videos/chunk-000/observation.images.cam_high/episode_{number:06d}.mp4"
            clips.append({"clip_id": "authored-" + str(number),
                "source_attribution": {"dataset": dataset, "revision": revision, "episode": number,
                    "remote_path": remote, "url": f"https://huggingface.co/datasets/{dataset}/resolve/{revision}/{remote}",
                    "license": "mit", "license_url": f"https://huggingface.co/datasets/{dataset}/blob/{revision}/README.md",
                    "source_role": "new_source_episode_same_exposed_dataset_family"},
                "original_file": str(original), "original_sha256": sha(original), "clip_file": str(clip),
                "seeds": [{"object_id": "target", "track_id": 1, "frame_index": 0,
                    "box": {"x1": 8, "y1": 6, "x2": 23, "y2": 23}},
                    {"object_id": "distractor", "track_id": 2, "frame_index": 0,
                     "box": {"x1": 35, "y1": 10, "x2": 50, "y2": 30}}],
                "contract": deepcopy(self.contract), "frames_directory": str(frame_directory),
                "checkpoints": [{"frame_index": index, "expected": "supported" if index % 2 == 0 else "unknown",
                    "target_box": self.reference_box if index % 2 == 0 else None,
                    "distractor_boxes": [[.55, .2, .8, .7]],
                    "condition": "clear" if index % 2 == 0 else "hidden_or_unidentifiable"} for index in range(8)]})
        dump(self.annotations, {"review_origin": "assistant_under_user_authorization", "reviewed_before_inference": True,
            "clips": clips})
        self._patch("proofloop.tracking_experiment._implementations", return_value=self.impl)
        self.reextract = self._patch("proofloop.tracking_experiment._reextract", side_effect=self._reextract)
        self._patch("proofloop.tracking_experiment._derivation", side_effect=lambda entry: {
            "authored_fixture_no_real_pixel_derivation": True,
            "original_sha256": entry["original_sha256"], "tracking_clip_sha256": entry["prepared"]["source"]["sha256"]})
        self._patch("proofloop.video_tracking._probe", side_effect=self._probe)
        self._patch("proofloop.meta._sam_decoder", return_value=(SimpleNamespace,
            lambda mask: b"\x01" * (mask.width * mask.height), ValueError))
        self._patch("proofloop.video_tracking._sam_decoder", return_value=None)
        self._patch("proofloop.tracking_guard._extractor_path", return_value="/authored/ffmpeg")
        self._patch("proofloop.tracking_guard._extractor_version", return_value="ffmpeg version authored-test")
        self._patch("proofloop.tracking_guard._crop", side_effect=self._crop)
        self._patch("proofloop.tracking_experiment.load_env_file")
        self._patch("proofloop.tracking_experiment.RetainingClient.from_env", side_effect=self._client)

    def _patch(self, target, **kwargs):
        handle = patch(target, **kwargs)
        value = handle.start()
        self.addCleanup(handle.stop)
        return value

    def _probe(self, path, ffprobe_path=None):
        return ({"width": 64, "height": 48, "codec": "h264", "container": "mp4_or_mov",
            "duration_ms": 1600, "frame_count": 8, "time_base": "1/5", "first_pts": 0,
            "frames": [{"frame_index": index, "pts": index, "timestamp_ms": index * 200} for index in range(8)]},
            "ffprobe version authored-test")

    def _crop(self, executable, source, target, box):
        # A deterministic authored JPEG header binds both crop geometry and
        # original frame bytes, without claiming actual pixel extraction.
        suffix = (sha(source) + json.dumps(box, sort_keys=True)).encode()
        Path(target).write_bytes(jpeg_header(box["x2"] - box["x1"] + 1,
            box["y2"] - box["y1"] + 1, suffix))

    def _reextract(self, entry):
        # Model the extraction boundary with independently retained authored
        # bytes. Updating a manifest SHA cannot turn an unrelated JPEG into a
        # frame from this authored clip. Production uses actual ffmpeg replay.
        for frame in entry["frames"]:
            expected = self.authored_frame_bytes[frame["file"]]
            if Path(frame["file"]).read_bytes() != expected or frame["sha256"] != hashlib.sha256(expected).hexdigest():
                raise ValidationError("Authored extraction fixture does not derive from its source")

    def _client(self, **kwargs):
        client = experiment.RetainingClient("authored-key-not-real", opener=self._opener, **kwargs)
        self.clients.append(client)
        return client

    def _opener(self, request, **kwargs):
        payload = json.loads(request.data)
        self.payloads.append(payload)
        contents = payload["input"][0]["content"]
        if payload["model"] == SAM_MODEL:
            if any(part["type"] == "input_video" for part in contents):
                text = "\n".join(sam_record(index) for index in range(8))
                return response(lane(text, SAM_MODEL, status=self.video_mode))
            return response(lane(sam_record(), SAM_MODEL, status=self.still_sam_mode))
        image_count = sum(part["type"] == "input_image" for part in contents)
        if image_count == 2:
            parsed = {"coordinate_system": "normalized_0_1000", "frames": [{"frame_index": index,
                "width": 64, "height": 48, "objects": [{"object_id": "target", "identity_ambiguous": False,
                    "fully_visible": True, "boundary_uncertain": False,
                    "candidates": [{"x1": 125, "y1": 125, "x2": 375, "y2": 500}]}]} for index in range(2)]}
            return response(lane(json.dumps(parsed), SPARK_MODEL, status=self.baseline_mode))
        parsed = {"candidate_identity": "target", "fully_visible": True, "identity_ambiguous": False,
            "attributes": {"fork_shape": "present"}, "exclusions": {"spoon": "absent"}}
        return response(lane(json.dumps(parsed), SPARK_MODEL, status=self.guard_mode))

    def freeze(self):
        self.plan = experiment.freeze(self.root, self.annotations, self.output)
        return self.plan

    def run_experiment(self):
        if not (self.output / "freeze.json").exists():
            self.freeze()
        return experiment.run(self.root, self.output / "freeze.json")

    def test_freeze_uses_no_inference_and_binds_code_labels_sources_and_selected_frames(self):
        plan = self.freeze()
        self.assertEqual([], self.payloads)
        self.assertEqual([], self.clients)
        self.assertEqual(2, self.reextract.call_count)
        self.assertFalse(plan["inference_started"])
        self.assertEqual(self.impl, plan["implementation_sha256"])
        self.assertEqual(sha(self.annotations), plan["annotations_sha256"])
        self.assertEqual(16, sum(len(entry["checkpoints"]) for entry in plan["clips"]))
        experiment.verify_plan(self.root, plan)

    def test_full_run_and_replay_use_real_budget_and_never_send_reference_labels(self):
        result = self.run_experiment()
        self.assertEqual(42, len(self.payloads))
        self.assertEqual(42, result["accounting"]["total_reserved_attempts"])
        self.assertEqual(0, result["accounting"]["remaining_attempts"])
        self.assertEqual(2, result["source_videos"])
        self.assertEqual(16, result["labeled_frames"])
        self.assertEqual(set(experiment.METHODS), set(result["methods"]))
        self.assertFalse(result["changes_live_placement_acceptance"])
        self.assertEqual(0, result["failed_operation_count"])
        wire = json.dumps(self.payloads)
        for label in ('"expected"', '"target_box"', '"distractor_boxes"', '"condition"', "REFERENCE_ONLY_DO_NOT_SEND"):
            self.assertNotIn(label, wire)
        self.assertEqual(2, sum(any(part["type"] == "input_video" for part in payload["input"][0]["content"]) for payload in self.payloads))
        self.assertEqual(16, sum(payload["model"] == SPARK_MODEL and
            sum(part["type"] == "input_image" for part in payload["input"][0]["content"]) == 1 for payload in self.payloads))
        self.assertTrue(all(client.retries == 0 for client in self.clients))
        calls = len(self.payloads)
        replay = experiment.replay(self.root, self.output / "freeze.json")
        self.assertEqual(result, replay)
        self.assertEqual(calls, len(self.payloads))

    def test_run_marker_prevents_a_second_request_set_and_global_allowance_survives_marker_removal(self):
        self.run_experiment()
        with self.assertRaises(FileExistsError):
            self.run_experiment()
        self.assertEqual(42, len(self.payloads))
        (self.output / "run-started.json").unlink()
        with self.assertRaises(FileExistsError):
            self.run_experiment()
        self.assertEqual(42, len(self.payloads))
        self.assertEqual(42, AttemptJournal(self.output / "attempts.sqlite3", request_limit=42).summary()["total_reserved_attempts"])

    def test_config_guard_methods_hash_and_implementation_changes_fail_before_any_request(self):
        plan = self.freeze()
        mutations = [lambda value: value["config"].update(max_requests=43),
            lambda value: value["guard_config"].update(max_requests=2),
            lambda value: value["methods"].append("unreviewed_method"),
            lambda value: value["implementation_sha256"].update(meta_py="a" * 64),
            lambda value: value.update(freeze_hash="a" * 64)]
        for mutation in mutations:
            changed = deepcopy(plan)
            mutation(changed)
            with self.assertRaises(ValidationError):
                experiment.verify_plan(self.root, changed)
        with patch("proofloop.tracking_experiment._implementations", return_value={"changed.py": "a" * 64}):
            with self.assertRaises(ValidationError):
                experiment.verify_plan(self.root, plan, files=False)
        self.assertEqual([], self.payloads)

    def test_annotation_original_clip_and_jpeg_mutations_fail_source_verification(self):
        plan = self.freeze()
        paths = [self.annotations, Path(plan["clips"][0]["original_file"]),
            Path(plan["clips"][0]["prepared"]["clip_file"]), Path(plan["clips"][0]["frames"][3]["file"])]
        for path in paths:
            before = path.read_bytes()
            path.write_bytes(before + b" changed")
            with self.assertRaises(ValidationError):
                experiment.verify_plan(self.root, plan)
            path.write_bytes(before)
        self.assertEqual([], self.payloads)

    def test_resealed_frame_hash_cannot_detach_a_jpeg_from_its_source_clip(self):
        plan = self.freeze()
        frame = plan["clips"][0]["frames"][2]
        path = Path(frame["file"])
        path.write_bytes(jpeg_header(64, 48, b"unrelated-authored-image"))
        frame["sha256"] = sha(path)
        plan["freeze_hash"] = experiment._digest({key: value for key, value in plan.items() if key != "freeze_hash"})
        with self.assertRaises(ValidationError):
            experiment.verify_plan(self.root, plan)
        self.assertEqual([], self.payloads)

    def test_resealed_frame_inventory_cannot_change_source_index_or_pts(self):
        original = self.freeze()
        for field, value in (("frame_index", 7), ("pts", 99), ("timestamp_ms", 999)):
            changed = deepcopy(original)
            changed["clips"][0]["frames"][2][field] = value
            changed["freeze_hash"] = experiment._digest({key: item for key, item in changed.items() if key != "freeze_hash"})
            with self.assertRaises(ValidationError):
                experiment.verify_plan(self.root, changed)
        self.assertEqual([], self.payloads)

    def test_per_request_guard_checks_frozen_code_and_annotations_before_admission(self):
        plan = self.freeze()
        self.run_experiment()
        guard_callback = self.clients[0]._guard
        with patch("proofloop.tracking_experiment._implementations", return_value={}):
            with self.assertRaises(ValidationError):
                guard_callback()
        self.annotations.write_text(self.annotations.read_text() + "\n")
        with self.assertRaises(ValidationError):
            guard_callback()
        self.assertEqual(42, len(self.payloads))

    def test_baseline_incomplete_responses_are_unscored_and_retained(self):
        self.baseline_mode = "incomplete"
        result = self.run_experiment()
        for method in experiment.METHODS[:2]:
            metrics = result["methods"][method]["metrics"]
            self.assertEqual(16, metrics["labels_unscored"])
            self.assertEqual(0, metrics["labels_completed"])
            self.assertEqual(0, metrics["unknowns_withheld"])
        saved = experiment._read(self.output / "authored-0-pair-0.json")
        self.assertEqual("unscored", saved["status"])
        self.assertEqual("incomplete_response", saved["error"]["code"])
        self.assertIsNone(saved["result"])
        self.assertEqual(1, len(saved["retained_responses"]))
        self.assertEqual("incomplete", saved["retained_responses"][0]["response"]["status"])
        self.assertEqual(26, result["accounting"]["total_reserved_attempts"])

    def test_later_sam_failure_preserves_completed_locator_and_unscored_pair(self):
        self.still_sam_mode = "incomplete"
        result = self.run_experiment()
        for method in experiment.METHODS[:2]:
            self.assertEqual(16, result["methods"][method]["metrics"]["labels_unscored"])
        saved = experiment._read(self.output / "authored-0-pair-0.json")
        self.assertEqual("unscored", saved["status"])
        self.assertEqual(2, saved["usage"]["request_attempts"])
        self.assertEqual(2, len(saved["retained_responses"]))
        self.assertEqual("completed", saved["retained_responses"][0]["response"]["status"])
        self.assertEqual(SPARK_MODEL, saved["retained_responses"][0]["response"]["model"])
        self.assertEqual("incomplete", saved["retained_responses"][1]["response"]["status"])
        self.assertEqual(34, result["accounting"]["total_reserved_attempts"])

    def test_guard_incomplete_response_is_unscored_and_keeps_reported_tokens(self):
        self.guard_mode = "incomplete"
        result = self.run_experiment()
        self.assertEqual(16, result["methods"]["semantic_reacquisition"]["metrics"]["labels_unscored"])
        self.assertEqual(0, result["methods"]["sam_video_presence"]["metrics"]["labels_unscored"])
        self.assertEqual(42, result["accounting"]["total_reserved_attempts"])
        saved = experiment._read(self.output / "authored-0-guard-0.json")
        self.assertEqual("unscored", saved["status"])
        self.assertEqual(20, saved["usage"]["output_tokens"])
        self.assertEqual("incomplete", saved["error"]["response"]["status"])

    def test_failed_video_does_not_create_synthetic_unknown_predictions_or_guard_calls(self):
        self.video_mode = "incomplete"
        result = self.run_experiment()
        self.assertEqual(16, result["methods"]["sam_video_presence"]["metrics"]["labels_unscored"])
        self.assertEqual(16, result["methods"]["semantic_reacquisition"]["metrics"]["labels_unscored"])
        self.assertEqual(26, result["accounting"]["total_reserved_attempts"])
        self.assertEqual(0, sum(payload["model"] == SPARK_MODEL and
            sum(part["type"] == "input_image" for part in payload["input"][0]["content"]) == 1 for payload in self.payloads))

    def test_tracker_response_preparation_cannot_be_swapped_between_frozen_clips(self):
        self.run_experiment()
        first = self.output / "authored-0-video.json"
        second = self.output / "authored-1-video.json"
        original = first.read_bytes()
        first.write_bytes(second.read_bytes())
        with self.assertRaises(ValidationError):
            experiment.replay(self.root, self.output / "freeze.json")
        first.write_bytes(original)

    def test_semantic_frame_candidate_source_and_contract_bindings_reject_tampering(self):
        self.run_experiment()
        path = self.output / "authored-0-guard-2.json"
        original = experiment._read(path)
        changes = [lambda value: value["prepared"].update(frame_index=3),
            lambda value: value["prepared"].update(track_id=2),
            lambda value: value["prepared"].update(source_sha256="b" * 64),
            lambda value: value["prepared"].update(frame_sha256="b" * 64),
            lambda value: value["prepared"]["candidate_box"].update(x2=25),
            lambda value: value["prepared"]["contract"].update(concept="other object")]
        for mutation in changes:
            value = deepcopy(original)
            mutation(value)
            dump(path, value)
            with self.assertRaises(ValidationError):
                experiment.replay(self.root, self.output / "freeze.json")
        dump(path, original)

    def test_reference_only_changes_are_not_accepted_as_a_new_frozen_comparison(self):
        plan = self.freeze()
        plan["clips"][0]["checkpoints"][0]["expected"] = "unknown"
        plan["clips"][0]["checkpoints"][0]["target_box"] = None
        with self.assertRaises(ValidationError):
            experiment.verify_plan(self.root, plan)
        plan["freeze_hash"] = experiment._digest({key: value for key, value in plan.items() if key != "freeze_hash"})
        with self.assertRaises(ValidationError):
            experiment.verify_plan(self.root, plan)
        self.assertEqual([], self.payloads)

    def test_admission_marker_must_name_the_exact_frozen_plan(self):
        plan = self.freeze()
        dump(self.output / "run-started.json", {"freeze_hash": "a" * 64, "automatic_retry": False})
        with self.assertRaises(ValidationError):
            experiment.verify_plan(self.root, plan)
        self.assertEqual([], self.payloads)

    def test_baseline_locator_provenance_coverage_and_retained_responses_replay_exactly(self):
        self.run_experiment()
        path = self.output / "authored-0-pair-0.json"
        original = experiment._read(path)
        frame_hash = next(iter(original["result"]["results"]))
        def record(value):
            return value["result"]["results"][frame_hash]["target"]
        changes = [lambda value: record(value)["guided_provenance"]["locator_box"].update(x1=7),
            lambda value: record(value)["guided_provenance"].update(sam_prompt="fork"),
            lambda value: record(value)["objects"][0].update(confidence=.99),
            lambda value: value["result"]["locator"]["parsed"]["frames"][0]["objects"][0].update(fully_visible=False),
            lambda value: value["result"]["results"].update({"a" * 64: {}}),
            lambda value: value["retained_responses"][0]["response"].update(model=SAM_MODEL)]
        for mutation in changes:
            changed = deepcopy(original)
            mutation(changed)
            dump(path, changed)
            with self.assertRaises(ValidationError):
                experiment.replay(self.root, self.output / "freeze.json")
        dump(path, original)

    def test_replay_rejects_tampered_baseline_and_guard_costs(self):
        self.run_experiment()
        for name in ("authored-0-pair-0.json", "authored-0-guard-0.json"):
            path = self.output / name
            original = experiment._read(path)
            value = deepcopy(original)
            value["usage"]["estimated_usd"] = -100
            dump(path, value)
            with self.assertRaises(ValidationError, msg=name):
                experiment.replay(self.root, self.output / "freeze.json")
            dump(path, original)


if __name__ == "__main__":
    unittest.main()
