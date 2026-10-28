"""Offline interactive state/anti-staleness tests; no calls to Meta or camera claims.

All encoded pictures, mask records and even recorded-report provenance below
are authored fixtures. They test the local contracts, not model performance.
"""
import base64
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import struct
import tempfile
import threading
import unittest
from unittest.mock import patch
import zlib

from proofloop.core import ValidationError
from proofloop.interactive import InteractiveEngine, FRESHNESS_MS, TOKEN_LIFETIME_MS, UPRIGHT_IMAGE_MESSAGE
from proofloop.meta import MetaAPIError, SAM_MODEL
from proofloop.session import approve_contract, contract_hash, evaluate_session, session_contract
from tests.test_session import capture_manifest, detection
from tests.test_guided import FakeClient as GuidedFakeClient, jpeg_header

REGIONS = {"left": [0.05, 0.05, 0.45, 0.95], "right": [0.55, 0.05, 0.95, 0.95]}


def picture(index=0):
    def chunk(kind, body):
        return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body) & 0xffffffff)
    header = struct.pack(">IIBBBBB", 100, 100, 8, 2, 0, 0, 0)
    pixels = b"".join(b"\0" + bytes([index % 256, 140, 180]) * 100 for _ in range(100))
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(pixels)) + chunk(b"IEND", b"")


def image(index):
    return {"data_url": "data:image/png;base64," + base64.b64encode(picture(index)).decode()}


def jpeg_header_fixture(orientation):
    """Header-only authored JPEG transport fixture; no usable photo or inference."""
    tiff = b"II" + struct.pack("<HIH", 42, 8, 1) + struct.pack("<HHI", 0x0112, 3, 1) + struct.pack("<H", orientation) + b"\0" * 6
    exif = b"Exif\0\0" + tiff
    sof = bytes([8]) + struct.pack(">HH", 100, 100) + b"\x01\x01\x11\x00"
    return (b"\xff\xd8\xff\xe1" + struct.pack(">H", len(exif) + 2) + exif +
            b"\xff\xc0" + struct.pack(">H", len(sof) + 2) + sof + b"\xff\xd9")


def sam_result(path, concept, *, wrong=False, covered=False):
    return {"source": {"sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()},
            "response": {"status": "completed", "model": SAM_MODEL, "id": "private-provider-id"},
            "mask_decoding": "official_meta_sam_parser_0.1.0",
            "objects": [] if covered else [detection("right" if wrong or concept == "spoon" else "left")]}


class FakeClient:
    def __init__(self, *, wrong=False, covered=False, fail_on=None, callback=None, transform=None):
        self.max_requests, self.retries, self.ledger = 4, 0, []
        self.wrong, self.covered, self.fail_on = wrong, covered, fail_on
        self.callback, self.transform = callback, transform

    def segment_image(self, path, concept, *, decode_masks):
        self.ledger.append({"concept": concept, "decode_masks": decode_masks})
        if self.callback:
            self.callback(len(self.ledger))
        if len(self.ledger) == self.fail_on:
            raise MetaAPIError("Secret upstream body must not escape", response={"private": "hidden-body"})
        result = sam_result(path, concept, wrong=self.wrong, covered=self.covered)
        return self.transform(result) if self.transform else result

    def usage_summary(self):
        return {"request_attempts": len(self.ledger), "estimated_usd": 0, "estimate_origin": "authored_mock_no_api_calls"}


class FakeCompileClient:
    """Authored Spark schema response, not a model-quality assertion."""
    def __init__(self, placements=None, failure=False, callback=None, unsupported=False):
        self.max_requests, self.retries, self.ledger = 1, 0, []
        self.placements = placements or {"fork": "right", "spoon": "left"}
        self.failure, self.callback, self.unsupported = failure, callback, unsupported

    def spark_json(self, prompt, schema, **kwargs):
        self.ledger.append({"prompt": prompt, "kwargs": kwargs})
        if self.callback:
            self.callback()
        if self.failure:
            raise MetaAPIError("Secret Spark response", response={"secret": "never-public"})
        objects = list(self.placements)
        return {"parsed": {"protocol": {"id": "proofloop-task", "version": "draft-1", "objects": objects,
            "regions": ["left", "right"], "steps": [], "checkpoints": [],
            "checks": [{"id": value + "-placement", "kind": "location", "object_id": value,
                        "region_id": region, "unique_target": True} for value, region in self.placements.items()]},
            "unsupported_requirements": ["Authored unsupported requirement"] if self.unsupported else [],
            "clarifications_needed": []}, "usage": {"input_tokens": 1, "output_tokens": 1}}

    def usage_summary(self):
        return {"request_attempts": len(self.ledger), "estimate_origin": "authored_mock_no_api_calls"}


class InteractiveTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.client = FakeClient()
        self.engine = InteractiveEngine(self.root, client_factory=lambda: self.client)

    def tearDown(self):
        self.temp.cleanup()

    def payload(self, *, index=1, source="upload"):
        state = self.engine.new_inspection()
        return {"capture_token": state["capture_token"], "source": source,
                "images": [image(index), image(index + 1)], "regions": deepcopy(REGIONS),
                "confirm_regions": True, "confirm_unique_targets": True,
                "confirm_fresh_capture": True, "confirm_pair_spacing": True}

    def test_missing_seed_starts_unknown_without_fake_mistake(self):
        state = self.engine.snapshot()
        self.assertEqual(("needs_evidence", "unknown", "none"), (state["phase"], state["status"], state["source_kind"]))
        self.assertEqual([], state["images"])
        self.assertIsNone(state["proposal"])
        self.assertIsNone(state["capture_token"])

    def test_fresh_pair_runs_four_calls_and_is_explicitly_injected(self):
        state = self.engine.inspect_pair(self.payload())
        self.assertEqual(("verified", "pass"), (state["phase"], state["status"]))
        self.assertEqual(4, len(self.client.ledger))
        self.assertTrue(all(item["decode_masks"] for item in self.client.ledger))
        self.assertEqual("injected_client_not_verified_live", state["inference_origin"])
        self.assertIn("declared by the operator", state["source_note"])
        self.assertIsNotNone(state["verified_at_ms"])
        self.assertIsNone(state["capture_token"])
        self.assertEqual(2, len(state["images"]))
        self.assertTrue(all(len(item["boxes"]) == 2 for item in state["images"]))
        self.assertNotIn("private-provider-id", json.dumps(state))
        self.assertNotIn("raster_base64", json.dumps(state))
        self.assertNotIn(str(self.root), json.dumps(state))

    def test_begin_repair_invalidates_pass_before_new_perception(self):
        before = self.engine.inspect_pair(self.payload())
        after = self.engine.begin_repair()
        self.assertEqual(before["revision"] + 1, after["revision"])
        self.assertEqual(("awaiting_capture", "unknown"), (after["phase"], after["status"]))
        self.assertTrue(all(check["status"] == "unknown" and check["observed_region"] is None for check in after["checks"]))
        self.assertIsNone(after["verified_at_ms"])
        self.assertIsNone(after["proposal"])
        self.assertEqual(4, len(self.client.ledger))

    def test_wrong_pair_plans_move_without_marking_execution_verified(self):
        self.client.wrong = True
        state = self.engine.inspect_pair(self.payload())
        self.assertEqual(("mistake", "fail"), (state["phase"], state["status"]))
        self.assertIn("move-fork-right-to-left", state["proposal"]["action_ids"])
        self.assertTrue(state["proposal"]["hypothetical"])
        self.assertFalse(state["proposal"]["execution_verified"])
        self.assertIsNone(state["verified_at_ms"])

    def test_occlusion_returns_unknown_and_no_move_or_pass(self):
        self.client.covered = True
        state = self.engine.inspect_pair(self.payload())
        self.assertEqual(("unknown", "unknown"), (state["phase"], state["status"]))
        self.assertTrue(all(check["observed_region"] is None for check in state["checks"]))
        self.assertIsNone(state["proposal"])

    def test_duplicate_pair_rejected_before_paid_calls(self):
        payload = self.payload()
        payload["images"][1] = deepcopy(payload["images"][0])
        with self.assertRaises(ValidationError):
            self.engine.inspect_pair(payload)
        self.assertEqual([], self.client.ledger)

    def test_old_token_is_invalidated_by_new_revision(self):
        first = self.payload()
        self.engine.new_inspection()
        with self.assertRaises(ValidationError):
            self.engine.inspect_pair(first)
        self.assertEqual([], self.client.ledger)

    def test_token_is_one_use_after_success(self):
        payload = self.payload()
        self.engine.inspect_pair(payload)
        with self.assertRaises(ValidationError):
            self.engine.inspect_pair(payload)
        self.assertEqual(4, len(self.client.ledger))

    def test_seen_upload_rejected_on_next_inspection(self):
        self.engine.inspect_pair(self.payload())
        self.client = FakeClient()
        repeated = self.payload()
        with self.assertRaises(ValidationError):
            self.engine.inspect_pair(repeated)
        self.assertEqual([], self.client.ledger)
        self.assertEqual("unknown", self.engine.snapshot()["status"])

    def test_all_confirmations_are_required(self):
        for name in ("confirm_regions", "confirm_unique_targets", "confirm_fresh_capture", "confirm_pair_spacing"):
            payload = self.payload()
            payload[name] = False
            with self.subTest(name=name), self.assertRaises(ValidationError):
                self.engine.inspect_pair(payload)
        self.assertEqual([], self.client.ledger)

    def test_region_overlap_is_rejected(self):
        payload = self.payload()
        payload["regions"]["left"] = [0, 0, 1, 1]
        with self.assertRaises(ValidationError):
            self.engine.inspect_pair(payload)
        self.assertEqual([], self.client.ledger)

    def test_token_expires_after_twenty_minutes(self):
        with patch("proofloop.interactive._now_ms", return_value=1000):
            payload = self.payload()
        with patch("proofloop.interactive._now_ms", return_value=1001 + TOKEN_LIFETIME_MS):
            with self.assertRaises(ValidationError):
                self.engine.inspect_pair(payload)
        self.assertEqual([], self.client.ledger)

    def test_camera_capture_before_token_or_too_close_is_rejected(self):
        with patch("proofloop.interactive._now_ms", return_value=10000):
            payload = self.payload(source="camera")
        for first, second in ((9000, 12000), (11000, 11500)):
            payload["images"][0]["captured_at_ms"] = first
            payload["images"][1]["captured_at_ms"] = second
            with patch("proofloop.interactive._now_ms", return_value=15000), self.assertRaises(ValidationError):
                self.engine.inspect_pair(payload)
        self.assertEqual([], self.client.ledger)

    def test_camera_reported_times_are_used_without_upload_attestation_time(self):
        with patch("proofloop.interactive._now_ms", return_value=10000):
            payload = self.payload(source="camera")
        payload["images"][0]["captured_at_ms"] = 11000
        payload["images"][1]["captured_at_ms"] = 12500
        with patch("proofloop.interactive._now_ms", return_value=15000):
            state = self.engine.inspect_pair(payload)
        self.assertEqual("pass", state["status"])
        self.assertIn("client-reported", state["source_note"])

    def test_camera_evidence_can_age_during_network_and_yield_unknown(self):
        clock = [10000]
        with patch("proofloop.interactive._now_ms", side_effect=lambda: clock[0]):
            payload = self.payload(source="camera")
            payload["images"][0]["captured_at_ms"] = 11000
            payload["images"][1]["captured_at_ms"] = 12500
            clock[0] = 15000
            self.client.callback = lambda _: clock.__setitem__(0, 15000 + FRESHNESS_MS)
            state = self.engine.inspect_pair(payload)
        self.assertEqual("unknown", state["status"])
        self.assertIsNone(state["verified_at_ms"])

    def test_pass_snapshot_ages_out_instead_of_claiming_current_pass_forever(self):
        with patch("proofloop.interactive._now_ms", return_value=10000):
            self.engine.inspect_pair(self.payload())
        with patch("proofloop.interactive._now_ms", return_value=10001 + FRESHNESS_MS):
            state = self.engine.snapshot()
        self.assertEqual(("unknown", "unknown"), (state["phase"], state["status"]))
        self.assertTrue(state["stale"])
        self.assertEqual(10000, state["verified_at_ms"])
        self.assertTrue(all(item["status"] == "unknown" for item in state["checks"]))

    def test_failed_api_erases_pass_and_keeps_raw_error_private(self):
        self.engine.inspect_pair(self.payload())
        self.client = FakeClient(fail_on=2)
        state = self.engine.inspect_pair(self.payload(index=3))
        self.assertEqual(("error", "unknown"), (state["phase"], state["status"]))
        self.assertFalse(state["busy"])
        self.assertIsNone(state["verified_at_ms"])
        self.assertNotIn("hidden-body", json.dumps(state))
        self.assertNotIn("Secret upstream", json.dumps(state))
        self.assertEqual(2, len(self.client.ledger))
        with self.assertRaises(ValidationError):
            self.engine.inspect_pair(self.payload(index=3))

    def test_wrong_model_source_and_geometry_cannot_verify(self):
        def wrong_model(value):
            value["response"]["model"] = "other-provider"
            return value
        def wrong_source(value):
            value["source"]["sha256"] = "0" * 64
            return value
        def wrong_geometry(value):
            value["objects"][0]["box"]["width"] = 500
            return value
        for index, transform in enumerate((wrong_model, wrong_source, wrong_geometry), start=10):
            self.client = FakeClient(transform=transform)
            state = self.engine.inspect_pair(self.payload(index=index * 2))
            self.assertEqual(("error", "unknown"), (state["phase"], state["status"]))
            self.assertEqual(1, len(self.client.ledger))

    def test_single_busy_inspection_cannot_be_overwritten_or_submitted_twice(self):
        entered, release = threading.Event(), threading.Event()
        self.client.callback = lambda number: (entered.set(), release.wait(3)) if number == 1 else None
        payload = self.payload()
        output = []
        worker = threading.Thread(target=lambda: output.append(self.engine.inspect_pair(payload)))
        worker.start()
        self.assertTrue(entered.wait(2))
        try:
            self.assertTrue(self.engine.snapshot()["busy"])
            self.assertEqual("unknown", self.engine.snapshot()["status"])
            self.assertIsNone(self.engine.snapshot()["inspection_request"])
            for mutation in (self.engine.begin_repair, self.engine.new_inspection, lambda: self.engine.inspect_pair(payload)):
                with self.assertRaises(ValidationError):
                    mutation()
        finally:
            release.set()
            worker.join(3)
        self.assertFalse(worker.is_alive())
        self.assertEqual("pass", output[0]["status"])
        self.assertEqual(4, len(self.client.ledger))

    def test_bad_image_signature_checksum_and_dimensions_are_rejected(self):
        variants = [b"not a png", picture(1)[:-1], picture(1).replace(b"IHDR", b"XXXX", 1)]
        for value in variants:
            payload = self.payload()
            payload["images"][0]["data_url"] = "data:image/png;base64," + base64.b64encode(value).decode()
            with self.assertRaises(ValidationError):
                self.engine.inspect_pair(payload)
        self.assertEqual([], self.client.ledger)

    def test_state_copy_cannot_modify_internal_result(self):
        state = self.engine.snapshot()
        state["checks"][0]["status"] = "pass"
        state["regions"]["left"][0] = 0.8
        self.assertEqual("unknown", self.engine.snapshot()["checks"][0]["status"])
        self.assertNotEqual(0.8, self.engine.snapshot()["regions"]["left"][0])

    def test_media_identifiers_do_not_accept_paths(self):
        for value in ("../.env.local", "/etc/passwd", "a" * 32):
            with self.assertRaises(ValidationError):
                self.engine.media_path(value)
        state = self.engine.inspect_pair(self.payload())
        path = self.engine.media_path(state["images"][0]["id"])
        self.assertTrue(path.is_file())
        self.assertEqual(0o600, path.stat().st_mode & 0o777)

    def test_media_replacement_is_rejected_instead_of_showing_unbound_pixels(self):
        state = self.engine.inspect_pair(self.payload())
        media_id = state["images"][0]["id"]
        path = self.engine.media_path(media_id)
        path.write_bytes(picture(99))
        with self.assertRaises(ValidationError):
            self.engine.media_path(media_id)

    def test_camera_snapshot_age_uses_capture_time_not_upload_arrival(self):
        with patch("proofloop.interactive._now_ms", return_value=10000):
            payload = self.payload(source="camera")
        payload["images"][0]["captured_at_ms"] = 11000
        payload["images"][1]["captured_at_ms"] = 12500
        with patch("proofloop.interactive._now_ms", return_value=35000):
            self.assertEqual("pass", self.engine.inspect_pair(payload)["status"])
        with patch("proofloop.interactive._now_ms", return_value=42000):
            self.assertEqual("unknown", self.engine.snapshot()["status"])

    def test_storage_directory_failure_releases_busy_and_can_start_again(self):
        payload = self.payload()
        with patch.object(Path, "mkdir", side_effect=OSError("Authored disk failure")):
            state = self.engine.inspect_pair(payload)
        self.assertEqual(("error", "unknown", False), (state["phase"], state["status"], state["busy"]))
        self.assertIsNotNone(self.engine.new_inspection()["capture_token"])
        self.assertEqual([], self.client.ledger)

    def test_rotated_jpeg_exif_is_rejected_before_paid_calls(self):
        for orientation in range(2, 9):
            payload = self.payload()
            payload["images"][0]["data_url"] = "data:image/jpeg;base64," + base64.b64encode(jpeg_header_fixture(orientation)).decode()
            with self.subTest(orientation=orientation), self.assertRaisesRegex(ValidationError, "orientation metadata requires rotation"):
                self.engine.inspect_pair(payload)
        self.assertEqual([], self.client.ledger)

    def test_seen_uploads_remain_blocked_after_server_restart(self):
        self.engine.inspect_pair(self.payload())
        self.engine = InteractiveEngine(self.root, client_factory=lambda: self.client)
        self.client = FakeClient()
        with self.assertRaises(ValidationError):
            self.engine.inspect_pair(self.payload())
        self.assertEqual([], self.client.ledger)

    def test_malformed_result_and_mask_cannot_leave_checking_or_mark_pass(self):
        def broken_object(value):
            value["objects"] = [None]
            return value
        self.client = FakeClient(transform=broken_object)
        state = self.engine.inspect_pair(self.payload())
        self.assertEqual(("error", "unknown", False), (state["phase"], state["status"], state["busy"]))
        def broken_mask(value):
            value["objects"][0]["mask"]["raster_base64"] = "invalid-raster"
            return value
        self.client = FakeClient(transform=broken_mask)
        state = self.engine.inspect_pair(self.payload(index=5))
        self.assertEqual(("unknown", "unknown", False), (state["phase"], state["status"], state["busy"]))

    def make_recorded_seed(self):
        """Authored provenance fixtures exercise binding, never establish a live claim."""
        directory = self.root / "inputs/local/phone/fixture"
        directory.mkdir(parents=True)
        manifest = capture_manifest()
        manifest["evidence_origin"] = "owned_phone"
        clip = directory.parent / "authored-fixture.mp4"
        clip.write_bytes(b"authored source binding fixture, not a playable clip")
        manifest["source"]["sha256"] = hashlib.sha256(clip.read_bytes()).hexdigest()
        results = {}
        counter = 50
        for stage in manifest["stages"]:
            results[stage["id"]] = {}
            for frame in stage["frames"]:
                path = directory / frame["file"]
                path.write_bytes(picture(counter))
                counter += 1
                frame["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
                results[stage["id"]][frame["sha256"]] = {obj: sam_result(path, obj,
                    wrong=stage["id"] == "wrong_region", covered=stage["id"] == "occluded") for obj in ("fork", "spoon")}
        (directory / "manifest.json").write_text(json.dumps(manifest))
        contract = session_contract(manifest, self.engine._protocol, deepcopy(REGIONS), {"fork": "fork", "spoon": "spoon"},
                                    manifest_file="inputs/local/phone/fixture/manifest.json")
        envelope = approve_contract(contract, contract_hash(contract), confirm_unique_targets=True)
        report = evaluate_session(envelope, results, inference_origin="live_meta_api_on_recorded_frames")
        report["failures"] = []
        output = self.root / "artifacts/local/recorded-fixture"
        output.mkdir()
        (output / "approval.json").write_text(json.dumps(envelope))
        (output / "report.json").write_text(json.dumps(report))
        stage = next(item for item in manifest["stages"] if item["id"] == "wrong_region")
        for index, frame in enumerate(stage["frames"]):
            for object_index, obj in enumerate(("fork", "spoon")):
                (output / f"wrong_region-{index}-{object_index}.json").write_text(json.dumps(results["wrong_region"][frame["sha256"]][obj]))
        return output / "report.json", manifest, directory

    def test_seed_loads_bound_recorded_mistake_and_blocks_all_prior_stage_frames(self):
        report, manifest, directory = self.make_recorded_seed()
        self.engine = InteractiveEngine(self.root, report, client_factory=lambda: self.client)
        state = self.engine.snapshot()
        self.assertEqual(("mistake", "fail", "recorded"), (state["phase"], state["status"], state["source_kind"]))
        self.assertTrue(state["stale"])
        self.assertIn("Historical", state["source_note"])
        self.assertIn("move-fork-right-to-left", state["proposal"]["action_ids"])
        for stage in manifest["stages"]:
            payload = self.payload()
            data = (directory / stage["frames"][0]["file"]).read_bytes()
            payload["images"][0]["data_url"] = "data:image/png;base64," + base64.b64encode(data).decode()
            with self.subTest(stage=stage["id"]), self.assertRaises(ValidationError):
                self.engine.inspect_pair(payload)
        self.assertEqual([], self.client.ledger)

    def test_seed_frame_changes_remove_reference_instead_of_trusting_old_report(self):
        report, manifest, directory = self.make_recorded_seed()
        (directory / manifest["stages"][0]["frames"][0]["file"]).write_bytes(picture(100))
        engine = InteractiveEngine(self.root, report, client_factory=lambda: self.client)
        state = engine.snapshot()
        self.assertEqual(("needs_evidence", "unknown", "none"), (state["phase"], state["status"], state["source_kind"]))
        self.assertEqual([], state["images"])
        self.assertIn("could not be verified", state["error"])

    def task_payload(self, objects=None):
        return {"title": "Reversed arrangement", "requirements": "Place fork in RIGHT and spoon in LEFT.",
                "objects": objects or ["fork", "spoon"], "confirm_unique_targets": True}

    def install_compiler(self, **kwargs):
        self.compiler = FakeCompileClient(**kwargs)
        self.engine._compile_client_factory = lambda: self.compiler

    def activate_payload(self, draft):
        return {"draft_id": draft["draft_id"], "draft_hash": draft["draft_hash"],
                "confirm_requirements": True, "confirm_unique_targets": True}

    def test_spark_draft_preserves_active_pass_until_exact_review(self):
        before = self.engine.inspect_pair(self.payload())
        self.install_compiler()
        state = self.engine.draft_task(self.task_payload())
        self.assertEqual("pass", state["status"])
        self.assertEqual(before["active_task"], state["active_task"])
        self.assertEqual(before["images"], state["images"])
        self.assertEqual(before["revision"], state["revision"])
        draft = state["task_draft"]
        self.assertFalse(draft["approved"])
        self.assertTrue(draft["requires_human_approval"])
        self.assertEqual("injected_compiler_not_verified_live", draft["origin"])
        self.assertEqual(1, len(self.compiler.ledger))
        self.assertIn('"predicate_kind": "location"', self.compiler.ledger[0]["prompt"])
        self.assertIsNone(state["draft_error"])
        self.assertNotIn("observations", draft)

    def test_activation_changes_spec_and_immediately_invalidates_old_evidence(self):
        self.engine.inspect_pair(self.payload())
        old_token = self.engine.begin_repair()["capture_token"]
        old_hash = self.engine.snapshot()["active_task"]["protocol_hash"]
        self.install_compiler()
        draft = self.engine.draft_task(self.task_payload())["task_draft"]
        state = self.engine.activate_task(self.activate_payload(draft))
        self.assertNotEqual(old_hash, state["active_task"]["protocol_hash"])
        self.assertEqual(draft["protocol_hash"], state["active_task"]["protocol_hash"])
        self.assertEqual(("awaiting_capture", "unknown", "none"), (state["phase"], state["status"], state["source_kind"]))
        self.assertEqual([], state["images"])
        self.assertIsNone(state["task_draft"])
        self.assertIsNone(state["proposal"])
        self.assertNotEqual(old_token, state["capture_token"])
        self.assertTrue(all(value["status"] == "unknown" for value in state["checks"]))
        self.assertEqual({"fork": "right", "spoon": "left"}, {value["object_id"]: value["expected_region"] for value in state["checks"]})
        self.assertTrue(state["active_task"]["human_reviewed"])
        self.assertEqual("local_operator", state["active_task"]["review_origin"])
        self.assertEqual(4, len(self.client.ledger))

    def test_activation_requires_current_id_hash_and_both_confirmations(self):
        self.install_compiler()
        draft = self.engine.draft_task(self.task_payload())["task_draft"]
        for field, value in (("draft_id", "wrong"), ("draft_hash", "0" * 64),
                             ("confirm_requirements", False), ("confirm_unique_targets", False)):
            payload = self.activate_payload(draft)
            payload[field] = value
            with self.subTest(field=field), self.assertRaises(ValidationError):
                self.engine.activate_task(payload)
        self.assertEqual("authored_default", self.engine.snapshot()["active_task"]["origin"])

    def test_assistant_activation_requires_private_authority_and_is_not_human_review(self):
        self.install_compiler()
        draft = self.engine.draft_task(self.task_payload())["task_draft"]
        payload = dict(self.activate_payload(draft), review_origin="assistant_under_user_authorization")
        with self.assertRaises(ValidationError):
            self.engine.activate_task(payload)
        payload["authorization_reference"] = "Explicit user instruction retained only in private approval"
        state = self.engine.activate_task(payload)
        self.assertTrue(state["active_task"]["reviewed"])
        self.assertFalse(state["active_task"]["human_reviewed"])
        self.assertEqual("assistant_under_user_authorization", state["active_task"]["review_origin"])
        self.assertNotIn(payload["authorization_reference"], json.dumps(state))
        approvals = list(self.engine._directory.glob("task-approval-*.json"))
        self.assertEqual(payload["authorization_reference"], json.loads(approvals[0].read_text())["authorization_reference"])

    def test_repeated_semantic_drafts_have_distinct_hashes_and_old_review_is_rejected(self):
        self.install_compiler()
        first = self.engine.draft_task(self.task_payload())["task_draft"]
        self.install_compiler()
        second = self.engine.draft_task(self.task_payload())["task_draft"]
        self.assertNotEqual(first["protocol_hash"], second["protocol_hash"])
        self.assertNotEqual(first["draft_hash"], second["draft_hash"])
        with self.assertRaises(ValidationError):
            self.engine.activate_task(self.activate_payload(first))

    def test_bad_draft_input_and_missing_unique_declaration_never_spend(self):
        self.install_compiler()
        for field, value in (("requirements", ""), ("requirements", "x" * 8193),
                             ("objects", ["fork"]), ("objects", ["fork", "fork"]),
                             ("objects", ["../key", "spoon"]), ("objects", [str(n) for n in range(7)]),
                             ("title", ""), ("confirm_unique_targets", False)):
            payload = self.task_payload()
            payload[field] = value
            with self.subTest(field=field, value=str(value)[:30]), self.assertRaises(ValidationError):
                self.engine.draft_task(payload)
        self.assertEqual([], self.compiler.ledger)
        self.assertFalse(self.engine.snapshot()["busy"])

    def test_failed_or_unsupported_draft_does_not_replace_task_or_erase_pass(self):
        before = self.engine.inspect_pair(self.payload())
        for kwargs in ({"failure": True}, {"unsupported": True}):
            self.install_compiler(**kwargs)
            state = self.engine.draft_task(self.task_payload())
            self.assertEqual("pass", state["status"])
            self.assertEqual(before["active_task"], state["active_task"])
            self.assertEqual(before["images"], state["images"])
            self.assertIsNone(state["task_draft"])
            self.assertTrue(state["draft_error"])
            self.assertFalse(state["busy"])
            self.assertNotIn("never-public", json.dumps(state))
            self.assertNotIn("Secret Spark", json.dumps(state))

    def test_task_busy_prevents_inspection_activation_and_second_draft(self):
        entered, release = threading.Event(), threading.Event()
        self.install_compiler(callback=lambda: (entered.set(), release.wait(3)))
        worker = threading.Thread(target=lambda: self.engine.draft_task(self.task_payload()))
        worker.start()
        self.assertTrue(entered.wait(2))
        try:
            state = self.engine.snapshot()
            self.assertEqual("task_draft", state["busy_operation"])
            self.assertEqual("unknown", state["status"])
            for mutation in (self.engine.new_inspection, lambda: self.engine.draft_task(self.task_payload()),
                             lambda: self.engine.activate_task({})):
                with self.assertRaises(ValidationError):
                    mutation()
        finally:
            release.set()
            worker.join(3)
        self.assertFalse(self.engine.snapshot()["busy"])

    def test_custom_three_object_task_uses_six_bound_sam_calls_and_generic_repair(self):
        objects = {"coffee-spoon": "right", "cup": "left", "fork": "left"}
        self.install_compiler(placements=objects)
        draft = self.engine.draft_task(self.task_payload(list(objects)))["task_draft"]
        self.engine.activate_task(self.activate_payload(draft))
        self.client = FakeClient()
        self.client.max_requests = 6
        state = self.engine.inspect_pair(self.payload())
        self.assertEqual(6, len(self.client.ledger))
        self.assertEqual("fail", state["status"])
        self.assertIn("move-coffee-spoon-left-to-right", state["proposal"]["action_ids"])
        self.assertIn("Move the coffee-spoon from LEFT to RIGHT.", state["proposal"]["steps"])
        self.assertEqual(set(objects), {value["object_id"] for value in state["checks"]})
        capture = next(self.engine._directory.glob("inspection-*/capture.json"))
        self.assertEqual(state["active_task"]["protocol_hash"], json.loads(capture.read_text())["protocol_hash"])

    def test_six_object_inventory_has_exact_twelve_request_budget(self):
        objects = {"cup": "left", "fork": "left", "spoon": "right", "plate": "left", "glass": "left", "napkin": "left"}
        self.install_compiler(placements=objects)
        draft = self.engine.draft_task(self.task_payload(list(objects)))["task_draft"]
        self.engine.activate_task(self.activate_payload(draft))
        self.client = FakeClient()
        self.client.max_requests = 12
        state = self.engine.inspect_pair(self.payload())
        self.assertEqual(12, len(self.client.ledger))
        self.assertEqual("pass", state["status"])
        self.assertIn("at most 12 SAM calls", state["trace"][-2]["label"])

    def test_compiler_client_must_have_one_request_no_retries_and_empty_ledger(self):
        for field, value in (("max_requests", 2), ("retries", 1), ("ledger", [{"used": True}])):
            self.install_compiler()
            setattr(self.compiler, field, value)
            state = self.engine.draft_task(self.task_payload())
            self.assertIsNone(state["task_draft"])
            self.assertTrue(state["draft_error"])
            self.assertFalse(state["busy"])

    def test_draft_storage_failure_releases_busy_and_never_changes_active_task(self):
        self.install_compiler()
        before = self.engine.snapshot()
        with patch.object(self.engine, "_persist", side_effect=OSError("Private disk failure")):
            state = self.engine.draft_task(self.task_payload())
        self.assertFalse(state["busy"])
        self.assertIsNone(state["busy_operation"])
        self.assertEqual(before["active_task"], state["active_task"])
        self.assertIsNone(state["task_draft"])
        self.assertEqual([], self.compiler.ledger)

    def test_failed_approval_storage_leaves_old_task_intact_and_no_new_token(self):
        self.install_compiler()
        state = self.engine.draft_task(self.task_payload())
        before = state["active_task"]
        with patch.object(self.engine, "_write", side_effect=OSError("Private approval failure")):
            with self.assertRaises(OSError):
                self.engine.activate_task(self.activate_payload(state["task_draft"]))
        self.assertEqual(before, self.engine.snapshot()["active_task"])
        self.assertIsNone(self.engine.snapshot()["capture_token"])

    def test_missing_fork_requests_only_fork_evidence_without_guessing_absence_or_repair(self):
        def missing_fork(value):
            if self.client.ledger[-1]["concept"] == "fork":
                value["objects"] = []
            return value
        self.client = FakeClient(transform=missing_fork)
        state = self.engine.inspect_pair(self.payload())
        self.assertEqual("unknown", state["status"])
        request = state["inspection_request"]
        self.assertEqual(["fork"], request["object_ids"])
        self.assertIn("entire fork", " ".join(request["instructions"]))
        self.assertIsNone(state["proposal"])
        text = json.dumps(request).lower()
        for unsupported_claim in ("fork is absent", "fork is hidden", "move the fork"):
            self.assertNotIn(unsupported_claim, text)
        fork = next(check for check in state["checks"] if check["object_id"] == "fork")
        self.assertIn("do not prove absence", fork["grounding_reason"])
        self.assertEqual("pass", next(check for check in state["checks"] if check["object_id"] == "spoon")["status"])

    def test_region_boundary_uncertainty_requests_calibration_review_not_a_physical_move(self):
        def straddling(value):
            if self.client.ledger[-1]["concept"] == "fork":
                value["objects"][0]["box"].update(x1=40, x2=60)
            return value
        self.client = FakeClient(transform=straddling)
        state = self.engine.inspect_pair(self.payload())
        request = state["inspection_request"]
        self.assertEqual("region_support_unconfirmed", request["reason"])
        self.assertEqual(["fork"], request["object_ids"])
        self.assertIn("rectangles", " ".join(request["instructions"]))
        self.assertIsNone(state["proposal"])

    def test_multiple_instances_request_identity_contract_review_instead_of_choosing_one(self):
        def multiple(value):
            if self.client.ledger[-1]["concept"] == "fork":
                extra = detection("right")
                extra["object_id"] = 1
                value["objects"].append(extra)
            return value
        self.client = FakeClient(transform=multiple)
        state = self.engine.inspect_pair(self.payload())
        self.assertEqual("target_identity_ambiguous", state["inspection_request"]["reason"])
        self.assertEqual(["fork"], state["inspection_request"]["object_ids"])
        self.assertIn("exactly one", " ".join(state["inspection_request"]["instructions"]))
        self.assertIsNone(state["proposal"])

    def test_unstable_pair_requests_consistent_fresh_views_not_repair(self):
        def changed_fork(value):
            if len(self.client.ledger) == 3:
                value["objects"] = [detection("right")]
            return value
        self.client = FakeClient(transform=changed_fork)
        state = self.engine.inspect_pair(self.payload())
        self.assertEqual("stable_pair_required", state["inspection_request"]["reason"])
        self.assertEqual(["fork"], state["inspection_request"]["object_ids"])
        self.assertIn("unchanged between views", " ".join(state["inspection_request"]["instructions"]))
        self.assertIsNone(state["proposal"])

    def test_low_confidence_asks_for_clearer_evidence_and_success_clears_request(self):
        def low_confidence(value):
            if self.client.ledger[-1]["concept"] == "fork":
                value["objects"][0]["confidence"] = 0.5
            return value
        self.client = FakeClient(transform=low_confidence)
        state = self.engine.inspect_pair(self.payload())
        self.assertEqual("confidence_below_threshold", state["inspection_request"]["reason"])
        self.assertIsNone(state["proposal"])
        self.client = FakeClient()
        state = self.engine.inspect_pair(self.payload(index=3))
        self.assertEqual("pass", state["status"])
        self.assertIsNone(state["inspection_request"])

    def test_expiration_and_task_activation_request_new_evidence_for_all_required_objects(self):
        with patch("proofloop.interactive._now_ms", return_value=10000):
            self.engine.inspect_pair(self.payload())
        with patch("proofloop.interactive._now_ms", return_value=10001 + FRESHNESS_MS):
            state = self.engine.snapshot()
        self.assertEqual("evidence_expired", state["inspection_request"]["reason"])
        self.assertEqual({"fork", "spoon"}, set(state["inspection_request"]["object_ids"]))
        self.install_compiler(placements={"cup": "left", "plate": "right"})
        draft = self.engine.draft_task(self.task_payload(["cup", "plate"]))["task_draft"]
        state = self.engine.activate_task(self.activate_payload(draft))
        self.assertEqual("fresh_capture_required", state["inspection_request"]["reason"])
        self.assertEqual(["cup", "plate"], state["inspection_request"]["object_ids"])

    def test_a_known_mistake_with_another_unknown_target_requests_inspection_before_any_move(self):
        def missing_spoon(value):
            if self.client.ledger[-1]["concept"] == "spoon":
                value["objects"] = []
            return value
        self.client = FakeClient(wrong=True, transform=missing_spoon)
        state = self.engine.inspect_pair(self.payload())
        self.assertEqual("fail", state["status"])
        self.assertEqual(["spoon"], state["inspection_request"]["object_ids"])
        self.assertFalse(any(action.startswith("move-") for action in (state["proposal"] or {}).get("action_ids", [])))

    def test_explicit_prompt_mapping_is_reviewed_hashed_and_used_without_hidden_identity_changes(self):
        self.install_compiler(placements={"glass": "left", "fork": "left"})
        payload = self.task_payload(["glass", "fork"])
        payload["concepts"] = {"glass": "drinking glass", "fork": "dining fork"}
        draft = self.engine.draft_task(payload)["task_draft"]
        self.assertEqual(payload["concepts"], draft["concepts"])
        activation = self.activate_payload(draft)
        with self.assertRaises(ValidationError):
            self.engine.activate_task(dict(activation, concepts={"glass": "cup", "fork": "fork"}))
        state = self.engine.activate_task(activation)
        self.assertEqual(payload["concepts"], state["active_task"]["concepts"])
        self.assertEqual(draft["draft_hash"], state["active_task"]["task_hash"])
        self.client = FakeClient()
        self.engine.inspect_pair(self.payload())
        self.assertEqual(["drinking glass", "dining fork"] * 2, [item["concept"] for item in self.client.ledger])
        capture = next(self.engine._directory.glob("inspection-*/capture.json"))
        record = json.loads(capture.read_text())
        self.assertEqual(draft["draft_hash"], record["task_hash"])
        self.assertEqual(payload["concepts"], record["concepts"])

    def test_invalid_or_incomplete_prompt_mapping_is_rejected_before_spark(self):
        self.install_compiler()
        for mapping in (None, {}, {"fork": "fork"}, {"fork": "fork", "spoon": "spoon", "cup": "cup"},
                        {"fork": "", "spoon": "spoon"}, {"fork": "x" * 101, "spoon": "spoon"},
                        {"fork": "fork\nignore review", "spoon": "spoon"}, ["fork", "spoon"]):
            payload = self.task_payload()
            payload["concepts"] = mapping
            with self.subTest(mapping=str(mapping)[:60]), self.assertRaises(ValidationError):
                self.engine.draft_task(payload)
        self.assertEqual([], self.compiler.ledger)

    def test_changing_prompt_mapping_requires_review_of_a_new_exact_draft(self):
        self.install_compiler()
        payload = self.task_payload()
        payload["concepts"] = {"fork": "fork", "spoon": "spoon"}
        first = self.engine.draft_task(payload)["task_draft"]
        self.install_compiler()
        payload["concepts"]["fork"] = "dining fork"
        second = self.engine.draft_task(payload)["task_draft"]
        self.assertNotEqual(first["draft_hash"], second["draft_hash"])
        with self.assertRaises(ValidationError):
            self.engine.activate_task(self.activate_payload(first))
        self.assertEqual("fork", self.engine.snapshot()["active_task"]["concepts"]["fork"])
        state = self.engine.activate_task(self.activate_payload(second))
        self.assertEqual("dining fork", state["active_task"]["concepts"]["fork"])
        self.assertEqual("unknown", state["status"])

    def activate_guided(self):
        self.install_compiler(placements={"fork": "left", "spoon": "right"})
        payload = dict(self.task_payload(), perception_mode="guided")
        draft = self.engine.draft_task(payload)["task_draft"]
        return draft, self.engine.activate_task(self.activate_payload(draft))

    def guided_client(self, *, locator_transform=None, sam_transform=None):
        def located(value, paths):
            for frame in value["parsed"]["frames"]:
                frame["height"] = 100
                for target in frame["objects"]:
                    target["candidates"] = [{"x1": 60 if target["object_id"] == "spoon" else 10,
                                               "y1": 10, "x2": 80 if target["object_id"] == "spoon" else 30, "y2": 30}]
            if locator_transform:
                locator_transform(value, paths)
        def refined(value, path):
            prompt = self.client.calls[-1][2]
            box = value["objects"][0]["box"]
            box["height"] = 100
            for key in ("x1", "y1", "x2", "y2"):
                box[key] = int(re.search(key + r"=(\d+)", prompt)[1])
            if sam_transform:
                sam_transform(value, path)
            return value
        client = GuidedFakeClient(concepts={"fork": "fork", "spoon": "spoon"}, transform=located, sam_transform=refined)
        client.max_requests = 5
        client.usage_summary = lambda: {"request_attempts": len(client.ledger), "estimate_origin": "authored_mock_no_api_calls"}
        self.client = client
        return client

    def guided_payload(self, *, offset=0):
        payload = self.payload()
        # Header-only transport fixtures, not real pictures or model evidence.
        for index, item in enumerate(payload["images"]):
            encoded = jpeg_header(width=100, height=100, suffix=b"")
            encoded = encoded[:2] + b"\xff\xfe\x00\x04" + bytes([index + offset, index + offset]) + encoded[2:]
            item["data_url"] = "data:image/jpeg;base64," + base64.b64encode(encoded).decode()
        return payload

    def test_guided_mode_requires_an_exact_new_review_and_invalidates_previous_evidence(self):
        self.engine.inspect_pair(self.payload())
        old = self.engine.snapshot()["active_task"]
        draft, state = self.activate_guided()
        self.assertEqual("concept", old["perception_mode"])
        self.assertEqual("guided", draft["perception_mode"])
        self.assertIn("does not independently verify identity", draft["perception_note"])
        self.assertEqual("guided", state["active_task"]["perception_mode"])
        self.assertEqual(draft["draft_hash"], state["active_task"]["task_hash"])
        self.assertNotEqual(old["protocol_hash"], state["active_task"]["protocol_hash"])
        self.assertEqual(("unknown", []), (state["status"], state["images"]))
        self.assertTrue(state["capture_token"])
        self.assertIsNone(state["perception_provenance"])

    def test_guided_runs_one_task_independent_locator_plus_four_actual_sam_calls(self):
        self.activate_guided()
        client = self.guided_client()
        state = self.engine.inspect_pair(self.guided_payload())
        self.assertEqual("pass", state["status"])
        self.assertEqual(["spark", "sam", "sam", "sam", "sam"], [value[0] for value in client.calls])
        self.assertEqual("injected_client_not_verified_live", state["inference_origin"])
        self.assertIn("Spark assigns object identity", state["source_note"])
        self.assertIn("without independently verifying identity", state["perception_provenance"]["identity_basis"])
        self.assertTrue(state["perception_provenance"]["task_independent_locator"])
        self.assertNotIn("Place fork", client.calls[0][1])
        self.assertNotIn('"regions"', client.calls[0][1])
        self.assertTrue(all(len(image["boxes"]) == 2 for image in state["images"]))
        capture = next(self.engine._directory.glob("inspection-*/capture.json"))
        self.assertEqual("guided", json.loads(capture.read_text())["perception_mode"])
        self.assertTrue(next(self.engine._directory.glob("inspection-*/guided.json")).is_file())

    def test_missing_guided_anchors_remain_unknown_without_fabricated_sam_completion(self):
        self.activate_guided()
        def missing(value, paths):
            for frame in value["parsed"]["frames"]:
                for target in frame["objects"]:
                    if target["object_id"] == "fork":
                        target["candidates"] = []
        client = self.guided_client(locator_transform=missing)
        state = self.engine.inspect_pair(self.guided_payload())
        self.assertEqual("unknown", state["status"])
        self.assertEqual(["spark", "sam", "sam"], [value[0] for value in client.calls])
        self.assertEqual(["fork"], state["inspection_request"]["object_ids"])
        self.assertEqual(2, state["perception_provenance"]["missing_or_ambiguous_anchors"])
        self.assertIsNone(state["proposal"])
        self.assertFalse(list(self.engine._directory.glob("inspection-*/result-*-fork.json")))

    def test_guided_wrong_locator_source_or_sam_source_stops_without_pass(self):
        for index, which in enumerate(("locator", "sam")):
            self.engine = InteractiveEngine(self.root, client_factory=lambda: self.client)
            self.activate_guided()
            def wrong_source(value, paths):
                value["sources"][0]["sha256"] = "0" * 64
            def wrong_sam(value, path):
                value["source"]["sha256"] = "0" * 64
            client = self.guided_client(locator_transform=wrong_source if which == "locator" else None,
                                        sam_transform=wrong_sam if which == "sam" else None)
            state = self.engine.inspect_pair(self.guided_payload(offset=10 * index))
            self.assertEqual(("error", "unknown"), (state["phase"], state["status"]))
            self.assertEqual(1 if which == "locator" else 2, len(client.calls))
            self.assertIsNone(state["verified_at_ms"])

    def test_guided_requires_five_request_budget_and_supported_images_before_paid_calls(self):
        self.activate_guided()
        client = self.guided_client()
        with self.assertRaisesRegex(ValidationError, "upright JPEG"):
            self.engine.inspect_pair(self.payload())
        self.assertEqual([], client.calls)
        client.max_requests = 4
        state = self.engine.inspect_pair(self.guided_payload())
        self.assertEqual("error", state["phase"])
        self.assertEqual([], client.calls)

    def test_invalid_or_unreviewed_perception_modes_cannot_silently_change_active_policy(self):
        self.install_compiler()
        for mode in (None, "automatic", "guided_fallback", True):
            with self.subTest(mode=mode), self.assertRaises(ValidationError):
                self.engine.draft_task(dict(self.task_payload(), perception_mode=mode))
        self.assertEqual([], self.compiler.ledger)
        self.assertEqual("concept", self.engine.snapshot()["active_task"]["perception_mode"])
        draft = self.engine.draft_task(self.task_payload())["task_draft"]
        with self.assertRaises(ValidationError):
            self.engine.activate_task(dict(self.activate_payload(draft), perception_mode="guided"))


if __name__ == "__main__":
    unittest.main()
