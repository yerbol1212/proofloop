"""Offline interactive state/anti-staleness tests; no calls to Meta or camera claims.

All encoded pictures, mask records and even recorded-report provenance below
are authored fixtures. They test the local contracts, not model performance.
"""
import base64
from copy import deepcopy
import hashlib
import json
from pathlib import Path
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


if __name__ == "__main__":
    unittest.main()
