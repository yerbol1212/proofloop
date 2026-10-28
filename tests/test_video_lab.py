"""Read-only recorded lab tests with authored, nonphotographic fixtures."""
from copy import deepcopy
import hashlib
import http.client
import json
from pathlib import Path
import struct
import tempfile
import threading
import unittest

from proofloop.benchmark import approve_manifest, evaluate_sam, evaluate_spark, manifest_hash, public_summary
from proofloop.server import create_server
from proofloop.video_lab import VideoLab
from tests.test_benchmark import approved, media_fixture, sam_results, spark_result


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def fixture(root):
    envelope = media_fixture(root, approved())
    contract = deepcopy(envelope["contract"])
    for index, frame in enumerate(frame for case in contract["cases"] for frame in case["frames"]):
        # Minimal authored JPEG header for dimension parsing, not a photograph.
        data = (b"\xff\xd8\xff\xfe\x00\x03" + bytes([index]) + b"\xff\xc0\x00\x11\x08" +
                struct.pack(">HH", 480, 640) + b"\x03\x01\x11\x00\x02\x11\x00\x03\x11\x00\xff\xd9")
        (root / frame["file"]).write_bytes(data)
        frame["sha256"] = hashlib.sha256(data).hexdigest()
    envelope = approve_manifest(contract, manifest_hash(contract), confirm_unique_targets=True)
    raw_spark = {case["id"]: spark_result(case, case["annotation"]["status"]) for case in contract["cases"]}
    report = {"schema_version": "1.0", "experiment": "licensed_recorded_video_comparison",
              "contract_hash": envelope["contract_hash"], "status": "completed", "inference_origin": "authored_fixture",
              "evidence_origin": "external_recorded_video", "conditions": {
                  "sam": {"status": "completed", "cases": evaluate_sam(envelope, sam_results(envelope, empty=True)),
                          "metrics": {}, "usage": {"estimated_usd": .01}, "wall_seconds": 1},
                  "spark": {"status": "completed", "cases": evaluate_spark(envelope, raw_spark),
                            "metrics": {}, "usage": {"estimated_usd": .02}, "wall_seconds": 2}}}
    run = root / "artifacts/local/online-benchmark-fixture"
    save(run / "approved-manifest.json", envelope)
    save(run / "report.json", report)
    summary = root / "artifacts/feasibility/online-fixture.json"
    save(summary, public_summary(envelope, report))
    return envelope, report, run, summary


class VideoLabTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.envelope, self.report, self.run, self.summary = fixture(self.root)
        self.lab = VideoLab(self.root)

    def tearDown(self):
        self.temp.cleanup()

    def test_source_bound_reviewed_run_projects_statuses_and_whitelisted_images(self):
        state = self.lab.snapshot()
        self.assertTrue(state["available"])
        self.assertEqual(1, len(state["runs"]))
        run = state["runs"][0]
        self.assertEqual("external_recorded_video", run["source_kind"])
        self.assertEqual(1, run["source_count"])
        self.assertEqual("unknown", run["cases"][0]["sam"])
        self.assertEqual("fail", run["cases"][0]["spark"])
        self.assertIn("Zero or unavailable detections", run["cases"][0]["reasons"]["sam"])
        image = run["cases"][0]["images"][0]
        self.assertEqual((640, 480), (image["width"], image["height"]))
        token = image["url"].rsplit("/", 1)[1]
        self.assertTrue(self.lab.media_path(token).is_file())
        self.assertNotIn("inputs/local", json.dumps(state))
        self.assertNotIn("authored-sam-shape", json.dumps(state))
        self.assertNotIn("sam-grounding://", json.dumps(state))

    def test_private_only_or_changed_public_summary_is_not_shown(self):
        self.summary.unlink()
        self.assertFalse(self.lab.snapshot()["available"])
        _, _, _, summary = fixture(self.root)
        data = json.loads(summary.read_text())
        data["conditions"]["sam"]["cases"][0]["status"] = "pass"
        save(summary, data)
        self.assertFalse(self.lab.snapshot()["available"])

    def test_media_change_fails_closed_and_stale_token_cannot_read_other_files(self):
        state = self.lab.snapshot()
        token = state["runs"][0]["cases"][0]["images"][0]["url"].rsplit("/", 1)[1]
        image = self.lab.media_path(token)
        image.write_bytes(b"changed")
        self.assertFalse(self.lab.snapshot()["available"])
        with self.assertRaises(KeyError):
            self.lab.media_path(token)
        for value in ("../.env.local", "unknown", "0" * 40):
            with self.assertRaises(KeyError):
                self.lab.media_path(value)

    def test_optional_guided_condition_keeps_its_identity_limit_and_unknown_result(self):
        report = deepcopy(self.report)
        report["conditions"]["guided"] = deepcopy(report["conditions"]["sam"])
        report["conditions"]["guided"]["cases"][0]["grounding"]["evidence"]["check_grounding"][0]["reason"] = (
            "No source images were supplied for this object")
        save(self.run / "report.json", report)
        save(self.summary, public_summary(self.envelope, report))
        run = self.lab.snapshot()["runs"][0]
        self.assertEqual("unknown", run["cases"][0]["guided"])
        self.assertIn("guided", run["cases"][0]["reasons"])
        self.assertIn("supported guided mask measurements", run["cases"][0]["reasons"]["guided"])
        self.assertNotIn("No source images", run["cases"][0]["reasons"]["guided"])
        self.assertEqual(2, len(run["cases"][0]["images"]))
        self.assertIn("does not independently prove", run["conditions"]["guided"]["method_note"])
        self.assertTrue(any("does not independently prove" in limit for limit in run["limits"]))

    def test_original_video_change_invalidates_a_cached_projection(self):
        self.assertTrue(self.lab.snapshot()["available"])
        video = self.root / self.envelope["contract"]["sources"][0]["file"]
        video.write_bytes(b"changed source recording")
        self.assertFalse(self.lab.snapshot()["available"])

    def test_report_hash_or_exact_source_pair_mismatch_prevents_projection(self):
        report = deepcopy(self.report)
        report["conditions"]["sam"]["cases"][0]["frame_hashes"].reverse()
        save(self.run / "report.json", report)
        # Regenerating a public summary does not conceal a bad private binding.
        save(self.summary, public_summary(self.envelope, report))
        self.assertFalse(self.lab.snapshot()["available"])
        report["contract_hash"] = "0" * 64
        save(self.run / "report.json", report)
        self.assertFalse(self.lab.snapshot()["available"])

    def test_explicit_paths_cannot_escape_fixed_roots_or_follow_symlinks(self):
        outside = self.root / "report.json"
        save(outside, self.report)
        self.assertFalse(VideoLab(self.root, report_paths=[outside]).snapshot()["available"])
        link = self.root / "artifacts/local/online-benchmark-link/report.json"
        link.parent.mkdir(parents=True)
        link.symlink_to(outside)
        self.assertFalse(VideoLab(self.root, report_paths=[link]).snapshot()["available"])

    def test_snapshot_is_a_copy_and_does_not_mutate_engine_or_make_requests(self):
        state = self.lab.snapshot()
        state["runs"][0]["cases"].clear()
        self.assertEqual(2, len(self.lab.snapshot()["runs"][0]["cases"]))

    def test_model_reason_is_plain_bounded_and_private_diagnostics_are_redacted(self):
        report = deepcopy(self.report)
        report["conditions"]["spark"]["cases"][0]["checks"][0]["reason"] = "/Users/private/key META_API_KEY"
        save(self.run / "report.json", report)
        state = self.lab.snapshot()
        self.assertTrue(state["available"])
        self.assertNotIn("/Users/private", json.dumps(state))
        self.assertIn("withheld", state["runs"][0]["cases"][0]["reasons"]["spark"])

    def test_loopback_routes_are_read_only_and_preserve_http_boundary(self):
        class Engine:
            def snapshot(self):
                return {"busy": False}
            def media_path(self, token):
                raise KeyError(token)
        web = self.root / "web/dist"
        web.mkdir(parents=True)
        (web / "index.html").write_text("Fixture")
        server = create_server(Engine(), web_root=web, video_lab=self.lab)
        thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": .01}, daemon=True)
        thread.start()
        def request(method, path, host=None):
            conn = http.client.HTTPConnection("127.0.0.1", server.server_address[1], timeout=3)
            conn.request(method, path, headers={"Host": host or f"127.0.0.1:{server.server_address[1]}"})
            response = conn.getresponse()
            body = response.read()
            result = (response.status, dict(response.getheaders()), body)
            conn.close()
            return result
        try:
            status, headers, body = request("GET", "/api/lab")
            self.assertEqual(200, status)
            self.assertEqual("no-store", headers["Cache-Control"])
            image_url = json.loads(body)["runs"][0]["cases"][0]["images"][0]["url"]
            self.assertEqual(200, request("GET", image_url)[0])
            self.assertEqual(b"", request("HEAD", image_url)[2])
            self.assertEqual(404, request("GET", "/api/lab/media/unknown")[0])
            self.assertEqual(404, request("GET", "/api/lab/media/%2e%2e/.env.local")[0])
            self.assertEqual(403, request("GET", "/api/lab", host="foreign.example")[0])
            self.assertEqual(403, request("POST", "/api/lab")[0])
        finally:
            server.shutdown()
            server.server_close()
            thread.join(2)


if __name__ == "__main__":
    unittest.main()
