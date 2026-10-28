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
from unittest.mock import patch

from proofloop.benchmark import approve_manifest, comparison_metrics, evaluate_sam, evaluate_spark, manifest_hash, public_summary
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
    for condition in report["conditions"].values():
        condition["metrics"] = comparison_metrics(envelope, condition["cases"])
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
        evaluation = run["evaluation"]
        self.assertEqual(len(self.envelope["contract"]["cases"]), evaluation["benchmark_case_count"])
        self.assertTrue(evaluation["pairwise"])
        self.assertFalse(evaluation["physical_action_execution_verified"])

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

    def test_normalized_version_and_declared_role_control_titles_without_relabeling_legacy(self):
        self.assertEqual("Text-prompt development pilot", self.lab.snapshot()["runs"][0]["title"])
        report = deepcopy(self.report)
        report["conditions"]["guided"] = deepcopy(report["conditions"]["sam"])
        report["conditions"]["guided"]["observation_version"] = "normalized_boxes_v2"
        for role, title in (("development_reprocessing", "Normalized grounding — development replay"),
                            ("new_source_holdback", "Normalized grounding — new sources")):
            report["evaluation_role"] = role
            save(self.run / "report.json", report)
            save(self.summary, public_summary(self.envelope, report))
            run = self.lab.snapshot()["runs"][0]
            self.assertEqual(title, run["title"])
            self.assertEqual(role, run["evaluation_role"])
            self.assertEqual("normalized_boxes_v2", run["conditions"]["guided"]["observation_version"])
        del report["evaluation_role"]
        del report["conditions"]["guided"]["observation_version"]
        save(self.run / "report.json", report)
        save(self.summary, public_summary(self.envelope, report))
        run = self.lab.snapshot()["runs"][0]
        self.assertEqual("Guided perception comparison", run["title"])
        self.assertNotIn("evaluation_role", run)
        self.assertNotIn("observation_version", run["conditions"]["guided"])

    def test_private_check_or_metric_tampering_is_rejected_without_republishing_summary(self):
        for mutation in (lambda r: r["conditions"]["sam"]["cases"][0]["checks"][0].update(status="pass"),
                         lambda r: r["conditions"]["sam"]["metrics"].update(accuracy=1)):
            report = deepcopy(self.report)
            mutation(report)
            save(self.run / "report.json", report)
            self.assertFalse(self.lab.snapshot()["available"])

    def test_sidecar_changes_invalidate_cached_projection_and_are_replayed(self):
        self.assertTrue(self.lab.snapshot()["available"])
        raw = sam_results(self.envelope, empty=True)
        sidecar = self.run / "sam/results.json"
        save(sidecar, raw)
        run = self.lab.snapshot()["runs"][0]
        self.assertTrue(run["report_integrity"]["metrics_recomputed"])
        self.assertEqual(1, run["report_integrity"]["inference_sidecars_replayed"])
        self.assertEqual(2, run["report_integrity"]["grounded_decisions_replayed"])
        # The same UNKNOWN verdict from a different/no input is insufficient.
        raw.pop("initial")
        save(sidecar, raw)
        self.assertFalse(self.lab.snapshot()["available"])

    def test_original_video_change_invalidates_a_cached_projection(self):
        self.assertTrue(self.lab.snapshot()["available"])
        video = self.root / self.envelope["contract"]["sources"][0]["file"]
        video.write_bytes(b"changed source recording")
        self.assertFalse(self.lab.snapshot()["available"])

    def test_report_hash_or_exact_source_pair_mismatch_prevents_projection(self):
        report = deepcopy(self.report)
        report["conditions"]["sam"]["cases"][0]["frame_hashes"].reverse()
        save(self.run / "report.json", report)
        # Export itself now rejects the bad binding before publication.
        from proofloop.core import ValidationError
        with self.assertRaises(ValidationError):
            public_summary(self.envelope, report)
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

    def _fixed_followup_fixture(self):
        # The controller uses real source-bound fixture observations, masks,
        # sidecars and request ledgers. Only JPEG decoding/locator transport is
        # replaced because these authored fixture bytes are not photographs.
        from proofloop.followup import replay_fixed_followup
        from tests.test_followup import fixture as followup_fixture
        envelope, report, folder, saved_replay = followup_fixture(self.root)
        report.update(schema_version="1.0", experiment="licensed_recorded_video_comparison",
                      evidence_origin="external_recorded_video")
        save(folder / "approved-manifest.json", envelope)
        save(folder / "report.json", report)
        with patch("proofloop.followup.guided_segment_pair", side_effect=saved_replay):
            replay = replay_fixed_followup(envelope, report, folder, self.root)
        summary = public_summary(envelope, report)
        summary["followup_policy_replay"] = replay
        public_path = self.root / "artifacts/feasibility/online-followup-fixture.json"
        save(public_path, summary)
        lab = VideoLab(self.root, report_paths=[folder / "report.json"], summary_paths=[public_path])
        return lab, folder, public_path, replay, saved_replay

    def test_followup_summary_extra_is_accepted_only_after_source_bound_policy_replay(self):
        lab, _, public_path, replay, saved_replay = self._fixed_followup_fixture()
        public = json.loads(public_path.read_text())
        public["unreviewed_note"] = "/private/unapproved/provider-body"
        save(public_path, public)
        with patch("proofloop.video_lab._dimensions", return_value=(100, 100)), \
                patch("proofloop.followup.guided_segment_pair", side_effect=saved_replay):
            state = lab.snapshot()
        self.assertTrue(state["available"])
        run = state["runs"][0]
        self.assertEqual("Specific evidence requests — fixed future pairs", run["title"])
        self.assertEqual(replay, run["followup_policy_replay"])
        self.assertEqual(1, replay["conditions"]["guided"]["resolved_trajectories"])
        self.assertFalse(replay["physical_action_execution_verified"])
        self.assertNotIn("unreviewed_note", json.dumps(state))
        self.assertNotIn("/private/unapproved", json.dumps(state))

    def test_tampered_followup_totals_hide_the_run_even_when_base_summary_matches(self):
        lab, _, public_path, _, saved_replay = self._fixed_followup_fixture()
        with patch("proofloop.video_lab._dimensions", return_value=(100, 100)), \
                patch("proofloop.followup.guided_segment_pair", side_effect=saved_replay):
            self.assertTrue(lab.snapshot()["available"])
            public = json.loads(public_path.read_text())
            public["followup_policy_replay"]["conditions"]["guided"]["resolved_trajectories"] = 999
            save(public_path, public)
            self.assertFalse(lab.snapshot()["available"])

    def test_missing_followup_policy_is_not_shown_as_a_reviewed_controller_run(self):
        lab, _, public_path, _, saved_replay = self._fixed_followup_fixture()
        public = json.loads(public_path.read_text())
        del public["followup_policy_replay"]
        save(public_path, public)
        with patch("proofloop.video_lab._dimensions", return_value=(100, 100)), \
                patch("proofloop.followup.guided_segment_pair", side_effect=saved_replay):
            self.assertFalse(lab.snapshot()["available"])

    def test_changed_pair_sidecar_invalidates_cached_followup_projection(self):
        lab, folder, _, _, saved_replay = self._fixed_followup_fixture()
        with patch("proofloop.video_lab._dimensions", return_value=(100, 100)), \
                patch("proofloop.followup.guided_segment_pair", side_effect=saved_replay):
            self.assertTrue(lab.snapshot()["available"])
            # Aggregate predictions and results.json still match. A changed
            # saved request latency breaks the independent usage-ledger replay.
            sidecar = folder / "guided/coffee-e00-initial.json"
            pair = json.loads(sidecar.read_text())
            pair["locator"]["latency_seconds"] += .1
            save(sidecar, pair)
            self.assertFalse(lab.snapshot()["available"])

    def test_candy_transfer_title_preserves_development_scope(self):
        contract = deepcopy(self.envelope["contract"])
        contract["dataset_id"] = "proofloop-candy-new-task-transfer-development-20261028"
        envelope = approve_manifest(contract, manifest_hash(contract), confirm_unique_targets=True)
        report = deepcopy(self.report)
        report.update(contract_hash=envelope["contract_hash"], evaluation_role="development_reprocessing")
        save(self.run / "approved-manifest.json", envelope)
        save(self.run / "report.json", report)
        save(self.summary, public_summary(envelope, report))
        run = self.lab.snapshot()["runs"][0]
        self.assertEqual("New task transfer — candy collection", run["title"])
        self.assertEqual("development_reprocessing", run["evaluation_role"])

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
