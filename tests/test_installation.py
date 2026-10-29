import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from proofloop import cli, installation, replay_bundle


class InstallationTests(unittest.TestCase):
    def test_explicit_and_environment_data_directories_override_checkout(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder).resolve()
            with patch.dict(os.environ, {"PROOFLOOP_DATA_DIR": str(path / "env")}):
                self.assertEqual(path / "explicit", installation.data_root(path / "explicit"))
                self.assertEqual(path / "env", installation.data_root())

    def test_installed_runtime_uses_user_data_not_site_packages(self):
        with patch.object(installation, "checkout_root", return_value=None), patch.dict(
                os.environ, {"XDG_DATA_HOME": "/tmp/authored-user-data"}):
            os.environ.pop("PROOFLOOP_DATA_DIR", None)
            self.assertEqual(Path("/tmp/authored-user-data/proofloop").resolve(), installation.data_root())

    def test_doctor_has_no_network_key_output_or_storage_side_effects(self):
        with tempfile.TemporaryDirectory() as folder, patch("proofloop.meta.MetaClient._request", side_effect=AssertionError("no API")), patch(
                "socket.socket.connect", side_effect=AssertionError("no network")):
            path = Path(folder) / "uncreated" / "data"
            with patch.dict(os.environ, {"META_API_KEY": "private-fixture-must-not-print"}):
                report = installation.doctor(path)
            self.assertFalse(path.exists())
            self.assertTrue(report["environment_key_configured"])
            self.assertFalse(report["model_authentication_checked"])
            self.assertEqual(0, report["provider_requests"])
            self.assertNotIn("private-fixture", json.dumps(report))
            self.assertNotIn(str(path), json.dumps(report))
            self.assertTrue(report["offline_replay_valid"])
            self.assertFalse(report["optional_live_prerequisites"]["model_authentication_checked"])
            self.assertFalse(report["optional_live_prerequisites"]["provider_connectivity_checked"])

    def test_doctor_reports_missing_frontend_and_binaries(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(installation, "web_root", return_value=Path(folder)), patch(
                "proofloop.installation.shutil.which", return_value=None), patch("proofloop.installation.os.access", return_value=False):
            report = installation.doctor(folder)
            self.assertFalse(report["frontend_available"])
            self.assertFalse(report["offline_workbench_ready"])
            self.assertFalse(report["ffmpeg_available"])
            self.assertFalse(report["ready"])
            self.assertIn("frontend_unavailable", report["reason_codes"])
            self.assertIn("data_directory_unwritable", report["reason_codes"])

    def test_missing_frontend_fails_default_workbench_but_explicit_core_passes(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(installation, "web_root", return_value=Path(folder)):
            report = installation.doctor(folder)
            self.assertEqual("workbench", report["profile"])
            self.assertFalse(report["ready"])
            self.assertEqual(["frontend_unavailable"], report["reason_codes"])
            core = installation.doctor(folder, profile="core")
            self.assertTrue(core["ready"])
            self.assertFalse(core["offline_workbench_ready"])
            self.assertEqual([], core["reason_codes"])
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(2, cli.main(["doctor", "--data-dir", folder]))
                self.assertEqual(0, cli.main(["doctor", "--profile", "core", "--data-dir", folder]))

    def test_complete_workbench_passes_without_optional_live_dependencies(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            (path / "index.html").write_text("<!doctype html><title>Authored UI fixture</title>")
            with patch.object(installation, "web_root", return_value=path), patch(
                    "proofloop.installation.shutil.which", return_value=None), patch(
                    "proofloop.installation.importlib.util.find_spec", return_value=None), patch.dict(
                    os.environ, {}, clear=True):
                with contextlib.redirect_stdout(io.StringIO()) as output:
                    self.assertEqual(0, cli.main(["doctor", "--data-dir", folder]))
                report = json.loads(output.getvalue())
            self.assertTrue(report["ready"])
            self.assertTrue(report["offline_workbench_ready"])
            self.assertTrue(report["offline_replay_valid"])
            self.assertFalse(report["sam_parser_discoverable"])
            self.assertFalse(report["environment_key_configured"])
            self.assertFalse(report["optional_live_prerequisites"]["required_for_selected_profile"])
            self.assertFalse(report["model_authentication_checked"])
            self.assertEqual(0, report["provider_requests"])

    def test_unwritable_data_directory_fails_workbench_and_cli(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            (path / "index.html").write_text("<!doctype html>")
            with patch.object(installation, "web_root", return_value=path), patch(
                    "proofloop.installation.os.access", return_value=False):
                report = installation.doctor(path)
                self.assertFalse(report["ready"])
                self.assertTrue(report["frontend_available"])
                self.assertEqual(["data_directory_unwritable"], report["reason_codes"])
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(2, cli.main(["doctor", "--data-dir", folder]))
                self.assertTrue(installation.doctor(path, profile="core")["ready"])

    def test_corrupted_bundle_is_not_available_even_when_its_file_exists(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            bundle = path / "corrupted.json"
            bundle.write_text("{}\n")
            (path / "index.html").write_text("<!doctype html>")
            with patch.object(installation, "web_root", return_value=path), patch.object(
                    replay_bundle, "default_bundle", return_value=bundle):
                report = installation.doctor(path)
                self.assertTrue(report["offline_replay_present"])
                self.assertFalse(report["offline_replay_available"])
                self.assertFalse(report["offline_replay_valid"])
                self.assertFalse(report["ready"])
                self.assertEqual(["offline_replay_invalid"], report["reason_codes"])
                for profile in ("core", "workbench"):
                    with contextlib.redirect_stdout(io.StringIO()):
                        self.assertEqual(2, cli.main(["doctor", "--profile", profile, "--data-dir", folder]))

    def test_missing_bundle_fails_both_profiles_without_creating_it(self):
        with tempfile.TemporaryDirectory() as folder:
            bundle = Path(folder) / "missing.json"
            with patch.object(replay_bundle, "default_bundle", return_value=bundle):
                for profile in ("core", "workbench"):
                    report = installation.doctor(folder, profile=profile)
                    self.assertFalse(report["offline_replay_present"])
                    self.assertFalse(report["ready"])
                    self.assertIn("offline_replay_invalid", report["reason_codes"])
            self.assertFalse(bundle.exists())

    def test_actual_replay_failure_is_redacted_and_fails_profile(self):
        with patch.object(replay_bundle, "replay", side_effect=ValueError("private fixture /Users/private/path secret")):
            report = installation.doctor(profile="core")
        self.assertTrue(report["offline_replay_present"])
        self.assertFalse(report["ready"])
        self.assertEqual(["offline_replay_invalid"], report["reason_codes"])
        self.assertNotIn("private fixture", json.dumps(report))
        self.assertNotIn("/Users/private", json.dumps(report))

    def test_valid_existing_bundle_is_actually_replayed(self):
        with patch.object(replay_bundle, "replay", wraps=replay_bundle.replay) as replay:
            report = installation.doctor(profile="core")
        replay.assert_called_once_with()
        self.assertTrue(report["ready"])

    def test_unsupported_profile_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "core or workbench"):
            installation.doctor(profile="live")

    def test_discovery_failure_is_optional_and_no_local_key_contents_are_read(self):
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder) / ".env.local").write_text("META_API_KEY=private-doctor-fixture")
            with patch("proofloop.installation.importlib.util.find_spec", side_effect=ValueError("broken optional module")):
                report = installation.doctor(folder, profile="core")
            self.assertTrue(report["ready"])
            self.assertFalse(report["sam_parser_discoverable"])
            self.assertTrue(report["local_env_file_present"])
            self.assertFalse(report["optional_live_prerequisites"]["local_env_key_contents_checked"])
            self.assertNotIn("private-doctor-fixture", json.dumps(report))

    def test_cli_doctor_and_demo_produce_versioned_zero_inference_reports(self):
        for command in (["doctor", "--profile", "core"], ["demo"]):
            with contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(0, cli.main(command))
            report = json.loads(output.getvalue())
            self.assertEqual(0, report["provider_requests"])

    def test_tracking_cli_publishes_recovery_key_before_lost_acceptance(self):
        from proofloop.client import ClientError
        output = io.StringIO()
        def lost_response(key):
            receipt = json.loads(output.getvalue().splitlines()[0])
            self.assertEqual(key, receipt["idempotency_key"])
            self.assertEqual("cached_demo", receipt["mode"])
            raise ClientError("Authored lost response")
        with patch("proofloop.client.ProofLoopClient") as client, contextlib.redirect_stdout(output):
            client.return_value.cached_tracking.side_effect = lost_response
            self.assertEqual(2, cli.main(["track-demo", "--idempotency-key", "stable-authored-demo-key"]))
        client.return_value.cached_tracking.assert_called_once_with("stable-authored-demo-key")


if __name__ == "__main__":
    unittest.main()
