import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from proofloop import cli, installation


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
        with tempfile.TemporaryDirectory() as folder, patch("proofloop.meta.MetaClient._request", side_effect=AssertionError("no API")):
            path = Path(folder) / "uncreated" / "data"
            with patch.dict(os.environ, {"META_API_KEY": "private-fixture-must-not-print"}):
                report = installation.doctor(path)
            self.assertFalse(path.exists())
            self.assertTrue(report["environment_key_configured"])
            self.assertFalse(report["model_authentication_checked"])
            self.assertEqual(0, report["provider_requests"])
            self.assertNotIn("private-fixture", json.dumps(report))
            self.assertNotIn(str(path), json.dumps(report))

    def test_doctor_reports_missing_frontend_and_binaries(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(installation, "web_root", return_value=Path(folder)), patch(
                "proofloop.installation.shutil.which", return_value=None), patch("proofloop.installation.os.access", return_value=False):
            report = installation.doctor(folder)
            self.assertFalse(report["frontend_available"])
            self.assertFalse(report["offline_workbench_ready"])
            self.assertFalse(report["ffmpeg_available"])

    def test_cli_doctor_and_demo_produce_versioned_zero_inference_reports(self):
        for command in ("doctor", "demo"):
            with contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(0, cli.main([command]))
            report = json.loads(output.getvalue())
            self.assertEqual(0, report["provider_requests"])


if __name__ == "__main__":
    unittest.main()
