"""Install a release wheel outside its checkout and exercise the offline service.

No credentials, network downloads, source overlays or model calls are used.
The only network traffic is the temporary local HTTP server.
"""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
from urllib.request import build_opener, ProxyHandler
import venv
import zipfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    args = parser.parse_args()
    wheel = args.wheel.resolve()
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        assert "proofloop/_web/index.html" in names, "Build the frontend before making a workbench wheel"
        assert "proofloop/data/contact_replay_v1.json" in names
        assert not any(".env" in name or "artifacts/local" in name or "inputs/local" in name for name in names)
    with tempfile.TemporaryDirectory(prefix="proofloop-install-") as temporary:
        root = Path(temporary)
        venv.EnvBuilder(with_pip=True).create(root / "venv")
        binaries = root / "venv" / ("Scripts" if os.name == "nt" else "bin")
        python = binaries / ("python.exe" if os.name == "nt" else "python")
        executable = binaries / ("proofloop.exe" if os.name == "nt" else "proofloop")
        environment = {key: value for key, value in os.environ.items() if key not in {"PYTHONPATH", "META_API_KEY", "PROOFLOOP_DATA_DIR"}}
        environment["PYTHONNOUSERSITE"] = "1"
        environment["XDG_DATA_HOME"] = str(root / "user-data")
        def run(*command):
            return subprocess.run(command, cwd=root, env=environment, check=True, capture_output=True, text=True).stdout
        run(str(python), "-m", "pip", "install", "--no-index", "--no-deps", str(wheel))
        doctor = json.loads(run(str(executable), "doctor", "--data-dir", str(root / "data")))
        assert doctor["frontend_available"] and doctor["offline_replay_available"]
        assert doctor["provider_requests"] == 0 and not doctor["environment_key_configured"]
        replay = json.loads(run(str(executable), "replay"))
        assert replay["provider_requests"] == 0
        assert replay["cohorts"]["exposed_episode_2"]["v2_veto_current"]["positives_retained"] == 3
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0)); port = listener.getsockname()[1]
        with (root / "server.log").open("w") as log:
            process = subprocess.Popen([str(executable), "serve", "--offline", "--port", str(port),
                "--data-dir", str(root / "data")], cwd=root, env=environment, stdout=log, stderr=log)
            opener = build_opener(ProxyHandler({}))
            try:
                for _ in range(100):
                    if process.poll() is not None:
                        raise AssertionError("Installed server stopped: " + (root / "server.log").read_text())
                    try:
                        with opener.open("http://127.0.0.1:" + str(port) + "/api/health", timeout=1) as response:
                            assert json.load(response)["ok"]
                        break
                    except OSError:
                        time.sleep(.05)
                else:
                    raise AssertionError("Installed server did not become available")
                with opener.open("http://127.0.0.1:" + str(port) + "/", timeout=2) as response:
                    assert b"<html" in response.read().lower()
                lines = run(str(executable), "track-demo", "--port", str(port), "--wait-seconds", "10").splitlines()
                job = json.loads(lines[-1])
                assert job["state"] == "succeeded" and job["result"]["request_attempts"] == 0
                assert job["result"]["mode"] == "cached_demo" and not job["physical_action_authorized"]
                assert job["result"]["frames"][-1]["status"] == "unknown"
                with opener.open("http://127.0.0.1:" + str(port) + "/api/v1/accounting", timeout=2) as response:
                    assert json.load(response)["total_reserved_attempts"] == 0
            finally:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill(); process.wait(timeout=5)
        print(json.dumps({"installed_outside_checkout": True, "packaged_frontend_served": True,
            "offline_replay_passed": True, "async_cached_tracking_passed": True, "provider_requests": 0}, sort_keys=True))


if __name__ == "__main__":
    main()
