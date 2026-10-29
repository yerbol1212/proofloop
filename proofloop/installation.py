"""Read-only installation checks and installed-resource discovery."""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import shutil
import sqlite3
import sys


def checkout_root():
    candidate = Path(__file__).resolve().parents[1]
    return candidate if (candidate / "pyproject.toml").is_file() and (candidate / "web/package.json").is_file() else None


def data_root(explicit=None):
    if explicit is not None:
        return Path(explicit).expanduser().resolve()
    configured = os.environ.get("PROOFLOOP_DATA_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    checkout = checkout_root()
    if checkout:
        return checkout
    base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local/share"))
    return (base / "proofloop").expanduser().resolve()


def web_root():
    checkout = checkout_root()
    candidates = ([checkout / "web/dist"] if checkout else []) + [Path(__file__).with_name("_web")]
    return next((path for path in candidates if (path / "index.html").is_file()), candidates[-1])


def doctor(root=None):
    directory = data_root(root)
    ancestor = directory
    while not ancestor.exists() and ancestor.parent != ancestor:
        ancestor = ancestor.parent
    binaries = {}
    for name in ("ffmpeg", "ffprobe"):
        nearby = Path(sys.executable).parent / name
        binaries[name] = bool(shutil.which(name) or nearby.is_file() and os.access(nearby, os.X_OK))
    package = importlib.util.find_spec("meta_sam_parser")
    # The official distribution currently exposes `meta_sam`, not necessarily
    # the normalized distribution name. No decoder is imported or downloaded.
    package = package or importlib.util.find_spec("meta_sam")
    writable = ancestor.is_dir() and os.access(ancestor, os.W_OK | os.X_OK)
    frontend = (web_root() / "index.html").is_file()
    return {"schema_version": "1.0", "python_supported": sys.version_info >= (3, 9),
            "sqlite_version": sqlite3.sqlite_version, "frontend_available": frontend,
            "data_directory_writable": writable, "ffmpeg_available": binaries["ffmpeg"],
            "ffprobe_available": binaries["ffprobe"], "sam_parser_discoverable": package is not None,
            "environment_key_configured": bool(os.environ.get("META_API_KEY")),
            "local_env_file_present": (directory / ".env.local").is_file(),
            "offline_replay_available": (Path(__file__).with_name("data") / "contact_replay_v1.json").is_file(),
            "offline_workbench_ready": frontend and writable,
            "model_authentication_checked": False, "provider_requests": 0,
            "scope": "read_only_local_installation_checks"}
