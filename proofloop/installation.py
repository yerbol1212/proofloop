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


def doctor(root=None, *, profile="workbench"):
    """Check one local installation profile without writes or provider access.

    A core-only install can deliberately omit frontend and storage. The default
    workbench profile additionally requires both, and every passing profile
    requires the actual packaged policy replay to validate successfully.
    """
    if profile not in ("core", "workbench"):
        raise ValueError("Use the core or workbench installation profile")
    directory = data_root(root)
    ancestor = directory
    while not ancestor.exists() and ancestor.parent != ancestor:
        ancestor = ancestor.parent
    binaries = {}
    for name in ("ffmpeg", "ffprobe"):
        nearby = Path(sys.executable).parent / name
        binaries[name] = bool(shutil.which(name) or nearby.is_file() and os.access(nearby, os.X_OK))
    try:
        package = importlib.util.find_spec("meta_sam_parser")
        # The official distribution currently exposes `meta_sam`, not
        # necessarily the normalized distribution name. No decoder is imported.
        package = package or importlib.util.find_spec("meta_sam")
    except (ImportError, AttributeError, ValueError):
        package = None
    writable = ancestor.is_dir() and os.access(ancestor, os.W_OK | os.X_OK)
    frontend = (web_root() / "index.html").is_file()
    from . import replay_bundle
    replay_present = replay_bundle.default_bundle().is_file()
    replay_valid = False
    try:
        replay_bundle.replay()
        replay_valid = True
    except (OSError, ValueError, KeyError, TypeError, AttributeError, ImportError):
        # Failure is a bounded diagnostic, never raw parser/provider/path text.
        pass
    python_supported = sys.version_info >= (3, 9)
    core_ready = python_supported and replay_valid
    workbench_ready = core_ready and frontend and writable
    reasons = []
    if not python_supported:
        reasons.append("python_unsupported")
    if not replay_valid:
        reasons.append("offline_replay_invalid")
    if profile == "workbench":
        if not frontend:
            reasons.append("frontend_unavailable")
        if not writable:
            reasons.append("data_directory_unwritable")
    return {"schema_version": "1.0", "profile": profile,
            "ready": core_ready if profile == "core" else workbench_ready,
            "reason_codes": reasons, "python_supported": python_supported,
            "sqlite_version": sqlite3.sqlite_version, "frontend_available": frontend,
            "data_directory_writable": writable, "ffmpeg_available": binaries["ffmpeg"],
            "ffprobe_available": binaries["ffprobe"], "sam_parser_discoverable": package is not None,
            "environment_key_configured": bool(os.environ.get("META_API_KEY")),
            "local_env_file_present": (directory / ".env.local").is_file(),
            "offline_replay_present": replay_present, "offline_replay_available": replay_valid,
            "offline_replay_valid": replay_valid, "core_ready": core_ready,
            "offline_workbench_ready": workbench_ready,
            "optional_live_prerequisites": {
                "required_for_selected_profile": False,
                "python_sam_supported": sys.version_info >= (3, 10),
                "ffmpeg_available": binaries["ffmpeg"], "ffprobe_available": binaries["ffprobe"],
                "sam_parser_discoverable": package is not None,
                "environment_key_configured": bool(os.environ.get("META_API_KEY")),
                "local_env_file_present": (directory / ".env.local").is_file(),
                "local_env_key_contents_checked": False,
                "model_authentication_checked": False, "provider_connectivity_checked": False,
            },
            "model_authentication_checked": False, "provider_requests": 0,
            "scope": "read_only_local_installation_checks"}
