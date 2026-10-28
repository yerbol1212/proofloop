"""Fixed-file, read-only projections of separately scoped research controls.

Opening this lab does not call a model, fetch media or change an active task.
Each experiment must replay its private evidence and exactly match its public
summary. Missing or changed evidence suppresses that experiment independently.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import re
from typing import Any, Mapping

from .benchmark import ROOT
from .contract_stress import contract_stress_summary
from .core import ValidationError


MAX_SUMMARY_BYTES = 2 * 1024 * 1024
MAX_RECORD_BYTES = 32 * 1024 * 1024
MAX_MEDIA_BYTES = 128 * 1024 * 1024
_PRIVATE_TEXT = re.compile(r"/Users/|/private/|\bMETA_API_KEY\b|\bBearer\s|\bsk-[A-Za-z0-9_-]{8,}")
_PUBLIC_FILES = {
    "contract_stress": "artifacts/feasibility/contract-stress-20261028.json",
    "semantic": "artifacts/feasibility/semantic-candidate-challenge-20261028.json",
}
_PRIVATE_DIRS = {
    "contract_stress": "artifacts/local/contract-stress-20261028-01",
    "semantic": "artifacts/local/semantic-review-20261028-03",
}
_TITLES = {"contract_stress": "Same evidence, changed requirement",
           "semantic": "Challenge the candidate's identity"}


def _same(first: Any, second: Any) -> bool:
    return json.dumps(first, sort_keys=True, separators=(",", ":"), allow_nan=False) == json.dumps(
        second, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _path(root: Path, value: str | Path, *, allowed: str) -> Path:
    if not isinstance(value, (str, Path)):
        raise ValidationError("Research evidence path is invalid")
    path = (root / value).resolve() if not Path(value).is_absolute() else Path(value).resolve()
    try:
        directory = (root / allowed).resolve()
        directory.relative_to(root)
        path.relative_to(directory)
    except ValueError:
        raise ValidationError("Research evidence is outside its fixed directory") from None
    return path


def _file(root: Path, value: str | Path, *, allowed: str, limit: int) -> Path:
    path = _path(root, value, allowed=allowed)
    if not path.is_file():
        raise FileNotFoundError("Research evidence is unavailable")
    if path.stat().st_size > limit:
        raise ValidationError("Research evidence exceeds its bounded size")
    return path


def _json(root: Path, value: str | Path, *, allowed: str, limit: int = MAX_RECORD_BYTES) -> dict[str, Any]:
    def pairs(items):
        result = {}
        for key, item in items:
            if key in result:
                raise ValidationError("Duplicate research record fields")
            result[key] = item
        return result
    def invalid_constant(_value):
        raise ValidationError("Research JSON must contain finite values")
    path = _file(root, value, allowed=allowed, limit=limit)
    data = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs, parse_constant=invalid_constant)
    if not isinstance(data, dict):
        raise ValidationError("Research evidence must be an object")
    return data


def _safe_public(value: Any, depth: int = 0) -> None:
    if depth > 20:
        raise ValidationError("Research public projection is too deeply nested")
    if isinstance(value, str):
        if (len(value) > 8000 or _PRIVATE_TEXT.search(value) or value.startswith(("data:", "file:", "/"))
                or re.match(r"^[A-Za-z]:\\", value)):
            raise ValidationError("Private diagnostic text cannot enter the research lab")
    elif isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValidationError("Research public keys must be strings")
            _safe_public(key, depth + 1)
            _safe_public(item, depth + 1)
    elif isinstance(value, (list, tuple)):
        if len(value) > 200:
            raise ValidationError("Research projection has too many entries")
        for item in value:
            _safe_public(item, depth + 1)


def _replay_semantic(root: Path) -> dict[str, Any]:
    """Bound local files before the semantic experiment's complete replay."""
    try:
        from .semantic_experiment import replay_public_experiment
    except ImportError:
        raise ValidationError("Complete semantic experiment replay is unavailable") from None
    for suffix in ("01", "02", "03"):
        folder = _path(root, f"artifacts/local/semantic-review-20261028-{suffix}", allowed="artifacts/local")
        files = list(folder.rglob("*.json"))
        if len(files) > 64:
            raise ValidationError("Semantic experiment contains too many local records")
        for path in files:
            _file(root, path, allowed="artifacts/local", limit=MAX_RECORD_BYTES)
        freeze = _json(root, folder / "freeze-approved.json", allowed="artifacts/local")
        if not isinstance(freeze.get("cases"), list) or len(freeze["cases"]) != 4:
            raise ValidationError("Semantic freeze must retain the four reviewed cases")
        for case in freeze["cases"]:
            prepared = _json(root, case["prepared_file"], allowed="artifacts/local")
            _json(root, case["prior_guided_artifact"], allowed="artifacts/local")
            if not isinstance(prepared.get("frames"), list) or len(prepared["frames"]) != 2:
                raise ValidationError("Semantic preparation requires two source/crop pairs")
            for frame in prepared["frames"]:
                _file(root, frame["source_file"], allowed="inputs", limit=MAX_MEDIA_BYTES)
                _file(root, frame["crop_file"], allowed="artifacts/local", limit=MAX_MEDIA_BYTES)
    return replay_public_experiment(root)


class ResearchLab:
    """Only two allowlisted local experiments; no browser-supplied paths."""

    def __init__(self, root: Path = ROOT):
        self.root = Path(root).resolve()

    def _contract_controls(self, public: Mapping[str, Any]) -> dict[str, Any]:
        folder = _path(self.root, _PRIVATE_DIRS["contract_stress"], allowed="artifacts/local")
        frozen = _json(self.root, folder / "frozen-controls.json", allowed="artifacts/local")
        _json(self.root, folder / "report.json", allowed="artifacts/local")
        plans = frozen.get("plans")
        if not isinstance(plans, list) or not 1 <= len(plans) <= 2:
            raise ValidationError("Research controls need a bounded frozen plan")
        for plan in plans:
            if not isinstance(plan, Mapping):
                raise ValidationError("Research control plan is invalid")
            original = _path(self.root, plan.get("original_run"), allowed="artifacts/local")
            case_id = plan.get("case_id")
            if not isinstance(case_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", case_id):
                raise ValidationError("Research control case is invalid")
            manifest = _json(self.root, original / "approved-manifest.json", allowed="artifacts/local")
            _json(self.root, original / "report.json", allowed="artifacts/local")
            for name in ("guided", "spark"):
                _json(self.root, original / name / "results.json", allowed="artifacts/local")
                for case in manifest["contract"]["cases"]:
                    identifier = case["id"]
                    if not isinstance(identifier, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", identifier):
                        raise ValidationError("Research source case is invalid")
                    _json(self.root, original / name / f"{identifier}.json", allowed="artifacts/local")
            for name in (f"{case_id}-comparison.json", f"{case_id}-spark.json"):
                _json(self.root, folder / name, allowed="artifacts/local")
            for source in manifest["contract"]["sources"]:
                _file(self.root, source["file"], allowed="inputs", limit=MAX_MEDIA_BYTES)
            for case in manifest["contract"]["cases"]:
                for frame in case["frames"]:
                    _file(self.root, frame["file"], allowed="inputs", limit=MAX_MEDIA_BYTES)
        recomputed = contract_stress_summary(folder, root=self.root)
        if not _same(recomputed, public):
            raise ValidationError("Research contract summary differs from replayed evidence")
        _safe_public(recomputed)
        return recomputed

    def _semantic_controls(self, public: Mapping[str, Any]) -> dict[str, Any]:
        # Aggregate validation binds frozen selection, original locator anchors,
        # source/crop bytes, model results, and every scored/unscored attempt.
        # A public classification validator alone cannot establish those links.
        recomputed = _replay_semantic(self.root)
        if not _same(recomputed, public):
            raise ValidationError("Research semantic summary differs from replayed evidence")
        _safe_public(recomputed)
        return recomputed

    def snapshot(self) -> dict[str, Any]:
        sections = {}
        for name, filename in _PUBLIC_FILES.items():
            section = {"available": False, "status": "unavailable", "title": _TITLES[name],
                       "note": "The complete local evidence for this experiment is unavailable."}
            try:
                public = _json(self.root, filename, allowed="artifacts/feasibility", limit=MAX_SUMMARY_BYTES)
                loader = self._contract_controls if name == "contract_stress" else self._semantic_controls
                summary = loader(public)
                section = {"available": True, "status": "validated", "title": _TITLES[name],
                           "note": "Saved evidence replayed and matched the public experiment summary.",
                           "summary": summary}
            except FileNotFoundError:
                pass
            except (OSError, ValueError, TypeError, KeyError, UnicodeError, RecursionError):
                section.update(status="invalid", note="This experiment is withheld because its saved evidence did not pass replay.")
            sections[name] = section
        return deepcopy({"schema_version": 1, "read_only": True, "paid_requests_on_open": 0,
                         "available": any(section["available"] for section in sections.values()),
                         "note": "Development controls on already exposed recordings. Opening or replaying this lab makes no model requests.",
                         **sections})
