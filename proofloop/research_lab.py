"""Fixed-file, read-only projections of separately scoped research controls.

Opening this lab does not call a model, fetch media or change an active task.
Each experiment must replay its private evidence and exactly match its public
summary. Missing or changed evidence suppresses that experiment independently.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import threading
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
    "completion_v2": "artifacts/feasibility/semantic-v2-completion-20261028.json",
    "distractors_v2": "artifacts/feasibility/semantic-v2-distractors-20261028.json",
    "crop_confirmation": "artifacts/feasibility/semantic-crop-confirmation-20261028.json",
}
_PRIVATE_DIRS = {
    "contract_stress": "artifacts/local/contract-stress-20261028-01",
    "semantic": "artifacts/local/semantic-review-20261028-03",
    "completion_v2": "artifacts/local/semantic-v2-completion-20261028-01",
    "distractors_v2": "artifacts/local/semantic-v2-distractors-20261028-01",
    "crop_confirmation": "artifacts/local/semantic-crop-confirmation-20261028-01",
}
_TITLES = {"contract_stress": "Same evidence, changed requirement",
           "semantic": "Historical candidate challenge — original completion failures",
           "completion_v2": "Low-reasoning challenger — paired completion check",
           "distractors_v2": "Real-video distractors — reviewed candidate boxes",
           "crop_confirmation": "Crop-only confirmation — development retest"}


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
    """Allowlisted local experiments; no browser-supplied file paths."""

    def __init__(self, root: Path = ROOT):
        self.root = Path(root).resolve()
        self._lock = threading.RLock()
        self._media: dict[str, tuple[Path, str]] = {}

    def _trial_controls(self, public: Mapping[str, Any], name: str) -> tuple[dict[str, Any], dict]:
        from .semantic_trial import public_trial_summary
        folder = _path(self.root, _PRIVATE_DIRS[name], allowed="artifacts/local")
        plan = _json(self.root, folder / "freeze.json", allowed="artifacts/local")
        if name == "completion_v2":
            # The historical replay binds saved model anchors to original
            # locator records; a generic prepared crop alone cannot do that.
            _replay_semantic(self.root)
            old = _json(self.root, "artifacts/local/semantic-review-20261028-03/freeze-approved.json",
                        allowed="artifacts/local")
            if [entry["case_id"] for entry in plan["cases"]] != [entry["case_id"] for entry in old["cases"]]:
                raise ValidationError("Paired completion selection changed")
            for new, prior in zip(plan["cases"], old["cases"]):
                for key in ("prepared_file", "prepared_hash", "expected_identity", "source_id", "source_timestamps_ms"):
                    if new[key] != prior[key]:
                        raise ValidationError("Paired completion must retain its exact historical inputs")
        else:
            from .distractor_media import validate_prepared_entries
            validate_prepared_entries(self.root, plan["cases"])
        recomputed = public_trial_summary(folder, root=self.root)
        if not _same(recomputed, public):
            raise ValidationError("Trial summary differs from its source-bound response replay")
        _safe_public(recomputed)
        projected, media = deepcopy(recomputed), {}
        for case, entry in zip(projected["cases"], plan["cases"]):
            prepared = _json(self.root, entry["prepared_file"], allowed="artifacts/local")
            images = []
            for frame in prepared["frames"]:
                urls = {}
                for role, field, hash_field in (("source", "source_file", "sha256"),
                                               ("crop", "crop_file", "crop_sha256")):
                    # Historical frames are in inputs; new reviewed frames
                    # and every derived crop remain under artifacts/local.
                    value = frame[field]
                    try:
                        path = _file(self.root, value, allowed="inputs/local", limit=MAX_SUMMARY_BYTES)
                    except ValidationError:
                        path = _file(self.root, value, allowed="artifacts/local", limit=MAX_SUMMARY_BYTES)
                    if hashlib.sha256(path.read_bytes()).hexdigest() != frame[hash_field]:
                        raise ValidationError("Trial image differs from its replayed source")
                    token = hashlib.sha256((name + case["case_id"] + role + frame[hash_field]).encode()).hexdigest()[:40]
                    media[token] = (path, frame[hash_field])
                    urls[role + "_url"] = "/api/research-lab/media/" + token
                images.append(dict(urls, width=frame["width"], height=frame["height"],
                                   candidate_box=frame["candidate_box"]))
            case["images"] = images
        return projected, media

    def _confirmation_controls(self, public: Mapping[str, Any]) -> dict[str, Any]:
        from .semantic_local import public_confirmation_summary
        # Bind the retest to the full original trial as well as its individual
        # supported reports. A local check cannot certify a new source/anchor.
        original = _json(self.root, _PUBLIC_FILES["distractors_v2"], allowed="artifacts/feasibility",
                         limit=MAX_SUMMARY_BYTES)
        parent, _media = self._trial_controls(original, "distractors_v2")
        expected = [case["case_id"] for case in parent["cases"] if case["attempt_status"] == "completed"
                    and case["challenge"]["challenge"]["decision"] == "supported"]
        folder = _path(self.root, _PRIVATE_DIRS["crop_confirmation"], allowed="artifacts/local")
        plan = _json(self.root, folder / "freeze.json", allowed="artifacts/local")
        if [case["case_id"] for case in plan["cases"]] != expected:
            raise ValidationError("Crop confirmation must retain all original supported candidates")
        for case in plan["cases"]:
            path = _path(self.root, case["global_report_file"], allowed=_PRIVATE_DIRS["distractors_v2"])
            if path.name != case["case_id"] + ".json":
                raise ValidationError("Crop confirmation has an unbound parent case")
        recomputed = public_confirmation_summary(folder, root=self.root)
        if not _same(recomputed, public):
            raise ValidationError("Crop confirmation summary differs from saved evidence")
        _safe_public(recomputed)
        return recomputed

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
        with self._lock:
            return self._snapshot_locked()

    def _snapshot_locked(self) -> dict[str, Any]:
        sections = {}
        media = {}
        for name, filename in _PUBLIC_FILES.items():
            section = {"available": False, "status": "unavailable", "title": _TITLES[name],
                       "note": "The complete local evidence for this experiment is unavailable."}
            try:
                public = _json(self.root, filename, allowed="artifacts/feasibility", limit=MAX_SUMMARY_BYTES)
                if name == "crop_confirmation":
                    summary = self._confirmation_controls(public)
                elif name.endswith("_v2"):
                    summary, new_media = self._trial_controls(public, name)
                    media.update(new_media)
                else:
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
        self._media = media
        return deepcopy({"schema_version": 1, "read_only": True, "paid_requests_on_open": 0,
                         "available": any(section["available"] for section in sections.values()),
                         "note": "Scoped development controls; cohorts and candidate origins remain separate. Opening or replaying makes no model requests.",
                         **sections})

    def media_path(self, media_id: str) -> Path:
        if not isinstance(media_id, str) or not re.fullmatch(r"[0-9a-f]{40}", media_id):
            raise KeyError("Unknown research image")
        with self._lock:
            if media_id not in self._media:
                raise KeyError("Research image is unavailable")
            original, expected = self._media[media_id]
            # URLs identify immutable bytes in the already validated snapshot.
            # Rehash the requested content, without spawning every experiment's
            # crop replay for each image. Refreshing results rebuilds the map.
            try:
                path = _file(self.root, original, allowed="inputs/local", limit=MAX_SUMMARY_BYTES)
            except ValidationError:
                path = _file(self.root, original, allowed="artifacts/local", limit=MAX_SUMMARY_BYTES)
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise KeyError("Research image no longer matches its source")
            return path
