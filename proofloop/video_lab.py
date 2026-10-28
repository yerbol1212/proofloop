"""Read-only, source-bound projection of reviewed recorded-video experiments.

No API calls, arbitrary browser-selected paths, media downloads or mutations.
Only private runs that match an allowlisted public summary can enter the lab.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import struct
import threading
from typing import Any, Mapping, Sequence

from .benchmark import AVAILABLE_CONDITIONS, GUIDED_NOTE, ROOT, STATUSES, _approved, _file_hash, public_summary, validate_report, verify_media
from .capture import _jpeg_dimensions
from .core import ValidationError
from .followup import replay_fixed_followup, response_paths

MAX_REPORT_BYTES = 32 * 1024 * 1024
MAX_SUMMARY_BYTES = 2 * 1024 * 1024
MAX_RUNS = 12


def _read(path: Path, limit: int) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size > limit:
        raise ValidationError("Lab record is missing or exceeds its local size limit")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, UnicodeError):
        raise ValidationError("Lab record is not valid JSON") from None
    if not isinstance(value, dict):
        raise ValidationError("Lab record must be an object")
    return value


def _inside(path: Path, allowed: Path) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(allowed.resolve())
    except ValueError:
        raise ValidationError("Lab path is outside its fixed local directory") from None
    return resolved


def _dimensions(path: Path) -> tuple[int, int]:
    data = path.read_bytes()
    if data.startswith(b"\xff\xd8"):
        return _jpeg_dimensions(data)
    if data.startswith(b"\x89PNG\r\n\x1a\n") and len(data) >= 24:
        width, height = struct.unpack(">II", data[16:24])
        if 0 < width <= 16000 and 0 < height <= 16000 and width * height <= 16_000_000:
            return width, height
    raise ValidationError("Lab image requires supported, bounded JPEG or PNG dimensions")


def _reason(value: Any) -> str:
    if not isinstance(value, str):
        return "No reviewable explanation is available."
    text = " ".join(value.split())[:1200]
    # Model text is untrusted explanatory text. Do not reflect local paths or
    # credential-shaped diagnostic strings even if a malformed report has them.
    if re.search(r"/Users/|/private/|Bearer\s|\bsk-[A-Za-z0-9_-]{8,}|\bMETA_API_KEY\b", text):
        return "Explanation withheld because it contains private diagnostic text."
    return text or "No reviewable explanation is available."


class VideoLab:
    """Fixed-directory media allowlist; unavailable or unreviewed runs fail closed."""

    def __init__(self, root: Path = ROOT, *, report_paths: Sequence[Path] | None = None,
                 summary_paths: Sequence[Path] | None = None):
        self.root = Path(root).resolve()
        self._report_paths = tuple(report_paths) if report_paths is not None else None
        self._summary_paths = tuple(summary_paths) if summary_paths is not None else None
        self._lock = threading.RLock()
        self._fingerprint: Any = None
        self._snapshot: dict[str, Any] = {"available": False, "runs": []}
        self._media: dict[str, tuple[Path, str]] = {}
        self._tracked_paths: tuple[Path, ...] = ()

    def _paths(self) -> tuple[list[Path], list[Path]]:
        private = self.root / "artifacts/local"
        public = self.root / "artifacts/feasibility"
        reports = (list(self._report_paths) if self._report_paths is not None else
                   sorted(private.glob("online-benchmark-*/report.json"), reverse=True)[:MAX_RUNS])
        summaries = (list(self._summary_paths) if self._summary_paths is not None else
                     sorted(public.glob("online*.json"))[:MAX_RUNS * 2])
        def bound(paths: list[Path], allowed: Path) -> list[Path]:
            result = []
            for path in paths:
                try:
                    result.append(_inside(path if path.is_absolute() else self.root / path, allowed))
                except (ValueError, OSError):
                    continue
            return result
        return bound(reports, private), bound(summaries, public)

    @staticmethod
    def _stats(paths: Sequence[Path]) -> tuple[Any, ...]:
        result = []
        for path in paths:
            try:
                stat = path.stat()
                result.append((str(path), stat.st_mtime_ns, stat.st_size))
            except OSError:
                result.append((str(path), None, None))
        return tuple(result)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            reports, summaries = self._paths()
            records = reports + summaries + [path.parent / "approved-manifest.json" for path in reports]
            records += [path.parent / name / "results.json" for path in reports for name in AVAILABLE_CONDITIONS]
            records += [dependency for path in reports for dependency in response_paths(path.parent)]
            key = self._stats(records + list(self._tracked_paths))
            if key == self._fingerprint:
                return deepcopy(self._snapshot)
            reviewed: dict[str, list[dict[str, Any]]] = {}
            for path in summaries:
                try:
                    value = _read(path, MAX_SUMMARY_BYTES)
                    if (value.get("experiment") == "licensed_recorded_video_comparison"
                            and isinstance(value.get("contract_hash"), str)):
                        reviewed.setdefault(value["contract_hash"], []).append(value)
                except (OSError, ValueError):
                    continue
            runs, media, tracked = [], {}, []
            for path in reports:
                try:
                    run, run_media, run_paths = self._load(path, reviewed)
                    runs.append(run)
                    media.update(run_media)
                    tracked.extend(run_paths)
                except (OSError, ValueError, KeyError, TypeError, struct.error):
                    # A private-only, changed or malformed run never leaks paths
                    # or partially reviewed predictions into the browser.
                    continue
            self._snapshot = {"available": bool(runs), "runs": runs,
                              "note": "Read-only recorded-video experiments. No camera or model requests run when this page opens."}
            self._media = media
            self._tracked_paths = tuple(dict.fromkeys(tracked))
            self._fingerprint = self._stats(records + list(self._tracked_paths))
            return deepcopy(self._snapshot)

    def _load(self, report_path: Path, reviewed: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, tuple[Path, str]], tuple[Path, ...]]:
        report = _read(report_path, MAX_REPORT_BYTES)
        envelope = _read(_inside(report_path.parent / "approved-manifest.json", self.root / "artifacts/local"), MAX_SUMMARY_BYTES)
        protocol = _approved(envelope)
        if (report.get("experiment") != "licensed_recorded_video_comparison" or report.get("status") != "completed"
                or report.get("evidence_origin") != "external_recorded_video"
                or report.get("contract_hash") != envelope["contract_hash"]):
            raise ValidationError("Lab requires a completed source-bound recorded-video run")
        summaries = reviewed.get(envelope["contract_hash"], [])
        expected_summary = public_summary(envelope, report)
        summary = next((candidate for candidate in summaries
                        if all(candidate.get(key) == value for key, value in expected_summary.items())), None)
        if summary is None:
            raise ValidationError("Private lab run differs from its reviewed public summary")
        paths = verify_media(envelope, project_root=self.root)
        sidecars = {}
        for name, condition in report["conditions"].items():
            candidate = report_path.parent / name / "results.json"
            if candidate.exists() and condition.get("status") == "completed":
                sidecars[name] = _read(_inside(candidate, self.root / "artifacts/local"), MAX_REPORT_BYTES)
        integrity = validate_report(envelope, report, raw_results=sidecars)
        conditions = report.get("conditions")
        if not isinstance(conditions, Mapping) or set(conditions) - set(AVAILABLE_CONDITIONS):
            raise ValidationError("Lab conditions are invalid")
        by_condition: dict[str, dict[str, Any]] = {}
        for name, condition in conditions.items():
            if condition.get("status") != "completed" or not isinstance(condition.get("cases"), list):
                raise ValidationError("Lab condition is incomplete")
            case_map = {}
            for result in condition["cases"]:
                identifier = result.get("case_id")
                if identifier in case_map or result.get("status") not in STATUSES:
                    raise ValidationError("Lab result IDs and statuses are invalid")
                case_map[identifier] = result
            by_condition[name] = case_map
        contract = envelope["contract"]
        followup = None
        if contract["dataset_id"] == "proofloop-fixed-future-evidence-development-20261028":
            followup = replay_fixed_followup(envelope, report, report_path.parent, root=self.root)
            if summary.get("followup_policy_replay") != followup:
                raise ValidationError("Public follow-up replay differs from its saved source-bound responses")
        declared_ids = {case["id"] for case in contract["cases"]}
        if any(set(case_map) != declared_ids for case_map in by_condition.values()):
            raise ValidationError("Lab results do not cover exactly the reviewed cases")
        run_id = "video-" + hashlib.sha256((report_path.parent.name + envelope["contract_hash"]).encode()).hexdigest()[:20]
        sources = {source["id"]: source for source in contract["sources"]}
        cases, media = [], {}
        for case in contract["cases"]:
            images = []
            for frame in case["frames"]:
                path = paths[frame["sha256"]]
                width, height = _dimensions(path)
                media_id = hashlib.sha256((run_id + case["id"] + frame["sha256"]).encode()).hexdigest()[:40]
                media[media_id] = (path, frame["sha256"])
                images.append({"url": "/api/lab/media/" + media_id, "width": width, "height": height,
                               "source_timestamp_ms": frame["source_timestamp_ms"]})
            statuses, reasons, checks = {}, {}, {}
            for name, case_map in by_condition.items():
                result = case_map[case["id"]]
                if result.get("frame_hashes") != [frame["sha256"] for frame in case["frames"]]:
                    raise ValidationError("Lab prediction does not bind the reviewed source pair")
                raw_checks = result.get("checks")
                if (not isinstance(raw_checks, list)
                        or {item.get("check_id") for item in raw_checks} != {check.id for check in protocol.checks}
                        or len(raw_checks) != len(protocol.checks)):
                    raise ValidationError("Lab prediction checks differ from the approved protocol")
                statuses[name] = result["status"]
                checks[name] = [{"id": check["check_id"], "status": check["status"], "reason": _reason(check.get("reason"))}
                                for check in raw_checks]
                reason_parts = [item["reason"] for item in checks[name]]
                if name in ("sam", "guided"):
                    grounding = result.get("grounding", {}).get("evidence", {})
                    reason_parts = [_reason(item.get("reason")) for item in grounding.get("check_grounding", [])] or reason_parts
                if name == "guided":
                    # The source pair exists even when an ambiguous locator
                    # yields no SAM measurement. Do not imply a missing upload.
                    reason_parts = [
                        "The supplied views did not yield enough supported guided mask measurements. "
                        "Inspect the target from a clearer view."
                        if "No source images were supplied" in reason else reason
                        for reason in reason_parts
                    ]
                reasons[name] = " ".join(dict.fromkeys(reason_parts))[:1600]
            source = sources[case["source_id"]]
            cases.append({"id": case["id"], "source_id": source["id"], "source_publisher": source["publisher"],
                          "source_url": source["page_url"], "source_media_url": source["media_url"],
                          "license": {key: source["license"][key] for key in ("name", "url", "attribution")},
                          "role": case["role"], "expected": case["annotation"]["status"] if case["annotation"] else None,
                          "sam": statuses.get("sam"), "spark": statuses.get("spark"), "guided": statuses.get("guided"),
                          "reasons": reasons, "checks": checks,
                          "images": images, "regions": deepcopy(case["regions"]),
                          "annotation_reason": case["annotation"]["reason"] if case["annotation"] else None})
        title = "Guided perception comparison" if "guided" in by_condition else "Text-prompt development pilot"
        evaluation_role = summary.get("evaluation_role")
        normalized = summary["conditions"].get("guided", {}).get("observation_version") == "normalized_boxes_v2"
        if normalized and evaluation_role == "development_reprocessing":
            title = "Normalized grounding — development replay"
        elif normalized and evaluation_role == "new_source_holdback":
            title = "Normalized grounding — new sources"
        if evaluation_role == "development_reprocessing" and contract["dataset_id"] == "proofloop-temporal-stress-development-20261028":
            title = "Temporal uncertainty — development stress test"
        if evaluation_role == "development_reprocessing" and contract["dataset_id"] == "proofloop-candy-new-task-transfer-development-20261028":
            title = "New task transfer — candy collection"
        if followup is not None:
            title = "Specific evidence requests — fixed future pairs"
        task_summary = " ".join(
            f"{check.object_id.replace('_', ' ')} must be in {check.region_id.upper()}."
            if check.kind == "location" else
            f"{check.object_id.replace('_', ' ')} must be {'visible' if check.expected else 'absent'} in {check.region_id.upper()}."
            for check in protocol.checks)
        return ({"id": run_id, "title": title, "contract_hash": envelope["contract_hash"],
                 **({"evaluation_role": evaluation_role} if evaluation_role is not None else {}),
                 **({"evaluation_note": "New-task development pilot: one newly tested candy recording and scene. This is not an independent benchmark or evidence of general task transfer."}
                    if contract["dataset_id"] == "proofloop-candy-new-task-transfer-development-20261028" else {}),
                 "source_kind": "external_recorded_video", "inference_origin": report.get("inference_origin"),
                 "report_integrity": integrity,
                 **({"followup_policy_replay": followup} if followup is not None else {}),
                 "requirements": contract["requirements"], "task_summary": task_summary, "objects": list(protocol.objects),
                 "concepts": deepcopy(contract["concepts"]), "case_count": len(cases), "source_count": len(sources),
                 "metrics": {name: deepcopy(summary["conditions"][name].get("metrics")) for name in by_condition},
                 "conditions": {name: {**({"method_note": GUIDED_NOTE} if name == "guided" else {}),
                                       **({"observation_version": "normalized_boxes_v2"}
                                          if name == "guided" and summary["conditions"][name].get("observation_version") == "normalized_boxes_v2" else {}),
                                       "estimated_usd": summary["conditions"][name].get("estimated_usd"),
                                       "wall_seconds": summary["conditions"][name].get("wall_seconds")}
                                for name in by_condition},
                 "limits": ["Prerecorded demonstration states with authored placement labels; no interactive recovery is verified.",
                            "Cases from the same source are correlated. This curated pilot is not held-out benchmark accuracy.",
                            "The acceptance gate covers image-zone placement, not robot control, product quality or safety.",
                            "UNKNOWN means the system lacks supported evidence. It is not successful task verification."]
                           + ([GUIDED_NOTE] if "guided" in by_condition else []),
                 "cases": cases}, media, tuple(paths.values()))

    def media_path(self, media_id: str) -> Path:
        if not isinstance(media_id, str) or not re.fullmatch(r"[a-f0-9]{40}", media_id):
            raise KeyError("Unknown lab image")
        with self._lock:
            self.snapshot()
            if media_id not in self._media:
                raise KeyError("Unknown lab image")
            original, expected_hash = self._media[media_id]
            path = _inside(original, self.root / "inputs/local")
            if not path.is_file() or _file_hash(path) != expected_hash:
                raise KeyError("Lab image no longer matches its reviewed source")
            return path
