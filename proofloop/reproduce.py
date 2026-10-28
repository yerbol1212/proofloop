"""Prepare a small pinned online-video demo without making model requests.

The default command prints a plan only. Downloads require --download; inference
is deliberately a separate benchmark command after exact-contract review.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, BinaryIO
from urllib.request import Request, urlopen

from .benchmark import approve_manifest, manifest_hash, validate_manifest
from .capture import CaptureError, _extract_frame, _tool
from .core import ValidationError

ROOT = Path(__file__).resolve().parents[1]
REVISION = "1c52c4c794b6f93d931489263cf15e4331600bfc"
DATASET = "https://huggingface.co/datasets/lerobot/aloha_static_coffee"
CARD_URL = f"{DATASET}/resolve/{REVISION}/README.md"
CARD_PAGE = f"{DATASET}/blob/{REVISION}/README.md"
CARD_SHA256 = "da783ef780c38b7a85c85db2a03dad304adfcd1de48b69b0618cf1eeab4024c8"
CARD_BYTES = 6505
ATTRIBUTION = ("ALOHA: Tony Z. Zhao, Vikash Kumar, Sergey Levine and Chelsea Finn (RSS 2023); "
               "LeRobot dataset conversion by Hugging Face. Dataset card explicitly declares MIT. "
               "Original ALOHA code copyright (c) 2023 Tony Z. Zhao. "
               "Paper: https://arxiv.org/abs/2304.13705; project: https://tonyzhaozh.github.io/aloha/.")
MIT_NOTICE = """MIT License

Copyright (c) 2023 Tony Z. Zhao

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
SOURCE_RECORDS = {
    0: (9964446, "02bc95dcc0922f5d7d759861e6e5151f4aa068631d5491e82badfc043f68cfd2"),
    1: (9842148, "20c258c99f463bd2451ab02b782f5f79bad3cba1bab7be7d45de6b43bb4f105e"),
    2: (10076093, "37d4930f315ac3a19288aa33b7446697be3e8a83cd31879674b40dbb8343d32f"),
}
# Timings are public, fixed before a new clone's model inference. Labels remain
# reviewable authored assertions; preparing this draft never claims human review.
PROFILES = {
    "coffee-placement": ((0, "staging", (500, 1500), "fail"),
                         (0, "dispense", (20000, 21000), "pass")),
    "coffee-acquisition": ((0, "initial", (16500, 18500), "unknown"),
                           (0, "attempt-1", (19100, 20100), "pass"),
                           (0, "attempt-2", (20700, 21700), "pass"),
                           (1, "initial", (18500, 20000), "unknown"),
                           (1, "attempt-1", (20500, 21500), "pass"),
                           (2, "terminal", (500, 21500), "unknown")),
}
REGIONS = {"dispense": [.38, .18, .64, .46], "staging": [.66, .18, .87, .46]}
PROTOCOL = {"id": "coffee-dispense-gate", "version": "authored-reviewed-1", "objects": ["glass"],
            "regions": ["dispense", "staging"], "checks": [
                {"id": "glass-under-spout", "kind": "location", "object_id": "glass", "region_id": "dispense",
                 "max_age_ms": 30000, "min_confidence": .8, "min_stable_samples": 2, "unique_target": True}],
            "steps": [], "checkpoints": []}


def profile_plan(profile: str = "coffee-placement") -> dict[str, Any]:
    if profile not in PROFILES:
        raise ValidationError("Unknown reproduction profile")
    episodes = sorted({row[0] for row in PROFILES[profile]})
    sources = []
    for episode in episodes:
        size, digest = SOURCE_RECORDS[episode]
        sources.append({"episode": episode, "bytes": size, "sha256": digest,
                        "media_url": f"{DATASET}/resolve/{REVISION}/videos/chunk-000/"
                                     f"observation.images.cam_high/episode_{episode:06d}.mp4"})
    return {"profile": profile, "revision": REVISION, "sources": sources,
            "download_bytes": sum(source["bytes"] for source in sources) + CARD_BYTES,
            "frames": sum(len(row[2]) for row in PROFILES[profile]),
            "model_requests": 0, "automatic_approval": False,
            "license": {"name": "MIT", "url": CARD_PAGE, "attribution": ATTRIBUTION},
            "fixed_cases": [{"case_id": f"coffee-e{ep:02d}-{stage}", "timestamps_ms": list(times),
                             "authored_status_for_review": status}
                            for ep, stage, times, status in PROFILES[profile]]}


def _private(path: Path, root: Path, folder: str) -> Path:
    resolved, allowed = path.resolve(), (root / folder).resolve()
    if resolved == allowed or allowed not in resolved.parents:
        raise ValidationError(f"Reproduction outputs must stay strictly inside {folder}")
    return resolved


def _write_new(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(data)


def _json_new(path: Path, value: Any) -> None:
    _write_new(path, (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode())


def _copy_bounded(stream: BinaryIO, path: Path, expected_bytes: int, expected_sha256: str) -> None:
    """Hash and bound the stream before publishing its fixed local filename."""
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    digest, total = hashlib.sha256(), 0
    try:
        with os.fdopen(descriptor, "wb") as target:
            while True:
                chunk = stream.read(min(1024 * 1024, expected_bytes - total + 1))
                if not chunk:
                    break
                total += len(chunk)
                if total > expected_bytes:
                    raise ValidationError("Pinned download exceeded its exact byte budget")
                digest.update(chunk)
                target.write(chunk)
        if total != expected_bytes or digest.hexdigest() != expected_sha256:
            raise ValidationError("Pinned download differs from its frozen size or SHA-256")
        # Hard-link publication is atomic and refuses to replace existing data.
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _obtain(record: dict[str, Any], path: Path, download: bool) -> None:
    if path.exists():
        if (not path.is_file() or path.stat().st_size != record["bytes"]
                or hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]):
            raise ValidationError("Existing pinned source differs; it will not be replaced")
        return
    if not download:
        raise ValidationError("Pinned source is missing; use --download to permit bounded network acquisition")
    request = Request(record["media_url"], headers={"User-Agent": "ProofLoop-reproduction/1.0"})
    with urlopen(request, timeout=60) as response:
        if not response.geturl().startswith("https://"):
            raise ValidationError("Pinned download redirected outside HTTPS")
        _copy_bounded(response, path, record["bytes"], record["sha256"])


def prepare(profile: str, output: Path, *, download: bool = False, root: Path = ROOT) -> dict[str, Any]:
    plan = profile_plan(profile)
    root = root.resolve()
    destination = _private(output if output.is_absolute() else root / output, root, "artifacts/local")
    if destination.exists():
        raise ValidationError("Use a new output folder; previous review or evidence cannot be overwritten")
    _tool("ffmpeg")  # Fail before downloading if the local decoder is unavailable.
    media = _private(root / "inputs/local/reproduction" / profile, root, "inputs/local")
    card = media / "dataset-card.md"
    _obtain({"media_url": CARD_URL, "bytes": CARD_BYTES, "sha256": CARD_SHA256}, card, download)
    if b"license: mit" not in card.read_bytes().lower():
        raise ValidationError("Pinned dataset card no longer carries the reviewed explicit MIT declaration")
    sources = []
    for source in plan["sources"]:
        path = media / f"episode_{source['episode']:06d}.mp4"
        _obtain(source, path, download)
        sources.append({"id": f"coffee-episode-{source['episode']:06d}", "publisher": "LeRobot / ALOHA authors",
                        "page_url": CARD_PAGE, "media_url": source["media_url"],
                        "license": deepcopy(plan["license"]), "file": str(path.relative_to(root)),
                        "sha256": source["sha256"], "duration_ms": 22000})
    destination.mkdir(parents=True, mode=0o700)
    notices = {"ATTRIBUTION.txt": ATTRIBUTION + "\nData license: MIT per pinned dataset card.\n" + CARD_PAGE + "\n",
               "MIT-aloha-LICENSE.txt": MIT_NOTICE}
    for name, text in notices.items():
        if not (media / name).exists():
            _write_new(media / name, text.encode())
    contract = {"schema_version": "1.0", "evidence_origin": "external_recorded_video",
                "dataset_id": f"proofloop-reproduction-{profile}", "sources": sources,
                "requirements": "Verify that the sole transparent drinking glass is in the calibrated dispense "
                                "region under the coffee machine outlet. Both distinct source frames must "
                                "consistently support placement (min_stable_samples=2). A uniquely detected "
                                "glass consistently in staging fails this scoped placement gate. Missing, "
                                "ambiguous, crossing-region or inconsistent evidence remains unknown. "
                                "This does not authorize dispensing or certify a whole procedure or physical safety.",
                "protocol": deepcopy(PROTOCOL), "concepts": {"glass": "transparent drinking glass"}, "cases": [],
                "notes": "Public fixed source/timestamp reproduction profile. All sources were exposed during "
                         "ProofLoop development. Labels are authored assertions pending exact-contract review. "
                         "Normal recorded robot demonstrations do not follow ProofLoop recommendations. "
                         "No independent holdback, continuous tracking or executed repair claim."}
    extraction = []
    by_id = {source["id"]: source for source in sources}
    for episode, stage, times, status in PROFILES[profile]:
        frames = []
        for timestamp in times:
            path = _private(media / "frames" / f"e{episode:02d}-{timestamp:06d}.jpg", root, "inputs/local")
            path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            if path.exists():
                raise ValidationError("Extracted evidence already exists; choose a fresh clone/profile folder")
            record = _extract_frame(root / by_id[f"coffee-episode-{episode:06d}"]["file"],
                                    timestamp / 1000, 22, path)
            record["file"] = str(path.relative_to(root))
            extraction.append(record)
            frames.append({key: record[key] for key in ("file", "sha256", "source_timestamp_ms")})
        rationale = {"unknown": "The first frame shows staging and the later frame dispense. Two consistent "
                                "location observations are not established; UNKNOWN labels the evidence pair.",
                     "fail": "Both frames show the unique glass in staging, contradicting the dispense region gate.",
                     "pass": "Both frames show the unique glass inside the dispense image region. Visible "
                             "gripper overlap does not certify amodal completeness or physical safety."}[status]
        contract["cases"].append({"id": f"coffee-e{episode:02d}-{stage}", "source_id": f"coffee-episode-{episode:06d}",
                                  "role": "benchmark", "regions": deepcopy(REGIONS), "frames": frames,
                                  "evaluation_timestamp_ms": frames[-1]["source_timestamp_ms"],
                                  "annotation": {"status": status, "checks": {"glass-under-spout": status},
                                                 "reason": rationale,
                                                 "origin": "source_frames_reviewed_before_inference"}})
    _json_new(destination / "contract-draft.json", contract)
    _json_new(destination / "extraction.json", {"records": extraction, "timestamp_basis": "decoded video PTS",
                                               "decoder": "ffmpeg; JPEG bytes may differ across decoder versions",
                                               "no_image_content_editing": True})
    result = {**plan, "status": "prepared_unapproved", "contract_hash": manifest_hash(contract),
              "manifest": str((destination / "contract-draft.json").relative_to(root)),
              "review_required": "View every selected frame and review labels, regions and unique-target assumption "
                                 "before approving this exact hash. No model inference has run."}
    _json_new(destination / "preparation.json", result)
    return result


def approve(output: Path, reviewed_hash: str, *, root: Path = ROOT) -> dict[str, Any]:
    root = root.resolve()
    destination = _private(output if output.is_absolute() else root / output, root, "artifacts/local")
    draft = destination / "contract-draft.json"
    if not draft.is_file() or draft.stat().st_size > 2 * 1024 * 1024:
        raise ValidationError("Prepared contract is missing or too large")
    contract = json.loads(draft.read_text(encoding="utf-8"))
    envelope = approve_manifest(contract, reviewed_hash, confirm_unique_targets=True, review_origin="local_operator")
    result = validate_manifest(envelope, project_root=root)
    _json_new(destination / "manifest-approved.json", envelope)
    return {"status": "approved_no_inference", **result}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command")
    plan = commands.add_parser("plan", help="Print fixed source sizes, hashes, license and timestamps; no network")
    plan.add_argument("--profile", choices=PROFILES, default="coffee-placement")
    create = commands.add_parser("prepare", help="Prepare a local unapproved draft; never calls a model")
    create.add_argument("--profile", choices=PROFILES, default="coffee-placement")
    create.add_argument("--output", type=Path, default=Path("artifacts/local/reproduction-coffee"))
    create.add_argument("--download", action="store_true", help="Permit exact bounded downloads from pinned URLs")
    review = commands.add_parser("approve", help="Assert you reviewed all frames, labels, calibration and unique target")
    review.add_argument("--output", type=Path, default=Path("artifacts/local/reproduction-coffee"))
    review.add_argument("--reviewed-hash", required=True, help="Exact prepared contract SHA-256 after visual review")
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            result = prepare(args.profile, args.output, download=args.download)
        elif args.command == "approve":
            result = approve(args.output, args.reviewed_hash)
        else:
            result = profile_plan(getattr(args, "profile", "coffee-placement"))
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (ValidationError, CaptureError, ValueError, OSError) as error:
        print(f"Reproduction stopped: {type(error).__name__}: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
