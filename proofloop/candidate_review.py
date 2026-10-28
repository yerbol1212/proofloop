"""Source-bound, explicitly authored candidate boxes for semantic stress tests.

These boxes test candidate classification, not automatic target localization.
The shared v1 preparation format describes inputs, not a Spark locator run.
No model response or locator confidence is fabricated for authored candidates.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .core import ValidationError
from .guided import _frames
from .semantic import (SEMANTIC_VERSION, _crop, _crop_geometry, _extractor_path,
                       _extractor_version, _hash, validate_contract, verify_prepared)

PREPARATION_VERSION = "authored_candidate_pair_v1"
CROP_RULE = "25_percent_context_min_8_pixels_clipped_no_upsampling_jpeg_q2_yuvj444p"
_FRAME_KEYS = {"frame_index", "sha256", "width", "height", "source_file", "crop_file",
               "candidate_box", "crop_box", "crop_sha256"}
_TOP_KEYS = {"method_version", "preparation_version", "anchor_origin", "author_review",
             "contract", "contract_hash", "crop_rule", "extractor", "frames", "prepared_hash"}


def _box(box: Any, width: int, height: int) -> dict[str, int]:
    if (not isinstance(box, Mapping) or set(box) != {"x1", "y1", "x2", "y2"}
            or any(type(value) is not int for value in box.values())
            or not 0 <= box["x1"] < box["x2"] < width
            or not 0 <= box["y1"] < box["y2"] < height):
        raise ValidationError("Author candidate must be a positive-area integer box inside its source")
    return deepcopy(dict(box))


def verify_authored_prepared(prepared: Mapping[str, Any]) -> None:
    """Verify honest anchor provenance as well as source and reproduced crop bytes."""
    try:
        review = prepared["author_review"]
        if (set(prepared) != _TOP_KEYS or prepared["preparation_version"] != PREPARATION_VERSION
                or prepared["anchor_origin"] != "authored_candidate" or not isinstance(review, Mapping)
                or set(review) != {"origin", "source_views_and_crops_reviewed", "note"}
                or review["origin"] != "assistant_under_user_authorization"
                or review["source_views_and_crops_reviewed"] is not True
                or not isinstance(review["note"], str) or not 1 <= len(review["note"].strip()) <= 1500
                or prepared["crop_rule"] != CROP_RULE
                or not isinstance(prepared["frames"], list) or len(prepared["frames"]) != 2):
            raise ValidationError("Authored candidate provenance is invalid")
        for frame in prepared["frames"]:
            if not isinstance(frame, Mapping) or set(frame) != _FRAME_KEYS:
                raise ValidationError("Authored candidate cannot contain model-locator claims")
            _box(frame["candidate_box"], frame["width"], frame["height"])
        verify_prepared(prepared)
    except (KeyError, TypeError, ValueError):
        raise ValidationError("Authored candidate preparation failed replay") from None


def prepare_authored_candidate_pair(image_paths: Sequence[str | Path], boxes: Sequence[Mapping[str, int]],
                                    contract: Mapping[str, Any], output_dir: str | Path, *,
                                    review_note: str) -> dict[str, Any]:
    """Create exact source crops for reviewed real or authored test images; zero inference."""
    contract = validate_contract(contract)
    paths, frames = _frames(image_paths)
    if not isinstance(boxes, (list, tuple)) or len(boxes) != 2:
        raise ValidationError("Author exactly one candidate box in each source view")
    reviewed = [_box(box, frame["width"], frame["height"]) for box, frame in zip(boxes, frames)]
    if not isinstance(review_note, str) or not 1 <= len(review_note.strip()) <= 1500:
        raise ValidationError("Record a concise source and crop review note")
    executable = _extractor_path()
    version = _extractor_version(executable)
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=False, mode=0o700)
    prepared_frames = []
    for path, frame, box in zip(paths, frames, reviewed):
        crop_box = _crop_geometry(box, frame["width"], frame["height"])
        crop_path = out / f"candidate-{frame['frame_index']}.jpg"
        _crop(executable, path, crop_path, crop_box)
        prepared_frames.append(dict(frame, source_file=str(path.resolve()), crop_file=str(crop_path),
            candidate_box=box, crop_box=crop_box, crop_sha256=hashlib.sha256(crop_path.read_bytes()).hexdigest()))
    prepared = {"method_version": SEMANTIC_VERSION, "preparation_version": PREPARATION_VERSION,
        "anchor_origin": "authored_candidate", "author_review": {
            "origin": "assistant_under_user_authorization", "source_views_and_crops_reviewed": True,
            "note": review_note}, "contract": contract, "contract_hash": _hash(contract),
        "crop_rule": CROP_RULE, "extractor": {"version": version}, "frames": prepared_frames}
    prepared["prepared_hash"] = _hash(prepared)
    verify_authored_prepared(prepared)
    target = out / "prepared.json"
    target.write_text(json.dumps(prepared, sort_keys=True, indent=2) + "\n")
    target.chmod(0o600)
    return prepared
