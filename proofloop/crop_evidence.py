"""Native context-crop evidence, with raw crop responses kept distinct from projection.

This experimental adapter makes no requests and grants no placement acceptance.
A low-scoring original concept proposal may locate a crop; it never becomes an
accepted observation. Cropped and full images remain one physical source view.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import struct
import subprocess
import tempfile
from typing import Any, Mapping

from . import benchmark, guided, mask_semantics, semantic
from .core import ValidationError
from .prospective_audit import _bound_text

VERSION = "native_context_crop_evidence_v1"
CODE_FILES = tuple(dict.fromkeys(mask_semantics.CODE_FILES + ("crop_evidence.py",)))
TRANSFORM = "original_JPEG_decode_to_RGB24_then_integer_crop_to_lossless_PNG_no_resize"
RULE = "side=max(256,4*max(inclusive_candidate_width,height)); if side>min(source_width,height), use full source; otherwise floor-centered clamped square"
LIMITS = ["Derived crop and original frame are one physical observation, not two.",
          "Crop confidence is the unmodified crop-SAM record score, not calibrated correctness.",
          "Full-source identity, uniqueness, visibility, gripper and region checks remain required.",
          "A candidate crop is selected without using placement labels, desired regions or its confidence."]


def _hash(value):
    return benchmark.manifest_hash(value)


def code_hashes():
    folder = Path(__file__).resolve().parent
    return {name: benchmark._file_hash(folder / name) for name in CODE_FILES}


def crop_geometry(box, width, height):
    """Whole-source fallback is declared when the requested square cannot fit."""
    if (type(width) is not int or type(height) is not int or width < 1 or height < 1
            or not isinstance(box, Mapping)
            or any(type(box.get(k)) is not int for k in ("x1", "y1", "x2", "y2"))
            or not 0 <= box["x1"] <= box["x2"] < width
            or not 0 <= box["y1"] <= box["y2"] < height):
        raise ValidationError("Context crop needs an exact inclusive source box")
    side = max(256, 4 * max(box["x2"] - box["x1"] + 1, box["y2"] - box["y1"] + 1))
    if side > min(width, height):
        return {"x1": 0, "y1": 0, "x2": width - 1, "y2": height - 1}
    x = max(0, min((box["x1"] + box["x2"] + 1 - side) // 2, width - side))
    y = max(0, min((box["y1"] + box["y2"] + 1 - side) // 2, height - side))
    return {"x1": x, "y1": y, "x2": x + side - 1, "y2": y + side - 1}


def _png_dimensions(path):
    if path.is_symlink() or not path.is_file() or not 24 <= path.stat().st_size <= 5 * 1024 * 1024:
        raise ValidationError("Crop must be a bounded regular PNG")
    with path.open("rb") as handle:
        header = handle.read(24)
    if header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
        raise ValidationError("Context crop is not a PNG")
    return struct.unpack(">II", header[16:24])


def _crop_png(executable, source, target, rectangle):
    w, h = rectangle["x2"] - rectangle["x1"] + 1, rectangle["y2"] - rectangle["y1"] + 1
    # Convert full decoded scene before cropping, avoiding crop-dependent chroma
    # resampling. Native pixels only; PNG adds no new lossy compression.
    filters = f"format=rgb24,crop=w={w}:h={h}:x={rectangle['x1']}:y={rectangle['y1']}:exact=1"
    args = [executable, "-hide_banner", "-loglevel", "error", "-nostdin", "-i", str(source),
            "-vf", filters, "-frames:v", "1", "-c:v", "png", "-pix_fmt", "rgb24",
            "-threads", "1", "-map_metadata", "-1", str(target)]
    try:
        subprocess.run(args, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       timeout=mask_semantics.MEDIA_TIMEOUT_SECONDS)
    except (OSError, subprocess.SubprocessError):
        raise ValidationError("Bounded native PNG crop extraction failed") from None
    if _png_dimensions(target) != (w, h):
        raise ValidationError("PNG crop dimensions differ from its declared source rectangle")


def _inputs(envelope, frame_index, cached_target_result, cached_locator, root):
    if type(frame_index) is not int or frame_index not in (0, 1):
        raise ValidationError("Context crop must name source view zero or one")
    benchmark._approved(envelope)
    contract = envelope["contract"]
    if len(contract["cases"]) != 1 or len(contract["concepts"]) != 1:
        raise ValidationError("Context crop requires one approved case and one target concept")
    case, = contract["cases"]
    object_id, = contract["concepts"]
    media = benchmark.verify_media(envelope, project_root=root)
    images, frames = guided._frames([media[frame["sha256"]] for frame in case["frames"]])
    _bound_text(cached_locator, [frame["sha256"] for frame in frames], guided.locator_schema([object_id]))
    located = guided._locator_bound(cached_locator, frames, contract["concepts"])
    frame, target = frames[frame_index], located[frame_index]["objects"][0]
    records = mask_semantics._sam(cached_target_result, frame["sha256"], frame["width"], frame["height"])
    vetoes = []
    if target["identity_ambiguous"]:
        vetoes.append("cached_identity_ambiguous")
    if not target["fully_visible"]:
        vetoes.append("cached_not_fully_visible")
    if len(target["candidates"]) != 1:
        vetoes.append("cached_candidate_count_not_one")
    rectangle = crop_geometry(records[0]["box"], frame["width"], frame["height"]) if len(records) == 1 else None
    return {"case_id": case["id"], "frame_index": frame_index, "object_id": object_id,
            "concept": contract["concepts"][object_id], "source_file": str(images[frame_index].resolve()),
            "source_sha256": frame["sha256"], "source_width": frame["width"], "source_height": frame["height"],
            "source_view_sha256s": [item["sha256"] for item in frames],
            "cached_proposal_sha256": _hash(cached_target_result), "cached_locator_sha256": _hash(cached_locator),
            "cached_record_count": len(records), "cached_vetoes": vetoes, "crop_rectangle": rectangle,
            "crop_available": rectangle is not None,
            "full_source_crop": rectangle == {"x1": 0, "y1": 0, "x2": frame["width"] - 1, "y2": frame["height"] - 1}}


def prepare_frame(envelope, frame_index, cached_target_result, cached_locator, out_dir, *, root=benchmark.ROOT):
    """Create crops for every valid single proposal, including low-score/vetoed ones.

    A cached veto still forbids projected support; retaining its diagnostic crop
    does not remove it. Absent/multiple proposals produce no selected crop.
    """
    root = Path(root)
    details = _inputs(envelope, frame_index, cached_target_result, cached_locator, root)
    output = benchmark._local_path(out_dir, root, output=True)
    if output.exists():
        raise ValidationError("Context-crop preparation requires a new private directory")
    executable = semantic._extractor_path() if details["crop_available"] else None
    version = mask_semantics._version(executable) if executable else None
    output.mkdir(parents=True, mode=0o700)
    crop = output / "context.png" if details["crop_available"] else None
    if crop:
        _crop_png(executable, Path(details["source_file"]), crop, details["crop_rectangle"])
    prepared = {"version": VERSION, "rule": RULE, "transform": TRANSFORM,
                "envelope": deepcopy(envelope), "cached_target_result": deepcopy(cached_target_result),
                "cached_locator": deepcopy(cached_locator), "details": details, "output_dir": str(output),
                "crop_file": str(crop) if crop else None,
                "crop_sha256": benchmark._file_hash(crop) if crop else None,
                "crop_bytes": crop.stat().st_size if crop else None, "extractor_version": version,
                "implementation_sha256": code_hashes(), "limits": list(LIMITS)}
    prepared["prepared_sha256"] = _hash(prepared)
    validate_prepared(prepared, root=root)
    benchmark._save(output / "prepared.json", prepared)
    return prepared


def validate_prepared(prepared, *, root=benchmark.ROOT):
    """Reproduce crop bytes and every source/locator/raw-SAM binding before use."""
    try:
        if (not isinstance(prepared, Mapping) or prepared["version"] != VERSION
                or prepared["rule"] != RULE or prepared["transform"] != TRANSFORM
                or prepared["limits"] != LIMITS or prepared["implementation_sha256"] != code_hashes()
                or prepared["prepared_sha256"] != _hash({k: v for k, v in prepared.items() if k != "prepared_sha256"})):
            raise ValidationError("Context-crop declaration or implementation changed")
        details = _inputs(prepared["envelope"], prepared["details"]["frame_index"],
                          prepared["cached_target_result"], prepared["cached_locator"], Path(root))
        if prepared["details"] != details:
            raise ValidationError("Context-crop source, proposal or locator binding changed")
        output = benchmark._local_path(prepared["output_dir"], Path(root), output=True)
        if details["crop_available"]:
            crop = output / "context.png"
            executable = semantic._extractor_path()
            if prepared["extractor_version"] != mask_semantics._version(executable):
                raise ValidationError("Context-crop extractor version changed")
            if (prepared["crop_file"] != str(crop) or crop.is_symlink()
                    or prepared["crop_sha256"] != benchmark._file_hash(crop)
                    or prepared["crop_bytes"] != crop.stat().st_size):
                raise ValidationError("Context-crop file binding changed")
            with tempfile.TemporaryDirectory() as temporary:
                reproduced = Path(temporary) / "context.png"
                _crop_png(executable, Path(details["source_file"]), reproduced, details["crop_rectangle"])
                if benchmark._file_hash(reproduced) != prepared["crop_sha256"]:
                    raise ValidationError("Context crop cannot be reproduced from the exact original source")
        elif any(prepared[key] is not None for key in ("crop_file", "crop_sha256", "crop_bytes", "extractor_version")):
            raise ValidationError("Missing or multiple proposals cannot select a crop")
    except (KeyError, TypeError, OSError, ValueError):
        raise ValidationError("Invalid or changed context-crop preparation") from None


def project_result(prepared, actual_crop_response, *, root=benchmark.ROOT):
    """Retain raw crop geometry, returning separately labeled source projection.

    It is intentionally not a MetaClient-shaped full-source result: no provider
    response is rewritten, and consumers must explicitly accept derived evidence.
    """
    validate_prepared(prepared, root=root)
    details = prepared["details"]
    if not details["crop_available"]:
        raise ValidationError("No crop response is admissible without a unique original proposal")
    rectangle = details["crop_rectangle"]
    width, height = rectangle["x2"] - rectangle["x1"] + 1, rectangle["y2"] - rectangle["y1"] + 1
    records = mask_semantics._sam(actual_crop_response, prepared["crop_sha256"], width, height)
    reasons = list(details["cached_vetoes"])
    if len(records) != 1:
        reasons.append("crop_sam_candidate_count_not_one")
    projected, boundary = [], []
    for record in records:
        box = record["box"]
        touched = [edge for edge, value, limit in (("left", box["x1"], 0), ("top", box["y1"], 0),
                   ("right", box["x2"], width - 1), ("bottom", box["y2"], height - 1)) if value == limit]
        artificial = [edge for edge in touched if ((edge == "left" and rectangle["x1"] > 0)
            or (edge == "top" and rectangle["y1"] > 0)
            or (edge == "right" and rectangle["x2"] < details["source_width"] - 1)
            or (edge == "bottom" and rectangle["y2"] < details["source_height"] - 1))]
        boundary.append({"object_id": record["object_id"], "touched_edges": touched, "artificial_edges": artificial})
        if artificial and "artificial_crop_edge_touched" not in reasons:
            reasons.append("artificial_crop_edge_touched")
        if record.get("confidence") is None:
            if "crop_sam_score_unavailable" not in reasons:
                reasons.append("crop_sam_score_unavailable")
        elif record["confidence"] < .8 and "crop_sam_score_below_0_8" not in reasons:
            reasons.append("crop_sam_score_below_0_8")
        translated = deepcopy(record)
        translated["box"] = {"x1": box["x1"] + rectangle["x1"], "y1": box["y1"] + rectangle["y1"],
                             "x2": box["x2"] + rectangle["x1"], "y2": box["y2"] + rectangle["y1"],
                             "width": details["source_width"], "height": details["source_height"]}
        projected.append(translated)
    result = {"version": VERSION, "prepared_sha256": prepared["prepared_sha256"],
              "source_sha256": details["source_sha256"], "crop_sha256": prepared["crop_sha256"],
              "frame_index": details["frame_index"], "object_id": details["object_id"],
              "crop_rectangle": deepcopy(rectangle), "actual_crop_response": deepcopy(actual_crop_response),
              "derived_full_source_geometry": projected, "crop_boundary_receipts": boundary,
              "withheld_reasons": reasons, "status": "eligible_geometry" if not reasons else "withheld",
              "physical_source_view_count": 1, "full_scene_semantic_assessment_required": True,
              "gripper_and_original_region_checks_required": True, "placement_accepted": False,
              "limits": list(LIMITS)}
    result["projection_sha256"] = _hash(result)
    return result


def validate_projection(prepared, projection, *, root=benchmark.ROOT):
    if (not isinstance(projection, Mapping) or "actual_crop_response" not in projection
            or project_result(prepared, projection["actual_crop_response"], root=root) != projection):
        raise ValidationError("Derived crop geometry or response binding changed")
