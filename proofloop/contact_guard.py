"""Experimental source-bound assessment of visible boundaries near contact.

The native full frame and context crop retain the v2 preparation contract. A
registered nearest-neighbor enlargement presents those same crop pixels; it is
not super-resolution or evidence of an invisible surface. Boundary uncertainty
withholds support without inventing a physical cause or authorizing an action.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import struct
import subprocess
import tempfile
from typing import Any, Mapping

from . import visibility_guard as native
from .core import ValidationError
from .meta import MAX_IMAGE_BYTES, MetaAPIError, MetaClient, SPARK_MODEL, output_text, validate_json

VERSION = "sam_video_boundary_contact_guard_v3"
SIDES = ("left", "right", "top", "bottom")
RELATIONS = ("clear_background_gap", "visible_target_boundary", "external_overlap", "unresolved_contact", "uncertain")
CONFIG = {"reasoning_effort": "low", "max_output_tokens": 4096, "max_requests": 1, "retries": 0,
          "crop_rule": native.CROP_RULE, "image_order": ["full_frame", "candidate_crop", "crop_enlargement"],
          "enlargement": {"factor": 3, "interpolation": "nearest_neighbor", "format": "png_rgb24",
                          "max_pixels": 16_000_000, "adds_source_detail": False},
          "boundary_rule": "four_projected_silhouette_regions_directly_resolved_v1"}

digest = native.digest


def _png_dimensions(data: bytes) -> tuple[int, int]:
    if len(data) < 24 or data[:8] != b"\x89PNG\r\n\x1a\n" or data[12:16] != b"IHDR":
        raise ValidationError("Contact enlargement must be a PNG image")
    return struct.unpack(">II", data[16:24])


def _enlarge(executable: str, source: Path, target: Path) -> None:
    """Present decoded native crop pixels at 3x with no interpolated detail."""
    args = [executable, "-hide_banner", "-loglevel", "error", "-nostdin", "-i", str(source),
            "-vf", "scale=iw*3:ih*3:flags=neighbor", "-frames:v", "1", "-pix_fmt", "rgb24",
            "-threads", "1", "-map_metadata", "-1", str(target)]
    try:
        subprocess.run(args, check=True, capture_output=True, timeout=20)
    except (subprocess.SubprocessError, OSError):
        raise ValidationError("Deterministic contact enlargement failed") from None
    target.chmod(0o600)


def schema(contract: Mapping[str, Any]) -> dict[str, Any]:
    result = native.schema(contract)
    boundary = {"type": "object", "additionalProperties": False,
        "required": ["relation", "neighbor", "observation"], "properties": {
            "relation": {"type": "string", "enum": list(RELATIONS)},
            "neighbor": {"type": "string", "maxLength": 120},
            "observation": {"type": "string", "minLength": 1, "maxLength": 200}}}
    result["required"].append("boundary_relations")
    result["properties"]["boundary_relations"] = {"type": "object", "additionalProperties": False,
        "required": list(SIDES), "properties": {side: deepcopy(boundary) for side in SIDES}}
    return result


def prompt(prepared: Mapping[str, Any]) -> str:
    # The v2 instructions remain the reference-free identity/visibility baseline;
    # this additional observational schema addresses contact ambiguity directly.
    registration = {"image_3": {"role": "nearest_neighbor_enlargement_of_image_2",
        "width": prepared["enlargement_width"], "height": prepared["enlargement_height"],
        "factor": 3, "adds_source_detail": False,
        "candidate_rectangle_inclusive_pixels": prepared["candidate_in_enlargement"]}}
    return (native.prompt(prepared)
        + "\nThere is also image 3: a deterministic 3x nearest-neighbor PNG enlargement of image 2. "
        "It repeats the SAME decoded source crop pixels. It supplies NO new detail, reference frame, "
        "restoration, segmentation mask or prior identity. Use all three registered views of this ONE "
        "frame to inspect boundary evidence; enlarged blocks cannot resolve information absent in the native frame.\n"
        "Before assigning physical visibility, inspect the actual target's current-view projected silhouette "
        "in four image-relative regions: left, right, top, bottom. These refer to the target's actual outline, "
        "NOT the proposal rectangle's edges, and do not require seeing a backside or all 3D surfaces. "
        "For EACH region report a relation, neighbor (empty if absent), and concise direct pixel observation. "
        "clear_background_gap means a visible strip of background separates the target's outline from "
        "nearby objects at that region. visible_target_boundary means the entire relevant outline is "
        "continuously traceable and directly distinguishable, even with adjacent objects. "
        "A visible dark gap can establish separation; proximity alone is not occlusion. "
        "external_overlap means a visible external object covers or interrupts a part of the target's "
        "projected outline; name that object and mark physical occlusion consistently. "
        "unresolved_contact means a neighbor abuts/merges with the target appearance and ownership or "
        "continuation of that boundary cannot be directly resolved, including low-contrast white-on-white "
        "contact with a gripper. Do NOT infer a complete boundary from expected shape, recognizable identity, "
        "a mostly visible body, or absence of obvious overlap. Name the contacting neighbor. "
        "unresolved_contact is uncertainty, not proof that a gripper physically hides the object. "
        "uncertain means insufficient pixels, blur, crop/full-frame extent or other ambiguity prevents "
        "resolving that region. Do not invent a gap or require one when a distinct complete outline is visible.\n"
        "clear requires all four relations to be clear_background_gap or visible_target_boundary, along "
        "with the earlier complete-boundary, extremity, blur and identity requirements. Any external_overlap, "
        "unresolved_contact or uncertain region forbids clear. For unresolved contact use uncertain physical "
        "visibility rather than asserting an unobserved physical cause. Actual external hiding remains "
        "occluded even if the target is easy to identify. Observations are direct visible evidence, not "
        "reasoning steps. Return only JSON matching the supplied schema, including boundary_relations.\n"
        + json.dumps({"additional_image_registration": registration}, sort_keys=True))


def prepare(frame_path: str | Path, candidate_box: Mapping[str, int], contract: Mapping[str, Any],
            output_dir: str | Path, *, source_sha256: str, frame_index: int, track_id: int) -> dict[str, Any]:
    base = native.prepare(frame_path, candidate_box, contract, output_dir, source_sha256=source_sha256,
                          frame_index=frame_index, track_id=track_id)
    if base["crop_width"] * base["crop_height"] * 9 > CONFIG["enlargement"]["max_pixels"]:
        raise ValidationError("Contact enlargement exceeds bounded pixel allocation")
    enlarged = Path(output_dir).resolve() / "candidate-nearest-3x.png"
    _enlarge(native._extractor_path(), Path(base["crop_file"]), enlarged)
    data = enlarged.read_bytes()
    width, height = _png_dimensions(data)
    if ((width, height) != (base["crop_width"] * 3, base["crop_height"] * 3)
            or not 0 < len(data) <= MAX_IMAGE_BYTES):
        raise ValidationError("Contact enlargement differs from its registered crop")
    result = deepcopy(base)
    result.update({"method_version": VERSION, "config": deepcopy(CONFIG),
        "native_preparation_hash": base["prepared_hash"], "enlargement_file": str(enlarged),
        "enlargement_sha256": hashlib.sha256(data).hexdigest(), "enlargement_width": width,
        "enlargement_height": height, "candidate_in_enlargement": {
            key: value * 3 + (2 if key in {"x2", "y2"} else 0)
            for key, value in base["candidate_in_crop"].items()}})
    result["prepared_hash"] = digest({key: value for key, value in result.items() if key != "prepared_hash"})
    verify(result)
    return result


def verify(prepared: Mapping[str, Any]) -> None:
    try:
        if (prepared["method_version"] != VERSION or prepared["config"] != CONFIG
                or prepared["prepared_hash"] != digest({key: value for key, value in prepared.items() if key != "prepared_hash"})):
            raise ValueError
        base = {key: deepcopy(value) for key, value in prepared.items() if key not in {
            "native_preparation_hash", "enlargement_file", "enlargement_sha256", "enlargement_width",
            "enlargement_height", "candidate_in_enlargement"}}
        base.update({"method_version": native.VERSION, "config": deepcopy(native.CONFIG),
                     "prepared_hash": prepared["native_preparation_hash"]})
        native.verify(base)
        data = Path(prepared["enlargement_file"]).read_bytes()
        width, height = prepared["crop_width"] * 3, prepared["crop_height"] * 3
        if (width * height > CONFIG["enlargement"]["max_pixels"]
                or not 0 < len(data) <= MAX_IMAGE_BYTES
                or hashlib.sha256(data).hexdigest() != prepared["enlargement_sha256"]
                or _png_dimensions(data) != (width, height)
                or (prepared["enlargement_width"], prepared["enlargement_height"]) != (width, height)
                or prepared["candidate_in_enlargement"] != {key: value * 3 + (2 if key in {"x2", "y2"} else 0)
                    for key, value in prepared["candidate_in_crop"].items()}):
            raise ValueError
        with tempfile.TemporaryDirectory(prefix="proofloop-contact-enlargement-replay-") as temporary:
            replay = Path(temporary) / "candidate-nearest-3x.png"
            _enlarge(native._extractor_path(), Path(prepared["crop_file"]), replay)
            if hashlib.sha256(replay.read_bytes()).hexdigest() != prepared["enlargement_sha256"]:
                raise ValueError
    except (OSError, ValueError, KeyError, TypeError):
        raise ValidationError("Contact source or preparation changed") from None


def evaluate(parsed: Any, contract: Mapping[str, Any]) -> dict[str, Any]:
    validate_json(parsed, schema(contract))
    base = native.evaluate({key: value for key, value in parsed.items() if key != "boundary_relations"}, contract)
    boundaries = parsed["boundary_relations"]
    if any(not row["observation"].strip() for row in boundaries.values()):
        raise ValueError("Contact boundary observations must contain current-view evidence")
    boundary_issues = []
    resolved = all(row["relation"] in {"clear_background_gap", "visible_target_boundary"}
                   for row in boundaries.values())
    visibility = parsed["physical_visibility"]
    if visibility["classification"] == "clear" and not resolved:
        boundary_issues.append("clear_visibility_conflicts_with_boundary_relations")
    for side, row in boundaries.items():
        if row["relation"] in {"external_overlap", "unresolved_contact"} and not row["neighbor"].strip():
            boundary_issues.append(side + "_contact_lacks_named_neighbor")
        if row["relation"] == "external_overlap" and not visibility["external_occluder_hides_target"]:
            boundary_issues.append(side + "_overlap_conflicts_with_physical_visibility")
    # Keep the physical diagnostic independent of semantic identity, as in v2.
    # Identity still gates supported; a mistaken identity cannot authorize it.
    issues = list(base["consistency_issues"]) + boundary_issues
    fully_visible = base["fully_visible"] and resolved and not boundary_issues
    supported = base["decision"] == "supported" and fully_visible
    base.update({"decision": "supported" if supported else "contradicted" if base["decision"] == "contradicted" else "unknown",
        "fully_visible": fully_visible, "boundary_relations": deepcopy(boundaries),
        "all_projected_boundary_regions_resolved": resolved, "consistency_issues": issues})
    return base


def _bound_parsed(response: Mapping[str, Any], prepared: Mapping[str, Any]) -> Any:
    if (response.get("source") is not None or [item.get("sha256") for item in response.get("sources", [])]
            != [prepared["frame_sha256"], prepared["crop_sha256"], prepared["enlargement_sha256"]]
            or response.get("response", {}).get("status") != "completed"
            or response["response"].get("model") != SPARK_MODEL):
        raise ValidationError("Contact response differs from its registered source images")
    text = output_text(response["response"])
    parsed = json.loads(text)
    if response.get("text") != text or response.get("parsed") != parsed:
        raise ValidationError("Contact classification differs from retained provider output")
    return parsed


def challenge(client: MetaClient, prepared: Mapping[str, Any]) -> dict[str, Any]:
    verify(prepared)
    if client.max_requests != 1 or client.retries != 0 or client.ledger:
        raise ValidationError("Contact guard requires one fresh zero-retry client")
    result = {"method_version": VERSION, "prepared": deepcopy(prepared), "status": "unscored",
              "response": None, "evaluation": None, "error": None, "usage": None}
    try:
        response = client.spark_json(prompt(prepared), schema(prepared["contract"]),
            name="proofloop_boundary_contact_guard_v3", image_paths=[prepared["source_file"], prepared["crop_file"],
                prepared["enlargement_file"]], reasoning_effort=CONFIG["reasoning_effort"],
            max_output_tokens=CONFIG["max_output_tokens"])
        result["response"] = response
        result["evaluation"] = evaluate(_bound_parsed(response, prepared), prepared["contract"])
        result["status"] = "completed"
    except MetaAPIError as error:
        result["error"] = {"code": error.code, "http_status": error.status, "response": error.response}
    except (ValidationError, ValueError, TypeError, KeyError):
        result["error"] = {"code": "local_validation_failed", "http_status": None, "response": result["response"]}
    result["usage"] = client.usage_summary()
    verify(prepared)
    return result


assess = challenge


def public_summary(result: Mapping[str, Any]) -> dict[str, Any]:
    """Recompute provider-bound flags; publish hashes of every free-form string."""
    verify(result["prepared"])
    prepared = result["prepared"]
    if result.get("method_version") != VERSION or result.get("status") not in {"completed", "unscored"}:
        raise ValidationError("Invalid contact result")
    evaluation = None
    if result["status"] == "completed":
        try:
            evaluation = evaluate(_bound_parsed(result["response"], prepared), prepared["contract"])
        except (MetaAPIError, ValueError, TypeError, KeyError):
            raise ValidationError("Contact result lost its registered provider evidence") from None
        if evaluation != result["evaluation"] or result.get("error") is not None:
            raise ValidationError("Contact result evaluation changed")
    elif result.get("evaluation") is not None:
        raise ValidationError("Unscored contact result cannot contain a classification")
    usage = result.get("usage") or {}
    numeric_usage = {key: usage[key] for key in ("request_attempts", "input_tokens", "output_tokens", "sam_images",
        "spark_requests_with_unreported_token_usage", "unsuccessful_spark_requests_with_reported_token_usage", "estimated_usd")
        if type(usage.get(key)) in {int, float} and math.isfinite(usage[key]) and usage[key] >= 0}
    classification = deepcopy(evaluation)
    if classification is not None:
        physical = classification["physical_visibility"]
        landmarks = physical.pop("visible_terminal_landmarks")
        physical["visible_terminal_landmarks_count"] = len(landmarks)
        physical["visible_terminal_landmarks_sha256"] = digest(landmarks)
        for field in ("occluder", "affected_boundary"):
            named = physical.pop(field)
            physical[field + "_named"] = bool(named.strip())
            physical[field + "_sha256"] = digest(named)
        for boundary in classification["boundary_relations"].values():
            for field in ("neighbor", "observation"):
                named = boundary.pop(field)
                boundary[field + "_named"] = bool(named.strip())
                boundary[field + "_sha256"] = digest(named)
    return {"method_version": VERSION, "status": result["status"],
        "prepared_hash": prepared["prepared_hash"], "native_preparation_hash": prepared["native_preparation_hash"],
        "source_sha256": prepared["source_sha256"], "frame_sha256": prepared["frame_sha256"],
        "crop_sha256": prepared["crop_sha256"], "enlargement_sha256": prepared["enlargement_sha256"],
        "frame_index": prepared["frame_index"], "track_id": prepared["track_id"],
        "candidate_box": deepcopy(prepared["candidate_box"]),
        "response_sha256": digest(result["response"]) if result.get("response") is not None else None,
        "classification": classification, "usage": numeric_usage, "physical_action_authorized": False}
