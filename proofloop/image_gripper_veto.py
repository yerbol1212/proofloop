"""Negative-only gripper proximity on exact-source still-image SAM masks.

A near proposal can withhold an existing decision. Far, missing or invalid
proposals cannot establish physical separation, visibility, absence or support.
"""
from __future__ import annotations

from copy import deepcopy
import json
import re
import time
from typing import Mapping

from . import benchmark, gripper_geometry, mask_semantics, tracking_geometry
from .core import ValidationError
from .meta import MetaAPIError

VERSION = "image_negative_only_gripper_veto_v1"
MAX_SOURCE_MAPPED_PIXELS = 32_000_000
MAX_GEOMETRY_SECONDS = 30.0
MAX_GRIPPER_PROPOSALS = 64
MAX_RESULT_BYTES = 64 * 1024 * 1024
CODE_FILES = tuple(dict.fromkeys(mask_semantics.CODE_FILES + (
    "gripper_geometry.py", "image_gripper_veto.py")))
CONFIG = {"source_pixel_margin": 3, "distance_cap": 4,
          "distance_metric": "chebyshev_foreground_pixel_centers",
          "mapping": "box_local_mask_nearest_source_pixel_center_inclusive_xyxy",
          "gripper_concept": "robot gripper", "max_gripper_proposals": MAX_GRIPPER_PROPOSALS,
          "max_source_mapped_pixels": MAX_SOURCE_MAPPED_PIXELS,
          "max_geometry_seconds": MAX_GEOMETRY_SECONDS,
          "confidence_usage": "reported_only_for_negative_risk_never_positive_qualification"}
REASON_CODES = frozenset({"geometry_time_limit", "geometry_pixel_limit", "target_candidate_count_not_one",
                         "gripper_missing_not_verified_absent", "mask_geometry_unavailable"} | {
    role + "_" + suffix for role in ("target", "gripper") for suffix in (
        "result_unavailable_or_excessive", "mask_invalid", "proposal_limit",
        "raw_mask_or_source_invalid", "instance_ids_invalid", "invalid_image_frame")})


def _receipt(value):
    try:
        raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    except (ValueError, TypeError, RecursionError):
        return None
    if len(raw) > MAX_RESULT_BYTES:
        return None
    return benchmark.manifest_hash(value)


def assess_frame(target_result, gripper_result, *, source_sha256, width, height,
                 target_object_id="target"):
    """Re-decode and compare actual masks; return only safe measurements/receipts.

    Caller-supplied concept/request identities require a separate frozen request
    binding. SAM instance ids alone do not authenticate semantic object classes.
    No image, invented score, source projection or positive observation is made.
    """
    benchmark._digest(source_sha256)
    if (type(width) is not int or type(height) is not int or not 1 <= width <= 1280
            or not 1 <= height <= 1280 or not isinstance(target_object_id, str)
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", target_object_id)):
        raise ValidationError("Declare exact bounded source dimensions and semantic target id")
    deadline = time.monotonic() + MAX_GEOMETRY_SECONDS
    receipts = {"target_raw_result_sha256": _receipt(target_result),
                "gripper_raw_result_sha256": _receipt(gripper_result)}
    base = {"method_version": VERSION, "config": deepcopy(CONFIG), "source_sha256": source_sha256,
            "width": width, "height": height, "target_object_id": target_object_id,
            "warning": "geometry_unavailable", "veto": False, "geometry_available": False,
            "comparisons": [], "reason_codes": [], "mapped_source_pixels": 0,
            "positive_support_can_be_granted": False, "separation_verified": False,
            "absence_verified": False, "full_visibility_verified": False,
            "identity_independently_verified": False, "physical_contact_observed": False,
            "physical_occlusion_observed": False, "acceptance_confidence_threshold_changed": False,
            "changes_production_acceptance": False, **receipts}

    def records(value, role):
        if receipts[role + "_raw_result_sha256"] is None:
            raise ValueError(role + "_result_unavailable_or_excessive")
        if not isinstance(value, Mapping) or not isinstance(value.get("objects"), list):
            raise ValueError(role + "_mask_invalid")
        if len(value["objects"]) > MAX_GRIPPER_PROPOSALS:
            raise ValueError(role + "_proposal_limit")
        try:
            parsed = mask_semantics._sam(value, source_sha256, width, height)
        except (ValidationError, MetaAPIError, ValueError, TypeError, KeyError):
            raise ValueError(role + "_raw_mask_or_source_invalid") from None
        if any(type(row.get("frame")) is not int or row["frame"] != 0 for row in parsed):
            raise ValueError(role + "_invalid_image_frame")
        ids = [row["object_id"] for row in parsed]
        if (len(ids) != len(set(ids)) or any(type(i) is not int or not 0 <= i <= 2 ** 31 - 1 for i in ids)):
            raise ValueError(role + "_instance_ids_invalid")
        if time.monotonic() > deadline:
            raise ValueError("geometry_time_limit")
        return parsed

    def mapped(record):
        box = tracking_geometry._box(record, width, height)
        area = (box["x2"] - box["x1"] + 1) * (box["y2"] - box["y1"] + 1)
        if base["mapped_source_pixels"] + area > MAX_SOURCE_MAPPED_PIXELS:
            raise ValueError("geometry_pixel_limit")
        if time.monotonic() > deadline:
            raise ValueError("geometry_time_limit")
        base["mapped_source_pixels"] += area
        value = tracking_geometry._mapped(record, width, height)
        return value, gripper_geometry._rows(value, deadline)

    try:
        targets = records(target_result, "target")
        if len(targets) != 1:
            raise ValueError("target_candidate_count_not_one")
        grippers = records(gripper_result, "gripper")
        if not grippers:
            raise ValueError("gripper_missing_not_verified_absent")
        target, = targets
        target_mapped, target_rows = mapped(target)
        for gripper in grippers:
            gripper_mapped, gripper_rows = mapped(gripper)
            distance = gripper_geometry._distance_capped(target_rows, gripper_rows, deadline)
            hashes = {}
            for role, value in (("target", target_mapped), ("gripper", gripper_mapped)):
                for field in ("raster_sha256", "mapped_mask_sha256", "foreground_pixels"):
                    hashes[role + "_" + field] = value[field]
            base["comparisons"].append({"target_sam_object_id": target["object_id"],
                "gripper_sam_object_id": gripper["object_id"],
                "target_confidence": target.get("confidence"), "gripper_confidence": gripper.get("confidence"),
                "min_chebyshev_distance_capped4": distance, "distance_is_lower_bound": distance == 4,
                "near_proposal": distance <= 3, **hashes})
        base["geometry_available"] = True
        base["veto"] = any(row["near_proposal"] for row in base["comparisons"])
        base["warning"] = "near_gripper_proposal" if base["veto"] else "no_near_proposal_observed"
    except (ValueError, MetaAPIError, ValidationError, TypeError, KeyError, OverflowError) as error:
        # Even a partial near measurement remains a negative warning. Incomplete
        # later comparisons cannot turn a measured near proposal into clear.
        reason = str(error) if str(error) in REASON_CODES else "mask_geometry_unavailable"
        base["reason_codes"] = [reason]
        base["veto"] = any(row["near_proposal"] for row in base["comparisons"])
        if base["veto"]:
            base["warning"] = "near_gripper_proposal"
    base["receipt_sha256"] = benchmark.manifest_hash(base)
    return base
