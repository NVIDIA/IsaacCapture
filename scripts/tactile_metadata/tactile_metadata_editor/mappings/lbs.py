# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Raw four-slot editing, validation and point-only LBS preview."""

from __future__ import annotations

from dataclasses import dataclass
import math

from ..domain import Issue


def _number(value):
    return not isinstance(value, bool) and (
        isinstance(value, int)
        or (isinstance(value, float) and math.isfinite(value))
    )


def validate_slots(
    indices, weights, segment_count, *, device_id=None, entity_id=None
):
    issues = []
    for field, row in (("indices", indices), ("weights", weights)):
        if not isinstance(row, (tuple, list)) or len(row) != 4:
            issues.append(Issue(
                "error", "slot_shape", "Expected four raw slots",
                device_id, entity_id, field,
            ))
            continue
        for slot, value in enumerate(row):
            if field == "indices":
                valid = (
                    _number(value) and value == int(value)
                    and 0 <= value < segment_count
                )
                reason = "Segment row is missing, nonintegral or out of range"
            else:
                valid = _number(value) and value >= 0
                reason = "Weight must be finite and nonnegative"
            if not valid:
                issues.append(Issue(
                    "error", "slot_value", reason, device_id, entity_id,
                    f"{field}[{slot}]",
                ))
    if (
        isinstance(weights, (tuple, list)) and len(weights) == 4
        and all(_number(x) for x in weights)
    ):
        try:
            total = sum(weights)
            invalid_sum = abs(total - 1) > 1e-5
        except OverflowError:
            invalid_sum = True
        if invalid_sum:
            issues.append(Issue(
                "error", "weight_sum", "Weights must sum to one",
                device_id, entity_id, "weights",
            ))
    return tuple(issues)


def edit_weights(
    weights, slot, value, *, source="number", participating=(0, 1, 2, 3)
):
    if type(slot) is not int or slot < 0:
        raise ValueError("Weight slot must be a nonnegative integer")
    if source == "number":
        if not isinstance(weights, (tuple, list)):
            raise ValueError(
                "Replace the malformed weight row before editing its fields"
            )
        if slot >= max(len(weights), 4):
            raise ValueError(
                "Only an existing field or a missing four-slot field can be edited"
            )
        result = list(weights)
        while len(result) <= slot:
            result.append(None)
        result[slot] = value
        return tuple(result)
    if source != "slider":
        raise ValueError("Weight source must be number or slider")
    if (
        not isinstance(weights, (tuple, list)) or len(weights) != 4
        or slot not in range(4)
    ):
        raise ValueError("Proportional slider editing requires four slots")
    result = list(weights)
    participants = tuple(participating)
    if (
        len(set(participants)) != len(participants) or slot not in participants
        or any(item not in range(4) for item in participants)
    ):
        raise ValueError("Invalid participating weight slots")
    if not _number(value) or not 0 <= value <= 1:
        raise ValueError("Slider value must be between zero and one")
    remaining = tuple(item for item in participants if item != slot)
    if not remaining:
        result[slot] = 1.0
        return tuple(result)
    if any(not _number(weights[item]) or weights[item] < 0 for item in remaining):
        raise ValueError(
            "Enter valid remaining weights before proportional slider editing"
        )
    try:
        total = sum(weights[item] for item in remaining)
    except OverflowError as error:
        raise ValueError("Remaining weights exceed numeric preview range") from error
    # Finite terms can overflow their floating-point sum.
    if isinstance(total, float) and not math.isfinite(total):
        raise ValueError("Remaining weights exceed numeric preview range")
    result[slot] = value
    for item in remaining:
        proportion = weights[item] / total if total else 1 / len(remaining)
        result[item] = (1 - value) * proportion
    return tuple(result)


@dataclass(frozen=True)
class LBSPreview:
    positions: object
    valid: object
    rigid_poses: tuple
    issues: tuple[Issue, ...]


def evaluate(
    segment_bind_poses, taxel_bind_poses, indices, weights, current_segments,
    *, current_valid=None,
):
    import numpy as np

    from ..geometry import immutable_array, matrix_pose, pose_matrix

    count = len(taxel_bind_poses)
    positions = np.full((count, 3), np.nan)
    valid = np.zeros(count, dtype=bool)
    rigid = [None] * count
    issues = []
    segment_transforms = {}
    for row in range(count):
        try:
            if row >= len(indices) or row >= len(weights):
                raise ValueError("Skinning row is missing")
            failures = validate_slots(
                indices[row], weights[row], len(segment_bind_poses)
            )
            if failures:
                raise ValueError(failures[0].message)
            bind = pose_matrix(taxel_bind_poses[row])
            point = np.append(bind[:3, 3], 1)
            blended = np.zeros(3)
            active = set()
            transforms = {}
            for slot in range(4):
                index = int(indices[row][slot])
                weight = weights[row][slot]
                if weight == 0:
                    continue
                if index >= len(current_segments) or (
                    current_valid is not None and not current_valid[index]
                ):
                    raise ValueError("A required current segment is unavailable")
                if index not in segment_transforms:
                    segment_transforms[index] = (
                        pose_matrix(current_segments[index])
                        @ np.linalg.inv(pose_matrix(segment_bind_poses[index]))
                    )
                transform = segment_transforms[index]
                blended += weight * (transform @ point)[:3]
                active.add(index)
                transforms[index] = transform
            positions[row] = blended
            valid[row] = True
            if len(active) == 1:
                rigid[row] = matrix_pose(transforms[next(iter(active))] @ bind)
        except (ValueError, TypeError, IndexError, np.linalg.LinAlgError) as error:
            issues.append(Issue(
                "error", "lbs_preview", str(error), entity_id=str(row),
                field="taxel_bind_poses",
            ))
    return LBSPreview(
        immutable_array(positions), immutable_array(valid, bool),
        tuple(rigid), tuple(issues),
    )


class LBSMapping:
    edit_weights = staticmethod(edit_weights)
    validate_slots = staticmethod(validate_slots)
    evaluate = staticmethod(evaluate)
