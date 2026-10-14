# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SOMA model checks shared by body and hand evaluators."""

from typing import Any

import numpy as np

from isaaccapture.schema import SomaBodyJoint, SomaHandJoint


SOMA_REFERENCE_VERSION = "v0.3.1"

_BODY_JOINT_NAMES = tuple(
    name
    for name, joint in sorted(
        SomaBodyJoint.__members__.items(), key=lambda item: int(item[1])
    )
    if name != "NUM_JOINTS"
)
_HAND_JOINT_NAMES = tuple(
    name
    for name, joint in sorted(
        SomaHandJoint.__members__.items(), key=lambda item: int(item[1])
    )
    if name != "NUM_JOINTS"
)

# Parent indices in the transport's 25-joint hand order. Wrist is self-parented
# in SOMA-X; each finger chain starts at Wrist.
_HAND_PARENT_IDS = (
    0,
    0,
    1,
    2,
    3,
    0,
    5,
    6,
    7,
    8,
    0,
    10,
    11,
    12,
    13,
    0,
    15,
    16,
    17,
    18,
    0,
    20,
    21,
    22,
    23,
)


def _normalized(name: Any) -> str:
    return str(name).replace("_", "").upper()


def _numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _validate_meters(layer: Any, label: str) -> None:
    output_unit = getattr(layer, "output_unit", None)
    if getattr(output_unit, "meters_per_unit", None) != 1.0:
        raise ValueError(f"{label} must use meters")


def _resolve_reference(layer: Any, joint_count: int, label: str) -> Any:
    resolver = getattr(layer, "get_reference_pose", None)
    if not callable(resolver):
        raise ValueError(f"{label} does not expose reference-pose selection")
    try:
        reference = resolver(version=SOMA_REFERENCE_VERSION)
    except (AttributeError, KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"{label} is incompatible with the SOMA {SOMA_REFERENCE_VERSION} reference"
        ) from error
    if tuple(reference.shape) not in ((joint_count, 3, 3), (joint_count, 4, 4)):
        raise ValueError(f"{label} returned an incompatible reference pose")
    return reference


def resolve_soma_body_contract(layer: Any) -> Any:
    """Validate body semantics and return the explicit transport reference."""

    actual_names = tuple(_normalized(name) for name in layer.public_joint_names)
    expected_names = ("ROOT", *(_normalized(name) for name in _BODY_JOINT_NAMES))
    if actual_names != expected_names:
        raise ValueError("SOMA body joint order differs from the transport contract")

    parent_ids = _numpy(layer.output_joint_parent_ids).astype(np.int64)
    if parent_ids.shape != (78,) or any(
        parent < 0 or parent >= child for child, parent in enumerate(parent_ids[1:], 1)
    ):
        raise ValueError("SOMA body hierarchy is incompatible with the transport")

    _validate_meters(layer, "SOMA body layer")
    # The public resolver also rejects an incompatible public name or hierarchy
    # mapping before returning the selected reference.
    return _resolve_reference(layer, 78, "SOMA body layer")


def resolve_soma_hand_contract(layer: Any, expected_hand_type: str) -> Any:
    """Validate hand semantics and return the explicit transport reference."""

    hand_type = getattr(layer, "hand_type", None)
    if hand_type != expected_hand_type:
        raise ValueError(
            f"SOMA hand layer side {hand_type!r} does not match {expected_hand_type!r}"
        )

    prefix = _normalized(f"{hand_type}Hand")
    actual_names = []
    for index, name in enumerate(layer.rig_data["joint_names"]):
        normalized = _normalized(name)
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix) :]
        actual_names.append("WRIST" if index == 0 and not normalized else normalized)
    if tuple(actual_names) != tuple(_normalized(name) for name in _HAND_JOINT_NAMES):
        raise ValueError("SOMA hand joint order differs from the transport contract")

    raw_parent_ids = _numpy(layer.joint_parent_ids)
    parent_ids = tuple(int(parent) for parent in raw_parent_ids)
    if (
        raw_parent_ids.shape != (25,)
        or not np.issubdtype(raw_parent_ids.dtype, np.integer)
        or parent_ids != _HAND_PARENT_IDS
    ):
        raise ValueError("SOMA hand hierarchy differs from the transport contract")

    _validate_meters(layer, "SOMA hand layer")
    return _resolve_reference(layer, 25, "SOMA hand layer")
