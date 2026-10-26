# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Rigid-pose boundaries in meters and XYZW, with no imported-value repair."""

from __future__ import annotations

import numpy as np
from trimesh import transformations


def immutable_array(value, dtype=np.float64):
    array = np.asarray(value, dtype=dtype)
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


def unit(vector, minimum=1e-6):
    try:
        vector = np.asarray(vector, dtype=np.float64)
    except (OverflowError, TypeError) as error:
        raise ValueError("Direction cannot be represented as finite numbers") from error
    if vector.shape != (3,) or not np.isfinite(vector).all():
        raise ValueError("Expected a finite three-component direction")
    length = np.linalg.norm(vector)
    if not np.isfinite(length) or length < minimum:
        raise ValueError("Direction is degenerate")
    return vector / length


def pose_matrix(pose):
    try:
        row = np.asarray(pose, dtype=np.float64)
    except (OverflowError, TypeError) as error:
        raise ValueError("Pose cannot be represented as finite numbers") from error
    if row.shape != (7,) or not np.isfinite(row).all():
        raise ValueError("Expected a finite XYZ/XYZW pose row")
    from .compiler import FLOAT32_MAX, float32, quaternion_norm_error

    target = tuple(float32(float(value)) for value in row)
    if np.any(np.abs(row) > FLOAT32_MAX) or not np.isfinite(target).all():
        raise ValueError("Pose cannot be represented as finite float32 numbers")
    # Validate target-dtype math; display matrices retain the raw float64 values.
    if quaternion_norm_error(target) > float32(1e-4):
        raise ValueError("Orientation must be a unit XYZW quaternion")
    quaternion = row[3:]
    matrix = transformations.quaternion_matrix(quaternion[[3, 0, 1, 2]])
    matrix[:3, 3] = row[:3]
    return matrix


def matrix_pose(matrix):
    try:
        matrix = np.asarray(matrix, dtype=np.float64)
    except (OverflowError, TypeError) as error:
        raise ValueError("Transform cannot be represented as finite numbers") from error
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("Expected a finite rigid transform")
    rotation = matrix[:3, :3]
    if (
        not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-8, rtol=0)
        or abs(np.linalg.det(rotation) - 1) > 1e-8
        or not np.allclose(matrix[3], (0, 0, 0, 1), atol=1e-10, rtol=0)
    ):
        raise ValueError("A pose requires a proper rigid rotation")
    quaternion = transformations.quaternion_from_matrix(matrix)[[1, 2, 3, 0]]
    return tuple(float(value) for value in (*matrix[:3, 3], *quaternion))


def transform_point(matrix, point):
    return np.asarray(matrix)[:3, :3] @ np.asarray(point) + np.asarray(matrix)[:3, 3]


def transform_direction(matrix, direction):
    return np.asarray(matrix)[:3, :3] @ np.asarray(direction)


def taxel_pose(position, normal, tangent, angle=0.0):
    normal = unit(normal)
    tangent = np.asarray(tangent, dtype=np.float64)
    tangent = unit(tangent - normal * np.dot(normal, tangent))
    tangent = np.cos(angle) * tangent + np.sin(angle) * np.cross(normal, tangent)
    matrix = np.eye(4)
    matrix[:3, :3] = np.column_stack((tangent, np.cross(normal, tangent), normal))
    matrix[:3, 3] = position
    return matrix_pose(matrix)
