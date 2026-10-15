# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pure data, geometry, and state-machine logic for the tactile visualizer."""

from __future__ import annotations

import math
import os
import tempfile
import threading
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Generic, TypeVar

import numpy as np

OPENXR_JOINT_NAMES = (
    "PALM",
    "WRIST",
    "THUMB_METACARPAL",
    "THUMB_PROXIMAL",
    "THUMB_DISTAL",
    "THUMB_TIP",
    "INDEX_METACARPAL",
    "INDEX_PROXIMAL",
    "INDEX_INTERMEDIATE",
    "INDEX_DISTAL",
    "INDEX_TIP",
    "MIDDLE_METACARPAL",
    "MIDDLE_PROXIMAL",
    "MIDDLE_INTERMEDIATE",
    "MIDDLE_DISTAL",
    "MIDDLE_TIP",
    "RING_METACARPAL",
    "RING_PROXIMAL",
    "RING_INTERMEDIATE",
    "RING_DISTAL",
    "RING_TIP",
    "LITTLE_METACARPAL",
    "LITTLE_PROXIMAL",
    "LITTLE_INTERMEDIATE",
    "LITTLE_DISTAL",
    "LITTLE_TIP",
)
WRIST = 1
ROOT_PARENT = np.uint32(0x7FFFFFFF)
MIRROR_X = np.diag([-1.0, 1.0, 1.0])


def normalize(vector: np.ndarray, *, label: str = "vector") -> np.ndarray:
    value = np.asarray(vector, dtype=np.float64)
    length = float(np.linalg.norm(value))
    if not math.isfinite(length) or length < 1e-8:
        raise ValueError(f"{label} must have a finite, nonzero length")
    return value / length


def orthonormalize_frame(
    normal: np.ndarray, tangent: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return a unit normal and a perpendicular unit tangent."""
    n = normalize(normal, label="normal")
    t = np.asarray(tangent, dtype=np.float64) - n * float(np.dot(n, tangent))
    if np.linalg.norm(t) < 1e-8:
        basis = np.zeros(3)
        basis[int(np.argmin(np.abs(n)))] = 1.0
        t = basis - n * float(np.dot(n, basis))
    return n.astype(np.float32), normalize(t, label="tangent").astype(np.float32)


def quaternion_xyzw_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    x, y, z, w = normalize(quaternion, label="quaternion")
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def matrix_to_quaternion_xyzw(matrix: np.ndarray) -> np.ndarray:
    """Convert a proper rotation matrix to a normalized XYZW quaternion."""
    m = np.asarray(matrix, dtype=np.float64)
    trace = float(np.trace(m))
    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2
        q = np.array(
            [
                (m[2, 1] - m[1, 2]) / s,
                (m[0, 2] - m[2, 0]) / s,
                (m[1, 0] - m[0, 1]) / s,
                0.25 * s,
            ]
        )
    else:
        axis = int(np.argmax(np.diag(m)))
        if axis == 0:
            s = math.sqrt(max(0.0, 1 + m[0, 0] - m[1, 1] - m[2, 2])) * 2
            q = np.array(
                [
                    0.25 * s,
                    (m[0, 1] + m[1, 0]) / s,
                    (m[0, 2] + m[2, 0]) / s,
                    (m[2, 1] - m[1, 2]) / s,
                ]
            )
        elif axis == 1:
            s = math.sqrt(max(0.0, 1 + m[1, 1] - m[0, 0] - m[2, 2])) * 2
            q = np.array(
                [
                    (m[0, 1] + m[1, 0]) / s,
                    0.25 * s,
                    (m[1, 2] + m[2, 1]) / s,
                    (m[0, 2] - m[2, 0]) / s,
                ]
            )
        else:
            s = math.sqrt(max(0.0, 1 + m[2, 2] - m[0, 0] - m[1, 1])) * 2
            q = np.array(
                [
                    (m[0, 2] + m[2, 0]) / s,
                    (m[1, 2] + m[2, 1]) / s,
                    0.25 * s,
                    (m[1, 0] - m[0, 1]) / s,
                ]
            )
    return normalize(q, label="rotation quaternion").astype(np.float32)


def frame_matrix(normal: np.ndarray, tangent: np.ndarray) -> np.ndarray:
    n, t = orthonormalize_frame(normal, tangent)
    return np.column_stack((t, np.cross(n, t), n)).astype(np.float64)


def frame_wxyz(normal: np.ndarray, tangent: np.ndarray) -> np.ndarray:
    q = matrix_to_quaternion_xyzw(frame_matrix(normal, tangent))
    return q[[3, 0, 1, 2]]


def mirror_pose(
    positions: np.ndarray, orientations: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    pos = np.asarray(positions, dtype=np.float64) @ MIRROR_X.T
    rotations = np.stack(
        [
            MIRROR_X @ quaternion_xyzw_to_matrix(q) @ MIRROR_X
            for q in np.asarray(orientations)
        ]
    )
    quat = np.stack([matrix_to_quaternion_xyzw(r) for r in rotations])
    return pos.astype(np.float32), quat.astype(np.float32)


@dataclass
class ShapeDocument:
    """Owned, mutable arrays corresponding exactly to ``TactileShape``."""

    joint_bind_positions: np.ndarray
    joint_bind_orientations: np.ndarray
    joint_parents: np.ndarray
    taxel_positions: np.ndarray
    taxel_normals: np.ndarray
    taxel_tangents: np.ndarray
    taxel_joint_indices: np.ndarray
    taxel_joint_weights: np.ndarray

    @property
    def num_joints(self) -> int:
        return len(self.joint_parents)

    @property
    def num_taxels(self) -> int:
        return len(self.taxel_positions)

    def copy(self) -> ShapeDocument:
        return ShapeDocument(
            **{name: np.asarray(value).copy() for name, value in self.arrays().items()}
        )

    def arrays(self) -> dict[str, np.ndarray]:
        return {
            "joint_bind_positions": self.joint_bind_positions,
            "joint_bind_orientations": self.joint_bind_orientations,
            "joint_parents": self.joint_parents,
            "taxel_positions": self.taxel_positions,
            "taxel_normals": self.taxel_normals,
            "taxel_tangents": self.taxel_tangents,
            "taxel_joint_indices": self.taxel_joint_indices,
            "taxel_joint_weights": self.taxel_joint_weights,
        }

    @classmethod
    def from_schema(cls, shape) -> ShapeDocument:
        document = cls(
            **{name: np.asarray(getattr(shape, name)).copy() for name in cls.fields()}
        )
        document.validate()
        return document

    @classmethod
    def from_bytes(cls, payload: bytes) -> ShapeDocument:
        from isaaccapture.schema import TactileShape

        return cls.from_schema(TactileShape.from_bytes(payload))

    @staticmethod
    def fields() -> tuple[str, ...]:
        return (
            "joint_bind_positions",
            "joint_bind_orientations",
            "joint_parents",
            "taxel_positions",
            "taxel_normals",
            "taxel_tangents",
            "taxel_joint_indices",
            "taxel_joint_weights",
        )

    def to_schema(self):
        from isaaccapture.schema import TactileShape

        self.validate()
        return TactileShape(**self.arrays())

    def to_bytes(self) -> bytes:
        return self.to_schema().to_bytes()

    def validate(self) -> None:
        arrays = self.arrays()
        expected = {
            "joint_bind_positions": (self.num_joints, 3),
            "joint_bind_orientations": (self.num_joints, 4),
            "joint_parents": (self.num_joints,),
            "taxel_positions": (self.num_taxels, 3),
            "taxel_normals": (self.num_taxels, 3),
            "taxel_tangents": (self.num_taxels, 3),
            "taxel_joint_indices": (self.num_taxels, 4),
            "taxel_joint_weights": (self.num_taxels, 4),
        }
        if not 1 <= self.num_joints <= len(OPENXR_JOINT_NAMES):
            raise ValueError("joint count must be in OpenXR range [1, 26]")
        for name, shape in expected.items():
            value = np.asarray(arrays[name])
            if value.shape != shape:
                raise ValueError(f"{name} must have shape {shape}, got {value.shape}")
            if value.dtype.kind == "f" and not np.isfinite(value).all():
                raise ValueError(f"{name} contains non-finite values")

        parents = np.asarray(self.joint_parents, dtype=np.uint32)
        for index, parent in enumerate(parents):
            if parent != ROOT_PARENT and int(parent) >= self.num_joints:
                raise ValueError(f"joint {index} has invalid parent {parent}")
            seen: set[int] = set()
            cursor = index
            while parents[cursor] != ROOT_PARENT:
                if cursor in seen:
                    raise ValueError("joint hierarchy contains a cycle")
                seen.add(cursor)
                cursor = int(parents[cursor])
        if not np.any(parents == ROOT_PARENT):
            raise ValueError("joint hierarchy has no root")

        q_norm = np.linalg.norm(self.joint_bind_orientations, axis=1)
        if not np.allclose(q_norm, 1.0, atol=1e-4):
            raise ValueError("joint bind orientations must be unit quaternions")
        if self.num_taxels == 0:
            return
        indices = np.asarray(self.taxel_joint_indices)
        if np.any(indices >= self.num_joints):
            raise ValueError("taxel joint index is outside the joint array")
        weights = np.asarray(self.taxel_joint_weights)
        if np.any(weights < 0) or not np.allclose(weights.sum(axis=1), 1.0, atol=1e-4):
            raise ValueError("taxel weights must be nonnegative and sum to one")
        if np.any(np.count_nonzero(weights > 0, axis=1) == 0):
            raise ValueError("every taxel must have at least one nonzero weight")
        n_norm = np.linalg.norm(self.taxel_normals, axis=1)
        t_norm = np.linalg.norm(self.taxel_tangents, axis=1)
        dot = np.sum(self.taxel_normals * self.taxel_tangents, axis=1)
        if not np.allclose(n_norm, 1.0, atol=1e-4):
            raise ValueError("taxel normals must be unit length")
        if not np.allclose(t_norm, 1.0, atol=1e-4):
            raise ValueError("taxel tangents must be unit length")
        if not np.allclose(dot, 0.0, atol=1e-4):
            raise ValueError("taxel normals and tangents must be perpendicular")


def default_openxr_shape(num_taxels: int = 0) -> ShapeDocument:
    """Build a tool-defined neutral right-hand pose in OpenXR ordinal order.

    OpenXR standardizes joint names and runtime poses, but not a universal bind
    pose. These approximate metric locations are therefore an explicit editor
    default, not a hardware calibration.
    """
    if num_taxels < 0:
        raise ValueError("num_taxels must be nonnegative")
    positions = np.zeros((26, 3), dtype=np.float32)
    positions[0] = (0.0, 0.042, 0.0)
    positions[1] = (0.0, 0.0, 0.0)
    positions[2:6] = (
        (-0.026, 0.026, 0.0),
        (-0.041, 0.041, 0.0),
        (-0.052, 0.054, 0.0),
        (-0.060, 0.066, 0.0),
    )
    finger_specs = (
        (6, -0.027, (0.042, 0.070, 0.096, 0.116, 0.133)),
        (11, -0.009, (0.044, 0.076, 0.105, 0.128, 0.147)),
        (16, 0.010, (0.043, 0.073, 0.100, 0.121, 0.139)),
        (21, 0.029, (0.039, 0.064, 0.087, 0.105, 0.120)),
    )
    for start, x, ys in finger_specs:
        positions[start : start + 5] = [(x, y, 0.0) for y in ys]

    parents = np.full(26, ROOT_PARENT, dtype=np.uint32)
    parents[0] = WRIST
    for start, length in ((2, 4), (6, 5), (11, 5), (16, 5), (21, 5)):
        parents[start] = WRIST
        parents[start + 1 : start + length] = np.arange(
            start, start + length - 1, dtype=np.uint32
        )

    attachments = np.array(
        [0, 4, 5, 8, 9, 10, 13, 14, 15, 18, 19, 20, 23, 24, 25], dtype=np.uint32
    )
    taxel_positions = np.empty((num_taxels, 3), dtype=np.float32)
    indices = np.empty((num_taxels, 4), dtype=np.uint32)
    weights = np.zeros((num_taxels, 4), dtype=np.float32)
    for taxel in range(num_taxels):
        joint = int(attachments[taxel % len(attachments)])
        ring = taxel // len(attachments)
        dx = ((ring % 5) - 2) * 0.0025
        dy = ((ring // 5) % 5 - 2) * 0.0025
        taxel_positions[taxel] = positions[joint] + (dx, dy, 0.002)
        indices[taxel] = joint
        weights[taxel, 0] = 1.0

    document = ShapeDocument(
        joint_bind_positions=positions,
        joint_bind_orientations=np.tile([0.0, 0.0, 0.0, 1.0], (26, 1)).astype(
            np.float32
        ),
        joint_parents=parents,
        taxel_positions=taxel_positions,
        taxel_normals=np.tile([0.0, 0.0, 1.0], (num_taxels, 1)).astype(np.float32),
        taxel_tangents=np.tile([1.0, 0.0, 0.0], (num_taxels, 1)).astype(np.float32),
        taxel_joint_indices=indices,
        taxel_joint_weights=weights,
    )
    document.validate()
    return document


def append_grid_taxels(document: ShapeDocument, amount: int) -> None:
    """Append the first N points from the documented x/y/z centimeter grid."""
    if not 0 <= amount <= 125:
        raise ValueError("taxel amount must be between 0 and 125")
    if amount == 0:
        return
    grid_cm = (-4.0, -2.0, 0.0, 2.0, 4.0)
    candidates = np.asarray(
        [(x, y, z) for x in grid_cm for y in grid_cm for z in grid_cm],
        dtype=np.float32,
    )
    positions = candidates[:amount] / 100.0
    normals = np.tile([0.0, 0.0, 1.0], (amount, 1)).astype(np.float32)
    tangents = np.tile([1.0, 0.0, 0.0], (amount, 1)).astype(np.float32)
    # TactileTaxel has no unbound sentinel: all four indices must be valid and
    # 1-4 weights must be nonzero. Bind to wrist with one unit weight.
    attachment = WRIST if document.num_joints > WRIST else 0
    indices = np.full((amount, 4), attachment, dtype=np.uint32)
    weights = np.zeros((amount, 4), dtype=np.float32)
    weights[:, 0] = 1.0
    document.taxel_positions = np.concatenate(
        (document.taxel_positions, positions), axis=0
    )
    document.taxel_normals = np.concatenate((document.taxel_normals, normals), axis=0)
    document.taxel_tangents = np.concatenate(
        (document.taxel_tangents, tangents), axis=0
    )
    document.taxel_joint_indices = np.concatenate(
        (document.taxel_joint_indices, indices), axis=0
    )
    document.taxel_joint_weights = np.concatenate(
        (document.taxel_joint_weights, weights), axis=0
    )
    document.validate()


def normalize_hand_to_wrist(
    positions: np.ndarray,
    orientations: np.ndarray,
    valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    pos = np.asarray(positions, dtype=np.float64).copy()
    quat = np.asarray(orientations, dtype=np.float64).copy()
    mask = np.asarray(valid, dtype=bool)
    if len(pos) <= WRIST or not mask[WRIST]:
        return pos.astype(np.float32), quat.astype(np.float32)
    wrist_rotation = quaternion_xyzw_to_matrix(quat[WRIST])
    inverse = wrist_rotation.T
    pos = (pos - pos[WRIST]) @ inverse.T
    for i, q in enumerate(quat):
        if mask[i]:
            quat[i] = matrix_to_quaternion_xyzw(inverse @ quaternion_xyzw_to_matrix(q))
    return pos.astype(np.float32), quat.astype(np.float32)


def resolve_current_joints(
    document: ShapeDocument,
    positions: np.ndarray | None,
    orientations: np.ndarray | None,
    valid: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    bind_pos = document.joint_bind_positions.astype(np.float64)
    bind_quat = document.joint_bind_orientations.astype(np.float64)
    if positions is None or orientations is None or valid is None:
        return bind_pos, bind_quat
    if len(valid) <= WRIST or not bool(valid[WRIST]):
        return bind_pos, bind_quat
    count = min(document.num_joints, len(positions), len(orientations), len(valid))
    current_pos = bind_pos.copy()
    current_quat = bind_quat.copy()
    normalized_pos, normalized_quat = normalize_hand_to_wrist(
        positions[:count], orientations[:count], valid[:count]
    )
    mask = np.asarray(valid[:count], dtype=bool)
    valid_indices = np.flatnonzero(mask)
    current_pos[valid_indices] = normalized_pos[valid_indices]
    current_quat[valid_indices] = normalized_quat[valid_indices]
    return current_pos, current_quat


def taxel_skinning_affine(
    document: ShapeDocument,
    current_positions: np.ndarray,
    current_orientations: np.ndarray,
    taxel: int,
) -> tuple[np.ndarray, np.ndarray]:
    linear = np.zeros((3, 3), dtype=np.float64)
    offset = np.zeros(3, dtype=np.float64)
    for joint, weight in zip(
        document.taxel_joint_indices[taxel],
        document.taxel_joint_weights[taxel],
    ):
        if weight <= 0:
            continue
        j = int(joint)
        bind_rotation = quaternion_xyzw_to_matrix(document.joint_bind_orientations[j])
        current_rotation = quaternion_xyzw_to_matrix(current_orientations[j])
        transform = current_rotation @ bind_rotation.T
        linear += float(weight) * transform
        offset += float(weight) * (
            current_positions[j] - transform @ document.joint_bind_positions[j]
        )
    return linear, offset


def skin_taxels(
    document: ShapeDocument,
    current_positions: np.ndarray,
    current_orientations: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    positions = np.empty_like(document.taxel_positions)
    normals = np.empty_like(document.taxel_normals)
    tangents = np.empty_like(document.taxel_tangents)
    for taxel in range(document.num_taxels):
        linear, offset = taxel_skinning_affine(
            document, current_positions, current_orientations, taxel
        )
        positions[taxel] = linear @ document.taxel_positions[taxel] + offset
        normal, tangent = orthonormalize_frame(
            linear @ document.taxel_normals[taxel],
            linear @ document.taxel_tangents[taxel],
        )
        normals[taxel], tangents[taxel] = normal, tangent
    return positions, normals, tangents


def apply_display_side(
    positions: np.ndarray,
    normals: np.ndarray | None,
    tangents: np.ndarray | None,
    side: str,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    if side == "right":
        return positions, normals, tangents
    pos = np.asarray(positions) @ MIRROR_X.T
    normal = None if normals is None else np.asarray(normals) @ MIRROR_X.T
    tangent = None if tangents is None else np.asarray(tangents) @ MIRROR_X.T
    return pos, normal, tangent


def set_taxel_display_position(
    document: ShapeDocument,
    current_positions: np.ndarray,
    current_orientations: np.ndarray,
    taxel: int,
    display_position: np.ndarray,
    side: str,
) -> None:
    desired = np.asarray(display_position, dtype=np.float64)
    if side == "left":
        desired = MIRROR_X @ desired
    linear, offset = taxel_skinning_affine(
        document, current_positions, current_orientations, taxel
    )
    document.taxel_positions[taxel] = np.linalg.pinv(linear) @ (desired - offset)


def set_taxel_display_frame(
    document: ShapeDocument,
    current_positions: np.ndarray,
    current_orientations: np.ndarray,
    taxel: int,
    display_normal: np.ndarray,
    display_tangent: np.ndarray,
    side: str,
) -> None:
    normal = np.asarray(display_normal, dtype=np.float64)
    tangent = np.asarray(display_tangent, dtype=np.float64)
    if side == "left":
        normal, tangent = MIRROR_X @ normal, MIRROR_X @ tangent
    linear, _ = taxel_skinning_affine(
        document, current_positions, current_orientations, taxel
    )
    inverse = np.linalg.pinv(linear)
    normal, tangent = orthonormalize_frame(inverse @ normal, inverse @ tangent)
    document.taxel_normals[taxel] = normal
    document.taxel_tangents[taxel] = tangent


def force_colors(forces: np.ndarray | None, count: int) -> np.ndarray:
    """Map 0..30 N linearly from blue to red, clamping larger forces."""
    if forces is None or len(forces) != count:
        return np.tile(np.array([0, 0, 255], dtype=np.uint8), (count, 1))
    value = np.clip(np.linalg.norm(forces, axis=1) / 30.0, 0.0, 1.0)
    colors = np.zeros((count, 3), dtype=np.float64)
    colors[:, 0] = 255.0 * value
    colors[:, 2] = 255.0 * (1.0 - value)
    return np.rint(colors).astype(np.uint8)


def project_points(
    points: np.ndarray,
    camera_position: np.ndarray,
    camera_wxyz: np.ndarray,
    vertical_fov: float,
    aspect: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Project world points to Viser normalized screen coordinates.

    Viser follows the OpenCV camera convention: +Z forward, +X right, +Y down.
    """
    w, x, y, z = normalize(camera_wxyz, label="camera quaternion")
    rotation = quaternion_xyzw_to_matrix(np.array([x, y, z, w]))
    camera = (np.asarray(points) - camera_position) @ rotation
    depth = camera[:, 2]
    tan_half = math.tan(float(vertical_fov) / 2)
    screen = np.empty((len(camera), 2), dtype=np.float64)
    screen[:, 0] = 0.5 + camera[:, 0] / (2 * depth * tan_half * aspect)
    screen[:, 1] = 0.5 + camera[:, 1] / (2 * depth * tan_half)
    return screen, depth > 0


class SessionState(str, Enum):
    OFFLINE = "OFFLINE"
    LIVE_PLAYING = "LIVE_PLAYING"
    LIVE_PAUSED = "LIVE_PAUSED"
    DISCONNECTED = "DISCONNECTED"


class SessionController:
    def __init__(self, live_source: bool, timeout_s: float) -> None:
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("liveness timeout must be finite and positive")
        self.live_source = live_source
        self.timeout_s = timeout_s
        self.paused = False
        self.ready = False
        self.last_message_time: float | None = None
        self.last_sample_time: float | None = None
        self.state = SessionState.OFFLINE

    def sample_received(self, now: float) -> SessionState:
        self.last_message_time = now
        self.last_sample_time = now
        self.ready = True
        self.state = (
            SessionState.LIVE_PAUSED if self.paused else SessionState.LIVE_PLAYING
        )
        return self.state

    def message_received(self, now: float, *, ready: bool) -> SessionState:
        self.last_message_time = now
        if ready:
            return self.sample_received(now)
        if not self.ready:
            self.state = SessionState.OFFLINE
        return self.state

    def requirements_changed(self, now: float, *, ready: bool) -> SessionState:
        self.ready = ready
        return self.tick(now)

    def pause(self) -> SessionState:
        self.paused = True
        if self.state is SessionState.LIVE_PLAYING:
            self.state = SessionState.LIVE_PAUSED
        return self.state

    def resume(self, now: float) -> SessionState:
        self.paused = False
        return self.tick(now)

    def tick(self, now: float) -> SessionState:
        if not self.live_source:
            self.state = SessionState.OFFLINE
        elif self.last_message_time is None or not self.ready:
            self.state = SessionState.OFFLINE
        elif now - self.last_message_time > self.timeout_s:
            self.state = SessionState.DISCONNECTED
        else:
            self.state = (
                SessionState.LIVE_PAUSED if self.paused else SessionState.LIVE_PLAYING
            )
        return self.state

    @property
    def editing_enabled(self) -> bool:
        return self.state is not SessionState.LIVE_PLAYING


T = TypeVar("T")


class LatestMailbox(Generic[T]):
    """Thread-safe, one-slot mailbox; publishing replaces any unread sample."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sequence = 0
        self._value: T | None = None
        self._error: BaseException | None = None

    def put(self, value: T) -> None:
        with self._lock:
            self._value = value
            self._error = None
            self._sequence += 1

    def fail(self, error: BaseException) -> None:
        with self._lock:
            self._value = None
            self._error = error
            self._sequence += 1

    def latest_after(self, sequence: int) -> tuple[int, T | None, BaseException | None]:
        with self._lock:
            if sequence == self._sequence:
                return sequence, None, None
            return self._sequence, self._value, self._error


def atomic_write_within(root: Path, destination: str, payload: bytes) -> Path:
    """Atomically write below an explicitly allowed local root."""
    root = root.expanduser().resolve()
    target = (root / destination).resolve()
    if not target.is_relative_to(root):
        raise ValueError("server destination must remain inside the allowed write root")
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{target.name}.", dir=target.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
    return target
