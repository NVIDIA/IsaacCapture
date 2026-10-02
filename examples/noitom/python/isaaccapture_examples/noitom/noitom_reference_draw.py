# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared Noitom mocap skeleton placement (cyan debug draw + wrist retargeting)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from isaaccapture.schema import BodyJoint

from .noitom_retargeting import noitom_position_to_isaac

# G1_29DOF_CFG spawn rot (0,0,0.7071,0.7071): pelvis +X aligns with world +Y (table forward).
G1_ROBOT_FORWARD_XY = np.array([0.0, 1.0], dtype=np.float64)


@dataclass(frozen=True)
class ReferenceSkeletonLengths:
    """G1 link lengths for posture-based reference skeleton (meters)."""

    upper_arm: float = 0.28
    forearm: float = 0.26
    torso_segment: float = 0.07
    neck: float = 0.08
    head: float = 0.12
    hand_extension: float = 0.05
    left_shoulder_offset: tuple[float, float, float] = (0.05, 0.19, 0.30)
    right_shoulder_offset: tuple[float, float, float] = (0.05, -0.19, 0.30)

    @classmethod
    def from_retargeting_settings(cls, settings: Any) -> ReferenceSkeletonLengths:
        return cls(
            upper_arm=float(settings.robot_upper_arm_length),
            forearm=float(settings.robot_forearm_length),
            left_shoulder_offset=tuple(
                float(v) for v in settings.robot_left_shoulder_offset
            ),
            right_shoulder_offset=tuple(
                float(v) for v in settings.robot_right_shoulder_offset
            ),
        )


ARM_CHAIN_JOINTS = frozenset(
    {
        int(BodyJoint.LEFT_COLLAR),
        int(BodyJoint.LEFT_SHOULDER),
        int(BodyJoint.LEFT_ELBOW),
        int(BodyJoint.LEFT_WRIST),
        int(BodyJoint.LEFT_HAND),
        int(BodyJoint.RIGHT_COLLAR),
        int(BodyJoint.RIGHT_SHOULDER),
        int(BodyJoint.RIGHT_ELBOW),
        int(BodyJoint.RIGHT_WRIST),
        int(BodyJoint.RIGHT_HAND),
    }
)


def extract_raw_yup_positions(
    frame: Any,
) -> tuple[dict[int, np.ndarray], np.ndarray | None]:
    """Read valid Noitom joint positions (Y-up meters) from a FullBodyPose frame."""
    if frame.joints is None:
        return {}, None
    positions: dict[int, np.ndarray] = {}
    for index in range(int(BodyJoint.NUM_JOINTS)):
        joint = frame.joints.joints(index)
        if not joint.is_valid:
            continue
        point = joint.pose.position
        positions[int(index)] = np.array([point.x, point.y, point.z], dtype=np.float64)
    return positions, positions.get(int(BodyJoint.PELVIS))


def reference_joint_scale(
    joint_index: int,
    draw_scale: float,
    calib_view: Any | None,
    arm_chain_joints: frozenset[int] = ARM_CHAIN_JOINTS,
) -> float:
    if calib_view is None:
        return draw_scale
    if joint_index in arm_chain_joints:
        return draw_scale * float(calib_view.arm_length_scale)
    return draw_scale * float(calib_view.body_height_scale)


def build_reference_skeleton_positions(
    raw_positions: dict[int, np.ndarray],
    pelvis_raw: np.ndarray,
    pelvis_anchor: np.ndarray,
    draw_scale: float,
    calib_view: Any | None,
    arm_chain_joints: frozenset[int] = ARM_CHAIN_JOINTS,
) -> dict[int, np.ndarray]:
    """Place Noitom joints at the robot pelvis anchor (Isaac Z-up, pelvis-relative)."""
    pelvis_isaac = noitom_position_to_isaac(pelvis_raw)
    anchor = np.asarray(pelvis_anchor, dtype=np.float64)
    positions: dict[int, np.ndarray] = {}
    for index, point_raw in raw_positions.items():
        joint_scale = reference_joint_scale(
            int(index), draw_scale, calib_view, arm_chain_joints
        )
        rel = (noitom_position_to_isaac(point_raw) - pelvis_isaac) * joint_scale
        positions[int(index)] = anchor + rel
    return positions


def reference_torso_forward_xy(
    positions: dict[int, np.ndarray],
    *,
    pelvis_index: int = int(BodyJoint.PELVIS),
    spine3_index: int = int(BodyJoint.SPINE3),
    left_shoulder_index: int = int(BodyJoint.LEFT_SHOULDER),
    right_shoulder_index: int = int(BodyJoint.RIGHT_SHOULDER),
) -> np.ndarray | None:
    pelvis = positions.get(pelvis_index)
    spine3 = positions.get(spine3_index)
    left_shoulder = positions.get(left_shoulder_index)
    right_shoulder = positions.get(right_shoulder_index)
    if (
        pelvis is None
        or spine3 is None
        or left_shoulder is None
        or right_shoulder is None
    ):
        return None
    up = spine3 - pelvis
    right = right_shoulder - left_shoulder
    if float(np.linalg.norm(up)) < 1e-5 or float(np.linalg.norm(right)) < 1e-5:
        return None
    up = up / np.linalg.norm(up)
    right = right / np.linalg.norm(right)
    forward = np.cross(up, right)
    forward_xy = forward[:2]
    norm = float(np.linalg.norm(forward_xy))
    if norm < 1e-5:
        return None
    return forward_xy / norm


def signed_yaw_xy(from_xy: np.ndarray, to_xy: np.ndarray) -> float:
    source = np.asarray(from_xy, dtype=np.float64)[:2]
    target = np.asarray(to_xy, dtype=np.float64)[:2]
    source_norm = float(np.linalg.norm(source))
    target_norm = float(np.linalg.norm(target))
    if source_norm < 1e-8 or target_norm < 1e-8:
        return 0.0
    source = source / source_norm
    target = target / target_norm
    cross = source[0] * target[1] - source[1] * target[0]
    dot = float(np.clip(np.dot(source, target), -1.0, 1.0))
    return float(np.arctan2(cross, dot))


def rotate_reference_about_anchor(
    positions: dict[int, np.ndarray],
    anchor: np.ndarray,
    yaw: float,
) -> dict[int, np.ndarray]:
    if abs(yaw) < 1e-6:
        return positions
    rot = Rotation.from_euler("z", yaw)
    anchor_vec = np.asarray(anchor, dtype=np.float64)
    return {
        index: anchor_vec + rot.apply(pos - anchor_vec)
        for index, pos in positions.items()
    }


def _unit_direction(
    vector: np.ndarray,
    fallback: np.ndarray | None = None,
) -> np.ndarray:
    vec = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(vec))
    if norm < 1e-6:
        if fallback is not None:
            return _unit_direction(fallback)
        return np.array([1.0, 0.0, 0.0], dtype=np.float64)
    return vec / norm


def _pelvis_frame_offset_to_world(offset: np.ndarray) -> np.ndarray:
    """Map G1 pelvis-frame offset to world when the robot faces scene +Y."""
    ox, oy, oz = np.asarray(offset, dtype=np.float64)
    return np.array([-oy, ox, oz], dtype=np.float64)


def apply_robot_link_lengths(
    positions: dict[int, np.ndarray],
    pelvis_anchor: np.ndarray,
    lengths: ReferenceSkeletonLengths,
    *,
    length_scale: float = 1.0,
    arm_length_scale: float = 1.0,
    shoulder_span_scale: float = 1.0,
) -> dict[int, np.ndarray]:
    """Rebuild upper body in order: head match -> shoulder width -> arm segment lengths."""
    anchor = np.asarray(pelvis_anchor, dtype=np.float64)
    scale = float(length_scale)
    arm_scale = float(arm_length_scale)
    span_scale = float(shoulder_span_scale)
    out = dict(positions)
    out[int(BodyJoint.PELVIS)] = anchor.copy()

    torso_nominal_total = (
        3.0 * lengths.torso_segment + lengths.neck + lengths.head
    ) * scale
    torso_scale = 1.0
    head_src = positions.get(int(BodyJoint.HEAD))
    pelvis_src = positions.get(int(BodyJoint.PELVIS))
    if head_src is not None and pelvis_src is not None and torso_nominal_total > 1e-6:
        src_head_dist = float(np.linalg.norm(head_src - pelvis_src))
        torso_scale = float(np.clip(src_head_dist / torso_nominal_total, 0.6, 1.8))

    torso_chain = (
        (int(BodyJoint.SPINE1), lengths.torso_segment),
        (int(BodyJoint.SPINE2), lengths.torso_segment),
        (int(BodyJoint.SPINE3), lengths.torso_segment),
        (int(BodyJoint.NECK), lengths.neck),
        (int(BodyJoint.HEAD), lengths.head),
    )
    parent_robot = anchor
    parent_mocap = positions.get(int(BodyJoint.PELVIS), anchor)
    up_fallback = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    for joint_index, segment_length in torso_chain:
        child_mocap = positions.get(joint_index)
        if child_mocap is None:
            continue
        direction = _unit_direction(child_mocap - parent_mocap, fallback=up_fallback)
        robot_pos = parent_robot + direction * (segment_length * scale * torso_scale)
        out[joint_index] = robot_pos
        parent_robot = robot_pos
        parent_mocap = child_mocap

    left_shoulder_src = positions.get(int(BodyJoint.LEFT_SHOULDER))
    right_shoulder_src = positions.get(int(BodyJoint.RIGHT_SHOULDER))
    spine3_robot = out.get(int(BodyJoint.SPINE3))
    spine3_src = positions.get(int(BodyJoint.SPINE3))

    default_left = anchor + _pelvis_frame_offset_to_world(
        np.asarray(lengths.left_shoulder_offset, dtype=np.float64)
    )
    default_right = anchor + _pelvis_frame_offset_to_world(
        np.asarray(lengths.right_shoulder_offset, dtype=np.float64)
    )
    default_span = float(np.linalg.norm(default_left - default_right))
    mocap_span_scale = 1.0
    shoulder_axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    shoulder_center = 0.5 * (default_left + default_right)
    if left_shoulder_src is not None and right_shoulder_src is not None:
        src_vec = left_shoulder_src - right_shoulder_src
        src_span = float(np.linalg.norm(src_vec))
        if src_span > 1e-6:
            shoulder_axis = src_vec / src_span
        if default_span > 1e-6:
            mocap_span_scale = float(np.clip(src_span / default_span, 0.7, 1.15))
        src_center = 0.5 * (left_shoulder_src + right_shoulder_src)
        if spine3_robot is not None and spine3_src is not None:
            shoulder_center = spine3_robot + (src_center - spine3_src)
        else:
            shoulder_center = src_center.copy()
    # Follow mocap shoulder height more aggressively; keep only a loose guard band
    # so shoulders can actually track instead of being locked near nominal.
    nominal_shoulder_center_z = float(0.5 * (default_left[2] + default_right[2]))
    blended_center_z = (
        0.85 * float(shoulder_center[2]) + 0.15 * nominal_shoulder_center_z
    )
    shoulder_center_z = float(
        np.clip(
            blended_center_z,
            nominal_shoulder_center_z - 0.12,
            nominal_shoulder_center_z + 0.10,
        )
    )
    shoulder_center = shoulder_center.copy()
    shoulder_center[2] = shoulder_center_z

    shoulder_half_span = 0.5 * default_span * mocap_span_scale * scale * span_scale
    left_shoulder_robot = shoulder_center + shoulder_axis * shoulder_half_span
    right_shoulder_robot = shoulder_center - shoulder_axis * shoulder_half_span

    arm_specs = (
        (
            left_shoulder_robot,
            int(BodyJoint.LEFT_SHOULDER),
            int(BodyJoint.LEFT_ELBOW),
            int(BodyJoint.LEFT_WRIST),
            int(BodyJoint.LEFT_HAND),
        ),
        (
            right_shoulder_robot,
            int(BodyJoint.RIGHT_SHOULDER),
            int(BodyJoint.RIGHT_ELBOW),
            int(BodyJoint.RIGHT_WRIST),
            int(BodyJoint.RIGHT_HAND),
        ),
    )
    for shoulder, shoulder_index, elbow_index, wrist_index, hand_index in arm_specs:
        elbow_mocap = positions.get(elbow_index)
        wrist_mocap = positions.get(wrist_index)
        shoulder_mocap = positions.get(shoulder_index, shoulder)
        if elbow_mocap is None or wrist_mocap is None:
            out[shoulder_index] = shoulder
            continue
        upper_dir = _unit_direction(
            elbow_mocap - shoulder_mocap,
            fallback=np.array([0.0, 0.0, -1.0], dtype=np.float64),
        )
        elbow = shoulder + upper_dir * (lengths.upper_arm * scale * arm_scale)
        fore_dir = _unit_direction(wrist_mocap - elbow_mocap, fallback=upper_dir)
        wrist = elbow + fore_dir * (lengths.forearm * scale * arm_scale)
        out[shoulder_index] = shoulder
        out[elbow_index] = elbow
        out[wrist_index] = wrist
        out[hand_index] = wrist + fore_dir * (
            lengths.hand_extension * scale * arm_scale
        )

    spine3 = out.get(int(BodyJoint.SPINE3))
    if spine3 is not None:
        for collar_index, shoulder_index in (
            (int(BodyJoint.LEFT_COLLAR), int(BodyJoint.LEFT_SHOULDER)),
            (int(BodyJoint.RIGHT_COLLAR), int(BodyJoint.RIGHT_SHOULDER)),
        ):
            shoulder_pos = out.get(shoulder_index)
            if shoulder_pos is not None:
                out[collar_index] = 0.5 * (spine3 + shoulder_pos)

    return out


def align_reference_skeleton_to_robot(
    positions: dict[int, np.ndarray],
    pelvis_anchor: np.ndarray,
    *,
    pelvis_index: int = int(BodyJoint.PELVIS),
    spine3_index: int = int(BodyJoint.SPINE3),
    left_shoulder_index: int = int(BodyJoint.LEFT_SHOULDER),
    right_shoulder_index: int = int(BodyJoint.RIGHT_SHOULDER),
    robot_forward_xy: np.ndarray | None = None,
) -> dict[int, np.ndarray]:
    """Rotate the skeleton about pelvis so it faces the G1 robot (+Y in scene)."""
    forward_xy = reference_torso_forward_xy(
        positions,
        pelvis_index=pelvis_index,
        spine3_index=spine3_index,
        left_shoulder_index=left_shoulder_index,
        right_shoulder_index=right_shoulder_index,
    )
    if forward_xy is None:
        return positions
    target_xy = G1_ROBOT_FORWARD_XY if robot_forward_xy is None else robot_forward_xy
    yaw = signed_yaw_xy(forward_xy, target_xy)
    return rotate_reference_about_anchor(positions, pelvis_anchor, yaw)


def place_aligned_reference_skeleton(
    raw_positions: dict[int, np.ndarray],
    pelvis_raw: np.ndarray,
    pelvis_anchor: np.ndarray,
    draw_scale: float = 1.0,
    calib_view: Any | None = None,
    *,
    use_robot_link_lengths: bool = True,
    link_lengths: ReferenceSkeletonLengths | None = None,
    length_scale: float = 1.0,
    arm_length_scale: float = 1.0,
    shoulder_span_scale: float = 1.0,
) -> dict[int, np.ndarray]:
    """Build pelvis-anchored skeleton and align facing with the G1 locomanipulation robot."""
    direction_scale = 1.0 if use_robot_link_lengths else draw_scale
    direction_calib = None if use_robot_link_lengths else calib_view
    positions = build_reference_skeleton_positions(
        raw_positions,
        pelvis_raw,
        pelvis_anchor,
        direction_scale,
        direction_calib,
    )
    positions = align_reference_skeleton_to_robot(positions, pelvis_anchor)
    if use_robot_link_lengths:
        lengths = link_lengths or ReferenceSkeletonLengths()
        positions = apply_robot_link_lengths(
            positions,
            pelvis_anchor,
            lengths,
            length_scale=length_scale * draw_scale,
            arm_length_scale=arm_length_scale,
            shoulder_span_scale=shoulder_span_scale,
        )
    return positions


def aligned_reference_skeleton_from_frame(
    frame: Any,
    pelvis_anchor: np.ndarray,
    draw_scale: float = 1.0,
    calib_view: Any | None = None,
    *,
    use_robot_link_lengths: bool = True,
    link_lengths: ReferenceSkeletonLengths | None = None,
    length_scale: float = 1.0,
    arm_length_scale: float = 1.0,
    shoulder_span_scale: float = 1.0,
) -> dict[int, np.ndarray]:
    """Full pipeline: FullBodyPose -> robot-aligned joint positions in Isaac world."""
    raw_positions, pelvis_raw = extract_raw_yup_positions(frame)
    if pelvis_raw is None or not raw_positions:
        return {}
    return place_aligned_reference_skeleton(
        raw_positions,
        pelvis_raw,
        pelvis_anchor,
        draw_scale,
        calib_view,
        use_robot_link_lengths=use_robot_link_lengths,
        link_lengths=link_lengths,
        length_scale=length_scale,
        arm_length_scale=arm_length_scale,
        shoulder_span_scale=shoulder_span_scale,
    )


__all__ = [
    "ARM_CHAIN_JOINTS",
    "G1_ROBOT_FORWARD_XY",
    "ReferenceSkeletonLengths",
    "align_reference_skeleton_to_robot",
    "aligned_reference_skeleton_from_frame",
    "apply_robot_link_lengths",
    "build_reference_skeleton_positions",
    "extract_raw_yup_positions",
    "place_aligned_reference_skeleton",
    "reference_torso_forward_xy",
    "rotate_reference_about_anchor",
    "signed_yaw_xy",
]
