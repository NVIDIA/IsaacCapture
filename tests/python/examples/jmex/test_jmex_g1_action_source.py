# SPDX-FileCopyrightText: Copyright (c) 2026 j-mex. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The IO contract of the j-mex G1 action source, on synthetic frames."""

from __future__ import annotations

import logging

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from isaaccapture.retargeting_engine.interface import ComputeContext, TensorGroup
from isaaccapture.schema import BodyJoint, BodyJoints, FullBodyPose

from isaaccapture_examples.jmex.jmex_g1_retargeting import (
    ACTION_OUTPUT,
    FULL_BODY_INPUT,
    POSE_DIM,
    STANDING_LOCOMOTION_COMMAND,
    FrameTargetCfg,
    G1ActionLayout,
    JmexG1ActionSource,
    PelvisPose,
    arm_frame_targets,
    forearm_frame,
    left_wrist_chain,
    right_wrist_chain,
)

# Human proportions, meters. Deliberately unlike the robot lengths below: every assertion
# about the reconstructed target has to come out the same whatever these are.
_HUMAN_PELVIS_TO_SHOULDER = 0.55
_HUMAN_SHOULDER_TO_ELBOW = 0.31
_HUMAN_ELBOW_TO_WRIST = 0.27
# Each hip this far to its side of the pelvis and a little below it; each shoulder this far to
# its side of the torso's line, per meter up it. The hips are what the heading is read off.
_HUMAN_HIP_HALF_WIDTH = 0.09
_HUMAN_HIP_DROP = 0.06
_HUMAN_SHOULDER_SPREAD = 0.35
# Each slot's distance up the torso's line, as a fraction of the pelvis-to-shoulder distance.
_HUMAN_SPINE = (
    (BodyJoint.SPINE1, 0.2),
    (BodyJoint.SPINE2, 0.45),
    (BodyJoint.SPINE3, 0.7),
    (BodyJoint.NECK, 0.9),
    (BodyJoint.HEAD, 1.2),
)

# Synthetic robot upper arm and forearm, and where the robot's shoulders sit in its own
# pelvis frame. Not the G1's -- the point is that the test needs no asset.
_ROBOT_SEGMENTS = (0.22, 0.19)
_ROBOT_LEFT_SHOULDER = (0.02, 0.12, 0.27)
_ROBOT_RIGHT_SHOULDER = (0.02, -0.12, 0.27)
# What the robot's upper body turns about, in the same frame: off the pelvis origin, and off the
# line under the shoulders, so a target turned about the wrong point cannot pass.
_ROBOT_WAIST_PIVOT = (-0.01, 0.0, 0.05)
# The robot's forearm frame in its wrist task's axes. Deliberately not the identity, so a
# target that forgets to carry it cannot pass.
_ROBOT_FOREARM_IN_TASK = tuple(
    Rotation.from_euler("xyz", [15.0, -25.0, 40.0], degrees=True).as_quat()
)

# A G1-like rest arm, in its pelvis frame: upper arm down and splayed out, forearm ahead.
_REST_SHOULDER = np.array([0.0, 0.10, 0.29])
_REST_ELBOW = np.array([0.016, 0.147, 0.105])
_REST_WRIST = np.array([0.20, 0.149, 0.095])

# Every slot above the pelvis that is not a leg: what turns when the operator twists.
_UPPER_BODY = (
    BodyJoint.SPINE1,
    BodyJoint.SPINE2,
    BodyJoint.SPINE3,
    BodyJoint.NECK,
    BodyJoint.HEAD,
    BodyJoint.LEFT_COLLAR,
    BodyJoint.RIGHT_COLLAR,
    BodyJoint.LEFT_SHOULDER,
    BodyJoint.RIGHT_SHOULDER,
    BodyJoint.LEFT_ELBOW,
    BodyJoint.RIGHT_ELBOW,
    BodyJoint.LEFT_WRIST,
    BodyJoint.RIGHT_WRIST,
    BodyJoint.LEFT_HAND,
    BodyJoint.RIGHT_HAND,
)

# The operator's arm slots, shoulder first, as the plugin fills them.
_OPERATOR_ARMS = (
    (BodyJoint.LEFT_SHOULDER, BodyJoint.LEFT_ELBOW, BodyJoint.LEFT_WRIST),
    (BodyJoint.RIGHT_SHOULDER, BodyJoint.RIGHT_ELBOW, BodyJoint.RIGHT_WRIST),
)

# How the standard human skeleton's WRIST slot is turned, in the axes it is published in: X
# runs from the elbow out through the wrist, and Y lies along the elbow's hinge -- the
# direction of (upper arm x forearm). Both are negative on the left arm and positive on the
# right, so one sign per arm carries the slot's whole layout.
_WRIST_SLOT_SIGN = {BodyJoint.LEFT_WRIST: -1.0, BodyJoint.RIGHT_WRIST: 1.0}

# OpenXR body axes, as the plugin publishes them: +X the operator's right, +Y up, +Z behind.
_OPENXR_FORWARD = np.array([0.0, 0.0, -1.0])
_OPENXR_UP = np.array([0.0, 1.0, 0.0])
_OPENXR_RIGHT = np.array([1.0, 0.0, 0.0])

# The same three, after the change of basis into Isaac's +X forward, +Y left, +Z up.
_ISAAC_FORWARD = np.array([1.0, 0.0, 0.0])
_ISAAC_UP = np.array([0.0, 0.0, 1.0])
_ISAAC_RIGHT = np.array([0.0, -1.0, 0.0])
_ISAAC_LEFT = -_ISAAC_RIGHT

# The torso's moves, as axes of turns in Isaac's: leaning forward tips up toward forward, bending
# to the right tips it toward the right, twisting to the left turns forward toward the left.
_LEANING_FORWARD = _ISAAC_LEFT
_BENDING_RIGHT = _ISAAC_FORWARD
_TWISTING_LEFT = _ISAAC_UP


def _identity_pelvis() -> PelvisPose:
    return PelvisPose(
        position=np.zeros(3), orientation_xyzw=np.array([0.0, 0.0, 0.0, 1.0])
    )


def _yawed_pelvis(yaw_deg: float, height: float = 0.75) -> PelvisPose:
    """The G1's spawn pose shape: a yaw about world Z at a standing height."""
    return PelvisPose(
        position=np.array([0.0, 0.0, height]),
        orientation_xyzw=Rotation.from_euler("z", yaw_deg, degrees=True).as_quat(),
    )


def _robot_shoulders_in_world(
    pelvis: PelvisPose, torso: Rotation | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Where the configured shoulders are once the pelvis is placed, in task order.

    With ``torso``, a turn in the pelvis frame, each is first turned that far about the waist.
    """
    rotation = Rotation.from_quat(pelvis.orientation_xyzw)
    turn = Rotation.identity() if torso is None else torso
    pivot = np.asarray(_ROBOT_WAIST_PIVOT)
    return tuple(
        pelvis.position
        + rotation.apply(pivot + turn.apply(np.asarray(shoulder) - pivot))
        for shoulder in (_ROBOT_LEFT_SHOULDER, _ROBOT_RIGHT_SHOULDER)
    )


def _torso_turn(axis_isaac: np.ndarray, degrees: float) -> Rotation:
    """A turn of ``degrees`` about ``axis_isaac``, in Isaac's axes."""
    return Rotation.from_rotvec(np.radians(degrees) * _unit(axis_isaac))


def _frame_target(
    name: str,
    chain: tuple[int, ...],
    *,
    anchor: tuple[float, float, float],
    forearm_frame_in_task_xyzw: tuple[float, float, float, float] | None,
    segments: tuple[float, ...] | None = None,
    neutral_orientation_xyzw: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
) -> FrameTargetCfg:
    """One task over ``chain``, with the robot's lengths along it unless told otherwise."""
    return FrameTargetCfg(
        name=name,
        chain=chain,
        anchor_position_pelvis=anchor,
        robot_segment_lengths=(
            _ROBOT_SEGMENTS[: len(chain) - 1] if segments is None else segments
        ),
        neutral_position_pelvis=(0.2, 0.2, 0.1),
        neutral_orientation_pelvis_xyzw=neutral_orientation_xyzw,
        forearm_frame_in_task_xyzw=forearm_frame_in_task_xyzw,
    )


def _g1_layout(
    neutral_position: tuple[float, float, float] = (0.2, 0.2, 0.1),
    neutral_orientation_xyzw: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
) -> G1ActionLayout:
    """The G1 task's frame tasks in its controller's order, every frame resting alike."""
    rest = (neutral_position, neutral_orientation_xyzw)
    arms = [
        arm_frame_targets(
            chain,
            anchor_position_pelvis=anchor,
            robot_segment_lengths=_ROBOT_SEGMENTS,
            rest_poses_pelvis=[rest] * len(chain),
            forearm_frame_in_task_xyzw=_ROBOT_FOREARM_IN_TASK,
        )
        for chain, anchor in (
            (left_wrist_chain(), _ROBOT_LEFT_SHOULDER),
            (right_wrist_chain(), _ROBOT_RIGHT_SHOULDER),
        )
    ]
    # Each arm comes shoulder first; the controller lists both wrists, both elbows, then
    # both shoulders, so tasks 0 and 1 are the wrists.
    return G1ActionLayout(
        frame_targets=tuple(arm[depth] for depth in (2, 1, 0) for arm in arms),
        waist_pivot_pelvis=_ROBOT_WAIST_PIVOT,
    )


def _task_index(layout: G1ActionLayout, name: str) -> int:
    return [target.name for target in layout.frame_targets].index(name)


def _unit(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    return vector / np.linalg.norm(vector)


def _in_isaac_axes(vector_openxr: np.ndarray) -> np.ndarray:
    """An OpenXR direction in Isaac's axes, built from the three pairs declared above."""
    x, y, z = np.asarray(vector_openxr, dtype=np.float64)
    return x * _ISAAC_RIGHT + y * _ISAAC_UP - z * _ISAAC_FORWARD


def _in_openxr_axes(vector_isaac: np.ndarray) -> np.ndarray:
    """An Isaac direction in OpenXR's axes: ``_in_isaac_axes`` the other way round."""
    x, y, z = np.asarray(vector_isaac, dtype=np.float64)
    return x * _OPENXR_FORWARD - y * _OPENXR_RIGHT + z * _OPENXR_UP


def _in_openxr(turn_isaac: Rotation) -> Rotation:
    """A turn given in Isaac's axes, as the same turn in OpenXR's."""
    return Rotation.from_rotvec(_in_openxr_axes(turn_isaac.as_rotvec()))


def _swung_bent_arm(swing_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """An upper arm and forearm (OpenXR) with the elbow bent, swung ``swing_deg``.

    The swing is about the operator's shoulder-to-wrist line: the hand stays where it was.
    """
    upper_arm = _unit(-_OPENXR_UP + 0.4 * _OPENXR_RIGHT)
    forearm = _unit(_OPENXR_FORWARD + 0.3 * _OPENXR_UP)
    to_wrist = _unit(
        _HUMAN_SHOULDER_TO_ELBOW * upper_arm + _HUMAN_ELBOW_TO_WRIST * forearm
    )
    swing = Rotation.from_rotvec(np.radians(swing_deg) * to_wrist)
    return swing.apply(upper_arm), swing.apply(forearm)


def _wrist_orientation(
    wrist_slot: BodyJoint, forearm_openxr: np.ndarray, hinge_openxr: np.ndarray
) -> np.ndarray:
    """The WRIST slot's published orientation (xyzw) for a forearm and hinge in OpenXR axes."""
    distal = _unit(forearm_openxr)
    hinge = _unit(hinge_openxr - (np.dot(hinge_openxr, distal)) * distal)
    sign = _WRIST_SLOT_SIGN[wrist_slot]
    along_x, along_y = sign * distal, sign * hinge
    matrix = np.column_stack([along_x, along_y, np.cross(along_x, along_y)])
    return Rotation.from_matrix(matrix).as_quat()


def _straight_arm_frame(direction_openxr: np.ndarray, **kwargs) -> FullBodyPose:
    """One frame whose two arms run straight out from the shoulder along ``direction_openxr``."""
    return _arm_frame(direction_openxr, direction_openxr, **kwargs)


def _arm_frame(
    upper_arm_openxr: np.ndarray,
    forearm_openxr: np.ndarray,
    *,
    human_segments: tuple[float, float, float] = (
        _HUMAN_PELVIS_TO_SHOULDER,
        _HUMAN_SHOULDER_TO_ELBOW,
        _HUMAN_ELBOW_TO_WRIST,
    ),
    pelvis_position_openxr: np.ndarray | None = None,
    shoulder_direction_openxr: np.ndarray = _OPENXR_UP,
    hinge_openxr: np.ndarray | None = None,
) -> FullBodyPose:
    """One frame whose two arms run along ``upper_arm_openxr``, then ``forearm_openxr``.

    The operator faces -Z. The spine runs ``shoulder_direction_openxr`` from the pelvis, each
    shoulder spread to its side of that line at the pelvis-to-shoulder distance; the hips stay
    level either side of the pelvis. Each WRIST is turned as the skeleton turns it for that
    forearm, its hinge toward ``hinge_openxr``.
    """
    upper = _unit(upper_arm_openxr)
    arm = _unit(forearm_openxr)
    if hinge_openxr is None:
        hinge_openxr = (
            _OPENXR_RIGHT if abs(np.dot(arm, _OPENXR_RIGHT)) < 0.9 else _OPENXR_FORWARD
        )
    torso = _unit(shoulder_direction_openxr)
    pelvis = (
        np.zeros(3)
        if pelvis_position_openxr is None
        else np.asarray(pelvis_position_openxr, dtype=np.float64)
    )
    to_shoulder, upper_arm, forearm = human_segments

    joints = BodyJoints()
    positions = np.zeros((int(BodyJoint.NUM_JOINTS), 3), dtype=np.float32)
    orientations = np.zeros((int(BodyJoint.NUM_JOINTS), 4), dtype=np.float32)
    orientations[:, 3] = 1.0
    positions[int(BodyJoint.PELVIS)] = pelvis
    for slot, side in ((BodyJoint.LEFT_HIP, -1.0), (BodyJoint.RIGHT_HIP, 1.0)):
        positions[int(slot)] = (
            pelvis
            + side * _HUMAN_HIP_HALF_WIDTH * _OPENXR_RIGHT
            - _HUMAN_HIP_DROP * _OPENXR_UP
        )
    for slot, fraction in _HUMAN_SPINE:
        positions[int(slot)] = pelvis + torso * to_shoulder * fraction
    across = _unit(_OPENXR_RIGHT - np.dot(_OPENXR_RIGHT, torso) * torso)
    for (shoulder_slot, elbow_slot, wrist_slot), side in zip(
        _OPERATOR_ARMS, (-1.0, 1.0)
    ):
        shoulder = pelvis + to_shoulder * _unit(
            torso + side * _HUMAN_SHOULDER_SPREAD * across
        )
        elbow = shoulder + upper * upper_arm
        positions[int(shoulder_slot)] = shoulder
        positions[int(elbow_slot)] = elbow
        positions[int(wrist_slot)] = elbow + arm * forearm
        orientations[int(wrist_slot)] = _wrist_orientation(
            wrist_slot, arm, hinge_openxr
        )
    joints.positions[:] = positions
    joints.orientations[:] = orientations
    joints.is_valid[:] = 1
    return FullBodyPose(joints, True)


def _twist_wrists(frame: FullBodyPose, degrees: float) -> FullBodyPose:
    """Turn ``frame``'s two WRIST slots in place about their own forearms, and return it."""
    positions = np.asarray(frame.joints.positions, dtype=np.float64)
    for _, elbow, wrist in _OPERATOR_ARMS:
        forearm = _unit(positions[int(wrist)] - positions[int(elbow)])
        twisted = Rotation.from_rotvec(
            np.radians(degrees) * forearm
        ) * Rotation.from_quat(frame.joints.orientations[int(wrist)])
        frame.joints.orientations[int(wrist)] = twisted.as_quat()
    return frame


def _turned(
    frame: FullBodyPose, degrees: float, *, about_openxr: np.ndarray | None = None
) -> FullBodyPose:
    """Turn the whole of ``frame``, hips and all, about the vertical through ``about_openxr``.

    Positions go round that line and every slot's orientation turns with them, the way an
    operator turning on the spot -- or walking in a circle -- arrives. Returns ``frame``.
    """
    turn = Rotation.from_rotvec(np.radians(degrees) * _OPENXR_UP)
    about = np.zeros(3) if about_openxr is None else np.asarray(about_openxr)
    positions = np.asarray(frame.joints.positions, dtype=np.float64)
    frame.joints.positions[:] = about + turn.apply(positions - about)
    frame.joints.orientations[:] = (
        turn * Rotation.from_quat(np.asarray(frame.joints.orientations))
    ).as_quat()
    return frame


def _tilted(
    frame: FullBodyPose,
    turn_isaac: Rotation,
    *,
    slots: tuple[BodyJoint, ...] | None = None,
) -> FullBodyPose:
    """Turn ``slots`` of ``frame`` -- all of them unless told -- about the operator's pelvis.

    ``turn_isaac`` is given in Isaac's axes; positions and orientations turn with it alike.
    Returns ``frame``.
    """
    turn = _in_openxr(turn_isaac)
    indices = [int(slot) for slot in (slots or range(int(BodyJoint.NUM_JOINTS)))]
    positions = np.asarray(frame.joints.positions, dtype=np.float64)
    orientations = np.asarray(frame.joints.orientations, dtype=np.float64)
    about = positions[int(BodyJoint.PELVIS)].copy()
    positions[indices] = about + turn.apply(positions[indices] - about)
    orientations[indices] = (turn * Rotation.from_quat(orientations[indices])).as_quat()
    frame.joints.positions[:] = positions
    frame.joints.orientations[:] = orientations
    return frame


def _run(source: JmexG1ActionSource, frame: FullBodyPose | None) -> np.ndarray:
    group = TensorGroup(source.input_spec()[FULL_BODY_INPUT])
    group[0] = frame
    outputs = source({FULL_BODY_INPUT: group}, ComputeContext())
    return np.asarray(outputs[ACTION_OUTPUT][0])


def _target_pose(action: np.ndarray, task_index: int) -> tuple[np.ndarray, np.ndarray]:
    start = task_index * POSE_DIM
    return action[start : start + 3], action[start + 3 : start + POSE_DIM]


def _robot_forearm_axes(
    action: np.ndarray, task_index: int, layout: G1ActionLayout, *, wxyz: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """Where the target puts the robot's forearm and elbow hinge, read from the action.

    ``wxyz`` reads the orientation field scalar-first instead, as an Isaac Lab 2.x copy would.
    """
    _, quaternion = _target_pose(action, task_index)
    if wxyz:
        quaternion = np.roll(quaternion, -1)
    target = Rotation.from_quat(quaternion) * Rotation.from_quat(
        layout.frame_targets[task_index].forearm_frame_in_task_xyzw
    )
    matrix = target.as_matrix()
    return matrix[:, 0], matrix[:, 1]


def test_a_straight_arm_reaches_exactly_the_robots_arm_length_from_its_shoulder():
    """Per-segment reconstruction: the operator's angles, the robot's lengths."""
    layout = _g1_layout()
    pelvis = _yawed_pelvis(90.0)
    source = JmexG1ActionSource(layout=layout, pelvis_pose_provider=lambda: pelvis)

    action = _run(source, _straight_arm_frame(_OPENXR_FORWARD))

    for task_index, shoulder in enumerate(_robot_shoulders_in_world(pelvis)):
        position, _ = _target_pose(action, task_index)
        reach = np.linalg.norm(position - shoulder)
        assert reach == pytest.approx(sum(_ROBOT_SEGMENTS), abs=1e-5)


def test_the_arm_hangs_from_the_robots_own_shoulder_never_from_the_operators():
    """The torso lies straight ahead, so the operator's shoulder is nowhere near the robot's."""
    layout = _g1_layout()
    pelvis = _yawed_pelvis(90.0)
    source = JmexG1ActionSource(layout=layout, pelvis_pose_provider=lambda: pelvis)

    action = _run(
        source,
        _straight_arm_frame(_OPENXR_FORWARD, shoulder_direction_openxr=_OPENXR_FORWARD),
    )

    lying_forward = _torso_turn(_LEANING_FORWARD, 90.0)
    for side, turned in zip(
        ("left", "right"), _robot_shoulders_in_world(pelvis, lying_forward)
    ):
        shoulder, _ = _target_pose(action, _task_index(layout, f"{side}_shoulder"))
        wrist, _ = _target_pose(action, _task_index(layout, f"{side}_wrist"))
        np.testing.assert_allclose(shoulder, turned, atol=1e-5)
        assert np.linalg.norm(wrist - shoulder) <= sum(_ROBOT_SEGMENTS) + 1e-6


@pytest.mark.parametrize("swing_deg", [0.0, 90.0, -60.0])
def test_the_elbow_target_follows_the_operators_upper_arm_through_the_swing(swing_deg):
    """With the hand held still the operator's elbow can still go round; the target follows it."""
    layout = _g1_layout()
    pelvis = _yawed_pelvis(90.0)
    source = JmexG1ActionSource(layout=layout, pelvis_pose_provider=lambda: pelvis)
    upper_arm, forearm = _swung_bent_arm(swing_deg)

    action = _run(source, _arm_frame(upper_arm, forearm))

    turn = Rotation.from_quat(pelvis.orientation_xyzw)
    for side, shoulder in zip(("left", "right"), _robot_shoulders_in_world(pelvis)):
        elbow, _ = _target_pose(action, _task_index(layout, f"{side}_elbow"))
        np.testing.assert_allclose(
            elbow,
            shoulder + _ROBOT_SEGMENTS[0] * turn.apply(_in_isaac_axes(upper_arm)),
            atol=1e-5,
        )


def test_the_wrist_target_is_one_robot_forearm_on_from_the_elbow_target():
    """Both are points of one rebuilt arm, so the IK is never handed a pair it cannot reach."""
    layout = _g1_layout()
    pelvis = _yawed_pelvis(90.0)
    source = JmexG1ActionSource(layout=layout, pelvis_pose_provider=lambda: pelvis)
    upper_arm, forearm = _swung_bent_arm(30.0)

    action = _run(source, _arm_frame(upper_arm, forearm))

    turn = Rotation.from_quat(pelvis.orientation_xyzw)
    for side in ("left", "right"):
        elbow, _ = _target_pose(action, _task_index(layout, f"{side}_elbow"))
        wrist, _ = _target_pose(action, _task_index(layout, f"{side}_wrist"))
        np.testing.assert_allclose(
            wrist - elbow,
            _ROBOT_SEGMENTS[1] * turn.apply(_in_isaac_axes(forearm)),
            atol=1e-5,
        )


def test_standing_in_the_reference_pose_hangs_each_arm_from_where_the_robots_shoulder_rests():
    """Upright and facing ahead is a torso pose of zero."""
    layout = _g1_layout()
    pelvis = _yawed_pelvis(90.0)
    source = JmexG1ActionSource(layout=layout, pelvis_pose_provider=lambda: pelvis)
    upper_arm, forearm = _swung_bent_arm(30.0)

    action = _run(source, _arm_frame(upper_arm, forearm))

    _assert_arms_hang_from_shoulders_turned_by(
        action, layout, pelvis, Rotation.identity(), upper_arm, forearm
    )
    for side, rest in zip(("left", "right"), _robot_shoulders_in_world(pelvis)):
        shoulder, _ = _target_pose(action, _task_index(layout, f"{side}_shoulder"))
        np.testing.assert_allclose(shoulder, rest, atol=1e-6)


def _assert_arms_hang_from_shoulders_turned_by(
    action: np.ndarray,
    layout: G1ActionLayout,
    pelvis: PelvisPose,
    torso: Rotation,
    upper_arm_openxr: np.ndarray,
    forearm_openxr: np.ndarray,
) -> None:
    """Each shoulder target is the robot's, turned ``torso`` about the waist; its elbow and
    wrist are the robot's arm hung from there along the operator's own segments."""
    turn = Rotation.from_quat(pelvis.orientation_xyzw)
    for side, turned in zip(
        ("left", "right"), _robot_shoulders_in_world(pelvis, torso)
    ):
        shoulder, _ = _target_pose(action, _task_index(layout, f"{side}_shoulder"))
        elbow, _ = _target_pose(action, _task_index(layout, f"{side}_elbow"))
        wrist, _ = _target_pose(action, _task_index(layout, f"{side}_wrist"))
        np.testing.assert_allclose(shoulder, turned, atol=1e-5)
        np.testing.assert_allclose(
            elbow,
            shoulder
            + _ROBOT_SEGMENTS[0] * turn.apply(_in_isaac_axes(_unit(upper_arm_openxr))),
            atol=1e-5,
        )
        np.testing.assert_allclose(
            wrist,
            elbow
            + _ROBOT_SEGMENTS[1] * turn.apply(_in_isaac_axes(_unit(forearm_openxr))),
            atol=1e-5,
        )


@pytest.mark.parametrize(
    ("axis", "degrees"),
    [
        (_LEANING_FORWARD, 25.0),
        (_LEANING_FORWARD, -15.0),
        (_BENDING_RIGHT, 20.0),
        (_BENDING_RIGHT, -20.0),
    ],
    ids=["leaning-forward", "leaning-back", "bending-right", "bending-left"],
)
def test_leaning_turns_the_shoulders_about_the_waist_and_the_arms_hang_from_them(
    axis, degrees
):
    """The whole operator tips, pelvis and all, the way a bend from the hips does."""
    layout = _g1_layout()
    pelvis = _yawed_pelvis(90.0)
    source = JmexG1ActionSource(layout=layout, pelvis_pose_provider=lambda: pelvis)
    torso = _torso_turn(axis, degrees)
    upper_arm, forearm = _swung_bent_arm(30.0)

    action = _run(source, _tilted(_arm_frame(upper_arm, forearm), torso))

    tipped = _in_openxr(torso)
    _assert_arms_hang_from_shoulders_turned_by(
        action, layout, pelvis, torso, tipped.apply(upper_arm), tipped.apply(forearm)
    )


@pytest.mark.parametrize(
    ("axis", "degrees"),
    [(_LEANING_FORWARD, 30.0), (_BENDING_RIGHT, -25.0)],
    ids=["leaning-forward", "bending-left"],
)
def test_a_lean_from_the_spine_alone_counts_the_same_as_one_from_the_hips(
    axis, degrees
):
    """Leaning is measured from upright, not from the operator's pelvis."""
    layout = _g1_layout()
    source = JmexG1ActionSource(
        layout=layout, pelvis_pose_provider=lambda: _yawed_pelvis(90.0)
    )
    torso = _torso_turn(axis, degrees)
    tipped = _in_openxr(torso)
    upper_arm, forearm = _swung_bent_arm(30.0)

    from_the_hips = _run(source, _tilted(_arm_frame(upper_arm, forearm), torso))
    from_the_spine = _run(
        source,
        _arm_frame(
            tipped.apply(upper_arm),
            tipped.apply(forearm),
            shoulder_direction_openxr=tipped.apply(_OPENXR_UP),
        ),
    )

    for task_index in range(len(layout.frame_targets)):
        np.testing.assert_allclose(
            _target_pose(from_the_spine, task_index)[0],
            _target_pose(from_the_hips, task_index)[0],
            atol=1e-5,
        )


@pytest.mark.parametrize("degrees", [35.0, -50.0])
def test_twisting_turns_the_shoulders_about_the_vertical(degrees):
    """The hips stay; the shoulders and everything above them turn."""
    layout = _g1_layout()
    pelvis = _yawed_pelvis(90.0)
    source = JmexG1ActionSource(layout=layout, pelvis_pose_provider=lambda: pelvis)
    torso = _torso_turn(_TWISTING_LEFT, degrees)
    upper_arm, forearm = _swung_bent_arm(30.0)

    action = _run(
        source, _tilted(_arm_frame(upper_arm, forearm), torso, slots=_UPPER_BODY)
    )

    turned = _in_openxr(torso)
    _assert_arms_hang_from_shoulders_turned_by(
        action, layout, pelvis, torso, turned.apply(upper_arm), turned.apply(forearm)
    )


@pytest.mark.parametrize("degrees", [45.0, 80.0])
def test_a_lean_past_the_waists_range_is_passed_on_whole(degrees):
    layout = _g1_layout()
    pelvis = _yawed_pelvis(90.0)
    source = JmexG1ActionSource(layout=layout, pelvis_pose_provider=lambda: pelvis)
    torso = _torso_turn(_LEANING_FORWARD, degrees)

    action = _run(source, _tilted(_straight_arm_frame(-_OPENXR_UP), torso))

    pivot = pelvis.place(np.asarray(_ROBOT_WAIST_PIVOT))
    axis = pelvis.rotation.apply(_LEANING_FORWARD)

    def square_to_the_axis(vector: np.ndarray) -> np.ndarray:
        return _unit(vector - np.dot(vector, axis) * axis)

    for side, rest in zip(("left", "right"), _robot_shoulders_in_world(pelvis)):
        shoulder, _ = _target_pose(action, _task_index(layout, f"{side}_shoulder"))
        swept = np.arccos(
            np.dot(
                square_to_the_axis(shoulder - pivot), square_to_the_axis(rest - pivot)
            )
        )
        assert np.degrees(swept) == pytest.approx(degrees, abs=1e-3)
    hanging = _in_openxr(torso).apply(-_OPENXR_UP)
    _assert_arms_hang_from_shoulders_turned_by(
        action, layout, pelvis, torso, hanging, hanging
    )


def test_a_lean_after_a_quarter_turn_is_still_a_lean():
    """Leaning is read against the operator's own heading, so it never becomes a side bend."""
    layout = _g1_layout()
    pelvis = _yawed_pelvis(90.0)
    source = JmexG1ActionSource(layout=layout, pelvis_pose_provider=lambda: pelvis)
    torso = _torso_turn(_LEANING_FORWARD, 25.0)
    upper_arm, forearm = _swung_bent_arm(30.0)

    facing = _run(source, _tilted(_arm_frame(upper_arm, forearm), torso))
    turned = _run(source, _turned(_tilted(_arm_frame(upper_arm, forearm), torso), 90.0))

    np.testing.assert_allclose(turned, facing, atol=1e-6)
    tipped = _in_openxr(torso)
    _assert_arms_hang_from_shoulders_turned_by(
        turned, layout, pelvis, torso, tipped.apply(upper_arm), tipped.apply(forearm)
    )


def test_reach_does_not_move_when_the_operator_gets_longer_arms():
    layout = _g1_layout()
    pelvis = _yawed_pelvis(90.0)
    source = JmexG1ActionSource(layout=layout, pelvis_pose_provider=lambda: pelvis)

    short = _run(
        source, _straight_arm_frame(_OPENXR_FORWARD, human_segments=(0.4, 0.2, 0.2))
    )
    tall = _run(
        source, _straight_arm_frame(_OPENXR_FORWARD, human_segments=(0.8, 0.5, 0.45))
    )

    np.testing.assert_allclose(short, tall, atol=1e-6)


def test_an_operator_standing_away_from_the_world_origin_retargets_to_the_same_target():
    layout = _g1_layout()
    pelvis = _yawed_pelvis(90.0)
    source = JmexG1ActionSource(layout=layout, pelvis_pose_provider=lambda: pelvis)

    centred = _run(source, _straight_arm_frame(_OPENXR_FORWARD))
    displaced = _run(
        source,
        _straight_arm_frame(
            _OPENXR_FORWARD, pelvis_position_openxr=np.array([1.5, 0.0, -2.0])
        ),
    )

    np.testing.assert_allclose(centred, displaced, atol=1e-6)


def _swung_arms_twisted_hands() -> FullBodyPose:
    return _twist_wrists(_arm_frame(*_swung_bent_arm(50.0)), 35.0)


def _reaching_across_with_a_tilted_hinge() -> FullBodyPose:
    return _straight_arm_frame(
        _OPENXR_FORWARD - 0.6 * _OPENXR_UP + 0.3 * _OPENXR_RIGHT,
        hinge_openxr=_OPENXR_RIGHT + 0.4 * _OPENXR_UP,
    )


@pytest.mark.parametrize(
    "pose", [_swung_arms_twisted_hands, _reaching_across_with_a_tilted_hinge]
)
@pytest.mark.parametrize("degrees", [37.0, 90.0, 180.0, -120.0])
@pytest.mark.parametrize(
    "about_openxr",
    [None, np.array([1.5, 0.0, -2.0])],
    ids=["on-the-spot", "round-a-point-across-the-room"],
)
def test_turning_moves_no_target(pose, degrees, about_openxr):
    """Turning round a point across the room also covers walking in a circle."""
    layout = _g1_layout()
    pelvis = _yawed_pelvis(90.0)
    source = JmexG1ActionSource(layout=layout, pelvis_pose_provider=lambda: pelvis)

    facing = _run(source, pose())
    turned = _run(source, _turned(pose(), degrees, about_openxr=about_openxr))

    np.testing.assert_allclose(turned, facing, atol=1e-6)


@pytest.mark.parametrize(
    "hips",
    [
        {BodyJoint.LEFT_HIP: [0.0, 0.1, 0.0], BodyJoint.RIGHT_HIP: [0.0, -0.1, 0.0]},
        {BodyJoint.LEFT_HIP: [0.02, 0.0, 0.0], BodyJoint.RIGHT_HIP: [0.02, 0.0, 0.0]},
    ],
    ids=["one-above-the-other", "one-point"],
)
def test_a_frame_whose_hips_give_no_heading_holds_every_target(hips):
    """No level line between the hips, no heading: nothing moves, orientations included."""
    seen = []
    layout = _g1_layout()
    source = JmexG1ActionSource(
        layout=layout,
        pelvis_pose_provider=lambda: _yawed_pelvis(90.0),
        overlay_sink=seen.append,
    )

    tracked = _run(source, _straight_arm_frame(_OPENXR_UP))
    frame = _swung_arms_twisted_hands()
    for slot, position in hips.items():
        frame.joints.positions[int(slot)] = position
    held = _run(source, frame)

    np.testing.assert_allclose(held, tracked, atol=1e-6)
    # Nor is there a way to face the skeleton on the robot.
    assert len(seen[-1].skeleton) == 0


def _spine_folded(frame: FullBodyPose) -> None:
    frame.joints.positions[int(BodyJoint.SPINE3)] = frame.joints.positions[
        int(BodyJoint.SPINE1)
    ]


def _shoulders_one_point(frame: FullBodyPose) -> None:
    frame.joints.positions[int(BodyJoint.RIGHT_SHOULDER)] = frame.joints.positions[
        int(BodyJoint.LEFT_SHOULDER)
    ]


def _shoulders_on_the_spines_line(frame: FullBodyPose) -> None:
    pelvis = np.asarray(frame.joints.positions[int(BodyJoint.PELVIS)])
    frame.joints.positions[int(BodyJoint.LEFT_SHOULDER)] = pelvis + 0.6 * _OPENXR_UP
    frame.joints.positions[int(BodyJoint.RIGHT_SHOULDER)] = pelvis + 0.5 * _OPENXR_UP


@pytest.mark.parametrize(
    "spoil", [_spine_folded, _shoulders_one_point, _shoulders_on_the_spines_line]
)
def test_a_frame_whose_upper_body_gives_no_torso_pose_holds_every_target(spoil):
    layout = _g1_layout()
    source = JmexG1ActionSource(
        layout=layout, pelvis_pose_provider=lambda: _yawed_pelvis(90.0)
    )

    tracked = _run(source, _straight_arm_frame(_OPENXR_UP))
    frame = _swung_arms_twisted_hands()
    spoil(frame)
    held = _run(source, frame)

    np.testing.assert_allclose(held, tracked, atol=1e-6)


@pytest.mark.parametrize(
    ("direction_openxr", "expected_isaac"),
    [
        (_OPENXR_FORWARD, _ISAAC_FORWARD),
        (_OPENXR_UP, _ISAAC_UP),
        (_OPENXR_RIGHT, _ISAAC_RIGHT),
    ],
)
def test_arm_direction_is_preserved_through_the_change_of_basis(
    direction_openxr, expected_isaac
):
    layout = _g1_layout()
    pelvis = _identity_pelvis()
    source = JmexG1ActionSource(layout=layout, pelvis_pose_provider=lambda: pelvis)

    action = _run(source, _straight_arm_frame(direction_openxr))

    for task_index, shoulder in enumerate(_robot_shoulders_in_world(pelvis)):
        position, _ = _target_pose(action, task_index)
        np.testing.assert_allclose(
            _unit(position - shoulder), expected_isaac, atol=1e-5
        )


def test_an_arm_hanging_at_the_side_points_the_robots_forearm_down():
    layout = _g1_layout()
    pelvis = _yawed_pelvis(90.0)
    source = JmexG1ActionSource(layout=layout, pelvis_pose_provider=lambda: pelvis)

    action = _run(source, _straight_arm_frame(-_OPENXR_UP))

    for task_index in range(2):
        forearm, _ = _robot_forearm_axes(action, task_index, layout)
        np.testing.assert_allclose(forearm, _ISAAC_UP * -1.0, atol=1e-5)


def test_the_robots_forearm_and_hinge_turn_where_the_operators_do_and_the_field_is_xyzw():
    """Asymmetric on every axis, on a robot turned 90 degrees, so neither the pelvis nor a
    scalar-first reading of the field can pass by accident."""
    layout = _g1_layout()
    pelvis = _yawed_pelvis(90.0)
    source = JmexG1ActionSource(layout=layout, pelvis_pose_provider=lambda: pelvis)
    reaching = _unit(_OPENXR_FORWARD - 0.6 * _OPENXR_UP + 0.3 * _OPENXR_RIGHT)
    hinge = _OPENXR_RIGHT + 0.4 * _OPENXR_UP

    action = _run(source, _straight_arm_frame(reaching, hinge_openxr=hinge))

    turn = Rotation.from_quat(pelvis.orientation_xyzw)
    hinge_across = _unit(hinge - np.dot(hinge, reaching) * reaching)
    for task_index in range(2):
        _, quaternion = _target_pose(action, task_index)
        assert np.linalg.norm(quaternion) == pytest.approx(1.0, abs=1e-6)
        forearm, robot_hinge = _robot_forearm_axes(action, task_index, layout)
        np.testing.assert_allclose(
            forearm, turn.apply(_in_isaac_axes(reaching)), atol=1e-5
        )
        np.testing.assert_allclose(
            robot_hinge, turn.apply(_in_isaac_axes(hinge_across)), atol=1e-5
        )
        misread, _ = _robot_forearm_axes(action, task_index, layout, wxyz=True)
        assert not np.allclose(misread, forearm, atol=1e-2)


def test_turning_the_forearm_turns_the_target_about_it_and_nothing_else():
    layout = _g1_layout()
    pelvis = _yawed_pelvis(90.0)
    source = JmexG1ActionSource(layout=layout, pelvis_pose_provider=lambda: pelvis)
    reaching = _OPENXR_FORWARD - _OPENXR_UP

    before = _run(source, _straight_arm_frame(reaching))
    after = _run(source, _twist_wrists(_straight_arm_frame(reaching), 40.0))

    for task_index in range(2):
        np.testing.assert_allclose(
            _target_pose(after, task_index)[0],
            _target_pose(before, task_index)[0],
            atol=1e-6,
        )
    axis = Rotation.from_quat(pelvis.orientation_xyzw).apply(
        _in_isaac_axes(_unit(reaching))
    )
    for task_index in range(2):
        turned = Rotation.from_quat(_target_pose(after, task_index)[1]) * (
            Rotation.from_quat(_target_pose(before, task_index)[1]).inv()
        )
        np.testing.assert_allclose(
            turned.as_rotvec(), np.radians(40.0) * axis, atol=1e-4
        )
        np.testing.assert_allclose(
            _robot_forearm_axes(after, task_index, layout)[0],
            _robot_forearm_axes(before, task_index, layout)[0],
            atol=1e-5,
        )


def test_bending_the_wrist_bends_the_target_off_the_forearms_line():
    """WRIST is read whole: a hand bent across the forearm is followed, not straightened."""
    layout = _g1_layout()
    pelvis = _yawed_pelvis(90.0)
    source = JmexG1ActionSource(layout=layout, pelvis_pose_provider=lambda: pelvis)
    reaching = _OPENXR_FORWARD - _OPENXR_UP
    bend = Rotation.from_rotvec(np.radians(20.0) * _OPENXR_RIGHT)
    frame = _straight_arm_frame(reaching)
    for _, _, wrist in _OPERATOR_ARMS:
        frame.joints.orientations[int(wrist)] = (
            bend * Rotation.from_quat(frame.joints.orientations[int(wrist)])
        ).as_quat()

    action = _run(source, frame)

    hand = Rotation.from_quat(pelvis.orientation_xyzw).apply(
        _in_isaac_axes(bend.apply(_unit(reaching)))
    )
    for task_index in range(2):
        forearm, _ = _robot_forearm_axes(action, task_index, layout)
        np.testing.assert_allclose(forearm, hand, atol=1e-5)


def test_the_hand_slots_orientation_does_not_reach_the_target():
    """HAND is rigid on WRIST in the skeleton; whatever it says, WRIST already said."""
    layout = _g1_layout()
    source = JmexG1ActionSource(
        layout=layout, pelvis_pose_provider=lambda: _yawed_pelvis(90.0)
    )

    plain = _run(source, _straight_arm_frame(_OPENXR_FORWARD))
    frame = _straight_arm_frame(_OPENXR_FORWARD)
    for slot in (BodyJoint.LEFT_HAND, BodyJoint.RIGHT_HAND):
        frame.joints.orientations[int(slot)] = Rotation.from_euler(
            "xyz", [70.0, -40.0, 25.0], degrees=True
        ).as_quat()
    turned_hands = _run(source, frame)

    np.testing.assert_allclose(turned_hands, plain, atol=1e-6)


def test_elbow_and_shoulder_targets_keep_the_robots_rest_orientation_whatever_the_hand_does():
    rest = Rotation.from_euler("xyz", [10.0, -20.0, 30.0], degrees=True)
    layout = _g1_layout(neutral_orientation_xyzw=tuple(rest.as_quat()))
    pelvis = _yawed_pelvis(90.0)
    source = JmexG1ActionSource(layout=layout, pelvis_pose_provider=lambda: pelvis)

    action = _run(
        source, _twist_wrists(_straight_arm_frame(_OPENXR_FORWARD - _OPENXR_UP), 40.0)
    )

    expected = Rotation.from_quat(pelvis.orientation_xyzw) * rest
    for name in ("left_elbow", "right_elbow", "left_shoulder", "right_shoulder"):
        _, quaternion = _target_pose(action, _task_index(layout, name))
        assert (Rotation.from_quat(quaternion) * expected.inv()).magnitude() < 1e-6


def test_every_orientation_field_is_a_unit_quaternion_even_where_the_ik_ignores_it():
    """The action term divides every field by its squared norm, position-only tasks' too."""
    layout = _g1_layout(neutral_orientation_xyzw=(0.0, 0.0, 3.0, 3.0))
    source = JmexG1ActionSource(
        layout=layout, pelvis_pose_provider=lambda: _yawed_pelvis(90.0)
    )

    for frame in (None, _straight_arm_frame(_OPENXR_FORWARD)):
        action = _run(source, frame)
        for task_index in range(len(layout.frame_targets)):
            _, quaternion = _target_pose(action, task_index)
            assert np.linalg.norm(quaternion) == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize(
    "field", ["neutral_orientation_xyzw", "forearm_frame_in_task_xyzw"]
)
def test_a_zero_orientation_is_refused_at_construction(field):
    orientations = {"forearm_frame_in_task_xyzw": _ROBOT_FOREARM_IN_TASK}
    orientations[field] = (0.0, 0.0, 0.0, 0.0)
    with pytest.raises(ValueError, match="non-zero quaternion"):
        _frame_target(
            "left_wrist",
            left_wrist_chain(),
            anchor=_ROBOT_LEFT_SHOULDER,
            **orientations,
        )


@pytest.mark.parametrize(
    "elbow_differs",
    [{"anchor": (0.02, 0.15, 0.27)}, {"segments": (0.25,)}],
    ids=["another-shoulder", "another-upper-arm"],
)
def test_an_elbow_that_disagrees_with_its_wrist_about_the_arm_is_refused(elbow_differs):
    """Cut from two different arms, the two targets need not be reachable together."""
    wrist = _frame_target(
        "left_wrist",
        left_wrist_chain(),
        anchor=_ROBOT_LEFT_SHOULDER,
        forearm_frame_in_task_xyzw=_ROBOT_FOREARM_IN_TASK,
    )
    elbow = _frame_target(
        "left_elbow",
        left_wrist_chain()[:2],
        forearm_frame_in_task_xyzw=None,
        **{"anchor": _ROBOT_LEFT_SHOULDER, **elbow_differs},
    )

    with pytest.raises(ValueError, match="left_wrist and left_elbow"):
        G1ActionLayout(
            frame_targets=(wrist, elbow), waist_pivot_pelvis=_ROBOT_WAIST_PIVOT
        )


@pytest.mark.parametrize(
    ("chain", "forearm_frame_in_task_xyzw"),
    [
        (left_wrist_chain(), None),
        (left_wrist_chain()[:2], _ROBOT_FOREARM_IN_TASK),
    ],
    ids=["wrist-without", "elbow-with"],
)
def test_a_forearm_frame_belongs_to_a_chain_that_ends_at_a_wrist_and_no_other(
    chain, forearm_frame_in_task_xyzw
):
    with pytest.raises(ValueError, match="forearm frame"):
        _frame_target(
            "arm",
            chain,
            anchor=_ROBOT_LEFT_SHOULDER,
            forearm_frame_in_task_xyzw=forearm_frame_in_task_xyzw,
        )


@pytest.mark.parametrize(
    ("task_count", "action_length"),
    [(2, 32), (4, 46), (6, 60)],
    ids=["wrists", "and-elbows", "and-shoulders"],
)
def test_the_action_grows_by_seven_with_each_frame_task(task_count, action_length):
    """Then 14 fingers and 4 locomotion: the G1's 32 and 60 are what that comes to."""
    layout = G1ActionLayout(
        frame_targets=_g1_layout().frame_targets[:task_count],
        waist_pivot_pelvis=_ROBOT_WAIST_PIVOT,
    )
    source = JmexG1ActionSource(
        layout=layout, pelvis_pose_provider=lambda: _yawed_pelvis(90.0)
    )

    action = _run(source, _straight_arm_frame(_OPENXR_FORWARD))

    assert action.shape == (action_length,)


def test_a_waist_pivot_is_a_point():
    with pytest.raises(ValueError, match="waist pivot"):
        G1ActionLayout(
            frame_targets=_g1_layout().frame_targets, waist_pivot_pelvis=(0.0, 0.0)
        )


def test_an_arm_needs_a_rest_pose_for_every_slot():
    with pytest.raises(ValueError, match="rest poses"):
        arm_frame_targets(
            left_wrist_chain(),
            anchor_position_pelvis=_ROBOT_LEFT_SHOULDER,
            robot_segment_lengths=_ROBOT_SEGMENTS,
            rest_poses_pelvis=[((0.2, 0.2, 0.1), (0.0, 0.0, 0.0, 1.0))] * 2,
            forearm_frame_in_task_xyzw=_ROBOT_FOREARM_IN_TASK,
        )


def test_a_layout_without_fingers_leaves_them_out_of_the_action():
    layout = G1ActionLayout(
        frame_targets=_g1_layout().frame_targets[:2],
        waist_pivot_pelvis=_ROBOT_WAIST_PIVOT,
        num_hand_joints=0,
    )
    source = JmexG1ActionSource(
        layout=layout, pelvis_pose_provider=lambda: _yawed_pelvis(90.0)
    )

    action = _run(source, _straight_arm_frame(_OPENXR_FORWARD))

    assert action.shape == (18,)


def test_lower_body_is_the_agreed_standing_command():
    layout = _g1_layout()
    source = JmexG1ActionSource(
        layout=layout, pelvis_pose_provider=lambda: _yawed_pelvis(90.0)
    )

    action = _run(source, _straight_arm_frame(_OPENXR_FORWARD))

    np.testing.assert_allclose(
        action[-layout.num_locomotion :], STANDING_LOCOMOTION_COMMAND, atol=0.0
    )
    assert STANDING_LOCOMOTION_COMMAND[:3] == (0.0, 0.0, 0.0)
    assert STANDING_LOCOMOTION_COMMAND[3] > 0.0


def test_fingers_are_zero_and_the_action_is_finite():
    layout = _g1_layout()
    source = JmexG1ActionSource(
        layout=layout, pelvis_pose_provider=lambda: _yawed_pelvis(90.0)
    )

    action = _run(source, _straight_arm_frame(_OPENXR_FORWARD))

    assert action.dtype == np.float32
    assert np.all(np.isfinite(action))
    finger_start = len(layout.frame_targets) * POSE_DIM
    np.testing.assert_array_equal(
        action[finger_start : finger_start + layout.num_hand_joints], 0.0
    )


def test_before_pairing_the_targets_are_the_robots_own_neutral_pose():
    """No frame has arrived yet: hold where the robot already is, not the world origin."""
    neutral = Rotation.from_euler("xyz", [20.0, -35.0, 50.0], degrees=True)
    layout = _g1_layout(
        neutral_position=(0.21, 0.19, 0.11),
        neutral_orientation_xyzw=tuple(neutral.as_quat()),
    )
    pelvis = _yawed_pelvis(90.0)
    source = JmexG1ActionSource(layout=layout, pelvis_pose_provider=lambda: pelvis)

    action = _run(source, None)

    turn = Rotation.from_quat(pelvis.orientation_xyzw)
    expected = pelvis.position + turn.apply((0.21, 0.19, 0.11))
    for task_index in range(len(layout.frame_targets)):
        position, quaternion = _target_pose(action, task_index)
        np.testing.assert_allclose(position, expected, atol=1e-5)
        assert (
            Rotation.from_quat(quaternion) * (turn * neutral).inv()
        ).magnitude() < 1e-5


def _told(caplog, level: int = logging.NOTSET) -> list[str]:
    """Messages at ``level`` or above under ``isaaccapture.``, the tree the console handler is on."""
    return [
        record.getMessage()
        for record in caplog.records
        if record.name.startswith("isaaccapture.") and record.levelno >= level
    ]


def test_with_the_logging_system_off_the_info_lines_still_print(monkeypatch, capsys):
    """``ISAACCAPTURE_LOGGING=off`` attaches no handler, so an info record shows nowhere."""
    monkeypatch.setenv("ISAACCAPTURE_LOGGING", "off")
    source = JmexG1ActionSource(
        layout=_g1_layout(), pelvis_pose_provider=_identity_pelvis
    )

    _run(source, _straight_arm_frame(_OPENXR_UP))

    printed = capsys.readouterr().out
    assert "paired" in printed
    assert "measured operator" in printed


def test_the_operator_is_told_the_plugin_has_not_paired(caplog):
    source = JmexG1ActionSource(
        layout=_g1_layout(), pelvis_pose_provider=_identity_pelvis
    )

    _run(source, None)

    warnings = "\n".join(_told(caplog, logging.WARNING))
    assert "not paired" in warnings
    assert source.settings.collection_id in warnings


@pytest.mark.parametrize(
    "tasks", [slice(0, 2), slice(4, 6)], ids=["wrists", "shoulders-alone"]
)
def test_a_lost_pairing_keeps_saying_the_last_target_is_held(caplog, tasks):
    """The repeat 300 frames later still describes what the robot is doing, not rest.

    Shoulder chains are one point whether rebuilt or at rest: chain length cannot tell which.
    """
    source = JmexG1ActionSource(
        layout=G1ActionLayout(
            frame_targets=_g1_layout().frame_targets[tasks],
            waist_pivot_pelvis=_ROBOT_WAIST_PIVOT,
        ),
        pelvis_pose_provider=_identity_pelvis,
    )

    _run(source, _straight_arm_frame(_OPENXR_UP))
    for _ in range(301):
        _run(source, None)

    reports = [line for line in _told(caplog, logging.WARNING) if "not paired" in line]
    assert len(reports) == 2
    assert all("last target" in line for line in reports)


def test_operator_segment_lengths_are_measured_from_the_first_frame():
    layout = _g1_layout()
    source = JmexG1ActionSource(
        layout=layout, pelvis_pose_provider=lambda: _yawed_pelvis(90.0)
    )
    assert source.operator_segment_lengths is None

    _run(
        source, _straight_arm_frame(_OPENXR_FORWARD, human_segments=(0.61, 0.33, 0.26))
    )

    measured = source.operator_segment_lengths
    assert measured is not None
    # Upper arm and forearm: the chain starts at the shoulder.
    np.testing.assert_allclose(measured["left_wrist"], (0.33, 0.26), atol=1e-4)
    # And they are not what drove the output.
    assert not np.allclose(measured["left_wrist"], _ROBOT_SEGMENTS)


def test_losing_the_pairing_holds_the_last_target_rather_than_dropping_the_arms():
    layout = _g1_layout()
    source = JmexG1ActionSource(
        layout=layout, pelvis_pose_provider=lambda: _yawed_pelvis(90.0)
    )

    tracked = _run(source, _straight_arm_frame(_OPENXR_UP))
    unpaired = _run(source, None)

    np.testing.assert_allclose(unpaired, tracked, atol=1e-6)


def test_the_measurement_is_taken_again_after_a_re_pairing():
    """Measured per pairing, not once per process -- Player may come back with another skeleton."""
    source = JmexG1ActionSource(
        layout=_g1_layout(), pelvis_pose_provider=_identity_pelvis
    )

    _run(
        source, _straight_arm_frame(_OPENXR_FORWARD, human_segments=(0.61, 0.33, 0.26))
    )
    _run(source, None)
    assert source.operator_segment_lengths is None

    _run(
        source, _straight_arm_frame(_OPENXR_FORWARD, human_segments=(0.50, 0.28, 0.22))
    )
    np.testing.assert_allclose(
        source.operator_segment_lengths["left_wrist"], (0.28, 0.22), atol=1e-4
    )


def test_a_frame_missing_any_slot_holds_every_target():
    """A missing foot holds both arms, orientation and all, though their own slots arrived."""
    layout = _g1_layout()
    source = JmexG1ActionSource(
        layout=layout, pelvis_pose_provider=lambda: _yawed_pelvis(90.0)
    )

    tracked = _run(source, _straight_arm_frame(_OPENXR_UP))
    held = _run(source, _twist_wrists(_model_without(BodyJoint.LEFT_FOOT), 60.0))

    np.testing.assert_allclose(held, tracked, atol=1e-6)


def test_a_wrist_that_cannot_be_rebuilt_holds_its_elbow_too():
    """Every slot valid, the upper arms moved, the forearms folded to nothing."""
    layout = _g1_layout()
    source = JmexG1ActionSource(
        layout=layout, pelvis_pose_provider=lambda: _yawed_pelvis(90.0)
    )
    tracked = _run(source, _arm_frame(*_swung_bent_arm(0.0)))

    folded = _arm_frame(*_swung_bent_arm(90.0))
    for _, elbow, wrist in _OPERATOR_ARMS:
        folded.joints.positions[int(wrist)] = folded.joints.positions[int(elbow)]
    held = _run(source, folded)

    np.testing.assert_allclose(held, tracked, atol=1e-6)


def _model_without(*slots: BodyJoint) -> FullBodyPose:
    """A frame from a model that has no joint for ``slots`` -- a robot URDF, say."""
    frame = _straight_arm_frame(_OPENXR_FORWARD)
    for slot in slots:
        frame.joints.is_valid[int(slot)] = 0
    return frame


# The torso slots a G1 URDF loaded in Player leaves empty; its arm slots come through.
_ROBOT_URDF_GAPS = (
    BodyJoint.PELVIS,
    BodyJoint.SPINE1,
    BodyJoint.SPINE2,
    BodyJoint.SPINE3,
    BodyJoint.NECK,
    BodyJoint.HEAD,
)


def test_a_model_that_never_fills_every_slot_is_named_on_the_console(caplog):
    source = JmexG1ActionSource(
        layout=_g1_layout(), pelvis_pose_provider=_identity_pelvis
    )
    caplog.clear()

    for _ in range(60):
        _run(source, _model_without(*_ROBOT_URDF_GAPS))

    (warning,) = [line for line in _told(caplog, logging.WARNING) if "skeleton" in line]
    assert "standard human skeleton" in warning
    for slot in _ROBOT_URDF_GAPS:
        assert slot.name in warning
    # The slots that did arrive are not on the list.
    assert BodyJoint.LEFT_WRIST.name not in warning


def test_the_wrong_model_leaves_the_robot_at_rest_even_with_its_arms_complete():
    layout = _g1_layout(neutral_position=(0.21, 0.19, 0.11))
    pelvis = _identity_pelvis()
    source = JmexG1ActionSource(layout=layout, pelvis_pose_provider=lambda: pelvis)

    for _ in range(60):
        action = _run(source, _model_without(*_ROBOT_URDF_GAPS))

    for task_index in range(len(layout.frame_targets)):
        position, quaternion = _target_pose(action, task_index)
        np.testing.assert_allclose(position, (0.21, 0.19, 0.11), atol=1e-6)
        np.testing.assert_allclose(quaternion, (0.0, 0.0, 0.0, 1.0), atol=1e-6)
    assert source.operator_segment_lengths is None


def test_a_single_bad_frame_is_not_reported_as_the_wrong_model(caplog):
    source = JmexG1ActionSource(
        layout=_g1_layout(), pelvis_pose_provider=_identity_pelvis
    )

    _run(source, _straight_arm_frame(_OPENXR_UP))
    _run(source, _model_without(BodyJoint.LEFT_ELBOW))
    for _ in range(60):
        _run(source, _straight_arm_frame(_OPENXR_UP))

    assert not [line for line in _told(caplog) if "standard human skeleton" in line]


def test_the_overlay_draws_the_rebuilt_arm_from_the_robots_shoulder_to_the_ik_target():
    seen = []
    pelvis = _yawed_pelvis(90.0)
    source = JmexG1ActionSource(
        layout=_g1_layout(),
        pelvis_pose_provider=lambda: pelvis,
        overlay_sink=seen.append,
    )

    action = _run(source, _straight_arm_frame(_OPENXR_FORWARD - _OPENXR_UP))

    (overlay,) = seen
    shoulders = _robot_shoulders_in_world(pelvis)
    for task_index, name in enumerate(("left_wrist", "right_wrist")):
        shoulder, elbow, wrist = overlay.chains[name]
        np.testing.assert_allclose(shoulder, shoulders[task_index], atol=1e-6)
        assert np.linalg.norm(elbow - shoulder) == pytest.approx(_ROBOT_SEGMENTS[0])
        np.testing.assert_allclose(
            wrist, _target_pose(action, task_index)[0], atol=1e-5
        )


def test_the_overlay_ends_every_tasks_chain_on_what_the_ik_is_told():
    """Wrist, elbow and shoulder alike, so each large dot drawn is an IK target."""
    seen = []
    layout = _g1_layout()
    source = JmexG1ActionSource(
        layout=layout,
        pelvis_pose_provider=lambda: _yawed_pelvis(90.0),
        overlay_sink=seen.append,
    )

    action = _run(source, _arm_frame(*_swung_bent_arm(30.0)))

    chains = seen[-1].chains
    for task_index, target in enumerate(layout.frame_targets):
        np.testing.assert_allclose(
            chains[target.name][-1], _target_pose(action, task_index)[0], atol=1e-5
        )


def test_the_overlay_draws_the_received_skeleton_unscaled_on_the_robots_pelvis():
    seen = []
    pelvis = _yawed_pelvis(90.0)
    source = JmexG1ActionSource(
        layout=_g1_layout(),
        pelvis_pose_provider=lambda: pelvis,
        overlay_sink=seen.append,
    )

    _run(
        source,
        _straight_arm_frame(
            _OPENXR_FORWARD, pelvis_position_openxr=np.array([1.5, 0.0, -2.0])
        ),
    )

    skeleton = seen[-1].skeleton
    operator_pelvis = skeleton[int(BodyJoint.PELVIS)]
    np.testing.assert_allclose(operator_pelvis, pelvis.position, atol=1e-5)
    for shoulder, elbow, wrist in _OPERATOR_ARMS:
        shoulder, elbow, wrist = (
            skeleton[int(slot)] for slot in (shoulder, elbow, wrist)
        )
        assert np.linalg.norm(shoulder - operator_pelvis) == pytest.approx(
            _HUMAN_PELVIS_TO_SHOULDER, abs=1e-5
        )
        assert np.linalg.norm(elbow - shoulder) == pytest.approx(
            _HUMAN_SHOULDER_TO_ELBOW, abs=1e-5
        )
        assert np.linalg.norm(wrist - elbow) == pytest.approx(
            _HUMAN_ELBOW_TO_WRIST, abs=1e-5
        )


@pytest.mark.parametrize("degrees", [0.0, 90.0, -150.0])
def test_the_overlay_draws_the_operator_facing_the_way_the_robot_does(degrees):
    """However the operator is turned, their left hip lands on the robot's left side."""
    seen = []
    pelvis = _yawed_pelvis(90.0)
    source = JmexG1ActionSource(
        layout=_g1_layout(),
        pelvis_pose_provider=lambda: pelvis,
        overlay_sink=seen.append,
    )

    _run(source, _swung_arms_twisted_hands())
    _run(source, _turned(_swung_arms_twisted_hands(), degrees))

    facing, turned = seen
    assert facing.skeleton.keys() == turned.skeleton.keys()
    for slot, point in facing.skeleton.items():
        np.testing.assert_allclose(turned.skeleton[slot], point, atol=1e-5)
    robots_left = Rotation.from_quat(pelvis.orientation_xyzw).apply(_ISAAC_RIGHT * -1.0)
    across = (
        turned.skeleton[int(BodyJoint.LEFT_HIP)]
        - turned.skeleton[int(BodyJoint.RIGHT_HIP)]
    )
    np.testing.assert_allclose(
        across, 2.0 * _HUMAN_HIP_HALF_WIDTH * robots_left, atol=1e-5
    )


def test_each_rebuilt_segment_points_where_its_received_segment_points():
    seen = []
    source = JmexG1ActionSource(
        layout=_g1_layout(),
        pelvis_pose_provider=lambda: _yawed_pelvis(90.0),
        overlay_sink=seen.append,
    )
    bent = _straight_arm_frame(_OPENXR_FORWARD - _OPENXR_UP)
    forearm = _HUMAN_ELBOW_TO_WRIST * _unit(_OPENXR_RIGHT - _OPENXR_UP)
    for _, elbow, wrist in _OPERATOR_ARMS:
        bent.joints.positions[int(wrist)] = bent.joints.positions[int(elbow)] + forearm

    _run(source, bent)

    overlay = seen[-1]
    for name, arm in zip(("left_wrist", "right_wrist"), _OPERATOR_ARMS):
        received = np.diff([overlay.skeleton[int(slot)] for slot in arm], axis=0)
        rebuilt = np.diff(overlay.chains[name], axis=0)
        assert not np.allclose(_unit(received[0]), _unit(received[1]), atol=1e-2)
        for got, want in zip(rebuilt, received):
            np.testing.assert_allclose(_unit(got), _unit(want), atol=1e-5)


def test_the_overlay_draws_no_skeleton_while_nothing_arrives():
    """Before the first pairing and after losing one; the held target is still drawn."""
    seen = []
    source = JmexG1ActionSource(
        layout=_g1_layout(),
        pelvis_pose_provider=lambda: _yawed_pelvis(90.0),
        overlay_sink=seen.append,
    )

    for frame in (None, _straight_arm_frame(_OPENXR_UP), None):
        action = _run(source, frame)
        overlay = seen[-1]
        assert (len(overlay.skeleton) == 0) == (frame is None)
        for task_index, name in enumerate(("left_wrist", "right_wrist")):
            np.testing.assert_allclose(
                overlay.chains[name][-1],
                _target_pose(action, task_index)[0],
                atol=1e-5,
            )


def test_an_unusable_slot_is_left_out_of_the_skeleton():
    seen = []
    source = JmexG1ActionSource(
        layout=_g1_layout(),
        pelvis_pose_provider=lambda: _yawed_pelvis(90.0),
        overlay_sink=seen.append,
    )
    frame = _straight_arm_frame(_OPENXR_FORWARD)
    frame.joints.is_valid[int(BodyJoint.LEFT_ELBOW)] = 0

    _run(source, frame)

    skeleton = seen[-1].skeleton
    assert int(BodyJoint.LEFT_ELBOW) not in skeleton
    assert int(BodyJoint.RIGHT_ELBOW) in skeleton


def test_without_the_operators_pelvis_there_is_no_skeleton_to_place():
    """The skeleton is placed by its pelvis; the held arms are drawn all the same."""
    seen = []
    source = JmexG1ActionSource(
        layout=_g1_layout(),
        pelvis_pose_provider=lambda: _yawed_pelvis(90.0),
        overlay_sink=seen.append,
    )

    action = _run(source, _straight_arm_frame(_OPENXR_FORWARD))
    frame = _straight_arm_frame(_OPENXR_UP)
    frame.joints.is_valid[int(BodyJoint.PELVIS)] = 0
    _run(source, frame)

    overlay = seen[-1]
    assert len(overlay.skeleton) == 0
    for task_index, name in enumerate(("left_wrist", "right_wrist")):
        np.testing.assert_allclose(
            overlay.chains[name][-1], _target_pose(action, task_index)[0], atol=1e-5
        )


@pytest.mark.parametrize("axis_sign", [1.0, -1.0])
def test_the_robots_forearm_frame_runs_out_the_forearm_with_its_hinge_toward_the_bend(
    axis_sign,
):
    """The hinge is the elbow joint's own axis, whichever way the model happens to sign it."""
    frame = forearm_frame(
        _REST_SHOULDER,
        _REST_ELBOW,
        _REST_WRIST,
        elbow_axis=axis_sign * np.array([0.0, 1.0, 0.0]),
    ).as_matrix()

    np.testing.assert_allclose(frame[:, 0], _unit(_REST_WRIST - _REST_ELBOW), atol=1e-9)
    bend = np.cross(_REST_ELBOW - _REST_SHOULDER, _REST_WRIST - _REST_ELBOW)
    assert np.dot(frame[:, 1], bend) > 0.0
    # The joint's axis, not the bend's direction: the splayed upper arm tilts the latter.
    assert abs(frame[1, 1]) > 0.999


def test_a_robot_resting_with_a_straight_elbow_has_no_forearm_frame():
    """Which way the elbow bends is read off the rest pose; a straight one does not say."""
    with pytest.raises(ValueError):
        forearm_frame(
            _REST_SHOULDER,
            _REST_ELBOW,
            _REST_ELBOW + (_REST_ELBOW - _REST_SHOULDER),
            elbow_axis=np.array([0.0, 1.0, 0.0]),
        )
