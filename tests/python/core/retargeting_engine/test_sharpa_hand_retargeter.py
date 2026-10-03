# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for SharpaHandRetargeter.

Resolves the generated mesh-free MJCF from the public V2D package staged
inside the Isaac Teleop wheel. The CMake test installs the ``grounding`` extra.
"""

import xml.etree.ElementTree as ET
from importlib.resources import files

import numpy as np
import pink
import pytest
from scipy.spatial.transform import Rotation

from isaaccapture.retargeters import (
    SharpaHandRetargeter,
    SharpaHandRetargeterConfig,
)
from isaaccapture.retargeting_engine.interface import (
    ComputeContext,
    ExecutionEvents,
    ExecutionState,
    OptionalTensorGroup,
    TensorGroup,
)
from isaaccapture.retargeting_engine.interface.retargeter_core_types import GraphTime
from isaaccapture.retargeting_engine.interface.tensor_group_type import (
    OptionalTensorGroupType,
)
from isaaccapture.retargeting_engine.tensor_types import (
    HandInput,
    HandInputIndex,
    HandJointIndex,
    NUM_HAND_JOINTS,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SHARPA_MJCF = str(
    files("robotic_grounding")
    / "assets"
    / "xmls"
    / "sharpawave"
    / "right_sharpawave_nomesh.xml"
)

# ---------------------------------------------------------------------------
# Helpers (mirror test_dual_input_retargeters.py patterns)
# ---------------------------------------------------------------------------

ID_QUAT = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)


def _make_context(*, reset: bool = False) -> ComputeContext:
    return ComputeContext(
        graph_time=GraphTime(sim_time_ns=0, real_time_ns=0),
        execution_events=ExecutionEvents(
            reset=reset, execution_state=ExecutionState.RUNNING
        ),
    )


def _build_io(retargeter):
    inputs = {}
    for k, v in retargeter.input_spec().items():
        if isinstance(v, OptionalTensorGroupType):
            inputs[k] = OptionalTensorGroup(v)
        else:
            inputs[k] = TensorGroup(v)
    outputs = {}
    for k, v in retargeter.output_spec().items():
        if isinstance(v, OptionalTensorGroupType):
            outputs[k] = OptionalTensorGroup(v)
        else:
            outputs[k] = TensorGroup(v)
    return inputs, outputs


def _compute_joint_values(
    retargeter: SharpaHandRetargeter, hand: TensorGroup
) -> np.ndarray:
    inputs, outputs = _build_io(retargeter)
    inputs["hand_right"] = hand
    retargeter.compute(inputs, outputs, _make_context())
    return np.asarray(
        [float(outputs["hand_joints"][i]) for i in range(len(EXPECTED_FINGER_JOINTS))]
    )


def _make_hand_input_open() -> TensorGroup:
    """Build a straight / open hand pose at the origin."""
    tg = TensorGroup(HandInput())
    positions = np.zeros((NUM_HAND_JOINTS, 3), dtype=np.float32)
    orientations = np.tile(ID_QUAT, (NUM_HAND_JOINTS, 1))
    valid = np.ones(NUM_HAND_JOINTS, dtype=np.uint8)

    # Wrist at origin
    positions[HandJointIndex.WRIST] = [0, 0, 0]

    # Thumb: straight along +X +Y
    positions[HandJointIndex.THUMB_METACARPAL] = [0.02, 0.02, 0]
    positions[HandJointIndex.THUMB_PROXIMAL] = [0.04, 0.04, 0]
    positions[HandJointIndex.THUMB_DISTAL] = [0.06, 0.06, 0]
    positions[HandJointIndex.THUMB_TIP] = [0.08, 0.08, 0]

    # Index: straight along +Z
    positions[HandJointIndex.INDEX_PROXIMAL] = [0, 0.03, 0.04]
    positions[HandJointIndex.INDEX_INTERMEDIATE] = [0, 0.03, 0.07]
    positions[HandJointIndex.INDEX_DISTAL] = [0, 0.03, 0.10]
    positions[HandJointIndex.INDEX_TIP] = [0, 0.03, 0.13]

    # Middle
    positions[HandJointIndex.MIDDLE_PROXIMAL] = [0, 0.01, 0.04]
    positions[HandJointIndex.MIDDLE_INTERMEDIATE] = [0, 0.01, 0.07]
    positions[HandJointIndex.MIDDLE_DISTAL] = [0, 0.01, 0.10]
    positions[HandJointIndex.MIDDLE_TIP] = [0, 0.01, 0.13]

    # Ring
    positions[HandJointIndex.RING_PROXIMAL] = [0, -0.01, 0.04]
    positions[HandJointIndex.RING_INTERMEDIATE] = [0, -0.01, 0.07]
    positions[HandJointIndex.RING_DISTAL] = [0, -0.01, 0.10]
    positions[HandJointIndex.RING_TIP] = [0, -0.01, 0.13]

    # Pinky
    positions[HandJointIndex.LITTLE_PROXIMAL] = [0, -0.03, 0.04]
    positions[HandJointIndex.LITTLE_INTERMEDIATE] = [0, -0.03, 0.07]
    positions[HandJointIndex.LITTLE_DISTAL] = [0, -0.03, 0.10]
    positions[HandJointIndex.LITTLE_TIP] = [0, -0.03, 0.13]

    tg[HandInputIndex.JOINT_POSITIONS] = positions
    tg[HandInputIndex.JOINT_ORIENTATIONS] = orientations
    tg[HandInputIndex.JOINT_RADII] = np.ones(NUM_HAND_JOINTS, dtype=np.float32) * 0.01
    tg[HandInputIndex.JOINT_VALID] = valid
    return tg


def _make_hand_input_curled() -> TensorGroup:
    """Build a hand pose with curled fingers."""
    tg = TensorGroup(HandInput())
    positions = np.zeros((NUM_HAND_JOINTS, 3), dtype=np.float32)
    orientations = np.tile(ID_QUAT, (NUM_HAND_JOINTS, 1))
    valid = np.ones(NUM_HAND_JOINTS, dtype=np.uint8)

    positions[HandJointIndex.WRIST] = [0, 0, 0]

    # Thumb (slightly curved)
    positions[HandJointIndex.THUMB_METACARPAL] = [0.02, 0.02, 0]
    positions[HandJointIndex.THUMB_PROXIMAL] = [0.04, 0.04, 0]
    positions[HandJointIndex.THUMB_DISTAL] = [0.05, 0.05, -0.01]
    positions[HandJointIndex.THUMB_TIP] = [0.055, 0.055, -0.025]

    # Index (curled)
    positions[HandJointIndex.INDEX_PROXIMAL] = [0, 0.03, 0.04]
    positions[HandJointIndex.INDEX_INTERMEDIATE] = [0, 0.03, 0.07]
    positions[HandJointIndex.INDEX_DISTAL] = [-0.02, 0.03, 0.08]
    positions[HandJointIndex.INDEX_TIP] = [-0.04, 0.03, 0.07]

    # Middle (curled)
    positions[HandJointIndex.MIDDLE_PROXIMAL] = [0, 0.01, 0.04]
    positions[HandJointIndex.MIDDLE_INTERMEDIATE] = [0, 0.01, 0.07]
    positions[HandJointIndex.MIDDLE_DISTAL] = [-0.02, 0.01, 0.08]
    positions[HandJointIndex.MIDDLE_TIP] = [-0.04, 0.01, 0.07]

    # Ring (curled)
    positions[HandJointIndex.RING_PROXIMAL] = [0, -0.01, 0.04]
    positions[HandJointIndex.RING_INTERMEDIATE] = [0, -0.01, 0.07]
    positions[HandJointIndex.RING_DISTAL] = [-0.02, -0.01, 0.08]
    positions[HandJointIndex.RING_TIP] = [-0.04, -0.01, 0.07]

    # Pinky (curled)
    positions[HandJointIndex.LITTLE_PROXIMAL] = [0, -0.03, 0.04]
    positions[HandJointIndex.LITTLE_INTERMEDIATE] = [0, -0.03, 0.07]
    positions[HandJointIndex.LITTLE_DISTAL] = [-0.02, -0.03, 0.08]
    positions[HandJointIndex.LITTLE_TIP] = [-0.04, -0.03, 0.07]

    tg[HandInputIndex.JOINT_POSITIONS] = positions
    tg[HandInputIndex.JOINT_ORIENTATIONS] = orientations
    tg[HandInputIndex.JOINT_RADII] = np.ones(NUM_HAND_JOINTS, dtype=np.float32) * 0.01
    tg[HandInputIndex.JOINT_VALID] = valid
    return tg


# ---------------------------------------------------------------------------
# Expected joint names from the MJCF (22 finger DOFs)
# ---------------------------------------------------------------------------

EXPECTED_FINGER_JOINTS = [
    "right_thumb_CMC_FE",
    "right_thumb_CMC_AA",
    "right_thumb_MCP_FE",
    "right_thumb_MCP_AA",
    "right_thumb_IP",
    "right_index_MCP_FE",
    "right_index_MCP_AA",
    "right_index_PIP",
    "right_index_DIP",
    "right_middle_MCP_FE",
    "right_middle_MCP_AA",
    "right_middle_PIP",
    "right_middle_DIP",
    "right_ring_MCP_FE",
    "right_ring_MCP_AA",
    "right_ring_PIP",
    "right_ring_DIP",
    "right_pinky_CMC",
    "right_pinky_MCP_FE",
    "right_pinky_MCP_AA",
    "right_pinky_PIP",
    "right_pinky_DIP",
]


# ===========================================================================
# Tests
# ===========================================================================


class TestSharpaHandRetargeter:
    @pytest.fixture()
    def retargeter(self):
        cfg = SharpaHandRetargeterConfig(
            robot_asset_path=SHARPA_MJCF,
            hand_side="right",
            input_convention="mano",
            max_iter=50,
            frequency=200.0,
        )
        return SharpaHandRetargeter(cfg, name="sharpa_test")

    def test_init_loads_model(self, retargeter):
        """Verify model loading discovers the expected finger joints."""
        spec = retargeter.output_spec()
        joint_names = [t.name for t in spec["hand_joints"].types]
        assert joint_names == EXPECTED_FINGER_JOINTS

    def test_input_spec_has_hand_key(self, retargeter):
        spec = retargeter.input_spec()
        assert "hand_right" in spec

    def test_absent_hand_outputs_zeros(self, retargeter):
        """Absent (optional) hand input should produce all-zero output."""
        inputs, outputs = _build_io(retargeter)
        retargeter.compute(inputs, outputs, _make_context())

        for i in range(len(EXPECTED_FINGER_JOINTS)):
            assert outputs["hand_joints"][i] == pytest.approx(0.0)

    def test_open_hand_produces_output(self, retargeter):
        """An open hand pose should produce finite joint angles."""
        values = _compute_joint_values(retargeter, _make_hand_input_open())
        assert all(np.isfinite(values)), f"Non-finite values: {values}"

    def test_open_and_curled_poses_are_meaningful(self, retargeter):
        open_values = _compute_joint_values(retargeter, _make_hand_input_open())
        retargeter._qpos_prev = None
        curled_values = _compute_joint_values(retargeter, _make_hand_input_curled())

        assert np.all(np.isfinite(open_values))
        assert np.all(np.isfinite(curled_values))
        assert np.any(np.abs(curled_values) > 0.01)

        finger_groups = {
            "thumb": slice(0, 5),
            "index": slice(5, 9),
            "middle": slice(9, 13),
            "ring": slice(13, 17),
            "pinky": slice(17, 22),
        }
        for finger, indices in finger_groups.items():
            delta = np.linalg.norm(curled_values[indices] - open_values[indices])
            assert delta > 1e-3, (
                f"{finger} joints did not respond to the curled target: {delta}"
            )

        ranges = {
            joint.attrib["name"]: np.fromstring(joint.attrib["range"], sep=" ")
            for joint in ET.parse(SHARPA_MJCF).getroot().iter("joint")
            if "name" in joint.attrib and "range" in joint.attrib
        }
        for values in (open_values, curled_values):
            for name, value in zip(EXPECTED_FINGER_JOINTS, values, strict=True):
                lower, upper = ranges[name]
                assert lower - 1e-6 <= value <= upper + 1e-6, (
                    f"{name}={value} is outside [{lower}, {upper}]"
                )

        kinematics = retargeter._kinematics
        solved_errors = np.asarray(
            [
                np.linalg.norm(
                    np.asarray(task.compute_error(kinematics.configuration))[:3]
                )
                for task in kinematics.frame_tasks.values()
            ]
        )
        neutral = pink.Configuration(
            kinematics.robot.model,
            kinematics.robot.data,
            kinematics.robot.q0,
        )
        neutral_errors = np.asarray(
            [
                np.linalg.norm(np.asarray(task.compute_error(neutral))[:3])
                for task in kinematics.frame_tasks.values()
            ]
        )
        assert solved_errors.mean() < neutral_errors.mean() * 0.8, (
            "solved fingertip frames are not materially closer to the curled targets"
        )

    def test_openxr_output_is_invariant_to_wrist_pose(self):
        mano_hand = _make_hand_input_curled()
        mano_positions = np.from_dlpack(
            mano_hand[HandInputIndex.JOINT_POSITIONS]
        ).copy()
        mano_rotations = Rotation.from_quat(
            np.from_dlpack(mano_hand[HandInputIndex.JOINT_ORIENTATIONS])
        ).as_matrix()

        mano_from_openxr = np.array(
            [[0.0, -1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, -1.0]]
        )

        def make_openxr_hand(wrist_position, wrist_rotation):
            openxr_hand = _make_hand_input_curled()
            openxr_positions = (
                wrist_position
                + (wrist_rotation @ mano_from_openxr.T @ mano_positions.T).T
            )
            openxr_rotations = (
                wrist_rotation @ mano_from_openxr.T @ mano_rotations @ mano_from_openxr
            )
            openxr_hand[HandInputIndex.JOINT_POSITIONS] = openxr_positions.astype(
                np.float32
            )
            openxr_hand[HandInputIndex.JOINT_ORIENTATIONS] = (
                Rotation.from_matrix(openxr_rotations).as_quat().astype(np.float32)
            )
            return openxr_hand

        def make_openxr_retargeter(name):
            return SharpaHandRetargeter(
                SharpaHandRetargeterConfig(
                    robot_asset_path=SHARPA_MJCF,
                    hand_side="right",
                    input_convention="openxr",
                    max_iter=50,
                    frequency=200.0,
                ),
                name=name,
            )

        reference = make_openxr_hand(np.zeros(3), np.eye(3))
        transformed = make_openxr_hand(
            np.array([0.4, -0.2, 0.7]),
            Rotation.from_euler("xyz", [0.3, -0.2, 0.4]).as_matrix(),
        )
        reference_values = _compute_joint_values(
            make_openxr_retargeter("sharpa_openxr_reference"), reference
        )
        transformed_values = _compute_joint_values(
            make_openxr_retargeter("sharpa_openxr_transformed"), transformed
        )

        assert transformed_values == pytest.approx(reference_values, abs=1e-6)

    def test_open_curl_open_sweep_is_smooth_and_repeatable(self, retargeter):
        samples = [
            _make_hand_input_open(),
            _make_hand_input_curled(),
            _make_hand_input_curled(),
            _make_hand_input_open(),
        ]
        outputs = [_compute_joint_values(retargeter, sample) for sample in samples]

        frame_deltas = [
            np.linalg.norm(current - previous)
            for previous, current in zip(outputs, outputs[1:])
        ]
        assert all(delta < 4.0 for delta in frame_deltas)
        assert np.linalg.norm(outputs[2] - outputs[1]) < 0.1
        assert np.linalg.norm(outputs[-1] - outputs[0]) < 0.25

    def test_warm_starting_persistence(self, retargeter):
        """Second frame should reuse warm-start from the first frame."""
        inputs, outputs = _build_io(retargeter)
        inputs["hand_right"] = _make_hand_input_curled()

        retargeter.compute(inputs, outputs, _make_context())
        assert retargeter._qpos_prev is not None, (
            "qpos_prev should be set after first frame"
        )

        qpos_after_first = retargeter._qpos_prev.copy()

        # Slightly perturbed pose for second frame
        inputs2, outputs2 = _build_io(retargeter)
        hand2 = _make_hand_input_curled()
        positions = np.from_dlpack(hand2[HandInputIndex.JOINT_POSITIONS]).copy()
        positions += 0.001  # small perturbation
        hand2[HandInputIndex.JOINT_POSITIONS] = positions
        inputs2["hand_right"] = hand2

        retargeter.compute(inputs2, outputs2, _make_context())

        qpos_after_second = retargeter._qpos_prev.copy()

        # The second qpos should be close to the first (warm-started)
        # but not identical (different input). Skip the 7-DOF FreeFlyer
        # prefix (3 translation + 4 quat) that the wrist pose is anchored to.
        FREEFLYER_NQ = 7
        diff = np.linalg.norm(
            qpos_after_second[FREEFLYER_NQ:] - qpos_after_first[FREEFLYER_NQ:]
        )
        assert diff < 1.0, (
            f"Warm-started second frame should be close to first, diff={diff}"
        )

    def test_absent_then_valid_resets_warm_start(self, retargeter):
        """Feeding absent input should reset warm-start state."""
        inputs, outputs = _build_io(retargeter)
        inputs["hand_right"] = _make_hand_input_curled()
        retargeter.compute(inputs, outputs, _make_context())
        assert retargeter._qpos_prev is not None

        # Feed absent input
        inputs_absent, outputs_absent = _build_io(retargeter)
        retargeter.compute(inputs_absent, outputs_absent, _make_context())
        assert retargeter._qpos_prev is None, (
            "qpos_prev should be reset after absent input"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
