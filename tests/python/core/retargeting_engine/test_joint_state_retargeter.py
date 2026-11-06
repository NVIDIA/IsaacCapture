# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sim-free unit tests for the generic JointStateRetargeter (leader arms, exoskeletons, ...).

Covers both modes at the ``BaseRetargeter.compute`` level and end-to-end through a
``TensorReorderer`` + ``OutputCombiner`` pipeline (the object an Isaac Lab pipeline_builder
returns), with no ``gym.make``, USD, GPU, or XR device:

* ``mode="joint"`` -- name-keyed remap + per-joint affine, hold-last, reset.
* ``mode="ee_pose"`` -- URDF forward kinematics (guarded on ``pinocchio``) + gripper command.
"""

import importlib.util
import math
import os
import tempfile

import numpy as np
import pytest

from isaaccapture.retargeting_engine.interface import (
    ComputeContext,
    ExecutionEvents,
    ExecutionState,
    OutputCombiner,
    OptionalTensorGroup,
    TensorGroup,
)
from isaaccapture.retargeting_engine.interface.retargeter_core_types import GraphTime
from isaaccapture.retargeting_engine.interface.tensor_group_type import (
    OptionalTensorGroupType,
)
from isaaccapture.retargeting_engine.tensor_types import TransformMatrix
from isaaccapture.retargeters import (
    JointStateRetargeter,
    JointStateRetargeterConfig,
    TensorReorderer,
)

_HAS_PINOCCHIO = importlib.util.find_spec("pinocchio") is not None

SO101_JOINTS = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]

# Minimal 2-revolute-joint arm with a tool frame, enough to exercise FK without external assets.
_MINIMAL_URDF = """<?xml version="1.0"?>
<robot name="mini">
  <link name="base"><inertial><mass value="0.1"/>
    <inertia ixx="1e-4" ixy="0" ixz="0" iyy="1e-4" iyz="0" izz="1e-4"/></inertial></link>
  <link name="l1"><inertial><mass value="0.1"/>
    <inertia ixx="1e-4" ixy="0" ixz="0" iyy="1e-4" iyz="0" izz="1e-4"/></inertial></link>
  <link name="tool"><inertial><mass value="0.1"/>
    <inertia ixx="1e-4" ixy="0" ixz="0" iyy="1e-4" iyz="0" izz="1e-4"/></inertial></link>
  <joint name="j1" type="revolute">
    <parent link="base"/><child link="l1"/>
    <origin xyz="0 0 0.1" rpy="0 0 0"/><axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" effort="10" velocity="10"/>
  </joint>
  <joint name="j2" type="revolute">
    <parent link="l1"/><child link="tool"/>
    <origin xyz="0.2 0 0" rpy="0 0 0"/><axis xyz="0 1 0"/>
    <limit lower="-3.14" upper="3.14" effort="10" velocity="10"/>
  </joint>
</robot>
"""


def _ctx(
    reset: bool = False, state: ExecutionState = ExecutionState.RUNNING
) -> ComputeContext:
    return ComputeContext(
        graph_time=GraphTime(sim_time_ns=0, real_time_ns=0),
        execution_events=ExecutionEvents(reset=reset, execution_state=state),
    )


def _build_io(node):
    inputs = {}
    for k, v in node.input_spec().items():
        inputs[k] = (
            OptionalTensorGroup(v)
            if isinstance(v, OptionalTensorGroupType)
            else TensorGroup(v)
        )
    outputs = {}
    for k, v in node.output_spec().items():
        outputs[k] = (
            OptionalTensorGroup(v)
            if isinstance(v, OptionalTensorGroupType)
            else TensorGroup(v)
        )
    return inputs, outputs


def _joints_group(node, device_joints, positions):
    """Build a present, name-keyed joints TensorGroup for the retargeter's JOINTS input."""
    inner = node.input_spec()[JointStateRetargeter.JOINTS].inner_type
    group = TensorGroup(inner)
    for i, name in enumerate(device_joints):
        group[i] = float(positions.get(name, 0.0))
    return group


def _world_T_ee(translation) -> TensorGroup:
    """Build a TransformMatrix group with identity rotation and the given translation."""
    group = TensorGroup(TransformMatrix())
    matrix = np.eye(4, dtype=np.float32)
    matrix[:3, 3] = np.asarray(translation, dtype=np.float32)
    group[0] = matrix
    return group


# ===========================================================================
# Joint mode
# ===========================================================================


class TestJointMode:
    def test_output_spec_matches_target_joints(self):
        r = JointStateRetargeter(
            "r", "joint", JointStateRetargeterConfig(device_joints=SO101_JOINTS)
        )
        spec = r.output_spec()
        assert list(spec) == ["joint_targets"]
        # Defaults target_joints to device_joints (identity mirror).
        names = [t.name for t in spec["joint_targets"].types]
        assert names == SO101_JOINTS

    def test_identity_mirror(self):
        r = JointStateRetargeter(
            "r", "joint", JointStateRetargeterConfig(device_joints=SO101_JOINTS)
        )
        inputs, outputs = _build_io(r)
        positions = {n: 0.1 * (i + 1) for i, n in enumerate(SO101_JOINTS)}
        inputs[JointStateRetargeter.JOINTS] = _joints_group(r, SO101_JOINTS, positions)
        r.compute(inputs, outputs, _ctx(reset=True))
        out = outputs["joint_targets"]
        for i, n in enumerate(SO101_JOINTS):
            assert float(out[i]) == pytest.approx(positions[n])

    def test_affine_scale_offset_sign(self):
        cfg = JointStateRetargeterConfig(
            device_joints=["a", "b"],
            target_joints=["a", "b"],
            scale={"a": 2.0},
            offset={"b": 0.5},
            sign={"a": -1.0},
        )
        r = JointStateRetargeter("r", "joint", cfg)
        inputs, outputs = _build_io(r)
        inputs[JointStateRetargeter.JOINTS] = _joints_group(
            r, ["a", "b"], {"a": 0.3, "b": 0.4}
        )
        r.compute(inputs, outputs, _ctx(reset=True))
        out = outputs["joint_targets"]
        assert float(out[0]) == pytest.approx(-1.0 * 2.0 * 0.3)  # sign * scale * value
        assert float(out[1]) == pytest.approx(0.5 + 0.4)  # offset + value

    def test_name_remap(self):
        # Device joint "lead_a" feeds robot joint "robot_a".
        cfg = JointStateRetargeterConfig(
            device_joints=["lead_a", "lead_b"],
            target_joints=["robot_a", "robot_b"],
            joint_map={"lead_a": "robot_a", "lead_b": "robot_b"},
        )
        r = JointStateRetargeter("r", "joint", cfg)
        inputs, outputs = _build_io(r)
        inputs[JointStateRetargeter.JOINTS] = _joints_group(
            r, ["lead_a", "lead_b"], {"lead_a": 0.7, "lead_b": -0.2}
        )
        r.compute(inputs, outputs, _ctx(reset=True))
        out = outputs["joint_targets"]
        assert float(out[0]) == pytest.approx(0.7)
        assert float(out[1]) == pytest.approx(-0.2)

    def test_hold_last_on_dropped_frame(self):
        r = JointStateRetargeter(
            "r", "joint", JointStateRetargeterConfig(device_joints=["a", "b"])
        )
        inputs, outputs = _build_io(r)
        inputs[JointStateRetargeter.JOINTS] = _joints_group(
            r, ["a", "b"], {"a": 0.9, "b": 0.1}
        )
        r.compute(inputs, outputs, _ctx())
        # Next frame: joints input absent -> hold last commanded targets.
        inputs2, outputs2 = _build_io(r)
        r.compute(inputs2, outputs2, _ctx())
        assert float(outputs2["joint_targets"][0]) == pytest.approx(0.9)
        assert float(outputs2["joint_targets"][1]) == pytest.approx(0.1)

    def test_reset_zeros_targets(self):
        r = JointStateRetargeter(
            "r", "joint", JointStateRetargeterConfig(device_joints=["a"])
        )
        inputs, outputs = _build_io(r)
        inputs[JointStateRetargeter.JOINTS] = _joints_group(r, ["a"], {"a": 0.5})
        r.compute(inputs, outputs, _ctx())
        # Reset with no input -> targets cleared to zero.
        inputs2, outputs2 = _build_io(r)
        r.compute(inputs2, outputs2, _ctx(reset=True))
        assert float(outputs2["joint_targets"][0]) == pytest.approx(0.0)


# ===========================================================================
# EE-pose mode (URDF forward kinematics) -- guarded on pinocchio
# ===========================================================================


@pytest.fixture(scope="module")
def minimal_urdf_path():
    fd, path = tempfile.mkstemp(suffix=".urdf")
    with os.fdopen(fd, "w") as f:
        f.write(_MINIMAL_URDF)
    yield path
    os.remove(path)


@pytest.mark.skipif(not _HAS_PINOCCHIO, reason="pinocchio not installed")
class TestEePoseMode:
    def _make(self, urdf_path, **overrides):
        cfg = JointStateRetargeterConfig(
            device_joints=["j1", "j2", "gripper"],
            urdf_path=urdf_path,
            ee_link="tool",
            gripper_joint="gripper",
            **overrides,
        )
        return JointStateRetargeter("ee", "ee_pose", cfg)

    def test_requires_urdf_and_ee_link(self):
        with pytest.raises(ValueError):
            JointStateRetargeter(
                "ee", "ee_pose", JointStateRetargeterConfig(device_joints=["j1"])
            )

    def test_output_spec_is_pose_plus_gripper(self, minimal_urdf_path):
        r = self._make(minimal_urdf_path)
        spec = r.output_spec()
        assert set(spec) == {"ee_pose", "gripper_command"}
        assert spec["ee_pose"].types[0].shape == (7,)

    def test_fk_unit_quaternion_and_shape(self, minimal_urdf_path):
        r = self._make(minimal_urdf_path)
        inputs, outputs = _build_io(r)
        inputs[JointStateRetargeter.JOINTS] = _joints_group(
            r, ["j1", "j2", "gripper"], {"j1": 0.0, "j2": 0.0, "gripper": 0.0}
        )
        r.compute(inputs, outputs, _ctx(reset=True))
        pose = np.asarray(outputs["ee_pose"][0], dtype=np.float64)
        assert pose.shape == (7,)
        assert np.linalg.norm(pose[3:7]) == pytest.approx(1.0, abs=1e-5)

    def test_loads_urdf_version_1_2(self, minimal_urdf_path, tmp_path):
        # Isaac Sim's URDF exporter stamps version="1.2" on <robot>.
        urdf_1_2 = tmp_path / "mini_1_2.urdf"
        urdf_1_2.write_text(
            _MINIMAL_URDF.replace(
                '<robot name="mini">', '<robot name="mini" version="1.2">'
            )
        )
        joints = {"j1": 0.3, "j2": -0.4, "gripper": 0.0}
        poses = []
        for path in (minimal_urdf_path, str(urdf_1_2)):
            r = self._make(path)
            inputs, outputs = _build_io(r)
            inputs[JointStateRetargeter.JOINTS] = _joints_group(
                r, ["j1", "j2", "gripper"], joints
            )
            r.compute(inputs, outputs, _ctx(reset=True))
            poses.append(np.asarray(outputs["ee_pose"][0], dtype=np.float64))
        np.testing.assert_allclose(poses[1], poses[0], atol=1e-6)

    def test_fk_moves_with_joints(self, minimal_urdf_path):
        r = self._make(minimal_urdf_path)
        inputs, outputs = _build_io(r)
        inputs[JointStateRetargeter.JOINTS] = _joints_group(
            r, ["j1", "j2", "gripper"], {"j1": 0.0, "j2": 0.0, "gripper": 0.0}
        )
        r.compute(inputs, outputs, _ctx(reset=True))
        pos0 = np.asarray(outputs["ee_pose"][0], dtype=np.float64)[:3].copy()

        inputs2, outputs2 = _build_io(r)
        inputs2[JointStateRetargeter.JOINTS] = _joints_group(
            r, ["j1", "j2", "gripper"], {"j1": math.pi / 2, "j2": 0.0, "gripper": 0.0}
        )
        r.compute(inputs2, outputs2, _ctx())
        pos1 = np.asarray(outputs2["ee_pose"][0], dtype=np.float64)[:3]
        assert np.linalg.norm(pos1 - pos0) > 0.05

    def test_gripper_raw_passthrough(self, minimal_urdf_path):
        r = self._make(minimal_urdf_path)
        inputs, outputs = _build_io(r)
        inputs[JointStateRetargeter.JOINTS] = _joints_group(
            r, ["j1", "j2", "gripper"], {"j1": 0.0, "j2": 0.0, "gripper": 0.42}
        )
        r.compute(inputs, outputs, _ctx(reset=True))
        assert float(outputs["gripper_command"][0]) == pytest.approx(0.42)

    def test_gripper_normalized_closedness(self, minimal_urdf_path):
        r = self._make(minimal_urdf_path, gripper_open=0.0, gripper_close=2.0)
        inputs, outputs = _build_io(r)
        inputs[JointStateRetargeter.JOINTS] = _joints_group(
            r, ["j1", "j2", "gripper"], {"j1": 0.0, "j2": 0.0, "gripper": 1.0}
        )
        r.compute(inputs, outputs, _ctx(reset=True))
        # 1.0 is halfway between open (0) and close (2) -> closedness 0.5.
        assert float(outputs["gripper_command"][0]) == pytest.approx(0.5)


@pytest.mark.skipif(not _HAS_PINOCCHIO, reason="pinocchio not installed")
class TestEeClutch:
    """Clutch rebasing: no jump on engage, then track FK deltas off the latched home."""

    def _make(self, urdf_path):
        return JointStateRetargeter(
            "ee",
            "ee_pose",
            JointStateRetargeterConfig(
                device_joints=["j1", "j2", "gripper"],
                urdf_path=urdf_path,
                ee_link="tool",
                clutch=True,
            ),
        )

    def _fk_pos(self, urdf_path, joint_values):
        """Absolute FK position from a non-clutch retargeter (reference for delta checks)."""
        nc = JointStateRetargeter(
            "nc",
            "ee_pose",
            JointStateRetargeterConfig(
                device_joints=["j1", "j2", "gripper"],
                urdf_path=urdf_path,
                ee_link="tool",
            ),
        )
        inputs, outputs = _build_io(nc)
        inputs[JointStateRetargeter.JOINTS] = _joints_group(
            nc, ["j1", "j2", "gripper"], joint_values
        )
        nc.compute(inputs, outputs, _ctx(reset=True))
        return np.asarray(outputs["ee_pose"][0], dtype=np.float64)[:3]

    def test_clutch_adds_robot_ee_pos_input(self, minimal_urdf_path):
        r = self._make(minimal_urdf_path)
        assert JointStateRetargeter.ROBOT_EE_POS_INPUT in r.input_spec()

    def test_engage_latches_robot_ee_home_no_jump(self, minimal_urdf_path):
        r = self._make(minimal_urdf_path)
        inputs, outputs = _build_io(r)
        inputs[JointStateRetargeter.JOINTS] = _joints_group(
            r, ["j1", "j2", "gripper"], {"j1": 0.5, "j2": -0.3, "gripper": 0.0}
        )
        home = [0.31, -0.12, 0.44]
        inputs[JointStateRetargeter.ROBOT_EE_POS_INPUT] = _world_T_ee(home)
        # First RUNNING frame: EE sits at the robot's current EE (home), not the leader's FK pose.
        r.compute(inputs, outputs, _ctx(reset=True, state=ExecutionState.RUNNING))
        pose = np.asarray(outputs["ee_pose"][0], dtype=np.float64)
        np.testing.assert_allclose(pose[:3], home, atol=1e-5)

    def test_not_running_holds_and_does_not_latch(self, minimal_urdf_path):
        r = self._make(minimal_urdf_path)
        inputs, outputs = _build_io(r)
        inputs[JointStateRetargeter.JOINTS] = _joints_group(
            r, ["j1", "j2", "gripper"], {"j1": 0.5, "j2": 0.0, "gripper": 0.0}
        )
        r.compute(inputs, outputs, _ctx(state=ExecutionState.STOPPED))
        pose = np.asarray(outputs["ee_pose"][0], dtype=np.float64)
        np.testing.assert_allclose(pose[:3], [0.0, 0.0, 0.0], atol=1e-9)  # held seed
        assert r._origin is None  # not latched while stopped

    def test_motion_after_engage_adds_fk_delta(self, minimal_urdf_path):
        j0 = {"j1": 0.2, "j2": -0.1, "gripper": 0.0}
        j1 = {"j1": 0.8, "j2": 0.3, "gripper": 0.0}
        fk0 = self._fk_pos(minimal_urdf_path, j0)
        fk1 = self._fk_pos(minimal_urdf_path, j1)
        home = np.array([0.3, 0.1, 0.5])

        r = self._make(minimal_urdf_path)
        inputs, outputs = _build_io(r)
        inputs[JointStateRetargeter.JOINTS] = _joints_group(
            r, ["j1", "j2", "gripper"], j0
        )
        inputs[JointStateRetargeter.ROBOT_EE_POS_INPUT] = _world_T_ee(home)
        r.compute(
            inputs, outputs, _ctx(reset=True, state=ExecutionState.RUNNING)
        )  # engage
        np.testing.assert_allclose(
            np.asarray(outputs["ee_pose"][0], dtype=np.float64)[:3], home, atol=1e-5
        )

        inputs2, outputs2 = _build_io(r)
        inputs2[JointStateRetargeter.JOINTS] = _joints_group(
            r, ["j1", "j2", "gripper"], j1
        )
        r.compute(inputs2, outputs2, _ctx(state=ExecutionState.RUNNING))
        pos = np.asarray(outputs2["ee_pose"][0], dtype=np.float64)[:3]
        np.testing.assert_allclose(pos, home + (fk1 - fk0), atol=1e-5)


# ===========================================================================
# End-to-end pipeline (retargeter -> TensorReorderer -> OutputCombiner)
# ===========================================================================


def _run_pipeline(combiner, leaf_name, joints_group, ctx):
    result = combiner.execute_pipeline(
        {leaf_name: {JointStateRetargeter.JOINTS: joints_group}}, ctx
    )
    return np.asarray(result["action"][0], dtype=np.float64)


class TestPipeline:
    def test_joint_pipeline_action_width_and_order(self):
        r = JointStateRetargeter(
            "leader", "joint", JointStateRetargeterConfig(device_joints=SO101_JOINTS)
        )
        reorderer = TensorReorderer(
            input_config={"joint_targets": SO101_JOINTS},
            output_order=SO101_JOINTS,
            name="action_reorderer",
            input_types={"joint_targets": "scalar"},
        )
        connected = reorderer.connect({"joint_targets": r.output("joint_targets")})
        combiner = OutputCombiner({"action": connected.output("output")})

        positions = {n: 0.1 * (i + 1) for i, n in enumerate(SO101_JOINTS)}
        jg = _joints_group(r, SO101_JOINTS, positions)
        action = _run_pipeline(combiner, r.name, jg, _ctx(reset=True))
        assert action.shape == (6,)
        np.testing.assert_allclose(
            action, [positions[n] for n in SO101_JOINTS], atol=1e-6
        )

    @pytest.mark.skipif(not _HAS_PINOCCHIO, reason="pinocchio not installed")
    def test_ee_pipeline_action_width(self, minimal_urdf_path):
        pose_labels = [
            "pos_x",
            "pos_y",
            "pos_z",
            "quat_x",
            "quat_y",
            "quat_z",
            "quat_w",
        ]
        r = JointStateRetargeter(
            "leader",
            "ee_pose",
            JointStateRetargeterConfig(
                device_joints=["j1", "j2", "gripper"],
                urdf_path=minimal_urdf_path,
                ee_link="tool",
            ),
        )
        reorderer = TensorReorderer(
            input_config={"ee_pose": pose_labels, "gripper_command": ["gripper_value"]},
            output_order=pose_labels + ["gripper_value"],
            name="action_reorderer",
            input_types={"ee_pose": "array", "gripper_command": "scalar"},
        )
        connected = reorderer.connect(
            {
                "ee_pose": r.output("ee_pose"),
                "gripper_command": r.output("gripper_command"),
            }
        )
        combiner = OutputCombiner({"action": connected.output("output")})

        jg = _joints_group(
            r, ["j1", "j2", "gripper"], {"j1": 0.2, "j2": -0.3, "gripper": 0.5}
        )
        action = _run_pipeline(combiner, r.name, jg, _ctx(reset=True))
        assert action.shape == (8,)
        assert action[7] == pytest.approx(0.5)  # gripper passthrough
