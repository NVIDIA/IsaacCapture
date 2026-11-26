# SPDX-FileCopyrightText: Copyright (c) 2026 RealHand. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
import sys

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from repo_paths import repo_root

from isaaccapture.schema import (
    ControllerInputState,
    ControllerPose,
    ControllerSnapshot,
    HandJoints,
    HandPose,
    Point,
    Pose,
    Quaternion,
)
from isaaccapture.retargeting_engine.interface import (
    ComputeContext,
    ExecutionEvents,
    ExecutionState,
    TensorGroup,
)
from isaaccapture.retargeting_engine.interface.retargeter_core_types import GraphTime
from isaaccapture_examples.realhand_p7 import p7_bimanual_example as example


def execute(
    pipeline,
    mode,
    *,
    position=(0.25, 1.15, -0.40),
    rotation=None,
    absent=False,
    grip_valid=True,
    time_ns=1_000_000_000,
    reset=False,
    anchor=None,
):
    rotation = rotation or Rotation.identity()
    inputs = {}
    for leaf in pipeline.get_leaf_nodes():
        values = {}
        for name, spec in leaf.input_spec().items():
            group = TensorGroup(spec)
            if leaf.name == "world_T_anchor":
                group[0] = example.STAGE_TO_P7 if anchor is None else anchor
            else:
                point = np.asarray(position).copy()
                if name.endswith("left"):
                    point[0] *= -1
                if absent:
                    payload = None
                elif mode == "controller":
                    pose = ControllerPose(
                        Pose(Point(*point), Quaternion(*rotation.as_quat())), grip_valid
                    )
                    state = ControllerInputState(
                        primary_click=False,
                        secondary_click=False,
                        thumbstick_click=False,
                        menu_click=False,
                        thumbstick_x=0.0,
                        thumbstick_y=0.0,
                        squeeze_value=0.0,
                        trigger_value=0.0,
                    )
                    payload = ControllerSnapshot(pose, pose, state)
                else:
                    joints = HandJoints()
                    joints.positions[:] = point
                    joints.orientations[:] = rotation.as_quat()
                    joints.radii[:] = 0.008
                    joints.is_valid[:] = 1
                    payload = HandPose(joints)
                group[0] = payload
            values[name] = group
        inputs[leaf.name] = values
    context = ComputeContext(
        graph_time=GraphTime(sim_time_ns=time_ns, real_time_ns=time_ns),
        execution_events=ExecutionEvents(
            reset=reset, execution_state=ExecutionState.RUNNING
        ),
    )
    return np.from_dlpack(
        pipeline.execute_pipeline(inputs, context)["action"][0]
    ).copy()


@pytest.mark.parametrize("mode", ("controller", "handtracking"))
def test_stage_axes_yaw_and_bilateral_targets(mode):
    pipeline, order = example.build_pipeline(mode)
    assert len(order) == 14
    for position in ((0.25, 1.15, -0.40), (0.25, 1.35, -0.40), (0.25, 1.15, -0.60)):
        rotation = Rotation.from_euler("Y", 30, degrees=True)
        action = execute(
            pipeline, mode, position=position, rotation=rotation, reset=True
        )
        np.testing.assert_allclose(
            action[:3], (-position[0], -position[2], position[1]), atol=1e-6
        )
        np.testing.assert_allclose(
            action[7:10], (position[0], -position[2], position[1]), atol=1e-6
        )
        expected_rotation = (
            Rotation.from_euler("X", 90, degrees=True).as_matrix()
            @ rotation.as_matrix()
            @ np.asarray(example._TOOL_FRAME).reshape(3, 3).T
        )
        np.testing.assert_allclose(
            Rotation.from_quat(action[10:14]).as_matrix(), expected_rotation, atol=1e-6
        )


@pytest.mark.parametrize("mode", ("controller", "handtracking"))
@pytest.mark.parametrize("fps", (30, 60, 90, 1000))
def test_motion_limits_use_elapsed_time(mode, fps):
    pipeline, _ = example.build_pipeline(mode)
    first = execute(pipeline, mode)
    second = execute(
        pipeline,
        mode,
        position=(0.25, 1.35, -0.40),
        rotation=Rotation.from_euler("Y", 30, degrees=True),
        time_ns=1_000_000_000 + round(1e9 / fps),
    )
    assert np.linalg.norm(second[7:10] - first[7:10]) == pytest.approx(
        0.25 / fps, abs=2e-7
    )
    delta = Rotation.from_quat(second[10:14]) * Rotation.from_quat(first[10:14]).inv()
    assert delta.magnitude() == pytest.approx(1.5 / fps, abs=3e-7)


@pytest.mark.parametrize("mode", ("controller", "handtracking"))
def test_external_anchor_and_reset_without_input(mode):
    pipeline, _ = example.build_pipeline(mode)
    anchor = np.eye(4, dtype=np.float32)
    anchor[:3, 3] = (0.1, 0.2, 0.3)
    action = execute(pipeline, mode, anchor=anchor)
    np.testing.assert_allclose(action[7:10], (0.35, 1.35, -0.1), atol=1e-6)
    reset = execute(pipeline, mode, absent=True, time_ns=1_020_000_000, reset=True)
    np.testing.assert_allclose(reset, (0, 0, 0, 0, 0, 0, 1) * 2)


@pytest.mark.parametrize("mode", ("controller", "handtracking"))
@pytest.mark.parametrize("flag", ("--help", "--dry-run"))
def test_command_outside_checkout(mode, flag, tmp_path):
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        (str(repo_root() / "examples/realhand_p7/python"), env.get("PYTHONPATH", ""))
    )
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "isaaccapture_examples.realhand_p7.p7_bimanual_example",
            "--input-mode",
            mode,
            flag,
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,  # Assert below so failures include the child's stderr.
    )
    assert result.returncode == 0, result.stderr


def test_rejects_nonstandard_device_mode():
    with pytest.raises(ValueError, match="Unknown input mode"):
        example.build_pipeline("raw_glove")
