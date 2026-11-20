# SPDX-FileCopyrightText: Copyright (c) 2026 RealHand. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
import sys

import numpy as np
import pytest
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
from isaaccapture_examples.realhand_hands.hand_example import build_pipeline
from isaaccapture_examples.realhand_hands._profiles import get_realhand_profile
from _samples import human_sample


def execute(
    pipeline,
    mode,
    *,
    triggers=(0.0, 0.0),
    grip_valid=True,
    curl=0.0,
    absent=False,
    reset=False,
    time_ns=1_000_000_000,
):
    inputs = {}
    for leaf in pipeline.get_leaf_nodes():
        values = {}
        for name, spec in leaf.input_spec().items():
            group = TensorGroup(spec)
            side = "left" if name.endswith("left") else "right"
            if absent:
                payload = None
            elif mode == "controller":
                pose = ControllerPose(
                    Pose(Point(0, 0, 0), Quaternion(0, 0, 0, 1)), grip_valid
                )
                state = ControllerInputState(
                    primary_click=False,
                    secondary_click=False,
                    thumbstick_click=False,
                    menu_click=False,
                    thumbstick_x=0.0,
                    thumbstick_y=0.0,
                    squeeze_value=0.0,
                    trigger_value=triggers[0 if side == "left" else 1],
                )
                payload = ControllerSnapshot(pose, pose, state)
            else:
                sample = human_sample(side, curl)
                joints = HandJoints()
                joints.positions[:] = [j["position"] for j in sample["joints"]]
                joints.orientations[:] = [j["orientation"] for j in sample["joints"]]
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
    return {
        key: np.asarray([float(value) for value in group])
        for key, group in pipeline.execute_pipeline(inputs, context).items()
    }


@pytest.mark.parametrize("model", ("l6", "o6", "l20"))
@pytest.mark.parametrize("mode", ("controller", "handtracking"))
def test_graph_is_standalone_and_outputs_only_fingers(model, mode):
    pipeline = build_pipeline(mode, model)
    result = execute(pipeline, mode)
    assert set(result) == {"left_hand", "right_hand"}
    profile = get_realhand_profile(model)
    for side in ("left", "right"):
        assert result[f"{side}_hand"].shape == (len(profile.joint_names(side)),)
        assert np.all(np.isfinite(result[f"{side}_hand"]))
    source_names = {leaf.name for leaf in pipeline.get_leaf_nodes()}
    assert source_names == ({"controllers"} if mode == "controller" else {"hands"})


@pytest.mark.parametrize("model", ("l6", "o6", "l20"))
def test_trigger_side_independence_invalid_pose_and_reset(model):
    pipeline = build_pipeline("controller", model)
    first = execute(pipeline, "controller", triggers=(1.0, 0.0))
    lost = execute(
        pipeline,
        "controller",
        triggers=(1.0, 0.0),
        grip_valid=False,
        time_ns=1_020_000_000,
    )
    assert np.max(first["left_hand"]) > 0.5
    np.testing.assert_allclose(first["right_hand"], 0.0)
    for side in ("left_hand", "right_hand"):
        np.testing.assert_allclose(lost[side], first[side])
    reset = execute(
        pipeline, "controller", absent=True, reset=True, time_ns=1_040_000_000
    )
    for target in reset.values():
        np.testing.assert_allclose(target, 0.0)


@pytest.mark.parametrize("fps", (30, 60, 90, 1000))
def test_joint_velocity_limit_uses_graph_time(fps):
    pipeline = build_pipeline("controller", "l20")
    first = execute(pipeline, "controller")
    second = execute(
        pipeline,
        "controller",
        triggers=(1.0, 1.0),
        time_ns=1_000_000_000 + round(1e9 / fps),
    )
    for side in first:
        assert np.max(np.abs(second[side] - first[side])) == pytest.approx(
            1.5 / fps, abs=3e-7
        )


@pytest.mark.parametrize("model", ("l6", "o6", "l20"))
@pytest.mark.parametrize("mode", ("controller", "handtracking"))
def test_command_builds_without_other_realhand_packages(model, mode, tmp_path):
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        (str(repo_root() / "examples/realhand_hands/python"), env.get("PYTHONPATH", ""))
    )
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "isaaccapture_examples.realhand_hands.hand_example",
            "--hand-model",
            model,
            "--input-mode",
            mode,
            "--dry-run",
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,  # Assert below so failures include the child's stderr.
    )
    assert result.returncode == 0, result.stderr


def test_rejects_raw_glove_mode():
    with pytest.raises(ValueError, match="Unknown input mode"):
        build_pipeline("raw_glove", "l6")
