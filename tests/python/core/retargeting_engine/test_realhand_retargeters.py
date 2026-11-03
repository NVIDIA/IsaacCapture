# SPDX-FileCopyrightText: Copyright (c) 2026 RealHand. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Simulation-free checks for P7 and RealHand retargeters."""

import numpy as np
import pytest
import subprocess
import sys

from isaaccapture.retargeters.realhand.arm import (
    P7ControllerPoseRetargeter,
    P7WorkspacePoseConfig,
)
from isaaccapture.retargeters.realhand.hand import (
    ControllerTriggerRealHandRetargeter,
    ControllerTriggerRealHandRetargeterConfig,
)
from isaaccapture.retargeters.realhand.profiles import get_realhand_profile
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
    ControllerInputIndex,
)


def _context(*, reset: bool = False) -> ComputeContext:
    return ComputeContext(
        graph_time=GraphTime(sim_time_ns=0, real_time_ns=0),
        execution_events=ExecutionEvents(
            reset=reset, execution_state=ExecutionState.RUNNING
        ),
    )


def _build_io(retargeter):
    def make_group(group_type):
        if isinstance(group_type, OptionalTensorGroupType):
            return OptionalTensorGroup(group_type)
        return TensorGroup(group_type)

    return (
        {name: make_group(spec) for name, spec in retargeter.input_spec().items()},
        {name: make_group(spec) for name, spec in retargeter.output_spec().items()},
    )


def _fill_controller(group, *, position=(0.0, 0.0, 0.0), trigger=0.0, squeeze=0.0):
    group[ControllerInputIndex.GRIP_POSITION] = np.asarray(position, dtype=np.float32)
    group[ControllerInputIndex.GRIP_ORIENTATION] = np.asarray(
        (0.0, 0.0, 0.0, 1.0), dtype=np.float32
    )
    group[ControllerInputIndex.GRIP_IS_VALID] = True
    group[ControllerInputIndex.TRIGGER_VALUE] = trigger
    group[ControllerInputIndex.SQUEEZE_VALUE] = squeeze


@pytest.mark.parametrize("model", ["l6", "o6", "l20"])
def test_profile_resolves_downloaded_urdf(model, tmp_path):
    profile = get_realhand_profile(model)
    urdf = tmp_path / profile.asset_dir_name / profile.urdf_name
    urdf.parent.mkdir(parents=True)
    urdf.touch()

    assert profile.resolve_urdf(tmp_path) == urdf.resolve()


def test_profile_reports_asset_fetch_command_when_urdf_is_missing(tmp_path):
    with pytest.raises(
        FileNotFoundError,
        match=r"python -m isaaccapture_examples\.teleop\.fetch_realhand_assets",
    ):
        get_realhand_profile("l6").resolve_urdf(tmp_path)


def test_profiles_have_unique_action_joint_names():
    for model in ("l6", "o6", "l20"):
        profile = get_realhand_profile(model)
        assert len(profile.action_joint_names) == len(set(profile.action_joint_names))


def test_hand_imports_do_not_require_scipy_and_arm_has_install_hint():
    script = """
import sys
import importlib.abc
class BlockScipy(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "scipy" or fullname.startswith("scipy."):
            raise ModuleNotFoundError("blocked scipy", name=fullname)
sys.meta_path.insert(0, BlockScipy())
from isaaccapture.retargeters import (
    ControllerTriggerRealHandRetargeter, ControllerTriggerRealHandRetargeterConfig,
)
from isaaccapture.retargeters.realhand import get_realhand_profile
assert get_realhand_profile("l6").model == "l6"
from isaaccapture import retargeters
try:
    retargeters.P7ControllerPoseRetargeter
except ModuleNotFoundError as error:
    assert "isaaccapture[retargeters-lite]" in str(error)
else:
    raise AssertionError("arm retargeter must require scipy")
"""
    subprocess.run([sys.executable, "-c", script], check=True)


def test_p7_controller_pose_maps_absolute_workspace_and_clamps():
    retargeter = P7ControllerPoseRetargeter(
        P7WorkspacePoseConfig(
            input_device="controller_left",
            fallback_position=(0.0, 0.0, 0.0),
            fallback_rotation=(0.0, 0.0, 0.0, 1.0),
            input_center=(0.0, 0.0, 0.0),
            workspace_center=(1.0, 2.0, 3.0),
            position_scale=(2.0, 2.0, 2.0),
            max_delta=(0.25, 0.25, 0.25),
        ),
        name="p7_left",
    )
    inputs, outputs = _build_io(retargeter)
    _fill_controller(inputs["controller_left"], position=(0.1, -0.2, 0.5))
    retargeter.compute(inputs, outputs, _context(reset=True))

    pose = np.from_dlpack(outputs["ee_pose"][0])
    np.testing.assert_allclose(pose[:3], (1.2, 1.75, 3.25), atol=1.0e-6)
    np.testing.assert_allclose(pose[3:], (0.0, 0.0, 0.0, 1.0), atol=1.0e-6)


@pytest.mark.parametrize("model", ["l6", "o6", "l20"])
@pytest.mark.parametrize("side", ["left", "right"])
def test_controller_trigger_closes_only_configured_hand(model, side):
    profile = get_realhand_profile(model)
    joint_names = profile.joint_names(side)
    input_name = f"controller_{side}"
    retargeter = ControllerTriggerRealHandRetargeter(
        ControllerTriggerRealHandRetargeterConfig(
            input_device=input_name,
            joint_names=joint_names,
            side=side,
            hand_model=model,
            smoothing_alpha=1.0,
        ),
        name=f"{model}_{side}",
    )
    inputs, outputs = _build_io(retargeter)
    _fill_controller(inputs[input_name], trigger=1.0)
    retargeter.compute(inputs, outputs, _context(reset=False))

    actual = np.asarray([float(value) for value in outputs["hand_joints"]])
    expected = np.asarray([joint.trigger_closed for joint in profile.joints(side)])
    np.testing.assert_allclose(actual, expected, atol=1.0e-6)
