# SPDX-FileCopyrightText: Copyright (c) 2026 RealHand. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate native glove output against the standard HandInput contract."""

import numpy as np
import pytest

from isaaccapture.schema import HandJoints, HandPose
from isaaccapture.retargeting_engine.interface import (
    ComputeContext,
    ExecutionEvents,
    ExecutionState,
    TensorGroup,
)
from isaaccapture.retargeting_engine.interface.retargeter_core_types import GraphTime
from isaaccapture.retargeting_engine.tensor_types import HandInputIndex
from isaaccapture_examples.realhand_ffg_glove.realhand_ffg_hand_demo_bridge import (
    build_pipeline,
)


@pytest.mark.parametrize("side", ("left", "right"))
def test_native_skeleton_passes_through_hands_source(side, skeleton_samples):
    pipeline = build_pipeline()
    outputs = []
    for sample in skeleton_samples[side]:
        joints = HandJoints()
        xyz = np.asarray([joint["position"] for joint in sample["joints"]])
        joints.positions[:] = xyz
        joints.orientations[:] = [joint["orientation"] for joint in sample["joints"]]
        joints.radii[:] = [joint["radius"] for joint in sample["joints"]]
        joints.is_valid[:] = 1
        inputs = {}
        for leaf in pipeline.get_leaf_nodes():
            values = {}
            for name, spec in leaf.input_spec().items():
                group = TensorGroup(spec)
                # DeviceIO leaf inputs are not the downstream HandInput output names.
                group[0] = (
                    HandPose(joints)
                    if leaf.name == "ffg_demo_hands" and name.endswith(side)
                    else None
                )
                values[name] = group
            inputs[leaf.name] = values
        context = ComputeContext(
            graph_time=GraphTime(sim_time_ns=0, real_time_ns=0),
            execution_events=ExecutionEvents(
                reset=False, execution_state=ExecutionState.RUNNING
            ),
        )
        result = pipeline.execute_pipeline(inputs, context)
        output = result[side]
        np.testing.assert_allclose(
            np.from_dlpack(output[HandInputIndex.JOINT_POSITIONS]), xyz, atol=1e-7
        )
        np.testing.assert_allclose(
            np.linalg.norm(
                np.from_dlpack(output[HandInputIndex.JOINT_ORIENTATIONS]), axis=1
            ),
            1,
            atol=1e-6,
        )
        assert result["right" if side == "left" else "left"].is_none
        outputs.append(xyz)
    assert np.max(np.abs(outputs[-1] - outputs[0])) > 0.01


@pytest.mark.parametrize("side", ("left", "right"))
def test_existing_human_calibration_produces_finite_26_joint_hands(
    side, calibrated_skeleton_samples
):
    samples = calibrated_skeleton_samples["poses"][side]
    for sample in samples:
        xyz = np.asarray([joint["position"] for joint in sample["joints"]])
        assert xyz.shape == (26, 3)
        assert np.all(np.isfinite(xyz))
