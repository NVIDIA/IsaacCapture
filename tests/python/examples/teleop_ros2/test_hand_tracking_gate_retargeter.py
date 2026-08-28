# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the tracking gate that withholds untracked hand joints."""

from isaaccapture.retargeting_engine.interface import (
    OptionalTensorGroup,
    OptionalTensorGroupType,
    OptionalType,
    TensorGroup,
    TensorGroupType,
)
from isaaccapture_examples.teleop_ros2.teleop_ros2_retargeters import (
    HandTrackingGateRetargeter,
)

JOINTS = ["thumb_j0", "thumb_j1"]


def _make_io(gate: HandTrackingGateRetargeter, joints, valid):
    inputs = {}
    for key, group_type in gate.input_spec().items():
        group = TensorGroup(group_type)
        inputs[key] = group
    for index, value in enumerate(joints):
        inputs["hand_joints"][index] = value
    inputs["hand_valid"][0] = valid
    outputs = {
        "hand_joints": OptionalTensorGroup(gate.output_spec()["hand_joints"]),
    }
    return inputs, outputs


def _gate() -> HandTrackingGateRetargeter:
    return HandTrackingGateRetargeter(joint_names=JOINTS, name="gate")


def test_output_is_optional() -> None:
    """The absent state is how a dropout is signalled downstream."""
    assert isinstance(_gate().output_spec()["hand_joints"], OptionalTensorGroupType)


def test_tracked_frames_pass_through() -> None:
    gate = _gate()
    inputs, outputs = _make_io(gate, [0.25, -0.5], valid=1.0)

    gate._compute_fn(inputs, outputs, None)

    assert not outputs["hand_joints"].is_none
    assert outputs["hand_joints"][0] == 0.25
    assert outputs["hand_joints"][1] == -0.5


def test_untracked_frames_emit_an_absent_group() -> None:
    """An all-zero pose is also a legitimate flat-open hand, so absence -- not
    zeros -- is what tells a consumer tracking was lost."""
    gate = _gate()
    inputs, outputs = _make_io(gate, [0.25, -0.5], valid=0.0)

    gate._compute_fn(inputs, outputs, None)

    assert outputs["hand_joints"].is_none


def test_tracking_resumes_after_a_dropout() -> None:
    gate = _gate()
    inputs, outputs = _make_io(gate, [0.25, -0.5], valid=0.0)
    gate._compute_fn(inputs, outputs, None)
    assert outputs["hand_joints"].is_none

    inputs["hand_valid"][0] = 1.0
    gate._compute_fn(inputs, outputs, None)

    assert not outputs["hand_joints"].is_none
    assert outputs["hand_joints"][0] == 0.25


def test_absent_validity_is_treated_as_untracked() -> None:
    gate = _gate()
    inputs, outputs = _make_io(gate, [0.25, -0.5], valid=1.0)
    inputs["hand_valid"] = OptionalTensorGroup(
        OptionalType(
            TensorGroupType(
                "hand_valid_input", list(gate.input_spec()["hand_valid"].types)
            )
        )
    )

    gate._compute_fn(inputs, outputs, None)

    assert outputs["hand_joints"].is_none
