# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock

import numpy as np
import pytest

from isaaccapture.deviceio_trackers import (
    SomaBodyJointPosesV0Tracker,
    SomaBodyJointRotationsV0Tracker,
)
from isaaccapture.retargeting_engine.deviceio_source_nodes import (
    DeviceIOFullBodyPoseTracked,
    DeviceIOSomaBodyJointPosesV0Tracked,
    DeviceIOSomaBodyJointRotationsV0Tracked,
    SomaBodyRepresentation,
    SomaBodySource,
)
from isaaccapture.retargeting_engine.interface.base_retargeter import _make_output_group
from isaaccapture.retargeting_engine.interface.tensor_group import TensorGroup
from isaaccapture.retargeting_engine.tensor_types import SomaBodyInputIndex
from isaaccapture.schema import (
    BodyJoints,
    FullBodyPose,
    Point,
    SomaBodyJointPoseArrayV0,
    SomaBodyJointPosesV0,
    SomaBodyJointRotationArrayV0,
    SomaBodyJointRotationsV0,
)
from isaaccapture.teleop_session_manager.helpers import _get_trackers_from_pipeline


def pose():
    joints = SomaBodyJointRotationArrayV0()
    joints.rotations[:] = (0, 0, 0, 1)
    joints.is_valid[:] = 1
    return SomaBodyJointRotationsV0(joints, Point(1, 2, 3), True)


def fake_layer():
    layer = MagicMock()
    layer.public_joint_names = ("Root", *(f"Joint{index}" for index in range(77)))
    layer.output_joint_parent_ids = np.array([0, 0, *range(1, 77)])
    return layer


def source():
    return SomaBodySource("body", "vendor.soma", fake_layer())


def evaluated_pose():
    joints = SomaBodyJointPoseArrayV0()
    joints.positions[:] = np.arange(231, dtype=np.float32).reshape(77, 3)
    joints.orientations[:] = np.tile([0, 0, 0, 1], (77, 1))
    joints.is_valid[:] = 1
    return SomaBodyJointPosesV0(joints)


def payload_group(group_type, data):
    group = TensorGroup(group_type)
    group[0] = data
    return group


def test_tracker_discovery_and_specs():
    soma_source = source()
    assert isinstance(soma_source.get_tracker(), SomaBodyJointRotationsV0Tracker)
    assert soma_source.get_vendor() is None
    assert _get_trackers_from_pipeline(soma_source) == [soma_source.get_tracker()]
    assert list(soma_source.input_spec()) == ["deviceio_soma_body"]
    assert list(soma_source.output_spec()) == [SomaBodySource.BODY]
    assert soma_source.output_spec()[SomaBodySource.BODY].is_optional
    assert len(soma_source.joint_names) == 77
    assert len(soma_source.bones) == 76


def test_polling_preserves_payload_before_source_evaluation():
    soma_source = source()
    tracker = MagicMock()
    soma_source._tracker = tracker
    raw = pose()
    tracker.get_data.return_value = raw
    session = object()
    inputs = soma_source.poll_tracker(session)
    tracker.get_data.assert_called_once_with(session)
    assert inputs["deviceio_soma_body"][0] is raw


def test_source_evaluates_received_payload_into_body_tensors():
    soma_source = source()
    raw = pose()
    expected = (
        np.ones((77, 3), dtype=np.float32),
        np.tile([0, 0, 0, 1], (77, 1)).astype(np.float32),
        np.ones(77, dtype=np.uint8),
    )
    soma_source._evaluator.evaluate = MagicMock(return_value=expected)
    inputs = payload_group(soma_source.input_spec()["deviceio_soma_body"], raw)

    evaluated = soma_source({"deviceio_soma_body": inputs})[SomaBodySource.BODY]

    soma_source._evaluator.evaluate.assert_called_once_with(raw)
    np.testing.assert_array_equal(
        evaluated[SomaBodyInputIndex.JOINT_POSITIONS], expected[0]
    )
    np.testing.assert_array_equal(
        evaluated[SomaBodyInputIndex.JOINT_ORIENTATIONS], expected[1]
    )
    np.testing.assert_array_equal(
        evaluated[SomaBodyInputIndex.JOINT_VALID], expected[2]
    )


def test_joint_pose_source_maps_received_payload_without_fk():
    soma_source = SomaBodySource(
        "body",
        "vendor.soma",
        representation=SomaBodyRepresentation.JOINT_POSES,
    )
    assert isinstance(soma_source.get_tracker(), SomaBodyJointPosesV0Tracker)
    raw = evaluated_pose()
    assert soma_source._evaluator is None
    inputs = payload_group(soma_source.input_spec()["deviceio_soma_body"], raw)

    evaluated = soma_source({"deviceio_soma_body": inputs})[SomaBodySource.BODY]

    np.testing.assert_array_equal(
        evaluated[SomaBodyInputIndex.JOINT_POSITIONS], raw.joint_poses.positions
    )
    np.testing.assert_array_equal(
        evaluated[SomaBodyInputIndex.JOINT_ORIENTATIONS],
        raw.joint_poses.orientations,
    )
    np.testing.assert_array_equal(
        evaluated[SomaBodyInputIndex.JOINT_VALID], raw.joint_poses.is_valid
    )


def test_absent_sample_clears_reused_output():
    soma_source = source()
    soma_source._evaluator.evaluate = MagicMock(
        return_value=(
            np.zeros((77, 3), dtype=np.float32),
            np.zeros((77, 4), dtype=np.float32),
            np.zeros(77, dtype=np.uint8),
        )
    )
    inputs = payload_group(soma_source.input_spec()["deviceio_soma_body"], pose())
    outputs = {
        SomaBodySource.BODY: _make_output_group(
            soma_source.output_spec()[SomaBodySource.BODY]
        )
    }
    soma_source.compute({"deviceio_soma_body": inputs}, outputs)
    assert not outputs[SomaBodySource.BODY].is_none
    inputs[0] = None
    soma_source.compute({"deviceio_soma_body": inputs}, outputs)
    assert outputs[SomaBodySource.BODY].is_none
    inputs[0] = pose()
    soma_source.compute({"deviceio_soma_body": inputs}, outputs)
    assert not outputs[SomaBodySource.BODY].is_none


def test_soma_and_fullbody_payloads_are_distinct():
    soma = DeviceIOSomaBodyJointRotationsV0Tracked()
    soma_poses = DeviceIOSomaBodyJointPosesV0Tracked()
    fullbody = DeviceIOFullBodyPoseTracked()
    with pytest.raises(ValueError, match="type mismatch"):
        soma.check_compatibility(fullbody)
    with pytest.raises(TypeError, match="SomaBodyJointRotationsV0"):
        payload_group(soma, FullBodyPose(BodyJoints(), False))
    with pytest.raises(TypeError, match="FullBodyPose"):
        payload_group(fullbody, pose())
    with pytest.raises(ValueError, match="type mismatch"):
        soma.check_compatibility(soma_poses)
    with pytest.raises(TypeError, match="SomaBodyJointPosesV0"):
        payload_group(soma_poses, pose())


def test_empty_collection_rejected():
    with pytest.raises(ValueError, match="collection_id"):
        SomaBodySource("body", "", fake_layer())


def test_joint_rotations_require_prepared_layer():
    with pytest.raises(ValueError, match="prepared SOMA layer"):
        SomaBodySource("body", "vendor.soma")
