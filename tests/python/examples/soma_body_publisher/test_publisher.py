# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from isaaccapture.retargeting_engine.utilities.soma_body_evaluator import (
    _dense_joint_poses,
    _SomaBodyEvaluator,
)
from isaaccapture.schema import (
    DeviceDataTimestamp,
    SomaBodyJointPosesRecord,
    SomaBodyJointRotationsRecord,
)
from isaaccapture_examples.soma_body_publisher import publisher
from isaaccapture_examples.soma_body_publisher.publisher import (
    create_layer,
    demo_controls,
    evaluate_demo_controls,
    soma_joint_poses,
    soma_joint_rotations,
)


def test_nested_soma_serialization_uses_payload_root():
    q = np.tile([0.0, 0.0, 0.0, 1.0], (77, 1))
    pose = soma_joint_rotations(q, [1, 2, 3])
    record = SomaBodyJointRotationsRecord(pose, DeviceDataTimestamp(10, 20, 30))
    assert record.data.to_bytes() == pose.to_bytes()
    assert record.to_bytes() != pose.to_bytes()

    evaluated = soma_joint_poses(
        np.zeros((77, 3), dtype=np.float32),
        np.tile([0.0, 0.0, 0.0, 1.0], (77, 1)),
    )
    evaluated_record = SomaBodyJointPosesRecord(
        evaluated, DeviceDataTimestamp(10, 20, 30)
    )
    assert evaluated_record.data.to_bytes() == evaluated.to_bytes()


def test_layer_uses_upstream_asset_cache(monkeypatch, tmp_path):
    soma = MagicMock()
    soma.get_assets_dir.return_value = tmp_path
    layer = soma.SOMALayer.return_value
    layer.public_joint_names = [
        "Root",
        *[joint.name.replace("_", "") for joint in publisher.BODY_JOINTS],
    ]
    layer.data_root = tmp_path
    monkeypatch.setitem(sys.modules, "soma", soma)
    monkeypatch.setitem(sys.modules, "torch", MagicMock())

    assert create_layer() is layer

    soma.get_assets_dir.assert_called_once_with()
    assert soma.SOMALayer.call_args.kwargs["data_root"] == str(tmp_path)
    layer.prepare_identity.assert_called_once()


@pytest.fixture
def demo(monkeypatch, tmp_path):
    monkeypatch.setattr(
        publisher, "create_layer", lambda: SimpleNamespace(data_root=tmp_path)
    )
    monkeypatch.setattr(
        publisher,
        "demo_controls",
        lambda *args: (np.tile([0, 0, 0, 1], (2, 77, 1)), [[0, 0, 0], [1, 2, 3]], None),
    )
    monkeypatch.setattr(
        publisher,
        "evaluate_demo_controls",
        lambda *args: (np.zeros((2, 77, 3)), np.tile([0, 0, 0, 1], (2, 77, 1))),
    )
    session = MagicMock()
    session.__enter__.return_value = session
    oxr_session = MagicMock()
    oxr_session.__enter__.return_value = oxr_session
    deviceio = MagicMock()
    deviceio.run.return_value = session
    oxr = MagicMock(return_value=oxr_session)
    tracker = MagicMock()
    tracker_factory = MagicMock(return_value=tracker)
    monkeypatch.setattr(publisher, "DeviceIOSession", deviceio)
    monkeypatch.setattr(publisher, "OpenXRSession", oxr)
    monkeypatch.setattr(publisher, "TensorPushTracker", tracker_factory)
    monkeypatch.setattr(publisher.time, "monotonic", lambda: 0.0)
    sleep = MagicMock()
    monkeypatch.setattr(publisher.time, "sleep", sleep)
    return SimpleNamespace(
        session=session,
        oxr_session=oxr_session,
        deviceio=deviceio,
        oxr=oxr,
        tracker=tracker,
        tracker_factory=tracker_factory,
        sleep=sleep,
    )


@pytest.mark.parametrize(
    "representation,capacity", [("joint-rotations", 2048), ("joint-poses", 4096)]
)
def test_publisher_pushes_selected_schema_in_process(demo, representation, capacity):
    assert publisher.main(["publisher", "--body-representation", representation]) == 0
    demo.tracker_factory.assert_called_once_with(
        "soma_body_demo", "soma_body_" + representation.replace("-", "_"), capacity
    )
    demo.deviceio.get_required_extensions.assert_called_once_with([demo.tracker])
    demo.oxr.assert_called_once_with(
        "SomaBodyDemoPublisher", demo.deviceio.get_required_extensions.return_value
    )
    demo.deviceio.run.assert_called_once_with(
        [demo.tracker], demo.oxr_session.get_handles.return_value
    )
    assert demo.session.update.call_count == 2
    assert demo.tracker.push.call_count == 2
    if representation == "joint-rotations":
        expected = soma_joint_rotations(
            np.tile([0, 0, 0, 1], (77, 1)), [1, 2, 3]
        ).to_bytes()
    else:
        expected = soma_joint_poses(
            np.zeros((77, 3)), np.tile([0, 0, 0, 1], (77, 1))
        ).to_bytes()
    demo.tracker.push.assert_called_with(demo.session, expected)
    demo.sleep.assert_called_once_with(1 / 30)
    demo.session.__exit__.assert_called_once()
    demo.oxr_session.__exit__.assert_called_once()


@pytest.mark.parametrize("representation", ["joint-rotations", "joint-poses"])
def test_validate_only_does_not_open_runtime_or_loop(demo, representation):
    assert (
        publisher.main(
            [
                "publisher",
                "--body-representation",
                representation,
                "--validate-only",
                "--loop",
            ]
        )
        == 0
    )
    demo.tracker_factory.assert_not_called()
    demo.deviceio.run.assert_not_called()
    demo.oxr.assert_not_called()
    demo.sleep.assert_not_called()


@pytest.mark.parametrize("error", [KeyboardInterrupt, RuntimeError])
def test_publisher_closes_sessions_and_preserves_push_failures(demo, error):
    demo.tracker.push.side_effect = error("push stopped")
    if error is KeyboardInterrupt:
        assert publisher.main(["publisher"]) == 0
    else:
        with pytest.raises(RuntimeError, match="push stopped"):
            publisher.main(["publisher"])
    demo.session.__exit__.assert_called_once()
    demo.oxr_session.__exit__.assert_called_once()


def test_session_initialization_failure_closes_openxr(demo):
    demo.deviceio.run.side_effect = RuntimeError("session failed")
    with pytest.raises(RuntimeError, match="session failed"):
        publisher.main(["publisher"])
    demo.oxr_session.__exit__.assert_called_once()
    demo.tracker.push.assert_not_called()


def test_publisher_loop_repeats_clip_until_interrupted(demo):
    demo.tracker.push.side_effect = [None, None, KeyboardInterrupt]
    assert publisher.main(["publisher", "--loop"]) == 0
    assert demo.tracker.push.call_count == 3
    assert (
        demo.tracker.push.call_args_list[0].args[1]
        == demo.tracker.push.call_args_list[2].args[1]
    )


@pytest.mark.parametrize(
    "representation,capacity", [("joint-rotations", 2048), ("joint-poses", 4096)]
)
def test_validate_only_rejects_oversized_payload(
    demo, monkeypatch, representation, capacity
):
    encoder = (
        "soma_joint_poses"
        if representation == "joint-poses"
        else "soma_joint_rotations"
    )
    monkeypatch.setattr(
        publisher,
        encoder,
        lambda *args: SimpleNamespace(to_bytes=lambda: bytes(capacity + 1)),
    )
    with pytest.raises(ValueError, match="exceeds transport capacity"):
        publisher.main(
            ["publisher", "--body-representation", representation, "--validate-only"]
        )


@pytest.mark.parametrize("rate", ["0", "-1", "nan", "inf"])
def test_invalid_rate_is_rejected_before_loading_model(monkeypatch, rate):
    layer = MagicMock()
    monkeypatch.setattr(publisher, "create_layer", layer)
    with pytest.raises(SystemExit) as error:
        publisher.main(["publisher", "--rate", rate])
    assert error.value.code == 2
    layer.assert_not_called()


def test_demo_to_fbs_to_native_soma_matches_upstream(soma_assets):
    import torch
    from soma.geometry.transforms import (
        matrix_to_quaternion_xyzw,
        quaternion_xyzw_to_matrix,
    )

    evaluator = _SomaBodyEvaluator(create_layer(soma_assets))
    q, t, matrices = demo_controls(
        soma_assets / "example_animation.npy", evaluator.layer
    )
    torch.testing.assert_close(
        quaternion_xyzw_to_matrix(torch.from_numpy(q)), matrices, atol=1e-4, rtol=1e-4
    )
    with torch.no_grad():
        expected = evaluator.layer.pose(
            matrices[:1],
            transl=torch.from_numpy(t[:1]),
            pose2rot=False,
            fk_only=True,
            apply_correctives=False,
        )
    payload = soma_joint_rotations(q[0], t[0])
    record = SomaBodyJointRotationsRecord(payload, DeviceDataTimestamp(10, 20, 30))
    positions, orientations, valid = evaluator.evaluate(record.data)
    np.testing.assert_allclose(positions, expected["joints"][0].numpy(), atol=1e-4)
    expected_orientations = matrix_to_quaternion_xyzw(
        expected["transforms"][0, 1:, :3, :3]
    )
    actual_rotations = quaternion_xyzw_to_matrix(torch.from_numpy(orientations.copy()))
    expected_rotations = quaternion_xyzw_to_matrix(expected_orientations)
    torch.testing.assert_close(
        actual_rotations, expected_rotations, atol=1e-4, rtol=1e-4
    )
    np.testing.assert_array_equal(valid, np.ones(77, dtype=np.uint8))

    evaluated_positions, evaluated_orientations = evaluate_demo_controls(
        evaluator.layer, matrices, t
    )
    evaluated_payload = soma_joint_poses(
        evaluated_positions[0], evaluated_orientations[0]
    )
    sparse_positions, sparse_orientations, sparse_valid = _dense_joint_poses(
        evaluated_payload.joint_poses, 77
    )
    np.testing.assert_allclose(sparse_positions, positions, atol=1e-4)
    assert sparse_valid.all()
    torch.testing.assert_close(
        quaternion_xyzw_to_matrix(torch.from_numpy(sparse_orientations)),
        actual_rotations,
        atol=1e-4,
        rtol=1e-4,
    )

    flipped_positions, _, _ = evaluator.evaluate(soma_joint_rotations(-q[0], t[0]))
    np.testing.assert_allclose(flipped_positions, positions, atol=1e-6)

    for rotations, translation in zip(q, t, strict=True):
        assert len(soma_joint_rotations(rotations, translation).to_bytes()) <= 2048
    for frame_positions, frame_orientations in zip(
        evaluated_positions, evaluated_orientations, strict=True
    ):
        assert (
            len(soma_joint_poses(frame_positions, frame_orientations).to_bytes())
            <= 4096
        )
