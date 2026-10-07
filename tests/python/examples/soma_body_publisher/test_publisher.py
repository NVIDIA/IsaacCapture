# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import signal
import struct
import subprocess
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


@pytest.mark.parametrize("exit_code", [0, -signal.SIGINT, 1])
def test_publisher_interrupt_preserves_real_failures(monkeypatch, tmp_path, exit_code):
    process = MagicMock()
    process.__enter__.return_value = process
    process.stdin.write.side_effect = KeyboardInterrupt
    process.wait.return_value = exit_code
    monkeypatch.setattr(publisher.subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(
        publisher, "create_layer", lambda: SimpleNamespace(data_root=tmp_path)
    )
    monkeypatch.setattr(
        publisher,
        "demo_controls",
        lambda *args: (np.tile([0, 0, 0, 1], (1, 77, 1)), [[0, 0, 0]], None),
    )
    args = ["soma_body_publisher", "--pusher", "pusher"]
    if exit_code == 1:
        with pytest.raises(RuntimeError, match="pusher failed"):
            publisher.main(args)
    else:
        assert publisher.main(args) == 0
    process.stdin.close.assert_called_once()


@pytest.mark.parametrize(
    "packet",
    [b"x", struct.pack("<IQ", 2049, 0), struct.pack("<IQ", 8, 0) + b"badbytes"],
)
def test_pusher_rejects_invalid_packets_without_runtime(request, packet):
    executable = request.config.getoption("--soma-pusher")
    if executable is None:
        pytest.skip("Pass --soma-pusher to exercise the native producer boundary")
    result = subprocess.run(
        [str(executable), "--validate-only"],
        input=packet,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 1


def test_pusher_rejects_mismatched_body_representation(request):
    executable = request.config.getoption("--soma-pusher")
    if executable is None:
        pytest.skip("Pass --soma-pusher to exercise the native producer boundary")
    rotations = soma_joint_rotations(
        np.tile([0.0, 0.0, 0.0, 1.0], (77, 1)), [0.0, 0.0, 0.0]
    ).to_bytes()
    packet = struct.pack("<IQ", len(rotations), 0) + rotations

    result = subprocess.run(
        [
            str(executable),
            "--body-representation",
            "joint-poses",
            "--validate-only",
        ],
        input=packet,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1


def test_demo_to_fbs_to_native_soma_matches_upstream(soma_assets, request):
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

    executable = request.config.getoption("--soma-pusher")
    if executable is not None:
        packets = []
        for frame, (rotations, translation) in enumerate(zip(q, t, strict=True)):
            encoded = soma_joint_rotations(rotations, translation).to_bytes()
            packets.append(struct.pack("<IQ", len(encoded), frame) + encoded)
        result = subprocess.run(
            [str(executable), "--validate-only"],
            input=b"".join(packets),
            capture_output=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr.decode()

        pose_packets = []
        for frame, (frame_positions, frame_orientations) in enumerate(
            zip(evaluated_positions, evaluated_orientations, strict=True)
        ):
            encoded = soma_joint_poses(frame_positions, frame_orientations).to_bytes()
            pose_packets.append(struct.pack("<IQ", len(encoded), frame) + encoded)
        result = subprocess.run(
            [
                str(executable),
                "--body-representation",
                "joint-poses",
                "--validate-only",
            ],
            input=b"".join(pose_packets),
            capture_output=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr.decode()
