# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
from mcap.reader import make_reader

from isaaccapture import schema
from isaaccapture.deviceio import McapReplayConfig, ReplaySession
from isaaccapture.retargeting_engine.deviceio_source_nodes import soma_body_source
from isaaccapture.retargeting_engine.tensor_types import SomaBodyInputIndex
from isaaccapture.teleop_session_manager import (
    SessionMode,
    TeleopSession,
    TeleopSessionConfig,
)
from isaaccapture_examples.mcap_record_replay import (
    common,
    soma_body_common,
)


def payload(representation, indices, sign=1.0, translation_valid=True):
    quaternion = schema.Quaternion(0.0, 0.0, sign * 0.6, sign * 0.8)
    if representation == "joint-rotations":
        return schema.SomaBodyJointRotations(
            [
                schema.SomaBodyJointRotation(schema.SomaBodyJoint(i), quaternion)
                for i in indices
            ],
            schema.Point(1.0, 2.0, 3.0),
            translation_valid,
        )
    return schema.SomaBodyJointPoses(
        [
            schema.SomaBodyJointPose(
                schema.SomaBodyJoint(i),
                schema.Pose(schema.Point(float(i), 2.0, 3.0), quaternion),
            )
            for i in indices
        ]
    )


@pytest.mark.parametrize("representation", ["joint-rotations", "joint-poses"])
def test_repeated_native_roundtrip(soma_mcap, tmp_path, representation):
    payloads = [
        payload(representation, range(77)),
        None,
        payload(representation, [0, 1, 76], -1),
        payload(representation, [], translation_valid=False),
    ]
    timestamps = [
        schema.DeviceDataTimestamp(100 + i * 100_000_000, 80 + i, 10 + i)
        for i in range(len(payloads))
    ]
    expected = [None if data is None else data.to_bytes() for data in payloads]
    record_type = (
        schema.SomaBodyJointRotationsRecord
        if representation == "joint-rotations"
        else schema.SomaBodyJointPosesRecord
    )
    topic = f"vendor.body/soma_body_{representation.replace('-', '_')}_tracked"
    for generation in range(3):
        path = soma_mcap(
            representation,
            payloads,
            timestamps,
            collection="vendor.body",
            path=tmp_path / f"generation_{generation}.mcap",
        )
        with path.open("rb") as stream:
            records = list(make_reader(stream).iter_messages(topics=[topic]))
        assert len(records) == len(payloads)
        for (_, _, message), data, timestamp in zip(
            records, payloads, timestamps, strict=True
        ):
            assert message.data == record_type(data, timestamp).to_bytes()
            assert (
                message.log_time
                == message.publish_time
                == timestamp.available_time_local_common_clock
            )
        tracker = soma_body_common.create_tracker("vendor.body", representation)
        replayed = []
        with ReplaySession.run(
            McapReplayConfig(str(path), [(tracker, "vendor.body")])
        ) as session:
            for encoded in expected:
                session.update()
                data = tracker.get_data(session)
                assert (None if data is None else data.to_bytes()) == encoded
                replayed.append(data)
            for _ in range(2):
                session.update()
                assert tracker.get_data(session) is None
        # Re-encode the actual received payloads, without changing their timestamps.
        payloads = replayed


@pytest.mark.parametrize("representation", ["joint-rotations", "joint-poses"])
def test_native_replay_pipeline_and_bounded_loop(
    soma_mcap, monkeypatch, representation
):
    entries = [
        payload(representation, range(77)),
        None,
        payload(representation, [0, 1, 76]),
    ]
    timestamps = [
        schema.DeviceDataTimestamp(1_000_000_000 + i * 100_000_000, 0, i)
        for i in range(3)
    ]
    path = soma_mcap(representation, entries, timestamps)
    evaluator = MagicMock()
    evaluator.evaluate.return_value = (
        np.ones((77, 3), dtype=np.float32),
        np.tile(np.array([0, 0, 0, 1], dtype=np.float32), (77, 1)),
        np.ones(77, dtype=np.uint8),
    )
    monkeypatch.setattr(soma_body_common, "create_layer", lambda: object())
    monkeypatch.setattr(soma_body_source, "_SomaBodyEvaluator", lambda _: evaluator)
    pipeline = soma_body_common.build_pipeline("soma_body_demo", representation)
    config = TeleopSessionConfig(
        app_name="SomaMcapRegression",
        pipeline=pipeline,
        mode=SessionMode.REPLAY,
        mcap_config=McapReplayConfig(str(path)),
    )
    for _ in range(2):
        with TeleopSession(config) as session:
            body = session.step()["soma_body"]
            assert not body.is_none
            assert np.count_nonzero(body[SomaBodyInputIndex.JOINT_VALID]) == 77
            assert session.step()["soma_body"].is_none
            body = session.step()["soma_body"]
            assert not body.is_none
            if representation == "joint-poses":
                valid = body[SomaBodyInputIndex.JOINT_VALID]
                assert np.flatnonzero(valid).tolist() == [0, 1, 76]
                assert body[SomaBodyInputIndex.JOINT_POSITIONS][76, 0] == 76.0
            assert session.step()["soma_body"].is_none
    evaluator.reset_mock()
    count, rate = soma_body_common.replay_info(path, "soma_body_demo", representation)
    assert count == 3
    assert rate == pytest.approx(10.0)
    viz = MagicMock()
    clock = [0.0]
    monkeypatch.setattr(
        common,
        "time",
        SimpleNamespace(
            monotonic=lambda: clock[0],
            sleep=lambda delay: clock.__setitem__(0, clock[0] + delay),
        ),
    )
    for _ in range(2):
        assert (
            common.replay_once(
                path,
                pipeline,
                count,
                lambda result: viz.update(result["soma_body"]),
                rate,
            )
            == 3
        )
    assert viz.update.call_count == 6
    assert clock[0] == pytest.approx(0.4)
    assert evaluator.evaluate.call_count == (
        4 if representation == "joint-rotations" else 0
    )
    if representation == "joint-rotations":
        assert evaluator.evaluate.call_args.args[0].to_bytes() == entries[-1].to_bytes()


def test_replay_rejects_missing_collection_or_representation(soma_mcap):
    path = soma_mcap(
        "joint-poses",
        [payload("joint-poses", [0])],
        [schema.DeviceDataTimestamp(1, 2, 3)],
    )
    for collection, representation in [
        ("wrong", "joint-poses"),
        ("soma_body_demo", "joint-rotations"),
    ]:
        with pytest.raises(ValueError, match="no samples"):
            soma_body_common.replay_info(path, collection, representation)
    count, rate = soma_body_common.replay_info(path, "soma_body_demo", "joint-poses")
    assert count == 1
    assert rate == 60.0


def test_resolve_only_matching_recordings(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    recordings = tmp_path / "recordings"
    recordings.mkdir()
    (recordings / "full_body_20261009.mcap").touch()
    (recordings / "soma_body_joint-rotations_20261009.mcap").touch()
    with pytest.raises(FileNotFoundError, match="No soma_body_joint-poses"):
        common.resolve_recording(None, "soma_body_joint-poses")
    expected = recordings / "soma_body_joint-poses_20261009.mcap"
    expected.touch()
    assert common.resolve_recording(None, "soma_body_joint-poses") == expected
    assert common.resolve_recording(str(expected), "soma_body_joint-poses") == expected
