# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
from mcap.reader import make_reader

from isaaccapture import schema
from isaaccapture.deviceio import McapReplayConfig, ReplaySession
from isaaccapture.teleop_session_manager import (
    SessionMode,
    TeleopSession,
    TeleopSessionConfig,
)
from isaaccapture.deviceio_trackers import (
    SomaHandJointPosesTracker,
    SomaHandJointRotationsTracker,
)
from isaaccapture.retargeting_engine.deviceio_source_nodes import soma_hand_source
from isaaccapture.retargeting_engine.interface.tensor_group import OptionalTensorGroup
from isaaccapture.retargeting_engine.interface.tensor_group_type import OptionalType
from isaaccapture.retargeting_engine.tensor_types import (
    SomaHandInput,
    SomaHandInputIndex,
)
from isaaccapture_examples.mcap_record_replay import (
    live_hand,
    record_hand,
    replay_hand,
    common,
    soma_hand_common,
)


def payload(representation, hand_side, indices=None, sign=1.0):
    indices = range(25) if indices is None else indices
    q = schema.Quaternion(0.0, sign * 0.6, 0.0, sign * 0.8)
    if representation == "joint-rotations":
        return schema.SomaHandJointRotations(
            [schema.SomaHandJointRotation(schema.SomaHandJoint(i), q) for i in indices],
            schema.Point(1.0, 2.0, 3.0),
            True,
            hand_side,
        )
    return schema.SomaHandJointPoses(
        [
            schema.SomaHandJointPose(
                schema.SomaHandJoint(i),
                schema.Pose(schema.Point(float(i), 2.0, 3.0), q),
            )
            for i in indices
        ],
        hand_side,
    )


def timestamps(count, start=1_000_000_000):
    return [
        schema.DeviceDataTimestamp(start + i * 100_000_000, 80 + i, 10 + i)
        for i in range(count)
    ]


@pytest.mark.parametrize("representation", ["joint-rotations", "joint-poses"])
def test_repeated_hand_roundtrip_preserves_sides_and_independent_ticks(
    soma_hand_mcap, tmp_path, representation
):
    sides = soma_hand_common.HAND_SIDES
    collection_ids = soma_hand_common.collections("vendor.left", "vendor.right")
    payloads = {
        "hand_left": [
            payload(representation, sides["hand_left"]),
            None,
            payload(representation, sides["hand_left"], [0, 1, 24], -1),
        ],
        "hand_right": [payload(representation, sides["hand_right"], [0, 5, 24], -1)],
    }
    times = {
        key: timestamps(
            len(values), start=1_000_000_000 if key == "hand_left" else 9_000_000_000
        )
        for key, values in payloads.items()
    }
    expected = {
        key: [None if data is None else data.to_bytes() for data in values]
        for key, values in payloads.items()
    }
    record_type = (
        schema.SomaHandJointRotationsRecord
        if representation == "joint-rotations"
        else schema.SomaHandJointPosesRecord
    )
    for generation in range(3):
        path = soma_hand_mcap(
            representation,
            payloads,
            times,
            collection_ids=collection_ids,
            path=tmp_path / f"hands_{generation}.mcap",
        )
        for key, values in payloads.items():
            topic = f"{collection_ids[key]}/soma_hand_{representation.replace('-', '_')}_tracked"
            with path.open("rb") as stream:
                records = list(make_reader(stream).iter_messages(topics=[topic]))
            for (_, _, message), data, stamp in zip(
                records, values, times[key], strict=True
            ):
                assert message.data == record_type(data, stamp).to_bytes()
                assert (
                    message.log_time
                    == message.publish_time
                    == stamp.available_time_local_common_clock
                )
        trackers = soma_hand_common.create_trackers(collection_ids, representation)
        replayed = {key: [] for key in trackers}
        config = McapReplayConfig(
            str(path),
            [(tracker, collection_ids[key]) for key, tracker in trackers.items()],
        )
        with ReplaySession.run(config) as session:
            for frame in range(3):
                session.update()
                for key, tracker in trackers.items():
                    data = tracker.get_data(session)
                    encoded = (
                        expected[key][frame] if frame < len(expected[key]) else None
                    )
                    assert (None if data is None else data.to_bytes()) == encoded
                    if data is not None:
                        assert data.hand_side == sides[key]
                    if frame < len(expected[key]):
                        replayed[key].append(data)
            session.update()
            assert all(
                tracker.get_data(session) is None for tracker in trackers.values()
            )
        payloads = replayed


@pytest.mark.parametrize("representation", ["joint-rotations", "joint-poses"])
@pytest.mark.parametrize("right_present", [True, False])
def test_hand_pipeline_repeats_with_missing_or_exhausted_hand(
    soma_hand_mcap, monkeypatch, representation, right_present
):
    collection_ids = soma_hand_common.collections(
        "soma_hand_left_demo", "soma_hand_right_demo"
    )
    values = {
        "hand_left": [
            payload(representation, schema.SomaHandSide.LEFT),
            None,
            payload(representation, schema.SomaHandSide.LEFT, [0, 1, 24]),
        ]
    }
    if right_present:
        values["hand_right"] = [payload(representation, schema.SomaHandSide.RIGHT)]
    times = {key: timestamps(len(data)) for key, data in values.items()}
    path = soma_hand_mcap(representation, values, times)
    evaluator = MagicMock()
    evaluator.evaluate.return_value = (
        np.ones((25, 3), dtype=np.float32),
        np.tile(np.array([0, 0, 0, 1], dtype=np.float32), (25, 1)),
        np.ones(25, dtype=np.uint8),
    )
    monkeypatch.setattr(soma_hand_source, "_SomaHandEvaluator", lambda *args: evaluator)
    monkeypatch.setattr(soma_hand_common, "create_layer", lambda _: object())
    pipeline = soma_hand_common.build_pipeline(collection_ids, representation)
    counts, rate = soma_hand_common.replay_info(path, collection_ids, representation)
    assert counts == {"hand_left": 3, "hand_right": int(right_present)}
    assert rate == pytest.approx(10)
    frames = {key: [] for key in collection_ids}

    def capture(key, group):
        frames[key].append(
            None
            if group.is_none
            else np.asarray(group[SomaHandInputIndex.JOINT_VALID]).copy()
        )

    visualizers = {
        key: MagicMock(update=lambda group, key=key: capture(key, group))
        for key in collection_ids
    }
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

        def update(result):
            for key, viz in visualizers.items():
                viz.update(result[key])

        assert (
            common.replay_once(path, pipeline, max(counts.values()), update, rate) == 3
        )
    assert clock[0] == pytest.approx(0.4)
    assert frames["hand_left"][1] is None and frames["hand_left"][4] is None
    assert np.count_nonzero(frames["hand_left"][0]) == 25
    if representation == "joint-poses":
        assert np.flatnonzero(frames["hand_left"][2]).tolist() == [0, 1, 24]
    for frame in (1, 2, 4, 5):
        assert frames["hand_right"][frame] is None
    assert (frames["hand_right"][0] is not None) == right_present
    assert evaluator.evaluate.call_count == (
        2 * (2 + right_present) if representation == "joint-rotations" else 0
    )


def test_hand_pose_pipeline_and_native_25_joint_renderer_need_no_model(monkeypatch):
    layer = MagicMock(side_effect=AssertionError("joint poses must not load models"))
    monkeypatch.setattr(soma_hand_common, "create_layer", layer)
    ids = soma_hand_common.collections("vendor.left", "vendor.right")
    sources = {
        source.name: source
        for source in soma_hand_common.build_pipeline(
            ids, "joint-poses"
        ).get_leaf_nodes()
    }
    assert set(sources) == set(ids.values())
    assert all(
        isinstance(source.get_tracker(), SomaHandJointPosesTracker)
        for source in sources.values()
    )
    assert sources["vendor.left"].hand_side == schema.SomaHandSide.LEFT
    assert sources["vendor.right"].hand_side == schema.SomaHandSide.RIGHT
    layer.assert_not_called()
    server = MagicMock()
    server.scene.add_point_cloud.side_effect = lambda **kwargs: MagicMock(**kwargs)
    server.scene.add_line_segments.side_effect = lambda **kwargs: MagicMock(**kwargs)
    visualizers = soma_hand_common.create_visualizers(server)
    group = OptionalTensorGroup(OptionalType(SomaHandInput()))
    group[SomaHandInputIndex.JOINT_POSITIONS] = np.arange(75, dtype=np.float32).reshape(
        25, 3
    )
    group[SomaHandInputIndex.JOINT_VALID] = np.ones(25, dtype=np.uint8)
    assert len(soma_hand_common.JOINT_NAMES) == 25
    for viz in visualizers.values():
        assert viz.update(group) == 25
        assert viz.points.points.shape == (25, 3)
        assert viz.bones.points.shape == (24, 2, 3)
    group[SomaHandInputIndex.JOINT_VALID] = np.array(
        [1, 1, *([0] * 23)], dtype=np.uint8
    )
    assert visualizers["hand_left"].update(group) == 2
    assert visualizers["hand_left"].bones.points.shape == (1, 2, 3)
    group.set_none()
    assert visualizers["hand_left"].update(group) == 0
    assert visualizers["hand_left"].points.points.shape == (0, 3)


@pytest.mark.parametrize("module", [live_hand, record_hand, replay_hand])
def test_hand_cli_help_does_not_load_models(module, monkeypatch, capsys):
    monkeypatch.setattr(
        soma_hand_common,
        "create_layer",
        MagicMock(side_effect=AssertionError("model loaded")),
    )
    with pytest.raises(SystemExit) as caught:
        module.main(["hands", "--help"])
    assert caught.value.code == 0
    help_text = capsys.readouterr().out
    assert "--hand-schema" in help_text
    assert "--soma-hand-representation" in help_text


@pytest.mark.parametrize("representation", ["joint-rotations", "joint-poses"])
def test_hand_recorder_registers_two_trackers_without_fk(
    representation, monkeypatch, tmp_path
):
    monkeypatch.setattr(
        soma_hand_common,
        "create_layer",
        MagicMock(side_effect=AssertionError("recording must not run FK")),
    )
    monkeypatch.setattr(common.CloudXRLauncher, "launch_context", MagicMock())
    monkeypatch.setattr(common, "OpenXRSession", MagicMock())
    deviceio = MagicMock()
    deviceio.run.return_value.__enter__.return_value.update.side_effect = (
        KeyboardInterrupt
    )
    monkeypatch.setattr(common, "DeviceIOSession", deviceio)
    path = tmp_path / "hands.mcap"
    assert (
        record_hand.main(
            [
                "record",
                "5",
                str(path),
                "--hand-schema",
                "soma",
                "--soma-hand-representation",
                representation,
            ]
        )
        == 0
    )
    trackers, _, config = deviceio.run.call_args.args
    assert len(trackers) == 2
    expected_type = (
        SomaHandJointRotationsTracker
        if representation == "joint-rotations"
        else SomaHandJointPosesTracker
    )
    assert all(isinstance(tracker, expected_type) for tracker in trackers)
    assert config.get_tracker_names() == list(
        zip(trackers, ["soma_hand_left_demo", "soma_hand_right_demo"], strict=True)
    )
    assert config.filename == str(path)
    deviceio.run.return_value.__exit__.assert_called_once()


def test_live_hand_preview_has_no_mcap_config(monkeypatch):
    monkeypatch.setattr(live_hand.viser, "ViserServer", MagicMock())
    monkeypatch.setattr(live_hand.CloudXRLauncher, "launch_context", MagicMock())
    session = MagicMock()
    session.__enter__.return_value.step.side_effect = KeyboardInterrupt
    factory = MagicMock(return_value=session)
    monkeypatch.setattr(live_hand, "TeleopSession", factory)
    assert (
        live_hand.main(
            [
                "live",
                "--hand-schema",
                "soma",
                "--soma-hand-representation",
                "joint-poses",
            ]
        )
        == 0
    )
    assert factory.call_args.args[0].mcap_config is None


def test_hand_replay_rejects_wrong_collections_or_profile(soma_hand_mcap):
    path = soma_hand_mcap(
        "joint-poses",
        {"hand_left": [payload("joint-poses", schema.SomaHandSide.LEFT)]},
        {"hand_left": timestamps(1)},
    )
    for ids, profile in [
        (soma_hand_common.collections("wrong.left", "wrong.right"), "joint-poses"),
        (
            soma_hand_common.collections("soma_hand_left_demo", "soma_hand_right_demo"),
            "joint-rotations",
        ),
    ]:
        with pytest.raises(ValueError, match="no SOMA hand samples"):
            soma_hand_common.replay_info(path, ids, profile)


def test_hand_replay_rejects_wrong_anatomical_side(soma_hand_mcap):
    path = soma_hand_mcap(
        "joint-poses",
        {"hand_left": [payload("joint-poses", schema.SomaHandSide.RIGHT)]},
        {"hand_left": timestamps(1)},
    )
    ids = soma_hand_common.collections("soma_hand_left_demo", "soma_hand_right_demo")
    pipeline = soma_hand_common.build_pipeline(ids, "joint-poses")
    with pytest.raises(ValueError, match="hand side"):
        common.replay_once(
            path,
            pipeline,
            1,
            MagicMock(),
            0,
        )


def test_hand_recording_resolution_does_not_select_body_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    recordings = tmp_path / "recordings"
    recordings.mkdir()
    (recordings / "soma_body_joint-poses_demo.mcap").touch()
    with pytest.raises(FileNotFoundError, match="No soma_hand_joint-poses"):
        common.resolve_recording(None, "soma_hand_joint-poses")
    expected = recordings / "soma_hand_joint-poses_demo.mcap"
    expected.touch()
    assert common.resolve_recording(None, "soma_hand_joint-poses") == expected


@pytest.mark.parametrize("module", [live_hand, record_hand, replay_hand])
def test_hand_commands_reject_duplicate_collections(module):
    with pytest.raises(SystemExit) as caught:
        module.main(
            [
                "hands",
                "--hand-schema",
                "soma",
                "--soma-left-hand-collection-id",
                "same",
                "--soma-right-hand-collection-id",
                "same",
            ]
        )
    assert caught.value.code == 2


def test_default_hand_native_replay_keeps_openxr_palm_and_26_joints(legacy_mcap):
    path = legacy_mcap("hand")
    config = TeleopSessionConfig(
        app_name="OpenXRHandMcapRegression",
        pipeline=common.build_hand_pipeline(),
        mode=SessionMode.REPLAY,
        mcap_config=McapReplayConfig(str(path)),
    )
    parser = argparse.ArgumentParser()
    common.add_hand_arguments(parser)
    args = parser.parse_args([])
    assert args.hand_schema == "openxr-hand-pose"
    visualizers = common.create_hand_visualizers(args, MagicMock())
    for _ in range(2):
        with TeleopSession(config) as session:
            for present in [True, False, True, False]:
                assert common.update_hand_visualizers(
                    args, session.step(), visualizers
                ) == [present, present]
                for viz in visualizers.values():
                    assert viz.points.points.shape == (26, 3)
                    if present:
                        np.testing.assert_array_equal(
                            viz.points.points[[0, 25]], [[0, 1, 2], [75, 76, 77]]
                        )


def test_default_hand_recording_keeps_existing_source_name(
    monkeypatch, tmp_path, command_clock
):
    monkeypatch.setattr(record_hand.CloudXRLauncher, "launch_context", MagicMock())
    session = MagicMock()
    session.__enter__.return_value.frame_count = 1
    factory = MagicMock(return_value=session)
    monkeypatch.setattr(record_hand, "TeleopSession", factory)
    command_clock(record_hand, [0, 0, 6])
    assert record_hand.main(["record", "5", str(tmp_path / "hand.mcap")]) == 0
    config = factory.call_args.args[0]
    assert {source.name for source in config.pipeline.get_leaf_nodes()} == {"hands"}
    assert config.mcap_config.filename == str(tmp_path / "hand.mcap")


def test_default_hand_replay_keeps_duration_and_60_hz(
    monkeypatch, tmp_path, command_clock
):
    session = MagicMock()
    session.__enter__.return_value.frame_count = 1
    session.__enter__.return_value.step.return_value = {
        "left_positions": [np.zeros((26, 3))],
        "right_positions": [np.zeros((26, 3))],
        "left_valid": [False],
        "right_valid": [False],
    }
    factory = MagicMock(return_value=session)
    monkeypatch.setattr(replay_hand, "TeleopSession", factory)
    clock = command_clock(replay_hand, [0, 0, 1])
    assert (
        replay_hand.run_once(tmp_path / "hand.mcap", 0.5, MagicMock(), MagicMock()) == 1
    )
    clock.sleep.assert_called_once_with(1 / 60)
    config = factory.call_args.args[0]
    assert {source.name for source in config.pipeline.get_leaf_nodes()} == {"hands"}


def test_hand_replay_cli_uses_native_soma_recording(soma_hand_mcap, monkeypatch):
    path = soma_hand_mcap(
        "joint-poses",
        {"hand_left": [payload("joint-poses", schema.SomaHandSide.LEFT), None]},
        {"hand_left": timestamps(2)},
    )
    server = MagicMock()
    monkeypatch.setattr(replay_hand.viser, "ViserServer", lambda **kwargs: server)
    frames = []
    real_replay = common.replay_once

    def once(path, pipeline, count, update, rate):
        def capture(result):
            frames.append([result[key].is_none for key in ("hand_left", "hand_right")])
            update(result)

        assert real_replay(path, pipeline, count, capture, rate) == 2
        raise KeyboardInterrupt

    monkeypatch.setattr(replay_hand, "replay_once", once)
    assert (
        replay_hand.main(
            [
                "replay",
                str(path),
                "--hand-schema",
                "soma",
                "--soma-hand-representation",
                "joint-poses",
                "--rate",
                "0",
            ]
        )
        == 0
    )
    assert frames == [[False, True], [True, True]]
    server.stop.assert_called_once()
