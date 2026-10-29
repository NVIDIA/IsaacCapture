# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
from unittest.mock import MagicMock

import numpy as np
import pytest

from isaaccapture.deviceio_trackers import (
    SomaBodyJointPosesTracker,
    SomaBodyJointRotationsTracker,
)
from isaaccapture.retargeting_engine.interface.tensor_group import OptionalTensorGroup
from isaaccapture.retargeting_engine.interface.tensor_group_type import OptionalType
from isaaccapture.retargeting_engine.tensor_types import (
    SomaBodyInput,
    SomaBodyInputIndex,
)
from isaaccapture.deviceio import McapReplayConfig, ControllerTracker
from isaaccapture.teleop_session_manager import (
    SessionMode,
    TeleopSession,
    TeleopSessionConfig,
)
from isaaccapture_examples.mcap_record_replay import (
    common,
    live_full_body,
    record_full_body,
    record_hand,
    replay_full_body,
    soma_body_common,
)
from isaaccapture_examples.mcap_record_replay.common import (
    BODY_BONES,
    FullBodyViz,
    build_full_body_pipeline,
)


def body_group(valid):
    # OptionalType describes the shape; OptionalTensorGroup supplies absence state.
    group = OptionalTensorGroup(OptionalType(SomaBodyInput()))
    group[SomaBodyInputIndex.JOINT_POSITIONS] = np.arange(
        77 * 3, dtype=np.float32
    ).reshape(77, 3)
    group[SomaBodyInputIndex.JOINT_VALID] = np.asarray(valid, dtype=np.uint8)
    return group


def test_soma_renderer_keeps_native_topology_and_hides_unavailable_joints():
    server = MagicMock()
    server.scene.add_point_cloud.side_effect = lambda **kwargs: MagicMock(**kwargs)
    server.scene.add_line_segments.side_effect = lambda **kwargs: MagicMock(**kwargs)
    ground = MagicMock()
    viz = soma_body_common.SomaBodyViz(server, ground)
    assert len(soma_body_common.JOINT_NAMES) == 77
    assert len(soma_body_common.BONES) == 76
    group = body_group(np.ones(77))
    assert viz.update(group) == 77
    ground.follow.assert_called_once()
    assert viz.points.points.shape == (77, 3)
    assert viz.bones.points.shape == (76, 2, 3)
    np.testing.assert_array_equal(
        viz.bones.points[-1], group[SomaBodyInputIndex.JOINT_POSITIONS][[75, 76]]
    )
    valid = np.zeros(77)
    valid[[0, 1, 76]] = 1
    assert viz.update(body_group(valid)) == 3
    assert viz.points.points.shape == (3, 3)
    assert viz.bones.points.shape == (1, 2, 3)
    group.set_none()
    assert viz.update(group) == 0
    assert viz.points.points.shape == (0, 3)
    assert viz.bones.points.shape == (0, 2, 3)


def test_joint_pose_pipeline_needs_no_model(monkeypatch):
    layer = MagicMock(
        side_effect=AssertionError("evaluated poses must not load a model")
    )
    monkeypatch.setattr(soma_body_common, "create_layer", layer)
    pipeline = soma_body_common.build_pipeline("vendor.body", "joint-poses")
    sources = list(pipeline.get_leaf_nodes())
    assert len(sources) == 1
    assert sources[0].name == "vendor.body"
    assert isinstance(sources[0].get_tracker(), SomaBodyJointPosesTracker)
    assert isinstance(
        soma_body_common.create_tracker("vendor.body", "joint-rotations"),
        SomaBodyJointRotationsTracker,
    )
    layer.assert_not_called()


def test_existing_full_body_pipeline_and_renderer_stay_24_joint():
    sources = {source.name for source in build_full_body_pipeline().get_leaf_nodes()}
    assert sources == {"full_body", "controllers"}
    server = MagicMock()
    viz = FullBodyViz(server)
    positions = np.zeros((24, 3), dtype=np.float32)
    viz.update(positions, np.ones(24, dtype=np.uint8))
    assert viz.points.points.shape == (24, 3)
    assert viz.bones.points.shape == (len(BODY_BONES), 2, 3)


@pytest.mark.parametrize("module", [live_full_body, record_full_body, replay_full_body])
def test_cli_help_does_not_load_models_or_start_sessions(module, monkeypatch, capsys):
    monkeypatch.setattr(
        soma_body_common,
        "create_layer",
        MagicMock(side_effect=AssertionError("model loaded")),
    )
    with pytest.raises(SystemExit) as caught:
        module.main(["example", "--help"])
    assert caught.value.code == 0
    help_text = capsys.readouterr().out
    assert "--body-schema" in help_text
    assert "--soma-body-representation" in help_text
    assert "--soma-body-collection-id" in help_text


@pytest.mark.parametrize("module", [live_full_body, record_full_body, replay_full_body])
def test_body_cli_rejects_controller_collection_collision(
    module, monkeypatch, tmp_path, capsys
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        module,
        "build_body_pipeline",
        MagicMock(side_effect=AssertionError("invalid arguments reached the pipeline")),
    )
    if module is record_full_body:
        monkeypatch.setattr(
            module,
            "record_trackers",
            MagicMock(
                side_effect=AssertionError("invalid arguments reached recording")
            ),
        )
    with pytest.raises(SystemExit) as caught:
        module.main(
            [
                "example",
                "--body-schema",
                "soma",
                "--soma-body-collection-id",
                "controllers",
                "--controllers",
                "--soma-body-representation",
                "joint-poses",
            ]
        )
    assert caught.value.code == 2
    assert "reserved" in capsys.readouterr().err


def test_live_preview_has_no_mcap_config(monkeypatch):
    monkeypatch.setattr(live_full_body.viser, "ViserServer", MagicMock())
    monkeypatch.setattr(live_full_body.CloudXRLauncher, "launch_context", MagicMock())
    session = MagicMock()
    session.__enter__.return_value.step.side_effect = KeyboardInterrupt
    session_factory = MagicMock(return_value=session)
    monkeypatch.setattr(live_full_body, "TeleopSession", session_factory)
    assert (
        live_full_body.main(
            [
                "live",
                "--body-schema",
                "soma",
                "--soma-body-representation",
                "joint-poses",
            ]
        )
        == 0
    )
    assert session_factory.call_args.args[0].mcap_config is None


@pytest.mark.parametrize("representation", ["joint-rotations", "joint-poses"])
def test_recorder_registers_only_raw_tracker_and_closes_on_interrupt(
    representation, monkeypatch, tmp_path
):
    monkeypatch.setattr(
        soma_body_common,
        "create_layer",
        MagicMock(side_effect=AssertionError("recording must not run FK")),
    )
    launcher = MagicMock()
    monkeypatch.setattr(common.CloudXRLauncher, "launch_context", launcher)
    oxr_factory = MagicMock()
    monkeypatch.setattr(common, "OpenXRSession", oxr_factory)
    deviceio = MagicMock()
    deviceio.run.return_value.__enter__.return_value.update.side_effect = (
        KeyboardInterrupt
    )
    monkeypatch.setattr(common, "DeviceIOSession", deviceio)
    path = tmp_path / "body.mcap"
    assert (
        record_full_body.main(
            [
                "record",
                "5",
                str(path),
                "--body-schema",
                "soma",
                "--soma-body-collection-id",
                "vendor.body",
                "--soma-body-representation",
                representation,
            ]
        )
        == 0
    )
    trackers, handles, config = deviceio.run.call_args.args
    assert len(trackers) == 1
    tracker_type = (
        SomaBodyJointRotationsTracker
        if representation == "joint-rotations"
        else SomaBodyJointPosesTracker
    )
    assert isinstance(trackers[0], tracker_type)
    assert config.filename == str(path)
    assert config.get_tracker_names() == [(trackers[0], "vendor.body")]
    assert (
        handles
        == oxr_factory.return_value.__enter__.return_value.get_handles.return_value
    )
    deviceio.run.return_value.__exit__.assert_called_once()
    oxr_factory.return_value.__exit__.assert_called_once()
    launcher.return_value.__exit__.assert_called_once()


@pytest.mark.parametrize(
    "module,schema_flag",
    [(record_full_body, "--body-schema"), (record_hand, "--hand-schema")],
)
@pytest.mark.parametrize("stage", ["launcher", "openxr", "deviceio"])
def test_recorder_startup_interrupt_does_not_report_success(
    module, schema_flag, stage, monkeypatch, tmp_path, capsys
):
    launcher = MagicMock()
    oxr_factory = MagicMock()
    deviceio = MagicMock()
    monkeypatch.setattr(common.CloudXRLauncher, "launch_context", launcher)
    monkeypatch.setattr(common, "OpenXRSession", oxr_factory)
    monkeypatch.setattr(common, "DeviceIOSession", deviceio)
    context = {
        "launcher": launcher.return_value,
        "openxr": oxr_factory.return_value,
        "deviceio": deviceio.run.return_value,
    }[stage]
    context.__enter__.side_effect = KeyboardInterrupt
    path = tmp_path / "cancelled.mcap"

    with pytest.raises(KeyboardInterrupt):
        module.main(["record", "5", str(path), schema_flag, "soma"])

    assert "[record] done" not in capsys.readouterr().out
    assert not path.exists()


@pytest.mark.parametrize("duration", ["0", "-1", "nan", "inf"])
def test_recorder_rejects_invalid_duration(duration):
    with pytest.raises(SystemExit) as caught:
        record_full_body.main(["record", duration])
    assert caught.value.code == 2


@pytest.mark.parametrize(
    "flags,names",
    [
        ([], {"full_body", "controllers"}),
        (["--no-controllers"], {"full_body"}),
        (["--body-schema", "soma"], {"soma_body_demo"}),
        (
            ["--body-schema", "soma", "--soma-body-collection-id", "controllers"],
            {"controllers"},
        ),
        (["--body-schema", "soma", "--controllers"], {"soma_body_demo", "controllers"}),
    ],
)
def test_body_schema_selection_preserves_controller_defaults(flags, names):
    parser = argparse.ArgumentParser()
    common.add_body_arguments(parser)
    args = parser.parse_args([*flags, "--soma-body-representation", "joint-poses"])
    common.validate_body_arguments(parser, args)
    pipeline = common.build_body_pipeline(args)
    assert {source.name for source in pipeline.get_leaf_nodes()} == names


def test_soma_recorder_can_opt_in_to_controllers(monkeypatch, tmp_path):
    record = MagicMock()
    monkeypatch.setattr(record_full_body, "record_trackers", record)
    assert (
        record_full_body.main(
            [
                "record",
                "5",
                str(tmp_path / "body.mcap"),
                "--body-schema",
                "soma",
                "--controllers",
            ]
        )
        == 0
    )
    trackers = record.call_args.args[2]
    assert len(trackers) == 2
    assert isinstance(trackers[0][0], SomaBodyJointRotationsTracker)
    assert isinstance(trackers[1][0], ControllerTracker)
    assert trackers[1][1] == "controllers"


def test_default_body_recording_keeps_existing_source_names(
    monkeypatch, tmp_path, command_clock
):
    monkeypatch.setattr(record_full_body.CloudXRLauncher, "launch_context", MagicMock())
    session = MagicMock()
    session.__enter__.return_value.frame_count = 1
    factory = MagicMock(return_value=session)
    monkeypatch.setattr(record_full_body, "TeleopSession", factory)
    command_clock(record_full_body, [0, 0, 6])
    assert record_full_body.main(["record", "5", str(tmp_path / "body.mcap")]) == 0
    config = factory.call_args.args[0]
    assert {source.name for source in config.pipeline.get_leaf_nodes()} == {
        "full_body",
        "controllers",
    }
    assert config.mcap_config.filename == str(tmp_path / "body.mcap")


def test_default_body_native_replay_and_renderer(legacy_mcap):
    path = legacy_mcap("body")
    config = TeleopSessionConfig(
        app_name="FullBodyMcapRegression",
        pipeline=build_full_body_pipeline(),
        mode=SessionMode.REPLAY,
        mcap_config=McapReplayConfig(str(path)),
    )
    server = MagicMock()
    viz = FullBodyViz(server)
    args = argparse.Namespace(body_schema="full-body-pose")
    for _ in range(2):
        with TeleopSession(config) as session:
            for expected in [24, 0, 24, 0]:
                assert (
                    common.update_body_visualizer(args, session.step(), viz) == expected
                )
                assert viz.points.points.shape == (24, 3)
                if expected:
                    np.testing.assert_array_equal(viz.points.points[23], [69, 70, 71])
                else:
                    assert viz.bones.points.shape == (0, 2, 3)


def test_default_body_replay_keeps_duration_and_60_hz(
    monkeypatch, tmp_path, command_clock
):
    session = MagicMock()
    session.__enter__.return_value.frame_count = 1
    session.__enter__.return_value.step.return_value = {
        "full_body": MagicMock(is_none=True)
    }
    factory = MagicMock(return_value=session)
    monkeypatch.setattr(replay_full_body, "TeleopSession", factory)
    clock = command_clock(replay_full_body, [0, 0, 1])
    assert replay_full_body.run_once(tmp_path / "body.mcap", 0.5, MagicMock()) == 1
    clock.sleep.assert_called_once_with(1 / 60)
    config = factory.call_args.args[0]
    assert {source.name for source in config.pipeline.get_leaf_nodes()} == {
        "full_body",
        "controllers",
    }


def test_body_replay_cli_uses_native_soma_recording(soma_mcap, monkeypatch):
    from test_soma_mcap import payload
    from isaaccapture import schema

    path = soma_mcap(
        "joint-poses",
        [payload("joint-poses", [0, 1, 76]), None],
        [schema.DeviceDataTimestamp(1, 2, 3), schema.DeviceDataTimestamp(2, 3, 4)],
    )
    server = MagicMock()
    monkeypatch.setattr(replay_full_body.viser, "ViserServer", lambda **kwargs: server)
    frames = []
    real_replay = common.replay_once

    def once(path, pipeline, count, update, rate):
        def capture(result):
            frames.append(result["soma_body"].is_none)
            update(result)

        assert real_replay(path, pipeline, count, capture, rate) == 2
        raise KeyboardInterrupt

    monkeypatch.setattr(replay_full_body, "replay_once", once)
    assert (
        replay_full_body.main(
            [
                "replay",
                str(path),
                "--body-schema",
                "soma",
                "--soma-body-representation",
                "joint-poses",
                "--rate",
                "0",
            ]
        )
        == 0
    )
    assert frames == [False, True]
    server.stop.assert_called_once()
