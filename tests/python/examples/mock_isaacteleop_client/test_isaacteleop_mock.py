# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the isaacteleop mock shim, mirroring GumanIsaacTeleopUsage.md's usage shape."""

import numpy as np

from isaacteleop.cloudxr import CloudXRLauncher
from isaacteleop.haptic_devices.glove import haptic_glove_device
from isaacteleop.retargeters.tactile_retargeters import TactileVectorToFingerPower
from isaacteleop.retargeting_engine.deviceio_source_nodes import (
    ControllersSource,
    HandsSource,
    HeadSource,
    JointStateSource,
)
from isaacteleop.retargeting_engine.interface import OutputCombiner
from isaacteleop.retargeting_engine.tensor_types import (
    ControllerInputIndex,
    HandInputIndex,
    HeadPoseIndex,
)
from isaacteleop.teleop_session_manager import (
    PluginConfig,
    TeleopSession,
    TeleopSessionConfig,
)


def _build_pipeline(
    *, with_hands: bool = True, with_joint_state: bool = True
) -> OutputCombiner:
    controllers = ControllersSource(name="controllers")
    head = HeadSource(name="head")
    outputs = {
        "controller_left": controllers.output(ControllersSource.LEFT),
        "controller_right": controllers.output(ControllersSource.RIGHT),
        "head": head.output("head"),
    }
    if with_hands:
        hands = HandsSource(name="hands")
        outputs["hand_left"] = hands.output(HandsSource.LEFT)
        outputs["hand_right"] = hands.output(HandsSource.RIGHT)
    if with_joint_state:
        source = JointStateSource(
            name="manus_sensors_left",
            collection_id="manus_sensors_left",
            joint_names=[f"joint_{i}" for i in range(35)],
        )
        outputs["manus_sensors_left"] = source.output(JointStateSource.JOINTS)
    return OutputCombiner(outputs)


def test_all_guman_imports_succeed():
    # Import success is the test; failing here means an import in
    # GumanIsaacTeleopUsage.md section 2 is missing from this shim.
    assert CloudXRLauncher is not None
    assert haptic_glove_device is not None
    assert TactileVectorToFingerPower is not None


def test_cloudxr_launcher_accepts_gumans_kwargs_and_stops():
    launcher = CloudXRLauncher(
        install_dir="/tmp/x",
        env_config=None,
        accept_eula=True,
        setup_oob=True,
        usb_local=True,
        run_embedded=True,
    )
    launcher.stop()  # must not raise


def test_plugin_config_accepted_without_launching_anything():
    plugin = PluginConfig(
        plugin_name="manus_hand_plugin",
        plugin_root_id="manus_hand_plugin",
        search_paths=[],
        plugin_args=["--datasets=human,sensors,haptic"],
    )
    config = TeleopSessionConfig(
        app_name="test", pipeline=_build_pipeline(), plugins=[plugin]
    )
    with TeleopSession(config) as session:
        session.step()  # must not crash, no process ever launched


def test_step_returns_dict_keyed_by_output_combiner_keys():
    pipeline = _build_pipeline()
    config = TeleopSessionConfig(app_name="test", pipeline=pipeline)
    with TeleopSession(config) as session:
        result = session.step()
        assert set(result.keys()) == set(pipeline.output_names)


def test_result_get_returns_none_for_unbuilt_output():
    pipeline = _build_pipeline(with_hands=False)
    config = TeleopSessionConfig(app_name="test", pipeline=pipeline)
    with TeleopSession(config) as session:
        result = session.step()
        assert result.get("hand_left") is None


def test_head_output_shape_and_validity():
    config = TeleopSessionConfig(app_name="test", pipeline=_build_pipeline())
    with TeleopSession(config) as session:
        head = session.step()["head"]
        assert not head.is_none
        assert bool(head[HeadPoseIndex.IS_VALID])
        assert np.asarray(head[HeadPoseIndex.POSITION]).shape == (3,)
        assert np.asarray(head[HeadPoseIndex.ORIENTATION]).shape == (4,)


def test_controller_output_shape_and_trigger_range():
    config = TeleopSessionConfig(app_name="test", pipeline=_build_pipeline())
    with TeleopSession(config) as session:
        result = session.step()
        for key in ("controller_left", "controller_right"):
            controller = result[key]
            assert not controller.is_none
            assert 0.0 <= controller[ControllerInputIndex.TRIGGER_VALUE] <= 1.0
            assert bool(controller[ControllerInputIndex.AIM_IS_VALID])


def test_hand_output_shape():
    config = TeleopSessionConfig(app_name="test", pipeline=_build_pipeline())
    with TeleopSession(config) as session:
        result = session.step()
        for key in ("hand_left", "hand_right"):
            hand = result[key]
            assert not hand.is_none
            positions = np.asarray(hand[HandInputIndex.JOINT_POSITIONS])
            assert positions.shape == (26, 3)
            assert np.asarray(hand[HandInputIndex.JOINT_VALID]).shape == (26,)


def test_joint_state_output_is_positional_and_matches_joint_count():
    config = TeleopSessionConfig(app_name="test", pipeline=_build_pipeline())
    with TeleopSession(config) as session:
        group = session.step()["manus_sensors_left"]
        assert not group.is_none
        values = [group[i] for i in range(35)]
        assert len(values) == 35


def test_haptic_sink_accepts_tactile_without_raising():
    device = haptic_glove_device()
    device.set_tactile(np.zeros(5, dtype=np.float32))
    retargeter = TactileVectorToFingerPower()
    retargeter.set_tactile(np.ones(5, dtype=np.float32))


def test_step_called_multiple_times_including_final_teardown_step():
    config = TeleopSessionConfig(app_name="test", pipeline=_build_pipeline())
    with TeleopSession(config) as session:
        session.step()
        session.step()
        session.step()  # mirrors the extra pre-teardown step in close()
