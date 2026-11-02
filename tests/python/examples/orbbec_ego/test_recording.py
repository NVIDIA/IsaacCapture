# SPDX-FileCopyrightText: Copyright (c) 2026 ORBBEC INC. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from types import SimpleNamespace

import pytest

from isaaccapture import plugin_manager
from isaaccapture.deviceio_session import McapRecordingConfig
from isaaccapture_examples.orbbec_ego import configure_recording
from isaaccapture.teleop_session_manager import (
    PluginConfig,
    SessionMode,
    TeleopSessionConfig,
)


def test_vendor_plugin_is_discovered_in_its_directory(monkeypatch):
    for parent in Path(__file__).resolve().parents:
        if (parent / "repo_paths.py").is_file():
            monkeypatch.syspath_prepend(str(parent))
            break
    from repo_paths import repo_root

    plugins = repo_root() / "src/plugins"
    manager = plugin_manager.PluginManager([str(plugins)])
    assert "orbbec_ego_camera" in manager.get_plugin_names()
    assert "ego_camera" not in manager.get_plugin_names()
    info = manager.get_plugin_info("orbbec_ego_camera")
    assert Path(info.working_dir).resolve() == plugins / "orbbec_ego"
    assert info.command == "./orbbec_ego_camera_plugin"
    assert [device.path for device in info.devices] == ["/camera/ego"]


@pytest.fixture
def setup(tmp_path, monkeypatch):
    final = tmp_path / "final.mcap"
    plugin = PluginConfig(
        "orbbec_ego_camera",
        "/camera/ego",
        [tmp_path],
        plugin_args=["--collection-prefix=ego"],
    )
    config = TeleopSessionConfig(
        "test",
        SimpleNamespace(get_leaf_nodes=list),
        plugins=[plugin],
        mcap_config=McapRecordingConfig(str(final)),
    )
    info = SimpleNamespace(working_dir=str(tmp_path), args=[])
    monkeypatch.setattr(
        plugin_manager,
        "PluginManager",
        lambda paths: SimpleNamespace(get_plugin_info=lambda name: info),
    )
    return SimpleNamespace(config=config, final=final)


@pytest.mark.parametrize("invalid", ["replay", "duplicate"])
def test_configure_recording_rejects_invalid_session(setup, invalid):
    if invalid == "replay":
        setup.config.mode = SessionMode.REPLAY
    elif invalid == "duplicate":
        setup.config.plugins.append(setup.config.plugins[0])
    with pytest.raises(ValueError):
        configure_recording(setup.config)


@pytest.mark.parametrize("value", [True, False])
def test_configure_recording_normalizes_policy_without_changing_caller(setup, value):
    original = list(setup.config.plugins[0].plugin_args)
    setup.config.plugins[0].plugin_args += [
        "--record-device-diagnostics=off",
        "--record-device-diagnostics=on",
    ]
    copied = configure_recording(setup.config, record_device_diagnostics=value)
    assert copied is not setup.config
    assert copied.plugins[0] is not setup.config.plugins[0]
    assert copied.plugins[0].plugin_args == original + [
        f"--record-device-diagnostics={'on' if value else 'off'}"
    ]
    assert setup.config.plugins[0].plugin_args[-1] == "--record-device-diagnostics=on"


def test_configure_recording_honors_effective_plugin_arguments(setup):
    setup.config.plugins[0].plugin_args += ["--record-device-diagnostics=off"]
    assert configure_recording(setup.config).plugins[0].plugin_args[-1].endswith("=off")
    setup.config.plugins[0].plugin_args[-1] = "--record-device-diagnostics=invalid"
    with pytest.raises(ValueError, match="on or off"):
        configure_recording(setup.config)


def test_off_rejects_explicit_diagnostic_recording_even_with_custom_topic(setup):
    from isaaccapture.deviceio_trackers import OrbbecEgoDeviceStateTracker

    tracker = OrbbecEgoDeviceStateTracker("arbitrary/DeviceState")
    setup.config.mcap_config = McapRecordingConfig(
        str(setup.final), [(tracker, "renamed_diagnostics")]
    )
    with pytest.raises(ValueError, match="renamed_diagnostics"):
        configure_recording(setup.config, record_device_diagnostics=False)


@pytest.mark.parametrize("root", ["pipeline", "teleop_control_pipeline", "sinks"])
def test_off_rejects_nv_automatic_recording_in_every_graph_root(
    setup, monkeypatch, root
):
    from isaaccapture.deviceio_trackers import OrbbecEgoDeviceStateTracker
    from isaaccapture.retargeting_engine import deviceio_source_nodes

    class Source:
        name = "diagnostic_side_effect"

        def get_tracker(self):
            return OrbbecEgoDeviceStateTracker("ego/DeviceState")

    monkeypatch.setattr(deviceio_source_nodes, "IDeviceIOSource", Source)
    graph = SimpleNamespace(get_leaf_nodes=lambda: [Source()])
    setattr(setup.config, root, [graph] if root == "sinks" else graph)
    with pytest.raises(ValueError, match="automatically recorded"):
        configure_recording(setup.config, record_device_diagnostics=False)


def test_off_keeps_manual_live_tracker(setup):
    from isaaccapture.deviceio_trackers import (
        OrbbecEgoCalibrationTracker,
        OrbbecEgoDeviceStateTracker,
    )

    tracker = OrbbecEgoDeviceStateTracker("ego/DeviceState")
    setup.config.trackers = [tracker]
    copied = configure_recording(setup.config, record_device_diagnostics=False)
    assert copied.trackers[0] is tracker
    assert isinstance(copied.trackers[1], OrbbecEgoCalibrationTracker)
    assert setup.config.trackers == [tracker]
    assert copied.plugins[0].plugin_args[-1] == "--record-device-diagnostics=off"


def test_recording_retains_calibration_without_mutating_caller(setup):
    from isaaccapture.deviceio_trackers import OrbbecEgoCalibrationTracker

    copied = configure_recording(setup.config)
    entries = copied.mcap_config.get_tracker_names()
    assert len(entries) == 1
    calibration, name = entries[0]
    assert isinstance(calibration, OrbbecEgoCalibrationTracker)
    assert name == "orbbec_ego_calibration"
    assert copied.trackers == [calibration]
    assert copied.mcap_config.filename == setup.config.mcap_config.filename
    assert setup.config.mcap_config.get_tracker_names() == []
    assert setup.config.trackers == []
    repeated = configure_recording(copied)
    assert repeated.mcap_config.get_tracker_names() == entries
    assert repeated.trackers == [calibration]


def test_recording_preserves_explicit_calibration_name_and_reference(setup):
    from isaaccapture.deviceio_trackers import OrbbecEgoCalibrationTracker

    calibration = OrbbecEgoCalibrationTracker("ego/Calibration")
    setup.config.mcap_config = McapRecordingConfig(
        str(setup.final), [(calibration, "custom_calibration")]
    )
    copied = configure_recording(setup.config)
    assert copied.mcap_config.get_tracker_names() == [
        (calibration, "custom_calibration")
    ]
    assert copied.trackers == [calibration]


def test_recording_reuses_manual_calibration(setup):
    from isaaccapture.deviceio_trackers import OrbbecEgoCalibrationTracker

    calibration = OrbbecEgoCalibrationTracker("ego/Calibration")
    setup.config.trackers = [calibration]
    copied = configure_recording(setup.config)
    assert copied.mcap_config.get_tracker_names() == [
        (calibration, "orbbec_ego_calibration")
    ]
    assert copied.trackers == [calibration]


def test_recording_does_not_duplicate_automatically_recorded_calibration(
    setup, monkeypatch
):
    from isaaccapture.deviceio_trackers import OrbbecEgoCalibrationTracker
    from isaaccapture.retargeting_engine import deviceio_source_nodes

    calibration = OrbbecEgoCalibrationTracker("ego/Calibration")

    class Source:
        name = "custom_calibration"

        def get_tracker(self):
            return calibration

    monkeypatch.setattr(deviceio_source_nodes, "IDeviceIOSource", Source)
    setup.config.pipeline = SimpleNamespace(get_leaf_nodes=lambda: [Source()])
    copied = configure_recording(setup.config)
    assert copied.mcap_config.get_tracker_names() == []
    assert copied.trackers == [calibration]


def test_recording_rejects_calibration_name_collision_before_capture(setup):
    from isaaccapture.deviceio_trackers import OrbbecEgoDeviceStateTracker

    tracker = OrbbecEgoDeviceStateTracker("ego/DeviceState")
    setup.config.mcap_config = McapRecordingConfig(
        str(setup.final), [(tracker, "orbbec_ego_calibration")]
    )
    with pytest.raises(ValueError, match="already used"):
        configure_recording(setup.config)


def test_recording_requires_collection_prefix_for_automatic_calibration(setup):
    setup.config.plugins[0].plugin_args = [
        item
        for item in setup.config.plugins[0].plugin_args
        if not item.startswith("--collection-prefix=")
    ]
    with pytest.raises(ValueError, match="collection-prefix"):
        configure_recording(setup.config)


def test_configure_without_mcap_does_not_add_calibration(setup):
    setup.config.mcap_config = None
    copied = configure_recording(setup.config)
    assert copied.mcap_config is None
    assert copied.trackers == []
