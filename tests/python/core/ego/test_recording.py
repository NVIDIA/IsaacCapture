# SPDX-FileCopyrightText: Copyright (c) 2026 ORBBEC INC. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from types import SimpleNamespace
import subprocess

import pytest

from isaaccapture import plugin_manager, teleop_session_manager
from isaaccapture.deviceio_session import McapRecordingConfig
from isaaccapture.ego import configure_recording, embedded_recording
from isaaccapture.ego import recording
from isaaccapture.teleop_session_manager import (
    PluginConfig,
    SessionMode,
    TeleopSessionConfig,
)


@pytest.fixture
def setup(tmp_path, monkeypatch):
    fragment = tmp_path / "capture.mcap"
    final = tmp_path / "final.mcap"
    executable = tmp_path / "ego_mcap_merge"
    executable.write_text("merge fixture")
    executable.chmod(0o755)
    plugin = PluginConfig(
        "ego_camera",
        "/camera/ego",
        [tmp_path],
        plugin_args=[
            "--collection-prefix=ego",
            "--mcap-media=embedded",
            f"--mcap-media-spool={fragment}",
        ],
    )
    config = TeleopSessionConfig(
        "test",
        SimpleNamespace(get_leaf_nodes=list),
        plugins=[plugin],
        mcap_config=McapRecordingConfig(str(final)),
    )
    events = []
    state = SimpleNamespace(
        config=config,
        fragment=fragment,
        final=final,
        executable=executable,
        events=events,
        finish="formal",
        cleanup_error=False,
        merger_error=False,
        session_config=None,
        command=None,
    )
    info = SimpleNamespace(working_dir=str(tmp_path), args=[])
    monkeypatch.setattr(
        plugin_manager,
        "PluginManager",
        lambda paths: SimpleNamespace(get_plugin_info=lambda name: info),
    )

    class Session:
        def __init__(self, session_config):
            state.session_config = session_config

        def __enter__(self):
            events.append("enter")
            Path(state.session_config.mcap_config.filename).write_bytes(
                b"session fixture"
            )
            return self

        def __exit__(self, *exception):
            events.append("exit")
            if state.finish == "formal":
                fragment.write_bytes(b"capture fixture")
            elif state.finish == "partial":
                Path(str(fragment) + ".partial").write_bytes(b"incomplete fixture")
            if state.cleanup_error:
                raise OSError("cleanup failed")
            return False

    def merge(command, *, check):
        assert events[-1] == "exit"
        assert check
        events.append("merge")
        state.command = command
        if state.merger_error:
            Path(str(final) + ".partial").write_bytes(b"partial merge")
            raise subprocess.CalledProcessError(1, command)
        final.write_bytes(b"merged fixture")

    monkeypatch.setattr(teleop_session_manager, "TeleopSession", Session)
    monkeypatch.setattr(recording.subprocess, "run", merge)
    return state


def test_merge_follows_complete_session_cleanup_without_mutating_config(setup):
    original = setup.config.mcap_config
    with embedded_recording(setup.config, media_fragment=setup.fragment) as session:
        assert session is not None
        setup.events.append("body")
    assert setup.events == ["enter", "body", "exit", "merge"]
    assert setup.config.mcap_config is original
    assert setup.config.mcap_config.filename == str(setup.final)
    assert not setup.config.plugins[0].required
    assert setup.session_config.plugins[0].required
    assert setup.command == [
        str(setup.executable),
        "--recording",
        str(setup.final.with_name("final.session.mcap")),
        "--media",
        str(setup.fragment),
        "--output",
        str(setup.final),
    ]
    assert setup.fragment.exists()
    assert setup.final.with_name("final.session.mcap").exists()


def test_body_error_propagates_and_keeps_inputs_without_merging(setup):
    with pytest.raises(RuntimeError, match="body failed"):
        with embedded_recording(setup.config, media_fragment=setup.fragment):
            raise RuntimeError("body failed")
    assert setup.events == ["enter", "exit"]
    assert not setup.final.exists()
    assert setup.fragment.exists()


def test_cleanup_error_is_not_reported_as_complete_recording(setup):
    setup.cleanup_error = True
    with pytest.raises(OSError, match="cleanup failed"):
        with embedded_recording(setup.config, media_fragment=setup.fragment):
            pass
    assert setup.events == ["enter", "exit"]
    assert not setup.final.exists()


@pytest.mark.parametrize("finish", ["partial", "missing"])
def test_stop_without_committed_fragment_does_not_merge(setup, finish):
    setup.finish = finish
    with pytest.raises(RuntimeError, match="did not commit"):
        with embedded_recording(setup.config, media_fragment=setup.fragment):
            pass
    assert setup.events == ["enter", "exit"]
    assert not setup.final.exists()


def test_merge_failure_propagates_and_retains_evidence(setup):
    setup.merger_error = True
    with pytest.raises(subprocess.CalledProcessError):
        with embedded_recording(setup.config, media_fragment=setup.fragment):
            pass
    assert setup.fragment.exists()
    assert setup.final.with_name("final.session.mcap").exists()
    assert Path(str(setup.final) + ".partial").exists()
    assert not setup.final.exists()


@pytest.mark.parametrize(
    "invalid",
    [
        "replay",
        "no_recording",
        "duplicate",
        "spool",
        "mode",
        "missing_merger",
        "existing_output",
    ],
)
def test_invalid_recording_fails_before_capture(setup, invalid):
    if invalid == "replay":
        setup.config.mode = SessionMode.REPLAY
    elif invalid == "no_recording":
        setup.config.mcap_config = None
    elif invalid == "duplicate":
        setup.config.plugins.append(setup.config.plugins[0])
    elif invalid == "spool":
        setup.config.plugins[0].plugin_args[-1] = "--mcap-media-spool=other.mcap"
    elif invalid == "mode":
        setup.config.plugins[0].plugin_args[1] = "--mcap-media=metadata-only"
    elif invalid == "missing_merger":
        setup.executable.unlink()
    elif invalid == "existing_output":
        setup.final.write_bytes(b"keep")
    with pytest.raises((ValueError, FileNotFoundError, FileExistsError)):
        with embedded_recording(setup.config, media_fragment=setup.fragment):
            pytest.fail("Capture must not start")
    assert not setup.events
    if invalid == "existing_output":
        assert setup.final.read_bytes() == b"keep"


def test_caught_interrupt_is_a_normal_stop(setup):
    with embedded_recording(setup.config, media_fragment=setup.fragment):
        try:
            raise KeyboardInterrupt
        except KeyboardInterrupt:
            pass
    assert setup.events == ["enter", "exit", "merge"]


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
    from isaaccapture.deviceio_trackers import EgoDeviceStateTracker

    tracker = EgoDeviceStateTracker("arbitrary/DeviceState")
    setup.config.mcap_config = McapRecordingConfig(
        str(setup.final), [(tracker, "renamed_diagnostics")]
    )
    with pytest.raises(ValueError, match="renamed_diagnostics"):
        with embedded_recording(
            setup.config, media_fragment=setup.fragment, record_device_diagnostics=False
        ):
            pytest.fail("Must reject before capture")
    assert not setup.events


@pytest.mark.parametrize("root", ["pipeline", "teleop_control_pipeline", "sinks"])
def test_off_rejects_nv_automatic_recording_in_every_graph_root(
    setup, monkeypatch, root
):
    from isaaccapture.deviceio_trackers import EgoDeviceStateTracker
    from isaaccapture.retargeting_engine import deviceio_source_nodes

    class Source:
        name = "diagnostic_side_effect"

        def get_tracker(self):
            return EgoDeviceStateTracker("ego/DeviceState")

    monkeypatch.setattr(deviceio_source_nodes, "IDeviceIOSource", Source)
    graph = SimpleNamespace(get_leaf_nodes=lambda: [Source()])
    setattr(setup.config, root, [graph] if root == "sinks" else graph)
    with pytest.raises(ValueError, match="automatically recorded"):
        configure_recording(setup.config, record_device_diagnostics=False)
    assert not setup.events


def test_off_keeps_manual_live_tracker_and_embedded_recording_lifecycle(setup):
    from isaaccapture.deviceio_trackers import (
        EgoCalibrationTracker,
        EgoDeviceStateTracker,
    )

    tracker = EgoDeviceStateTracker("ego/DeviceState")
    setup.config.trackers = [tracker]
    with embedded_recording(
        setup.config, media_fragment=setup.fragment, record_device_diagnostics=False
    ):
        assert setup.session_config.trackers[0] is tracker
        assert isinstance(setup.session_config.trackers[1], EgoCalibrationTracker)
        assert setup.config.trackers == [tracker]
        assert (
            setup.session_config.plugins[0].plugin_args[-1]
            == "--record-device-diagnostics=off"
        )
    assert setup.events == ["enter", "exit", "merge"]


def test_recording_retains_calibration_without_mutating_caller(setup):
    from isaaccapture.deviceio_trackers import EgoCalibrationTracker

    copied = configure_recording(setup.config)
    entries = copied.mcap_config.get_tracker_names()
    assert len(entries) == 1
    calibration, name = entries[0]
    assert isinstance(calibration, EgoCalibrationTracker)
    assert name == "ego_calibration"
    assert copied.trackers == [calibration]
    assert copied.mcap_config.filename == setup.config.mcap_config.filename
    assert setup.config.mcap_config.get_tracker_names() == []
    assert setup.config.trackers == []
    repeated = configure_recording(copied)
    assert repeated.mcap_config.get_tracker_names() == entries
    assert repeated.trackers == [calibration]
    assert not setup.events


def test_recording_preserves_explicit_calibration_name_and_reference(setup):
    from isaaccapture.deviceio_trackers import EgoCalibrationTracker

    calibration = EgoCalibrationTracker("ego/Calibration")
    setup.config.mcap_config = McapRecordingConfig(
        str(setup.final), [(calibration, "custom_calibration")]
    )
    copied = configure_recording(setup.config)
    assert copied.mcap_config.get_tracker_names() == [
        (calibration, "custom_calibration")
    ]
    assert copied.trackers == [calibration]


def test_recording_reuses_manual_calibration(setup):
    from isaaccapture.deviceio_trackers import EgoCalibrationTracker

    calibration = EgoCalibrationTracker("ego/Calibration")
    setup.config.trackers = [calibration]
    copied = configure_recording(setup.config)
    assert copied.mcap_config.get_tracker_names() == [(calibration, "ego_calibration")]
    assert copied.trackers == [calibration]


def test_recording_does_not_duplicate_automatically_recorded_calibration(
    setup, monkeypatch
):
    from isaaccapture.deviceio_trackers import EgoCalibrationTracker
    from isaaccapture.retargeting_engine import deviceio_source_nodes

    calibration = EgoCalibrationTracker("ego/Calibration")

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
    from isaaccapture.deviceio_trackers import EgoDeviceStateTracker

    tracker = EgoDeviceStateTracker("ego/DeviceState")
    setup.config.mcap_config = McapRecordingConfig(
        str(setup.final), [(tracker, "ego_calibration")]
    )
    with pytest.raises(ValueError, match="already used"):
        configure_recording(setup.config)
    assert not setup.events


def test_recording_requires_collection_prefix_for_automatic_calibration(setup):
    setup.config.plugins[0].plugin_args = [
        item
        for item in setup.config.plugins[0].plugin_args
        if not item.startswith("--collection-prefix=")
    ]
    with pytest.raises(ValueError, match="collection-prefix"):
        configure_recording(setup.config)
    assert not setup.events


def test_configure_without_mcap_does_not_add_calibration(setup):
    setup.config.mcap_config = None
    copied = configure_recording(setup.config)
    assert copied.mcap_config is None
    assert copied.trackers == []
