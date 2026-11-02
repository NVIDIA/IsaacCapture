# SPDX-FileCopyrightText: Copyright (c) 2026 ORBBEC INC. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare a TeleopSession configuration for EGO MCAP recording."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaccapture.teleop_session_manager import TeleopSessionConfig


def _option(args: list[str], name: str) -> str | None:
    for argument in reversed(args):
        if argument.startswith(name + "="):
            return argument[len(name) + 1 :]
    return None


def _with_calibration(config: TeleopSessionConfig, arguments: list[str]):
    from isaaccapture.deviceio_session import McapRecordingConfig
    from isaaccapture.deviceio_trackers import OrbbecEgoCalibrationTracker
    from isaaccapture.retargeting_engine.deviceio_source_nodes import IDeviceIOSource

    if not isinstance(config.mcap_config, McapRecordingConfig):
        return config.mcap_config, config.trackers
    entries = list(config.mcap_config.get_tracker_names())
    automatic = []
    for graph in (config.pipeline, config.teleop_control_pipeline, *config.sinks):
        if graph is not None:
            for source in graph.get_leaf_nodes():
                if isinstance(source, IDeviceIOSource):
                    automatic.append((source.get_tracker(), source.name))
    recorded = [
        tracker
        for tracker, _ in entries + automatic
        if isinstance(tracker, OrbbecEgoCalibrationTracker)
    ]
    if recorded:
        calibration = recorded[0]
    else:
        if any(name == "orbbec_ego_calibration" for _, name in entries + automatic):
            raise ValueError(
                "MCAP name 'orbbec_ego_calibration' is already used by another tracker"
            )
        manual = [
            tracker
            for tracker in config.trackers
            if isinstance(tracker, OrbbecEgoCalibrationTracker)
        ]
        if len(manual) > 1:
            raise ValueError(
                "Choose an explicit MCAP name for each EGO calibration tracker"
            )
        if manual:
            calibration = manual[0]
        else:
            prefix = _option(arguments, "--collection-prefix")
            if not prefix:
                raise ValueError(
                    "EGO MCAP recording requires --collection-prefix to record calibration"
                )
            calibration = OrbbecEgoCalibrationTracker(prefix + "/Calibration")
        entries.append((calibration, "orbbec_ego_calibration"))
    trackers = list(config.trackers)
    # McapRecordingConfig holds raw tracker pointers; the session config owns this reference.
    if all(tracker is not calibration for tracker in trackers):
        trackers.append(calibration)
    return McapRecordingConfig(config.mcap_config.filename, entries), trackers


def configure_recording(
    config: TeleopSessionConfig,
    *,
    record_device_diagnostics: bool | None = None,
) -> TeleopSessionConfig:
    """Copy a live config, retain calibration, and validate diagnostic recording.

    Explicit choice overrides plugin arguments, then the default is on. Off
    rejects diagnostic trackers recorded explicitly or through any source graph;
    manual live-only trackers remain usable. No session or device is opened.
    """
    from isaaccapture.deviceio_session import McapRecordingConfig
    from isaaccapture.deviceio_trackers import OrbbecEgoDeviceStateTracker
    from isaaccapture.plugin_manager import PluginManager
    from isaaccapture.retargeting_engine.deviceio_source_nodes import IDeviceIOSource
    from isaaccapture.teleop_session_manager import SessionMode

    if config.mode != SessionMode.LIVE:
        raise ValueError("configure_recording requires a live session")
    if (
        record_device_diagnostics is not None
        and type(record_device_diagnostics) is not bool
    ):
        raise TypeError("record_device_diagnostics must be bool or None")
    plugins = [
        item
        for item in config.plugins
        if item.enabled and item.plugin_name == "orbbec_ego_camera"
    ]
    if len(plugins) != 1:
        raise ValueError(
            "configure_recording requires exactly one enabled orbbec_ego_camera plugin"
        )
    plugin = plugins[0]
    manager = PluginManager(
        [str(path) for path in plugin.search_paths if path.is_dir()]
    )
    info = manager.get_plugin_info(plugin.plugin_name)
    arguments = list(info.args) + list(plugin.plugin_args)
    flag = "--record-device-diagnostics"
    value = _option(arguments, flag)
    if record_device_diagnostics is None:
        if value not in (None, "on", "off", "true", "false", "1", "0"):
            raise ValueError(f"{flag} must be on or off")
        record_device_diagnostics = value not in ("off", "false", "0")
    if not record_device_diagnostics and isinstance(
        config.mcap_config, McapRecordingConfig
    ):
        for tracker, name in config.mcap_config.get_tracker_names():
            if isinstance(tracker, OrbbecEgoDeviceStateTracker):
                raise ValueError(
                    f"Diagnostic recording is off but MCAP tracker {name!r} records OrbbecEgoDeviceState"
                )
        # Match NV's public source graph roots, without altering session discovery.
        roots = [config.pipeline, config.teleop_control_pipeline, *config.sinks]
        for graph in roots:
            if graph is None:
                continue
            for source in graph.get_leaf_nodes():
                if isinstance(source, IDeviceIOSource) and isinstance(
                    source.get_tracker(), OrbbecEgoDeviceStateTracker
                ):
                    raise ValueError(
                        f"Diagnostic recording is off but source {source.name!r} is automatically recorded by NV"
                    )
    args = [arg for arg in plugin.plugin_args if not arg.startswith(flag + "=")]
    args.append(f"{flag}={'on' if record_device_diagnostics else 'off'}")
    mcap_config, trackers = _with_calibration(config, arguments)
    return replace(
        config,
        mcap_config=mcap_config,
        trackers=trackers,
        plugins=[
            replace(item, plugin_args=args) if item is plugin else item
            for item in config.plugins
        ],
    )
