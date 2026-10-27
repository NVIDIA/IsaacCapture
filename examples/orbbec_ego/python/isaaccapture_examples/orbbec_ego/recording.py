# SPDX-FileCopyrightText: Copyright (c) 2026 ORBBEC INC. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Finalize EGO media after the official session has released its resources."""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING
import subprocess

if TYPE_CHECKING:
    from collections.abc import Iterator

    from isaaccapture.teleop_session_manager import TeleopSession, TeleopSessionConfig

logger = logging.getLogger("isaaccapture.examples.orbbec_ego.recording")


def _option(args: list[str], name: str) -> str | None:
    for argument in reversed(args):
        if argument.startswith(name + "="):
            return argument[len(name) + 1 :]
    return None


def _fresh_paths(paths: tuple[Path, ...]) -> None:
    if len(set(paths)) != len(paths):
        raise ValueError("Recording, media fragment, and final output must be distinct")
    for path in paths:
        if not path.parent.is_dir():
            raise ValueError(
                f"Create the output directory before recording: {path.parent}"
            )
        for candidate in (path, Path(str(path) + ".partial")):
            if candidate.exists() or candidate.is_symlink():
                raise FileExistsError(
                    f"Refusing to overwrite recording output: {candidate}"
                )


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


@contextmanager
def embedded_recording(
    config: TeleopSessionConfig,
    *,
    media_fragment: str | os.PathLike[str],
    merger: str | os.PathLike[str] | None = None,
    record_device_diagnostics: bool | None = None,
) -> Iterator[TeleopSession]:
    """Record through TeleopSession, then atomically merge the EGO capture fragment.

    Configure one EGO plugin with collection-prefix, embedded media, and a matching
    mcap-media-spool. Catch KeyboardInterrupt inside the context for a clean stop.
    Failures preserve intermediate files and propagate to the caller.
    """
    from isaaccapture.deviceio_session import McapRecordingConfig
    from isaaccapture.plugin_manager import PluginManager
    from isaaccapture.teleop_session_manager import TeleopSession

    config = configure_recording(
        config, record_device_diagnostics=record_device_diagnostics
    )
    if not isinstance(config.mcap_config, McapRecordingConfig):
        raise ValueError("embedded_recording requires McapRecordingConfig")
    plugins = [
        plugin
        for plugin in config.plugins
        if plugin.enabled and plugin.plugin_name == "orbbec_ego_camera"
    ]
    if len(plugins) != 1:
        raise ValueError(
            "embedded_recording requires exactly one enabled orbbec_ego_camera plugin"
        )
    plugin = plugins[0]
    manager = PluginManager(
        [str(path) for path in plugin.search_paths if path.is_dir()]
    )
    info = manager.get_plugin_info(plugin.plugin_name)
    working_dir = Path(info.working_dir).resolve()
    arguments = list(info.args) + list(plugin.plugin_args)
    fragment = Path(media_fragment).resolve()
    spool = _option(arguments, "--mcap-media-spool")
    if not _option(arguments, "--collection-prefix"):
        raise ValueError("The EGO plugin requires --collection-prefix=PREFIX")
    if _option(arguments, "--mcap-media") != "embedded" or not spool:
        raise ValueError(
            "The EGO plugin requires embedded media and --mcap-media-spool=PATH"
        )
    if (working_dir / spool).resolve() != fragment:
        raise ValueError("media_fragment must match the EGO plugin's mcap-media-spool")
    if _option(arguments, "--mcap-filename"):
        raise ValueError(
            "The EGO plugin cannot use --mcap-filename with a live collection"
        )
    executable = (
        Path(merger).resolve() if merger is not None else working_dir / "ego_mcap_merge"
    )
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise FileNotFoundError(
            f"Build or install the EGO merge tool before capture: {executable}"
        )

    final = Path(config.mcap_config.filename).resolve()
    recording = final.with_name(final.stem + ".session" + final.suffix)
    _fresh_paths((recording, fragment, final))
    recording_config = McapRecordingConfig(
        str(recording), config.mcap_config.get_tracker_names()
    )
    session_config = replace(
        config,
        mcap_config=recording_config,
        plugins=[
            replace(item, required=True) if item is plugin else item
            for item in config.plugins
        ],
    )
    with TeleopSession(session_config) as session:
        yield session

    # Explicit Plugin.stop() clears exit status, so only a committed fragment proves completion.
    if not fragment.is_file() or Path(str(fragment) + ".partial").exists():
        raise RuntimeError(
            f"EGO capture did not commit a complete fragment: {fragment}"
        )
    subprocess.run(
        [
            str(executable),
            "--recording",
            str(recording),
            "--media",
            str(fragment),
            "--output",
            str(final),
        ],
        check=True,
    )
    if not final.is_file() or Path(str(final) + ".partial").exists():
        raise RuntimeError(f"The merge tool did not commit a complete output: {final}")
    logger.info("Finalized EGO recording: %s", final)
