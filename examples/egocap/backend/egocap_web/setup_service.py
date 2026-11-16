# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""The start-of-day flow: its steps, the last device connection test, and whether station setup allows a take.

The flow is station, devices, calibration, record, wrap-up. This service reports where each step stands and is
the recording gate: with a station in use, a take is refused until the connection test has passed for that
station in the current capture session and the system checks hold. With no stations set up nothing here gates
recording, the same configuration-free default the station registry keeps; the steps are then advisory.
"""

from __future__ import annotations

import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from capture_server import protocol as p

from .capture_status import CaptureStatusMonitor
from .config import CaptureStatusConfig
from .livekit_recording import unrecordable_reason
from .setup_checks import (
    FAIL,
    SKIP,
    CameraFact,
    CaptureFacts,
    CheckItem,
    DeviceFact,
    connection_items,
    first_failure,
    system_items,
)
from .station_service import StationService

DONE, TODO, ATTENTION, SKIPPED = "done", "todo", "attention", "skipped"

#: How old a sample may be and still count as arriving; the same window recording readiness uses.
_FRESH_SECONDS = 2.0
#: The system checks touch the filesystem, and the gate is asked many times a second; one answer serves this long.
_SYSTEM_CACHE_SECONDS = 2.0

_SOURCE_KINDS = {p.SourceType.HEAD: "headset", p.SourceType.CONTROLLERS: "controller"}


def capture_facts(monitor: CaptureStatusMonitor, config: CaptureStatusConfig) -> CaptureFacts:
    """The capture session's current state, as the checks read it."""
    native = monitor.native
    now = time.monotonic()
    status = native.latest_status
    running = native.stopped is None and native.status is not None and native.status.running
    gloves = p.PluginType.MANUS in config.plugins
    requested = {_SOURCE_KINDS[source] for source in config.sources if source in _SOURCE_KINDS}
    # Glove devices exist only when the glove plugin supplies the hands.
    if gloves:
        requested.add("hand")
    tracking = monitor.preview.tracking()
    cameras: dict[str, CameraFact] = {}
    for camera in status.cameras if status is not None else ():
        slot = monitor.preview.camera(camera.name)
        cameras[camera.name] = CameraFact(
            connected=camera.status is p.DeviceState.CONNECTED,
            fresh=slot is not None and now - slot.received_monotonic <= _FRESH_SECONDS,
            detail=camera.error or camera.status.name.lower(),
        )
    started = native.started
    livekit = [s for s in started.streams if isinstance(s, p.LiveKitCameraStream)] if started is not None else []
    return CaptureFacts(
        running=running,
        error=None if running else (monitor.snapshot().error_message or "capture is not running"),
        mock=started is not None and started.mode is p.CaptureMode.MOCK,
        session_id=started.session_id if started is not None else None,
        requested=frozenset(requested),
        tracking_fresh=tracking is not None and now - tracking.received_monotonic <= _FRESH_SECONDS,
        devices=tuple(
            DeviceFact(
                kind=device.type,
                label=device.description or device.id,
                connected=device.status is p.DeviceState.CONNECTED,
                detail=device.error or device.status.name.lower(),
            )
            for device in (status.devices if status is not None else ())
            # A device switched off by configuration is not part of this station.
            if device.status is not p.DeviceState.DISABLED
        ),
        cameras=cameras,
        livekit_cameras=frozenset(stream.name for stream in livekit),
        livekit_blocked=unrecordable_reason(started, config.capture_server_url) if livekit else None,
        glove_plugin_requested=gloves,
        glove_plugin_error=_glove_plugin_error(status) if gloves else None,
    )


def _glove_plugin_error(status: p.StatusSnapshot | None) -> str | None:
    """Why the glove plugin is not available, or None when it is."""
    providers = [provider for provider in (status.providers if status is not None else ()) if provider.plugin]
    if not providers:
        return "not reported by the capture session"
    provider = providers[0]
    if provider.status is p.ProviderState.AVAILABLE:
        return None
    return provider.error or f"plugin is {provider.status.name.lower()}"


@dataclass(frozen=True)
class ConnectionTest:
    """One run of the device connection test, and what it was run against."""

    ran_at: float
    station: str
    session_id: str | None
    items: tuple[CheckItem, ...]

    @property
    def passed(self) -> bool:
        """Whether no item failed."""
        return all(item.status != FAIL for item in self.items)

    def to_json(self) -> dict[str, Any]:
        """The wire form."""
        return {
            "ran_at": self.ran_at,
            "station": self.station,
            "passed": self.passed,
            "items": [item.to_json() for item in self.items],
        }


def _step(step_id: str, label: str, status: str, detail: str | None = None) -> dict[str, Any]:
    """One step of the flow in its wire form."""
    return {"id": step_id, "label": label, "status": status, "detail": detail}


class SetupService:
    """The start-of-day state for the station in use, and the gate recording asks."""

    def __init__(
        self,
        *,
        facts: Callable[[], CaptureFacts],
        session_id: Callable[[], str | None],
        stations: StationService,
        calibration_dir: Callable[[], Path],
        output_root: Path,
        free_bytes: Callable[[Path], int] = lambda path: shutil.disk_usage(path).free,
        clock: Callable[[], float] = time.time,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        """``facts`` reads the capture session now and ``session_id`` only its identity, which is all the gate
        needs most of the time; ``calibration_dir`` is read each time, since the directory is an environment
        setting a relaunch may change."""
        self._facts = facts
        self._session_id = session_id
        self._stations = stations
        self._calibration_dir = calibration_dir
        self._output_root = output_root
        self._free_bytes = free_bytes
        self._clock = clock
        self._monotonic = monotonic
        self._test: ConnectionTest | None = None
        self._system_cache: tuple[float, str, list[CheckItem]] | None = None

    def _roles(self) -> dict[str, str | None] | None:
        """The camera each role of the station in use is filled by, or None with no station in use."""
        selected = self._stations.registry.selected
        return dict(selected.camera_roles) if selected is not None else None

    def run_connection_test(self) -> ConnectionTest:
        """Test every device the station needs now, and keep the result for the gate."""
        facts = self._facts()
        self._test = ConnectionTest(
            ran_at=self._clock(),
            station=self._stations.label(),
            session_id=facts.session_id,
            items=tuple(connection_items(facts, self._roles())),
        )
        return self._test

    def current_test(self) -> ConnectionTest | None:
        """The last test, while it still describes this station and this capture session; else None."""
        test = self._test
        # A different station or a restarted capture session has devices this result never looked at.
        if test is None or test.station != self._stations.label() or test.session_id != self._session_id():
            return None
        return test

    def system(self) -> list[CheckItem]:
        """The system checks, served from a short cache: they read the disk and the gate asks constantly."""
        now, station = self._monotonic(), self._stations.label()
        cached = self._system_cache
        if cached is not None and cached[1] == station and now - cached[0] < _SYSTEM_CACHE_SECONDS:
            return cached[2]
        selected = self._stations.registry.selected
        items = system_items(
            self._facts(),
            self._roles(),
            selected.camera_type if selected is not None else None,
            calibration_dir=self._calibration_dir(),
            output_root=self._output_root,
            free_bytes=self._free_bytes,
        )
        self._system_cache = (now, station, items)
        return items

    def block_reason(self) -> str | None:
        """Why station setup does not allow a take now, or None when it does."""
        blocked = self._stations.block_reason()
        if blocked is not None:
            return blocked
        # With no station in use the flow is advisory: the configuration-free default records as before.
        if self._stations.registry.selected is None:
            return None
        test = self.current_test()
        if test is None:
            return "run the device connection test before recording"
        failure = first_failure(list(test.items))
        if failure is not None:
            return f"the device connection test failed: {failure}"
        return first_failure(self.system())

    def status(self, *, saved_count: int, headset_address_changed: bool) -> dict[str, Any]:
        """The flow's steps, the last test and the system checks, for the page."""
        test = self.current_test()
        system = self.system()
        by_id = {item.id: item for item in system}
        return {
            "steps": [
                self._station_step(headset_address_changed),
                self._devices_step(test, [by_id["disk"], by_id["producer"]]),
                self._calibration_step([by_id["calibration"], by_id["gloves"]]),
                _step(
                    "record",
                    "Record",
                    DONE if saved_count > 0 else TODO,
                    f"{saved_count} saved" if saved_count else None,
                ),
                _step("wrap_up", "Wrap up", TODO),
            ],
            "connection_test": test.to_json() if test is not None else None,
            "system": [item.to_json() for item in system],
            "blocked_reason": self.block_reason(),
            # Whether the flow gates recording, or only advises (no station in use).
            "enforced": self._stations.registry.selected is not None,
        }

    def _station_step(self, headset_address_changed: bool) -> dict[str, Any]:
        """Station: skipped with no stations set up, done once the station in use can record."""
        registry = self._stations.registry
        if not registry.stations:
            return _step("station", "Station", SKIPPED, "no stations set up")
        blocked = self._stations.block_reason()
        if blocked is not None:
            return _step("station", "Station", TODO if registry.selected is None else ATTENTION, blocked)
        # Advisory: the headset cannot reach a host whose address changed, but that is not for the gate to judge.
        if headset_address_changed:
            return _step("station", "Station", ATTENTION, "this host's address changed; update the headset")
        return _step("station", "Station", DONE, self._stations.label())

    @staticmethod
    def _devices_step(test: ConnectionTest | None, system: list[CheckItem]) -> dict[str, Any]:
        """Devices: the connection test, then the disk and the camera producer."""
        if test is None:
            return _step("devices", "Devices", TODO, "run the connection test")
        failure = first_failure(list(test.items)) or first_failure(system)
        if failure is not None:
            return _step("devices", "Devices", ATTENTION, failure)
        return _step("devices", "Devices", DONE, "connection test passed")

    @staticmethod
    def _calibration_step(items: list[CheckItem]) -> dict[str, Any]:
        """Calibration: the camera calibration files and the glove plugin; skipped when neither applies."""
        failure = first_failure(items)
        if failure is not None:
            return _step("calibration", "Calibration", ATTENTION, failure)
        if all(item.status == SKIP for item in items):
            return _step("calibration", "Calibration", SKIPPED, items[0].detail)
        return _step("calibration", "Calibration", DONE)
