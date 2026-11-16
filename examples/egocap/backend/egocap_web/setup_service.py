# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""The start-of-day flow: its steps, the live station checklist, and whether station setup allows a take.

The flow is station, devices, record, wrap-up. The checklist is judged continuously from the capture session, so
there is nothing to run and no result to go stale. With a station in use it is the recording gate: a take is
refused while any item fails. With no stations set up nothing here gates recording, the same configuration-free
default the station registry keeps; the checklist is then advisory.
"""

from __future__ import annotations

import logging
import shutil
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from capture_server import protocol as p

from .capture_status import CaptureStatusMonitor
from .config import CaptureStatusConfig
from .livekit_recording import unrecordable_reason
from .setup_checks import (
    FAIL,
    SYSTEM,
    CameraFact,
    CaptureFacts,
    CheckItem,
    DeviceFact,
    checklist,
    first_failure,
    free_bytes_under,
)
from .station_service import StationService

logger = logging.getLogger("isaaccapture.egocap.setup_service")

DONE, TODO, ATTENTION, SKIPPED = "done", "todo", "attention", "skipped"

#: How old a sample may be and still count as arriving; the same window recording readiness uses.
_FRESH_SECONDS = 2.0
#: The gate is asked many times a second by every viewer; one evaluation of the checklist serves this long.
_CHECKLIST_SECONDS = 1.0


def capture_facts(monitor: CaptureStatusMonitor, config: CaptureStatusConfig) -> CaptureFacts:
    """The capture session's current state, as the checklist reads it."""
    native = monitor.native
    now = time.monotonic()
    status = native.latest_status
    running = native.stopped is None and native.status is not None and native.status.running
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
        tracking_fresh=tracking is not None and now - tracking.received_monotonic <= _FRESH_SECONDS,
        devices=tuple(
            DeviceFact(
                label=device.description or device.id,
                connected=device.status is p.DeviceState.CONNECTED,
                detail=device.error or device.status.name.lower(),
                headset=device.type == "headset",
            )
            for device in (status.devices if status is not None else ())
            # A device switched off by configuration is not part of this station.
            if device.status is not p.DeviceState.DISABLED
        ),
        cameras=cameras,
        livekit_cameras=frozenset(stream.name for stream in livekit),
        livekit_blocked=unrecordable_reason(started, config.capture_server_url) if livekit else None,
    )


def _step(step_id: str, label: str, status: str, detail: str | None = None) -> dict[str, Any]:
    """One step of the flow in its wire form."""
    return {"id": step_id, "label": label, "status": status, "detail": detail}


class SetupService:
    """The start-of-day state for the station in use, and the gate recording asks."""

    def __init__(
        self,
        *,
        facts: Callable[[], CaptureFacts],
        stations: StationService,
        calibration_dir: Callable[[], Path],
        output_root: Path,
        free_bytes: Callable[[Path], int] = lambda path: free_bytes_under(path, shutil.disk_usage),
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        """``facts`` reads the capture session now; ``calibration_dir`` is read each time, since the directory
        is an environment setting a relaunch may change."""
        self._facts = facts
        self._stations = stations
        self._calibration_dir = calibration_dir
        self._output_root = output_root
        self._free_bytes = free_bytes
        self._monotonic = monotonic
        self._cache: tuple[float, Any, CaptureFacts, list[CheckItem]] | None = None
        self._last_failure: str | None = None

    def _roles(self) -> dict[str, str | None] | None:
        """The camera each role of the station in use is filled by, or None with no station in use."""
        selected = self._stations.registry.selected
        return dict(selected.camera_roles) if selected is not None else None

    def _evaluated(self) -> tuple[CaptureFacts, list[CheckItem]]:
        """The capture facts and the checklist judged from them, from a short cache keyed by what was judged."""
        roles = self._roles()
        key = tuple(sorted(roles.items())) if roles is not None else None
        now = self._monotonic()
        cached = self._cache
        if cached is not None and cached[1] == key and now - cached[0] < _CHECKLIST_SECONDS:
            return cached[2], cached[3]
        try:
            facts = self._facts()
            items = checklist(
                facts,
                roles,
                calibration_dir=self._calibration_dir(),
                output_root=self._output_root,
                free_bytes=self._free_bytes,
            )
            self._last_failure = None
        except Exception as exc:  # noqa: BLE001 -- the gate runs in every viewer's loop and must never raise there
            facts = CaptureFacts()
            items = [
                CheckItem("checklist", SYSTEM, "Station checks", FAIL, "could not be evaluated; see backend logs")
            ]
            # Logged when the cause changes, not on every evaluation, or one fault would fill the log.
            if repr(exc) != self._last_failure:
                self._last_failure = repr(exc)
                logger.exception("The station checklist could not be evaluated")
        self._cache = (now, key, facts, items)
        return facts, items

    def checks(self) -> list[CheckItem]:
        """The station checklist as it stands now."""
        return self._evaluated()[1]

    def headset_connected(self) -> bool:
        """Whether a headset is connected now, which is what shows the host address it was given still works."""
        return self._evaluated()[0].headset_connected

    def block_reason(self) -> str | None:
        """Why station setup does not allow a take now, or None when it does."""
        blocked = self._stations.block_reason()
        if blocked is not None:
            return blocked
        # With no station in use the checklist is advisory: the configuration-free default records as before.
        if self._stations.registry.selected is None:
            return None
        return first_failure(self.checks())

    def status(
        self, *, took_today: bool, saved_count: int, report_saved: bool, headset_address_changed: bool
    ) -> dict[str, Any]:
        """The flow's steps, the checklist and what holds a take, for the page.

        ``took_today`` is whether the station in use has recorded a take today; ``saved_count`` is the takes
        kept in the current batch, which is only a detail of the step.
        """
        items = self.checks()
        return {
            "steps": [
                self._station_step(headset_address_changed),
                self._devices_step(items),
                self._record_step(took_today, saved_count),
                _step(
                    "wrap_up", "Wrap up", DONE if report_saved else TODO, "report saved" if report_saved else None
                ),
            ],
            "checks": [item.to_json() for item in items],
            "blocked_reason": self.block_reason(),
            # Whether the checklist gates recording, or only advises (no station in use).
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
        # Advisory: a headset pointed at an address this host no longer has cannot connect.
        if headset_address_changed:
            return _step("station", "Station", ATTENTION, "this host's address changed; update the headset")
        return _step("station", "Station", DONE, self._stations.label())

    @staticmethod
    def _record_step(took_today: bool, saved_count: int) -> dict[str, Any]:
        """Record: done once the station has recorded today. The batch's own count restarts with every new
        batch and outlives midnight, so it never decides the step."""
        if not took_today:
            return _step("record", "Record", TODO)
        batch = f" · {saved_count} saved in this batch" if saved_count else ""
        return _step("record", "Record", DONE, f"recorded today{batch}")

    @staticmethod
    def _devices_step(items: list[CheckItem]) -> dict[str, Any]:
        """Devices: needs attention while any item of the checklist fails."""
        failure = first_failure(items)
        if failure is not None:
            return _step("devices", "Devices", ATTENTION, failure)
        return _step("devices", "Devices", DONE, "all checks pass")
