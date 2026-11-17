# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""The start-of-day flow: its steps, the live station checklist, and whether station setup allows a take.

The flow is station, devices, record, wrap-up. The checklist is judged continuously from the capture session, so
there is nothing to run. With a station in use it is the recording gate: a take is refused while any item fails;
an item that only warns never holds one. With no stations set up nothing here gates recording, the same
configuration-free default the station registry keeps; the checklist is then advisory.

The checklist reads the calibration directory and the recording disk, which can stall on a slow mount, so it is
judged in a worker thread (:meth:`SetupService.refresh`, run as a background duty and by the route). The gate and
every other synchronous reader take the last result from memory and never touch the filesystem.
"""

from __future__ import annotations

import asyncio
import logging
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
    SYSTEM,
    CameraFact,
    CaptureFacts,
    CheckItem,
    DeviceFact,
    checklist,
    first_failure,
    first_warning,
    free_bytes_under,
)
from .station_service import StationService

logger = logging.getLogger("isaaccapture.egocap.setup_service")

DONE, TODO, ATTENTION, SKIPPED = "done", "todo", "attention", "skipped"

#: How old a sample may be and still count as arriving; the same window recording readiness uses.
_FRESH_SECONDS = 2.0
#: How often the application judges the checklist afresh (see app.py's duties).
CHECKLIST_SECONDS = 1.0
#: A result older than this no longer speaks for the station: the evaluation is stuck, on a stalled disk for one.
_STALE_SECONDS = 10.0
#: Why a take is held while no current result exists: before the first one, just after the station in use
#: changed, or while an evaluation is stuck.
CHECKS_PENDING = "the station checks are being evaluated"


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


@dataclass(frozen=True)
class Evaluation:
    """One judgement of the checklist: what was judged, what was found, and when."""

    #: The camera per role of the station judged, as a comparable value; None with no station in use.
    key: tuple[tuple[str, str | None], ...] | None
    facts: CaptureFacts
    items: tuple[CheckItem, ...]
    #: Monotonic time its facts were sampled, so slow file checks age it.
    at: float


class SetupService:
    """The start-of-day state for the station in use, and the gate recording asks."""

    def __init__(
        self,
        *,
        facts: Callable[[], CaptureFacts],
        stations: StationService,
        calibration_dir: Callable[[], Path],
        output_root: Path,
        require_calibration: Callable[[], bool] = lambda: False,
        free_bytes: Callable[[Path], int] = lambda path: free_bytes_under(path, shutil.disk_usage),
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        """``facts`` reads the capture session now, on the event loop. ``calibration_dir`` and
        ``require_calibration`` are read at each evaluation, in its worker thread: the directory is an
        environment setting a relaunch may change, and the requirement is a setting the page can change."""
        self._facts = facts
        self._stations = stations
        self._calibration_dir = calibration_dir
        self._output_root = output_root
        self._require_calibration = require_calibration
        self._free_bytes = free_bytes
        self._monotonic = monotonic
        self._latest: Evaluation | None = None
        # One evaluation at a time, so a stalled disk holds one worker thread however many callers wait.
        self._refreshing = asyncio.Lock()
        self._finished = 0
        self._last_failure: str | None = None

    def _roles(self) -> dict[str, str | None] | None:
        """The camera each role of the station in use is filled by, or None with no station in use."""
        selected = self._stations.registry.selected
        return dict(selected.camera_roles) if selected is not None else None

    @staticmethod
    def _key_of(roles: dict[str, str | None] | None) -> tuple[tuple[str, str | None], ...] | None:
        """The comparable value an evaluation of these roles is kept under."""
        return tuple(sorted(roles.items())) if roles is not None else None

    def _key(self) -> tuple[tuple[str, str | None], ...] | None:
        """What an evaluation for the station in use now would be keyed by."""
        return self._key_of(self._roles())

    async def refresh(self) -> Evaluation:
        """Judge the checklist now and keep the result for the synchronous readers.

        A caller that waited for another evaluation takes that one's result when it is for the same station:
        it finished after this caller asked, so it is as fresh as its own would be.
        """
        asked = self._finished
        async with self._refreshing:
            latest = self._latest
            if latest is not None and self._finished > asked and latest.key == self._key():
                return latest
            roles = self._roles()
            # Stamped when the facts are sampled, not when the files answer: a stalled disk must age the result.
            sampled = self._monotonic()
            try:
                # The capture facts are plain memory and belong to the event loop; only the files leave it.
                facts = self._facts()
                items = await asyncio.to_thread(self._judged, facts, roles)
                self._last_failure = None
            except Exception as exc:  # noqa: BLE001 -- the gate must hold with a reason, never raise
                facts = CaptureFacts()
                items = [
                    CheckItem(
                        "checklist", SYSTEM, "Station checks", FAIL, "could not be evaluated; see backend logs"
                    )
                ]
                # Logged when the cause changes, not on every evaluation, or one fault would fill the log.
                if repr(exc) != self._last_failure:
                    self._last_failure = repr(exc)
                    logger.exception("The station checklist could not be evaluated")
            self._latest = Evaluation(self._key_of(roles), facts, tuple(items), sampled)
            self._finished += 1
            return self._latest

    def _judged(self, facts: CaptureFacts, roles: dict[str, str | None] | None) -> list[CheckItem]:
        """The checklist for these facts; blocking file work, run in a worker thread."""
        return checklist(
            facts,
            roles,
            calibration_dir=self._calibration_dir(),
            output_root=self._output_root,
            free_bytes=self._free_bytes,
            require_calibration=self._require_calibration(),
        )

    def _current(self) -> Evaluation | None:
        """The last evaluation while it still speaks for the station in use, else None."""
        latest = self._latest
        if latest is None or latest.key != self._key():
            return None
        return latest if self._monotonic() - latest.at <= _STALE_SECONDS else None

    def headset_connected(self) -> bool:
        """Whether a headset was connected at the last evaluation, which is what shows the host address it
        was given still works. The headset is the same whichever station is in use."""
        latest = self._latest
        return latest is not None and latest.facts.headset_connected

    def block_reason(self) -> str | None:
        """Why station setup does not allow a take now, or None when it does. Reads memory only: it is asked
        on the event loop, by every viewer's sender and by the uptime sample."""
        blocked = self._stations.block_reason()
        if blocked is not None:
            return blocked
        # With no station in use the checklist is advisory: the configuration-free default records as before.
        if self._stations.registry.selected is None:
            return None
        current = self._current()
        # Held rather than guessed: a take admitted on no result could fill a disk nobody measured.
        if current is None:
            return CHECKS_PENDING
        return first_failure(list(current.items))

    def status(
        self, *, took_today: bool, saved_count: int, report_saved: bool, headset_address_changed: bool
    ) -> dict[str, Any]:
        """The flow's steps, the checklist and what holds a take, for the page.

        ``took_today`` is whether the station in use has recorded a take today; ``saved_count`` is the takes
        kept in the current batch, which is only a detail of the step.
        """
        current = self._current()
        items = list(current.items) if current is not None else []
        return {
            "steps": [
                self._station_step(headset_address_changed),
                # Without a current result nothing is known about the devices; "all checks pass" would be a guess.
                self._devices_step(items)
                if current is not None
                else _step("devices", "Devices", TODO, CHECKS_PENDING),
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
        """Devices: needs attention while any item of the checklist fails. A warning leaves the step done
        and is named in its detail, since it does not hold a take."""
        failure = first_failure(items)
        if failure is not None:
            return _step("devices", "Devices", ATTENTION, failure)
        warning = first_warning(items)
        if warning is not None:
            return _step("devices", "Devices", DONE, f"{warning}; takes are not held")
        return _step("devices", "Devices", DONE, "all checks pass")
