# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""The station checklist: what a station must show before it records.

A list of items, each passed, failed with the reason, warned about, or skipped. It covers every tracking device
the capture session reports, every camera the station uses, the calibration files of those cameras, and room on
the recording disk. A device or check the station does not use is skipped, never failed.

Only a failed item holds a take. An item fails for what recording readiness itself refuses (capture not running,
no recent tracking sample or camera frame, a camera that cannot be recorded) and for a full disk. What goes beyond
that is a warning: a device's reported state, and calibration files unless the deployment requires them.

This module is pure: it judges a :class:`CaptureFacts` value and never reads the capture session itself.
setup_service.py gathers the facts, keeps the checklist fresh and turns it into the recording gate.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

#: ``warn`` is advisory: shown with its reason, and never a reason to hold a take.
PASS, FAIL, WARN, SKIP = "pass", "fail", "warn", "skip"

#: The groups items are listed under, in display order.
TRACKING, CAMERAS, SYSTEM = "tracking", "cameras", "system"

#: Free space, under the recording output, below which a take is refused: a take that fills the disk is lost.
MIN_FREE_BYTES = 10 * 1024**3

#: Operator-facing role names, shared by the items below.
ROLE_NAMES = {"ego": "Ego (head)", "wrist_left": "Left wrist", "wrist_right": "Right wrist"}

#: The calibration file each camera role needs before its video can be related to the tracked pose.
ROLE_CALIBRATION = {
    "ego": ("head_camera_intrinsics.json", "head_camera_extrinsics.json"),
    "wrist_left": ("left_wrist_camera_intrinsics.json",),
    "wrist_right": ("right_wrist_camera_intrinsics.json",),
}
#: Needed once a station has any wrist camera.
WRIST_OFFSETS = "wrist_calibration_offsets.json"


@dataclass(frozen=True)
class CheckItem:
    """One device or check, and how it stands."""

    id: str
    group: str
    label: str
    status: str
    #: Why it failed, warns or was skipped; None for a pass that needs no comment.
    detail: str | None = None

    def to_json(self) -> dict[str, Any]:
        """The wire form."""
        return {
            "id": self.id,
            "group": self.group,
            "label": self.label,
            "status": self.status,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class DeviceFact:
    """One tracking device as the capture status reports it."""

    label: str
    connected: bool
    #: The device's own error or state, shown when it is not connected.
    detail: str | None = None
    #: Whether this is the headset, whose connection is what proves the host address it was given.
    headset: bool = False


@dataclass(frozen=True)
class CameraFact:
    """One CaptureServer camera: whether it is connected and whether a frame arrived recently."""

    connected: bool
    fresh: bool
    detail: str | None = None


@dataclass(frozen=True)
class CaptureFacts:
    """Everything the checklist reads from the capture session, taken at one moment."""

    running: bool = False
    #: Why capture is not running, when it said.
    error: str | None = None
    #: A mock session has no real cameras to calibrate.
    mock: bool = False
    tracking_fresh: bool = False
    devices: tuple[DeviceFact, ...] = ()
    cameras: Mapping[str, CameraFact] | None = None
    #: Names of the camera streams offered over LiveKit.
    livekit_cameras: frozenset[str] = frozenset()
    #: Why an offered LiveKit stream cannot be recorded, or None.
    livekit_blocked: str | None = None

    @property
    def headset_connected(self) -> bool:
        """Whether a headset is connected right now."""
        return any(device.headset and device.connected for device in self.devices)


def _tracking_items(facts: CaptureFacts) -> list[CheckItem]:
    """One item per tracking device the session reports, and one for whether their data is arriving.

    A device that is not connected warns and does not fail. Recording readiness judges the tracking data, not
    the state each device reports, and a station may leave a controller or a glove unused.
    """
    items = [
        CheckItem(
            f"device:{index}",
            TRACKING,
            device.label,
            PASS if device.connected else WARN,
            None if device.connected else (device.detail or "not connected"),
        )
        for index, device in enumerate(facts.devices)
    ]
    # Connected devices that deliver nothing would record an empty take.
    if items and not facts.tracking_fresh:
        items.append(CheckItem("tracking:samples", TRACKING, "Tracking data", FAIL, "no recent sample"))
    return items


def _camera_item(item_id: str, label: str, name: str, facts: CaptureFacts) -> CheckItem:
    """How one named camera stands: a CaptureServer camera must be delivering frames, a LiveKit camera offered
    and recordable (its video is joined when a take starts). A camera that delivers frames while reporting
    another state than connected warns: its frames are what recording readiness judges."""
    if name in facts.livekit_cameras:
        blocked = facts.livekit_blocked
        return CheckItem(item_id, CAMERAS, label, FAIL if blocked else PASS, blocked)
    camera = (facts.cameras or {}).get(name)
    if camera is None:
        return CheckItem(item_id, CAMERAS, label, FAIL, f"camera {name!r} is not offered by the capture session")
    if not camera.fresh:
        # The camera's own reason, when it has one, says more than the missing frame it caused.
        detail = "no recent frame" if camera.connected else (camera.detail or "not connected")
        return CheckItem(item_id, CAMERAS, label, FAIL, detail)
    if not camera.connected:
        return CheckItem(item_id, CAMERAS, label, WARN, camera.detail or "not connected")
    return CheckItem(item_id, CAMERAS, label, PASS)


def _camera_items(facts: CaptureFacts, roles: Mapping[str, str | None] | None) -> list[CheckItem]:
    """The station's cameras by role; with no station in use, every camera the session offers by name."""
    if roles is None:
        names = sorted({*(facts.cameras or {}), *facts.livekit_cameras})
        return [_camera_item(f"camera:{name}", name, name, facts) for name in names]
    items = []
    for role, camera in roles.items():
        label = ROLE_NAMES.get(role, role)
        if camera is None:
            # A station without this camera has nothing to check here.
            items.append(CheckItem(f"camera:{role}", CAMERAS, label, SKIP, "not used by this station"))
        else:
            items.append(_camera_item(f"camera:{role}", f"{label}: {camera}", camera, facts))
    return items


def _calibration_item(
    facts: CaptureFacts, roles: Mapping[str, str | None] | None, calibration_dir: Path, required: bool
) -> CheckItem:
    """The calibration file each camera role in use needs; skipped for a mock session or with no station.

    Calibration files are optional unless the deployment requires them: a missing or unreadable one fails
    when ``required`` and otherwise warns, so a station that has not captured its calibration still records.
    """
    absent = FAIL if required else WARN
    if facts.mock or roles is None:
        why = "mock capture has nothing to calibrate" if facts.mock else "no station in use"
        return CheckItem("calibration", SYSTEM, "Camera calibration", SKIP, why)
    needed = [name for role, names in ROLE_CALIBRATION.items() if roles.get(role) is not None for name in names]
    if roles.get("wrist_left") is not None or roles.get("wrist_right") is not None:
        needed.append(WRIST_OFFSETS)
    if not needed:
        return CheckItem("calibration", SYSTEM, "Camera calibration", SKIP, "the station has no cameras assigned")
    try:
        missing = [name for name in needed if not (calibration_dir / name).is_file()]
    except OSError as exc:
        # A directory that cannot be searched is reported with its cause, never an exception for the caller.
        detail = f"cannot read the calibration directory: {exc.strerror}"
        return CheckItem("calibration", SYSTEM, "Camera calibration", absent, detail)
    detail = f"missing: {', '.join(missing)}" if missing else None
    return CheckItem("calibration", SYSTEM, "Camera calibration", absent if missing else PASS, detail)


def _disk_item(output_root: Path, free_bytes: Callable[[Path], int], min_free_bytes: int) -> CheckItem:
    """Whether the recording output has room for takes."""
    try:
        free = free_bytes(output_root)
    except OSError as exc:
        return CheckItem(
            "disk", SYSTEM, "Free disk space", FAIL, f"cannot read the recording output: {exc.strerror}"
        )
    gib = 1024**3
    detail = f"{free / gib:.1f} GB free"
    if free < min_free_bytes:
        return CheckItem(
            "disk", SYSTEM, "Free disk space", FAIL, f"{detail}; {min_free_bytes / gib:.0f} GB needed"
        )
    return CheckItem("disk", SYSTEM, "Free disk space", PASS, detail)


def checklist(
    facts: CaptureFacts,
    roles: Mapping[str, str | None] | None,
    *,
    calibration_dir: Path,
    output_root: Path,
    free_bytes: Callable[[Path], int],
    min_free_bytes: int = MIN_FREE_BYTES,
    require_calibration: bool = False,
) -> list[CheckItem]:
    """Every item of the station checklist, in display order.

    ``roles`` is the station in use's camera per role, or None when no station is in use.
    ``require_calibration`` is the deployment's choice of whether missing calibration files hold a take.
    Reads the calibration directory and the disk, so it is called from a worker thread.
    """
    system = [
        _calibration_item(facts, roles, calibration_dir, require_calibration),
        _disk_item(output_root, free_bytes, min_free_bytes),
    ]
    if not facts.running:
        # Nothing can be judged device by device without a running session; say that once.
        capture = CheckItem("capture", TRACKING, "Capture session", FAIL, facts.error or "capture is not running")
        return [capture, *system]
    return [*_tracking_items(facts), *_camera_items(facts, roles), *system]


def _first(items: list[CheckItem], status: str) -> str | None:
    """The first item with ``status`` as one phrase, or None when there is none."""
    for item in items:
        if item.status == status:
            return f"{item.label}: {item.detail}" if item.detail else item.label
    return None


def first_failure(items: list[CheckItem]) -> str | None:
    """The first failing item as one phrase, or None when nothing failed."""
    return _first(items, FAIL)


def first_warning(items: list[CheckItem]) -> str | None:
    """The first item that warns as one phrase, or None when nothing does."""
    return _first(items, WARN)


def free_bytes_under(path: Path, disk_usage: Callable[[Path], Any]) -> int:
    """Free space on the filesystem that holds ``path``, which may not exist yet.

    The recording output is created by the first take, so before that the nearest existing ancestor is measured:
    it is on the same filesystem, and a missing directory is not a full disk.
    """
    current = path.resolve()
    while not current.exists() and current != current.parent:
        current = current.parent
    return disk_usage(current).free
