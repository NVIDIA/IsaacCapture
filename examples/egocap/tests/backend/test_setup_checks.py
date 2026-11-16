# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""The station checklist: what passes, what fails with which reason, what is skipped."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from egocap_web.setup_checks import (
    FAIL,
    PASS,
    SKIP,
    CameraFact,
    CaptureFacts,
    CheckItem,
    DeviceFact,
    checklist,
    first_failure,
    free_bytes_under,
)

HEADSET = DeviceFact("Headset", True, headset=True)
LEFT = DeviceFact("Left controller", True)
RIGHT = DeviceFact("Right controller", True)
ROLES = {"ego": "cam_head", "wrist_left": "cam_left", "wrist_right": None}
CALIBRATED = (
    "head_camera_intrinsics.json",
    "head_camera_extrinsics.json",
    "left_wrist_camera_intrinsics.json",
    "wrist_calibration_offsets.json",
)


def _facts(**overrides) -> CaptureFacts:
    base = dict(
        running=True,
        tracking_fresh=True,
        devices=(HEADSET, LEFT, RIGHT),
        cameras={"cam_head": CameraFact(True, True), "cam_left": CameraFact(True, True)},
    )
    return CaptureFacts(**{**base, **overrides})


def _items(tmp_path, facts=None, roles=ROLES, free=50 * 1024**3) -> list[CheckItem]:
    def free_bytes(path):
        if isinstance(free, OSError):
            raise free
        return free

    return checklist(
        facts or _facts(),
        roles,
        calibration_dir=tmp_path / "calibration",
        output_root=tmp_path / "recordings",
        free_bytes=free_bytes,
    )


def _by_id(items: list[CheckItem]) -> dict[str, tuple[str, str | None]]:
    return {item.id: (item.status, item.detail) for item in items}


def _calibrate(tmp_path, *names: str) -> None:
    (tmp_path / "calibration").mkdir(exist_ok=True)
    for name in names:
        (tmp_path / "calibration" / name).write_text("{}")


def test_every_device_the_station_needs_passes_and_an_unused_wrist_is_skipped(tmp_path):
    _calibrate(tmp_path, *CALIBRATED)
    items = _items(tmp_path)
    assert [(item.group, item.label, item.status) for item in items] == [
        ("tracking", "Headset", PASS),
        ("tracking", "Left controller", PASS),
        ("tracking", "Right controller", PASS),
        ("cameras", "Ego (head): cam_head", PASS),
        ("cameras", "Left wrist: cam_left", PASS),
        ("cameras", "Right wrist", SKIP),
        ("system", "Camera calibration", PASS),
        ("system", "Free disk space", PASS),
    ]
    assert _by_id(items)["camera:wrist_right"] == (SKIP, "not used by this station")
    assert first_failure(items) is None


def test_a_stopped_capture_session_is_one_failure_and_the_system_checks_still_show(tmp_path):
    stopped = _by_id(_items(tmp_path, _facts(running=False, error="connection refused", mock=True)))
    assert stopped == {
        "capture": (FAIL, "connection refused"),
        "calibration": (SKIP, "mock capture has nothing to calibrate"),
        "disk": (PASS, "50.0 GB free"),
    }
    assert _by_id(_items(tmp_path, CaptureFacts(), None))["capture"] == (FAIL, "capture is not running")


def test_a_disconnected_device_fails_with_its_own_reason(tmp_path):
    devices = (HEADSET, DeviceFact("Left controller", False, "battery empty"), DeviceFact("Left glove", False))
    items = _items(tmp_path, _facts(devices=devices, mock=True))
    assert _by_id(items)["device:1"] == (FAIL, "battery empty")
    assert _by_id(items)["device:2"] == (FAIL, "not connected")
    assert first_failure(items) == "Left controller: battery empty"


def test_only_devices_the_session_reports_are_judged(tmp_path):
    # No glove is reported, so there is no glove item to fail, whatever the deployment asked for.
    items = _by_id(_items(tmp_path, _facts(devices=(HEADSET,), mock=True)))
    assert [key for key in items if key.startswith("device:")] == ["device:0"]


def test_connected_devices_that_deliver_nothing_fail(tmp_path):
    assert _by_id(_items(tmp_path, _facts(tracking_fresh=False)))["tracking:samples"] == (FAIL, "no recent sample")
    # A cameras-only deployment has no tracking to be late.
    assert "tracking:samples" not in _by_id(_items(tmp_path, _facts(tracking_fresh=False, devices=())))


@pytest.mark.parametrize(
    ("camera", "detail"),
    [
        (None, "camera 'cam_left' is not offered by the capture session"),
        (CameraFact(False, False, "unplugged"), "unplugged"),
        (CameraFact(False, True), "not connected"),
        (CameraFact(True, False), "no recent frame"),
    ],
)
def test_a_station_camera_fails_with_the_specific_reason(tmp_path, camera, detail):
    cameras = {"cam_head": CameraFact(True, True)}
    if camera is not None:
        cameras["cam_left"] = camera
    assert _by_id(_items(tmp_path, _facts(cameras=cameras)))["camera:wrist_left"] == (FAIL, detail)


def test_a_livekit_camera_passes_when_offered_and_fails_with_why_it_cannot_be_recorded(tmp_path):
    offered = _facts(cameras={}, livekit_cameras=frozenset({"cam_head", "cam_left"}))
    assert _by_id(_items(tmp_path, offered))["camera:ego"] == (PASS, None)
    blocked = _facts(
        cameras=None, livekit_cameras=frozenset({"cam_head", "cam_left"}), livekit_blocked="no capture time"
    )
    assert _by_id(_items(tmp_path, blocked))["camera:ego"] == (FAIL, "no capture time")


def test_without_a_station_every_offered_camera_is_checked_by_name(tmp_path):
    items = _items(tmp_path, _facts(livekit_cameras=frozenset({"lk_cam"})), roles=None)
    assert [item.id for item in items if item.group == "cameras"] == [
        "camera:cam_head",
        "camera:cam_left",
        "camera:lk_cam",
    ]
    assert _by_id(items)["calibration"] == (SKIP, "no station in use")


def test_calibration_needs_the_files_of_the_roles_in_use(tmp_path):
    assert _by_id(_items(tmp_path))["calibration"] == (FAIL, "missing: " + ", ".join(CALIBRATED))
    _calibrate(tmp_path, *CALIBRATED[:3])
    assert _by_id(_items(tmp_path))["calibration"] == (FAIL, "missing: wrist_calibration_offsets.json")
    _calibrate(tmp_path, CALIBRATED[3])
    assert _by_id(_items(tmp_path))["calibration"] == (PASS, None)
    # A station with only the right wrist needs that wrist's file, not the left's.
    right_only = {"ego": None, "wrist_left": None, "wrist_right": "cam_right"}
    assert _by_id(_items(tmp_path, roles=right_only))["calibration"] == (
        FAIL,
        "missing: right_wrist_camera_intrinsics.json",
    )
    # A station without wrist cameras needs no wrist calibration.
    assert _by_id(_items(tmp_path, roles={"ego": "cam_head", "wrist_left": None, "wrist_right": None}))[
        "calibration"
    ] == (PASS, None)


@pytest.mark.parametrize(
    ("facts", "roles", "detail"),
    [
        (_facts(mock=True), ROLES, "mock capture has nothing to calibrate"),
        (_facts(), None, "no station in use"),
        (_facts(), {"ego": None, "wrist_left": None, "wrist_right": None}, "the station has no cameras assigned"),
    ],
)
def test_calibration_is_skipped_where_it_does_not_apply(tmp_path, facts, roles, detail):
    assert _by_id(_items(tmp_path, facts=facts, roles=roles))["calibration"] == (SKIP, detail)


def test_a_calibration_directory_that_cannot_be_searched_fails_the_check_instead_of_raising(tmp_path, monkeypatch):
    from pathlib import Path

    def denied(self):
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(Path, "is_file", denied)
    assert _by_id(_items(tmp_path))["calibration"] == (
        FAIL,
        "cannot read the calibration directory: Permission denied",
    )


def test_disk_space_must_leave_room_for_takes(tmp_path):
    assert _by_id(_items(tmp_path))["disk"] == (PASS, "50.0 GB free")
    assert _by_id(_items(tmp_path, free=3 * 1024**3))["disk"] == (FAIL, "3.0 GB free; 10 GB needed")
    assert _by_id(_items(tmp_path, free=OSError(5, "Input/output error")))["disk"] == (
        FAIL,
        "cannot read the recording output: Input/output error",
    )


def test_free_space_is_measured_on_the_nearest_existing_directory(tmp_path):
    asked = []

    def disk_usage(path):
        asked.append(path)
        return SimpleNamespace(free=123)

    # A fresh install has no recording output yet; its parent is on the same disk.
    assert free_bytes_under(tmp_path / "recordings" / "nested", disk_usage) == 123
    assert asked == [tmp_path.resolve()]
    assert free_bytes_under(tmp_path, disk_usage) == 123 and asked[-1] == tmp_path.resolve()


def test_whether_a_headset_is_connected_is_read_from_the_devices():
    assert _facts().headset_connected is True
    assert _facts(devices=(DeviceFact("Headset", False, headset=True), LEFT)).headset_connected is False
    assert _facts(devices=(LEFT, RIGHT)).headset_connected is False


def test_the_first_failure_reads_as_one_phrase():
    items = [
        CheckItem("a", "system", "Disk", PASS),
        CheckItem("b", "system", "Gloves", FAIL),
        CheckItem("c", "system", "Cam", FAIL, "x"),
    ]
    assert first_failure(items) == "Gloves"
    assert items[2].to_json() == {"id": "c", "group": "system", "label": "Cam", "status": "fail", "detail": "x"}
