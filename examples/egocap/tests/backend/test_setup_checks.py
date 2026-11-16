# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""The device connection test and the system checks: what passes, what fails with which reason, what is skipped."""

from __future__ import annotations

import pytest

from egocap_web.setup_checks import (
    FAIL,
    PASS,
    SKIP,
    CameraFact,
    CaptureFacts,
    CheckItem,
    DeviceFact,
    connection_items,
    first_failure,
    system_items,
)

HEADSET = DeviceFact("headset", "Headset", True)
LEFT = DeviceFact("controller", "Left controller", True)
RIGHT = DeviceFact("controller", "Right controller", True)
ROLES = {"ego": "cam_head", "wrist_left": "cam_left", "wrist_right": None}


def _facts(**overrides) -> CaptureFacts:
    base = dict(
        running=True,
        session_id="s1",
        requested=frozenset({"headset", "controller"}),
        tracking_fresh=True,
        devices=(HEADSET, LEFT, RIGHT),
        cameras={"cam_head": CameraFact(True, True), "cam_left": CameraFact(True, True)},
    )
    return CaptureFacts(**{**base, **overrides})


def _by_id(items: list[CheckItem]) -> dict[str, tuple[str, str | None]]:
    return {item.id: (item.status, item.detail) for item in items}


def test_every_device_the_station_needs_passes_and_an_unused_wrist_is_skipped():
    items = connection_items(_facts(), ROLES)
    assert [(item.group, item.label, item.status) for item in items] == [
        ("tracking", "Headset", PASS),
        ("tracking", "Left controller", PASS),
        ("tracking", "Right controller", PASS),
        ("cameras", "Ego (head): cam_head", PASS),
        ("cameras", "Left wrist: cam_left", PASS),
        ("cameras", "Right wrist", SKIP),
    ]
    assert items[-1].detail == "not used by this station"
    assert first_failure(items) is None


def test_a_stopped_capture_session_is_the_one_failure():
    stopped = connection_items(_facts(running=False, error="connection refused"), ROLES)
    assert _by_id(stopped) == {"capture": (FAIL, "connection refused")}
    assert _by_id(connection_items(CaptureFacts(), None)) == {"capture": (FAIL, "capture is not running")}


def test_a_disconnected_device_fails_with_its_own_reason():
    devices = (
        HEADSET,
        DeviceFact("controller", "Left controller", False, "battery empty"),
        DeviceFact("hand", "Left hand", False),
    )
    items = _by_id(connection_items(_facts(devices=devices), ROLES))
    assert items["device:controller:1"] == (FAIL, "battery empty")
    assert items["device:hand:2"] == (FAIL, "not connected")
    assert first_failure(connection_items(_facts(devices=devices), ROLES)) == "Left controller: battery empty"


def test_a_requested_kind_the_session_never_reports_fails():
    facts = _facts(requested=frozenset({"headset", "controller", "hand"}), devices=(HEADSET,))
    items = _by_id(connection_items(facts, ROLES))
    assert items["device:controller"] == (FAIL, "not reported by the capture session")
    assert items["device:hand"] == (FAIL, "not reported by the capture session")
    assert "device:headset" not in items


def test_connected_devices_that_deliver_nothing_fail():
    items = _by_id(connection_items(_facts(tracking_fresh=False), ROLES))
    assert items["tracking:samples"] == (FAIL, "no recent sample")
    # A cameras-only deployment has no tracking to be late.
    assert "tracking:samples" not in _by_id(
        connection_items(_facts(tracking_fresh=False, devices=(), requested=frozenset()), ROLES)
    )


@pytest.mark.parametrize(
    ("camera", "detail"),
    [
        (None, "camera 'cam_left' is not offered by the capture session"),
        (CameraFact(False, False, "unplugged"), "unplugged"),
        (CameraFact(False, True), "not connected"),
        (CameraFact(True, False), "no recent frame"),
    ],
)
def test_a_station_camera_fails_with_the_specific_reason(camera, detail):
    cameras = {"cam_head": CameraFact(True, True)}
    if camera is not None:
        cameras["cam_left"] = camera
    assert _by_id(connection_items(_facts(cameras=cameras), ROLES))["camera:wrist_left"] == (FAIL, detail)


def test_a_livekit_camera_passes_when_offered_and_fails_with_why_it_cannot_be_recorded():
    facts = _facts(cameras={}, livekit_cameras=frozenset({"cam_head", "cam_left"}))
    assert _by_id(connection_items(facts, ROLES))["camera:ego"] == (PASS, None)
    blocked = _facts(
        cameras=None, livekit_cameras=frozenset({"cam_head", "cam_left"}), livekit_blocked="no capture time"
    )
    assert _by_id(connection_items(blocked, ROLES))["camera:ego"] == (FAIL, "no capture time")


def test_without_a_station_every_offered_camera_is_tested_by_name():
    facts = _facts(livekit_cameras=frozenset({"lk_cam"}))
    items = connection_items(facts, None)
    assert [item.id for item in items if item.group == "cameras"] == [
        "camera:cam_head",
        "camera:cam_left",
        "camera:lk_cam",
    ]
    assert all(item.status == PASS for item in items)


def _system(tmp_path, facts=None, roles=ROLES, camera_type=None, free=50 * 1024**3):
    def free_bytes(path):
        if isinstance(free, OSError):
            raise free
        return free

    return _by_id(
        system_items(
            facts or _facts(),
            roles,
            camera_type,
            calibration_dir=tmp_path / "calibration",
            output_root=tmp_path / "recordings",
            free_bytes=free_bytes,
        )
    )


def _calibrate(tmp_path, *names: str) -> None:
    (tmp_path / "calibration").mkdir(exist_ok=True)
    for name in names:
        (tmp_path / "calibration" / name).write_text("{}")


def test_calibration_needs_the_files_of_the_roles_in_use(tmp_path):
    assert _system(tmp_path)["calibration"] == (
        FAIL,
        "missing: head_camera_intrinsics.json, head_camera_extrinsics.json, "
        "left_wrist_camera_intrinsics.json, wrist_calibration_offsets.json",
    )
    _calibrate(
        tmp_path, "head_camera_intrinsics.json", "head_camera_extrinsics.json", "left_wrist_camera_intrinsics.json"
    )
    assert _system(tmp_path)["calibration"] == (FAIL, "missing: wrist_calibration_offsets.json")
    _calibrate(tmp_path, "wrist_calibration_offsets.json")
    assert _system(tmp_path)["calibration"] == (PASS, None)
    # A station with only the right wrist needs that wrist's file, not the left's.
    right_only = {"ego": None, "wrist_left": None, "wrist_right": "cam_right"}
    assert _system(tmp_path, roles=right_only)["calibration"] == (
        FAIL,
        "missing: right_wrist_camera_intrinsics.json",
    )


def test_a_station_without_wrist_cameras_needs_no_wrist_calibration(tmp_path):
    _calibrate(tmp_path, "head_camera_intrinsics.json", "head_camera_extrinsics.json")
    assert _system(tmp_path, roles={"ego": "cam_head", "wrist_left": None, "wrist_right": None})[
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
    assert _system(tmp_path, facts=facts, roles=roles)["calibration"] == (SKIP, detail)


def test_the_glove_plugin_is_checked_only_where_gloves_are_used(tmp_path):
    assert _system(tmp_path)["gloves"] == (SKIP, "gloves are not part of this deployment")
    assert _system(tmp_path, facts=_facts(glove_plugin_requested=True, mock=True))["gloves"] == (
        SKIP,
        "mock capture needs no glove software",
    )
    assert _system(tmp_path, facts=_facts(glove_plugin_requested=True))["gloves"] == (PASS, None)
    unlicensed = _facts(glove_plugin_requested=True, glove_plugin_error="licence dongle not found")
    assert _system(tmp_path, facts=unlicensed)["gloves"] == (FAIL, "licence dongle not found")


def test_disk_space_must_leave_room_for_takes(tmp_path):
    assert _system(tmp_path)["disk"] == (PASS, "50.0 GB free")
    assert _system(tmp_path, free=3 * 1024**3)["disk"] == (FAIL, "3.0 GB free; 10 GB needed")
    assert _system(tmp_path, free=OSError(2, "No such file or directory"))["disk"] == (
        FAIL,
        "cannot read the recording output: No such file or directory",
    )


def test_the_camera_producer_is_checked_only_for_cameras_that_arrive_over_livekit(tmp_path):
    assert _system(tmp_path)["producer"] == (SKIP, "cameras are read by CaptureServer")
    assert _system(tmp_path, camera_type="sensing_gmsl")["producer"] == (
        FAIL,
        "no camera stream is offered by the producer",
    )
    offered = _facts(livekit_cameras=frozenset({"cam_head"}))
    assert _system(tmp_path, facts=offered, camera_type="sensing_gmsl")["producer"] == (PASS, None)
    blocked = _facts(
        livekit_cameras=frozenset({"cam_head"}), livekit_blocked="CaptureServer is not on this machine"
    )
    assert _system(tmp_path, facts=blocked)["producer"] == (FAIL, "CaptureServer is not on this machine")


def test_the_first_failure_reads_as_one_phrase():
    items = [
        CheckItem("a", "system", "Disk", PASS),
        CheckItem("b", "system", "Gloves", FAIL),
        CheckItem("c", "system", "Cam", FAIL, "x"),
    ]
    assert first_failure(items) == "Gloves"
    assert items[2].to_json() == {"id": "c", "group": "system", "label": "Cam", "status": "fail", "detail": "x"}
