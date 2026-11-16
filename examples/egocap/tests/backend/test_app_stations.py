# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""The app's wiring of the station in use: when it may change, how the batch rotates, and a session's label."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from capture_server import protocol as p
from egocap_web import app as app_module
from egocap_web.capture_status import CaptureStatusMonitor
from egocap_web.config import CaptureStatusConfig
from egocap_web.recording_session import CoordinatorBusyError
from egocap_web.station_service import TAKE_IN_PROGRESS
from egocap_web.stations import StationError
from recording_service import RecordingPhase


def _coordinator(phase: RecordingPhase | None, operation=None, refusal: Exception | None = None):
    """A stand-in coordinator in one recording phase; ``phase`` None means no session was ever opened."""
    rotations: list[str] = []

    async def new_batch() -> None:
        if refusal is not None:
            raise refusal
        rotations.append("rotated")

    service = None if phase is None else SimpleNamespace(phase=phase)
    snapshot = SimpleNamespace(service=service, active_operation=operation)
    return SimpleNamespace(snapshot=lambda: snapshot, new_batch=new_batch, rotations=rotations)


@pytest.mark.parametrize(
    "phase", [None, RecordingPhase.IDLE, RecordingPhase.RECOVERY_REQUIRED, RecordingPhase.CLOSED]
)
def test_the_station_may_change_whenever_no_take_is_being_written(phase):
    assert app_module._station_change_blocked(_coordinator(phase)) is None


@pytest.mark.parametrize(
    "phase",
    [RecordingPhase.STARTING, RecordingPhase.RECORDING, RecordingPhase.STOPPING, RecordingPhase.FINALIZING],
)
def test_the_station_cannot_change_while_a_take_is_being_written(phase):
    assert app_module._station_change_blocked(_coordinator(phase)) == TAKE_IN_PROGRESS


def test_the_station_cannot_change_while_a_recording_command_is_still_running():
    assert app_module._station_change_blocked(_coordinator(RecordingPhase.IDLE, operation="NEW_BATCH")) == (
        TAKE_IN_PROGRESS
    )


async def test_a_station_change_rotates_the_batch_and_a_refused_rotation_says_why():
    coordinator = _coordinator(RecordingPhase.IDLE)
    await app_module._rotate_batch_for_station(coordinator)
    assert coordinator.rotations == ["rotated"]
    busy = _coordinator(RecordingPhase.IDLE, refusal=CoordinatorBusyError("an episode is active"))
    with pytest.raises(StationError) as refused:
        await app_module._rotate_batch_for_station(busy)
    assert refused.value.public_message == "the recording session cannot change stations now: an episode is active"


def test_a_session_is_labelled_with_the_station_in_use_when_it_opens(tmp_path):
    monitor = SimpleNamespace(camera_descriptors=(), native=SimpleNamespace(started=None))
    label = ["Bench A"]
    factory = app_module._recording_config_factory(monitor, lambda: label[0], tmp_path)
    assert factory().station_label == "Bench A"
    label[0] = "Cart B"  # the operator changed stations; the next session to open carries the new name
    assert factory().station_label == "Cart B"


def _camera(name: str) -> p.CameraStream:
    return p.CameraStream(
        stream_id=1,
        kind=p.StreamKind.IMAGE,
        name=name,
        type=p.CameraType.OAK_D_COLOR,
        required=True,
        width=8,
        height=4,
        fps=30.0,
        codec=p.ImageCodec.JPEG,
        jpeg_quality=80,
        orientation=p.ImageOrientation(),
    )


def _livekit(name: str) -> p.LiveKitCameraStream:
    return p.LiveKitCameraStream(
        stream_id=16,
        name=name,
        url="wss://lk",
        room="r",
        publisher="o",
        track="t",
        timestamp_basis=p.LiveKitTimestampBasis.SENSOR_CAPTURE_MONOTONIC_US,
        width=8,
        height=4,
        fps=30.0,
    )


def test_the_monitor_offers_every_camera_of_the_accepted_session_by_name():
    monitor = CaptureStatusMonitor(CaptureStatusConfig(), backend_instance_id="b1")
    assert monitor.offered_camera_names() is None  # nothing accepted yet: what is offered is unknown
    tracking = SimpleNamespace(name="tracking")
    monitor.native = SimpleNamespace(
        started=SimpleNamespace(streams=(tracking, _camera("cam_head"), _livekit("cam_wrist")))
    )
    assert monitor.offered_camera_names() == frozenset({"cam_head", "cam_wrist"})


def test_a_changed_station_label_is_announced_to_every_viewer_once():
    monitor = CaptureStatusMonitor(CaptureStatusConfig(station_label="Deployment"), backend_instance_id="b1")
    revision = monitor.revision
    monitor.set_station_label("Deployment")
    assert monitor.revision == revision
    monitor.set_station_label("Bench A")
    assert (monitor.revision, monitor.snapshot().station_label) == (revision + 1, "Bench A")
