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


# --- the station's observed state, the headset address it remembers, and the background duties -------------------


def _state_of(*, episode=None, phase=RecordingPhase.IDLE, blocked=None, ready=(True, None)):
    """The observed state for a coordinator, gate and readiness in the given condition."""
    service = SimpleNamespace(episode=episode, phase=phase) if episode is not False else None
    coordinator = SimpleNamespace(snapshot=lambda: SimpleNamespace(service=service), readiness=lambda: ready)
    context = SimpleNamespace(setup_block=lambda: blocked)
    stations = SimpleNamespace(label=lambda: "Bench A")
    return app_module._station_state(coordinator, context, stations)


def test_the_station_is_recording_ready_or_down_with_the_reason():
    assert _state_of(
        episode=object(), phase=RecordingPhase.RECORDING, blocked="ignored while a take is written"
    ) == ("Bench A", "recording", None)
    # An interrupted take is still the service's episode, but nothing is being recorded until it is resolved.
    assert _state_of(episode=object(), phase=RecordingPhase.RECOVERY_REQUIRED) == (
        "Bench A",
        "down",
        "an interrupted take needs recovery",
    )
    assert _state_of() == ("Bench A", "ready", None)
    assert _state_of(episode=False) == ("Bench A", "ready", None)  # no session was ever opened
    assert _state_of(blocked="Left wrist: cam_left: unplugged") == (
        "Bench A",
        "down",
        "Left wrist: cam_left: unplugged",
    )
    assert _state_of(ready=(False, "capture is not running")) == ("Bench A", "down", "capture is not running")
    assert _state_of(ready=(False, None)) == ("Bench A", "down", "capture is not ready")
    # Station setup is the earlier step, so its reason is the one logged.
    assert _state_of(blocked="select a station before recording", ready=(False, "capture is not running"))[2] == (
        "select a station before recording"
    )


class _Stations:
    """A stand-in station service that records what it was asked to remember."""

    def __init__(self, saved: str | None, in_use: bool = True):
        selected = SimpleNamespace(headset_address=saved) if in_use else None
        self.registry = SimpleNamespace(selected=selected)
        self.remembered: list[str] = []

    async def remember_headset_address(self, address: str) -> None:
        self.remembered.append(address)


async def _remember(
    monkeypatch, stations: _Stations, *, connected: bool = True, host=("10.0.0.42", "192.168.1.10")
):
    from egocap_web.host_addresses import HostAddress

    monkeypatch.setattr(
        app_module, "headset_addresses", lambda url: [HostAddress(a, "wlp2s0", "wireless") for a in host]
    )
    setup = SimpleNamespace(headset_connected=lambda: connected)
    await app_module._remember_headset_address(stations, setup, "ws://127.0.0.1:8082")
    return stations.remembered


async def test_a_connected_headset_has_its_host_address_remembered_for_the_station_in_use(monkeypatch):
    assert await _remember(monkeypatch, _Stations(None)) == ["10.0.0.42"]
    assert await _remember(monkeypatch, _Stations("10.0.0.7")) == [
        "10.0.0.42"
    ]  # the old one is gone from the host


async def test_a_remembered_address_the_host_still_has_is_never_overwritten(monkeypatch):
    # The operator may have chosen the wired address; a connected headset does not say which one it uses.
    assert await _remember(monkeypatch, _Stations("192.168.1.10")) == []


async def test_nothing_is_remembered_without_a_connected_headset_a_station_or_an_address(monkeypatch):
    assert await _remember(monkeypatch, _Stations(None), connected=False) == []
    assert await _remember(monkeypatch, _Stations(None, in_use=False)) == []
    assert await _remember(monkeypatch, _Stations(None), host=()) == []


async def test_a_background_duty_keeps_running_after_a_failed_pass(caplog):
    import asyncio

    passes: list[int] = []

    async def duty() -> None:
        passes.append(len(passes))
        if len(passes) == 1:
            raise OSError("disk full")
        if len(passes) == 3:
            raise asyncio.CancelledError  # the application is shutting down

    with pytest.raises(asyncio.CancelledError):
        await app_module._every(0, duty, "the uptime sample")
    assert passes == [0, 1, 2] and caplog.text.count("the uptime sample failed") == 1
