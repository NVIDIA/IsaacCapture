# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""The app's wiring of recordable LiveKit streams: the recorder's token and the recording config."""

from __future__ import annotations

import time
from types import SimpleNamespace

import numpy as np
import pytest

from capture_server import protocol as p
from egocap_web import app as app_module
from egocap_web.livekit_receiver import RECEIVER_FRAMES_PER_STREAM
from egocap_web.recording_adapter import RecordingIngress
from recording_service import PixelFormat, VideoSample


def _stream(stream_id: int, basis, width=6) -> p.LiveKitCameraStream:
    return p.LiveKitCameraStream(
        stream_id=stream_id,
        name=f"cam{stream_id}",
        url="wss://lk",
        room="r",
        publisher="o",
        track=f"t{stream_id}",
        timestamp_basis=basis,
        width=width,
        height=4,
        fps=30.0,
    )


def _monitor(streams):
    return SimpleNamespace(camera_descriptors=(), native=SimpleNamespace(started=SimpleNamespace(streams=streams)))


def test_config_factory_declares_and_requires_each_recordable_livekit_stream(tmp_path):
    capture = p.LiveKitTimestampBasis.SENSOR_CAPTURE_MONOTONIC_US
    monitor = _monitor(
        [
            _stream(16, capture),
            _stream(17, p.LiveKitTimestampBasis.PUBLISHER_PROCESSING_UNIX_US),
            _stream(18, capture, width=None),
        ]
    )
    config = app_module._recording_config_factory(monitor, None, tmp_path)()
    assert [(v.name, v.width, v.height, v.fps, v.intrinsics) for v in config.video_sources] == [
        ("cam16", 6, 4, 30.0, False)
    ]
    assert "cam16" in config.qc.required_sources


def test_config_factory_without_a_started_session_has_no_livekit_sources(tmp_path):
    monitor = SimpleNamespace(camera_descriptors=(), native=SimpleNamespace(started=None))
    assert app_module._recording_config_factory(monitor, None, tmp_path)().video_sources == ()


def test_recorder_token_is_signed_when_the_key_pair_is_set(monkeypatch):
    monkeypatch.setenv("LIVEKIT_API_KEY", "key")
    monkeypatch.setenv("LIVEKIT_API_SECRET", "secret")
    assert app_module._recorder_livekit_token("room").count(".") == 2


@pytest.mark.parametrize("missing", ["LIVEKIT_API_KEY", "LIVEKIT_API_SECRET"])
def test_recorder_token_names_the_missing_variables(monkeypatch, missing):
    monkeypatch.setenv("LIVEKIT_API_KEY", "key")
    monkeypatch.setenv("LIVEKIT_API_SECRET", "secret")
    monkeypatch.delenv(missing)
    with pytest.raises(RuntimeError, match="LIVEKIT_API_KEY and LIVEKIT_API_SECRET"):
        app_module._recorder_livekit_token("room")


# --- recording readiness -------------------------------------------------------


def _ready_monitor(streams, cameras=()):
    now = time.monotonic()
    return SimpleNamespace(
        tracking_descriptor=object(),
        camera_descriptors=tuple(cameras),
        native=SimpleNamespace(status=SimpleNamespace(running=True), started=SimpleNamespace(streams=streams)),
        preview=SimpleNamespace(
            tracking=lambda: SimpleNamespace(received_monotonic=now),
            camera=lambda name: SimpleNamespace(received_monotonic=now),
        ),
    )


@pytest.fixture
def accepted_tracking(monkeypatch):
    monkeypatch.setattr(app_module, "validate_tracking_descriptor", lambda descriptor: None)


def test_readiness_accepts_recordable_livekit_streams_without_a_jpeg_camera(accepted_tracking):
    capture = p.LiveKitTimestampBasis.SENSOR_CAPTURE_MONOTONIC_US
    check = app_module._recording_readiness_check(_ready_monitor([_stream(16, capture)]), None)
    assert check() == (True, None)


def test_readiness_without_any_stream_says_no_camera(accepted_tracking):
    assert app_module._recording_readiness_check(_ready_monitor([]), None)() == (False, "no camera selected")


def test_readiness_names_a_stream_whose_timestamps_are_not_monotonic(accepted_tracking):
    streams = [
        _stream(16, p.LiveKitTimestampBasis.SENSOR_CAPTURE_MONOTONIC_US),
        _stream(17, p.LiveKitTimestampBasis.PUBLISHER_PROCESSING_UNIX_US),
    ]
    check = app_module._recording_readiness_check(_ready_monitor(streams), None)
    assert check() == (
        False,
        "LiveKit stream 'cam17': its timestamps are not on the publisher's monotonic clock, "
        "so it cannot be recorded",
    )


def test_readiness_names_a_stream_without_geometry(accepted_tracking):
    stream = _stream(16, p.LiveKitTimestampBasis.SENSOR_CAPTURE_MONOTONIC_US, width=None)
    check = app_module._recording_readiness_check(_ready_monitor([stream]), None)
    assert check() == (False, "LiveKit stream 'cam16': width, height and fps must be configured to record it")


# --- memory budget with full-size decoded frames -------------------------------

FULL_WIDTH, FULL_HEIGHT = 2560, 1984


def _full_size_streams(count):
    capture = p.LiveKitTimestampBasis.SENSOR_CAPTURE_UNIX_US
    return [_full(16 + i, capture) for i in range(count)]


def _full(stream_id, basis):
    return p.LiveKitCameraStream(
        stream_id=stream_id,
        name=f"cam{stream_id}",
        url="wss://lk",
        room="r",
        publisher="o",
        track=f"t{stream_id}",
        timestamp_basis=basis,
        width=FULL_WIDTH,
        height=FULL_HEIGHT,
        fps=30.0,
    )


def test_full_size_livekit_frames_fit_ordinary_back_pressure_in_the_ingress():
    started = SimpleNamespace(streams=_full_size_streams(2))
    frame_bytes = FULL_WIDTH * FULL_HEIGHT * 3
    ingress = RecordingIngress(
        queue_capacity=4096,
        queue_bytes=app_module._ingress_queue_bytes(started),
        submit=lambda sample: None,
    )
    image = np.zeros((FULL_HEIGHT, FULL_WIDTH, 3), dtype=np.uint8)
    sample = VideoSample(
        source="cam16", image=image, pixel_format=PixelFormat.RGB8, capture_time=1.0, receive_time=1.0
    )
    # About 200 ms of a stalled consumer at 30 fps on two streams.
    assert all(ingress.offer(sample) for _ in range(12))
    assert ingress.failure is None
    assert app_module._ingress_queue_bytes(started) >= 12 * frame_bytes


def test_config_factory_with_two_full_size_streams_fits_its_total_budget(tmp_path):
    started = SimpleNamespace(streams=_full_size_streams(2))
    monitor = SimpleNamespace(camera_descriptors=(), native=SimpleNamespace(started=started))
    config = app_module._recording_config_factory(monitor, None, tmp_path)()
    frame_bytes = FULL_WIDTH * FULL_HEIGHT * 3
    total = (
        app_module._RECORDING_QUEUE_MEMORY_BUDGET_BYTES
        + app_module._livekit_ingress_bytes(started)
        + app_module._livekit_receiver_bytes(started)
    )
    held = (
        app_module._ingress_queue_bytes(started)
        + app_module._livekit_receiver_bytes(started)
        + max(app_module.MAX_CAMERA_JPEG_BYTES, frame_bytes)
        + (config.queue_capacity + 2) * frame_bytes
    )
    assert config.queue_capacity >= 1
    assert held <= total


def test_owned_bytes_of_two_full_size_streams_never_exceed_the_allowances(tmp_path):
    started = SimpleNamespace(streams=_full_size_streams(2))
    monitor = SimpleNamespace(camera_descriptors=(), native=SimpleNamespace(started=started))
    config = app_module._recording_config_factory(monitor, None, tmp_path)()
    frame_bytes = FULL_WIDTH * FULL_HEIGHT * 3
    owned = (
        app_module._ingress_queue_bytes(started)
        + app_module._livekit_receiver_bytes(started)
        + max(app_module.MAX_CAMERA_JPEG_BYTES, frame_bytes)
        + 2 * frame_bytes
        + config.queue_capacity * frame_bytes
    )
    allowed = (
        app_module._RECORDING_QUEUE_MEMORY_BUDGET_BYTES
        + app_module._livekit_ingress_bytes(started)
        + app_module._livekit_receiver_bytes(started)
    )
    assert owned <= allowed
    # The capacity is the largest that fits: one more decoded frame would exceed the allowance.
    assert owned + frame_bytes > allowed


def test_receiver_queues_are_part_of_the_livekit_budget():
    started = SimpleNamespace(streams=_full_size_streams(2))
    frame_bytes = FULL_WIDTH * FULL_HEIGHT * 3
    assert app_module._livekit_receiver_bytes(started) == 2 * frame_bytes * RECEIVER_FRAMES_PER_STREAM


def test_without_livekit_streams_the_ingress_budget_is_the_jpeg_allowance():
    assert app_module._ingress_queue_bytes(None) == app_module._INGRESS_QUEUE_BYTES


# --- recording needs CaptureServer on this machine -----------------------------


def test_readiness_refuses_livekit_recording_when_capture_server_is_remote(accepted_tracking):
    capture = p.LiveKitTimestampBasis.SENSOR_CAPTURE_UNIX_US
    check = app_module._recording_readiness_check(
        _ready_monitor([_stream(16, capture)]), None, "ws://10.0.0.5:8082"
    )
    ready, reason = check()
    assert not ready and "CaptureServer on this machine" in reason


def test_readiness_accepts_livekit_recording_with_a_loopback_capture_server(accepted_tracking):
    capture = p.LiveKitTimestampBasis.SENSOR_CAPTURE_UNIX_US
    check = app_module._recording_readiness_check(
        _ready_monitor([_stream(16, capture)]), None, "ws://127.0.0.1:8082"
    )
    assert check() == (True, None)
