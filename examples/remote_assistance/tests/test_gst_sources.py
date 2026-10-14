# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Source selection and validation without loading camera vendor runtimes."""

from fractions import Fraction
from types import SimpleNamespace

import pytest

from remote_assistance.gst_camera import GstCapture, parameters
from remote_assistance.gst_sources import camera_source


class Element:
    def __init__(self):
        self.properties = {}

    def set_property(self, key, value):
        self.properties[key] = value


@pytest.mark.parametrize("kind", ["v4l2", "argus"])
def test_sources_feed_shared_nvmm_encoder(kind):
    spec = dict(
        type=kind,
        device="/dev/video4",
        sensor_id=1,
        sensor_mode=0,
        flip_method=2,
        width=1920,
        height=1080,
        fps=30,
    )
    _, w, h, fps, _ = parameters({}, spec)
    source = camera_source(spec)
    pipeline = source.pipeline(w, h, f"{fps.numerator}/{fps.denominator}")
    element = Element()
    source.configure(element)
    assert "video/x-raw(memory:NVMM),format=NV12" in pipeline
    assert "framerate=30/1" in pipeline
    if kind == "v4l2":
        assert "v4l2src" in pipeline and "format=YUY2" in pipeline
        assert element.properties == {"device": "/dev/video4"}
    else:
        assert "nvarguscamerasrc" in pipeline and "flip-method=2" in pipeline
        assert "videoconvert" not in pipeline
        assert element.properties == {
            "sensor-id": 1,
            "sensor-mode": 0,
            "event-wait": 10_000_000_000,
            "acquire-wait": 15_000_000_000,
        }


@pytest.mark.parametrize(
    "updates",
    [
        {"sensor_id": -1},
        {"sensor_mode": "0"},
        {"flip_method": 8},
        {"flip_method": True},
        {"stereo": True},
    ],
)
def test_argus_rejects_invalid_source_options(updates):
    with pytest.raises(ValueError):
        camera_source(dict(type="argus", **updates))


def test_source_and_encoder_options_are_validated_independently():
    with pytest.raises(ValueError, match="Unsupported"):
        camera_source({"type": "depthai"})
    with pytest.raises(ValueError, match="YUYV"):
        camera_source({"type": "v4l2", "fourcc": "MJPG"})
    assert parameters({}, dict(type="argus", width=1280, height=720, fps="60000/1001"))[
        3
    ] == Fraction(60000, 1001)


def test_argus_startup_grace_does_not_extend_running_watchdog(monkeypatch):
    capture = GstCapture.__new__(GstCapture)
    capture.Gst = SimpleNamespace(MessageType=SimpleNamespace(ERROR=1, EOS=2))
    capture.bus = SimpleNamespace(timed_pop_filtered=lambda *_: None)
    capture.error = None
    capture.last_frame = 0
    capture.count = 0
    capture.startup_timeout = 20
    monkeypatch.setattr("remote_assistance.gst_camera.time.monotonic", lambda: 15)
    capture.check()
    capture.count = 1
    with pytest.raises(RuntimeError, match="10 seconds"):
        capture.check()
    capture.count = 0
    monkeypatch.setattr("remote_assistance.gst_camera.time.monotonic", lambda: 21)
    with pytest.raises(RuntimeError, match="20 seconds"):
        capture.check()
