# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest

from remote_assistance.oak_camera import H264AccessUnits, parameters


SPS = b"\x00\x00\x00\x01\x67abc"
PPS = b"\x00\x00\x01\x68def"
IDR = b"\x00\x00\x01\x65xyz"
DELTA = b"\x00\x00\x01\x41xyz"


def test_recovery_includes_parameter_sets():
    stream = H264AccessUnits()
    stream.push(SPS + PPS + IDR, 10, 0)
    assert stream.units.pop()[1]
    stream.push(DELTA, 20, 2)
    assert stream.units.pop() is None
    stream.push(IDR, 30, 3)
    payload, key, stamp, sequence = stream.units.pop()
    assert SPS in payload and b"\x68def" in payload
    assert key and stamp == 30 and sequence == 3


def test_idr_without_headers_is_not_decodable():
    stream = H264AccessUnits()
    stream.push(IDR, 10, 0)
    assert stream.units.pop() is None
    stream.push(SPS + PPS, 20, 1)
    stream.push(DELTA, 30, 2)
    assert stream.units.pop() is None
    stream.push(IDR, 40, 3)
    assert stream.units.pop()[1]


@pytest.mark.parametrize("size", [(1280, 720), (1920, 1080)])
def test_supported_profiles(size):
    assert (
        parameters({}, dict(type="depthai", width=size[0], height=size[1], fps=60))[3]
        == 60
    )


def test_rejects_unsupported_codec():
    with pytest.raises(ValueError, match="h264"):
        parameters(
            dict(codec="h265"), dict(type="depthai", width=1280, height=720, fps=60)
        )


@pytest.mark.parametrize("width,height,fps", [(3840, 2160, 30), (2016, 1520, 60)])
def test_large_profiles(width, height, fps):
    assert parameters({}, dict(type="depthai", width=width, height=height, fps=fps))[
        1:4
    ] == (width, height, fps)


@pytest.mark.parametrize("width,height,fps", [(3840, 2160, 60), (2024, 1520, 60)])
def test_sensor_mode_does_not_override_encoder_limits(width, height, fps):
    with pytest.raises(ValueError):
        parameters({}, dict(type="depthai", width=width, height=height, fps=fps))
