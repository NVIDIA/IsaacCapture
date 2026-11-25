# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reject software/unknown decoding rather than mislabel a hardware measurement."""

import pytest
from livekit.rtc._proto.stats_pb2 import RtcStats
from isaac_transport_livekit.video import VideoReceiver


def receiver(monkeypatch):
    monkeypatch.delenv("LK_DISABLE_NVDEC", raising=False)
    return VideoReceiver(
        {"video_decoder": "hardware"}, identity="test", publisher="camera", track="cam"
    )


def report(implementation, frames=10):
    stat = RtcStats()
    stat.inbound_rtp.inbound.frames_decoded = frames
    stat.inbound_rtp.inbound.decoder_implementation = implementation
    return stat


@pytest.mark.parametrize("name", ["NVIDIA H264 Decoder", "NVIDIA H265 Decoder"])
def test_nvdec_stats_accepted(monkeypatch, name):
    assert receiver(monkeypatch)._verify_decoder([report(name)], "cam.left")


@pytest.mark.parametrize("name", ["FFmpeg", "", "ExternalDecoder", "libvpx"])
def test_software_or_unknown_rejected(monkeypatch, name):
    with pytest.raises(RuntimeError, match="NVDEC required"):
        receiver(monkeypatch)._verify_decoder([report(name)], "cam.right")


def test_startup_stats_do_not_confirm_decoder(monkeypatch):
    r = receiver(monkeypatch)
    assert not r._verify_decoder([], "cam")
    assert not r._verify_decoder([report("", frames=0)], "cam")


def test_disable_flag_rejected(monkeypatch):
    monkeypatch.setenv("LK_DISABLE_NVDEC", "1")
    with pytest.raises(ValueError, match="fresh viewer"):
        VideoReceiver(
            {"video_decoder": "hardware"},
            identity="test",
            publisher="camera",
            track="cam",
        )
