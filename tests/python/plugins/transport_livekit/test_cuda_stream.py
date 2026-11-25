# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Encoded LiveKit receive must preserve metadata and recover at an IDR after loss."""

import asyncio
from types import SimpleNamespace

import pytest

from isaac_transport_livekit.livekit_cuda import (
    CudaStream,
    EncodedEvent,
    configure_native,
)


def packet(index, key):
    e = EncodedEvent()
    e.encoded.payload = b"access-unit"
    e.encoded.keyframe = key
    e.encoded.metadata.frame_id = index
    e.encoded.metadata.user_timestamp = index * 1000000 + 999
    return e.encoded


def test_encoded_extension_round_trip():
    event = EncodedEvent(stream_handle=42)
    event.encoded.CopyFrom(packet(7, True))
    decoded = EncodedEvent.FromString(event.SerializeToString())
    assert decoded.stream_handle == 42
    assert decoded.encoded.metadata.frame_id == 7
    assert decoded.encoded.keyframe


def test_missing_native_library_fails_before_connect(tmp_path):
    with pytest.raises(ValueError, match="livekit_cuda_library"):
        configure_native({"livekit_cuda_library": str(tmp_path / "missing.so")})


def test_sequence_gap_skips_delta_until_keyframe():
    async def scenario():
        stream = object.__new__(CudaStream)
        stream.closed = False
        stream.waiting = True
        stream.previous = None
        stream.frames = 0
        stream.name = "cam.left"
        stream.packets = asyncio.Queue()
        decoded = []
        resets = []

        def decode(payload, stamp, index):
            assert stamp == index * 1000000 + 999
            decoded.append(index)
            return [SimpleNamespace(frame="CUDA image", metadata=(index, stamp))]

        stream.decoder = SimpleNamespace(
            reset=lambda: resets.append(True), decode=decode
        )
        for item in (packet(1, True), packet(3, False), packet(4, True)):
            stream.packets.put_nowait(item)
        await anext(stream)
        await anext(stream)
        assert decoded == [1, 4]
        assert len(resets) == 2

    asyncio.run(scenario())
