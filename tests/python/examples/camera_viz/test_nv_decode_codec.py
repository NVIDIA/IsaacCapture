# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Codec selection preserves the existing H.264 source API."""

import sys
from types import SimpleNamespace

import pytest

from sources._nv_decode import NvH264Decoder, NvVideoDecoder


@pytest.mark.parametrize(
    "name, expected", [("h264", "h264"), ("h265", "hevc"), ("hevc", "hevc")]
)
def test_native_codec_selection_and_reset(monkeypatch, name, expected):
    calls = []

    class NativeDecoder:
        def __init__(self, cfg):
            calls.append(cfg)

        def decode(self, packet, output, width, height):
            calls.append((packet, output, width, height))
            return True

        def reset(self):
            calls.append("reset")

    monkeypatch.setitem(
        sys.modules,
        "codec",
        SimpleNamespace(
            DecoderConfig=SimpleNamespace,
            DecoderCodec=SimpleNamespace(H264="h264", HEVC="hevc"),
            VideoDecoder=NativeDecoder,
        ),
    )
    decoder = NvVideoDecoder(1600, 1300, gpu_id=2, codec=name)
    assert calls == []
    output = object()
    assert decoder.decode(b"packet", output)
    assert calls[0].codec == expected
    assert calls[0].gpu_id == 2
    assert calls[1] == (b"packet", output, 1600, 1300)
    decoder.reset()
    assert calls[-1] == "reset"


def test_legacy_h264_default_and_invalid_codec():
    assert NvH264Decoder(2, 1)._codec == "h264"
    with pytest.raises(ValueError, match="unsupported decoder codec"):
        NvVideoDecoder(2, 1, codec="vp9")
