# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Contract boundaries, feedback mapping, and native publication cleanup."""

import asyncio
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from isaac_media_transport.transport import EncoderFeedback, RateControl, get_transport
from isaac_transport_livekit.livekit_transport import LiveKitVideoPublication


def test_contract_import_does_not_load_native_sdk():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import isaac_media_transport.transport; assert 'livekit.rtc' not in sys.modules; assert 'isaaccapture' not in sys.modules",
        ],
        check=True,
    )


def test_backend_selection_is_explicit():
    assert (
        get_transport().capabilities
        == get_transport({"transport": "livekit"}).capabilities
    )
    with pytest.raises(ValueError, match="not implemented"):
        get_transport({"transport": "unknown"})


def test_feedback_and_cleanup_do_not_expose_protobuf():
    async def scenario():
        rate = SimpleNamespace(target_bitrate_bps=1234, framerate_fps=29.97)
        feedback = SimpleNamespace(
            keyframe_requested=True, rate_control=rate, HasField=lambda _: True
        )
        source = SimpleNamespace(feedback=lambda: feedback, aclose=AsyncMock())
        participant = SimpleNamespace(
            unpublish_track=AsyncMock(side_effect=RuntimeError("lost"))
        )
        room = SimpleNamespace(isconnected=lambda: True, local_participant=participant)
        publication = LiveKitVideoPublication(room, SimpleNamespace(sid="test"), source)
        assert publication.feedback() == EncoderFeedback(True, RateControl(1234, 29.97))
        feedback.HasField = lambda _: False
        assert publication.feedback() == EncoderFeedback(True, None)
        with pytest.raises(RuntimeError, match="lost"):
            await publication.close()
        source.aclose.assert_awaited_once()

    asyncio.run(scenario())


def test_failed_publication_releases_source(monkeypatch):
    async def scenario():
        source = SimpleNamespace(aclose=AsyncMock())

        def make_source(width, height, codec, *, settings, track):
            assert (width, height, codec, settings, track) == (
                320,
                180,
                "H264",
                {},
                "cam",
            )
            return source

        monkeypatch.setattr(
            "isaac_transport_livekit.encoded_source.EncodedSource", make_source
        )
        monkeypatch.setattr(
            "isaac_transport_livekit.livekit_transport.rtc.LocalVideoTrack.create_video_track",
            lambda *args: object(),
        )
        session = get_transport().video_sender({}, asyncio.Event())
        session.room = SimpleNamespace(
            local_participant=SimpleNamespace(
                publish_track=AsyncMock(side_effect=RuntimeError("publish failed"))
            )
        )
        with pytest.raises(RuntimeError, match="publish failed"):
            await session.publish(
                dict(name="cam", width=320, height=180, fps=30), encoded=True
            )
        source.aclose.assert_awaited_once()

    asyncio.run(scenario())
