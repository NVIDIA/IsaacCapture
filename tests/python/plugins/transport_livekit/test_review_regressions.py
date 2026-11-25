# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
from collections import OrderedDict, Counter
from concurrent.futures import ThreadPoolExecutor
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from livekit.rtc._proto import ffi_pb2

from isaac_transport_livekit import connection, network_setup
from isaac_transport_livekit.data import DataSession
from isaac_transport_livekit.livekit_cuda import CudaStream, EncodedEvent
from isaac_transport_livekit.livekit_transport import LiveKitVideoPublication
from isaac_media_transport.transport import EncoderFeedback


def test_late_connection_is_not_cancelled_and_is_disconnected(monkeypatch):
    async def exercise():
        gate = asyncio.Event()

        async def join(*args, **kwargs):
            await gate.wait()

        room = SimpleNamespace(connect=join, disconnect=AsyncMock())
        monkeypatch.setattr(connection, "CONNECT_TIMEOUT_S", 0.01)
        monkeypatch.setattr(connection, "token_for", lambda *args, **kwargs: "token")
        with pytest.raises(TimeoutError):
            await connection.connect(room, {"url": "wss://example.test"}, "client")
        gate.set()
        await asyncio.gather(*connection._pending_cleanup)
        room.disconnect.assert_awaited_once()

    asyncio.run(exercise())


def test_production_token_requires_tls(monkeypatch):
    monkeypatch.setenv("LIVEKIT_TOKEN", "secret")
    with pytest.raises(ValueError, match="wss"):
        connection.token_for({"url": "ws://example.test"}, "client")


def test_closed_session_rejects_incoming_and_outgoing():
    session = DataSession.__new__(DataSession)
    session.closed = asyncio.Event()
    session.closed.set()
    session.stats = Counter()
    session._send_lock = asyncio.Lock()
    assert not session.ready
    session._received(object())  # No packet fields or native room should be touched.
    assert not asyncio.run(session.send("ra/v1/test", b"data"))


def test_native_callback_queue_is_bounded_before_event_loop_delivery():
    stream = CudaStream.__new__(CudaStream)
    stream.handle = SimpleNamespace(handle=42)
    stream.timing = SimpleNamespace(enabled=False)
    stream._callback_lock = threading.Lock()
    stream._callback_times = OrderedDict()
    stream._pending_encoded = 0
    native = EncodedEvent(stream_handle=42)
    native.encoded.payload = b"payload"
    native.encoded.keyframe = True
    event = ffi_pb2.FfiEvent()
    event.video_stream_event.ParseFromString(native.SerializeToString())
    assert all(stream._filter_event(event) for _ in range(64))
    assert not stream._filter_event(event)
    event.video_stream_event.eos.SetInParent()
    assert stream._filter_event(event)


def test_raw_feedback_and_null_latency_are_supported():
    publication = LiveKitVideoPublication(
        None, None, object(), settings={"latency": None}
    )
    assert publication.feedback() == EncoderFeedback()


def test_nonreplacing_bundle_generation_is_serialized(tmp_path, monkeypatch):
    entered, release = threading.Event(), threading.Event()

    def generate(output, *args, **kwargs):
        entered.set()
        assert release.wait(2)
        output.mkdir()
        (output / "sentinel").write_text("first")
        return output

    monkeypatch.setattr(network_setup, "_generate", generate)
    output = tmp_path / "bundle"
    with ThreadPoolExecutor(2) as pool:
        first = pool.submit(
            network_setup._generate_bundle, output, "", "", "", replace=False
        )
        assert entered.wait(2)
        second = pool.submit(
            network_setup._generate_bundle, output, "", "", "", replace=False
        )
        release.set()
        assert first.result() == output
        with pytest.raises(FileExistsError):
            second.result()
    assert (output / "sentinel").read_text() == "first"
    assert not list(tmp_path.glob("bundle.backup.*"))
