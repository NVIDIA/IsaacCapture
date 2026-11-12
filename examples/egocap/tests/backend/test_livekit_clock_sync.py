# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""ClockSyncClient against a fake room and an injected clock, and its wiring into LiveKitFrameReceiver."""

from __future__ import annotations

import asyncio
import struct
from types import SimpleNamespace

import pytest

from egocap_web import livekit_receiver as lr
from egocap_web.clock_exchange import DEFAULT_MAX_RTT_NS, DEFAULT_STALE_AFTER_NS
from egocap_web.livekit_clock_sync import CLOCK_SYNC_TOPIC, ClockSyncClient

PUBLISHER = "orin"
REMOTE_AHEAD_NS = 3_000_000_000
ONE_WAY_NS = 5_000_000
_REQUEST = struct.Struct("<cIq")
_REPLY = struct.Struct("<cIqqq")


class FakeRoom:
    """Records publish_data calls and handler (un)registration; ``deliver`` plays a data_received event."""

    def __init__(self) -> None:
        self.handlers: dict[str, object] = {}
        self.off_calls: list[tuple[str, object]] = []
        self.published: list[dict] = []
        self.local_participant = SimpleNamespace(publish_data=self._publish_data)
        self.remote_participants: dict[str, object] = {}

    async def _publish_data(self, data: bytes, **kwargs: object) -> None:
        self.published.append({"data": data, **kwargs})

    def on(self, name: str, handler: object) -> None:
        self.handlers[name] = handler

    def off(self, name: str, handler: object) -> None:
        self.off_calls.append((name, handler))
        self.handlers.pop(name, None)

    async def connect(self, url: str, token: str, options: object = None) -> None:
        pass

    async def disconnect(self) -> None:
        pass

    def deliver(self, data: bytes, *, identity: str = PUBLISHER, topic: str = CLOCK_SYNC_TOPIC) -> None:
        packet = SimpleNamespace(data=data, participant=SimpleNamespace(identity=identity), topic=topic)
        self.handlers["data_received"](packet)


class Clock:
    def __init__(self, now: int = 1_000_000_000) -> None:
        self.now = now

    def __call__(self) -> int:
        return self.now


def reply_to(request: bytes, t2: int, t3: int) -> bytes:
    kind, sequence, t1 = _REQUEST.unpack(request)
    assert kind == b"Q"
    return _REPLY.pack(b"R", sequence, t1, t2, t3)


async def settle() -> None:
    for _ in range(5):
        await asyncio.sleep(0)


@pytest.fixture
async def sync():
    room, clock = FakeRoom(), Clock()
    # A long interval: exactly one request is sent, at start.
    client = ClockSyncClient(room, PUBLISHER, interval_s=3600, monotonic_ns=clock)
    client.start()
    await settle()
    yield SimpleNamespace(room=room, clock=clock, client=client)
    await client.stop()


def answer(s: SimpleNamespace, *, one_way_ns: int = ONE_WAY_NS) -> bytes:
    """Delivers the pending request's reply from a remote clock REMOTE_AHEAD_NS ahead; returns the request."""
    request = s.room.published[-1]["data"]
    t1 = _REQUEST.unpack(request)[2]
    t2 = t1 + one_way_ns + REMOTE_AHEAD_NS
    t3 = t2 + 1_000
    s.clock.now = t3 - REMOTE_AHEAD_NS + one_way_ns
    s.room.deliver(reply_to(request, t2, t3))
    return request


async def test_sends_one_request_with_destination_topic_and_t1(sync):
    assert len(sync.room.published) == 1
    sent = sync.room.published[0]
    assert sent["reliable"] is False
    assert sent["destination_identities"] == [PUBLISHER]
    assert sent["topic"] == CLOCK_SYNC_TOPIC
    kind, sequence, t1 = _REQUEST.unpack(sent["data"])
    assert (kind, sequence, t1) == (b"Q", 1, sync.clock.now)


async def test_reply_yields_offset_and_rtt(sync):
    assert sync.client.mapping() is None
    answer(sync)
    mapping = sync.client.mapping()
    assert mapping is not None
    assert abs(mapping.offset_ns - REMOTE_AHEAD_NS) < 1_000
    assert abs(mapping.rtt_ns - 2 * ONE_WAY_NS) < 1_000
    assert sync.client.rejected == 0
    assert not sync.client.discontinuity


def bad_replies(request: bytes) -> dict[str, tuple[bytes, dict]]:
    _, sequence, t1 = _REQUEST.unpack(request)
    t2 = t1 + REMOTE_AHEAD_NS
    good = _REPLY.pack(b"R", sequence, t1, t2, t2)
    return {
        "wrong topic": (good, {"topic": "other"}),
        "other participant": (good, {"identity": "intruder"}),
        "short": (good[:-1], {}),
        "long": (good + b"\0", {}),
        "wrong kind": (_REPLY.pack(b"Q", sequence, t1, t2, t2), {}),
        "unknown sequence": (_REPLY.pack(b"R", sequence + 7, t1, t2, t2), {}),
        "wrong t1": (_REPLY.pack(b"R", sequence, t1 + 1, t2, t2), {}),
    }


@pytest.mark.parametrize(
    "case", ["wrong topic", "other participant", "short", "long", "wrong kind", "unknown sequence", "wrong t1"]
)
async def test_unrelated_or_malformed_replies_are_ignored(sync, case):
    request = sync.room.published[0]["data"]
    data, kwargs = bad_replies(request)[case]
    sync.clock.now += 10_000_000
    sync.room.deliver(data, **kwargs)
    assert sync.client.mapping() is None
    assert sync.client.rejected == 0
    # The request is still outstanding, even after a reply with the wrong t1: the genuine reply is accepted.
    answer(sync)
    assert sync.client.mapping() is not None


async def test_slow_round_trip_is_counted_as_rejected(sync):
    answer(sync, one_way_ns=DEFAULT_MAX_RTT_NS)  # round trip is twice the limit
    assert sync.client.rejected == 1
    assert sync.client.mapping() is None


async def test_mapping_goes_stale_on_the_injected_clock(sync):
    answer(sync)
    assert sync.client.mapping() is not None
    sync.clock.now += DEFAULT_STALE_AFTER_NS + 1
    assert sync.client.mapping() is None


async def test_stop_cancels_task_and_removes_handler():
    room, clock = FakeRoom(), Clock()
    client = ClockSyncClient(room, PUBLISHER, interval_s=3600, monotonic_ns=clock)
    client.start()
    await settle()
    task = client._task
    handler = room.handlers["data_received"]
    await client.stop()
    assert task.cancelled()
    assert client._task is None
    assert room.off_calls == [("data_received", handler)]
    assert "data_received" not in room.handlers


# ---- receiver wiring ----


def receiver(monkeypatch: pytest.MonkeyPatch, **kwargs: object) -> lr.LiveKitFrameReceiver:
    monkeypatch.setattr(
        lr,
        "rtc",
        SimpleNamespace(
            Room=FakeRoom,
            RoomOptions=SimpleNamespace,
            TrackKind=SimpleNamespace(KIND_VIDEO="video"),
        ),
    )
    stream = lr.p.LiveKitCameraStream(
        stream_id=1, name="cam", track="left", url="ws://lk.test", room="room", publisher=PUBLISHER
    )
    return lr.LiveKitFrameReceiver([stream], "tok", **kwargs)


async def test_receiver_runs_clock_sync_when_asked(monkeypatch):
    rx = receiver(monkeypatch, clock_sync=True)
    assert rx.clock_sync is None
    await rx.start()
    client = rx.clock_sync
    assert isinstance(client, ClockSyncClient)
    assert client._task is not None
    await settle()
    assert "data_received" in rx._room.handlers
    await rx.stop()
    assert rx.clock_sync is None
    assert client._task is None


async def test_receiver_does_not_create_clock_sync_by_default(monkeypatch):
    rx = receiver(monkeypatch)
    await rx.start()
    assert rx.clock_sync is None
    assert "data_received" not in rx._room.handlers
    await rx.stop()
