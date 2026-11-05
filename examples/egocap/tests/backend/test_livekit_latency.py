# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""summarize, LiveKitLatencyProbe against a fake frame receiver, and GET /api/livekit-latency."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from capture_server import protocol as p
from egocap_web import livekit_latency as ll
from egocap_web.livekit_latency import LiveKitLatencyProbe, summarize

UNIX = p.LiveKitTimestampBasis.SENSOR_CAPTURE_UNIX_US
MONO = p.LiveKitTimestampBasis.PUBLISHER_MONOTONIC_US
STAMP_US = 1_000_000
NOW_US = 1_050_000
S = 1_000_000_000


def livekit_stream(stream_id: int, *, basis=UNIX, room="room-a"):
    return p.LiveKitCameraStream(
        stream_id=stream_id,
        name=f"cam{stream_id}",
        url="wss://lk",
        room=room,
        publisher="o",
        track=f"t{stream_id}",
        timestamp_basis=basis,
        width=6,
        height=4,
        fps=30.0,
    )


class FakeReceiver:
    instances: list[FakeReceiver] = []

    def __init__(self, streams, token, **kwargs):
        self.streams = list(streams)
        self.queues = {s.stream_id: asyncio.Queue() for s in streams}
        self.failed = asyncio.Event()
        self.stops = 0
        FakeReceiver.instances.append(self)

    async def start(self):
        pass

    async def stop(self):
        self.stops += 1

    async def next_frame_or_lost(self, stream_id):
        return await self.queues[stream_id].get()


@pytest.fixture(autouse=True)
def fake_receiver(monkeypatch):
    FakeReceiver.instances = []
    monkeypatch.setattr(ll, "LiveKitFrameReceiver", FakeReceiver)


def frame(stamp_us=STAMP_US):
    return SimpleNamespace(user_timestamp_us=stamp_us)


async def settle():
    for _ in range(5):
        await asyncio.sleep(0)


def make_probe(streams, clock):
    return LiveKitLatencyProbe(
        lambda: streams, lambda room: "tok", clock_ns=lambda: clock[0], unix_ns=lambda: NOW_US * 1000
    )


async def feed(probe, stream_id, *stamps):
    for stamp in stamps:
        FakeReceiver.instances[-1].queues[stream_id].put_nowait(frame(stamp))
    await settle()


# --- summarize -------------------------------------------------------------------


def test_summarize_empty_is_none():
    assert summarize([]) is None


def test_summarize_median_p95_and_count():
    samples = [float(v) for v in range(100, 0, -1)]
    assert summarize(samples) == {"median_ms": 51.0, "p95_ms": 96.0, "samples": 100}


# --- probe -----------------------------------------------------------------------


async def test_sample_is_unix_now_minus_stamp_in_ms():
    clock = [10 * S]
    probe = make_probe([livekit_stream(1)], clock)
    try:
        await probe.snapshot()
        await feed(probe, 1, STAMP_US)
        out = await probe.snapshot()
        assert out["cam1"]["summary"] == {"median_ms": 50.0, "p95_ms": 50.0, "samples": 1}
        assert out["cam1"]["note"] == ""
        assert len(FakeReceiver.instances) == 1
    finally:
        await probe.stop()


async def test_implausible_dropped_and_small_negative_kept():
    clock = [10 * S]
    probe = make_probe([livekit_stream(1)], clock)
    try:
        await probe.snapshot()
        await feed(probe, 1, NOW_US + 2_000_000, NOW_US - 61_000_000, NOW_US + 5_000, None)
        out = await probe.snapshot()
        assert out["cam1"]["summary"] == {"median_ms": -5.0, "p95_ms": -5.0, "samples": 1}
    finally:
        await probe.stop()


async def test_monotonic_stream_has_note_and_no_summary():
    clock = [10 * S]
    probe = make_probe([livekit_stream(1, basis=MONO)], clock)
    try:
        await probe.snapshot()
        await feed(probe, 1, STAMP_US)
        out = await probe.snapshot()
        assert out["cam1"]["summary"] is None
        assert "monotonic" in out["cam1"]["note"]
    finally:
        await probe.stop()


async def test_no_samples_reports_note_and_old_samples_age_out():
    clock = [10 * S]
    probe = make_probe([livekit_stream(1)], clock)
    try:
        out = await probe.snapshot()
        assert out["cam1"]["summary"] is None
        assert out["cam1"]["note"] == "no frames received recently"
        await feed(probe, 1, STAMP_US)
        assert (await probe.snapshot())["cam1"]["summary"]["samples"] == 1
        clock[0] += int((ll.WINDOW_S + 0.5) * S)
        out = await probe.snapshot()
        assert out["cam1"]["summary"] is None
        assert out["cam1"]["note"] == "no frames received recently"
    finally:
        await probe.stop()


async def test_restarts_when_offered_streams_change_or_receiver_failed():
    clock = [10 * S]
    streams = [livekit_stream(1)]
    probe = make_probe(streams, clock)
    try:
        await probe.snapshot()
        await probe.snapshot()
        assert len(FakeReceiver.instances) == 1
        streams.append(livekit_stream(2))
        out = await probe.snapshot()
        assert len(FakeReceiver.instances) == 2
        assert FakeReceiver.instances[0].stops == 1
        assert set(out) == {"cam1", "cam2"}
        FakeReceiver.instances[1].failed.set()
        await probe.snapshot()
        assert len(FakeReceiver.instances) == 3
        assert FakeReceiver.instances[1].stops == 1
    finally:
        await probe.stop()


# --- route -----------------------------------------------------------------------


def test_route_returns_probe_snapshot():
    class StubProbe:
        async def snapshot(self):
            return {"cam1": {"basis": "X", "summary": None, "note": "n"}}

    app = FastAPI()
    app.include_router(ll.router)
    app.state.livekit_latency = StubProbe()
    response = TestClient(app).get("/api/livekit-latency")
    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    assert response.json() == {"streams": {"cam1": {"basis": "X", "summary": None, "note": "n"}}}
