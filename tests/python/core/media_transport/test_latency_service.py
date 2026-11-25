# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import os
import socket
import struct
from types import SimpleNamespace

import pytest
from isaac_media_transport.latency_bus import EventBuffer, SocketTail
from isaac_media_transport.latency_online import expire_stale_tracks
from isaac_media_transport.latency_monitor import TOTAL_SPAN, wait_for_traces


def test_event_service_survives_nonmapping_request_and_detects_zero_cursor_gap(
    tmp_path,
):
    path = tmp_path / "private/events.sock"
    buffer = EventBuffer(path, capacity=2)
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as conn:
            conn.settimeout(1)
            conn.connect(str(path))
            conn.sendall(b"[]\n")
            assert conn.recv(1) == b""
        tail = SocketTail(path)
        assert tail.follow() == 0
        for i in range(3):
            buffer.append({"record_type": "stage", "monotonic_ns": i})
        with pytest.raises(ValueError, match="overrun"):
            tail.read()
    finally:
        buffer.close()


def test_socket_tail_rejects_foreign_server_before_request(monkeypatch):
    class Foreign:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def settimeout(self, timeout):
            pass

        def connect(self, path):
            pass

        def getsockopt(self, *args):
            return struct.pack("3i", 1, os.getuid() + 1, 1)

        def sendall(self, data):
            raise AssertionError("request sent to foreign peer")

    monkeypatch.setattr(socket, "socket", lambda *args: Foreign())
    with pytest.raises(PermissionError):
        SocketTail("/unused").read()


def test_fresh_eye_does_not_keep_stale_eye_totals_visible():
    def snapshot(left, right):
        return {
            "tracks": {
                "cam.left": {"valid_totals": left},
                "cam.right": {"valid_totals": right},
            },
            "metrics": {
                f"{track}/totals/{TOTAL_SPAN}": {"mean_ms": 20}
                for track in ("cam.left", "cam.right")
            },
        }

    freshness = {}
    initial = snapshot(5, 5)
    expire_stale_tracks(initial, freshness, 0)
    later = snapshot(5, 10)
    expire_stale_tracks(later, freshness, 4)
    assert later["stale_tracks"] == ["cam.left"]
    assert list(later["metrics"]) == [f"cam.right/totals/{TOTAL_SPAN}"]
    assert later["fresh_total_count"] == 10
    resumed = snapshot(6, 11)
    expire_stale_tracks(resumed, freshness, 5)
    assert resumed["stale_tracks"] == []


def test_readiness_does_not_retain_a_disappeared_trace(monkeypatch):
    async def exercise():
        tick = 0

        def status(path, stamp):
            return {
                "path": path,
                "run_id": "run" if stamp is not None else None,
                "stage_monotonic_ns": stamp,
            }

        async def request(kind):
            nonlocal tick
            tick += 1
            return (
                None,
                None,
                {
                    "traces": [
                        status("sender", 1 if tick == 1 else 2 if tick == 2 else None)
                    ]
                },
            )

        host = SimpleNamespace(status=lambda: status("viewer", 1 if tick < 3 else 2))

        async def no_sleep(seconds):
            pass

        monkeypatch.setattr(asyncio, "sleep", no_sleep)
        import isaac_media_transport.latency_monitor as monitor

        monkeypatch.setattr(monitor.time, "monotonic", lambda: tick)
        with pytest.raises(ValueError, match="No fresh frame traces"):
            await wait_for_traces(
                SimpleNamespace(request=request), [host], "run", timeout=3
            )

    asyncio.run(exercise())
