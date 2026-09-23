# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Async unit tests for :mod:`oob_teleop_hub.OOBControlHub`.

Run from this directory (after ``pip install pytest``)::

    pytest -q

No CloudXR runtime, TLS, or ``isaaccapture`` install required — ``conftest.py`` adds
``src/python/isaaccapture/cloudxr`` to ``sys.path``.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from oob_teleop_hub import OOBControlHub


class QueueWS:
    """Minimal async-iterable WebSocket stand-in for :meth:`OOBControlHub.handle_connection`."""

    def __init__(self) -> None:
        self._q: asyncio.Queue[str | None] = asyncio.Queue()
        self.sent: list[str] = []
        self.close_calls: list[tuple[Any, ...]] = []

    async def inject(self, message: str) -> None:
        await self._q.put(message)

    async def end_stream(self) -> None:
        await self._q.put(None)

    def __aiter__(self) -> QueueWS:
        return self

    async def __anext__(self) -> str:
        item = await self._q.get()
        if item is None:
            raise StopAsyncIteration
        return item

    async def send(self, data: str) -> None:
        self.sent.append(data)

    async def close(self, *_args: Any, **_kwargs: Any) -> None:
        self.close_calls.append(_args)


def _loads_sent(ws: QueueWS) -> list[dict]:
    return [json.loads(s) for s in ws.sent]


def test_check_token_no_requirement() -> None:
    hub = OOBControlHub(control_token=None)
    assert hub.check_token(None) is True
    assert hub.check_token("anything") is True


def test_check_token_required() -> None:
    hub = OOBControlHub(control_token="secret")
    assert hub.check_token(None) is False
    assert hub.check_token("wrong") is False
    assert hub.check_token("secret") is True


@pytest.mark.asyncio
async def test_get_snapshot_empty() -> None:
    hub = OOBControlHub()
    snap = await hub.get_snapshot()
    assert snap["configVersion"] == 0
    assert snap["config"] == {}
    assert snap["headsets"] == []
    assert "updatedAt" in snap


@pytest.mark.asyncio
async def test_headset_register_hello_and_snapshot() -> None:
    hub = OOBControlHub(initial_config={"serverIP": "1.2.3.4", "port": 1111})
    ws = QueueWS()
    task = asyncio.create_task(hub.handle_connection(ws))

    await ws.inject(
        json.dumps(
            {"type": "register", "payload": {"role": "headset", "deviceLabel": "Q3"}}
        )
    )
    await asyncio.sleep(0)
    hello = json.loads(ws.sent[0])
    assert hello["type"] == "hello"
    assert hello["payload"]["config"]["serverIP"] == "1.2.3.4"
    assert hello["payload"]["config"]["port"] == 1111
    hid = hello["payload"]["clientId"]

    snap = await hub.get_snapshot()
    assert len(snap["headsets"]) == 1
    assert snap["headsets"][0]["clientId"] == hid
    assert snap["headsets"][0]["deviceLabel"] == "Q3"

    await ws.end_stream()
    await task


@pytest.mark.asyncio
async def test_baseline_client_can_register_stream_without_health_probe_reply() -> None:
    """The pre-healthProbe client still sends its existing OOB messages."""
    hub = OOBControlHub()
    ws = QueueWS()
    task = asyncio.create_task(hub.handle_connection(ws))
    await ws.inject(json.dumps({"type": "register", "payload": {"role": "headset"}}))
    await asyncio.sleep(0)
    await ws.inject(
        json.dumps({"type": "streamStatus", "payload": {"streaming": True}})
    )
    await asyncio.sleep(0)

    state = await hub.get_snapshot()
    assert state["headsets"][0]["streaming"] is True
    assert await hub.probe_browser(1, 0, timeout=0.02) is None
    assert any(msg["type"] == "healthProbe" for msg in _loads_sent(ws))
    assert (await hub.get_snapshot())["headsets"][0]["streaming"] is True
    await ws.end_stream()
    await task


@pytest.mark.asyncio
async def test_generation_probe_requires_matching_fresh_client() -> None:
    hub = OOBControlHub()
    ws = QueueWS()
    handler = asyncio.create_task(hub.handle_connection(ws))
    await ws.inject(json.dumps({"type": "register", "payload": {"role": "headset"}}))
    await asyncio.sleep(0)
    registered_at = (await hub.get_snapshot())["headsets"][0]["registeredAt"] / 1000
    assert await hub.probe_browser(1, registered_at + 1, timeout=0.02) is None

    pending = asyncio.create_task(hub.probe_browser(2, registered_at - 1, timeout=0.2))
    for _ in range(10):
        await asyncio.sleep(0)
        probes = [msg for msg in _loads_sent(ws) if msg["type"] == "healthProbe"]
        if probes:
            break
    assert probes
    probe = probes[-1]["payload"]
    await ws.inject(
        json.dumps(
            {"type": "healthReport", "payload": {**probe, "lifecycleGeneration": 1}}
        )
    )
    await asyncio.sleep(0)
    assert not pending.done()
    await ws.inject(
        json.dumps({"type": "healthReport", "payload": {**probe, "probeId": "old"}})
    )
    await asyncio.sleep(0)
    assert not pending.done()
    await ws.inject(
        json.dumps(
            {
                "type": "clientMetrics",
                "payload": {"cadence": "network", "metrics": {"rtt": 1}},
            }
        )
    )
    await ws.inject(json.dumps({"type": "healthReport", "payload": probe}))
    report = await pending
    assert report is not None
    assert report["lastMetricsAt"] is not None
    state = await hub.get_snapshot()
    assert state["headsets"][0]["lastSeenAt"] is not None
    assert state["headsets"][0]["lastMetricsAt"] is not None
    await hub.set_lifecycle_snapshot({"health": "browser_ready"})
    assert (await hub.get_snapshot())["lifecycle"]["health"] == "browser_ready"
    await ws.end_stream()
    await handler


@pytest.mark.asyncio
async def test_register_rejects_bad_token() -> None:
    hub = OOBControlHub(control_token="ok")
    ws = QueueWS()
    task = asyncio.create_task(hub.handle_connection(ws))

    await ws.inject(
        json.dumps({"type": "register", "payload": {"role": "headset", "token": "bad"}})
    )
    await asyncio.sleep(0)
    err = json.loads(ws.sent[0])
    assert err["type"] == "error"
    assert err["payload"]["code"] == "UNAUTHORIZED"
    assert ws.close_calls

    await ws.end_stream()
    await task


@pytest.mark.asyncio
async def test_register_rejects_non_headset_role() -> None:
    hub = OOBControlHub()
    ws = QueueWS()
    task = asyncio.create_task(hub.handle_connection(ws))

    await ws.inject(json.dumps({"type": "register", "payload": {"role": "dashboard"}}))
    await asyncio.sleep(0)
    err = json.loads(ws.sent[0])
    assert err["type"] == "error"
    assert err["payload"]["code"] == "BAD_REQUEST"

    await ws.end_stream()
    await task


@pytest.mark.asyncio
async def test_first_message_must_be_register() -> None:
    hub = OOBControlHub()
    ws = QueueWS()
    task = asyncio.create_task(hub.handle_connection(ws))

    await ws.inject(json.dumps({"type": "clientMetrics", "payload": {}}))
    await asyncio.sleep(0)
    err = json.loads(ws.sent[0])
    assert err["payload"]["code"] == "BAD_REQUEST"

    await ws.end_stream()
    await task


@pytest.mark.asyncio
async def test_client_metrics_stored_in_snapshot() -> None:
    hub = OOBControlHub()
    ws = QueueWS()
    task = asyncio.create_task(hub.handle_connection(ws))

    await ws.inject(json.dumps({"type": "register", "payload": {"role": "headset"}}))
    await asyncio.sleep(0)
    await ws.inject(
        json.dumps(
            {
                "type": "clientMetrics",
                "payload": {
                    "t": 12345000,
                    "cadence": "frame",
                    "metrics": {"StreamingFramerate": 72.5},
                },
            }
        )
    )
    await asyncio.sleep(0)

    snap = await hub.get_snapshot()
    m = snap["headsets"][0]["metricsByCadence"]["frame"]
    assert m["at"] == 12345000
    assert m["metrics"]["StreamingFramerate"] == 72.5

    await ws.end_stream()
    await task


@pytest.mark.asyncio
async def test_stream_status_updates_snapshot() -> None:
    hub = OOBControlHub()
    ws = QueueWS()
    task = asyncio.create_task(hub.handle_connection(ws))

    await ws.inject(json.dumps({"type": "register", "payload": {"role": "headset"}}))
    await asyncio.sleep(0)

    # Snapshot before any streamStatus: defaults to streaming=False, since=None.
    snap = await hub.get_snapshot()
    assert snap["headsets"][0]["streaming"] is False
    assert snap["headsets"][0]["streamingSince"] is None

    # Rising edge sets the timestamp.
    await ws.inject(
        json.dumps({"type": "streamStatus", "payload": {"streaming": True}})
    )
    await asyncio.sleep(0)
    snap = await hub.get_snapshot()
    assert snap["headsets"][0]["streaming"] is True
    first_since = snap["headsets"][0]["streamingSince"]
    assert isinstance(first_since, int) and first_since > 0

    # Repeat True must NOT reset the timestamp (only rising edges count).
    await asyncio.sleep(0.005)
    await ws.inject(
        json.dumps({"type": "streamStatus", "payload": {"streaming": True}})
    )
    await asyncio.sleep(0)
    snap = await hub.get_snapshot()
    assert snap["headsets"][0]["streamingSince"] == first_since

    # Falling edge clears the timestamp.
    await ws.inject(
        json.dumps({"type": "streamStatus", "payload": {"streaming": False}})
    )
    await asyncio.sleep(0)
    snap = await hub.get_snapshot()
    assert snap["headsets"][0]["streaming"] is False
    assert snap["headsets"][0]["streamingSince"] is None

    await ws.end_stream()
    await task


@pytest.mark.asyncio
async def test_wait_for_streaming_returns_on_rising_edge() -> None:
    hub = OOBControlHub()
    ws = QueueWS()
    task = asyncio.create_task(hub.handle_connection(ws))

    await ws.inject(json.dumps({"type": "register", "payload": {"role": "headset"}}))
    await asyncio.sleep(0)

    waiter = asyncio.create_task(hub.wait_for_streaming(poll_seconds=0.01))
    # Give the waiter a chance to start polling without a True yet.
    await asyncio.sleep(0.02)
    assert not waiter.done()

    await ws.inject(
        json.dumps({"type": "streamStatus", "payload": {"streaming": True}})
    )
    client_id, since = await asyncio.wait_for(waiter, timeout=1.0)
    assert isinstance(client_id, str) and client_id
    assert isinstance(since, float) and since > 0

    await ws.end_stream()
    await task


@pytest.mark.asyncio
async def test_unknown_message_type_returns_error() -> None:
    hub = OOBControlHub()
    ws = QueueWS()
    task = asyncio.create_task(hub.handle_connection(ws))

    await ws.inject(json.dumps({"type": "register", "payload": {"role": "headset"}}))
    await asyncio.sleep(0)
    ws.sent.clear()

    await ws.inject(json.dumps({"type": "bogus", "payload": {}}))
    await asyncio.sleep(0)
    errs = [m for m in _loads_sent(ws) if m["type"] == "error"]
    assert errs and errs[0]["payload"]["code"] == "BAD_REQUEST"

    await ws.end_stream()
    await task


@pytest.mark.asyncio
async def test_http_oob_set_config_noop_returns_changed_false() -> None:
    hub = OOBControlHub(initial_config={"serverIP": "10.0.0.1", "port": 9000})
    status, body = await hub.http_oob_set_config(
        {"config": {"serverIP": "10.0.0.1", "port": 9000}, "token": None}
    )
    assert status == 200
    assert body.get("changed") is False


@pytest.mark.asyncio
async def test_http_oob_set_config_updates_and_pushes() -> None:
    hub = OOBControlHub(initial_config={"serverIP": "127.0.0.1", "port": 49100})
    hw = QueueWS()
    th = asyncio.create_task(hub.handle_connection(hw))
    await hw.inject(json.dumps({"type": "register", "payload": {"role": "headset"}}))
    await asyncio.sleep(0)
    hw.sent.clear()

    status, body = await hub.http_oob_set_config(
        {"config": {"serverIP": "10.0.0.2"}, "token": None}
    )
    assert status == 200
    assert body.get("changed") is True
    cfg_msgs = [m for m in _loads_sent(hw) if m["type"] == "config"]
    assert len(cfg_msgs) == 1
    assert cfg_msgs[0]["payload"]["config"]["serverIP"] == "10.0.0.2"

    await hw.end_stream()
    await th
