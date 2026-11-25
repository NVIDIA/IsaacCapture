# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import threading
from types import SimpleNamespace

import pytest

from remote_assistance import gst_camera


def test_cancel_during_startup_joins_worker_before_stopping(monkeypatch):
    async def exercise():
        entered, release = threading.Event(), threading.Event()
        events = []

        def start():
            entered.set()
            assert release.wait(2)
            events.append("started")

        async def close():
            events.append("closed")

        async def publish(*args, **kwargs):
            return SimpleNamespace(close=close)

        monkeypatch.setattr(
            gst_camera, "parameters", lambda *args: ("h264", 2, 2, 30, 1000)
        )
        monkeypatch.setattr(
            gst_camera,
            "GstCapture",
            lambda *args: SimpleNamespace(
                start=start, stop=lambda: events.append("stopped")
            ),
        )
        task = asyncio.create_task(
            gst_camera._publish_track(
                {}, {"name": "cam"}, asyncio.Event(), SimpleNamespace(publish=publish)
            )
        )
        try:
            assert await asyncio.to_thread(entered.wait, 2)
            task.cancel()
            await asyncio.sleep(0.01)
            assert events == []
        finally:
            release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert events == ["started", "stopped", "closed"]

    asyncio.run(exercise())
