# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""WSS starts its host listeners before waiting for an OOB headset."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from isaacteleop.cloudxr import wss
from isaacteleop.cloudxr.oob_teleop_lifecycle import RecoveryConfig


@pytest.mark.asyncio
async def test_wss_lifecycle_waits_without_headset_and_cleans_up_in_order(
    monkeypatch, tmp_path
):
    events = []
    pending = asyncio.Event()
    stop = asyncio.get_running_loop().create_future()

    @asynccontextmanager
    async def serving(*args, **kwargs):
        events.append("wss listening")
        yield
        events.append("wss closed")

    async def lifecycle_run():
        events.append("lifecycle started")
        try:
            await pending.wait()
        finally:
            events.append("lifecycle stopped")

    lifecycle = MagicMock()
    lifecycle.run = lifecycle_run

    with (
        patch.object(wss, "ws_serve", side_effect=serving),
        patch.object(wss, "ensure_certificate"),
        patch.object(
            wss,
            "default_cert_paths",
            return_value=SimpleNamespace(cert_file="c", key_file="k"),
        ),
        patch.object(wss, "build_ssl_context", return_value=object()),
        patch(
            "isaacteleop.cloudxr.oob_teleop_env.require_web_client_static_dir",
            return_value=tmp_path,
        ),
        patch(
            "isaacteleop.cloudxr.oob_teleop_env.start_usb_local_https_server",
            side_effect=lambda *a, **kw: (events.append("https started"), object()),
        ),
        patch(
            "isaacteleop.cloudxr.oob_teleop_env.stop_usb_local_https_server",
            side_effect=lambda *a: events.append("https stopped"),
        ),
        patch(
            "isaacteleop.cloudxr.oob_teleop_lifecycle.OobLifecycle",
            return_value=lifecycle,
        ) as factory,
    ):
        task = asyncio.create_task(
            wss.run(
                None,
                stop,
                setup_oob=True,
                usb_local=True,
                recovery_config=RecoveryConfig(),
                on_listening=lambda: events.append("callback"),
            )
        )
        for _ in range(10):
            await asyncio.sleep(0)
            if "lifecycle started" in events:
                break
        assert not task.done()
        assert events[:3] == ["wss listening", "https started", "callback"]
        factory.assert_called_once()
        stop.set_result(None)
        await task
    assert events.index("lifecycle stopped") < events.index("https stopped")
    assert events.index("https stopped") < events.index("wss closed")


@pytest.mark.asyncio
async def test_hub_only_creates_no_lifecycle(monkeypatch):
    monkeypatch.setenv("TELEOP_OOB_HUB_ONLY", "1")
    stop = asyncio.get_running_loop().create_future()

    @asynccontextmanager
    async def serving(*args, **kwargs):
        yield

    with (
        patch.object(wss, "ws_serve", side_effect=serving),
        patch.object(wss, "ensure_certificate"),
        patch.object(
            wss,
            "default_cert_paths",
            return_value=SimpleNamespace(cert_file="c", key_file="k"),
        ),
        patch.object(wss, "build_ssl_context", return_value=object()),
        patch("isaacteleop.cloudxr.oob_teleop_lifecycle.OobLifecycle") as factory,
    ):
        task = asyncio.create_task(wss.run(None, stop, setup_oob=True))
        await asyncio.sleep(0)
        assert not task.done()
        stop.set_result(None)
        await task
    factory.assert_not_called()
