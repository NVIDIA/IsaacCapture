# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""WSS starts its host listeners before waiting for an OOB headset."""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from isaaccapture.cloudxr import wss
from isaaccapture.cloudxr.oob_teleop_lifecycle import RecoveryConfig


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
        # Match the package loggers that WSS forwards into its session file.
        logging.getLogger("isaaccapture.cloudxr.oob_teleop_lifecycle").info(
            "lifecycle test transition"
        )
        logging.getLogger("isaaccapture.cloudxr.oob_teleop_hub").info(
            "hub test registration"
        )
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
            "isaaccapture.cloudxr.oob_teleop_env.require_web_client_static_dir",
            return_value=tmp_path,
        ),
        patch(
            "isaaccapture.cloudxr.oob_teleop_lifecycle.OobLifecycle",
            return_value=lifecycle,
        ) as factory,
    ):
        task = asyncio.create_task(
            wss.run(
                tmp_path / "wss.log",
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
        assert events[:2] == ["wss listening", "callback"]
        factory.assert_called_once()
        stop.set_result(None)
        await task
    assert events.index("lifecycle stopped") < events.index("wss closed")
    assert "lifecycle test transition" in (tmp_path / "wss.log").read_text()
    assert "hub test registration" in (tmp_path / "wss.log").read_text()


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
        patch("isaaccapture.cloudxr.oob_teleop_lifecycle.OobLifecycle") as factory,
    ):
        task = asyncio.create_task(wss.run(None, stop, setup_oob=True))
        await asyncio.sleep(0)
        assert not task.done()
        stop.set_result(None)
        await task
    factory.assert_not_called()


@pytest.mark.asyncio
async def test_wss_reports_unexpected_lifecycle_failure_before_cleanup(
    tmp_path, capsys
):
    events = []
    failures = []
    stop = asyncio.get_running_loop().create_future()
    error = RuntimeError(
        "lifecycle crashed at https://headset.example/client/?controlToken=secret-value&mode=oob"
    )

    @asynccontextmanager
    async def serving(*args, **kwargs):
        events.append("wss listening")
        try:
            yield
        finally:
            events.append("wss closed")

    async def lifecycle_run():
        events.append("lifecycle started")
        raise error

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
            "isaaccapture.cloudxr.oob_teleop_lifecycle.OobLifecycle",
            return_value=lifecycle,
        ),
    ):
        with pytest.raises(RuntimeError, match="lifecycle crashed") as raised:
            await wss.run(
                tmp_path / "wss.log",
                stop,
                setup_oob=True,
                recovery_config=RecoveryConfig(),
                on_oob_fatal=failures.append,
            )
    assert raised.value is error
    assert failures == [error]
    assert events == ["wss listening", "lifecycle started", "wss closed"]
    log_text = (tmp_path / "wss.log").read_text()
    assert log_text.count("OOB lifecycle worker failed") == 1
    assert "Traceback (most recent call last)" in log_text
    assert "RuntimeError: lifecycle crashed" in log_text
    assert "controlToken=<REDACTED>" in log_text
    assert "secret-value" not in log_text
    terminal = capsys.readouterr()
    assert "OOB lifecycle worker failed" not in terminal.err
    assert "secret-value" not in terminal.err


@pytest.mark.asyncio
async def test_wss_reports_unexpected_lifecycle_return(tmp_path):
    stop = asyncio.get_running_loop().create_future()
    failures = []

    @asynccontextmanager
    async def serving(*args, **kwargs):
        yield

    async def lifecycle_run():
        return

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
            "isaaccapture.cloudxr.oob_teleop_lifecycle.OobLifecycle",
            return_value=lifecycle,
        ),
    ):
        with pytest.raises(
            RuntimeError, match="OOB lifecycle worker exited unexpectedly"
        ):
            await wss.run(
                tmp_path / "wss.log",
                stop,
                setup_oob=True,
                recovery_config=RecoveryConfig(),
                on_oob_fatal=failures.append,
            )
    assert len(failures) == 1
    assert str(failures[0]) == "OOB lifecycle worker exited unexpectedly"
    log_text = (tmp_path / "wss.log").read_text()
    assert log_text.count("OOB lifecycle worker exited unexpectedly") == 1
    assert "Traceback (most recent call last)" not in log_text


@pytest.mark.asyncio
async def test_wss_reports_unexpected_lifecycle_self_cancellation(tmp_path, capsys):
    stop = asyncio.get_running_loop().create_future()
    failures = []

    @asynccontextmanager
    async def serving(*args, **kwargs):
        yield

    async def lifecycle_run():
        raise asyncio.CancelledError

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
            "isaaccapture.cloudxr.oob_teleop_lifecycle.OobLifecycle",
            return_value=lifecycle,
        ),
    ):
        with pytest.raises(
            RuntimeError, match="OOB lifecycle worker was cancelled unexpectedly"
        ):
            await wss.run(
                tmp_path / "wss.log",
                stop,
                setup_oob=True,
                recovery_config=RecoveryConfig(),
                on_oob_fatal=failures.append,
            )
    assert not stop.done()
    assert len(failures) == 1
    assert isinstance(failures[0], RuntimeError)
    log_text = (tmp_path / "wss.log").read_text()
    assert log_text.count("OOB lifecycle worker failed") == 1
    assert "asyncio.exceptions.CancelledError" in log_text
    assert "OOB lifecycle worker failed" not in capsys.readouterr().err


@pytest.mark.asyncio
async def test_wss_top_level_failure_after_listening_records_traceback(
    tmp_path, capsys
):
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
        patch(
            "isaaccapture.cloudxr.oob_teleop_lifecycle.OobLifecycle",
            side_effect=KeyError("lifecycle construction failed"),
        ),
    ):
        with pytest.raises(KeyError, match="lifecycle construction failed"):
            await wss.run(
                tmp_path / "wss.log",
                stop,
                setup_oob=True,
                recovery_config=RecoveryConfig(),
            )
    log_text = (tmp_path / "wss.log").read_text()
    assert log_text.count("WSS proxy failed after listening") == 1
    assert "Traceback (most recent call last)" in log_text
    assert "KeyError: 'lifecycle construction failed'" in log_text
    assert "WSS proxy failed after listening" not in capsys.readouterr().err


@pytest.mark.asyncio
async def test_wss_intentional_stop_does_not_report_simultaneous_worker_exit(tmp_path):
    stop = asyncio.get_running_loop().create_future()
    failures = []

    @asynccontextmanager
    async def serving(*args, **kwargs):
        yield

    async def lifecycle_run():
        stop.set_result(None)

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
            "isaaccapture.cloudxr.oob_teleop_lifecycle.OobLifecycle",
            return_value=lifecycle,
        ),
    ):
        await wss.run(
            tmp_path / "wss.log",
            stop,
            setup_oob=True,
            recovery_config=RecoveryConfig(),
            on_oob_fatal=failures.append,
        )
    assert failures == []


@pytest.mark.asyncio
async def test_wss_preserves_lifecycle_error_when_fatal_callback_fails(tmp_path):
    stop = asyncio.get_running_loop().create_future()
    error = RuntimeError("original lifecycle failure")

    @asynccontextmanager
    async def serving(*args, **kwargs):
        yield

    async def lifecycle_run():
        raise error

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
            "isaaccapture.cloudxr.oob_teleop_lifecycle.OobLifecycle",
            return_value=lifecycle,
        ),
    ):
        with pytest.raises(RuntimeError, match="original lifecycle failure") as raised:
            await wss.run(
                tmp_path / "wss.log",
                stop,
                setup_oob=True,
                recovery_config=RecoveryConfig(),
                on_oob_fatal=MagicMock(side_effect=RuntimeError("callback failed")),
            )
    assert raised.value is error
