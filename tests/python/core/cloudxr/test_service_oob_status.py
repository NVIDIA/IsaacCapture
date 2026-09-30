# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Thread-safe OOB status publication and fatal service handoff."""

from __future__ import annotations

import json
import os
import threading
import asyncio
from collections import deque
from unittest.mock import MagicMock, patch

import pytest

from isaaccapture.cloudxr.launcher import CloudXRLauncher
from isaaccapture.cloudxr.service import CloudXRService


def _service_for_status(tmp_path):
    service = object.__new__(CloudXRService)
    service._oob_lock = threading.Lock()
    service._oob_publish_lock = asyncio.Lock()
    service._oob_snapshot = None
    service._oob_updates = deque(maxlen=128)
    service._oob_status_path = tmp_path / "run" / "oob_status.json"
    service._oob_session_id = "test-session"
    service._fatal_error = None
    service._fatal_supervisor = None
    service._runtime_proc = MagicMock()
    service._runtime_proc.pid = os.getpid()
    service._runtime_proc.poll.return_value = None
    service._wss_thread = None
    return service


@pytest.mark.asyncio
async def test_status_file_is_atomic_and_bound_to_session(tmp_path):
    service = _service_for_status(tmp_path)
    await service._publish_oob_status(
        {
            "schemaVersion": 1,
            "health": "degraded",
            "reason": "Waiting for headset",
            "selectedSerial": "original",
            "explicitSerial": False,
            "ignoredSerials": ["other"],
        }
    )
    status = json.loads(service._oob_status_path.read_text())
    assert status["sessionId"] == "test-session"
    assert status["runtimePid"] == os.getpid()
    assert status["health"] == "degraded"
    assert status["ignoredSerials"] == ["other"]
    assert not service._oob_status_path.with_suffix(".json.tmp").exists()
    assert service.oob_status() == status
    assert service.drain_oob_updates() == [status]
    assert service.drain_oob_updates() == []
    assert service.oob_status() == status
    service.health_check()


@pytest.mark.asyncio
async def test_status_persistence_is_nonblocking_ordered_and_drained(tmp_path):
    service = _service_for_status(tmp_path)
    entered = threading.Event()
    release = threading.Event()
    persisted = []
    original = service._persist_oob_status

    def stalled(payload):
        if payload["sequence"] == 1:
            entered.set()
            assert release.wait(timeout=2)
        original(payload)
        persisted.append(payload["sequence"])

    service._persist_oob_status = stalled
    first = asyncio.create_task(service._publish_oob_status({"sequence": 1}))
    await asyncio.to_thread(entered.wait, 2)
    heartbeat = False

    async def tick():
        nonlocal heartbeat
        await asyncio.sleep(0)
        heartbeat = True

    await tick()
    second = asyncio.create_task(service._publish_oob_status({"sequence": 2}))
    await asyncio.sleep(0)
    assert heartbeat is True
    assert persisted == []
    release.set()
    await asyncio.gather(first, second)
    assert persisted == [1, 2]
    assert json.loads(service._oob_status_path.read_text())["sequence"] == 2
    assert [item["sequence"] for item in service.drain_oob_updates()] == [1, 2]


@pytest.mark.asyncio
async def test_cancelled_status_publish_drains_worker_before_return(tmp_path):
    service = _service_for_status(tmp_path)
    entered = threading.Event()
    release = threading.Event()
    original = service._persist_oob_status

    def stalled(payload):
        entered.set()
        assert release.wait(timeout=2)
        original(payload)

    service._persist_oob_status = stalled
    task = asyncio.create_task(service._publish_oob_status({"sequence": 1}))
    await asyncio.to_thread(entered.wait, 2)
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    status = json.loads(service._oob_status_path.read_text())
    assert status["sequence"] == 1


@pytest.mark.asyncio
async def test_status_persistence_failure_preserves_last_valid_file(tmp_path):
    service = _service_for_status(tmp_path)
    await service._publish_oob_status({"sequence": 1})
    with patch(
        "isaaccapture.cloudxr.service._service.os.replace", side_effect=OSError("disk")
    ):
        await service._publish_oob_status({"sequence": 2})
    assert json.loads(service._oob_status_path.read_text())["sequence"] == 1
    assert service.oob_status()["sequence"] == 2
    assert not service._oob_status_path.with_suffix(".json.tmp").exists()


def test_stop_retains_writer_until_blocked_persistence_drains(tmp_path):
    service = _service_for_status(tmp_path)
    service._stop_lock = threading.RLock()
    service._stopping = False
    service._restore_signal_handlers = MagicMock()
    service._terminate_runtime = MagicMock()
    service._setup_oob = True
    service._usb_local = False
    service._host_client = False
    service._recovery_config = None
    entered = threading.Event()
    release = threading.Event()
    original = service._persist_oob_status

    def stalled(payload):
        if payload["sequence"] == 2:
            entered.set()
            assert release.wait(timeout=5)
        original(payload)

    async def proxy(*, stop_future, on_listening, on_oob_status, **_kwargs):
        on_listening()
        await on_oob_status({"sequence": 1})
        publication = asyncio.create_task(on_oob_status({"sequence": 2}))
        await asyncio.wait(
            (publication, stop_future), return_when=asyncio.FIRST_COMPLETED
        )
        if stop_future.done():
            publication.cancel()
        try:
            await publication
        except asyncio.CancelledError:
            pass

    service._persist_oob_status = stalled
    with patch("isaaccapture.cloudxr.wss.run", side_effect=proxy):
        service._start_wss_proxy_thread(tmp_path / "wss.log")
    writer = service._wss_thread
    real_join = writer.join
    try:
        assert entered.wait(timeout=2)
        with patch.object(writer, "join", side_effect=lambda timeout: real_join(0.01)):
            with pytest.raises(RuntimeError, match=r"status persistence.*draining"):
                service.stop()
        assert service._wss_thread is writer
        assert service._wss_loop is not None
        assert service._runtime_proc is not None
        service._terminate_runtime.assert_not_called()
        assert json.loads(service._oob_status_path.read_text())["sequence"] == 1
        release.set()
        real_join(timeout=2)
        assert not writer.is_alive()
        assert json.loads(service._oob_status_path.read_text())["sequence"] == 2
        service.stop()
        assert service._wss_thread is None
        assert service._runtime_proc is None
        assert not service._oob_status_path.exists()
    finally:
        release.set()
        real_join(timeout=2)


def test_fatal_callback_is_one_shot_and_health_preserves_cause(tmp_path):
    service = _service_for_status(tmp_path)
    service.stop = MagicMock()
    error = RuntimeError("fatal lifecycle failure")
    service._on_oob_fatal(error)
    service._on_oob_fatal(RuntimeError("another failure"))
    service._fatal_supervisor.join(timeout=2)
    service.stop.assert_called_once()
    with pytest.raises(RuntimeError, match="fatal lifecycle failure"):
        service.health_check()


def test_stop_only_cleans_status_owned_by_its_session(tmp_path):
    service = _service_for_status(tmp_path)
    service._stop_lock = threading.RLock()
    service._stopping = False
    service._oob_snapshot = {"health": "degraded"}
    service._runtime_proc = None
    service._stop_wss_proxy = MagicMock()
    service._restore_signal_handlers = MagicMock()
    service._oob_status_path.parent.mkdir(parents=True)
    service._oob_status_path.write_text('{"sessionId":"other-service"}')
    service.stop()
    assert service._oob_status_path.exists()
    service._oob_status_path.write_text('{"sessionId":"test-session"}')
    with patch.object(
        type(service._oob_status_path), "unlink", side_effect=OSError("read-only")
    ):
        service.stop()
    assert service._oob_status_path.exists()


def test_attached_launcher_rejects_stale_writer_or_runtime(tmp_path):
    launcher = object.__new__(CloudXRLauncher)
    launcher._service = None
    launcher._run_dir = str(tmp_path)
    path = tmp_path / "oob_status.json"
    path.write_text(
        json.dumps(
            {"schemaVersion": 1, "writerPid": 999999999, "runtimePid": os.getpid()}
        )
    )
    with patch("isaaccapture.cloudxr.launcher.is_runtime_live", return_value=True):
        assert launcher.oob_status() is None
        path.write_text(
            json.dumps(
                {"schemaVersion": 1, "writerPid": os.getpid(), "runtimePid": 999999999}
            )
        )
        assert launcher.oob_status() is None
        path.write_text(
            json.dumps(
                {
                    "schemaVersion": 1,
                    "writerPid": os.getpid(),
                    "runtimePid": os.getpid(),
                    "health": "degraded",
                }
            )
        )
        assert launcher.oob_status()["health"] == "degraded"
