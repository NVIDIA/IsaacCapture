# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Thread-safe OOB status publication and fatal service handoff."""

from __future__ import annotations

import json
import os
import threading
from collections import deque
from unittest.mock import MagicMock, patch

import pytest

from isaacteleop.cloudxr.oob_teleop_lifecycle import DeviceReplacedError
from isaacteleop.cloudxr.launcher import CloudXRLauncher
from isaacteleop.cloudxr.service import CloudXRService


def _service_for_status(tmp_path):
    service = object.__new__(CloudXRService)
    service._oob_lock = threading.Lock()
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


def test_status_file_is_atomic_and_bound_to_session(tmp_path):
    service = _service_for_status(tmp_path)
    service._publish_oob_status(
        {"schemaVersion": 1, "health": "degraded", "reason": "Waiting for headset"}
    )
    status = json.loads(service._oob_status_path.read_text())
    assert status["sessionId"] == "test-session"
    assert status["runtimePid"] == os.getpid()
    assert status["health"] == "degraded"
    assert not service._oob_status_path.with_suffix(".json.tmp").exists()
    assert service.oob_status() == status
    assert service.drain_oob_updates() == [status]
    assert service.drain_oob_updates() == []
    assert service.oob_status() == status
    service.health_check()


def test_fatal_callback_is_one_shot_and_health_preserves_cause(tmp_path):
    service = _service_for_status(tmp_path)
    service.stop = MagicMock()
    error = DeviceReplacedError("old", "new")
    service._on_oob_fatal(error)
    service._on_oob_fatal(DeviceReplacedError("old", "another"))
    service._fatal_supervisor.join(timeout=2)
    service.stop.assert_called_once()
    with pytest.raises(
        RuntimeError, match="DEVICE_REPLACED: selected old, observed new"
    ):
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
    with patch("isaacteleop.cloudxr.launcher.is_runtime_live", return_value=True):
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
