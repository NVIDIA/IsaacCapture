# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Own coturn files with the process that can reopen them."""

import asyncio
import gc
import os
import secrets
import signal
import stat
import subprocess
import sys
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock

import pytest

# Match the lifecycle import so mocks reach the same module instance.
from isaaccapture.cloudxr import oob_teleop_adb as adb
from isaaccapture.cloudxr.oob_teleop_lifecycle import OobLifecycle, RecoveryConfig


@pytest.fixture
def relay(tmp_path, monkeypatch):
    binary = tmp_path / "turnserver"
    mode = tmp_path / "mode"
    binary.write_text(
        f"#!{sys.executable}\n"
        "from pathlib import Path\n"
        "import signal, sys, time\n"
        "config = Path(sys.argv[2])\n"
        "values = dict(line.split('=', 1) for line in config.read_text().splitlines() if '=' in line)\n"
        "mode = Path(__file__).with_name('mode')\n"
        "mode = mode.read_text() if mode.exists() else 'live'\n"
        "Path(values['log-file']).write_text('relay ready\\n')\n"
        "if mode == 'early': sys.exit(7)\n"
        "if mode == 'ignore-term': signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        "config.with_name('ready').touch()\n"
        "while True: time.sleep(1)\n"
    )
    binary.chmod(0o700)
    directories = []

    def directory(**kwargs):
        owned = TemporaryDirectory(dir=tmp_path, **kwargs)
        directories.append(Path(owned.name))
        return owned

    monkeypatch.setattr(adb, "TemporaryDirectory", directory)
    monkeypatch.setattr(adb, "coturn_binary_path", lambda: str(binary))
    credential = secrets.token_urlsafe(24)
    processes = []
    start = adb.start_coturn

    def launch(port=3478, user="test-user", secret=None):
        proc = start(port, user, credential if secret is None else secret)
        if proc is not None:
            processes.append(proc)
            deadline = time.monotonic() + 5
            while not (Path(proc.directory.name) / "ready").exists():
                assert proc.poll() is None, "stand-in exited before becoming ready"
                assert time.monotonic() < deadline, "stand-in did not become ready"
                time.sleep(0.01)
        return proc

    yield launch, processes, directories, mode, credential
    for proc in processes:
        adb.stop_coturn(proc)


def test_private_files_live_until_the_owned_process_stops(relay, caplog):
    launch, _, directories, _, credential = relay
    proc = launch()
    directory = directories[0]
    assert stat.S_IMODE(directory.stat().st_mode) == 0o700
    for name in ("turnserver.conf", "coturn.log"):
        assert stat.S_IMODE((directory / name).stat().st_mode) == 0o600
    assert credential in (directory / "turnserver.conf").read_text()
    assert credential not in caplog.text
    assert adb._tail_file(proc.log_path, 20) == "relay ready"

    adb.stop_coturn(proc)
    assert proc.poll() is not None
    assert not directory.exists()


def test_same_port_launches_have_separate_files(relay):
    launch, _, directories, _, _ = relay
    first = launch()
    second = launch()
    assert first.log_path != second.log_path
    assert len(set(directories)) == 2
    assert all(directory.exists() for directory in directories)


@pytest.mark.parametrize("failure", ["directory", "config", "spawn"])
def test_start_failure_removes_its_private_files(relay, monkeypatch, failure):
    launch, _, directories, _, _ = relay

    def fail(*args, **kwargs):
        raise OSError("injected startup failure")

    target = {
        "directory": "TemporaryDirectory",
        "config": "_open_private",
        "spawn": "_CoturnProcess",
    }[failure]
    monkeypatch.setattr(adb, target, fail)
    assert launch() is None
    assert not any(directory.exists() for directory in directories)


def test_early_exit_reports_tail_and_removes_files(relay, caplog):
    launch, _, directories, mode, _ = relay
    mode.write_text("early")
    assert launch() is None
    assert not directories[0].exists()
    assert "relay ready" in caplog.text


def test_stop_waits_for_forced_exit_before_removing_files(relay, monkeypatch):
    launch, _, directories, mode, _ = relay
    mode.write_text("ignore-term")
    proc = launch()
    wait = proc.wait
    first = True

    def interrupted_stop(timeout):
        nonlocal first
        if first:
            first = False
            raise subprocess.TimeoutExpired(proc.args, timeout)
        return wait(timeout=timeout)

    monkeypatch.setattr(proc, "wait", interrupted_stop)
    adb.stop_coturn(proc)
    assert proc.returncode == -signal.SIGKILL
    assert not directories[0].exists()


def test_failed_stop_keeps_live_process_files(tmp_path):
    directory = TemporaryDirectory(dir=tmp_path)
    proc = MagicMock(pid=123)
    proc.directory = directory
    proc.poll.return_value = None
    proc.cleanup_files.side_effect = directory.cleanup
    proc.terminate.side_effect = OSError("stop failed")
    proc.kill.side_effect = OSError("kill failed")
    try:
        adb.stop_coturn(proc)
        assert Path(directory.name).exists()
    finally:
        directory.cleanup()


def test_dead_process_tail_is_reported_before_cleanup(relay, caplog):
    launch, _, directories, _, _ = relay
    proc = launch()
    proc.terminate()
    proc.wait(timeout=5)
    adb.stop_coturn(proc)
    assert "exited before shutdown" in caplog.text
    assert "relay ready" in caplog.text
    assert not directories[0].exists()


def test_repeated_stop_leaves_a_replacement_at_the_old_path_alone(relay, caplog):
    launch, _, directories, _, _ = relay
    proc = launch()
    adb.stop_coturn(proc)
    directory = directories[0]
    directory.mkdir()
    replacement = directory / "coturn.log"
    replacement.write_text("replacement must not be read or deleted")
    caplog.clear()

    adb.stop_coturn(proc)

    assert replacement.read_text() == "replacement must not be read or deleted"
    assert "replacement must not be read or deleted" not in caplog.text


async def test_restart_uses_new_files_and_cleans_both_dead_processes(
    relay, monkeypatch
):
    launch, _, directories, _, credential = relay
    first = launch()
    first.terminate()
    first.wait(timeout=5)
    proc_box = [first]
    monkeypatch.setattr(adb, "start_coturn", launch)
    task = asyncio.create_task(
        adb.watch_coturn(
            proc_box,
            turn_port=3478,
            user="test-user",
            credential=credential,
            poll_seconds=0.001,
        )
    )
    try:
        deadline = time.monotonic() + 5
        while proc_box[0] is first:
            assert not task.done(), task.exception()
            assert time.monotonic() < deadline
            await asyncio.sleep(0.01)
        second = proc_box[0]
        assert not directories[0].exists()
        assert Path(second.directory.name).exists()
        second.terminate()
        second.wait(timeout=5)
        await asyncio.wait_for(task, timeout=5)
        assert not any(directory.exists() for directory in directories)
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


def test_tail_refuses_fifo_without_waiting_for_a_writer(tmp_path):
    fifo = tmp_path / "log"
    os.mkfifo(fifo, 0o600)
    assert adb._tail_file(str(fifo), 20) == ""


def test_tail_refuses_symlink(tmp_path):
    target = tmp_path / "target"
    target.write_text("private")
    link = tmp_path / "log"
    link.symlink_to(target)
    assert adb._tail_file(str(link), 20) == ""


async def test_recovery_retains_a_relay_that_cannot_be_stopped(relay, monkeypatch):
    launch, processes, directories, _, _ = relay
    lifecycle = OobLifecycle(
        hub=None,
        resolved_port=48322,
        usb_local=True,
        host_client=False,
        config=RecoveryConfig(60, 5),
        turn_port=3478,
    )
    lifecycle.coturn = launch()
    processes.clear()  # The lifecycle must be the only owner of the process handle.
    starts = []
    monkeypatch.setattr(adb, "verify_coturn_listening", lambda _port: False)
    monkeypatch.setattr(adb, "start_coturn", lambda *args: starts.append(args))

    def refuse_stop(self):
        raise OSError("injected failure to signal relay")

    try:
        with monkeypatch.context() as stop_failure:
            stop_failure.setattr(adb._CoturnProcess, "terminate", refuse_stop)
            stop_failure.setattr(adb._CoturnProcess, "kill", refuse_stop)
            for _ in range(2):
                with pytest.raises(adb.OobAdbError, match="could not be stopped"):
                    await lifecycle._ensure_coturn()
                gc.collect()
                assert lifecycle.coturn.poll() is None
                assert directories[0].exists()
                assert (directories[0] / "turnserver.conf").is_file()
                assert not starts
                assert not lifecycle._turn_listener_ready
    finally:
        lifecycle.coturn.terminate()
        lifecycle.coturn.wait(timeout=5)
        adb.stop_coturn(lifecycle.coturn)
    assert not directories[0].exists()
