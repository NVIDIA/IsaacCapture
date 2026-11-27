# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import os
import socket
import threading
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from isaac_media_transport.capture import CaptureEndpoint
from isaac_media_transport.gpu_ipc import (
    IpcServer,
    PROTOCOL,
    finish_import,
    lock_endpoint,
    receive,
    send,
    require_same_user,
    validate_allocation,
)
from isaac_media_transport.latency import open_trace


def endpoint():
    result = CaptureEndpoint.__new__(CaptureEndpoint)
    result.lock = threading.Lock()
    result.closed = threading.Event()
    result.job = None
    result.tracks = {}
    return result


@pytest.mark.parametrize("failure", [None, "write", "deactivate", "directory"])
def test_capture_releases_admission_after_deactivation(tmp_path, failure):
    capture = endpoint()
    transitions = []

    def active(value):
        transitions.append(value)
        assert capture.job is not None
        if not value:
            with pytest.raises(RuntimeError, match="already in progress"):
                capture.capture({"output": str(tmp_path / "overlap")})
            if failure == "deactivate":
                raise RuntimeError("callback failed")

    capture.register("cam", stage="test", width=1, height=1, fps=30, on_active=active)

    def write(job, spec, manifest, output, seconds):
        if failure == "write":
            raise RuntimeError("write failed")
        manifest["frames"].append({})

    capture._write = write
    output = tmp_path / "capture"
    if failure == "directory":
        output.mkdir()
    if failure:
        with pytest.raises((RuntimeError, FileExistsError)):
            capture.capture({"output": str(output)})
    else:
        capture.capture({"output": str(output)})
    assert transitions == ([] if failure == "directory" else [True, False])
    assert capture.job is None


def test_trace_rejects_symlink_and_hardlink(tmp_path):
    target = tmp_path / "target"
    target.write_text("keep")
    link = tmp_path / "trace"
    link.symlink_to(target)
    with pytest.raises(OSError):
        open_trace(link)
    link.unlink()
    os.link(target, link)
    with pytest.raises(PermissionError):
        open_trace(link)
    assert target.read_text() == "keep"


def test_trace_private_permissions(tmp_path):
    path = tmp_path / "trace"
    with open_trace(path) as stream:
        stream.write("event\n")
    assert path.stat().st_mode & 0o777 == 0o600


def test_endpoint_reclaims_stale_socket_but_not_live_owner(tmp_path):
    path = tmp_path / "ipc.sock"
    with socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET) as stale:
        stale.bind(str(path))
    fd = lock_endpoint(path)
    try:
        assert not path.exists()
        with pytest.raises(BlockingIOError):
            lock_endpoint(path)
    finally:
        os.close(fd)
    with socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET) as live:
        live.bind(str(path))
        live.listen(1)
        with pytest.raises(FileExistsError):
            lock_endpoint(path)
        assert path.exists()


def test_import_authenticates_exporter():
    fake = SimpleNamespace(
        getsockopt=lambda *args: __import__("struct").pack("3i", 1, os.getuid() + 1, 1)
    )
    with pytest.raises(PermissionError):
        require_same_user(fake)


def test_extent_checked_independently_of_descriptor(monkeypatch):
    def query(base, extent, pointer):
        base._obj.value = 100
        extent._obj.value = 64
        return 0

    monkeypatch.setattr(
        "isaac_media_transport.gpu_ipc._address_range_function", lambda: query
    )
    validate_allocation(100, 64)
    with pytest.raises(ValueError):
        validate_allocation(100, 65)


def test_bye_racing_offer_does_not_quarantine(tmp_path):
    server = IpcServer.__new__(IpcServer)
    server.path = tmp_path / "ipc.sock"
    server.cp = SimpleNamespace(cuda=SimpleNamespace(Device=lambda gpu: nullcontext()))
    server.receiver = SimpleNamespace(latest=lambda: object(), overwritten=0)
    server.tracks, server.gpu = ["cam"], 0
    server.quarantine, server.slots, server.sent = [], object(), 0
    exporting, release = threading.Event(), threading.Event()

    def export(sample, seq):
        exporting.set()
        assert release.wait(2)
        return {"protocol": PROTOCOL, "seq": seq}

    server._export = export
    stop = threading.Event()
    errors = []

    def run():
        try:
            server.run(stop)
        except Exception as exc:
            errors.append(exc)

    thread = threading.Thread(target=run)
    thread.start()
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET) as client:
            import time

            deadline = time.monotonic() + 2
            while True:
                try:
                    client.connect(str(server.path))
                    break
                except (FileNotFoundError, ConnectionRefusedError):
                    assert time.monotonic() < deadline
                    time.sleep(0.01)
            send(client, {"protocol": PROTOCOL, "tracks": ["cam"], "gpu": 0})
            assert exporting.wait(2)
            # Ensure bye reaches the server while an offer is being prepared.
            send(client, {"bye": True})
            release.set()
            message = receive(client)
            send(client, {"decline": message["seq"]})
            assert receive(client) == {"bye": True}
        assert server.quarantine == []
    finally:
        stop.set()
        release.set()
        thread.join(3)
    assert not thread.is_alive()
    assert not errors


def test_import_shutdown_declines_unmapped_offer():
    client, server = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    with client, server:
        thread = threading.Thread(target=finish_import, args=(client,))
        thread.start()
        assert receive(server) == {"bye": True}
        send(server, {"protocol": PROTOCOL, "seq": 4})
        assert receive(server) == {"decline": 4}
        send(server, {"bye": True})
        thread.join(2)
        assert not thread.is_alive()
