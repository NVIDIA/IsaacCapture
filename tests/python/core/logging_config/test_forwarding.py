# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The cross-process transport: the frame, the receiver, and losing either end.

Records are handed to ``ForwardingHandler.emit`` directly rather than through a
logger: the receiver re-emits under the sender's own logger name, so a handler
attached to that name in this process would ship the record straight back.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import socket
import stat
import struct
import subprocess
import sys
import tempfile
import threading

import pytest
from conftest import (
    clean_env,
    cpp_logs,
    emitter_path,
    requires_emitter,
    run_emitter,
    run_python,
    wait_until,
)

from isaaccapture.logging_config import _forwarding

_HEADER = struct.Struct(">I")


def emit(handler, name, level=logging.INFO, message="msg", args=None, exc_info=None):
    handler.emit(logging.LogRecord(name, level, __file__, 1, message, args, exc_info))


def raw_frames(path: str, *payloads: bytes) -> None:
    """Send pre-built frames down one connection, as a C++ sender would."""
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        sock.settimeout(5.0)
        sock.connect(path)
        for payload in payloads:
            sock.sendall(_HEADER.pack(len(payload)) + payload)


def payload(name: str, message: str = "msg", levelno: int = logging.INFO) -> bytes:
    return json.dumps(
        {
            "name": name,
            "levelno": levelno,
            "msg": message,
            "created": 1_700_000_000.5,
            "process": 4242,
        }
    ).encode("utf-8")


class TestFrame:
    def test_a_record_arrives_whole(self, receiver):
        name = "isaaccapture.test.forward.whole"
        handler = _forwarding.ForwardingHandler(receiver.path)
        try:
            raise RuntimeError("boom")
        except RuntimeError:
            emit(handler, name, logging.ERROR, "failed %s", ("hard",), sys.exc_info())

        record = receiver.collector.wait_for(name, 1)[0]
        assert record.getMessage() == "failed hard"
        assert record.levelno == logging.ERROR
        assert record.levelname == "ERROR"
        assert record.process == os.getpid()
        # A traceback object does not survive JSON, so the child renders it.
        assert "RuntimeError: boom" in record.exc_text

    def test_the_senders_milliseconds_are_not_replaced_by_the_receivers(self, receiver):
        name = "isaaccapture.test.forward.clock"
        handler = _forwarding.ForwardingHandler(receiver.path)
        record = logging.LogRecord(name, logging.INFO, __file__, 1, "tick", None, None)
        record.created = 1_700_000_000.25
        handler.emit(record)

        arrived = receiver.collector.wait_for(name, 1)[0]
        assert int(arrived.created) == 1_700_000_000
        assert abs(arrived.msecs - 250.0) < 1.0

    def test_bytes_that_are_not_utf8_arrive_replaced_rather_than_dropped(
        self, receiver
    ):
        # The C++ sender copies >= 0x80 through verbatim; the strings it carries
        # (vendor text, strerror, paths) are not guaranteed UTF-8.
        name = "isaaccapture.test.forward.mojibake"
        raw_frames(
            receiver.path, payload(name, "cafLATIN1").replace(b"LATIN1", b"\xe9")
        )
        assert "\ufffd" in receiver.collector.wait_for(name, 1)[0].getMessage()


class TestReceiver:
    def test_serves_concurrent_senders(self, receiver):
        name = "isaaccapture.test.forward.concurrent"
        senders, per_sender = 8, 25
        start = threading.Barrier(senders)

        # Connections are established and confirmed before the measurement.
        # A first connect gives up after a second rather than queue, and on a
        # loaded machine it does; that drop is the sender's documented
        # best-effort contract, not what this test is about.
        handlers = [
            _forwarding.ForwardingHandler(receiver.path) for _ in range(senders)
        ]
        for index, handler in enumerate(handlers):
            marker = f"{name}.warmup.{index}"
            for _attempt in range(30):
                emit(handler, marker, message="ready")
                if wait_until(
                    lambda m=marker: receiver.collector.named(m), timeout=2.0
                ):
                    break
            else:
                pytest.fail(f"sender {index} never reached the receiver")

        def ship(index: int) -> None:
            start.wait()
            for record in range(per_sender):
                emit(handlers[index], name, message=f"sender {index} record {record}")

        threads = [threading.Thread(target=ship, args=(i,)) for i in range(senders)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=60)
            assert not thread.is_alive()

        arrived = receiver.collector.wait_for(name, senders * per_sender)
        assert len(arrived) == senders * per_sender

    def test_a_malformed_frame_is_dropped_and_the_connection_kept(self, receiver):
        name = "isaaccapture.test.forward.malformed"
        raw_frames(receiver.path, b"{not json", payload(name, "after the bad one"))
        assert receiver.collector.wait_for(name, 1)[0].getMessage() == (
            "after the bad one"
        )

    def test_an_oversized_length_prefix_costs_only_its_connection(self, receiver):
        name = "isaaccapture.test.forward.oversized"
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.connect(receiver.path)
            sock.sendall(_HEADER.pack(8 * 1024 * 1024) + b"x" * 64)

        raw_frames(receiver.path, payload(name))
        assert len(receiver.collector.wait_for(name, 1)) == 1

    def test_a_sender_killed_mid_frame_leaves_the_receiver_serving(
        self, receiver, tmp_path
    ):
        name = "isaaccapture.test.forward.sigkill"
        half = payload(name, "never arrives")
        child = run_python(
            "import os, signal, socket, struct, sys\n"
            "sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)\n"
            "sock.connect(sys.argv[1])\n"
            "body = sys.argv[2].encode()\n"
            "sock.sendall(struct.pack('>I', len(body)) + body[:8])\n"
            "os.kill(os.getpid(), signal.SIGKILL)\n",
            clean_env(tmp_path),
            receiver.path,
            half.decode(),
        )
        assert child.returncode == -9

        raw_frames(receiver.path, payload(name, "after the kill"))
        arrived = receiver.collector.wait_for(name, 1)
        assert [r.getMessage() for r in arrived] == ["after the kill"]


class TestAddressVerification:
    def test_a_dead_address_is_unset_rather_than_believed(self, monkeypatch, tmp_path):
        # A forwarding process installs no console and no file handler, so one
        # that trusted a dead address would emit nothing, anywhere.
        dead = tmp_path / "dead.sock"
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.bind(str(dead))
        sock.close()

        monkeypatch.setattr(_forwarding, "_verified_path", None)
        monkeypatch.setenv("ISAACCAPTURE_LOG_SOCKET", str(dead))
        assert _forwarding.socket_path() is None
        assert "ISAACCAPTURE_LOG_SOCKET" not in os.environ

    def test_a_live_address_is_accepted(self, monkeypatch, receiver):
        monkeypatch.setattr(_forwarding, "_verified_path", None)
        monkeypatch.setenv("ISAACCAPTURE_LOG_SOCKET", receiver.path)
        assert _forwarding.socket_path() == receiver.path

    def test_the_handler_drops_records_it_cannot_deliver(self, tmp_path):
        # Best-effort: dropped, never raised, never queued -- losing a line
        # during a hiccup beats blocking the emitting thread.
        handler = _forwarding.ForwardingHandler(str(tmp_path / "nothing.sock"))
        emit(handler, "isaaccapture.test.forward.nowhere", message="dropped")
        emit(handler, "isaaccapture.test.forward.nowhere", message="dropped again")
        assert handler._sock is None


class TestReleaseReceiver:
    """Only the process that created a receiver may stop or unlink it."""

    @staticmethod
    def _serving():
        directory = tempfile.mkdtemp(prefix="itlog", dir="/tmp")
        path = os.path.join(directory, "r.sock")
        server = _forwarding.ThreadingUnixStreamServer(path, _forwarding.RequestHandler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        info = os.lstat(path)
        return server, path, directory, (info.st_dev, info.st_ino)

    def test_a_process_that_did_not_create_it_leaves_it_running(self, receiver):
        # fork() copies the atexit registration but not serve_forever's thread:
        # shutdown() there waits on an Event nothing will set, and the unlink
        # takes the socket the leader is still listening on.
        name = "isaaccapture.test.forward.notmine"
        _forwarding._release_receiver(
            receiver.server, receiver.path, os.getpid() + 1, (0, 0)
        )
        assert stat.S_ISSOCK(os.lstat(receiver.path).st_mode)
        raw_frames(receiver.path, payload(name))
        assert len(receiver.collector.wait_for(name, 1)) == 1

    def test_it_will_not_unlink_whatever_took_its_name(self):
        server, path, directory, identity = self._serving()
        try:
            os.unlink(path)
            with open(path, "w", encoding="utf-8") as replacement:
                replacement.write("someone else's\n")

            _forwarding._release_receiver(server, path, os.getpid(), identity)
            assert os.path.isfile(path)
        finally:
            server.server_close()
            shutil.rmtree(directory, ignore_errors=True)


@requires_emitter
class TestAbnormalTermination:
    """The transport a standalone C++ process shares with the Python half."""

    def test_a_record_sent_before_abort_still_reaches_the_leader(
        self, receiver, tmp_path
    ):
        # sink_it_ sends synchronously, so nothing is waiting in a buffer the
        # abort would take with it.
        name = "isaaccapture.test.cpp.Abort"
        result = run_emitter(
            ["abort", name, "error", "last words"],
            clean_env(tmp_path, ISAACCAPTURE_LOG_SOCKET=receiver.path),
        )
        assert result.returncode == -6

        record = receiver.collector.wait_for(name, 1)[0]
        assert record.getMessage() == "last words"
        assert record.levelno == logging.ERROR

    def test_a_leader_that_dies_mid_stream_does_not_take_the_child_down(
        self, leader, tmp_path
    ):
        # socket_path() only covers startup; a leader killed mid-run takes its
        # children's remaining records with it, and must cost them nothing else.
        name = "isaaccapture.test.cpp.Spin"
        child_logs = tmp_path / "child-logs"
        child_logs.mkdir()
        child = subprocess.Popen(
            [emitter_path(), "spin", name, "60", "50"],
            env=clean_env(child_logs, ISAACCAPTURE_LOG_SOCKET=leader.socket),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            assert "spin 1" in leader.wait_for("spin 1")
            leader.kill()
            assert child.wait(timeout=90) == 0
        finally:
            if child.poll() is None:
                child.kill()
                child.wait()

        # No local fallback is grown on the way down: local_sinks() resolved to
        # the forwarding sink alone, once, at the first record.
        assert cpp_logs(child_logs) == []


def test_receiver_permissions_are_private_before_listening(monkeypatch):
    activate = _forwarding.ThreadingUnixStreamServer.server_activate
    observed = []

    def inspect_before_listen(server):
        mode = stat.S_IMODE(os.stat(server.server_address).st_mode)
        observed.append(mode)
        assert mode == 0o600
        activate(server)

    monkeypatch.setattr(
        _forwarding.ThreadingUnixStreamServer, "server_activate", inspect_before_listen
    )
    with tempfile.TemporaryDirectory(prefix="itlog") as directory:
        os.chmod(directory, 0o755)
        previous = os.umask(0)
        try:
            with _forwarding.ThreadingUnixStreamServer(
                os.path.join(directory, "r.sock"), _forwarding.RequestHandler
            ):
                assert observed == [0o600]
        finally:
            os.umask(previous)


def test_ensure_receiver_is_idempotent_and_publishes_its_address():
    # This process is already a leader; a second call must return the same
    # address rather than binding a second socket.
    published = os.environ.get("ISAACCAPTURE_LOG_SOCKET")
    if not published:
        pytest.skip("this interpreter could not start a receiver")
    assert _forwarding.ensure_receiver() == published
    assert stat.S_ISSOCK(os.lstat(published).st_mode)
    assert stat.S_IMODE(os.lstat(published).st_mode) == 0o600
