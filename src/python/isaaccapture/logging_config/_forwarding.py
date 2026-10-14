# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Forward Python and C++ records to one session leader over a Unix socket.

The leader owns console and file handlers and publishes
``ISAACCAPTURE_LOG_SOCKET``. Python children use :class:`ForwardingHandler`;
C++ processes use the same framed JSON protocol.
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import select
import socket
import socketserver
import stat
import struct
import threading
import time
from pathlib import Path

from ._core import _move_above_std, ensure_private_dir, root_logger

# Match the POSIX-only C++ forwarding transport.
_HAS_UNIX_SOCKETS = os.name == "posix" and hasattr(socket, "AF_UNIX")

# Common sockaddr_un path limit, excluding the terminator.
_MAX_SOCKET_PATH = 103


def _runtime_dir() -> Path:
    """Return a short per-user directory independent of the log directory."""
    xdg = os.environ.get("XDG_RUNTIME_DIR")
    if xdg:
        return Path(xdg) / "isaaccapture"
    return Path(f"/tmp/isaaccapture-{os.getuid()}")


_FRAME_HEADER = struct.Struct(">I")  # 4-byte big-endian payload length prefix

# Only ever used for its formatException(); the leader applies the real format.
_EXC_FORMATTER = logging.Formatter()


def _format_exception(record: logging.LogRecord) -> str | None:
    """The record's traceback as text, or ``None`` when it carries no exception."""
    if record.exc_text:
        return record.exc_text
    if record.exc_info:
        return _EXC_FORMATTER.formatException(record.exc_info)
    return None


# Bound allocation from malformed frame lengths.
_MAX_FRAME_SIZE = 1 * 1024 * 1024  # 1 MiB

_lock = threading.Lock()


_verified_path: str | None = None


def _move_socket_above_std(sock: socket.socket) -> socket.socket:
    """Keep a logging socket from occupying a host-closed fd 0/1/2."""
    if sock.fileno() > 2:
        return sock
    family, kind, proto, timeout = sock.family, sock.type, sock.proto, sock.gettimeout()
    fd = _move_above_std(sock.detach())
    try:
        moved = socket.socket(family, kind, proto, fileno=fd)
        moved.settimeout(timeout)
        return moved
    except BaseException:
        try:
            os.close(fd)
        except OSError:
            pass
        raise


def _unix_socket() -> socket.socket:
    return _move_socket_above_std(socket.socket(socket.AF_UNIX, socket.SOCK_STREAM))


def _reachable(path: str) -> bool:
    """Whether anything is still accepting connections at *path*."""
    try:
        with _unix_socket() as probe:
            probe.settimeout(1.0)
            probe.connect(path)
        return True
    except OSError:
        return False


def socket_path() -> str | None:
    """Return a verified inherited socket, or ``None`` to become the leader."""
    global _verified_path
    if not _HAS_UNIX_SOCKETS:
        return None
    path = os.environ.get("ISAACCAPTURE_LOG_SOCKET") or None
    if path is None or path == _verified_path:
        return path
    if _reachable(path):
        _verified_path = path
        return path
    # Unset dead addresses for both Python and C++; pop tolerates racing callers.
    os.environ.pop("ISAACCAPTURE_LOG_SOCKET", None)
    return None


class ForwardingHandler(logging.Handler):
    """Best-effort sender that drops records when the leader is unreachable."""

    def __init__(self, path: str) -> None:
        super().__init__()
        self._socket_path = path
        self._sock: socket.socket | None = None
        self._send_lock = threading.Lock()

    def _connect(self) -> socket.socket | None:
        try:
            sock = _unix_socket()
            sock.settimeout(1.0)
            sock.connect(self._socket_path)
            return sock
        except OSError:
            return None

    def emit(self, record: logging.LogRecord) -> None:
        try:
            payload = json.dumps(
                {
                    "name": record.name,
                    "levelno": record.levelno,
                    "msg": record.getMessage(),
                    "created": record.created,
                    "process": record.process,
                    # Traceback objects do not survive JSON.
                    "exc_text": _format_exception(record),
                }
            ).encode("utf-8")
        except Exception:  # noqa: BLE001 -- Handler.emit()'s own documented contract
            self.handleError(record)
            return
        if len(payload) > _MAX_FRAME_SIZE:
            return
        frame = _FRAME_HEADER.pack(len(payload)) + payload
        with self._send_lock:
            if self._sock is None:
                self._sock = self._connect()
                if self._sock is None:
                    return
            try:
                self._sock.sendall(frame)
            except OSError:
                self._sock.close()
                self._sock = None


_handler: logging.Handler | None = None


def ensure_handler(path: str) -> logging.Handler:
    """Create and attach this process's single forwarding handler; idempotent."""
    global _handler
    if _handler is not None:
        return _handler
    with _lock:
        if _handler is not None:
            return _handler
        handler = ForwardingHandler(path)
        root_logger.addHandler(handler)
        _handler = handler
        return _handler


# Open receiver connections, and frames taken off them but not yet re-emitted:
# _drain() waits on both so records already sent are not lost at exit.
_connections: set[socket.socket] = set()
_in_flight = 0
_drain_lock = threading.Lock()


class RequestHandler(socketserver.StreamRequestHandler):
    """Re-emit one child's framed JSON records through the leader's logger tree."""

    # Unbuffered: a frame read ahead into Python would be invisible to _drain().
    rbufsize = 0

    def handle(self) -> None:
        global _in_flight
        while True:
            try:
                # Wait for the next frame without consuming it, so it is never
                # both off the socket and uncounted.
                if not self.request.recv(1, socket.MSG_PEEK):
                    return
            except OSError:
                return
            with _drain_lock:
                _in_flight += 1
            try:
                if not self._handle_frame():
                    return
            finally:
                with _drain_lock:
                    _in_flight -= 1

    def _handle_frame(self) -> bool:
        """Re-emit one frame; ``False`` ends the connection."""
        header = self._recv_exact(_FRAME_HEADER.size)
        if header is None:
            return False
        (length,) = _FRAME_HEADER.unpack(header)
        if length > _MAX_FRAME_SIZE:
            return False  # corrupted stream; drop the connection rather than buffer it
        body = self._recv_exact(length)
        if body is None:
            return False
        try:
            # Preserve records containing non-UTF-8 vendor text.
            payload = json.loads(body.decode("utf-8", errors="replace"))
            record = logging.makeLogRecord(
                {
                    "name": payload["name"],
                    "levelno": payload["levelno"],
                    "levelname": logging.getLevelName(payload["levelno"]),
                    "msg": payload["msg"],
                    "created": payload["created"],
                    # Recompute milliseconds from the sender's timestamp.
                    "msecs": (payload["created"] - int(payload["created"])) * 1000,
                    "process": payload["process"],
                    # Absent from the C++ sender, which has no exceptions.
                    "exc_text": payload.get("exc_text"),
                }
            )
            # The sender already applied its logger level; handlers filter here.
            logging.getLogger(payload["name"]).handle(record)
        except (KeyError, ValueError, UnicodeDecodeError, TypeError):
            pass  # malformed frame; drop it and keep the connection alive
        return True

    def _recv_exact(self, n: int) -> bytes | None:
        buf = bytearray()
        while len(buf) < n:
            chunk = self.rfile.read(n - len(buf))
            if not chunk:
                return None
            buf.extend(chunk)
        return bytes(buf)


# Defining this class is itself unsafe where UnixStreamServer is absent.
if _HAS_UNIX_SOCKETS:

    class ThreadingUnixStreamServer(
        socketserver.ThreadingMixIn, socketserver.UnixStreamServer
    ):
        daemon_threads = True

        def server_bind(self) -> None:
            self.socket = _move_socket_above_std(self.socket)
            super().server_bind()

        def get_request(self):
            # Accept and register as one step, so _drain() never sees a
            # connection that has left the backlog but is not yet counted.
            with _drain_lock:
                request, address = super().get_request()
                request = _move_socket_above_std(request)
                _connections.add(request)
            return request, address

        def shutdown_request(self, request) -> None:
            with _drain_lock:
                _connections.discard(request)
            super().shutdown_request(request)


_receiver_socket: str | None = None

# Bounds how long exit waits on a sender that keeps writing.
_DRAIN_TIMEOUT = 1.0


def _has_unread(sock: socket.socket) -> bool:
    try:
        return bool(sock.recv(1, socket.MSG_PEEK | socket.MSG_DONTWAIT))
    except OSError:  # nothing queued, or the connection is gone
        return False


def _drain(server) -> None:
    """Wait until every record already sent here has been re-emitted.

    In-process C++ loggers reach Python only through this receiver, so a record
    logged just before exit is otherwise lost from both console and file.
    """
    backlog = select.poll()
    backlog.register(server.socket, select.POLLIN)
    deadline = time.monotonic() + _DRAIN_TIMEOUT
    while True:
        with _drain_lock:
            idle = (
                _in_flight == 0
                and not backlog.poll(0)
                and not any(_has_unread(s) for s in _connections)
            )
        if idle or time.monotonic() >= deadline:
            return
        time.sleep(0.01)


def _release_receiver(
    server, path: str, owner_pid: int, socket_identity: tuple[int, int]
) -> None:
    """Stop and unlink the receiver only in its creator process."""
    if os.getpid() != owner_pid:
        return
    _drain(server)
    server.shutdown()
    try:
        current = os.lstat(path)
        if (
            stat.S_ISSOCK(current.st_mode)
            and (
                current.st_dev,
                current.st_ino,
            )
            == socket_identity
        ):
            os.unlink(path)
    except OSError:
        pass  # already gone, replaced, or a directory we can no longer write


def _no_receiver(reason: str) -> str:
    """Warn that forwarding is disabled and return the empty-path sentinel."""
    root_logger.warning(
        "Log forwarding disabled: %s. Each process will keep its own console "
        "and log file.",
        reason,
    )
    return ""


def ensure_receiver() -> str:
    """Start the leader's receiver once, publish its path, and never raise."""
    global _receiver_socket
    if not _HAS_UNIX_SOCKETS:
        return ""
    if _receiver_socket is not None:
        return _receiver_socket
    with _lock:
        if _receiver_socket is not None:
            return _receiver_socket
        try:
            directory = ensure_private_dir(_runtime_dir())
        except OSError as exc:
            return _no_receiver(f"cannot use the runtime directory ({exc})")
        # A live PID is unique and keeps the constrained socket path short.
        path = str(directory / f"isaaccapture.{os.getpid()}.sock")
        path_bytes = len(os.fsencode(path))
        if path_bytes > _MAX_SOCKET_PATH:
            return _no_receiver(
                f"socket path is {path_bytes} bytes, over the {_MAX_SOCKET_PATH} "
                f"a Unix domain socket allows ({path})"
            )
        server = None
        try:
            # lexists also removes stale symlinks that still occupy the name.
            if os.path.lexists(path):
                os.unlink(path)
            server = ThreadingUnixStreamServer(path, RequestHandler)
            # Restrict the socket before accepting records.
            os.chmod(path, 0o600)
            socket_info = os.lstat(path)
            socket_identity = (socket_info.st_dev, socket_info.st_ino)
        except OSError as exc:
            if server is not None:
                try:
                    server.server_close()
                except OSError:
                    pass
            try:
                os.unlink(path)
            except OSError:
                pass
            return _no_receiver(f"cannot bind {path} ({exc})")
        thread = threading.Thread(
            target=server.serve_forever, name="isaaccapture-log-receiver", daemon=True
        )
        try:
            thread.start()
        except RuntimeError as exc:
            try:
                server.server_close()
            except OSError:
                pass
            try:
                os.unlink(path)
            except OSError:
                pass
            return _no_receiver(f"cannot start the receiver thread ({exc})")
        atexit.register(_release_receiver, server, path, os.getpid(), socket_identity)
        os.environ["ISAACCAPTURE_LOG_SOCKET"] = path
        _receiver_socket = path
        return path
