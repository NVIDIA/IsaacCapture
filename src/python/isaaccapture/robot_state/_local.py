# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The local connection between a reader program and the Teleop Session Server.

A Unix socket of this user, on the server machine. Its format is private to this package and may
change in any release; it is not an interface. Use the calls of `isaaccapture.robot_state`.
"""

from __future__ import annotations

import logging
import os
import selectors
import socket
import stat
import struct
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING

from isaaccapture.logging_config._core import ensure_private_dir

if TYPE_CHECKING:
    from ._robot_state import RobotStateStore

logger = logging.getLogger(__name__)

__all__ = ["RobotStateLocalService", "default_socket_path"]

MAGIC = b"ICRS"
VERSION = 1
OP_SNAPSHOT = 1
OP_ANSWER = 2
# magic, version, op, zero, known_setup_generation
REQUEST = struct.Struct("<4sBBHQ")
# magic, version, op, zero, length of the snapshot that follows
ANSWER = struct.Struct("<4sBBHI")

MAX_CLIENTS = 16
MAX_ANSWER_BYTES = 32 * 1024 * 1024
# Common sockaddr_un path limit, excluding the terminator (as logging_config uses).
MAX_SOCKET_PATH = 103
SOCKET_NAME = "robot_state.sock"
THREAD_NAME = "isaaccapture-robot-state-readers"

_HAS_UNIX_SOCKETS = os.name == "posix" and hasattr(socket, "AF_UNIX")
_NO_UNIX_SOCKETS = "this platform has no Unix domain sockets"
_ACCEPT = "accept"
_WAKE = "wake"
_CLOSE_JOIN_S = 5.0


class NotThisVersion(Exception):
    """The server answered in a format this version of the package does not read."""


def default_socket_path() -> str:
    """Where the Teleop Session Server serves robot state to this user's readers by default."""
    xdg = os.environ.get("XDG_RUNTIME_DIR")
    if xdg and os.path.isdir(xdg):
        return str(Path(xdg) / "isaaccapture" / SOCKET_NAME)
    uid = os.getuid() if hasattr(os, "getuid") else 0
    return str(Path(f"/tmp/isaaccapture-{uid}") / SOCKET_NAME)


def _path_problem(path: str) -> str | None:
    """Why `path` cannot name a Unix socket here; None when it can."""
    if not _HAS_UNIX_SOCKETS:
        return _NO_UNIX_SOCKETS
    size = len(os.fsencode(path))
    if size > MAX_SOCKET_PATH:
        return (
            f"socket path is {size} bytes, over the {MAX_SOCKET_PATH} a Unix domain "
            f"socket allows ({path})"
        )
    return None


def _check_private_dir(directory: Path) -> None:
    """Raise OSError unless `directory` belongs to this user and is closed to everybody else."""
    info = os.stat(directory)
    if not stat.S_ISDIR(info.st_mode):
        raise NotADirectoryError(f"{directory} is not a directory")
    if info.st_uid != os.getuid():
        raise PermissionError(
            f"{directory} belongs to another user; robot state is served only from a "
            "directory of the user who runs the Teleop Session Server"
        )
    if info.st_mode & 0o077:
        raise PermissionError(
            f"{directory} is open to group or others (mode {stat.S_IMODE(info.st_mode):o}); "
            "robot state is served only from a directory closed to them (mode 700)"
        )


def _remove_stale_socket(path: str) -> None:
    """Remove a socket file nobody serves. OSError when one is served or `path` is no socket."""
    try:
        info = os.lstat(path)
    except FileNotFoundError:
        return
    if not stat.S_ISSOCK(info.st_mode):
        raise FileExistsError(f"{path} exists and is not a socket")
    probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        probe.settimeout(1.0)
        probe.connect(path)
    except (ConnectionRefusedError, FileNotFoundError):
        pass
    except OSError as error:
        raise OSError(
            f"another Teleop Session Server serves robot state at {path} ({error})"
        ) from None
    else:
        raise OSError(f"another Teleop Session Server serves robot state at {path}")
    finally:
        probe.close()
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass


def _peer_uid(connection: socket.socket) -> int | None:
    """The user at the other end; None where the platform does not tell."""
    option = getattr(socket, "SO_PEERCRED", None)
    if option is None:
        return None
    credentials = connection.getsockopt(
        socket.SOL_SOCKET, option, struct.calcsize("3i")
    )
    _pid, uid, _gid = struct.unpack("3i", credentials)
    return uid


class _Reader:
    """One accepted reader: its unread request bytes and its unsent answer."""

    def __init__(self) -> None:
        self.inbox = bytearray()
        self.outbox = b""
        self.sent = 0
        self.events = selectors.EVENT_READ


class RobotStateLocalService:
    """Serves snapshots of a `RobotStateStore` to this user's reader programs on this machine.

    One thread of its own answers every reader. The socket's directory must belong to this user
    and be closed to group and others.
    """

    def __init__(
        self, store: RobotStateStore, socket_path: str | os.PathLike | None = None
    ) -> None:
        self._store = store
        self._path = (
            default_socket_path() if socket_path is None else os.fspath(socket_path)
        )
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._listener: socket.socket | None = None
        self._wake: tuple[socket.socket, socket.socket] | None = None
        self._identity: tuple[int, int] | None = None
        self._uid = 0
        self._closed = False
        self._reported: set[str] = set()

    @property
    def path(self) -> str:
        """The socket's path."""
        return self._path

    def start(self) -> None:
        """Bind the socket and start the thread; returns at once.

        OSError when the path is too long, its directory is not this user's alone, or another
        server serves robot state there. A socket file nobody serves is replaced.
        """
        with self._lock:
            if self._closed:
                raise RuntimeError("RobotStateLocalService is closed")
            if self._thread is not None:
                return
            path = self._path
            problem = _path_problem(path)
            if problem is not None:
                raise OSError(problem)
            directory = ensure_private_dir(Path(path).parent)
            _check_private_dir(directory)
            _remove_stale_socket(path)
            self._uid = os.getuid()
            listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            wake = None
            bound = False
            try:
                listener.bind(path)
                bound = True
                os.chmod(path, 0o600)
                info = os.lstat(path)
                listener.listen(MAX_CLIENTS)
                listener.setblocking(False)
                wake = socket.socketpair()
                thread = threading.Thread(
                    target=self._serve,
                    args=(listener, wake[0]),
                    name=THREAD_NAME,
                    daemon=True,
                )
                thread.start()
            except BaseException:
                listener.close()
                for end in wake or ():
                    end.close()
                if bound:
                    _unlink_quietly(path)
                raise
            self._listener = listener
            self._wake = wake
            self._identity = (info.st_dev, info.st_ino)
            self._thread = thread
        logger.info("Serving robot state to reader programs at %s", path)

    def close(self) -> None:
        """Stop the thread, close every reader's connection and remove the socket file. Idempotent."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            thread, self._thread = self._thread, None
            listener, self._listener = self._listener, None
            wake, self._wake = self._wake, None
            identity, self._identity = self._identity, None
        if thread is None:
            return
        try:
            wake[1].send(b"\0")
        except OSError:
            pass
        thread.join(_CLOSE_JOIN_S)
        listener.close()
        for end in wake:
            end.close()
        # Another server may have taken the path over; its file is not this one's to remove.
        try:
            info = os.lstat(self._path)
        except OSError:
            return
        if (info.st_dev, info.st_ino) == identity:
            _unlink_quietly(self._path)

    def _serve(self, listener: socket.socket, wake: socket.socket) -> None:
        readers: dict[socket.socket, _Reader] = {}
        selector = selectors.DefaultSelector()
        try:
            selector.register(listener, selectors.EVENT_READ, _ACCEPT)
            selector.register(wake, selectors.EVENT_READ, _WAKE)
            while True:
                for key, events in selector.select():
                    if key.data is _WAKE:
                        return
                    if key.data is _ACCEPT:
                        self._accept(listener, selector, readers)
                        continue
                    connection = key.fileobj
                    try:
                        keep = self._exchange(
                            selector, connection, readers[connection], events
                        )
                    except OSError:
                        keep = False
                    if not keep:
                        selector.unregister(connection)
                        del readers[connection]
                        connection.close()
        except Exception:
            logger.exception(
                "Serving robot state at %s stopped; reader programs get no answer",
                self._path,
            )
        finally:
            for connection in readers:
                connection.close()
            selector.close()

    def _accept(
        self,
        listener: socket.socket,
        selector: selectors.BaseSelector,
        readers: dict[socket.socket, _Reader],
    ) -> None:
        try:
            connection, _ = listener.accept()
        except OSError:
            return
        try:
            uid = _peer_uid(connection)
            if uid is not None and uid != self._uid:
                self._report_once(
                    "user",
                    "Closed a robot-state reader of another user (uid %d) at %s",
                    uid,
                    self._path,
                )
                connection.close()
                return
            if len(readers) >= MAX_CLIENTS:
                self._report_once(
                    "clients",
                    "Closed a robot-state reader at %s: %d are connected already",
                    self._path,
                    len(readers),
                )
                connection.close()
                return
            connection.setblocking(False)
            reader = _Reader()
            selector.register(connection, reader.events)
        except OSError:
            connection.close()
            return
        readers[connection] = reader

    def _exchange(
        self,
        selector: selectors.BaseSelector,
        connection: socket.socket,
        reader: _Reader,
        events: int,
    ) -> bool:
        """Read requests and send answers as far as the socket allows; False to close it."""
        if events & selectors.EVENT_READ:
            try:
                data = connection.recv(4096)
            except BlockingIOError:
                data = None
            if data is not None:
                if not data:
                    return False
                reader.inbox += data
        while True:
            if reader.sent < len(reader.outbox):
                try:
                    reader.sent += connection.send(
                        memoryview(reader.outbox)[reader.sent :]
                    )
                except BlockingIOError:
                    break
                if reader.sent < len(reader.outbox):
                    break
            if len(reader.inbox) < REQUEST.size:
                break
            answer = self._answer(bytes(reader.inbox[: REQUEST.size]))
            if answer is None:
                return False
            del reader.inbox[: REQUEST.size]
            reader.outbox = answer
            reader.sent = 0
        # While an answer waits for the reader to take it, no further request is read.
        wanted = (
            selectors.EVENT_WRITE
            if reader.sent < len(reader.outbox)
            else selectors.EVENT_READ
        )
        if wanted != reader.events:
            selector.modify(connection, wanted)
            reader.events = wanted
        return True

    def _answer(self, request: bytes) -> bytes | None:
        """The answer to one request; None for a request of another format."""
        magic, version, op, _zero, known_setup_generation = REQUEST.unpack(request)
        if magic != MAGIC or version != VERSION or op != OP_SNAPSHOT:
            return None
        snapshot = self._store.export_snapshot(known_setup_generation)
        if len(snapshot) > MAX_ANSWER_BYTES:
            self._report_once(
                "size",
                "Robot state of %d bytes is above the %d a reader program takes at %s",
                len(snapshot),
                MAX_ANSWER_BYTES,
                self._path,
            )
            return None
        return ANSWER.pack(MAGIC, VERSION, OP_ANSWER, 0, len(snapshot)) + snapshot

    def _report_once(self, what: str, message: str, *values) -> None:
        if what not in self._reported:
            self._reported.add(what)
            logger.warning(message, *values)


def _unlink_quietly(path: str) -> None:
    try:
        os.unlink(path)
    except OSError:
        pass


class LocalClient:
    """A reader's end of the connection: asks for snapshots. Not thread-safe."""

    def __init__(
        self, socket_path: str | os.PathLike | None, timeout_s: float = 1.0
    ) -> None:
        self._path = (
            default_socket_path() if socket_path is None else os.fspath(socket_path)
        )
        self._timeout_s = max(float(timeout_s), 0.001)
        self._socket: socket.socket | None = None
        self._problem = _path_problem(self._path)
        if self._problem is not None:
            logger.warning("Robot state cannot be read: %s", self._problem)

    @property
    def path(self) -> str:
        return self._path

    @property
    def is_open(self) -> bool:
        return self._socket is not None

    def open(self) -> None:
        """Connect. OSError when nothing serves robot state at the path."""
        self.close()
        if self._problem is not None:
            raise OSError(self._problem)
        connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            connection.settimeout(self._timeout_s)
            connection.connect(self._path)
        except OSError:
            connection.close()
            raise
        self._socket = connection

    def request(self, known_setup_generation: int) -> bytes:
        """One snapshot of the server's store.

        OSError when the server does not answer in time, NotThisVersion when it answers in
        another format; the connection is closed then.
        """
        connection = self._socket
        if connection is None:
            raise OSError("not connected")
        deadline = time.monotonic() + self._timeout_s
        try:
            self._wait(connection, deadline)
            connection.sendall(
                REQUEST.pack(MAGIC, VERSION, OP_SNAPSHOT, 0, known_setup_generation)
            )
            header = self._receive(connection, ANSWER.size, deadline)
            magic, version, op, _zero, length = ANSWER.unpack(header)
            if magic != MAGIC or version != VERSION or op != OP_ANSWER:
                raise NotThisVersion(
                    "its answer is in a format this version of isaaccapture does not read"
                )
            if length > MAX_ANSWER_BYTES:
                raise OSError(
                    f"an answer of {length} bytes is above the limit of {MAX_ANSWER_BYTES}"
                )
            return self._receive(connection, length, deadline)
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        connection, self._socket = self._socket, None
        if connection is not None:
            try:
                connection.close()
            except OSError:
                pass

    def _wait(self, connection: socket.socket, deadline: float) -> None:
        """Give the next socket call the time left; TimeoutError when none is."""
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            raise TimeoutError(f"no answer within {self._timeout_s:g} s")
        connection.settimeout(remaining)

    def _receive(self, connection: socket.socket, count: int, deadline: float) -> bytes:
        data = bytearray(count)
        view = memoryview(data)
        received = 0
        while received < count:
            self._wait(connection, deadline)
            try:
                got = connection.recv_into(view[received:])
            except TimeoutError:
                raise TimeoutError(f"no answer within {self._timeout_s:g} s") from None
            if got == 0:
                raise ConnectionError("the server closed the connection")
            received += got
        return bytes(data)
