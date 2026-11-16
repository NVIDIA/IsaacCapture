# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The local API's Unix socket: its records, its listener and the client's carrier.

A record is one byte of type number, four bytes of big-endian payload length and
the payload. Type numbers and payloads are those of the robot connection, plus
the local API's own (protocol/local_messages.py).
"""

from __future__ import annotations

import contextlib
import fcntl
import logging
import os
import select
import socket
import stat
import struct
import threading
import time
from collections import OrderedDict, deque
from collections.abc import Callable
from typing import NamedTuple

from ..protocol.local_messages import (
    ROLE_CONSOLE,
    LocalHello,
    LocalNotice,
    LocalStatus,
    decode_local,
    is_local,
)
from ..protocol.messages import (
    MAX_PAYLOAD_BYTES,
    NEWEST_WINS_TYPES,
    PROTOCOL_VERSION,
    ProtocolError,
    Stop,
)
from ..robot_app.text import MAX_REASON_CHARS, printable

_log = logging.getLogger("isaaccapture.examples.agentic_remote.local")

HEADER = struct.Struct(">BI")  # type number, payload length
SOCKET_FOLDER = "agentic_remote"
SOCKET_NAME = "robot_link.sock"
# sockaddr_un holds 108 bytes, the ending zero byte among them.
MAX_SOCKET_PATH_BYTES = 107
# Seconds a send to the other end of the socket may take before that end counts
# as stuck.
SEND_TIMEOUT_S = 2.0
# Records a LocalCarrier keeps for receive() before it stops reading the socket.
INBOX_MESSAGES = 4096
INBOX_BYTES = 32 * 1024 * 1024
# Notices a LocalCarrier keeps, newest last.
NOTICES_KEPT = 64
_RECV_BYTES = 65536
_PROBE_S = 0.5


class LocalSocketError(OSError):
    """The local API's socket cannot be made at the given path."""


class LocalSocketInUse(LocalSocketError):
    """A live robot link service already has the socket."""


class LocalRefused(ConnectionError):
    """The robot link service did not take this client; the text says why."""


class Stalled(Exception):
    """The other end sent less than a whole record in the time allowed."""


def default_socket_path() -> str:
    """$XDG_RUNTIME_DIR/agentic_remote/robot_link.sock, else under /tmp per user."""
    runtime = os.environ.get("XDG_RUNTIME_DIR")
    if runtime:
        return os.path.join(runtime, SOCKET_FOLDER, SOCKET_NAME)
    return os.path.join("/tmp", f"{SOCKET_FOLDER}-{os.getuid()}", SOCKET_NAME)


def newest_wins_locally(msg_type: int) -> bool:
    """True for a type of which only the newest waiting record is kept."""
    return msg_type in NEWEST_WINS_TYPES or msg_type == LocalStatus.TYPE


def pack_record(msg_type: int, payload: bytes) -> bytes:
    """One record for the socket; ProtocolError if it cannot be one."""
    if not 0 <= msg_type <= 255:
        raise ProtocolError(f"type number {msg_type!r} does not fit one byte")
    if len(payload) > MAX_PAYLOAD_BYTES:
        raise ProtocolError(f"{len(payload)} bytes of payload is over the limit")
    return HEADER.pack(msg_type, len(payload)) + payload


def read_exact(sock: socket.socket, count: int, deadline: float | None = None) -> bytes:
    """The next `count` bytes of the socket.

    Raises EOFError when the other end closed, and Stalled when `deadline`
    (time.monotonic()) passes first. Without a deadline it waits for as long as
    the socket stays open.
    """
    parts: list[bytes] = []
    left = count
    while left > 0:
        if deadline is not None:
            wait = deadline - time.monotonic()
            if wait <= 0:
                raise Stalled
            try:
                ready, _, _ = select.select([sock], [], [], wait)
            except (OSError, ValueError):
                raise EOFError from None
            if not ready:
                raise Stalled
        try:
            part = sock.recv(min(left, _RECV_BYTES))
        except TimeoutError:
            continue
        except OSError:
            raise EOFError from None
        if not part:
            raise EOFError
        parts.append(part)
        left -= len(part)
    return b"".join(parts)


class RecordQueue:
    """A bounded backlog of (type number, payload) in the order of arrival.

    A newest-wins record replaces a waiting one of its type and takes the newer
    place. Not thread-safe.
    """

    def __init__(
        self,
        max_messages: int,
        max_bytes: int,
        newest_wins: Callable[[int], bool] = newest_wins_locally,
    ) -> None:
        self.max_messages = int(max_messages)
        self.max_bytes = int(max_bytes)
        self._newest_wins = newest_wins
        # Arrival number -> (type number, payload), oldest first.
        self._waiting: OrderedDict[int, tuple[int, bytes]] = OrderedDict()
        # Type number -> arrival number of its waiting newest-wins record.
        self._newest: dict[int, int] = {}
        self._arrivals = 0
        self._bytes = 0

    def __len__(self) -> int:
        return len(self._waiting)

    def add(self, msg_type: int, payload: bytes) -> bool:
        """Keep a record for later; False, and it is not kept, when full."""
        newest = self._newest_wins(msg_type)
        replaced = self._newest.get(msg_type) if newest else None
        freed = 0 if replaced is None else len(self._waiting[replaced][1])
        count = len(self._waiting) - (replaced is not None)
        if count >= self.max_messages or (
            self._bytes - freed + len(payload) > self.max_bytes
        ):
            return False
        if replaced is not None:
            del self._waiting[replaced]
            self._bytes -= freed
        if newest:
            self._newest[msg_type] = self._arrivals
        self._waiting[self._arrivals] = (msg_type, payload)
        self._bytes += len(payload)
        self._arrivals += 1
        return True

    def take(self) -> tuple[int, bytes] | None:
        """The oldest waiting record, or None when none waits."""
        if not self._waiting:
            return None
        arrival, taken = self._waiting.popitem(last=False)
        if self._newest.get(taken[0]) == arrival:
            del self._newest[taken[0]]
        self._bytes -= len(taken[1])
        return taken

    def clear(self) -> None:
        self._waiting.clear()
        self._newest.clear()
        self._bytes = 0


class LocalListener:
    """The service's end of the local API: a Unix stream socket that listens.

    The socket's folder is made with mode 0700 and the socket has mode 0600. A
    socket file left by a dead service is replaced; a live service keeps its
    socket and LocalSocketInUse is raised.
    """

    def __init__(self, path: str | None = None, *, backlog: int = 8) -> None:
        self.path = os.path.abspath(path or default_socket_path())
        if len(os.fsencode(self.path)) > MAX_SOCKET_PATH_BYTES:
            raise LocalSocketError(
                f"socket path {self.path} is longer than {MAX_SOCKET_PATH_BYTES} bytes"
            )
        _private_folder(os.path.dirname(self.path))
        self._lock_fd = _take_lock(self.path)
        try:
            _clear_stale(self.path)
            self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                self._sock.bind(self.path)
                # Nobody can connect before listen(), so the mode is in place first.
                os.chmod(self.path, 0o600)
                self._inode = os.stat(self.path).st_ino
                self._sock.listen(backlog)
            except BaseException:
                self._sock.close()
                raise
        except BaseException:
            os.close(self._lock_fd)
            raise
        self._closed = False

    def accept(self, timeout_s: float) -> socket.socket | None:
        """The next client's socket, or None after timeout_s or once closed."""
        try:
            ready, _, _ = select.select([self._sock], [], [], max(0.0, timeout_s))
            if not ready:
                return None
            client, _ = self._sock.accept()
        except (OSError, ValueError):
            return None
        return client

    def close(self) -> None:
        """Stop listening and remove the socket file, if it is still this one."""
        if self._closed:
            return
        self._closed = True
        self._sock.close()
        with contextlib.suppress(OSError):
            if os.stat(self.path).st_ino == self._inode:
                os.unlink(self.path)
        os.close(self._lock_fd)  # the lock file stays; only the lock ends


def _private_folder(folder: str) -> None:
    """Make the socket's folder with mode 0700, or check the one that exists."""
    try:
        os.makedirs(os.path.dirname(folder), exist_ok=True)
        os.mkdir(folder, 0o700)
        os.chmod(folder, 0o700)  # whatever the umask said
        return
    except FileExistsError:
        pass
    except OSError as error:
        raise LocalSocketError(f"cannot make the folder {folder}: {error}") from error
    found = os.stat(folder)
    if not stat.S_ISDIR(found.st_mode):
        raise LocalSocketError(f"{folder} is not a folder")
    if found.st_uid != os.getuid():
        raise LocalSocketError(f"the folder {folder} belongs to another user")
    if found.st_mode & 0o077:
        raise LocalSocketError(
            f"the folder {folder} is open to other users (mode "
            f"{stat.S_IMODE(found.st_mode):04o}); the socket needs a folder of mode "
            "0700"
        )


def _take_lock(path: str) -> int:
    """Lock the file beside the socket for this process; a live service holds it."""
    try:
        fd = os.open(f"{path}.lock", os.O_RDWR | os.O_CREAT | os.O_CLOEXEC, 0o600)
    except OSError as error:
        raise LocalSocketError(f"cannot open {path}.lock: {error}") from error
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        os.close(fd)
        raise LocalSocketInUse(f"a robot link service already runs on {path}") from None
    return fd


def _dial(sock: socket.socket, path: str) -> None:
    """Open the socket to the one that listens at `path`; OSError when none does.

    Written with connect_ex(): test_wiring.py keeps every method call named
    "connect" in this package for the arm.
    """
    error = sock.connect_ex(path)
    if error:
        raise OSError(error, os.strerror(error), path)


def _clear_stale(path: str) -> None:
    """Remove the socket file of a dead service; leave anything else alone."""
    try:
        found = os.lstat(path)
    except FileNotFoundError:
        return
    if not stat.S_ISSOCK(found.st_mode):
        raise LocalSocketError(f"{path} exists and is not a socket; it is not replaced")
    probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    probe.settimeout(_PROBE_S)
    try:
        _dial(probe, path)
    except (ConnectionRefusedError, FileNotFoundError):
        pass  # nobody listens: stale
    except OSError as error:
        raise LocalSocketError(f"cannot tell whether {path} is in use: {error}") from (
            error
        )
    else:
        raise LocalSocketInUse(f"a robot link service already runs on {path}")
    finally:
        probe.close()
    _log.info("replacing the socket file of a dead service: %s", path)
    with contextlib.suppress(FileNotFoundError):
        os.unlink(path)


class LocalCarrier:
    """A local client's end of the local API, as a Carrier.

    Link messages of the robot connection pass through receive(). The local
    API's own messages do not: the newest LOCAL_STATUS is `status` and the
    LOCAL_NOTICEs are `notices`. When receive() is not called the carrier stops
    reading its socket once its inbox is full. On a dead carrier receive() still
    waits out its timeout.
    """

    def __init__(
        self,
        sock: socket.socket,
        *,
        send_timeout_s: float = SEND_TIMEOUT_S,
        inbox_messages: int = INBOX_MESSAGES,
        inbox_bytes: int = INBOX_BYTES,
    ) -> None:
        self._sock = sock
        sock.settimeout(send_timeout_s)
        self._send_lock = threading.Lock()
        self._stop_limit_s: float | None = None  # see bound_stop()
        self._cond = threading.Condition()
        self._inbox = RecordQueue(inbox_messages, inbox_bytes)
        self._status: LocalStatus | None = None
        self._notices: deque[LocalNotice] = deque(maxlen=NOTICES_KEPT)
        self._alive = True
        self._close_reason = ""
        self._reader = threading.Thread(
            target=self._read_loop, name="local-carrier", daemon=True
        )
        self._reader.start()

    def send_reliable(self, msg_type: int, payload: bytes) -> None:
        self._send(msg_type, payload)

    def send_latest(self, msg_type: int, payload: bytes) -> None:
        # The socket keeps the order; the service keeps only the newest where
        # that matters.
        self._send(msg_type, payload)

    def receive(self, timeout_s: float) -> tuple[int, bytes] | None:
        with self._cond:
            if not len(self._inbox) and timeout_s > 0:
                self._cond.wait_for(lambda: bool(len(self._inbox)), timeout_s)
            taken = self._inbox.take()
            if taken is not None:
                self._cond.notify_all()  # the reader may have waited for room
            return taken

    @property
    def alive(self) -> bool:
        with self._cond:
            return self._alive

    @property
    def close_reason(self) -> str:
        """Why the carrier died; "" while it is alive."""
        with self._cond:
            return self._close_reason

    @property
    def status(self) -> LocalStatus | None:
        """The newest LOCAL_STATUS, None before the first one."""
        with self._cond:
            return self._status

    @property
    def notices(self) -> list[LocalNotice]:
        """The LOCAL_NOTICEs so far, oldest first (the newest NOTICES_KEPT)."""
        with self._cond:
            return list(self._notices)

    def wait_status(
        self,
        predicate: Callable[[LocalStatus], bool] | None = None,
        timeout_s: float = 5.0,
    ) -> LocalStatus:
        """Wait in real time for a LOCAL_STATUS the predicate accepts."""

        def found() -> bool:
            status = self._status
            return status is not None and (predicate is None or bool(predicate(status)))

        with self._cond:
            if not self._cond.wait_for(found, timeout_s):
                raise TimeoutError(
                    f"no such LOCAL_STATUS within {timeout_s:g} s; newest: "
                    f"{self._status}"
                )
            return self._status

    def wait_notice(
        self,
        predicate: Callable[[LocalNotice], bool] | None = None,
        timeout_s: float = 5.0,
    ) -> LocalNotice:
        """Wait in real time for a LOCAL_NOTICE the predicate accepts."""

        def found() -> LocalNotice | None:
            for notice in self._notices:
                if predicate is None or predicate(notice):
                    return notice
            return None

        with self._cond:
            if not self._cond.wait_for(lambda: found() is not None, timeout_s):
                raise TimeoutError(f"no such LOCAL_NOTICE within {timeout_s:g} s")
            return found()

    def wait_dead(self, timeout_s: float = 5.0) -> bool:
        """Wait in real time until the carrier is dead; whether it is."""
        with self._cond:
            return self._cond.wait_for(lambda: not self._alive, timeout_s)

    def wait_answer(self, timeout_s: float) -> bool:
        """Wait for the first LOCAL_STATUS or the carrier's death; False if neither."""
        with self._cond:
            return self._cond.wait_for(
                lambda: self._status is not None or not self._alive, timeout_s
            )

    def close(self) -> None:
        """End the local connection."""
        self._die("closed by this end")
        if self._reader is not threading.current_thread():
            self._reader.join(SEND_TIMEOUT_S)
        self._sock.close()

    def bound_stop(self, limit_s: float) -> None:
        """Let a STOP wait at most limit_s for this socket, from now on.

        A STOP that another send or a socket nobody reads holds up for longer
        ends the carrier with nothing of it written. For a client that then
        writes the STOP another way (send_raw_stop); without this call a STOP
        waits as any other send does.
        """
        self._stop_limit_s = float(limit_s)

    def _send(self, msg_type: int, payload: bytes) -> None:
        record = pack_record(msg_type, payload)
        limit = self._stop_limit_s if msg_type == Stop.TYPE else None
        end = time.monotonic() + (limit or 0.0)
        if not self._send_lock.acquire(timeout=-1 if limit is None else limit):
            self._die(f"a STOP waited {limit:g} s behind another send")
            return
        try:
            if not self.alive:
                return
            if limit is not None and not self._writable(end - time.monotonic()):
                self._die(f"the socket did not take a STOP within {limit:g} s")
                return
            try:
                self._sock.sendall(record)
            except OSError as error:
                # Part of a record may be out: the stream cannot be used again.
                self._die(f"send failed: {error}")
        finally:
            self._send_lock.release()

    def _writable(self, wait_s: float) -> bool:
        """Whether the socket takes a record within wait_s; a send would wait."""
        try:
            return bool(select.select([], [self._sock], [], max(0.0, wait_s))[1])
        except (OSError, ValueError):
            return False

    def _die(self, reason: str) -> None:
        with self._cond:
            if not self._alive:
                return
            self._alive = False
            self._close_reason = reason
            self._cond.notify_all()
        with contextlib.suppress(OSError):
            self._sock.shutdown(socket.SHUT_RDWR)

    def _read_loop(self) -> None:
        reason = "closed by the robot link service"
        try:
            while True:
                msg_type, length = HEADER.unpack(read_exact(self._sock, HEADER.size))
                if length > MAX_PAYLOAD_BYTES:
                    reason = f"a record of {length} bytes is over the limit"
                    break
                payload = read_exact(self._sock, length)
                if is_local(msg_type):
                    self._take_local(msg_type, payload)
                elif not self._keep(msg_type, payload):
                    return
        except EOFError:
            pass
        except Exception as error:
            _log.exception("reading the local API socket failed")
            reason = f"read failed: {error}"
        self._die(reason)

    def _keep(self, msg_type: int, payload: bytes) -> bool:
        """Put a record into the inbox, waiting for room; False once dead."""
        with self._cond:
            while self._alive:
                if self._inbox.add(msg_type, payload):
                    self._cond.notify_all()
                    return True
                self._cond.wait()
            return False

    def _take_local(self, msg_type: int, payload: bytes) -> None:
        try:
            message = decode_local(msg_type, payload)
        except ProtocolError as error:
            _log.warning(
                "dropped a local API message of type %d: %s",
                msg_type,
                printable(error, MAX_REASON_CHARS),
            )
            return
        with self._cond:
            if isinstance(message, LocalStatus):
                self._status = message
            elif isinstance(message, LocalNotice):
                self._notices.append(message)
            self._cond.notify_all()


def connect_local(
    path: str | None = None,
    role: str = "tool",
    name: str = "",
    timeout_s: float = 2.0,
) -> LocalCarrier:
    """Attach to the robot link service at `path` in this role.

    Returns once the service has answered with its first LOCAL_STATUS. Raises
    LocalRefused when the service refuses the client, TimeoutError when it does
    not answer in timeout_s and OSError when nothing listens at the path.
    """
    path = path or default_socket_path()
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        sock.settimeout(timeout_s)
        _dial(sock, path)
    except BaseException:
        sock.close()
        raise
    carrier = LocalCarrier(sock)
    hello = LocalHello(role=role, name=name, protocol_version=PROTOCOL_VERSION)
    carrier.send_reliable(hello.TYPE, hello.to_payload())
    answered = carrier.wait_answer(timeout_s)
    if carrier.status is None:
        notices, why = carrier.notices, carrier.close_reason
        carrier.close()
        if not answered:
            raise TimeoutError(
                f"the robot link service at {path} did not answer in {timeout_s:g} s"
            )
        raise LocalRefused(notices[-1].text if notices else why)
    return carrier


class RawStop(NamedTuple):
    """What became of a raw STOP; it is never known to be delivered."""

    written: bool  # both records left this process
    notice: str  # the last LOCAL_NOTICE read back, "" if none; why, if not written


def send_raw_stop(
    path: str | None = None,
    *,
    reason: str,
    sender: str,
    name: str = "console",
    timeout_s: float = 1.0,
) -> RawStop:
    """Write a LOCAL_HELLO and a STOP to the service in one go, without a carrier.

    For when connect_local() fails: the service forwards a STOP that follows
    the LOCAL_HELLO at once even when it turns the client away. One attempt;
    it raises nothing. Then it reads until the service closes the socket or
    timeout_s has passed and keeps the newest LOCAL_NOTICE.
    """
    path = path or default_socket_path()
    hello = LocalHello(role=ROLE_CONSOLE, name=name, protocol_version=PROTOCOL_VERSION)
    stop = Stop(reason=reason, sender=sender)
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        try:
            records = pack_record(hello.TYPE, hello.to_payload())
            records += pack_record(stop.TYPE, stop.to_payload())
            sock.settimeout(timeout_s)
            _dial(sock, path)
        except (OSError, ProtocolError) as error:
            return RawStop(False, f"no robot link service at {path}: {error}")
        try:
            sock.sendall(records)
        except OSError as error:
            return RawStop(False, f"writing to {path} failed: {error}")
        notice = ""
        by = time.monotonic() + timeout_s
        try:
            while True:
                msg_type, length = HEADER.unpack(read_exact(sock, HEADER.size, by))
                if length > MAX_PAYLOAD_BYTES:
                    break
                payload = read_exact(sock, length, by)
                if msg_type == LocalNotice.TYPE:
                    with contextlib.suppress(ProtocolError):
                        notice = decode_local(msg_type, payload).text
        except (EOFError, Stalled):
            pass
        return RawStop(True, printable(notice, MAX_REASON_CHARS))
    finally:
        sock.close()
