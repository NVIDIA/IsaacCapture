# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The records between the camera frame bridge and the agent program.

Each record is one line of JSON on a Unix socket. A "pair" record is followed by
the raw NV12 bytes of the left and then the right camera frame, as many as its
`left_bytes` and `right_bytes` say. Standard library only: the bridge runs in
the camera consumer's Python environment.
"""

from __future__ import annotations

import json
import os
import socket
import time
from typing import Any

VERSION = 1
MAX_LINE_BYTES = 65536
# sockaddr_un holds 108 bytes, the ending zero byte among them.
MAX_SOCKET_PATH_BYTES = 107
SOCKET_FOLDER = "agentic_remote"
_RECV_BYTES = 1 << 20

HELLO = "hello"  # bridge -> agent program, once, right after the socket opens
NEXT = "next"  # agent program -> bridge: one stereo pair newer than the one named
PAIR = "pair"  # bridge -> agent program: a verified stereo pair; raw bytes follow
NONE = "none"  # bridge -> agent program: no pair within the time asked for
ERROR = "error"  # bridge -> agent program: the bridge gives up; it ends after this
CLOSE = "close"  # agent program -> bridge: end

_INT_PAIR = "two whole numbers"
# Per record type: field name -> the type (or types) its value must have.
_FIELDS: dict[str, dict[str, Any]] = {
    HELLO: {
        "version": int,
        "width": int,
        "height": int,
        "pixel_format": str,
        "ring_depth": int,
        "gpu_id": int,
        "streams": list,
    },
    NEXT: {
        "after_session": (int, type(None)),
        "after_seq": (int, type(None)),
        "timeout_s": (int, float),
    },
    PAIR: {
        "session_id": int,
        "seq": int,
        "sof_wall_ns": _INT_PAIR,
        "device_index": _INT_PAIR,
        "rotation": int,
        "receiver_rx_mono_ns": int,
        "check": dict,
        "left_bytes": int,
        "right_bytes": int,
    },
    NONE: {"reason": str},
    ERROR: {"reason": str},
    CLOSE: {},
}


class WireError(Exception):
    """A record that is not one of this module's."""


class WireClosed(Exception):
    """The other end closed the socket."""


class WireTimeout(Exception):
    """The deadline passed before a whole record or all bytes were in."""


def hello(
    *,
    width: int,
    height: int,
    pixel_format: str,
    ring_depth: int,
    gpu_id: int,
    streams: tuple[str, str],
) -> dict[str, Any]:
    return {
        "type": HELLO,
        "version": VERSION,
        "width": width,
        "height": height,
        "pixel_format": pixel_format,
        "ring_depth": ring_depth,
        "gpu_id": gpu_id,
        "streams": list(streams),
    }


def next_request(
    after_session: int | None, after_seq: int | None, timeout_s: float
) -> dict[str, Any]:
    return {
        "type": NEXT,
        "after_session": after_session,
        "after_seq": after_seq,
        "timeout_s": float(timeout_s),
    }


def none(reason: str, counters: dict[str, int] | None = None) -> dict[str, Any]:
    return {"type": NONE, "reason": reason, "counters": dict(counters or {})}


def error(reason: str) -> dict[str, Any]:
    return {"type": ERROR, "reason": reason}


def close() -> dict[str, Any]:
    return {"type": CLOSE}


def pack(record: dict[str, Any]) -> bytes:
    """The record as one line; WireError when it is not a valid record."""
    check(record)
    line = json.dumps(record, separators=(",", ":"), allow_nan=False).encode() + b"\n"
    if len(line) > MAX_LINE_BYTES:
        raise WireError(f"a {record['type']} record of {len(line)} bytes is too long")
    return line


def parse(line: bytes) -> dict[str, Any]:
    """The record of one line (without its newline); WireError when it is none."""
    try:
        record = json.loads(line.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as problem:
        raise WireError(f"not a JSON line: {problem}") from None
    check(record)
    return record


def check(record: Any) -> None:
    """Raise WireError unless `record` has a known type and that type's fields."""
    if not isinstance(record, dict):
        raise WireError("a record must be a JSON object")
    fields = _FIELDS.get(record.get("type"))
    if fields is None:
        raise WireError(f"unknown record type {record.get('type')!r}")
    for name, kind in fields.items():
        value = record.get(name)
        if kind is _INT_PAIR:
            good = (
                isinstance(value, list)
                and len(value) == 2
                and all(_is(item, int) for item in value)
            )
        else:
            good = _is(value, kind)
        if not good:
            raise WireError(f"{record['type']} record: bad field {name}: {value!r}")
    if record["type"] == PAIR and min(record["left_bytes"], record["right_bytes"]) <= 0:
        raise WireError("pair record: no bytes")
    if record["type"] == NEXT and not 0 <= record["timeout_s"] < float("inf"):
        raise WireError(f"next record: bad timeout_s {record['timeout_s']!r}")


def _is(value: Any, kind: Any) -> bool:
    # bool is an int for Python; it is none here.
    return isinstance(value, kind) and not isinstance(value, bool)


class Reader:
    """Reads records and raw bytes from one socket, in the order they were sent."""

    def __init__(self, sock: socket.socket) -> None:
        self._sock = sock
        self._buffer = bytearray()

    def record(self, deadline: float | None = None) -> dict[str, Any]:
        """The next record. `deadline` is a time.monotonic() value.

        Raises WireClosed, WireTimeout (what was read so far is kept) or WireError.
        """
        while True:
            end = self._buffer.find(b"\n")
            if end >= 0:
                line = bytes(self._buffer[:end])
                del self._buffer[: end + 1]
                return parse(line)
            if len(self._buffer) > MAX_LINE_BYTES:
                raise WireError("a line without an end")
            self._buffer += self._receive(_RECV_BYTES, deadline)

    def exact(self, count: int, deadline: float | None = None) -> bytearray:
        """The next `count` raw bytes; WireClosed or WireTimeout."""
        out = bytearray(count)
        have = min(count, len(self._buffer))
        out[:have] = self._buffer[:have]
        del self._buffer[:have]
        view = memoryview(out)
        while have < count:
            self._wait(deadline)
            try:
                got = self._sock.recv_into(view[have:], count - have)
            except TimeoutError:
                raise WireTimeout from None
            except OSError:
                raise WireClosed from None
            if got == 0:
                raise WireClosed
            have += got
        return out

    def _receive(self, count: int, deadline: float | None) -> bytes:
        self._wait(deadline)
        try:
            chunk = self._sock.recv(count)
        except TimeoutError:
            raise WireTimeout from None
        except OSError:
            raise WireClosed from None
        if not chunk:
            raise WireClosed
        return chunk

    def _wait(self, deadline: float | None) -> None:
        if deadline is None:
            self._sock.settimeout(None)
            return
        left = deadline - time.monotonic()
        if left <= 0:
            raise WireTimeout
        try:
            self._sock.settimeout(left)
        except OSError:
            raise WireClosed from None


def send(sock: socket.socket, record: dict[str, Any], *payloads: Any) -> None:
    """Send one record and the raw bytes that follow it; WireClosed if that fails."""
    try:
        sock.settimeout(None)
        sock.sendall(pack(record))
        for payload in payloads:
            sock.sendall(payload)
    except OSError:
        raise WireClosed from None


def dial(path: str) -> socket.socket:
    """A socket to the one that listens at `path`; OSError when none does.

    Written with connect_ex(): test_wiring.py keeps every method call named
    "connect" in this package for the arm.
    """
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    problem = sock.connect_ex(path)
    if problem:
        sock.close()
        raise OSError(problem, os.strerror(problem), path)
    return sock


def socket_path(folder: str, name: str) -> str:
    """Where a Unix socket called `name` goes: in `folder` when the path fits.

    A run folder deep in the tree gives a path longer than a socket address can
    hold; the socket then goes into the user's own runtime folder (mode 0700).
    """
    path = os.path.join(folder, name)
    if len(os.fsencode(path)) <= MAX_SOCKET_PATH_BYTES:
        return path
    runtime = os.environ.get("XDG_RUNTIME_DIR")
    if runtime:
        short = os.path.join(runtime, SOCKET_FOLDER)
    else:
        short = os.path.join("/tmp", f"{SOCKET_FOLDER}-{os.getuid()}")
    os.makedirs(short, mode=0o700, exist_ok=True)
    if os.stat(short).st_uid != os.getuid():
        raise OSError(f"{short} belongs to another user")
    path = os.path.join(short, name)
    if len(os.fsencode(path)) > MAX_SOCKET_PATH_BYTES:
        raise OSError(f"socket path too long: {path}")
    return path
