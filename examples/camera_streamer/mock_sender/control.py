# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The camera producer's management channel, as the mock serves it.

Wire protocol and status shape follow src/sensing_control (control_handler.cpp,
producer_status.hpp): an authenticated WebSocket, one JSON object per text
message, requests `ping` and `get_status`, pushed `state_changed` and `error`
events, and a coded reply for every failure.

The clock exchange is the one egocap's clock-sync client sends (little-endian,
nanoseconds): a binary message `Q`, u32 sequence, i64 t1 (13 bytes) is answered
with `R`, sequence, t1, t2 (receipt), t3 (reply send) (29 bytes), both times on
the mock's synthetic sender clock. Any other binary message closes the channel.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import socket
import struct
import threading
import time
from dataclasses import dataclass
from typing import Callable

logger = logging.getLogger("isaaccapture.camera_streamer.mock_control")

GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
MAX_MESSAGE_BYTES = 64 * 1024
MAX_CLIENTS = 8
STATES = ("starting", "streaming", "camera_fault", "error", "stopping")


@dataclass
class StreamStatus:
    stream: str
    sensor: int
    host: str = ""
    port: int = 0
    connected: bool = False
    error: str = ""
    frames: int = 0
    published: int = 0


class ProducerStatus:
    """State shared between the media loop (writer) and the channel threads (readers)."""

    def __init__(self, release: str, streams: list[StreamStatus]) -> None:
        self._lock = threading.Lock()
        self._release = release
        self._streams = streams
        self._state = "starting"
        self._fault: dict | None = None
        self._started = time.monotonic()
        self._listeners: list[Callable[[dict], None]] = []

    def subscribe(self, listener: Callable[[dict], None]) -> None:
        self._listeners.append(listener)

    def set_state(self, state: str, fault: dict | None = None) -> None:
        assert state in STATES
        with self._lock:
            changed = (state, fault) != (self._state, self._fault)
            self._state, self._fault = (
                state,
                fault if state in ("camera_fault", "error") else None,
            )
        if changed:
            self._push(
                {"type": "event", "event": "state_changed", "status": self.snapshot()}
            )
            if fault and state in ("camera_fault", "error"):
                self._push({"type": "event", "event": "error", "error": fault})

    def update_stream(self, index: int, **fields) -> None:
        with self._lock:
            for key, value in fields.items():
                setattr(self._streams[index], key, value)

    def add_frames(self, index: int, count: int = 1) -> None:
        with self._lock:
            self._streams[index].frames += count
            self._streams[index].published += count

    def snapshot(self) -> dict:
        with self._lock:
            out = {
                "state": self._state,
                "release": self._release,
                "uptime_seconds": int(time.monotonic() - self._started),
                "streams": [
                    {
                        "stream": s.stream,
                        "sensor": s.sensor,
                        "direct": {
                            "host": s.host,
                            "port": s.port,
                            "connected": s.connected,
                            "error": s.error,
                        },
                        "frames": s.frames,
                        "published": s.published,
                    }
                    for s in self._streams
                ],
            }
            if self._fault:
                out["fault"] = dict(self._fault)
            return out

    def _push(self, message: dict) -> None:
        for listener in list(self._listeners):
            listener(message)


def _failure(request_id, code: str, message: str) -> str:
    return json.dumps(
        {"id": request_id, "ok": False, "error": {"code": code, "message": message}}
    )


REQUEST = struct.Struct("<cIq")
REPLY = struct.Struct("<cIqqq")


def clock_sync_reply(data: bytes, t2: int, now_ns: Callable[[], int]) -> bytes | None:
    """The reply to a clock-sync request, or None unless `data` is exactly one."""
    if len(data) != REQUEST.size or data[:1] != b"Q":
        return None
    _, sequence, t1 = REQUEST.unpack(data)
    return REPLY.pack(
        b"R", sequence, t1, t2, now_ns()
    )  # t3 read last, just before the send


def handle_message(text: str, status: ProducerStatus) -> str:
    """The reply to one received text message."""
    try:
        request = json.loads(text)
    except ValueError:
        request = None
    if not isinstance(request, dict):
        return _failure(None, "bad_json", "the message must be one JSON object")
    request_id = request.get("id")
    kind = request.get("type")
    if not isinstance(kind, str):
        return _failure(request_id, "missing_type", 'the request needs a string "type"')
    if kind == "ping":
        return json.dumps({"id": request_id, "ok": True, "type": "pong"})
    if kind == "get_status":
        return json.dumps(
            {
                "id": request_id,
                "ok": True,
                "type": "status",
                "status": status.snapshot(),
            }
        )
    return _failure(
        request_id,
        "unknown_type",
        f'unknown request type "{kind}"; supported: ping, get_status',
    )


# --- WebSocket server (standard library, text frames only) -----------------


def _read_exact(sock: socket.socket, count: int) -> bytes:
    data = b""
    while len(data) < count:
        chunk = sock.recv(count - len(data))
        if not chunk:
            raise ConnectionError("closed")
        data += chunk
    return data


def _frame(opcode: int, payload: bytes) -> bytes:
    head = bytes([0x80 | opcode])
    if len(payload) < 126:
        return head + bytes([len(payload)]) + payload
    if len(payload) < 65536:
        return head + bytes([126]) + struct.pack(">H", len(payload)) + payload
    return head + bytes([127]) + struct.pack(">Q", len(payload)) + payload


def _close_frame(code: int, reason: str) -> bytes:
    return _frame(0x8, struct.pack(">H", code) + reason.encode())


class ControlServer:
    """Authenticated WebSocket server; every text message is answered by `on_message`."""

    def __init__(
        self,
        bind: str,
        port: int,
        token: str,
        on_message: Callable[[str], str],
        now_ns: Callable[[], int],
    ) -> None:
        if not token:
            raise ValueError("the control channel needs a non-empty token")
        self._token = token
        self._on_message = on_message
        self._now_ns = now_ns
        self._listener = socket.create_server((bind, port))
        self.port = self._listener.getsockname()[1]
        self._clients: list[tuple[socket.socket, threading.Lock]] = []
        self._clients_lock = threading.Lock()
        self._stop = threading.Event()
        threading.Thread(target=self._accept_loop, daemon=True).start()

    def close(self) -> None:
        self._stop.set()
        self._listener.close()
        with self._clients_lock:
            for sock, _ in self._clients:
                try:
                    sock.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass

    def broadcast(self, message: dict) -> None:
        data = _frame(0x1, json.dumps(message).encode())
        with self._clients_lock:
            clients = list(self._clients)
        for sock, lock in clients:
            try:
                with lock:
                    sock.sendall(data)
            except OSError:
                pass

    def _accept_loop(self) -> None:
        while not self._stop.is_set():
            try:
                sock, _ = self._listener.accept()
            except OSError:
                return
            threading.Thread(target=self._serve, args=(sock,), daemon=True).start()

    def _handshake(self, sock: socket.socket) -> bool:
        head = b""
        while b"\r\n\r\n" not in head:
            chunk = sock.recv(4096)
            if not chunk or len(head) > 8192:
                return False
            head += chunk
        lines = head.split(b"\r\n\r\n", 1)[0].decode(errors="replace").split("\r\n")
        headers = {}
        for line in lines[1:]:
            name, _, value = line.partition(":")
            headers[name.strip().lower()] = value.strip()
        target = lines[0].split(" ")[1] if len(lines[0].split(" ")) > 1 else "/"
        bearer = headers.get("authorization", "")
        presented = bearer[7:] if bearer.lower().startswith("bearer ") else ""
        if not presented and "token=" in target:
            presented = target.split("token=", 1)[1].split("&", 1)[0]
        key = headers.get("sec-websocket-key", "")
        refusal = None
        if headers.get("upgrade", "").lower() != "websocket" or not key:
            refusal = ("400 Bad Request", "not a WebSocket handshake")
        elif presented != self._token:
            refusal = ("401 Unauthorized", "missing or wrong token")
        elif len(self._clients) >= MAX_CLIENTS:
            refusal = ("503 Service Unavailable", "too many clients")
        if refusal:
            body = refusal[1]
            sock.sendall(
                f"HTTP/1.1 {refusal[0]}\r\nContent-Length: {len(body)}\r\nConnection: close\r\n\r\n{body}".encode()
            )
            return False
        accept = base64.b64encode(hashlib.sha1((key + GUID).encode()).digest()).decode()
        sock.sendall(
            (
                "HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\n"
                f"Connection: Upgrade\r\nSec-WebSocket-Accept: {accept}\r\n\r\n"
            ).encode()
        )
        return True

    def _serve(self, sock: socket.socket) -> None:
        lock = threading.Lock()
        try:
            if not self._handshake(sock):
                return
            with self._clients_lock:
                self._clients.append((sock, lock))
            while True:
                first, second = _read_exact(sock, 2)
                t2 = self._now_ns()  # receipt, as soon as the frame header arrives
                opcode, length = first & 0x0F, second & 0x7F
                if length == 126:
                    (length,) = struct.unpack(">H", _read_exact(sock, 2))
                elif length == 127:
                    (length,) = struct.unpack(">Q", _read_exact(sock, 8))
                if length > MAX_MESSAGE_BYTES:
                    with lock:
                        sock.sendall(_close_frame(1009, "message too large"))
                    return
                mask = _read_exact(sock, 4) if second & 0x80 else b"\0\0\0\0"
                payload = bytes(
                    b ^ mask[i % 4] for i, b in enumerate(_read_exact(sock, length))
                )
                if opcode == 0x8:
                    return
                if opcode == 0x9:
                    with lock:
                        sock.sendall(_frame(0xA, payload))
                elif opcode == 0x1:
                    reply = self._on_message(payload.decode(errors="replace"))
                    with lock:
                        sock.sendall(_frame(0x1, reply.encode()))
                elif opcode == 0x2:
                    reply = clock_sync_reply(payload, t2, self._now_ns)
                    with lock:
                        if reply is None:
                            sock.sendall(_close_frame(1003, "text messages only"))
                            return
                        sock.sendall(_frame(0x2, reply))
        except (OSError, ConnectionError):
            pass
        finally:
            with self._clients_lock:
                self._clients = [c for c in self._clients if c[0] is not sock]
            sock.close()
