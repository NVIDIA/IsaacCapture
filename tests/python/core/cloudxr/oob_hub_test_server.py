# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Standalone real OOBControlHub, for a real HeadsetControlChannel to connect to.

Not a pytest test: a small asyncio script a JS/TS test spawns as a subprocess, so a
real ``HeadsetControlChannel`` (deps/cloudxr/webxr_client/helpers/controlChannel.ts) can
talk to a real ``OOBControlHub`` over an actual (plain ws://, no TLS - the proxy's TLS
termination is orthogonal to the hub's own protocol) loopback WebSocket, with neither
side mocked.

Uses only the standard library (a minimal RFC 6455 server, just enough framing to satisfy
OOBControlHub.handle_connection's duck-typed ws interface: async iteration, send(), close())
rather than the third-party ``websockets`` package, so this runs under the plain ``python3``
already present on any CI runner - no extra install step, no virtual environment.

Protocol on stdout (one line per event, flushed immediately):
  READY <port>    - listening, ready for a client to connect
  SNAPSHOT <json> - hub.get_snapshot(), printed on every change to the
                    (clientId, connected, streaming, metricsByCadence) tuple of any
                    headset - i.e. on register/disconnect, sendStreamStatus, and
                    clientMetrics
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import struct
import sys
from pathlib import Path

# Not run under pytest (conftest.py never loads), so add tests/python to sys.path
# ourselves the same way conftest.py does, to reach the repo_paths helper.
_TESTS_PYTHON = Path(__file__).resolve().parents[2]
if str(_TESTS_PYTHON) not in sys.path:
    sys.path.insert(0, str(_TESTS_PYTHON))

from repo_paths import repo_root  # noqa: E402

# oob_teleop_hub.py has no relative imports, so it's importable flat without installing
# isaaccapture as a package (mirrors tests/python/core/cloudxr/conftest.py's approach).
_CLOUDXR_PY = repo_root() / "src" / "python" / "isaaccapture" / "cloudxr"
if str(_CLOUDXR_PY) not in sys.path:
    sys.path.insert(0, str(_CLOUDXR_PY))

from oob_teleop_hub import OOBControlHub  # noqa: E402

POLL_INTERVAL_S = 0.05
_WS_HANDSHAKE_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
_OPCODE_TEXT, _OPCODE_CLOSE, _OPCODE_PING, _OPCODE_PONG = 0x1, 0x8, 0x9, 0xA


class _MinimalWebSocket:
    """Bare client<->text-frame RFC 6455 connection - text and close/ping only, no
    fragmentation support - real WebXR/browser clients never need more than this for the
    small, single-frame JSON messages OOBControlHub's protocol uses."""

    def __init__(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        self._reader = reader
        self._writer = writer
        self._closed = False

    def __aiter__(self) -> _MinimalWebSocket:
        return self

    async def __anext__(self) -> str:
        while True:
            frame = await self._read_frame()
            if frame is None:
                raise StopAsyncIteration
            opcode, payload = frame
            if opcode == _OPCODE_CLOSE:
                await self.close()
                raise StopAsyncIteration
            if opcode == _OPCODE_PING:
                await self._send_frame(_OPCODE_PONG, payload)
                continue
            if opcode == _OPCODE_TEXT:
                return payload.decode("utf-8")
            # Binary/pong/continuation frames: not part of this protocol, ignore and
            # keep waiting for the next frame instead of raising.

    async def send(self, data: str) -> None:
        await self._send_frame(_OPCODE_TEXT, data.encode("utf-8"))

    async def close(self, code: int = 1000, reason: str = "") -> None:
        if self._closed:
            return
        self._closed = True
        try:
            await self._send_frame(
                _OPCODE_CLOSE, struct.pack("!H", code) + reason.encode("utf-8")
            )
        except (ConnectionError, OSError):
            pass  # Peer may already be gone - closing our end is still required below.
        self._writer.close()

    async def _read_frame(self) -> tuple[int, bytes] | None:
        header = await self._reader.readexactly(2)
        opcode = header[0] & 0x0F
        masked = bool(header[1] & 0x80)
        length = header[1] & 0x7F
        if length == 126:
            (length,) = struct.unpack("!H", await self._reader.readexactly(2))
        elif length == 127:
            (length,) = struct.unpack("!Q", await self._reader.readexactly(8))
        mask = await self._reader.readexactly(4) if masked else b""
        payload = await self._reader.readexactly(length) if length else b""
        if masked:
            payload = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))
        return opcode, payload

    async def _send_frame(self, opcode: int, payload: bytes) -> None:
        header = bytes([0x80 | opcode])  # FIN=1, no fragmentation
        length = len(payload)
        if length < 126:
            header += bytes([length])
        elif length < 1 << 16:
            header += bytes([126]) + struct.pack("!H", length)
        else:
            header += bytes([127]) + struct.pack("!Q", length)
        self._writer.write(header + payload)  # Server->client frames are never masked.
        await self._writer.drain()


async def _websocket_handshake(
    reader: asyncio.StreamReader, writer: asyncio.StreamWriter
) -> bool:
    """Perform the HTTP Upgrade handshake; return whether it succeeded."""
    request_line = await reader.readline()
    if not request_line:
        return False
    key = None
    while True:
        line = await reader.readline()
        if line in (b"\r\n", b""):
            break
        name, _, value = line.decode("latin-1").partition(":")
        if name.strip().lower() == "sec-websocket-key":
            key = value.strip()
    if key is None:
        return False
    accept = base64.b64encode(
        hashlib.sha1((key + _WS_HANDSHAKE_GUID).encode("ascii")).digest()
    ).decode("ascii")
    writer.write(
        (
            "HTTP/1.1 101 Switching Protocols\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            f"Sec-WebSocket-Accept: {accept}\r\n\r\n"
        ).encode("ascii")
    )
    await writer.drain()
    return True


def _fingerprint(snapshot: dict) -> tuple:
    """Reduce a snapshot to the fields whose change should trigger a new SNAPSHOT print."""
    return tuple(
        sorted(
            (
                h["clientId"],
                h["connected"],
                h["streaming"],
                # Sorted so key order in the JSON payload can't change the fingerprint.
                tuple(sorted(h["metricsByCadence"].items())),
            )
            for h in snapshot["headsets"]
        )
    )


async def main() -> None:
    """Start the hub, announce readiness, then print a snapshot on every state change."""
    hub = OOBControlHub()

    async def handle_tcp(
        reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        """Upgrade each new TCP connection to WebSocket, then hand it to the real hub."""
        if not await _websocket_handshake(reader, writer):
            writer.close()
            return
        await hub.handle_connection(_MinimalWebSocket(reader, writer))

    server = await asyncio.start_server(handle_tcp, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    print(f"READY {port}", flush=True)

    last = _fingerprint(await hub.get_snapshot())  # empty at startup - not printed

    try:
        while True:
            await asyncio.sleep(POLL_INTERVAL_S)
            snap = await hub.get_snapshot()
            fp = _fingerprint(snap)
            if fp != last:
                last = fp
                print("SNAPSHOT " + json.dumps(snap), flush=True)
    except asyncio.CancelledError:
        pass


if __name__ == "__main__":
    asyncio.run(main())
