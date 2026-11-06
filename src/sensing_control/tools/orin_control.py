#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Talk to a camera producer's management channel (camera_plugin_sensing or sensing_mock_producer).

    orin_control.py HOST[:PORT] --token-file token.txt status
    orin_control.py HOST[:PORT] --token-file token.txt set-target sensing-left 192.168.1.205 5600
    orin_control.py HOST[:PORT] --token-file token.txt set-target sensing-left ""     # stop the direct link
    orin_control.py HOST[:PORT] --token-file token.txt watch                          # print pushed events

Standard library only. The token goes in the Authorization header, never on the command line or in the URL.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import socket
import struct
import sys
from pathlib import Path

DEFAULT_PORT = 5610
GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"


class ChannelError(Exception):
    """The channel could not be used; the message says why."""


def recv_exact(sock: socket.socket, count: int) -> bytes:
    data = b""
    while len(data) < count:
        chunk = sock.recv(count - len(data))
        if not chunk:
            raise ChannelError("the producer closed the connection")
        data += chunk
    return data


class Connection:
    def __init__(self, host: str, port: int, token: str, timeout: float = 5.0) -> None:
        try:
            self._sock = socket.create_connection((host, port), timeout=timeout)
        except OSError as exc:
            raise ChannelError(f"cannot connect to {host}:{port}: {exc}") from exc
        key = base64.b64encode(os.urandom(16)).decode()
        request = (
            f"GET / HTTP/1.1\r\nHost: {host}:{port}\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\nAuthorization: Bearer {token}\r\n\r\n"
        )
        self._sock.sendall(request.encode())
        head = b""
        while b"\r\n\r\n" not in head:
            chunk = self._sock.recv(4096)
            if not chunk:
                raise ChannelError(
                    "the producer closed the connection during the handshake"
                )
            head += chunk
        status_line = head.split(b"\r\n", 1)[0].decode(errors="replace")
        if " 101 " not in status_line:
            body = head.split(b"\r\n\r\n", 1)[1].decode(errors="replace").strip()
            raise ChannelError(
                f"the producer refused the connection ({status_line}): {body}"
            )
        expected = base64.b64encode(
            hashlib.sha1((key + GUID).encode()).digest()
        ).decode()
        if (
            f"sec-websocket-accept: {expected}".lower()
            not in head.decode(errors="replace").lower()
        ):
            raise ChannelError(
                "the server's handshake answer is wrong; this is not a management channel"
            )
        self._pending = head.split(b"\r\n\r\n", 1)[1]

    def send(self, message: dict) -> None:
        payload = json.dumps(message).encode()
        mask = os.urandom(4)
        header = bytes([0x81])
        if len(payload) < 126:
            header += bytes([0x80 | len(payload)])
        elif len(payload) < 65536:
            header += bytes([0x80 | 126]) + struct.pack(">H", len(payload))
        else:
            header += bytes([0x80 | 127]) + struct.pack(">Q", len(payload))
        masked = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))
        self._sock.sendall(header + mask + masked)

    def _read(self, count: int) -> bytes:
        take = self._pending[:count]
        self._pending = self._pending[count:]
        return take + (
            recv_exact(self._sock, count - len(take)) if len(take) < count else b""
        )

    def receive(self, timeout: float | None = None) -> dict:
        self._sock.settimeout(timeout)
        while True:
            first, second = self._read(2)
            opcode, length = first & 0x0F, second & 0x7F
            if length == 126:
                (length,) = struct.unpack(">H", self._read(2))
            elif length == 127:
                (length,) = struct.unpack(">Q", self._read(8))
            payload = self._read(length)
            if opcode == 0x8:
                raise ChannelError(
                    "the producer closed the channel: "
                    + payload[2:].decode(errors="replace")
                )
            if opcode == 0x1:
                return json.loads(payload)
            # Pings and other frames are ignored; the server sends none that need an answer.

    def request(self, message: dict) -> dict:
        message = {"id": 1, **message}
        self.send(message)
        while True:
            reply = self.receive(timeout=10)
            if reply.get("id") == message["id"]:
                return reply


def describe(status: dict) -> str:
    lines = [
        f"state: {status['state']}   release: {status['release']}   up {status['uptime_seconds']} s"
    ]
    fault = status.get("fault")
    if fault:
        lines.append(f"PROBLEM [{fault['code']}]: {fault['message']}")
    for stream in status["streams"]:
        direct = stream["direct"]
        target = f"{direct['host']}:{direct['port']}" if direct["host"] else "off"
        link = "connected" if direct["connected"] else "not connected"
        lines.append(
            f"  {stream['stream']}: sensor {stream['sensor']}, {stream['frames']} frames, "
            f"direct link {target} ({link if direct['host'] else '-'})"
        )
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "target", help=f"producer HOST or HOST:PORT (default port {DEFAULT_PORT})"
    )
    ap.add_argument(
        "--token-file", required=True, type=Path, help="file holding the shared secret"
    )
    sub = ap.add_subparsers(dest="command", required=True)
    sub.add_parser("status", help="print the producer's state")
    sub.add_parser("watch", help="print pushed events until interrupted")
    setter = sub.add_parser(
        "set-target",
        help="send a stream's video to HOST PORT (empty HOST turns it off)",
    )
    setter.add_argument("stream")
    setter.add_argument("host")
    setter.add_argument("port", nargs="?", type=int, default=0)
    args = ap.parse_args(argv)

    host, _, port = args.target.partition(":")
    try:
        token = args.token_file.read_text().strip()
        if not token:
            raise ChannelError(f"{args.token_file} is empty")
        conn = Connection(host, int(port) if port else DEFAULT_PORT, token)
        if args.command == "status":
            reply = conn.request({"type": "get_status"})
        elif args.command == "set-target":
            reply = conn.request(
                {
                    "type": "set_direct_target",
                    "stream": args.stream,
                    "host": args.host,
                    "port": args.port,
                }
            )
        else:
            print(describe(conn.request({"type": "get_status"})["status"]))
            while True:
                print(json.dumps(conn.receive()))
        if not reply["ok"]:
            print(
                f"Error [{reply['error']['code']}]: {reply['error']['message']}",
                file=sys.stderr,
            )
            return 1
        print(describe(reply["status"]))
        return 0
    except (ChannelError, OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
