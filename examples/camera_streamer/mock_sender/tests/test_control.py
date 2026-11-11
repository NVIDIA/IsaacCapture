#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The mock's management channel: protocol, authentication, events and faults."""

from __future__ import annotations

import base64
import json
import os
import shutil
import socket
import struct
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import control  # noqa: E402
import mock_sender as ms  # noqa: E402

TOKEN = "secret"


class Client:
    """A minimal masked-frame WebSocket client."""

    def __init__(self, port: int, token: str | None = TOKEN) -> None:
        self.sock = socket.create_connection(("127.0.0.1", port), timeout=5)
        auth = f"Authorization: Bearer {token}\r\n" if token is not None else ""
        key = base64.b64encode(os.urandom(16)).decode()
        self.sock.sendall(
            (
                f"GET / HTTP/1.1\r\nHost: x\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
                f"Sec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n{auth}\r\n"
            ).encode()
        )
        head = b""
        while b"\r\n\r\n" not in head:
            head += self.sock.recv(4096)
        self.status_line = head.split(b"\r\n", 1)[0].decode()

    def close(self) -> None:
        self.sock.close()

    def send(self, payload: bytes, opcode: int = 0x1) -> None:
        mask = os.urandom(4)
        masked = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))
        self.sock.sendall(bytes([0x80 | opcode, 0x80 | len(payload)]) + mask + masked)

    def _exact(self, count: int) -> bytes:
        data = b""
        while len(data) < count:
            data += self.sock.recv(count - len(data))
        return data

    def receive(self) -> tuple[int, bytes]:
        first, second = self._exact(2)
        length = second & 0x7F
        if length == 126:
            (length,) = struct.unpack(">H", self._exact(2))
        elif length == 127:
            (length,) = struct.unpack(">Q", self._exact(8))
        return first & 0x0F, self._exact(length)

    def request(self, message: dict) -> dict:
        self.send(json.dumps(message).encode())
        while True:
            opcode, data = self.receive()
            reply = json.loads(data)
            if opcode == 0x1 and reply.get("id") == message["id"]:
                return reply

    def event(self) -> dict:
        while True:
            _, data = self.receive()
            reply = json.loads(data)
            if reply.get("type") == "event":
                return reply


def _status() -> control.ProducerStatus:
    return control.ProducerStatus(
        "mock-1", [control.StreamStatus("head", 0), control.StreamStatus("wrist", 1)]
    )


class HandlerTest(unittest.TestCase):
    def setUp(self):
        self.status = _status()

    def handle(self, message) -> dict:
        text = message if isinstance(message, str) else json.dumps(message)
        return json.loads(control.handle_message(text, self.status))

    def test_ping_and_status_shape(self):
        self.assertEqual(
            self.handle({"id": 1, "type": "ping"}),
            {"id": 1, "ok": True, "type": "pong"},
        )
        reply = self.handle({"id": 2, "type": "get_status"})
        status = reply["status"]
        self.assertEqual(status["state"], "starting")
        self.assertEqual(status["release"], "mock-1")
        self.assertEqual([s["stream"] for s in status["streams"]], ["head", "wrist"])
        self.assertEqual(
            set(status["streams"][0]),
            {"stream", "sensor", "direct", "frames", "published"},
        )
        self.assertNotIn("fault", status)

    def test_every_failure_has_a_code(self):
        self.assertEqual(self.handle("[1]")["error"]["code"], "bad_json")
        self.assertEqual(self.handle({"id": 3})["error"]["code"], "missing_type")
        reply = self.handle({"id": 4, "type": "set_direct_target"})
        self.assertEqual(reply["error"]["code"], "unknown_type")
        self.assertEqual(reply["id"], 4)

    def test_fault_is_kept_only_for_fault_states(self):
        fault = {"code": "camera_power_missing", "message": "m"}
        self.status.set_state("camera_fault", fault)
        self.assertEqual(self.status.snapshot()["fault"], fault)
        self.status.set_state("streaming", fault)
        self.assertNotIn("fault", self.status.snapshot())


class ServerTest(unittest.TestCase):
    def setUp(self):
        self.status = _status()
        self.clock = ms.MockClock(7_000_000_000, 0.0)
        self.server = control.ControlServer(
            "127.0.0.1",
            0,
            TOKEN,
            lambda text: control.handle_message(text, self.status),
            self.clock.now_ns,
        )
        self.status.subscribe(self.server.broadcast)
        self.addCleanup(self.server.close)

    def test_wrong_or_missing_token_is_refused(self):
        for token in ("nope", None):
            client = Client(self.server.port, token=token)
            self.addCleanup(client.close)
            self.assertIn("401", client.status_line)

    def test_requests_and_pushed_events(self):
        client = Client(self.server.port)
        self.addCleanup(client.close)
        self.assertIn("101", client.status_line)
        self.assertEqual(client.request({"id": 1, "type": "ping"})["type"], "pong")
        self.status.set_state("streaming")
        event = client.event()
        self.assertEqual(event["event"], "state_changed")
        self.assertEqual(event["status"]["state"], "streaming")
        fault = {"code": "camera_no_frames", "message": "m"}
        self.status.set_state("camera_fault", fault)
        self.assertEqual(client.event()["event"], "state_changed")
        self.assertEqual(client.event()["error"], fault)

    def test_clock_sync_is_egocaps_binary_exchange(self):
        client = Client(self.server.port)
        self.addCleanup(client.close)
        before = self.clock.now_ns()
        client.send(struct.pack("<cIq", b"Q", 7, 123456789), opcode=0x2)
        opcode, data = client.receive()
        after = self.clock.now_ns()
        self.assertEqual((opcode, len(data)), (0x2, 29))
        kind, sequence, t1, t2, t3 = struct.unpack("<cIqqq", data)
        self.assertEqual((kind, sequence, t1), (b"R", 7, 123456789))
        self.assertTrue(before <= t2 <= t3 <= after)
        # The channel stays usable after an exchange.
        self.assertEqual(client.request({"id": 1, "type": "ping"})["type"], "pong")

    def test_other_binary_messages_are_refused(self):
        client = Client(self.server.port)
        self.addCleanup(client.close)
        client.send(b"Q", opcode=0x2)
        opcode, data = client.receive()
        self.assertEqual((opcode, struct.unpack(">H", data[:2])[0]), (0x8, 1003))

    def test_empty_token_is_rejected(self):
        with self.assertRaises(ValueError):
            control.ControlServer("127.0.0.1", 0, "", lambda text: text, time.time_ns)


@unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg with libx265 is required")
class MockFaultTest(unittest.TestCase):
    def test_status_answers_idle_then_follows_connection_and_fault(self):
        spec = ms.StreamSpec("t", 128, 96, 10)
        with tempfile.TemporaryDirectory() as tmp:
            sources = ms.ensure_sources([spec], Path(tmp))
            status = control.ProducerStatus("mock", [control.StreamStatus("t", 0)])
            sender = ms.MockSender(
                [spec],
                sources,
                ms.MockClock(0, 0.0),
                [],
                [],
                [ms.Fault("t", 0.5, 0.5, "camera_power_missing")],
                status,
            )
            listener = socket.create_server(("127.0.0.1", 0))
            thread = threading.Thread(
                target=sender.serve, args=(listener, 20.0), daemon=True
            )
            thread.start()
            self.addCleanup(thread.join, 5)
            self.addCleanup(listener.close)
            self.addCleanup(sender.stop_event.set)
            snap = status.snapshot()
            self.assertEqual(snap["state"], "starting")  # no receiver, still answering
            self.assertFalse(snap["streams"][0]["direct"]["connected"])
            conn = socket.create_connection(listener.getsockname(), timeout=5)
            self.addCleanup(conn.close)
            seen = set()
            deadline = time.monotonic() + 5
            while (
                time.monotonic() < deadline
                and not {"streaming", "camera_fault"} <= seen
            ):
                conn.recv(65536)
                snap = status.snapshot()
                seen.add(snap["state"])
                if snap["state"] == "camera_fault":
                    self.assertEqual(snap["fault"]["code"], "camera_power_missing")
            self.assertTrue({"streaming", "camera_fault"} <= seen, seen)
            self.assertGreater(status.snapshot()["streams"][0]["frames"], 0)


if __name__ == "__main__":
    unittest.main()
