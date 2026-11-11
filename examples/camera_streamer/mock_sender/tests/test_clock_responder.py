# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import socket
import struct
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import clock_responder  # noqa: E402


class ClockResponderTest(unittest.TestCase):
    def setUp(self):
        self.now = 1_000
        self.muted = False

        def tick():
            self.now += 10
            return self.now

        self.responder = clock_responder.ClockResponder(
            "127.0.0.1", 0, tick, lambda: self.muted
        )
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.settimeout(0.3)
        self.addr = ("127.0.0.1", self.responder.port)

    def tearDown(self):
        self.sock.close()
        self.responder.close()

    def test_reply_layout(self):
        self.sock.sendto(struct.pack("<cIq", b"Q", 7, -5), self.addr)
        data = self.sock.recv(64)
        self.assertEqual(len(data), 29)
        tag, seq, t1, t2, t3 = struct.unpack("<cIqqq", data)
        self.assertEqual((tag, seq, t1), (b"R", 7, -5))
        self.assertLess(t2, t3)

    def test_bad_datagrams_are_ignored(self):
        for bad in (
            b"",
            b"Q",
            struct.pack("<cIq", b"X", 1, 1),
            struct.pack("<cIq", b"Q", 1, 1) + b"\0",
        ):
            self.sock.sendto(bad, self.addr)
        with self.assertRaises(TimeoutError):
            self.sock.recv(64)
        self.sock.sendto(struct.pack("<cIq", b"Q", 2, 1), self.addr)
        self.assertEqual(len(self.sock.recv(64)), 29)

    def test_muted_window_gets_no_reply(self):
        self.muted = True
        self.sock.sendto(struct.pack("<cIq", b"Q", 1, 1), self.addr)
        with self.assertRaises(TimeoutError):
            self.sock.recv(64)


if __name__ == "__main__":
    unittest.main()
