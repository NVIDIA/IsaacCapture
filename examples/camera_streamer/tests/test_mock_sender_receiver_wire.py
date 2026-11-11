#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The mock sender's records must satisfy the receiver's AVS1 parser."""

from __future__ import annotations

import shutil
import socket
import struct
import sys
import tempfile
import threading
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "mock_sender"))
sys.path.insert(0, str(ROOT / "camera_receiver"))

import avs_wire  # noqa: E402
import mock_sender as ms  # noqa: E402

SPECS = [ms.StreamSpec("head", 128, 96, 10), ms.StreamSpec("wrist", 128, 96, 5)]


def _read_record(conn: socket.socket) -> bytes:
    def exact(n: int) -> bytes:
        data = b""
        while len(data) < n:
            chunk = conn.recv(n - len(data))
            if not chunk:
                raise EOFError
            data += chunk
        return data

    (length,) = struct.unpack("!I", exact(4))
    return exact(length)


@unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg with libx265 is required")
class MockSenderWireTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._dir = tempfile.TemporaryDirectory()
        cls.sources = ms.ensure_sources(SPECS, Path(cls._dir.name))

    @classmethod
    def tearDownClass(cls):
        cls._dir.cleanup()

    def _serve(self, **kwargs):
        sender = ms.MockSender(
            SPECS,
            self.sources,
            ms.MockClock(5_000_000_000, 0.0),
            kwargs.get("pauses", []),
            kwargs.get("disconnects", []),
        )
        listener = socket.create_server(("127.0.0.1", 0))
        thread = threading.Thread(
            target=sender.serve, args=(listener, 20.0), daemon=True
        )
        thread.start()
        self.addCleanup(thread.join, 5)
        self.addCleanup(listener.close)
        self.addCleanup(sender.stop_event.set)
        return sender, listener.getsockname()[1]

    def test_receiver_parser_accepts_records(self):
        _, port = self._serve()
        with socket.create_connection(("127.0.0.1", port), timeout=5) as conn:
            bundles = [
                avs_wire.parse_avs1_record(_read_record(conn)) for _ in range(12)
            ]
        self.assertEqual([b.sequence for b in bundles], list(range(1, 13)))
        self.assertEqual(len({b.session_id for b in bundles}), 1)
        first = bundles[0]
        self.assertEqual([s.codec_name for s in first.streams], ["hevc", "hevc"])
        self.assertTrue(first.streams[0].present)
        # The slower stream is present on every other bundle.
        present = [bundles[i].streams[1].present for i in range(4)]
        self.assertEqual(present, [False, True, False, True])
        for bundle in bundles:
            for stream in bundle.streams:
                for sample in stream.samples:
                    if sample.present:
                        self.assertIsNone(sample.sof_tsc_ns)
                        self.assertIsNone(sample.frame_timestamp_ns)

    def test_paused_stream_is_absent_and_resumes_at_idr(self):
        _, port = self._serve(pauses=[ms.Pause("head", 0.5, 1.0)])
        with socket.create_connection(("127.0.0.1", port), timeout=5) as conn:
            bundles = [
                avs_wire.parse_avs1_record(_read_record(conn)) for _ in range(30)
            ]
        head = [b.streams[0] for b in bundles]
        self.assertTrue(any(not s.present for s in head))
        resumed = next(
            i
            for i in range(1, len(head))
            if head[i].present and not head[i - 1].present
        )
        payload = head[resumed].samples[0].payload
        nal_types = {
            (payload[i + 3] >> 1) & 0x3F
            for i in range(len(payload) - 4)
            if payload[i : i + 3] == b"\x00\x00\x01"
        }
        self.assertTrue(nal_types & {19, 20}, "resume must start at an IDR")

    def test_reconnect_starts_a_new_session_at_idr(self):
        _, port = self._serve(disconnects=[0.6])
        sessions = []
        for _ in range(2):
            with socket.create_connection(("127.0.0.1", port), timeout=5) as conn:
                first = avs_wire.parse_avs1_record(_read_record(conn))
                sessions.append(first.session_id)
                self.assertEqual(first.sequence, 1)
                with self.assertRaises((EOFError, OSError)):
                    while True:
                        _read_record(conn)
        self.assertNotEqual(sessions[0], sessions[1])


if __name__ == "__main__":
    unittest.main()
