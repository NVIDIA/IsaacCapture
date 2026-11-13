#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mock sender: access-unit splitting, stream scheduling and pause behaviour."""

from __future__ import annotations

import argparse
import shutil
import socket
import sys
import tempfile
import threading
import time
import unittest
import unittest.mock
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mock_sender as ms  # noqa: E402

SPEC = ms.StreamSpec("t", 128, 96, 10)


@unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg with libx265 is required")
class MockSenderTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._dir = tempfile.TemporaryDirectory()
        cls.path = Path(cls._dir.name) / "t.hevc"
        ms.make_stream_file(cls.path, SPEC)
        cls.units = ms.split_access_units(cls.path.read_bytes())

    @classmethod
    def tearDownClass(cls):
        cls._dir.cleanup()

    def test_one_unit_per_frame_and_gop_period(self):
        self.assertEqual(len(self.units), ms.LOOP_SECONDS * SPEC.fps)
        idr_positions = [i for i, (_, idr) in enumerate(self.units) if idr]
        self.assertEqual(idr_positions[:2], [0, SPEC.fps * ms.GOP_SECONDS])

    def test_units_reassemble_to_the_stream(self):
        self.assertEqual(b"".join(au for au, _ in self.units), self.path.read_bytes())

    def test_cursor_starts_and_restarts_at_idr(self):
        cursor = ms._Cursor(SPEC, self.units)
        self.assertEqual(cursor.next_unit(), self.units[0][0])
        cursor.position = 3  # mid-GOP
        cursor.need_idr = True
        self.assertEqual(cursor.next_unit(), self.units[SPEC.fps * ms.GOP_SECONDS][0])

    def test_mixed_rate_streams_are_due_at_their_own_rate(self):
        slow = ms.StreamSpec("s", 128, 96, 5)
        base = max(SPEC.fps, slow.fps)
        due = [(t + 1) * slow.fps // base > t * slow.fps // base for t in range(10)]
        self.assertEqual(sum(due), 5)


# One IDR access unit: AUD followed by an IDR_W_RADL slice with first_slice_segment set.
_AU = b"\x00\x00\x00\x01\x46\x01\x50" + b"\x00\x00\x00\x01\x26\x01\x80\xaa\xbb"


def _write_valid(path: Path) -> None:
    path.write_bytes(_AU * 4)


class EnsureSourcesTest(unittest.TestCase):
    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._dir.cleanup)
        self.cache = Path(self._dir.name)
        self.final = self.cache / f"t_{SPEC.width}x{SPEC.height}_{SPEC.fps}.hevc"
        which = unittest.mock.patch.object(
            ms.shutil, "which", return_value="/usr/bin/ffmpeg"
        )
        which.start()
        self.addCleanup(which.stop)

    def _ensure(self, encoder):
        with unittest.mock.patch.object(
            ms, "make_stream_file", side_effect=encoder
        ) as enc:
            sources = ms.ensure_sources([SPEC], self.cache)
        return sources, enc

    def test_valid_cache_is_not_re_encoded(self):
        _write_valid(self.final)
        before = self.final.read_bytes()
        sources, enc = self._ensure(lambda *_: self.fail("re-encoded a valid cache"))
        enc.assert_not_called()
        self.assertEqual(sources[SPEC], self.final)
        self.assertEqual(self.final.read_bytes(), before)

    def test_empty_cache_file_is_regenerated(self):
        self.final.write_bytes(b"")
        sources, enc = self._ensure(lambda path, spec: _write_valid(path))
        enc.assert_called_once()
        self.assertTrue(ms._has_access_units(self.final))
        self.assertEqual(sources[SPEC], self.final)
        self.assertEqual([p.name for p in self.cache.iterdir()], [self.final.name])

    def test_cache_file_without_access_unit_is_regenerated(self):
        self.final.write_bytes(b"not an hevc stream")
        _, enc = self._ensure(lambda path, spec: _write_valid(path))
        enc.assert_called_once()
        self.assertTrue(ms._has_access_units(self.final))
        self.assertFalse(list(self.cache.glob("*.partial")))

    def test_missing_cache_is_encoded_through_partial_file(self):
        seen = []

        def encoder(path, spec):
            seen.append(path)
            _write_valid(path)

        self._ensure(encoder)
        self.assertEqual(seen[0].name, self.final.name + ".partial")
        self.assertFalse(seen[0].exists())

    def test_empty_encode_raises_and_leaves_no_cache_file(self):
        with self.assertRaises(SystemExit):
            self._ensure(lambda path, spec: path.write_bytes(b""))
        self.assertFalse(self.final.exists())

    def test_missing_ffmpeg_raises(self):
        with unittest.mock.patch.object(ms.shutil, "which", return_value=None):
            with self.assertRaises(SystemExit):
                ms.ensure_sources([SPEC], self.cache)


class ServeDurationTest(unittest.TestCase):
    def _sender(self):
        path = Path(self._dir.name) / "t.hevc"
        _write_valid(path)
        return ms.MockSender(
            [SPEC], {SPEC: path}, ms.MockClock(0, 0.0), pauses=[], disconnects_s=[]
        )

    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._dir.cleanup)
        self.listener = socket.socket()
        self.addCleanup(self.listener.close)
        self.listener.bind(("127.0.0.1", 0))
        self.listener.listen(1)
        self.port = self.listener.getsockname()[1]

    def _serve_timed(self, duration_s, sender):
        started = time.monotonic()
        thread = threading.Thread(
            target=sender.serve, args=(self.listener, duration_s), daemon=True
        )
        thread.start()
        return started, thread

    def test_returns_after_duration_with_a_reading_receiver(self):
        sender = self._sender()
        started, thread = self._serve_timed(1.0, sender)
        received = 0
        with socket.create_connection(("127.0.0.1", self.port)) as rx:
            rx.settimeout(5)
            while True:
                data = rx.recv(65536)
                if not data:
                    break
                received += len(data)
        thread.join(5)
        elapsed = time.monotonic() - started
        self.assertFalse(thread.is_alive())
        self.assertGreater(received, 0)
        self.assertGreaterEqual(elapsed, 0.9)
        self.assertLess(elapsed, 2.0)

    def test_returns_after_duration_when_receiver_never_reads(self):
        sender = self._sender()
        # Padded records fill the socket buffers so sendall blocks on the idle receiver.
        build = ms.build_record
        padding = b"\x00" * (1 << 20)

        def padded(*args, **kwargs):
            return build(*args, **kwargs) + padding

        with unittest.mock.patch.object(ms, "build_record", side_effect=padded):
            started, thread = self._serve_timed(1.0, sender)
            rx = socket.create_connection(("127.0.0.1", self.port))
            self.addCleanup(rx.close)
            thread.join(5)
        elapsed = time.monotonic() - started
        self.assertFalse(thread.is_alive())
        self.assertLess(elapsed, 2.5)
        self.assertGreater(sender.bundles_sent, 0)


class ParseArgsTest(unittest.TestCase):
    def test_stream_spec(self):
        self.assertEqual(
            ms.parse_stream_spec("head:2560x1984@30"),
            ms.StreamSpec("head", 2560, 1984, 30),
        )
        with self.assertRaises(argparse.ArgumentTypeError):
            ms.parse_stream_spec("head:2561x1984@30")

    def test_pause(self):
        self.assertEqual(ms.parse_pause("wrist1@10+2.5"), ms.Pause("wrist1", 10.0, 2.5))


if __name__ == "__main__":
    unittest.main()
