#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mock sender: access-unit splitting, stream scheduling and pause behaviour."""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import unittest
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
