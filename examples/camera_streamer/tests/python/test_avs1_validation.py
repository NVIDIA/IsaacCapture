# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Protocol-only regressions runnable without the GPU receiver or pytest."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "camera_receiver"))

from avs_wire import HEADER, SAMPLE, STREAM, parse_avs1_record  # noqa: E402


def record(first_samples=(0, 1), extra_sample=False):
    streams = b"".join(
        STREAM.pack(i, 1, 1, 1, 0, 1280, 720, 30, 1, 1, first, 1, 0)
        for i, first in enumerate(first_samples)
    )
    samples = b"".join(
        SAMPLE.pack(i, 0, 0, 1, 0, i, 0, 1, 0, 0, 0, 0, 0, i, 1) for i in range(2)
    )
    if extra_sample:
        samples += bytes(SAMPLE.size)
    header = HEADER.pack(
        b"AVS1",
        1,
        0,
        HEADER.size,
        2,
        2 + extra_sample,
        1,
        1,
        0,
        0,
        len(streams) + len(samples),
        2,
        0,
    )
    return header + streams + samples + b"LR"


class DescriptorCoverageTests(unittest.TestCase):
    def test_valid_mono_streams(self):
        bundle = parse_avs1_record(record())
        self.assertEqual(
            [bytes(stream.samples[0].payload) for stream in bundle.streams],
            [b"L", b"R"],
        )

    def test_rejects_unreferenced_descriptor(self):
        with self.assertRaisesRegex(ValueError, "not fully referenced"):
            parse_avs1_record(record(extra_sample=True))

    def test_rejects_noncontiguous_stream_ranges(self):
        for first_samples in [(1, 0), (0, 0), (0, 2)]:
            with self.subTest(first_samples=first_samples):
                with self.assertRaisesRegex(ValueError, "not contiguous"):
                    parse_avs1_record(record(first_samples))

    def test_rejects_range_past_descriptor_table(self):
        packet = bytearray(record())
        fields = list(STREAM.unpack_from(packet, HEADER.size + STREAM.size))
        fields[1] |= 2  # Stereo requires two samples, but only one remains.
        fields[11] = 2
        STREAM.pack_into(packet, HEADER.size + STREAM.size, *fields)
        with self.assertRaisesRegex(ValueError, "exceeds descriptor table"):
            parse_avs1_record(packet)


if __name__ == "__main__":
    unittest.main()
