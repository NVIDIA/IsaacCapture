# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest

from nv_video_decode import NvVideoDecoder


class _FakeNativeDecoder:
    def __init__(self) -> None:
        self.calls = []

    def decode(self, *args):
        self.calls.append(("rgba", args))
        return True

    def decode_nv12(self, *args):
        self.calls.append(("nv12", args))
        return True


class NvVideoDecoderRotationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.decoder = NvVideoDecoder(2560, 1984, codec="hevc")
        self.native = _FakeNativeDecoder()
        self.decoder._decoder = self.native

    def test_rgba_rotation_is_forwarded(self) -> None:
        output = object()
        self.assertTrue(self.decoder.decode(b"au", output, rotation=180))
        self.assertEqual(
            self.native.calls,
            [("rgba", (b"au", output, 2560, 1984, 180))],
        )

    def test_nv12_rotation_is_forwarded(self) -> None:
        output = object()
        self.assertTrue(self.decoder.decode_nv12(b"au", output, rotation=180))
        self.assertEqual(
            self.native.calls,
            [("nv12", (b"au", output, 2560, 1984, 180))],
        )


if __name__ == "__main__":
    unittest.main()
