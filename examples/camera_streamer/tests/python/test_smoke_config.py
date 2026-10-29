# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "hardware"))
from avs1_smoke import configured_port  # noqa: E402


class SmokeConfigTests(unittest.TestCase):
    def test_configured_port_and_sender_default(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "camera.yaml"
            for text, expected in [("network: {port: 54321}", 54321), ("{}", 5000)]:
                path.write_text(text)
                self.assertEqual(configured_port(str(path)), expected)
            for value in [0, 65536, True, "not-a-port"]:
                path.write_text(f"network: {{port: {value}}}")
                with self.assertRaises(ValueError):
                    configured_port(str(path))
