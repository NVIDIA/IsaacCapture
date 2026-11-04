# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check launcher routing and one-shot replay ownership without native libraries."""

from contextlib import redirect_stdout
import io
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from isaaccapture_examples.synthetic_xr.launch import client_url
from isaaccapture_examples.synthetic_xr import replay_once


class LauncherTests(unittest.TestCase):
    def test_preserves_service_route_and_application(self):
        self.assertEqual(
            client_url(
                "https://host:48322/client/?serverIP=gpu&codec=h264#/sim/isaaclab"
            ),
            "https://host:48322/client/?serverIP=gpu&codec=h264&syntheticXR=1&immersiveMode=vr#/sim/isaaclab",
        )

    def test_replaces_disabled_flag_and_preserves_explicit_mode(self):
        self.assertEqual(
            client_url("https://host/?syntheticXR=0&immersiveMode=ar"),
            "https://host/?syntheticXR=1&immersiveMode=ar",
        )

    def test_rejects_local_files_and_credentials(self):
        for url in (
            "file:///tmp/index.html",
            "javascript:alert(1)",
            "https://user:secret@host/",
            "https:///client",
        ):
            with self.subTest(url=url), self.assertRaises(ValueError):
                client_url(url)


class ReplayTests(unittest.TestCase):
    def invoke(self, frames=12, error=None, duration=1.0):
        server = Mock()
        replay = SimpleNamespace(
            resolve_mcap=Mock(return_value=Path("input.mcap")),
            mcap_duration_s=Mock(return_value=duration),
            viser=SimpleNamespace(ViserServer=Mock(return_value=server)),
            setup_scene=Mock(),
            ControllerViz=Mock(),
            LEFT_COLOR=1,
            RIGHT_COLOR=2,
            run_once=Mock(return_value=frames, side_effect=error),
        )
        module = ModuleType("isaaccapture_examples.mcap_record_replay")
        module.replay_controller = replay
        patches = patch.dict(sys.modules, {module.__name__: module})
        with (
            patches,
            patch.object(
                sys, "argv", ["replay_once", "input.mcap", "--hold-seconds", "0"]
            ),
            redirect_stdout(io.StringIO()),
        ):
            try:
                replay_once.main()
            finally:
                self.replay, self.server = replay, server

    def test_replays_exactly_once_and_closes_viewer(self):
        self.invoke()
        self.replay.run_once.assert_called_once()
        self.server.stop.assert_called_once()

    def test_zero_frames_cannot_pass(self):
        with self.assertRaisesRegex(RuntimeError, "no frames"):
            self.invoke(frames=0)
        self.server.stop.assert_called_once()

    def test_native_failure_propagates_and_closes_viewer(self):
        with self.assertRaisesRegex(RuntimeError, "native failure"):
            self.invoke(error=RuntimeError("native failure"))
        self.server.stop.assert_called_once()

    def test_empty_recording_is_rejected_before_viewer_start(self):
        with redirect_stdout(io.StringIO()), self.assertRaises(SystemExit):
            self.invoke(duration=0)
        self.replay.viser.ViserServer.assert_not_called()


if __name__ == "__main__":
    unittest.main()
