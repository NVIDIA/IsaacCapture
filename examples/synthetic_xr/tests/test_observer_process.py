# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shutdown retains process ownership and writes completion evidence."""

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, call, patch

from isaaccapture_examples.synthetic_xr.franka import observer_process as observer


@unittest.skipUnless(os.name == "posix", "POSIX process groups")
class ObserverShutdownTests(unittest.TestCase):
    def test_signal_handler_neither_reaps_nor_signals_the_child(self):
        process = Mock(pid=4242)
        process.poll.return_value = None
        process.wait.return_value = 0
        handlers = {}

        def install(signum, handler):
            previous = handlers.get(signum, signal.SIG_DFL)
            handlers[signum] = handler
            return previous

        def interrupt(_seconds):
            polls = process.poll.call_count
            handlers[signal.SIGTERM](signal.SIGTERM, None)
            self.assertEqual(process.poll.call_count, polls)
            send.assert_not_called()

        with (
            TemporaryDirectory() as directory,
            patch.object(observer.subprocess, "Popen", return_value=process),
            patch.object(observer.signal, "signal", side_effect=install),
            patch.object(observer.time, "sleep", side_effect=interrupt),
            patch.object(observer.os, "killpg") as send,
        ):
            root = Path(directory) / "run"
            self.assertEqual(observer.supervise(root, [], timeout=60), 1)
            send.assert_called_once_with(process.pid, signal.SIGINT)
            self.assertTrue(
                json.loads((root / "worker-exit.json").read_text())["interrupted"]
            )
        self.assertEqual(
            handlers, {signal.SIGINT: signal.SIG_DFL, signal.SIGTERM: signal.SIG_DFL}
        )

    def test_exit_between_poll_and_signal_still_reaps_and_writes_snapshots(self):
        process = Mock(pid=4242)
        process.poll.return_value = None
        process.wait.return_value = 7
        with (
            TemporaryDirectory() as directory,
            patch.object(observer.subprocess, "Popen", return_value=process),
            patch.object(observer.os, "killpg", side_effect=ProcessLookupError),
        ):
            root = Path(directory) / "run"
            self.assertEqual(observer.supervise(root, [], timeout=0), 1)
            process.wait.assert_called_once_with(timeout=5)
            exit_record = json.loads((root / "worker-exit.json").read_text())
            self.assertEqual(exit_record["returncode"], 7)
            self.assertEqual(exit_record["shutdown_signals"], ["SIGINT"])
            state = json.loads((root / "sim-state.json").read_text())
            self.assertEqual(state["phase"], "completed")
            self.assertEqual(state["worker_returncode"], 7)
            self.assertEqual(
                json.loads((root / "sim-feedback.jsonl").read_text()), state
            )

    def test_timeout_keeps_group_escalation(self):
        process = Mock(pid=4242)
        process.poll.return_value = None
        process.wait.side_effect = [subprocess.TimeoutExpired("worker", 20), -15, -15]
        with (
            TemporaryDirectory() as directory,
            patch.object(observer.subprocess, "Popen", return_value=process),
            patch.object(observer.os, "killpg") as send,
        ):
            root = Path(directory) / "run"
            self.assertEqual(observer.supervise(root, [], timeout=0), 1)
            self.assertEqual(
                send.call_args_list,
                [call(4242, signal.SIGINT), call(4242, signal.SIGTERM)],
            )
            self.assertEqual(
                json.loads((root / "worker-exit.json").read_text())["shutdown_signals"],
                ["SIGINT", "SIGTERM"],
            )

    def test_real_session_leader_is_stopped_and_reaped(self):
        popen = subprocess.Popen
        processes = []

        def stand_in(_command, **kwargs):
            self.assertTrue(kwargs["start_new_session"])
            process = popen(
                [sys.executable, "-c", "import time; time.sleep(60)"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                **kwargs,
            )
            processes.append(process)
            return process

        with TemporaryDirectory() as directory:
            try:
                root = Path(directory) / "run"
                with patch.object(observer.subprocess, "Popen", side_effect=stand_in):
                    self.assertEqual(observer.supervise(root, [], timeout=0), 1)
                self.assertIsNotNone(processes[0].returncode)
                self.assertEqual(
                    json.loads((root / "worker-exit.json").read_text())["returncode"],
                    processes[0].returncode,
                )
            finally:
                for process in processes:
                    if process.poll() is None:
                        process.kill()
                    process.wait(timeout=5)


if __name__ == "__main__":
    unittest.main()
