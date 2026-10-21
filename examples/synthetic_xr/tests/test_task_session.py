# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Failure controls for application hooks and the loopback task boundary."""

import json
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
import time
import unittest
from unittest.mock import Mock, patch
from urllib.error import URLError

import numpy as np

# Keep helper imports above fixtures to preserve import-only test collection.
from isaaccapture_examples.synthetic_xr.franka.observer_process import (
    completion_snapshot,
)
from isaaccapture_examples.synthetic_xr.franka.browser_command import BrowserClient
from isaaccapture_examples.synthetic_xr.franka.session import (
    FrankaTaskSession,
    validate_observation,
)
from isaaccapture_examples.synthetic_xr.franka.task_hooks import (
    check_driver_lease,
    evaluate_success,
    load_success_hook,
)
from isaaccapture_examples.synthetic_xr.franka.xr_controller_bridge import (
    XRControllerBridge,
)


def observation(root):
    return {
        "schema_version": 1,
        "run_id": "one",
        "coordinate_convention": "metres; world Z-up; quaternion xyzw",
        "run_dir": str(root),
        "phase": "running",
        "timestamp": time.time(),
        "simulation_dt": 0.05,
        "ee_positions_world": [[[0.4, 0, 0.2]]],
        "ee_quaternions_world": [[[0, 0, 0, 1]]],
        "cubes": {
            "cube_1": {
                "position_world": [[0.4, 0, 0.02]],
                "quaternion_world": [[0, 0, 0, 1]],
                "linear_velocity_world": [[0, 0, 0]],
            }
        },
        "exported_successful_episodes": 0,
    }


class TaskTests(unittest.TestCase):
    def test_observer_help_needs_no_lab_or_output(self):
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "isaaccapture_examples.synthetic_xr.franka.observe_record_demos",
                "--help",
            ],
            check=False,  # Assert the exit code below, retaining stderr on failure.
            capture_output=True,
            text=True,
            timeout=10,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--success", result.stdout)
        self.assertIn("Additional arguments are forwarded", result.stdout)

    def test_hook_cannot_modify_original_and_must_return_bool(self):
        original = {"nested": {"value": 4}}

        def callback(value):
            value["nested"]["value"] = 9
            return False

        self.assertFalse(evaluate_success(callback, original))
        self.assertEqual(original["nested"]["value"], 4)
        for result in (None, 1, "PASS", np.bool_(True), [False]):
            with self.subTest(result=result), self.assertRaises(TypeError):
                # Bind each negative-control value; deferred callbacks must not capture the loop cell.
                evaluate_success(lambda _, result=result: result, original)

    def test_driver_lease_detects_crash_and_wrong_session(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "heartbeat.json"
            check_driver_lease(path, "one", 30)
            path.write_text(
                json.dumps({"run_id": "one", "monotonic": time.monotonic()})
            )
            check_driver_lease(path, "one", 30)
            for identity, age in (("two", 0), ("one", 31), ("one", -10)):
                path.write_text(
                    json.dumps(
                        {"run_id": identity, "monotonic": time.monotonic() - age}
                    )
                )
                with self.assertRaises(RuntimeError):
                    check_driver_lease(path, "one", 30)

    def test_hook_exception_propagates(self):
        def broken(_):
            raise RuntimeError("criterion failed")

        with self.assertRaisesRegex(RuntimeError, "criterion failed"):
            evaluate_success(broken, {})
        with self.assertRaises(ValueError):
            load_success_hook("not_a_module_function")

    def test_stale_future_and_missing_timestamp_rejected(self):
        client = BrowserClient()
        for timestamp in (0, time.time() - 60, time.time() + 60, float("nan")):
            with (
                patch.object(
                    client, "_read_json", return_value={"timestamp": timestamp}
                ),
                self.assertRaises(RuntimeError),
            ):
                client.state()

    def test_endpoints_reject_remote_hosts_and_userinfo(self):
        for url in (
            "http://example.com:9222",
            "https://localhost",
            "http://user:secret@localhost",
            "http://localhost#fragment",
        ):
            with self.subTest(url=url), self.assertRaises(ValueError):
                BrowserClient(url)

    def test_sessions_do_not_share_browser_connection(self):
        first, second = BrowserClient(), BrowserClient()
        first._socket = Mock()
        second.close()
        first._socket.close.assert_not_called()
        first.close()
        self.assertIsNone(first._socket)

    def test_invalid_feedback_shape_and_nan_rejected(self):
        for bad in ([[[1, 2]]], [[[1, 2, float("nan")]]]):
            value = observation(Path("/tmp/observer"))
            value["ee_positions_world"] = bad
            with self.assertRaises(ValueError):
                validate_observation(value)

    def test_native_tool_and_finger_frames_are_accepted(self):
        value = observation(Path("/tmp/observer"))
        value["ee_positions_world"] = [
            [[0.4, 0, 0.2], [0.4, 0.04, 0.2], [0.4, -0.04, 0.2]]
        ]
        value["ee_quaternions_world"] = [[[0, 0, 0, 1]] * 3]
        self.assertEqual(
            validate_observation(value)["ee_positions_world"][0][0], [0.4, 0, 0.2]
        )
        value["ee_quaternions_world"] = [[[0, 0, 0, 1]]]
        with self.assertRaisesRegex(ValueError, "counts differ"):
            validate_observation(value)

    def test_terminal_snapshot_requires_matching_identity_and_clean_completion(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            session = FrankaTaskSession(observation_dir=root, output=root / "task")
            session.run_id = "one"
            session.client.state = Mock(side_effect=URLError("disconnected"))
            for phase, identity, error, passes in (
                ("running", "one", None, False),
                ("completed", "two", None, False),
                ("completed", "one", "failure", False),
                ("completed", "one", None, True),
            ):
                value = observation(root)
                value.update(phase=phase, run_id=identity, completion_error=error)
                (root / "sim-state.json").write_text(json.dumps(value))
                if passes:
                    self.assertEqual(session.state()["phase"], "completed")
                else:
                    with self.assertRaises(RuntimeError):
                        session.state()

    def test_export_requires_native_completion_and_never_writes_success(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            session = FrankaTaskSession(observation_dir=root, output=root / "task")
            value = observation(root)
            value.update(phase="completed")
            session.state = Mock(return_value=value)
            with self.assertRaisesRegex(RuntimeError, "without a native successful"):
                session.wait_for_export()
            self.assertFalse((root / "task/result.json").exists())
            value.update(phase="running", exported_successful_episodes=1)
            with self.assertRaises(TimeoutError):
                session.wait_for_export(timeout=0.001)

    def test_terminal_worker_error_is_reported_without_polling(self):
        for offline in (False, True):
            for initialized in (False, True):
                with (
                    self.subTest(offline=offline, initialized=initialized),
                    TemporaryDirectory() as directory,
                ):
                    root = Path(directory)
                    if initialized:
                        value = observation(root)
                        value["exported_successful_episodes"] = 1
                        (root / "sim-state.json").write_text(json.dumps(value))
                    value = completion_snapshot(root, "one", 23)
                    (root / "sim-state.json").write_text(json.dumps(value))
                    session = FrankaTaskSession(
                        observation_dir=root, output=root / "task"
                    )
                    session.client.state = (
                        Mock(side_effect=URLError("disconnected"))
                        if offline
                        else Mock(return_value=value)
                    )
                    with (
                        patch(
                            "isaaccapture_examples.synthetic_xr.franka.session.time.sleep",
                            side_effect=AssertionError(
                                "Terminal failures must not retry"
                            ),
                        ) as sleep,
                        self.assertRaisesRegex(
                            RuntimeError,
                            "Observer failed: Worker exited with status 23",
                        ),
                    ):
                        session.wait_for_export()
                    sleep.assert_not_called()
                    self.assertFalse((root / "task/result.json").exists())

    def test_export_retries_nonterminal_snapshot_until_orderly_completion(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "task"
            output.mkdir()
            value = observation(root)
            (root / "sim-state.json").write_text(json.dumps(value))
            session = FrankaTaskSession(observation_dir=root, output=output)
            session.run_id = "one"
            session.client.state = Mock(side_effect=URLError("disconnected"))
            session.mark = Mock()

            def finish_worker(_):
                value.update(
                    exported_successful_episodes=1, native_environment_closed=True
                )
                (root / "sim-state.json").write_text(json.dumps(value))
                (root / "sim-state.json").write_text(
                    json.dumps(completion_snapshot(root, "one", 0))
                )

            with patch(
                "isaaccapture_examples.synthetic_xr.franka.session.time.sleep",
                side_effect=finish_worker,
            ) as sleep:
                self.assertEqual(session.wait_for_export()["phase"], "completed")
            sleep.assert_called_once()
            self.assertEqual(
                json.loads((output / "result.json").read_text())["status"], "PASS"
            )

    def test_invalid_tool_and_gripper_inputs_never_send_motion(self):
        session = FrankaTaskSession(observation_dir="/tmp/observer", output="/tmp/task")
        session.controller = Mock()
        for point, quat in (
            ([0, 0], [0, 0, 0, 1]),
            ([0, 0, 1], [0, 0, 0, 0]),
            ([np.inf, 0, 1], [0, 0, 0, 1]),
        ):
            with self.assertRaises(ValueError):
                session.move_tool(point, quat)
        for closed in (1, "false", None):
            with self.assertRaises(TypeError):
                session.set_gripper(closed=closed)
        session.controller.move_world.assert_not_called()

    def test_improper_rotation_rejected_before_browser_command(self):
        bridge = XRControllerBridge.__new__(XRControllerBridge)
        bridge.client = Mock()
        with self.assertRaises(ValueError):
            bridge.target_base([0.4, 0, 0.2], np.diag([1, 1, -1]))
        bridge.client.command.assert_not_called()

    def test_cleanup_failure_overrides_pass_and_still_requests_observer_stop(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "task"
            output.mkdir()
            (root / "sim-state.json").write_text(json.dumps(observation(root)))
            (output / "result.json").write_text('{"status":"PASS"}')
            session = FrankaTaskSession(observation_dir=root, output=output)
            session.run_id = "one"
            session.client.command = Mock(side_effect=RuntimeError("disconnect"))
            with self.assertRaisesRegex(RuntimeError, "disconnect"):
                session.__exit__(None, None, None)
            self.assertTrue((root / "stop").exists())
            self.assertEqual(
                json.loads((output / "result.json").read_text())["status"], "FAIL"
            )

    def test_export_alone_cannot_mask_worker_failure_or_missing_close(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            value = observation(root)
            value["exported_successful_episodes"] = 1
            for closed, code, interrupted, valid in (
                (False, 0, False, False),
                (True, 1, False, False),
                (True, 0, True, False),
                (True, 0, False, True),
            ):
                value["native_environment_closed"] = closed
                (root / "sim-state.json").write_text(json.dumps(value))
                result = completion_snapshot(root, "one", code, interrupted)
                self.assertEqual(result["completion_error"] is None, valid)

    def test_second_driver_cannot_send_input_or_stop_first_driver(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "driver-owner.json").write_text('{"run_id":"one","pid":1}')
            session = FrankaTaskSession(observation_dir=root, output=root / "second")
            session.client.state = Mock(return_value=observation(root))
            session.client.command = Mock()
            with self.assertRaises(FileExistsError):
                session.__enter__()
            session.client.command.assert_not_called()
            self.assertFalse((root / "stop").exists())

    def test_unbounded_timeouts_rejected_before_io(self):
        session = FrankaTaskSession(observation_dir="/tmp/observer", output="/tmp/task")
        for timeout in (-1, 0, float("nan"), float("inf"), 301):
            with self.subTest(timeout=timeout):
                with self.assertRaises(ValueError):
                    session.wait_for_export(timeout)
                with self.assertRaises(ValueError):
                    session.client.command("inspect", timeout=timeout)

    def test_stack_module_is_importable_without_starting_task(self):
        import isaaccapture_examples.synthetic_xr.franka.stack_logic as stack

        self.assertTrue(callable(stack.main))


if __name__ == "__main__":
    unittest.main()
