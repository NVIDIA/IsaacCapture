# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Counterexamples for motion admission, measured poses and episode boundaries."""

import copy
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

import numpy as np

from isaaccapture_examples.synthetic_xr.franka.session import (
    FrankaTaskSession,
    validate_observation,
)
from isaaccapture_examples.synthetic_xr.franka.xr_controller_bridge import (
    XRControllerBridge,
)
from isaaccapture_examples.synthetic_xr.franka.task_hooks import install_reset_observer
from test_task_session import observation

BRIDGE = "isaaccapture_examples.synthetic_xr.franka.xr_controller_bridge"


def bridge(client=None):
    value = XRControllerBridge.__new__(XRControllerBridge)
    value.base_to_world = np.eye(4)
    value.base_to_anchor = np.eye(4)
    value.aim_to_grip = np.eye(4)
    value.offset = np.eye(3)
    value.target = np.eye(4)
    value.target[:3, 3] = [0.4, 0, 0.2]
    value.trigger = 0.0
    value.ack_trace = io.StringIO()
    value.client = client or Mock()
    return value


def feedback(step):
    return dict(
        steps=step,
        timestamp=101.0,
        simulation_dt=0.05,
        exported_successful_episodes=0,
        last_action=[[0.4, 0, 0.2, 0, 0, 0, 1, 1]],
    )


class MotionGuards(unittest.TestCase):
    def test_invalid_endpoint_or_trigger_has_no_input_side_effects(self):
        for point, rotation, trigger in (
            ([1.6, 0, 0.2], np.eye(3), None),
            ([0.4, 0, 0.2], np.eye(3), float("nan")),
            ([0.4, 0, 0.2], np.eye(3), 2),
            ([0.4, 0, 0.2], np.diag([1, 1, -1]), None),
        ):
            with self.subTest(point=point, trigger=trigger):
                value = bridge()
                initial = value.target.copy()
                value.client.state.return_value = feedback(10)
                real_target = value.target_base
                # Bind the callable to this subtest, not the loop variable.
                value.target_base = lambda p, r, t, target=real_target: target(
                    p, r, t, wait_receipt=False
                )
                with patch(BRIDGE + ".wait_steps"), self.assertRaises(ValueError):
                    value.move_world(point, rotation, trigger)
                value.client.command.assert_not_called()
                np.testing.assert_array_equal(value.target, initial)
                self.assertEqual(value.trigger, 0)

    def test_workspace_is_checked_after_world_to_base_transform(self):
        value = bridge()
        value.base_to_world[:3, 3] = [-2, 0, 0]
        value.client.state.return_value = feedback(10)
        value.target_base = Mock()
        with patch(BRIDGE + ".wait_steps"):
            value.move_world([2.5, 0, 0.2], np.eye(3))
        np.testing.assert_allclose(value.target_base.call_args.args[0], [0.5, 0, 0.2])
        value.target_base.reset_mock()
        with self.assertRaises(ValueError):
            value.move_world([0.4, 0, 0.2], np.eye(3))
        value.target_base.assert_not_called()

    def test_public_tool_offset_endpoint_is_checked_before_input(self):
        value = bridge()
        value.client.state.return_value = feedback(10)
        real_target = value.target_base
        value.target_base = lambda p, r, t: real_target(p, r, t, wait_receipt=False)
        session = FrankaTaskSession(observation_dir="/tmp/observer", output="/tmp/task")
        session.controller = value
        # This tool target maps outside the hand workspace after its offset.
        with patch(BRIDGE + ".wait_steps"), self.assertRaises(ValueError):
            session.move_tool([0, 0, -1.45], [0, 0, 0, 1])
        value.client.command.assert_not_called()

    def test_position_alone_cannot_claim_pose_convergence(self):
        session = FrankaTaskSession(observation_dir="/tmp/observer", output="/tmp/task")
        session.controller = bridge()
        session.controller.move_world = Mock()
        session.state = Mock(return_value=observation(Path("/tmp/observer")))
        with self.assertRaisesRegex(RuntimeError, "converge"):
            session.move_tool([0.4, 0, 0.2], [1, 0, 0, 0])

    def test_orientation_only_move_waits_and_quaternion_sign_is_equivalent(self):
        session = FrankaTaskSession(observation_dir="/tmp/observer", output="/tmp/task")
        session.controller = bridge()
        session.controller.move_world = Mock()
        wrong = observation(Path("/tmp/observer"))
        correct = copy.deepcopy(wrong)
        correct["ee_quaternions_world"] = [[[-1, 0, 0, 0]]]
        session.state = Mock(side_effect=[wrong, correct])
        self.assertIs(session.move_tool([0.4, 0, 0.2], [1, 0, 0, 0]), correct)
        self.assertEqual(session.controller.move_world.call_count, 2)

    def test_measured_quaternion_and_counters_are_validated(self):
        for name, invalid in (
            ("ee_quaternions_world", [[[0, 0, 0, 0]]]),
            ("episode_id", -1),
            ("steps", True),
            ("exported_successful_episodes", 0.5),
        ):
            with self.subTest(name=name):
                value = observation(Path("/tmp/observer"))
                value[name] = invalid
                with self.assertRaises(ValueError):
                    validate_observation(value)

    def test_reset_and_step_regression_latch_failure_and_release_input(self):
        for changes in ({"episode_id": 2, "steps": 11}, {"steps": 9}):
            with self.subTest(changes=changes), TemporaryDirectory() as directory:
                root = Path(directory)
                session = FrankaTaskSession(observation_dir=root, output=root / "task")
                session.run_id = "one"
                value = observation(root)
                session.client.state = Mock(return_value=value)
                session.client.command = Mock(return_value={})
                session.observe()
                value.update(changes)
                with self.assertRaisesRegex(RuntimeError, "Episode reset"):
                    session.observe()
                with self.assertRaisesRegex(RuntimeError, "Episode reset"):
                    session.command("pose", buttons={"trigger": 1})
                session.client.command.assert_called_once_with("stop")

    def test_reset_during_native_ack_aborts_remaining_interpolation(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "task"
            output.mkdir()
            session = FrankaTaskSession(observation_dir=root, output=output)
            session.run_id = "one"
            value = observation(root)
            session.client.state = Mock(return_value=value)
            session.observe()
            session.controller = bridge(session)
            actions = []

            def send(action, **kwargs):
                actions.append(action)
                if action == "pose":
                    value.update(episode_id=2, steps=11)
                return {"timestamp": 0, "id": "command"}

            session.client.command = Mock(side_effect=send)
            (root / "sim-state.json").write_text(json.dumps(value))
            with self.assertRaisesRegex(RuntimeError, "Episode reset") as caught:
                session.move_tool([0.5, 0, 0.2], [0, 0, 0, 1])
            self.assertEqual(actions, ["pose", "stop"])
            session.__exit__(RuntimeError, caught.exception, None)
            self.assertEqual(
                json.loads((output / "result.json").read_text())["status"], "FAIL"
            )
            self.assertTrue((root / "stop").exists())

    def test_reset_during_wait_is_rejected(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            session = FrankaTaskSession(observation_dir=root, output=root / "task")
            session.run_id = "one"
            value = observation(root)
            reset = dict(value, episode_id=2, steps=11)
            session.client.state = Mock(side_effect=[value, value, reset])
            session.client.command = Mock(return_value={})
            with self.assertRaisesRegex(RuntimeError, "Episode reset"):
                session.wait(1)
            session.client.command.assert_called_once_with("stop")

    def test_terminal_export_reset_does_not_rearm_motion(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            session = FrankaTaskSession(observation_dir=root, output=root / "task")
            session.run_id = "one"
            value = observation(root)
            session.client.state = Mock(return_value=value)
            session.client.command = Mock(return_value={})
            session.observe()
            value.update(episode_id=2, steps=0, exported_successful_episodes=1)
            session.controller = bridge(session)
            session.controller.move_world = Mock()
            self.assertIs(session.move_tool([0.4, 0, 0.2], [0, 0, 0, 1]), value)
            with self.assertRaisesRegex(RuntimeError, "already completed"):
                session.command("pose", buttons={"trigger": 1})
            session.client.command.assert_not_called()

    def test_export_between_preflight_and_command_sends_no_input(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            session = FrankaTaskSession(observation_dir=root, output=root / "task")
            session.run_id = "one"
            value = observation(root)
            exported = dict(value, episode_id=2, exported_successful_episodes=1)
            session.client.state = Mock(side_effect=[value, exported])
            session.client.command = Mock()
            value_bridge = bridge(session)
            self.assertIsNone(value_bridge.target_base([0.4, 0, 0.2], np.eye(3)))
            session.client.command.assert_not_called()

    def test_render_timestamp_cannot_acknowledge_unchanged_native_step(self):
        value = bridge()
        value.client.state.side_effect = [feedback(7), feedback(7), feedback(8)]
        value.client.command.return_value = {"timestamp": 100.0, "id": "command"}
        with patch(BRIDGE + ".time.sleep"):
            value.target_base([0.4, 0, 0.2], np.eye(3))
        receipt = json.loads(value.ack_trace.getvalue())
        self.assertEqual(receipt["native_step"], 8)
        self.assertEqual(receipt["native_step_before_send"], 7)

    def test_newer_session_command_preflight_is_used_for_ack_baseline(self):
        value = bridge()
        value.client.state.side_effect = [feedback(7), feedback(8), feedback(9)]
        value.client.command.return_value = {
            "timestamp": 100.0,
            "id": "command",
            "native_step_before_send": 8,
        }
        with patch(BRIDGE + ".time.sleep"):
            value.target_base([0.4, 0, 0.2], np.eye(3))
        receipt = json.loads(value.ack_trace.getvalue())
        self.assertEqual(receipt["native_step"], 9)
        self.assertEqual(receipt["native_step_before_send"], 8)

    def test_stale_matching_action_times_out_without_receipt(self):
        value = bridge()
        value.client.state.return_value = feedback(7)
        value.client.command.return_value = {"timestamp": 100.0, "id": "command"}
        with (
            patch(BRIDGE + ".time.sleep"),
            patch(BRIDGE + ".time.monotonic", side_effect=[0, 1, 9]),
            self.assertRaises(TimeoutError),
        ):
            value.target_base([0.4, 0, 0.2], np.eye(3))
        self.assertEqual(value.ack_trace.getvalue(), "")

    def test_reset_observer_covers_explicit_and_internal_resets_once(self):
        class Environment:
            def __init__(self):
                self.epoch = 0
                self.seen = []
                self.result = object()

            def _reset_idx(self, ids, *, reason):
                self.seen.append((ids, reason, self.epoch))
                return self.result

            def reset(self):
                return self._reset_idx([0], reason="manual")

            def step(self):
                return self._reset_idx([0], reason="termination")

        env = Environment()
        install_reset_observer(env, lambda: setattr(env, "epoch", env.epoch + 1))
        self.assertIs(env.reset(), env.result)
        self.assertIs(env.step(), env.result)
        self.assertIs(env._reset_idx([], reason="empty"), env.result)
        self.assertEqual(
            env.seen, [([0], "manual", 1), ([0], "termination", 2), ([], "empty", 2)]
        )

    def test_native_default_and_partial_reset_slices_preserve_arguments(self):
        class Environment:
            num_envs = 3

            def __init__(self):
                self.seen = []

            def _reset_idx(self, ids):
                self.seen.append(ids)
                return ids

            def reset(self, env_ids=slice(None)):
                if env_ids is None:
                    env_ids = slice(None)
                return self._reset_idx(env_ids)

        env = Environment()
        callback = Mock()
        install_reset_observer(env, callback)
        for selection in (slice(None), None, slice(1, 2), slice(None, None, -1)):
            result = env.reset(selection)
            self.assertEqual(result, slice(None) if selection is None else selection)
        self.assertEqual(callback.call_count, 4)
        callback.reset_mock()
        for empty in (slice(0, 0), slice(3, 3), slice(2, 1)):
            self.assertIs(env.reset(empty), empty)
        callback.assert_not_called()
        # Preserve None as well if a native caller accepts it directly.
        self.assertIsNone(env._reset_idx(None))
        callback.assert_called_once_with()

    def test_internal_reset_blocks_next_command_without_explicit_reset(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            session = FrankaTaskSession(observation_dir=root, output=root / "task")
            session.run_id = "one"
            value = observation(root)
            session.client.state = Mock(return_value=value)
            session.client.command = Mock(return_value={})
            session.observe()

            class Environment:
                def reset(self):
                    raise AssertionError("Automatic reset must not call env.reset")

                def _reset_idx(self, ids):
                    return ids

                def step(self):
                    self._reset_idx([0])

            env = Environment()
            install_reset_observer(
                env, lambda: value.update(episode_id=value["episode_id"] + 1)
            )
            env.step()
            value["steps"] += 1
            with self.assertRaisesRegex(RuntimeError, "Episode reset"):
                session.command("pose", buttons={"trigger": 1})
            session.client.command.assert_called_once_with("stop")

    def test_reset_observer_preserves_native_exception(self):
        env = Mock()
        error = RuntimeError("native reset failed")
        env._reset_idx = Mock(side_effect=error)
        callback = Mock()
        install_reset_observer(env, callback)
        with self.assertRaises(RuntimeError) as caught:
            env._reset_idx([0])
        self.assertIs(caught.exception, error)
        callback.assert_called_once_with()

    def test_session_receipt_records_pre_send_physics_step(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            session = FrankaTaskSession(observation_dir=root, output=root / "task")
            session.run_id = "one"
            session.client.state = Mock(return_value=observation(root))
            session.client.command = Mock(return_value={"id": "command"})
            self.assertEqual(
                session.command("pose", buttons={"trigger": 0})[
                    "native_step_before_send"
                ],
                10,
            )


if __name__ == "__main__":
    unittest.main()
