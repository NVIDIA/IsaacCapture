# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Task guards and adapter contracts without native runtime or injected joints."""

import ast
import copy
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from isaaccapture_examples.synthetic_xr.franka import pick_place_success
from isaaccapture_examples.synthetic_xr.franka.task_plan import execute
from isaaccapture_examples.synthetic_xr.franka.xr_backend import XRBackend


class SharedTaskTests(unittest.TestCase):
    def test_lift_and_carry_failures_release_input(self):
        initial = {
            "object": [0.5, 0.1, 0.02],
            "object_quaternion_xyzw": [0, 0, 0, 1],
            "ee": [0.5, 0.1, 0.3],
        }
        lifted = dict(initial, object=[0.5, 0.1, 0.19], ee=[0.5, 0.1, 0.19])
        for snapshots, error in [
            ([initial] * 5, "did not lift"),
            ([initial] * 4 + [lifted, initial], "dropped during carry"),
        ]:
            with self.subTest(error=error):
                backend = Mock(name="backend")
                backend.observe.side_effect = snapshots
                with self.assertRaisesRegex(RuntimeError, error):
                    execute(backend, lambda _: None)
                backend.neutral.assert_called_once()
                backend.wait.assert_not_called()
                self.assertEqual(backend.set_gripper.call_args.kwargs, {"closed": True})

    def test_success_callback_requires_lift_release_and_retreat(self):
        initial = dict(
            episode_id=7,
            cubes={
                "cube_2": {
                    "position_world": [[0.6, 0, 0.02]],
                    "linear_velocity_world": [[0, 0, 0]],
                }
            },
            ee_positions_world=[[[0.6, 0, 0.02]]],
            joint_names=["panda_finger_joint1", "panda_finger_joint2"],
            joints=[[0.04, 0.04]],
        )
        pick_place_success._history.clear()
        self.assertFalse(pick_place_success.is_success(initial))
        goal = pick_place_success.destination(initial)
        placed = copy.deepcopy(initial)
        placed["cubes"]["cube_2"]["position_world"] = [goal.tolist()]
        placed["ee_positions_world"] = [[[goal[0], goal[1], 0.22]]]
        self.assertFalse(
            pick_place_success.is_success(placed), "placement without lift cannot pass"
        )
        lift = copy.deepcopy(initial)
        lift["cubes"]["cube_2"]["position_world"][0][2] = 0.20
        self.assertFalse(pick_place_success.is_success(lift))
        closed = copy.deepcopy(placed)
        closed["joints"] = [[0.01, 0.01]]
        self.assertFalse(pick_place_success.is_success(closed))
        self.assertTrue(pick_place_success.is_success(placed))
        reset = copy.deepcopy(initial)
        reset["episode_id"] += 1
        self.assertFalse(
            pick_place_success.is_success(reset),
            "reset cannot retain the lift history",
        )

    def test_xr_neutral_releases_clutch_without_dropping_held_cube(self):
        session = Mock()
        adapter = XRBackend(session)
        adapter.trigger = 1.0
        adapter.neutral()
        session.command.assert_called_once_with(
            "pose", hand="right", buttons={"squeeze": 0, "trigger": 1.0}
        )

    def test_xr_adapter_rejects_reset_and_early_export(self):
        observation = dict(
            run_id="run",
            episode_id=0,
            steps=1,
            simulation_dt=0.02,
            cubes={
                "cube_2": {
                    "position_world": [[0.5, 0, 0.02]],
                    "quaternion_world": [[0, 0, 0, 1]],
                    "linear_velocity_world": [[0, 0, 0]],
                }
            },
            ee_positions_world=[[[0.5, 0, 0.2]]],
            ee_quaternions_world=[[[1, 0, 0, 0]]],
            joints=[[0] * 7 + [0.04, 0.04]],
        )
        adapter = XRBackend(SimpleNamespace(observe=lambda: observation))
        adapter.observe()
        observation.update(episode_id=1, steps=2)
        with self.assertRaisesRegex(RuntimeError, "episode changed"):
            adapter.observe()
        observation["exported_successful_episodes"] = 1
        with self.assertRaisesRegex(RuntimeError, "before the shared task finished"):
            adapter.observe()


class XRExportRaceTests(unittest.TestCase):
    class Session:
        # Pinned record_demos with --num_demos 1 exits before handle_reset.
        def __init__(self, settle_steps=150):
            self.steps = 1
            self.settle_steps = settle_steps
            self.exported = False
            self.closed = False
            self.ee = [0.5, 0.1, 0.3]
            self.cube = [0.5, 0.1, 0.02]
            self.commands = []

        def observe(self):
            return dict(
                joint_names=[f"panda_joint{i}" for i in range(7)]
                + ["panda_finger_joint1", "panda_finger_joint2"],
                run_id="run",
                episode_id=0,
                steps=self.steps,
                simulation_dt=0.02,
                exported_successful_episodes=int(self.exported),
                phase="completed" if self.exported else "running",
                cubes={
                    "cube_2": {
                        "position_world": [self.cube[:]],
                        "quaternion_world": [[0, 0, 0, 1]],
                        "linear_velocity_world": [[0, 0, 0]],
                    }
                },
                ee_positions_world=[[self.ee[:]]],
                ee_quaternions_world=[[[1, 0, 0, 0]]],
                joints=[[0] * 7 + ([0.02, 0.02] if self.closed else [0.04, 0.04])],
            )

        def move_tool(self, position, quaternion, *, seconds):
            self.steps += 150
            self.ee = list(position)
            if self.closed:
                self.cube = [position[0], position[1], position[2] - 0.003]

        def set_gripper(self, *, closed):
            self.closed = closed
            if not closed:
                self.cube[2] = 0.02

        def wait(self, seconds):
            if seconds == 3:
                self.steps += self.settle_steps
                self.exported = True
            else:
                self.steps += 50

        def command(self, action, **kwargs):
            from isaaccapture_examples.synthetic_xr.franka.xr_controller_bridge import (
                EpisodeCompleted,
            )

            if action == "pose" and self.exported:
                raise EpisodeCompleted("Native export raced cleanup")
            self.commands.append((action, kwargs))

    def test_complete_task_accepts_export_after_full_neutral_wait(self):
        session = self.Session()
        result = execute(XRBackend(session), lambda _: None)
        self.assertTrue(result["task_completed"])
        self.assertEqual(session.commands[-1], ("stop", {}))
        self.assertTrue(session.exported)

    def test_export_one_step_before_settling_finishes_is_failure(self):
        session = self.Session(settle_steps=149)
        with self.assertRaisesRegex(RuntimeError, "before the shared task finished"):
            execute(XRBackend(session), lambda _: None)
        self.assertEqual(session.commands[-1], ("stop", {}))

    def test_cleanup_failure_is_not_hidden_after_successful_export(self):
        session = self.Session()
        original = session.command

        def command(action, **kwargs):
            if action == "stop":
                raise OSError("stop failed")
            return original(action, **kwargs)

        session.command = command
        with self.assertRaisesRegex(OSError, "stop failed"):
            execute(XRBackend(session), lambda _: None)

    def test_destination_choice_cannot_flip_with_scene_settling(self):
        observation = self.Session().observe()
        goal = pick_place_success.destination(observation)
        observation["cubes"]["cube_2"]["position_world"][0][1] += 0.001
        moved = pick_place_success.destination(observation)
        self.assertEqual(list(goal[:2]), list(moved[:2]))
        observation["cubes"]["cube_1"] = {"position_world": [goal.tolist()]}
        with self.assertRaisesRegex(RuntimeError, "occupied"):
            pick_place_success.destination(observation)

    def test_callback_returns_false_for_blocked_new_episode(self):
        observation = self.Session().observe()
        pick_place_success._history.clear()
        self.assertFalse(pick_place_success.is_success(observation))
        observation["episode_id"] += 1
        observation["cubes"]["cube_1"] = {"position_world": [[0.45, -0.20, 0.02]]}
        self.assertFalse(pick_place_success.is_success(observation))
        observation["cubes"]["cube_2"]["position_world"][0][2] += 0.2
        self.assertFalse(pick_place_success.is_success(observation))

    def test_native_wait_returns_export_before_requested_step_count(self):
        from isaaccapture_examples.synthetic_xr.franka.xr_controller_bridge import (
            wait_steps,
        )

        terminal = {"steps": 149, "exported_successful_episodes": 1}
        client = SimpleNamespace(state=Mock(side_effect=[{"steps": 1}, terminal]))
        self.assertIs(wait_steps(count=150, client=client), terminal)


class XRSuccessWindowTests(unittest.TestCase):
    class TimedSession(XRExportRaceTests.Session):
        def __init__(self, success_steps):
            super().__init__()
            self.success_steps = success_steps
            self.success_onset = None
            self.released = False
            self.latency_applied = False

        def set_gripper(self, *, closed):
            self.released = self.released or (self.closed and not closed)
            super().set_gripper(closed=closed)

        def move_tool(self, position, quaternion, *, seconds):
            super().move_tool(position, quaternion, seconds=seconds)
            if self.released:
                # Conservative onset before the end of retreat, plus later transport latency.
                self.success_onset = self.steps - 65

        def command(self, action, **kwargs):
            if self.success_onset is not None and not self.latency_applied:
                self.steps += 45
                self.latency_applied = True
            return super().command(action, **kwargs)

        def wait(self, seconds):
            if self.success_onset is None:
                return super().wait(seconds)
            export_at = self.success_onset + self.success_steps
            self.steps = min(self.steps + round(seconds / 0.02), export_at)
            self.exported = self.steps == export_at

    def test_300_step_window_preserves_full_wait_despite_early_onset_and_latency(self):
        session = self.TimedSession(300)
        self.assertTrue(execute(XRBackend(session), lambda _: None)["task_completed"])
        self.assertFalse(session.exported)
        session.wait(1)
        self.assertTrue(session.exported)

    def test_200_step_window_can_export_too_early_and_must_fail(self):
        session = self.TimedSession(200)
        with self.assertRaisesRegex(RuntimeError, "before the shared task finished"):
            execute(XRBackend(session), lambda _: None)

    def test_success_requires_near_final_vertical_retreat(self):
        observation = XRExportRaceTests.Session().observe()
        pick_place_success._history.clear()
        self.assertFalse(pick_place_success.is_success(observation))
        observation["cubes"]["cube_2"]["position_world"][0][2] = 0.19
        self.assertFalse(pick_place_success.is_success(observation))
        goal = [0.45, -0.20, 0.02]
        observation["cubes"]["cube_2"]["position_world"] = [goal]
        observation["ee_positions_world"] = [[[0.45, -0.20, 0.13]]]
        self.assertFalse(pick_place_success.is_success(observation))
        observation["ee_positions_world"] = [[[0.45, -0.20, 0.20]]]
        self.assertTrue(pick_place_success.is_success(observation))


class SuccessSnapshotTests(unittest.TestCase):
    def test_callback_refreshes_scene_after_episode_boundary(self):
        from isaaccapture_examples.synthetic_xr.franka import task_hooks

        source = Path(task_hooks.__file__).with_name("observe_record_demos.py")
        tree = ast.parse(source.read_text())
        callback = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "custom_success"
        )
        # Reproduce the live failure: reset ID advanced, old scene still cached
        # because render publication was throttled. The old goal is blocked.
        old = XRExportRaceTests.Session().observe()
        old["cubes"]["cube_2"]["position_world"] = [[0.45, -0.04, 0.02]]
        fresh = XRExportRaceTests.Session().observe()
        stale = copy.deepcopy(old)
        pick_place_success._history.clear()
        self.assertFalse(pick_place_success.is_success(stale))
        self.assertIsNone(pick_place_success._history["goal"])
        pick_place_success._history.clear()

        def publish(*, force):
            self.assertTrue(force)
            stale.update(copy.deepcopy(fresh))

        namespace = {
            "state": stale,
            "publish": publish,
            "evaluate_success": task_hooks.evaluate_success,
            "success_callback": pick_place_success.is_success,
        }
        compiled = compile(
            ast.Module(body=[callback], type_ignores=[]), str(source), "exec"
        )
        exec(compiled, namespace)
        torch = SimpleNamespace(tensor=lambda value, **_: value, bool=bool)
        with patch.dict("sys.modules", {"torch": torch}):
            self.assertEqual(
                namespace["custom_success"](SimpleNamespace(device="cpu")), [False]
            )
        self.assertIsNotNone(pick_place_success._history["goal"])
        self.assertEqual(
            pick_place_success._history["start"].tolist(),
            fresh["cubes"]["cube_2"]["position_world"][0],
        )


if __name__ == "__main__":
    unittest.main()
