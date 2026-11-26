# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Negative controls for the input adapters and independent outcome oracle."""

import time
import unittest
from unittest.mock import patch
from isaaccapture_examples.synthetic_xr.franka.native_backend import NativeBackend
from isaaccapture_examples.synthetic_xr.franka.task_plan import execute


def state():
    return dict(
        run_id="test",
        step=1,
        time=time.time(),
        sim_time=0.02,
        quaternion_order="xyzw",
        episode_id=0,
        tool_frame={
            "sensor": "ee_frame",
            "index": 0,
            "body_name": "panda_hand",
            "offset_position": [0.0, 0.0, 0.1034],
        },
        object_quaternion=[0, 0, 0, 1],
        ee=[0.4, 0, 0.3],
        ee_quaternion=[1, 0, 0, 0],
        object=[0.4, 0, 0.025],
        joints=[0] * 7 + [0.04, 0.04],
        object_velocity=[0, 0, 0],
        action=[0] * 6 + [1],
        terminated=[False],
        truncated=[False],
    )


class Contracts(unittest.TestCase):
    def backend(self, s, **kwargs):
        sent = []
        b = NativeBackend(
            "keyboard", lambda: s, lambda x: sent.append(x), "test", **kwargs
        )
        return b, sent

    def test_reject_foreign_run(self):
        s = state()
        s["run_id"] = "other"
        b, _ = self.backend(s)
        with self.assertRaisesRegex(RuntimeError, "Foreign"):
            b.observe()

    def test_reject_stale(self):
        s = state()
        s["time"] -= 4
        b, _ = self.backend(s)
        with self.assertRaisesRegex(RuntimeError, "Stale"):
            b.observe()

    def test_reject_reset(self):
        for flag in ["terminated", "truncated"]:
            s = state()
            s[flag] = [True]
            b, _ = self.backend(s)
            with self.assertRaisesRegex(RuntimeError, "Reset"):
                b.observe()

    def test_reject_step_regression(self):
        s = state()
        b, _ = self.backend(s)
        b.observe()
        s["step"] = 0
        with self.assertRaisesRegex(RuntimeError, "Reset"):
            b.observe()

    def test_reject_quaternion_order(self):
        s = state()
        s["quaternion_order"] = "wxyz"
        b, _ = self.backend(s)
        with self.assertRaisesRegex(RuntimeError, "convention"):
            b.observe()

    def test_xyzw_is_not_reordered(self):
        b, _ = self.backend(state())
        self.assertEqual(b.observe()["tool_quaternion_xyzw"], [1, 0, 0, 0])

    def test_cancel(self):
        b, _ = self.backend(state(), cancelled=lambda: True)
        with self.assertRaisesRegex(RuntimeError, "Cancelled"):
            b.observe()

    def test_expired_observation_deadline_times_out(self):
        b, _ = self.backend(state())
        b.observe()
        with patch(
            "isaaccapture_examples.synthetic_xr.franka.native_backend.time.monotonic",
            side_effect=[0, 4],
        ):
            with self.assertRaisesRegex(TimeoutError, "advance"):
                b.observe()

    def test_invalid_target_emits_nothing(self):
        b, sent = self.backend(state())
        with self.assertRaises(ValueError):
            b.move_tool([float("nan"), 0, 0], [1, 0, 0, 0])
        self.assertEqual(sent, [])

    def test_gripper_open_is_idempotent(self):
        s = state()
        b, sent = self.backend(s)
        with patch.object(b, "observe", return_value=s), patch.object(b, "wait"):
            b.set_gripper(closed=False)
            b.set_gripper(closed=False)
        self.assertEqual(sent, [])

    def test_missing_gripper_ack_no_retry_and_release(self):
        s = state()
        b, sent = self.backend(s)
        with (
            patch.object(b, "observe", return_value=s),
            patch.object(b, "wait"),
            patch(
                "isaaccapture_examples.synthetic_xr.franka.native_backend.time.sleep"
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "no blind retry"):
                b.set_gripper(closed=True)
        self.assertEqual(sent, [["k"], []])

    def test_task_failure_stops_input(self):
        b, sent = self.backend(state())
        with patch.object(b, "observe", side_effect=RuntimeError("stale")):
            with self.assertRaises(RuntimeError):
                execute(b, lambda p: None)
        self.assertEqual(sent, [[]])

    def test_unreachable_target_times_out_and_neutralizes(self):
        b, sent = self.backend(state())
        with (
            patch.object(b, "observe", return_value=state()),
            patch(
                "isaaccapture_examples.synthetic_xr.franka.native_backend.time.monotonic",
                side_effect=[0, 2],
            ),
        ):
            with self.assertRaisesRegex(TimeoutError, "wall timeout"):
                b.move_tool([0.6, 0.2, 0.4], [1, 0, 0, 0], timeout=1)
        self.assertEqual(sent, [[]])

    def test_cleanup_preserves_primary_failure(self):
        b, _ = self.backend(state())
        with (
            patch.object(b, "observe", side_effect=RuntimeError("primary")),
            patch.object(b, "neutral", side_effect=RuntimeError("cleanup")),
        ):
            with self.assertRaisesRegex(RuntimeError, "primary") as caught:
                execute(b, lambda p: None)
        self.assertIn("cleanup", caught.exception.cleanup_error)

    def test_wrong_tool_point_rejected(self):
        s = state()
        s["tool_frame"]["offset_position"] = [0, 0, 0]
        b, _ = self.backend(s)
        with self.assertRaisesRegex(RuntimeError, "tool frame"):
            b.observe()

    def test_skipped_reset_detected(self):
        s = state()
        b, _ = self.backend(s)
        b.observe()
        s.update(step=10, episode_id=1)
        with self.assertRaisesRegex(RuntimeError, "Reset"):
            b.observe()


if __name__ == "__main__":
    unittest.main(verbosity=2)
