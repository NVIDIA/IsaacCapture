# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression controls for release drift without weaker pose acceptance."""

import unittest
from unittest.mock import Mock, patch
from test_input_backends import state
from isaaccapture_examples.synthetic_xr.franka.native_backend import NativeBackend


def observed(x=0.0, q=(1.0, 0.0, 0.0, 0.0), sim=0.02):
    return {
        "ee": [x, 0.0, 0.0],
        "tool_quaternion_xyzw": list(q),
        "step": 10,
        "sim_time": sim,
    }


class Settling(unittest.TestCase):
    def backend(self, kind="keyboard"):
        sent = []
        return NativeBackend(kind, lambda: None, sent.append, "test"), sent

    def test_corrects_release_drift_using_input_then_accepts_settled_pose(self):
        for kind in ["keyboard", "gamepad"]:
            with self.subTest(kind=kind):
                b, sent = self.backend(kind)
                samples = [
                    observed(),
                    observed(),
                    observed(0.012),
                    observed(0.012),
                    observed(),
                    observed(0.002),
                ]
                with (
                    patch.object(b, "observe", side_effect=samples),
                    patch.object(b, "wait"),
                    patch(
                        "isaaccapture_examples.synthetic_xr.franka.native_backend.time.sleep"
                    ),
                ):
                    final = b.move_tool([0, 0, 0], [1, 0, 0, 0])
                self.assertEqual(final["ee"][0], 0.002)
                self.assertEqual(b.moves[0]["settle_attempts"], 2)
                if kind == "keyboard":
                    self.assertIn(["s"], sent)
                    self.assertEqual(sent[-1], [])
                else:
                    self.assertTrue(any(x.get("ly", 0) > 0 for x in sent))
                    self.assertEqual(
                        set(sent[-1]), {"lx", "ly", "rx", "ry", "hx", "hy", "x"}
                    )
                    self.assertTrue(all(v == 0 for v in sent[-1].values()))

    def test_three_failed_settles_remain_failure(self):
        b, sent = self.backend()
        samples = [
            observed(),
            observed(),
            observed(0.0091),
            observed(0.012),
            observed(),
            observed(0.0091),
            observed(0.012),
            observed(),
            observed(0.0091),
        ]
        with (
            patch.object(b, "observe", side_effect=samples),
            patch.object(b, "wait") as wait,
        ):
            with self.assertRaisesRegex(RuntimeError, "did not remain reached"):
                b.move_tool([0, 0, 0], [1, 0, 0, 0])
        self.assertEqual(wait.call_count, 3)
        self.assertEqual(sum(bool(x) for x in sent), 2)
        self.assertEqual(b.moves, [])
        self.assertEqual(sent[-1], [])

    def test_wrong_settled_orientation_remains_failure(self):
        b, sent = self.backend()
        samples = [
            observed(),
            observed(),
            observed(q=(0, 0, 0, 1)),
            observed(q=(0, 0, 0, 1)),
            observed(),
            observed(q=(0, 0, 0, 1)),
            observed(q=(0, 0, 0, 1)),
            observed(),
            observed(q=(0, 0, 0, 1)),
        ]
        with (
            patch.object(b, "observe", side_effect=samples),
            patch.object(b, "wait") as wait,
            patch(
                "isaaccapture_examples.synthetic_xr.franka.native_backend.time.sleep"
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "did not remain reached"):
                b.move_tool([0, 0, 0], [1, 0, 0, 0])
        self.assertEqual(wait.call_count, 3)
        self.assertEqual(sum(bool(x) for x in sent), 2)
        self.assertEqual(b.moves, [])
        self.assertEqual(sent[-1], [])

    def test_correction_keeps_original_wall_deadline(self):
        b, sent = self.backend()
        with (
            patch.object(
                b, "observe", side_effect=[observed(), observed(), observed(0.02)]
            ),
            patch.object(b, "wait"),
            patch(
                "isaaccapture_examples.synthetic_xr.franka.native_backend.time.monotonic",
                side_effect=[0, 0.1, 0.2, 1.1],
            ),
        ):
            with self.assertRaisesRegex(TimeoutError, "wall timeout"):
                b.move_tool([0, 0, 0], [1, 0, 0, 0], timeout=1)
        self.assertEqual(sent[-1], [])

    def test_reset_during_correction_releases_input(self):
        b, sent = self.backend()
        with (
            patch.object(
                b,
                "observe",
                side_effect=[
                    observed(),
                    observed(),
                    observed(0.02),
                    RuntimeError("Reset detected"),
                ],
            ),
            patch.object(b, "wait"),
        ):
            with self.assertRaisesRegex(RuntimeError, "Reset detected"):
                b.move_tool([0, 0, 0], [1, 0, 0, 0])
        self.assertEqual(b.moves, [])
        self.assertEqual(sent[-1], [])

    def test_correction_keeps_original_simulation_deadline(self):
        b, sent = self.backend()
        with (
            patch.object(
                b,
                "observe",
                side_effect=[observed(), observed(), observed(0.02), observed(sim=2)],
            ),
            patch.object(b, "wait"),
        ):
            with self.assertRaisesRegex(TimeoutError, "simulation timeout"):
                b.move_tool([0, 0, 0], [1, 0, 0, 0], timeout=1)
        self.assertEqual(sent[-1], [])

    def test_settle_wait_uses_move_deadline(self):
        b, _ = self.backend()
        with (
            patch.object(
                b, "observe", side_effect=[observed(), observed(), observed()]
            ),
            patch.object(b, "wait") as wait,
            patch(
                "isaaccapture_examples.synthetic_xr.franka.native_backend.time.monotonic",
                return_value=0,
            ),
        ):
            b.move_tool([0, 0, 0], [1, 0, 0, 0], timeout=1)
        wait.assert_called_once_with(0.35, deadline=1)

    def test_wait_itself_stops_at_parent_deadline(self):
        b, _ = self.backend()
        with (
            patch.object(b, "observe", return_value=observed()) as observe,
            patch(
                "isaaccapture_examples.synthetic_xr.franka.native_backend.time.monotonic",
                side_effect=[0, 0.1, 0.3],
            ),
            patch(
                "isaaccapture_examples.synthetic_xr.franka.native_backend.time.sleep"
            ),
        ):
            with self.assertRaisesRegex(TimeoutError, "wall backstop"):
                b.wait(0.35, deadline=0.25)
        self.assertTrue(
            all(c.kwargs["deadline"] == 0.25 for c in observe.call_args_list)
        )

    def test_frozen_observation_uses_parent_deadline(self):
        b, _ = self.backend()
        snapshot = state()
        b.last_step = snapshot["step"]
        b.read = Mock(return_value=snapshot)
        with (
            patch(
                "isaaccapture_examples.synthetic_xr.franka.native_backend.time.monotonic",
                side_effect=[0, 0.1, 0.2, 0.3],
            ),
            patch(
                "isaaccapture_examples.synthetic_xr.franka.native_backend.time.sleep"
            ),
        ):
            with self.assertRaisesRegex(TimeoutError, "did not advance"):
                b.observe(deadline=0.25)
        self.assertEqual(b.read.call_count, 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
