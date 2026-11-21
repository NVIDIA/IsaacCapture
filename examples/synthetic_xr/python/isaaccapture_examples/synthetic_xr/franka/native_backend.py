# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Translate common tool targets into existing XTEST/evemu inputs only."""

import logging
import math
import time
from scipy.spatial.transform import Rotation


_LOG = logging.getLogger("isaaccapture.examples.synthetic_xr.franka.native_backend")


class NativeBackend:
    """Lab Se3Keyboard/Se3Gamepad mapping; the caller owns read/send and devices."""

    def __init__(self, kind, read, send, run_id, cancelled=lambda: False):
        if kind not in {"keyboard", "gamepad"}:
            raise ValueError("Unsupported native backend")
        self.name, self.read, self.send, self.run_id = kind, read, send, run_id
        self.cancelled = cancelled
        self.last_step = -1
        self.episode = None
        self.moves = []

    def observe(self, *, deadline=None):
        local_deadline = time.monotonic() + 3
        deadline = local_deadline if deadline is None else min(local_deadline, deadline)
        while time.monotonic() < deadline:
            if self.cancelled():
                raise RuntimeError("Cancelled")
            s = self.read()
            if s.get("run_id") != self.run_id:
                raise RuntimeError("Foreign observation run")
            if s.get("tool_frame") != {
                "sensor": "ee_frame",
                "index": 0,
                "body_name": "panda_hand",
                "offset_position": [0.0, 0.0, 0.1034],
            }:
                raise RuntimeError("Unqualified tool frame")
            if self.episode is None:
                self.episode = s["episode_id"]
            if s["episode_id"] != self.episode:
                raise RuntimeError("Reset detected")
            if (
                s["step"] < self.last_step
                or any(s.get("terminated", []))
                or any(s.get("truncated", []))
            ):
                raise RuntimeError("Reset detected")
            if time.time() - s["time"] > 3 or s["time"] > time.time() + 1:
                raise RuntimeError("Stale observation")
            if s["step"] > self.last_step:
                self.last_step = s["step"]
                # Pinned Lab 91a0906 uses Warp xyzw (BaseFrameTransformerData).
                if s.get("quaternion_order") != "xyzw":
                    raise RuntimeError("Unqualified quaternion convention")
                s = dict(
                    s,
                    tool_quaternion_xyzw=list(s["ee_quaternion"]),
                    object_quaternion_xyzw=list(s["object_quaternion"]),
                )
                for key in [
                    "ee",
                    "tool_quaternion_xyzw",
                    "object_quaternion_xyzw",
                    "object",
                    "joints",
                    "object_velocity",
                    "action",
                ]:
                    if not all(math.isfinite(v) for v in s[key]):
                        raise RuntimeError("Nonfinite observation: " + key)
                if abs(sum(v * v for v in s["tool_quaternion_xyzw"]) - 1) > 0.001:
                    raise RuntimeError("Invalid tool quaternion")
                return s
            time.sleep(0.01)
        raise TimeoutError("Observation step did not advance")

    def neutral(self):
        self.send(
            []
            if self.name == "keyboard"
            else {"lx": 0, "ly": 0, "rx": 0, "ry": 0, "hx": 0, "hy": 0, "x": 0}
        )

    def wait(self, seconds, *, deadline=None):
        wall = time.monotonic() + max(10, seconds * 10)
        if deadline is not None:
            wall = min(wall, deadline)
        start = self.observe(deadline=wall)["sim_time"]
        while True:
            if time.monotonic() >= wall:
                raise TimeoutError("Simulation wait exceeded wall backstop")
            if self.observe(deadline=wall)["sim_time"] - start >= seconds:
                return
            time.sleep(0.02)

    def move_tool(self, target, quaternion, *, timeout=60):
        if (
            len(target) != 3
            or len(quaternion) != 4
            or not all(math.isfinite(v) for v in [*target, *quaternion])
        ):
            raise ValueError("Invalid pose")
        if abs(sum(v * v for v in quaternion) - 1) > 1e-5:
            raise ValueError("Target quaternion must be unit xyzw")
        desired = Rotation.from_quat(quaternion)
        start = self.observe()
        wall = time.monotonic() + timeout
        # Reaching the pose while input is held is not settled convergence.
        # Correct bounded release drift without relaxing pose tolerances or
        # resetting the original wall/simulation budgets. All corrections still
        # pass through the native keyboard/gamepad input path.
        try:
            for attempt in range(3):
                while time.monotonic() < wall:
                    s = self.observe(deadline=wall)
                    error = [target[i] - s["ee"][i] for i in range(3)]
                    rotation = (
                        desired * Rotation.from_quat(s["tool_quaternion_xyzw"]).inv()
                    ).as_rotvec()
                    if s["sim_time"] - start["sim_time"] > timeout:
                        raise TimeoutError("Movement simulation timeout")
                    if max(map(abs, error)) < 0.004 and max(map(abs, rotation)) < 0.035:
                        break
                    axis = max(range(3), key=lambda i: abs(rotation[i]))
                    if self.name == "keyboard":
                        command = [
                            pair[int(error[i] < 0)]
                            for i, pair in enumerate(
                                [("w", "s"), ("a", "d"), ("q", "e")]
                            )
                            if abs(error[i]) > 0.0025
                        ]
                        if abs(rotation[axis]) > 0.035:
                            command.append(
                                [("z", "x"), ("t", "g"), ("c", "v")][axis][
                                    int(rotation[axis] < 0)
                                ]
                            )
                    else:
                        v = [
                            max(-0.3, min(0.3, e * 30)) if abs(e) > 0.002 else 0
                            for e in error
                        ]
                        command = {
                            "ly": -v[0],
                            "lx": v[1],
                            "ry": -v[2],
                            "rx": 0,
                            "hx": 0,
                            "hy": 0,
                            "x": 0,
                        }
                        if abs(rotation[axis]) > 0.035:
                            sign = 1 if rotation[axis] > 0 else -1
                            command.update(
                                [
                                    {"hx": -sign},
                                    {"hy": sign},
                                    {"rx": max(-0.5, min(0.5, rotation[2] * 5))},
                                ][axis]
                            )
                    self.send(command)
                    time.sleep(0.025 if self.name == "keyboard" else 0.08)
                else:
                    raise TimeoutError("Movement wall timeout")
                self.neutral()
                self.wait(0.35, deadline=wall)
                final = self.observe(deadline=wall)
                if time.monotonic() >= wall:
                    raise TimeoutError("Movement wall timeout")
                if final["sim_time"] - start["sim_time"] > timeout:
                    raise TimeoutError("Movement simulation timeout")
                rot_error = (
                    desired * Rotation.from_quat(final["tool_quaternion_xyzw"]).inv()
                ).magnitude()
                pos_error = max(abs(target[i] - final["ee"][i]) for i in range(3))
                if pos_error <= 0.009 and rot_error <= 0.07:
                    self.moves.append(
                        {
                            "target": list(target),
                            "position_error": pos_error,
                            "orientation_error": rot_error,
                            "step": final["step"],
                            "settle_attempts": attempt + 1,
                        }
                    )
                    return final
                _LOG.info(
                    "Target settling attempt %d/3: position error=%.6fm, orientation error=%.6frad",
                    attempt + 1,
                    pos_error,
                    rot_error,
                )
            raise RuntimeError(
                f"Target did not remain reached: position error={pos_error:.6f}m, orientation error={rot_error:.6f}rad"
            )
        finally:
            self.neutral()

    def set_gripper(self, *, closed):
        if type(closed) is not bool:
            raise TypeError("closed must be bool")
        s = self.observe()
        action_closed = s["action"][-1] < 0
        if closed != action_closed:
            try:
                self.send(["k"] if self.name == "keyboard" else {"x": 1})
                time.sleep(0.18)
            finally:
                self.neutral()
        self.wait(1)
        final = self.observe()
        if (final["action"][-1] < 0) != closed:
            raise RuntimeError("Native gripper command not confirmed; no blind retry")
        aperture = sum(final["joints"][-2:])
        # Contact with the cube prevents full closure; lifting is a separate task guard.
        if (not closed and min(final["joints"][-2:]) <= 0.035) or (
            closed and aperture >= 0.075
        ):
            raise RuntimeError("Gripper physical response not confirmed")
        return final
