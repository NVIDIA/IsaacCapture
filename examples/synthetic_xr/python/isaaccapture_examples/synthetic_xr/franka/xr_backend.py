# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Same task contract, through the existing public Synthetic XR session API."""

import math
from .xr_controller_bridge import EpisodeCompleted


class XRBackend:
    name = "xr"

    def __init__(self, session, object_name="cube_2"):
        self.session = session
        self.object_name = object_name
        self.episode = None
        self.last_step = -1
        self.trigger = 0.0
        self.terminal_step = None

    def observe(self):
        s = self.session.observe()
        terminal = s.get("phase") == "completed" or s.get(
            "exported_successful_episodes", 0
        )
        if terminal and (
            not s.get("exported_successful_episodes", 0)
            or self.terminal_step is None
            or s["steps"] < self.terminal_step
        ):
            raise RuntimeError("XR exported before the shared task finished")
        if self.episode is None:
            self.episode = s["episode_id"]
        if s["episode_id"] != self.episode or s["steps"] < self.last_step:
            raise RuntimeError("XR episode changed during task")
        self.last_step = s["steps"]
        obj = s["cubes"][self.object_name]
        return dict(
            run_id=s["run_id"],
            step=s["steps"],
            sim_time=s["steps"] * s["simulation_dt"],
            ee=s["ee_positions_world"][0][0],
            tool_quaternion_xyzw=s["ee_quaternions_world"][0][0],
            object=obj["position_world"][0],
            object_velocity=obj["linear_velocity_world"][0],
            object_quaternion_xyzw=obj["quaternion_world"][0],
            joints=s["joints"][0],
        )

    def move_tool(self, target, quaternion):
        self.terminal_step = None
        # The session owns measured position/orientation convergence and its limits.
        self.session.move_tool(target, quaternion, seconds=3)
        return self.observe()

    def set_gripper(self, *, closed):
        self.terminal_step = None
        self.session.set_gripper(closed=closed)
        self.trigger = float(closed)
        self.session.wait(1)
        s = self.observe()
        if (not closed and min(s["joints"][-2:]) <= 0.035) or (
            closed and sum(s["joints"][-2:]) >= 0.075
        ):
            raise RuntimeError("XR gripper physical response missing")
        return s

    def wait(self, seconds):
        if not math.isfinite(seconds) or not 0 < seconds <= 120:
            raise ValueError("Wait duration must be in (0, 120] simulation seconds")
        self.terminal_step = None
        current = self.session.observe()
        self.observe()  # Reject export/reset before beginning the neutral wait.
        self.terminal_step = current["steps"] + math.ceil(
            seconds / current["simulation_dt"]
        )
        self.session.wait(seconds)
        self.observe()  # Native export may end wait_steps early; still require the duration.

    def neutral(self):
        # Clutch release retains the trigger state; do not drop a carried object.
        try:
            self.session.command(
                "pose", hand="right", buttons={"squeeze": 0, "trigger": self.trigger}
            )
        except EpisodeCompleted:
            # Export can race with cleanup; stop is allowed after terminal export.
            self.session.command("stop")
