# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Convert Franka hand-body targets to IWER inputs for the native SE3 retargeter."""

import json
from pathlib import Path
import time

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from .browser_command import command, state


def transform(position, quaternion):
    result = np.eye(4)
    result[:3, 3] = position
    result[:3, :3] = Rotation.from_quat(quaternion).as_matrix()
    return result


def wait_steps(count=3, timeout=20):
    first = state()["steps"]
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        current = state()
        if current["steps"] >= first + count or current.get(
            "exported_successful_episodes", 0
        ):
            return current
        time.sleep(0.05)
    raise TimeoutError("Native simulation did not advance")


def play():
    # Start through the existing client handler, not a layout-dependent controller ray.
    command("start")
    return wait_steps()


class XRControllerBridge:
    def __init__(self):
        initial = state()
        self.base_to_world = (
            np.eye(4)
            if initial["base_T_world"] is None
            else np.asarray(initial["base_T_world"])
        )
        self.base_to_anchor = self.base_to_world @ np.asarray(initial["world_T_anchor"])
        # The native Franka task config applies +90 degrees intrinsic X to grip orientation.
        self.offset = Rotation.from_euler("X", 90, degrees=True).as_matrix()
        command("pose", hand="right", buttons={"squeeze": 1, "trigger": 0})
        time.sleep(0.3)
        frame = command("inspect")["browser"]["frame"]
        right = next(s for s in frame["sources"] if s["hand"] == "right")
        self.aim_to_grip = np.linalg.inv(transform(**right["aim"])) @ transform(
            **right["grip"]
        )
        index = initial["body_names"].index("panda_hand")
        self.target = self.base_to_world @ transform(
            initial["body_positions_world"][0][index],
            initial["body_quaternions_world"][0][index],
        )
        self.trigger = 0.0
        self.ack_trace = (Path(initial["run_dir"]) / "controller-acks.jsonl").open(
            "a", buffering=1
        )
        # Place the virtual controller at the current robot target before starting native physics.
        self.target_base(self.target[:3, 3], self.target[:3, :3], wait_receipt=False)

    def target_base(self, position, rotation, trigger=None, wait_receipt=True):
        p, rot = np.asarray(position), np.asarray(rotation)
        if (
            p.shape != (3,)
            or rot.shape != (3, 3)
            or not np.isfinite(p).all()
            or not np.isfinite(rot).all()
        ):
            raise ValueError("Invalid target pose")
        if np.linalg.norm(p) > 1.5:
            raise ValueError("Target outside declared Franka test workspace")
        self.trigger = self.trigger if trigger is None else float(trigger)
        if not 0 <= self.trigger <= 1:
            raise ValueError("Invalid trigger")
        base_grip = np.eye(4)
        base_grip[:3, 3] = p
        base_grip[:3, :3] = rot @ self.offset.T
        aim = (
            np.linalg.inv(self.base_to_anchor)
            @ base_grip
            @ np.linalg.inv(self.aim_to_grip)
        )
        result = command(
            "pose",
            position=aim[:3, 3].tolist(),
            quaternion=Rotation.from_matrix(aim[:3, :3]).as_quat().tolist(),
            buttons={"squeeze": 1, "trigger": self.trigger},
        )
        self.target = transform(p, Rotation.from_matrix(rot).as_quat())
        if not wait_receipt:
            return result
        wanted_q = Rotation.from_matrix(rot).as_quat()
        gripper = -1.0 if self.trigger > 0.5 else 1.0
        deadline = time.monotonic() + 8
        while time.monotonic() < deadline:
            observed = state()
            if observed.get("exported_successful_episodes", 0):
                return result
            action = np.asarray(observed["last_action"])[0]
            if (
                observed["timestamp"] >= result["timestamp"]
                and np.linalg.norm(action[:3] - p) < 1e-4
                and abs(float(action[3:7] @ wanted_q)) > 0.99999
                and abs(action[7] - gripper) < 0.001
            ):
                self.ack_trace.write(
                    json.dumps(
                        {
                            "timestamp": time.time(),
                            "browser_command_id": result["id"],
                            "browser_ack_timestamp": result["timestamp"],
                            "native_step": observed["steps"],
                            "native_timestamp": observed["timestamp"],
                            "raw_trigger": self.trigger,
                            "expected_native_action": [
                                *p.tolist(),
                                *wanted_q.tolist(),
                                gripper,
                            ],
                            "received_native_action": action.tolist(),
                        }
                    )
                    + "\n"
                )
                return result
            time.sleep(0.025)
        raise TimeoutError(
            f"XR target did not reach native Franka action: expected={p.tolist()}, {wanted_q.tolist()}, {gripper}; actual={action.tolist()}"
        )

    def move_world(self, position, rotation=None, trigger=None, seconds=2):
        current_world = np.linalg.inv(self.base_to_world) @ self.target
        rotation = current_world[:3, :3] if rotation is None else rotation
        target = self.base_to_world @ transform(
            position, Rotation.from_matrix(rotation).as_quat()
        )
        origin = self.target.copy()
        steps = max(2, round(seconds * 5))
        orientations = Slerp(
            [0, 1], Rotation.from_matrix([origin[:3, :3], target[:3, :3]])
        )
        for fraction in np.linspace(0, 1, steps + 1)[1:]:
            if state().get("exported_successful_episodes", 0):
                return state()
            self.target_base(
                origin[:3, 3] * (1 - fraction) + target[:3, 3] * fraction,
                orientations(fraction).as_matrix(),
                trigger,
            )
            wait_steps(count=max(1, round(seconds / steps / state()["simulation_dt"])))
        return wait_steps()

    def release(self):
        try:
            command("pose", buttons={"squeeze": 0, "trigger": self.trigger})
        finally:
            self.ack_trace.close()
