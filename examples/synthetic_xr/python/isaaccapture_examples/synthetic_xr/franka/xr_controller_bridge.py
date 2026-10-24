# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Convert Franka hand-body targets to IWER inputs for the native SE3 retargeter."""

import json
from pathlib import Path
import time

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from . import browser_command


class EpisodeCompleted(RuntimeError):
    """Native export ended a command while the task was in progress."""


def transform(position, quaternion):
    result = np.eye(4)
    result[:3, 3] = position
    result[:3, :3] = Rotation.from_quat(quaternion).as_matrix()
    return result


def validate_pose(position, rotation):
    p, rot = np.asarray(position, dtype=float), np.asarray(rotation, dtype=float)
    if (
        p.shape != (3,)
        or rot.shape != (3, 3)
        or not np.isfinite(p).all()
        or not np.isfinite(rot).all()
    ):
        raise ValueError("Invalid target pose")
    if not np.allclose(rot.T @ rot, np.eye(3), atol=1e-6) or not np.isclose(
        np.linalg.det(rot), 1.0, atol=1e-6
    ):
        raise ValueError("Target rotation must be orthonormal with determinant +1")
    return p, rot


def validate_target(position, rotation, trigger):
    p, rot = validate_pose(position, rotation)
    if np.linalg.norm(p) > 1.5:
        raise ValueError("Target outside declared Franka test workspace")
    trigger = float(trigger)
    if not 0 <= trigger <= 1:
        raise ValueError("Invalid trigger")
    return p, rot, trigger


def wait_steps(count=3, timeout=20, client=browser_command):
    first = client.state()["steps"]
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        current = client.state()
        if current["steps"] >= first + count or current.get(
            "exported_successful_episodes", 0
        ):
            return current
        time.sleep(0.05)
    raise TimeoutError("Native simulation did not advance")


def play(client=browser_command):
    # Start through the existing client handler, not a layout-dependent controller ray.
    client.command("start")
    return wait_steps(client=client)


class XRControllerBridge:
    def __init__(self, client=browser_command, acknowledgement_path=None):
        self.client = client
        initial = self.client.state()
        self.base_to_world = (
            np.eye(4)
            if initial["base_T_world"] is None
            else np.asarray(initial["base_T_world"])
        )
        self.base_to_anchor = self.base_to_world @ np.asarray(initial["world_T_anchor"])
        # The native Franka task config applies +90 degrees intrinsic X to grip orientation.
        self.offset = Rotation.from_euler("X", 90, degrees=True).as_matrix()
        self.client.command("pose", hand="right", buttons={"squeeze": 1, "trigger": 0})
        time.sleep(0.3)
        frame = self.client.command("inspect")["browser"]["frame"]
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
        self.ack_trace = (
            Path(acknowledgement_path)
            if acknowledgement_path is not None
            else Path(initial["run_dir"]) / "controller-acks.jsonl"
        ).open("a", buffering=1)
        # Place the virtual controller at the current robot target before starting native physics.
        self.target_base(self.target[:3, 3], self.target[:3, :3], wait_receipt=False)

    def target_base(self, position, rotation, trigger=None, wait_receipt=True):
        p, rot, next_trigger = validate_target(
            position, rotation, self.trigger if trigger is None else trigger
        )
        base_grip = np.eye(4)
        base_grip[:3, 3] = p
        base_grip[:3, :3] = rot @ self.offset.T
        aim = (
            np.linalg.inv(self.base_to_anchor)
            @ base_grip
            @ np.linalg.inv(self.aim_to_grip)
        )
        before = self.client.state()
        if before.get("exported_successful_episodes", 0):
            return None
        step_at_send = before["steps"]
        try:
            result = self.client.command(
                "pose",
                position=aim[:3, 3].tolist(),
                quaternion=Rotation.from_matrix(aim[:3, :3]).as_quat().tolist(),
                buttons={"squeeze": 1, "trigger": next_trigger},
            )
        except EpisodeCompleted:
            return None
        step_at_send = max(
            step_at_send, result.get("native_step_before_send", step_at_send)
        )
        self.trigger = next_trigger
        self.target = transform(p, Rotation.from_matrix(rot).as_quat())
        if not wait_receipt:
            return result
        wanted_q = Rotation.from_matrix(rot).as_quat()
        gripper = -1.0 if self.trigger > 0.5 else 1.0
        deadline = time.monotonic() + 8
        while time.monotonic() < deadline:
            observed = self.client.state()
            if observed.get("exported_successful_episodes", 0):
                return result
            action = np.asarray(observed["last_action"])[0]
            if (
                observed["steps"] > step_at_send
                and observed["timestamp"] >= result["timestamp"]
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
                            "native_step_before_send": step_at_send,
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
        if not np.isfinite(seconds) or not 0 < seconds <= 120:
            raise ValueError("Movement duration must be in (0, 120] seconds")
        current_world = np.linalg.inv(self.base_to_world) @ self.target
        rotation = current_world[:3, :3] if rotation is None else rotation
        position, rotation = validate_pose(position, rotation)
        target = self.base_to_world @ transform(
            position, Rotation.from_matrix(rotation).as_quat()
        )
        origin = self.target.copy()
        # The base-frame workspace is convex: admit both endpoints before any input.
        validate_target(
            target[:3, 3], target[:3, :3], self.trigger if trigger is None else trigger
        )
        validate_target(origin[:3, 3], origin[:3, :3], self.trigger)
        steps = max(2, round(seconds * 5))
        orientations = Slerp(
            [0, 1], Rotation.from_matrix([origin[:3, :3], target[:3, :3]])
        )
        for fraction in np.linspace(0, 1, steps + 1)[1:]:
            if self.client.state().get("exported_successful_episodes", 0):
                return self.client.state()
            self.target_base(
                origin[:3, 3] * (1 - fraction) + target[:3, 3] * fraction,
                orientations(fraction).as_matrix(),
                trigger,
            )
            wait_steps(
                count=max(
                    1, round(seconds / steps / self.client.state()["simulation_dt"])
                ),
                client=self.client,
            )
        return wait_steps(client=self.client)

    def release(self):
        try:
            self.client.command("pose", buttons={"squeeze": 0, "trigger": self.trigger})
        finally:
            self.ack_trace.close()
