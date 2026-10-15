# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Scripted Franka stacking through browser XR input, with read-only scene feedback."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .browser_command import command, configure, state
from .xr_controller_bridge import XRControllerBridge, play, wait_steps

parser = argparse.ArgumentParser()
parser.add_argument(
    "--stop-after", choices=["approach", "grasp", "stack"], default="stack"
)
parser.add_argument("--cdp-url", default="http://127.0.0.1:9222")
parser.add_argument("--feedback-url", default="http://127.0.0.1:8765/state")
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
configure(args.cdp_url, args.feedback_url, args.output)
run = args.output / ("stack-logic-" + str(time.time_ns()))
run.mkdir(parents=True)
trace = (run / "stages.jsonl").open("x", buffering=1)
TOOL_OFFSET = np.array(
    [0.0, 0.0, 0.1034]
)  # Native Franka ee_frame offset from panda_hand.


def mark(stage):
    s = state()
    trace.write(json.dumps({"stage": stage, "state": s}) + "\n")
    print(
        stage,
        json.dumps(
            {
                "steps": s["steps"],
                "ee": s["ee_positions_world"][0][0],
                "joints": s["joints"][0],
                "cubes": {k: v["position_world"][0] for k, v in s["cubes"].items()},
            }
        ),
        flush=True,
    )
    return s


def cube(name):
    return np.asarray(state()["cubes"][name]["position_world"])[0]


def move_tool(controller, point, rotation, seconds=2):
    point = np.asarray(point)
    controller.move_world(point - rotation @ TOOL_OFFSET, rotation, seconds=seconds)
    for _ in range(4):
        if state().get("exported_successful_episodes", 0):
            return
        error = point - np.asarray(state()["ee_positions_world"])[0, 0]
        if np.linalg.norm(error) < 0.002:
            return
        target = np.linalg.inv(controller.base_to_world) @ controller.target
        controller.move_world(
            target[:3, 3] + np.clip(error, -0.01, 0.01), rotation, seconds=0.6
        )
    if np.linalg.norm(point - np.asarray(state()["ee_positions_world"])[0, 0]) > 0.01:
        raise RuntimeError("Gripper did not converge to the requested approach point")


def transfer(controller, name, support):
    start = cube(name)
    yaw = Rotation.from_quat(state()["cubes"][name]["quaternion_world"][0]).as_euler(
        "xyz"
    )[2]
    orientation = (
        Rotation.from_euler("z", yaw) * Rotation.from_euler("x", np.pi)
    ).as_matrix()
    ee = np.asarray(state()["ee_positions_world"])[0, 0]
    move_tool(controller, [ee[0], ee[1], max(0.23, ee[2])], orientation, seconds=2)
    move_tool(controller, start + [0, 0, 0.14], orientation, seconds=3)
    mark(name + "_above")
    if args.stop_after == "approach":
        return False
    move_tool(controller, start + [0, 0, 0.003], orientation, seconds=3)
    mark(name + "_at_grasp")
    controller.target_base(
        controller.target[:3, 3], controller.target[:3, :3], trigger=1.0
    )
    wait_steps(count=20)
    mark(name + "_closed")
    move_tool(controller, start + [0, 0, 0.16], orientation, seconds=3)
    lifted = mark(name + "_lifted")
    if cube(name)[2] - start[2] < 0.08:
        raise RuntimeError(f"{name} did not lift")
    if args.stop_after == "grasp":
        return False
    goal = cube(support) + [0, 0, 0.0468]
    held_offset = cube(name) - np.asarray(lifted["ee_positions_world"])[0, 0]
    move_tool(controller, goal - held_offset + [0, 0, 0.12], orientation, seconds=3)
    move_tool(controller, goal - held_offset + [0, 0, 0.003], orientation, seconds=3)
    for _ in range(3):
        error = goal + [0, 0, 0.003] - cube(name)
        if np.linalg.norm(error) < 0.003:
            break
        target = np.linalg.inv(controller.base_to_world) @ controller.target
        controller.move_world(
            target[:3, 3] + np.clip(error, -0.01, 0.01), orientation, seconds=0.6
        )
    mark(name + "_placed")
    controller.target_base(
        controller.target[:3, 3], controller.target[:3, :3], trigger=0.0
    )
    wait_steps(count=5)
    ee = np.asarray(state()["ee_positions_world"])[0, 0]
    move_tool(controller, ee + [0, 0, 0.13], orientation, seconds=3)
    wait_steps(count=30)
    mark(name + "_released")
    delta = cube(name) - cube(support)
    if np.linalg.norm(delta[:2]) > 0.015 or abs(delta[2] - 0.0468) > 0.006:
        raise RuntimeError(f"{name} did not settle on {support}: {delta.tolist()}")
    return True


controller = None
try:
    command("head", quaternion=Rotation.from_euler("x", -0.55).as_quat().tolist())
    controller = XRControllerBridge()
    if not state().get("control", {}).get("is_active"):
        play()
    controller.target_base(controller.target[:3, 3], controller.target[:3, :3])
    mark("started")
    if transfer(controller, "cube_2", "cube_1"):
        if transfer(controller, "cube_3", "cube_2"):
            final = wait_steps(count=30)
            if not final.get("exported_successful_episodes", 0):
                raise RuntimeError(
                    "Stack sequence ended without native successful export"
                )
    print("STACK_LOGIC_END", run, flush=True)
except Exception as error:
    (run / "failure.txt").write_text(repr(error) + "\n")
    print("STACK_LOGIC_FAILED", repr(error), flush=True)
    raise
finally:
    if controller is not None:
        controller.release()
    trace.close()
