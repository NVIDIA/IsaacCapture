# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Scripted Franka stacking through browser XR input, with read-only scene feedback."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .session import FrankaTaskSession


def transfer(session, name, support, stop_after):
    initial = session.observe()
    start = np.array(initial["cubes"][name]["position_world"][0])
    yaw = Rotation.from_quat(initial["cubes"][name]["quaternion_world"][0]).as_euler(
        "xyz"
    )[2]
    orientation = (
        Rotation.from_euler("z", yaw) * Rotation.from_euler("x", np.pi)
    ).as_quat()
    ee = np.array(initial["ee_positions_world"][0][0])
    session.move_tool([ee[0], ee[1], max(0.23, ee[2])], orientation)
    session.move_tool(start + [0, 0, 0.14], orientation, seconds=3)
    session.mark(name + "_above")
    if stop_after == "approach":
        return False
    session.move_tool(start + [0, 0, 0.003], orientation, seconds=3)
    session.set_gripper(closed=True)
    session.wait(1)
    session.move_tool(start + [0, 0, 0.16], orientation, seconds=3)
    lifted = session.mark(name + "_lifted")
    cube = np.array(lifted["cubes"][name]["position_world"][0])
    if cube[2] - start[2] < 0.08:
        raise RuntimeError(f"{name} did not lift")
    if stop_after == "grasp":
        return False
    goal = np.array(lifted["cubes"][support]["position_world"][0]) + [0, 0, 0.0468]
    held_offset = cube - lifted["ee_positions_world"][0][0]
    session.move_tool(goal - held_offset + [0, 0, 0.12], orientation, seconds=3)
    session.move_tool(goal - held_offset + [0, 0, 0.003], orientation, seconds=3)
    for _ in range(3):
        current = session.observe()
        if current.get("exported_successful_episodes", 0):
            return True
        error = goal + [0, 0, 0.003] - current["cubes"][name]["position_world"][0]
        if np.linalg.norm(error) < 0.003:
            break
        session.move_tool(
            np.array(current["ee_positions_world"][0][0]) + np.clip(error, -0.01, 0.01),
            orientation,
            seconds=0.6,
        )
    session.mark(name + "_placed")
    session.set_gripper(closed=False)
    session.wait(0.25)
    ee = np.array(session.observe()["ee_positions_world"][0][0])
    session.move_tool(ee + [0, 0, 0.13], orientation, seconds=3)
    session.wait(1.5)
    final = session.mark(name + "_released")
    if not final.get("exported_successful_episodes", 0):
        delta = (
            np.array(final["cubes"][name]["position_world"][0])
            - final["cubes"][support]["position_world"][0]
        )
        if np.linalg.norm(delta[:2]) > 0.015 or abs(delta[2] - 0.0468) > 0.006:
            raise RuntimeError(f"{name} did not settle on {support}: {delta.tolist()}")
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stop-after", choices=["approach", "grasp", "stack"], default="stack"
    )
    parser.add_argument("--cdp-url", default="http://127.0.0.1:9222")
    parser.add_argument("--feedback-url", default="http://127.0.0.1:8765/state")
    parser.add_argument("--observation-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run = args.output / ("stack-logic-" + str(time.time_ns()))
    with FrankaTaskSession(
        observation_dir=args.observation_dir,
        output=run,
        cdp_url=args.cdp_url,
        feedback_url=args.feedback_url,
    ) as session:
        if transfer(session, "cube_2", "cube_1", args.stop_after) and transfer(
            session, "cube_3", "cube_2", args.stop_after
        ):
            final = session.wait_for_export()
            print(
                "STACK_LOGIC_END",
                json.dumps(
                    {
                        "run": str(run),
                        "native_exports": final["exported_successful_episodes"],
                    }
                ),
                flush=True,
            )
        else:
            print("STACK_LOGIC_PARTIAL", args.stop_after, run, flush=True)


if __name__ == "__main__":
    main()
