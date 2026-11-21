# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Gamepad Printer Example.

Prints the gamepad's sticks, triggers and held buttons each frame, via GamepadSource's "gamepad"
output, and what the gamepad retargeters make of them: the end-effector delta
(GamepadToSe3RelRetargeter), the base velocity (GamepadToSe2Retargeter) and the gripper command
(GamepadGripperRetargeter, toggled by the west face button: X on Xbox, Square on PlayStation).
Buttons print by their field name in the standard layout, so the names are the same on every pad.

Like a downstream app driving a robot, it also integrates the end-effector delta into an absolute
pose and the base velocity into a base pose, treating both as rates per second. Menu resets them.
Stick drift above --dead-zone shows up as a pose that creeps with the pad at rest.

The session starts the gamepad plugin bundled with isaaccapture; nothing else needs to run. Like
every TeleopSession it opens an OpenXR session; the CloudXR launcher arguments start a runtime, and
no headset needs to connect. GamepadToSe3RelRetargeter needs scipy (the retargeters-lite extra).
"""

import argparse
import math
import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation

from isaaccapture.cloudxr import CloudXRLauncher
from isaaccapture.retargeters import (
    GamepadGripperRetargeter,
    GamepadToSe2Retargeter,
    GamepadToSe2RetargeterConfig,
    GamepadToSe3RelRetargeter,
    GamepadToSe3RelRetargeterConfig,
)
from isaaccapture.retargeting_engine.deviceio_source_nodes import GamepadSource
from isaaccapture.retargeting_engine.interface import OutputCombiner
from isaaccapture.retargeting_engine.tensor_types import GamepadInputIndex
from isaaccapture.teleop_session_manager import (
    BUNDLED_PLUGINS_DIR,
    PluginConfig,
    TeleopSession,
    TeleopSessionConfig,
)

STATUS_LINES = 3


def build_pipeline(dead_zone: float) -> OutputCombiner:
    """The source's raw outputs next to the SE(3), SE(2) and gripper retargeters' outputs."""
    source = GamepadSource(name="gamepad")
    gamepad = {"gamepad": source.output("gamepad")}
    se3 = GamepadToSe3RelRetargeter(
        GamepadToSe3RelRetargeterConfig(dead_zone=dead_zone), name="se3"
    )
    se2 = GamepadToSe2Retargeter(
        GamepadToSe2RetargeterConfig(dead_zone=dead_zone), name="se2"
    )
    gripper = GamepadGripperRetargeter(name="gripper")
    return OutputCombiner(
        {
            "gamepad": source.output("gamepad"),
            "ee_delta": se3.connect(gamepad).output("ee_delta"),
            "base_command": se2.connect(gamepad).output("base_command"),
            "gripper_command": gripper.connect(gamepad).output("gripper_command"),
        }
    )


class IntegratedPoses:
    """End-effector and base poses accumulated from the retargeters' rate outputs."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.ee_position = np.zeros(3)
        self.ee_rotation = Rotation.identity()
        self.base = np.zeros(3)  # x, y, yaw

    def step(self, ee_delta, base_command, dt: float) -> None:
        delta = np.asarray(ee_delta, dtype=np.float64)
        self.ee_position += delta[:3] * dt
        # Left-multiplied: the delta rotates about the base frame's axes, as a relative IK
        # controller applies it.
        self.ee_rotation = Rotation.from_rotvec(delta[3:] * dt) * self.ee_rotation

        v_x, v_y, omega_z = np.asarray(base_command, dtype=np.float64)
        yaw = self.base[2]
        self.base[0] += (v_x * math.cos(yaw) - v_y * math.sin(yaw)) * dt
        self.base[1] += (v_x * math.sin(yaw) + v_y * math.cos(yaw)) * dt
        self.base[2] = math.remainder(yaw + omega_z * dt, math.tau)

    def describe(self) -> str:
        rpy = self.ee_rotation.as_euler("XYZ", degrees=True)
        return (
            f"EE pose: pos [{_fmt(self.ee_position)}]  rpy [{_fmt(rpy, 1)}] deg  "
            f"Base pose: [{_fmt(self.base[:2])}] yaw {math.degrees(self.base[2]):+.1f} deg"
        )


# GamepadInput's buttons, printed without the "_click" suffix, and its sticks and triggers.
BUTTONS = [i for i in GamepadInputIndex if i.name.endswith("_CLICK")]
AXES = [i for i in GamepadInputIndex if not i.name.endswith("_CLICK")]


def _name(button) -> str:
    return button.name.removesuffix("_CLICK").lower()


def _fmt(values, decimals: int = 2) -> str:
    return " ".join(f"{v:+.{decimals}f}" for v in values)


def _show(lines: list[str]) -> None:
    """Rewrite the status block in place: move up to its first line, then print each line."""
    print(
        f"\033[{STATUS_LINES - 1}F" + "\n".join(f"{line}\033[K" for line in lines),
        end="",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--player",
        type=int,
        default=1,
        help="Read the Nth connected gamepad (default: the first)",
    )
    parser.add_argument(
        "--dead-zone",
        type=float,
        default=GamepadToSe3RelRetargeterConfig().dead_zone,
        help="Stick deflection below which the retargeters read zero",
    )
    parser.add_argument("--duration", type=float, default=30.0, help="Seconds to run")
    CloudXRLauncher.add_launcher_arguments(parser)
    args = parser.parse_args()

    session_config = TeleopSessionConfig(
        app_name="GamepadPrinterExample",
        pipeline=build_pipeline(args.dead_zone),
        plugins=[
            PluginConfig(
                plugin_name="gamepad",
                plugin_root_id="gamepad",
                search_paths=[BUNDLED_PLUGINS_DIR],
                plugin_args=[f"--player={args.player}"],
                required=True,
            )
        ],
    )

    print(
        "Move the sticks or D-pad, or press a button. west toggles the gripper, menu resets the poses."
    )
    print("\n" * (STATUS_LINES - 1), end="")
    poses = IntegratedPoses()
    with CloudXRLauncher.launch_context(args), TeleopSession(session_config) as session:
        start_time = time.time()
        last_step = time.monotonic()
        prev_pressed: set = set()

        while time.time() - start_time < args.duration:
            result = session.step()
            now = time.monotonic()
            poses.step(
                result["ee_delta"][0], result["base_command"][0], now - last_step
            )
            last_step = now

            elapsed = session.get_elapsed_time()
            gripper = "closed" if result["gripper_command"][0] < 0 else "open"
            retargeted = (
                f"EE delta: [{_fmt(result['ee_delta'][0])}]  "
                f"Base cmd: [{_fmt(result['base_command'][0])}]  Gripper: {gripper}"
            )
            if result["gamepad"].is_none:
                _show([f"[{elapsed:5.1f}s] (no gamepad)", retargeted, poses.describe()])
                time.sleep(0.01)
                continue

            gamepad = result["gamepad"]
            pressed = {b for b in BUTTONS if float(gamepad[b]) > 0.5}
            if GamepadInputIndex.MENU_CLICK in pressed - prev_pressed:
                poses.reset()
            held = " ".join(_name(i) for i in sorted(pressed)) or "-"
            _show(
                [
                    f"[{elapsed:5.1f}s] Axes: [{_fmt(float(gamepad[a]) for a in AXES)}]  Held: {held}",
                    retargeted,
                    poses.describe(),
                ]
            )
            # Log every transition above the status block: a quick tap can flash by in it.
            events = [
                f"[{elapsed:5.1f}s] {_name(i)} down"
                for i in sorted(pressed - prev_pressed)
            ]
            events += [
                f"[{elapsed:5.1f}s] {_name(i)} up"
                for i in sorted(prev_pressed - pressed)
            ]
            if events:
                print("\n" + "\n".join(events) + "\n" * STATUS_LINES, end="")
            prev_pressed = pressed

            time.sleep(0.01)

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
