# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Gamepad Printer Example.

Prints every currently-held button and the full axis array each frame, via
GamepadSource's "gamepad_buttons" and "gamepad_axes" outputs. Carries no semantic
mapping (stick, trigger, toggle) -- that belongs in a retargeter (e.g.
GamepadToSe3RelRetargeter) consuming this source's output.

The gamepad is read in process from the first connected joystick (/dev/input/jsN), or from
``--device``. It needs no plugin process, and udev grants the logged-in user access to
joysticks without the ``input`` group. Like every TeleopSession it opens an OpenXR session; the
CloudXR launcher arguments start a runtime, and no headset needs to connect.
"""

import sys
import time

from isaaccapture.cloudxr import CloudXRLauncher
from isaaccapture.retargeting_engine.deviceio_source_nodes import GamepadSource
from isaaccapture.teleop_session_manager import TeleopSession, TeleopSessionConfig


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Joystick device to read (e.g. /dev/input/js0); default: the first connected one.",
    )
    CloudXRLauncher.add_launcher_arguments(parser)
    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("  Gamepad Printer Example")
    print("=" * 80)
    print("Press any button or move a stick on the connected gamepad.")
    print("=" * 80 + "\n")

    # ==================================================================
    # Setup: Create gamepad source
    # ==================================================================
    gamepad_source = GamepadSource(name="gamepad", device_path=args.device)

    # ==================================================================
    # Create and run TeleopSession
    # ==================================================================

    session_config = TeleopSessionConfig(
        app_name="GamepadPrinterExample",
        trackers=[],
        pipeline=gamepad_source,
    )

    with CloudXRLauncher.launch_context(args), TeleopSession(session_config) as session:
        start_time = time.time()
        prev_pressed: set[int] = set()

        while time.time() - start_time < 30.0:
            result = session.step()
            buttons_group = result["gamepad_buttons"]
            axes_group = result["gamepad_axes"]

            elapsed = session.get_elapsed_time()
            if buttons_group.is_none:
                print(
                    f"[{elapsed:5.1f}s] (no gamepad connected)",
                    end="\r",
                    flush=True,
                )
                time.sleep(0.01)
                continue

            bitmap = buttons_group[0]
            axes = axes_group[0]
            pressed = {code for code in range(len(bitmap)) if bitmap[code]}
            axes_str = " ".join(f"{v:+.2f}" for v in axes)

            # Live status line (overwritten each frame).
            names = [f"btn{code}" for code in sorted(pressed)]
            print(
                f"[{elapsed:5.1f}s] Axes: [{axes_str}]  Held: {' '.join(names) or '-'}"
                + " " * 20,
                end="\r",
                flush=True,
            )

            # Permanent, scrollable log of every press/release transition -- a
            # quick tap can flash by on the status line above before you notice
            # it, but every transition is logged here.
            for code in sorted(pressed - prev_pressed):
                print(f"[{elapsed:5.1f}s] btn{code} down")
            for code in sorted(prev_pressed - pressed):
                print(f"[{elapsed:5.1f}s] btn{code} up")
            prev_pressed = pressed

            time.sleep(0.01)  # ~100 FPS

        print("\nTime limit reached.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
