# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
SpaceMouse Printer Example.

Prints the translation axes, rotation axes, and every currently-held button each
frame, via SpaceMouseSource's "spacemouse_translation", "spacemouse_rotation", and
"spacemouse_buttons" outputs. Carries no semantic mapping (a position delta, a
rotation delta, a gripper toggle) -- that belongs in a retargeter (e.g.
SpaceMouseToSe3RelRetargeter) consuming this source's output.

The SpaceMouse is read in process from the first connected device of a validated model
(/dev/hidrawN), or from ``--device``. It needs no plugin process. Like every TeleopSession it
opens an OpenXR session; the CloudXR launcher arguments start a runtime, and no headset needs
to connect. Reading /dev/hidraw* needs a udev rule granting
your user access, e.g. /etc/udev/rules.d/70-spacemouse.rules containing:

    KERNEL=="hidraw*", ATTRS{idVendor}=="256f", TAG+="uaccess"
    KERNEL=="hidraw*", ATTRS{idVendor}=="046d", ATTRS{idProduct}=="c62[0-9a-f]", TAG+="uaccess"
"""

import sys
import time

from isaaccapture.cloudxr import CloudXRLauncher
from isaaccapture.retargeting_engine.deviceio_source_nodes import SpaceMouseSource
from isaaccapture.teleop_session_manager import TeleopSession, TeleopSessionConfig


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--device",
        default=None,
        help="HID device to read (e.g. /dev/hidraw3); default: the first connected SpaceMouse.",
    )
    parser.add_argument(
        "--combined-report",
        action="store_true",
        help="With --device: the device is a 3Dconnexion Universal Receiver (combined report).",
    )
    CloudXRLauncher.add_launcher_arguments(parser)
    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("  SpaceMouse Printer Example")
    print("=" * 80)
    print("Move or twist the connected SpaceMouse, or press a button.")
    print("=" * 80 + "\n")

    # ==================================================================
    # Setup: Create spacemouse source
    # ==================================================================
    spacemouse_source = SpaceMouseSource(
        name="spacemouse",
        device_path=args.device,
        combined_report=args.combined_report,
    )

    # ==================================================================
    # Create and run TeleopSession
    # ==================================================================

    session_config = TeleopSessionConfig(
        app_name="SpaceMousePrinterExample",
        trackers=[],
        pipeline=spacemouse_source,
    )

    with CloudXRLauncher.launch_context(args), TeleopSession(session_config) as session:
        start_time = time.time()
        prev_pressed: set[int] = set()

        while time.time() - start_time < 30.0:
            result = session.step()
            translation_group = result["spacemouse_translation"]
            rotation_group = result["spacemouse_rotation"]
            buttons_group = result["spacemouse_buttons"]

            elapsed = session.get_elapsed_time()
            if translation_group.is_none:
                print(
                    f"[{elapsed:5.1f}s] (no SpaceMouse connected)",
                    end="\r",
                    flush=True,
                )
                time.sleep(0.01)
                continue

            translation = translation_group[0]
            rotation = rotation_group[0]
            bitmap = buttons_group[0]
            pressed = {code for code in range(len(bitmap)) if bitmap[code]}

            t_str = " ".join(f"{v:+.2f}" for v in translation)
            r_str = " ".join(f"{v:+.2f}" for v in rotation)

            # Live status line (overwritten each frame).
            names = [f"btn{code}" for code in sorted(pressed)]
            print(
                f"[{elapsed:5.1f}s] T: [{t_str}]  R: [{r_str}]  Held: {' '.join(names) or '-'}"
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
