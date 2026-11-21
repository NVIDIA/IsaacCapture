# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
SpaceMouse Printer Example.

Prints the SpaceMouse's translation and rotation axes and held buttons each frame, via
SpaceMouseSource's "spacemouse_translation", "spacemouse_rotation" and "spacemouse_buttons"
outputs. Mapping them to motion belongs in a retargeter (e.g. SpaceMouseToSe3RelRetargeter).

The session starts the spacemouse plugin bundled with isaaccapture; nothing else needs to run.
Reading /dev/hidraw* needs the plugin's udev rule (src/plugins/spacemouse/install_udev_rules.sh,
run once on the host). Like every TeleopSession it opens an OpenXR session; the CloudXR launcher
arguments start a runtime, and no headset needs to connect.
"""

import argparse
import sys
import time

from isaaccapture.cloudxr import CloudXRLauncher
from isaaccapture.retargeting_engine.deviceio_source_nodes import SpaceMouseSource
from isaaccapture.teleop_session_manager import (
    BUNDLED_PLUGINS_DIR,
    PluginConfig,
    TeleopSession,
    TeleopSessionConfig,
)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--device",
        help="HID node to read (e.g. /dev/hidraw3); default: the first connected SpaceMouse.",
    )
    parser.add_argument(
        "--combined-report",
        action="store_true",
        help="With --device: the device is a 3Dconnexion Universal Receiver.",
    )
    parser.add_argument("--duration", type=float, default=30.0, help="Seconds to run")
    CloudXRLauncher.add_launcher_arguments(parser)
    args = parser.parse_args()

    plugin_args = []
    if args.device:
        plugin_args.append(f"--device={args.device}")
        if args.combined_report:
            plugin_args.append("--combined-report")

    session_config = TeleopSessionConfig(
        app_name="SpaceMousePrinterExample",
        pipeline=SpaceMouseSource(name="spacemouse"),
        plugins=[
            PluginConfig(
                plugin_name="spacemouse",
                plugin_root_id="spacemouse",
                search_paths=[BUNDLED_PLUGINS_DIR],
                plugin_args=plugin_args,
                required=True,
            )
        ],
    )

    print("Move or twist the SpaceMouse, or press a button.")
    with CloudXRLauncher.launch_context(args), TeleopSession(session_config) as session:
        start_time = time.time()
        prev_pressed: set[int] = set()

        while time.time() - start_time < args.duration:
            result = session.step()
            elapsed = session.get_elapsed_time()
            if result["spacemouse_translation"].is_none:
                print(f"[{elapsed:5.1f}s] (no SpaceMouse)", end="\r", flush=True)
                time.sleep(0.01)
                continue

            translation = " ".join(
                f"{v:+.2f}" for v in result["spacemouse_translation"][0]
            )
            rotation = " ".join(f"{v:+.2f}" for v in result["spacemouse_rotation"][0])
            bitmap = result["spacemouse_buttons"][0]
            pressed = {i for i in range(len(bitmap)) if bitmap[i]}
            held = " ".join(f"btn{i}" for i in sorted(pressed)) or "-"
            print(
                f"[{elapsed:5.1f}s] T: [{translation}]  R: [{rotation}]  Held: {held}"
                + " " * 20,
                end="\r",
                flush=True,
            )
            # Log every transition: a quick tap can flash by on the status line.
            for i in sorted(pressed - prev_pressed):
                print(f"\n[{elapsed:5.1f}s] btn{i} down")
            for i in sorted(prev_pressed - pressed):
                print(f"\n[{elapsed:5.1f}s] btn{i} up")
            prev_pressed = pressed

            time.sleep(0.01)

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
