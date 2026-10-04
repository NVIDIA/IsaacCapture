# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Keyboard Printer Example.

Prints every currently-held key (not just the SE3-relevant subset) each frame,
via KeyboardSource's "keyboard_all_keys" bitmap output. The keyboard plugin
self-discovers its device and is auto-launched by TeleopSession -- no external
process to start manually.
"""

import sys
import time
from pathlib import Path

from isaaccapture.cloudxr import CloudXRLauncher
from isaaccapture.retargeting_engine.deviceio_source_nodes import KeyboardSource
from isaaccapture.teleop_session_manager import (
    TeleopSession,
    TeleopSessionConfig,
    PluginConfig,
)


PLUGIN_ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent / "plugins"
PLUGIN_NAME = "keyboard"
PLUGIN_ROOT_ID = "keyboard"

# Evdev key codes (linux/input-event-codes.h) -> display name, for the keys on a
# standard PC keyboard. Codes not listed here still show up (as "code<N>") --
# this table is for readability, not a completeness gate.
KEY_NAMES = {
    1: "ESC",
    14: "BACKSPACE",
    15: "TAB",
    28: "ENTER",
    29: "LCTRL",
    42: "LSHIFT",
    54: "RSHIFT",
    56: "LALT",
    57: "SPACE",
    58: "CAPSLOCK",
    69: "NUMLOCK",
    70: "SCROLLLOCK",
    97: "RCTRL",
    100: "RALT",
    102: "HOME",
    103: "UP",
    104: "PAGEUP",
    105: "LEFT",
    106: "RIGHT",
    107: "END",
    108: "DOWN",
    109: "PAGEDOWN",
    110: "INSERT",
    111: "DELETE",
    **{2 + i: str((i + 1) % 10) for i in range(10)},  # KEY_1..KEY_0 -> "1".."9","0"
    **{59 + i: f"F{i + 1}" for i in range(10)},  # KEY_F1..KEY_F10
    87: "F11",
    88: "F12",
    16: "Q",
    17: "W",
    18: "E",
    19: "R",
    20: "T",
    21: "Y",
    22: "U",
    23: "I",
    24: "O",
    25: "P",
    30: "A",
    31: "S",
    32: "D",
    33: "F",
    34: "G",
    35: "H",
    36: "J",
    37: "K",
    38: "L",
    44: "Z",
    45: "X",
    46: "C",
    47: "V",
    48: "B",
    49: "N",
    50: "M",
}

ALL_KEYS_BITMAP_SIZE = 256


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    CloudXRLauncher.add_launcher_arguments(parser)
    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("  Keyboard Printer Example")
    print("=" * 80)
    print("Press any key on the machine running this example.")
    print("=" * 80 + "\n")

    # ==================================================================
    # Setup: Create keyboard source
    # ==================================================================
    keyboard_source = KeyboardSource(name="keyboard")

    # ==================================================================
    # Configure Plugins
    # ==================================================================

    plugins = []
    if PLUGIN_ROOT_DIR.exists():
        plugins.append(
            PluginConfig(
                plugin_name=PLUGIN_NAME,
                plugin_root_id=PLUGIN_ROOT_ID,
                search_paths=[PLUGIN_ROOT_DIR],
            )
        )

    # ==================================================================
    # Create and run TeleopSession
    # ==================================================================

    session_config = TeleopSessionConfig(
        app_name="KeyboardPrinterExample",
        trackers=[],
        pipeline=keyboard_source,
        plugins=plugins,
    )

    with CloudXRLauncher.launch_context(args):
        with TeleopSession(session_config) as session:
            start_time = time.time()
            prev_pressed: set[int] = set()

            while time.time() - start_time < 30.0:
                result = session.step()
                bitmap_group = result["keyboard_all_keys"]

                elapsed = session.get_elapsed_time()
                if bitmap_group.is_none:
                    print(
                        f"[{elapsed:5.1f}s] (no keyboard data yet)",
                        end="\r",
                        flush=True,
                    )
                    time.sleep(0.01)
                    continue

                bitmap = bitmap_group[0]
                pressed = {code for code in range(ALL_KEYS_BITMAP_SIZE) if bitmap[code]}
                names = [KEY_NAMES.get(code, f"code{code}") for code in sorted(pressed)]

                # Live status line (overwritten each frame).
                print(
                    f"[{elapsed:5.1f}s] Held: {' '.join(names) or '-'}" + " " * 20,
                    end="\r",
                    flush=True,
                )

                # Permanent, scrollable log of every press/release transition -- a
                # quick tap can flash by on the status line above before you notice
                # it, but every transition is logged here.
                for code in sorted(pressed - prev_pressed):
                    print(
                        f"[{elapsed:5.1f}s] {KEY_NAMES.get(code, f'code{code}')} down"
                    )
                for code in sorted(prev_pressed - pressed):
                    print(f"[{elapsed:5.1f}s] {KEY_NAMES.get(code, f'code{code}')} up")
                prev_pressed = pressed

                time.sleep(0.01)  # ~100 FPS

            print("\nTime limit reached.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
