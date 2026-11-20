# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Robot state on one machine: a Teleop Session Server, a stand-in robot and a stand-in harness.

The server runs in its test mode (no CloudXR, no headset) on a loopback port the system
picks, with its reader socket in a temporary folder, so it disturbs nothing else that runs
on this machine.
"""

import argparse
import logging
import re
import subprocess
import sys
import tempfile
import time
from contextlib import ExitStack
from pathlib import Path

from isaaccapture.robot_state import load_robot_description

from . import stand_in_harness
from .stand_in_robot import description_path

logger = logging.getLogger("isaaccapture.examples.robot_state")

BANNER = re.compile(r"^Teleop Session Server listening on (tcp://\S+)$", re.MULTILINE)
SERVER_START_S = 60.0


def _stop(process: subprocess.Popen) -> None:
    if process.poll() is None:
        process.terminate()
    try:
        process.wait(timeout=15.0)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def _start_server(stack: ExitStack, folder: Path, socket_path: Path) -> str:
    """Start the server and return the address it listens on."""
    command = [
        sys.executable,
        "-m",
        "isaaccapture.teleop_session_server",
        "--no-launch-cloudxr-runtime",
        "--placeholder-openxr",
        "--listen",
        "tcp://127.0.0.1:0",
        "--robot-state-socket",
        str(socket_path),
    ]
    # The server prints the address it listens on to its standard output.
    output = folder / "server.out"
    with open(output, "wb") as file:
        server = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=file)
    stack.callback(_stop, server)
    deadline = time.monotonic() + SERVER_START_S
    while time.monotonic() < deadline and server.poll() is None:
        match = BANNER.search(output.read_text(errors="replace"))
        if match:
            return match.group(1)
        time.sleep(0.05)
    raise RuntimeError("the Teleop Session Server did not start")


def _start_robot(stack: ExitStack, endpoint: str, description: Path) -> None:
    command = [
        sys.executable,
        "-m",
        f"{__package__}.stand_in_robot",
        "--endpoint",
        endpoint,
        "--description",
        str(description),
    ]
    robot = subprocess.Popen(command, stdin=subprocess.DEVNULL)
    stack.callback(_stop, robot)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m isaaccapture_examples.robot_state",
        description=(
            "Start a Teleop Session Server in its test mode and a stand-in robot that "
            "publishes robot state to it, and print what a stand-in harness reads."
        ),
    )
    parser.add_argument(
        "--description",
        default="six_joint_arm",
        metavar="NAME_OR_FILE",
        help="six_joint_arm, two_arm_robot, or a description file (default: %(default)s).",
    )
    parser.add_argument(
        "--duration-s",
        type=float,
        default=None,
        help="Stop after this many seconds (default: run until Ctrl+C).",
    )
    parser.add_argument(
        "--period-s",
        type=float,
        default=1.0,
        help="Seconds between two printouts (default: %(default)s).",
    )
    args = parser.parse_args(argv)
    description = description_path(args.description)
    try:
        load_robot_description(description)
    except (OSError, ValueError) as error:
        parser.error(str(error))

    with ExitStack() as stack:
        folder = Path(
            stack.enter_context(
                tempfile.TemporaryDirectory(prefix="isaaccapture-robot-state-")
            )
        )
        socket_path = folder / "rs.sock"
        try:
            endpoint = _start_server(stack, folder, socket_path)
            _start_robot(stack, endpoint, description)
            seen = stand_in_harness.watch(
                socket_path, period_s=args.period_s, duration_s=args.duration_s
            )
        except KeyboardInterrupt:
            return 0
        except RuntimeError as error:
            logger.error("%s", error)
            return 1
    if not seen:
        logger.error("The stand-in harness read no robot description")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
