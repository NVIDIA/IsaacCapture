# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A stand-in harness: reads what a robot is and its newest readings on the server machine.

It knows no robot in advance: the robot's description says which data series, joints and
grippers exist, and the calls of `isaaccapture.robot_state` read them by name.
"""

import argparse
import os
import sys
import time

from isaaccapture.robot_state import RobotState, SeriesDescription


def _age(reading) -> str:
    """How old a reading is; without a clock estimate, the time since it arrived."""
    if reading.age_s is None:
        return f"arrived {reading.receive_age_s:.3f} s ago"
    return f"{reading.age_s:.3f} s old"


def _joint_lines(series: SeriesDescription, reading) -> list[str]:
    lines = []
    limits = series.limits
    for joint in series.joints:
        value = reading.get(joint.name)
        if value is None or not value.valid:
            lines.append(f"    {joint.name:<16} not read")
            continue
        text = f"    {joint.name:<16} {value.position:8.3f} {joint.unit}".rstrip()
        if value.velocity is not None:
            text += f"  velocity {value.velocity:.3f}"
        if value.effort is not None:
            text += f"  effort {value.effort:.3f}"
        if joint.name in limits:
            text += "  limits {:.3f} to {:.3f}".format(*limits[joint.name])
        lines.append(text)
    return lines


def _status_lines(reading) -> list[str]:
    lines = [f"    {name:<16} {value!r}" for name, value in reading.items()]
    for fault in reading.faults:
        where = f" at {fault.source}" if fault.source else ""
        lines.append(f"    {fault.severity} {fault.code}{where}: {fault.text}")
    return lines


def _record_lines(reading) -> list[str]:
    if reading.values is None:
        return [f"    {len(reading.data)} bytes of {reading.schema_name}"]
    return [f"    {name:<16} {value!r}" for name, value in reading.values.items()]


def describe(robot: RobotState) -> list[str]:
    """Lines for a person: what the robot is, then every series and gripper with its newest reading."""
    what = robot.description()
    if what is None:
        if not robot.server_reachable:
            return ["no Teleop Session Server serves robot state here yet"]
        return ["no robot app has sent a robot description"]
    lines = [
        f"robot {what.robot}"
        + ("" if robot.connected else " (robot app not connected)")
    ]
    for name, series in what.series.items():
        reading = robot.record(name)
        if reading is None:
            lines.append(f"  {name} ({series.shape_name}): no fresh reading")
            continue
        lines.append(f"  {name} ({series.shape_name}), {_age(reading)}")
        if series.shape == "joints":
            lines += _joint_lines(series, reading)
        elif series.shape == "status":
            lines += _status_lines(reading)
        else:
            lines += _record_lines(reading)
    for name in what.grippers:
        gripper = robot.gripper(name)
        if gripper is None or not gripper.valid:
            lines.append(f"  gripper {name}: no fresh reading")
        else:
            lines.append(f"  gripper {name}: {100.0 * gripper.opening:.0f} % open")
    return lines


def watch(
    socket_path: str | os.PathLike | None = None,
    *,
    period_s: float = 1.0,
    duration_s: float | None = None,
) -> bool:
    """Print what `describe()` reads every `period_s`; True when a robot description was read."""
    seen = False
    end = None if duration_s is None else time.monotonic() + duration_s
    # One snapshot for each printout: every read between two refresh() calls sees the same one.
    with RobotState.connect(socket_path, auto_refresh=False) as robot:
        while True:
            robot.refresh()
            seen = seen or robot.description() is not None
            print("\n".join(describe(robot)), end="\n\n", flush=True)
            if end is not None and time.monotonic() + period_s > end:
                return seen
            time.sleep(period_s)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m isaaccapture_examples.robot_state.stand_in_harness",
        description=(
            "Stand-in harness: prints what the robot connected to this machine's "
            "Teleop Session Server is and what it last read, until Ctrl+C."
        ),
    )
    parser.add_argument(
        "--robot-state-socket",
        default=None,
        metavar="PATH",
        help="The path the server was given with --robot-state-socket (default: its default).",
    )
    parser.add_argument(
        "--period-s",
        type=float,
        default=1.0,
        help="Seconds between two printouts (default: %(default)s).",
    )
    args = parser.parse_args(argv)
    try:
        watch(args.robot_state_socket, period_s=args.period_s)
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
