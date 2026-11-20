# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A stand-in robot app: publishes moving joint angles through a remote teleop session.

The robot is whatever its description file says. Every value is a function of the record
number, so a reader can check what it reads against what was published.
"""

import argparse
import logging
import math
import signal
import sys
import threading
import time
from pathlib import Path

from isaaccapture.robot_state import (
    Fault,
    RobotDescription,
    SeriesDescription,
    load_robot_description,
)
from isaaccapture.teleop_session_manager import (
    SessionMode,
    TeleopSession,
    TeleopSessionConfig,
)

logger = logging.getLogger("isaaccapture.examples.robot_state.stand_in_robot")

DESCRIPTIONS = Path(__file__).resolve().parent / "descriptions"
RATE_HZ = 20.0
# Records in one sweep of every value; all values repeat after this many records.
SWEEP_RECORDS = 200
# The stand-in fault is reported for this many records, then absent for as many.
FAULT_RECORDS = 50
FAULT_CODE = "overtemperature"


def description_path(name: str) -> Path:
    """The file of a bundled description (`six_joint_arm`, `two_arm_robot`), else `name` as a path."""
    bundled = DESCRIPTIONS / f"{name}.toml"
    return bundled if bundled.is_file() else Path(name)


def _wave(record: int, index: int) -> float:
    """-1 to 1: one cycle per SWEEP_RECORDS records, shifted for each index."""
    return math.sin(2.0 * math.pi * (record / SWEEP_RECORDS + index / 8.0))


def joint_positions(series: SeriesDescription, record: int) -> list[float]:
    """Each joint sweeps the middle 80 % of its range; a joint without limits sweeps -1 to 1."""
    positions = []
    for index, joint in enumerate(series.joints):
        if joint.min is None or joint.max is None:
            positions.append(_wave(record, index))
        else:
            middle = 0.5 * (joint.min + joint.max)
            positions.append(
                middle + 0.4 * (joint.max - joint.min) * _wave(record, index)
            )
    return positions


def joint_velocities(series: SeriesDescription, record: int) -> list[float]:
    return [
        0.5 * (joint.max_velocity or 1.0) * _wave(record, index + 2)
        for index, joint in enumerate(series.joints)
    ]


def joint_efforts(series: SeriesDescription, record: int) -> list[float]:
    return [
        0.5 * (joint.max_effort or 1.0) * _wave(record, index + 4)
        for index, joint in enumerate(series.joints)
    ]


def fault_active(record: int) -> bool:
    return (record // FAULT_RECORDS) % 2 == 1


def status_values(series: SeriesDescription, record: int) -> list:
    """`motors_on` is True and `fault` names the active fault; other readings follow their kind."""
    values = []
    for index, reading in enumerate(series.values):
        if reading.kind == "flag":
            values.append(fault_active(record) if reading.name == "fault" else True)
        elif reading.kind == "number":
            values.append(24.0 + 0.5 * _wave(record, index))
        elif reading.name == "fault":
            values.append(FAULT_CODE if fault_active(record) else "")
        elif reading.name == "motors_on":
            values.append(True)
        elif reading.kind == "text":
            values.append(f"record {record % SWEEP_RECORDS}")
        else:
            values.append(24.0 + 0.5 * _wave(record, index))
    return values


def status_faults(description: RobotDescription, record: int) -> list[Fault]:
    """One warning on the robot's first joint while the stand-in fault is active."""
    if not fault_active(record):
        return []
    source = next(
        (
            series.joints[0].name
            for series in description.series.values()
            if series.joints
        ),
        "",
    )
    return [Fault(FAULT_CODE, "stand-in fault: a motor is hot", source, "warning")]


def custom_record(series: SeriesDescription, record: int) -> dict:
    """One record of a shape the robot defines: a value for each field, by the field's kind."""
    values = {}
    for index, field in enumerate(series.values):
        wave = _wave(record, index)
        if field.kind == "flag":
            values[field.name] = wave >= 0.0
        elif field.kind == "text":
            values[field.name] = f"record {record % SWEEP_RECORDS}"
        elif field.kind == "numbers":
            values[field.name] = [wave, -wave]
        else:
            values[field.name] = 10.0 * wave
    return values


def publish_all(
    session: TeleopSession, description: RobotDescription, record: int
) -> bool:
    """Publish record number `record` of every series; False when none was sent."""
    sent = False
    for name, series in description.series.items():
        if series.shape == "joints":
            sent |= session.publish_robot_state(
                name,
                joint_positions(series, record),
                velocities=joint_velocities(series, record)
                if series.has_velocity
                else None,
                efforts=joint_efforts(series, record) if series.has_effort else None,
            )
        elif series.shape == "status":
            sent |= session.publish_robot_state(
                name,
                status_values(series, record),
                faults=status_faults(description, record),
            )
        else:
            sent |= session.publish_robot_record(name, custom_record(series, record))
    return sent


class _NoHumanInput:
    """A pipeline without sources: the stand-in robot reads no headset or controller data."""

    def get_leaf_nodes(self):
        return []

    def execute_pipeline(self, inputs, context=None):
        return {}


def run(
    endpoint: str,
    description_file: str | Path,
    *,
    rate_hz: float = RATE_HZ,
    records: int | None = None,
    stop: threading.Event | None = None,
    connect_timeout_s: float | None = 60.0,
) -> int:
    """Publish until `stop` is set or `records` are published; the number published.

    A lost link raises from `session.step()`.
    """
    if rate_hz <= 0.0:
        raise ValueError("rate_hz must be above zero")
    stop = stop or threading.Event()
    description = load_robot_description(description_file)
    config = TeleopSessionConfig(
        app_name="StandInRobot",
        pipeline=_NoHumanInput(),
        mode=SessionMode.REMOTE,
        remote_endpoint=endpoint,
        remote_connect_timeout_s=connect_timeout_s,
        robot_description=description,
    )
    record = 0
    with TeleopSession(config) as session:
        logger.info(
            "Stand-in robot '%s' publishes %s at %g records a second",
            description.robot,
            ", ".join(description.series),
            rate_hz,
        )
        due = time.monotonic()
        while not stop.is_set() and (records is None or record < records):
            session.step()
            publish_all(session, description, record)
            record += 1
            due += 1.0 / rate_hz
            stop.wait(max(0.0, due - time.monotonic()))
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m isaaccapture_examples.robot_state.stand_in_robot",
        description=(
            "Stand-in robot app: publishes the data series of a robot description "
            "through a remote teleop session until Ctrl+C."
        ),
    )
    parser.add_argument(
        "--endpoint",
        required=True,
        metavar="tcp://HOST:PORT",
        help="Address of the Teleop Session Server.",
    )
    parser.add_argument(
        "--description",
        default="six_joint_arm",
        metavar="NAME_OR_FILE",
        help="six_joint_arm, two_arm_robot, or a description file (default: %(default)s).",
    )
    parser.add_argument(
        "--rate-hz",
        type=float,
        default=RATE_HZ,
        help="Records a second of every series (default: %(default)s).",
    )
    args = parser.parse_args(argv)
    stop = threading.Event()

    def leave(*_):
        # The first signal ends the session; one that follows must not cut its closing short.
        if not stop.is_set():
            stop.set()
            raise KeyboardInterrupt

    signal.signal(signal.SIGINT, leave)
    signal.signal(signal.SIGTERM, leave)
    try:
        run(
            args.endpoint,
            description_path(args.description),
            rate_hz=args.rate_hz,
            stop=stop,
        )
    except KeyboardInterrupt:
        pass
    except (OSError, RuntimeError, ValueError) as error:
        logger.error("%s", error)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
