# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Writes the label sidecar from the windows the panel recorded.

Opens the finished recording read-only, resolves each window's record numbers to
``sample_time_local_common_clock`` timestamps, and writes the sidecar the checker loads
beside the file.
"""

from __future__ import annotations

import json
import math
import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from full_body_cts.frames import Frame
from full_body_cts.mcap_source import McapFrameSource
from full_body_cts.profile import FULL_BODY

from steps import Window

J = {
    name: FULL_BODY.joint_names.index(name)
    for name in ("PELVIS", "LEFT_ANKLE", "RIGHT_ANKLE", "LEFT_HAND", "RIGHT_HAND")
}

NOTE = "Step labels for the recording beside this file."
CLOCK_DOMAIN = "sample_time_local_common_clock (system monotonic, nanoseconds)"

#: One row of the report the panel shows after the file closes: name, ok, detail.
Check = tuple[str, bool, str]


@dataclass(frozen=True, slots=True)
class Geometry:
    """The few distances ``verify()`` re-derives a window's motion from."""

    separation: float
    left_hand_y: float
    right_hand_y: float
    left_ankle_y: float
    right_ankle_y: float
    pelvis_y: float


@dataclass(frozen=True, slots=True)
class Record:
    """One record of the body channel, addressed by its position in the file."""

    sequence: int
    sample_ns: int | None
    geometry: Geometry | None


def read(recording: Path) -> list[Record]:
    """Every record in file order; none is skipped, so a record number indexes the list."""
    return [
        Record(
            sequence=frame.sequence,
            sample_ns=frame.sample_time_ns,
            geometry=_geometry(frame),
        )
        for frame in McapFrameSource(str(recording))
    ]


def _geometry(frame: Frame) -> Geometry | None:
    joints = frame.joints
    if joints is None or not all(joints[index].is_valid for index in J.values()):
        return None
    left, right = joints[J["LEFT_HAND"]], joints[J["RIGHT_HAND"]]
    return Geometry(
        separation=math.dist(left.position, right.position),
        left_hand_y=left.position[1],
        right_hand_y=right.position[1],
        left_ankle_y=joints[J["LEFT_ANKLE"]].position[1],
        right_ankle_y=joints[J["RIGHT_ANKLE"]].position[1],
        pelvis_y=joints[J["PELVIS"]].position[1],
    )


def audit(records: Sequence[Record], windows: Sequence[Window]) -> list[Check]:
    """What has to hold before a frame number may be read as a record number.

    Reported, not raised; never map the numbers onto the nearest records on a mismatch.
    """
    reach = max((window.end_frame for window in windows), default=0)
    renumbered = sum(
        1 for position, record in enumerate(records) if record.sequence != position
    )
    unstamped = sum(1 for record in records if record.sample_ns is None)
    return [
        (
            "frames recorded",
            reach <= len(records),
            f"{len(records)} records, windows reach {reach}",
        ),
        (
            "sequence in file order",
            renumbered == 0,
            "every sequence equals its position"
            if renumbered == 0
            else f"{renumbered} records are numbered otherwise",
        ),
        (
            "sample clock present",
            unstamped == 0,
            "every record stamped"
            if unstamped == 0
            else f"{unstamped} records carry no sample time",
        ),
    ]


def lay_out(records: Sequence[Record], windows: Sequence[Window]) -> list[dict]:
    """The sidecar's step list: each window's record numbers resolved to timestamps."""
    first_ns = next(
        (record.sample_ns for record in records if record.sample_ns is not None), None
    )
    if first_ns is None:
        return []

    steps = []
    for window in windows:
        start_ns = _stamp(records, window.start_frame)
        end_ns = _stamp(records, window.end_frame)
        if start_ns is None or end_ns is None:
            continue
        steps.append(
            {
                "index": window.index,
                "label": window.label,
                "start_ns": start_ns,
                "end_ns": end_ns,
                "start_s_from_first_sample": round((start_ns - first_ns) / 1e9, 6),
                "end_s_from_first_sample": round((end_ns - first_ns) / 1e9, 6),
                "is_still_window": window.is_still_window,
                "start_frame": window.start_frame,
                "end_frame": window.end_frame,
                # Which input opened the window.
                "boundary_source": window.source,
            }
        )
    return steps


def _stamp(records: Sequence[Record], frame: int) -> int | None:
    """The sample time of record ``frame``; one past the last record resolves to the final sample plus 1 ns."""
    if 0 <= frame < len(records):
        return records[frame].sample_ns
    if frame == len(records) and records and records[-1].sample_ns is not None:
        return records[-1].sample_ns + 1
    return None


def _mean(window: Sequence[Geometry], field: str) -> float:
    return statistics.mean(getattr(geometry, field) for geometry in window)


def verify(records: Sequence[Record], steps: Sequence[dict]) -> list[Check]:
    """Re-derives each window's motion from signals the press did not use."""
    measurable = [record.geometry for record in records if record.geometry is not None]
    if not measurable:
        return [("all steps", False, "no frames with all needed joints valid")]
    standing = statistics.median(geometry.pelvis_y for geometry in measurable)

    checks: list[Check] = []
    for step in steps:
        window = [
            record.geometry
            for record in records[step["start_frame"] : step["end_frame"]]
            if record.geometry is not None
        ]
        label = step["label"]
        if not window:
            checks.append((label, False, "no measurable frames in the window"))
            continue

        separation = _mean(window, "separation")
        left_hand = _mean(window, "left_hand_y")
        right_hand = _mean(window, "right_hand_y")
        left_ankle = _mean(window, "left_ankle_y")
        right_ankle = _mean(window, "right_ankle_y")
        dip = standing - min(geometry.pelvis_y for geometry in window)

        if label in ("t_pose_hold_open", "t_pose_hold_close"):
            checks.append(
                (label, separation > 1.30, f"hands {separation * 100:.0f} cm apart")
            )
        elif label == "left_arm_raise":
            checks.append(
                (
                    label,
                    left_hand > right_hand + 0.25,
                    f"left hand {left_hand:.2f} m vs right {right_hand:.2f} m",
                )
            )
        elif label == "right_arm_raise":
            checks.append(
                (
                    label,
                    right_hand > left_hand + 0.25,
                    f"right hand {right_hand:.2f} m vs left {left_hand:.2f} m",
                )
            )
        elif label == "left_leg_raise":
            checks.append(
                (
                    label,
                    left_ankle > right_ankle + 0.04,
                    f"left ankle {left_ankle:+.3f} m vs right {right_ankle:+.3f} m",
                )
            )
        elif label == "right_leg_raise":
            checks.append(
                (
                    label,
                    right_ankle > left_ankle + 0.04,
                    f"right ankle {right_ankle:+.3f} m vs left {left_ankle:+.3f} m",
                )
            )
        elif label == "squat_x2":
            checks.append((label, dip > 0.15, f"pelvis drops {dip * 100:.0f} cm"))
    return checks


def write(
    recording: Path, windows: Sequence[Window]
) -> tuple[Path | None, list[Check]]:
    """Reads the take back and writes ``<stem>.labels.json`` beside it.

    Returns the sidecar and every check run on the way; a sidecar with no steps is not written.
    """
    records = read(recording)
    checks = audit(records, windows)
    steps = lay_out(records, windows)
    checks.append(
        (
            "windows labelled",
            len(steps) == len(windows),
            f"{len(steps)} of {len(windows)} resolved to timestamps",
        )
    )
    checks.extend(verify(records, steps))
    if not steps:
        return (None, checks)

    stamped = [record.sample_ns for record in records if record.sample_ns is not None]
    span_s = (stamped[-1] - stamped[0]) / 1e9 if len(stamped) > 1 else 0.0

    payload = {
        "note": NOTE,
        "clock_domain": CLOCK_DOMAIN,
        "nominal_rate_hz": round(len(stamped) / span_s, 1) if span_s > 0 else None,
        "records": len(records),
        "steps": steps,
        "mcap": recording.name,
    }
    sidecar = recording.with_name(recording.stem + ".labels.json")
    sidecar.write_text(json.dumps(payload, indent=2) + "\n")
    return (sidecar, checks)
