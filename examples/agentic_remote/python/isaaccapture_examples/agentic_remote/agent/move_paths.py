# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The planned joint path of each MOVE, for the run folder and the debug recording.

The agent plans a move's setpoints on the workstation and sends them to the
robot app in one MOVE. The arm adapter hands the MOVE's own arrays to the
launcher's Measures when the MOVE has left, and how the move ended when it has
ended. Measures' thread makes one line of MOVE_PATHS_FILE per MOVE and, with
the debug recording, one PATH_EVENT, with the functions of this module.

Nothing here runs on a thread that drives the arm. The setpoints are made into
text and into one array CHUNK at a time with a pause between, so that the
thread which does it never holds the interpreter for long against the thread
that sends the PERMITs.

numpy is imported inside points_array(), so the package imports without it.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, is_dataclass
from typing import Any

# One line per MOVE the agent sent or could not send.
MOVE_PATHS_FILE = "move-paths.jsonl"
# The event of the debug recording with one MOVE's whole path.
PATH_EVENT = "robot.move_path"
PATH_NOTE = (
    "The PLANNED joint path of one MOVE: every setpoint the agent sent to the "
    "robot app in that one link message, in order, period_s apart. These are "
    "no single targets timed at the motors, which the experiment records as "
    "robot.call of send_joint_target: the robot app sends them on the Thor. "
    "How many of them it sent is move_index in robot.state; how the move "
    "ended is in move-paths.jsonl of the run folder."
)
# Setpoints made into text at one go, about 0.1 ms of work, and the pause
# behind them, in s.
CHUNK = 16
PAUSE_S = 1e-4
# What the adapter hands over of a MOVE, and what a line names it.
_AS_SENT = (
    "session_id",
    "owner_number",
    "stop_number",
    "period_s",
    "start_tol_rad",
    "evidence_wall_s",
    "expires_in_s",
    "skill",
    "trace_step",
)
# What it hands over of the move's end.
_ENDING = ("status", "sent", "reason", "outcome", "sent_seen", "error")


def _floats(row: Any) -> list[float]:
    """A pose as the link sends it: a list of floats, each value whole."""
    if row is None:
        return []
    values = row.tolist() if hasattr(row, "tolist") else row
    return [float(value) for value in values]


def _points(facts: dict[str, Any]) -> Sequence[Any]:
    """The setpoints the adapter handed over, as it holds them."""
    setpoints = facts.get("setpoints")
    return () if setpoints is None else setpoints


def _region(region: Any) -> dict[str, Any] | None:
    """The MOVE's grasp region as a dict; None when it carried none."""
    if region is None:
        return None
    fields = asdict(region) if is_dataclass(region) else dict(region)
    return {
        name: _floats(value) if isinstance(value, (list, tuple)) else value
        for name, value in fields.items()
    }


def _head(facts: dict[str, Any]) -> dict[str, Any]:
    """What a line and an event say of a MOVE apart from its setpoints."""
    problem = facts.get("not_sent")
    head: dict[str, Any] = {
        "move_id": facts.get("move_id"),
        "sent_mono_s": facts.get("sent_mono_s"),
        "sent": problem is None,
        "not_sent": None if problem is None else str(problem),
    }
    head.update({name: facts.get(name) for name in _AS_SENT})
    head["grasp_region"] = _region(facts.get("grasp_region"))
    head["count"] = len(_points(facts))
    head["start_q"] = _floats(facts.get("start_q"))
    return head


def path_row(facts: dict[str, Any], utc: str | None) -> dict[str, Any]:
    """The line of MOVE_PATHS_FILE for one MOVE, without setpoints and end.

    facts is what the adapter handed over (arm_adapter.MEASURED_PATH); utc is
    its sent_mono_s as a date.
    """
    head = _head(facts)
    return {"move_id": head.pop("move_id"), "utc": utc, **head, "end": None}


def end_row(facts: dict[str, Any], utc: str, mono_s: float) -> dict[str, Any]:
    """What a line says under "end": how the move ended, and when.

    facts is what the adapter handed over (arm_adapter.MEASURED_PATH_END).
    """
    row: dict[str, Any] = {"utc": utc, "mono_s": mono_s}
    row.update({name: facts.get(name) for name in _ENDING})
    end_q = facts.get("end_q")
    row["end_q"] = None if end_q is None else _floats(end_q)
    return row


def points_text(
    facts: dict[str, Any], pause: Callable[[float], None] = time.sleep
) -> str:
    """The MOVE's setpoints as JSON, each value whole, in their order."""
    setpoints = _points(facts)
    pieces = []
    for at in range(0, len(setpoints), CHUNK):
        if at:
            pause(PAUSE_S)
        rows = [_floats(point) for point in setpoints[at : at + CHUNK]]
        pieces.append(json.dumps(rows)[1:-1])
    return "[" + ", ".join(pieces) + "]"


def line_text(row: dict[str, Any], points: str) -> str:
    """One line of MOVE_PATHS_FILE: the row, then its setpoints."""
    return json.dumps(row, default=str)[:-1] + ', "setpoints": ' + points + "}\n"


def points_array(
    facts: dict[str, Any], pause: Callable[[float], None] = time.sleep
) -> Any:
    """The MOVE's setpoints as one array of float64, a row each."""
    import numpy as np

    setpoints = _points(facts)
    width = len(setpoints[0]) if len(setpoints) else 0
    array = np.empty((len(setpoints), width), dtype=float)
    step = 4 * CHUNK
    for at in range(0, len(setpoints), step):
        if at:
            pause(PAUSE_S)
        array[at : at + step] = setpoints[at : at + step]
    return array


def path_event(facts: dict[str, Any]) -> dict[str, Any]:
    """The data of the PATH_EVENT for one MOVE: the setpoints as one array."""
    return {"what": PATH_NOTE, **_head(facts), "setpoints": points_array(facts)}
