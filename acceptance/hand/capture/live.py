# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Both hands out of a running ``TeleopSession``, as the checker's ``Frame``.

Live frames feed the panel only. The recorder writes every update on both channels,
active or not, and the labels are stamped on the session clock, so nothing here counts
records.
"""

from __future__ import annotations

from typing import Any

from isaacteleop.retargeting_engine.deviceio_source_nodes import (
    ControllersSource,
    HandsSource,
)
from isaacteleop.retargeting_engine.interface import OutputCombiner

from hand_acceptance.frames import Frame, JointPose


def build_pipeline() -> tuple[OutputCombiner, HandsSource]:
    """The two hand channels, and the controllers beside them.

    The controllers are recorded for the wrist-anchor comparison: gloves that fall
    back to the controller for the wrist are placed by it.
    """
    controllers = ControllersSource(name="controllers")
    hands = HandsSource(name="hands")
    pipeline = OutputCombiner(
        {
            "controller_left": controllers.output(ControllersSource.LEFT),
            "controller_right": controllers.output(ControllersSource.RIGHT),
            "hand_left": hands.output(HandsSource.LEFT),
            "hand_right": hands.output(HandsSource.RIGHT),
        }
    )
    return pipeline, hands


class LiveHands:
    def __init__(self, session: Any, hands: HandsSource) -> None:
        self._session = session
        self._tracker = hands.get_tracker()
        self._steps = 0

    def step(self) -> dict[str, Frame | None]:
        self._session.step()
        device = self._session.deviceio_session
        frames = {
            "left": _frame(self._steps, "left", self._tracker.get_left_hand(device)),
            "right": _frame(self._steps, "right", self._tracker.get_right_hand(device)),
        }
        self._steps += 1
        return frames


def _frame(sequence: int, side: str, pose: Any) -> Frame | None:
    """None for an inactive hand. Time fields stay empty: a live pose carries none."""
    if pose is None:
        return None
    joints = pose.joints
    decoded = tuple(
        JointPose(
            position=(float(p[0]), float(p[1]), float(p[2])),
            orientation=(float(q[0]), float(q[1]), float(q[2]), float(q[3])),
            is_valid=bool(valid),
            radius=float(r),
        )
        for p, q, valid, r in zip(
            joints.positions, joints.orientations, joints.is_valid, joints.radii
        )
    )
    return Frame(
        sequence=sequence,
        log_time_ns=0,
        publish_time_ns=0,
        has_payload=True,
        side=side,
        joints=decoded,
    )
