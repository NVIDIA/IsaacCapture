# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The motion script as performed by a human, shared by the panel and the labeller."""

from __future__ import annotations

# (label, duration_s, spoken cue), in performed order.
# A duration covers the motion only: the performer opens each window already in the pose.
STEPS: list[tuple[str, float, str]] = [
    ("a_pose_still", 5.0, "Arms down. Stand still."),
    ("t_pose_hold_open", 6.0, "T pose. Hold."),
    ("neutral_stance", 7.0, "Arms down. Relax."),
    ("left_arm_raise", 5.0, "Left arm up, overhead."),
    ("right_arm_raise", 5.0, "Right arm up, overhead."),
    ("left_leg_raise", 5.0, "Left knee up."),
    ("right_leg_raise", 5.0, "Right knee up."),
    ("squat_x2", 10.0, "Squat twice, slowly."),
    ("march_in_place", 6.0, "March in place."),
    ("t_pose_hold_close", 6.0, "T pose again. Hold."),
]

STILL_LABELS = frozenset(
    {"a_pose_still", "t_pose_hold_open", "neutral_stance", "t_pose_hold_close"}
)
