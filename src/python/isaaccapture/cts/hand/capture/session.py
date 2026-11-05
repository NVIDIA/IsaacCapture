# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The hand script as performed: hold time and spoken cue for each of the checker's steps.

Both hands perform every step together, except the two rolls, where one hand turns
against the other's still fingertip.
"""

from __future__ import annotations

from hand_cts.labels import SCRIPT, STILL_LABELS

# (label, hold_s, spoken cue), in performed order; hold_s covers the pose, not the move into it.
STEPS: list[tuple[str, float, str]] = [
    (
        "flat_on_table_open",
        4.0,
        "Rest both hands flat on the table, palms down, fingers together.",
    ),
    ("fist", 4.0, "Make two tight fists."),
    ("pinch_index", 3.0, "Thumb to index fingertip."),
    ("pinch_middle", 3.0, "Thumb to middle fingertip."),
    ("pinch_ring", 3.0, "Thumb to ring fingertip."),
    ("pinch_little", 3.0, "Thumb to little fingertip."),
    ("palms_together", 4.0, "Press your palms together, fingers up."),
    (
        "tips_together",
        3.0,
        "Raise both hands, palms facing you. Touch middle fingertips.",
    ),
    (
        "right_tip_roll",
        6.0,
        "Fingertips touching. At the tone, turn the right palm out, then back.",
    ),
    (
        "left_tip_roll",
        6.0,
        "Fingertips touching. At the tone, turn the left palm out, then back.",
    ),
    ("flat_on_table_close", 4.0, "Hands flat on the table again."),
]

COUNTDOWN_BEATS = 3
BEAT_S = 1.0

assert tuple(label for label, _, _ in STEPS) == SCRIPT, "session and checker disagree"

__all__ = ["BEAT_S", "COUNTDOWN_BEATS", "STEPS", "STILL_LABELS"]
