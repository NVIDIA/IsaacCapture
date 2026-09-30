# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The hand script as performed, shared by the panel and the labeller.

Both hands perform every step together. The order and labels are the checker's
``SCRIPT``; this adds how long each is held and what the performer hears.
"""

from __future__ import annotations

from hand_acceptance.labels import SCRIPT, STILL_LABELS

# (label, hold_s, spoken cue), in performed order. The move into each pose happens
# during the cue and the 3-2-1 countdown, so a hold covers the pose only.
STEPS: list[tuple[str, float, str]] = [
    ("open_hand_open", 4.0, "Both hands open, palms down."),
    ("fist", 4.0, "Make two tight fists."),
    ("right_grasps_left_fist", 4.0, "Left fist. Right hand grips it."),
    ("left_grasps_right_fist", 4.0, "Right fist. Left hand grips it."),
    ("pinch_index", 3.0, "Thumb to index fingertip."),
    ("pinch_middle", 3.0, "Thumb to middle fingertip."),
    ("pinch_ring", 3.0, "Thumb to ring fingertip."),
    ("pinch_little", 3.0, "Thumb to little fingertip."),
    ("wrist_rotate", 6.0, "Open hands. Turn palms up, then back down."),
    ("open_hand_close", 4.0, "Open hands again, palms down."),
]

COUNTDOWN_BEATS = 3
BEAT_S = 1.0

assert tuple(label for label, _, _ in STEPS) == SCRIPT, "session and checker disagree"

__all__ = ["BEAT_S", "COUNTDOWN_BEATS", "STEPS", "STILL_LABELS"]
