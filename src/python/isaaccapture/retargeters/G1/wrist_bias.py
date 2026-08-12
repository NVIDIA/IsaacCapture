# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Per-headset wrist bias for the G1, in commanded joint space.

Companion to the skeleton orientation correction in
``retargeting_engine.utilities.full_body_transform``. That correction puts a
Quest skeleton onto the ByteDance convention; this covers what it cannot. A
correct skeleton still leaves the G1's wrists visibly off-neutral, because the
residual is in the mapping from wrist orientation to G1 joint commands rather
than in the skeleton.

Lives under ``retargeters/G1`` rather than beside the skeleton correction
because these are **G1 joint angles**, not skeleton-space rotations: they are
specific to this robot, while the skeleton correction is specific to the
headset.

Applies to a consumer that derives G1 wrist roll/pitch/yaw from the retargeted
arm chain and adds these as a constant offset. The values encode that
consumer's sign conventions, so a retargeter that computes wrist angles
differently needs its own fit rather than these numbers.
"""

from typing import Dict, Tuple

Bias = Tuple[Tuple[float, float, float], Tuple[float, float, float]]

# Radians, added to the commanded G1 wrist joints as (roll, pitch, yaw) per side.
#
# Measured at `01_wrist_neutral` -- forearms forward, elbows 90, wrists straight
# and relaxed -- where all three channels should read the same on both headsets.
# They did not:
#
#     channel     Pico     Quest    bias applied
#     L roll     -33.6     -24.5       +24.24   (zeroed, not matched)
#     L pitch     +7.2      +7.1       +11.44
#     L yaw      +15.2     +44.1       -29.88
#     R roll     +22.3     +19.7       -19.47   (zeroed, not matched)
#     R pitch     +9.4      +9.4       +16.87
#     R yaw      -19.9     -48.2       +28.13
#
# ROLL is the exception: it is driven to ZERO rather than matched to the Pico.
# The Pico commands -33.6 / +22.2 deg of wrist roll at a neutral wrist, which
# renders as visibly rolled palms, and it is asymmetric -- the left sits 11.4 deg
# further round than the right. An operator sees exactly that: both palms rolled
# up, the left more. Matching the Pico faithfully reproduced the flaw, so roll
# alone departs from the reference. Because a constant bias shifts the operating
# point without compressing travel, this costs nothing in responsiveness: roll
# span across the wrist sweep is 91 / 86 deg either way.
#
# Caveat: zero commanded roll is a HYPOTHESIS about where the robot's neutral
# lies, not a measurement. There is no ground truth here for "thumbs facing each
# other" in joint terms. If a trial still shows residual roll, adjust these two
# numbers by the observed amount -- the rest of the table does not depend on them.
#
# YAW carried the visible error, not pitch. An earlier pitch-only bias was
# already exact (delta -0.0 and +0.1 deg) yet the robot's hands still sat
# visibly extended at neutral, because roughly 29 deg of error per side was
# arriving through the yaw channel. That is consistent with operator wrist
# flexion largely landing in the yaw channel rather than pitch.
#
# Pico is zero by intent, not oversight. Its teleop has been tuned by people
# over a long time, and its non-zero neutral may be correct for the robot's
# mechanical neutral rather than an error; this data cannot distinguish those.
#
# This is a BIAS at neutral. It does not change how much the wrist travels; the
# usable range is limited separately, in shared retargeting code.
WRIST_BIAS_RAD: Dict[str, Bias] = {
    # (roll, pitch, yaw) per side
    "pico": ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),  # intentionally untouched
    "quest": (
        (+0.423036, +0.199678, -0.521456),
        (-0.339744, +0.294511, +0.490959),
    ),
}

_ZERO: Bias = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))


def wrist_bias_for(profile: str) -> Bias:
    """
    Wrist bias for a headset, as ``((l_roll, l_pitch, l_yaw), (r_...))`` radians.

    Args:
        profile: Source headset, matching the skeleton profiles in
            ``full_body_transform`` (``"pico"`` / ``"quest"``).

    Returns:
        The bias per side. Unknown profiles return zeros rather than raising:
        an unbiased wrist is the pre-existing behaviour and is safe, whereas
        applying another headset's bias is not.
    """
    return WRIST_BIAS_RAD.get(profile, _ZERO)
