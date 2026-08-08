# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Identify the headset behind a streaming runtime from its interaction profile.

A remote-streaming runtime presents every headset through the same OpenXR
surface, so system and device-profile fields do not distinguish them: CloudXR
reports ``NV_DEVICE_PROFILE=Quest3`` with a PICO connected. The interaction
profile the runtime binds for the controllers does track the real hardware,
measured on a PICO 4 Ultra and a Quest 3.

Callers that drive hardware from this should treat ``None`` as "ask the
operator", never as a default: applying a skeleton correction meant for the
other headset feeds a robot permuted joint axes.
"""

from typing import Optional

# OpenXR interaction profile -> headset key.
#
# bytedance/pico4_controller is a positive match: only PICO hardware binds it.
# oculus/touch_controller is identification by elimination -- it is the generic
# Meta-compatible fallback, and a Quest 3 binds it when no Meta-specific profile
# is suggested. Anything else stays unknown rather than guessing.
HEADSET_BY_INTERACTION_PROFILE = {
    "/interaction_profiles/bytedance/pico4_controller": "pico",
    "/interaction_profiles/bytedance/pico_neo3_controller": "pico",
    "/interaction_profiles/bytedance/pico_g3_controller": "pico",
    "/interaction_profiles/meta/touch_controller_plus": "quest",
    "/interaction_profiles/meta/touch_controller_quest_2": "quest",
    "/interaction_profiles/facebook/touch_controller_pro": "quest",
    "/interaction_profiles/oculus/touch_controller": "quest",
}


def identify_headset(interaction_profile: Optional[str]) -> Optional[str]:
    """
    Map an OpenXR interaction profile to a headset key.

    Args:
        interaction_profile: Profile path from
            :meth:`ControllerTracker.get_interaction_profile`. Empty or ``None``
            means the runtime has not bound one yet.

    Returns:
        ``"pico"``, ``"quest"``, or ``None`` when the headset cannot be
        determined. ``None`` is a normal early-session state: nothing is bound
        until actions have synced at least once.
    """
    if not interaction_profile:
        return None
    return HEADSET_BY_INTERACTION_PROFILE.get(interaction_profile)
