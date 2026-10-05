# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""XR controller bindings for camera_viz.

Four modules behind one door, in the same spirit as :mod:`placements`:

``bindings``
    What each button and stick does, and the state each camera carries.
``shapes``
    The left stick, per surface shape -- one strategy class per shape.
``stereo``
    The gap-to-perceived-distance geometry, as pure functions.
``hud``
    The head-locked panel that shows what a press just did.

Import from the package, not the modules: ``from controls import
ControllerControls``.
"""

from .bindings import (
    DEFAULT_IPD_MM,
    FAR_TARGET_M,
    LOCK_MODE_CYCLE,
    MAX_OFFSET_FRACTION_OF_IPD,
    PLANE_DISTANCE_STEP_CM,
    SHAPE_CYCLE,
    SHAPE_PARAMS,
    ControllerControls,
    ControlsConfig,
    ControlTarget,
    controls_config_from_yaml,
    summarize,
)
from .hud import make_hud, split_message

__all__ = [
    "DEFAULT_IPD_MM",
    "FAR_TARGET_M",
    "LOCK_MODE_CYCLE",
    "MAX_OFFSET_FRACTION_OF_IPD",
    "PLANE_DISTANCE_STEP_CM",
    "SHAPE_CYCLE",
    "SHAPE_PARAMS",
    "ControlTarget",
    "ControllerControls",
    "ControlsConfig",
    "controls_config_from_yaml",
    "make_hud",
    "split_message",
    "summarize",
]
