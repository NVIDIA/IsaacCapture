# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Placement strategies for camera_viz.

Strategies are app policy (per the viz design): they live in the
example, not in viz_layers.
"""

# Re-exported: the same yaw maths positions an equirect sphere, which has no
# lock-mode strategy of its own (see controls.shapes).
from ._math import heading_deg, yaw_quat
from .lock_modes import (
    HeadLocked,
    LazyLocked,
    Placement,
    PlacementConfig,
    PlacementStrategy,
    WorldLocked,
    build,
)

__all__ = [
    "HeadLocked",
    "LazyLocked",
    "Placement",
    "PlacementConfig",
    "PlacementStrategy",
    "WorldLocked",
    "build",
    "heading_deg",
    "yaw_quat",
]
