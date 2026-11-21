# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""YAML -> SourceEntry parsing and validation."""

from __future__ import annotations

import pytest

from isaaccapture_examples.camera_viz import config


def test_shape_config_carries_the_equirect_heading():
    shape, _, _, _, yaw = config._shape_for(
        "sky", {"sky": {"shape": "equirect", "equirect_yaw_deg": -90.0}}
    )
    assert (shape, yaw) == ("equirect", -90.0)
    assert config._shape_for("cam", {})[4] == 0.0


def test_a_curved_shape_is_refused_before_the_runtime_starts():
    """Also pins the shape-config tuple against its callers: an extra field
    once slipped past a positional unpack here."""
    cfg = {
        "cameras": [{"name": "sky", "enabled": True}],
        "display": {"placements": {"sky": {"shape": "equirect"}}},
    }
    config.check_shapes_are_displayable(cfg, "xr")
    with pytest.raises(SystemExit, match="requires XR mode"):
        config.check_shapes_are_displayable(cfg, "window")
