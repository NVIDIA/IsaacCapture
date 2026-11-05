# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""What the left stick does, per surface shape.

One class per shape, in the same spirit as ``placements.lock_modes``: the two
axis labels live next to the behaviour they describe, so adding a shape means
adding a class rather than editing an if/elif and a separate label table.

The rule every shape follows: **the two axes must not collapse into one
control.** Apparent size is ``2*atan((w/2)/d)`` and a cylinder's arc width is
``radius * angle``, so moving either surface further away enlarges it by the
same factor and looks identical to resizing it. Distance and radius are
therefore YAML placement choices, not stick axes -- they would be a second
copy of the size axis. What they *do* change is the real distance to the
surface, which :mod:`stereo` works from.
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import List, Optional, Tuple

from ..placements import yaw_quat


def _clamp(value: float, limits: Tuple[float, float]) -> float:
    low, high = limits
    return min(max(value, low), high)


def retune(target, strategy, **updates) -> None:
    """Swap in a modified :class:`PlacementConfig`.

    It is frozen, so retuning means replacing it and handing the replacement
    to the live strategy. Rebuilding the strategy instead would drop its lazy
    anchor and re-snap the plane on every nudge.
    """
    target.placement_config = replace(target.placement_config, **updates)
    if strategy is not None:
        strategy.retune(target.placement_config)


class ShapeControl:
    """Maps the left stick's two axes onto one shape's parameters."""

    #: (X label, Y label), shown when the shape is selected.
    AXES: Tuple[str, str] = ("", "")

    def adjust(
        self, target, cfg, strategy, ax: float, ay: float, dt: float
    ) -> List[str]:
        """Apply one frame of stick input; return a description per change."""
        raise NotImplementedError

    # Height is common to the flat and curved surfaces that have a placement.
    @staticmethod
    def _adjust_height(target, cfg, strategy, ay: float, dt: float) -> List[str]:
        placement = target.placement_config
        if ay == 0.0 or placement is None:
            return []
        offset = _clamp(
            placement.offset_y + ay * cfg.offset_rate_m_per_s * dt, cfg.offset_y_range_m
        )
        if offset == placement.offset_y:
            return []
        retune(target, strategy, offset_y=offset)
        return [f"height: {offset:+.2f} m"]


class QuadControl(ShapeControl):
    AXES = ("size", "height")

    def adjust(self, target, cfg, strategy, ax, ay, dt) -> List[str]:
        placement = target.placement_config
        if placement is None:
            return []
        out: List[str] = []
        if ax != 0.0:
            width, height = placement.size_meters
            new_width = _clamp(
                width + ax * cfg.size_rate_m_per_s * dt, cfg.size_range_m
            )
            if new_width != width:
                # Aspect preserved: the source's pixels can't be re-shaped
                # after the fact, so overall scale is the only sane knob.
                retune(
                    target,
                    strategy,
                    size_meters=(new_width, new_width * height / width),
                )
                out.append(f"size: {new_width:.2f} m")
        return out + self._adjust_height(target, cfg, strategy, ay, dt)


class CylinderControl(ShapeControl):
    AXES = ("arc", "height")

    def adjust(self, target, cfg, strategy, ax, ay, dt) -> List[str]:
        out: List[str] = []
        angle = _clamp(
            target.cylinder_angle_deg + ax * cfg.angle_rate_deg_per_s * dt,
            cfg.cylinder_angle_range_deg,
        )
        if angle != target.cylinder_angle_deg:
            target.cylinder_angle_deg = angle
            apply_cylinder(target)
            out.append(f"arc: {angle:.0f}°")
        return out + self._adjust_height(target, cfg, strategy, ay, dt)


class EquirectControl(ShapeControl):
    AXES = ("h-span", "v-span")

    def adjust(self, target, cfg, strategy, ax, ay, dt) -> List[str]:
        horizontal = _clamp(
            target.equirect_h_deg + ax * cfg.angle_rate_deg_per_s * dt,
            cfg.equirect_h_range_deg,
        )
        vertical = _clamp(
            target.equirect_v_half_deg + ay * cfg.angle_rate_deg_per_s * dt,
            cfg.equirect_v_half_range_deg,
        )
        if (horizontal, vertical) == (
            target.equirect_h_deg,
            target.equirect_v_half_deg,
        ):
            return []
        target.equirect_h_deg, target.equirect_v_half_deg = horizontal, vertical
        apply_equirect(target)
        return [f"span: {horizontal:.0f}° x {2 * vertical:.0f}°"]


_CONTROLS = {
    "quad": QuadControl(),
    "cylinder": CylinderControl(),
    "equirect": EquirectControl(),
}


def for_shape(shape: str) -> Optional[ShapeControl]:
    return _CONTROLS.get(shape)


def axes(shape: str) -> Optional[Tuple[str, str]]:
    control = _CONTROLS.get(shape)
    return control.AXES if control is not None else None


# ── Pushing a target's shaped params onto its layer ───────────────────
#
# Read-modify-write in both cases: the runner rewrites only ``.pose`` each
# frame, so these survive it; writing a fresh placement here would race it.


def apply_cylinder(target) -> None:
    layer = target.shape_layers.get("cylinder", target.layer)
    if layer is None:
        return
    placement = layer.placement()
    placement.radius_m = target.cylinder_radius_m
    placement.central_angle_rad = math.radians(target.cylinder_angle_deg)
    layer.set_placement(placement)


def apply_equirect(target) -> None:
    layer = target.shape_layers.get("equirect", target.layer)
    if layer is None:
        return
    placement = layer.placement()
    # The texture's horizontal center maps to the pose's -z, so yawing the
    # pose is what aims the middle of the panorama. This is the sphere's only
    # useful pose knob: its radius is infinite, where translation does
    # nothing. Vertically the center sits on the pose's horizon.
    placement.pose.orientation = yaw_quat(math.radians(target.equirect_yaw_deg))
    placement.central_horizontal_angle_rad = math.radians(target.equirect_h_deg)
    # Kept symmetric about the horizon, which also keeps upper > lower.
    placement.upper_vertical_angle_rad = math.radians(target.equirect_v_half_deg)
    placement.lower_vertical_angle_rad = -math.radians(target.equirect_v_half_deg)
    layer.set_placement(placement)


def apply_all(target) -> None:
    """Push every shaped param onto its layer (used by the reset button)."""
    if "cylinder" in target.shape_layers or target.shape == "cylinder":
        apply_cylinder(target)
    if "equirect" in target.shape_layers or target.shape == "equirect":
        apply_equirect(target)


__all__ = [
    "CylinderControl",
    "EquirectControl",
    "QuadControl",
    "ShapeControl",
    "apply_all",
    "apply_cylinder",
    "apply_equirect",
    "axes",
    "for_shape",
    "retune",
]
