# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""World / head / lazy / gimbal locked placement strategies."""

from __future__ import annotations

import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple

from ._math import (
    Quat,
    Vec3,
    angle_between_xz_deg,
    normalize_angle,
    project_forward_xz,
    rotate_vec,
    smoothstep,
    yaw_quat,
)


@dataclass(frozen=True)
class PlacementConfig:
    """Per-plane placement tuning.

    ``size_meters`` is plane width/height in world units. The lazy-mode
    fields are no-ops for World / Head locks.
    """

    size_meters: Tuple[float, float] = (1.0, 0.5625)
    distance: float = 1.5
    offset_x: float = 0.0
    offset_y: float = 0.0
    look_away_angle_deg: float = 45.0
    reposition_distance: float = 0.5
    reposition_delay_s: float = 0.5
    transition_duration_s: float = 0.3


@dataclass
class Placement:
    """Output of a strategy.

    ``position`` / ``orientation`` / ``size_meters`` feed straight into
    ``viz.QuadLayerPlacement``. ``anchor_position`` / ``anchor_orientation``
    are the head-anchored pose ``distance`` behind the plane, oriented so
    local −z points at the plane — the pose a curved surface centered on
    the viewer (``viz.CylinderLayerPlacement``) should use. Each strategy
    computes the anchor itself because the facing conventions differ
    (world/lazy/gimbal build yaw-only quats whose +z faces the head;
    head-locked uses the full head orientation, pitch and roll included).
    """

    position: Vec3
    orientation: Quat  # (w, x, y, z)
    size_meters: Tuple[float, float]
    distance: float = 1.5
    anchor_position: Vec3 = (0.0, 0.0, 0.0)
    anchor_orientation: Quat = (1.0, 0.0, 0.0, 0.0)


class PlacementStrategy(ABC):
    """Per-layer placement policy. ``update`` is called from the render
    thread each frame; implementations must be cheap and pure-CPU."""

    @abstractmethod
    def update(self, head_pos: Vec3, head_orientation: Quat) -> Placement: ...

    def retune(self, config: PlacementConfig) -> None:
        """Swap the tuning without resetting the strategy's live state.

        ``PlacementConfig`` is frozen, so a caller adjusting size or
        distance at runtime builds a replacement with ``dataclasses.replace``
        and hands it over here. Rebuilding the strategy instead would drop
        the lazy anchor and re-snap the plane on every change.
        """
        self._config = config


def _target_position(head_pos: Vec3, forward_xz: Vec3, cfg: PlacementConfig) -> Vec3:
    """Place the quad ``distance`` ahead, with right-vector / up-vector offsets.
    Mirrors ``CameraPlane::compute_target_position``."""
    right_x = -forward_xz[2]
    right_z = forward_xz[0]
    return (
        head_pos[0] + forward_xz[0] * cfg.distance + right_x * cfg.offset_x,
        head_pos[1] + cfg.offset_y,
        head_pos[2] + forward_xz[2] * cfg.distance + right_z * cfg.offset_x,
    )


def _shift_y(position: Vec3, delta: float) -> Vec3:
    return (position[0], position[1] + delta, position[2])


def _yaw_to_face(target: Vec3, plane_pos: Vec3) -> float:
    """Yaw rotation that aims the plane's front at ``target``.
    Mirrors ``CameraPlane::compute_yaw_to_face``."""
    return math.atan2(target[0] - plane_pos[0], target[2] - plane_pos[2])


class WorldLocked(PlacementStrategy):
    """Place once in front of the user; never move thereafter.
    Mirrors ``CameraPlane::update_world``."""

    def __init__(self, config: PlacementConfig) -> None:
        self._config = config
        # The head pose the plane was placed around, frozen on first update.
        # Everything else is recomputed from it each frame so a retune (size,
        # height) takes effect at once; caching the finished Placement instead
        # made this mode ignore retuning entirely.
        self._anchor_head: Optional[Vec3] = None
        self._anchor_forward_xz: Optional[Vec3] = None

    def update(self, head_pos: Vec3, head_orientation: Quat) -> Placement:
        if self._anchor_head is None:
            self._anchor_head = head_pos
            self._anchor_forward_xz = project_forward_xz(head_orientation)

        position = _target_position(
            self._anchor_head, self._anchor_forward_xz, self._config
        )
        orientation = yaw_quat(_yaw_to_face(self._anchor_head, position))
        # Anchor = the head point the plane was placed around: push the plane
        # back along its local +z (which faces the head).
        back = rotate_vec(orientation, (0.0, 0.0, 1.0))
        anchor = tuple(position[i] + back[i] * self._config.distance for i in range(3))
        return Placement(
            position,
            orientation,
            self._config.size_meters,
            self._config.distance,
            anchor,
            orientation,
        )


class HeadLocked(PlacementStrategy):
    """Follow the head every frame, full 6-DoF (pitch and roll included,
    unlike the yaw-only world / gimbal / lazy modes)."""

    def __init__(self, config: PlacementConfig) -> None:
        self._config = config

    def update(self, head_pos: Vec3, head_orientation: Quat) -> Placement:
        forward = rotate_vec(head_orientation, (0.0, 0.0, -1.0))
        right = rotate_vec(head_orientation, (1.0, 0.0, 0.0))
        up = rotate_vec(head_orientation, (0.0, 1.0, 0.0))
        d, ox, oy = self._config.distance, self._config.offset_x, self._config.offset_y
        position = (
            head_pos[0] + forward[0] * d + right[0] * ox + up[0] * oy,
            head_pos[1] + forward[1] * d + right[1] * ox + up[1] * oy,
            head_pos[2] + forward[2] * d + right[2] * ox + up[2] * oy,
        )
        # No flip: every mode returns the rotation that faces the plane back
        # at the viewer, which is identity for a level head. Rotating by 180
        # deg here pointed the quad away, and an OpenXR quad layer is
        # single-sided, so the feed went black.
        orientation = head_orientation
        # Anchor = the head itself (plus the configured offsets, already
        # baked into ``position``): pull the plane back by ``distance``.
        anchor = (
            position[0] - forward[0] * d,
            position[1] - forward[1] * d,
            position[2] - forward[2] * d,
        )
        return Placement(
            position,
            orientation,
            self._config.size_meters,
            self._config.distance,
            anchor,
            head_orientation,
        )


class GimbalLocked(PlacementStrategy):
    """Translation head-locked, rotation world-locked: the surface follows
    your position every frame but keeps the yaw captured at first sight —
    walk and it comes with you, turn your head and you look around it.
    The "virtual gimbal" mode for wide cylinder feeds."""

    def __init__(self, config: PlacementConfig) -> None:
        self._config = config
        self._forward_xz: Optional[Vec3] = None
        self._yaw = 0.0

    def update(self, head_pos: Vec3, head_orientation: Quat) -> Placement:
        if self._forward_xz is None:
            # Capture the world-locked heading once: face where the user
            # is looking at first update.
            self._forward_xz = project_forward_xz(head_orientation)
            probe = _target_position(head_pos, self._forward_xz, self._config)
            self._yaw = _yaw_to_face(head_pos, probe)
        position = _target_position(head_pos, self._forward_xz, self._config)
        orientation = yaw_quat(self._yaw)
        # Anchor = the head itself (plus offsets, already baked into
        # ``position``): pull back along the fixed heading.
        anchor = tuple(
            position[i] - self._forward_xz[i] * self._config.distance for i in range(3)
        )
        return Placement(
            position,
            orientation,
            self._config.size_meters,
            self._config.distance,
            anchor,
            orientation,
        )


class LazyLocked(PlacementStrategy):
    """World-locked, but smoothly re-snaps in front of the user when they
    look away (or drift) past a threshold for ``reposition_delay_s``.

    Mirrors ``CameraPlane::update_lazy`` + ``::update_transition``.
    """

    def __init__(self, config: PlacementConfig) -> None:
        self._config = config
        self._initialized = False
        self._position: Vec3 = (0.0, 0.0, 0.0)
        self._yaw: float = 0.0
        self._is_looking_away = False
        self._look_away_start_t = 0.0
        self._is_transitioning = False
        self._transition_start_t = 0.0
        self._transition_start_position: Vec3 = (0.0, 0.0, 0.0)
        self._transition_start_yaw = 0.0
        self._target_position: Vec3 = (0.0, 0.0, 0.0)
        self._target_yaw = 0.0

    def retune(self, config: PlacementConfig) -> None:
        """Carry a height change onto the position already placed.

        Unlike the other modes this one cannot recompute from an anchor -- its
        position is wherever the last re-snap and transition left it. Without
        this, a height change sat unapplied until the next re-snap, which is
        the whole point of lazy mode not happening.
        """
        delta_y = config.offset_y - self._config.offset_y
        super().retune(config)
        if delta_y:
            self._position = _shift_y(self._position, delta_y)
            self._target_position = _shift_y(self._target_position, delta_y)
            self._transition_start_position = _shift_y(
                self._transition_start_position, delta_y
            )

    def update(self, head_pos: Vec3, head_orientation: Quat) -> Placement:
        now = time.monotonic()
        forward_xz = project_forward_xz(head_orientation)

        if not self._initialized:
            self._position = _target_position(head_pos, forward_xz, self._config)
            self._target_position = self._position
            self._yaw = _yaw_to_face(head_pos, self._position)
            self._target_yaw = self._yaw
            self._initialized = True
            return self._placement()

        # Look-away check: angle between head forward and head→plane vector.
        head_to_plane = (
            self._position[0] - head_pos[0],
            0.0,
            self._position[2] - head_pos[2],
        )
        angle = angle_between_xz_deg(forward_xz, head_to_plane)
        angle_triggered = angle > self._config.look_away_angle_deg

        # Position drift check: user has moved far from the ideal placement.
        ideal = _target_position(head_pos, forward_xz, self._config)
        drift = math.sqrt(sum((self._position[i] - ideal[i]) ** 2 for i in range(3)))
        position_triggered = (
            self._config.reposition_distance > 0.0
            and drift > self._config.reposition_distance
        )

        if angle_triggered or position_triggered:
            if not self._is_looking_away:
                self._is_looking_away = True
                self._look_away_start_t = now
            elif not self._is_transitioning:
                if (now - self._look_away_start_t) >= self._config.reposition_delay_s:
                    self._target_position = ideal
                    self._target_yaw = _yaw_to_face(head_pos, self._target_position)
                    self._transition_start_position = self._position
                    self._transition_start_yaw = self._yaw
                    self._transition_start_t = now
                    self._is_transitioning = True
        else:
            self._is_looking_away = False

        if self._is_transitioning:
            dur = self._config.transition_duration_s
            t = min((now - self._transition_start_t) / dur, 1.0) if dur > 0.0 else 1.0
            s = smoothstep(t)
            self._position = tuple(
                self._transition_start_position[i]
                + (self._target_position[i] - self._transition_start_position[i]) * s
                for i in range(3)
            )  # type: ignore[assignment]
            yaw_diff = normalize_angle(self._target_yaw - self._transition_start_yaw)
            self._yaw = self._transition_start_yaw + yaw_diff * s
            if t >= 1.0:
                self._is_transitioning = False
                self._is_looking_away = False

        return self._placement()

    def _placement(self) -> Placement:
        """Current (possibly mid-transition) pose as a Placement. The anchor
        glides with the smoothed plane pose so a cylinder re-snaps along the
        same eased path a quad does."""
        orientation = yaw_quat(self._yaw)
        back = rotate_vec(orientation, (0.0, 0.0, 1.0))
        anchor = tuple(
            self._position[i] + back[i] * self._config.distance for i in range(3)
        )
        return Placement(
            self._position,
            orientation,
            self._config.size_meters,
            self._config.distance,
            anchor,
            orientation,
        )


def build(lock_mode: str, config: PlacementConfig) -> PlacementStrategy:
    """Factory used by the YAML loader.

    ``"world"`` → WorldLocked, ``"head"`` → HeadLocked, ``"gimbal"`` →
    GimbalLocked (translation follows the head, rotation stays
    world-locked), anything else (including ``"lazy"``) → LazyLocked.
    """
    if lock_mode == "world":
        return WorldLocked(config)
    if lock_mode == "head":
        return HeadLocked(config)
    if lock_mode == "gimbal":
        return GimbalLocked(config)
    return LazyLocked(config)
