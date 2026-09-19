# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Building the display side: the VizSession and one layer per surface.

Everything here consumes the parsed :class:`~config.SourceEntry` values and
allocates — a session in the configured display mode, then a layer per shape.
Kept apart from the parsing so the config half stays free of Vulkan.
"""

from __future__ import annotations

import math
from typing import List, Optional

import isaacteleop.viz as viz

from config import SourceEntry, VALID_SHAPES
from placements import yaw_quat

# ImageLayerBase::kSlotCount (kMaxFramesInFlight + 2). Only used to
# report the VRAM that shape switching adds.
_MAILBOX_SLOTS = 7


# display.xr.system_wait_seconds default: block in xrGetSystem until the
# headset shows up. CloudXR brings the runtime up long before the operator
# has the Quest on their head, so failing fast (VizSessionConfig's own
# default of 0) just means losing the race and re-running. Matches
# isaacteleop.viz.robot.session.WAIT_FOR_HEADSET.
#
# The cost: VizSession.create holds the GIL through the wait, so Ctrl-C is
# queued and not delivered until a headset connects. Set this to a positive
# number of seconds (or pass --xr-wait) to get a bounded wait back.
WAIT_FOR_HEADSET = -1


def make_session(
    cfg: dict,
    mode_override: Optional[str] = None,
    required_extensions: Optional[List[str]] = None,
    xr_wait_override: Optional[int] = None,
) -> viz.VizSession:
    display = cfg.get("display", {})
    # --mode overrides display.mode when given.
    mode_str = (mode_override or display.get("mode", "xr")).lower()
    session_cfg = viz.VizSessionConfig()
    if mode_str == "window":
        session_cfg.mode = viz.DisplayMode.kWindow
        w = display.get("window", {})
        session_cfg.window_width = int(w.get("width", 1280))
        session_cfg.window_height = int(w.get("height", 720))
    elif mode_str == "xr":
        session_cfg.mode = viz.DisplayMode.kXr
        x = display.get("xr", {})
        session_cfg.xr_near_z = float(x.get("near_z", 0.05))
        session_cfg.xr_far_z = float(x.get("far_z", 100.0))
        wait = (
            xr_wait_override
            if xr_wait_override is not None
            else int(x.get("system_wait_seconds", WAIT_FOR_HEADSET))
        )
        session_cfg.xr_system_wait_seconds = wait
    else:
        raise ValueError(
            f"camera_viz: display.mode must be window|xr, got {mode_str!r}"
        )
    if "clear_color" in display:
        session_cfg.clear_color = tuple(display["clear_color"])
    session_cfg.app_name = display.get("app_name", "camera_viz")
    # Televiz creates the XrInstance, so anything downstream needs (here the
    # controller tracker's action-context extension) has to be declared now.
    if required_extensions:
        session_cfg.required_extensions = list(required_extensions)
    return viz.VizSession.create(session_cfg)


def _build_layer(session: viz.VizSession, entry: SourceEntry, shape: str):
    """One layer of ``shape`` for ``entry``.

    quad     → QuadLayer, composited by the OpenXR runtime by default
               (``compositor: televiz`` opts into the built-in compositor);
               the placement strategy positions it per frame.
    cylinder → CylinderLayer: the feed wrapped on an arc facing the user
               (``cylinder_radius_m`` / ``cylinder_angle_deg``, aspect from
               the source). Runtime-composited always.
    equirect → EquirectLayer: full 360x180 sphere (the source is expected
               to be an equirect panorama), aimed by ``equirect_yaw_deg``.
               Runtime-composited always.
    """
    spec = entry.source.spec
    if shape == "cylinder":
        layer_cfg = viz.CylinderLayerConfig()
        layer_cfg.name = spec.name
        layer_cfg.resolution = viz.Resolution(spec.width, spec.height)
        layer_cfg.stereo = entry.stereo
        layer_cfg.stereo_baseline_mm = entry.stereo_plane_distance_cm * 10.0
        # aspect_ratio 0 = derived from the source resolution (square texels).
        layer_cfg.placement = viz.CylinderLayerPlacement(
            radius_m=entry.cylinder_radius_m,
            central_angle_rad=math.radians(entry.cylinder_angle_deg),
        )
        return session.add_cylinder_layer(layer_cfg)
    if shape == "equirect":
        layer_cfg = viz.EquirectLayerConfig()
        layer_cfg.name = spec.name
        layer_cfg.resolution = viz.Resolution(spec.width, spec.height)
        layer_cfg.stereo = entry.stereo
        # Baseline only matters at finite sphere radius; harmless at the
        # default infinite-radius placement (full 360x180 sphere).
        layer_cfg.stereo_baseline_mm = entry.stereo_plane_distance_cm * 10.0
        # Set explicitly rather than leaning on the default so the controls
        # have a known starting point to adjust from and reset to. The pose's
        # -z is where the middle of the panorama lands, so yawing it aims the
        # feed; position is irrelevant on the default infinite-radius sphere.
        layer_cfg.placement = viz.EquirectLayerPlacement(
            pose=viz.Pose3D(
                (0.0, 0.0, 0.0), yaw_quat(math.radians(entry.equirect_yaw_deg))
            )
        )
        return session.add_equirect_layer(layer_cfg)

    layer_cfg = viz.QuadLayerConfig()
    layer_cfg.name = spec.name
    layer_cfg.resolution = viz.Resolution(spec.width, spec.height)
    layer_cfg.format = viz.PixelFormat.kRGBA8
    if entry.stereo:
        layer_cfg.stereo = True
        layer_cfg.stereo_baseline_mm = entry.stereo_plane_distance_cm * 10.0
    # OpenXR-runtime composition is the default (kXr only; window mode is
    # always composited by Televiz). Requires a placement, which the
    # placement strategy applies below.
    layer_cfg.openxr_composition = entry.compositor == "openxr"
    return session.add_quad_layer(layer_cfg)


def add_layers(
    session: viz.VizSession, entry: SourceEntry, all_shapes: bool
) -> "dict[str, object]":
    """Returns ``{shape: layer}`` for ``entry``.

    With ``all_shapes`` every shape is built up front and all but the
    configured one start hidden, so switching later is just an atomic
    ``set_visible`` — no reallocation and no ``vkDeviceWaitIdle`` mid-demo,
    which is what removing and re-adding a layer would cost.
    """
    shapes = VALID_SHAPES if all_shapes else (entry.shape,)
    layers = {}
    for shape in shapes:
        layer = _build_layer(session, entry, shape)
        layer.set_visible(shape == entry.shape)
        layers[shape] = layer
    return layers


def estimate_layer_bytes(entry: SourceEntry, shape: str) -> int:
    """Rough VRAM for one layer's mailbox: kSlotCount images, doubled for
    stereo, plus the quad's mip chain."""
    spec = entry.source.spec
    per_image = spec.width * spec.height * 4
    total = per_image * _MAILBOX_SLOTS * (2 if entry.stereo else 1)
    if shape == "quad":
        total = int(total * 4 / 3)  # capped mip chain ≈ +33%
    return total
