# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The YAML, turned into the objects the viewer runs on.

Everything here reads ``cameras`` and ``display.placements`` and produces
:class:`SourceEntry` values — one per camera stream, carrying its source, its
placement strategy and its surface config. Nothing here touches Vulkan, the
CloudXR runtime or a display: this is the parse-and-validate half, so a
malformed config fails before anything is allocated.

Unknown keys warn rather than falling back silently — a typo'd
``cylinder_radius`` should not run at the 2 m default with no hint.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import List, Optional, Tuple

from .pipeline import FrameSource
from .placements import PlacementConfig, PlacementStrategy, build as build_placement
from .sources import PairedFrameSource, RtpH264Source, build_local_camera


@dataclass
class SourceEntry:
    """source + placement + stereo + shape cfg; drives layer construction."""

    source: FrameSource
    placement: Optional[PlacementStrategy]
    stereo: bool = False
    # display.placements.<name>.stereo_plane_distance_cm — the gap between
    # the left-eye and right-eye planes.
    stereo_plane_distance_cm: float = 0.0
    # Kept so the controls can rebuild a strategy when the lock mode is
    # cycled at runtime; None outside XR (no placement to rebuild).
    lock_mode: str = "lazy"
    placement_config: Optional[PlacementConfig] = None
    # display.placements.<name>.shape: quad | cylinder | equirect.
    shape: str = "quad"
    # Who composites the layer (display.placements.<name>.compositor):
    # "openxr" (default — the OpenXR runtime) or "televiz" (built-in
    # compositor; quads only).
    compositor: str = "openxr"
    # Cylinder shape parameters (display.placements.<name>).
    cylinder_radius_m: float = 2.0
    cylinder_angle_deg: float = 90.0
    # Equirect shape parameter: heading the middle of the panorama points
    # at, degrees about +Y (0 = the reference space's forward, positive to
    # the left). The sphere has no lock-mode strategy, so this is how a feed
    # whose camera does not face the way the headset started gets aimed.
    equirect_yaw_deg: float = 0.0


VALID_SHAPES = ("quad", "cylinder", "equirect")

_VALID_COMPOSITORS = ("openxr", "televiz")

# Every key the placements.<name> block understands (lock-mode strategy
# knobs + surface-shape keys). Unknown keys warn instead of silently
# falling back to defaults — a typo'd `cylinder_radius` should not run
# with a 2 m default and no hint.
_KNOWN_PLACEMENT_KEYS = frozenset(
    {
        "lock_mode",
        "distance",
        "offset_x",
        "offset_y",
        "look_away_angle_deg",
        "reposition_distance",
        "reposition_delay_s",
        "transition_duration_s",
        "size",
        "stereo_plane_distance_cm",
        "shape",
        "compositor",
        "cylinder_radius_m",
        "cylinder_angle_deg",
        "equirect_yaw_deg",
    }
)


def _warn_unknown_placement_keys(cam_name: str, pspec: dict) -> None:
    import difflib

    for key in pspec:
        if key in _KNOWN_PLACEMENT_KEYS:
            continue
        hint = difflib.get_close_matches(key, _KNOWN_PLACEMENT_KEYS, n=1)
        suggestion = f" (did you mean {hint[0]!r}?)" if hint else ""
        print(
            f"camera_viz: warning: placements.{cam_name}: unknown key "
            f"{key!r}{suggestion} — ignored",
            file=sys.stderr,
            flush=True,
        )


def _shape_for(
    cam_name: str, placements_cfg: dict
) -> Tuple[str, str, float, float, float]:
    """Per-camera surface config from ``display.placements.<name>``:
    ``shape`` (quad | cylinder | equirect, default quad), ``compositor``
    (openxr — the default — or televiz; quads only), ``cylinder_radius_m``
    / ``cylinder_angle_deg`` (cylinder only), ``equirect_yaw_deg``
    (equirect only)."""
    pspec = placements_cfg.get(cam_name) or {}
    _warn_unknown_placement_keys(cam_name, pspec)
    shape = str(pspec.get("shape", "quad")).lower()
    if shape not in VALID_SHAPES:
        raise ValueError(
            f"camera_viz: placements.{cam_name}.shape must be one of "
            f"{'|'.join(VALID_SHAPES)}, got {shape!r}"
        )
    compositor = str(pspec.get("compositor", "openxr")).lower()
    if compositor not in _VALID_COMPOSITORS:
        raise ValueError(
            f"camera_viz: placements.{cam_name}.compositor must be "
            f"{'|'.join(_VALID_COMPOSITORS)}, got {compositor!r}"
        )
    if compositor == "televiz" and shape != "quad":
        raise ValueError(
            f"camera_viz: placements.{cam_name}: compositor: televiz only "
            f"applies to shape: quad — {shape} layers are composited by the "
            "OpenXR runtime always."
        )
    radius_m = float(pspec.get("cylinder_radius_m", 2.0))
    angle_deg = float(pspec.get("cylinder_angle_deg", 90.0))
    yaw_deg = float(pspec.get("equirect_yaw_deg", 0.0))
    return shape, compositor, radius_m, angle_deg, yaw_deg


_VALID_LOCK_MODES = ("world", "head", "lazy", "gimbal")


def _build_placement(
    spec: Optional[dict], is_xr: bool
) -> Tuple[Optional[PlacementStrategy], str, Optional[PlacementConfig]]:
    """Returns (strategy, lock_mode, config). The last two let the
    controls rebuild a strategy when the lock mode changes at runtime."""
    if spec is not None:
        # Validate in every display mode — a typo'd lock_mode shouldn't
        # silently become lazy (XR) or pass unnoticed (window).
        lock_mode = str(spec.get("lock_mode", "lazy")).lower()
        if lock_mode not in _VALID_LOCK_MODES:
            raise ValueError(
                f"camera_viz: lock_mode must be {'|'.join(_VALID_LOCK_MODES)}, "
                f"got {lock_mode!r}"
            )
    if not is_xr or spec is None:
        return None, "lazy", None
    cfg_kwargs = {}
    if "size" in spec:
        cfg_kwargs["size_meters"] = tuple(spec["size"])
    for key in (
        "distance",
        "offset_x",
        "offset_y",
        "look_away_angle_deg",
        "reposition_distance",
        "reposition_delay_s",
        "transition_duration_s",
    ):
        if key in spec:
            cfg_kwargs[key] = spec[key]
    cfg = PlacementConfig(**cfg_kwargs)
    # The normalised form, not the raw one: the controls cycle it through
    # LOCK_MODE_CYCLE.index(), so a config saying `WORLD` would validate here
    # and then raise ValueError on the first press of A.
    lock_mode = str(spec.get("lock_mode", "lazy")).lower()
    return build_placement(lock_mode, cfg), lock_mode, cfg


def _enabled_cameras(cfg: dict) -> List[dict]:
    return [c for c in cfg.get("cameras", []) if c.get("enabled", True)]


# Default plane width when ``size`` is omitted from a placement block.
# Height is derived from the camera's pixel aspect ratio so the rendered
# plane keeps the picture's shape.
_DEFAULT_PLANE_WIDTH_M = 1.0


def _placement_with_aspect(
    spec: Optional[dict], width: int, height: int, is_xr: bool
) -> Tuple[Optional[PlacementStrategy], str, Optional[PlacementConfig]]:
    """Build the placement, filling in ``size`` from the source's aspect
    ratio when the YAML doesn't pin it. Width defaults to 1.0 m so a
    16:9 source lands at 1.0 x 0.5625, a 3.55:1 SBS at 1.0 x 0.281."""
    if spec is not None and "size" not in spec:
        spec = {
            **spec,
            "size": [_DEFAULT_PLANE_WIDTH_M, _DEFAULT_PLANE_WIDTH_M * height / width],
        }
    return _build_placement(spec, is_xr)


def _stereo_for(cam: dict, placements_cfg: dict) -> Tuple[bool, float]:
    """``cameras.<cam>.stereo`` (producer toggle) plus the placement's
    ``placements.<cam>.stereo_plane_distance_cm`` — the gap between the
    left-eye and right-eye planes in 3D."""
    stereo = bool(cam.get("stereo", False))
    pspec = placements_cfg.get(cam["name"]) or {}
    return stereo, float(pspec.get("stereo_plane_distance_cm", 0.0))


def build_local_entries(cfg: dict, is_xr: bool) -> List[SourceEntry]:
    """source=local: open each enabled camera directly."""
    placements_cfg = cfg.get("display", {}).get("placements", {})
    entries: List[SourceEntry] = []
    for cam in _enabled_cameras(cfg):
        cam_sources = build_local_camera(cam)
        # Aspect comes from the built source's spec, not the YAML — video
        # sources may omit width/height and size themselves from the file.
        first = cam_sources[0].spec
        placement, lock_mode, placement_cfg = _placement_with_aspect(
            placements_cfg.get(cam["name"]), first.width, first.height, is_xr
        )
        stereo, plane_distance_cm = _stereo_for(cam, placements_cfg)
        shape, compositor, radius_m, angle_deg, yaw_deg = _shape_for(
            cam["name"], placements_cfg
        )
        for source in cam_sources:
            entries.append(
                SourceEntry(
                    source=source,
                    placement=placement,
                    stereo=stereo,
                    stereo_plane_distance_cm=plane_distance_cm,
                    shape=shape,
                    compositor=compositor,
                    cylinder_radius_m=radius_m,
                    cylinder_angle_deg=angle_deg,
                    equirect_yaw_deg=yaw_deg,
                    lock_mode=lock_mode,
                    placement_config=placement_cfg,
                )
            )
    return entries


def build_rtp_entries(cfg: dict, is_xr: bool) -> List[SourceEntry]:
    """One RTP listener per camera; stereo uses rtp.port + rtp.port_right
    and pairs them at the receiver (no wire-level sync — drift OK)."""
    placements_cfg = cfg.get("display", {}).get("placements", {})
    entries: List[SourceEntry] = []
    for cam in _enabled_cameras(cfg):
        rtp = cam.get("rtp", {})
        if "port" not in rtp:
            raise ValueError(
                f"camera_viz: camera {cam.get('name')!r} missing rtp.port; "
                "required when source: rtp"
            )
        if "width" not in cam or "height" not in cam:
            raise ValueError(
                f"camera_viz: camera {cam.get('name')!r} needs explicit "
                "width/height when source: rtp — the receiver sizes its "
                "decoder from the YAML, not from the wire"
            )
        placement, lock_mode, placement_cfg = _placement_with_aspect(
            placements_cfg.get(cam["name"]),
            int(cam["width"]),
            int(cam["height"]),
            is_xr,
        )
        stereo, plane_distance_cm = _stereo_for(cam, placements_cfg)

        if stereo:
            if "port_right" not in rtp:
                raise ValueError(
                    f"camera_viz: stereo camera {cam.get('name')!r} missing "
                    "rtp.port_right (required when stereo + source: rtp)"
                )
            left = RtpH264Source(
                name=f"{cam['name']}.left",
                width=int(cam["width"]),
                height=int(cam["height"]),
                port=int(rtp["port"]),
                rtp_buffer_size=int(rtp.get("rtp_buffer_size", 212992)),
                gpu_id=int(rtp.get("gpu_id", 0)),
            )
            right = RtpH264Source(
                name=f"{cam['name']}.right",
                width=int(cam["width"]),
                height=int(cam["height"]),
                port=int(rtp["port_right"]),
                rtp_buffer_size=int(rtp.get("rtp_buffer_size", 212992)),
                gpu_id=int(rtp.get("gpu_id", 0)),
            )
            source: FrameSource = PairedFrameSource(
                name=cam["name"], left=left, right=right
            )
        else:
            source = RtpH264Source(
                name=cam["name"],
                width=int(cam["width"]),
                height=int(cam["height"]),
                port=int(rtp["port"]),
                rtp_buffer_size=int(rtp.get("rtp_buffer_size", 212992)),
                gpu_id=int(rtp.get("gpu_id", 0)),
            )

        shape, compositor, radius_m, angle_deg, yaw_deg = _shape_for(
            cam["name"], placements_cfg
        )
        entries.append(
            SourceEntry(
                source=source,
                placement=placement,
                stereo=stereo,
                stereo_plane_distance_cm=plane_distance_cm,
                shape=shape,
                compositor=compositor,
                cylinder_radius_m=radius_m,
                cylinder_angle_deg=angle_deg,
                equirect_yaw_deg=yaw_deg,
                lock_mode=lock_mode,
                placement_config=placement_cfg,
            )
        )
    return entries


def check_shapes_are_displayable(cfg: dict, effective_mode: str) -> None:
    """Shaped layers are composited by the OpenXR runtime, so they need XR
    mode. Checked here, before the runtime is launched, rather than as a
    failure part-way through building the session."""
    placements_cfg = cfg.get("display", {}).get("placements", {})
    for cam in _enabled_cameras(cfg):
        shape = _shape_for(cam["name"], placements_cfg)[0]
        if shape != "quad" and effective_mode != "xr":
            raise SystemExit(
                f"camera_viz: placements.{cam['name']}.shape: {shape} is "
                "composited by the OpenXR runtime and requires XR mode; "
                "use --mode xr or shape: quad in window mode."
            )
