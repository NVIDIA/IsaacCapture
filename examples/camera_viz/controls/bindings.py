# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Controller bindings for camera_viz (Quest / Pico).

Right hand -- where the feed sits and how it reads in depth:

    thumbstick X   gap between the eyes' stereo planes, held to ramp;
                   on equirect, which has no gap to set, it pans the
                   panorama instead
    thumbstick     recenter: aim the surface at wherever you are looking
      click
    A              cycle lock mode (world -> head -> lazy)
    B              toggle mono / stereo

Left hand -- what surface it is mapped onto:

    X              cycle shape (quad -> cylinder -> equirect)
    Y              reset every parameter to the YAML values
    thumbstick     shape-dependent, see controls.shapes:
                     quad      X = size,   Y = height
                     cylinder  X = arc,    Y = height
                     equirect  X = h-span, Y = v-span

Input arrives through a ``ControllerTracker`` on a ``DeviceIOSession`` that
shares the OpenXR session Televiz owns, so there is one CloudXR connection
for rendering and input both. XR only -- window mode has no controllers.

Every limit below sits strictly inside what the layers validate, because
``set_placement`` raises on a bad value and this runs on the render thread.
"""

from __future__ import annotations

import math
import sys
import threading
from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence

from . import shapes, stereo
from .hud import split_message
from placements import (
    PlacementConfig,
    PlacementStrategy,
    build as build_placement,
    heading_deg,
)

# Re-exported so ``controls`` stays the one import for callers and tests.
DEFAULT_IPD_MM = stereo.DEFAULT_IPD_MM
MAX_OFFSET_FRACTION_OF_IPD = stereo.MAX_FRACTION_OF_IPD
FAR_TARGET_M = stereo.FAR_TARGET_M
PLANE_DISTANCE_STEP_CM = stereo.STEP_CM

# isaacteleop is imported lazily inside the methods that touch the device
# session: everything else here is policy (deadzone, clamping, cycle order)
# and is unit-testable without the SDK or a headset.

# Cycle order for the A button. Starts from whatever the YAML set, so the
# first press moves to the next entry after it.
LOCK_MODE_CYCLE = ("world", "head", "gimbal", "lazy")

# Cycle order for the left X button.
SHAPE_CYCLE = ("quad", "cylinder", "equirect")

# Per-shape stick bindings, for the log line and the docs.
#: Each shape's two stick axes, from the control that implements them.
SHAPE_PARAMS = {shape: shapes.axes(shape) for shape in SHAPE_CYCLE}


@dataclass
class ControlsConfig:
    """``display.controls`` in the YAML."""

    enabled: bool = True
    # Held-stick ramp rate. At 2 cm/s that is 20 of the 0.1 cm steps a
    # second: fine enough to land on a value, quick enough to cross the
    # usable range in about three.
    plane_distance_rate_cm_per_s: float = 2.0
    plane_distance_min_cm: float = -10.0
    # The ceiling is also bounded by the measured IPD, so a stick held to the
    # stop can never reach divergent parallax.
    plane_distance_max_cm: float = 10.0
    # Quest/Pico sticks rest a few percent off centre; below this the axis
    # reads as zero so an untouched stick never drifts the plane distance.
    deadzone: float = 0.2

    # In-headset readout of what each press changed. Opaque, head-locked
    # below the eyeline, auto-hiding a couple of seconds after the change.
    hud: bool = True

    # Keep every shape resident so the left X button is an atomic
    # set_visible instead of a layer rebuild. Costs the extra shapes' VRAM;
    # camera_viz prints the figure at startup.
    shape_switching: bool = True

    # Left-stick rates, per second of full deflection.
    size_rate_m_per_s: float = 0.5
    offset_rate_m_per_s: float = 0.5
    angle_rate_deg_per_s: float = 40.0

    # Limits, all strictly inside the layers' own validation.
    size_range_m: tuple = (0.2, 8.0)
    offset_y_range_m: tuple = (-2.0, 2.0)
    cylinder_angle_range_deg: tuple = (15.0, 350.0)
    equirect_h_range_deg: tuple = (30.0, 360.0)
    equirect_v_half_range_deg: tuple = (10.0, 90.0)


@dataclass
class ControlTarget:
    """One controllable layer."""

    name: str
    layer: Any
    shape: str
    # Camera produces per-eye frames. False disables both the baseline and
    # the mono/stereo toggle: neither means anything on a mono source.
    stereo: bool
    # ``placements.<cam>.stereo_plane_distance_cm``: the gap between the
    # left-eye and right-eye planes in 3D. Applied as-is -- the HUD suggests
    # a value rather than computing one behind your back.
    plane_distance_cm: float
    lock_mode: str
    placement_config: Optional[PlacementConfig] = None
    # Submit thread reads this; render thread writes it.
    force_mono: threading.Event = field(default_factory=threading.Event)

    # {shape: layer} for every resident shape. Just {shape: layer} when
    # shape switching is off, which makes the X button a no-op.
    shape_layers: dict = field(default_factory=dict)
    # Set while mono so the gap can be put back on the way out.
    plane_distance_before_mono: Optional[float] = None
    # Live shaped-layer params; the quad's live in placement_config, which
    # the strategy already reads every frame.
    cylinder_radius_m: float = 2.0
    cylinder_angle_deg: float = 90.0
    equirect_h_deg: float = 360.0
    equirect_v_half_deg: float = 90.0
    # Heading the middle of the panorama points at, degrees about +Y. The
    # sphere has no strategy to place it, so this is its whole pose.
    equirect_yaw_deg: float = 0.0

    def __post_init__(self) -> None:
        # Snapshot for the Y button. Tuples/floats only, so a reset can't be
        # aliased by a later edit.
        self._initial = {
            "shape": self.shape,
            "lock_mode": self.lock_mode,
            "plane_distance_cm": self.plane_distance_cm,
            "cylinder_radius_m": self.cylinder_radius_m,
            "cylinder_angle_deg": self.cylinder_angle_deg,
            "equirect_h_deg": self.equirect_h_deg,
            "equirect_v_half_deg": self.equirect_v_half_deg,
            "equirect_yaw_deg": self.equirect_yaw_deg,
            "size_meters": (
                tuple(self.placement_config.size_meters)
                if self.placement_config is not None
                else None
            ),
            "distance": (
                self.placement_config.distance
                if self.placement_config is not None
                else None
            ),
            "offset_y": (
                self.placement_config.offset_y
                if self.placement_config is not None
                else 0.0
            ),
        }


class ControllerControls:
    """Owns the input session and applies its events to the layers.

    ``strategies`` is the *same list object* the runner iterates, mutated in
    place on a lock-mode change. Both that mutation and the read happen on
    the render thread, so no lock is needed; the mono/stereo flag crosses to
    the submit thread and uses an Event instead.
    """

    def __init__(
        self,
        viz_session: Any,
        targets: Sequence[ControlTarget],
        strategies: List[Optional[PlacementStrategy]],
        config: Optional[ControlsConfig] = None,
        tracker: Optional[Any] = None,
        hud: Optional[Any] = None,
        log_to_stderr: bool = True,
    ) -> None:
        """``tracker`` overrides the ControllerTracker; tests inject a fake
        so the policy can be exercised without the SDK or a headset.

        ``log_to_stderr`` is off when the status panel is live: it redraws
        stderr in place by moving the cursor, and a second writer on the same
        stream lands mid-panel and leaves a trail of half-erased copies. The
        panel shows the same events on its own line, so nothing is lost.
        """
        if len(targets) != len(strategies):
            raise ValueError(
                f"targets / strategies length mismatch: "
                f"{len(targets)} vs {len(strategies)}"
            )
        self._session = viz_session
        self._targets = list(targets)
        self._strategies = strategies
        self._cfg = config or ControlsConfig()
        self._hud = hud
        self._log = _log if log_to_stderr else _discard
        if tracker is None:
            import isaacteleop.deviceio as deviceio

            tracker = deviceio.ControllerTracker()
        self._tracker = tracker
        self._device_session: Optional[Any] = None
        self._device_ctx: Optional[Any] = None
        self._prev_a = False
        self._prev_b = False
        self._prev_stick = False
        self._prev_x = False
        self._prev_y = False
        # Held-stick messages would otherwise print once per rendered frame.
        self._elapsed = 0.0
        self._last_log: dict = {}
        self._ipd_mm = DEFAULT_IPD_MM
        self._last_event = ""
        # Live baseline needs a viz newer than the released wheel. A and B
        # are pure Python and work either way, so drop only this binding
        # rather than failing the whole run.
        self._baseline_supported = all(
            hasattr(t.layer, "set_stereo_baseline_mm") for t in self._targets
        )
        if not self._baseline_supported and any(t.stereo for t in self._targets):
            self._log(
                "installed isaacteleop has no Layer.set_stereo_baseline_mm; "
                "thumbstick baseline disabled (A and B still work)"
            )

    @staticmethod
    def required_extensions() -> List[str]:
        """OpenXR extensions to declare on ``VizSessionConfig`` *before*
        creating the session -- Televiz owns the XrInstance."""
        import isaacteleop.deviceio as deviceio

        return list(
            deviceio.DeviceIOSession.get_required_extensions(
                [deviceio.ControllerTracker()]
            )
        )

    def __enter__(self) -> "ControllerControls":
        import isaacteleop.deviceio as deviceio
        import isaacteleop.oxr as oxr

        handles = self._session.get_oxr_handles()
        if handles is None:
            raise RuntimeError("camera_viz: controls require XR mode")
        self._device_ctx = deviceio.DeviceIOSession.run(
            [self._tracker], oxr.OpenXRSessionHandles(*handles)
        )
        self._device_session = self._device_ctx.__enter__()
        return self

    def __exit__(self, *exc) -> None:
        if self._device_ctx is not None:
            self._device_ctx.__exit__(*exc)
            self._device_ctx = None
            self._device_session = None

    def active_layer(self, index: int) -> Any:
        """Layer currently visible for source ``index``. The runner calls this
        every pass -- it is the single source of truth for the active shape."""
        return self._targets[index].layer

    def strategy(self, index: int) -> Optional[PlacementStrategy]:
        """Placement strategy for source ``index``, following A-button swaps."""
        return self._strategies[index]

    def status(self, index: int) -> dict:
        """Per-camera state for the status panel."""
        target = self._targets[index]
        cfg = target.placement_config
        gap = None if target.shape == "equirect" else target.plane_distance_cm
        return {
            # The heading rides along with the shape: it is the sphere's
            # whole placement, and the lock / size / gap columns are all
            # blank for it.
            "shape": (
                f"{target.shape} {target.equirect_yaw_deg:+.0f}°"
                if target.shape == "equirect"
                else target.shape
            ),
            "lock_mode": target.lock_mode,
            "stereo": target.stereo and not target.force_mono.is_set(),
            "size_m": cfg.size_meters[0] if cfg is not None else None,
            "offset_y_m": cfg.offset_y if cfg is not None else None,
            "plane_distance_cm": gap,
            "suggested_cm": self._suggested_plane_distance_cm(target)
            if gap is not None
            else None,
        }

    @property
    def ipd_mm(self) -> float:
        return self._ipd_mm

    @property
    def last_event(self) -> str:
        return self._last_event

    def force_mono(self, index: int) -> bool:
        """Submit thread: send the left frame to both eyes for this layer."""
        return self._targets[index].force_mono.is_set()

    def step(self, dt: float, ipd_mm: Optional[float] = None) -> None:
        """Render thread: pump the device session and apply one frame of input.

        ``ipd_mm`` is the headset's reported eye separation. It converts
        stereo distances and bounds the offset below divergence.
        """
        if ipd_mm is not None and abs(ipd_mm - self._ipd_mm) > 0.1:
            self._ipd_mm = ipd_mm
            self._log(f"headset IPD: {ipd_mm:.1f} mm")
        if self._device_session is None:
            return
        self._elapsed += dt
        self._device_session.update()
        self._step_right(dt)
        self._step_left(dt)
        if self._hud is not None:
            self._hud.step(dt, self._session.head_pose_now())

    def _step_right(self, dt: float) -> None:
        # The accessor hands back the ControllerSnapshot itself, or None when
        # the device is inactive -- a headset streaming hand tracking only
        # never binds controllers at all, so None is a steady state here, not
        # a blip.
        controller = self._tracker.get_right_controller(self._device_session)
        if controller is None:
            # Controller asleep or out of range. Drop the edge state so
            # waking it mid-press doesn't fire a phantom transition.
            self._prev_a = self._prev_b = self._prev_stick = False
            return

        inputs = controller.inputs
        a, b = bool(inputs.primary_click), bool(inputs.secondary_click)
        click = bool(inputs.thumbstick_click)
        if a and not self._prev_a:
            self._cycle_lock_modes()
        if b and not self._prev_b:
            self._toggle_stereo()
        if click and not self._prev_stick:
            self._recenter()
        self._prev_a, self._prev_b, self._prev_stick = a, b, click

        # One stick, two shapes' worth of meaning: a target is either an
        # equirect or it isn't, so exactly one of these acts on it.
        self._adjust_plane_distance(float(inputs.thumbstick_x), dt)
        self._adjust_equirect_yaw(float(inputs.thumbstick_x), dt)

    def _step_left(self, dt: float) -> None:
        controller = self._tracker.get_left_controller(self._device_session)
        if controller is None:
            self._prev_x = self._prev_y = False
            return

        inputs = controller.inputs
        x, y = bool(inputs.primary_click), bool(inputs.secondary_click)
        if x and not self._prev_x:
            self._cycle_shapes()
        if y and not self._prev_y:
            self._reset_params()
        self._prev_x, self._prev_y = x, y

        self._adjust_shape_params(
            self._axis(float(inputs.thumbstick_x)),
            self._axis(float(inputs.thumbstick_y)),
            dt,
        )

    def _axis(self, value: float) -> float:
        """Deadzoned, rescaled so the ramp starts at zero just past the edge
        rather than jumping to deadzone * rate."""
        if abs(value) < self._cfg.deadzone:
            return 0.0
        span = 1.0 - self._cfg.deadzone
        return math.copysign((abs(value) - self._cfg.deadzone) / span, value)

    def _surface_distance_m(self, target) -> Optional[float]:
        """How far the target's surface sits from the viewer, or None when the
        shape has no meaningful one (an infinite sphere)."""
        if target.shape == "cylinder":
            return target.cylinder_radius_m
        if target.shape == "equirect":
            return None  # centred on the viewer, effectively at infinity
        cfg = target.placement_config
        return cfg.distance if cfg is not None else None

    def _gap_limits(self) -> tuple:
        return (self._cfg.plane_distance_min_cm, self._cfg.plane_distance_max_cm)

    def _max_plane_distance_cm(self) -> float:
        return stereo.max_gap_cm(self._ipd_mm, self._cfg.plane_distance_max_cm)

    def _clamp_plane_distance(self, value: float) -> float:
        return stereo.clamp_gap_cm(value, self._ipd_mm, self._gap_limits())

    @staticmethod
    def _stepped(value: float) -> float:
        return stereo.step(value)

    def _suggested_plane_distance_cm(self, target) -> Optional[float]:
        return stereo.suggested_gap_cm(
            self._surface_distance_m(target), self._ipd_mm, self._gap_limits()
        )

    def _stereo_distance_cm(self, target) -> Optional[float]:
        return stereo.perceived_distance_cm(
            self._surface_distance_m(target), self._ipd_mm, target.plane_distance_cm
        )

    def _adjust_plane_distance(self, axis: float, dt: float) -> None:
        """Right stick: widen or narrow the gap between the eyes' surfaces."""
        if not self._baseline_supported:
            return
        scaled = self._axis(axis)
        if scaled == 0.0:
            return
        delta = scaled * self._cfg.plane_distance_rate_cm_per_s * dt

        changed, suggestions = [], []
        for target in self._targets:
            # Equirect is skipped, not clamped to nothing: the gap works by
            # shifting each eye's surface, and camera_viz's sphere is at
            # infinite radius, where translating it changes nothing at all.
            if not target.stereo or target.shape == "equirect":
                continue
            if not self._nudge_gap(target, delta):
                continue
            changed.append(
                (target.name, f"{self._stepped(target.plane_distance_cm):.1f} cm")
            )
            suggested = self._suggested_plane_distance_cm(target)
            if suggested is not None:
                suggestions.append((target.name, f"{suggested:.1f} cm"))
        if changed:
            self._notify(
                f"stereo planes: {summarize(changed)}",
                self._gap_detail(suggestions),
                log_key="plane_distance",
            )

    def _nudge_gap(self, target, delta: float) -> bool:
        """Move one target's gap by ``delta``. False when it stayed on the
        same step, in which case the sub-step remainder is kept so a slow ramp
        still gets there, but nothing is re-applied or re-announced."""
        new = self._clamp_plane_distance(target.plane_distance_cm + delta)
        if self._stepped(new) == self._stepped(target.plane_distance_cm):
            target.plane_distance_cm = new
            return False
        self._set_plane_distance(target, new)
        return True

    def _gap_detail(self, suggestions: List[tuple]) -> str:
        """Second HUD line: the suggestion, and the IPD it came from -- which
        is how a headset whose IPD setting is wrong becomes visible."""
        ipd = f"IPD: {self._ipd_mm:.0f} mm"
        if not suggestions:
            return ipd
        # The count already appears on the first line; repeating it is noise.
        values = {value for _, value in suggestions}
        shown = suggestions[0][1] if len(values) == 1 else summarize(suggestions)
        return f"suggested: {shown} · {ipd}"

    def _adjust_equirect_yaw(self, axis: float, dt: float) -> None:
        """Right stick on an equirect: pan the panorama.

        The gap binding skips this shape (an infinite sphere cannot be
        translated), so the stick is free here and panning is what an
        operator actually reaches for on a 360 feed -- a camera is rarely
        mounted facing exactly the way the headset was when it recentered.
        """
        scaled = self._axis(axis)
        if scaled == 0.0:
            return
        delta = scaled * self._cfg.angle_rate_deg_per_s * dt
        changed = []
        for target in self._targets:
            if target.shape != "equirect":
                continue
            # Stick right pans the view right: the middle of the texture
            # swings left, which is +heading.
            target.equirect_yaw_deg = _wrap_deg(target.equirect_yaw_deg + delta)
            shapes.apply_equirect(target)
            changed.append((target.name, f"{target.equirect_yaw_deg:+.0f}°"))
        if changed:
            self._notify(f"pan: {summarize(changed)}", log_key="equirect_yaw")

    def _recenter(self) -> None:
        """Thumbstick click: put the surface back where you are looking.

        One gesture, read per shape: an equirect yaws so the middle of the
        panorama lands dead ahead, and anything with a placement re-snaps its
        anchor -- which is the way out of a world-locked plane left behind in
        another part of the room.
        """
        head = self._session.head_pose_now()
        if head is None:
            return
        heading = heading_deg(head.orientation)
        for index, target in enumerate(self._targets):
            if target.shape == "equirect":
                target.equirect_yaw_deg = _wrap_deg(heading)
                shapes.apply_equirect(target)
            elif target.placement_config is not None:
                # A fresh strategy re-snaps on its next update; retuning the
                # live one would keep the anchor it is holding.
                self._strategies[index] = build_placement(
                    target.lock_mode, target.placement_config
                )
        self._notify("recentered on your view")

    def _cycle_lock_modes(self) -> None:
        changed = []
        for i, target in enumerate(self._targets):
            # An equirect sphere is centred on the operator; there is no
            # placement to re-lock, so it has no strategy to swap.
            if self._strategies[i] is None or target.placement_config is None:
                continue
            nxt = LOCK_MODE_CYCLE[
                (LOCK_MODE_CYCLE.index(target.lock_mode) + 1) % len(LOCK_MODE_CYCLE)
            ]
            target.lock_mode = nxt
            self._strategies[i] = build_placement(nxt, target.placement_config)
            changed.append((target.name, nxt))
        if changed:
            self._notify("lock mode: " + summarize(changed))

    def _toggle_stereo(self) -> None:
        targets = [t for t in self._targets if t.stereo]
        if not targets:
            self._notify("mono/stereo: no stereo-capable camera")
            return
        # One shared decision, so a mixed set can't end up half-toggled:
        # if any layer is still stereo, the press collapses all to mono.
        to_mono = any(not t.force_mono.is_set() for t in targets)
        for target in targets:
            if to_mono:
                target.force_mono.set()
                # Both eyes get the same image now, so a plane gap would only
                # shove that flat image to some other depth. Park it at zero
                # and put the operator's value back on the way out.
                target.plane_distance_before_mono = target.plane_distance_cm
                self._set_plane_distance(target, 0.0)
            else:
                target.force_mono.clear()
                if target.plane_distance_before_mono is not None:
                    self._set_plane_distance(target, target.plane_distance_before_mono)
                    target.plane_distance_before_mono = None
        self._notify("eyes: mono" if to_mono else "eyes: stereo")

    def _set_plane_distance(self, target, value: float) -> None:
        """``value`` is the running total; the layer only ever sees whole
        steps of it."""
        target.plane_distance_cm = value
        if self._baseline_supported and target.stereo:
            # viz still speaks millimetres; this is the only conversion.
            target.layer.set_stereo_baseline_mm(self._stepped(value) * 10.0)

    # ── Left hand: shape + shape params ───────────────────────────────

    def _cycle_shapes(self) -> None:
        """Swap which shape is visible. Both set_visible calls land before
        the render() that follows on this thread, so no frame ever sees the
        pair half-applied."""
        changed = []
        for i, target in enumerate(self._targets):
            if len(target.shape_layers) < 2:
                continue
            nxt = SHAPE_CYCLE[(SHAPE_CYCLE.index(target.shape) + 1) % len(SHAPE_CYCLE)]
            new_layer = target.shape_layers.get(nxt)
            if new_layer is None:
                continue
            target.shape_layers[target.shape].set_visible(False)
            new_layer.set_visible(True)
            target.shape = nxt
            target.layer = new_layer
            # The gap is tracked per camera but applied per layer, so the
            # newly visible one is still on whatever the YAML built it with.
            self._set_plane_distance(target, target.plane_distance_cm)
            changed.append((target.name, nxt))
        if changed:
            shape = self._targets[0].shape if self._targets else ""
            hint = SHAPE_PARAMS.get(shape)
            suffix = f"  (stick: X {hint[0]}, Y {hint[1]})" if hint else ""
            self._notify("shape: " + summarize(changed) + suffix)

    def _adjust_shape_params(self, ax: float, ay: float, dt: float) -> None:
        """Left stick. One entry per camera with its changed parameters joined:
        summarize() collapses one value across cameras, so handing it two
        parameters from the same camera would read as cameras disagreeing and
        prefix every one with the camera's name."""
        if ax == 0.0 and ay == 0.0:
            return
        changed = []
        for i, target in enumerate(self._targets):
            control = shapes.for_shape(target.shape)
            if control is None:
                continue
            parts = control.adjust(target, self._cfg, self._strategies[i], ax, ay, dt)
            if parts:
                changed.append((target.name, "   ".join(parts)))
        if changed:
            self._notify(summarize(changed), log_key="shape_params")

    def _reset_params(self) -> None:
        """Y button: back to the YAML values -- the way out of a demo that has
        been knocked out of shape."""
        for index, target in enumerate(self._targets):
            self._reset_target(index, target)
        self._notify("reset to config defaults")

    def _reset_target(self, index: int, target) -> None:
        initial = target._initial
        strategy = self._strategies[index]

        if target.shape != initial["shape"] and initial["shape"] in target.shape_layers:
            target.shape_layers[target.shape].set_visible(False)
            target.shape = initial["shape"]
            target.layer = target.shape_layers[target.shape]
            target.layer.set_visible(True)

        target.force_mono.clear()
        target.plane_distance_before_mono = None
        self._set_plane_distance(target, initial["plane_distance_cm"])

        if target.lock_mode != initial["lock_mode"]:
            target.lock_mode = initial["lock_mode"]
            if target.placement_config is not None and strategy is not None:
                # A fresh strategy is right here: reset is meant to re-snap.
                strategy = build_placement(target.lock_mode, target.placement_config)
                self._strategies[index] = strategy

        if target.placement_config is not None and initial["size_meters"]:
            shapes.retune(
                target,
                strategy,
                size_meters=initial["size_meters"],
                distance=initial["distance"],
                offset_y=initial["offset_y"],
            )

        target.cylinder_radius_m = initial["cylinder_radius_m"]
        target.cylinder_angle_deg = initial["cylinder_angle_deg"]
        target.equirect_h_deg = initial["equirect_h_deg"]
        target.equirect_v_half_deg = initial["equirect_v_half_deg"]
        target.equirect_yaw_deg = initial["equirect_yaw_deg"]
        shapes.apply_all(target)

    def _notify(
        self,
        message: str,
        detail: str = "",
        log_key: Optional[str] = None,
        log_period: float = 0.25,
    ) -> None:
        """Single place every control message goes: stderr for whoever is at
        the workstation, HUD for whoever is wearing the headset.

        The HUD always updates -- it is a live readout, and throttling it made
        a value moving in 0.1 cm steps look like it moved in 0.5 cm ones.
        ``log_key`` throttles only the stderr line, which is a log and would
        otherwise scroll a page per second while a stick is held.

        ``detail`` takes the panel's second line verbatim instead of letting
        the message wrap into it.
        """
        self._last_event = f"{message}  ·  {detail}" if detail else message
        if (
            log_key is None
            or self._elapsed - self._last_log.get(log_key, -1e9) >= log_period
        ):
            if log_key is not None:
                self._last_log[log_key] = self._elapsed
            self._log(f"{message}  |  {detail}" if detail else message)
        if self._hud is not None:
            self._hud.show([message, detail] if detail else split_message(message))


def summarize(items: List[tuple]) -> str:
    """Collapse per-camera ``(name, value)`` pairs for the message line.

    Every binding applies to every camera at once, so the values normally
    agree -- say it once with a count instead of repeating it N times and
    overflowing the HUD.
    """
    if not items:
        return ""
    values = {value for _, value in items}
    if len(values) == 1:
        value = items[0][1]
        return value if len(items) == 1 else f"{value}  ({len(items)} cameras)"
    return "  ".join(f"{name} {value}" for name, value in items)


def _clamp(value: float, limits) -> float:
    lo, hi = limits
    return min(max(value, lo), hi)


def _wrap_deg(angle: float) -> float:
    """Into (-180, 180], so a panned-all-the-way-round readout stays legible."""
    return (angle + 180.0) % 360.0 - 180.0


def _log(message: str) -> None:
    # Operator feedback for a run without the panel: confirmation at the
    # workstation that a press registered.
    print(f"camera_viz: controls: {message}", file=sys.stderr, flush=True)


def _discard(message: str) -> None:
    """Sink for when the status panel owns stderr."""


def controls_config_from_yaml(display: dict) -> ControlsConfig:
    """Parse ``display.controls``; unknown keys raise rather than silently
    doing nothing."""
    spec = display.get("controls", {})
    if not isinstance(spec, dict):
        raise ValueError("camera_viz: display.controls must be a mapping")
    known = {f.name for f in ControlsConfig.__dataclass_fields__.values()}
    unknown = set(spec) - known
    if unknown:
        raise ValueError(
            f"camera_viz: display.controls: unknown key(s) {sorted(unknown)}; "
            f"valid keys are {sorted(known)}"
        )
    return ControlsConfig(**spec)
