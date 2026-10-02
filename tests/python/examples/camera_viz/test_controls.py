# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the right-controller bindings.

The input side is faked: these exercise the policy (edge detection,
deadzone, clamping, cycle order, mixed-camera toggling), which is where
the bugs live. Driving a real ControllerTracker needs a headset.
"""

from __future__ import annotations

import math
from dataclasses import replace as dataclasses_replace

import pytest

from controls import (
    LOCK_MODE_CYCLE,
    SHAPE_CYCLE,
    ControllerControls,
    ControlsConfig,
    ControlTarget,
)
from controls import controls_config_from_yaml
from placements import (
    PlacementConfig,
    build as build_placement,
    heading_deg,
    yaw_quat,
)


class FakeLayer:
    def __init__(self) -> None:
        self.baseline_mm = 0.0

    def set_stereo_baseline_mm(self, value: float) -> None:
        self.baseline_mm = value


class FakeInputs:
    def __init__(self, a=False, b=False, x=0.0, y=0.0, click=False) -> None:
        self.primary_click = a
        self.secondary_click = b
        self.thumbstick_x = x
        self.thumbstick_y = y
        self.thumbstick_click = click


class FakeSnapshot:
    def __init__(self, inputs) -> None:
        self.inputs = inputs


class FakeTracker:
    """Stands in for ControllerTracker; per-hand state is swapped per step.

    Shaped to match the real binding exactly, because an earlier version of
    this fake did not and hid a crash: it wrapped every read in an extra
    object carrying the snapshot on a ``data`` attribute. The real accessor
    returns a ``ControllerSnapshot`` (fields ``inputs`` / ``grip_pose`` /
    ``aim_pose``, no ``data``) or ``None`` when the controller is inactive.
    """

    def __init__(self) -> None:
        self.right = FakeInputs()
        self.left = FakeInputs()

    def get_right_controller(self, _session):
        return None if self.right is None else FakeSnapshot(self.right)

    def get_left_controller(self, _session):
        return None if self.left is None else FakeSnapshot(self.left)


class FakeSession:
    """DeviceIOSession stub: step() only calls update()."""

    def update(self) -> None:
        pass


class FakePose3D:
    def __init__(self, orientation) -> None:
        self.position = (0.0, 0.0, 0.0)
        self.orientation = orientation


class FakeVizSession:
    """Only the bit the controls touch per frame."""

    def __init__(self, head=None) -> None:
        self.head = head

    def head_pose_now(self):
        return self.head


def _make(targets, config=None):
    """Real ControllerControls with only the device boundary faked, so the
    policy under test is the shipping code path."""
    # Real strategies: the quad path calls retune() on them.
    strategies = [
        build_placement(t.lock_mode, t.placement_config)
        if t.placement_config is not None
        else None
        for t in targets
    ]
    controls = ControllerControls(
        FakeVizSession(),
        targets,
        strategies,
        config or ControlsConfig(),
        tracker=FakeTracker(),
    )
    controls._device_session = FakeSession()
    return controls, strategies


def _stereo_target(name="cam", plane_distance=0.0, lock_mode="lazy"):
    return ControlTarget(
        name=name,
        layer=FakeLayer(),
        shape="quad",
        stereo=True,
        plane_distance_cm=plane_distance,
        lock_mode=lock_mode,
        placement_config=PlacementConfig(),
    )


def _press(controls, dt=1.0, **kwargs):
    """Right hand: a, b, x (stick), click (stick press)."""
    controls._tracker.right = FakeInputs(**kwargs)
    controls._tracker.left = FakeInputs()
    controls.step(dt)


def _press_left(controls, dt=1.0, **kwargs):
    """Left hand: a = X button, b = Y button, x / y = stick."""
    controls._tracker.right = FakeInputs()
    controls._tracker.left = FakeInputs(**kwargs)
    controls.step(dt)


# ── A: lock mode ──────────────────────────────────────────────────────


def test_a_cycles_lock_mode_in_order():
    target = _stereo_target(lock_mode="world")
    controls, strategies = _make([target])
    before = strategies[0]

    _press(controls, a=True)
    assert target.lock_mode == "head"
    # A new strategy object must land in the shared list the runner reads.
    assert strategies[0] is not before

    _press(controls, a=False)
    _press(controls, a=True)
    assert target.lock_mode == "gimbal"


def test_a_wraps_around_the_cycle():
    target = _stereo_target(lock_mode=LOCK_MODE_CYCLE[-1])
    controls, _ = _make([target])
    _press(controls, a=True)
    assert target.lock_mode == LOCK_MODE_CYCLE[0]


def test_held_a_fires_once():
    """Rising edge only — a held button must not cycle every frame."""
    target = _stereo_target(lock_mode="world")
    controls, _ = _make([target])
    for _ in range(5):
        _press(controls, a=True)
    assert target.lock_mode == "head"


def test_equirect_without_strategy_is_skipped():
    target = ControlTarget(
        name="sky",
        layer=FakeLayer(),
        shape="equirect",
        stereo=True,
        plane_distance_cm=0.0,
        lock_mode="lazy",
        placement_config=None,
    )
    controls, strategies = _make([target])
    _press(controls, a=True)
    assert strategies[0] is None
    assert target.lock_mode == "lazy"


# ── B: mono / stereo ──────────────────────────────────────────────────


def test_b_toggles_mono_then_back():
    target = _stereo_target()
    controls, _ = _make([target])

    _press(controls, b=True)
    assert target.force_mono.is_set()

    _press(controls, b=False)
    _press(controls, b=True)
    assert not target.force_mono.is_set()


def test_b_is_a_noop_for_mono_cameras():
    target = ControlTarget(
        name="mono",
        layer=FakeLayer(),
        shape="quad",
        stereo=False,
        plane_distance_cm=0.0,
        lock_mode="lazy",
        placement_config=PlacementConfig(),
    )
    controls, _ = _make([target])
    _press(controls, b=True)
    assert not target.force_mono.is_set()


def test_b_keeps_mixed_cameras_in_step():
    """One press must not leave half the cameras mono and half stereo."""
    a, b = _stereo_target("a"), _stereo_target("b")
    b.force_mono.set()
    controls, _ = _make([a, b])

    # 'a' is still stereo, so the press collapses everything to mono.
    _press(controls, b=True)
    assert a.force_mono.is_set() and b.force_mono.is_set()

    _press(controls, b=False)
    _press(controls, b=True)
    assert not a.force_mono.is_set() and not b.force_mono.is_set()


# ── Thumbstick: baseline ──────────────────────────────────────────────


def test_stick_inside_deadzone_does_not_drift():
    target = _stereo_target(plane_distance=1.0)
    controls, _ = _make([target], ControlsConfig(deadzone=0.2))
    _press(controls, x=0.19)
    assert target.plane_distance_cm == 1.0
    assert target.layer.baseline_mm == 0.0  # setter never called


def test_stick_ramps_and_reaches_the_layer():
    target = _stereo_target(plane_distance=0.0)
    controls, _ = _make(
        [target], ControlsConfig(deadzone=0.0, plane_distance_rate_cm_per_s=4.0)
    )
    _press(controls, x=1.0)  # dt == 1.0
    assert target.plane_distance_cm == pytest.approx(4.0)
    # viz still speaks millimetres, so the layer sees 10x the cm value.
    assert target.layer.baseline_mm == pytest.approx(40.0)

    _press(controls, x=-1.0)
    assert target.plane_distance_cm == pytest.approx(0.0)


def test_deadzone_rescale_starts_the_ramp_at_zero():
    """Just past the deadzone should crawl, not jump to deadzone * rate."""
    target = _stereo_target()
    controls, _ = _make(
        [target], ControlsConfig(deadzone=0.5, plane_distance_rate_cm_per_s=10.0)
    )
    _press(controls, x=0.51)
    assert target.plane_distance_cm == pytest.approx(0.2, abs=0.05)


def test_offset_can_never_reach_divergent_parallax():
    """At an offset equal to the IPD the eyes' rays are parallel; past it they
    must diverge, which they physically cannot do. The ceiling is derived from
    the measured IPD, so a stick held to the stop stays clear of it however
    generous the configured limit is."""
    from controls import MAX_OFFSET_FRACTION_OF_IPD

    target = _stereo_target(plane_distance=0.0)
    controls, _ = _make(
        [target],
        ControlsConfig(
            deadzone=0.0, plane_distance_rate_cm_per_s=1e5, plane_distance_max_cm=1000.0
        ),
    )
    ipd_mm = 63.0
    for _ in range(5):
        _press(controls, dt=1.0, x=1.0)
        controls.step(0.0, ipd_mm)
    ipd_cm = ipd_mm / 10.0
    assert target.plane_distance_cm == pytest.approx(
        ipd_cm * MAX_OFFSET_FRACTION_OF_IPD
    )
    assert target.plane_distance_cm < ipd_cm


def test_offset_clamps_to_the_configured_range_when_it_is_the_tighter_one():
    target = _stereo_target(plane_distance=0.0)
    controls, _ = _make(
        [target],
        ControlsConfig(
            deadzone=0.0, plane_distance_rate_cm_per_s=100.0, plane_distance_max_cm=2.0
        ),
    )
    _press(controls, x=1.0)
    assert target.plane_distance_cm == pytest.approx(2.0)
    # Already clamped: no further change, and no redundant setter call.
    target.layer.baseline_mm = -1.0
    _press(controls, x=1.0)
    assert target.plane_distance_cm == pytest.approx(2.0)
    assert target.layer.baseline_mm == -1.0


def test_mono_camera_baseline_untouched():
    target = ControlTarget(
        name="mono",
        layer=FakeLayer(),
        shape="quad",
        stereo=False,
        plane_distance_cm=0.0,
        lock_mode="lazy",
        placement_config=PlacementConfig(),
    )
    controls, _ = _make([target], ControlsConfig(deadzone=0.0))
    _press(controls, x=1.0)
    assert target.plane_distance_cm == 0.0


# ── Controller loss ───────────────────────────────────────────────────


def test_absent_controller_does_not_fire_a_phantom_press():
    """Dropping out mid-press then returning must not read as a new press."""
    target = _stereo_target(lock_mode="world")
    controls, _ = _make([target])

    _press(controls, a=True)
    assert target.lock_mode == "head"

    controls._tracker.right = None  # controller asleep
    controls.step(1.0)

    _press(controls, a=True)  # still held on return
    assert target.lock_mode == "gimbal"


def test_step_survives_both_controllers_inactive():
    """Neither hand present: step() must return quietly, not raise.

    The regression this pins is a crash, not a behaviour. ``step`` runs on the
    render thread, so an AttributeError here is not a dropped frame -- the
    runner re-raises it out of ``wait`` and the whole viewer exits, which is
    what an idle pair of controllers used to do at session start.
    """
    target = _stereo_target(lock_mode="world")
    controls, _ = _make([target])

    controls._tracker.right = None
    controls._tracker.left = None
    controls.step(1.0)

    assert target.lock_mode == "world"


def test_fake_tracker_matches_the_real_snapshot_shape():
    """Pin FakeSnapshot to the binding it stands in for.

    Every other test here runs against the fake, so a fake that drifts from
    ControllerTracker proves nothing -- that is exactly how a read of a
    ``data`` attribute the real ControllerSnapshot has never had survived a
    green suite and crashed the render thread in XR. Skipped when the
    compiled package is absent (ctest supplies it on PYTHONPATH).
    """
    schema = pytest.importorskip("isaacteleop.schema._schema")

    real = schema.ControllerSnapshot
    assert hasattr(real, "inputs"), "binding lost 'inputs'"
    assert hasattr(FakeSnapshot(FakeInputs()), "inputs"), "fake lost 'inputs'"
    assert not hasattr(real, "data"), (
        "ControllerSnapshot grew a 'data' attribute; the accessor contract "
        "this fake encodes has changed"
    )


# ── Config parsing ────────────────────────────────────────────────────


def test_baseline_disabled_when_viz_lacks_the_setter():
    """An older installed isaacteleop must degrade, not raise every frame."""
    target = _stereo_target(plane_distance=0.0)

    class OldLayer:  # released wheel: no set_stereo_baseline_mm
        pass

    target.layer = OldLayer()
    controls, _ = _make([target], ControlsConfig(deadzone=0.0))
    # Detected by the constructor's probe, not forced by the test.
    assert controls._baseline_supported is False
    _press(controls, x=1.0)
    assert target.plane_distance_cm == 0.0

    # A still works on the same wheel.
    _press(controls, a=True)
    assert target.lock_mode == "world"


def test_controls_config_defaults_and_overrides():
    assert controls_config_from_yaml({}).enabled is True
    cfg = controls_config_from_yaml({"controls": {"enabled": False, "deadzone": 0.3}})
    assert cfg.enabled is False and cfg.deadzone == 0.3


def test_controls_config_rejects_unknown_keys():
    with pytest.raises(ValueError, match="unknown key"):
        controls_config_from_yaml({"controls": {"deadzoneee": 0.3}})


# ── Left X: shape cycling ─────────────────────────────────────────────


class FakeShapedLayer(FakeLayer):
    """Layer with visibility + a mutable placement, like the shaped layers."""

    def __init__(self, placement) -> None:
        super().__init__()
        self.visible = False
        self._placement = placement

    def set_visible(self, visible: bool) -> None:
        self.visible = bool(visible)

    def placement(self):
        return self._placement

    def set_placement(self, placement) -> None:
        self._placement = placement


class FakeCylinderPlacement:
    def __init__(self) -> None:
        self.radius_m = 2.0
        self.central_angle_rad = math.radians(90.0)


class FakePose:
    """viz.Pose3D's surface: the placement's ``pose`` is a live reference, so
    writing ``pose.orientation`` reaches the placement."""

    def __init__(self) -> None:
        self.position = (0.0, 0.0, 0.0)
        self.orientation = (1.0, 0.0, 0.0, 0.0)


class FakeEquirectPlacement:
    def __init__(self) -> None:
        self.pose = FakePose()
        self.central_horizontal_angle_rad = math.radians(360.0)
        self.upper_vertical_angle_rad = math.radians(90.0)
        self.lower_vertical_angle_rad = math.radians(-90.0)


def _switchable_target(shape="quad", name="cam"):
    shape_layers = {
        "quad": FakeShapedLayer(None),
        "cylinder": FakeShapedLayer(FakeCylinderPlacement()),
        "equirect": FakeShapedLayer(FakeEquirectPlacement()),
    }
    shape_layers[shape].visible = True
    return ControlTarget(
        name=name,
        layer=shape_layers[shape],
        shape=shape,
        stereo=True,
        plane_distance_cm=0.0,
        lock_mode="lazy",
        placement_config=PlacementConfig(),
        shape_layers=shape_layers,
    )


def test_left_x_cycles_shape_and_swaps_visibility():
    target = _switchable_target("quad")
    controls, _ = _make([target])

    _press_left(controls, a=True)
    assert target.shape == "cylinder"
    assert target.shape_layers["cylinder"].visible
    assert not target.shape_layers["quad"].visible
    # This is what the runner asks for each pass.
    assert controls.active_layer(0) is target.shape_layers["cylinder"]

    _press_left(controls, a=False)
    _press_left(controls, a=True)
    assert target.shape == "equirect"


def test_left_x_wraps_and_holding_fires_once():
    target = _switchable_target(SHAPE_CYCLE[-1])
    controls, _ = _make([target])
    for _ in range(4):
        _press_left(controls, a=True)  # held
    assert target.shape == SHAPE_CYCLE[0]


def test_left_x_is_a_noop_without_shape_switching():
    """Only the configured shape is resident, so there is nothing to cycle."""
    target = _stereo_target()
    controls, _ = _make([target])
    _press_left(controls, a=True)
    assert target.shape == "quad"


def test_exactly_one_shape_visible_through_a_full_cycle():
    target = _switchable_target("quad")
    controls, _ = _make([target])
    for _ in range(len(SHAPE_CYCLE) + 1):
        visible = [s for s, layer in target.shape_layers.items() if layer.visible]
        assert visible == [target.shape]
        _press_left(controls, a=False)
        _press_left(controls, a=True)


# ── Left stick: per-shape parameters ──────────────────────────────────


def test_quad_x_sizes_the_plane_without_touching_its_distance():
    """Width in metres, matching the YAML's own unit. Distance is fixed at
    runtime, so width maps 1:1 to how big it looks."""
    target = _switchable_target("quad")
    d0 = target.placement_config.distance
    w0, h0 = target.placement_config.size_meters
    controls, strategies = _make(
        [target], ControlsConfig(deadzone=0.0, size_rate_m_per_s=1.0)
    )

    _press_left(controls, x=1.0, dt=0.5)
    cfg = target.placement_config
    w1, h1 = cfg.size_meters
    assert cfg.distance == d0  # X does not move the plane
    assert w1 == pytest.approx(w0 + 0.5)
    assert w1 / h1 == pytest.approx(w0 / h0)  # aspect preserved
    assert strategies[0]._config is cfg


def test_quad_y_slides_the_plane_up_and_down():
    """The axis distance used to occupy, doing something distance cannot."""
    target = _switchable_target("quad")
    size0 = target.placement_config.size_meters
    controls, _ = _make([target], ControlsConfig(deadzone=0.0, offset_rate_m_per_s=1.0))

    _press_left(controls, y=1.0, dt=0.25)
    cfg = target.placement_config
    assert cfg.offset_y == pytest.approx(0.25)
    assert cfg.size_meters == size0  # and it does not resize


def test_cylinder_x_widens_the_arc_and_y_slides_it():
    """Radius is not a stick axis: arc width is radius*angle, so growing the
    radius grows the surface by the same factor and nothing looks different."""
    target = _switchable_target("cylinder")
    radius0 = target.cylinder_radius_m
    controls, _ = _make(
        [target],
        ControlsConfig(
            deadzone=0.0, angle_rate_deg_per_s=40.0, offset_rate_m_per_s=1.0
        ),
    )

    _press_left(controls, x=-1.0, dt=1.0)
    assert target.cylinder_angle_deg == pytest.approx(50.0)
    assert target.layer.placement().central_angle_rad == pytest.approx(
        math.radians(50.0)
    )
    assert target.cylinder_radius_m == radius0  # untouched

    _press_left(controls, y=1.0, dt=0.25)
    assert target.placement_config.offset_y == pytest.approx(0.25)


def test_equirect_stick_drives_spans_symmetrically():
    target = _switchable_target("equirect")
    controls, _ = _make(
        [target], ControlsConfig(deadzone=0.0, angle_rate_deg_per_s=40.0)
    )
    _press_left(controls, x=-1.0, y=-1.0, dt=1.0)
    assert target.equirect_h_deg == pytest.approx(320.0)
    assert target.equirect_v_half_deg == pytest.approx(50.0)
    placement = target.layer.placement()
    assert placement.upper_vertical_angle_rad == pytest.approx(
        -placement.lower_vertical_angle_rad
    )


@pytest.mark.parametrize(
    "shape,axis,attr,limit_name,to_max",
    [
        ("cylinder", "x", "cylinder_angle_deg", "cylinder_angle_range_deg", True),
        ("cylinder", "x", "cylinder_angle_deg", "cylinder_angle_range_deg", False),
        ("equirect", "x", "equirect_h_deg", "equirect_h_range_deg", True),
        ("equirect", "y", "equirect_v_half_deg", "equirect_v_half_range_deg", False),
    ],
)
def test_shape_params_clamp_inside_layer_validation(
    shape, axis, attr, limit_name, to_max
):
    """Out-of-range values make set_placement raise on the render thread, so
    the clamp is what keeps a held stick from killing the run."""
    target = _switchable_target(shape)
    cfg = ControlsConfig(deadzone=0.0, size_rate_m_per_s=1e4, angle_rate_deg_per_s=1e4)
    controls, _ = _make([target], cfg)
    kwargs = {axis: 1.0 if to_max else -1.0}
    _press_left(controls, dt=1.0, **kwargs)
    lo, hi = getattr(cfg, limit_name)
    assert getattr(target, attr) == pytest.approx(hi if to_max else lo)


def test_default_limits_sit_inside_the_layer_specs():
    """cylinder needs (0, 2pi) exclusive; equirect (0, 2pi] and |v| <= pi/2."""
    cfg = ControlsConfig()
    lo, hi = cfg.cylinder_angle_range_deg
    assert lo > 0.0 and hi < 360.0
    lo_h, hi_h = cfg.equirect_h_range_deg
    assert lo_h > 0.0 and hi_h <= 360.0
    lo_v, hi_v = cfg.equirect_v_half_range_deg
    assert lo_v > 0.0 and hi_v <= 90.0


# ── Left Y: reset ─────────────────────────────────────────────────────


def test_left_y_resets_shape_params_and_modes():
    target = _switchable_target("quad")
    cfg = target.placement_config
    controls, _ = _make(
        [target],
        ControlsConfig(
            deadzone=0.0,
            size_rate_m_per_s=1.0,
            plane_distance_rate_cm_per_s=10.0,
        ),
    )
    size0, distance0 = cfg.size_meters, cfg.distance

    _press_left(controls, a=True)  # -> cylinder
    _press(controls, a=True)  # lock mode moves on
    _press(controls, b=True)  # -> mono
    _press(controls, x=1.0, dt=1.0)  # baseline drifts
    _press_left(controls, x=1.0, dt=1.0)
    assert target.shape == "cylinder"

    _press_left(controls, b=True)
    assert target.shape == "quad"
    assert target.shape_layers["quad"].visible
    assert not target.shape_layers["cylinder"].visible
    assert target.lock_mode == "lazy"
    assert target.plane_distance_cm == 0.0
    assert not target.force_mono.is_set()
    assert cfg.size_meters == size0 and cfg.distance == distance0
    assert target.cylinder_radius_m == pytest.approx(2.0)


# ── Logging ───────────────────────────────────────────────────────────


class RecordingHud:
    def __init__(self):
        self.lines = []

    def show(self, lines):
        self.lines.append(lines[0])

    def step(self, dt, head_pose):
        pass


def test_hud_shows_every_step_not_the_log_rate():
    """The readout is live. Throttling it made a value moving in 0.1 cm steps
    look like it moved in whatever the log period times the rate came to."""
    target = _stereo_target(plane_distance=0.0)
    controls, _ = _make(
        [target], ControlsConfig(deadzone=0.0, plane_distance_rate_cm_per_s=2.0)
    )
    hud = RecordingHud()
    controls._hud = hud
    for _ in range(60):  # one second at 60 Hz
        _press(controls, x=1.0, dt=1.0 / 60.0)

    values = [float(line.split()[2]) for line in hud.lines]
    assert len(values) == 20  # 2 cm/s in 0.1 cm steps
    assert {round(b - a, 3) for a, b in zip(values, values[1:])} == {0.1}


def test_held_stick_log_is_throttled(capsys):
    """At display rate an unthrottled log would print ~90 lines a second."""
    target = _stereo_target()
    controls, _ = _make(
        [target], ControlsConfig(deadzone=0.0, plane_distance_rate_cm_per_s=1.0)
    )
    for _ in range(60):
        _press(controls, x=1.0, dt=1.0 / 60.0)
    lines = [ln for ln in capsys.readouterr().err.splitlines() if "stereo planes" in ln]
    assert 1 <= len(lines) <= 6


# ── Message summarising (multi-camera) ────────────────────────────────


def test_summarize_says_a_shared_value_once():
    """Bindings apply to every camera at once, so values normally agree —
    repeating them per camera is what overflowed the HUD bar."""
    from controls import summarize

    assert summarize([("zed", "52.0 mm")]) == "52.0 mm"
    assert summarize([("a", "52.0 mm"), ("b", "52.0 mm")]) == "52.0 mm  (2 cameras)"


def test_summarize_lists_cameras_that_disagree():
    from controls import summarize

    out = summarize([("front", "52.0 mm"), ("rear", "10.0 mm")])
    assert "front 52.0 mm" in out and "rear 10.0 mm" in out
    assert "cameras)" not in out


def test_summarize_handles_nothing_changing():
    from controls import summarize

    assert summarize([]) == ""


def test_multi_camera_baseline_message_stays_one_line(capsys):
    """End-to-end: three cameras must not produce three repeated values."""
    from controls.hud import _TEXT_W, _TITLE, split_message

    targets = [_stereo_target(n) for n in ("front", "left", "right")]
    controls, _ = _make(targets, ControlsConfig(deadzone=0.0))
    _press(controls, x=1.0, dt=1.0)

    line = [ln for ln in capsys.readouterr().err.splitlines() if "stereo planes" in ln][
        -1
    ]
    # stderr joins the panel's two lines with " | "; the HUD gets them
    # separately, so the headline is what has to fit.
    headline = line.split("controls: ", 1)[1].split("  |  ")[0]
    assert "(3 cameras)" in headline
    assert len(split_message(headline)) == 1
    assert _TITLE.getlength(headline) <= _TEXT_W


# ── Suggested plane distance ──────────────────────────────────────────


def test_suggestion_comes_from_the_plane_distance_and_ipd():
    """d = ipd * (1 - Z / FAR_TARGET_M): a 1 m plane at a 63 mm IPD suggests
    6.3 * 0.9 = 5.67 cm."""
    from controls import FAR_TARGET_M

    target = _stereo_target()
    target.placement_config = PlacementConfig(distance=1.0)
    controls, _ = _make([target])
    controls.step(0.0, 63.0)
    from controls import PLANE_DISTANCE_STEP_CM

    raw = 6.3 * (1.0 - 1.0 / FAR_TARGET_M)
    expected = round(raw / PLANE_DISTANCE_STEP_CM) * PLANE_DISTANCE_STEP_CM
    # Stepped, so it is a value the stick can actually land on.
    assert controls._suggested_plane_distance_cm(target) == pytest.approx(expected)


def test_suggestion_moves_with_the_plane_distance():
    """A nearer plane needs a wider gap to reach the same far end."""
    near, far = _stereo_target("near"), _stereo_target("far")
    near.placement_config = PlacementConfig(distance=0.8)
    far.placement_config = PlacementConfig(distance=3.0)
    controls, _ = _make([near, far])
    controls.step(0.0, 63.0)
    assert controls._suggested_plane_distance_cm(
        near
    ) > controls._suggested_plane_distance_cm(far)


def test_suggestion_moves_with_the_ipd():
    target = _stereo_target()
    target.placement_config = PlacementConfig(distance=1.0)
    controls, _ = _make([target])
    controls.step(0.0, 58.0)
    narrow = controls._suggested_plane_distance_cm(target)
    controls.step(0.0, 70.0)
    assert controls._suggested_plane_distance_cm(target) > narrow


def test_suggestion_never_exceeds_the_divergence_ceiling():
    target = _stereo_target()
    target.placement_config = PlacementConfig(distance=0.05)  # absurdly near
    controls, _ = _make([target])
    controls.step(0.0, 63.0)
    from controls import PLANE_DISTANCE_STEP_CM

    # Rounding to a step can land half a step above the raw ceiling.
    assert controls._suggested_plane_distance_cm(target) <= (
        controls._max_plane_distance_cm() + PLANE_DISTANCE_STEP_CM / 2
    )


def test_suggestion_is_only_advice_never_applied():
    """The value in use stays whatever was configured or trimmed to."""
    target = _stereo_target(plane_distance=0.0)
    target.placement_config = PlacementConfig(distance=1.0)
    controls, _ = _make([target])
    for _ in range(5):
        controls.step(0.0, 63.0)
    assert target.plane_distance_cm == 0.0


def test_suggestion_is_shown_next_to_the_value(capsys):
    target = _stereo_target()
    target.placement_config = PlacementConfig(distance=1.0)
    controls, _ = _make([target], ControlsConfig(deadzone=0.0))
    _press(controls, x=1.0, dt=0.5)
    err = capsys.readouterr().err
    assert "stereo planes" in err and "suggested" in err and "IPD" in err


def test_equirect_has_no_suggestion():
    """A sphere centred on the viewer has no plane distance to work from."""
    target = _switchable_target("equirect")
    controls, _ = _make([target])
    assert controls._suggested_plane_distance_cm(target) is None


def test_stick_still_moves_at_display_frame_rates():
    """One 60 Hz frame moves less than half a step, so rounding the running
    total each frame would pin the value at its start forever."""
    target = _stereo_target(plane_distance=0.0)
    controls, _ = _make(
        [target], ControlsConfig(deadzone=0.0, plane_distance_rate_cm_per_s=2.0)
    )
    for _ in range(60):  # one second at 60 Hz
        _press(controls, x=1.0, dt=1.0 / 60.0)
    assert target.plane_distance_cm == pytest.approx(2.0, abs=0.05)


def test_the_layer_only_ever_sees_whole_steps():
    from controls import PLANE_DISTANCE_STEP_CM

    target = _stereo_target(plane_distance=0.0)
    controls, _ = _make(
        [target], ControlsConfig(deadzone=0.0, plane_distance_rate_cm_per_s=2.0)
    )
    for _ in range(60):
        _press(controls, x=1.0, dt=1.0 / 60.0)
        mm = target.layer.baseline_mm
        steps = mm / (PLANE_DISTANCE_STEP_CM * 10.0)
        assert steps == pytest.approx(round(steps), abs=1e-6), mm


# ── Mono / stereo and the plane gap ───────────────────────────────────


def test_mono_parks_the_gap_and_stereo_puts_it_back():
    """Both eyes see one image in mono, so a gap would only shove that flat
    image to another depth."""
    target = _stereo_target(plane_distance=5.0)
    controls, _ = _make([target])

    _press(controls, b=True)  # -> mono
    assert target.plane_distance_cm == 0.0
    assert target.layer.baseline_mm == 0.0

    _press(controls, b=False)
    _press(controls, b=True)  # -> stereo
    assert target.plane_distance_cm == pytest.approx(5.0)
    assert target.layer.baseline_mm == pytest.approx(50.0)


def test_a_gap_set_while_mono_is_not_clobbered_on_the_way_out():
    target = _stereo_target(plane_distance=5.0)
    controls, _ = _make(
        [target], ControlsConfig(deadzone=0.0, plane_distance_rate_cm_per_s=2.0)
    )
    _press(controls, b=True)
    _press(controls, b=False)
    _press(controls, b=True)
    assert target.plane_distance_cm == pytest.approx(5.0)


def test_reset_clears_the_remembered_mono_gap():
    target = _stereo_target(plane_distance=5.0)
    controls, _ = _make([target])
    _press(controls, b=True)  # -> mono, gap parked
    _press_left(controls, b=True)  # Y: reset
    assert target.plane_distance_cm == pytest.approx(5.0)
    assert target.plane_distance_before_mono is None
    assert not target.force_mono.is_set()


def test_equirect_gap_is_skipped_because_it_does_nothing():
    """The gap shifts each eye's surface, and camera_viz's sphere is at
    infinite radius — translating that changes nothing, so the stick must not
    pretend otherwise."""
    target = _switchable_target("equirect")
    controls, _ = _make([target], ControlsConfig(deadzone=0.0))
    _press(controls, x=1.0, dt=1.0)
    assert target.plane_distance_cm == 0.0
    assert target.layer.baseline_mm == 0.0


def test_quad_and_cylinder_axes_do_different_things():
    """The point of the rework: each shape's two axes must not collapse into
    one control."""
    from controls import SHAPE_PARAMS

    for shape, (x_name, y_name) in SHAPE_PARAMS.items():
        assert x_name != y_name, shape


# ── Retuning must land immediately in every lock mode ─────────────────


@pytest.mark.parametrize("lock_mode", ["world", "head", "gimbal", "lazy"])
def test_height_change_moves_the_plane_now_not_at_the_next_resnap(lock_mode):
    """world cached the finished placement and ignored retuning outright;
    lazy only recomputed on a re-snap, so a height change sat unapplied until
    you happened to look away."""
    from placements import PlacementConfig, build

    head_pos, head_q = (0.0, 1.5, 0.0), (1.0, 0.0, 0.0, 0.0)
    cfg = PlacementConfig(distance=1.0)
    strategy = build(lock_mode, cfg)
    before = strategy.update(head_pos, head_q).position[1]

    strategy.retune(dataclasses_replace(cfg, offset_y=0.4))
    after = strategy.update(head_pos, head_q).position[1]
    assert after == pytest.approx(before + 0.4, abs=1e-6), lock_mode


@pytest.mark.parametrize("lock_mode", ["world", "head", "gimbal", "lazy"])
def test_size_change_lands_immediately_too(lock_mode):
    from placements import PlacementConfig, build

    head_pos, head_q = (0.0, 1.5, 0.0), (1.0, 0.0, 0.0, 0.0)
    cfg = PlacementConfig(distance=1.0)
    strategy = build(lock_mode, cfg)
    strategy.update(head_pos, head_q)

    strategy.retune(dataclasses_replace(cfg, size_meters=(2.0, 1.125)))
    assert strategy.update(head_pos, head_q).size_meters == (2.0, 1.125), lock_mode


def test_world_lock_still_pins_the_plane_across_a_retune():
    """Retuning must not become an excuse to re-place a world-locked plane."""
    from placements import PlacementConfig, build

    cfg = PlacementConfig(distance=1.0)
    strategy = build("world", cfg)
    placed = strategy.update((0.0, 1.5, 0.0), (1.0, 0.0, 0.0, 0.0)).position

    strategy.retune(dataclasses_replace(cfg, size_meters=(2.0, 1.125)))
    # Head has since moved; the plane must not follow it.
    after = strategy.update((3.0, 1.5, -2.0), (1.0, 0.0, 0.0, 0.0)).position
    assert after == pytest.approx(placed)


def test_the_plane_gap_follows_a_shape_switch():
    """It is tracked per camera but applied per layer, so a switch has to
    carry it onto the layer that just became visible."""
    target = _switchable_target("quad")
    target.plane_distance_cm = 5.0
    controls, _ = _make([target], ControlsConfig(deadzone=0.0))
    for layer in target.shape_layers.values():
        layer.baseline_mm = 99.0  # as if built from a different YAML value

    _press_left(controls, a=True)  # -> cylinder
    assert target.layer.baseline_mm == pytest.approx(50.0)


def test_a_shape_switch_while_mono_keeps_the_gap_parked():
    target = _switchable_target("quad")
    target.plane_distance_cm = 5.0
    controls, _ = _make([target])
    for layer in target.shape_layers.values():
        layer.baseline_mm = 99.0

    _press(controls, b=True)  # -> mono, gap parked at 0
    _press_left(controls, a=True)  # -> cylinder
    assert target.layer.baseline_mm == 0.0

    _press(controls, b=False)
    _press(controls, b=True)  # -> stereo again
    assert target.layer.baseline_mm == pytest.approx(50.0)


def test_reset_restores_height_too():
    """It restored size and distance but not offset_y, so a nudged plane
    stayed nudged."""
    target = _switchable_target("quad")
    controls, _ = _make([target], ControlsConfig(deadzone=0.0, offset_rate_m_per_s=1.0))
    _press_left(controls, y=1.0, dt=0.5)
    assert target.placement_config.offset_y == pytest.approx(0.5)

    _press_left(controls, b=True)  # Y: reset
    assert target.placement_config.offset_y == pytest.approx(0.0)


# ── Equirect aiming: pan + recenter ───────────────────────────────────


def _looking(yaw_deg: float) -> FakeVizSession:
    return FakeVizSession(FakePose3D(yaw_quat(math.radians(yaw_deg))))


def test_right_stick_pans_an_equirect_instead_of_setting_a_gap():
    """The sphere has no gap to set, so the stick is free -- and panning is
    what a 360 feed actually needs, since the camera rarely faces the way the
    headset did when it recentered."""
    target = _switchable_target("equirect")
    controls, _ = _make(
        [target], ControlsConfig(deadzone=0.0, angle_rate_deg_per_s=40.0)
    )

    _press(controls, x=1.0, dt=0.5)
    assert target.equirect_yaw_deg == pytest.approx(20.0)
    placement = target.shape_layers["equirect"].placement()
    assert placement.pose.orientation == pytest.approx(yaw_quat(math.radians(20.0)))


def test_panning_wraps_instead_of_running_off():
    target = _switchable_target("equirect")
    target.equirect_yaw_deg = 170.0
    controls, _ = _make(
        [target], ControlsConfig(deadzone=0.0, angle_rate_deg_per_s=40.0)
    )
    _press(controls, x=1.0, dt=1.0)
    assert target.equirect_yaw_deg == pytest.approx(-150.0)


def test_the_stick_does_not_pan_a_shape_that_has_a_gap():
    target = _switchable_target("quad")
    controls, _ = _make([target], ControlsConfig(deadzone=0.0))
    _press(controls, x=1.0, dt=1.0)
    assert target.equirect_yaw_deg == 0.0


def test_stick_click_aims_the_panorama_where_you_are_looking():
    target = _switchable_target("equirect")
    controls, _ = _make([target])
    controls._session = _looking(35.0)

    _press(controls, click=True)
    assert target.equirect_yaw_deg == pytest.approx(35.0, abs=1e-3)


def test_stick_click_resnaps_a_placed_layer():
    """Same gesture, read per shape: a world-locked plane left in another
    part of the room comes back."""
    target = _switchable_target("quad", name="cam")
    target.lock_mode = "world"
    controls, strategies = _make([target])
    controls._session = _looking(0.0)
    before = strategies[0]

    _press(controls, click=True)
    assert strategies[0] is not before


def test_recenter_is_a_press_not_a_hold():
    target = _switchable_target("equirect")
    controls, _ = _make([target])
    controls._session = _looking(35.0)
    _press(controls, click=True)
    target.equirect_yaw_deg = 0.0
    _press(controls, click=True)  # still held
    assert target.equirect_yaw_deg == 0.0


def test_recenter_does_nothing_without_a_head_pose():
    """Tracking loss must not slam the panorama to a default heading."""
    target = _switchable_target("equirect")
    target.equirect_yaw_deg = 90.0
    controls, _ = _make([target])  # FakeVizSession head is None
    _press(controls, click=True)
    assert target.equirect_yaw_deg == 90.0


def test_reset_restores_the_panorama_heading():
    target = _switchable_target("equirect")
    controls, _ = _make(
        [target], ControlsConfig(deadzone=0.0, angle_rate_deg_per_s=40.0)
    )
    _press(controls, x=1.0, dt=1.0)
    assert target.equirect_yaw_deg != 0.0

    _press_left(controls, b=True)  # Y: reset
    assert target.equirect_yaw_deg == pytest.approx(0.0)
    placement = target.shape_layers["equirect"].placement()
    assert placement.pose.orientation == pytest.approx((1.0, 0.0, 0.0, 0.0))


def test_heading_is_the_inverse_of_the_yaw_it_is_built_from():
    for degrees in (-179.0, -90.0, 0.0, 45.0, 179.0):
        assert heading_deg(yaw_quat(math.radians(degrees))) == pytest.approx(
            degrees, abs=1e-3
        )


# ── The panel owns stderr while it is live ────────────────────────────


def test_control_messages_stay_off_stderr_when_the_panel_owns_it(capsys):
    """Two writers on one stream is what left a column of half-erased
    headers scrolling up the terminal."""
    target = _switchable_target("quad")
    controls = ControllerControls(
        FakeVizSession(),
        [target],
        [build_placement(target.lock_mode, target.placement_config)],
        ControlsConfig(),
        tracker=FakeTracker(),
        log_to_stderr=False,
    )
    controls._device_session = FakeSession()
    _press(controls, a=True)

    assert capsys.readouterr().err == ""
    # The event still reaches the panel, which is what draws it.
    assert "lock" in controls.last_event
