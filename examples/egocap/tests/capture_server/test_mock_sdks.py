# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the two mock SDK approximation files
(capture_server.mock.mock_isaac_capture, capture_server.mock.mock_depthai)
and the production adapters that consume them -- retargeted from an
earlier generator-based mock design that exercised five separate
generator files (operator/tracking/status/camera/faults.py), since
removed in favor of the two SDK-shaped mock files this suite exercises.

Deliberately NOT protocol-shaped-generator tests: most assertions here
either call the mock SDK classes directly in their own native shape
(TeleopSession.step()/get_status(), Pipeline/MessageQueue), or go through
the real IsaacCaptureAdapter/DepthAIAdapter/Acquisition path via
capture_server.launch.build_mock_source_factory -- the same adapter path
production code uses, not a protocol-shaped generator called directly.
"""

from __future__ import annotations

import numpy as np
import pytest

from capture_server import protocol as p
from capture_server.acquisition import Acquisition, CameraSourceError, DepthAIAdapter, SourceError, isaac_capture
from capture_server.acquisition.tensor_indices import (
    ControllerInputIndex,
    HandInputIndex,
    HandJointIndex,
    HeadInputIndex,
)
from capture_server.launch import build_mock_source_factory
from capture_server.mock import mock_depthai as dai
from capture_server.mock import mock_isaac_capture as mic

# ---------------------------------------------------------------------------
# mock_isaac_capture: deterministic motion math (private, white-box).
# ---------------------------------------------------------------------------


def test_operator_rejects_unknown_scenario():
    with pytest.raises(ValueError, match="unknown scenario"):
        mic._Operator(scenario="not-a-scenario")


@pytest.mark.parametrize("scenario", mic.SCENARIOS)
def test_operator_core_is_deterministic_for_seed_and_time(scenario):
    a = mic._Operator(scenario=scenario, seed=7)
    b = mic._Operator(scenario=scenario, seed=7)
    head_a, left_a, right_a = a.core(2.5)
    head_b, left_b, right_b = b.core(2.5)
    np.testing.assert_array_equal(head_a, head_b)
    np.testing.assert_array_equal(left_a.joints, left_b.joints)
    np.testing.assert_array_equal(right_a.joints, right_b.joints)


def test_operator_different_seeds_diverge():
    a = mic._Operator(scenario="demo", seed=1)
    b = mic._Operator(scenario="demo", seed=2)
    head_a, _, _ = a.core(3.0)
    head_b, _, _ = b.core(3.0)
    assert not np.array_equal(head_a, head_b)


def test_operator_head_height_is_near_configured_constant():
    operator = mic._Operator(scenario="idle", seed=0)
    head, _, _ = operator.core(0.0)
    assert abs(float(head[1]) - mic.HEAD_HEIGHT_M) < 0.05


def test_rotation_helpers_round_trip_through_quaternion():
    for fn, angle in ((mic._rot_x, 0.3), (mic._rot_y, -0.6), (mic._rot_z, 1.1)):
        m = fn(angle)
        q = mic._mat_to_quat(m)
        assert q.shape == (4,)
        np.testing.assert_allclose(np.linalg.norm(q), 1.0, atol=1e-9)
        assert q[3] >= 0.0


def test_mat_to_quat_covers_every_largest_diagonal_branch():
    for matrix in (
        np.diag([1.0, -1.0, -1.0]),
        np.diag([-1.0, 1.0, -1.0]),
        np.diag([-1.0, -1.0, 1.0]),
    ):
        q = mic._mat_to_quat(matrix)
        assert q.shape == (4,)
        np.testing.assert_allclose(np.linalg.norm(q), 1.0, atol=1e-9)


def test_min_jerk_clamps_and_is_monotonic_increasing():
    assert mic._min_jerk(-1.0) == 0.0
    assert mic._min_jerk(2.0) == 1.0
    xs = [0.0, 0.25, 0.5, 0.75, 1.0]
    ys = [mic._min_jerk(x) for x in xs]
    assert ys == sorted(ys)


def test_hand_joint_radii_matches_joint_name_count_and_order():
    assert len(mic.HAND_JOINT_RADII) == 26
    assert mic.HAND_JOINT_RADII[0] == 0.045  # PALM
    assert mic.HAND_JOINT_RADII[1] == 0.014  # WRIST
    assert mic.HAND_JOINT_RADII[-1] == 0.005  # LITTLE_TIP


def test_joint_radius_rejects_unrecognized_name():
    with pytest.raises(AssertionError):
        mic._joint_radius("NOT_A_JOINT")


def test_hand_joint_index_matches_protocol_openxr_layout():
    assert p.HAND_JOINT_NAMES[0] == "PALM"
    assert p.HAND_JOINT_NAMES[1] == "WRIST"
    assert [j.name for j in HandJointIndex] == list(p.HAND_JOINT_NAMES)


# ---------------------------------------------------------------------------
# OptionalTensorGroup: the real is_none/__getitem__/__setitem__ contract.
# ---------------------------------------------------------------------------


def test_optional_tensor_group_starts_absent_and_raises_on_read():
    group = mic.OptionalTensorGroup(len(HeadInputIndex))
    assert group.is_none
    with pytest.raises(ValueError):
        _ = group[HeadInputIndex.POSITION]


def test_optional_tensor_group_setitem_makes_it_present():
    group = mic.OptionalTensorGroup(len(HeadInputIndex))
    group[HeadInputIndex.POSITION] = np.zeros(3, dtype=np.float32)
    assert not group.is_none
    np.testing.assert_array_equal(group[HeadInputIndex.POSITION], np.zeros(3, dtype=np.float32))


# ---------------------------------------------------------------------------
# TeleopSession: construction, step(), fault injection, status.
# ---------------------------------------------------------------------------


def _full_config() -> mic.TeleopSessionConfig:
    head = mic.HeadSource(name="head")
    controllers = mic.ControllersSource(name="controllers")
    hands = mic.HandsSource(name="hands")
    pipeline = mic.OutputCombiner(
        {
            "head": head.output("head"),
            "controller_left": controllers.output(mic.ControllersSource.LEFT),
            "controller_right": controllers.output(mic.ControllersSource.RIGHT),
            "hand_left": hands.output(mic.HandsSource.LEFT),
            "hand_right": hands.output(mic.HandsSource.RIGHT),
        }
    )
    plugins = [mic.PluginConfig(plugin_name=mic.MANUS_PLUGIN_NAME, plugin_root_id="manus", search_paths=[])]
    return mic.TeleopSessionConfig(app_name="test", pipeline=pipeline, plugins=plugins)


def test_mock_manus_plugin_name_is_the_manifest_name_the_adapter_requests():
    assert mic.MANUS_PLUGIN_NAME == isaac_capture.MANUS_PLUGIN_NAME == "manus_hand_plugin"


def test_mock_ignores_a_plugin_request_that_does_not_use_the_manifest_name():
    plugins = [mic.PluginConfig(plugin_name="Manus", plugin_root_id="manus", search_paths=[])]
    config = mic.TeleopSessionConfig(
        app_name="test",
        pipeline=mic.OutputCombiner({"head": mic.HeadSource(name="head").output("head")}),
        plugins=plugins,
    )
    with mic.TeleopSession(config, scenario="demo", seed=0) as session:
        status = session.get_status()
    assert all(pr.type != mic.ProviderType.PLUGIN for pr in status.providers)


def test_teleop_session_step_before_enter_raises():
    session = mic.TeleopSession(_full_config(), scenario="demo", seed=0)
    with pytest.raises(RuntimeError, match="before __enter__"):
        session.step()


def test_teleop_session_step_returns_only_configured_leaves():
    config = mic.TeleopSessionConfig(
        app_name="test",
        pipeline=mic.OutputCombiner({"head": mic.HeadSource(name="head").output("head")}),
    )
    with mic.TeleopSession(config, scenario="demo", seed=0) as session:
        result = session.step()
    assert set(result) == {"head"}
    assert not result["head"].is_none
    assert result["head"][HeadInputIndex.POSITION].shape == (3,)


def test_teleop_session_step_populates_every_group_shape():
    with mic.TeleopSession(_full_config(), scenario="demo", seed=1) as session:
        result = session.step()
    assert result["head"][HeadInputIndex.ORIENTATION].shape == (4,)
    assert len(ControllerInputIndex) == 14
    assert result["controller_left"][ControllerInputIndex.GRIP_POSITION].shape == (3,)
    assert result["hand_right"][HandInputIndex.JOINT_POSITIONS].shape == (26, 3)
    assert result["hand_right"][HandInputIndex.JOINT_RADII].shape == (26,)


def test_teleop_session_step_is_deterministic_for_a_fixed_clock():
    class _FixedClock:
        def monotonic(self) -> float:
            return 10.0

    config = mic.TeleopSessionConfig(
        app_name="test",
        pipeline=mic.OutputCombiner({"head": mic.HeadSource(name="head").output("head")}),
    )
    with mic.TeleopSession(config, scenario="demo", seed=3, clock=_FixedClock()) as session:
        a = session.step()
    with mic.TeleopSession(config, scenario="demo", seed=3, clock=_FixedClock()) as session:
        b = session.step()
    np.testing.assert_array_equal(a["head"][HeadInputIndex.POSITION], b["head"][HeadInputIndex.POSITION])


def test_startup_error_fault_raises_on_enter():
    session = mic.TeleopSession(_full_config(), scenario="demo", seed=0, faults={"startup_error"})
    with pytest.raises(RuntimeError, match="startup_error"):
        session.__enter__()


def test_fatal_acquisition_error_fault_raises_after_threshold_steps():
    config = mic.TeleopSessionConfig(
        app_name="test",
        pipeline=mic.OutputCombiner({"head": mic.HeadSource(name="head").output("head")}),
    )
    with mic.TeleopSession(config, scenario="demo", seed=0, faults={"fatal_acquisition_error"}) as session:
        for _ in range(session._FATAL_AFTER_STEPS - 1):
            session.step()
        with pytest.raises(RuntimeError, match="exited"):
            session.step()


def test_set_faults_takes_effect_live_on_an_already_entered_session():
    config = mic.TeleopSessionConfig(
        app_name="test",
        pipeline=mic.OutputCombiner({"head": mic.HeadSource(name="head").output("head")}),
    )
    with mic.TeleopSession(config, scenario="demo", seed=0) as session:
        session.step()  # no fault yet
        session.set_faults({"fatal_acquisition_error"})
        session.frame_count = session._FATAL_AFTER_STEPS
        with pytest.raises(RuntimeError):
            session.step()


def test_get_status_omits_manus_without_plugin_request():
    config = mic.TeleopSessionConfig(
        app_name="test",
        pipeline=mic.OutputCombiner({"head": mic.HeadSource(name="head").output("head")}),
    )
    with mic.TeleopSession(config, scenario="demo", seed=0) as session:
        status = session.get_status()
    assert all(pr.name != mic.MANUS_PLUGIN_NAME for pr in status.providers)
    assert all("hand" not in d.id for d in status.devices)


def test_get_status_includes_manus_when_plugin_enabled():
    with mic.TeleopSession(_full_config(), scenario="demo", seed=0) as session:
        status = session.get_status()
    manus = next(pr for pr in status.providers if pr.name == mic.MANUS_PLUGIN_NAME)
    assert manus.status == mic.ProviderState.AVAILABLE
    hand_devices = [d for d in status.devices if d.provider_id == manus.id]
    assert len(hand_devices) == 2


def test_get_status_declares_controllers_regardless_of_leaf_label():
    """Controller device inventory must be derived from the configured
    ControllersSource + its LEFT/RIGHT selector, not from whatever leaf
    label an application's OutputCombiner dict happens to use -- an
    application is free to key its graph's outputs with arbitrary
    strings; that labeling must never make an otherwise-selected real
    controller vanish from inventory."""
    controllers = mic.ControllersSource(name="controllers")
    config = mic.TeleopSessionConfig(
        app_name="test",
        pipeline=mic.OutputCombiner(
            {
                "my_left_stick": controllers.output(mic.ControllersSource.LEFT),
                "weird-key-2": controllers.output(mic.ControllersSource.RIGHT),
            }
        ),
    )
    with mic.TeleopSession(config, scenario="demo", seed=0) as session:
        status = session.get_status()
        fields = session.step()
    device_ids = {d.id for d in status.devices}
    assert "device-controller-left" in device_ids
    assert "device-controller-right" in device_ids
    assert set(fields) == {"my_left_stick", "weird-key-2"}


def test_provider_failed_fault_degrades_manus_and_both_hand_devices():
    with mic.TeleopSession(_full_config(), scenario="demo", seed=0, faults={"provider_failed"}) as session:
        status = session.get_status()
    manus = next(pr for pr in status.providers if pr.name == mic.MANUS_PLUGIN_NAME)
    assert manus.status == mic.ProviderState.FAILED
    hands = [d for d in status.devices if d.provider_id == manus.id]
    assert all(d.status == mic.DeviceState.UNAVAILABLE for d in hands)


def test_sample_freeze_fault_repeats_head_position():
    config = mic.TeleopSessionConfig(
        app_name="test",
        pipeline=mic.OutputCombiner({"head": mic.HeadSource(name="head").output("head")}),
    )
    positions = []

    class _AdvancingClock:
        def __init__(self) -> None:
            self.t = 0.0

        def monotonic(self) -> float:
            self.t += 0.05
            return self.t

    clock = _AdvancingClock()
    with mic.TeleopSession(config, scenario="demo", seed=0, faults={"sample_freeze"}, clock=clock) as session:
        for _ in range(3):
            positions.append(np.array(session.step()["head"][HeadInputIndex.POSITION]))
    assert all(np.array_equal(positions[0], pos) for pos in positions[1:])


# ---------------------------------------------------------------------------
# mock_depthai: Pipeline/MessageQueue, faults, pacing, calibration.
# ---------------------------------------------------------------------------


def test_pipeline_rejects_unknown_device():
    pipeline = dai.Pipeline(device_id="not-a-real-device", available_device_ids=("mock-oakd-0",))
    with pytest.raises(RuntimeError, match="not found"):
        pipeline.__enter__()


def test_camera_required_failure_fault_raises_on_enter():
    pipeline = dai.Pipeline(device_id="mock-oakd-0", faults={"camera_required_failure"})
    with pytest.raises(RuntimeError, match="camera_required_failure"):
        pipeline.__enter__()


class _FakeClock:
    """A fully controllable clock -- no real-time wall-clock race:
    ``advance()`` is the only thing that ever moves time forward."""

    def __init__(self, t: float = 0.0) -> None:
        self.t = t

    def monotonic(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


def _build_pipeline(**kwargs) -> tuple[dai.Pipeline, "dai.MessageQueue", _FakeClock]:
    clock = _FakeClock()
    pipeline = dai.Pipeline(device_id="mock-oakd-0", seed=1, clock=clock, **kwargs)
    pipeline.__enter__()
    camera = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
    output = camera.requestOutput((64, 48))
    encoder = pipeline.create(dai.node.VideoEncoder).build(
        output, frameRate=10.0, profile=dai.VideoEncoderProperties.Profile.MJPEG
    )
    queue = encoder.out.createOutputQueue()
    pipeline.start()
    return pipeline, queue, clock


def test_pipeline_produces_a_decodable_jpeg_once_running():
    pipeline, queue, clock = _build_pipeline()
    try:
        # The first poll always fires immediately (pacing starts counting
        # from the first tryGet()/has() call, not from start()).
        frame = queue.tryGet()
        assert frame is not None
        assert queue.tryGet() is None  # nothing new due yet
        clock.advance(0.1)  # exactly one interval at frameRate=10.0
        frame = queue.tryGet()
        assert frame is not None
        data = frame.getData()
        assert data[:2] == b"\xff\xd8"
        assert data[-2:] == b"\xff\xd9"
        assert frame.getWidth() == 64
        assert frame.getHeight() == 48
    finally:
        pipeline.stop()


def test_queue_returns_none_before_pipeline_is_running():
    clock = _FakeClock()
    pipeline = dai.Pipeline(device_id="mock-oakd-0", seed=1, clock=clock)
    pipeline.__enter__()
    camera = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
    output = camera.requestOutput((64, 48))
    encoder = pipeline.create(dai.node.VideoEncoder).build(
        output, frameRate=10.0, profile=dai.VideoEncoderProperties.Profile.MJPEG
    )
    queue = encoder.out.createOutputQueue()
    # Deliberately never call pipeline.start().
    clock.advance(1.0)
    assert queue.tryGet() is None


def test_queue_skips_overdue_slots_without_catch_up_bursting():
    pipeline, queue, clock = _build_pipeline()
    try:
        clock.advance(0.1)
        first = queue.tryGet()
        assert first is not None
        clock.advance(0.5)  # several intervals at frameRate=10.0
        second = queue.tryGet()
        assert second is not None
        # Only one new frame should be available per poll, never a
        # backlog -- immediately polling again (before the next interval)
        # must be empty, not another queued catch-up frame.
        assert queue.tryGet() is None
    finally:
        pipeline.stop()


def test_queue_honors_max_size_by_dropping_the_oldest_buffered_frame():
    pipeline, queue, clock = _build_pipeline()
    try:
        queue.setMaxSize(2)
        # Advance past three intervals before ever draining -- only the
        # two most recent buffered frames should survive.
        clock.advance(0.1)
        queue.has()  # drive _advance() without consuming, to buffer a frame
        clock.advance(0.1)
        queue.has()
        clock.advance(0.1)
        queue.has()
        sequence_nums = []
        while True:
            frame = queue.tryGet()
            if frame is None:
                break
            sequence_nums.append(frame.getSequenceNum())
        assert len(sequence_nums) == 2
        assert sequence_nums == sorted(sequence_nums)  # oldest-first, newest two retained
    finally:
        pipeline.stop()


def test_camera_optional_failure_fault_raises_from_fill():
    pipeline, queue, clock = _build_pipeline(faults={"camera_optional_failure"})
    try:
        clock.advance(0.1)
        with pytest.raises(RuntimeError, match="camera_optional_failure"):
            queue.tryGet()
    finally:
        pipeline.stop()


def test_queue_closes_and_stops_producing_after_pipeline_stop():
    pipeline, queue, clock = _build_pipeline()
    pipeline.stop()
    clock.advance(1.0)
    assert queue.tryGet() is None


def test_get_camera_intrinsics_for_bound_socket():
    pipeline = dai.Pipeline(device_id="mock-oakd-0")
    pipeline.__enter__()
    pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
    intrinsics = pipeline.getCameraIntrinsics(dai.CameraBoardSocket.CAM_A, 640, 480)
    assert len(intrinsics) == 3
    assert intrinsics[0][0] == 640.0  # focal length == max(width, height)


def test_get_camera_intrinsics_rejects_unbound_socket():
    pipeline = dai.Pipeline(device_id="mock-oakd-0")
    pipeline.__enter__()
    with pytest.raises(RuntimeError, match="unbound socket"):
        pipeline.getCameraIntrinsics(dai.CameraBoardSocket.CAM_B, 640, 480)


# ---------------------------------------------------------------------------
# Same-adapter path: through IsaacCaptureAdapter/DepthAIAdapter/Acquisition,
# not a protocol-shaped generator called directly.
# ---------------------------------------------------------------------------


async def test_adapter_path_produces_a_valid_tracking_payload():
    factory = build_mock_source_factory(scenario="demo", seed=1)
    begin = p.BeginRequest(
        tracking=p.TrackingConfig(sources=(p.SourceType.HEAD, p.SourceType.CONTROLLERS, p.SourceType.HANDS)),
        plugins=(p.PluginRequest(plugin=p.PluginType.MANUS, enabled=True),),
    )
    acq: Acquisition = factory(begin, ())
    await acq.enter()
    try:
        _, fields = await acq.next_tracking_sample()
        payload = p.encode_tracking_frame(fields)
        assert len(payload) == p.IC_TRACKING_V1_PAYLOAD_SIZE
        status = await acq.status(123)
        assert any(pr.plugin == p.PluginType.MANUS for pr in status.providers)
    finally:
        await acq.close()


async def test_adapter_path_capture_time_ns_is_frozen_at_the_real_sample_point():
    """capture_time_ns marks the host SAMPLING step (step()), not later
    publish/assembly time. Make the mapping work that
    runs AFTER step() artificially slow (advances the injected clock),
    and confirm the returned timestamp reflects the moment BEFORE that
    slow work ran, not after -- proving the value is captured eagerly at
    the real sample point and never re-read later."""
    factory = build_mock_source_factory(scenario="demo", seed=1)
    begin = p.BeginRequest(tracking=p.TrackingConfig(sources=(p.SourceType.HEAD,)))
    acq: Acquisition = factory(begin, ())
    await acq.enter()
    try:
        adapter = acq.tracking
        assert adapter is not None
        # build_mock_source_factory wires the real wall-clock MonotonicClock
        # by default -- swap in a fully controllable one for this test. Only
        # the sync methods are called directly below, which don't depend on
        # adapter._deadline (only the async pacing wrapper does), so this
        # swap needs no further adapter state fixup.
        fake_clock = _FakeClock(t=100.0)
        adapter._clock = fake_clock
        before_ns = int(fake_clock.monotonic() * 1_000_000_000)
        real_map_tracking = adapter._map_tracking

        def slow_map_tracking(native):
            # Simulate a slow mapping/encode step by advancing the SAME
            # clock the timestamp was (or wasn't) captured from.
            fake_clock.advance(5.0)
            return real_map_tracking(native)

        adapter._map_tracking = slow_map_tracking
        sample_time_ns, _ = adapter._step_and_map_sync()
        # The returned timestamp must equal BEFORE the 5s advance exactly,
        # not 5s later -- proves it was stamped prior to step()/mapping,
        # not read again afterward.
        assert sample_time_ns == before_ns
    finally:
        adapter._map_tracking = real_map_tracking
        await acq.close()


async def test_adapter_path_camera_capture_time_ns_is_frozen_at_the_real_sample_point():
    """Same invariant as the tracking case above, for the camera path:
    capture_time_ns must reflect the real tryGet() poll, not whatever
    time validation/copying finished at."""
    factory = build_mock_source_factory(scenario="demo", seed=1)
    begin = p.BeginRequest(tracking=p.TrackingConfig(sources=()))
    cameras = (
        p.CameraStreamConfig(
            stream_id=1,
            name="cam0",
            type=p.CameraType.OAK_D_COLOR,
            required=True,
            width=64,
            height=48,
            fps=120.0,
            jpeg_quality=80,
        ),
    )
    acq = factory(begin, cameras)
    await acq.enter()
    try:
        camera_adapter = acq.cameras[0]
        # The mock pipeline's OWN frame pacing uses its own clock (real
        # wall-clock, unaffected by this swap) -- only the adapter's own
        # capture_time_ns stamping (taken after a successful tryGet())
        # reads camera_adapter._clock, so swapping it here doesn't change
        # whether/when tryGet() actually returns a frame.
        fake_clock = _FakeClock(t=100.0)
        camera_adapter._clock = fake_clock
        real_validate_dimensions = camera_adapter._validate_dimensions

        def slow_validate_dimensions(frame):
            fake_clock.advance(5.0)
            return real_validate_dimensions(frame)

        camera_adapter._validate_dimensions = slow_validate_dimensions
        result = None
        for _ in range(50):
            before_ns = int(fake_clock.monotonic() * 1_000_000_000)
            result = camera_adapter._poll_sync()
            if result is not None:
                break
        assert result is not None
        sample_time_ns, _ = result
        # The returned timestamp must equal the real tryGet() moment, not
        # the 5s-later time _validate_dimensions (called after the real
        # sample point) advanced the clock to.
        assert sample_time_ns == before_ns
    finally:
        camera_adapter._validate_dimensions = real_validate_dimensions
        await acq.close()


def test_depthai_adapter_encoded_frame_ceiling_matches_the_client_s_2mib_limit():
    """The wire protocol's JPEG payload ceiling is a fixed 2 MiB -- the
    complete binary image message is at most 2 MiB plus the 32-byte
    header -- and CaptureClient enforces exactly that as its own
    websockets max_size. A frame just at the 2 MiB boundary must be
    accepted; one byte over must be rejected. Construct the adapter
    directly (sdk/pipeline_factory are unused by these pure validation
    methods) rather than routing an actual oversized frame through the
    full mock pipeline."""
    camera = p.CameraStreamConfig(
        stream_id=1,
        name="cam0",
        type=p.CameraType.OAK_D_COLOR,
        required=True,
        width=1280,
        height=720,
        fps=30.0,
        jpeg_quality=80,
    )
    adapter = DepthAIAdapter(sdk=None, pipeline_factory=lambda: None, camera=camera)
    max_bytes = adapter._max_encoded_bytes()
    assert max_bytes == 2 * 1024 * 1024
    assert adapter._validate_size(max_bytes) is None
    assert adapter._validate_size(max_bytes + 1) is not None


def test_depthai_adapter_small_frame_ceiling_keeps_its_header_overhead_margin():
    """The 2 MiB spec clamp must only ever bind at larger resolutions --
    a small frame's ceiling still gets the raw-pixel-plus-64KiB margin
    the original design intended (a tiny valid JPEG's header/metadata
    overhead can exceed its own raw-pixel count), not a flat 2 MiB for
    everything regardless of resolution."""
    camera = p.CameraStreamConfig(
        stream_id=1,
        name="cam0",
        type=p.CameraType.OAK_D_COLOR,
        required=True,
        width=64,
        height=48,
        fps=30.0,
        jpeg_quality=80,
    )
    adapter = DepthAIAdapter(sdk=None, pipeline_factory=lambda: None, camera=camera)
    expected_margin_based = 64 * 48 * 3 + 65536
    assert expected_margin_based < 2 * 1024 * 1024
    assert adapter._max_encoded_bytes() == expected_margin_based


async def test_adapter_path_produces_a_valid_camera_jpeg():
    factory = build_mock_source_factory(scenario="demo", seed=1)
    begin = p.BeginRequest(tracking=p.TrackingConfig(sources=()))
    cameras = (
        p.CameraStreamConfig(
            stream_id=1,
            name="cam0",
            type=p.CameraType.OAK_D_COLOR,
            required=True,
            width=64,
            height=48,
            fps=120.0,
            jpeg_quality=80,
        ),
    )
    acq = factory(begin, cameras)
    await acq.enter()
    try:
        result = None
        for _ in range(50):
            result = await acq.next_camera_jpeg(1)
            if result is not None:
                break
        assert result is not None
        capture_time_ns, jpeg = result
        assert capture_time_ns > 0
        assert jpeg[:2] == b"\xff\xd8"
        assert jpeg[-2:] == b"\xff\xd9"
    finally:
        await acq.close()


async def test_adapter_path_startup_error_raises_source_error():
    factory = build_mock_source_factory(scenario="demo", seed=1, ic_faults={"startup_error"})
    begin = p.BeginRequest(tracking=p.TrackingConfig(sources=(p.SourceType.HEAD,)))
    acq = factory(begin, ())
    with pytest.raises(SourceError):
        await acq.enter()


async def test_adapter_path_required_manus_plugin_load_failure_raises_source_error():
    """A required Manus plugin that fails to load must fail session
    establishment entirely (IC_ERROR) -- distinct from
    test_adapter_path_manus_requested_but_unavailable_with_head_and_
    controllers above, which uses provider_failed to degrade an already
    -loaded plugin post-entry without failing enter() at all."""
    factory = build_mock_source_factory(scenario="demo", seed=1, ic_faults={"manus_plugin_load_failed"})
    begin = p.BeginRequest(
        tracking=p.TrackingConfig(sources=(p.SourceType.HEAD,)),
        plugins=(p.PluginRequest(plugin=p.PluginType.MANUS, enabled=True, required=True),),
    )
    acq = factory(begin, ())
    with pytest.raises(SourceError):
        await acq.enter()


async def test_adapter_path_optional_manus_plugin_undiscovered_omits_inventory_without_raising():
    """The same load-failure fault on an OPTIONAL (required=False, the
    PluginRequest default) Manus plugin must not fail session
    establishment -- enter() succeeds and head tracking proceeds
    normally. Unlike test_adapter_path_manus_requested_but_unavailable_
    with_head_and_controllers below (provider_failed: a plugin that DID
    launch and later failed, genuinely FAILED/UNAVAILABLE), an optional
    plugin the real SDK could never discover at startup in the first
    place gets NO inventory entry at all -- same as never having been
    requested -- matching TeleopSession._resolve_plugin's silent skip."""
    factory = build_mock_source_factory(scenario="demo", seed=1, ic_faults={"manus_plugin_load_failed"})
    begin = p.BeginRequest(
        tracking=p.TrackingConfig(sources=(p.SourceType.HEAD,)),
        plugins=(p.PluginRequest(plugin=p.PluginType.MANUS, enabled=True, required=False),),
    )
    acq = factory(begin, ())
    await acq.enter()
    try:
        _, fields = await acq.next_tracking_sample()
        assert fields["head.present"] == 1
        status = await acq.status(1)
        assert not any(pr.plugin == p.PluginType.MANUS for pr in status.providers)
        assert "device-hand-left" not in {d.id for d in status.devices}
        assert "device-hand-right" not in {d.id for d in status.devices}
    finally:
        await acq.close()


async def test_adapter_path_undiscovered_vs_launched_then_failed_manus_are_distinct():
    """The two "no usable Manus" scenarios must map to genuinely
    different statuses, not collapse into one: an optional plugin that
    could never be discovered (manus_plugin_load_failed) gets absent
    inventory, while a plugin that DID launch and later failed
    (provider_failed) is reported FAILED/UNAVAILABLE and keeps its
    inventory entries."""
    begin = p.BeginRequest(
        tracking=p.TrackingConfig(sources=(p.SourceType.HEAD,)),
        plugins=(p.PluginRequest(plugin=p.PluginType.MANUS, enabled=True, required=False),),
    )

    undiscovered_factory = build_mock_source_factory(
        scenario="demo", seed=1, ic_faults={"manus_plugin_load_failed"}
    )
    acq = undiscovered_factory(begin, ())
    await acq.enter()
    try:
        status = await acq.status(1)
        assert not any(pr.plugin == p.PluginType.MANUS for pr in status.providers)
    finally:
        await acq.close()

    failed_factory = build_mock_source_factory(scenario="demo", seed=1, ic_faults={"provider_failed"})
    acq = failed_factory(begin, ())
    await acq.enter()
    try:
        status = await acq.status(1)
        manus = next(pr for pr in status.providers if pr.plugin == p.PluginType.MANUS)
        assert manus.status == p.ProviderState.FAILED
        assert "device-hand-left" in {d.id for d in status.devices}
    finally:
        await acq.close()


async def test_adapter_path_required_camera_failure_raises_camera_source_error():
    factory = build_mock_source_factory(scenario="demo", seed=1, camera_faults={"camera_required_failure"})
    begin = p.BeginRequest(tracking=p.TrackingConfig(sources=()))
    cameras = (
        p.CameraStreamConfig(
            stream_id=1,
            name="cam0",
            type=p.CameraType.OAK_D_COLOR,
            required=True,
            width=64,
            height=48,
            fps=30.0,
            jpeg_quality=80,
        ),
    )
    acq = factory(begin, cameras)
    with pytest.raises(CameraSourceError):
        await acq.enter()


async def test_adapter_path_manus_requested_but_unavailable_with_head_and_controllers():
    """(a) head+controllers available but a requested Manus plugin is
    unavailable -- provider_failed fault through the real adapter path,
    with controllers independently selected and healthy."""
    factory = build_mock_source_factory(scenario="demo", seed=1, ic_faults={"provider_failed"})
    begin = p.BeginRequest(
        tracking=p.TrackingConfig(sources=(p.SourceType.HEAD, p.SourceType.CONTROLLERS)),
        plugins=(p.PluginRequest(plugin=p.PluginType.MANUS, enabled=True),),
    )
    acq = factory(begin, ())
    await acq.enter()
    try:
        _, fields = await acq.next_tracking_sample()
        assert fields["head.present"] == 1
        assert fields["controller_left.present"] == 1
        assert fields["controller_right.present"] == 1
        status = await acq.status(1)
        manus = next(pr for pr in status.providers if pr.plugin == p.PluginType.MANUS)
        assert manus.status == p.ProviderState.FAILED
        # Controller devices are registered independently of Manus's state.
        assert {d.id for d in status.devices} >= {
            "device-headset",
            "device-controller-left",
            "device-controller-right",
        }
    finally:
        await acq.close()


@pytest.mark.parametrize("absent_side", ["left", "right"])
async def test_adapter_path_one_controller_side_absent_independently(absent_side):
    """(b) independent left/right controller absence -- tested in both
    directions through the real adapter path. The absent side's device
    stays registered (it was selected/present a moment ago); only this
    sample's presence is affected."""
    factory = build_mock_source_factory(scenario="demo", seed=1, ic_faults={f"controller_{absent_side}_absent"})
    present_side = "right" if absent_side == "left" else "left"
    begin = p.BeginRequest(tracking=p.TrackingConfig(sources=(p.SourceType.CONTROLLERS,)))
    acq = factory(begin, ())
    await acq.enter()
    try:
        _, fields = await acq.next_tracking_sample()
        assert fields[f"controller_{absent_side}.present"] == 0
        assert fields[f"controller_{present_side}.present"] == 1
        status = await acq.status(1)
        device_ids = {d.id for d in status.devices}
        assert f"device-controller-{absent_side}" in device_ids
        assert f"device-controller-{present_side}" in device_ids
    finally:
        await acq.close()


@pytest.mark.parametrize("absent_side", ["left", "right"])
async def test_adapter_path_one_hand_absent_independently(absent_side):
    """A selected hand output can return an absent OptionalTensorGroup
    while every other selected group (head, controllers, the other
    hand) continues reporting present/valid normally -- same pattern as
    controller_left_absent/controller_right_absent, tested in both
    directions. Independent of Manus/provider state: no vendor
    inference, this is plain HandsSource output absence."""
    factory = build_mock_source_factory(scenario="demo", seed=1, ic_faults={f"hand_{absent_side}_absent"})
    present_side = "right" if absent_side == "left" else "left"
    begin = p.BeginRequest(
        tracking=p.TrackingConfig(sources=(p.SourceType.HEAD, p.SourceType.CONTROLLERS, p.SourceType.HANDS)),
        plugins=(p.PluginRequest(plugin=p.PluginType.MANUS, enabled=True),),
    )
    acq = factory(begin, ())
    await acq.enter()
    try:
        _, fields = await acq.next_tracking_sample()
        assert fields[f"hand_{absent_side}.present"] == 0
        assert fields[f"hand_{present_side}.present"] == 1
        # Everything else keeps reporting present/valid, unaffected.
        assert fields["head.present"] == 1
        assert fields["head.is_valid"] == 1
        assert fields["controller_left.present"] == 1
        assert fields["controller_left.grip_is_valid"] == 1
        assert fields["controller_right.present"] == 1
        assert fields["controller_right.grip_is_valid"] == 1
    finally:
        await acq.close()


async def test_adapter_path_present_controller_with_invalid_pose():
    """(c) a present group with an invalid pose -- not absent, just
    unusable this sample."""
    factory = build_mock_source_factory(scenario="demo", seed=1, ic_faults={"controller_pose_invalid"})
    begin = p.BeginRequest(tracking=p.TrackingConfig(sources=(p.SourceType.CONTROLLERS,)))
    acq = factory(begin, ())
    await acq.enter()
    try:
        _, fields = await acq.next_tracking_sample()
        assert fields["controller_left.present"] == 1
        assert fields["controller_left.grip_is_valid"] == 0
        assert fields["controller_left.aim_is_valid"] == 0
    finally:
        await acq.close()


async def test_adapter_path_present_hand_with_invalid_joints():
    """(c) a present hand group with EVERY joint invalid -- the whole
    hand untracked, not absent."""
    factory = build_mock_source_factory(scenario="demo", seed=1, ic_faults={"hand_pose_invalid"})
    begin = p.BeginRequest(tracking=p.TrackingConfig(sources=(p.SourceType.HANDS,)))
    acq = factory(begin, ())
    await acq.enter()
    try:
        _, fields = await acq.next_tracking_sample()
        assert fields["hand_left.present"] == 1
        assert int(fields["hand_left.joint_valid"].sum()) == 0
    finally:
        await acq.close()


async def test_adapter_path_present_hand_with_mixed_joint_validity():
    """Distinct from the all-invalid case above: SOME joints untracked
    (every fingertip, deterministically), the rest still valid -- a
    present, partially-usable hand, not an all-or-nothing one. Generic
    hand output absence/validity here never implies anything about
    Manus provenance -- no vendor filtering, this is plain HandsSource
    output."""
    factory = build_mock_source_factory(scenario="demo", seed=1, ic_faults={"hand_pose_partially_invalid"})
    begin = p.BeginRequest(tracking=p.TrackingConfig(sources=(p.SourceType.HANDS,)))
    acq = factory(begin, ())
    await acq.enter()
    try:
        _, fields = await acq.next_tracking_sample()
        valid = fields["hand_left.joint_valid"]
        assert fields["hand_left.present"] == 1
        assert 0 < int(valid.sum()) < 26  # neither all-valid nor all-invalid
        for index in HandJointIndex:
            expected = 0 if "TIP" in index.name else 1
            assert valid[int(index)] == expected, index.name
    finally:
        await acq.close()
