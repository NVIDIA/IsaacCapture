# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What the agent side takes for granted about APAI and about the robot app.

The agent side (isaaccapture_examples.agentic_remote.agent) stands in for
APAI's motor driver and camera and reads a few of APAI's private names. Each
test here pins one of those call sites on an unmodified APAI checkout
(AGENTIC_REMOTE_APAI_SRC), on a runtime APAI built itself for its mock scene
with a required obstacle grid. The last tests pin the robot app's texts the
arm adapter matches; they need no APAI.
"""

from __future__ import annotations

import importlib
import inspect
import re
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest import mock

import numpy as np
import pytest
from isaaccapture_examples.agentic_remote.link_service import service
from isaaccapture_examples.agentic_remote.protocol import messages
from isaaccapture_examples.agentic_remote.robot_app import app as robot_app
from isaaccapture_examples.agentic_remote.robot_app import executor

from conftest import FakeGridClient, apai_runtime_env

apai = pytest.mark.apai

# The keywords of ArmBase.stream_to and stream_path after self, with defaults.
STREAM_KEYWORDS = {
    "duration_s": inspect.Parameter.empty,
    "rate_hz": 50.0,
    "approve": None,
    "settle_tol": None,
    "settle_timeout_s": None,
    "preflight": None,
    "before_stream": None,
    "contact_monitor": None,
}
# Names APAI looks for on an arm to switch to another mode; the arm adapter
# must have none of them.
MODE_PROBES = (
    "motion_status",
    "ensure_motion_ready",
    "check_resume_ready",
    "physics_engine",
    "validate_simulation_clock",
    "reset_props",
    "attachment_feedback_version",
    "stream_profile",
    "close_gripper_torque",
)
# The contact monitor's trip text, as the arm adapter parses it.
CONTACT_TEXT = re.compile(
    r"contact detected at joint (\d+): tracking error ([0-9.]+) rad for ([0-9.]+) ms"
)


def parameters(function: Any) -> dict[str, Any]:
    """A function's parameters after self: name to default."""
    found = inspect.signature(function).parameters
    return {name: p.default for name, p in found.items() if name != "self"}


def takes_no_other_keywords(function: Any) -> bool:
    kinds = {p.kind for p in inspect.signature(function).parameters.values()}
    return inspect.Parameter.VAR_KEYWORD not in kinds


def mock_scene_config(folder: Path) -> Any:
    """APAI's mock scene with the obstacle grid on and required."""
    config = importlib.import_module("apai.config")
    loaded = config.load_demo_config(camera="mock", arm="mock", llm="mock")
    data = loaded.as_dict()  # a deep copy: Cfg itself is read-only
    data["memory"]["experience_path"] = str(folder / "experience.json")
    views = [data, *(arm["resolved"] for arm in data["arms"] if "resolved" in arm)]
    for view in views:
        view["occupancy"]["enabled"] = True
        view["occupancy"]["required"] = True
    return type(loaded)(data)


@pytest.fixture(scope="module")
def runtime(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Any]:
    """A runtime APAI built for its mock scene, on a fake obstacle grid."""
    demo = importlib.import_module("apai.apps.demo")
    occupancy = importlib.import_module("apai.perception.occupancy")
    with apai_runtime_env(tmp_path_factory.mktemp("apai-contract")) as folder:
        with mock.patch.object(occupancy, "OccupancyClient", FakeGridClient):
            built, raw = demo.build_runtime(mock_scene_config(folder), folder / "run")
        try:
            assert built.arm.harness.occupancy.wait_until_usable(10.0)
            yield built
        finally:
            demo.shutdown_runtime(built, raw)


@pytest.fixture
def harness(runtime: Any) -> Any:
    """The workstation checker of the runtime's arm: an APAI SafetyHarness."""
    return runtime.arm.harness


class StreamSpy:
    """Stands between SafeArm and the mock arm's stream_to for one test."""

    def __init__(self, raw: Any, harness: Any) -> None:
        self.raw, self.harness = raw, harness
        self.keywords: list[dict[str, Any]] = []  # of every stream_to call
        # Per call of the checker callback: (q_prev, q_i, dt, positional count).
        self.approvals: list[tuple[np.ndarray, np.ndarray, float, int]] = []
        self.order: list[str] = []  # "approve" and "send", as they happened
        self.counts: list[int] = []  # approved_commands after each approval
        self.thread: threading.Thread | None = None

    def stream_to(
        self,
        q_target,
        duration_s,
        rate_hz=50.0,
        approve=None,
        settle_tol=None,
        settle_timeout_s=None,
        preflight=None,
        before_stream=None,
        contact_monitor=None,
    ) -> bool:
        given = {
            "rate_hz": rate_hz,
            "approve": approve,
            "settle_tol": settle_tol,
            "settle_timeout_s": settle_timeout_s,
            "preflight": preflight,
            "before_stream": before_stream,
            "contact_monitor": contact_monitor,
        }
        self.keywords.append(given)
        self.thread = threading.current_thread()

        def recorded(*args: Any) -> None:
            approve(*args)
            self.approvals.append((args[0].copy(), args[1].copy(), args[2], len(args)))
            self.order.append("approve")
            self.counts.append(self.harness.approved_commands)

        return type(self.raw).stream_to(
            self.raw, q_target, duration_s, **{**given, "approve": recorded}
        )

    def send_joint_target(self, q) -> None:
        self.order.append("send")
        type(self.raw).send_joint_target(self.raw, q)


@pytest.fixture
def spy(runtime: Any, harness: Any) -> Iterator[StreamSpy]:
    """The runtime's arm with a StreamSpy on its driver; the arm ends where it began."""
    raw = runtime.arm.raw
    start = np.asarray(runtime.arm.get_state().q, dtype=float).copy()
    made = StreamSpy(raw, harness)
    with (
        mock.patch.object(raw, "stream_to", made.stream_to),
        mock.patch.object(raw, "send_joint_target", made.send_joint_target),
    ):
        yield made
    type(raw).send_joint_target(raw, start)


def nearby(runtime: Any, joint: int = 0, by: float = 0.03) -> np.ndarray:
    """A pose a little away from where the runtime's arm is."""
    q = np.asarray(runtime.arm.get_state().q, dtype=float).copy()
    q[joint] += by
    return q


# The arm interface


@apai
def test_a_driver_must_define_these_six_methods_and_nothing_else() -> None:
    arm_base = importlib.import_module("apai.control.arm_base")
    assert arm_base.ArmBase.__abstractmethods__ == frozenset(
        {
            "connect",
            "disconnect",
            "get_state",
            "send_joint_target",
            "set_gripper",
            "stop",
        }
    )
    assert parameters(arm_base.ArmBase.set_gripper) == {
        "pos": inspect.Parameter.empty,
        "effort": 1.0,
    }
    assert not hasattr(arm_base.ArmBase, "resume")  # only some drivers have one


@apai
@pytest.mark.parametrize("name", ["stream_to", "stream_path"])
def test_a_stream_call_takes_these_keywords_and_no_others(name: str) -> None:
    arm_base = importlib.import_module("apai.control.arm_base")
    method = getattr(arm_base.ArmBase, name)
    first, *rest = parameters(method).items()
    assert first[1] is inspect.Parameter.empty  # the goal, or the path
    assert dict(rest) == STREAM_KEYWORDS
    assert takes_no_other_keywords(method)


@apai
def test_the_driver_base_class_gives_these_defaults() -> None:
    arm_base = importlib.import_module("apai.control.arm_base")
    base = arm_base.ArmBase
    assert base.motion_rate_hz == 50.0 and type(base.motion_rate_hz) is float
    assert base.n_joints == 6
    assert (base.settle_tol, base.settle_timeout_s) == (0.03, 2.0)
    assert base.connect_cuts_torque is False
    for name in MODE_PROBES:
        assert not hasattr(base, name), name


@apai
def test_apai_still_looks_for_the_mode_names_on_the_driver() -> None:
    """If a probe is gone from APAI, drop it from MODE_PROBES and the contract."""
    package = Path(importlib.import_module("apai").__file__).parent
    text = "".join(path.read_text() for path in sorted(package.rglob("*.py")))
    for name in MODE_PROBES:
        assert re.search(rf"""["']{name}["']|\.{name}\b""", text), name
    runtime_source = (package / "skills" / "runtime.py").read_text()
    assert 'hasattr(raw, "close_gripper_two_stage")' in runtime_source
    assert 'getattr(raw, "_stopped", False)' in runtime_source


def test_the_arm_adapter_has_none_of_the_mode_names() -> None:
    try:
        adapter = importlib.import_module(
            "isaaccapture_examples.agentic_remote.agent.arm_adapter"
        )
    except ModuleNotFoundError as missing:
        pytest.skip(f"the arm adapter cannot be imported: {missing}")
    for name in MODE_PROBES:
        assert not hasattr(adapter.RobotLinkArm, name), name


@apai
def test_the_arm_factory_knows_no_other_type_and_demo_holds_its_own_name() -> None:
    arm_base = importlib.import_module("apai.control.arm_base")
    config = importlib.import_module("apai.config")
    demo = importlib.import_module("apai.apps.demo")
    assert parameters(arm_base.make_arm) == {
        "cfg": inspect.Parameter.empty,
        "kinematics": None,
    }
    with pytest.raises(ValueError, match="unknown arm type 'robot_link'"):
        arm_base.make_arm(config.Cfg({"type": "robot_link"}))
    # demo bound the name at import: the launcher patches demo.make_arm.
    assert demo.make_arm is arm_base.make_arm


@apai
def test_the_arm_state_has_these_fields(runtime: Any) -> None:
    types = importlib.import_module("apai.types")
    state = types.RobotState(
        q=np.zeros(6), gripper_pos=0.5, gripper_valid=False, t=12.0
    )
    assert (state.gripper_pos, state.gripper_valid, state.t) == (0.5, False, 12.0)
    read = runtime.arm.get_state()
    assert np.asarray(read.q).shape == (6,)
    assert isinstance(read.t, float)


@apai
def test_prepare_stream_binds_the_start_to_the_readers_pose() -> None:
    arm_base = importlib.import_module("apai.control.arm_base")
    types = importlib.import_module("apai.types")
    assert parameters(arm_base.prepare_stream) == {
        "arm": inspect.Parameter.empty,
        "q_target": inspect.Parameter.empty,
        "duration_s": inspect.Parameter.empty,
        "preflight": None,
        "before_stream": None,
    }
    calls: list[Any] = []

    class Reader:  # all it asks of the arm is get_state().q
        pose = np.array([0.1, 0.2, 0.3, 0.0, 0.0, 0.0])

        def get_state(self) -> Any:
            calls.append("read")
            return types.RobotState(q=self.pose.copy(), gripper_pos=0.0)

    reader = Reader()
    start, target = arm_base.prepare_stream(
        reader,
        [0.2, 0.2, 0.3, 0.0, 0.0, 0.0],
        1.5,
        preflight=lambda q, duration_s: calls.append(("preflight", duration_s)),
        before_stream=lambda: calls.append("before"),
    )
    assert start == pytest.approx(reader.pose) and start is not reader.pose
    assert target == pytest.approx([0.2, 0.2, 0.3, 0.0, 0.0, 0.0])
    assert calls == ["read", "before", ("preflight", 1.5), "read", "before", "before"]

    # A pose that keeps changing by more than 1 mrad is refused, nothing sent.
    def drift() -> None:
        reader.pose = reader.pose + 0.002

    with pytest.raises(types.SafetyViolation, match="no motion sent"):
        arm_base.prepare_stream(reader, reader.pose, 1.0, lambda q, d: drift())
    assert arm_base.PREFLIGHT_MAX_DRIFT_RAD == 0.001


# The checker callback


@apai
def test_the_checker_callback_is_called_per_setpoint_before_each_send(
    runtime: Any, spy: StreamSpy
) -> None:
    start = np.asarray(runtime.arm.get_state().q, dtype=float)
    goal = nearby(runtime)
    before = spy.harness.approved_commands
    assert runtime.arm.move_joints(goal, duration_s=0.4) is True
    steps = 20  # max(2, int(duration_s * rate_hz))
    assert len(spy.approvals) == steps
    assert spy.order == ["approve", "send"] * steps
    assert {count for *_, count in spy.approvals} == {3}  # approve(q_prev, q_i, dt)
    assert [dt for _, _, dt, _ in spy.approvals] == [pytest.approx(0.4 / steps)] * steps
    previous = start
    for q_prev, q_i, _, _ in spy.approvals:
        assert q_prev == pytest.approx(previous)
        previous = q_i
    assert previous == pytest.approx(goal)
    # The counter APAI's skills read goes up once per setpoint the callback cleared.
    assert spy.counts == list(range(before + 1, before + steps + 1))
    assert spy.thread is threading.current_thread()  # the skill thread


@apai
def test_safe_arm_passes_the_callback_by_keyword_and_no_settle_numbers(
    runtime: Any, spy: StreamSpy
) -> None:
    runtime.arm.move_joints(nearby(runtime), duration_s=0.1)
    (given,) = spy.keywords
    assert callable(given["approve"])
    assert given["rate_hz"] == 50.0
    assert given["settle_tol"] is None and given["settle_timeout_s"] is None
    monitor = given["contact_monitor"]
    contact = importlib.import_module("apai.safety.contact")
    assert monitor is None or isinstance(monitor, contact.ContactMonitor)


@apai
def test_a_hint_the_driver_does_not_take_is_dropped_and_the_move_runs(
    runtime: Any, spy: StreamSpy
) -> None:
    assert runtime.arm.move_joints(
        nearby(runtime), duration_s=0.1, bias_compensate=np.zeros(6)
    )
    assert len(spy.keywords) == 1  # the call with the hint never reached the driver
    assert len(spy.approvals) == 5


@apai
def test_a_safety_keyword_the_driver_does_not_take_ends_the_move(
    runtime: Any, spy: StreamSpy
) -> None:
    types = importlib.import_module("apai.types")
    before = spy.harness.approved_commands
    with pytest.raises(types.SafetyViolation, match="no unguarded retry"):
        runtime.arm.move_joints(
            nearby(runtime), duration_s=0.1, feedback_guard=lambda: None
        )
    assert spy.keywords == [] and spy.harness.approved_commands == before


@apai
def test_the_checker_has_these_methods(harness: Any) -> None:
    step = {
        "q_prev": inspect.Parameter.empty,
        "q_next": inspect.Parameter.empty,
        "dt": inspect.Parameter.empty,
        "joint_margin": None,
    }
    assert parameters(harness.vet_step) == step
    assert parameters(harness.approve) == step
    assert parameters(harness.clear_grasp_exemption) == {}
    assert parameters(harness.clear_contact_zones) == {"reason": "cleared"}
    assert parameters(harness.estop) == {"reason": "manual"}
    assert parameters(harness.halt) == {"reason": inspect.Parameter.empty}
    assert parameters(harness.record_contact_motion) == {
        "monitor": inspect.Parameter.empty,
        "exc": None,
    }
    assert parameters(harness.check_contact_episode) == {"gripper": False}


@apai
def test_vet_step_returns_a_text_and_counts_nothing(harness: Any, runtime: Any) -> None:
    q = np.asarray(runtime.arm.get_state().q, dtype=float)
    counters = (harness.approved_commands, harness.motions_begun)
    assert type(harness.approved_commands) is int
    assert harness.vet_step(q, nearby(runtime, by=0.002), 0.02) is None
    refusal = harness.vet_step(q, nearby(runtime, by=1.0), 0.02)  # 50 rad/s
    assert isinstance(refusal, str) and refusal
    assert (harness.approved_commands, harness.motions_begun) == counters
    assert isinstance(harness.violations, list)


@apai
def test_the_checkers_private_names_are_there(harness: Any) -> None:
    assert harness._grasp_exempt is None  # no grasp region is open
    assert harness.last_contact is None
    assert harness.estopped is False
    assert harness.occupancy is not None  # CheckAhead refuses a checker without


# The obstacle grid


@apai
def test_the_grids_lock_and_counters_are_there(harness: Any) -> None:
    grid = harness.occupancy
    assert grid.required is True
    lock = grid._refresh_lock
    assert lock.acquire(timeout=1.0)
    try:
        assert lock.acquire(blocking=False)  # re-entrant
        lock.release()
        stamp = (grid.refreshes, grid.cache_clears)
        assert all(type(count) is int for count in stamp)
        assert stamp[0] >= 1
        assert isinstance(grid._last_refresh, float)
    finally:
        lock.release()
    end = time.monotonic() + 5.0
    while grid.refreshes == stamp[0] and time.monotonic() < end:
        time.sleep(0.02)
    assert grid.refreshes > stamp[0]  # every camera frame of the watcher counts


@apai
def test_a_new_grid_keeps_these_values() -> None:
    occupancy = importlib.import_module("apai.perception.occupancy")
    grid = occupancy.OccupancyMap(
        client=FakeGridClient(),
        region_min=np.full(3, -0.5),
        region_max=np.full(3, 0.5),
        required=True,
    )
    assert (grid._last_refresh, grid.refreshes, grid.cache_clears) == (None, 0, 0)
    assert parameters(grid.wait_until_usable) == {"timeout_s": None}


@apai
def test_the_grids_masking_and_integration_take_these_arguments() -> None:
    occupancy = importlib.import_module("apai.perception.occupancy")
    grid = occupancy.OccupancyMap
    assert parameters(grid._robot_depth_mask) == {
        "depth": inspect.Parameter.empty,
        "K": inspect.Parameter.empty,
        "T_base_cam": inspect.Parameter.empty,
        "frame": None,
    }
    frame = inspect.signature(grid._robot_depth_mask).parameters["frame"]
    assert frame.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameters(grid._integrate_points) == {
        "frame": inspect.Parameter.empty,
        "T_base_cam": inspect.Parameter.empty,
    }
    assert parameters(grid.refresh) == parameters(grid._integrate_points)


@apai
def test_the_grid_client_can_be_replaced_while_the_runtime_is_built() -> None:
    """Without pyzmq a required grid cannot be built; with the fake it can."""
    occupancy = importlib.import_module("apai.perception.occupancy")
    assert inspect.isclass(occupancy.OccupancyClient)
    assert issubclass(occupancy.OccupancyError, RuntimeError)
    source = inspect.getsource(occupancy.OccupancyMap.from_config)
    assert "client = OccupancyClient(" in source  # looked up by name at each call


# The contact monitor


@apai
def test_the_contact_summary_reads_these_keys_of_the_trip_record() -> None:
    contact = importlib.import_module("apai.safety.contact")
    monitor = contact.ContactMonitor(
        contact.ContactMonitorConfig(enabled=True), label="move_joints"
    )
    assert (monitor.tripped, monitor.commands, monitor.hold) == (None, 0, None)
    assert monitor.started is False and monitor.label == "move_joints"
    end_q = [0.0, 0.1, 0.1, 0.0, 0.0, 0.0]
    record = {
        "joint": 2,
        "error_rad": 0.2,
        "duration_ms": 80.0,
        "phase": "stream",
        "motion": monitor.label,
        "measured_q": end_q,
        "hold_q": end_q,
    }
    monitor.tripped = record
    monitor.commands = 7
    monitor.hold = "measured pose (reported by the robot app)"
    summary = monitor.summary()  # begin() was never called: the short form
    assert summary["tripped"] == "joint 2 tracking error 0.200 rad (stream)"
    assert (summary["motion"], summary["commands"]) == ("move_joints", 7)
    assert summary["hold"] == monitor.hold
    assert "samples" not in summary
    # A record without one of the three keys it formats hides the contact.
    for key in ("joint", "error_rad", "phase"):
        monitor.tripped = {name: value for name, value in record.items() if name != key}
        with pytest.raises(KeyError, match=key):
            monitor.summary()


@apai
def test_the_contact_text_is_the_one_the_adapter_parses() -> None:
    contact = importlib.import_module("apai.safety.contact")
    runtime_module = importlib.import_module("apai.skills.runtime")
    types = importlib.import_module("apai.types")
    source = inspect.getsource(contact.ContactMonitor._trip)
    assert (
        'f"contact detected at joint {j + 1}: tracking error {err:.3f} rad for "'
    ) in source
    assert "{dur_ms:.0f} ms" in source
    text = "contact detected at joint 3: tracking error 0.214 rad for 85 ms -- x"
    assert CONTACT_TEXT.match(text).groups() == ("3", "0.214", "85")
    raised = types.ContactDetected(text, {"joint": 3})
    assert isinstance(raised, types.SafetyViolation) and raised.event == {"joint": 3}
    assert runtime_module._is_contact(raised) is True


@apai
def test_a_contact_reaches_the_checker_through_the_monitor(
    harness: Any, runtime: Any
) -> None:
    contact = importlib.import_module("apai.safety.contact")
    types = importlib.import_module("apai.types")
    end_q = np.asarray(runtime.arm.get_state().q, dtype=float).tolist()
    monitor = contact.ContactMonitor(
        contact.ContactMonitorConfig(enabled=True), label="move_joints"
    )
    text = "contact detected at joint 2: tracking error 0.200 rad for 80 ms"
    monitor.tripped = {
        "joint": 2,
        "error_rad": 0.2,
        "duration_ms": 80.0,
        "phase": "stream",
        "motion": monitor.label,
        "measured_q": end_q,
        "hold_q": end_q,
    }
    try:
        summary = harness.record_contact_motion(
            monitor, types.ContactDetected(text, monitor.tripped)
        )
        assert summary["tripped"] == "joint 2 tracking error 0.200 rad (stream)"
        assert harness.last_contact is not None
    finally:
        harness.clear_contact_zones("test")
        harness.last_contact = None


# The camera interface


@apai
def test_a_camera_must_define_these_four_names() -> None:
    camera_base = importlib.import_module("apai.perception.camera_base")
    assert camera_base.CameraBase.__abstractmethods__ == frozenset(
        {"open", "close", "_grab", "has_depth"}
    )
    assert isinstance(camera_base.CameraBase.has_depth, property)
    assert issubclass(camera_base.CameraError, RuntimeError)


@apai
def test_get_frame_numbers_what_grab_returns() -> None:
    camera_base = importlib.import_module("apai.perception.camera_base")
    types = importlib.import_module("apai.types")

    class Camera(camera_base.CameraBase):
        has_depth = True

        def open(self) -> None: ...

        def close(self) -> None: ...

        def _grab(self) -> Any:
            return types.Frame(
                rgb=np.zeros((4, 4, 3), dtype=np.uint8),
                depth_m=np.ones((4, 4), dtype=np.float32),
                K=np.eye(3),
                t=5.0,
                observed_t=5.0,
                depth_source="sensor",
                capture={"backend": "link_stereo", "seq": 1},
            )

    camera = Camera()
    first, second = camera.get_frame(), camera.get_frame()
    assert (first.frame_id, second.frame_id) == (1, 2)
    assert first.capture["backend"] == "link_stereo" and first.observed_t == 5.0


@apai
def test_a_frame_of_any_other_backend_is_a_local_frame() -> None:
    freshness = importlib.import_module("apai.perception.freshness")
    types = importlib.import_module("apai.types")
    assert freshness._PRODUCER_BACKENDS == ("ovrtx", "isaac")
    frame = types.Frame(
        rgb=np.zeros((2, 2, 3), dtype=np.uint8),
        depth_m=None,
        K=np.eye(3),
        t=7.5,
        capture={"backend": "link_stereo"},
    )
    assert freshness.capture_marker(frame) == {
        "channel": "local_frame",
        "clock": "client_monotonic",
        "t": 7.5,
    }


@apai
def test_the_camera_factory_knows_no_other_type_and_demo_holds_its_own_name() -> None:
    camera_base = importlib.import_module("apai.perception.camera_base")
    config = importlib.import_module("apai.config")
    demo = importlib.import_module("apai.apps.demo")
    assert parameters(camera_base.make_camera) == {"cfg": inspect.Parameter.empty}
    with pytest.raises(Exception, match="link_stereo"):
        camera_base.make_camera(config.Cfg({"type": "link_stereo"}))
    assert demo.make_camera is camera_base.make_camera


# The names the launcher patches


@apai
def test_demo_has_the_names_the_launcher_patches() -> None:
    demo = importlib.import_module("apai.apps.demo")
    orchestrator = importlib.import_module("apai.agent.orchestrator")
    colors = importlib.import_module("apai.perception.colors")
    assert parameters(demo.build_runtime) == {
        "cfg": inspect.Parameter.empty,
        "run_dir": inspect.Parameter.empty,
        "view": False,
        "lazy_arm": False,
        "serve": False,
    }
    assert list(parameters(demo._build_arm)) == [
        "acfg",
        "lazy_arm",
        "occupancy",
        "fallback_cfg",
        "safety",
    ]
    assert parameters(demo.main) == {"argv": None}
    assert list(parameters(demo.shutdown_runtime)) == ["runtime", "arm"]
    assert list(parameters(demo._print_report)) == ["report"]
    assert callable(demo._park_move) and callable(demo.load_demo_config)
    assert demo.AgentOrchestrator is orchestrator.AgentOrchestrator
    assert list(parameters(orchestrator.AgentOrchestrator.run_task)) == [
        "task",
        "cancel_event",
    ]
    assert parameters(orchestrator.tool_specs_for_config) == {"cfg": None}
    assert list(parameters(colors.mask_color))[:2] == ["bgr", "mask"]
    # demo calls the builtin input(): a patch of demo.input needs create=True.
    assert not hasattr(demo, "input")


@apai
def test_a_configuration_is_read_only_and_copied_to_change_it() -> None:
    config = importlib.import_module("apai.config")
    loaded = config.load_demo_config(camera="mock", arm="mock", llm="mock")
    with pytest.raises(TypeError):
        loaded.memory["experience_path"] = "x"
    data = loaded.as_dict()
    data["occupancy"]["required"] = True
    assert loaded.occupancy.required is False  # as_dict() is a deep copy
    assert config.Cfg(data).occupancy.required is True
    assert loaded.arm.type == "mock" and loaded.camera.type == "mock"


@apai
def test_the_errors_the_adapter_raises_are_what_apai_expects() -> None:
    types = importlib.import_module("apai.types")
    assert issubclass(types.SafetyViolation, Exception)
    assert issubclass(types.ContactDetected, types.SafetyViolation)
    assert inspect.isclass(types.MotionHalted)
    assert parameters(types.ContactDetected.__init__) == {
        "message": inspect.Parameter.empty,
        "event": None,
    }


# The robot app's texts the arm adapter matches


def test_the_final_result_of_a_move_names_these_outcomes() -> None:
    assert robot_app._MOVE_STATUS == {
        "completed": "completed",
        "stopped": "cancelled",
        "refused": "refused",
        "late": "failed",
        "not_settled": "failed",
        "contact": "failed",
        "failed": "failed",
    }
    source = inspect.getsource(robot_app)
    for key in ('"outcome": outcome.status', '"sent": outcome.index', '"contact"'):
        assert key in source, key


def test_the_reasons_the_adapter_matches_by_their_beginning() -> None:
    assert executor.HOLD_FAILED == "; the hold at the measured pose failed: "
    assert robot_app.HOLD_FAILED is executor.HOLD_FAILED
    assert 'f"fault: {' in inspect.getsource(executor)
    app_source = inspect.getsource(robot_app)
    assert 'f"fault: {fault}"' in app_source
    assert 'f"robot app error in {' in app_source
    assert 'f"robot app error: {error!r}"' in app_source
    assert "; its last permit was refused: " in inspect.getsource(executor)


def test_a_lost_robot_connection_reads_the_same_from_the_app_and_the_service() -> None:
    assert robot_app.LINK_LOST == service.LOST == "robot connection lost"
    assert service.AGENT_GONE == "agent program gone"


def test_the_checker_state_actions_and_the_move_stages() -> None:
    assert robot_app.CLOSE_GRASP == "close_grasp"
    assert robot_app.CLEAR_CONTACT_SPOTS == "clear_contact_spots"
    assert messages.MOVE_ARMED == "armed"
    state = {field.name: field for field in _fields(messages.RobotState)}
    assert state["move_stage"].default == ""


def _fields(kind: type) -> tuple[Any, ...]:
    import dataclasses

    return dataclasses.fields(kind)
