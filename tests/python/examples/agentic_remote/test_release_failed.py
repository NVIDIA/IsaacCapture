# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A torque release whose driver call raises is never reported as motors off.

A torque release switches the motors off at measured rest; on APAI's drivers
the driver's disconnect does it. When that call raises, the driver may have
switched some or all of the motors off before it failed: the motion state is
"release_failed", ROBOT_STATE keeps torque_on true, a fault-stop latches (the
hold after a fault, until a person types reset) and nothing moves the arm. A
later release at measured rest is tried again; a shutdown closes the robot
app only once a release has worked. The texts name a new start of the robot
app as the way on: it works with every driver, and the reBot arm's driver
takes no second release (test_apai_backend.py).

On the simulated arm, and on APAI's mock driver as test_apai_app.py sets it up.
"""

from __future__ import annotations

import signal
import threading
from collections.abc import Callable, Iterator
from typing import Any

import pytest
from isaaccapture_examples.agentic_remote.robot_app import __main__ as robot_main
from isaaccapture_examples.agentic_remote.robot_app import rest as rest_module
from isaaccapture_examples.agentic_remote.robot_app.app import (
    MOTORS_OFF,
    MOTORS_UNKNOWN,
    NEW_START,
    RELEASE_FAILED,
    TORQUE_UNKNOWN,
)
from isaaccapture_examples.agentic_remote.robot_app.clock import FakeClock
from isaaccapture_examples.agentic_remote.robot_app.config import RobotAppConfig
from isaaccapture_examples.agentic_remote.robot_app.interfaces import (
    ArmFault,
    GripperOutcome,
    MotionState,
)
from isaaccapture_examples.agentic_remote.robot_app.terminal import TypedLines

import test_apai_app as on_apai
from test_apai_app import ApaiHarness, BackendSpy
from test_app import APP_LOGGER, RAISED, REST, WAIT_S, Done, closing_records, final
from test_main import (
    in_the_background,
    left_threads,
    sim_program,
    started_apps,
    until,
)

apai = pytest.mark.apai

# What the simulated arm's failed release and the mock driver's raise say.
SIM_FAULT = f"{RELEASE_FAILED}: bus off"
APAI_FAULT = f"{RELEASE_FAILED}: the driver's disconnect failed: bus off"
SHUTDOWN_DONE = "shutdown: the motors are off at rest; the robot app closes"


def fail_release(h: Any) -> Callable[[], None]:
    """Make the simulated arm's release raise before it switches anything off.

    Returns the call that makes the release work again.
    """
    real = h.arm.release_torque

    def failing() -> None:
        raise ArmFault("bus off")

    def undo() -> None:
        h.arm.release_torque = real

    h.arm.release_torque = failing
    return undo


def fail_disconnect(h: ApaiHarness, after_cut: bool = False) -> Callable[[], None]:
    """Make the mock driver's disconnect raise, before or after it disconnects.

    Returns the call that makes the disconnect work again.
    """
    spied = h.spy.backend.disconnect  # the spy's: it records, then disconnects

    def disconnect() -> None:
        if after_cut:
            spied()
        raise OSError("bus off")

    def undo() -> None:
        h.spy.backend.disconnect = spied

    h.spy.backend.disconnect = disconnect
    return undo


def latched(h: Any) -> Any:
    """The ROBOT_STATE that shows the fault-stop of a failed release."""
    return h.client.wait_state(lambda s: s.stop_level == "fault")


def reset(h: Any) -> Any:
    """A RESET that clears the fault-stop; the ROBOT_STATE that shows it."""
    latched(h)
    result = h.client.wait_result(h.client.reset(), timeout_s=WAIT_S)
    assert result.status == "completed", result.reason
    return h.client.wait_state(lambda s: s.stop_level == "none")


def said_motors_off(h: Any) -> list[str]:
    """The operator events so far that say the motors are off."""
    return [text for _, text, _ in h.local_events if "motors are off" in text]


# What the mock driver's jaws are sent by the close before a torque release,
# once per release that got as far: the close, then the light hold.
CLOSED_AT_REST = [
    (on_apai.JAWS_CLOSED, 1.0),
    (on_apai.JAWS_CLOSED, pytest.approx(on_apai.LIGHT)),
]


def mock_driver_connected(h: ApaiHarness) -> bool:
    """Whether APAI's mock driver still takes a joint target: it is connected."""
    try:
        h.spy.place(REST)
    except RuntimeError as refused:
        assert "not connected" in str(refused)
        return False
    return True


@pytest.fixture
def apai_h(clock: FakeClock) -> Iterator[ApaiHarness]:
    """A started robot app on APAI's mock driver at rest, with a client.

    After the test: the driver was connected once, never stopped, and each
    of its disconnects was asked for at measured rest.
    """
    profile = on_apai.apai_backend.load_apai_profile("mock")
    harness = ApaiHarness(on_apai.make_config(profile), profile, clock)
    harness.start()
    yield harness
    harness.close()
    spy, rest = harness.spy, harness.config.rest
    assert not set(BackendSpy.FORBIDDEN) & set(spy.calls)
    assert spy.calls.count("connect") == 1
    for pose in spy.released_at:
        off = max(abs(at - goal) for at, goal in zip(pose, rest.rest_q))
        assert off <= rest.rest_tol_rad, f"disconnected {off:.3f} rad from rest"


# On the simulated arm


def test_a_failed_release_then_reset_and_shutdown_on_the_simulated_arm(robot_app):
    h = robot_app
    undo = fail_release(h)
    result = final(h, h.client.release())
    assert (result.status, result.reason) == ("failed", f"fault: {SIM_FAULT}")
    state = latched(h)
    assert (state.motion, state.torque_on) == ("release_failed", True)
    assert state.fault_reason == SIM_FAULT
    assert h.app.motion_state is MotionState.RELEASE_FAILED
    # A reset clears the fault-stop and nothing else.
    state = reset(h)
    assert (state.motion, state.torque_on) == ("release_failed", True)
    # A shutdown tries the release again: it fails again, and the app runs on.
    assert h.app.shutdown_moves_arm() is False
    result = final(h, h.client.shutdown())
    assert (result.status, result.reason) == ("failed", f"fault: {SIM_FAULT}")
    assert h.app.running and not h.app.wait_closed(0.0)
    assert h.arm.torque_on() and h.arm.released_at == []
    assert said_motors_off(h) == []
    # Once the release works the same shutdown closes the app.
    undo()
    reset(h)
    assert final(h, h.client.shutdown()).status == "completed"
    assert h.app.wait_closed(WAIT_S)
    assert h.heard("shutdown")[-1][1] == SHUTDOWN_DONE
    assert not h.arm.torque_on() and len(h.arm.released_at) == 1
    assert h.heard("rest") == [] and h.arm.sent == []  # no return to rest


def test_after_a_failed_release_nothing_moves_the_arm(robot_app):
    h = robot_app
    fail_release(h)
    assert final(h, h.client.release()).status == "failed"
    reset(h)
    # As an arm that sagged because some of its motors are off.
    h.arm.set_pose(RAISED)
    refused = h.client.start_task()
    assert (refused.status, refused.reason) == ("refused", MOTORS_UNKNOWN)
    refused = final(h, h.client.open_jaws())
    assert (refused.status, refused.reason) == ("refused", MOTORS_UNKNOWN)
    refused = final(h, h.client.rest())
    assert (refused.status, refused.reason) == ("refused", MOTORS_UNKNOWN)
    assert h.heard("rest")[-1][1] == f"return to rest refused: {MOTORS_UNKNOWN}"
    done = Done()
    assert h.app.request_rest(True, done) is None
    assert (done.wait().status, done.results[0].reason) == ("refused", MOTORS_UNKNOWN)
    # A shutdown starts no return to rest either, and asks for no clear path.
    assert h.app.shutdown_moves_arm() is False
    result = final(h, h.client.shutdown(confirmed=False))
    assert result.status == "failed"
    assert result.reason.startswith(
        f"a torque release failed and {TORQUE_UNKNOWN}; release refused: the arm "
        "is not at rest: joint index 0"
    )
    assert h.heard("shutdown")[-1][1] == (
        f"shutdown failed: {result.reason}; the robot app keeps running; {NEW_START}"
    )
    # The one jaw command is the first release's own close, before it failed.
    assert h.arm.sent == []
    assert [command.opening for command in h.arm.jaw_commands] == [0.0]
    assert h.app.running and h.arm.torque_on()
    assert h.app.motion_state is MotionState.RELEASE_FAILED


def test_a_second_release_at_measured_rest_is_tried_again(robot_app):
    h = robot_app
    undo = fail_release(h)
    assert final(h, h.client.release()).status == "failed"
    latched(h)
    # While the fault-stop is latched a release is refused, with the truth.
    done = Done()
    assert h.app.request_release(done) is None
    refused = done.wait()
    assert refused.reason == f"fault-stop latched: {SIM_FAULT}"
    assert h.heard("refused")[-1][1] == (
        f"release refused: {refused.reason}; {TORQUE_UNKNOWN}; {NEW_START}"
    )
    reset(h)
    undo()
    assert final(h, h.client.release()).status == "completed"
    state = h.client.wait_state(lambda s: s.motion == "rest_unpowered")
    assert not state.torque_on
    assert h.arm.released_at == [pytest.approx(REST, abs=1e-6)]
    assert h.heard("release")[-1][1] == rest_module.RELEASED
    refused = h.client.start_task()
    assert (refused.status, refused.reason) == ("refused", MOTORS_OFF)


def test_a_failed_release_shows_the_motors_off_when_the_arm_port_knows_it(robot_app):
    h = robot_app
    cut = h.arm.release_torque

    def cut_then_fail() -> None:
        cut()
        raise ArmFault("bus off")

    h.arm.release_torque = cut_then_fail
    result = final(h, h.client.release())
    # The simulated arm knows that its motors are off: nothing is unknown.
    assert (result.status, result.reason) == ("failed", "fault: bus off")
    state = latched(h)
    assert (state.motion, state.torque_on) == ("rest_unpowered", False)
    assert len(h.arm.released_at) == 1


def test_close_after_a_failed_release_logs_the_torque_as_not_known(robot_app, caplog):
    h = robot_app
    fail_release(h)
    assert final(h, h.client.release()).status == "failed"
    with caplog.at_level("INFO", logger=APP_LOGGER):
        h.app.close()
    assert closing_records(caplog) == [
        "robot app closed; torque on: not known, a torque release failed: the "
        "motors may be on or off"
    ]


def test_a_fault_stop_after_a_failed_release_names_a_new_start_not_a_reset(robot_app):
    h = robot_app
    fail_release(h)
    assert final(h, h.client.release()).status == "failed"
    state = latched(h)
    assert h.heard("fault")[-1][1] == (
        f"FAULT-STOP: {SIM_FAULT}; end the robot app with SIGTERM and start it "
        f"again with the arm at rest, stop number {state.stop_number}"
    )
    reset(h)
    # A later fault in that state: the arm may not hold, and a reset that
    # clears the fault-stop still leaves nothing that moves the arm.
    h.arm.set_fault("motor 3 over temperature")
    h.step()
    state = latched(h)
    assert h.heard("fault")[-1][1] == (
        f"FAULT-STOP: motor 3 over temperature; {MOTORS_UNKNOWN}, stop number "
        f"{state.stop_number}"
    )
    assert not [text for _, text, _ in h.local_events if "types reset" in text]


def test_sigterm_after_a_failed_release_does_not_say_that_the_motors_are_left_on(
    monkeypatch,
):
    """The robot app as a program on the simulated arm, in real time."""
    # No dial: the program is given a port that nothing may be dialed on.
    monkeypatch.setattr(robot_main.RobotDialer, "start", lambda self: None)
    apps = started_apps(monkeypatch)
    program = sim_program(RobotAppConfig.default())
    arm = program[1]

    def failing() -> None:
        raise ArmFault("bus off")

    arm.release_torque = failing
    # What the program's handler notes at a SIGTERM; no signal is raised here.
    signals = robot_main._StartSignals()
    typed, said = TypedLines(), []
    thread, ended = in_the_background(
        lambda: robot_main.run_connected(
            *program, typed=typed, say=said.append, driver="sim", signals=signals
        )
    )
    try:
        until(lambda: apps and apps[0].running, "the start")
        app = apps[0]
        assert app.request_release() is None
        until(lambda: app.motion_state is MotionState.RELEASE_FAILED, "the release")
        signals.terminated.append(signal.SIGTERM)
        thread.join(10.0)
    finally:
        typed.end()
        for app in apps:
            app.close()
    assert ended == [0]
    assert (
        "robot app: SIGTERM; closing after a torque release that failed: the "
        "motors may be on or off"
    ) in said
    assert not [line for line in said if "motors left on" in line]
    assert arm.torque_on() and arm.released_at == []
    assert left_threads() == []


# On APAI's mock driver


@apai
def test_a_failed_release_then_reset_and_shutdown_on_apais_mock_driver(
    apai_h: ApaiHarness,
) -> None:
    h = apai_h
    jaws_at_start = list(h.spy.jaw_commands)  # the jaw reference
    undo = fail_disconnect(h)
    result = final(h, h.client.release())
    assert (result.status, result.reason) == ("failed", f"fault: {APAI_FAULT}")
    state = latched(h)
    assert (state.motion, state.torque_on) == ("release_failed", True)
    assert state.fault_reason == APAI_FAULT
    assert h.arm.torque_on() and not h.arm.release_confirmed()
    assert mock_driver_connected(h)
    state = reset(h)
    assert (state.motion, state.torque_on) == ("release_failed", True)
    # A shutdown does not close the app on a release that fails again.
    result = h.client.wait_result(h.client.shutdown(), timeout_s=WAIT_S)
    assert (result.status, result.reason) == ("failed", f"fault: {APAI_FAULT}")
    assert h.app.running and not h.app.wait_closed(0.0)
    assert said_motors_off(h) == [] and h.heard("shutdown") == []
    assert mock_driver_connected(h)
    # Once the driver's disconnect works the same shutdown closes the app.
    undo()
    reset(h)
    result = h.client.wait_result(h.client.shutdown(), timeout_s=WAIT_S)
    assert result.status == "completed"
    assert h.app.wait_closed(WAIT_S)
    assert h.heard("shutdown")[-1][1] == SHUTDOWN_DONE
    assert not h.arm.torque_on() and h.arm.release_confirmed()
    assert not mock_driver_connected(h)
    assert h.spy.targets == []
    assert h.spy.jaw_commands == [*jaws_at_start, *CLOSED_AT_REST]


@apai
def test_a_release_that_fails_after_the_driver_disconnected_is_not_known_as_off(
    apai_h: ApaiHarness,
) -> None:
    h = apai_h
    jaws_at_start = list(h.spy.jaw_commands)  # the jaw reference
    undo = fail_disconnect(h, after_cut=True)
    result = final(h, h.client.release())
    assert (result.status, result.reason) == ("failed", f"fault: {APAI_FAULT}")
    assert not mock_driver_connected(h)
    # The adapter was not told so: the torque is not known, and never "off".
    state = latched(h)
    assert (state.motion, state.torque_on) == ("release_failed", True)
    assert not h.arm.release_confirmed()
    refused = h.client.start_task()
    assert (refused.status, refused.reason) == ("refused", MOTORS_UNKNOWN)
    # A second release, at measured rest, is tried again and works.
    undo()
    reset(h)
    assert final(h, h.client.release()).status == "completed"
    state = h.client.wait_state(lambda s: s.motion == "rest_unpowered")
    assert not state.torque_on
    assert h.heard("release")[-1][1] == rest_module.RELEASED
    assert h.spy.calls.count("disconnect") == 2
    assert h.spy.released_at == [pytest.approx(REST), pytest.approx(REST)]
    assert h.spy.targets == []
    assert h.spy.jaw_commands == [*jaws_at_start, *CLOSED_AT_REST]
    refused = h.client.start_task()
    assert (refused.status, refused.reason) == ("refused", MOTORS_OFF)


@apai
def test_the_adapter_counts_a_release_only_once_the_drivers_call_returned(
    clock: FakeClock,
) -> None:
    profile = on_apai.apai_backend.load_apai_profile("mock")
    config = on_apai.make_config(profile)
    arm = on_apai.apai_backend.build_apai_arm(config, profile, clock)
    arm.connect()
    arm.gripper_reference()
    real = arm.backend.disconnect
    jaw_commands: list[tuple[Any, ...]] = []
    arm.backend.set_gripper = lambda *args: jaw_commands.append(args)

    def failing() -> None:
        raise OSError("bus off")

    arm.backend.disconnect = failing
    with pytest.raises(ArmFault, match=r"the driver's disconnect failed: bus off"):
        arm.release_torque()
    assert arm.torque_on() and not arm.release_confirmed()
    # The driver is sent no joint target and no jaw command in that state.
    unknown = "a torque release failed: the motors may be on or off"
    with pytest.raises(ArmFault, match=unknown):
        arm.send_target(REST)
    arm.gripper_light_hold()
    refused = arm.gripper_command(1.0, 0.5, "plain", threading.Event())
    assert (refused.status, refused.detail) == ("refused", unknown)
    assert len(arm.backend.commands) == 0 and jaw_commands == []
    # Its joints are still read: a second release needs them measured at rest.
    assert len(arm.read_all()) == config.n_joints
    arm.backend.disconnect = real
    arm.release_torque()
    assert not arm.torque_on() and arm.release_confirmed()
    refused = arm.gripper_command(1.0, 0.5, "plain", threading.Event())
    off = "the arm is not connected or its motors are off"
    assert refused == GripperOutcome("refused", None, off)
