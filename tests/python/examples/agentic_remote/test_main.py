# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The robot app's command line: arguments, logging, the self-test and --arm apai.

The self-test runs on the machine's clock, so these tests take real seconds.
"""

from __future__ import annotations

import dataclasses
import logging
import os
import runpy
import signal
import subprocess
import sys
import threading
import time
import types
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import yaml
from isaaccapture_examples.agentic_remote.agent import config as agent_config
from isaaccapture_examples.agentic_remote.protocol.carrier import loopback_pair
from isaaccapture_examples.agentic_remote.protocol.messages import SIMULATED_DRIVERS
from isaaccapture_examples.agentic_remote.robot_app import __main__ as robot_main
from isaaccapture_examples.agentic_remote.robot_app.authority import Authority
from isaaccapture_examples.agentic_remote.robot_app.clock import SystemClock
from isaaccapture_examples.agentic_remote.robot_app.config import (
    JawsConfig,
    RobotAppConfig,
)
from isaaccapture_examples.agentic_remote.robot_app.permits import PermitGate
from isaaccapture_examples.agentic_remote.robot_app.sim import BasicChecker, SimArm
from isaaccapture_examples.agentic_remote.robot_app.terminal import TypedLines
from isaaccapture_examples.agentic_remote.testing.client import WorkstationClient
from repo_paths import repo_root

CONFIG_YAML = repo_root() / "examples" / "agentic_remote" / "configs" / "robot_app.yaml"
BACKEND = "isaaccapture_examples.agentic_remote.robot_app.apai_backend"
LOGGER = "isaaccapture.examples.agentic_remote"
# Real seconds the whole self-test may take; it needs about 5 on a free machine.
SELFTEST_BOUND_S = 30.0
STEPS = 11
# Threads the self-test starts; none may outlive it.
THREADS = ("robot-app-", "gripper-worker", "thor-terminal-", "workstation-client")
NO_MODE = "--arm sim needs --connect HOST:PORT, --selftest or --selftest-link"


@pytest.fixture(autouse=True)
def console_logging_undone(monkeypatch) -> Iterator[None]:
    """Keep main()'s logging setup inside its test, and Isaac Capture out of it."""
    monkeypatch.setitem(sys.modules, "isaaccapture", None)  # not importable
    logger = logging.getLogger(LOGGER)
    level, handlers = logger.level, list(logger.handlers)
    yield
    logger.setLevel(level)
    for handler in list(logger.handlers):
        if handler not in handlers:
            logger.removeHandler(handler)
            handler.close()


@pytest.fixture
def releases(monkeypatch) -> list[list[float]]:
    """The pose at every torque release of a simulated arm, in rad."""
    seen: list[list[float]] = []
    real = SimArm.release_torque

    def release_torque(self: SimArm) -> None:
        seen.append(self.pose())
        real(self)

    monkeypatch.setattr(SimArm, "release_torque", release_torque)
    return seen


def quick_config(tmp_path, changes: dict[str, object] | None = None) -> str:
    """The shipped file with quicker moves, rest and permit wait, then changed.

    changes maps a dotted setting such as "permits.first_wait_s" to its value.
    """
    data = yaml.safe_load(CONFIG_YAML.read_text(encoding="utf-8"))
    quick = {
        "rest.min_duration_s": 0.3,
        "rest.recovery_speed": 0.4,
        "rest.read_gap_s": 0.05,
        "limits.recovery.max_joint_speed": 0.4,
        "limits.agent.max_joint_speed": 1.2,
        "permits.first_wait_s": 0.05,
        "gripper.read_gap_s": 0.01,
    }
    for dotted, value in {**quick, **(changes or {})}.items():
        *blocks, name = dotted.split(".")
        block = data
        for key in blocks:
            block = block[key]
        block[name] = value
    path = tmp_path / "robot_app.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return str(path)


def selftest(config: str, capsys) -> tuple[int, list[str]]:
    """Run the self-test in this process: its exit code and what it printed."""
    code = robot_main.main(["--config", config, "--arm", "sim", "--selftest"])
    return code, capsys.readouterr().out.splitlines()


def step_lines(lines: list[str]) -> list[str]:
    """The one line of each step that ran, without the Thor terminal's lines."""
    return [line for line in lines if line.startswith("selftest ") and "/" in line]


def left_threads() -> list[str]:
    return [
        thread.name
        for thread in threading.enumerate()
        if thread.name.startswith(THREADS)
    ]


# Arguments


@pytest.mark.parametrize(
    "argv",
    [
        [],
        ["--arm", "sim"],
        ["--config", str(CONFIG_YAML)],
        ["--config", str(CONFIG_YAML), "--arm", "robot"],
        ["--config", str(CONFIG_YAML), "--arm", "sim", "--log-level", "loud"],
        ["--config", str(CONFIG_YAML), "--arm", "sim", "--selftest", "--fast"],
        ["--config", str(CONFIG_YAML), "--arm", "sim", "selftest"],
    ],
)
def test_arguments_that_cannot_be_used_give_exit_code_2(argv, capsys, monkeypatch):
    monkeypatch.setattr(robot_main, "run_selftest", None)  # must not be reached
    assert robot_main.main(argv) == 2
    captured = capsys.readouterr()
    assert captured.err.startswith(
        "usage: python -m isaaccapture_examples.agentic_remote.robot_app"
    )
    assert captured.out == ""


def test_help_names_every_argument_and_exits_with_0(capsys):
    assert robot_main.main(["--help"]) == 0
    out = capsys.readouterr().out
    for argument in (
        "--config PATH",
        "--arm {sim,apai}",
        "--connect HOST:PORT",
        "--selftest",
        "--selftest-link",
        "--log-level",
        "--driver {mock,real}",
        "--apai-profile NAME",
        "--calibration NAME",
    ):
        assert argument in out


def test_a_missing_configuration_file_gives_exit_code_2(tmp_path, capsys):
    missing = str(tmp_path / "nowhere.yaml")
    assert robot_main.main(["--config", missing, "--arm", "sim", "--selftest"]) == 2
    err = capsys.readouterr().err
    assert err.startswith("robot app: the configuration cannot be used: ")
    assert "nowhere.yaml" in err


def test_a_configuration_that_fails_its_checks_gives_exit_code_2(tmp_path, capsys):
    config = quick_config(tmp_path, {"waits.assistance_s": 60.0})
    assert robot_main.main(["--config", config, "--arm", "sim", "--selftest"]) == 2
    err = capsys.readouterr().err
    assert "waits.assistance_s must be at least 90 s" in err
    assert left_threads() == []


def test_arm_sim_without_selftest_has_nothing_to_connect_to(capsys, monkeypatch):
    monkeypatch.setattr(robot_main, "RobotApp", None)  # no app may be built
    assert robot_main.main(["--config", str(CONFIG_YAML), "--arm", "sim"]) == 2
    captured = capsys.readouterr()
    assert captured.err == f"robot app: {NO_MODE}\n"
    assert captured.out == ""


def test_the_selftest_is_for_the_simulated_arm_only(capsys, monkeypatch):
    monkeypatch.setitem(sys.modules, BACKEND, None)  # must not be imported
    argv = ["--config", str(CONFIG_YAML), "--arm", "apai", "--selftest"]
    assert robot_main.main(argv) == 2
    assert capsys.readouterr().err == (
        "robot app: --selftest runs on the simulated arm only: use --arm sim\n"
    )


@pytest.mark.filterwarnings("ignore:.*found in sys.modules:RuntimeWarning")
def test_the_package_runs_as_a_program_with_mains_exit_code(capsys, monkeypatch):
    argv = ["robot_app", "--config", str(CONFIG_YAML), "--arm", "sim"]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as exit_:
        runpy.run_module(
            "isaaccapture_examples.agentic_remote.robot_app", run_name="__main__"
        )
    assert exit_.value.code == 2
    assert NO_MODE in capsys.readouterr().err


# Logging


def test_without_isaac_capture_one_stream_handler_is_attached(monkeypatch):
    monkeypatch.setattr(logging, "basicConfig", None)  # must not be called
    logger = logging.getLogger(LOGGER)
    before = list(logger.handlers)
    elsewhere = [logging.getLogger(), logging.getLogger("isaaccapture")]
    handlers_elsewhere = [list(other.handlers) for other in elsewhere]
    argv = ["--config", str(CONFIG_YAML), "--arm", "sim", "--log-level"]
    assert robot_main.main([*argv, "info"]) == 2
    added = [handler for handler in logger.handlers if handler not in before]
    assert len(added) == 1 and type(added[0]) is logging.StreamHandler
    assert logger.level == logging.INFO
    # A second run in the same process changes the level and adds no handler.
    assert robot_main.main([*argv, "DEBUG"]) == 2
    assert [handler for handler in logger.handlers if handler not in before] == added
    assert logger.level == logging.DEBUG
    assert [list(other.handlers) for other in elsewhere] == handlers_elsewhere


def test_the_default_log_level_is_warning():
    assert robot_main.main(["--config", str(CONFIG_YAML), "--arm", "sim"]) == 2
    assert logging.getLogger(LOGGER).level == logging.WARNING


def test_with_isaac_capture_its_console_level_is_set_and_no_handler(monkeypatch):
    levels: list[str] = []
    isaaccapture = types.ModuleType("isaaccapture")
    isaaccapture.logging_config = types.SimpleNamespace(set_console_level=levels.append)
    monkeypatch.setitem(sys.modules, "isaaccapture", isaaccapture)
    logger = logging.getLogger(LOGGER)
    before, level = list(logger.handlers), logger.level
    argv = ["--config", str(CONFIG_YAML), "--arm", "sim"]
    assert robot_main.main([*argv, "--log-level", "Error"]) == 2
    assert robot_main.main(argv) == 2
    assert levels == ["error", "warning"]
    assert logger.handlers == before and logger.level == level


# The self-test


def test_the_selftest_passes_on_the_shipped_configuration(capsys, releases):
    clock = SystemClock()
    began = clock.monotonic()
    code, lines = selftest(str(CONFIG_YAML), capsys)
    took = clock.monotonic() - began
    assert code == 0, "\n".join(lines)
    assert took < SELFTEST_BOUND_S
    steps = step_lines(lines)
    assert [line.split()[1:3] for line in steps] == [
        [f"{number}/{STEPS}", "ok"] for number in range(1, STEPS + 1)
    ]
    titles = [line.split(" ok  ")[1].split(":")[0] for line in steps]
    assert titles == [
        "start the robot app",
        "take ownership for the agent",
        "a move with permits",
        "a move without a permit",
        "STOP during a move",
        "a new task after the STOP, no reset",
        "the agent asks for help",
        "stop typed at the Thor terminal",
        "rest typed at the Thor terminal",
        "release at rest",
        "quit typed at the Thor terminal",
    ]
    assert "refused, no setpoint sent" in steps[3]
    assert steps[4].endswith("the arm holds, stop number 1")
    assert steps[5].endswith(
        "the agent owns the arm, owner number 3; a move quoting the old stop "
        "number is refused"
    )
    assert lines[-1].startswith(f"selftest: all {STEPS} steps passed in ")
    assert not any("FAILED" in line for line in lines)
    # What the Thor terminal showed during the run.
    shown = [line.split(" | ", 1)[1] for line in lines if " thor terminal | " in line]
    for line in (
        "> stop",
        'Return to rest: is the path to the rest pose clear? Type "path is clear" '
        "to confirm.",
        "> path is clear",
        "rest: completed",
        "quit: completed; the robot app closes",
    ):
        assert line in shown
    asked = (
        'Return to rest: is the path to the rest pose clear? Type "path is clear" '
        "to confirm."
    )
    assert shown.index("> rest") < shown.index(asked) < shown.index("> path is clear")
    # The quit comes with the motors off at rest: it moves nothing and asks nothing.
    assert shown.count(asked) == 1
    assert "nothing was asked: the arm did not move" in steps[10]
    typed_stop = "[stop] STOP from thor_terminal: typed stop; the arm holds"
    assert any(line.startswith(typed_stop) for line in shown)
    # One release, at rest, and nothing left running.
    rest = RobotAppConfig.default().rest
    assert len(releases) == 1
    assert releases[0] == pytest.approx(list(rest.rest_q), abs=rest.rest_tol_rad)
    assert left_threads() == []


def test_the_selftest_follows_another_configuration(tmp_path, capsys, releases):
    rest_q = [0.3, -0.2, 0.1, 0.0, -0.4, 0.25]
    code, lines = selftest(quick_config(tmp_path, {"rest.rest_q": rest_q}), capsys)
    assert code == 0, "\n".join(lines)
    assert len(step_lines(lines)) == STEPS
    assert len(releases) == 1
    assert releases[0] == pytest.approx(rest_q, abs=0.05)


def test_a_configuration_without_permits_fails_the_no_permit_step(
    tmp_path, capsys, releases
):
    config = quick_config(tmp_path, {"permits.required_for_agent": False})
    code, lines = selftest(config, capsys)
    assert code == 1
    steps = step_lines(lines)
    assert [line.split()[1:3] for line in steps] == [
        [f"1/{STEPS}", "ok"],
        [f"2/{STEPS}", "ok"],
        [f"3/{STEPS}", "ok"],
        [f"4/{STEPS}", "FAILED"],
    ]
    assert steps[-1].endswith(
        "a move without a permit: the move ended completed, not refused: "
        "this configuration switches permits off"
    )
    assert lines[-1] == f"selftest: FAILED at step 4 of {STEPS}"
    assert releases == []
    assert left_threads() == []


def test_an_arm_that_advances_without_a_permit_fails_the_selftest(
    tmp_path, capsys, releases, monkeypatch
):
    real = PermitGate.covers

    def covers(self: PermitGate, move_id: str, index: int, now_mono: float):
        # A broken gate: the first setpoint of every move counts as permitted.
        return None if index == 0 else real(self, move_id, index, now_mono)

    monkeypatch.setattr(PermitGate, "covers", covers)
    code, lines = selftest(quick_config(tmp_path), capsys)
    assert code == 1
    assert step_lines(lines)[-1] == (
        f"selftest  4/{STEPS} FAILED  a move without a permit: a setpoint was sent "
        "without a permit"
    )
    assert releases == []
    assert left_threads() == []


def test_a_stop_that_does_not_end_the_move_fails_the_selftest(
    tmp_path, capsys, releases, monkeypatch
):
    monkeypatch.setattr(WorkstationClient, "stop", lambda self, *args: None)
    code, lines = selftest(quick_config(tmp_path), capsys)
    assert code == 1
    assert step_lines(lines)[-1] == (
        f"selftest  5/{STEPS} FAILED  STOP during a move: RESULT completed, not "
        "cancelled"
    )
    assert releases == []
    assert left_threads() == []


def test_a_robot_app_that_takes_an_old_stop_number_fails_the_selftest(
    tmp_path, capsys, releases, monkeypatch
):
    real = Authority.admit

    def admit(self: Authority, session_id, owner_number, stop_number, required):
        # A broken rule: every command counts as quoting the current stop number.
        current = self.snapshot().stop_number
        return real(self, session_id, owner_number, current, required)

    monkeypatch.setattr(Authority, "admit", admit)
    code, lines = selftest(quick_config(tmp_path), capsys)
    assert code == 1
    assert step_lines(lines)[-1] == (
        f"selftest  6/{STEPS} FAILED  a new task after the STOP, no reset: RESULT "
        "cancelled, not refused: stop number is now 1, the move was admitted at 0"
    )
    assert lines[-1] == f"selftest: FAILED at step 6 of {STEPS}"
    assert releases == []
    assert left_threads() == []


def test_a_refused_return_to_rest_fails_the_selftest_with_the_motors_on(
    tmp_path, capsys, releases, monkeypatch
):
    blocked = "the gripper tip is below the table clearance"
    monkeypatch.setattr(BasicChecker, "tip_blocked", lambda self, q, region: blocked)
    code, lines = selftest(quick_config(tmp_path), capsys)
    assert code == 1
    assert step_lines(lines)[-1] == (
        f"selftest  9/{STEPS} FAILED  rest typed at the Thor terminal: the return to "
        f"rest ended refused: {blocked}"
    )
    assert any(line.endswith(f"| rest: refused: {blocked}") for line in lines)
    assert releases == []
    assert left_threads() == []


def test_a_release_away_from_rest_fails_the_selftest(tmp_path, capsys, monkeypatch):
    real = SimArm.release_torque

    def release_torque(self: SimArm) -> None:
        real(self)
        self.released_at[-1] = [0.3] * self.n_joints  # as if the arm were raised

    monkeypatch.setattr(SimArm, "release_torque", release_torque)
    code, lines = selftest(quick_config(tmp_path), capsys)
    assert code == 1
    assert step_lines(lines)[-1] == (
        f"selftest 10/{STEPS} FAILED  release at rest: torque was released 0.300 rad "
        "away from rest"
    )
    assert lines[-1] == f"selftest: FAILED at step 10 of {STEPS}"
    assert left_threads() == []


def test_a_faulty_arm_fails_the_selftest_and_nothing_is_released(
    tmp_path, capsys, releases, monkeypatch
):
    monkeypatch.setattr(SimArm, "faults", lambda self: ["motor 2 is too hot"])
    code, lines = selftest(quick_config(tmp_path), capsys)
    assert code == 1
    steps = step_lines(lines)
    assert len(steps) == 2  # no later step ran
    assert f" 2/{STEPS} FAILED  take ownership for the agent: " in steps[1]
    assert "fault-stop latched: motor 2 is too hot" in steps[1]
    assert lines[-1] == f"selftest: FAILED at step 2 of {STEPS}"
    assert any("[fault] FAULT-STOP: motor 2 is too hot" in line for line in lines)
    assert releases == []
    assert left_threads() == []


def test_a_step_that_raises_fails_the_selftest(tmp_path, capsys, releases, monkeypatch):
    def run_move(self, *args, **kwargs):
        raise RuntimeError("the workstation fell over")

    monkeypatch.setattr(WorkstationClient, "run_move", run_move)
    code, lines = selftest(quick_config(tmp_path), capsys)
    assert code == 1
    assert step_lines(lines)[-1] == (
        f"selftest  3/{STEPS} FAILED  a move with permits: the workstation fell over"
    )
    assert lines[-1] == f"selftest: FAILED at step 3 of {STEPS}"
    assert releases == []
    assert left_threads() == []


def test_a_robot_app_that_does_not_start_fails_the_first_step(
    tmp_path, capsys, releases, monkeypatch
):
    def connect(self):
        raise OSError("no motor answers")

    monkeypatch.setattr(SimArm, "connect", connect)
    code, lines = selftest(quick_config(tmp_path), capsys)
    assert code == 1
    assert step_lines(lines) == [
        f"selftest  1/{STEPS} FAILED  start the robot app: no motor answers"
    ]
    assert releases == []
    assert left_threads() == []


def test_the_simulated_checkers_take_the_configured_reads_per_step():
    config = RobotAppConfig.default()
    checkers = robot_main.sim_checkers(config, SystemClock())
    assert [type(checker) for checker in checkers.all()] == [BasicChecker] * 3
    assert [checker.name for checker in checkers.all()] == [
        "agent",
        "human",
        "recovery",
    ]
    reads = config.executor.reads_per_step
    assert [checker.reads_per_step for checker in checkers.all()] == [reads] * 3
    # Nothing can feed the recovery checker's watchdog, so it has none.
    assert checkers.agent.watchdog_s is not None
    assert checkers.recovery.watchdog_s is None


# --arm apai


def test_arm_apai_without_apai_says_what_is_missing(capsys, monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, BACKEND, None)  # its import fails
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setenv(robot_main.APAI_SRC_ENV, str(tmp_path))
    assert robot_main.main(["--config", str(CONFIG_YAML), "--arm", "apai"]) == 2
    captured = capsys.readouterr()
    assert captured.err.startswith(
        "robot app: --arm apai needs APAI on the Python path (set "
        "AGENTIC_REMOTE_APAI_SRC to its src folder): "
    )
    assert captured.out == ""
    # The folder the variable names was tried.
    assert sys.path[0] == str(tmp_path)


@pytest.fixture
def apai_yaml(tmp_path) -> str:
    """The shipped configuration with the human joint speed at 0.8 rad/s.

    The shipped 1.5 rad/s is above the speed cap of both APAI profiles used
    here (mock 1.2, rebot_rs 0.8), and a limit set above the cap is refused.
    """
    text = CONFIG_YAML.read_text(encoding="utf-8")
    assert text.count("max_joint_speed: 1.5") == 1
    path = tmp_path / "robot_app_apai.yaml"
    path.write_text(
        text.replace("max_joint_speed: 1.5", "max_joint_speed: 0.8"), encoding="utf-8"
    )
    return str(path)


@pytest.mark.apai
@pytest.mark.parametrize(("profile", "cap"), [("mock", "1.2"), ("rebot_rs", "0.8")])
def test_arm_apai_refuses_a_limit_set_above_the_profiles_speed_cap(
    capsys, monkeypatch, profile, cap
):
    monkeypatch.setattr(sys, "path", list(sys.path))  # the driver adds its SDK path
    argv = ["--config", str(CONFIG_YAML), "--arm", "apai", "--apai-profile", profile]
    assert robot_main.main(argv) == 2
    captured = capsys.readouterr()
    assert captured.err == (
        f"robot app: APAI arm profile {profile!r} cannot be used with {CONFIG_YAML}: "
        "limits.human.max_joint_speed 1.5 rad/s is above the APAI profile's speed "
        f"cap of {cap} rad/s (safety.max_joint_vel)\n"
    )
    assert captured.out == ""


@pytest.mark.apai
def test_arm_apai_builds_the_backend_and_never_connects(capsys, monkeypatch, apai_yaml):
    from isaaccapture_examples.agentic_remote.robot_app import apai_backend
    from apai.control.mock_arm import MockArm

    touched: list[str] = []

    def forbidden(name: str):
        def call(self, *args, **kwargs):
            touched.append(name)
            raise AssertionError(f"{name} was called")

        return call

    for name in ("connect", "release_torque", "send_target", "gripper_reference"):
        monkeypatch.setattr(apai_backend.ApaiArm, name, forbidden(f"ApaiArm.{name}"))
    for name in ("connect", "disconnect"):
        monkeypatch.setattr(MockArm, name, forbidden(f"MockArm.{name}"))
    built: list[object] = []
    for name in ("build_apai_checkers", "build_apai_arm"):
        real = getattr(apai_backend, name)

        def build(*args, real=real, **kwargs):
            built.append(real(*args, **kwargs))
            return built[-1]

        monkeypatch.setattr(apai_backend, name, build)

    assert robot_main.main(["--config", apai_yaml, "--arm", "apai"]) == 2
    assert touched == []
    checkers, arm = built
    assert type(arm) is apai_backend.ApaiArm and type(arm.backend) is MockArm
    assert not arm.torque_on()
    assert len(checkers.all()) == 3
    captured = capsys.readouterr()
    lines = captured.out.splitlines()
    assert lines[0] == "robot app with --arm apai would run:"
    assert "  APAI arm profile mock: driver MockArm, built, not connected" in lines
    # The mock arm's jaw position is no angle: the configured fraction stays.
    assert lines[-3] == (
        "  jaws proven empty: within 0.03 of the jaw travel of the last jaw target"
    )
    # The mock profile names no rest check tolerance: APAI's default, 0.12 rad.
    assert lines[-2] == (
        "  at rest at start: within 0.12 rad of the rest pose (the profile's); "
        "a release: within 0.05 rad"
    )
    assert lines[-1] == (
        "no --connect HOST:PORT was given: the arm was not connected and nothing "
        "runs (exit code 2)"
    )
    assert captured.err == ""
    assert left_threads() == []


@pytest.mark.apai
def test_arm_apai_gives_the_jaw_tolerance_of_a_real_arm_profile_in_rad(
    capsys, monkeypatch, apai_yaml
):
    monkeypatch.setattr(sys, "path", list(sys.path))  # the driver adds its SDK path
    argv = ["--config", apai_yaml, "--arm", "apai", "--apai-profile", "rebot_rs"]
    assert robot_main.main(argv) == 2
    lines = capsys.readouterr().out.splitlines()
    assert (
        "  APAI arm profile rebot_rs: driver RebotRSArm, built, not connected" in lines
    )
    assert lines[-3] == (
        "  jaws proven empty: within 0.0161 of the jaw travel of the last jaw target "
        "(0.1 rad of 6.2 rad)"
    )
    assert left_threads() == []


@pytest.mark.apai
def test_arm_apai_with_a_profile_that_cannot_be_used_gives_exit_code_2(capsys):
    argv = ["--config", str(CONFIG_YAML), "--arm", "apai", "--apai-profile", "no_arm"]
    assert robot_main.main(argv) == 2
    captured = capsys.readouterr()
    assert captured.err.startswith(
        f"robot app: APAI arm profile 'no_arm' cannot be used with {CONFIG_YAML}: "
    )
    assert captured.out == ""


@pytest.mark.apai
@pytest.mark.parametrize(("profile", "cap"), [("mock", "1.2"), ("rebot_rs", "0.8")])
def test_arm_apai_names_the_profiles_speed_cap(
    capsys, monkeypatch, apai_yaml, profile, cap
):
    monkeypatch.setattr(sys, "path", list(sys.path))  # the driver adds its SDK path
    argv = ["--config", apai_yaml, "--arm", "apai", "--apai-profile", profile]
    assert robot_main.main(argv) == 2
    lines = capsys.readouterr().out.splitlines()
    assert lines[-4] == (
        f"  the profile's speed cap: {cap} rad/s; the limit sets' joint speeds: "
        "agent 0.4, human 0.8, recovery 0.1 rad/s"
    )


# What WELCOME declares, and the setting jaws.allow_reference


class Dialed:
    """Stands in for LinkCarrier in the program: each dial is a loopback link."""

    def __init__(self) -> None:
        self.workstation_ends: list[Any] = []
        self._made = threading.Condition()

    def connect(self, endpoint: str, timeout_s: float, *, name: str = "robot") -> Any:
        robot_end, workstation_end = loopback_pair()
        with self._made:
            self.workstation_ends.append(workstation_end)
            self._made.notify_all()
        return robot_end

    def first(self) -> Any:
        """The workstation's end of the first dial."""
        with self._made:
            assert self._made.wait_for(lambda: self.workstation_ends, 10.0)
            return self.workstation_ends[0]


def built_apps(monkeypatch) -> list[dict[str, Any]]:
    """The keywords of every RobotApp the program builds from here on."""
    built: list[dict[str, Any]] = []
    real = robot_main.RobotApp

    def robot_app(*args: Any, **kwargs: Any) -> Any:
        built.append(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(robot_main, "RobotApp", robot_app)
    return built


def unconnected_arm(config: RobotAppConfig, clock: SystemClock) -> SimArm:
    """A simulated arm whose connect fails: the program ends before it dials."""
    arm = SimArm(n_joints=6, clock=clock, q0=list(config.rest.rest_q))

    def no_bus() -> None:
        raise RuntimeError("the CAN bus is down")

    arm.connect = no_bus
    return arm


def invented_times_settings(folder: Path) -> Any:
    """Agent settings for --camera mock: its capture times are made up."""
    data = {"camera": {"allow_invented_times": True}, "state_dir": str(folder)}
    path = folder / "agent.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return agent_config.load_settings(path)


@pytest.mark.parametrize(
    ("arm", "driver"),
    [
        (["--arm", "sim"], "sim"),
        pytest.param(["--arm", "apai"], "apai-mock", marks=pytest.mark.apai),
        pytest.param(
            ["--arm", "apai", "--driver", "mock"], "apai-mock", marks=pytest.mark.apai
        ),
    ],
)
def test_sim_and_mock_arms_declare_a_simulated_driver(
    monkeypatch, tmp_path, apai_yaml, arm, driver
):
    links, typed, returned = Dialed(), TypedLines(), []
    built = built_apps(monkeypatch)
    monkeypatch.setattr(robot_main, "LinkCarrier", links)
    monkeypatch.setattr(robot_main, "_remote_link_missing", lambda: None)
    monkeypatch.setattr(robot_main, "typed_lines", lambda: typed)
    argv = ["--config", apai_yaml, *arm, "--connect", "127.0.0.1:47610"]
    program = threading.Thread(
        target=lambda: returned.append(robot_main.main(argv)), daemon=True
    )
    program.start()
    client = WorkstationClient(links.first(), SystemClock())
    try:
        welcome = client.wait_welcome(10.0)
        assert welcome.driver == driver
        assert driver in SIMULATED_DRIVERS
        # The agent program takes --camera mock against this robot app; against
        # one that declares no driver it ends.
        settings = invented_times_settings(tmp_path)
        agent_config.check_robot_driver(settings, welcome.driver, invented=True)
        with pytest.raises(agent_config.ConfigError, match="allow_invented_times"):
            agent_config.check_robot_driver(settings, "", invented=True)
        state = client.wait_state(lambda s: s.motion == "rest_hold", 10.0)
        assert state.torque_on
        typed.type("quit")
        program.join(20.0)
        assert returned == [0]
    finally:
        typed.end()
        client.close()
    assert [kwargs["driver"] for kwargs in built] == [driver]
    assert built[0]["allow_reference"] is False
    assert left_threads() == []


def test_the_two_selftests_declare_the_simulated_driver(monkeypatch, tmp_path):
    built = built_apps(monkeypatch)
    config = RobotAppConfig.default()
    said: list[str] = []
    for run in (
        robot_main._SelfTest(config, said.append),
        robot_main._LinkSelfTest(config, said.append, str(tmp_path)),
    ):
        run._close()  # nothing was started
    assert [kwargs["driver"] for kwargs in built] == ["sim", "sim"]
    assert [kwargs["allow_reference"] for kwargs in built] == [False, False]
    assert left_threads() == []


@pytest.mark.parametrize("allowed", [False, True])
def test_every_robot_app_takes_the_files_jaw_reference_setting(
    monkeypatch, tmp_path, allowed
):
    built = built_apps(monkeypatch)
    config = dataclasses.replace(
        RobotAppConfig.default(), jaws=JawsConfig(allow_reference=allowed)
    )
    clock = SystemClock()
    code = robot_main.run_connected(
        config,
        unconnected_arm(config, clock),
        robot_main.sim_checkers(config, clock),
        clock,
        "127.0.0.1:47610",
        typed=iter(()),
        driver="sim",
    )
    assert code == 1
    for run in (
        robot_main._SelfTest(config, lambda line: None),
        robot_main._LinkSelfTest(config, lambda line: None, str(tmp_path)),
    ):
        run._close()
    assert [kwargs["allow_reference"] for kwargs in built] == [allowed] * 3
    assert left_threads() == []


def test_run_connected_declares_the_driver_its_caller_names(monkeypatch):
    built = built_apps(monkeypatch)
    config, clock = RobotAppConfig.default(), SystemClock()
    arm = unconnected_arm(config, clock)
    checkers = robot_main.sim_checkers(config, clock)
    # Without a name WELCOME declares nothing, which every reader takes for real.
    assert (
        robot_main.run_connected(config, arm, checkers, clock, "h:1", typed=iter(()))
        == 1
    )
    with pytest.raises(ValueError, match="driver must be one of"):
        robot_main.run_connected(config, arm, checkers, clock, "h:1", driver="mock")
    assert [kwargs["driver"] for kwargs in built] == ["", "mock"]
    assert left_threads() == []


def test_ctrl_c_while_the_arm_connects_is_a_stop_once_the_app_has_started(monkeypatch):
    config, clock = RobotAppConfig.default(), SystemClock()
    arm = SimArm(n_joints=6, clock=clock, q0=list(config.rest.rest_q))
    connect = arm.connect

    def interrupted() -> None:
        connect()
        signal.raise_signal(signal.SIGINT)  # what Ctrl+C sends

    arm.connect = interrupted
    monkeypatch.setattr(robot_main.RobotDialer, "start", lambda self: None)
    apps: list[Any] = []
    real = robot_main.RobotApp

    def robot_app(*args: Any, **kwargs: Any) -> Any:
        apps.append(real(*args, **kwargs))
        return apps[-1]

    monkeypatch.setattr(robot_main, "RobotApp", robot_app)
    stops: list[int] = []

    def close_after_the_stop() -> None:
        by = time.monotonic() + 10.0
        while time.monotonic() < by and not (apps and apps[0].snapshot().stop_number):
            time.sleep(0.02)
        for app in apps:
            stops.append(app.snapshot().stop_number)
            app.close()

    typed = TypedLines()
    before = signal.getsignal(signal.SIGINT)
    closer = threading.Thread(target=close_after_the_stop, daemon=True)
    closer.start()
    try:
        code = robot_main.run_connected(
            config,
            arm,
            robot_main.sim_checkers(config, clock),
            clock,
            "127.0.0.1:47610",
            typed=typed,
            say=lambda line: None,
            driver="sim",
        )
    except KeyboardInterrupt:
        code = None
    finally:
        typed.end()
        closer.join(15.0)
    assert code == 0 and stops == [1]
    assert arm.torque_on() and arm.released_at == []
    assert signal.getsignal(signal.SIGINT) is before
    assert left_threads() == []


# The Thor start script

EXAMPLE = CONFIG_YAML.parents[1]


def test_run_robot_app_sh_sets_the_environment_and_names_what_is_missing(tmp_path):
    script = EXAMPLE / "run_robot_app.sh"
    text = script.read_text(encoding="utf-8")
    assert os.access(script, os.X_OK)
    assert "[ " not in text.replace("[[ ", "")  # [[ ]] only
    assert "set -euo pipefail" in text
    assert "SPDX-License-Identifier: Apache-2.0" in text
    env = {"PATH": os.environ["PATH"], "APAI_ARM_HOLD_DIR": "/elsewhere"}
    fake = tmp_path / "python"
    fake.write_text(
        "#!/usr/bin/env bash\n"
        'echo "${APAI_ROBOT:-unset} ${APAI_BASE:-unset} ${APAI_ARM_HOLD_DIR:-unset} '
        '$PYTHONDONTWRITEBYTECODE $AGENTIC_REMOTE_APAI_SRC $PYTHONPATH $*"\n',
        encoding="utf-8",
    )
    fake.chmod(0o755)
    package = tmp_path / "isaac_capture_py"
    package.mkdir()
    needed = {
        "AGENTIC_REMOTE_THOR_PYTHON": (str(fake), "the Python of APAI's environment"),
        "AGENTIC_REMOTE_APAI_SRC": (str(tmp_path), "the src folder of the APAI"),
        "AGENTIC_REMOTE_ISAAC_CAPTURE_PY": (
            str(package),
            "the folder of the Isaac Capture Python package",
        ),
    }
    for name, (value, said) in needed.items():
        # One line and exit code 2 without the variable, and with a wrong one.
        for wrong in ({}, {name: str(tmp_path / "missing")}):
            done = subprocess.run(
                [str(script), "--help"],
                check=False,
                env={**env, **wrong},
                capture_output=True,
                text=True,
                timeout=60,
            )
            assert done.returncode == 2 and done.stdout == ""
            assert done.stderr.startswith(f"run_robot_app.sh: {name} must name {said}")
            assert done.stderr.count("\n") == 1
        env[name] = value
    env.update(APAI_ROBOT="another", APAI_BASE="another", PYTHONPATH="/before")
    done = subprocess.run(
        [str(script), "--config", "configs/robot_app.yaml", "--arm", "sim"],
        check=False,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        cwd=tmp_path,
    )
    assert done.returncode == 0, done.stderr
    assert done.stdout.split() == [
        *("unset", "unset", "unset", "1"),
        str(tmp_path),
        f"{EXAMPLE}/python:{package}:/before",
        *("-m", "isaaccapture_examples.agentic_remote.robot_app"),
        *("--config", "configs/robot_app.yaml", "--arm", "sim"),
    ]
