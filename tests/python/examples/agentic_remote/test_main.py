# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The robot app's command line: arguments, logging, the self-test and --arm apai.

The self-test runs on the machine's clock, so these tests take real seconds.
"""

from __future__ import annotations

import logging
import runpy
import sys
import threading
import types
from collections.abc import Iterator

import pytest
import yaml
from isaaccapture_examples.agentic_remote.robot_app import __main__ as robot_main
from isaaccapture_examples.agentic_remote.robot_app.authority import Authority
from isaaccapture_examples.agentic_remote.robot_app.clock import SystemClock
from isaaccapture_examples.agentic_remote.robot_app.config import RobotAppConfig
from isaaccapture_examples.agentic_remote.robot_app.permits import PermitGate
from isaaccapture_examples.agentic_remote.robot_app.sim import BasicChecker, SimArm
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
