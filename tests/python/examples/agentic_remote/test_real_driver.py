# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The robot app's --driver real: its checks, its three typed confirmations and
the connect.

The connect cuts motor torque. The first half runs without APAI, on a stand-in
for the backend module. The tests marked "apai" run APAI's real driver class,
its connect() included, over a stand-in motor SDK that records every call made
to it: no test here talks to a motor bus.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import pty
import signal
import sys
import threading
import time
import tty
import types
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml
from isaaccapture_examples.agentic_remote.protocol import confirm
from isaaccapture_examples.agentic_remote.robot_app import __main__ as robot_main
from isaaccapture_examples.agentic_remote.robot_app.clock import SystemClock
from isaaccapture_examples.agentic_remote.robot_app.config import load_config
from isaaccapture_examples.agentic_remote.robot_app.terminal import TypedLines
from isaaccapture_examples.agentic_remote.testing.client import WorkstationClient
from repo_paths import repo_root

from test_apai_backend import (
    LAUNCHER_SPEED,
    RIG_REST_READINGS,
    GuardTime,
    JawGroup,
    Motor,
    MotorGroup,
)
from test_main import quick_config
from test_takeover import setpoint

CONFIGS = repo_root() / "examples" / "agentic_remote" / "configs"
RIG_YAML = CONFIGS / "robot_app_rebot.yaml"
BACKEND = "isaaccapture_examples.agentic_remote.robot_app.apai_backend"
LOGGER = "isaaccapture.examples.agentic_remote"
ENDPOINT = "127.0.0.1:47610"
WAIT_S = 20.0  # real seconds a test gives the program before it fails
# Real seconds a typed rest may take on the rig's file: the leg up to the
# ready pose takes 12.5 s at the file's 0.2 rad/s, the four fold steps 3 s each.
REST_WAIT_S = 90.0
# How the line begins that the program writes once the real driver is
# connected at measured rest. No joint target is in force then: the motors are
# switched on, and whether they hold is not known from a reading.
CONNECTED = (
    "robot app: the real driver is connected at measured rest: torque on, hold unknown"
)
QUESTIONS = [
    "Real driver, 1 of 3: Is the arm powered? Type yes to confirm.",
    "Real driver, 2 of 3: Connecting cuts motor torque. Is the arm at rest and "
    "supported? Type yes to confirm.",
    "Real driver, 3 of 3: Is the calibration in use 'bench-7'? Type yes to confirm.",
]
NOT_CONFIRMED = "robot app: not confirmed: nothing was connected\n"
NOT_CONNECTED = "robot app: the real driver was not connected: "
BUILT = (
    "robot app: APAI arm profile rebot_rs, the real driver on can4: built, not "
    "connected. No hold marker; calibration bench-7; speed cap 0.1 rad/s; at "
    "rest at start within 0.05 rad"
)
# Threads of the program; none may outlive it.
THREADS = ("robot-app-", "robot-dialer", "gripper-worker", "thor-terminal-")
# What the stand-in backend is asked at each look into APAI's hold folder.
LOOKS = ["hold", "folder", "pending"]
# Stands in for the route of APAI's rest tool (rest.functional_route()).
ROUTE = "the functional rest's route"


@pytest.fixture(autouse=True)
def console_logging_undone() -> Iterator[None]:
    """Keep a program's logging setup inside its test."""
    logger = logging.getLogger(LOGGER)
    level, handlers = logger.level, list(logger.handlers)
    yield
    logger.setLevel(level)
    for handler in list(logger.handlers):
        if handler not in handlers:
            logger.removeHandler(handler)
            handler.close()


@pytest.fixture(autouse=True)
def a_persons_environment(monkeypatch, tmp_path) -> None:
    """No mark of an AI agent in the environment, and APAI's files in tmp_path."""
    for name in list(os.environ):
        if name in confirm.AGENT_ENV or name.startswith(confirm.AGENT_ENV_PREFIXES):
            monkeypatch.delenv(name)
    monkeypatch.setenv("APAI_ARM_HOLD_DIR", str(tmp_path / "arm_holds"))
    monkeypatch.setenv("APAI_REST_LOG", str(tmp_path / "rest_readings.jsonl"))


def left_threads() -> list[str]:
    return [
        thread.name
        for thread in threading.enumerate()
        if thread.name.startswith(THREADS)
    ]


# Without APAI: a stand-in for the backend module


def fake_backend(
    monkeypatch,
    *,
    kind: str = "rebot_rs",
    hold: dict[str, Any] | None = None,
    pending: tuple[str, ...] = (),
    unfit: str = "",
    unlistable: str = "",
) -> list[str]:
    """Stand in for the APAI backend module; what was asked of it, in order.

    hold is the hold marker of the arm's motor bus, pending the names of the
    confirmation files in APAI's hold folder, unfit why the file does not fit
    the profile, unlistable why APAI's hold folder cannot be listed.
    """
    asked: list[str] = []
    module = types.ModuleType(BACKEND)
    profile = types.SimpleNamespace(
        arm_cfg={"type": kind}, apai_cfg=types.SimpleNamespace(safety={})
    )

    def load_apai_profile(name: str) -> Any:
        if name == "no_arm":
            raise ValueError("there is no arm profile no_arm")
        asked.append(f"load {name}")
        return profile

    def config_for_profile(config: Any, of: Any) -> Any:
        asked.append("config")
        return config

    def build_apai_checkers(config: Any, of: Any, clock: Any) -> str:
        if unfit:
            raise ValueError(unfit)
        asked.append("checkers")
        return "the checkers"

    def build_apai_arm(config: Any, of: Any, clock: Any) -> str:
        asked.append("arm")
        return "the arm"

    def unresolved_hold(of: Any) -> dict[str, Any] | None:
        asked.append("hold")
        return hold

    def hold_folder_unlistable() -> str | None:
        asked.append("folder")
        return unlistable or None

    def confirmation_on_file() -> str | None:
        asked.append("pending")
        if unlistable:
            return "/holds"  # the folder itself, as the backend gives it
        return f"/holds/{pending[0]}" if pending else None

    module.load_apai_profile = load_apai_profile
    module.config_for_profile = config_for_profile
    module.build_apai_checkers = build_apai_checkers
    module.build_apai_arm = build_apai_arm
    module.unresolved_hold = unresolved_hold
    module.hold_bus = lambda of: "can4"
    module.hold_folder_unlistable = hold_folder_unlistable
    module.confirmation_on_file = confirmation_on_file
    module.speed_cap = lambda of: 0.1
    monkeypatch.setitem(sys.modules, BACKEND, module)
    # The route of APAI's rest tool, which the program reads from APAI itself.
    monkeypatch.setattr(robot_main, "functional_route", lambda home_q=None: ROUTE)
    return asked


@pytest.fixture
def started(monkeypatch) -> list[dict[str, Any]]:
    """The robot apps the program went on to run; none is run here."""
    calls: list[dict[str, Any]] = []

    def run_connected(config, arm, checkers, clock, endpoint, **options) -> int:
        calls.append({"arm": arm, "checkers": checkers, "endpoint": endpoint})
        calls[-1].update(options)
        return 0

    monkeypatch.setattr(robot_main, "run_connected", run_connected)
    monkeypatch.setattr(robot_main, "_remote_link_missing", lambda: None)
    on_a_terminal(monkeypatch)
    return calls


def on_a_terminal(monkeypatch) -> None:
    """The program takes its standard input and output for a terminal."""
    monkeypatch.setattr(robot_main, "_stdin_is_terminal", lambda: True)
    monkeypatch.setattr(robot_main, "_stdout_is_terminal", lambda: True)


@pytest.fixture
def bench_yaml(tmp_path) -> str:
    """The shipped file, naming the calibration bench-7."""
    return quick_config(tmp_path, {"camera_calibration_id": "bench-7"})


def answer(monkeypatch, *typed: str) -> Iterator[str]:
    """What the operator types at the Thor terminal, then the input ends."""
    lines = iter(typed)
    monkeypatch.setattr(robot_main, "typed_lines", lambda: lines)
    return lines


def real_argv(config: str, *, calibration: str = "bench-7") -> list[str]:
    return [
        *("--config", config, "--arm", "apai", "--driver", "real"),
        *("--apai-profile", "rebot_rs", "--calibration", calibration),
        *("--connect", ENDPOINT),
    ]


def real_driver(capsys, argv: list[str]) -> tuple[int, list[str], str]:
    code = robot_main.main(argv)
    captured = capsys.readouterr()
    return code, captured.out.splitlines(), captured.err


def test_driver_real_without_connect_is_refused_before_any_question(
    capsys, monkeypatch, bench_yaml
):
    monkeypatch.setitem(sys.modules, BACKEND, None)  # must not be imported
    monkeypatch.setattr(robot_main, "typed_lines", None)  # nothing may be asked
    monkeypatch.setattr(robot_main, "RobotApp", None)  # no app may be built
    argv = real_argv(bench_yaml)[:-2]
    code, out, err = real_driver(capsys, argv)
    assert (code, out) == (2, [])
    assert err == (
        "robot app: --driver real needs the robot link service's address: "
        "--connect HOST:PORT\n"
    )


@pytest.mark.parametrize(
    "marked", ["CLAUDECODE", "CODEX_SANDBOX", "AI_AGENT", "OPENCLAW_HOME"]
)
def test_an_agent_environment_is_refused(
    capsys, monkeypatch, bench_yaml, started, marked
):
    asked = fake_backend(monkeypatch, hold={"key": "can4"}, unfit="too fast")
    monkeypatch.setattr(robot_main, "typed_lines", None)
    monkeypatch.setenv(marked, "1")
    code, out, err = real_driver(capsys, real_argv(bench_yaml, calibration="other"))
    assert (code, out) == (2, [])
    assert err == (
        "robot app: not confirmed: nothing was connected: the real driver is "
        f"started by a person, outside an agent environment ({marked} is set)\n"
    )
    # Not the hold, not the confirmations on file, and nothing was built.
    assert asked == ["load rebot_rs"]
    assert started == []


def test_standard_input_that_is_no_terminal_is_refused(
    capsys, monkeypatch, bench_yaml, started
):
    asked = fake_backend(monkeypatch)
    monkeypatch.setattr(robot_main, "typed_lines", None)
    monkeypatch.setattr(robot_main, "_stdin_is_terminal", lambda: False)
    code, out, err = real_driver(capsys, real_argv(bench_yaml))
    assert (code, out) == (2, [])
    assert err == (
        "robot app: not confirmed: nothing was connected: the confirmations are "
        "typed by a person: standard input is no terminal\n"
    )
    assert asked == ["load rebot_rs"] and started == []


def test_standard_output_that_is_no_terminal_is_refused(
    capsys, monkeypatch, bench_yaml, started
):
    asked = fake_backend(monkeypatch)
    monkeypatch.setattr(robot_main, "typed_lines", None)
    monkeypatch.setattr(robot_main, "_stdout_is_terminal", lambda: False)
    code, out, err = real_driver(capsys, real_argv(bench_yaml))
    assert (code, out) == (2, [])
    assert err == (
        "robot app: not confirmed: nothing was connected: the questions are "
        "shown to a person: standard output is no terminal\n"
    )
    assert asked == ["load rebot_rs"] and started == []


def test_the_test_runners_standard_input_and_output_are_no_terminals(monkeypatch):
    class Closed:
        def isatty(self) -> bool:
            raise ValueError("I/O operation on closed file")

    assert robot_main._stdin_is_terminal() is False  # pytest's own input
    assert robot_main._stdout_is_terminal() is False  # and its captured output
    for stream in (None, Closed()):
        monkeypatch.setattr(sys, "stdin", stream)
        monkeypatch.setattr(sys, "stdout", stream)
        assert robot_main._stdin_is_terminal() is False
        assert robot_main._stdout_is_terminal() is False


@pytest.mark.parametrize(("typed_at", "shown_at"), [(True, False), (False, True)])
def test_each_terminal_check_looks_at_its_own_stream(monkeypatch, typed_at, shown_at):
    class Stream:
        def __init__(self, terminal: bool) -> None:
            self.terminal = terminal

        def isatty(self) -> bool:
            return self.terminal

        def write(self, text: str) -> int:
            return len(text)

        def flush(self) -> None:
            pass

    monkeypatch.setattr(sys, "stdin", Stream(typed_at))
    monkeypatch.setattr(sys, "stdout", Stream(shown_at))
    assert robot_main._stdin_is_terminal() is typed_at
    assert robot_main._stdout_is_terminal() is shown_at


def test_a_terminal_on_the_standard_input_alone_is_not_enough(
    capsys, monkeypatch, bench_yaml
):
    master, slave = pty.openpty()
    try:
        asked = fake_backend(monkeypatch)
        monkeypatch.setattr(robot_main, "typed_lines", None)  # nothing may be asked
        monkeypatch.setattr(robot_main, "run_connected", None)  # and nothing run
        monkeypatch.setattr(robot_main, "_remote_link_missing", lambda: None)
        monkeypatch.setattr(sys, "stdin", os.fdopen(slave, closefd=False))
        # The program's own two checks: a terminal to type at, a captured output.
        assert robot_main._stdin_is_terminal() is True
        assert robot_main._stdout_is_terminal() is False
        code, out, err = real_driver(capsys, real_argv(bench_yaml))
    finally:
        os.close(master)
        os.close(slave)
    assert (code, out) == (2, [])
    assert err == (
        "robot app: not confirmed: nothing was connected: the questions are "
        "shown to a person: standard output is no terminal\n"
    )
    assert asked == ["load rebot_rs"]


@pytest.mark.parametrize(
    ("in_file", "named", "shown"),
    [
        ("bench-7", "bench-8", "'bench-8' but camera_calibration_id of the file is"),
        ("bench-7", "Bench-7", "'Bench-7' but camera_calibration_id of the file is"),
        ("bench-7", "bench-7 ", "'bench-7 ' but camera_calibration_id of the file"),
        ("", "bench-7", "'bench-7' but camera_calibration_id of the file is ''"),
    ],
)
def test_a_calibration_name_that_is_not_the_files_is_refused(
    capsys, monkeypatch, tmp_path, started, in_file, named, shown
):
    asked = fake_backend(monkeypatch, hold={"key": "can4"}, unfit="too fast")
    monkeypatch.setattr(robot_main, "typed_lines", None)
    config = quick_config(tmp_path, {"camera_calibration_id": in_file})
    code, out, err = real_driver(capsys, real_argv(config, calibration=named))
    assert (code, out) == (2, [])
    assert err.startswith(f"{NOT_CONNECTED}--calibration names {shown}")
    assert err.endswith("the two must be the same name, and not empty\n")
    assert f"{in_file!r}" in err
    assert asked == ["load rebot_rs"] and started == []


def test_an_unresolved_hold_is_refused_and_names_the_bus(
    capsys, monkeypatch, bench_yaml, started
):
    marker = {
        "key": "can4",
        "reason": "left raised by a run",
        "at_text": "2026-11-14 16:02:11",
    }
    asked = fake_backend(monkeypatch, hold=marker, unfit="too fast")
    monkeypatch.setattr(robot_main, "typed_lines", None)
    code, out, err = real_driver(capsys, real_argv(bench_yaml))
    assert (code, out) == (2, [])
    assert err == (
        f"{NOT_CONNECTED}the arm on can4 has an unresolved hold: APAI's hold "
        "marker of 2026-11-14 16:02:11 says 'left raised by a run'. Connecting "
        "would cut the torque of an arm that may be raised. Do APAI's operator "
        "recovery first; a typed yes never releases a hold\n"
    )
    assert asked == ["load rebot_rs", "hold"] and started == []
    # A marker that cannot be read names no time: it is a hold all the same.
    fake_backend(monkeypatch, hold={"key": "can4", "reason": "x", "unreadable": True})
    code, out, err = real_driver(capsys, real_argv(bench_yaml))
    assert (code, out) == (2, [])
    assert "APAI's hold marker of an unknown time says 'x'" in err


def on_file(name: str) -> str:
    """The refusal that names a confirmation file in APAI's hold folder."""
    return (
        f"{NOT_CONNECTED}an operator's confirmation for APAI's connect is on "
        f"file ({name}): with one on file APAI's driver would connect an arm it "
        "does not measure at rest. Revoke it with APAI's own tool (arm_hold.py "
        "revoke), then start again\n"
    )


@pytest.mark.parametrize("name", ["_any.confirm.json", "can4.confirm.json"])
def test_an_operators_confirmation_on_file_is_refused(
    capsys, monkeypatch, bench_yaml, started, name
):
    asked = fake_backend(monkeypatch, pending=(name,), unfit="too fast")
    monkeypatch.setattr(robot_main, "typed_lines", None)
    code, out, err = real_driver(capsys, real_argv(bench_yaml))
    assert (code, out) == (2, [])
    assert err == on_file(f"/holds/{name}")
    assert asked == ["load rebot_rs", *LOOKS] and started == []


def test_a_confirmation_for_another_bus_refuses_this_arm_too(
    capsys, monkeypatch, bench_yaml, started
):
    fake_backend(monkeypatch, pending=("can0.confirm.json",))
    monkeypatch.setattr(robot_main, "typed_lines", None)
    code, out, err = real_driver(capsys, real_argv(bench_yaml))
    assert (code, out) == (2, [])
    assert err == on_file("/holds/can0.confirm.json")
    assert started == []


def not_listed(problem: str) -> str:
    """The refusal of a hold folder of APAI's that cannot be listed."""
    return (
        f"{NOT_CONNECTED}APAI's hold folder cannot be listed ({problem}): an "
        "operator's confirmation may lie in it, and APAI's driver would find it "
        "by its name. Let the user who starts this program read and enter the "
        "folder (its permissions), then start again\n"
    )


def test_a_hold_folder_that_cannot_be_listed_is_refused_and_named_as_that(
    capsys, monkeypatch, bench_yaml, started
):
    problem = "[Errno 13] Permission denied: '/holds'"
    asked = fake_backend(monkeypatch, unlistable=problem, unfit="too fast")
    monkeypatch.setattr(robot_main, "typed_lines", None)
    code, out, err = real_driver(capsys, real_argv(bench_yaml))
    assert (code, out) == (2, [])
    assert err == not_listed(problem)
    # No file is named and no revoke is advised: APAI's tool cannot list it either.
    assert "revoke" not in err and "is on file" not in err
    assert asked == ["load rebot_rs", "hold", "folder"] and started == []


def test_a_file_that_does_not_fit_the_profile_is_refused_before_any_question(
    capsys, monkeypatch, bench_yaml, started
):
    asked = fake_backend(monkeypatch, unfit="the limit set is above the speed cap")
    monkeypatch.setattr(robot_main, "typed_lines", None)
    code, out, err = real_driver(capsys, real_argv(bench_yaml))
    assert (code, out) == (2, [])
    assert err == (
        f"{NOT_CONNECTED}APAI arm profile 'rebot_rs' cannot be used with "
        f"{bench_yaml}: the limit set is above the speed cap\n"
    )
    assert asked == ["load rebot_rs", *LOOKS, "config"]
    assert started == []


def test_without_isaac_capture_nothing_is_asked(
    capsys, monkeypatch, bench_yaml, started
):
    fake_backend(monkeypatch)
    monkeypatch.setattr(robot_main, "typed_lines", None)
    monkeypatch.setattr(
        robot_main, "_remote_link_missing", lambda: "no remote link here"
    )
    code, out, err = real_driver(capsys, real_argv(bench_yaml))
    assert (code, out) == (2, [])
    assert err == f"{NOT_CONNECTED}--connect cannot run: no remote link here\n"
    assert started == []


def test_three_yes_hand_the_built_arm_and_the_same_input_to_the_robot_app(
    capsys, monkeypatch, bench_yaml, started
):
    asked = fake_backend(monkeypatch)
    lines = answer(monkeypatch, "yes", " Yes ", "YES", "stop", "quit")
    code, out, err = real_driver(capsys, real_argv(bench_yaml))
    assert (code, err) == (0, "")
    assert out == [BUILT, *QUESTIONS, robot_main.CONNECTING]
    # Every check came before anything was built, and before the first question;
    # the hold and the confirmations on file are looked at again after the last.
    built = ["config", "checkers", "arm"]
    assert asked == ["load rebot_rs", *LOOKS, *built, *LOOKS]
    (call,) = started
    assert (call["arm"], call["checkers"]) == ("the arm", "the checkers")
    assert (call["endpoint"], call["driver"]) == (ENDPOINT, "real")
    # The Thor terminal reads on where the answers ended.
    assert call["typed"] is lines
    assert list(lines) == ["stop", "quit"]
    # The robot app gets the functional rest's route, and a way to build the
    # arm port anew for the motors to be switched on again after a rest.
    assert call["rest_route"] is ROUTE
    assert call["rebuild_arm"]() == "the arm" and asked[-1] == "arm"


@pytest.mark.parametrize("late", ["hold", "confirmation"])
def test_a_hold_or_a_confirmation_written_during_the_questions_is_refused(
    capsys, monkeypatch, bench_yaml, started, late
):
    asked = fake_backend(monkeypatch)
    backend = sys.modules[BACKEND]
    answer(monkeypatch, "yes", "yes", "yes")

    def after_the_questions(found: Any) -> Any:
        return found if "arm" in asked else None

    marker = {"key": "can4", "reason": "left raised", "at_text": "just now"}
    waiting = "/holds/can4.confirm.json"
    if late == "hold":
        backend.unresolved_hold = lambda of: after_the_questions(marker)
    else:
        backend.confirmation_on_file = lambda: after_the_questions(waiting)
    code, out, err = real_driver(capsys, real_argv(bench_yaml))
    assert code == 2
    assert out == [BUILT, *QUESTIONS]
    said = "has an unresolved hold" if late == "hold" else "confirmation for APAI's"
    assert err.startswith(NOT_CONNECTED) and said in err
    assert started == []


@pytest.mark.parametrize(
    "wrong", ["no", "", "y", "yes please", "yes yes", "ok", "path is clear", "stop"]
)
@pytest.mark.parametrize("at", [0, 1, 2])
def test_each_question_needs_a_typed_yes(
    capsys, monkeypatch, bench_yaml, started, at, wrong
):
    fake_backend(monkeypatch)
    typed = ["yes", "yes", "yes", "yes"]
    typed[at] = wrong
    answer(monkeypatch, *typed)
    code, out, err = real_driver(capsys, real_argv(bench_yaml))
    assert code == 2
    # The questions after the one that was not confirmed are not asked.
    assert out == [BUILT, *QUESTIONS[: at + 1]]
    assert err == NOT_CONFIRMED
    assert started == []


@pytest.mark.parametrize("answers", [0, 1, 2])
def test_an_input_that_ends_confirms_nothing(
    capsys, monkeypatch, bench_yaml, started, answers
):
    fake_backend(monkeypatch)
    answer(monkeypatch, *["yes"] * answers)
    code, out, err = real_driver(capsys, real_argv(bench_yaml))
    assert code == 2
    assert out[1:] == QUESTIONS[: answers + 1]
    assert err == NOT_CONFIRMED and started == []


@pytest.mark.parametrize("gone", [OSError("the terminal is gone"), KeyboardInterrupt()])
def test_a_terminal_that_is_gone_and_ctrl_c_confirm_nothing(
    capsys, monkeypatch, bench_yaml, started, gone
):
    fake_backend(monkeypatch)

    def lines() -> Iterator[str]:
        yield "yes"
        raise gone

    monkeypatch.setattr(robot_main, "typed_lines", lines)
    code, out, err = real_driver(capsys, real_argv(bench_yaml))
    assert code == 2
    assert out[1:] == QUESTIONS[:2]
    assert err == NOT_CONFIRMED and started == []


def test_the_calibration_is_shown_as_one_printable_word():
    said: list[str] = []
    name = "bench\x07-7\n" + "x" * 100
    assert robot_main.confirm_real_driver(name, iter(["yes"] * 3), said.append)
    assert len(said) == 3
    shown = said[2].removeprefix("Real driver, 3 of 3: Is the calibration in use ")
    assert shown.startswith("'bench -7 xxx")
    assert shown.isprintable() and len(shown) < 100


def test_the_real_driver_is_not_asked_for_with_apais_software_arm(
    capsys, monkeypatch, bench_yaml, started
):
    fake_backend(monkeypatch, kind="mock")
    monkeypatch.setattr(robot_main, "typed_lines", None)  # nothing may be asked
    code, out, err = real_driver(capsys, real_argv(bench_yaml))
    assert (code, out) == (2, [])
    assert err == (
        "robot app: APAI arm profile 'rebot_rs' is APAI's software arm: use "
        "--driver mock\n"
    )
    assert started == []


def test_a_real_driver_profile_that_cannot_be_loaded_asks_nothing(
    capsys, monkeypatch, bench_yaml, started
):
    fake_backend(monkeypatch)
    monkeypatch.setattr(robot_main, "typed_lines", None)
    argv = real_argv(bench_yaml)
    argv[argv.index("rebot_rs")] = "no_arm"
    code, out, err = real_driver(capsys, argv)
    assert (code, out) == (2, [])
    assert err == (
        "robot app: APAI arm profile 'no_arm' cannot be used: there is no arm "
        "profile no_arm\n"
    )
    assert started == []


def test_the_checks_refuse_in_turn(monkeypatch, bench_yaml):
    config = load_config(bench_yaml)
    marker = {"key": "can4", "reason": "held", "at_text": "then"}
    waiting = ("_any.confirm.json",)
    asked = fake_backend(monkeypatch, hold=marker, pending=waiting)
    backend = sys.modules[BACKEND]
    profile = backend.load_apai_profile("rebot_rs")

    def refusal(calibration: str, environ: dict[str, str]) -> str:
        return robot_main.real_driver_refusal(
            config, profile, calibration, backend, environ
        )

    # Every check fails: each is reported once the ones before it pass.
    assert "outside an agent environment" in refusal("other", {"CLAUDECODE": "1"})
    assert "--calibration names 'other'" in refusal("other", {"CLAUDECODE": " "})
    assert "has an unresolved hold" in refusal("bench-7", {})
    marker_gone = fake_backend(monkeypatch, pending=waiting)
    backend = sys.modules[BACKEND]
    assert "an operator's confirmation" in refusal("bench-7", {})
    fake_backend(monkeypatch)
    backend = sys.modules[BACKEND]
    assert refusal("bench-7", {}) is None
    assert asked == ["load rebot_rs", "hold"]
    assert marker_gone == LOOKS


# With APAI: its real driver class and its connect(), over a stand-in motor SDK

rig = pytest.mark.apai
CUTS = ("mode_mit", "clear_error", "enable", "disconnect", "send_mit")
RAISED = [0.0, 0.6, 0.5, 0.0, 0.0, 0.0]


class RigMotor(Motor):
    """A motor that notes each call made to it."""

    def __init__(self, name: str, calls: list[str]) -> None:
        super().__init__()
        self.name, self.calls = name, calls

    def robstride_get_param_f32(
        self, param: int, timeout_ms: int | None = None
    ) -> float:
        self.calls.append(f"read {self.name}")
        return super().robstride_get_param_f32(param, timeout_ms)

    def clear_error(self) -> None:
        self.calls.append(f"clear_error {self.name}")


class _Noted:
    """What the driver's connect asks of a motor group, noted."""

    def note(self, kind: str, calls: list[str]) -> None:
        self.kind, self.calls = kind, calls
        self._mm = {name: RigMotor(name, calls) for name in self.joint_names}
        self._mit_kp = np.full(len(self.joint_names), 30.0)
        self._mit_kd = np.ones(len(self.joint_names))

    def mode_mit(self) -> None:
        self.calls.append(f"mode_mit {self.kind}")

    def enable(self) -> None:
        self.calls.append("enable")

    def send_mit(self, q: Any, kp: Any = None, kd: Any = None, tau: Any = None) -> None:
        self.calls.append(f"send_mit {self.kind}")
        super().send_mit(q, kp, kd, tau)


class RigArmGroup(_Noted, MotorGroup):
    pass


class RigJawGroup(_Noted, JawGroup):
    pass


class _Bus:
    def __init__(self, calls: list[str]) -> None:
        self.calls = calls

    def close_bus(self) -> None:
        self.calls.append("close_bus")

    def close(self) -> None:
        self.calls.append("close")


class RigSdk:
    """Stands in for reBotArm_control_py's RebotArm: no bus, every call noted."""

    has_gripper = True

    def __init__(self, bench: Bench, hw_yaml: str) -> None:
        self.calls = bench.calls
        self.hw_yaml = hw_yaml
        self.arm = RigArmGroup([f"joint{i}" for i in range(1, 7)])
        self.arm.note("arm", self.calls)
        self.gripper = RigJawGroup(["jaw"])
        self.gripper.note("gripper", self.calls)
        self._motor_map = {**self.arm._mm, **self.gripper._mm}
        self._ctrl_map = {"robstride": _Bus(self.calls)}
        self.disconnects = 0
        self.released_at: list[list[float]] = []
        for motor, value in zip(self.arm._mm.values(), bench.reading):
            motor.value = value
        self.gripper._mm["jaw"].value = 0.5  # the jaws start a little open

    def connect(self) -> None:
        self.calls.append("open the bus")

    def disconnect(self) -> None:
        self.calls.append("disconnect")
        self.disconnects += 1
        self.released_at.append([motor.value for motor in self.arm._mm.values()])


class Bench:
    """The rig on the bench: where its arm lies, and every SDK call made."""

    def __init__(self, folder: Path) -> None:
        self.calls: list[str] = []
        self.reading = [0.0] * 6
        self.sdks: list[RigSdk] = []
        self.holds = folder / "arm_holds"
        self.rest_log = folder / "rest_readings.jsonl"
        # The arm profile lies as the rig's does on the Thor: in the configs
        # folder of the experiment, whose launcher runs it. The experiment is
        # the on-device agent this example was split from.
        self.profile = str(folder / "experiment" / "configs" / "thor_arm")
        # Changes a new SDK object before the driver uses it.
        self.prepare: Any = None

    def make(self, hw_yaml: str) -> RigSdk:
        sdk = RigSdk(self, hw_yaml)
        if self.prepare is not None:
            self.prepare(sdk)
        self.sdks.append(sdk)
        return sdk

    def cuts(self) -> list[str]:
        """The calls that cut torque, write a mode, enable or command a motor."""
        return [call for call in self.calls if call.startswith(CUTS)]

    def rest_rows(self) -> list[dict[str, Any]]:
        """What APAI logged of its rest checks."""
        if not self.rest_log.exists():
            return []
        text = self.rest_log.read_text(encoding="utf-8")
        return [json.loads(line) for line in text.splitlines()]

    def argv(self, *extra: str) -> list[str]:
        named = load_config(RIG_YAML).camera_calibration_id
        return [
            *("--config", str(RIG_YAML), "--arm", "apai", "--driver", "real"),
            *("--apai-profile", self.profile, "--calibration", named),
            *extra,
        ]


@pytest.fixture
def bench(monkeypatch, tmp_path) -> Bench:
    """APAI's reBot RS driver on a stand-in SDK, with a profile like the rig's.

    The profile extends APAI's rebot_rs as the rig's does: motor bus can4, a
    fallback speed cap of 0.1 rad/s. The robot app loads it as on the Thor:
    the experiment's launcher runs it, at the joint speed of its
    rebot_workstation file, which is the profile's speed cap. The gripper
    guard's clock does not wait.
    """
    import apai.control.rs_gripper as rs_gripper

    bench = Bench(tmp_path)
    hw_yaml = tmp_path / "sdk-can4.yaml"
    hw_yaml.write_text("channel: can4\n", encoding="utf-8")
    profile = {
        "extends": "rebot_rs",
        "name": "rebot_thor",
        "hw_yaml": str(hw_yaml),
        "channel": "can4",
        "overrides": {"safety": {"max_joint_vel": 0.1}},
    }
    configs = Path(bench.profile).parent
    configs.mkdir(parents=True)
    files = {
        "thor_arm.yaml": profile,
        "thor_runtime.yaml": {"arm_profile": "configs/thor_arm.yaml"},
        "rebot_workstation.yaml": {
            "extends": "rebot_rs",
            "overrides": {"safety": {"max_joint_vel": LAUNCHER_SPEED}},
        },
    }
    for name, content in files.items():
        (configs / name).write_text(yaml.safe_dump(content), encoding="utf-8")
    actuator = types.ModuleType("reBotArm_control_py.actuator")
    actuator.RebotArm = bench.make
    package = types.ModuleType("reBotArm_control_py")
    package.actuator = actuator
    monkeypatch.setitem(sys.modules, "reBotArm_control_py", package)
    monkeypatch.setitem(sys.modules, "reBotArm_control_py.actuator", actuator)
    monkeypatch.setattr(sys, "path", list(sys.path))  # the driver adds its SDK path
    monkeypatch.setattr(rs_gripper, "time", GuardTime())
    on_a_terminal(monkeypatch)
    return bench


def refused_before_the_bus(bench: Bench, capsys, monkeypatch) -> str:
    """Run the program to an exit code 2; what it wrote to the standard error."""
    # A program that asked its questions all the same gets three yes, goes on
    # to APAI's connect and does not end with exit code 2.
    monkeypatch.setattr(robot_main, "typed_lines", lambda: iter(["yes"] * 3))
    no_dial(monkeypatch)
    closed_once(monkeypatch, lambda app: True)  # a program that ran on ends at once
    assert robot_main.main(bench.argv("--connect", ENDPOINT)) == 2
    captured = capsys.readouterr()
    assert captured.out == ""  # nothing was asked
    # Nothing was opened, read or sent, and APAI measured nothing.
    assert bench.calls == [] and bench.sdks == []
    assert bench.rest_rows() == []
    return captured.err


@rig
def test_an_agent_environment_is_refused_before_the_real_driver_is_built(
    bench, capsys, monkeypatch
):
    monkeypatch.setenv("CODEX_THREAD_ID", "t-1")
    err = refused_before_the_bus(bench, capsys, monkeypatch)
    assert err.startswith("robot app: not confirmed: nothing was connected: ")
    assert "(CODEX_THREAD_ID is set)" in err


@rig
def test_a_calibration_name_that_is_not_the_rig_files_is_refused(
    bench, capsys, monkeypatch
):
    argv = bench.argv("--connect", ENDPOINT)
    argv[argv.index("--calibration") + 1] = "stereo-000000000000+mount-000000000000"
    monkeypatch.setattr(bench, "argv", lambda *extra: argv)
    err = refused_before_the_bus(bench, capsys, monkeypatch)
    assert err.startswith(
        f"{NOT_CONNECTED}--calibration names "
        "'stereo-000000000000+mount-000000000000' but camera_calibration_id of "
        "the file is 'stereo-c50bf5b2b899+mount-7f474001bc0f'"
    )


@rig
def test_an_unresolved_hold_of_the_rigs_bus_is_refused_and_stays(
    bench, capsys, monkeypatch
):
    from apai.control import arm_hold

    bench.reading = list(RAISED)
    written = arm_hold.record("can4", arm="rebot_rs", reason="left raised by a run")
    err = refused_before_the_bus(bench, capsys, monkeypatch)
    assert err.startswith(
        f"{NOT_CONNECTED}the arm on can4 has an unresolved hold: APAI's hold "
        f"marker of {written['at_text']} says 'left raised by a run'."
    )
    # The marker is read, never cleared; a marker of another bus does not count.
    assert json.loads(arm_hold.marker_path("can4").read_text()) == written
    arm_hold.marker_path("can4").rename(arm_hold.marker_path("can0"))
    monkeypatch.setattr(robot_main, "typed_lines", lambda: iter(["no"]))
    assert robot_main.main(bench.argv("--connect", ENDPOINT)) == 2
    assert capsys.readouterr().err == NOT_CONFIRMED
    assert bench.calls == []


@rig
@pytest.mark.parametrize("key", [None, "can4", "can4.", " can4", "can4/", "can0"])
def test_an_operators_confirmation_is_never_used_by_the_robot_app(
    bench, capsys, monkeypatch, key
):
    from apai.control import arm_hold

    bench.reading = list(RAISED)
    # What `arm_hold.py confirm [--arm KEY]` writes after the operator's phrase.
    confirmation = arm_hold.confirm("holding the arm", key)
    err = refused_before_the_bus(bench, capsys, monkeypatch)
    assert err == on_file(str(confirmation))
    # It was not claimed: it is still on file for APAI's own tools.
    assert confirmation.is_file()
    assert sorted(path.name for path in bench.holds.iterdir()) == [confirmation.name]


LAPSED = '{"key": "can4", "at": 1.0}'  # written in 1970: APAI takes it for stale


@rig
@pytest.mark.parametrize(
    ("content", "mode"),
    [("not json", 0), ("not json", 0o600), (LAPSED, 0), (LAPSED, 0o600)],
    ids=["unreadable", "no-json", "lapsed-unreadable", "lapsed"],
)
def test_an_unreadable_or_lapsed_confirmation_file_is_refused_too(
    bench, capsys, monkeypatch, content, mode
):
    from apai.control import arm_hold

    bench.reading = list(RAISED)
    bench.holds.mkdir()
    confirmation = bench.holds / "can4.confirm.json"
    confirmation.write_text(content)
    assert arm_hold.pending_confirmations() == []  # APAI lists none of them
    confirmation.chmod(mode)
    err = refused_before_the_bus(bench, capsys, monkeypatch)
    assert err == on_file(str(confirmation))
    assert confirmation.is_file()
    if mode:
        assert confirmation.read_text() == content  # left as it was


@rig
def test_a_confirmation_written_during_the_questions_never_reaches_the_connect(
    bench, capsys, monkeypatch
):
    from apai.control import arm_hold

    bench.reading = list(RAISED)
    written: list[Path] = []

    def answers() -> Iterator[str]:
        yield "yes"
        yield "yes"
        written.append(arm_hold.confirm("holding the arm", "can4."))
        yield "yes"

    monkeypatch.setattr(robot_main, "typed_lines", answers)
    no_dial(monkeypatch)
    closed_once(monkeypatch, lambda app: True)  # a program that ran on ends at once
    assert robot_main.main(bench.argv("--connect", ENDPOINT)) == 2
    captured = capsys.readouterr()
    assert captured.err == on_file(str(written[0]))
    assert robot_main.CONNECTING not in captured.out
    # The raised arm's bus was never opened, and the file was not claimed.
    assert bench.calls == [] and bench.sdks == []
    assert written[0].is_file()


def three_yes(
    bench: Bench, capsys, monkeypatch, before_the_connect: Any = None
) -> tuple[int, Any]:
    """Answer the three questions; the program's exit code and its output.

    before_the_connect runs after the program's last look into APAI's hold
    folder and before APAI's driver is asked to connect.
    """
    monkeypatch.setattr(robot_main, "typed_lines", lambda: iter(["yes"] * 3))
    no_dial(monkeypatch)
    closed_once(monkeypatch, lambda app: True)  # a program that ran on ends at once
    if before_the_connect is not None:
        build = robot_main.RobotApp

        def robot_app(*args: Any, **kwargs: Any) -> Any:
            before_the_connect()
            return build(*args, **kwargs)

        monkeypatch.setattr(robot_main, "RobotApp", robot_app)
    code = robot_main.main(bench.argv("--connect", ENDPOINT))
    return code, capsys.readouterr()


def refused_with_no_torque_cut(bench: Bench, code: int, captured: Any) -> None:
    """APAI's driver refused the connect: nothing cut torque, nothing ran on."""
    assert captured.out.splitlines()[-1] == robot_main.CONNECTING
    assert captured.err.splitlines()[-1] == f"robot app: {robot_main.NOT_AT_REST}"
    assert code == 1
    assert bench.cuts() == []
    assert [sdk.disconnects for sdk in bench.sdks] == [0] * len(bench.sdks)
    assert left_threads() == []


@rig
@pytest.mark.parametrize("key", [None, "can4"])
def test_a_confirmation_written_while_the_bus_opens_is_not_claimed(
    bench, capsys, monkeypatch, key
):
    from apai.control import arm_hold

    bench.reading = list(RAISED)
    written: list[Path] = []
    # The driver builds its SDK object after the program's last look.
    bench.prepare = lambda sdk: written.append(arm_hold.confirm("holding the arm", key))
    code, captured = three_yes(bench, capsys, monkeypatch)
    refused_with_no_torque_cut(bench, code, captured)
    assert "The arm on can4 is not folded at rest (" in captured.err
    reads = [f"read joint{i}" for i in range(1, 7)]
    assert bench.calls == ["open the bus", *reads, *reads, "close_bus", "close"]
    # The confirmation is still on file for APAI's own tools: nothing claimed it.
    (confirmation,) = written
    assert confirmation.is_file()
    assert sorted(path.name for path in bench.holds.iterdir()) == [confirmation.name]
    assert arm_hold.pending_confirmations()[0]["key"] == key


@rig
def test_a_hold_and_a_confirmation_written_after_the_last_look_open_nothing(
    bench, capsys, monkeypatch
):
    from apai.control import arm_hold

    def written_late() -> None:
        arm_hold.record("can4", arm="rebot_rs", reason="left raised by a run")
        arm_hold.confirm("holding the arm", "can4")

    bench.reading = list(RAISED)
    code, captured = three_yes(
        bench, capsys, monkeypatch, before_the_connect=written_late
    )
    refused_with_no_torque_cut(bench, code, captured)
    # APAI's hold gate refused over the marker before the bus was opened.
    assert "left raised by a run" in captured.err
    assert bench.calls == [] and bench.sdks == [] and bench.rest_rows() == []
    # The marker was not cleared and the confirmation was not claimed.
    assert sorted(path.name for path in bench.holds.iterdir()) == [
        "can4.confirm.json",
        "can4.json",
    ]


@rig
def test_a_confirmation_another_connect_gives_back_is_not_claimed(
    bench, capsys, monkeypatch
):
    import importlib

    from apai.control import arm_hold

    confirmation = arm_hold.confirm("holding the arm", "can4")
    held = arm_hold.claim_confirmation("can4")  # as another APAI connect holds it
    assert held is not None and not confirmation.exists()
    # Under its claimed name it is no confirmation on file at the two looks.
    assert importlib.import_module(BACKEND).confirmation_on_file() is None
    bench.reading = list(RAISED)
    bench.prepare = lambda sdk: held.release()  # that connect failed meanwhile
    code, captured = three_yes(bench, capsys, monkeypatch)
    refused_with_no_torque_cut(bench, code, captured)
    assert sorted(path.name for path in bench.holds.iterdir()) == [confirmation.name]


@rig
@pytest.mark.parametrize("reading", [[0.0] * 6, RAISED], ids=["connected", "refused"])
def test_apais_claim_is_switched_off_for_the_connect_only(
    bench, capsys, monkeypatch, reading
):
    from apai.control import arm_hold

    claim = arm_hold.claim_confirmation
    during: list[Any] = []

    def prepare(sdk: RigSdk) -> None:
        arm_hold.confirm("holding the arm")  # for any arm, while the connect runs
        during.append(arm_hold.claim_confirmation)
        during.append(arm_hold.claim_confirmation("can4"))
        during.append(arm_hold.take_confirmation(None))

    bench.reading = list(reading)
    bench.prepare = prepare
    code, _ = three_yes(bench, capsys, monkeypatch)
    assert code == (1 if reading == RAISED else 0)
    switched_off, *claimed = during
    assert switched_off is not claim and claimed == [None, None]
    assert arm_hold.claim_confirmation is claim  # APAI's own again
    assert sorted(path.name for path in bench.holds.iterdir()) == ["_any.confirm.json"]
    assert left_threads() == []


@rig
@pytest.mark.parametrize(
    ("owner", "speed", "refusal"),
    [
        # The agent and the return to rest: no faster than the launcher.
        (
            "agent",
            0.5,
            "limits.agent.max_joint_speed 0.5 rad/s is above the APAI profile's "
            "speed cap of 0.4 rad/s (safety.max_joint_vel)",
        ),
        (
            "recovery",
            0.5,
            "limits.recovery.max_joint_speed 0.5 rad/s is above the APAI profile's "
            "speed cap of 0.4 rad/s (safety.max_joint_vel)",
        ),
        # A person: no faster than the robot app's own speed cap for them,
        # which is above the cap of APAI's rebot_rs profile.
        (
            "human",
            2.1,
            "limits.human.max_joint_speed 2.1 rad/s is above the robot app's "
            "speed cap for a person of 2 rad/s (apai_backend.HUMAN_SPEED_CAP)",
        ),
    ],
)
def test_a_file_faster_than_the_profiles_cap_is_refused_before_any_question(
    bench, capsys, monkeypatch, tmp_path, owner, speed, refusal
):
    assert LAUNCHER_SPEED == 0.4
    data = yaml.safe_load(RIG_YAML.read_text(encoding="utf-8"))
    data["limits"][owner]["max_joint_speed"] = speed
    fast = tmp_path / "robot_app_fast.yaml"
    fast.write_text(yaml.safe_dump(data), encoding="utf-8")
    argv = bench.argv("--connect", ENDPOINT)
    argv[argv.index("--config") + 1] = str(fast)
    monkeypatch.setattr(bench, "argv", lambda *extra: argv)
    err = refused_before_the_bus(bench, capsys, monkeypatch)
    assert err == (
        f"{NOT_CONNECTED}APAI arm profile {bench.profile!r} cannot be used with "
        f"{fast}: {refusal}\n"
    )


@rig
def test_a_rest_pose_that_is_not_the_profiles_park_pose_is_refused(
    bench, capsys, monkeypatch, tmp_path
):
    data = yaml.safe_load(RIG_YAML.read_text(encoding="utf-8"))
    data["rest"]["rest_q"][3] = 0.2
    other = tmp_path / "robot_app_other.yaml"
    other.write_text(yaml.safe_dump(data), encoding="utf-8")
    argv = bench.argv("--connect", ENDPOINT)
    argv[argv.index("--config") + 1] = str(other)
    monkeypatch.setattr(bench, "argv", lambda *extra: argv)
    err = refused_before_the_bus(bench, capsys, monkeypatch)
    assert "differs from the APAI profile's park_q" in err


def without_a_park_pose(bench: Bench, how: str) -> None:
    """Rewrite the bench's arm profile so that it names no park pose."""
    from apai.config import CONFIG_DIR

    path = Path(bench.profile + ".yaml")
    profile = yaml.safe_load(path.read_text(encoding="utf-8"))
    if how == "null":
        profile["park_q"] = None
    else:
        base = CONFIG_DIR / "arms" / "rebot_rs.yaml"
        parent = yaml.safe_load(base.read_text(encoding="utf-8"))
        del parent["park_q"]
        unparked = path.with_name("rebot_rs_without_park.yaml")
        unparked.write_text(yaml.safe_dump(parent), encoding="utf-8")
        profile["extends"] = str(unparked.with_suffix(""))
    path.write_text(yaml.safe_dump(profile), encoding="utf-8")


@rig
@pytest.mark.parametrize("how", ["null", "absent"])
@pytest.mark.parametrize(
    "rest_q",
    [None, [0.0, 0.4, 0.3, 0.0, 0.0, 0.0]],
    ids=["the-rig-file", "another-rest-pose"],
)
def test_a_hardware_profile_that_names_no_park_pose_is_refused(
    bench, capsys, monkeypatch, tmp_path, how, rest_q
):
    without_a_park_pose(bench, how)
    argv = bench.argv("--connect", ENDPOINT)
    used = str(RIG_YAML)
    if rest_q is not None:
        data = yaml.safe_load(RIG_YAML.read_text(encoding="utf-8"))
        data["rest"]["rest_q"] = rest_q
        other = tmp_path / "robot_app_other.yaml"
        other.write_text(yaml.safe_dump(data), encoding="utf-8")
        used = argv[argv.index("--config") + 1] = str(other)
    monkeypatch.setattr(bench, "argv", lambda *extra: argv)
    # APAI's driver would measure such an arm against all zeros.
    err = refused_before_the_bus(bench, capsys, monkeypatch)
    assert err == (
        f"{NOT_CONNECTED}APAI arm profile {bench.profile!r} cannot be used with "
        f"{used}: the APAI profile names no park pose (park_q): a profile that "
        "drives hardware must name it, and rest.rest_q must be that pose\n"
    )
    # Without --connect the program describes nothing either.
    described = ["--config", used, "--arm", "apai", "--apai-profile", bench.profile]
    assert robot_main.main(described) == 2
    captured = capsys.readouterr()
    assert captured.out == "" and "names no park pose (park_q)" in captured.err


@rig
@pytest.mark.parametrize("mode", [0o300, 0o100])
def test_a_hold_folder_that_cannot_be_listed_is_refused_before_the_bus(
    bench, capsys, monkeypatch, mode
):
    from apai.control import arm_hold

    bench.reading = list(RAISED)
    # Fresh and for any arm: APAI's driver could claim it by its name.
    confirmation = arm_hold.confirm("holding the arm")
    bench.holds.chmod(mode)  # the folder can be entered, not listed
    try:
        err = refused_before_the_bus(bench, capsys, monkeypatch)
    finally:
        bench.holds.chmod(0o700)
    assert err == not_listed(f"[Errno 13] Permission denied: '{bench.holds}'")
    assert sorted(path.name for path in bench.holds.iterdir()) == [confirmation.name]


@rig
@pytest.mark.parametrize("at", [0, 1, 2])
def test_a_question_that_is_not_confirmed_leaves_the_real_driver_unconnected(
    bench, capsys, monkeypatch, at
):
    typed = ["yes", "yes", "yes"]
    typed[at] = "no"
    monkeypatch.setattr(robot_main, "typed_lines", lambda: iter(typed))
    monkeypatch.setattr(robot_main, "_remote_link_missing", lambda: None)
    assert robot_main.main(bench.argv("--connect", ENDPOINT)) == 2
    captured = capsys.readouterr()
    out = captured.out.splitlines()
    assert out[0] == (
        f"robot app: APAI arm profile {bench.profile}, the real driver on can4: "
        "built, not connected. No hold marker; calibration "
        "stereo-c50bf5b2b899+mount-7f474001bc0f; speed cap 0.4 rad/s; at rest at "
        "start within 0.12 rad"
    )
    assert len(out) == 2 + at and out[-1].startswith(f"Real driver, {at + 1} of 3: ")
    assert captured.err == NOT_CONFIRMED
    # The driver was built; its SDK was not even made.
    assert bench.calls == [] and bench.sdks == []
    assert left_threads() == []


def unmeasured(sdk: RigSdk, how: str) -> None:
    """Make the arm one that APAI's rest check cannot take for an arm at rest."""
    motor = sdk.arm._mm["joint3"]
    if how == "silent":
        motor.missed = 2
    elif how == "torn":
        motor.wrong = [0.0, 0.3]
    elif how == "implausible":
        motor.value = float("nan")


@rig
@pytest.mark.parametrize(
    ("how", "reading", "why"),
    [
        ("raised", RAISED, "joint 2 is 0.600 rad from park_q"),
        ("raised", [0.0, 0.0, 0.0, 0.0, 0.0, 0.13], "joint 6 is 0.130 rad from park_q"),
        ("silent", [0.0] * 6, r"joint\(s\) \[3\] did not answer"),
        ("torn", [0.0] * 6, "the two reads disagree by more than 0.02 rad"),
        ("implausible", [0.0] * 6, r"joint\(s\) \[3\] gave an implausible reading"),
    ],
)
def test_a_raised_arm_is_refused_by_the_measured_rest_and_no_torque_was_cut(
    bench, capsys, monkeypatch, how, reading, why
):
    import re

    bench.reading = list(reading)
    bench.prepare = lambda sdk: unmeasured(sdk, how)
    monkeypatch.setattr(robot_main, "typed_lines", lambda: iter(["yes"] * 3))
    monkeypatch.setattr(robot_main, "_remote_link_missing", lambda: None)
    dialed: list[str] = []
    monkeypatch.setattr(
        robot_main.RobotDialer, "start", lambda self: dialed.append("dialed")
    )
    assert robot_main.main(bench.argv("--connect", ENDPOINT)) == 1
    assert dialed == []
    captured = capsys.readouterr()
    assert captured.out.splitlines()[-1] == robot_main.CONNECTING
    started, advice = captured.err.splitlines()[-2:]
    assert started.startswith("robot app: the robot app did not start: ")
    assert "The arm on can4 is not folded at rest (" in started
    assert re.search(why, started)
    assert "Only parameter reads were sent and the bus is released" in started
    assert advice == f"robot app: {robot_main.NOT_AT_REST}"
    # The bus was opened, every joint was read twice, the bus was let go of:
    # no mode write, no fault clear, no enable, no command and no disconnect.
    (sdk,) = bench.sdks
    reads = [f"read joint{i}" for i in range(1, 7)]
    assert bench.calls == ["open the bus", *reads, *reads, "close_bus", "close"]
    assert bench.cuts() == [] and sdk.disconnects == 0
    (row,) = bench.rest_rows()
    assert row["key"] == "can4" and row["verdict"] != "rest"
    assert (row["marker"], row["tol_rad"]) == (False, 0.12)
    # No marker was written and none of APAI's confirmations was looked for.
    assert not bench.holds.exists()
    assert left_threads() == []


class Session:
    """The program under way: what is typed at it and a console client."""

    def __init__(self, typed: TypedLines, client: WorkstationClient, running: Any):
        self.typed, self.client, self.running = typed, client, running
        self.apps: list[Any] = []


@contextlib.contextmanager
def program(
    bench: Bench, folder: Path, monkeypatch, *answers: str
) -> Iterator[Session]:
    """Run --driver real against a real robot link service on 127.0.0.1."""
    from isaaccapture_examples.agentic_remote.link_service.listen import RobotListener
    from isaaccapture_examples.agentic_remote.link_service.local import connect_local
    from isaaccapture_examples.agentic_remote.link_service.service import (
        RobotLinkService,
    )

    from test_link_programs import Running

    typed = TypedLines()
    for line in answers:
        typed.type(line)
    monkeypatch.setattr(robot_main, "typed_lines", lambda: typed)
    apps: list[Any] = []
    real_app = robot_main.RobotApp

    def robot_app(*args: Any, **kwargs: Any) -> Any:
        apps.append(real_app(*args, **kwargs))
        return apps[-1]

    monkeypatch.setattr(robot_main, "RobotApp", robot_app)
    listener = RobotListener("127.0.0.1", 0)
    path = str(folder / "s.sock")
    service = RobotLinkService(listener.accept, socket_path=path)
    service.start()
    console = client = None
    try:
        running = Running(robot_main.main, bench.argv("--connect", listener.endpoint))
        console = connect_local(path, "console", "test", WAIT_S)
        client = WorkstationClient(console, SystemClock())
        session = Session(typed, client, running)
        session.apps = apps
        console.wait_status(lambda s: s.robot_connected, WAIT_S)
        yield session
    finally:
        for app in apps:
            app.close()  # ends a program a failed test left running
        typed.end()
        if client is not None:
            client.close()
        if console is not None:
            console.close()
        service.close()
        listener.close()


@pytest.fixture
def folder(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp("rd")  # a short path, mode 0700


@rig
@pytest.mark.link
@pytest.mark.parametrize("reading", RIG_REST_READINGS)
def test_three_yes_connect_the_arm_at_measured_rest_and_welcome_says_real(
    bench, folder, capsys, monkeypatch, reading
):
    bench.reading = list(reading)
    rig_file = load_config(RIG_YAML)
    with program(bench, folder, monkeypatch, "yes", "yes", "yes") as session:
        welcome = session.client.wait_welcome(WAIT_S)
        assert welcome.driver == "real"
        assert welcome.camera_calibration_id == rig_file.camera_calibration_id
        assert welcome.profile_id == "rebot-b601-rs-thor"
        assert list(welcome.joint_names) == [f"joint{i}" for i in range(1, 7)]
        state = session.client.wait_state(lambda s: s.gripper.referenced, WAIT_S)
        # At rest for APAI's connect (0.12 rad), not for a release (0.05 rad).
        assert (state.motion, state.torque_on) == ("hold", True)
        assert state.gripper.load == "empty"
        (app,) = session.apps
        assert app.allow_reference is False  # the rig file's jaws.allow_reference

        # The connect: both reads of every joint came before anything else.
        (sdk,) = bench.sdks
        assert sdk.hw_yaml.endswith("sdk-can4.yaml")
        reads = [f"read joint{i}" for i in range(1, 7)]
        clears = [f"clear_error {name}" for name in sorted(sdk._motor_map)]
        assert bench.calls[: 15 + len(clears)] == [
            "open the bus",
            *reads,
            *reads,
            "mode_mit arm",
            "mode_mit gripper",
            *clears,
        ]
        assert bench.calls.count("enable") == 1
        assert bench.calls.count("open the bus") == 1
        (row,) = bench.rest_rows()
        assert (row["verdict"], row["key"], row["marker"]) == ("rest", "can4", False)
        assert row["q"] == pytest.approx(reading)
        # No joint was commanded by the start; the jaws were referenced.
        assert "send_mit arm" not in bench.calls
        assert "send_mit gripper" in bench.calls

        # A typed rest takes the route of the functional rest, in real time:
        # up to the ready pose, then the fold in steps. It ends inside the
        # 0.05 rad of a release with the jaws closed and the motors off.
        jaw_mark = len(sdk.gripper.sent)
        session.typed.type("rest")
        session.typed.type("path is clear")
        state = session.client.wait_state(
            lambda s: s.motion == "rest_unpowered", REST_WAIT_S
        )
        assert not state.torque_on and sdk.disconnects == 1
        closes = sdk.gripper.sent[jaw_mark:]
        assert closes and all(q == [0.0] for q, _ in closes)
        # The wanted motors-off brings no fault-stop, also half a second on.
        time.sleep(0.6)
        state = session.client.wait_state(lambda s: s.seq > state.seq, WAIT_S)
        assert (state.stop_level, state.fault_reason) == ("none", "")
        session.typed.type("quit")
        assert session.running.result() == 0
    assert sdk.disconnects == 1
    (released_at,) = sdk.released_at
    assert max(abs(value) for value in released_at) <= 0.05
    out = capsys.readouterr().out.splitlines()
    assert out[1:4] == [
        question.replace("bench-7", rig_file.camera_calibration_id)
        for question in QUESTIONS
    ]
    # APAI's gripper guard writes its own lines between the program's.
    said = [line for line in out[4:] if line.startswith("robot app: ")]
    assert said[0] == robot_main.CONNECTING
    assert said[1] == f"{CONNECTED}, motion hold, jaws referenced"
    # What the Thor terminal showed of the rest, and of the motors going off.
    told = [line for line in out if line.startswith(("[rest]", "[jaws]", "[release]"))]
    assert told[0].startswith("[rest] return to rest: up to the ready pose: the arm")
    assert [line.split(": ")[1] for line in told[1:5]] == [
        f"fold step {step} of 4" for step in range(1, 5)
    ]
    assert told[5:] == [
        "[rest] return to rest: the arm is at rest, motors on",
        "[jaws] at rest: closing the jaws with the driver's close",
        "[jaws] at rest: the jaws are closed",
        "[release] torque release sent at rest: the driver does not confirm that "
        "the motors are off",
    ]
    assert "rest: completed" in out
    assert said[2].startswith("robot app: session ")
    assert left_threads() == []


@rig
@pytest.mark.link
def test_the_thor_terminal_reads_on_after_the_three_answers(
    bench, folder, capsys, monkeypatch
):
    with program(bench, folder, monkeypatch, "YES", " yes ", "Yes") as session:
        session.client.wait_welcome(WAIT_S)
        before = session.client.wait_state(lambda s: s.gripper.referenced, WAIT_S)
        # An arm that reads exactly the rest pose holds at rest.
        assert (before.motion, before.stop_number) == ("rest_hold", 0)
        session.typed.type("stop")
        state = session.client.wait_state(lambda s: s.stop_number == 1, WAIT_S)
        assert (state.torque_on, state.owner) == (True, "nobody")
        (sdk,) = bench.sdks
        assert sdk.disconnects == 0
        # A fourth yes is no command and confirms nothing.
        session.typed.type("yes")
        session.typed.type("quit")
        assert session.running.result() == 0
    assert sdk.disconnects == 1
    assert sdk.released_at == [[0.0] * 6]
    out = capsys.readouterr().out
    assert "unknown command 'yes'; type help" in out
    assert left_threads() == []


@rig
@pytest.mark.link
def test_the_real_driver_runs_the_typed_reference_only_with_the_setting(
    bench, folder, capsys, monkeypatch, tmp_path
):
    data = yaml.safe_load(RIG_YAML.read_text(encoding="utf-8"))
    assert data["jaws"] == {"allow_reference": False}
    data["jaws"]["allow_reference"] = True
    allowed = tmp_path / "robot_app_reference.yaml"
    allowed.write_text(yaml.safe_dump(data), encoding="utf-8")
    argv = bench.argv
    monkeypatch.setattr(
        bench,
        "argv",
        lambda *extra: [
            str(allowed) if word == str(RIG_YAML) else word for word in argv(*extra)
        ],
    )
    with program(bench, folder, monkeypatch, "yes", "yes", "yes") as session:
        session.client.wait_state(lambda s: s.gripper.referenced, WAIT_S)
        (app,) = session.apps
        assert app.allow_reference is True
        session.typed.type("quit")
        assert session.running.result() == 0
    assert left_threads() == []


# A person on the real driver. A takeover makes a person the control owner; a
# human drive is one stretch of human motion toward the joint targets of
# HUMAN_SETPOINT link messages. The live view is the viewer's page on the
# workstation with its live picture showing; the viewer reports its age in
# VIEW_STATUS link messages. The rig file requires it. No viewer runs below:
# no VIEW_STATUS is sent.

NO_LIVE_VIEW_LINE = (
    "robot app: human driving is not tied to a live view "
    "(human.require_live_view false)"
)
VIEW_NOT_FRESH = (
    "the live view is not fresh: no VIEW_STATUS has arrived; show the viewer's "
    "live picture in a visible browser tab"
)
# Off the two joint stops the arm rests on: the first motion of a squeeze.
OFF_THE_STOPS = [0.0, 0.03, 0.03, 0.0, 0.0, 0.0]


def a_person_on_the_rig_file(bench: Bench, monkeypatch, tmp_path, **human: Any) -> None:
    """Run the program on a copy of the rig file with human driving switched on."""
    data = yaml.safe_load(RIG_YAML.read_text(encoding="utf-8"))
    # As shipped: human driving off, and the live view required.
    assert data["human"]["enabled"] is False
    assert data["human"]["require_live_view"] is True
    data["human"].update({"enabled": True, **human})
    changed = tmp_path / "robot_app_person.yaml"
    changed.write_text(yaml.safe_dump(data), encoding="utf-8")
    argv = bench.argv
    monkeypatch.setattr(
        bench,
        "argv",
        lambda *extra: [
            str(changed) if word == str(RIG_YAML) else word for word in argv(*extra)
        ],
    )


class Controller:
    """Stands in for the human program: 50 HUMAN_SETPOINTs per second.

    Attached to the robot link service in the role human, with the headset's
    source. Each quotes the numbers of the newest ROBOT_STATE. goal is the
    joint target of a squeezed grip; None: the grip is released.
    """

    def __init__(self, socket_path: str) -> None:
        from isaaccapture_examples.agentic_remote.link_service.local import (
            connect_local,
        )

        self.goal: list[float] | None = None
        self.clock = SystemClock()
        self.end = connect_local(socket_path, "human", "test-human", WAIT_S)
        self.ws = WorkstationClient(self.end, self.clock, name="human")
        self._ended = threading.Event()
        self._thread = threading.Thread(target=self._feed, daemon=True)
        self._thread.start()

    def _feed(self) -> None:
        while not self._ended.wait(0.02):
            state = self.ws.state
            if state is None:
                continue
            quoted = types.SimpleNamespace(snapshot=lambda state=state: state)
            self.ws.send(
                setpoint(types.SimpleNamespace(app=quoted, clock=self.clock), self.goal)
            )

    def right_b(self) -> Any:
        """Ask for the takeover, as the human program does on right B; the RESULT."""
        asked = self.ws.request_owner("human", reason="right B held 1.0 s")
        return self.ws.wait_result(asked, timeout_s=WAIT_S)

    def close(self) -> None:
        self._ended.set()
        self._thread.join(WAIT_S)
        self.ws.close()
        self.end.close()


def human(state: Any) -> dict[str, Any]:
    return state.health["human"]


@rig
@pytest.mark.link
def test_a_person_takes_over_and_drives_on_the_real_driver_with_no_live_view(
    bench, folder, capsys, caplog, monkeypatch, tmp_path
):
    # The robot app takes a file that requires no live view with any driver.
    a_person_on_the_rig_file(bench, monkeypatch, tmp_path, require_live_view=False)
    with (
        caplog.at_level("INFO", logger=LOGGER),
        program(bench, folder, monkeypatch, "yes", "yes", "yes") as session,
    ):
        assert session.client.wait_welcome(WAIT_S).driver == "real"
        session.client.wait_state(lambda s: s.gripper.referenced, WAIT_S)
        (app,) = session.apps
        (sdk,) = bench.sdks
        hand = Controller(str(folder / "s.sock"))
        try:
            # Fresh controller data are all a takeover waits for: no view.
            ready = session.client.wait_state(
                lambda s: s.teleop_stage == "ready" and human(s)["takeover"] == "",
                WAIT_S,
            )
            assert ready.view_age_s is None
            assert human(ready)["live_view_required"] is False
            taken = hand.right_b()
            assert taken.status == "completed", taken.reason
            owned = session.client.wait_state(
                lambda s: s.owner == "human" and not human(s)["needs_release"], WAIT_S
            )
            assert human(owned)["takeover_open"] and "send_mit arm" not in bench.calls
            # A squeeze: the arm follows the hand off its stops. The released
            # grip ends the drive in a hold.
            hand.goal = list(OFF_THE_STOPS)
            session.client.wait_until(
                lambda: (
                    bool(sdk.arm.sent)
                    and sdk.arm.sent[-1][0] == pytest.approx(OFF_THE_STOPS, abs=1e-9)
                ),
                WAIT_S,
            )
            hand.goal = None
            held = session.client.wait_state(
                lambda s: not human(s)["driving"] and human(s)["ended"] != "", WAIT_S
            )
            assert human(held)["ended"] == "the grip was released"
            assert (held.owner, held.motion, held.torque_on) == ("human", "hold", True)
            assert human(held)["refusal"] == ""
            # Every setpoint lay within the Thor's speed clamp for a person.
            rig_file = load_config(RIG_YAML)
            clamp = rig_file.limits.human.max_joint_speed * rig_file.executor.period_s
            sent = np.array([[0.0] * 6, *[q for q, _ in sdk.arm.sent]])
            assert np.abs(np.diff(sent, axis=0)).max() <= clamp + 1e-12
            # A STOP on right B wins: nobody owns the arm, which holds.
            hand.ws.stop(reason="right B", sender="controller")
            stopped = session.client.wait_state(lambda s: s.stop_number == 1, WAIT_S)
            assert (stopped.owner, stopped.torque_on) == ("nobody", True)
        finally:
            hand.close()
        # No VIEW_STATUS arrived at any time.
        assert app._view is None
        # The program ends where the arm holds, off the rest pose with the
        # takeover open: nothing moves and no torque is cut.
        app.close()
        assert session.running.result() == 0
    assert sdk.disconnects == 0 and sdk.released_at == []
    out = capsys.readouterr().out.splitlines()
    said = [line for line in out if line.startswith("robot app: ")]
    # After the line on the connect, before the session's.
    at = said.index(robot_main.CONNECTING)
    assert said[at + 1].startswith(CONNECTED)
    assert said[at + 2] == NO_LIVE_VIEW_LINE and said.count(NO_LIVE_VIEW_LINE) == 1
    assert said[at + 3].startswith("robot app: session ")
    logged = [record.getMessage() for record in caplog.records]
    assert logged.count(NO_LIVE_VIEW_LINE.removeprefix("robot app: ")) == 1
    # Nothing told the operator of a live view that is not wanted.
    assert not any("live view is not fresh" in line for line in out)
    assert left_threads() == []


@rig
@pytest.mark.link
def test_the_same_takeover_is_refused_on_the_real_driver_where_the_view_is_required(
    bench, folder, capsys, monkeypatch, tmp_path
):
    # The rig file as it ships, with human driving switched on: without the
    # viewer's page and its live picture no takeover is taken.
    a_person_on_the_rig_file(bench, monkeypatch, tmp_path)
    with program(bench, folder, monkeypatch, "yes", "yes", "yes") as session:
        session.client.wait_state(lambda s: s.gripper.referenced, WAIT_S)
        hand = Controller(str(folder / "s.sock"))
        try:
            waiting = session.client.wait_state(
                lambda s: (
                    s.teleop_stage == "ready"
                    and human(s)["takeover"].startswith("the live view")
                ),
                WAIT_S,
            )
            assert human(waiting)["takeover"] == VIEW_NOT_FRESH
            assert human(waiting)["live_view_required"] is True
            refused = hand.right_b()
            assert (refused.status, refused.reason) == ("refused", VIEW_NOT_FRESH)
            # A squeeze moves nothing: nobody took the arm over.
            hand.goal = list(OFF_THE_STOPS)
            time.sleep(0.3)
            state = session.client.wait_state(lambda s: s.seq > waiting.seq, WAIT_S)
            assert (state.owner, human(state)["takeover_open"]) == ("nobody", False)
            assert "send_mit arm" not in bench.calls
        finally:
            hand.close()
        session.typed.type("quit")
        assert session.running.result() == 0
    out = capsys.readouterr().out.splitlines()
    assert NO_LIVE_VIEW_LINE not in out
    said = [line for line in out if line.startswith("robot app: ")]
    at = said.index(robot_main.CONNECTING)
    assert said[at + 1].startswith(CONNECTED)
    assert said[at + 2].startswith("robot app: session ")
    assert left_threads() == []


@rig
@pytest.mark.link
def test_no_hold_is_said_of_joints_that_stopped_answering(bench, folder, monkeypatch):
    bench.reading = list(RIG_REST_READINGS[0])
    with program(bench, folder, monkeypatch, "yes", "yes", "yes") as session:
        client = session.client
        client.wait_welcome(WAIT_S)
        client.wait_state(lambda s: s.gripper.referenced, WAIT_S)
        (sdk,) = bench.sdks
        # A typed rest stopped on its way: a joint target stands and the
        # joints read at it.
        session.typed.type("rest")
        session.typed.type("path is clear")
        client.wait_state(lambda s: s.motion == "moving" and s.target_q, WAIT_S)
        time.sleep(1.0)
        session.typed.type("stop")
        client.wait_state(lambda s: s.health["motors"]["state"] == "holding", WAIT_S)
        # Every joint motor stops answering, as when the 48 V supply goes.
        for motor in sdk.arm._mm.values():
            motor.missed = 10**9
        client.wait_state(lambda s: s.stop_level == "fault", WAIT_S)
        state = client.wait_state(
            lambda s: s.health["motors"]["state"] != "holding", WAIT_S
        )
        assert state.health["motors"]["state"] == "unknown"
        assert state.health["motors"]["reason"].endswith("not read for more than 1 s")
        assert state.fault_reason.startswith("joint feedback lost for ")
        # The program has not switched the motors off; it only cannot see them.
        assert state.torque_on and sdk.disconnects == 0
        # The joints answer again: they are read at the target they hold.
        for motor in sdk.arm._mm.values():
            motor.missed = 0
        client.wait_state(lambda s: s.health["motors"]["state"] == "holding", WAIT_S)
    assert sdk.disconnects == 0


@rig
@pytest.mark.link
def test_a_joint_read_without_an_answer_just_before_a_wanted_motors_off_is_no_fault(
    bench, folder, monkeypatch
):
    armed, in_flight = threading.Event(), threading.Event()
    read = RigMotor.robstride_get_param_f32

    def unanswered_once(self, param, timeout_ms=None):
        # Only a read of one joint passes a timeout.
        one_joint = timeout_ms is not None and self.name.startswith("joint")
        if one_joint and armed.is_set() and not in_flight.is_set():
            in_flight.set()
            time.sleep(0.03)
            raise OSError("no answer")
        return read(self, param, timeout_ms)

    monkeypatch.setattr(RigMotor, "robstride_get_param_f32", unanswered_once)

    def half_a_second(sdk):
        """The driver's disconnect takes as long as it was seen to on the arm."""
        disconnect = sdk.disconnect

        def slow():
            time.sleep(0.5)
            disconnect()

        sdk.disconnect = slow

    bench.prepare = half_a_second
    with program(bench, folder, monkeypatch, "yes", "yes", "yes") as session:
        client = session.client
        client.wait_welcome(WAIT_S)
        client.wait_state(lambda s: s.gripper.referenced, WAIT_S)
        (app,) = session.apps
        (sdk,) = bench.sdks
        paused = app._rest._reads_paused

        def paused_after_that_read():
            # The idle read under way when the release pauses the reads.
            armed.set()
            assert in_flight.wait(2.0), "no idle read came"
            return paused()

        app._rest._reads_paused = paused_after_that_read
        session.typed.type("release")
        state = client.wait_state(lambda s: s.motion == "rest_unpowered", REST_WAIT_S)
        time.sleep(0.6)
        state = client.wait_state(lambda s: s.seq > state.seq, WAIT_S)
        assert (state.stop_level, state.fault_reason) == ("none", "")
        assert in_flight.is_set() and sdk.disconnects == 1
        assert max(abs(value) for value in sdk.released_at[0]) <= 0.05
        # No reset is needed: the motors come on again and the joints are read.
        session.typed.type("on")
        session.typed.type("yes")
        on = client.wait_state(lambda s: s.torque_on, WAIT_S)
        assert app.wait_motion_idle(WAIT_S)  # the jaw reference of `on` ended

        def read_again(s) -> bool:
            age = s.health["reading_age_s"]
            return s.seq > on.seq and age is not None and age < 0.5

        state = client.wait_state(read_again, WAIT_S)
        assert (state.stop_level, state.fault_reason) == ("none", "")
        # ROBOT_STATE's jaw reading is the new driver's, read as a joint is.
        jaw = bench.sdks[1].gripper._mm["jaw"]
        by = time.monotonic() + WAIT_S
        while (0x7019, 50) not in jaw.reads:
            assert time.monotonic() < by, "no jaw reading after on"
            time.sleep(0.02)
        session.typed.type("quit")
        assert session.running.result() == 0
    assert left_threads() == []


# Typed-ahead input: the questions on a pseudo-terminal


@pytest.fixture
def terminal_pair() -> Iterator[tuple[int, int]]:
    """A pseudo-terminal: the test types at the first, the program reads the second."""
    master, slave = pty.openpty()
    yield master, slave
    os.close(master)
    os.close(slave)


def test_three_yes_typed_before_the_questions_confirm_nothing(terminal_pair):
    master, slave = terminal_pair
    os.write(master, b"yes\nyes\nyes\n")
    lines = robot_main._FdLines(slave)
    said: list[str] = []

    def say(text: str) -> None:
        said.append(text)
        # Nobody answers the question: the input ends a little later.
        threading.Timer(0.3, lines.close).start()

    assert robot_main.confirm_real_driver("bench-7", lines, say) is False
    assert said == QUESTIONS[:1]


def typed_at_each_question(lines: Any, type_answer: Any) -> list[list[str]]:
    """Call type_answer() once a question waits for its answer.

    That is after the question was written and what was typed before it was
    dropped. Returns what each drop gave back.
    """
    drop, dropped = lines.drop_typed_ahead, []

    def drop_then_type() -> list[str]:
        gone = drop()
        dropped.append(gone)
        type_answer()
        return gone

    lines.drop_typed_ahead = drop_then_type
    return dropped


def test_what_is_typed_once_a_question_shows_answers_it(terminal_pair):
    master, slave = terminal_pair
    os.write(master, b"no\n")  # typed ahead of the first question
    lines = robot_main._FdLines(slave)
    said: list[str] = []
    # The answer, and a line typed ahead of the question after it.
    dropped = typed_at_each_question(lines, lambda: os.write(master, b"yes\nno\n"))
    assert robot_main.confirm_real_driver("bench-7", lines, said.append) is True
    assert said == QUESTIONS
    assert dropped == [["no"]] * 3


def test_a_yes_typed_while_a_question_is_being_written_answers_nothing(
    terminal_pair,
):
    master, slave = terminal_pair
    lines = robot_main._FdLines(slave)
    said: list[str] = []

    def say(text: str) -> None:
        # The output is held up, as after Ctrl+S: typed before the question shows.
        os.write(master, b"yes\n")
        said.append(text)

    # Nobody answers once it shows: the input ends.
    dropped = typed_at_each_question(lines, lines.close)
    assert robot_main.confirm_real_driver("bench-7", lines, say) is False
    assert said == QUESTIONS[:1] and dropped == [["yes"]]


def test_a_question_that_cannot_be_written_is_not_confirmed():
    said: list[str] = []

    def say(text: str) -> None:
        said.append(text)
        if len(said) == 2:
            raise OSError(5, "Input/output error")  # the terminal is gone

    assert robot_main.confirm_real_driver("bench-7", iter(["yes"] * 3), say) is False
    assert said == QUESTIONS[:2]


def test_lines_read_before_the_terminal_went_away_answer_no_question():
    master, slave = pty.openpty()
    tty.setraw(slave)  # not in line mode: one read hands over every line typed
    try:
        os.write(master, b"x\nyes\nyes\nyes\n")
        lines = robot_main._FdLines(slave)
        assert next(lines) == "x"
        assert lines._buffer == b"yes\nyes\nyes\n"  # read with the line before
        os.close(master)  # the terminal goes away
        master = None
        said: list[str] = []
        assert robot_main.confirm_real_driver("bench-7", lines, said.append) is False
        assert said == QUESTIONS[:1]
    finally:
        for fd in (master, slave):
            if fd is not None:
                os.close(fd)


def test_lines_already_read_from_a_terminal_are_dropped_before_a_question(
    terminal_pair,
):
    _, slave = terminal_pair
    lines = robot_main._FdLines(slave)
    lines._buffer = b"yes\nyes\nyes\n"  # read from the terminal with a line before
    said: list[str] = []

    def say(text: str) -> None:
        said.append(text)
        lines.close()

    assert robot_main.confirm_real_driver("bench-7", lines, say) is False
    assert said == QUESTIONS[:1]


@pytest.mark.parametrize(
    ("typed", "confirmed"), [(b"yes\nyes\nyes", False), (b"yes\nyes\nyes\n", True)]
)
def test_a_last_line_without_a_newline_confirms_nothing(typed, confirmed):
    read_end, write_end = os.pipe()
    try:
        os.write(write_end, typed)
        os.close(write_end)
        lines = robot_main._FdLines(read_end)
        said: list[str] = []
        assert (
            robot_main.confirm_real_driver("bench-7", lines, said.append) is confirmed
        )
        assert said == QUESTIONS
    finally:
        os.close(read_end)


def test_a_yes_ended_by_the_end_of_a_terminals_input_confirms_nothing(terminal_pair):
    master, slave = terminal_pair
    lines = robot_main._FdLines(slave)
    said: list[str] = []

    def answer() -> None:
        # Ctrl+D twice after the third yes: the line is handed over without a
        # newline, then the input ends.
        os.write(master, b"yes\n" if len(said) < 3 else b"yes\x04\x04")

    typed_at_each_question(lines, answer)
    assert robot_main.confirm_real_driver("bench-7", lines, said.append) is False
    assert said == QUESTIONS


def test_the_program_drops_what_was_typed_before_its_questions(
    capsys, monkeypatch, bench_yaml, started, terminal_pair
):
    master, slave = terminal_pair
    fake_backend(monkeypatch)
    os.write(master, b"yes\nyes\nyes\n")  # typed while the program still loads
    monkeypatch.setattr(sys, "stdin", os.fdopen(slave, closefd=False))
    ended = threading.Event()

    def refuse() -> None:
        # Typed again and again: a no that came before the question is dropped.
        while not ended.wait(0.3):
            os.write(master, b"no\n")

    typist = threading.Thread(target=refuse, daemon=True)
    typist.start()
    try:
        code, out, err = real_driver(capsys, real_argv(bench_yaml))
    finally:
        ended.set()
        typist.join(WAIT_S)
    assert code == 2
    assert out == [BUILT, QUESTIONS[0]]
    assert err == NOT_CONFIRMED and started == []


# Signals and failures around the start, on APAI's driver over the stand-in SDK


def no_dial(monkeypatch) -> None:
    """The program dials nothing: no robot link service is needed."""
    monkeypatch.setattr(robot_main, "_remote_link_missing", lambda: None)
    monkeypatch.setattr(robot_main.RobotDialer, "start", lambda self: None)


def closed_once(monkeypatch, ready: Any) -> list[Any]:
    """Close every RobotApp the program builds, motors left on, once ready(app).

    It is closed after WAIT_S at the latest, so no test leaves a program running.
    """
    apps: list[Any] = []
    real_app = robot_main.RobotApp

    def close_when_ready(app: Any) -> None:
        by = time.monotonic() + WAIT_S
        while time.monotonic() < by and not ready(app):
            time.sleep(0.02)
        app.close()

    def robot_app(*args: Any, **kwargs: Any) -> Any:
        app = real_app(*args, **kwargs)
        apps.append(app)
        start = app.start

        def started() -> None:
            start()
            threading.Thread(target=close_when_ready, args=(app,), daemon=True).start()

        app.start = started
        return app

    monkeypatch.setattr(robot_main, "RobotApp", robot_app)
    return apps


def signal_during_the_jaw_reference(bench: Bench, number: int) -> list[str]:
    """Raise the signal on the second jaw command of the start; when it was."""
    seen: list[str] = []

    def prepare(sdk: RigSdk) -> None:
        send = sdk.gripper.send_mit

        def send_mit(q: Any, kp: Any = None, kd: Any = None, tau: Any = None) -> None:
            send(q, kp, kd, tau)
            if sdk.calls.count("send_mit gripper") == 2:
                seen.append(f"after {sdk.calls.count('enable')} enable")
                signal.raise_signal(number)

        sdk.gripper.send_mit = send_mit

    bench.prepare = prepare
    return seen


@pytest.fixture
def signals_kept() -> Iterator[list[int]]:
    """The signals that reached the handlers in place before the program."""
    reached: list[int] = []
    numbers = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
    before = [signal.signal(n, lambda n, _: reached.append(n)) for n in numbers]
    mine = [signal.getsignal(n) for n in numbers]
    yield reached
    # The program put back what it found.
    after = [signal.getsignal(n) for n in numbers]
    for number, handler in zip(numbers, before):
        signal.signal(number, handler)
    assert after == mine


@rig
def test_ctrl_c_during_the_start_is_a_stop_once_the_arm_is_connected(
    bench, capsys, monkeypatch, signals_kept
):
    seen = signal_during_the_jaw_reference(bench, signal.SIGINT)
    no_dial(monkeypatch)
    apps = closed_once(monkeypatch, lambda app: app.snapshot().stop_number >= 1)
    monkeypatch.setattr(robot_main, "typed_lines", lambda: iter(["yes"] * 3))
    assert robot_main.main(bench.argv("--connect", ENDPOINT)) == 0
    assert seen == ["after 1 enable"] and signals_kept == []
    (sdk,) = bench.sdks
    (app,) = apps
    state = app.snapshot()
    assert (state.stop_number, state.stop_level.value) == (1, "stop")
    # The start ran to its end: the jaws are referenced and the motors are on.
    assert app._arm.gripper_referenced()
    assert bench.calls.count("enable") == 1 and sdk.disconnects == 0
    out = capsys.readouterr().out
    assert CONNECTED in out
    assert "Ctrl+C" in out
    assert left_threads() == []


# What a SIGTERM leaves once the start has run to its end: at measured rest the
# jaws closed and the motors off, as the experiment's launcher ends; away from
# rest nothing moves and the motors stay on.
SIGTERM_AT_REST = (
    "robot app: SIGTERM; closing; the arm lies at measured rest: the motors are off"
)
SIGTERM_AWAY = "robot app: SIGTERM; closing with the motors left on"


@rig
@pytest.mark.parametrize("at_rest", [True, False], ids=["at rest", "away from rest"])
def test_sigterm_during_the_start_lets_the_start_end_and_cuts_torque_only_at_rest(
    bench, capsys, monkeypatch, signals_kept, at_rest
):
    if not at_rest:
        # Inside the 0.12 rad of APAI's connect, outside the 0.05 of a release.
        bench.reading = list(RIG_REST_READINGS[0])
    seen = signal_during_the_jaw_reference(bench, signal.SIGTERM)
    no_dial(monkeypatch)
    apps = closed_once(monkeypatch, lambda app: not app.running)
    monkeypatch.setattr(robot_main, "typed_lines", lambda: iter(["yes"] * 3))
    assert robot_main.main(bench.argv("--connect", ENDPOINT)) == 0
    assert seen == ["after 1 enable"] and signals_kept == []
    (sdk,) = bench.sdks
    (app,) = apps
    assert bench.calls.count("enable") == 1
    out = capsys.readouterr().out.splitlines()
    if at_rest:
        assert SIGTERM_AT_REST in out and SIGTERM_AWAY not in out
        assert sdk.disconnects == 1 and sdk.released_at == [[0.0] * 6]
    else:
        assert SIGTERM_AWAY in out and SIGTERM_AT_REST not in out
        why = out[out.index(SIGTERM_AWAY) + 1]
        assert why.startswith(
            "robot app: the motors stay on because the arm is not at rest: joint "
        )
        assert sdk.disconnects == 0 and app._arm.torque_on()
        assert app._arm.gripper_referenced()
        assert "send_mit arm" not in bench.calls  # nothing moved the arm
    assert left_threads() == []


@pytest.mark.parametrize("number", [signal.SIGINT, signal.SIGTERM])
def test_a_signal_while_the_checks_are_made_once_more_connects_nothing(
    capsys, monkeypatch, bench_yaml, started, signals_kept, number
):
    asked = fake_backend(monkeypatch)
    backend = sys.modules[BACKEND]

    def unresolved_hold(of: Any) -> None:
        if "arm" in asked:  # the look after the questions
            signal.raise_signal(number)

    backend.unresolved_hold = unresolved_hold
    answer(monkeypatch, "yes", "yes", "yes")
    code, out, err = real_driver(capsys, real_argv(bench_yaml))
    assert code == 2
    assert out == [BUILT, *QUESTIONS]
    assert err == NOT_CONFIRMED
    assert started == [] and signals_kept == []


@rig
@pytest.mark.parametrize("number", [signal.SIGINT, signal.SIGTERM])
def test_a_signal_after_the_connecting_line_is_kept_until_the_arm_is_connected(
    bench, capsys, monkeypatch, signals_kept, number
):
    no_dial(monkeypatch)
    stopped = number == signal.SIGINT
    apps = closed_once(
        monkeypatch,
        (lambda app: app.snapshot().stop_number >= 1)
        if stopped
        else (lambda app: not app.running),
    )
    build = robot_main.RobotApp
    seen: list[list[str]] = []

    def robot_app(*args: Any, **kwargs: Any) -> Any:
        # The line "connecting the arm" is out; the bus is not opened yet.
        seen.append(list(bench.calls))
        signal.raise_signal(number)
        return build(*args, **kwargs)

    monkeypatch.setattr(robot_main, "RobotApp", robot_app)
    monkeypatch.setattr(robot_main, "typed_lines", lambda: iter(["yes"] * 3))
    assert robot_main.main(bench.argv("--connect", ENDPOINT)) == 0
    assert seen == [[]] and signals_kept == []
    captured = capsys.readouterr()
    out = captured.out.splitlines()
    # The signal does not end the program with "not confirmed": the arm is
    # connected first, its joints read twice before anything that cuts torque.
    assert NOT_CONFIRMED not in captured.err
    assert CONNECTED in captured.out
    (sdk,) = bench.sdks
    reads = [f"read joint{i}" for i in range(1, 7)]
    assert bench.calls[:14] == ["open the bus", *reads, *reads, "mode_mit arm"]
    assert bench.calls.count("enable") == 1
    (app,) = apps
    if stopped:
        assert app.snapshot().stop_number == 1 and sdk.disconnects == 0
        assert any("STOP from thor_terminal: Ctrl+C" in line for line in out)
    else:
        # The arm lies at measured rest: the SIGTERM switches the motors off.
        assert SIGTERM_AT_REST in out and sdk.disconnects == 1
        assert sdk.released_at == [[0.0] * 6]
    assert left_threads() == []


@rig
def test_a_terminal_that_closes_during_the_start_does_not_end_it_half_way(
    bench, capsys, monkeypatch, signals_kept
):
    during: list[tuple[str, Any]] = []

    def hang_up(at: str) -> None:
        during.append((at, signal.getsignal(signal.SIGHUP)))
        signal.raise_signal(signal.SIGHUP)  # what a closed terminal sends

    def prepare(sdk: RigSdk) -> None:
        mode_mit, enable, send = sdk.arm.mode_mit, sdk.arm.enable, sdk.gripper.send_mit

        def first_frame() -> None:
            hang_up("the mode write")
            mode_mit()

        def enabled() -> None:
            enable()
            hang_up("the enable")

        def send_mit(q: Any, kp: Any = None, kd: Any = None, tau: Any = None) -> None:
            send(q, kp, kd, tau)
            if sdk.calls.count("send_mit gripper") == 2:
                hang_up("the jaw reference")

        sdk.arm.mode_mit, sdk.arm.enable = first_frame, enabled
        sdk.gripper.send_mit = send_mit

    bench.prepare = prepare
    no_dial(monkeypatch)
    apps = closed_once(monkeypatch, lambda app: True)
    monkeypatch.setattr(robot_main, "typed_lines", lambda: iter(["yes"] * 3))
    assert robot_main.main(bench.argv("--connect", ENDPOINT)) == 0
    # None of the three hang-ups reached the handler found at the start.
    assert signals_kept == []
    assert during == [
        ("the mode write", signal.SIG_IGN),
        ("the enable", signal.SIG_IGN),
        ("the jaw reference", signal.SIG_IGN),
    ]
    (sdk,) = bench.sdks
    (app,) = apps
    assert app._arm.gripper_referenced()
    assert bench.calls.count("enable") == 1 and sdk.disconnects == 0
    out = capsys.readouterr().out
    assert CONNECTED in out
    assert left_threads() == []


def test_a_terminal_that_is_gone_at_the_connecting_line_connects_nothing(
    capsys, monkeypatch, bench_yaml, started, signals_kept
):
    fake_backend(monkeypatch)
    answer(monkeypatch, "yes", "yes", "yes")
    write = robot_main._Lines.__call__

    def gone_at_the_line(self: Any, text: str) -> None:
        if text == robot_main.CONNECTING:
            raise OSError(5, "Input/output error")  # nobody is there to read it
        write(self, text)

    monkeypatch.setattr(robot_main._Lines, "__call__", gone_at_the_line)
    try:
        code, out, err = real_driver(capsys, real_argv(bench_yaml))
    except OSError as error:
        code, out, err = error, [], ""
    assert code == 2
    assert out == [BUILT, *QUESTIONS]
    assert err == NOT_CONFIRMED
    assert started == [] and signals_kept == []


@rig
def test_ctrl_c_at_a_question_connects_nothing_and_leaves_no_handler(
    bench, capsys, monkeypatch, signals_kept
):
    def answers() -> Iterator[str]:
        yield "yes"
        raise KeyboardInterrupt

    monkeypatch.setattr(robot_main, "typed_lines", answers)
    monkeypatch.setattr(robot_main, "_remote_link_missing", lambda: None)
    assert robot_main.main(bench.argv("--connect", ENDPOINT)) == 2
    assert capsys.readouterr().err == NOT_CONFIRMED
    assert bench.calls == [] and bench.sdks == [] and signals_kept == []


CLOSED_UNKNOWN = (
    "robot app closed; torque on: not known, the arm's connect did not end: the "
    "motors may be on"
)


def closing_records(caplog) -> list[str]:
    """What the robot app logged of its close."""
    said = [record.getMessage() for record in caplog.records]
    return [line for line in said if line.startswith("robot app closed")]


@rig
def test_a_connect_that_fails_after_the_enable_says_that_the_motors_may_be_on(
    bench, capsys, monkeypatch, caplog
):
    def prepare(sdk: RigSdk) -> None:
        motor = sdk.arm._mm["joint3"]
        read = motor.robstride_get_param_f32
        count = [0]

        def counted(param: int, timeout_ms: int | None = None) -> float:
            count[0] += 1
            if count[0] == 4:  # two rest reads, one before the enable, this one
                raise OSError("no answer")
            return read(param, timeout_ms)

        motor.robstride_get_param_f32 = counted

    bench.prepare = prepare
    no_dial(monkeypatch)
    monkeypatch.setattr(robot_main, "typed_lines", lambda: iter(["yes"] * 3))
    with caplog.at_level(logging.INFO, logger=LOGGER):
        assert robot_main.main(bench.argv("--connect", ENDPOINT)) == 1
    err = capsys.readouterr().err.splitlines()
    (sdk,) = bench.sdks
    # The motors were enabled and nothing switched them off again.
    assert "enable" in bench.calls and sdk.disconnects == 0
    assert [line.partition(": no answer")[0] for line in err[-2:]] == [
        "robot app: the robot app did not start",
        f"robot app: {robot_main.MOTORS_MAY_BE_ON}",
    ]
    # The log does not call motors off that the connect left enabled.
    assert closing_records(caplog) == [CLOSED_UNKNOWN]
    assert robot_main.MOTORS_MAY_BE_ON == (
        "the start failed after the connect began: the motors may be enabled and "
        "holding, and this program ends without releasing them. APAI's driver "
        "enables them only at measured rest: with the arm lying at rest, switch "
        "the 48 V off (safe only there or with the arm held by hand), then start "
        "again"
    )
    assert robot_main.NOT_AT_REST not in "\n".join(err)
    assert left_threads() == []
