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
import sys
import threading
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
    RIG_REST_READINGS,
    GuardTime,
    JawGroup,
    Motor,
    MotorGroup,
)
from test_main import quick_config

CONFIGS = repo_root() / "examples" / "agentic_remote" / "configs"
RIG_YAML = CONFIGS / "robot_app_rebot.yaml"
BACKEND = "isaaccapture_examples.agentic_remote.robot_app.apai_backend"
LOGGER = "isaaccapture.examples.agentic_remote"
ENDPOINT = "127.0.0.1:47610"
WAIT_S = 20.0  # real seconds a test gives the program before it fails
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
    import os

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
    pending: tuple[dict[str, Any], ...] = (),
    unfit: str = "",
) -> list[str]:
    """Stand in for the APAI backend module; what was asked of it, in order.

    hold is the hold marker of the arm's motor bus, pending the operator
    confirmations on file, unfit why the file does not fit the profile.
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

    def pending_confirmation(of: Any) -> bool:
        asked.append("pending")
        return any(found.get("key") in (None, "can4") for found in pending)

    module.load_apai_profile = load_apai_profile
    module.config_for_profile = config_for_profile
    module.build_apai_checkers = build_apai_checkers
    module.build_apai_arm = build_apai_arm
    module.unresolved_hold = unresolved_hold
    module.hold_bus = lambda of: "can4"
    module.pending_confirmation = pending_confirmation
    module.speed_cap = lambda of: 0.1
    monkeypatch.setitem(sys.modules, BACKEND, module)
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
    monkeypatch.setattr(robot_main, "_stdin_is_terminal", lambda: True)
    return calls


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


def test_the_test_runners_standard_input_is_no_terminal(monkeypatch):
    class Closed:
        def isatty(self) -> bool:
            raise ValueError("I/O operation on closed file")

    assert robot_main._stdin_is_terminal() is False  # pytest's own input
    for stdin in (None, Closed()):
        monkeypatch.setattr(sys, "stdin", stdin)
        assert robot_main._stdin_is_terminal() is False


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


@pytest.mark.parametrize("key", [None, "can4"])
def test_an_operators_confirmation_on_file_is_refused(
    capsys, monkeypatch, bench_yaml, started, key
):
    waiting = ({"key": key, "age_s": 3.0, "left_s": 117.0},)
    asked = fake_backend(monkeypatch, pending=waiting, unfit="too fast")
    monkeypatch.setattr(robot_main, "typed_lines", None)
    code, out, err = real_driver(capsys, real_argv(bench_yaml))
    assert (code, out) == (2, [])
    assert err == (
        f"{NOT_CONNECTED}an operator's confirmation for APAI's connect is on "
        "file for the arm on can4: with it APAI's driver would connect an arm it "
        "does not measure at rest. Revoke it with APAI's own tool, or wait until "
        "it has lapsed\n"
    )
    assert asked == ["load rebot_rs", "hold", "pending"] and started == []


def test_a_confirmation_for_another_bus_does_not_refuse_this_arm(
    capsys, monkeypatch, bench_yaml, started
):
    fake_backend(monkeypatch, pending=({"key": "can0", "age_s": 3.0},))
    answer(monkeypatch, "yes", "yes", "yes")
    assert real_driver(capsys, real_argv(bench_yaml))[0] == 0
    assert len(started) == 1


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
    assert asked == ["load rebot_rs", "hold", "pending", "config"]
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
    assert asked == ["load rebot_rs", "hold", "pending", *built, "hold", "pending"]
    (call,) = started
    assert (call["arm"], call["checkers"]) == ("the arm", "the checkers")
    assert (call["endpoint"], call["driver"]) == (ENDPOINT, "real")
    # The Thor terminal reads on where the answers ended.
    assert call["typed"] is lines
    assert list(lines) == ["stop", "quit"]


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
    waiting = [{"key": "can4", "age_s": 1.0}]
    if late == "hold":
        backend.unresolved_hold = lambda of: after_the_questions(marker)
    else:
        backend.pending_confirmation = lambda of: bool(after_the_questions(waiting))
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
    waiting = ({"key": None, "age_s": 1.0},)
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
    assert marker_gone == ["hold", "pending"]


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
        self.profile = str(folder / "thor_arm")
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
    speed cap of 0.1 rad/s. The gripper guard's clock does not wait.
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
    Path(bench.profile + ".yaml").write_text(yaml.safe_dump(profile), encoding="utf-8")
    actuator = types.ModuleType("reBotArm_control_py.actuator")
    actuator.RebotArm = bench.make
    package = types.ModuleType("reBotArm_control_py")
    package.actuator = actuator
    monkeypatch.setitem(sys.modules, "reBotArm_control_py", package)
    monkeypatch.setitem(sys.modules, "reBotArm_control_py.actuator", actuator)
    monkeypatch.setattr(sys, "path", list(sys.path))  # the driver adds its SDK path
    monkeypatch.setattr(rs_gripper, "time", GuardTime())
    monkeypatch.setattr(robot_main, "_stdin_is_terminal", lambda: True)
    return bench


def refused_before_the_bus(bench: Bench, capsys, monkeypatch) -> str:
    """Run the program to an exit code 2; what it wrote to the standard error."""
    monkeypatch.setattr(robot_main, "typed_lines", None)  # nothing may be asked
    monkeypatch.setattr(robot_main, "_remote_link_missing", lambda: None)
    assert robot_main.main(bench.argv("--connect", ENDPOINT)) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
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
        "the file is 'stereo-c50bf5b2b899+mount-41d3ce701628'"
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
@pytest.mark.parametrize("key", [None, "can4"])
def test_an_operators_confirmation_is_never_used_by_the_robot_app(
    bench, capsys, monkeypatch, key
):
    from apai.control import arm_hold

    bench.reading = list(RAISED)
    confirmation = arm_hold.confirm("holding the arm", key)
    err = refused_before_the_bus(bench, capsys, monkeypatch)
    assert err.startswith(
        f"{NOT_CONNECTED}an operator's confirmation for APAI's connect is on "
        "file for the arm on can4"
    )
    # It was not claimed: it is still on file for APAI's own tools.
    assert confirmation.is_file()
    assert sorted(path.name for path in bench.holds.iterdir()) == [confirmation.name]


@rig
def test_a_file_faster_than_the_profiles_cap_is_refused_before_any_question(
    bench, capsys, monkeypatch, tmp_path
):
    data = yaml.safe_load(RIG_YAML.read_text(encoding="utf-8"))
    data["limits"]["human"]["max_joint_speed"] = 0.2
    fast = tmp_path / "robot_app_fast.yaml"
    fast.write_text(yaml.safe_dump(data), encoding="utf-8")
    argv = bench.argv("--connect", ENDPOINT)
    argv[argv.index("--config") + 1] = str(fast)
    monkeypatch.setattr(bench, "argv", lambda *extra: argv)
    err = refused_before_the_bus(bench, capsys, monkeypatch)
    assert err == (
        f"{NOT_CONNECTED}APAI arm profile {bench.profile!r} cannot be used with "
        f"{fast}: limits.human.max_joint_speed 0.2 rad/s is above the APAI "
        "profile's speed cap of 0.1 rad/s (safety.max_joint_vel)\n"
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
        "stereo-c50bf5b2b899+mount-41d3ce701628; speed cap 0.1 rad/s; at rest at "
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

        # A typed rest brings the arm inside the 0.05 rad of a release.
        session.typed.type("rest")
        session.typed.type("path is clear")
        session.client.wait_state(lambda s: s.motion == "rest_hold", WAIT_S)
        assert sdk.disconnects == 0
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
    assert said[1] == (
        "robot app: the real driver is connected at measured rest: motors on, "
        "motion hold, jaws referenced"
    )
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
