# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The programs of the robot connection: the robot link service and the robot app.

The arguments, the real driver's confirmations and the robot app with a dialer
are tested without Isaac Capture. The tests marked "link" run the programs over
a real link on 127.0.0.1: in this process, and as processes of their own that
are ended with signals. Every process a test signals is one it started.
"""

from __future__ import annotations

import logging
import os
import signal
import socket
import sys
import threading
import time
import types
from collections.abc import Callable, Iterator
from typing import Any

import pytest
import yaml
from isaaccapture_examples.agentic_remote.link_service import __main__ as service_main
from isaaccapture_examples.agentic_remote.link_service.listen import (
    RobotListener,
    plain_peer,
)
from isaaccapture_examples.agentic_remote.link_service.local import (
    LocalListener,
    connect_local,
)
from isaaccapture_examples.agentic_remote.link_service.service import RobotLinkService
from isaaccapture_examples.agentic_remote.protocol.carrier import loopback_pair
from isaaccapture_examples.agentic_remote.protocol.link_carrier import LinkCarrier
from isaaccapture_examples.agentic_remote.robot_app import __main__ as robot_main
from isaaccapture_examples.agentic_remote.robot_app.clock import SystemClock
from isaaccapture_examples.agentic_remote.robot_app.config import RobotAppConfig
from isaaccapture_examples.agentic_remote.robot_app.interfaces import (
    MotionState,
    Owner,
)
from isaaccapture_examples.agentic_remote.robot_app.sim import SimArm
from isaaccapture_examples.agentic_remote.robot_app.terminal import TypedLines
from isaaccapture_examples.agentic_remote.testing.client import WorkstationClient
from repo_paths import repo_root

from link_stack import (
    CONNECT_S,
    HOST,
    PEER_TIMEOUT_S,
    WAIT_S,
    Program,
    ServiceProcess,
    Stack,
    until,
)
from test_main import quick_config

link = pytest.mark.link

CONFIGS = repo_root() / "examples" / "agentic_remote" / "configs"
ROBOT_YAML = str(CONFIGS / "robot_app.yaml")
SERVICE_YAML = str(CONFIGS / "link_service.yaml")
ROBOT_MODULE = "isaaccapture_examples.agentic_remote.robot_app"
BACKEND = "isaaccapture_examples.agentic_remote.robot_app.apai_backend"
LOGGER = "isaaccapture.examples.agentic_remote"
SIM = ["--config", ROBOT_YAML, "--arm", "sim"]
APAI = ["--config", ROBOT_YAML, "--arm", "apai"]
REAL = [*APAI, "--driver", "real", "--apai-profile", "rebot_rs"]
REST = [0.0] * 6
PATH = [[0.004 * (index + 1)] * 6 for index in range(25)]
P2_LINE = "hardware runs start in phase P2"
QUESTIONS = [
    "Real driver, 1 of 3: Is the arm powered? Type yes to confirm.",
    "Real driver, 2 of 3: Connecting cuts motor torque. Is the arm at rest and "
    "supported? Type yes to confirm.",
    "Real driver, 3 of 3: Is the calibration in use 'bench-7'? Type yes to confirm.",
]
# Threads of the programs; none may outlive one that ran in this process.
THREADS = (
    "robot-app-",
    "robot-dialer",
    "gripper-worker",
    "thor-terminal-",
    "link-service-",
)


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


@pytest.fixture
def no_isaac_capture(monkeypatch) -> None:
    """Isaac Capture cannot be imported."""
    for name in ("isaaccapture", "isaaccapture.remote_link"):
        monkeypatch.setitem(sys.modules, name, None)


@pytest.fixture
def folder(tmp_path_factory):
    return tmp_path_factory.mktemp("ls")  # a short path, mode 0700


def left_threads() -> list[str]:
    return [
        thread.name
        for thread in threading.enumerate()
        if thread.name.startswith(THREADS)
    ]


def service_yaml(folder, port: int = 0, **settings: Any) -> str:
    """A service configuration on 127.0.0.1 with its socket in the folder."""
    data = {
        "listen": {"address": HOST, "port": port},
        "allowed_peers": [HOST],
        "socket_path": str(folder / "s.sock"),
        **settings,
    }
    path = folder / "link_service.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return str(path)


class Running:
    """A call under way on a thread of its own; result() waits for what it returns."""

    def __init__(self, call: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
        self._returned: list[Any] = []
        self._thread = threading.Thread(
            target=lambda: self._returned.append(call(*args, **kwargs)), daemon=True
        )
        self._thread.start()

    def result(self) -> Any:
        self._thread.join(2 * WAIT_S)
        assert self._returned, "the call did not return"
        return self._returned[0]


# The robot app's arguments


@pytest.mark.parametrize(
    "connect", ["127.0.0.1", "127.0.0.1:", ":47610", "host:0", "host:65536", "h:4x"]
)
def test_a_connect_address_that_is_not_host_and_port_gives_exit_code_2(
    connect, capsys, monkeypatch
):
    monkeypatch.setattr(robot_main, "RobotApp", None)  # no app may be built
    assert robot_main.main([*SIM, "--connect", connect]) == 2
    err = capsys.readouterr().err
    assert f"argument --connect: {connect!r} is not HOST:PORT" in err


@pytest.mark.parametrize(
    ("argv", "problem"),
    [
        (
            [*SIM, "--selftest", "--selftest-link"],
            "--connect, --selftest and --selftest-link exclude each other",
        ),
        (
            [*SIM, "--connect", "127.0.0.1:47610", "--selftest-link"],
            "--connect, --selftest and --selftest-link exclude each other",
        ),
        (
            [*APAI, "--selftest-link"],
            "--selftest-link runs on the simulated arm only: use --arm sim",
        ),
        ([*SIM, "--driver", "mock"], "--driver can be used with --arm apai only"),
        (
            [*SIM, "--apai-profile", "mock", "--calibration", "bench-7"],
            "--apai-profile, --calibration can be used with --arm apai only",
        ),
        (
            [*APAI, "--calibration", "bench-7"],
            "--calibration can be used with --driver real only",
        ),
        (
            [*APAI, "--driver", "real", "--calibration", "bench-7"],
            "--driver real needs the arm's profile: --apai-profile NAME",
        ),
        (REAL, "--driver real needs the calibration's name: --calibration NAME"),
        (
            [*REAL, "--calibration", " "],
            "--driver real needs the calibration's name: --calibration NAME",
        ),
    ],
)
def test_arguments_that_do_not_go_together_give_exit_code_2(
    argv, problem, capsys, monkeypatch
):
    monkeypatch.setattr(robot_main, "RobotApp", None)
    monkeypatch.setitem(sys.modules, BACKEND, None)  # must not be imported
    assert robot_main.main(argv) == 2
    captured = capsys.readouterr()
    assert captured.err == f"robot app: {problem}\n"
    assert captured.out == ""


def test_the_connect_examples_name_the_services_default_port():
    shipped = yaml.safe_load((CONFIGS / "link_service.yaml").read_text("utf-8"))
    assert shipped["listen"]["port"] == 47610
    readme = (CONFIGS.parent / "README.md").read_text(encoding="utf-8")
    assert "--connect 127.0.0.1:47610" in readme


def test_connect_without_isaac_capture_says_what_is_missing(
    capsys, monkeypatch, no_isaac_capture
):
    monkeypatch.setattr(robot_main, "RobotApp", None)
    assert robot_main.main([*SIM, "--connect", "127.0.0.1:47610"]) == 2
    err = capsys.readouterr().err
    assert err.startswith(
        "robot app: --connect cannot run: Isaac Capture's remote link cannot be "
        "imported (isaaccapture.remote_link on the Python path): "
    )


def test_the_link_selftest_without_isaac_capture_is_skipped_with_exit_code_2(
    capsys, monkeypatch, no_isaac_capture
):
    monkeypatch.setattr(robot_main, "RobotApp", None)
    assert robot_main.main([*SIM, "--selftest-link"]) == 2
    out = capsys.readouterr().out.splitlines()
    assert len(out) == 1
    assert out[0].startswith(
        "selftest-link: skipped: Isaac Capture's remote link cannot be imported"
    )
    assert out[0].endswith("(exit code 2)")
    assert left_threads() == []


# The real driver's confirmations


def fake_backend(monkeypatch, kind: str = "rebot_rs") -> list[str]:
    """Stand in for the APAI backend module; the profile names it was asked for.

    It loads a profile of this arm type and can build nothing.
    """
    loaded: list[str] = []
    module = types.ModuleType(BACKEND)

    def load_apai_profile(name: str) -> Any:
        if name == "no_arm":
            raise ValueError("there is no arm profile no_arm")
        loaded.append(name)
        return types.SimpleNamespace(arm_cfg={"type": kind})

    def forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("the backend was built")

    module.load_apai_profile = load_apai_profile
    for name in ("config_for_profile", "build_apai_checkers", "build_apai_arm"):
        setattr(module, name, forbidden)
    monkeypatch.setitem(sys.modules, BACKEND, module)
    return loaded


def answer(monkeypatch, *typed: str) -> None:
    """What the operator types at the Thor terminal, then the input ends."""
    monkeypatch.setattr(robot_main, "typed_lines", lambda: iter(typed))


def real_driver(capsys, *extra: str) -> tuple[int, list[str], str]:
    code = robot_main.main([*REAL, "--calibration", "bench-7", *extra])
    captured = capsys.readouterr()
    return code, captured.out.splitlines(), captured.err


def test_three_typed_confirmations_then_the_real_driver_stops_before_connecting(
    capsys, monkeypatch
):
    monkeypatch.setattr(robot_main, "RobotApp", None)  # no app may be built
    monkeypatch.setattr(robot_main, "RobotDialer", None)
    loaded = fake_backend(monkeypatch)
    answer(monkeypatch, "yes", " Yes ", "YES")
    code, out, err = real_driver(capsys, "--connect", "127.0.0.1:47610")
    assert code == 2
    assert loaded == ["rebot_rs"]
    assert out == [
        "robot app: APAI arm profile rebot_rs, the real driver",
        *QUESTIONS,
        "robot app: confirmed three times; the arm was not connected",
        P2_LINE,
    ]
    assert err == ""
    assert left_threads() == []


@pytest.mark.parametrize("wrong", ["no", "", "y", "yes please", "yes yes", "ok"])
@pytest.mark.parametrize("at", [0, 1, 2])
def test_any_other_answer_ends_the_program_before_anything_is_connected(
    capsys, monkeypatch, at, wrong
):
    monkeypatch.setattr(robot_main, "RobotApp", None)
    fake_backend(monkeypatch)
    typed = ["yes", "yes", "yes"]
    typed[at] = wrong
    answer(monkeypatch, *typed)
    code, out, err = real_driver(capsys)
    assert code == 2
    # The questions after the one that was not confirmed are not asked.
    assert out[1:] == QUESTIONS[: at + 1]
    assert err == "robot app: not confirmed: nothing was connected\n"
    assert P2_LINE not in out


@pytest.mark.parametrize("answers", [0, 1, 2])
def test_an_input_that_ends_confirms_nothing(capsys, monkeypatch, answers):
    monkeypatch.setattr(robot_main, "RobotApp", None)
    fake_backend(monkeypatch)
    answer(monkeypatch, *["yes"] * answers)
    code, out, err = real_driver(capsys)
    assert code == 2
    assert out[1:] == QUESTIONS[: answers + 1]
    assert err == "robot app: not confirmed: nothing was connected\n"


def test_a_terminal_that_cannot_be_read_confirms_nothing(capsys, monkeypatch):
    fake_backend(monkeypatch)

    def gone() -> Iterator[str]:
        yield "yes"
        raise OSError("the terminal is gone")

    monkeypatch.setattr(robot_main, "typed_lines", gone)
    code, out, err = real_driver(capsys)
    assert code == 2
    assert out[1:] == QUESTIONS[:2]
    assert err == "robot app: not confirmed: nothing was connected\n"


def test_the_calibration_is_shown_as_one_printable_word(capsys):
    said: list[str] = []
    name = "bench\x07-7\n" + "x" * 100
    assert robot_main.confirm_real_driver(name, iter(["yes"] * 3), said.append)
    assert len(said) == 3
    shown = said[2].removeprefix("Real driver, 3 of 3: Is the calibration in use ")
    assert shown.startswith("'bench -7 xxx")
    assert shown.isprintable() and len(shown) < 100


def test_the_real_driver_is_not_asked_for_with_apais_software_arm(capsys, monkeypatch):
    fake_backend(monkeypatch, kind="mock")
    monkeypatch.setattr(robot_main, "typed_lines", None)  # nothing may be asked
    code, out, err = real_driver(capsys)
    assert code == 2
    assert out == []
    assert err == (
        "robot app: APAI arm profile 'rebot_rs' is APAI's software arm: use "
        "--driver mock\n"
    )


def test_a_real_driver_profile_that_cannot_be_loaded_asks_nothing(capsys, monkeypatch):
    fake_backend(monkeypatch)
    monkeypatch.setattr(robot_main, "typed_lines", None)
    argv = [*APAI, "--driver", "real", "--apai-profile", "no_arm"]
    assert robot_main.main([*argv, "--calibration", "bench-7"]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == (
        "robot app: APAI arm profile 'no_arm' cannot be used: there is no arm "
        "profile no_arm\n"
    )


def test_a_hardware_profile_is_not_run_as_the_mock_driver(capsys, monkeypatch):
    monkeypatch.setattr(robot_main, "RobotApp", None)
    monkeypatch.setattr(robot_main, "_remote_link_missing", lambda: None)
    fake_backend(monkeypatch)  # building anything fails the test
    argv = [*APAI, "--apai-profile", "rebot_rs", "--connect", "127.0.0.1:47610"]
    for driver in ([], ["--driver", "mock"]):
        assert robot_main.main([*argv, *driver]) == 2
        captured = capsys.readouterr()
        assert captured.err == (
            "robot app: APAI arm profile 'rebot_rs' drives hardware: it needs "
            "--driver real\n"
        )
        assert captured.out == ""


@pytest.mark.apai
def test_the_real_driver_of_an_apai_profile_is_confirmed_and_never_connected(
    capsys, monkeypatch
):
    from isaaccapture_examples.agentic_remote.robot_app import apai_backend

    def forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("the arm was built or connected")

    monkeypatch.setattr(sys, "path", list(sys.path))  # the driver adds its SDK path
    monkeypatch.setattr(apai_backend, "build_apai_arm", forbidden)
    monkeypatch.setattr(apai_backend.ApaiArm, "connect", forbidden)
    answer(monkeypatch, "yes", "yes", "yes")
    code, out, err = real_driver(capsys)
    assert (code, err) == (2, "")
    assert out[1:4] == QUESTIONS
    assert out[-1] == P2_LINE
    # The mock profile is no real driver.
    argv = [*APAI, "--driver", "real", "--apai-profile", "mock"]
    assert robot_main.main([*argv, "--calibration", "bench-7"]) == 2
    assert "is APAI's software arm: use --driver mock" in capsys.readouterr().err


# Lines typed at a file descriptor


def test_typed_lines_are_read_from_a_descriptor_until_it_ends():
    read_end, write_end = os.pipe()
    try:
        lines = robot_main._FdLines(read_end)
        os.write(write_end, b"stop\nre")
        assert next(lines) == "stop"
        os.write(write_end, b"st\n\nlast")
        assert [next(lines), next(lines)] == ["rest", ""]
        os.close(write_end)
        write_end = -1
        assert next(lines) == "last"  # the input ended without a line end
        with pytest.raises(StopIteration):
            next(lines)
    finally:
        os.close(read_end)
        if write_end >= 0:
            os.close(write_end)


def test_closing_typed_lines_ends_a_wait_for_the_next_line():
    read_end, write_end = os.pipe()
    try:
        lines = robot_main._FdLines(read_end)
        waiting = Running(lambda: next(lines, None))
        time.sleep(0.05)
        lines.close()
        assert waiting.result() is None
    finally:
        os.close(read_end)
        os.close(write_end)


def test_without_a_descriptor_the_standard_input_itself_is_read(monkeypatch):
    monkeypatch.setattr(sys, "stdin", iter(["stop\n"]))  # it has no fileno()
    assert list(robot_main.typed_lines()) == ["stop\n"]


# The robot app with a dialer, on loopback links


class FakeLinks:
    """Stands in for LinkCarrier in the robot app's program: loopback links.

    connect() fails while `error` is set; else it gives the robot end of a new
    loopback link, whose other end is in `workstation_ends`.
    """

    def __init__(self) -> None:
        self.error: str | None = "cannot connect to 127.0.0.1:47610: refused"
        self.dialed: list[tuple[str, float, str]] = []
        self.workstation_ends: list[Any] = []

    def connect(self, endpoint: str, timeout_s: float, *, name: str) -> Any:
        self.dialed.append((endpoint, timeout_s, name))
        if self.error is not None:
            raise RuntimeError(self.error)
        robot_end, workstation_end = loopback_pair()
        self.workstation_ends.append(workstation_end)
        return robot_end


class ConnectedRun:
    """run_connected() on the simulated arm, on a thread; the test types its lines."""

    def __init__(self, monkeypatch, tmp_path, changes: dict | None = None) -> None:
        self.links = FakeLinks()
        monkeypatch.setattr(robot_main, "LinkCarrier", self.links)
        changes = {"connection.retry_s": 0.2, **(changes or {})}
        self.config = robot_main.load_config(quick_config(tmp_path, changes))
        self.clock = SystemClock()
        self.arm = SimArm(
            n_joints=6, clock=self.clock, q0=list(self.config.rest.rest_q)
        )
        self.typed = TypedLines()
        self._said = threading.Condition()
        self.lines: list[str] = []
        self.running = Running(
            robot_main.run_connected,
            self.config,
            self.arm,
            robot_main.sim_checkers(self.config, self.clock),
            self.clock,
            "127.0.0.1:47610",
            typed=self.typed,
            say=self._say,
        )

    def _say(self, line: str) -> None:
        with self._said:
            self.lines.append(line)
            self._said.notify_all()

    def wait_line(self, part: str, count: int = 1) -> str:
        def found() -> list[str]:
            return [line for line in self.lines if part in line]

        with self._said:
            assert self._said.wait_for(lambda: len(found()) >= count, WAIT_S), (
                f"no line number {count} with {part!r}; said: {self.lines}"
            )
            return found()[count - 1]

    def client(self) -> WorkstationClient:
        """A workstation client on the newest link the program dialed."""
        until(lambda: self.links.workstation_ends, "a robot connection")
        return WorkstationClient(self.links.workstation_ends[-1], self.clock)


def test_the_program_dials_serves_the_thor_terminal_and_ends_with_a_typed_quit(
    monkeypatch, tmp_path
):
    run = ConnectedRun(monkeypatch, tmp_path)
    run.wait_line("dialing the robot link service at 127.0.0.1:47610")
    # No service listens: said once, however often the dial fails.
    failed = run.wait_line("no robot connection")
    assert failed == (
        "robot app: no robot connection: cannot connect to 127.0.0.1:47610: "
        "refused; dialing again every 0.2 s"
    )
    until(lambda: len(run.links.dialed) >= 3, "three dials")
    assert run.links.dialed[0] == ("127.0.0.1:47610", 2.0, "robot-app")
    # The Thor terminal is served with no robot connection.
    run.typed.type("stop")
    run.wait_line("[stop] STOP from thor_terminal: typed stop; the arm holds")

    run.links.error = None
    client = run.client()
    try:
        welcome = client.wait_welcome(WAIT_S)
        run.wait_line("[connection] ")
        state = client.wait_state(lambda s: s.stop_number == 1, WAIT_S)
        assert (state.session_id, state.torque_on) == (welcome.session_id, True)
        run.typed.type("quit")
        assert run.running.result() == 0
    finally:
        client.close()
    run.wait_line("quit: completed; the robot app closes")
    assert len([line for line in run.lines if "no robot connection" in line]) == 1
    # One release, at rest; the link the dialer attached is closed with it.
    assert len(run.arm.released_at) == 1
    assert not run.links.workstation_ends[-1].alive
    assert left_threads() == []


def test_the_program_runs_on_when_the_thor_terminals_input_ends(monkeypatch, tmp_path):
    run = ConnectedRun(monkeypatch, tmp_path)
    run.links.error = None
    client = run.client()
    try:
        client.wait_welcome(WAIT_S)
        run.typed.end()
        run.wait_line("the Thor terminal's input ended; the robot app runs on")
        # It still serves the robot connection: a SHUTDOWN at rest ends it.
        assert client.start_task(timeout_s=WAIT_S).status == "completed"
        gave = client.request_owner("nobody")
        assert client.wait_result(gave, timeout_s=WAIT_S).status == "completed"
        client.wait_state(lambda s: s.owner == "nobody" and s.owner_number == 2)
        down = client.shutdown()
        assert client.wait_result(down, timeout_s=WAIT_S).status == "completed"
        assert run.running.result() == 0
    finally:
        client.close()
    assert len(run.arm.released_at) == 1
    assert left_threads() == []


def test_a_lost_robot_connection_is_dialed_again_and_the_arm_keeps_holding(
    monkeypatch, tmp_path
):
    run = ConnectedRun(monkeypatch, tmp_path)
    run.links.error = None
    client = run.client()
    try:
        client.wait_welcome(WAIT_S)
        assert client.start_task(timeout_s=WAIT_S).status == "completed"
        run.links.workstation_ends[0].break_link()
        run.wait_line("[connection] ", 2)  # made, then lost
        until(lambda: len(run.links.workstation_ends) == 2, "the second dial")
        again = run.client()
        try:
            state = again.wait_state(lambda s: s.owner_number == 2, WAIT_S)
            assert (state.owner, state.motion) == ("nobody", "rest_hold")
            assert state.torque_on and run.arm.sent == []
        finally:
            again.close()
        run.typed.type("quit")
        assert run.running.result() == 0
    finally:
        client.close()
    assert left_threads() == []


def test_a_robot_app_that_does_not_start_gives_exit_code_1(monkeypatch, capsys):
    monkeypatch.setattr(robot_main, "LinkCarrier", FakeLinks())
    config, clock = RobotAppConfig.default(), SystemClock()
    arm = SimArm(n_joints=6, clock=clock, q0=list(config.rest.rest_q))

    def no_bus() -> None:
        raise RuntimeError("the CAN bus is down")

    arm.connect = no_bus
    checkers = robot_main.sim_checkers(config, clock)
    said: list[str] = []
    code = robot_main.run_connected(
        config, arm, checkers, clock, "127.0.0.1:47610", typed=iter(()), say=said.append
    )
    assert code == 1
    assert capsys.readouterr().err == (
        "robot app: the robot app did not start: the CAN bus is down\n"
    )
    assert said == []
    assert left_threads() == []


# The robot link service's command line


@pytest.mark.parametrize(
    "argv",
    [[], ["--config"], ["--config", SERVICE_YAML, "--log-level", "loud"], ["x"]],
)
def test_service_arguments_that_cannot_be_used_give_exit_code_2(argv, capsys):
    assert service_main.main(argv) == 2
    err = capsys.readouterr().err
    assert err.startswith(
        "usage: python -m isaaccapture_examples.agentic_remote.link_service"
    )


def test_the_services_help_names_its_arguments(capsys):
    assert service_main.main(["--help"]) == 0
    out = capsys.readouterr().out
    assert "--config PATH" in out and "--log-level L" in out


def test_a_service_configuration_that_cannot_be_used_gives_exit_code_2(folder, capsys):
    assert service_main.main(["--config", str(folder / "nowhere.yaml")]) == 2
    err = capsys.readouterr().err
    assert err.startswith("robot link service: the configuration cannot be used: ")
    assert "nowhere.yaml" in err
    bad = service_yaml(folder, allowed_peers=[])
    assert service_main.main(["--config", bad]) == 2
    assert "allowed_peers must be a non-empty list" in capsys.readouterr().err
    assert not (folder / "s.sock").exists()


def test_the_service_without_isaac_capture_says_what_is_missing(
    folder, capsys, no_isaac_capture
):
    code = service_main.main(["--config", service_yaml(folder)], wait=lambda _: "")
    assert code == 2
    captured = capsys.readouterr()
    assert captured.err.startswith(
        "robot link service: Isaac Capture's remote link cannot be imported "
        "(isaaccapture.remote_link on the Python path): "
    )
    assert captured.out == ""
    # Nothing is left behind: the local socket was closed again.
    assert not (folder / "s.sock").exists()
    assert left_threads() == []


@link
def test_a_second_service_on_a_live_socket_exits_with_a_clear_line(folder, capsys):
    config = service_yaml(folder)
    live = LocalListener(str(folder / "s.sock"))
    try:
        assert service_main.main(["--config", config], wait=lambda _: "") == 1
    finally:
        live.close()
    captured = capsys.readouterr()
    assert captured.err == (
        f"robot link service: a robot link service already runs on {folder}/s.sock\n"
    )
    assert captured.out == ""
    assert left_threads() == []


def test_signals_are_waited_for_on_the_main_thread_only(folder, capsys):
    code = Running(service_main.main, ["--config", service_yaml(folder)]).result()
    assert code == 2
    assert "on the main thread only" in capsys.readouterr().err
    assert not (folder / "s.sock").exists()


@pytest.mark.parametrize(
    ("peer", "plain"),
    [
        ("127.0.0.1:5000", "127.0.0.1:5000"),
        ("[::ffff:10.0.0.2]:5000", "10.0.0.2:5000"),
        ("[::1]:5000", "[::1]:5000"),
        ("[2001:db8::2]:5000", "[2001:db8::2]:5000"),
        ("[::ffff:0:1]:5000", "[::ffff:0:1]:5000"),
        ("", ""),
    ],
)
def test_an_ipv4_peer_of_an_ipv6_listener_is_written_as_ipv4(peer, plain):
    assert plain_peer(peer) == plain


# The service's program, in this process, over a real link


class Hold:
    """The `wait` of the service's main(): holds it open until the test ends it."""

    def __init__(self) -> None:
        self.started = threading.Event()
        self.end = threading.Event()
        self.program: Any = None

    def __call__(self, program: Any) -> str:
        self.program = program
        self.started.set()
        self.end.wait(3 * WAIT_S)
        return "the test ended it"


@link
def test_the_service_program_takes_the_robot_connection_and_closes_cleanly(
    folder, capsys
):
    hold = Hold()
    running = Running(service_main.main, ["--config", service_yaml(folder)], wait=hold)
    assert hold.started.wait(WAIT_S)
    port = hold.program.port
    assert port > 0
    path = str(folder / "s.sock")
    assert os.stat(path).st_mode & 0o777 == 0o600
    tool = connect_local(path, "tool", "test", WAIT_S)
    robot = None
    try:
        assert not tool.status.robot_connected
        robot = LinkCarrier.connect(f"{HOST}:{port}", CONNECT_S, name="robot-app")
        tool.wait_status(lambda s: s.robot_connected, WAIT_S)
        hold.end.set()
        assert running.result() == 0
        # Closing the service ends its connections; nothing is sent to the robot.
        assert tool.wait_dead(WAIT_S)
        until(lambda: not robot.alive, "the end of the robot's link")
        assert robot.receive(0) is None
    finally:
        tool.close()
        if robot is not None:
            robot.close()
    out = capsys.readouterr().out.splitlines()
    assert out == [
        f"robot link service: waits for the robot app on {HOST}:{port} (taken from "
        f"{HOST}); local API at {path}",
        "robot link service: the test ended it; closing. The robot holds on its own.",
    ]
    assert not os.path.exists(path)
    assert left_threads() == []


@link
def test_an_address_that_cannot_be_listened_on_gives_exit_code_1(folder, capsys):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as taken:
        taken.bind((HOST, 0))
        taken.listen(1)
        config = service_yaml(folder, port=taken.getsockname()[1])
        assert service_main.main(["--config", config], wait=lambda _: "") == 1
    err = capsys.readouterr().err
    assert err.startswith(f"robot link service: cannot listen on {HOST}:")
    assert not (folder / "s.sock").exists()
    assert left_threads() == []


# The link self-test


@link
def test_the_link_selftest_passes_on_the_shipped_configuration(capsys, folder):
    released: list[list[float]] = []
    real = SimArm.release_torque

    def release_torque(self: SimArm) -> None:
        released.append(self.pose())
        real(self)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(SimArm, "release_torque", release_torque)
        code = robot_main.main([*SIM, "--selftest-link"])
    lines = capsys.readouterr().out.splitlines()
    assert code == 0, "\n".join(lines)
    steps = [line for line in lines if line.startswith("selftest-link ")]
    titles = [line.split(" ok  ")[1].split(":")[0] for line in steps]
    assert titles == [
        "start the robot link service",
        "start the robot app; it dials the service",
        "take ownership for the agent",
        "a move with permits",
        "a move without a permit",
        "STOP from the console during a move",
        "a new task after the STOP, no reset",
        "the robot connection is cut during a move",
        "the robot app dials again",
        "rest from the console",
        "release from the console",
        "close the robot app",
    ]
    assert "robot connection up after 1 dial(s)" in steps[1]
    assert "failed: robot connection lost" in steps[7]
    assert "after 2 dial(s) in all, the same session; nothing moved" in steps[8]
    assert lines[-1].startswith("selftest-link: all 12 steps passed in ")
    rest = RobotAppConfig.default().rest
    assert len(released) == 1
    assert released[0] == pytest.approx(list(rest.rest_q), abs=rest.rest_tol_rad)
    assert left_threads() == []


@link
def test_a_failed_step_of_the_link_selftest_gives_exit_code_1_and_closes_everything(
    tmp_path, folder
):
    config = robot_main.load_config(
        quick_config(tmp_path, {"permits.required_for_agent": False})
    )
    said: list[str] = []
    assert robot_main.run_link_selftest(config, said.append, str(folder)) == 1
    assert said[-1] == "selftest-link: FAILED at step 5 of 12"
    assert "FAILED  a move without a permit" in said[-2]
    assert not (folder / "robot_link.sock").exists()
    assert left_threads() == []


# The programs as processes of their own


@pytest.fixture
def programs() -> Iterator[list[Program]]:
    """Processes a test started; each is ended after the test."""
    started: list[Program] = []
    yield started
    for program in started:
        program.close()


def free_port() -> int:
    """A TCP port on 127.0.0.1 that nothing listens on now."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind((HOST, 0))
        return probe.getsockname()[1]


@link
@pytest.mark.parametrize("number", [signal.SIGTERM, signal.SIGINT])
def test_the_service_process_closes_cleanly_on_a_signal(folder, programs, number):
    path = str(folder / "s.sock")
    service = ServiceProcess(folder, path)
    programs.append(service)
    port = service.wait_ready()
    tool = connect_local(path, "tool", "test", WAIT_S)
    try:
        # A second service on the same socket ends at once and harms nothing.
        second = ServiceProcess(folder, path, free_port())
        programs.append(second)
        assert second.wait_exit() == 1
        assert second.lines == [
            f"robot link service: a robot link service already runs on {path}"
        ]
        assert tool.alive
        service.signal(number)
        assert service.wait_exit() == 0
        assert tool.wait_dead(WAIT_S)
    finally:
        tool.close()
    ready = f"robot link service: waits for the robot app on {HOST}:{port} "
    assert any(line.startswith(ready) for line in service.lines)
    assert (
        f"robot link service: {signal.Signals(number).name}; closing. The robot "
        "holds on its own."
    ) in service.lines
    assert not os.path.exists(path)


@link
def test_the_robot_app_process_stops_on_ctrl_c_and_closes_on_sigterm(folder, programs):
    path = str(folder / "s.sock")
    service = ServiceProcess(folder, path)
    programs.append(service)
    port = service.wait_ready()
    app = Program(ROBOT_MODULE, *SIM, "--connect", f"{HOST}:{port}")
    programs.append(app)
    app.wait_line(f"dialing the robot link service at {HOST}:{port}")
    console = connect_local(path, "console", "test", WAIT_S)
    client = WorkstationClient(console, SystemClock())
    try:
        console.wait_status(lambda s: s.robot_connected, WAIT_S)
        client.wait_state(lambda s: s.motion == "rest_hold" and s.torque_on, WAIT_S)
        # Ctrl+C is a STOP and never ends the robot app.
        app.signal(signal.SIGINT)
        app.wait_line("[stop] STOP from thor_terminal: Ctrl+C; the arm holds")
        state = client.wait_state(lambda s: s.stop_number == 1, WAIT_S)
        assert state.torque_on
        assert app.process.poll() is None
        app.signal(signal.SIGTERM)
        assert app.wait_exit() == 0
        assert "robot app: SIGTERM; closing with the motors left on" in app.lines
        status = console.wait_status(lambda s: not s.robot_connected, WAIT_S)
        assert status.close_reason
    finally:
        client.close()
        console.close()
    service.signal(signal.SIGTERM)
    assert service.wait_exit() == 0


@link
def test_the_robot_app_process_dials_until_a_service_listens_and_quits_when_told(
    folder, programs
):
    path = str(folder / "s.sock")
    port = free_port()
    app = Program(ROBOT_MODULE, *SIM, "--connect", f"{HOST}:{port}")
    programs.append(app)
    app.wait_line(f"robot app: no robot connection: cannot connect to {HOST}:{port}")
    service = ServiceProcess(folder, path, port)
    programs.append(service)
    assert service.wait_ready() == port
    app.wait_line("[connection] ")
    tool = connect_local(path, "tool", "test", WAIT_S)
    try:
        tool.wait_status(lambda s: s.robot_connected, WAIT_S)
        app.type("quit")
        assert app.wait_exit() == 0
        tool.wait_status(lambda s: not s.robot_connected, WAIT_S)
    finally:
        tool.close()
    assert "quit: completed; the robot app closes" in app.lines
    said = [line for line in app.lines if "no robot connection" in line]
    assert len(said) == 1


@pytest.fixture
def process_stack(make_robot_app, folder, programs) -> Iterator[Stack]:
    """A robot app on the simulated arm that dials a service process."""
    stack = Stack(make_robot_app(connected=False), folder)
    service = ServiceProcess(folder, stack.path)
    programs.append(service)
    stack.port = service.wait_ready()
    stack.dial()
    stack.agent()
    yield stack
    stack.close()


@link
def test_a_service_process_killed_and_started_again_moves_nothing(
    process_stack, folder, programs
):
    stack, h = process_stack, process_stack.h
    agent, service = stack.stations[0], programs[0]
    h.new_task()
    moved = agent.ws.run_move(PATH, WAIT_S, start_q=REST)
    assert moved.status == "completed", moved.reason

    stack.held = True
    events = stack.connection_events()
    service.signal(signal.SIGKILL)
    assert service.wait_exit() == -signal.SIGKILL
    # The Thor learns at once that the process is gone, and holds.
    h.wait_local_events("connection", events + 1)
    assert agent.end.wait_dead(WAIT_S)
    h.idle()
    now = h.app.snapshot()
    assert (now.owner, now.owner_number) == (Owner.NOBODY, 2)
    assert h.app.motion_state is MotionState.HOLD
    assert h.arm.torque_on() and h.arm.released_at == []
    assert os.path.exists(stack.path)  # the dead service's socket file

    again = ServiceProcess(folder, stack.path, stack.port)
    programs.append(again)
    assert again.wait_ready() == stack.port  # it replaced the socket file
    stack.held = False
    agent = stack.agent()
    state = agent.ws.wait_state(lambda s: s.owner_number == 2)
    assert (state.owner, state.motion, state.torque_on) == ("nobody", "hold", True)
    h.step()
    h.idle()
    assert h.sent() == PATH
    h.new_task()
    onward = [[q + 0.004 * (index + 1) for q in PATH[-1]] for index in range(10)]
    more = agent.ws.run_move(onward, WAIT_S, start_q=PATH[-1])
    assert more.status == "completed", more.reason
    again.signal(signal.SIGTERM)
    assert again.wait_exit() == 0
    assert not os.path.exists(stack.path)


@link
def test_a_service_process_that_hangs_is_noticed_within_the_peer_timeout(
    process_stack, programs
):
    stack, h = process_stack, process_stack.h
    agent, service = stack.stations[0], programs[0]
    h.new_task()
    reached, release = h.hold_move_after(6)
    move_id = agent.ws.move(PATH, start_q=REST)
    agent.ws.permit(move_id, len(PATH) - 1)
    assert reached.wait(WAIT_S)

    stack.held = True
    events = stack.connection_events()
    service.signal(signal.SIGSTOP)  # its sockets stay open; it sends nothing
    began = time.monotonic()
    until(
        lambda: stack.connection_events() > events,
        "the loss of the silent link",
        PEER_TIMEOUT_S + 3.0,
    )
    took = time.monotonic() - began
    assert PEER_TIMEOUT_S - 1.0 < took < PEER_TIMEOUT_S + 1.5
    # The move ends in a hold where it was; nobody owns the arm.
    release.set()
    h.idle()
    assert h.sent() == PATH[:6]
    assert h.app.motion_state is MotionState.HOLD
    now = h.app.snapshot()
    assert (now.owner, now.owner_number) == (Owner.NOBODY, 2)
    assert h.arm.torque_on() and h.arm.released_at == []


# On APAI's mock arm


@link
@pytest.mark.apai
def test_the_robot_app_on_apais_mock_driver_dials_the_service(
    folder, tmp_path, capsys, monkeypatch
):
    text = (CONFIGS / "robot_app.yaml").read_text(encoding="utf-8")
    apai_yaml = tmp_path / "robot_app_apai.yaml"
    apai_yaml.write_text(
        text.replace("max_joint_speed: 1.5", "max_joint_speed: 0.8"), encoding="utf-8"
    )
    monkeypatch.setattr(robot_main, "typed_lines", lambda: iter(()))  # a closed input
    listener = RobotListener(HOST, 0)
    path = str(folder / "s.sock")
    service = RobotLinkService(listener.accept, socket_path=path)
    service.start()
    console = None
    client = None
    try:
        argv = ["--config", str(apai_yaml), "--arm", "apai", "--driver", "mock"]
        running = Running(robot_main.main, [*argv, "--connect", listener.endpoint])
        console = connect_local(path, "console", "test", WAIT_S)
        client = WorkstationClient(console, SystemClock())
        console.wait_status(lambda s: s.robot_connected, WAIT_S)
        welcome = client.wait_welcome(WAIT_S)
        assert len(welcome.joint_names) == 6
        state = client.wait_state(lambda s: s.motion == "rest_hold", WAIT_S)
        assert state.torque_on
        # A SHUTDOWN at rest switches the motors off there and ends the program.
        down = client.shutdown()
        assert client.wait_result(down, timeout_s=WAIT_S).status == "completed"
        assert running.result() == 0
    finally:
        if client is not None:
            client.close()
        if console is not None:
            console.close()
        service.close()
        listener.close()
    out = capsys.readouterr().out
    assert f"dialing the robot link service at {listener.endpoint}" in out
    assert left_threads() == []
