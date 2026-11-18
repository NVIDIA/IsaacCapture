# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The programs of the robot connection: the robot link service and the robot app.

The arguments and the robot app with a dialer are tested without Isaac Capture;
the real driver's start is tested in test_real_driver.py. The tests marked
"link" run the programs over a real link on 127.0.0.1: in this process, and as
processes of their own that are ended with signals. Every process a test
signals is one it started.
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
from isaaccapture_examples.agentic_remote.robot_app import rest as rest_module
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
# What the robot app's process says of a SIGTERM: with the arm at measured
# rest, where it switches the motors off, and anywhere else.
SIGTERM_AT_REST = (
    "robot app: SIGTERM; closing; the arm lies at measured rest: the motors are off"
)
SIGTERM_MOTORS_ON = "robot app: SIGTERM; closing with the motors left on"
SIGTERM_WHY = "robot app: the motors stay on because "
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
        (
            [*REAL, "--calibration", "bench-7"],
            "--driver real needs the robot link service's address: --connect HOST:PORT",
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


# A hardware profile without --driver real


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
        "close the robot app",
    ]
    assert "robot connection up after 1 dial(s)" in steps[1]
    assert "failed: robot connection lost" in steps[7]
    assert "after 2 dial(s) in all, the same session; nothing moved" in steps[8]
    # The console's rest ends the session at rest: it switches the motors off.
    assert steps[9].endswith(" rad from rest")
    assert ": the motors are off, released 0.0" in steps[9]
    assert lines[-1].startswith("selftest-link: all 11 steps passed in ")
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
    assert said[-1] == "selftest-link: FAILED at step 5 of 11"
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
        # SIGTERM with the arm at measured rest: the jaws close, the motors go
        # off there, and the program ends.
        app.signal(signal.SIGTERM)
        assert app.wait_exit() == 0
        status = console.wait_status(lambda s: not s.robot_connected, WAIT_S)
        assert status.close_reason
    finally:
        client.close()
        console.close()
    said = app.lines
    closed = said.index(f"[{rest_module.EVENT_JAWS}] {rest_module.JAWS_CLOSED}")
    released = said.index(f"[{rest_module.EVENT_RELEASE}] {rest_module.RELEASED}")
    ended = said.index(SIGTERM_AT_REST)
    # The jaws close before the motors go off. The program's own line and the
    # release's line come from two threads: either can be written first.
    assert closed < released and closed < ended
    assert not [line for line in said if "motors left on" in line]
    service.signal(signal.SIGTERM)
    assert service.wait_exit() == 0


@link
def test_sigterm_away_from_rest_ends_the_robot_app_with_the_motors_left_on(
    folder, programs
):
    path = str(folder / "s.sock")
    service = ServiceProcess(folder, path)
    programs.append(service)
    port = service.wait_ready()
    app = Program(ROBOT_MODULE, *SIM, "--connect", f"{HOST}:{port}")
    programs.append(app)
    app.wait_line(f"dialing the robot link service at {HOST}:{port}")
    agent = connect_local(path, "agent", "test", WAIT_S)
    client = WorkstationClient(agent, SystemClock())
    try:
        agent.wait_status(lambda s: s.robot_connected, WAIT_S)
        client.wait_state(lambda s: s.motion == "rest_hold" and s.torque_on, WAIT_S)
        assert client.start_task(timeout_s=WAIT_S).status == "completed"
        moved = client.run_move(PATH, WAIT_S, start_q=REST)
        assert moved.status == "completed", moved.reason
        client.wait_state(lambda s: s.motion == "hold" and s.torque_on, WAIT_S)
        app.signal(signal.SIGTERM)
        assert app.wait_exit() == 0
    finally:
        client.close()
        agent.close()
    said = app.lines
    first = said.index(SIGTERM_MOTORS_ON)
    # The Thor terminal writes on a thread of its own: a line of it can come
    # between the program's two.
    [why] = [line for line in said[first + 1 :] if line.startswith(SIGTERM_WHY)]
    assert "rest" in why  # the arm is not measured at rest
    # Nothing was sent to the jaws or the motors on the way out.
    for kind in (rest_module.EVENT_JAWS, rest_module.EVENT_RELEASE):
        assert not [line for line in said if line.startswith(f"[{kind}] ")], kind
    assert SIGTERM_AT_REST not in said
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


# The README

README = CONFIGS.parent / "README.md"
TESTS = repo_root() / "tests" / "python" / "examples" / "agentic_remote"
# The heading the operator console and the agent program's task gate send the
# operator to when a takeover, a person's ownership of the arm, cannot be
# ended with the controller.
NO_RIGHT_A = "A takeover that cannot be ended with right A"


def readme_section(heading: str) -> str:
    """The README from this heading line to the next heading of its level."""
    readme = README.read_text(encoding="utf-8")
    start = readme.index(f"\n{heading}\n")
    level = heading.split()[0]
    return readme[start : readme.index(f"\n{level} ", start + 1)]


def config_files_named(text: str) -> set[str]:
    """The names of the example's config files that a text mentions."""
    import re

    programs = r"robot_app|agent|link_service|view|human|camera_(?:sender|receiver)"
    return set(re.findall(rf"\b(?:{programs})(?:_\w+)?\.yaml\b", text))


def test_the_readme_has_the_section_the_console_and_the_task_gate_name():
    from isaaccapture_examples.agentic_remote.agent import task_gate
    from isaaccapture_examples.agentic_remote.console import status

    assert f'README, "{NO_RIGHT_A}"' in status.OPEN_NO_CONTROLLER
    remedies = [remedy for _, remedy in task_gate._REMEDIES]
    assert any(f'README, "{NO_RIGHT_A}"' in remedy for remedy in remedies)
    assert f"\n#### {NO_RIGHT_A}\n" in README.read_text(encoding="utf-8")


def test_the_readme_lists_every_workstation_start_script_with_the_programs():
    where = readme_section("### What runs where")
    table = [line for line in where.splitlines() if line.startswith("|")]
    gpu = where[where.index("use one GPU") :]
    scripts = sorted(CONFIGS.parent.glob("run_*.sh"))
    assert len(scripts) >= 6
    for script in scripts:
        if script.name == "run_robot_app.sh":  # the Thor's; under its own heading
            continue
        assert any(script.name in row for row in table), script.name
        if "CUDA_VISIBLE_DEVICES=0" in script.read_text(encoding="utf-8"):
            assert f"`{script.name}`" in gpu, script.name


def test_the_readme_names_every_test_file_of_the_human_path():
    tests = readme_section("## Run the tests")
    names = sorted(path.name for path in TESTS.glob("test_human_*.py"))
    assert len(names) >= 6
    for name in [*names, "test_takeover.py", "test_demo2_programs.py"]:
        assert (TESTS / name).is_file(), name
        assert f"`{name}`" in tests, name


def test_the_readme_names_only_config_files_that_are_there():
    named = config_files_named(README.read_text(encoding="utf-8"))
    assert {"robot_app_demo2.yaml", "robot_app_human_check.yaml"} <= named
    assert sorted(name for name in named if not (CONFIGS / name).is_file()) == []
    # The scan finds a file that is not there, however it is written.
    assert config_files_named("see `configs/robot_app_demo3.yaml`, (view_x.yaml)") == {
        "robot_app_demo3.yaml",
        "view_x.yaml",
    }


# The README on the human path. The human joint box is limits.human.joint_min to
# joint_max of the robot app's file: where a person may drive the arm. The
# recovery region is rest.region_min to region_max: the start poses from which
# a return to rest is taken. The rig file is the robot app's file for the rig,
# the real arm.

RIG_ROBOT_YAML = CONFIGS / "robot_app_rebot.yaml"
# The two tests that run the recipes of "Run demo 2 on one machine" as processes.
RECIPE_TESTS = (
    "test_the_recipe_of_demo_2_runs_as_processes_with_the_shipped_files",
    "test_demo_2_with_the_agent_failing_first_runs_as_processes",
)
# The refusal of a takeover from a pose outside the human joint box, as quoted.
NO_TAKEOVER = "`the arm is outside the human limits: ...; type rest`"


def one_line(text: str) -> str:
    """A text with every line break and its indent as one blank."""
    return " ".join(text.split())


def readme_row(start: str) -> str:
    """The one table row of the README that begins with this text."""
    lines = README.read_text(encoding="utf-8").splitlines()
    [found] = [line for line in lines if line.startswith(start)]
    return found


def readme_item(start: str) -> str:
    """The list item that begins with this text, on one line, up to the next."""
    text = one_line(README.read_text(encoding="utf-8"))
    at = text.index(start)
    return text[at : text.index(" - **", at + 1)]


def test_the_readme_says_what_the_rig_margin_does_for_a_typed_rest():
    # The row of the typed rest's refusal for a start pose outside the region.
    cell = readme_row("| `joint index N at X rad is outside the tested recovery region")
    assert "In the rig file the box lies 0.02 rad inside the region" in cell
    assert "this refusal does not come" in cell
    # What a box that equals its region leaves after a STOP at its edge.
    assert "That needs the person as control owner" in cell
    assert "a new takeover is refused too" in cell and NO_TAKEOVER in cell
    assert "no command moves the arm then" in cell
    assert "[Known gaps](#known-gaps)" in cell


def test_the_readme_rig_files_say_what_the_margin_is_for_and_what_it_costs():
    rig = one_line(readme_section("## The rig files"))
    assert "A STOP at such a pose leaves no squeeze" in rig
    assert "a new takeover is refused there too" in rig and NO_TAKEOVER in rig
    assert "so the typed `rest` is taken" in rig
    cost = "In the outer 0.02 rad of the agent's working box a takeover is refused"
    assert cost in rig
    assert "No rule of the loader asks for the margin" in rig


def test_the_readme_known_gaps_name_what_the_margin_leaves_and_a_box_without_one():
    left = readme_item("- **In the outer 0.02 rad of the agent's working box")
    assert "a takeover is refused" in left and NO_TAKEOVER in left
    assert "The typed `rest` is the way out" in left
    assert "If its path is refused there" in left
    assert "no command moves the arm" in left
    assert "the 48 V switch with the arm held by hand" in left
    equal = readme_item("- **A robot file whose human joint box equals its region")
    assert "a new takeover is refused" in equal
    assert "the typed `rest` and a shutdown are refused for the start pose" in equal
    assert "the 48 V switch with the arm held by hand" in equal


def test_the_readme_gives_the_rig_files_human_joint_box_and_its_margin():
    rig = yaml.safe_load(RIG_ROBOT_YAML.read_text(encoding="utf-8"))
    box, rest = rig["limits"]["human"], rig["rest"]
    told = one_line(readme_section("## The rig files"))

    def listed(values: list[float]) -> str:
        return "`[" + ", ".join(f"{value:g}" for value in values) + "]`"

    assert f"`limits.human.joint_min` is {listed(box['joint_min'])}" in told
    assert f"`limits.human.joint_max` is {listed(box['joint_max'])}" in told
    region = f"{listed(rest['region_min'])} to {listed(rest['region_max'])}"
    assert f"the region runs from {region}" in told
    # The margin the README names is the file's, and so is where there is none.
    above = [edge - high for high, edge in zip(box["joint_max"], rest["region_max"])]
    below = [low - edge for low, edge in zip(box["joint_min"], rest["region_min"])]
    margin = round(above[0], 6)
    assert margin > 0.0
    assert [round(value, 6) for value in above] == [margin] * 6
    assert [round(value, 6) for value in below] == [margin, 0, 0, *[margin] * 3]
    heading = f"**The human joint box lies {margin:g} rad inside `rest.region_*`.**"
    assert heading in told
    assert (
        "The lower bound of the second and the third joint is the region's own, "
        "with no margin"
    ) in told
    # Every other place of the README that names the margin names the file's.
    import re

    whole = one_line(README.read_text(encoding="utf-8"))
    inside = r"([\d.]+) rad inside (?:`rest\.region_\*`|the region)"
    named = re.findall(inside, whole) + re.findall(r"outer ([\d.]+) rad of", whole)
    assert len(named) >= 8 and set(named) == {f"{margin:g}"}


def test_the_readme_names_the_tests_that_ran_the_recipes_of_demo_2():
    recipe = one_line(readme_section("## Run demo 2 on one machine"))
    run = one_line(readme_section("## What was run"))
    by_tests, never = run.split(" Never run: ")
    tests = (TESTS / "test_demo2_programs.py").read_text(encoding="utf-8")
    for name in RECIPE_TESTS:
        assert f"\ndef {name}(" in tests, name
        assert f"`{name}`" in recipe, name
        assert f"`{name}`" in by_tests, name
    # Neither recipe is said to be not run, other than by a person.
    assert "was not run" not in recipe and "not run either" not in recipe
    assert recipe.count("A person has not run them.") == 1
    assert recipe.count("a person has not run it.") == 1
    assert "as processes or by hand" not in never
    assert "Tests ran both as processes." in never


def test_the_readme_tells_a_failed_pass_by_where_its_error_came():
    told = readme_item(
        "- **A pass whose logic raises an error does not end the program.**"
    )
    assert "Right B is still a STOP on such a pass" not in told
    assert "nothing follows the hand until a new squeeze. Right B" not in told
    assert "a STOP or a request that was due on that pass still goes out" in told
    assert "An error before the buttons are read" in told
    assert "is a STOP on the next pass that reads it" in told


def test_the_readme_row_of_unproven_jaws_names_jaws_that_are_not_referenced():
    cell = readme_row("| `the jaws are not proven empty (...)` |")
    assert "`open` is refused too" in cell
    assert "(unknown): the jaws are not referenced`" in cell


def test_the_readme_defines_the_checkers_and_the_limit_set_at_their_first_use():
    text = one_line(README.read_text(encoding="utf-8"))
    for term in ("agent's checker", "recovery checker", "limit set"):
        found = [text.find(term), text.find(term.capitalize())]
        first = min(at for at in found if at >= 0)
        assert text[first - 2 : first] == "**", term
    assert "jaw load" not in text


def test_the_readme_says_in_one_way_that_the_human_program_writes_no_raw_stop():
    text = one_line(README.read_text(encoding="utf-8"))
    assert "writes no raw STOP, as the console does" not in text
    assert (
        "unlike the console, which writes a raw STOP when it is not attached, "
        "the human program writes none"
    ) in text


# The README on where the human program runs, on the jaws under a person and on
# a low gripper tip. On the rig the human program runs on the Thor, the machine
# on the robot, attached at a Unix socket that the robot app serves when it is
# started with --human-socket; on one machine it attaches to the robot link
# service. The jaws are the gripper's fingers; a jaw opening is a share of
# their travel, and the grip level is the jaw opening in force when a person
# takes the arm over. The gripper tip is low below the table clearance or
# inside the agent's last grasp region: a person lifts it out.

HUMAN_YAML = CONFIGS / "human.yaml"
RIG_HUMAN_YAML = CONFIGS / "human_rebot.yaml"
HUMAN_SOCKET = "--human-socket"
ON_THOR = "### The human program on the Thor"
# The item of "A takeover, step by step" on a takeover with a low gripper tip.
LOW_TIP_ITEM = "- **A low gripper tip: a person lifts the arm out.**"


def test_the_readme_places_the_human_program_on_the_thor_on_the_rig():
    text = one_line(README.read_text(encoding="utf-8"))
    opening = text[: text.index(" ## Terms ")]
    assert (
        "On the rig the human program runs on the Thor, attached at the robot "
        f"app's own socket (`{HUMAN_SOCKET}`)"
    ) in opening
    assert (
        "On one machine and in the tests the human program attaches to the robot "
        "link service"
    ) in opening
    # No place calls it a workstation program.
    for old in (
        "**human program** on the workstation",
        "the workstation program in the role `human`",
        "| Human program | Workstation |",
        "| `configs/human_rebot.yaml` | Workstation |",
        "this folder does it on the workstation",
    ):
        assert old not in text, old
    where = readme_row("| Human program |").split("|")
    assert where[2].strip().startswith("Thor on the rig")
    assert "--config configs/human_rebot.yaml`" in where[3]
    assert f"started with `{HUMAN_SOCKET}`" in where[3]
    assert readme_row("| `configs/human_rebot.yaml` |").split("|")[2].strip() == "Thor"
    # The two files place it so: the rig's names a socket of its own, and the
    # one-machine file leaves the robot link service's default.
    rig = yaml.safe_load(RIG_HUMAN_YAML.read_text(encoding="utf-8"))
    demo = yaml.safe_load(HUMAN_YAML.read_text(encoding="utf-8"))
    assert rig["socket_path"] and demo["socket_path"] == ""
    assert f'`socket_path: "{rig["socket_path"]}"`' in text
    assert f'`teleop.endpoint: "{rig["teleop"]["endpoint"]}"`' in text


def test_the_readme_gives_the_robot_app_the_socket_the_rig_human_file_names():
    path = yaml.safe_load(RIG_HUMAN_YAML.read_text(encoding="utf-8"))["socket_path"]
    # The robot app's command line takes the argument, with --connect only.
    served = [*SIM, "--connect", "127.0.0.1:9", HUMAN_SOCKET, path]
    args = robot_main._parser().parse_args(served)
    assert args.human_socket == path and robot_main._unusable(args) is None
    alone = robot_main._parser().parse_args([*SIM, HUMAN_SOCKET, path])
    refusal = robot_main._unusable(alone)
    text = one_line(README.read_text(encoding="utf-8"))
    assert refusal is not None and f"`{refusal}`" in text
    # Every command of the README that starts the robot app for the human
    # program on the Thor gives it the path of the rig's human file.
    given = f"{HUMAN_SOCKET} {path}"
    thor = readme_section(ON_THOR)
    assert f"    {given}\n" in thor  # the command for the Thor
    assert f"Started with `{given}`" in one_line(thor)
    assert f"{given}` |" in readme_row("| F0 |")
    assert f"with `{given}` added" in readme_row("| 15 |")
    assert f"`{given}`" in one_line(readme_section("### The real driver"))
    # And no command names another path.
    import re

    paths = set(re.findall(rf"{HUMAN_SOCKET} (~?/[^\s`]+)", text))
    assert paths == {path}


def test_the_readme_says_how_fast_the_arm_follows_the_hand_as_the_programs_have_it():
    # A person drives the arm: how fast it follows, what a hand that is ahead
    # does, how to stop it and what is printed, by the programs' own numbers
    # and words.
    import math

    from isaaccapture_examples.agentic_remote.human import logic
    from isaaccapture_examples.agentic_remote.human import types as human_types
    from isaaccapture_examples.agentic_remote.robot_app.human import SPEED_MARGIN

    text = one_line(README.read_text(encoding="utf-8"))
    section = one_line(readme_section("### How fast the arm follows the hand"))
    rig = yaml.safe_load(RIG_HUMAN_YAML.read_text(encoding="utf-8"))
    robot = yaml.safe_load((CONFIGS / "robot_app_rebot.yaml").read_text("utf-8"))
    human = robot["limits"]["human"]
    # The joints: the owner's limit, the Thor's clamp and the program's share.
    assert (human["max_joint_speed"], human["max_step_rad"]) == (0.8, 0.05)
    assert (SPEED_MARGIN, logic.THOR_SPEED_MARGIN, logic.JOINT_SPEED_SHARE) == (
        0.98,
        0.98,
        0.9,
    )
    clamp = SPEED_MARGIN * human["max_joint_speed"]
    own = logic.JOINT_SPEED_SHARE * clamp
    assert (f"{clamp:.2f}", f"{own:.2f}") == ("0.78", "0.71")
    assert (
        "The Thor follows a person's joint target at 0.98 of "
        "`limits.human.max_joint_speed`, 0.8 rad/s by the rig owner's decision of "
        "2026-11-16: 0.78 rad/s, which is 0.9 degrees with each setpoint."
    ) in section
    assert "names that limit in WELCOME (`human_max_joint_speed`)" in section
    assert (
        "at 0.9 of the Thor's speed: on the joint that has the farthest to go 0.71 "
        "rad/s, 40 degrees a second, and every other joint by the same share of "
        "its way"
    ) in section
    unnamed = logic.JOINT_SPEED_SHARE * SPEED_MARGIN * logic.UNNAMED_JOINT_SPEED
    assert f"names no such limit is followed at {unnamed:.2f} rad/s" in section
    # The bounds of the tip's target, and the hand that is ahead.
    retarget, ik = rig["retarget"], rig["ik"]
    assert (
        f"`retarget.max_linear_velocity` ({retarget['max_linear_velocity']:g} m/s) "
        f"and `retarget.max_angular_velocity` ({retarget['max_angular_velocity']:.1f} "
        "rad/s) of `configs/human_rebot.yaml` bound the tip's target"
    ) in section
    note = human_types.NOTE_CATCHING_UP.format(cm="N")
    assert note == "hand N cm ahead: the arm is catching up"
    assert f"`{note}` from {human_types.AHEAD_NOTE_CM} cm on" in section
    assert f"| `{note}` |" in text and f"[; {note.replace('N', '<N>')}]" in text
    far = round(100 * ik["max_ahead_m"])
    assert f"believed up to `ik.max_ahead_m`, {far} cm, from the gripper tip" in section
    too_far = f"`the hand is N cm from the gripper tip, over the {far} cm believed`"
    assert too_far in section
    # What still ends an engagement has a section of its own.
    again = one_line(readme_section("### What still asks for a new squeeze"))
    assert "(#what-still-asks-for-a-new-squeeze)" in section
    turned = round(math.degrees(logic.MAX_AHEAD_TURN_RAD))
    assert f"more than {far} cm from the gripper tip (`ik.max_ahead_m`)" in again
    assert f"or turned more than {turned} degrees ahead of it" in again
    assert too_far in again
    assert "**Both programs are to be deployed together.**" in again
    for gone in ("ahead of the arm, over the", "past the workspace limit"):
        assert gone not in text
    # How to stop it, and the squeeze.
    assert "**To stop the arm, release the grip.**" in section
    assert "Stopping the hand does not stop the arm at once" in section
    wait = f"{logic.FOLLOW_WAIT_S:g} s"
    assert f"`the arm did not follow the squeeze within {wait}`" in section
    # The lead and the Thor's own step limit.
    assert f"(`ik.max_ahead_rad`, {ik['max_ahead_rad']} rad on the rig)" in section
    assert f"`limits.human.max_step_rad`, {human['max_step_rad']} rad" in section
    # The program's line of its speeds, with the rig's files.
    assert (
        f"`human program: the arm follows the hand at up to {own:.2f} rad/s a joint "
        "(the robot app's limit for a person is 0.8 rad/s), its tip at up to "
        f"{retarget['max_linear_velocity']:g} m/s and "
        f"{retarget['max_angular_velocity']:g} rad/s`"
    ) in text
    # The rig's human file, and the table of the settings.
    assert "It differs in exactly seven keys:" in text
    for key in ("max_linear_velocity", "max_angular_velocity"):
        assert f"`retarget.{key}: {retarget[key]}`" in text, key
    assert f"`ik.max_ahead_m: {ik['max_ahead_m']}`" in text
    assert "`retarget.max_linear_velocity: 0.01`" not in text
    # The tip's speeds, as the rig's human file gives them beside its bound.
    assert "moves at 7 to 19 cm a second from the raised pose" in section
    assert "up to the 25 cm a second of the bound" in section
    beside = one_line(RIG_HUMAN_YAML.read_text(encoding="utf-8").replace("#", " "))
    assert "7 to 19 cm a second from the raised pose" in beside
    assert "up to the 25 cm a second of the bound below" in beside
    # The straight way, and a release counted from the hand.
    assert "the arm goes straight to where the hand is now" in section
    assert (
        "From the hand, a release takes about 0.05 s and up to three setpoints"
        in section
    )
    assert "`teleop.max_tick_age_s`: up to 0.2 s" in section
    assert rig["teleop"]["max_tick_age_s"] == 0.15
    # A second squeeze, the walls of the Thor's box, a slow Thor: the notes
    # and the reasons as the human program has them.
    assert f"the note says `{logic.NOTE_STOPPING}`" in section
    assert f"| `{logic.NOTE_STOPPING}` |" in text
    wall = human_types.NOTE_AT_LIMIT.format(walls="front")
    assert wall == "at the workspace limit: front"
    names = [name for pair in human_types.WALL_NAMES for name in pair]
    assert sorted(names) == ["back", "bottom", "front", "left", "right", "top"]
    assert (
        f"`{wall}` (`back`, `left`, `right`, `top`, `bottom`; two walls as" in section
    )
    assert f"| `{wall}` (`back`, `left`, `right`, `top`, `bottom`;" in text
    assert "[; at the workspace limit: <wall>[, <wall>]]" in text
    margin_cm = round(100 * logic.TIP_BOX_MARGIN_M)
    assert (
        f"keeps the tip of the pose it solves for {margin_cm} cm inside it" in section
    )
    assert "(`human_tip_box`)" in section and "**`human_tip_box` in WELCOME**" in text
    # The table, the joint box and the reach: named in WELCOME, and the notes.
    for field in ("human_tip_floor", "human_joint_box"):
        assert f"(`{field}`)" in section and f"**`{field}` in WELCOME**" in text
    assert human_types.WALL_TABLE == "table"
    table = human_types.NOTE_AT_LIMIT.format(walls=human_types.WALL_TABLE)
    assert f"the note says `{table}`" in section and f"| `{table}` |" in text
    reach = human_types.NOTE_OUT_OF_REACH
    limit = human_types.NOTE_AT_JOINT_LIMIT.format(joint="N")
    assert (reach, limit) == ("out of reach", "at a joint limit: joint N")
    assert f"the note says `{limit}`" in section and f"| `{limit}` |" in text
    assert f"the note says `{reach}`" in section and f"| `{reach}` |" in text
    assert "[; out of reach | ; at a joint limit: joint <N>]" in text
    joint_cm = f"{logic.JOINT_BOX_MARGIN_RAD:g} rad inside"
    assert joint_cm == "0.01 rad inside" and f"stops {joint_cm} it" in section
    assert "from the fifth failed solve in a row (`ik.max_failures`)" in section
    assert rig["ik"]["max_failures"] == 5
    # The wait for the arm to be at rest, and a robot app that keeps its drive.
    assert f"agree within {logic.REST_TOL_RAD:g} rad" in section
    assert f"{logic.REST_WAIT_S:g} s after the engagement ended" in section
    assert f"the note says `{logic.NOTE_DRIVE_ON}`" in section
    assert f"| `{logic.NOTE_DRIVE_ON}` |" in text
    # A low gripper: what the note tells the person to do.
    raise_notes = f"| `{logic.NOTE_RAISE_TABLE}`, `{logic.NOTE_RAISE}` |"
    assert raise_notes in text
    assert f"the note says `{logic.NOTE_RAISE_TABLE}`" in text
    behind = human_types.NOTE_THOR_BEHIND
    assert behind == "the robot app is behind: the arm follows slower"
    assert f"the note says `{behind}`" in section and f"| `{behind}` |" in text
    assert f"[; {behind}]" in text
    steps = logic.HOLD_BACK_STEPS * clamp * robot["executor"]["period_s"]
    assert f"no further than two of the Thor's steps, {steps:.3f} rad, ahead" in section
    assert logic.HOLD_BACK_STEPS == 2 and logic.NO_FOLLOW_S == 1.0
    assert f"`{logic.NOT_FOLLOWED}`" in section and f"`{logic.NOT_FOLLOWED}`" in text
    # What a blocked joint costs before the arm holds.
    assert "the motors are asked for about 6 degrees more" in section
    # A ROBOT_STATE behind every setpoint, and when the next one is overdue.
    assert "sends a ROBOT_STATE behind every setpoint" in section
    due = logic.STATE_DUE_PERIODS * robot["executor"]["period_s"]
    assert logic.STATE_DUE_PERIODS == 1.5 and f"in this drive, {due:g} s" in section
    # The program's lines of the box and of the joints, with the rig's robot app.
    assert (
        "`human program: the gripper tip stops at the robot app's workspace limits: "
        "x 0.11 to 0.49, y -0.29 to 0.29, z 0.03 to 0.54 m from the arm's base; its "
        "lowest is the table clearance`"
    ) in text
    assert (
        "`human program: the joints stop at the robot app's joint limits for a "
        "person, and the arm stands while the hand is out of reach`"
    ) in text
    # The comment on the limit that sets the arm's pace says so.
    pace = one_line(
        (CONFIGS / "robot_app_rebot.yaml").read_text("utf-8").replace("#", " ")
    )
    assert "This limit sets how fast the arm follows a person." in pace
    assert f"at 0.9 of that clamp, {own:.2f} rad/s a joint" in pace
    assert "only bound the gripper tip, at 25 cm/s" in pace
    assert "decide how fast the joint targets" not in pace


def test_the_readme_gives_the_two_jaw_settings_of_every_robot_app_file():
    files = sorted(CONFIGS.glob("robot_app*.yaml"))
    assert len(files) >= 5
    shipped = set()
    for file in files:
        human = yaml.safe_load(file.read_text(encoding="utf-8"))["human"]
        shipped.add((human["trigger_jaws"], human["jaw_match_tol"]))
    [(on, tol)] = shipped  # one pair of values in every file
    default = RobotAppConfig.default().human
    assert (default.trigger_jaws, default.jaw_match_tol) == (on, tol)
    switch = f"`human.trigger_jaws: {str(on).lower()}`"
    both = f"{switch} and `human.jaw_match_tol: {tol:g}` in every robot app file"
    # The row of "The settings that ship off", and the rule of the jaws.
    assert readme_row("| `human.trigger_jaws: ").startswith(f"| {both} |")
    text = one_line(README.read_text(encoding="utf-8"))
    assert f"With {switch} in the robot app's file, the shipped value" in text
    near = f"is within `human.jaw_match_tol` ({tol:g} of the jaw travel) of the grip"
    assert near in text
    rig = one_line(readme_section("## The rig files"))
    assert f"({switch}, `human.jaw_match_tol: {tol:g}`)" in rig
    # The other position is told as what it is, not as what ships.
    off = f"`human.trigger_jaws: {str(not on).lower()}`"
    assert f"With {off} the trigger does nothing" in text
    assert f"{off} in every robot app file" not in text


def test_the_readme_quotes_the_rule_of_the_lift_as_the_robot_app_has_it():
    from isaaccapture_examples.agentic_remote.robot_app import app, human

    steps = one_line(readme_section("### A takeover, step by step"))
    rule = steps[steps.index(LOW_TIP_ITEM) :]
    # What a takeover with a low gripper tip adds to its line.
    assert f"the takeover's line ends `{app.LOW_TIP}`" in rule
    assert f"the line goes on `{app.LOW_TIP}`" in readme_row("| 5 | B stays down")
    # The refusal of a target that would lower the tip, and its tolerance.
    ended = "`human drive ended: " + human.LIFT + "<why the tip is low>, and the target"
    assert ended in rule
    tolerance = f"{human.TIP_TOL_M * 1000:g} mm"
    assert "The robot app keeps a **floor** for the gripper tip" in rule
    assert f"puts the tip more than {tolerance} under that floor" in rule
    lowered = readme_row("| While the measured gripper tip is low")
    assert f"`human drive ended: {human.LIFT}...`" in lowered
    # APAI's own rule under its table clearance, and what stays refused.
    assert "Below APAI's own table clearance every step must rise" in rule
    assert "The typed `rest` stays refused there" in rule
    # The old rule is gone, and the test file of the rule is named.
    whole = one_line(README.read_text(encoding="utf-8"))
    assert "Below the table clearance nobody can move the arm" not in whole
    assert "Below the table clearance a takeover is refused too" not in whole
    assert (TESTS / "test_lift_out.py").is_file()
    assert "`test_lift_out.py`" in readme_section("## Run the tests")


def test_the_readme_gives_the_joint_target_of_a_waiting_drive_as_the_logic_has_it():
    # A drive of the robot app waits for a lift while it sends nothing for an
    # arm that has settled under its highest gripper tip.
    from isaaccapture_examples.agentic_remote.human import logic
    from isaaccapture_examples.agentic_remote.robot_app.human import SPEED_MARGIN

    steps = one_line(readme_section("### A takeover, step by step"))
    rule = steps[steps.index(LOW_TIP_ITEM) :]
    rig = yaml.safe_load(RIG_HUMAN_YAML.read_text(encoding="utf-8"))
    robot = yaml.safe_load((CONFIGS / "robot_app_rebot.yaml").read_text("utf-8"))
    # When the human program takes the drive to wait, and its two notes.
    assert f"for {logic.LIFT_AFTER_S:g} s, a drive that has this engagement's" in rule
    for note in (logic.NOTE_RAISE_TABLE, logic.NOTE_RAISE):
        assert f"`{note}`" in rule
    # One step of the robot app from the measured arm, with the rig's numbers.
    one_step = "no further than the robot app reaches from the measured arm in one step"
    assert one_step in rule
    speed, period = robot["limits"]["human"]["max_joint_speed"], 0.02
    assert robot["executor"]["period_s"] == period
    step = logic.JOINT_SPEED_SHARE * SPEED_MARGIN * speed * period
    share = f"{logic.JOINT_SPEED_SHARE:g} of one step of the robot app"
    assert f"{share} ({step:.3f} rad on the rig, 0.8 degrees)" in rule
    assert f"`ik.max_ahead_rad` less {logic.LIFT_SLACK_RAD:g} rad" in rule
    tip_mm = 1000 * rig["retarget"]["max_linear_velocity"] * period
    assert f"allows in one setpoint period ({tip_mm:g} mm on the rig)" in rule
    assert logic.LIFT_SHARES == (1.0, 0.5, 0.25, 0.125)
    assert "is tried at a half, a quarter and an eighth of its length" in rule
    assert "This holds wherever the tip is" in rule
    # Under the named height: the hand's pose above the tip.
    tol_mm = 1000 * rig["ik"]["tol"]
    assert logic.LIFT_ASKED_TOLS == 2.0
    assert (
        f"above that tip by more than {logic.LIFT_ASKED_TOLS * tol_mm:g} mm, twice "
        f"what a solve may leave of its pose (`ik.tol`, {tol_mm:g} mm)"
    ) in rule
    assert "goes sideways or goes down moves nothing there" in rule
    # What it leaves open is under the known gaps, and the old bound is gone.
    gaps = one_line(readme_section("## Known gaps"))
    for gap in (
        "An arm that settled far under its highest point is not lifted by a squeeze.",
        "A slow start of the Thor's drive is taken for a wait for a lift.",
    ):
        assert f"**{gap}**" in gaps
    assert "the depth shrinks with that limit" in gaps
    taken = f"From {logic.LIFT_AFTER_S:g} s on the human program takes such a drive"
    assert taken in gaps
    whole = one_line(README.read_text(encoding="utf-8"))
    assert "0.04 rad on the rig" not in whole
    assert "the joint target follows the hand" not in whole
    assert (TESTS / "test_rig_configs.py").is_file()


def test_the_readme_quotes_what_the_trigger_is_told_as_the_programs_say_it():
    from isaaccapture_examples.agentic_remote.console import status
    from isaaccapture_examples.agentic_remote.human import logic
    from isaaccapture_examples.agentic_remote.robot_app import gripper

    text = one_line(README.read_text(encoding="utf-8"))
    # The human program: its two notes and the trigger's two dead zones.
    for note in (logic.NOTE_TRIGGER_PULL, logic.NOTE_TRIGGER_EASE):
        assert f"`{note}`" in text, note
    low, high = logic.TRIGGER_OPEN_MAX, logic.TRIGGER_CLOSED_MIN
    assert (logic.jaw_opening(low), logic.jaw_opening(high)) == (1.0, 0.0)
    assert logic.jaw_opening(0.25) == 0.75
    zones = (
        f"A trigger at or below {low:g} asks for fully open jaws, one at or above "
        f"{high:g} for closed jaws"
    )
    assert zones in text
    # The console's jaws line.
    for line in (
        status.JAWS_FOLLOW,
        status.JAWS_BRING.format(asked="X", level="Y"),
        f"{status.JAWS_NOTHING}: <reason>",
    ):
        assert f"`{line}`" in text, line
    # The robot app's reasons that the rule of the jaws quotes.
    for reason in (gripper.NOT_MATCHED, gripper.NO_TIGHTER):
        assert f"`{reason}`" in text, reason
    assert f"(`{gripper.NO_CLOSING}the live view is not fresh: ...`)" in text


# The README on the workstation's start script, on the agent program's
# arguments and on three refusals of the robot app. tmux is a program that
# splits a terminal into panes; the script starts the workstation's programs of
# demo 1 in the panes of one tmux window. The experiment's launcher is the
# program of the experiment folder whose arguments the agent program takes. A
# GRIPPER is the link message of one jaw command and a MOVE that of one move;
# the expiry of either is a time of the Thor's monotonic clock. A torque
# release switches the motors off at measured rest.

WORKSTATION_SCRIPT = CONFIGS.parent / "run_workstation.sh"
START_SCRIPTS = "### The start scripts"
LAUNCHER_OPTIONS = "### The options of the experiment's launcher"
AGENT_YAMLS = (CONFIGS / "agent.yaml", CONFIGS / "agent_rebot.yaml")
TORQUE_RULE = "## The rule: torque is never cut away from rest"
RESULT_WAIT_S = 5.0  # real seconds a test waits for a RESULT


def workstation_variables() -> list[str]:
    """The variables run_workstation.sh asks for, from its own list."""
    import re

    script = WORKSTATION_SCRIPT.read_text(encoding="utf-8")
    listed = re.search(r"^names=\(\n(.*?)\n\)$", script, re.M | re.S)
    assert listed is not None
    return listed.group(1).split()


def workstation_script(folder: Any, *arguments: str, **changed: str | None) -> Any:
    """Run run_workstation.sh --print; the process as it ended.

    With --print the script starts nothing. Its variables name this Python
    and `folder`; `changed` replaces one, or with None leaves it out.
    """
    import subprocess

    env = {name: str(folder) for name in workstation_variables()}
    env["AGENTIC_REMOTE_AGENT_PYTHON"] = sys.executable
    env = {name: value for name, value in {**env, **changed}.items() if value}
    return subprocess.run(
        ["bash", str(WORKSTATION_SCRIPT), "--print", *arguments],
        env={**env, "PATH": os.environ["PATH"]},
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


def test_the_readme_describes_the_workstation_script_as_it_prints_its_commands(
    tmp_path,
):
    import re
    import subprocess

    script = WORKSTATION_SCRIPT.read_text(encoding="utf-8")
    told = readme_section(START_SCRIPTS)
    parsed = subprocess.run(
        ["bash", "-n", str(WORKSTATION_SCRIPT)], capture_output=True, check=False
    )
    assert parsed.returncode == 0, parsed.stderr

    def commands(*arguments: str) -> tuple[dict[str, str], list[str]]:
        """Each pane's command as --print gives it, and the lines printed."""
        done = workstation_script(tmp_path, "--thor", "192.0.2.7", *arguments)
        assert (done.returncode, done.stderr) == (0, "")
        lines = done.stdout.splitlines()
        named = {
            line[2:].split()[0].rstrip(":"): command
            for line, command in zip(lines, lines[1:])
            if line.startswith("# ")
        }
        return named, lines

    printed, lines = commands()
    # The panes the script makes are the rows of the README's table: five. The
    # viewer's page is the live view.
    panes = re.findall(r"-T (\w+)$", script, re.M)
    panes += re.findall(r"\$\(pane (\w+)\)", script)
    rows = dict(re.findall(r"^\| `([a-z]+)` \| `([^`]+)`", told, re.M))
    assert sorted(rows) == sorted(panes) == sorted(printed)
    assert len(rows) == 5 and rows["viewer"] == "./run_viewer.sh"
    assert "The window has five panes, all visible together" in one_line(told)
    assert "**The pane `viewer` is the live view.**" in told
    said = (
        "`live view: the viewer's page with its live picture showing, in a visible "
        "browser tab; a person's takeover needs it`"
    )
    assert said in one_line(told)
    for line in said.strip("`").split(", in a visible browser tab; "):
        assert line in one_line(script), line
    variables = workstation_variables()
    assert len(variables) == 6
    for pane, command in rows.items():
        given = command.removeprefix("python ").replace("ADDRESS", "192.0.2.7")
        assert printed[pane].endswith(f" {given}"), pane
        # Each pane's command carries the variables itself, as the README says.
        for name in variables:
            assert f" {name}=" in printed[pane], (pane, name)
    for name in variables:
        assert f"`{name}`" in told, name
    # Without --thor the camera receiver's command names no address.
    no_address = workstation_script(tmp_path).stdout
    assert "./run_camera_receiver.sh --host THOR_ADDRESS\n" in no_address
    told = one_line(told)
    assert "with `THOR_ADDRESS` in the address's place" in told
    # The refusals: one line, exit code 2, nothing printed.
    unknown = workstation_script(tmp_path, "--bogus")
    assert (unknown.returncode, unknown.stdout) == (2, "")
    assert unknown.stderr.count("\n") == 1
    arguments = (
        "--env FILE",
        "--thor ADDRESS",
        "--session NAME",
        "--socket-wait SECONDS",
        "--print",
    )
    for argument in arguments:
        assert argument in unknown.stderr and f"[{argument}]" in told, argument
    # The wait for the service's socket: its default, and what cannot be one.
    waited = re.search(r"^socket_wait_s=(\d+)$", script, re.M)
    assert waited is not None
    assert f"at most {waited.group(1)} s; `--socket-wait SECONDS`" in told
    no_number = workstation_script(tmp_path, "--socket-wait", "soon")
    assert (no_number.returncode, no_number.stdout) == (2, "")
    assert no_number.stderr.count("\n") == 1
    # Only the console waits for the service: the viewer attaches by itself.
    once = [line for line in lines if "once the service takes a connection on " in line]
    assert len(once) == 1 and once[0].startswith("# console ")
    assert "**The console starts once the service listens.**" in told
    assert "The viewer attaches to the service by itself" in told
    # An option of before is an unknown argument now.
    gone = workstation_script(tmp_path, "--live-view")
    assert (gone.returncode, gone.stdout) == (2, "")
    assert gone.stderr.startswith(
        "run_workstation.sh: unknown argument '--live-view' ("
    )
    for name in variables:
        unset = workstation_script(tmp_path, **{name: None})
        assert (unset.returncode, unset.stdout) == (2, ""), name
        assert name in unset.stderr and unset.stderr.count("\n") == 1, name
    nowhere = str(tmp_path / "no-python")
    no_python = workstation_script(tmp_path, AGENTIC_REMOTE_AGENT_PYTHON=nowhere)
    assert (no_python.returncode, no_python.stdout) == (2, "")
    # The optional variable: the address the viewer listens on, with no token.
    host = "AGENTIC_REMOTE_VIEWER_HOST"
    assert f"#   {host} " in script and host not in variables
    open_to = workstation_script(tmp_path, **{host: "0.0.0.0"}).stdout
    assert " ./run_viewer.sh --host 0.0.0.0 --no-auth\n" in open_to
    assert f"`{host}` names the address the experiment's viewer listens on" in told
    assert "is `./run_viewer.sh --host <that address> --no-auth`" in told
    assert "without a token. Without it the viewer listens on `127.0.0.1`" in told
    not_plain = workstation_script(tmp_path, **{host: "0.0.0.0:8766"})
    assert (not_plain.returncode, not_plain.stdout) == (2, "")
    assert host in not_plain.stderr and not_plain.stderr.count("\n") == 1
    step = readme_section("### 3. The variables of each machine, in one file")
    assert f"\n# {host}=0.0.0.0\n" in step
    # The session's name, and the row of the programs' table.
    session = re.search(r'^session="(\w+)"$', script, re.M)
    assert session is not None
    assert f"names the tmux session (`{session.group(1)}`)" in told
    assert f"`attach with: tmux attach -t {session.group(1)}`" in told
    where = readme_row("| The console, the agent program, the robot link service,")
    assert "`bash run_workstation.sh --env FILE --thor THOR_ADDRESS`" in where
    never = one_line(readme_section("## What was run")).split(" Never run: ")[1]
    assert "`run_workstation.sh` with tmux: a test runs it with `--print` only" in never


def test_the_readme_names_every_argument_the_agent_program_takes(monkeypatch):
    import re
    from pathlib import Path

    from isaaccapture_examples.agentic_remote.agent import config as agent_config
    from isaaccapture_examples.agentic_remote.agent import launcher

    for name in agent_config.PATH_VARIABLES:
        monkeypatch.setenv(name, "/absolute")
    parser = launcher._parser()
    taken = {
        option
        for action in parser._actions
        for option in action.option_strings
        if option.startswith("--") and option != "--help"
    }
    program = readme_section("### The agent program")
    usage = program[program.index("```text") : program.index("\n```\n")]
    assert set(re.findall(r"--[a-z-]+", usage)) == taken
    told = readme_section(LAUNCHER_OPTIONS)
    rows = dict(re.findall(r"^\| `(--[a-z-]+)[^|]*\| (.*) \|$", told, re.M))
    # Every argument of the launcher's lines of the usage has its row.
    theirs = usage[usage.index("[--batch-milestones]") :]
    assert set(rows) == set(re.findall(r"--[a-z-]+", theirs))
    assert len(rows) == 9 and set(rows) <= taken

    def run(settings: Any, *arguments: str) -> Any:
        args = parser.parse_args(["--config", "unused", *arguments])
        return launcher._run_options(args, settings)

    # The command the README gives is the one the workstation's start script
    # types, and it is taken on the rig's file. Of its arguments --operate and
    # the stereo matcher decide nothing about what the run records.
    rig = agent_config.load_settings(CONFIGS / "agent_rebot.yaml")
    command = re.search(r"^\./run_agent\.sh --config \S+ (.*?)\n```", told, re.M | re.S)
    assert command is not None
    given = command.group(1).replace("\\\n", " ").split()
    assert {"--operate", "--depth-backend", "--motion-approval"} < set(given)
    typed = " ".join(given)
    script = one_line(WORKSTATION_SCRIPT.read_text(encoding="utf-8"))
    assert typed in script.replace('" agent+=" ', " ")
    kept = [word for word in given if word not in ("--operate", "ffs")]
    kept.remove("--depth-backend")
    took = run(rig, *given)
    assert took == run(rig, *kept)
    assert (took.record_debug, took.batch_milestones) == (True, True)
    # The arguments' values, as the parser has them.
    parsed = parser.parse_args(["--config", "unused", *given])
    assert (parsed.operate, parsed.depth_backend) == (True, "ffs")
    assert (parsed.speed_profile, parsed.mounting) == (None, None)
    assert rows["--operate"].startswith("Accepted: ")
    assert "The stereo matcher of this run" in rows["--depth-backend"]
    profiles = agent_config.SPEED_PROFILES
    assert sorted(profiles) == ["conservative", "reference"]
    for name in profiles:
        assert f"`{name}`" in rows["--speed-profile"], name
        assert parser.parse_args(["--config", "x", "--speed-profile", name])
    assert f"`{agent_config.SPEED_REFERENCE}` (the default)" in rows["--speed-profile"]
    slow = f"`conservative` is {agent_config.CONSERVATIVE_SPEED:g} rad/s"
    assert slow in rows["--speed-profile"]
    assert "`thor.agent_max_joint_speed`" in rows["--speed-profile"]
    assert "`camera.hand_eye_json`" in rows["--mounting"]
    assert "`accepted_fit` is true" in rows["--mounting"]
    # What the run does not take, as the README quotes it.
    text = one_line(told)
    mock = agent_config.load_settings(CONFIGS / "agent_mock.yaml")
    for settings, arguments in (
        (rig, ["--motion-approval", "tool"]),
        (rig, ["--record-debug", "--no-timing"]),
        (mock, ["--no-timing"]),
    ):
        with pytest.raises(agent_config.ConfigError) as refused:
            run(settings, *arguments)
        reason = str(refused.value)
        head = reason[: reason.index(":" if ":" in reason else ";") + 1]
        head = head.removeprefix("--no-timing ")
        assert f"{head} ...`" in text, reason
    # The defaults: the size, and the two recordings of both shipped files.
    assert run(rig).debug_max_gib == agent_config.DEBUG_MAX_GIB
    assert f"({agent_config.DEBUG_MAX_GIB:g})" in rows["--debug-max-gib"]
    keys = one_line(readme_section("### `configs/agent.yaml`"))
    for file in AGENT_YAMLS:
        written = yaml.safe_load(file.read_text(encoding="utf-8"))["experiment"]
        assert (written["timing"], written["model_trace"]) == (True, True), file
        assert f"`configs/{file.name}`" in text
    assert "`timing` (true) and `model_trace` (true)" in keys
    assert (run(rig).timing, run(rig).model_trace) == (True, True)
    assert run(mock) == agent_config.RunOptions()
    # The program's first lines, and what the debug recording leaves out.
    lines = launcher._option_lines(took, Path("run"))
    for line in lines:
        assert f"`{line.split(':')[0]}:`" in text, line
    [debug] = [line for line in lines if line.startswith("debug recording:")]
    for event in re.search(r"\(([^)]+)\)", debug).group(1).split(", "):
        assert f"`{event}`" in text, event


@pytest.mark.apai
def test_the_readme_quotes_operate_without_the_operators_terminal():
    import re

    from isaaccapture_examples.agentic_remote.agent import config as agent_config
    from isaaccapture_examples.agentic_remote.agent import launcher

    told = readme_section(LAUNCHER_OPTIONS)
    rows = dict(re.findall(r"^\| `(--[a-z-]+)[^|]*\| (.*) \|$", told, re.M))
    args = launcher._parser().parse_args(["--config", "unused", "--operate"])
    # A test has no terminal: the program's own question could not be answered.
    with pytest.raises(agent_config.ConfigError) as refused:
        launcher._operator_terminal(args, launcher.terminal_ask)
    assert f"exit code 2 and `{refused.value}`" in rows["--operate"]
    # Without the argument, or with another way to ask, nothing is refused.
    launcher._operator_terminal(args, lambda prompt: "yes")
    plain = launcher._parser().parse_args(["--config", "unused"])
    launcher._operator_terminal(plain, launcher.terminal_ask)


def test_the_readme_quotes_a_grippers_expiry_as_the_robot_app_refuses_it(robot_app):
    import dataclasses
    import re

    from isaaccapture_examples.agentic_remote.protocol.messages import Gripper

    h = robot_app
    h.new_task()
    late = h.client.gripper(0.0, expires_mono_s=h.clock.monotonic() - 0.25)
    refused = h.client.wait_result(late, timeout_s=RESULT_WAIT_S)
    assert refused.status == "refused" and h.arm.jaw_commands == []
    quoted = re.sub(r"\d+\.\d{3}", "X.XXX", refused.reason)
    jaws = one_line(readme_section("## What the jaws do"))
    assert f"with `{quoted}`; the jaws are sent nothing" in jaws
    # The field as the README names it, and a GRIPPER that carries none.
    fields = {field.name: field for field in dataclasses.fields(Gripper)}
    assert fields["expires_mono_s"].default is None
    assert "**A GRIPPER can carry an expiry**, `expires_mono_s`" in jaws
    h.clock.advance(1000.0)
    plain = h.client.gripper(0.5)
    assert h.client.wait_result(plain, timeout_s=RESULT_WAIT_S).status == "completed"
    assert "A GRIPPER without the field never expires" in jaws
    # The waits of the agent's files that the README gives in seconds.
    move = one_line(readme_section("### A move, its permits and the camera evidence"))
    for file in AGENT_YAMLS:
        agent = yaml.safe_load(file.read_text(encoding="utf-8"))
        wait = f"{agent['jaws']['result_timeout_s']:g} s"
        assert f"`jaws.result_timeout_s` ({wait})" in jaws, file
        assert f"no RESULT within {wait}; the arm holds" in jaws, file
        assert f"`move.expires_in_s` ({agent['move']['expires_in_s']:g} s)" in move


def test_the_readme_quotes_a_failed_torque_release_as_the_robot_app_says_it(robot_app):
    from isaaccapture_examples.agentic_remote.console import status
    from isaaccapture_examples.agentic_remote.robot_app.interfaces import ArmFault

    h = robot_app

    def failing() -> None:
        raise ArmFault("bus off")

    h.arm.release_torque = failing
    failed = h.client.wait_result(h.client.release(), timeout_s=RESULT_WAIT_S)
    state = h.client.wait_state(lambda s: s.stop_level == "fault")
    assert failed.status == "failed" and state.torque_on is True
    rule = one_line(readme_section(TORQUE_RULE))
    commands = one_line(readme_section("## What the operator's commands do"))
    carried = one_line(
        readme_section(
            "### What WELCOME, ROBOT_STATE and VIEW_STATUS carry for the operator"
        )
    )
    # The motion word, and how the fault-stop's reason begins.
    assert f"The motion state becomes `{state.motion}`" in rule
    assert f"The motion state is `{state.motion}`" in commands
    assert f"and `{state.motion}`. The last one follows a torque release" in carried
    begins = state.fault_reason.removesuffix("bus off")
    assert begins != state.fault_reason and failed.reason == f"fault: {begins}bus off"
    assert f"with the reason `{begins}the driver's disconnect failed: " in rule
    # What the console shows of the motors then.
    shown = status.status_line(None, state, None, 0.0)
    [torque] = [part for part in shown.split(" | ") if part.startswith("torque ")]
    assert f"The console shows `{torque}`" in rule
    assert f"`{torque}` while the motion is `{state.motion}`" in one_line(
        readme_section("### The operator console")
    )
    # What a return to rest is told once the fault-stop was reset.
    reset = h.client.wait_result(h.client.reset(), timeout_s=RESULT_WAIT_S)
    assert reset.status == "completed"
    h.client.wait_state(lambda s: s.stop_level == "none")
    rest = h.client.wait_result(h.client.rest(), timeout_s=RESULT_WAIT_S)
    assert rest.status == "refused" and h.arm.sent == []
    assert f"refused with `{rest.reason}`" in rule
    assert f"refused with `{rest.reason}`" in commands
    assert "`test_release_failed.py`" in readme_section("## Run the tests")


# The README on the experiment's viewer. The experiment's viewer is
# thor_viewer.py of the experiment folder: a web page for a browser on what a
# run of the agent program recorded. run_viewer.sh starts it. The debug
# recording is what a run started with --record-debug keeps of the obstacle
# map's inputs and of the joint feedback.

VIEWER_SCRIPT = CONFIGS.parent / "run_viewer.sh"
VIEWER = "### The experiment's viewer"


def test_the_readme_tells_the_experiments_viewer_as_its_start_script_starts_it():
    import re

    from isaaccapture_examples.agentic_remote.agent import config as agent_config

    script = VIEWER_SCRIPT.read_text(encoding="utf-8")
    head = script[: script.index("\nset -euo pipefail")]
    section = readme_section(VIEWER)
    told = one_line(section)
    # The usage line takes the arguments the script's head names.
    usage = section[section.index("```text") : section.index("\n```\n")]
    named = set(re.findall(r"^#   (--[a-z-]+)", head, re.M))
    assert named == set(re.findall(r"--[a-z-]+", usage))
    live = {"--no-live", "--live-fps", "--live-width", "--receiver-socket"}
    live |= {"--link-socket"}  # the live view: where the robot link service is
    assert named == {"--runs", "--host", "--port", "--token-file", "--no-auth"} | live
    # What the script gives the viewer: this machine, and its own two places.
    assert "defaults=(--host 127.0.0.1)" in script
    assert 'runs="${state}/runs"' in script
    assert 'token="${state}/viewer.token"' in script
    own = "behind the script's own `--host 127.0.0.1`, `--runs` and `--token-file`"
    assert own in told
    assert "The default is `.state/runs` beside the script" in told
    assert "The file is `.state/viewer.token` beside the script" in told
    link = '`echo "http://127.0.0.1:8766/?token=$(cat .state/viewer.token)"`'
    assert link in told
    assert "default: the viewer's own, 8766" in head
    # Its three variables and the one of the live picture, and every refusal
    # the script has.
    variables = re.findall(r"^#   (AGENTIC_REMOTE_\w+)", head, re.M)
    assert len(variables) == 4 and variables[3] == "AGENTIC_REMOTE_CAMERA_PR"
    for name in variables:
        assert f"`{name}`" in told, name
    refusals = told[told.index("**Exit code 2 and one line, nothing started**") :]
    refusals = refusals[: refusals.index("**What differs")]
    assert len(re.findall(r"\brefuse \"", script)) == 7
    for piece in (
        "one of the three variables cannot be used",
        "the experiment folder holds no `thor_viewer.py`",
        "an argument that takes a value has none",
        "`--runs` names no folder",
        "the state folder cannot be made",
    ):
        assert piece in refusals, piece
    # The viewer's rule for a run, with the file the agent program writes.
    listed = f"holds `debug/events.jsonl` and `{agent_config.RESOLVED_CONFIG}`"
    assert listed in told
    assert "**The picture of the obstacle map needs `--record-debug`.**" in told
    # The live picture: shown and not stored, its buttons and its first values,
    # as the launcher and the page's block have them.
    package = CONFIGS.parent / "python" / "isaaccapture_examples" / "agentic_remote"
    program = (package / "viewer_live" / "live.py").read_text(encoding="utf-8")
    values = (package / "viewer_live" / "settings.py").read_text(encoding="utf-8")
    block = (package / "viewer_live" / "page_block.html").read_text(encoding="utf-8")
    assert "**It only looks.**" in told
    assert "It is **shown and not stored**" in told
    assert "DEFAULT_FPS = 30.0" in values and "up to 30 a second" in told
    fps = "`--live-fps N` is the most pictures a second, 1 to 60, 30 by default"
    assert "FPS_RANGE = (1.0, 60.0)" in values and fps in told
    width = "`--live-width PIXELS` is the picture's greatest width, 64 to 4096, 960"
    assert "WIDTH_RANGE_PX = (64, 4096)" in values and width in told
    assert "DEFAULT_WIDTH_PX = 960" in values
    # The pause after a failed try, and the line that names it.
    pauses = "5 s, then 10, 20 and 40 s, and 60 s from then on; once pictures arrived"
    assert "RETRY_PAUSES_S = (5.0, 10.0, 20.0, 40.0, 60.0)" in program
    assert f"{pauses} it is 5 s again" in told
    assert "trying again in {pause_s:g} s" in program
    assert "(`trying again in 5 s`), once for each reason and each pause" in told
    # The viewer beside a running agent: its bridge loads before it subscribes.
    assert "It can be started and ended at any time" not in told
    assert "loads CuPy and makes its CUDA context before it subscribes" in told
    assert "STALE_S = 2.0" in program and "no camera picture younger than 2 s" in told
    for path in ("/api/live.mjpg", "/api/live.json", "/api/live.jpg"):
        assert f'"{path}"' in program and path in told, path
    assert '"/api/live/shown"' in program and "`/api/live/shown`" in told
    # The page fetches numbered pictures and acknowledges them: no stream.
    for path in ("/api/live.json", "/api/live.jpg", "/api/live/shown"):
        assert f"'{path}'" in block, path
    assert "/api/live.mjpg" not in block
    # The live view: the viewer's page with its live picture, and its word to
    # the robot app.
    assert "The page with its live picture showing is the **live view**" in told
    link = (package / "viewer_live" / "view_link.py").read_text(encoding="utf-8")
    assert "RETRY_PAUSES_S = (5.0, 10.0, 20.0, 40.0, 60.0)" in link
    view = one_line(readme_section("### The live view"))
    assert view.startswith(
        "### The live view The live view is the page of the experiment's viewer "
        "with its live picture showing, in a visible browser tab."
    )
    for piece in (
        "`viewer: live view: not reported to the robot: cannot attach to the robot "
        "link service: ...; trying again every 5 to 60 s. The page is served as "
        "before`",
        "`viewer: live view: attached to the robot link service at <socket>; the "
        "robot is told how fresh the page's live picture is`",
        "Its tries are 5, 10, 20 and 40 s apart and then 60 s",
        "`--link-socket PATH` names another socket",
        "`GET /api/live.jpg?after=N`",
        "`POST /api/live/shown`",
        "`The robot is told: live view, N s old.`",
        # The rule and the token are the viewer's own, in its model.
        "under the rules of **The view is live** and **The view token** below "
        "(`viewer_live/view_model.py`)",
        "- **The view is live** only while five things hold:",
        "- **The view token** is a secret the viewer gives to one page:",
        "The viewer sends VIEW_STATUS every 0.2 s, also while the view is not live.",
        "These times are fixed values of `viewer_live/settings.py`",
    ):
        assert piece in view, piece
    for value in (
        "STATUS_PERIOD_S = 0.2",
        "ACK_TIMEOUT_S = 1.0",
        "STATE_MAX_AGE_S = 1.0",
    ):
        assert value in values, value
    model = (package / "viewer_live" / "view_model.py").read_text(encoding="utf-8")
    assert "CLOCK_STEP_QUIET_S = 3.0" in model and "FUTURE_TOLERANCE_S = 0.05" in model
    for piece in (
        "not reported to the robot: {why}; trying",
        "attached to the robot link service at ",
        "the robot is told how fresh the page's live",
    ):
        assert piece in link, piece
    for piece in (
        "The robot is told: live view, ${view.age_s.toFixed(2)} s old.",
        # The note of a page whose viewer gives no answer, or no new picture.
        "`${failed}; trying again every second${old ? `; this picture is ${age} old`",
        "`the viewer answered ${answer.status}`",
        "`no new picture from the viewer; this picture is ${age} old`",
    ):
        assert piece in block, piece
    assert "This page was loaded again" not in block + view
    # Why the page does not tell the robot "live": the block's words, the
    # share of the picture that must be inside the window, and the watch-only
    # page of another machine. The longer ones are the note under the button:
    # a page that a newer load replaced, and one that was open across a
    # restart of the viewer.
    for words in (
        "Map picture showing",
        "tab hidden",
        "scroll the live picture into view",
        "watch only: see below",
        "watch only: the live view is the page opened on the workstation",
        "viewer not reached: see below",
        "viewer restarted: see below",
        "no new picture: see below",
        "the viewer cannot be reached",
        "the viewer was restarted: this page loads itself again",
        "the viewer was restarted: this page loads itself again once Live picture is "
        "pressed",
        "watch only: the viewer was restarted; reload this page to make it the live "
        "view",
        "watch only: another page on the workstation is the live view; reload this "
        "page to make it the live view",
        "watch only: no page is the live view now; reload this page to make it the "
        "live view",
    ):
        assert f"'{words}'" in block and f"`{words}" in view, words
    for piece in (
        "- **What the page's note says** when its picture is not live or does not "
        "count.",
        "`the viewer cannot be reached; trying again every second; this picture is N "
        "s old`",
        "`the viewer answered 401; trying again every second`",
        "`no new picture from the viewer; this picture is N s old`",
        "a page whose note ends with `reload this page to make it the live view` is "
        "reloaded",
        "After a restart of the viewer nothing is done: the page loads itself again",
        # The viewer instance, and what the viewer says of the page that counts.
        "7. A token is valid in one **viewer instance**: one start of the viewer",
        "`X-Viewer-Instance`",
        "It does so once for each instance",
        # A load that a page makes by itself takes the view token from no page.
        "It does so only while `loaded` of that viewer is false, in a visible tab, "
        "and not if it was replaced before",
        "the page makes it only while the new viewer has given no page a view token",
        "From then on it acknowledges nothing more and asks for pictures to watch "
        "(`&watch=1`)",
    ):
        assert piece in view, piece
    assert 'INSTANCE_HEADER = "X-Viewer-Instance"' in program
    for key in ("instance", "counting_page", "loaded", "is_this_page", "asks"):
        assert f'"{key}"' in program and f"`{key}`" in view and f"`{key}`" in told
    assert "if (!REMOTE && vacantFor !== other) return;" in block
    assert "vacantFor = other && counting.loaded === false ? other : '';" in block
    assert "FETCH_WATCH_S = 2.0" in program and "in the last 2 s (`asks`)" in view
    trouble = troubleshooting()
    for piece in (
        "| Viewer's page: the live picture stands still, dimmed, and the note under "
        "the button says `the viewer cannot be reached; trying again every second; "
        "this picture is N s old` |",
        "| Viewer's page: `watch only: another page on the workstation is the live "
        "view; reload this page to make it the live view`, or `watch only: no page "
        "is the live view now; ...` |",
        "Reload a page whose note ends with `reload this page to make it the live "
        "view`.",
    ):
        assert piece in trouble, piece
    assert "const SHARE = 0.75;" in block
    for piece in (
        "with at least three quarters of it inside the browser window",
        "less than three quarters of the live picture are inside the browser window",
        "**Only a page opened on the workstation is the live view.**",
        "the client's address is a loopback address or the address the request "
        "arrived on, both read off the connection and not off a header",
        "`GET /api/live.jpg?after=N&watch=1`, which counts for nothing",
        "A second page opened on the workstation still replaces the first.",
    ):
        assert piece in view, piece
    assert "it is then the page loaded last" not in view
    for key in ("available", "reason", "fps_asked", "fps_measured", "width", "age_s"):
        assert f'"{key}":' in program and f"`{key}`" in told, key
    for text in ("Map picture", "Live picture", "no live picture"):
        assert f"'{text}'" in block and text in told, text
    assert "`viewer: no live picture: <why>`" in told
    assert 'f"{PROGRAM}: no live picture: {why}.' in program
    said = "`viewer: live picture: the camera's pictures arrive`"
    assert said in told and said.strip("`").split(": ", 1)[1] in program
    step = readme_section("### 10. After a run: the viewer and the run's files")
    step = one_line(step)
    assert "**The page's picture is live.**" in step and "`no live picture`" in step
    assert "`./run_viewer.sh --no-live` starts the viewer without" in step
    # The live arm: what moves live and at which rate, what stays from the
    # recording, and the line under the 3D view, as the launcher's robot.py
    # and the page's block have them.
    robot = (package / "viewer_live" / "robot.py").read_text(encoding="utf-8")
    start = (package / "viewer_live" / "__main__.py").read_text(encoding="utf-8")
    assert "then no live picture, no live arm and no live view" in told
    assert "no live arm either: the page shows" in script
    assert "- **The live arm.**" in told
    rig = yaml.safe_load((CONFIGS / "robot_app_rebot.yaml").read_text())
    assert rig["state"]["publish_period_s"] == 0.05
    assert "20 a second on the rig (`state.publish_period_s` 0.05)" in told
    for piece in (
        "**What moves live, and at which rate**: the camera picture, up to 30 a "
        "second, and the arm with the gripper's collision volumes, 20 a second.",
        "**What stays from the recording, at the page's own rate of twice a "
        "second**: the obstacle map, the refusals",
        "**The line under the 3D view says which arm is drawn.**",
        "**The arm is drawn by the experiment's own code.**",
        "(`RunIndex.geometry`)",
        "`GET /api/live/robot.json?after=N`",
        "none is queued",
        "All five follow the viewer's own access rule",
        "It needs no view token and a watch-only page of another machine reads it",
    ):
        assert piece in told, piece
    assert 'ROBOT_PATH = "/api/live/robot.json"' in program
    assert "const ROBOT = '/api/live/robot.json';" in block
    assert "self._viewer.RunIndex(path)" in robot and ".geometry(q)" in robot
    # The four things the line says, and the two words for no map.
    for words, script_words in (
        ("arm: live, N a second; map: M s old", "`arm: live, ${rate()}`"),
        (
            "arm: last read N s ago (<why>); map: M s old",
            "`arm: last read ${seconds(age)} ago (",
        ),
        ("arm: recorded", "'arm: recorded'"),
        ("arm: no live state: <why>", "`arm: no live state: ${why}`"),
        ("map: no recording", "'no recording'"),
        ("map: none recorded", "'none recorded'"),
    ):
        assert f"`{words}`" in told and script_words in block, words
    assert "STALE_S = 1.0" in robot and "ROBOT_STATE is older than 1 s" in told
    # Live only with fresh readings: the limit, why readings are old, the jaws.
    for piece in (
        "**The arm is called live only while its readings are fresh.**",
        "plus the time since the ROBOT_STATE arrived is at most 1 s",
        "`arm: last read N s ago (torque off: the joints are not read)`",
        "`(joint feedback lost)`",
        "`jaws: last read N s ago (M % open)`",
        "; arm model of run <name>`",
        "the ROBOT_STATE has N joints and the arm model of run <name> has M`",
    ):
        assert piece in told, piece
    assert 'TORQUE_OFF_WHY = "torque off: the joints are not read"' in robot
    assert 'FEEDBACK_LOST = "joint feedback lost"' in robot
    assert "joints and the arm model of run " in robot
    for piece in ("`jaws: last read ${seconds(jaws.age_s)} ago (${open})`",):
        assert piece in block, piece
    # The share of the page's time, and the figures of the rig's recorded run.
    for piece in (
        "**Drawing the arm takes at most about half of the page's time.**",
        "`arm: live, 14 of 20 a second (a draw takes 25 ms); map: M s old`",
        "- **What the live arm costs** grows with the map",
        "whose map has 3,837 cells: 20 states a second drawn",
        "Each answer goes out in one write",
    ):
        assert piece in told, piece
    assert "of ${drawn + missed} a second" in block
    assert "(a draw takes ${Math.round(cost)} ms)" in block
    for key in ("stale_s", "pose_age_s", "last_read", "model_run"):
        assert f'"{key}":' in robot and f"`{key}`" in told, key
    for key in ("n", "available", "why", "age_s", "per_second", "joints", "target_q"):
        assert f'"{key}":' in robot and f"`{key}`" in told, key
    for key in ("gripper", "owner", "stop_level", "motion", "torque_on", "torque"):
        assert f'"{key}":' in robot and f"`{key}`" in told, key
    said = (
        "`viewer: live arm: on the page and at /api/live/robot.json: each "
        "ROBOT_STATE of that robot link service as it arrives; shown, not stored`"
    )
    assert said in told
    assert 'f"{PROGRAM}: live arm: on the page and at {ROBOT_PATH}: each "' in start
    assert "**The page's arm is live as well.**" in step
    assert "`arm: live, N a second; map: M s old`" in step
    assert "`./run_viewer.sh --no-live` starts the viewer without the live arm" in step
    ran = readme_row("| The live arm of the experiment's viewer |")
    assert "`test_viewer_live.py`" in ran and "Not run: with a person" in ran
    folders = one_line(readme_section("### `state_dir` and the run folders"))
    assert f"`{agent_config.RESOLVED_CONFIG}` (APAI's whole configuration" in folders


def test_the_readme_says_what_the_viewer_lacks_against_the_thors_recordings():
    import re
    from pathlib import Path

    from isaaccapture_examples.agentic_remote.agent import config as agent_config
    from isaaccapture_examples.agentic_remote.agent import launcher, self_mask

    told = one_line(readme_section(VIEWER))
    differs = told[told.index("**What differs from the viewer") :]
    # Where the measured joints come from, as the recording names it.
    assert f"(source `{launcher.FEEDBACK_SOURCE}`)" in differs
    assert f"(source `{self_mask.ALIGNMENT_SOURCE}`)" in differs
    assert "receipt on the workstation less the age of its joint readings" in differs
    assert "receipt on the workstation's monotonic clock" in launcher.FEEDBACK_CLOCK
    # What the debug recording leaves out, as the program's first lines say it.
    options = agent_config.RunOptions(timing=True, record_debug=True)
    lines = launcher._option_lines(options, Path("run"))
    [debug] = [line for line in lines if line.startswith("debug recording:")]
    left_out = re.search(r"\(([^)]+)\)", debug).group(1).split(", ")
    assert left_out == ["robot.call", "robot.return", "mask.state_source"]
    for event in left_out:
        assert f"`{event}`" in differs, event
    assert "./run_viewer.sh shows it in a browser" in debug
    assert "robot.feedback, is what this program read from ROBOT_STATE" in debug
    told_options = one_line(readme_section(LAUNCHER_OPTIONS))
    assert "Its joint feedback (`robot.feedback`) is not the motors' own" in (
        told_options
    )
    # No sentence is left that says the feedback is missing, or that nothing
    # shows a run's recordings.
    whole = one_line(README.read_text(encoding="utf-8"))
    assert "holds no motor feedback" not in whole
    assert "the motors' feedback is not in the debug recording" not in whole
    # The other places that name the viewer, and what was not run.
    assert "`./run_viewer.sh`" in readme_row("| The experiment's viewer |")
    gaps = one_line(readme_section("## Known gaps"))
    only = (
        "**The experiment's viewer shows a run only when it was recorded with "
        "`--record-debug`.**"
    )
    assert only in gaps
    never = one_line(readme_section("## What was run")).split(" Never run: ")[1]
    assert "The experiment's viewer in a browser, on port 8766" in never
    assert "Not run: the page in a browser" in told
    assert (TESTS / "test_viewer.py").is_file()
    assert "`test_viewer.py`" in readme_section("## Run the tests")


# The README on setting the rig up. The rig is the real arm, its camera and
# the Thor, the machine on the robot; the workstation is the GPU machine. A
# git bundle is one file of commits that git clones from. The model client is
# the program the agent asks its cloud model through. The calibration's name
# is what both machines call the camera calibration in use.

SETUP = "## Set up and start the rig"
TROUBLE = "### Troubleshooting: what a program printed, and what to do"
RIG_SERVICE_YAML = CONFIGS / "link_service_rebot.yaml"
RIG_AGENT_YAML = CONFIGS / "agent_rebot.yaml"
SETUP_SCRIPT = CONFIGS.parent / "setup_workstation_env.sh"


def troubleshooting() -> str:
    """The table of lines the programs print for a rig that is set up wrongly."""
    setup = readme_section(SETUP)
    return one_line(setup[setup.index(f"\n{TROUBLE}\n") :])


def test_the_readme_quotes_the_service_whose_file_names_no_robot(folder, capsys):
    # The rig's file listens on every address; a list without a peer is refused.
    rig = yaml.safe_load(RIG_SERVICE_YAML.read_text(encoding="utf-8"))
    everywhere = rig["listen"]["address"]
    bad = service_yaml(
        folder, listen={"address": everywhere, "port": 0}, allowed_peers=[]
    )
    assert service_main.main(["--config", bad]) == 2
    printed = capsys.readouterr().err.strip()
    line = printed[: printed.index(", got ")]
    assert line.endswith("allowed_peers must be a non-empty list")
    assert not (folder / "s.sock").exists()
    assert f"| `{line}, ...` |" in troubleshooting()
    edits = one_line(
        readme_section("### 5. The edits each machine needs, which are not committed")
    )
    assert f"`{line}, ...`" in edits
    assert f"(`listen.address: {everywhere}`)" in edits
    assert "**Make each edit in the checkout the programs are started from.**" in edits
    port = rig["listen"]["port"]
    start = one_line(readme_section("### 7. Start the workstation's programs"))
    assert f"waits for the robot app on {everywhere}:{port} (taken from" in start


def test_the_readme_quotes_the_agent_program_that_meets_another_robot_app():
    from types import SimpleNamespace

    from isaaccapture_examples.agentic_remote.agent import arm_adapter
    from isaaccapture_examples.agentic_remote.protocol.messages import PROTOCOL_VERSION

    link = yaml.safe_load(RIG_AGENT_YAML.read_text(encoding="utf-8"))["link"]
    assert link["allow_unnamed_calibration"] is False
    sim = RobotAppConfig.default()  # what the robot app on the simulated arm says
    arm = SimpleNamespace(
        _settings=SimpleNamespace(
            expected_profile_id=link["expected_profile_id"],
            allow_unnamed_calibration=link["allow_unnamed_calibration"],
            calibration_id="stereo-x+mount-y",
        ),
        n_joints=sim.n_joints,
        motion_rate_hz=1.0 / sim.executor.period_s,
    )
    welcome = SimpleNamespace(
        protocol_version=PROTOCOL_VERSION,
        joint_names=["joint"] * sim.n_joints,
        setpoint_period_s=sim.executor.period_s,
        profile_id=sim.profile_id,
        camera_calibration_id=sim.camera_calibration_id,
    )
    check = arm_adapter.RobotLinkArm.check_welcome
    # The rig's agent file against the simulated arm: refused for its profile.
    refusal = check(arm, welcome)
    assert refusal is not None and repr(sim.profile_id) in refusal
    told = troubleshooting()
    assert f"| Agent program: `{refusal}` |" in told
    robot = one_line(readme_section("### 9. Start the robot app on the Thor"))
    assert f"with `{refusal}`" in robot
    assert f"`{refusal}`" in readme_row("| 10 | Thor, then console |")
    # The rig's profile with another calibration: refused for its name.
    welcome.profile_id = link["expected_profile_id"]
    welcome.camera_calibration_id = "stereo-a+mount-b"
    other = check(arm, welcome) or ""
    for name in (welcome.camera_calibration_id, arm._settings.calibration_id):
        assert other.count(repr(name)) == 1, other
        other = other.replace(repr(name), "'...'")
    assert f"| Agent program: `{other}` |" in told
    name = one_line(readme_section("### 6. The camera calibration and its name"))
    assert f"`{other}`" in name
    welcome.camera_calibration_id = arm._settings.calibration_id
    assert check(arm, welcome) is None


def test_the_readme_names_the_model_client_as_the_files_and_the_script_have_it(
    tmp_path,
):
    import re
    from types import SimpleNamespace

    from isaaccapture_examples.agentic_remote.agent import config as agent_config

    step = one_line(readme_section("### 4. The model client on the workstation"))
    # The version the script pins, in the command the README gives for step 4.
    script = SETUP_SCRIPT.read_text(encoding="utf-8")
    pinned = re.search(r'^model_client="([^"]+)"$', script, re.M)
    assert pinned is not None
    by_script = 'npm install --prefix "${env_dir}/codex" --no-audit --no-fund'
    assert f'{by_script} "${{model_client}}"' in one_line(script)
    by_hand = by_script.replace("${env_dir}", "$AGENTIC_REMOTE_ENV_DIR")
    assert f"{by_hand} {pinned.group(1)}" in step
    assert 'refuse "${env_dir}/venv exists;' in script
    # Where the rig's agent file looks for it.
    rig = yaml.safe_load(RIG_AGENT_YAML.read_text(encoding="utf-8"))["experiment"]
    assert rig["codex_bin"].startswith("${AGENTIC_REMOTE_ENV_DIR}/codex/")
    assert f"names `{rig['codex_bin']}`" in step

    def named(path: str) -> Any:
        return SimpleNamespace(experiment=SimpleNamespace(codex_bin=path))

    # An empty key takes the experiment's own; one that names nothing is refused.
    theirs = tmp_path / "theirs"
    assert agent_config._codex_bin(named(""), theirs) == str(theirs)
    with pytest.raises(agent_config.ConfigError) as refused:
        agent_config._codex_bin(named(str(tmp_path / "none")), theirs)
    head = str(refused.value).split(": ")[0]
    assert f"`{head}: ...`" in step


def thor_script(name: str, *arguments: str) -> Any:
    """Run one of the Thor's two start scripts with none of its variables set."""
    import subprocess

    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("AGENTIC_REMOTE_", "APAI_", "PYTHON"))
    }
    return subprocess.run(
        ["bash", str(CONFIGS.parent / name), *arguments],
        env=env,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


def test_the_readme_gives_each_machine_a_file_of_the_variables_its_scripts_need(
    tmp_path,
):
    import re

    step = readme_section("### 3. The variables of each machine, in one file")
    told = one_line(step)
    # The workstation: the six variables of run_workstation.sh, one per line.
    for name in workstation_variables():
        assert f"\n{name}=" in step, name
    assert "`run_workstation.sh --env FILE` reads the file itself" in told
    # The Thor: each of its two start scripts reads such a file when --env
    # FILE is its first argument, and the README's file holds every variable
    # the scripts' heads name.
    both = "`run_robot_app.sh --env FILE` and `run_human.sh --env FILE` read the file"
    assert both in told
    assert "`--env FILE` must be the first argument" in told
    nowhere = tmp_path / "no-python"
    file = tmp_path / "thor.env"
    file.write_text(
        f"AGENTIC_REMOTE_THOR_PYTHON={nowhere}\nAGENTIC_REMOTE_HUMAN_PYTHON={nowhere}\n",
        encoding="utf-8",
    )
    for name in ("run_robot_app.sh", "run_human.sh"):
        script = (CONFIGS.parent / name).read_text(encoding="utf-8")
        head = script[: script.index("\nset -euo pipefail")]
        needs = re.findall(r"^#   (AGENTIC_REMOTE_\w+)", head, re.M)
        assert len(needs) == 3, name
        for variable in needs:
            assert f"\n{variable}=" in step, (name, variable)
        assert f"\n./{name} --env ~/agentic_remote_thor.env " in step, name
        # Without the file the script names the variable it misses.
        bare = thor_script(name, "--help")
        assert (bare.returncode, bare.stdout) == (2, ""), name
        assert bare.stderr.count("\n") == 1 and "(it is '')" in bare.stderr, name
        # With it the value is the file's: the refusal names that Python.
        read = thor_script(name, "--env", str(file), "--help")
        assert (read.returncode, read.stdout) == (2, ""), name
        assert f"(it is '{nowhere}')" in read.stderr, name
        # A file that is not there: one line, and nothing is started.
        missing = thor_script(name, "--env", str(tmp_path / "none.env"))
        assert (missing.returncode, missing.stdout) == (2, ""), name
        line = f"{name}: --env needs a file of KEY=value lines (it is '"
        assert missing.stderr.startswith(line) and missing.stderr.count("\n") == 1
    quoted = "`run_robot_app.sh: --env needs a file of KEY=value lines (it is '...')`"
    assert quoted in told
    # The robot app itself does not get the argument: with a Python that
    # runs, the file's variables start it, and it prints its own help.
    file.write_text(
        f"AGENTIC_REMOTE_THOR_PYTHON={sys.executable}\n"
        f"AGENTIC_REMOTE_APAI_SRC={tmp_path}\n"
        f"AGENTIC_REMOTE_ISAAC_CAPTURE_PY={tmp_path}\n",
        encoding="utf-8",
    )
    started = thor_script("run_robot_app.sh", "--env", str(file), "--help")
    assert started.returncode == 0, started.stderr
    assert "--connect" in started.stdout and "--env" not in started.stdout
    # Anywhere but in front the script does not read the file.
    late = thor_script("run_robot_app.sh", "--help", "--env", str(file))
    assert late.returncode == 2 and "AGENTIC_REMOTE_THOR_PYTHON" in late.stderr


def test_the_readme_builds_isaac_capture_with_settings_the_build_has():
    import re

    root = repo_root()
    step = readme_section("### 2. Build Isaac Capture on the Thor")
    command = step[step.index("```bash") : step.index("\n```\n")]
    flags = re.findall(r"-D([A-Z_]+)=", command)
    assert len(flags) == 4 and len(set(flags)) == 4
    options = (root / "CMakeLists.txt").read_text(encoding="utf-8")
    for flag in flags:
        assert re.search(rf"^\s*(option|set)\({flag}\b", options, re.M), flag
        assert f"| `-D{flag}=" in step, flag
    # The two stops of a configure without them, as the build's files say them.
    told = troubleshooting()
    for file, line in (
        (
            "ClangFormat.cmake",
            "clang-format not found but ENABLE_CLANG_FORMAT_CHECK is ON",
        ),
        (
            "CheckBuildDeps.cmake",
            "glslangValidator (BUILD_VIZ=ON -- compiles Televiz shaders to SPIR-V)",
        ),
    ):
        assert line in (root / "cmake" / file).read_text(encoding="utf-8"), file
        assert f"`{line}`" in told, line
    # The clone: no file of the example is one that Git LFS keeps.
    kept = (root / ".gitattributes").read_text(encoding="utf-8")
    suffixes = set(re.findall(r"^\*(\.\w+) filter=lfs", kept, re.M))
    assert ".png" in suffixes
    example = CONFIGS.parent
    tracked = [
        path
        for folder in (example / "configs", example / "python")
        for path in folder.rglob("*")
        if path.suffix in suffixes
    ]
    assert tracked == [] and not [p for p in example.glob("*") if p.suffix in suffixes]
    commit = one_line(readme_section("### 1. The same commit on both machines"))
    assert "No file under `examples/agentic_remote` is kept in Git LFS" in commit
    assert "GIT_LFS_SKIP_SMUDGE=1 git clone" in commit


def test_the_readme_tells_the_camera_as_its_two_files_and_programs_have_it():
    sender = yaml.safe_load(
        (CONFIGS / "camera_sender_rebot.yaml").read_text(encoding="utf-8")
    )["network"]
    receiver = yaml.safe_load(
        (CONFIGS / "camera_receiver_rebot.yaml").read_text(encoding="utf-8")
    )
    dialed = f"{receiver['receiver']['host']}:{receiver['receiver']['port']}"
    assert sender["port"] == receiver["receiver"]["port"]
    camera = one_line(
        readme_section("### 8. The camera, from the Thor to the workstation")
    )
    told = troubleshooting()
    # The receiver's file names itself as the sender: --host is the Thor's.
    assert receiver["receiver"]["host"] == "127.0.0.1"
    assert f"`waiting for AVS1 sender {dialed} (...)`" in camera
    assert f"| Camera receiver: `waiting for AVS1 sender {dialed} (...)` |" in told
    listening = f"`Listening for TCP receiver on {sender['bind']}:{sender['port']}`"
    assert listening in camera
    assert f"socket={receiver['ipc']['socket_path']}`" in camera
    script = (CONFIGS.parent / "run_camera_receiver.sh").read_text(encoding="utf-8")
    assert 'config="${here}/configs/camera_receiver_rebot.yaml"' in script
    # The texts of the camera streamer and of the experiment folder, where
    # those folders are named: both are outside this repository.
    streamer = os.environ.get("AGENTIC_REMOTE_CAMERA_PR")
    if streamer:
        from pathlib import Path

        sources = "".join(
            path.read_text(encoding="utf-8", errors="replace")
            for folder in ("argus_sender/src", "camera_receiver")
            for path in sorted(Path(streamer, folder).glob("*.*"))
            if path.suffix in (".cpp", ".py")
        )
        for text in (
            "No camera devices are available",
            "Listening for TCP receiver on ",
            "Receiver connected from ",
            "connected to AVS1 sender ",
            "waiting for AVS1 sender ",
        ):
            assert text in sources, text
            assert text.strip() in camera, text
    experiment = os.environ.get("AGENTIC_REMOTE_EXPERIMENT_DIR")
    if experiment:
        from pathlib import Path

        bring_up = Path(experiment, "thor_camera_bringup.sh").read_text(
            encoding="utf-8"
        )
        assert "systemctl start sensing-camera-argus.service" in bring_up
        [last] = [line for line in bring_up.splitlines() if "are ready for" in line]
        assert f"`{last.split(chr(39))[1]}`" in camera
    assert "sudo systemctl stop sensing-camera-argus.service" in camera


# The README on rest, release and the motors on again, on the question about
# the jaws, on the speeds of the rig and on the debug recording's size. A
# typed rest returns the arm to rest, closes the jaws and switches the motors
# off there; "on" switches them on again without a new start of the robot
# app. The rig files are the robot app's and the agent program's files for
# the real arm. The debug recording is what a run of the agent program
# started with --record-debug keeps for the experiment's viewer.

COMMANDS = "## What the operator's commands do"
RIG_FILES = "## The rig files"


@link
def test_the_readme_quotes_what_rest_on_release_and_quit_print_at_the_thor_terminal(
    programs,
):
    from isaaccapture_examples.agentic_remote.robot_app import app as app_module
    from isaaccapture_examples.agentic_remote.robot_app import terminal

    # No robot link service listens: the Thor terminal needs none.
    app = Program(ROBOT_MODULE, *SIM, "--connect", f"{HOST}:{free_port()}")
    programs.append(app)
    app.wait_line("dialing the robot link service at ")
    app.type("rest")
    app.wait_line(terminal.CONFIRM_REST)
    app.type("path is clear")
    app.wait_line("rest: completed")
    app.type("on")
    app.wait_line(terminal.CONFIRM_POWER_ON)
    app.type("yes")
    app.wait_line("on: completed")
    app.type("on")  # the motors are on: nothing is asked, so nothing confirms
    app.wait_line("on: refused: ")
    app.type("release")
    app.wait_line("release: completed")
    app.type("quit")
    assert app.wait_exit() == 0
    said = [line for line in app.lines if not line.startswith("robot app: ")]
    assert said[0].startswith(terminal.BANNER)
    # Every line of the terminal, in its order: the rest ends with the jaws
    # closed and the motors off, "on" asks first, and a quit with the motors
    # off at rest asks nothing and releases nothing more.
    jaws = [
        f"[{rest_module.EVENT_JAWS}] {rest_module.CLOSING_JAWS}",
        f"[{rest_module.EVENT_JAWS}] {rest_module.JAWS_CLOSED}",
        f"[{rest_module.EVENT_RELEASE}] {rest_module.RELEASED}",
    ]
    why = app_module.CONFIRM_POWER_ON
    assert said[1:-1] == [
        terminal.CONFIRM_REST,
        "[rest] return to rest: the arm is at rest, motors on",
        *jaws,
        "rest: completed",
        terminal.CONFIRM_POWER_ON,
        f"[rest] {rest_module.MOTORS_ON_AGAIN}: motion rest_hold",
        "on: completed",
        f"[refused] on typed at the Thor terminal refused: {why}",
        f"on: refused: {why}",
        *jaws,
        "release: completed",
        "[shutdown] shutdown: the motors are off at rest; the robot app closes",
        "quit: completed; the robot app closes",
    ]
    assert said[-1].startswith(terminal.INPUT_ENDED)
    told = one_line(readme_section(COMMANDS))
    typed = ("[", "rest:", "on:", "release")
    quoted = [line for line in said if line.startswith(typed)]
    first = [terminal.BANNER, terminal.CONFIRM_POWER_ON, "quit: completed; "]
    for line in [*first, *quoted]:
        if line.startswith(("[shutdown]", "[refused]")):
            continue  # the shutdown's line is in the recipes; a refusal's twice
        assert f"`{line}" in told, line


def test_the_readme_quotes_the_questions_of_rest_on_and_the_jaws_as_they_are_asked():
    from isaaccapture_examples.agentic_remote.agent import console_cmd, task_gate
    from isaaccapture_examples.agentic_remote.robot_app import app, terminal

    commands = one_line(readme_section(COMMANDS))
    console = one_line(readme_section("### The operator console"))
    task = one_line(readme_section("### A task from its start to its end"))
    # The question about the jaws, at the Thor terminal and at the console.
    assert f"`{terminal.ASK_JAWS_EMPTY.format(what='rest')}`" in commands
    assert f"`{console_cmd.ASK_JAWS_EMPTY.format(command='rest')}`" in console
    assert f"`{console_cmd.JAWS_NOT_CONFIRMED}`" in console
    word = app.JAWS_EMPTY_WORD.format(what="a return to rest")
    assert f"`[jaws] {word}`" in commands and f"`{word}`" in console
    # The lines of a rest that the simulated arm at rest does not print: the
    # legs of APAI's route, a release the driver only sent, jaws left as they
    # were, a takeover that keeps the motors on, and motors that stay on.
    fold = rest_module.LEG_FOLD.format(step="N", steps=4)
    for line in (
        f"[rest] return to rest: {rest_module.LEG_UP}: the arm moves for ",
        f"[rest] return to rest: {fold}: the arm moves for ",
        f"[release] {rest_module.RELEASE_SENT}`",
        "[jaws] " + rest_module.JAWS_LEFT.format(why="<why>") + "`",
        f"[rest] return to rest: {app.REST_KEEPS_MOTORS}`",
        f"rest: failed: {app.AT_REST_MOTORS_ON}; release refused: <why>`",
        "[refused] " + rest_module.NOT_ON_AGAIN.format(why="<the driver's error>"),
    ):
        assert f"`{line}" in commands, line
    # What the robot app's process says of a SIGTERM; the process tests of
    # this file wait for the same lines.
    for line in (SIGTERM_AT_REST, SIGTERM_MOTORS_ON, f"{SIGTERM_WHY}<why>"):
        assert f"`{line}`" in commands, line
    # With the motors off already, and after a torque release that failed, a
    # SIGTERM asks the robot app for nothing: this stand-in has no release.
    for state, off in (
        (MotionState.REST_UNPOWERED, True),
        (MotionState.RELEASE_FAILED, False),
    ):
        stand_in = types.SimpleNamespace(motion_state=state, motors_off_at_rest=off)
        [line] = robot_main._sigterm_lines(stand_in)
        assert f"`{line}`" in commands, line
    # The motors on again: the console's question, and what refuses meanwhile.
    assert f"`{console_cmd.ASK_ON}`" in console
    for told in (commands, console):
        assert f"`{app.MOTORS_OFF}`" in told
    assert console_cmd.question("on", None) == (console_cmd.ASK_ON, "yes")
    # What the console command says when it stops waiting for a RESULT.
    command = one_line(readme_section("### The console command"))
    assert f"(...). {console_cmd.STILL_LIVE}`" in command
    waits = console_cmd.RESULT_WAIT_S
    assert f"`rest` {waits['rest']:g} s, `shutdown` {waits['shutdown']:g} s" in command
    assert {waits["on"], waits["open"], waits["close"]} == {60.0}
    assert "`on`, `open` and `close` 60 s" in command
    # The agent program's two questions and what it says of the arm's end.
    asked = f"{task_gate.MOTORS_OFF_NOTE} {task_gate.CONNECT_QUESTION} Type yes to"
    assert f"`{asked} proceed: `" in task
    assert f"`{task_gate.END_QUESTION}`" in task
    assert f"`{task_gate.CONNECT_DECLINED}`" in task
    assert f"; {task_gate.END_HOLDS}`" in task
    assert f"{int(task_gate.PARK_WAIT_S)} s for its RESULT" in task
    refusal = task_gate.NOT_ADMITTED.format(name="<skill>")
    assert f"`{refusal}: <why>`" in task
    # The rest tool is one of the skills that move, and what the model is told.
    from isaaccapture_examples.agentic_remote.agent import launcher

    assert f"the rest tool `{task_gate.REST_SKILL}` among them" in task
    assert f"`{launcher.TOOLS_GUIDANCE}`" in task
    assert task_gate.REST_SKILL not in launcher.TOOLS_GUIDANCE
    # What the three request words are, and who may send them.
    from isaaccapture_examples.agentic_remote.link_service import service
    from isaaccapture_examples.agentic_remote.protocol import messages
    from isaaccapture_examples.agentic_remote.protocol.local_messages import (
        ROLE_AGENT,
        ROLE_CONSOLE,
    )

    words = (
        messages.REQUEST_MOTORS_OFF,
        messages.REQUEST_MOTORS_ON,
        messages.REQUEST_PARK,
    )
    assert words == ("motors_off", "motors_on", "park")
    assert set(words) <= service.OWNER_REQUESTS[ROLE_AGENT]
    assert service.OWNER_REQUESTS[ROLE_CONSOLE] & set(words) == {"motors_on"}
    roles = readme_section("### The local API and its roles")
    [agent] = [line for line in roles.splitlines() if line.startswith("| `agent` |")]
    [console_row] = [
        line for line in roles.splitlines() if line.startswith("| `console` |")
    ]
    for word in words:
        assert f'"{word}"' in agent, word
    assert '"motors_on"' in console_row and '"park"' not in console_row
    assert '"motors_off"' not in console_row


class PlainConfig:
    """Stands in for APAI's configuration: a mapping that gives itself as a dict."""

    def __init__(self, data: dict) -> None:
        self.data = data

    def as_dict(self) -> dict:
        import copy

        return copy.deepcopy(self.data)


def test_the_readme_gives_the_rig_speeds_as_the_two_rig_files_hold_them(monkeypatch):
    from isaaccapture_examples.agentic_remote.agent import config as agent_config

    for name in agent_config.PATH_VARIABLES:
        monkeypatch.setenv(name, "/absolute")
    robot = yaml.safe_load(RIG_ROBOT_YAML.read_text(encoding="utf-8"))
    agent = yaml.safe_load(RIG_AGENT_YAML.read_text(encoding="utf-8"))
    limits, rest = robot["limits"], robot["rest"]
    # One speed for the agent and the return to rest, on both machines.
    speeds = {
        "limits.agent.max_joint_speed": limits["agent"]["max_joint_speed"],
        "limits.recovery.max_joint_speed": limits["recovery"]["max_joint_speed"],
        "rest.recovery_speed": rest["recovery_speed"],
        "thor.agent_max_joint_speed": agent["thor"]["agent_max_joint_speed"],
    }
    # The rig owner's decisions of 2026-11-16: 0.2 rad/s there, 0.8 for a person.
    assert speeds == dict.fromkeys(speeds, 0.2)
    speed, human = 0.2, limits["human"]["max_joint_speed"]
    assert human == 0.8
    rig = one_line(readme_section(RIG_FILES))
    heading = (
        f"**The speeds: the agent and the return to rest run at {speed:g} rad/s, and "
        f"a person's drive is capped at {human:g} rad/s.**"
    )
    assert heading in rig
    for key in speeds:
        assert f"`{key}`" in rig, key
    held = f"`limits.human.max_joint_speed` of the robot app's file holds the {human:g}"
    assert held in rig
    assert "**Change the speeds on both machines together.**" in rig
    assert "**None of these speeds has run on the arm.**" in rig
    # The agent never plans faster than the Thor's own limit, whatever its
    # file says: the line it prints, on the file of the mock recipe, whose two
    # speeds are higher than the rig's.
    mock = agent_config.load_settings(CONFIGS / "agent_mock.yaml")
    theirs = mock.thor.agent_max_joint_speed
    assert mock.minimal and theirs > speed
    built = PlainConfig({"safety": {"max_joint_vel": theirs}})
    agreed, capped, line = agent_config.agree_joint_speed(mock, built, speed)
    assert agreed.thor.agent_max_joint_speed == speed
    assert capped.data["safety"]["max_joint_vel"] == speed
    head = f"joint speed: the agent plans at {speed:g} rad/s, the lowest of "
    tail = line[line.index("; the Thor's own agent speed limit is ") :]
    assert line.startswith(head) and f"is {speed:g} (limits.agent." in tail
    assert f"`{head}" in rig and f"(thor.agent_max_joint_speed){tail}`" in rig
    # A robot app that names no limit: the file's key stands for it.
    alone, _, _ = agent_config.agree_joint_speed(mock, built, None)
    assert alone.thor.agent_max_joint_speed == theirs
    # The idle joint reads, as often as the file has them.
    period = robot["state"]["idle_read_period_s"]
    assert f"`state.idle_read_period_s: {period:g}`" in rig
    # What ran on the arm is said once, at the top, with its speed.
    opening = one_line(README.read_text(encoding="utf-8")).split(" ## Terms ")[0]
    assert "**One thing ran on the real arm**" in opening
    assert "a pick, a lift, a hold and a place of an object, at 0.1 rad/s" in opening


def test_the_readme_lists_the_motors_state_as_the_programs_print_it():
    from isaaccapture_examples.agentic_remote.robot_app import app, text

    # The six texts, by what ROBOT_STATE says of the motors and of the hold.
    said = {
        text.torque_text(True, text.MOTORS_HOLDING): "torque on (holding)",
        text.torque_text(True, text.MOTORS_NOT_HOLDING): "torque NOT HOLDING",
        text.torque_text(True, text.MOTORS_UNKNOWN): "torque on, hold unknown",
        text.torque_text(True, None): "torque on",
        text.torque_text(False, text.MOTORS_HOLDING): "torque off",
        text.torque_text(True, text.MOTORS_HOLDING, True): "torque unknown",
    }
    assert list(said) == list(said.values())
    console = one_line(readme_section("### The operator console"))
    assert "one of six texts" in console
    for line in said:
        assert f"| `{line}` |" in console, line
    words = (text.MOTORS_HOLDING, text.MOTORS_NOT_HOLDING, text.MOTORS_UNKNOWN)
    carried = one_line(
        readme_section(
            "### What WELCOME, ROBOT_STATE and VIEW_STATUS carry for the operator"
        )
    )
    assert "(" + ", ".join(f"`{word}`" for word in words[:2]) in carried
    assert f"or `{words[2]}`) and `reason`" in carried
    # The real driver's line after its connect, and the line on the bus.
    real = one_line(readme_section("### The real driver"))
    unknown = text.torque_text(True, text.MOTORS_UNKNOWN)
    assert f"at measured rest: {unknown}, motion hold, jaws referenced`" in real
    reads = app.READ_TIMING_READS
    assert f"measures its first {reads} idle joint reads" in real
    assert f"on the motor bus (median of {reads} idle reads; worst Y ms)" in real
    period = yaml.safe_load(RIG_ROBOT_YAML.read_text(encoding="utf-8"))["state"]
    asked = 1.0 / period["idle_read_period_s"]
    assert f"state.idle_read_period_s asks for {asked:.0f}`" in real


def test_the_readme_gives_the_debug_recordings_size_limit_and_how_to_raise_it(
    monkeypatch,
):
    from isaaccapture_examples.agentic_remote.agent import config as agent_config
    from isaaccapture_examples.agentic_remote.agent import launcher

    limit = agent_config.DEBUG_MAX_GIB
    for name in agent_config.PATH_VARIABLES:
        monkeypatch.setenv(name, "/absolute")
    rig = agent_config.load_settings(CONFIGS / "agent_rebot.yaml")
    parser = launcher._parser()

    def run(*arguments: str) -> Any:
        args = parser.parse_args(["--config", "unused", "--record-debug", *arguments])
        return launcher._run_options(args, rig)

    # The start script records every run and names no size: the default holds.
    script = one_line(WORKSTATION_SCRIPT.read_text(encoding="utf-8"))
    assert " --record-debug " in script and "--debug-max-gib" not in script
    assert run().debug_max_gib == limit
    assert run("--debug-max-gib", "64").debug_max_gib == 64.0
    options = one_line(readme_section(LAUNCHER_OPTIONS))
    after = one_line(
        readme_section("### 10. After a run: the viewer and the run's files")
    )
    assert f"`--debug-max-gib`, {limit:g} GiB by default" in options
    assert f"`--debug-max-gib`, {limit:g} GiB by default" in after
    for told in (options, after):
        assert "`--debug-max-gib 64`" in told
        # One GiB is 2**30 bytes: the limit in the unit a disk is sold in.
        assert f"about {limit * 2**30 / 1e9:.0f} GB with the default" in told
        assert "by hand" in told
    assert f"**The debug recording stops at {limit:g} GiB" in after
    assert f"reached its size limit, {limit:g} GiB by default" in troubleshooting()


def test_the_readme_names_the_planned_paths_among_the_run_folders_files():
    from isaaccapture_examples.agentic_remote.agent import (
        debug_recording,
        launcher,
        move_paths,
    )

    name = launcher.MOVE_PATHS_FILE
    assert name == "move-paths.jsonl" and name in launcher.MEASURE_FILES
    files = (*launcher.MEASURE_FILES, launcher.SUMMARY_FILE)
    # The three places that list the files of a run folder name all seven.
    folders = one_line(readme_section("### `state_dir` and the run folders"))
    measures = one_line(readme_section("### What a run records and measures"))
    step = one_line(
        readme_section("### 10. After a run: the viewer and the run's files")
    )
    for told in (folders, measures, step):
        for file in files:
            assert f"`{file}`" in told, file
    assert len(files) == 7 and "the seven files of" in folders
    assert "Seven files go into the run folder" in measures
    assert "**Seven files are written in every run**" in step
    assert "in the six `.jsonl` files" in measures
    # The file's row, and each field of a line by the name the program writes.
    row = readme_row(f"| `{name}` |")
    assert "MOVE the agent sent, or could not send" in row
    line = move_paths.path_row({"move_id": "m", "setpoints": []}, None)
    end = move_paths.end_row({}, "", 0.0)
    for key in (*line, "setpoints", *end):
        assert f"`{key}`" in measures, key
    assert f"`path_file`, in a line with a `move_id`: `{name}`" in measures
    for key in ("scout_opening", "timeout_s", "sent_mono_s", "sent_utc", "skill"):
        assert f"`{key}`" in readme_row("| `jaw-commands.jsonl` |"), key
    # What a re-run can take from it, and what it cannot.
    for said in (
        "**The planned path of a move, in `move-paths.jsonl`.**",
        "**how many of the setpoints it sent to the motors**",
        "The setpoints `0` to `sent - 1` are what the robot app commanded",
        "**The move is not delayed by it**",
        "**What a run's record is good for, and what it is not.**",
        "(`model-trace/`, with the full stack)",
        "(`debug/`, with `--record-debug`)",
        "What the arm did is in `robot-state.jsonl`",
        "**It is not a closed-loop re-run.**",
        "that needs a simulator or the rig",
        "**Not run on the rig**",
    ):
        assert said in measures, said
    assert move_paths.CHUNK == 16 and "16 setpoints at a time" in measures
    # The event of the debug recording, by its name and by what it says it is.
    event = move_paths.PATH_EVENT
    assert f"`{event}` is one event per MOVE" in measures
    assert "the **planned** path the agent sent in one MOVE" in measures
    assert "PLANNED joint path of one MOVE" in move_paths.PATH_NOTE
    assert "and no such event is made up" in measures
    options = one_line(readme_section(LAUNCHER_OPTIONS))
    assert f"`{event}` (the planned joint path of each MOVE)" in options
    # The rule for the wait for a task, where the debug recording is told.
    assert "**While the program waits for a task.**" in options
    assert f"`{debug_recording.IDLE_OPTION} 0` stores none" in options
    for key in ("idle_map_s", "idle_seconds", "idle_skipped", "debug_idle_map_s"):
        assert f"`{key}`" in options, key
    assert f"`{debug_recording.IDLE_OPTION} SECONDS` sets that time" in step


def test_the_readme_tells_when_a_wait_is_thinned_and_how_a_path_line_is_read():
    from isaaccapture_examples.agentic_remote.agent import debug_recording

    readme = README.read_text(encoding="utf-8")
    options = one_line(readme_section(LAUNCHER_OPTIONS))
    measures = one_line(readme_section("### What a run records and measures"))
    step = one_line(
        readme_section("### 10. After a run: the viewer and the run's files")
    )
    # A wait for a task is thinned only while nothing happens at the arm, by
    # the numbers the program has; a person's takeover is not.
    assert "\n## Demo 2: a person takes over\n" in readme
    travel, jaw = debug_recording.ARM_TRAVEL_RAD, debug_recording.ARM_JAW_TRAVEL
    quiet, age = debug_recording.ARM_QUIET_S, debug_recording.ARM_STATE_AGE_S
    for said in (
        "and nothing happens at the arm: the newest ROBOT_STATE shows nobody as "
        "the control owner, and the arm does not move",
        f"a measured joint is more than {travel:g} rad from the pose the arm last "
        "stood at",
        f"the jaw opening is more than {jaw:g} of its range from where it stood",
        "**As soon as a ROBOT_STATE shows an owner (the agent, a person, recovery) "
        "or motion, the rate of `--debug-map-hz` is back**: the next map refresh "
        "is stored at once",
        f"without an owner and without motion for {quiet:g} s while the program "
        "still waits",
        "**a takeover in [Demo 2](#demo-2-a-person-takes-over) is recorded at the "
        "normal rate**",
        f"to {quiet:g} s after the hand-back the recording stores "
        f"{debug_recording.DEBUG_MAP_HZ:g} map pictures a second",
        f"**Without a ROBOT_STATE newer than {age:g} s**",
        "takes the safe side for the record: the rate of `--debug-map-hz`",
        "a person who moves an object by hand while nobody owns the arm and it "
        "stands is recorded once per idle period",
    ):
        assert said in options, said
    for word in debug_recording.ARM_STILL:
        assert f"`{word}`" in options, word
    # What the recording says of it, by the names of its first event and of
    # its status block.
    causes = (
        debug_recording.ARM_OWNER,
        debug_recording.ARM_MOTION,
        debug_recording.ARM_NO_STATE,
    )
    assert causes == ("owner", "motion", "no_state")
    for key in (
        "arm_quiet_s",
        "arm_state_age_s",
        "arm_travel_rad",
        "arm_jaw_travel",
        "wait_seconds",
        "idle_interrupted",
        "idle_interrupted_seconds",
        "idle_interrupted_by",
        *causes,
    ):
        assert f"`{key}`" in options, key
    assert "nobody owns or moves the arm" in step
    assert "a person who takes the arm over in that wait is recorded at 2 a" in step
    # A line of move-paths.jsonl is read by its count, not by its status.
    for said in (
        "**Go by `end.sent`, not by `end.status`**",
        "ends `refused` with `sent` above 0",
        "`sent` is `null` when the RESULT carried no count",
        "refused at its admission, none was sent",
        "at least `sent_seen` were",
        # What the hand-over costs a move, as it was measured.
        "some tens of millionths of a second per move",
        "a median of 24 and at most 49 were measured",
        # One failed write shuts the file, as it shuts the others.
        "After one failed write `move-paths.jsonl` is shut for the rest of the "
        "run, as the other measurement files are",
        # The files exist for a real run of the rig.
        "the six measurement files exist for a real run",
        "The run folder `20261118T022210Z-3147525` of the second session",
    ):
        assert said in measures, said
    assert "2 to 3 millionths" not in readme
    assert "has been seen yet" not in measures
