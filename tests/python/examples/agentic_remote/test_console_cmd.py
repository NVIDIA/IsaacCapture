# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""console_cmd: one operator command through the robot link service, then exit.

Most tests run the command against a real service whose robot side the test
scripts, so every link message the service forwarded is seen. The last ones put
it in front of the real robot app on a link of 127.0.0.1.
"""

from __future__ import annotations

import ast
import dataclasses
import os
import pty
import select
import signal
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
from isaaccapture_examples.agentic_remote.agent import console_cmd
from isaaccapture_examples.agentic_remote.protocol.messages import (
    Message,
    Open,
    OwnerRequest,
    Release,
    Reset,
    Result,
    Shutdown,
    Stop,
    decode,
)
from link_bench import Bench, until, welcome

link = pytest.mark.link

SOURCE = Path(console_cmd.__file__)
# The numbers of the newest ROBOT_STATE in these tests.
NUMBERS = {"session_id": "s1", "owner_number": 4, "stop_number": 3}
# What each command is on the link: the link message and what it must carry.
ON_THE_LINK: dict[str, tuple[type, dict[str, Any]]] = {
    "reset": (Reset, {}),
    "rest": (OwnerRequest, {"request": "rest", "confirmed": True}),
    "open": (Open, {"confirmed": True}),
    "release": (Release, {}),
    "shutdown": (Shutdown, {"confirmed": True}),
}
CONFIRMED = ("rest", "open", "release", "shutdown")

Reply = Callable[[Message], list[Message]]


def completed(message: Message) -> list[Message]:
    return [Result(message.command_id, "completed", detail={"owner": "nobody"})]


class Console:
    """Runs console_cmd on a service whose robot side answers as scripted."""

    def __init__(self, bench: Bench, robot: bool = True) -> None:
        self.bench = bench
        self.thor = bench.robot(None) if robot else None
        if self.thor is not None:
            self.thor.send(welcome())
            self.state = self.thor.state(owner_number=4, stop_number=3)
            until(lambda: bench.service.robot_state == self.state, "the ROBOT_STATE")
        self.forwarded: list[Message] = []  # what the service sent to the robot
        self.said: list[str] = []
        self.asked: list[str] = []
        self.answer = "yes"
        self.roles: list[str] = []  # the role of every attach
        self.reply: Reply = completed
        self._off = threading.Event()
        self._thread = threading.Thread(target=self._answer, daemon=True)
        self._thread.start()

    def _answer(self) -> None:
        while not self._off.is_set():
            got = self.thor.carrier.receive(0.02) if self.thor else None
            if got is None:
                self._off.wait(0.02 if self.thor is None else 0)
                continue
            message = decode(*got)
            self.forwarded.append(message)
            for answer in self.reply(message):
                self.thor.send(answer)

    def ask(self, prompt: str) -> str:
        self.asked.append(prompt)
        return self.answer

    def run(self, *argv: str, ask: Any = None) -> int:
        real = console_cmd.connect_local

        def attach(path: str | None, role: str, name: str, timeout_s: float) -> Any:
            self.roles.append(role)
            return real(path, role, name, timeout_s)

        console_cmd.connect_local = attach
        try:
            return console_cmd.main(
                ["--socket", self.bench.path, *argv],
                ask=ask or self.ask,
                say=self.said.append,
            )
        finally:
            console_cmd.connect_local = real

    def settled(self) -> list[Message]:
        """What was forwarded, once nothing more comes."""
        count = -1
        while count != len(self.forwarded):
            count = len(self.forwarded)
            self._off.wait(0.1)
        return self.forwarded

    def close(self) -> None:
        self._off.set()
        self._thread.join(5.0)


@pytest.fixture
def bench(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Bench]:
    made = Bench(tmp_path_factory.mktemp("cc"))
    yield made
    made.close()


@pytest.fixture
def console(bench: Bench) -> Iterator[Console]:
    made = Console(bench)
    yield made
    made.close()


# One command, with the console role


@pytest.mark.parametrize("command", list(ON_THE_LINK))
def test_a_command_is_one_link_message_with_the_console_role_and_its_result_is_printed(
    console: Console, command: str
) -> None:
    kind, carries = ON_THE_LINK[command]
    assert console.run(command) == console_cmd.EXIT_OK
    (message,) = console.settled()  # exactly one
    assert type(message) is kind
    # It quotes the numbers of the newest ROBOT_STATE.
    for name, value in {**NUMBERS, **carries}.items():
        assert getattr(message, name) == value
    assert console.roles == ["console"]
    assert console.said[-1] == f'{command}: RESULT completed {{"owner": "nobody"}}'
    # Only a console may send these: the service forwarded it for that role.
    assert console.bench.service.robot_state == console.state


@pytest.mark.parametrize("command", CONFIRMED)
def test_rest_open_release_and_shutdown_ask_for_a_typed_yes_first(
    console: Console, command: str
) -> None:
    assert console.run(command) == console_cmd.EXIT_OK
    (prompt,) = console.asked
    assert prompt == console_cmd.CONFIRMATIONS[command] and "Type yes" in prompt
    words = {
        "rest": "Is the path clear?",
        "open": "whatever they hold is let go",
        "release": "only with the arm measured at rest",
        "shutdown": "is the path clear?",
    }
    assert words[command] in prompt


@pytest.mark.parametrize("command", CONFIRMED)
@pytest.mark.parametrize("answer", ["", "no", "y", "yes."])
def test_without_the_typed_yes_nothing_is_attached_or_sent(
    console: Console, command: str, answer: str
) -> None:
    console.answer = answer
    assert console.run(command) == console_cmd.EXIT_FAILED
    assert console.said == [f"{command} not confirmed; nothing sent"]
    assert console.roles == [] and console.settled() == []


@pytest.mark.parametrize("command", CONFIRMED)
def test_a_confirmation_cannot_come_from_a_pipe(console: Console, command: str) -> None:
    # Under pytest there is no terminal on stdin: the real prompt refuses.
    code = console.run(command, ask=console_cmd.terminal_ask)
    assert code == console_cmd.EXIT_FAILED
    assert console.said == [
        f"{command} not confirmed (this command needs a terminal to type the "
        "confirmation at); nothing sent"
    ]
    assert console.roles == [] and console.settled() == []


def on_a_terminal(
    child: Callable[[], Any],
    typed_before: bytes = b"",
    typed_after: bytes = b"",
    marker: bytes = b"Type yes",
    delay_s: float = 0.5,
) -> tuple[str, bool]:
    """Run child() in a new process whose terminal this test owns.

    typed_before is typed at once and child() starts delay_s later;
    typed_after is typed once `marker` showed on the terminal. Returns (the
    repr of what child() returned or raised, whether the marker showed).
    """
    result = tempfile.NamedTemporaryFile(prefix="on-a-terminal-", delete=False)
    result.close()
    pid, master = pty.fork()
    if pid == 0:  # stdin, stdout and the controlling terminal are the new one
        try:
            # pytest holds sys.stdin and sys.stdout; the terminal is 0 and 1.
            sys.stdin = open(0, closefd=False)  # noqa: SIM115
            sys.stdout = open(1, "w", closefd=False)  # noqa: SIM115
            time.sleep(delay_s)
            text = repr(child())
        except BaseException as problem:  # noqa: BLE001
            text = f"raised {problem!r}"
        try:
            Path(result.name).write_text(text)
        finally:
            os._exit(0)
    try:
        os.write(master, typed_before)
        seen, answered, ended = b"", False, False
        end = time.monotonic() + delay_s + 10.0
        while time.monotonic() < end and not ended:
            ready, _, _ = select.select([master], [], [], 0.05)
            if ready:
                try:
                    seen += os.read(master, 4096)
                except OSError:  # the child closed the terminal
                    pass
            if marker in seen and not answered:
                answered = True
                time.sleep(0.1)  # the child reads by now
                os.write(master, typed_after)
            ended = os.waitpid(pid, os.WNOHANG)[0] != 0
        if not ended:
            os.kill(pid, signal.SIGKILL)
            os.waitpid(pid, 0)
        return Path(result.name).read_text(), marker in seen
    finally:
        os.close(master)
        os.unlink(result.name)


def release_at_a_terminal(tmp_path: Path, agent_marker: str = "") -> str:
    """console_cmd release, with no service to attach to; for on_a_terminal()."""
    said: list[str] = []
    for name in list(os.environ):
        if console_cmd.agent_environment({name: os.environ[name]}):
            del os.environ[name]
    if agent_marker:
        os.environ[agent_marker] = "1"
    code = console_cmd.main(
        ["release", "--socket", str(tmp_path / "no-service.sock")], say=said.append
    )
    return f"exit {code}: {said}"


def test_a_yes_typed_before_the_question_does_not_answer_it(tmp_path: Path) -> None:
    def run() -> str:
        return release_at_a_terminal(tmp_path)

    # Typed ahead, the yes is dropped; what is typed at the question counts.
    answer, shown = on_a_terminal(run, b"yes\n", b"no\n")
    assert shown and "release not confirmed; nothing sent" in answer
    answer, shown = on_a_terminal(run, b"", b"yes\n")
    assert shown and "cannot attach to the robot link service" in answer
    answer, shown = on_a_terminal(run, b"no\n\n", b"yes\n")
    assert shown and "cannot attach to the robot link service" in answer


@pytest.mark.parametrize("marker", ["CLAUDECODE", "AI_AGENT", "OPENCLAW_HOME"])
def test_a_confirmation_is_not_asked_in_an_agents_environment(
    tmp_path: Path, marker: str
) -> None:
    answer, shown = on_a_terminal(
        lambda: release_at_a_terminal(tmp_path, marker), b"yes\n", b"yes\n"
    )
    assert not shown  # the question was never put
    assert "release not confirmed (the confirmation is typed by the operator" in answer
    assert "outside an agent environment); nothing sent" in answer
    assert "exit 1: " in answer


def test_the_marks_of_an_agents_environment() -> None:
    marked = console_cmd.agent_environment
    assert marked({"CLAUDECODE": "1", "PATH": "/bin"}) == "CLAUDECODE"
    assert marked({"OPENCLAW_ANYTHING": "x"}) == "OPENCLAW_ANYTHING"
    assert marked({"CLAUDECODE": " ", "PATH": "/bin"}) is None
    assert marked({}) is None


@pytest.mark.apai
def test_the_marks_of_an_agents_environment_are_apais() -> None:
    import importlib

    arm_hold = importlib.import_module("apai.control.arm_hold")
    assert console_cmd.AGENT_ENV == arm_hold.AGENT_ENV
    assert console_cmd.AGENT_ENV_PREFIXES == arm_hold.AGENT_ENV_PREFIXES


@pytest.mark.parametrize("command", ["state", "stop", "reset"])
def test_state_stop_and_reset_ask_nothing(console: Console, command: str) -> None:
    console.reply = lambda message: (
        [dataclasses.replace(console.state, seq=9, stop_number=4, stop_level="stop")]
        if isinstance(message, Stop)
        else completed(message)
    )
    assert console.run(command) == console_cmd.EXIT_OK
    assert console.asked == []


def test_a_refusal_is_printed_with_its_reason_and_ends_with_exit_code_1(
    console: Console,
) -> None:
    reason = "control owner is agent, not nobody"
    console.reply = lambda message: [Result(message.command_id, "refused", reason)]
    assert console.run("open") == console_cmd.EXIT_FAILED
    assert console.said[-1] == f"open: RESULT refused: {reason}"
    assert len(console.settled()) == 1


def test_the_result_waited_for_is_the_one_with_the_commands_own_id(
    console: Console,
) -> None:
    def reply(message: Message) -> list[Message]:
        return [
            Result("another-console-7", "completed"),  # a console gets every RESULT
            Result(message.command_id, "accepted"),
            Result(message.command_id, "failed", "the return to rest was stopped"),
        ]

    console.reply = reply
    assert console.run("rest") == console_cmd.EXIT_FAILED
    assert console.said[-2].startswith("accepted; waiting for the end")
    assert console.said[-1] == "rest: RESULT failed: the return to rest was stopped"


def test_no_answer_ends_the_command_after_its_wait(console: Console) -> None:
    console.reply = lambda message: []
    assert console.run("reset", "--timeout", "0.3") == console_cmd.EXIT_FAILED
    assert console.said[-1].startswith("reset: no final RESULT within 0.3 s")
    assert len(console.settled()) == 1  # sent once, not again


# STOP and state


def test_stop_is_one_stop_from_the_console_and_a_higher_stop_number_answers_it(
    console: Console,
) -> None:
    stopped = dataclasses.replace(
        console.state, seq=9, stop_number=4, stop_level="stop", motion="hold"
    )
    console.reply = lambda message: [stopped]
    assert console.run("stop") == console_cmd.EXIT_OK
    (message,) = console.settled()
    assert message == Stop(reason=console_cmd.STOP_REASON, sender="console")
    assert console.roles == ["console"]
    assert console.said[-1].startswith("stop: taken. ROBOT_STATE: control owner nobody")
    assert "stop number 4, stop level stop, motion hold" in console.said[-1]


def test_a_stop_the_robot_does_not_show_is_said_so(console: Console) -> None:
    console.reply = lambda message: []
    assert console.run("stop", "--timeout", "0.3") == console_cmd.EXIT_FAILED
    assert "no ROBOT_STATE showed a higher stop number within 0.3 s" in console.said[-1]
    assert len(console.settled()) == 1


def test_a_stop_without_a_robot_connection_prints_the_services_notice(
    bench: Bench,
) -> None:
    console = Console(bench, robot=False)
    try:
        assert console.run("stop") == console_cmd.EXIT_FAILED
        assert console.said[-1].startswith("stop: STOP was not delivered")
        # And a command that quotes numbers is not sent at all.
        assert console.run("reset") == console_cmd.EXIT_FAILED
        assert console.said[-1].startswith("reset not sent: no ROBOT_STATE to quote")
        assert console.run("state") == console_cmd.EXIT_FAILED
        assert console.said[-1] == "ROBOT_STATE: none has arrived"
        assert "robot connection lost" in console.said[-2]
    finally:
        console.close()


def test_state_prints_the_newest_state_and_sends_nothing(console: Console) -> None:
    assert console.run("state") == console_cmd.EXIT_OK
    service, state = console.said
    assert service.startswith("robot link service: robot connection up, session s1")
    assert "clients: 1 console" in service
    assert state == (
        "ROBOT_STATE: control owner nobody (owner number 4), stop number 3, stop "
        "level none, motion rest_hold, jaws unknown, torque on, assistance wait none"
    )
    assert console.settled() == []


def test_no_service_is_said_so(tmp_path: Path) -> None:
    said: list[str] = []
    code = console_cmd.main(
        ["--socket", str(tmp_path / "none.sock"), "state"], say=said.append
    )
    assert code == console_cmd.EXIT_FAILED
    assert said[0].startswith(
        "console command: cannot attach to the robot link service"
    )


def test_an_unknown_command_is_not_run(console: Console) -> None:
    with pytest.raises(SystemExit) as raised:
        console.run("move")
    assert raised.value.code == 2 and console.settled() == []


# The source


def test_each_command_sends_through_one_call_and_there_is_no_other_send() -> None:
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    sends = sorted(
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and ast.unparse(node.func.value) in ("client", "carrier")
        and node.func.attr
        not in ("close", "results", "state", "status", "notices", "alive")
    )
    assert sends == ["open_jaws", "release", "reset", "rest", "shutdown", "stop"]
    assert "connect_local(args.socket, ROLE_CONSOLE" in SOURCE.read_text()
    # Nothing here starts motion of its own or stands in for the agent.
    names = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    assert not names & {"move", "run_move", "permit", "request_owner", "send"}
    calls = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert "gripper" not in calls and "checker_state" not in calls


# On the real robot app


@pytest.fixture
def stack(make_robot_app: Callable[..., Any], tmp_path_factory: Any) -> Iterator[Any]:
    from link_stack import Stack

    made = Stack(make_robot_app(connected=False), tmp_path_factory.mktemp("cs"))
    made.start_service()
    made.dial()
    made.connected(made.client("viewer"))
    yield made
    made.close()


def command(stack: Any, *argv: str, answer: str = "yes") -> tuple[int, list[str]]:
    said: list[str] = []
    code = console_cmd.main(
        ["--socket", stack.path, *argv], ask=lambda prompt: answer, say=said.append
    )
    return code, said


@link
def test_stop_and_reset_reach_the_real_robot_app(stack: Any) -> None:
    code, said = command(stack, "stop")
    assert code == console_cmd.EXIT_OK and "stop number 1, stop level stop" in said[-1]
    assert stack.h.app.snapshot().stop_number == 1
    code, said = command(stack, "reset")
    assert code == console_cmd.EXIT_FAILED
    assert said[-1] == "reset: RESULT refused: no fault-stop is latched"
    stack.h.app._fault("motor fault")
    code, said = command(stack, "state")
    assert code == console_cmd.EXIT_OK and "stop level fault (motor fault)" in said[-1]
    code, said = command(stack, "reset")
    assert code == console_cmd.EXIT_OK and said[-1].startswith(
        "reset: RESULT completed"
    )
    assert stack.h.arm.torque_on() and stack.h.arm.sent == []  # nothing moved


@link
def test_a_typed_release_is_the_robot_apps_guarded_release(stack: Any) -> None:
    # Not confirmed: nothing reaches the robot app.
    code, said = command(stack, "release", answer="no")
    assert code == console_cmd.EXIT_FAILED and stack.h.arm.torque_on()
    # Confirmed, with the arm measured at rest: the robot app releases.
    code, said = command(stack, "release")
    assert code == console_cmd.EXIT_OK, said
    assert not stack.h.arm.torque_on()
    (released_at,) = stack.h.arm.released_at
    assert released_at == pytest.approx(
        list(stack.h.config.rest.rest_q), abs=stack.h.config.rest.rest_tol_rad
    )


@link
def test_a_release_away_from_rest_is_refused_by_the_robot_app(stack: Any) -> None:
    stack.h.hold_at([0.0, 0.0, 0.4, 0.0, 0.0, 0.0])
    code, said = command(stack, "release")
    assert code == console_cmd.EXIT_FAILED
    assert said[-1].startswith("release: RESULT ") and "completed" not in said[-1]
    assert stack.h.arm.torque_on() and stack.h.arm.released_at == []
