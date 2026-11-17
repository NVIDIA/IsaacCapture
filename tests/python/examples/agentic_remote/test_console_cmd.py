# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""console_cmd: one operator command through the robot link service, then exit.

Most tests run the command against a real service whose robot side the test
scripts, so every link message the service forwarded is seen. The last ones put
it in front of the real robot app on a link of 127.0.0.1.
"""

from __future__ import annotations

import ast
import contextlib
import dataclasses
import os
import pty
import select
import signal
import socket
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
from isaaccapture_examples.agentic_remote.agent import console_cmd
from isaaccapture_examples.agentic_remote.link_service.local import (
    HEADER,
    LocalRefused,
    pack_record,
)
from isaaccapture_examples.agentic_remote.protocol import confirm
from isaaccapture_examples.agentic_remote.protocol.confirm import PATH_IS_CLEAR, YES
from isaaccapture_examples.agentic_remote.protocol.local_messages import (
    LocalHello,
    LocalStatus,
)
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
from link_bench import Bench, robot_state, until, welcome

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
    # The arm of these tests is at rest: a typed "yes", and nothing may move.
    "shutdown": (Shutdown, {"confirmed": False}),
}
# The commands that ask for a typed phrase in every state. Two more ask in
# some: "close" while the jaws are not proven empty, "on" with the motors off.
CONFIRMED = ("rest", "open", "release", "shutdown")
# What each question asks for with the arm at rest.
PHRASE = {"rest": PATH_IS_CLEAR, "open": YES, "release": YES, "shutdown": YES}
# Shows on the terminal once a question that asks for "yes" is put.
ASKS_YES = b'Type "yes"'
# A takeover makes a person the control owner; the hand-back ends it at rest.
# Their typed fallback, and the OWNER_REQUEST each sends. The robot link
# service forwards them only where its file sets operator.typed_takeover.
TYPED = [("takeover", "human"), ("handback", "handback")]

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
        # What is typed at a question; None types the phrase it asks for.
        self.answer: str | None = None
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
        if self.answer is not None:
            return self.answer
        return PATH_IS_CLEAR if f'"{PATH_IS_CLEAR}"' in prompt else YES

    def show(self, **changes: Any) -> Any:
        """Send a newer ROBOT_STATE and wait until the service has it."""
        seq = self.state.seq + 1
        self.state = dataclasses.replace(self.state, seq=seq, **changes)
        self.thor.send(self.state)
        until(lambda: self.bench.service.robot_state == self.state, "the state")
        return self.state

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


class OddService:
    """A Unix socket that takes connections as the robot link service does.

    answers=False: every connection is read and nothing is answered, as by a
    service that hangs. answers=True: the first connection gets a LOCAL_STATUS,
    a WELCOME and a ROBOT_STATE for its LOCAL_HELLO and is not read after that,
    as by a service that stopped reading one client; later connections are
    read, and closed once a STOP was read on them.
    """

    WAIT_S = 10.0

    def __init__(self, path: str, answers: bool) -> None:
        self.path, self.answers = path, answers
        # (time.monotonic(), connection number, type number) of each record read.
        self.records: list[tuple[float, int, int]] = []
        self.connections = 0
        self._cond = threading.Condition()
        self._closed = threading.Event()
        self._socks: list[socket.socket] = []
        self._listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._listener.bind(path)
        self._listener.listen(32)
        self._threads = [threading.Thread(target=self._accept, daemon=True)]
        self._threads[0].start()

    def _accept(self) -> None:
        while not self._closed.is_set():
            try:
                if not select.select([self._listener], [], [], 0.05)[0]:
                    continue
                sock, _ = self._listener.accept()
            except (OSError, ValueError):
                return
            with self._cond:
                self.connections += 1
                number = self.connections
                self._socks.append(sock)
                self._cond.notify_all()
            thread = threading.Thread(
                target=self._serve, args=(sock, number), daemon=True
            )
            self._threads.append(thread)
            thread.start()

    def _serve(self, sock: socket.socket, number: int) -> None:
        pending = b""
        while not self._closed.is_set():
            try:
                if not select.select([sock], [], [], 0.05)[0]:
                    continue
                data = sock.recv(65536)
            except (OSError, ValueError):
                return
            if not data:
                return
            pending += data
            while len(pending) >= HEADER.size:
                msg_type, length = HEADER.unpack(pending[: HEADER.size])
                if len(pending) < HEADER.size + length:
                    break
                pending = pending[HEADER.size + length :]
                with self._cond:
                    self.records.append((time.monotonic(), number, msg_type))
                    self._cond.notify_all()
                if not self.answers:
                    continue
                if number == 1 and msg_type == LocalHello.TYPE:
                    status = LocalStatus(True, "", "s1", None, False, {"console": 1})
                    for message in (status, welcome(), robot_state(seq=1)):
                        sock.sendall(pack_record(message.TYPE, message.to_payload()))
                    return  # and this socket is never read again
                if msg_type == Stop.TYPE:
                    sock.close()  # a raw STOP's sender sees the end at once
                    return

    def stops(self) -> list[tuple[float, int]]:
        """(arrival time, connection number) of every STOP read so far."""
        with self._cond:
            return [(at, n) for at, n, kind in self.records if kind == Stop.TYPE]

    def wait_stop(self, count: int) -> tuple[float, int]:
        """(arrival time, connection number) of the count-th STOP."""
        with self._cond:
            found = self._cond.wait_for(
                lambda: sum(kind == Stop.TYPE for _, _, kind in self.records) >= count,
                self.WAIT_S,
            )
        assert found, f"{len(self.stops())} of {count} STOPs came"
        return self.stops()[count - 1]

    def wait_connections(self, count: int) -> None:
        with self._cond:
            assert self._cond.wait_for(lambda: self.connections >= count, self.WAIT_S)

    def close(self) -> None:
        self._closed.set()
        self._listener.close()
        with self._cond:
            socks = list(self._socks)
        for sock in socks:
            with contextlib.suppress(OSError):
                sock.close()
        for thread in self._threads:
            thread.join(2.0)


def odd_services(
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[Callable[[bool], OddService]]:
    """make(answers): an OddService on a socket of its own; closed at the end."""
    made: list[OddService] = []

    def make(answers: bool) -> OddService:
        path = str(tmp_path_factory.mktemp("os") / "s.sock")
        made.append(OddService(path, answers))
        return made[-1]

    yield make
    for service in made:
        service.close()


@pytest.fixture
def odd_service(
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[Callable[[bool], OddService]]:
    yield from odd_services(tmp_path_factory)


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


@pytest.fixture
def allowed(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Console]:
    """The command on a service whose file allows the typed takeover."""
    allowing = Bench(tmp_path_factory.mktemp("ct"), typed_takeover=True)
    made = Console(allowing)
    yield made
    made.close()
    allowing.close()


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
def test_rest_open_release_and_shutdown_ask_for_a_typed_phrase_first(
    console: Console, command: str
) -> None:
    assert console.run(command) == console_cmd.EXIT_OK
    (prompt,) = console.asked
    assert (prompt, PHRASE[command]) == console_cmd.question(command, console.state)
    assert f'Type "{PHRASE[command]}" to ' in prompt
    words = {
        "rest": "Is the path clear?",
        "open": "whatever they hold is let go",
        "release": "only with the arm measured at rest",
        "shutdown": "the arm is at rest",
    }
    assert words[command] in prompt
    # The state the answer is given for is on the screen before the question.
    assert console.said[0] == console_cmd.state_text(console.state)


@pytest.mark.parametrize("command", CONFIRMED)
@pytest.mark.parametrize("answer", ["", "no", "y", "yes.", "path clear"])
def test_without_the_typed_phrase_nothing_is_sent(
    console: Console, command: str, answer: str
) -> None:
    console.answer = answer
    assert console.run(command) == console_cmd.EXIT_FAILED
    assert console.said[-1] == f"{command} not confirmed; nothing sent"
    assert len(console.asked) == 1 and console.settled() == []


def test_rest_asks_for_path_is_clear_and_yes_sends_nothing(console: Console) -> None:
    console.answer = "yes"
    assert console.run("rest") == console_cmd.EXIT_FAILED
    assert console.asked == [console_cmd.ASK_REST]
    assert 'Type "path is clear" to start' in console.asked[0]
    assert console.said[-1] == "rest not confirmed; nothing sent"
    assert console.settled() == []
    # Case and spacing do not count; the words do.
    console.answer = "  Path  is CLEAR \n"
    assert console.run("rest") == console_cmd.EXIT_OK
    (message,) = console.settled()
    assert isinstance(message, OwnerRequest) and message.confirmed is True


def test_shutdown_at_rest_asks_for_yes_and_is_sent_unconfirmed(
    console: Console,
) -> None:
    assert console.state.motion == "rest_hold"
    reason = "confirm the path is clear"  # what the Thor says if the arm must move
    console.reply = lambda message: [Result(message.command_id, "refused", reason)]
    assert console.run("shutdown") == console_cmd.EXIT_FAILED
    assert console.asked == [console_cmd.ASK_SHUTDOWN_AT_REST]
    (message,) = console.settled()
    assert isinstance(message, Shutdown) and message.confirmed is False
    assert console.said[-1] == f"shutdown: RESULT refused: {reason}"
    # "path is clear" is not what was asked at rest: nothing more is sent.
    console.answer = PATH_IS_CLEAR
    assert console.run("shutdown") == console_cmd.EXIT_FAILED
    assert console.said[-1] == "shutdown not confirmed; nothing sent"
    assert len(console.settled()) == 1


def test_shutdown_away_from_rest_asks_for_path_is_clear(console: Console) -> None:
    console.show(motion="hold")
    console.answer = "yes"
    assert console.run("shutdown") == console_cmd.EXIT_FAILED
    assert console.asked == [console_cmd.ASK_SHUTDOWN_MOVES]
    assert 'Type "path is clear" to shut down' in console.asked[0]
    assert console.settled() == []
    console.answer = None
    assert console.run("shutdown") == console_cmd.EXIT_OK
    (message,) = console.settled()
    assert isinstance(message, Shutdown) and message.confirmed is True
    # No ROBOT_STATE at all asks the same: the arm may have to move.
    assert console_cmd.question("shutdown", None)[1] == PATH_IS_CLEAR


def test_a_command_quotes_the_state_shown_with_its_question(console: Console) -> None:
    shown = console.state

    def ask(prompt: str) -> str:
        # While the operator types, the owner number and the stop number move on.
        console.show(owner_number=5, stop_number=4)
        return YES

    assert console.run("open", ask=ask) == console_cmd.EXIT_OK
    (message,) = console.settled()
    assert (message.owner_number, message.stop_number) == (4, 3)
    assert console.said[0] == console_cmd.state_text(shown)


def test_send_command_sends_nothing_without_the_phrase_of_its_question() -> None:
    class Client:
        def __init__(self) -> None:
            self.sent: list[tuple[str, dict[str, Any]]] = []

        def __getattr__(self, name: str) -> Any:
            def call(*request: str, **fields: Any) -> str:
                if request:  # an owner request's word is given by place
                    (fields["request"],) = request
                self.sent.append((name, fields))
                return f"{name}-1"

            return call

    client, state = Client(), robot_state(owner_number=4, stop_number=3)
    assert console_cmd.CONFIRMED == (
        "rest",
        "open",
        "close",
        "release",
        "on",
        "shutdown",
        "takeover",
        "handback",
    )
    assert set(CONFIRMED) < set(console_cmd.CONFIRMED)
    # "on" switches the motors on again after a rest or a release.
    assert console_cmd.question("on", state) == (console_cmd.ASK_ON, YES)
    assert console_cmd.send_command("on", client, state, phrase_typed="yes")
    name, fields = client.sent.pop()
    assert (name, fields["request"]) == ("request_owner", "motors_on")
    assert (fields["owner_number"], fields["stop_number"]) == (4, 3)
    for command in console_cmd.CONFIRMED:
        for typed in ("", "no", "stop"):
            with pytest.raises(console_cmd.NotConfirmed):
                console_cmd.send_command(command, client, state, phrase_typed=typed)
    with pytest.raises(ValueError, match="'stop' is not a command"):
        console_cmd.send_command("stop", client, state, phrase_typed="")
    assert client.sent == []
    sent = console_cmd.send_command("reset", client, state, phrase_typed="")
    assert sent == "reset-1" and client.sent == [("reset", NUMBERS)]


@pytest.mark.parametrize("command", CONFIRMED)
def test_a_confirmation_cannot_come_from_a_pipe(console: Console, command: str) -> None:
    # Under pytest there is no terminal on stdin: the real prompt refuses.
    code = console.run(command, ask=console_cmd.terminal_ask)
    assert code == console_cmd.EXIT_FAILED
    assert console.said[-1] == (
        f"{command} not confirmed (this command needs a terminal to type the "
        "confirmation at); nothing sent"
    )
    assert console.settled() == []


# The typed takeover and hand-back


@pytest.mark.parametrize(("command", "sends"), TYPED)
def test_takeover_and_handback_ask_yes_and_send_the_owner_request(
    allowed: Console, command: str, sends: str
) -> None:
    assert allowed.run(command) == console_cmd.EXIT_OK
    (prompt,) = allowed.asked
    assert (prompt, YES) == console_cmd.question(command, allowed.state)
    assert f'Type "{YES}" to ' in prompt
    # The state the answer is given for is on the screen before the question.
    assert allowed.said[0] == console_cmd.state_text(allowed.state)
    (message,) = allowed.settled()  # exactly one
    assert message == OwnerRequest(
        message.command_id,
        **NUMBERS,
        request=sends,
        reason="typed at the operator console",
    )
    assert allowed.roles == ["console"]
    assert allowed.said[-1] == f'{command}: RESULT completed {{"owner": "nobody"}}'
    # Any answer but yes sends nothing.
    for answer in ("", "no", PATH_IS_CLEAR):
        allowed.answer = answer
        assert allowed.run(command) == console_cmd.EXIT_FAILED
        assert allowed.said[-1] == f"{command} not confirmed; nothing sent"
    # A confirmation cannot come from a pipe either.
    code = allowed.run(command, ask=console_cmd.terminal_ask)
    assert code == console_cmd.EXIT_FAILED and len(allowed.settled()) == 1
    # Each has its wait for the robot app's answer.
    assert set(console_cmd.RESULT_WAIT_S) == {"reset", *console_cmd.CONFIRMED}
    assert set(console_cmd.COMMANDS) == {"state", "stop", *console_cmd.RESULT_WAIT_S}


@pytest.mark.parametrize(("command", "sends"), TYPED)
def test_a_refusal_by_the_service_is_shown_with_the_buttons_to_use(
    console: Console, allowed: Console, command: str, sends: str
) -> None:
    # The service of `console` is made as its file ships: the setting is off.
    assert console.run(command) == console_cmd.EXIT_FAILED
    assert console.said[-1] == (
        f"{command}: RESULT refused: the console role may not send "
        f'OWNER_REQUEST "{sends}"; use right B / right A on the controller'
    )
    assert len(console.asked) == 1 and console.settled() == []
    # A refusal of the robot app is printed as it is: it names no buttons.
    reason = "control owner is agent, not nobody"
    allowed.reply = lambda message: [Result(message.command_id, "refused", reason)]
    assert allowed.run(command) == console_cmd.EXIT_FAILED
    assert allowed.said[-1] == f"{command}: RESULT refused: {reason}"
    assert len(allowed.settled()) == 1
    # Nor does the service's refusal of anything else, or a RESULT that is none.
    not_allowed = f"{console_cmd.NOT_ALLOWED}OWNER_REQUEST"
    for other, status in (("rest", "refused"), (command, "completed")):
        result = Result("x", status, not_allowed)
        assert console_cmd.buttons_hint(other, result) == ""


# The number of the read system call, by machine.
_READ_CALL = {"x86_64": 0, "aarch64": 63}


def reads_a_terminal(pid: int) -> bool | None:
    """Whether the process's main thread waits in read() on a terminal.

    None where that cannot be told: no /proc/<pid>/syscall that may be read,
    or a machine whose call numbers are not listed.
    """
    number = _READ_CALL.get(os.uname().machine)
    try:
        fields = Path(f"/proc/{pid}/syscall").read_text().split()
    except OSError:
        return None
    if number is None or not fields:
        return None
    if fields[0] != str(number) or len(fields) < 2:
        return False  # running, or in another call
    try:
        target = os.readlink(f"/proc/{pid}/fd/{int(fields[1], 16)}")
    except (OSError, ValueError):
        return False
    return target.startswith("/dev/pts/") or target == "/dev/tty"


def await_read(pid: int, wait_s: float = 2.0, unknown_s: float = 0.1) -> None:
    """Wait until the process reads its terminal; where unknown, unknown_s.

    It must be seen reading twice, 5 ms apart: a read of a line that was
    typed ahead ends at once and is followed by a flush.
    """
    end = time.monotonic() + wait_s
    seen = 0
    while time.monotonic() < end and seen < 2:
        reading = reads_a_terminal(pid)
        if reading is None:
            time.sleep(unknown_s)
            return
        seen = seen + 1 if reading else 0
        time.sleep(0.005)


def on_a_terminal(
    child: Callable[[], Any],
    typed_before: bytes = b"",
    typed_after: bytes = b"",
    marker: bytes = b"Type yes",
    delay_s: float = 0.5,
    then: tuple[bytes, bytes] | None = None,
) -> tuple[str, bool]:
    """Run child() in a new process whose terminal this test owns.

    typed_before is typed at once and child() starts delay_s later;
    typed_after is typed once `marker` showed on the terminal and the child
    reads the terminal again (what it flushes before that would be lost).
    `then` is a second (marker, what is typed once it showed after the
    first). Returns (the repr of what child() returned or raised, whether the
    last marker showed).
    """
    steps = [(marker, typed_after), *([then] if then else [])]
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
        seen, answered, ended = b"", 0, False
        end = time.monotonic() + delay_s + 10.0
        while time.monotonic() < end and not ended:
            ready, _, _ = select.select([master], [], [], 0.05)
            if ready:
                try:
                    seen += os.read(master, 4096)
                except OSError:  # the child closed the terminal
                    pass
            if answered < len(steps) and steps[answered][0] in seen:
                await_read(pid)
                os.write(master, steps[answered][1])
                answered += 1
            ended = os.waitpid(pid, os.WNOHANG)[0] != 0
        if not ended:
            os.kill(pid, signal.SIGKILL)
            os.waitpid(pid, 0)
        return Path(result.name).read_text(), steps[-1][0] in seen
    finally:
        os.close(master)
        os.unlink(result.name)


def release_at_a_terminal(socket_path: str, agent_marker: str = "") -> str:
    """console_cmd release on the service at socket_path; for on_a_terminal()."""
    said: list[str] = []
    for name in list(os.environ):
        if console_cmd.agent_environment({name: os.environ[name]}):
            del os.environ[name]
    if agent_marker:
        os.environ[agent_marker] = "1"
    code = console_cmd.main(["release", "--socket", socket_path], say=said.append)
    return f"exit {code}: {said}"


def test_a_yes_typed_before_the_question_does_not_answer_it(console: Console) -> None:
    def run() -> str:
        return release_at_a_terminal(console.bench.path)

    # Typed ahead, the yes is dropped; what is typed at the question counts.
    answer, shown = on_a_terminal(run, b"yes\n", b"no\n", ASKS_YES)
    assert shown and "release not confirmed; nothing sent" in answer
    assert console.settled() == []
    answer, shown = on_a_terminal(run, b"", b"yes\n", ASKS_YES)
    assert shown and "exit 0: " in answer and "release: RESULT completed" in answer
    answer, shown = on_a_terminal(run, b"no\n\n", b"yes\n", ASKS_YES)
    assert shown and "exit 0: " in answer
    assert [type(message) for message in console.settled()] == [Release, Release]


@pytest.mark.parametrize("marker", ["CLAUDECODE", "AI_AGENT", "OPENCLAW_HOME"])
def test_a_confirmation_is_not_asked_in_an_agents_environment(
    console: Console, marker: str
) -> None:
    answer, shown = on_a_terminal(
        lambda: release_at_a_terminal(console.bench.path, marker),
        b"yes\n",
        b"yes\n",
        ASKS_YES,
    )
    assert not shown  # the question was never put
    assert "release not confirmed (the confirmation is typed by the operator" in answer
    assert "outside an agent environment); nothing sent" in answer
    assert "exit 1: " in answer
    assert console.settled() == []


def test_the_marks_of_an_agents_environment() -> None:
    marked = console_cmd.agent_environment
    assert marked({"CLAUDECODE": "1", "PATH": "/bin"}) == "CLAUDECODE"
    assert marked({"OPENCLAW_ANYTHING": "x"}) == "OPENCLAW_ANYTHING"
    assert marked({"CLAUDECODE": " ", "PATH": "/bin"}) is None
    assert marked({}) is None
    # One copy, in protocol/confirm.py; the old names still import from here.
    assert marked is confirm.agent_environment
    assert console_cmd.AGENT_ENV is confirm.AGENT_ENV
    assert console_cmd.AGENT_ENV_PREFIXES is confirm.AGENT_ENV_PREFIXES
    assert "AGENT_ENV = " not in SOURCE.read_text(encoding="utf-8")


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
    # A command carries no expiry: one that reaches the robot app later is
    # still done there. The operator is told, with what ends it.
    console.answer = PATH_IS_CLEAR
    assert console.run("rest", "--timeout", "0.3") == console_cmd.EXIT_FAILED
    assert console.said[-1].startswith("rest: no final RESULT within 0.3 s (")
    assert console.said[-1].endswith(
        "). What was sent is not withdrawn: the robot app may still do it. Run "
        "the command stop before anyone goes near the arm"
    )
    assert len(console.settled()) == 2


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


def test_a_new_session_does_not_confirm_a_stop(console: Console) -> None:
    fresh = dataclasses.replace(
        robot_state(seq=50, session_id="s2", owner_number=0, stop_number=0),
        motion="hold",
    )
    # The robot side drops the STOP and shows a restarted robot app.
    console.reply = lambda message: [welcome("s2"), fresh]
    assert console.run("stop") == console_cmd.EXIT_FAILED
    assert console.said[-1] == (
        "stop: NOT CONFIRMED: the robot app shows a new session (s2), which says "
        "nothing about this STOP; look at the Thor terminal"
    )
    assert len(console.settled()) == 1


def test_a_stop_sent_before_any_state_waits_for_a_higher_stop_number(
    bench: Bench,
) -> None:
    console = Console(bench, robot=False)
    console.thor = bench.robot(None)  # connected; no ROBOT_STATE yet
    old = dataclasses.replace(
        robot_state(seq=1, stop_number=7), stop_level="stop", motion="hold"
    )
    try:
        # The first state shows a stop of an earlier STOP: it answers nothing.
        console.reply = lambda message: [welcome(), old]
        assert console.run("stop", "--timeout", "0.5") == console_cmd.EXIT_FAILED
        assert "no ROBOT_STATE showed a higher stop number" in console.said[-1]
    finally:
        console.close()
    later = dataclasses.replace(old, seq=2, stop_number=8)
    assert not console_cmd.stop_shows(None, old)
    assert not console_cmd.stop_shows(None, later)
    assert console_cmd.stop_shows(old, later)
    assert not console_cmd.stop_shows(old, dataclasses.replace(old, seq=3))
    other = dataclasses.replace(later, session_id="s2")
    assert not console_cmd.stop_shows(old, other)
    assert console_cmd.new_session(old, other)
    assert not console_cmd.new_session(old, later)
    assert not console_cmd.new_session(None, other)


def test_a_stop_is_sent_at_once_while_no_robot_state_has_arrived(
    bench: Bench,
) -> None:
    console = Console(bench, robot=False)
    console.thor = bench.robot(None)  # connected; no ROBOT_STATE comes
    arrived: list[float] = []
    console.reply = lambda message: arrived.append(time.monotonic()) or []
    try:
        began = time.monotonic()
        assert console.run("stop", "--timeout", "0.5") == console_cmd.EXIT_FAILED
        (message,) = console.settled()
        assert message == Stop(reason=console_cmd.STOP_REASON, sender="console")
        took = arrived[0] - began
        assert took < 1.0, f"the STOP reached the robot side after {took:.2f} s"
        assert "no ROBOT_STATE showed a higher stop number" in console.said[-1]
    finally:
        console.close()


def test_a_stop_sent_before_any_state_is_answered_by_a_later_higher_stop_number(
    bench: Bench,
) -> None:
    console = Console(bench, robot=False)
    console.thor = thor = bench.robot(None)  # connected; no ROBOT_STATE yet
    old = dataclasses.replace(
        robot_state(seq=1, stop_number=7), stop_level="stop", motion="hold"
    )
    later = dataclasses.replace(old, seq=2, stop_number=8)

    def reply(message: Message) -> list[Message]:
        # The first state is of an earlier STOP; this one shows 0.3 s later.
        threading.Timer(0.3, thor.send, (later,)).start()
        return [welcome(), old]

    console.reply = reply
    try:
        assert console.run("stop", "--timeout", "3") == console_cmd.EXIT_OK
        assert console.said[-1].startswith("stop: taken. ROBOT_STATE: ")
        assert "stop number 8, stop level stop" in console.said[-1]
    finally:
        console.close()


def test_a_stop_does_not_wait_out_a_service_that_takes_the_connection_and_is_silent(
    odd_service: Callable[[bool], OddService],
) -> None:
    service = odd_service(False)
    said: list[str] = []
    began = time.monotonic()
    code = console_cmd.main(["--socket", service.path, "stop"], say=said.append)
    at, connection = service.wait_stop(1)
    took = at - began
    assert took < 1.0, f"the STOP was written {took:.2f} s after the command began"
    assert connection == 2  # raw, after the attach was given up
    assert code == console_cmd.EXIT_FAILED
    assert said == [
        "stop: cannot attach to the robot link service: the robot link service "
        f"at {service.path} did not answer in 0.5 s. STOP written; not confirmed: "
        "look at the Thor terminal"
    ]
    # Every other command gives the service its 5 s.
    assert (console_cmd.STOP_ATTACH_S, console_cmd.ATTACH_S) == (0.5, 5.0)


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
    # A stop says so too: its raw STOP found nobody to write to.
    code = console_cmd.main(
        ["--socket", str(tmp_path / "none.sock"), "stop"], say=said.append
    )
    assert code == console_cmd.EXIT_FAILED
    assert said[-1].startswith("stop: cannot attach to the robot link service: ")
    assert "STOP NOT written: no robot link service at " in said[-1]


def test_the_one_shot_stop_falls_back_to_the_raw_path(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    full = Bench(tmp_path_factory.mktemp("cf"), max_clients=1)
    try:
        thor = full.robot()
        full.client("viewer")  # the one client slot
        with pytest.raises(LocalRefused):
            console_cmd.connect_local(full.path, "console", "console_cmd", 2.0)
        said: list[str] = []
        code = console_cmd.main(["--socket", full.path, "stop"], say=said.append)
        # Exit code 1: a raw STOP cannot be confirmed.
        assert code == console_cmd.EXIT_FAILED
        assert said == [
            "stop: cannot attach to the robot link service: too many local "
            "clients are attached. STOP written; not confirmed: look at the Thor "
            "terminal (the robot link service said: too many local clients are "
            "attached)"
        ]
        assert thor.next_message() == Stop(console_cmd.STOP_REASON, "console")
        thor.nothing()
        # Any other command is not sent that way.
        code = console_cmd.main(["--socket", full.path, "reset"], say=said.append)
        assert code == console_cmd.EXIT_FAILED
        assert said[-1].startswith("console command: cannot attach")
        thor.nothing()
    finally:
        full.close()


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
    assert sends == [
        "open_jaws",
        "release",
        "request_owner",
        "reset",
        "rest",
        "shutdown",
        "stop",
    ]
    # The six that quote numbers go out in _run() and nowhere else.
    (run,) = (
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_run"
    )
    calls_in_run = [
        node
        for node in ast.walk(run)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    ]
    assert sorted(node.func.attr for node in calls_in_run) == [
        "open_jaws",
        "release",
        "request_owner",
        "reset",
        "rest",
        "shutdown",
    ]
    assert "connect_local(args.socket, ROLE_CONSOLE" in SOURCE.read_text()
    # The one owner request it writes itself is a typed takeover or hand-back,
    # or "on", which switches the motors on again.
    (request,) = (node for node in calls_in_run if node.func.attr == "request_owner")
    assert ast.unparse(request.args[0]) == "OWNER_WORDS[command]"
    assert console_cmd.TYPED_REQUESTS == dict(TYPED)
    assert console_cmd.OWNER_WORDS == {**dict(TYPED), "on": "motors_on"}
    # Nothing here starts motion of its own or stands in for the agent.
    names = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    assert not names & {"move", "run_move", "permit", "send"}
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


def command(stack: Any, *argv: str, answer: str = YES) -> tuple[int, list[str]]:
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
