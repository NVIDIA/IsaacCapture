# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The Thor terminal: typed lines, operator lines and Ctrl+C.

The first part drives it with a stand-in for the robot app, the second with the
robot app itself on the simulated arm.
"""

from __future__ import annotations

import os
import pty
import signal
import threading
import time
import tty
from collections.abc import Callable, Iterator

import pytest
from isaaccapture_examples.agentic_remote.protocol import confirm
from isaaccapture_examples.agentic_remote.protocol.messages import Result
from isaaccapture_examples.agentic_remote.robot_app import __main__ as robot_main
from isaaccapture_examples.agentic_remote.robot_app import terminal as terminal_module
from isaaccapture_examples.agentic_remote.robot_app.app import (
    CONFIRM_JAWS_EMPTY,
    CONFIRM_PATH,
    NOT_RUNNING,
    REFERENCE_OFF,
)
from isaaccapture_examples.agentic_remote.robot_app.interfaces import MotionState, Owner
from isaaccapture_examples.agentic_remote.robot_app.terminal import (
    BANNER,
    BELL_CHAR,
    CONFIRM_REFERENCE,
    CONFIRM_REST,
    CTRL_C_NOTE,
    HELP,
    STOP_CTRL_C,
    STOP_TYPED,
    ThorTerminal,
    TypedLines,
)

WAIT_S = 5.0  # real seconds a test gives another thread before it fails
REST = [0.0] * 6
# 25 setpoints at 0.2 rad/s on every joint, inside the agent's limits.
PATH = [[0.004 * (index + 1)] * 6 for index in range(25)]
# A pose the agent may hold; the return to rest from it is a real move.
RAISED = [0.2, -0.15, 0.1, 0.0, 0.05, -0.05]
ENDED_RUNNING = "the Thor terminal's input ended; the robot app runs on"
THOR = "thor_terminal"


class Lines:
    """The terminal's writer: keeps every line and can be waited on."""

    def __init__(self) -> None:
        self._written = threading.Condition()
        self.lines: list[str] = []

    def __call__(self, line: str) -> None:
        with self._written:
            self.lines.append(line)
            self._written.notify_all()

    def wait_for(self, wanted: str | Callable[[str], bool]) -> str:
        """The first line that equals `wanted`, or that it accepts."""

        def accepts(line: str) -> bool:
            return line == wanted if isinstance(wanted, str) else wanted(line)

        def found() -> str | None:
            return next((line for line in self.lines if accepts(line)), None)

        with self._written:
            assert self._written.wait_for(lambda: found() is not None, WAIT_S), (
                f"no such line; written: {self.lines}"
            )
            return found()


class FakeApp:
    """What the terminal needs of the robot app; records every call."""

    def __init__(self) -> None:
        self.running = True
        self.moves_arm = False  # whether a shutdown would have to move the arm
        self.allow_reference = False  # the setting jaws.allow_reference
        self.listeners: list[Callable[[str, str, dict], None]] = []
        self.calls: list[tuple] = []
        self.threads: list[str] = []  # the thread each call was made on
        self.release_result = Result("thor-terminal-release-1", "completed")
        self.shutdown_result = Result("thor-terminal-shutdown-1", "completed")
        self.raises: Exception | None = None
        self._called = threading.Condition()

    def add_event_listener(self, listener: Callable[[str, str, dict], None]) -> None:
        self.listeners.append(listener)

    def stop(self, reason: str, sender: str = THOR) -> int:
        self._record(("stop", reason, sender))
        return len([call for call in self.calls if call[0] == "stop"])

    def request_rest(self, confirmed, on_done=None) -> str | None:
        self._record(("rest", confirmed))
        if confirmed:
            on_done(Result("thor-terminal-rest-1", "completed"))
            return None
        on_done(Result("thor-terminal-rest-1", "refused", CONFIRM_PATH))
        return CONFIRM_PATH

    def request_release(self, on_done=None) -> str | None:
        self._record(("release",))
        on_done(self.release_result)
        refused = self.release_result.status == "refused"
        return self.release_result.reason if refused else None

    def request_reference(self, confirmed, on_done=None) -> str | None:
        self._record(("reference", confirmed))
        reason = None if self.allow_reference else REFERENCE_OFF
        reason = reason or (None if confirmed else CONFIRM_JAWS_EMPTY)
        if reason is None:
            on_done(Result("thor-terminal-reference-1", "completed"))
        else:
            on_done(Result("thor-terminal-reference-1", "refused", reason))
        return reason

    def shutdown_moves_arm(self) -> bool:
        return self.moves_arm

    def request_shutdown(self, confirmed, on_done=None) -> str | None:
        self._record(("shutdown", confirmed))
        if self.moves_arm and not confirmed:
            on_done(Result("thor-terminal-shutdown-1", "refused", CONFIRM_PATH))
            return CONFIRM_PATH
        on_done(self.shutdown_result)
        refused = self.shutdown_result.status == "refused"
        return self.shutdown_result.reason if refused else None

    def emit(self, kind: str, text: str, data: dict | None = None) -> None:
        for listener in self.listeners:
            listener(kind, text, data or {})

    def wait_calls(self, count: int) -> list[tuple]:
        with self._called:
            assert self._called.wait_for(lambda: len(self.calls) >= count, WAIT_S), (
                f"{len(self.calls)} calls, not {count}: {self.calls}"
            )
            return list(self.calls)

    def _record(self, call: tuple) -> None:
        with self._called:
            self.calls.append(call)
            self.threads.append(threading.current_thread().name)
            self._called.notify_all()
        if self.raises is not None:
            raise self.raises


class Session:
    """A terminal whose input is typed by the test, read on a thread of its own."""

    def __init__(self, app) -> None:
        self.typed = TypedLines()
        self.out = Lines()
        self.terminal = ThorTerminal(app, self.typed, self.out)
        self._reader = threading.Thread(target=self.terminal.run, daemon=True)
        self._reader.start()
        self.out.wait_for(BANNER)

    def type(self, *lines: str) -> None:
        for line in lines:
            self.typed.type(line)

    def end_input(self) -> None:
        self.typed.end()
        self._reader.join(WAIT_S)
        assert not self._reader.is_alive(), "run() did not return"

    def close(self) -> None:
        self.end_input()
        self.terminal.close()


@pytest.fixture
def fake() -> FakeApp:
    return FakeApp()


@pytest.fixture
def out() -> Lines:
    return Lines()


@pytest.fixture
def open_terminal() -> Iterator[Callable[..., ThorTerminal]]:
    """Builds terminals that are closed after the test."""
    made: list[ThorTerminal] = []

    def make(app, lines=(), output=None, clock=None) -> ThorTerminal:
        made.append(ThorTerminal(app, lines, output, clock=clock))
        return made[-1]

    yield make
    for terminal in made:
        terminal._replaced = None  # the test's stand-in handlers need no restoring
        terminal.close()


@pytest.fixture
def session() -> Iterator[Callable[..., Session]]:
    """Builds terminals with typed input that are closed after the test."""
    made: list[Session] = []

    def make(app) -> Session:
        made.append(Session(app))
        return made[-1]

    yield make
    for one in made:
        one.close()


def typed(app, *lines: str) -> list[str]:
    """Everything a terminal wrote for these typed lines, the banner left out."""
    out = Lines()
    terminal = ThorTerminal(app, iter(lines), out)
    terminal.run()
    assert terminal.flush(WAIT_S)
    terminal.close()
    assert out.lines[0] == BANNER
    return out.lines[1:]


def held_move(h, count: int = 6) -> tuple[str, threading.Event]:
    """An agent MOVE along PATH, held once `count` setpoints were sent."""
    reached, release = h.hold_move_after(count)
    command_id = h.client.move(PATH, start_q=REST)
    h.client.permit(command_id, len(PATH) - 1)
    assert reached.wait(WAIT_S)
    return command_id, release


def nothing_switched_off(h) -> bool:
    return h.arm.torque_on() and h.arm.released_at == []


# Typed lines


@pytest.mark.parametrize("line", ["", "   ", "\n", " \t \n"])
def test_an_empty_line_does_nothing(fake, line):
    assert typed(fake, line) == [ENDED_RUNNING]
    assert fake.calls == []


@pytest.mark.parametrize("line", ["stop", "stop\n", "  STOP  \r\n", "Stop"])
def test_stop_is_a_stop_from_the_thor_terminal(fake, line):
    assert typed(fake, line) == [ENDED_RUNNING]
    assert fake.calls == [("stop", STOP_TYPED, THOR)]
    assert STOP_TYPED == "typed stop"


def test_help_lists_every_command(fake):
    lines = typed(fake, "help\n")
    assert lines[: len(HELP)] == list(HELP)
    words = [line.split()[0] for line in HELP]
    assert words == ["stop", "rest", "release", "quit", "reference", "help"]
    # The banner leaves out the command that a setting switches on.
    assert all(word in BANNER for word in words if word != "reference")
    assert '"path is clear"' in HELP[1]
    assert '"jaws are empty"' in HELP[4] and "jaws.allow_reference" in HELP[4]
    assert lines[len(HELP) :] == ["an empty line does nothing", ENDED_RUNNING]
    assert fake.calls == []


@pytest.mark.parametrize("line", ["restt", "releases", "open", "yes", "stop now"])
def test_an_unknown_line_is_named_and_does_nothing(fake, line):
    assert typed(fake, line + "\n")[0] == f"unknown command {line!r}; type help"
    assert fake.calls == []


def test_an_unknown_line_is_shown_without_its_control_characters(fake):
    shown = typed(fake, "re\ast\x1b[2J" + "x" * 100)[0]
    assert BELL_CHAR not in shown and "\x1b" not in shown
    assert len(shown) < 80


def test_rest_needs_the_typed_path_is_clear(fake):
    lines = typed(fake, "rest\n", "  Path  is CLEAR \n")
    assert lines == [CONFIRM_REST, "rest: completed", ENDED_RUNNING]
    assert CONFIRM_REST == (
        "Return to rest: is the path to the rest pose clear? "
        'Type "path is clear" to confirm.'
    )
    assert fake.calls == [("rest", True)]


@pytest.mark.parametrize("answer", ["yes", " YES ", "y"])
def test_yes_no_longer_confirms_a_rest(fake, answer):
    lines = typed(fake, "rest\n", answer + "\n")
    assert lines[1] == "rest: refused: confirm the path is clear"
    assert fake.calls == [("rest", False)]
    # The entry point's own questions still take it.
    assert terminal_module.YES == confirm.YES == "yes"


@pytest.mark.parametrize(
    "answer",
    ["no", "", "path clear", "path is clear now", "pathisclear", "rest", "quit"],
)
def test_rest_without_the_typed_phrase_is_passed_on_as_not_confirmed(fake, answer):
    lines = typed(fake, "rest\n", answer + "\n")
    assert lines == [
        CONFIRM_REST,
        "rest: refused: confirm the path is clear",
        ENDED_RUNNING,
    ]
    # The answer is used up: it is not served as a command of its own.
    assert fake.calls == [("rest", False)]


def test_stop_typed_as_the_answer_is_a_stop(fake):
    typed(fake, "rest\n", " STOP\n")
    assert fake.calls == [("stop", STOP_TYPED, THOR), ("rest", False)]


@pytest.mark.parametrize(
    ("answer", "phrase", "confirmed"),
    [
        ("path is clear", confirm.PATH_IS_CLEAR, True),
        ("  Path   IS\tclear \r\n", confirm.PATH_IS_CLEAR, True),
        ("JAWS ARE EMPTY", confirm.JAWS_ARE_EMPTY, True),
        (" Yes\n", confirm.YES, True),
        ("yes", confirm.PATH_IS_CLEAR, False),
        ("pathisclear", confirm.PATH_IS_CLEAR, False),
        ("path is clear.", confirm.PATH_IS_CLEAR, False),
        ("", confirm.YES, False),
        (None, confirm.YES, False),
    ],
)
def test_confirms_ignores_case_and_spacing(answer, phrase, confirmed):
    assert confirm.confirms(answer, phrase) is confirmed
    assert (confirm.YES, confirm.PATH_IS_CLEAR, confirm.JAWS_ARE_EMPTY) == (
        "yes",
        "path is clear",
        "jaws are empty",
    )


def test_agent_environment_names_the_variable(monkeypatch):
    assert confirm.agent_environment({}) is None
    assert confirm.agent_environment({"PATH": "/bin", "CLAUDECODE": " "}) is None
    assert confirm.agent_environment({"CLAUDECODE": "1"}) == "CLAUDECODE"
    # The fixed names are looked at first, in their order, then the prefixes.
    both = {"OPENCLAW_HOME": "x", "CODEX_THREAD_ID": "7", "AI_AGENT": "a"}
    assert confirm.agent_environment(both) == "AI_AGENT"
    assert confirm.agent_environment({"OPENCLAW_HOME": "x"}) == "OPENCLAW_HOME"
    assert confirm.agent_environment({"OPENCLAW_HOME": ""}) is None
    # Without a mapping it reads the process environment.
    for name in confirm.AGENT_ENV:
        monkeypatch.delenv(name, raising=False)
    for name in list(os.environ):
        if name.startswith(confirm.AGENT_ENV_PREFIXES):
            monkeypatch.delenv(name)
    assert confirm.agent_environment() is None
    monkeypatch.setenv("HERMES_AGENT", "1")
    assert confirm.agent_environment() == "HERMES_AGENT"
    assert "CLAUDECODE" in confirm.AGENT_ENV
    assert confirm.AGENT_ENV_PREFIXES == ("OPENCLAW_",)


def test_reference_is_refused_unasked_while_the_setting_is_off(fake):
    lines = typed(fake, "reference\n", "jaws are empty\n")
    assert lines[0] == f"reference: refused: {REFERENCE_OFF}"
    assert "jaws.allow_reference" in REFERENCE_OFF
    assert CONFIRM_REFERENCE not in lines
    # The second line was not used up as an answer: it is an unknown command.
    assert lines[1] == "unknown command 'jaws are empty'; type help"
    assert fake.calls == [("reference", False)]


def test_reference_asks_that_the_jaws_are_empty(fake):
    fake.allow_reference = True
    lines = typed(fake, "Reference\n", " Jaws are  EMPTY\n")
    assert lines == [CONFIRM_REFERENCE, "reference: completed", ENDED_RUNNING]
    assert CONFIRM_REFERENCE == (
        "Reference the jaws: they close slowly to their stop. Are the jaws empty? "
        'Type "jaws are empty" to confirm.'
    )
    assert fake.calls == [("reference", True)]


@pytest.mark.parametrize("answer", ["yes", "path is clear", "", "empty"])
def test_reference_is_refused_without_the_typed_phrase(fake, answer):
    fake.allow_reference = True
    lines = typed(fake, "reference\n", answer + "\n")
    assert lines == [
        CONFIRM_REFERENCE,
        "reference: refused: confirm that the jaws are empty",
        ENDED_RUNNING,
    ]
    assert fake.calls == [("reference", False)]


def test_stop_typed_where_reference_asks_is_a_stop_and_confirms_nothing(fake):
    fake.allow_reference = True
    typed(fake, "reference\n", "stop\n")
    assert fake.calls == [("stop", STOP_TYPED, THOR), ("reference", False)]


def test_the_input_ending_at_the_confirmation_confirms_nothing(fake):
    lines = typed(fake, "rest\n")
    assert lines == [
        CONFIRM_REST,
        "rest: refused: confirm the path is clear",
        ENDED_RUNNING,
    ]
    assert fake.calls == [("rest", False)]


def test_an_ended_input_is_not_read_again(fake, out):
    class Once:
        """Ends after one line; a second read after the end fails the test."""

        def __init__(self) -> None:
            self.left = ["rest\n"]
            self.ended = False

        def __iter__(self) -> Once:
            return self

        def __next__(self) -> str:
            assert not self.ended, "read after the end of the input"
            if self.left:
                return self.left.pop()
            self.ended = True
            raise StopIteration

    terminal = ThorTerminal(fake, Once(), out)
    terminal.run()
    terminal.close()
    assert fake.calls == [("rest", False)]


def test_release_is_the_release_path_and_asks_nothing(fake):
    assert typed(fake, "Release\n") == ["release: completed", ENDED_RUNNING]
    assert fake.calls == [("release",)]


def test_a_refused_release_says_why(fake):
    reason = "the arm is not at rest: joint index 0 is at 0.2000 rad"
    fake.release_result = Result("thor-terminal-release-1", "refused", reason)
    assert typed(fake, "release\n")[0] == f"release: refused: {reason}"
    # Nothing else was asked of the app: no return to rest, no second try.
    assert fake.calls == [("release",)]


def test_quit_is_the_shutdown_path_and_asks_nothing_when_the_arm_need_not_move(fake):
    assert typed(fake, "quit\n") == [
        "quit: completed; the robot app closes",
        ENDED_RUNNING,
    ]
    assert fake.calls == [("shutdown", False)]


def test_quit_asks_that_the_path_is_clear_when_the_arm_has_to_move(fake):
    fake.moves_arm = True
    lines = typed(fake, "quit\n", " PATH is clear \n")
    assert lines == [
        CONFIRM_REST,
        "quit: completed; the robot app closes",
        ENDED_RUNNING,
    ]
    assert fake.calls == [("shutdown", True)]


@pytest.mark.parametrize("answer", ["no", "", "yes", "path clear", "rest", "quit"])
def test_quit_without_the_typed_phrase_is_passed_on_as_not_confirmed(fake, answer):
    fake.moves_arm = True
    lines = typed(fake, "quit\n", answer + "\n")
    assert lines == [
        CONFIRM_REST,
        "quit: refused: confirm the path is clear; the robot app keeps running",
        ENDED_RUNNING,
    ]
    # The answer is used up: it is not served as a command of its own.
    assert fake.calls == [("shutdown", False)]


def test_stop_typed_where_quit_asks_is_a_stop_and_confirms_nothing(fake):
    fake.moves_arm = True
    typed(fake, "quit\n", "stop\n")
    assert fake.calls == [("stop", STOP_TYPED, THOR), ("shutdown", False)]


def test_the_input_ending_where_quit_asks_confirms_nothing(fake):
    fake.moves_arm = True
    assert typed(fake, "quit\n")[:2] == [
        CONFIRM_REST,
        "quit: refused: confirm the path is clear; the robot app keeps running",
    ]
    assert fake.calls == [("shutdown", False)]


def test_a_refused_quit_says_why_and_that_the_app_keeps_running(fake):
    reason = "the return to rest ended refused: the jaws are not proven empty"
    fake.shutdown_result = Result("thor-terminal-shutdown-1", "refused", reason)
    assert typed(fake, "quit\n")[0] == (
        f"quit: refused: {reason}; the robot app keeps running"
    )
    # Nothing else was asked of the app: no stop, no second try.
    assert fake.calls == [("shutdown", False)]


@pytest.mark.parametrize("status", ["cancelled", "failed"])
def test_a_quit_that_ends_early_says_how(fake, status):
    fake.shutdown_result = Result("thor-terminal-shutdown-1", status, "STOP from x")
    assert typed(fake, "quit\n")[0] == (
        f"quit: {status}: STOP from x; the robot app keeps running"
    )


def test_a_quit_after_the_app_has_closed_does_not_say_it_keeps_running(fake):
    fake.running = False
    fake.shutdown_result = Result("thor-terminal-shutdown-1", "refused", NOT_RUNNING)
    assert typed(fake, "quit\n") == [
        "quit: refused: the robot app is not running",
        "the Thor terminal's input ended; the robot app has closed",
    ]


def test_a_request_that_raises_is_reported_and_the_terminal_goes_on(fake):
    fake.raises = RuntimeError("boom")
    lines = typed(fake, "rest\n", "yes\n", "quit\n", "stop\n", "help\n")
    assert lines[1] == "rest: the robot app raised RuntimeError('boom')"
    assert lines[2] == "quit: the robot app raised RuntimeError('boom')"
    assert lines[3] == "STOP: the robot app raised RuntimeError('boom')"
    assert lines[4] == HELP[0]
    assert [call[0] for call in fake.calls] == ["rest", "shutdown", "stop"]


def test_a_result_reason_cannot_ring_the_bell(fake):
    fake.shutdown_result = Result("id", "refused", "no\a\nway")
    assert typed(fake, "quit\n")[0] == (
        "quit: refused: no  way; the robot app keeps running"
    )


# The end of the input


def test_the_end_of_the_input_stops_nothing_and_asks_nothing(fake):
    assert typed(fake) == [ENDED_RUNNING]
    assert fake.calls == []


def test_after_the_input_ended_events_and_ctrl_c_still_work(fake, out, open_terminal):
    terminal = open_terminal(fake, iter(()), out)
    terminal.run()
    out.wait_for(ENDED_RUNNING)
    fake.emit("stop", "later")
    assert out.wait_for("[stop] later")
    terminal.interrupt()
    assert fake.wait_calls(1) == [("stop", STOP_CTRL_C, THOR)]


@pytest.mark.parametrize("error", [OSError(5, "Input/output error"), ValueError("x")])
def test_an_input_that_cannot_be_read_counts_as_ended(fake, error):
    def broken() -> Iterator[str]:
        yield "stop\n"
        raise error

    out = Lines()
    terminal = ThorTerminal(fake, broken(), out)
    terminal.run()
    terminal.close()
    assert out.lines == [BANNER, ENDED_RUNNING]
    assert fake.calls == [("stop", STOP_TYPED, THOR)]


# Operator events


def test_events_are_written_as_they_come_in_their_order(fake, out, open_terminal):
    open_terminal(fake, (), out).start()
    events = [
        ("help_needed", "HELP NEEDED: stuck"),
        ("countdown", "assistance wait: 120 s left"),
        ("refused", "open refused: confirm the jaw opening"),
        ("rest", "return to rest: the arm is at rest, motors on"),
    ]
    for kind, text in events:
        fake.emit(kind, text, {"n": 1})
    out.wait_for(f"[{events[-1][0]}] {events[-1][1]}")
    assert out.lines == [f"[{kind}] {text}" for kind, text in events]


def test_only_the_waits_bell_events_ring_the_bell(fake, out, open_terminal):
    open_terminal(fake, (), out).start()
    kinds = [
        "countdown",
        "bell",
        "expired",
        "stop",
        "fault",
        "reset",
        "rest",
        "release",
        "help_needed",
        "owner",
        "connection",
        "refused",
        "shutdown",
        "bell",
    ]
    for kind in kinds:
        fake.emit(kind, "ARM WILL MOVE TO REST in 30 s")
    fake.emit("end", "of the test")
    out.wait_for("[end] of the test")
    rang = [line for line in out.lines if BELL_CHAR in line]
    assert rang == [BELL_CHAR + "[bell] ARM WILL MOVE TO REST in 30 s"] * 2
    assert len(out.lines) == len(kinds) + 1
    assert BELL_CHAR == "\a"


def test_an_events_text_cannot_ring_the_bell_or_start_a_line(fake, out, open_terminal):
    open_terminal(fake, (), out).start()
    fake.emit("stop", "STOP from console: a\ab\nquit: completed\r\x1b[2Jc")
    fake.emit("be\all", "x")
    fake.emit("end", "of the test")
    out.wait_for("[end] of the test")
    assert out.lines[0] == "[stop] STOP from console: a b quit: completed  [2Jc"
    assert out.lines[1] == "[be ll] x"


OPEN_REFUSED = "open refused: confirm the jaw opening"
REST_REFUSED = "owner request 'rest' refused: confirm the path is clear"


def test_a_refusal_line_that_repeats_is_written_once_a_second_and_counted(
    fake, out, open_terminal, clock
):
    open_terminal(fake, (), out, clock).start()
    for _ in range(500):
        fake.emit("refused", OPEN_REFUSED)
    fake.emit("refused", REST_REFUSED)  # another refusal: its first line is shown
    fake.emit("countdown", "assistance wait: 30 s left")
    out.wait_for("[countdown] assistance wait: 30 s left")
    assert out.lines == [
        f"[refused] {OPEN_REFUSED}",
        f"[refused] {REST_REFUSED}",
        "[countdown] assistance wait: 30 s left",
    ]
    # Once the second is over, the next line of any kind tells the count first.
    clock.advance(1.0)
    fake.emit("countdown", "assistance wait: 20 s left")
    out.wait_for("[countdown] assistance wait: 20 s left")
    assert out.lines[3:] == [
        f"[refused] 499 more like: {OPEN_REFUSED}",
        "[countdown] assistance wait: 20 s left",
    ]
    # In a new second the same refusal is written again.
    fake.emit("refused", OPEN_REFUSED)
    out.wait_for(
        lambda line: line == f"[refused] {OPEN_REFUSED}" and len(out.lines) > 5
    )
    assert out.lines[5:] == [f"[refused] {OPEN_REFUSED}"]


def test_only_refusal_lines_are_limited(fake, out, open_terminal, clock):
    open_terminal(fake, (), out, clock).start()
    for kind in ("stop", "countdown", "bell", "rest", "fault"):
        for _ in range(3):
            fake.emit(kind, "the same text")
    fake.emit("end", "of the test")
    out.wait_for("[end] of the test")
    assert len(out.lines) == 16


def test_refusals_of_more_kinds_than_are_told_apart_are_counted_together(
    fake, out, open_terminal, clock
):
    open_terminal(fake, (), out, clock).start()
    for index in range(20):
        fake.emit("refused", f"command {index} refused")
    clock.advance(1.0)
    fake.emit("end", "of the test")
    out.wait_for("[end] of the test")
    assert out.lines == [
        *(f"[refused] command {index} refused" for index in range(16)),
        "[refused] 4 more refusals of other kinds",
        "[end] of the test",
    ]


def test_without_a_clock_the_terminal_counts_on_the_machines(fake, out, open_terminal):
    terminal = open_terminal(fake, (), out)
    assert type(terminal._refusals._clock).__name__ == "SystemClock"


def test_events_before_start_are_kept_and_written_then(fake, out, open_terminal):
    terminal = open_terminal(fake, (), out)
    fake.emit("stop", "early")
    assert not terminal.flush(0.0)  # nothing writes yet
    assert out.lines == []
    terminal.start()
    assert terminal.flush(WAIT_S)
    assert out.lines == ["[stop] early"]


def test_a_writer_that_raises_loses_lines_but_nothing_else(fake, open_terminal):
    written: list[str] = []
    failing = [True]

    def write(line: str) -> None:
        if failing[0]:
            raise OSError(5, "Input/output error")
        written.append(line)

    terminal = open_terminal(fake, iter(["stop\n", "quit\n"]), write)
    terminal.run()
    assert terminal.flush(WAIT_S)
    assert fake.calls == [("stop", STOP_TYPED, THOR), ("shutdown", False)]
    failing[0] = False
    fake.emit("stop", "after the terminal came back")
    assert terminal.flush(WAIT_S)
    assert written == ["[stop] after the terminal came back"]


def test_a_writer_that_blocks_holds_up_neither_events_nor_stops(fake, open_terminal):
    blocked, release = threading.Event(), threading.Event()
    written: list[str] = []

    def write(line: str) -> None:
        blocked.set()
        assert release.wait(WAIT_S)
        written.append(line)

    terminal = open_terminal(fake, TypedLines(), write)
    terminal.start()
    fake.emit("stop", "first")
    assert blocked.wait(WAIT_S)
    # The listener returns at once, as it must under the robot app's lock.
    for index in range(50):
        fake.emit("countdown", f"line {index}")
    terminal.interrupt()
    assert fake.wait_calls(1) == [("stop", STOP_CTRL_C, THOR)]
    assert written == []
    release.set()
    assert terminal.flush(WAIT_S)
    assert written[0] == "[stop] first" and len(written) == 51


# Ctrl+C


def test_ctrl_c_is_a_stop_raised_on_the_terminals_stop_thread(fake, open_terminal):
    terminal = open_terminal(fake)
    terminal.start()
    terminal.interrupt()
    assert fake.wait_calls(1) == [("stop", STOP_CTRL_C, THOR)]
    assert fake.threads == ["thor-terminal-stop"]
    assert STOP_CTRL_C == "Ctrl+C"


def test_the_sigint_handler_itself_calls_nothing(fake, open_terminal):
    terminal = open_terminal(fake)
    terminal._on_sigint(signal.SIGINT, None)
    terminal._on_sigint(signal.SIGINT, None)
    # No thread serves the requests yet, and the handler did not serve them.
    assert fake.calls == []
    terminal.start()
    assert len(fake.wait_calls(2)) == 2
    assert fake.threads == ["thor-terminal-stop"] * 2


def test_three_ctrl_c_are_three_stops_and_nothing_else(fake, out, open_terminal):
    terminal = open_terminal(fake, TypedLines(), out)
    terminal.start()
    for _ in range(3):
        terminal.interrupt()
    assert fake.wait_calls(3) == [("stop", STOP_CTRL_C, THOR)] * 3
    assert terminal.flush(WAIT_S)
    assert fake.calls == [("stop", STOP_CTRL_C, THOR)] * 3  # no quit, no rest
    assert terminal._writer.is_alive() and terminal._stopper.is_alive()


def test_a_stop_that_raises_does_not_end_the_stop_thread(fake, out, open_terminal):
    terminal = open_terminal(fake, (), out)
    terminal.start()
    fake.raises = RuntimeError("boom")
    terminal.interrupt()
    out.wait_for("STOP: the robot app raised RuntimeError('boom')")
    fake.raises = None
    terminal.interrupt()
    assert len(fake.wait_calls(2)) == 2


class Signals:
    """Stands in for signal.signal: records, and changes no real handler."""

    def __init__(self) -> None:
        self.installed: dict[int, object] = {}
        self.calls: list[tuple[int, object]] = []

    def __call__(self, number: int, handler: object) -> object:
        self.calls.append((number, handler))
        before = self.installed.get(number, f"before-{int(number)}")
        self.installed[number] = handler
        return before


@pytest.fixture
def signals(monkeypatch) -> Signals:
    stand_in = Signals()
    monkeypatch.setattr(terminal_module.signal, "signal", stand_in)
    return stand_in


def test_install_makes_sigint_a_stop_and_ignores_sighup(fake, signals, open_terminal):
    terminal = open_terminal(fake)
    terminal.install_signal_handlers()
    assert set(signals.installed) == {signal.SIGINT, signal.SIGHUP}
    assert signals.installed[signal.SIGHUP] is signal.SIG_IGN
    handler = signals.installed[signal.SIGINT]
    # The threads run already: a Ctrl+C before run() is served too.
    handler(signal.SIGINT, None)
    assert fake.wait_calls(1) == [("stop", STOP_CTRL_C, THOR)]
    assert fake.threads == ["thor-terminal-stop"]
    terminal.install_signal_handlers()  # a second call installs nothing new
    assert len(signals.calls) == 2


def test_the_banner_and_help_name_ctrl_c_only_once_it_is_installed(fake, signals):
    assert CTRL_C_NOTE not in "".join(typed(fake, "help\n"))
    out = Lines()
    terminal = ThorTerminal(fake, iter(["help\n"]), out)
    terminal.install_signal_handlers()
    terminal.run()
    assert terminal.flush(WAIT_S)
    assert out.lines[0] == f"{BANNER}; Ctrl+C is STOP"
    assert out.lines[-2] == "an empty line does nothing; Ctrl+C is STOP"
    terminal.close()


def test_install_is_for_the_main_thread_only(fake, signals, open_terminal):
    terminal = open_terminal(fake)
    raised: list[BaseException] = []

    def install() -> None:
        try:
            terminal.install_signal_handlers()
        except RuntimeError as error:
            raised.append(error)
        try:
            terminal.restore_signal_handlers()
        except RuntimeError as error:
            raised.append(error)

    thread = threading.Thread(target=install)
    thread.start()
    thread.join(WAIT_S)
    assert [str(error) for error in raised] == [
        "install_signal_handlers() must be called from the main thread",
        "restore_signal_handlers() must be called from the main thread",
    ]
    assert signals.calls == []
    assert not terminal._stopper.is_alive()  # nothing was started either


def test_install_on_a_closed_terminal_is_refused(fake, signals):
    terminal = ThorTerminal(fake, ())
    terminal.close()
    with pytest.raises(RuntimeError, match=r"closed: Ctrl\+C would do nothing"):
        terminal.install_signal_handlers()
    assert signals.calls == []


def test_close_on_the_main_thread_puts_the_handlers_back(fake, signals):
    terminal = ThorTerminal(fake, ())
    terminal.install_signal_handlers()
    terminal.close()
    assert signals.installed == {
        signal.SIGINT: f"before-{int(signal.SIGINT)}",
        signal.SIGHUP: f"before-{int(signal.SIGHUP)}",
    }
    assert not terminal._stopper.is_alive()
    terminal.restore_signal_handlers()  # nothing left to put back
    assert len(signals.calls) == 4


def test_a_handler_that_was_not_set_from_python_is_put_back_as_default(
    fake, monkeypatch
):
    calls: list[tuple[int, object]] = []

    def stand_in(number: int, handler: object) -> None:
        calls.append((number, handler))

    monkeypatch.setattr(terminal_module.signal, "signal", stand_in)
    terminal = ThorTerminal(fake, ())
    terminal.install_signal_handlers()
    terminal.close()
    assert calls[2:] == [
        (signal.SIGINT, signal.SIG_DFL),
        (signal.SIGHUP, signal.SIG_DFL),
    ]


def test_closed_off_the_main_thread_ctrl_c_stays_a_stop(fake, signals, out):
    terminal = ThorTerminal(fake, (), out)
    terminal.install_signal_handlers()
    closer = threading.Thread(target=terminal.close)
    closer.start()
    closer.join(WAIT_S)
    assert not terminal._writer.is_alive()
    # The handler is still installed, so the stop thread must still serve it.
    assert signals.installed[signal.SIGINT] == terminal._on_sigint
    signals.installed[signal.SIGINT](signal.SIGINT, None)
    assert fake.wait_calls(1) == [("stop", STOP_CTRL_C, THOR)]
    terminal.close()  # on the main thread: handlers back, stop thread ended
    assert signals.installed[signal.SIGINT] == f"before-{int(signal.SIGINT)}"
    terminal._stopper.join(WAIT_S)
    assert not terminal._stopper.is_alive()


# Life cycle


def test_close_ends_the_threads_after_the_queued_lines(fake, out):
    terminal = ThorTerminal(fake, (), out)
    terminal.start()
    for index in range(20):
        fake.emit("countdown", str(index))
    terminal.close()
    assert out.lines == [f"[countdown] {index}" for index in range(20)]
    assert not terminal._writer.is_alive() and not terminal._stopper.is_alive()
    fake.emit("stop", "after close")
    assert terminal._out.empty()  # a closed terminal keeps no lines
    terminal.close()  # a second close is harmless
    terminal.start()  # and a closed terminal does not start again
    assert out.lines[-1] == "[countdown] 19"


def test_flush_without_a_writer_returns_false_and_waits_for_nothing(fake):
    terminal = ThorTerminal(fake, ())
    assert not terminal.flush(WAIT_S)  # never started
    assert terminal._out.empty()
    terminal.start()
    assert terminal.flush(WAIT_S)
    terminal.close()
    assert not terminal.flush(WAIT_S)  # closed
    assert terminal._out.empty()


def test_a_terminal_that_was_never_started_closes(fake):
    terminal = ThorTerminal(fake, ())
    terminal.close()
    assert not terminal._writer.is_alive() and not terminal._stopper.is_alive()


def test_without_a_writer_the_lines_go_to_the_standard_output(fake, capsys):
    terminal = ThorTerminal(fake, iter(["help\n"]))
    terminal.run()
    terminal.close()
    assert capsys.readouterr().out.splitlines() == [
        BANNER,
        *HELP,
        "an empty line does nothing",
        ENDED_RUNNING,
    ]


def test_typed_lines_hand_over_in_order_and_end():
    lines = TypedLines()
    lines.type("one")
    lines.type("two")
    lines.end()
    lines.type("after the end")
    assert list(lines) == ["one", "two"]
    assert list(lines) == []  # an ended input stays ended


def test_typed_lines_wait_for_the_next_line():
    lines = TypedLines()
    got: list[str] = []
    reader = threading.Thread(target=lambda: got.extend(lines), daemon=True)
    reader.start()
    lines.type("late")
    lines.end()
    reader.join(WAIT_S)
    assert got == ["late"]


# Typed-ahead input at a question, on a pseudo-terminal and on a pipe

QUESTIONS = [
    pytest.param(b"rest", CONFIRM_REST, b"path is clear", "rest", id="rest"),
    pytest.param(b"quit", CONFIRM_REST, b"path is clear", "shutdown", id="quit"),
    pytest.param(
        b"reference", CONFIRM_REFERENCE, b"jaws are empty", "reference", id="reference"
    ),
]


@pytest.fixture
def terminal_pair() -> Iterator[tuple[int, int]]:
    """A pseudo-terminal: the test types at the first, the program reads the second."""
    master, slave = pty.openpty()
    yield master, slave
    os.close(master)
    os.close(slave)


@pytest.fixture
def pipe_pair() -> Iterator[tuple[int, int]]:
    """A pipe: the program reads the first, the test writes the second."""
    read_end, write_end = os.pipe()
    yield read_end, write_end
    os.close(write_end)
    os.close(read_end)


@pytest.fixture
def served(fake) -> Iterator[Callable[[int], Lines]]:
    """Serve the lines typed at a descriptor; returns what the terminal writes.

    Ask for the descriptors' fixture first: the reading ends before they close.
    """
    fake.moves_arm = fake.allow_reference = True  # so every command asks
    made: list[tuple] = []

    def serve(fd: int, output: Callable[[str], None] | None = None) -> Lines:
        out, lines = Lines(), robot_main._FdLines(fd)
        drop = lines.drop_typed_ahead
        out.typed = lines
        # Counts the questions that wait for their answer: what is typed from
        # then on was typed at the question.
        out.at_question = threading.Semaphore(0)

        def dropped() -> list[str]:
            gone = drop()
            out.at_question.release()
            return gone

        lines.drop_typed_ahead = dropped
        terminal = ThorTerminal(fake, lines, out if output is None else output)
        reader = threading.Thread(target=terminal.run, daemon=True)
        made.append((lines, reader, terminal))
        reader.start()
        return out

    yield serve
    for lines, reader, terminal in made:
        lines.close()
        reader.join(WAIT_S)
        terminal.close()


def answered_from_here(out: Lines, question: str) -> None:
    """Wait until the question is written and what was typed before it is dropped."""
    out.wait_for(question)
    assert out.at_question.acquire(timeout=WAIT_S), "the question does not wait"


@pytest.mark.parametrize(("command", "question", "phrase", "call"), QUESTIONS)
def test_an_answer_typed_before_its_question_is_dropped(
    fake, terminal_pair, served, command, question, phrase, call
):
    master, slave = terminal_pair
    os.write(master, command + b"\n" + phrase + b"\n")  # both before the question
    out = served(slave)
    answered_from_here(out, question)
    out.wait_for("dropped what was typed before the question: 1 line")
    os.write(master, b"no\n")  # typed at the question
    assert fake.wait_calls(1) == [(call, False)]
    os.write(master, b"help\n")  # the phrase was not left over as a command
    out.wait_for(HELP[0])
    assert not any(line.startswith("unknown command") for line in out.lines)


@pytest.mark.parametrize(("command", "question", "phrase", "call"), QUESTIONS)
def test_an_answer_typed_at_its_question_counts_on_a_terminal(
    fake, terminal_pair, served, command, question, phrase, call
):
    master, slave = terminal_pair
    os.write(master, command + b"\nno\nno\n")
    out = served(slave)
    answered_from_here(out, question)
    out.wait_for("dropped what was typed before the question: 2 lines")
    os.write(master, phrase + b"\n")
    assert fake.wait_calls(1) == [(call, True)]


def held_output(
    out: Lines, held: str
) -> tuple[Callable[[str], None], threading.Event, threading.Event]:
    """A writer that does not take the line `held` until it is let go.

    Returns (write, reached, release): as a terminal whose output is stopped
    (Ctrl+S) when that line is to be written.
    """
    reached, release = threading.Event(), threading.Event()

    def write(line: str) -> None:
        if line == held:
            reached.set()
            release.wait(WAIT_S)
        out(line)

    return write, reached, release


@pytest.mark.parametrize(("command", "question", "phrase", "call"), QUESTIONS)
def test_a_phrase_typed_while_its_question_is_not_written_confirms_nothing(
    fake, terminal_pair, served, monkeypatch, command, question, phrase, call
):
    monkeypatch.setattr(terminal_module, "QUESTION_WRITE_S", 0.2)
    master, slave = terminal_pair
    shown = Lines()
    write, reached, release = held_output(shown, question)
    served(slave, write)
    shown.wait_for(BANNER)
    os.write(master, command + b"\n")
    assert reached.wait(WAIT_S)  # asked, and the terminal does not show it
    os.write(master, phrase + b"\n")  # typed blind
    assert fake.wait_calls(1) == [(call, False)]
    assert question not in shown.lines
    release.set()
    # The phrase answered nothing: it is read as a command once the output runs.
    shown.wait_for(lambda line: line.startswith("unknown command"))
    assert fake.calls == [(call, False)]


def test_a_stop_typed_while_a_question_is_not_written_is_served_as_a_stop(
    fake, terminal_pair, served, monkeypatch
):
    monkeypatch.setattr(terminal_module, "QUESTION_WRITE_S", 0.2)
    master, slave = terminal_pair
    shown = Lines()
    write, reached, release = held_output(shown, CONFIRM_REST)
    served(slave, write)
    shown.wait_for(BANNER)
    os.write(master, b"rest\n")
    assert reached.wait(WAIT_S)
    os.write(master, b"stop\n")
    assert fake.wait_calls(2) == [("rest", False), ("stop", STOP_TYPED, THOR)]
    release.set()


def test_a_question_that_cannot_be_written_confirms_nothing(fake):
    fake.moves_arm = True

    def gone(line: str) -> None:
        raise OSError(5, "Input/output error")

    terminal = ThorTerminal(fake, iter(["rest\n", "path is clear\n"]), gone)
    terminal.run()
    terminal.close()
    # Nobody saw the question: the phrase is no answer to it.
    assert fake.calls == [("rest", False)]


def test_a_stop_typed_before_a_question_is_a_stop_and_the_question_is_withdrawn(
    fake, terminal_pair, served
):
    master, slave = terminal_pair
    os.write(master, b"rest\nstop\npath is clear\n")
    out = served(slave)
    assert fake.wait_calls(2) == [("stop", STOP_TYPED, THOR), ("rest", False)]
    out.wait_for(
        "a stop typed before the question was raised as a STOP; the question is "
        "withdrawn"
    )
    out.wait_for("rest: refused: confirm the path is clear")
    # The phrase typed after that stop went with it: it is no command either.
    os.write(master, b"help\n")
    out.wait_for(HELP[0])
    assert not any(line.startswith("unknown command") for line in out.lines)
    assert len(fake.calls) == 2


def test_a_stop_typed_at_a_question_is_a_stop_on_a_terminal(
    fake, terminal_pair, served
):
    master, slave = terminal_pair
    os.write(master, b"rest\n")
    out = served(slave)
    answered_from_here(out, CONFIRM_REST)
    assert fake.calls == []
    os.write(master, b"stop\n")
    assert fake.wait_calls(2) == [("stop", STOP_TYPED, THOR), ("rest", False)]
    assert not any(line.startswith("dropped what was typed") for line in out.lines)


@pytest.mark.parametrize(("command", "question", "phrase", "call"), QUESTIONS)
def test_a_phrase_the_end_of_the_input_cut_short_confirms_nothing(
    fake, terminal_pair, served, command, question, phrase, call
):
    master, slave = terminal_pair
    os.write(master, command + b"\n")
    out = served(slave)
    answered_from_here(out, question)
    # Typed and not entered: Ctrl+D hands the line over, a second ends the input.
    os.write(master, phrase + b"\x04\x04")
    assert fake.wait_calls(1) == [(call, False)]
    out.wait_for(ENDED_RUNNING)


def test_a_stop_the_end_of_the_input_cut_short_is_still_a_stop(
    fake, terminal_pair, served
):
    master, slave = terminal_pair
    os.write(master, b"rest\n")
    out = served(slave)
    answered_from_here(out, CONFIRM_REST)
    os.write(master, b"stop\x04\x04")
    assert fake.wait_calls(2) == [("stop", STOP_TYPED, THOR), ("rest", False)]


def test_a_phrase_typed_and_not_entered_when_the_terminal_goes_confirms_nothing(
    fake, served
):
    master, slave = pty.openpty()
    try:
        os.write(master, b"rest\n")
        out = served(slave)
        answered_from_here(out, CONFIRM_REST)
        os.write(master, b"path is clear\x04")  # handed over without its Enter
        by = time.monotonic() + WAIT_S
        while out.typed._buffer != b"path is clear":
            assert time.monotonic() < by, "the program did not read the phrase"
            time.sleep(0.01)
        os.close(master)  # then the window closes
        master = None
        assert fake.wait_calls(1) == [("rest", False)]
        assert out.typed.unterminated
    finally:
        for fd in (master, slave):
            if fd is not None:
                os.close(fd)


def test_a_last_line_of_a_pipe_without_a_line_end_confirms_nothing(fake, served):
    read_end, write_end = os.pipe()
    try:
        os.write(write_end, b"rest\npath is clear")
        os.close(write_end)
        served(read_end)
        assert fake.wait_calls(1) == [("rest", False)]
    finally:
        os.close(read_end)


def test_lines_read_before_the_terminal_went_away_answer_no_question():
    master, slave = pty.openpty()
    tty.setraw(slave)  # not in line mode: one read hands over every line typed
    try:
        os.write(master, b"rest\npath is clear\n")
        lines = robot_main._FdLines(slave)
        assert next(lines) == "rest"
        assert lines._buffer == b"path is clear\n"  # read with the line before
        os.close(master)  # the terminal goes away
        master = None
        assert lines.drop_typed_ahead() == ["path is clear"]
        assert lines._buffer == b"" and next(lines, None) is None
    finally:
        for fd in (master, slave):
            if fd is not None:
                os.close(fd)


def test_the_terminal_going_away_before_a_question_leaves_it_unanswered(fake):
    fake.moves_arm = True
    master, slave = pty.openpty()
    tty.setraw(slave)
    lines = robot_main._FdLines(slave)
    drop, out = lines.drop_typed_ahead, Lines()

    def gone_and_dropped() -> list[str]:
        nonlocal master
        os.close(master)  # just before the program looks at its input
        master = None
        return drop()

    lines.drop_typed_ahead = gone_and_dropped
    terminal = ThorTerminal(fake, lines, out)
    try:
        os.write(master, b"rest\npath is clear\n")
        terminal.run()
        assert fake.calls == [("rest", False)]
    finally:
        terminal.close()
        for fd in (master, slave):
            if fd is not None:
                os.close(fd)


@pytest.mark.parametrize(("command", "question", "phrase", "call"), QUESTIONS)
def test_commands_and_answers_from_a_pipe_are_all_taken(
    fake, pipe_pair, served, command, question, phrase, call
):
    read_end, write_end = pipe_pair
    os.write(write_end, command + b"\n" + phrase + b"\nstop\n")
    out = served(read_end)
    out.wait_for(question)
    assert fake.wait_calls(2) == [(call, True), ("stop", STOP_TYPED, THOR)]


def test_input_without_a_way_to_drop_typed_ahead_lines_is_served_as_before(fake):
    fake.moves_arm = True
    assert typed(fake, "rest", "path is clear", "quit", "path is clear") == [
        CONFIRM_REST,
        "rest: completed",
        CONFIRM_REST,
        "quit: completed; the robot app closes",
        ENDED_RUNNING,
    ]


# With the robot app itself, on the simulated arm


def test_typed_stop_ends_a_running_move_of_the_robot_app(robot_app, session):
    h = robot_app
    terminal = session(h.app)
    h.new_task()
    move_id, release = held_move(h)
    terminal.type("stop\n")
    terminal.out.wait_for(
        "[stop] STOP from thor_terminal: typed stop; the arm holds, stop number 1"
    )
    release.set()
    result = h.client.wait_result(move_id, timeout_s=WAIT_S)
    assert (result.status, result.reason) == (
        "cancelled",
        "STOP from thor_terminal: typed stop",
    )
    assert h.sent() == PATH[:6]
    assert h.app.snapshot().owner is Owner.NOBODY
    assert h.app.running and nothing_switched_off(h)


def test_typed_rest_with_the_phrase_returns_the_arm_to_rest(robot_app, session):
    h = robot_app
    h.arm.set_pose(RAISED)
    terminal = session(h.app)
    terminal.type("rest\n")
    terminal.out.wait_for(CONFIRM_REST)
    h.idle()
    assert h.arm.sent == []  # nothing moves before the answer
    terminal.type("path is clear\n")
    terminal.out.wait_for("rest: completed")
    assert h.arm.pose() == pytest.approx(REST, abs=h.config.rest.rest_tol_rad)
    assert h.app.motion_state is MotionState.REST_HOLD
    assert h.app.running and nothing_switched_off(h)
    assert "[rest] return to rest: the arm is at rest, motors on" in terminal.out.lines


@pytest.mark.parametrize("answer", ["no", "yes"])
def test_typed_rest_without_the_phrase_moves_nothing(robot_app, session, answer):
    h = robot_app
    h.arm.set_pose(RAISED)
    terminal = session(h.app)
    terminal.type("rest\n", answer + "\n")
    terminal.out.wait_for("rest: refused: confirm the path is clear")
    h.idle()
    assert h.arm.sent == []
    assert h.arm.pose() == pytest.approx(RAISED)
    assert h.app.running and nothing_switched_off(h)
    assert (
        "[refused] rest typed at the Thor terminal refused: confirm the path is clear"
    ) in terminal.out.lines


def test_typed_release_at_rest_switches_the_motors_off_and_the_app_runs_on(
    robot_app, session
):
    h = robot_app
    terminal = session(h.app)
    terminal.type("release\n")
    terminal.out.wait_for("release: completed")
    assert len(h.arm.released_at) == 1  # the fixture checks that it was at rest
    assert not h.arm.torque_on()
    assert h.app.running and not h.app.wait_closed(0.0)
    assert "[release] torque released at rest: the motors are off" in (
        terminal.out.lines
    )
    assert h.arm.sent == []


def test_typed_release_away_from_rest_is_refused_and_moves_nothing(robot_app, session):
    h = robot_app
    h.arm.set_pose(RAISED)
    terminal = session(h.app)
    terminal.type("release\n")
    answer = terminal.out.wait_for(lambda line: line.startswith("release: "))
    assert answer.startswith("release: refused: the arm is not at rest: joint index 0")
    h.idle()
    assert h.app.running and nothing_switched_off(h)
    assert h.arm.sent == []
    assert h.arm.pose() == pytest.approx(RAISED)


def test_quit_with_the_phrase_returns_to_rest_releases_there_and_ends_the_app(
    robot_app, session
):
    h = robot_app
    h.hold_at(RAISED)
    terminal = session(h.app)
    terminal.type("quit\n")
    terminal.out.wait_for(CONFIRM_REST)
    h.idle()
    assert h.arm.sent == []  # nothing moves before the answer
    terminal.type("path is clear\n")
    terminal.out.wait_for("quit: completed; the robot app closes")
    assert h.app.wait_closed(WAIT_S)
    assert not h.app.running
    assert len(h.arm.released_at) == 1  # the fixture checks that it was at rest
    assert not h.arm.torque_on()
    assert terminal.out.lines.index(
        "[shutdown] shutdown: the motors are off at rest; the robot app closes"
    ) < terminal.out.lines.index("quit: completed; the robot app closes")


@pytest.mark.parametrize("answer", ["no", "yes"])
def test_quit_without_the_phrase_moves_nothing_and_the_app_runs_on(
    robot_app, session, answer
):
    h = robot_app
    h.hold_at(RAISED)
    terminal = session(h.app)
    terminal.type("quit\n", answer + "\n")
    terminal.out.wait_for(
        "quit: refused: confirm the path is clear; the robot app keeps running"
    )
    h.idle()
    assert h.arm.sent == []
    assert h.arm.pose() == pytest.approx(RAISED)
    assert h.app.running and nothing_switched_off(h)
    refused = "[refused] shutdown typed at the Thor terminal refused: confirm the path"
    assert any(line.startswith(refused) for line in terminal.out.lines)


def test_quit_at_rest_asks_nothing(robot_app, session):
    h = robot_app
    terminal = session(h.app)
    terminal.type("quit\n")
    terminal.out.wait_for("quit: completed; the robot app closes")
    assert h.app.wait_closed(WAIT_S)
    assert CONFIRM_REST not in terminal.out.lines
    assert h.arm.sent == [] and len(h.arm.released_at) == 1


def test_a_refused_quit_says_why_and_the_app_keeps_running_and_holding(
    make_robot_app, session
):
    # Started away from rest the jaws are not referenced: no return to rest.
    h = make_robot_app(pose=RAISED)
    terminal = session(h.app)
    terminal.type("quit\n", "path is clear\n")
    answer = terminal.out.wait_for(lambda line: line.startswith("quit: "))
    assert answer.startswith(
        "quit: refused: the return to rest ended refused: the jaws are not proven "
        "empty (unknown)"
    )
    assert answer.endswith("; the robot app keeps running")
    h.idle()
    assert h.app.running and not h.app.wait_closed(0.0)
    assert nothing_switched_off(h)
    assert h.arm.sent == []
    assert h.arm.pose() == pytest.approx(RAISED)
    # The terminal is still served.
    terminal.type("stop\n")
    terminal.out.wait_for(lambda line: line.startswith("[stop] STOP from thor_"))


def test_quit_while_a_move_runs_is_refused_as_busy(robot_app, session):
    h = robot_app
    terminal = session(h.app)
    h.new_task()
    move_id, release = held_move(h)
    terminal.type("quit\n", "path is clear\n")
    terminal.out.wait_for(
        "quit: refused: busy: a move is running; the robot app keeps running"
    )
    release.set()
    assert h.client.wait_result(move_id, timeout_s=WAIT_S).status == "completed"
    assert h.app.running and nothing_switched_off(h)


@pytest.mark.parametrize("how", ["typed stop", "Ctrl+C"])
def test_a_stop_during_a_quit_ends_it_and_nothing_is_released(robot_app, session, how):
    h = robot_app
    h.hold_at(RAISED)
    reached, release = h.hold_move_after(5)
    terminal = session(h.app)
    terminal.type("quit\n", "path is clear\n")
    assert reached.wait(WAIT_S)  # the return to rest is under way
    if how == "Ctrl+C":
        terminal.terminal.interrupt()
    else:
        terminal.type("stop\n")  # typed lines are still served
    stop = f"STOP from thor_terminal: {how}"
    terminal.out.wait_for(f"[stop] {stop}; the arm holds, stop number 1")
    release.set()
    answer = terminal.out.wait_for(lambda line: line.startswith("quit: "))
    assert answer.startswith(
        f"quit: cancelled: the return to rest ended cancelled: {stop}"
    )
    assert answer.endswith("; the robot app keeps running")
    h.idle()
    assert len(h.arm.sent) == 5  # no setpoint after the STOP
    assert h.app.running and not h.app.wait_closed(0.0)
    assert nothing_switched_off(h)


def test_three_ctrl_c_are_three_stops_of_the_robot_app(robot_app, open_terminal, out):
    h = robot_app
    terminal = open_terminal(h.app, TypedLines(), out)
    terminal.start()
    h.new_task()
    move_id, release = held_move(h)
    for _ in range(3):
        terminal.interrupt()
    h.client.wait_state(lambda s: s.stop_number == 3)
    release.set()
    assert h.client.wait_result(move_id, timeout_s=WAIT_S).status == "cancelled"
    assert h.sent() == PATH[:6]
    assert terminal.flush(WAIT_S)
    stops = [line for line in out.lines if line.startswith("[stop] ")]
    assert [line.split("; ")[0] for line in stops] == [
        "[stop] STOP from thor_terminal: Ctrl+C"
    ] * 3
    assert [line.split("stop number ")[1][0] for line in stops] == ["1", "2", "3"]
    h.idle()
    assert h.app.running and not h.app.wait_closed(0.0)
    assert nothing_switched_off(h)
    assert h.arm.last_target()[0] == PATH[5]  # still holding its last target


def test_ctrl_c_and_typed_lines_work_with_the_robot_connection_down(robot_app, session):
    h = robot_app
    h.arm.set_pose(RAISED)
    terminal = session(h.app)
    h.robot_end.break_link()
    h.wait_local_event("connection")
    terminal.terminal.interrupt()
    terminal.out.wait_for(
        "[stop] STOP from thor_terminal: Ctrl+C; the arm holds, stop number 1"
    )
    terminal.type("rest\n", "path is clear\n")
    terminal.out.wait_for("rest: completed")
    assert h.arm.pose() == pytest.approx(REST, abs=h.config.rest.rest_tol_rad)
    assert h.app.running and nothing_switched_off(h)


def hold_dispatcher_after_a_result(h) -> None:
    """Hold the dispatcher where it has sent a final RESULT and not yet stepped.

    It goes on when the fake clock moves or the test waits for a dispatcher
    pass, whichever the test does first.
    """
    go = threading.Event()
    send, advance = h.robot_end.send_reliable, h.clock.advance
    wait_step = h.app.wait_step

    def send_reliable(msg_type: int, payload: bytes) -> None:
        send(msg_type, payload)
        if msg_type == Result.TYPE and b'"accepted"' not in payload:
            go.wait(WAIT_S)

    def advance_clock(seconds: float) -> None:
        advance(seconds)
        go.set()

    def wait_for_step(timeout_s: float = 2.0) -> bool:
        go.set()
        return wait_step(timeout_s)

    h.robot_end.send_reliable = send_reliable
    h.clock.advance = advance_clock
    h.app.wait_step = wait_for_step


@pytest.mark.parametrize(
    "held", [False, True], ids=["as it comes", "the RESULT is read before the step"]
)
def test_the_assistance_waits_bell_rings_at_the_thor_terminal(robot_app, session, held):
    h = robot_app
    terminal = session(h.app)
    h.new_task()
    if held:
        hold_dispatcher_after_a_result(h)
    command_id = h.client.request_owner("help_needed", reason="the agent is stuck")
    assert h.client.wait_result(command_id, timeout_s=WAIT_S).status == "completed"
    h.step()  # the wait's first step writes its start line; then the clock moves
    h.advance_until(lambda: h.heard("bell"))
    terminal.out.wait_for(BELL_CHAR + "[bell] ARM WILL MOVE TO REST in 30 s")
    assert terminal.terminal.flush(WAIT_S)
    lines = terminal.out.lines
    assert len([line for line in lines if BELL_CHAR in line]) == 1
    countdowns = [line for line in lines if line.startswith("[countdown] ")]
    assert countdowns[0] == (
        "[countdown] assistance wait: 120 s left, then the arm moves to rest "
        "(the agent is stuck)"
    )
    assert len(countdowns) == 10  # 120 s, then every 10 s down to 30 s
    assert any(line.startswith("[help_needed] HELP NEEDED") for line in lines)
    # A typed stop ends the wait: nothing moves by itself afterwards.
    terminal.type("stop\n")
    terminal.out.wait_for(
        lambda line: line.endswith("the assistance wait is cancelled")
    )
    h.advance(40.0)
    assert h.heard("expired") == [] and h.arm.sent == []


def test_the_end_of_the_input_changes_nothing_in_the_robot_app(robot_app, session):
    h = robot_app
    terminal = session(h.app)
    h.new_task()
    assert h.client.run_move(PATH, start_q=REST).status == "completed"
    before = h.app.snapshot()
    terminal.end_input()
    terminal.out.wait_for(ENDED_RUNNING)
    h.step()
    assert h.app.snapshot() == before  # no STOP, the agent still owns the arm
    assert h.app.running and not h.app.wait_closed(0.0)
    assert nothing_switched_off(h)
    assert h.arm.last_target()[0] == PATH[-1]  # still holding its last target
    # The agent's commands are still admitted.
    command_id = h.client.checker_state("clear_contact_spots")
    assert h.client.wait_result(command_id, timeout_s=WAIT_S).status == "completed"
