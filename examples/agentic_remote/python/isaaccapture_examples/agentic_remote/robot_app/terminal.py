# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The Thor terminal: what the operator types at the robot, and what it shows.

See sections 9, 11 and 13 of docs/design/claude_agentic_remote_execution.md.
"""

from __future__ import annotations

import logging
import queue
import signal
import threading
from collections.abc import Callable, Iterable, Iterator
from typing import TYPE_CHECKING, Any, ClassVar

from ..protocol.confirm import JAWS_ARE_EMPTY, PATH_IS_CLEAR, confirms
from ..protocol.confirm import YES as YES  # re-exported: the entry point imports it
from .app import (
    COMPLETED,
    EVENT_REFUSED,
    NOT_RUNNING,
    PATH_CLEAR_PROMPT,
    POWER_ON_PROMPT,
    THOR_TERMINAL,
)
from .clock import SystemClock
from .text import LineLimit
from .waits import BELL

if TYPE_CHECKING:
    from ..protocol.messages import Result
    from .app import RobotApp
    from .clock import Clock

    # Told the final RESULT of a typed request.
    DoneFn = Callable[[Result], None]

_log = logging.getLogger("isaaccapture.examples.agentic_remote.terminal")

BELL_CHAR = "\a"
# Reasons of the STOPs raised here.
STOP_TYPED = "typed stop"
STOP_CTRL_C = "Ctrl+C"

CONFIRM_REST = f'{PATH_CLEAR_PROMPT} Type "{PATH_IS_CLEAR}" to confirm.'
CONFIRM_REFERENCE = (
    "Reference the jaws: they close slowly to their stop. Are the jaws empty? "
    f'Type "{JAWS_ARE_EMPTY}" to confirm.'
)
CONFIRM_OPEN = (
    f'Open the jaws: whatever they hold is let go. Type "{YES}" to open the jaws.'
)
# Asked before a typed close only while the jaws are not proven empty.
CONFIRM_CLOSE = (
    "Close the jaws: they are not proven empty ({why}) and close on whatever "
    f'they hold. Type "{YES}" to close the jaws.'
)
# How the robot app's refusal reads when only the jaws stand in the way of a
# rest, a release or a quit (rest.py), and what the terminal asks then. The
# typed word goes with the command, which is asked again.
JAWS_NOT_PROVEN = "the jaws are not proven empty"
ASK_JAWS_EMPTY = (
    "{what} was refused because the jaws are not proven empty. Look at the "
    f"jaws. If they hold something: stop, open, {YES}. If they are empty: type "
    f'"{JAWS_ARE_EMPTY}" and {{what}} is asked again on your word.'
)
# Asked before the motors are switched on again after a rest or a release.
CONFIRM_POWER_ON = f'{POWER_ON_PROMPT} Type "{YES}" to switch the motors on.'
BANNER = "Thor terminal: type stop, rest, release, on, reset, open, close, quit or help"
CTRL_C_NOTE = "Ctrl+C is STOP"
INPUT_ENDED = "the Thor terminal's input ended"
# Said in place of an answer when a stop was typed ahead of a question.
STOP_AHEAD = (
    "a stop typed before the question was raised as a STOP; the question is withdrawn"
)
HELP = (
    "stop       STOP: the arm holds where it is, motors on",
    f'rest       asks first for a typed "{PATH_IS_CLEAR}"; then the arm returns '
    "to rest, rest is measured, the jaws close and the motors go off",
    "release    at measured rest: the jaws close and the motors go off; away "
    "from rest nothing happens",
    "on         switch the motors on again after a rest or a release; asks "
    f'first for a typed "{YES}"',
    "reset      clear a fault-stop; the arm holds and nothing is started",
    "quit       as rest (asks only if the arm moves), then the robot app ends",
    "reference  make the jaw position known again: the jaws close slowly to "
    f'their stop; asks first for a typed "{JAWS_ARE_EMPTY}"; only with the '
    "robot app setting jaws.allow_reference",
    f'open       open the jaws in one go; asks first for a typed "{YES}"; type '
    "stop first: nobody may own the arm",
    "close      close the jaws; asks first while they are not proven empty; "
    "type stop first: nobody may own the arm",
    "help       this list",
)

# Longest close() waits, in real time, for one of the terminal's threads.
_JOIN_S = 2.0
# Longest a question waits, in real time, to be written before it counts as
# not asked. The thread that serves the typed lines serves none meanwhile.
QUESTION_WRITE_S = 1.0
# How much of an unknown command is shown back.
_SHOWN = 40


class _Question:
    """A question on its way to the output thread, which notes how its write went."""

    def __init__(self, line: str) -> None:
        self.line = line
        self.written = False
        self.done = threading.Event()


class TypedLines:
    """Lines a script types in place of a person, for the self-test and tests.

    Iterating waits for the next line; end() ends the input as a closed
    terminal does.
    """

    def __init__(self) -> None:
        self._lines: queue.SimpleQueue[str | None] = queue.SimpleQueue()
        self._ended = False

    def type(self, line: str) -> None:
        self._lines.put(line)

    def end(self) -> None:
        self._lines.put(None)

    def __iter__(self) -> Iterator[str]:
        return self

    def __next__(self) -> str:
        if not self._ended:
            line = self._lines.get()
            if line is not None:
                return line
            self._ended = True
        raise StopIteration


class ThorTerminal:
    """Typed commands in, operator lines out, Ctrl+C as a STOP.

    run() serves the typed lines. One thread of the terminal writes its lines,
    so a terminal that blocks or is gone never holds up the robot app; another
    turns each Ctrl+C into a STOP. The terminal itself never ends the program
    or switches the motors off: release and quit ask the robot app.
    """

    def __init__(
        self,
        app: RobotApp,
        input_lines: Iterable[str],
        output: Callable[[str], None] | None = None,
        *,
        clock: Clock | None = None,
    ) -> None:
        """input_lines yields the typed lines; output writes one line of text.

        output None prints to the standard output. Operator events are queued
        from here on and written once start() or run() was called. A refusal
        line that repeats is written once per second of `clock` (None: the
        machine's clock) and its repeats are counted.
        """
        self._app = app
        self._refusals = LineLimit(SystemClock() if clock is None else clock)
        self._input = iter(input_lines)
        self._input_ended = False
        self._write = _print_line if output is None else output
        self._write_failed = False
        # A line, a question, a flush() marker, or None to end the output thread.
        self._out: queue.SimpleQueue[str | _Question | threading.Event | None] = (
            queue.SimpleQueue()
        )
        # Filled by the SIGINT handler: SimpleQueue.put() is reentrant, which a
        # threading.Event's set() is not.
        self._interrupts: queue.SimpleQueue[str | None] = queue.SimpleQueue()
        self._life = threading.Lock()
        self._started = False
        self._closed = False
        self._stopper_told = False  # the stop thread was told to end
        # The handlers install_signal_handlers() replaced; None while not installed.
        self._replaced: dict[int, Any] | None = None
        # The command the robot app refused because the jaws are not proven
        # empty, while the operator may answer that they are; else None.
        self._jaws_offer: str | None = None
        self._writer = threading.Thread(
            target=self._write_loop, name="thor-terminal-output", daemon=True
        )
        self._stopper = threading.Thread(
            target=self._stop_loop, name="thor-terminal-stop", daemon=True
        )
        app.add_event_listener(self._on_event)

    # Life cycle

    def start(self) -> None:
        """Start the output thread and the stop thread; run() does it too."""
        with self._life:
            if self._started:
                return
            self._started = True
            self._writer.start()
            self._stopper.start()

    def run(self) -> None:
        """Serve the typed lines until the input ends.

        The end of the input, a closed terminal, changes nothing: the robot app
        is not stopped and nothing is switched off.
        """
        self.start()
        self._say(self._with_note(BANNER))
        while (line := self._next_line()) is not None:
            self._serve(line)
        running = bool(self._app.running)
        after = "the robot app runs on" if running else "the robot app has closed"
        _log.info("%s; %s", INPUT_ENDED, after)
        self._say(f"{INPUT_ENDED}; {after}")

    def flush(self, timeout_s: float = 2.0) -> bool:
        """Wait in real time until the lines queued so far are written."""
        with self._life:
            if not self._started or self._closed:
                return False
            written = threading.Event()
            self._out.put(written)
        return written.wait(timeout_s)

    def close(self) -> None:
        """End the terminal's threads once the queued lines are written.

        On the main thread its signal handlers are put back first. While its
        SIGINT handler stays installed the stop thread runs on, so Ctrl+C stays
        a STOP.
        """
        if threading.current_thread() is threading.main_thread():
            self.restore_signal_handlers()
        ending: list[threading.Thread] = []
        with self._life:
            if not self._closed:
                self._closed = True
                self._out.put(None)
                ending.append(self._writer)
            if self._replaced is None and not self._stopper_told:
                self._stopper_told = True
                self._interrupts.put(None)
                ending.append(self._stopper)
            started = self._started
        for thread in ending:
            if started and thread is not threading.current_thread():
                thread.join(_JOIN_S)

    # Ctrl+C

    def install_signal_handlers(self) -> None:
        """Make Ctrl+C a STOP and ignore a closed terminal's SIGHUP.

        Only the main thread may call it. The handler only queues the request;
        the stop thread raises the STOP, so Ctrl+C never ends the program.
        """
        _main_thread_only("install_signal_handlers()")
        if self._replaced is not None:
            return
        if self._closed:
            raise RuntimeError("the terminal is closed: Ctrl+C would do nothing")
        self.start()
        replaced = {signal.SIGINT: signal.signal(signal.SIGINT, self._on_sigint)}
        self._replaced = replaced
        hangup = getattr(signal, "SIGHUP", None)  # absent on Windows
        if hangup is not None:
            replaced[hangup] = signal.signal(hangup, signal.SIG_IGN)

    def restore_signal_handlers(self) -> None:
        """Put back the handlers install_signal_handlers() replaced.

        Only the main thread may call it; it does nothing when none is installed.
        """
        _main_thread_only("restore_signal_handlers()")
        replaced, self._replaced = self._replaced, None
        for number, handler in (replaced or {}).items():
            # None stands for a handler that was not installed from Python.
            signal.signal(number, signal.SIG_DFL if handler is None else handler)

    def interrupt(self) -> None:
        """What Ctrl+C does: ask the stop thread for a STOP.

        Safe in a signal handler: it takes no lock.
        """
        self._interrupts.put(STOP_CTRL_C)

    def _on_sigint(self, signum: int, stack: object) -> None:
        self.interrupt()

    def _stop_loop(self) -> None:
        while (reason := self._interrupts.get()) is not None:
            self._stop(reason)

    # Typed lines

    def _next_line(self) -> str | None:
        """The next typed line; None once the input has ended."""
        if self._input_ended:
            return None
        try:
            return next(self._input)
        except StopIteration:
            pass
        except (OSError, ValueError) as error:  # the terminal is gone
            _log.warning("the Thor terminal cannot be read: %r", error)
        self._input_ended = True
        return None

    def _serve(self, line: str) -> None:
        word = line.strip().lower()
        if not word:
            return
        # The question about the jaws stands for the next typed line only.
        offered, self._jaws_offer = self._jaws_offer, None
        if word == "stop":
            self._stop(STOP_TYPED)
        elif word == "rest":
            self._rest()
        elif word == "release":
            self._release()
        elif word == "on":
            self._power_on()
        elif word == "reset":
            self._request("reset", self._app.request_reset)
        elif word == "quit":
            self._quit()
        elif word == "reference":
            self._reference()
        elif word == "open":
            self._open()
        elif word == "close":
            self._close_jaws()
        elif offered is not None and confirms(line, JAWS_ARE_EMPTY):
            # The answer to the question a refusal for the jaws put.
            self._ASKED_AGAIN[offered](self, True)
        elif word == "help":
            for text in HELP:
                self._say(text)
            self._say(self._with_note("an empty line does nothing"))
        else:
            self._say(f"unknown command {line.strip()[:_SHOWN]!r}; type help")

    def _rest(self, jaws_empty: bool = False) -> None:
        """A typed rest: ask that the path is clear and pass the answer on.

        jaws_empty, here and below: the operator typed that the jaws are
        empty, after the robot app refused the command for jaws that are not
        proven empty.
        """
        confirmed = self._path_is_clear()
        word = _word(jaws_empty)
        self._request(
            "rest", lambda done: self._app.request_rest(confirmed, done, **word)
        )

    def _release(self, jaws_empty: bool = False) -> None:
        """A typed release: the robot app does it only at measured rest."""
        word = _word(jaws_empty)
        self._request("release", lambda done: self._app.request_release(done, **word))

    def _quit(self, jaws_empty: bool = False) -> None:
        """A typed quit: the same question, asked only when the arm has to move."""
        confirmed = self._app.shutdown_moves_arm() and self._path_is_clear()
        word = _word(jaws_empty)
        self._request(
            "quit", lambda done: self._app.request_shutdown(confirmed, done, **word)
        )

    def _power_on(self) -> None:
        """A typed on: asked only where the motors were switched off at rest."""
        confirmed = self._app.motors_off_at_rest and self._asked(CONFIRM_POWER_ON, YES)
        self._request("on", lambda done: self._app.request_power_on(confirmed, done))

    # The commands asked again on the operator's word that the jaws are empty.
    _ASKED_AGAIN: ClassVar[dict[str, Callable[..., None]]] = {
        "rest": _rest,
        "release": _release,
        "quit": _quit,
    }

    def _open(self) -> None:
        """A typed open: the driver's own opening, after a typed confirmation."""
        confirmed = self._asked(CONFIRM_OPEN, YES)
        self._request("open", lambda done: self._app.request_open(confirmed, done))

    def _close_jaws(self) -> None:
        """A typed close: asked first only while the jaws are not proven empty."""
        why = self._app.jaws_not_proven()
        confirmed = why is not None and self._asked(
            CONFIRM_CLOSE.format(why=_printable(why)), YES
        )
        self._request("close", lambda done: self._app.request_close(confirmed, done))

    def _reference(self) -> None:
        """A typed reference: asked only where the setting allows the command."""
        confirmed = self._app.allow_reference and self._asked(
            CONFIRM_REFERENCE, JAWS_ARE_EMPTY
        )
        self._request(
            "reference", lambda done: self._app.request_reference(confirmed, done)
        )

    def _path_is_clear(self) -> bool:
        """Ask that the path to rest is clear; only the typed phrase confirms it."""
        return self._asked(CONFIRM_REST, PATH_IS_CLEAR)

    def _asked(self, question: str, phrase: str) -> bool:
        """Ask a question; True only when the next typed line is the phrase.

        The question is written first; one not written in time confirms nothing.
        On a terminal, what was typed before it was written is dropped, and a
        stop among it is a STOP that withdraws the question. A line that the
        end of the input cut short confirms nothing.
        """
        if not self._written(question):
            return False
        dropped = drop_typed_ahead(self._input)
        if any(confirms(line, "stop") for line in dropped):
            self._stop(STOP_TYPED)
            self._say(STOP_AHEAD)
            return False
        if dropped:
            self._say(_dropped_ahead(len(dropped)))
        answer = self._next_line()
        if confirms(answer, "stop"):
            self._stop(STOP_TYPED)  # where a question waits a stop is a STOP
            return False
        if getattr(self._input, "unterminated", False):
            return False  # typed and not entered
        return confirms(answer, phrase)

    def _written(self, question: str) -> bool:
        """Queue a question; True once the output thread has written it."""
        asked = _Question(question)
        with self._life:
            if not self._started or self._closed:
                return False
            self._out.put(asked)
        return asked.done.wait(QUESTION_WRITE_S) and asked.written

    def _request(self, what: str, request: Callable[[DoneFn], object]) -> None:
        """Ask the robot app for something; its final RESULT becomes one line."""

        def done(result: Result) -> None:
            # Can run under the robot app's lock: it only queues the lines.
            self._say(_answer(what, result))
            if what in self._ASKED_AGAIN and _for_the_jaws(result):
                # The operator is asked; the next typed line may be the word.
                self._jaws_offer = what
                self._say(ASK_JAWS_EMPTY.format(what=what))

        try:
            request(done)
        except Exception as error:
            _log.exception("%s typed at the Thor terminal raised", what)
            self._say(f"{what}: the robot app raised {error!r}")

    def _stop(self, reason: str) -> None:
        """STOP through the robot app, whose event line tells the operator."""
        try:
            self._app.stop(reason, THOR_TERMINAL)
        except Exception as error:
            _log.exception("a STOP from the Thor terminal raised")
            self._say(f"STOP: the robot app raised {error!r}")

    # Output

    def _on_event(self, kind: str, text: str, data: dict[str, Any]) -> None:
        """One operator event as a line; only the wait's bell rings the bell.

        It can run under the robot app's lock: it only queues the line.
        """
        line = f"[{_printable(kind)}] {_printable(text)}"
        for repeated, count in self._refusals.ended():
            self._say(_more_refusals(count, repeated))
        if kind == EVENT_REFUSED and not self._refusals.allow(line):
            return  # counted; said with the next line once the second is over
        self._say(BELL_CHAR + line if kind == BELL else line)

    def _say(self, line: str) -> None:
        if not self._closed:
            self._out.put(line)

    def _with_note(self, line: str) -> str:
        """The line, with the Ctrl+C note while its handler is installed."""
        return line if self._replaced is None else f"{line}; {CTRL_C_NOTE}"

    def _write_loop(self) -> None:
        while (item := self._out.get()) is not None:
            if isinstance(item, threading.Event):
                item.set()
            elif isinstance(item, _Question):
                item.written = self._write_line(item.line)
                item.done.set()
            else:
                self._write_line(item)

    def _write_line(self, line: str) -> bool:
        """Write one line; False when the terminal did not take it."""
        try:
            self._write(line)
        except Exception as error:
            # A closed terminal: the robot app runs on without its lines.
            if not self._write_failed:
                _log.warning("the Thor terminal cannot be written to: %r", error)
            self._write_failed = True
        else:
            self._write_failed = False
        return not self._write_failed


def drop_typed_ahead(lines: object) -> list[str]:
    """Drop the input typed so far, where the input can; the whole lines dropped.

    Called once a question is written. Input with a drop_typed_ahead() of its
    own, the lines typed at a terminal, drops it; any other input, a pipe or a
    script's lines, is left as it is.
    """
    drop = getattr(lines, "drop_typed_ahead", None)
    return [] if drop is None else list(drop() or ())


def _dropped_ahead(count: int) -> str:
    """The line that tells of typed-ahead lines that were dropped."""
    lines = "1 line" if count == 1 else f"{count} lines"
    return f"dropped what was typed before the question: {lines}"


def _main_thread_only(what: str) -> None:
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError(f"{what} must be called from the main thread")


def _print_line(text: str) -> None:
    print(text, flush=True)


def _printable(text: object) -> str:
    """Text for one line: control characters, the bell among them, become spaces."""
    return "".join(ch if ch.isprintable() else " " for ch in str(text))


def _more_refusals(count: int, line: object) -> str:
    """The line that stands for `count` refusal lines that were not written."""
    head = f"[{EVENT_REFUSED}] "
    if line is None:
        return f"{head}{count} more refusals of other kinds"
    return f"{head}{count} more like: {str(line).removeprefix(head)}"


def _word(jaws_empty: bool) -> dict[str, bool]:
    """The keyword that carries the operator's word to the robot app, if given."""
    return {"jaws_empty": True} if jaws_empty else {}


def _for_the_jaws(result: Result) -> bool:
    """Whether the robot app refused a request for jaws that are not proven empty."""
    return result.status != COMPLETED and JAWS_NOT_PROVEN in result.reason


def _answer(what: str, result: Result) -> str:
    """The final RESULT of a typed request as the terminal's line."""
    line = f"{what}: {_printable(result.status)}"
    if result.reason:
        line += f": {_printable(result.reason)}"
    if what != "quit":
        return line
    if result.status == COMPLETED:
        return f"{line}; the robot app closes"
    if result.reason == NOT_RUNNING:
        return line
    return f"{line}; the robot app keeps running"
