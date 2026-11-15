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
from typing import TYPE_CHECKING, Any

from .app import (
    COMPLETED,
    EVENT_REFUSED,
    NOT_RUNNING,
    PATH_CLEAR_PROMPT,
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
# The one answer that confirms that the path to rest is clear.
YES = "yes"
# Reasons of the STOPs raised here.
STOP_TYPED = "typed stop"
STOP_CTRL_C = "Ctrl+C"

CONFIRM_REST = f"{PATH_CLEAR_PROMPT} Type yes to confirm."
BANNER = "Thor terminal: type stop, rest, release, quit or help"
CTRL_C_NOTE = "Ctrl+C is STOP"
INPUT_ENDED = "the Thor terminal's input ended"
HELP = (
    "stop     STOP: the arm holds where it is, motors on",
    "rest     return the arm to rest; asks first that the path is clear",
    "release  switch the motors off; only at rest with the jaws proven empty",
    "quit     return to rest (asks like rest if the arm must move), motors off, exit",
    "help     this list",
)

# Longest close() waits, in real time, for one of the terminal's threads.
_JOIN_S = 2.0
# How much of an unknown command is shown back.
_SHOWN = 40


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
        # A line, a flush() marker, or None to end the output thread.
        self._out: queue.SimpleQueue[str | threading.Event | None] = queue.SimpleQueue()
        # Filled by the SIGINT handler: SimpleQueue.put() is reentrant, which a
        # threading.Event's set() is not.
        self._interrupts: queue.SimpleQueue[str | None] = queue.SimpleQueue()
        self._life = threading.Lock()
        self._started = False
        self._closed = False
        self._stopper_told = False  # the stop thread was told to end
        # The handlers install_signal_handlers() replaced; None while not installed.
        self._replaced: dict[int, Any] | None = None
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
        if word == "stop":
            self._stop(STOP_TYPED)
        elif word == "rest":
            self._rest()
        elif word == "release":
            self._request("release", self._app.request_release)
        elif word == "quit":
            self._quit()
        elif word == "help":
            for text in HELP:
                self._say(text)
            self._say(self._with_note("an empty line does nothing"))
        else:
            self._say(f"unknown command {line.strip()[:_SHOWN]!r}; type help")

    def _rest(self) -> None:
        """A typed rest: ask that the path is clear and pass the answer on."""
        confirmed = self._path_is_clear()
        self._request("rest", lambda done: self._app.request_rest(confirmed, done))

    def _quit(self) -> None:
        """A typed quit: the same question, asked only when the arm has to move."""
        confirmed = self._app.shutdown_moves_arm() and self._path_is_clear()
        self._request("quit", lambda done: self._app.request_shutdown(confirmed, done))

    def _path_is_clear(self) -> bool:
        """Ask that the path to rest is clear; only a typed yes confirms it."""
        self._say(CONFIRM_REST)
        answer = self._next_line()
        word = "" if answer is None else answer.strip().lower()
        if word == "stop":
            self._stop(STOP_TYPED)  # a stop is a STOP wherever it is typed
        return word == YES

    def _request(self, what: str, request: Callable[[DoneFn], object]) -> None:
        """Ask the robot app for something; its final RESULT becomes one line."""

        def done(result: Result) -> None:
            # Can run under the robot app's lock: it only queues the line.
            self._say(_answer(what, result))

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
            else:
                self._write_line(item)

    def _write_line(self, line: str) -> None:
        try:
            self._write(line)
        except Exception as error:
            # A closed terminal: the robot app runs on without its lines.
            if not self._write_failed:
                _log.warning("the Thor terminal cannot be written to: %r", error)
            self._write_failed = True
        else:
            self._write_failed = False


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
