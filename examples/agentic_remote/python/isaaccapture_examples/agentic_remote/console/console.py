# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""python -m isaaccapture_examples.agentic_remote.console: the operator console.

A line printer on the local API in the role "console": it prints the robot
connection, the ROBOT_STATE, every EVENT and every refusal, and sends what the
operator types. RESET, OPEN, RELEASE, SHUTDOWN and rest go out through
agent/console_cmd.py, which stays their only sender. It imports no APAI and
no Isaac Capture.
"""

from __future__ import annotations

import argparse
import contextlib
import logging
import queue
import signal
import sys
import termios
import threading
from collections.abc import Callable, Iterable, Iterator, Sequence
from typing import TYPE_CHECKING, NamedTuple

from ..agent.console_cmd import (
    CONFIRMED,
    NOT_CONFIRMED,
    SENDER,
    STATE_WAIT_S,
    STOP_REASON,
    UNDELIVERED,
    NotConfirmed,
    question,
    raw_stop_text,
    result_text,
    send_command,
    state_text,
    status_text,
    stop_shows,
)
from ..console_log import set_console_level
from ..link_service.local import LocalCarrier, RawStop, connect_local, send_raw_stop
from ..protocol.confirm import agent_environment, confirms
from ..protocol.local_messages import ROLE_CONSOLE
from ..protocol.messages import Event, Result, RobotState, Welcome
from ..robot_app.clock import SystemClock
from ..robot_app.text import MAX_OUT_CHARS, MAX_WORD_CHARS, LineLimit, printable
from ..testing.client import ACCEPTED, WorkstationClient
from .status import (
    KIND_BELL,
    KIND_HELP_NEEDED,
    KIND_REFUSED,
    help_block,
    status_key,
    status_line,
    welcome_line,
)

if TYPE_CHECKING:
    from ..protocol.local_messages import LocalNotice
    from ..protocol.messages import Message
    from ..robot_app.clock import Clock

_log = logging.getLogger("isaaccapture.examples.agentic_remote.console")

EXIT_OK = 0
EXIT_UNUSABLE = 2  # unusable arguments, no terminal, or an agent's environment

PROGRAM = "operator console"
NAME = "console"
LOG_LEVELS = ("debug", "info", "warning", "error", "critical")
STOP_CTRL_C = "Ctrl+C at the operator console"
BELL_CHAR = "\a"

# A status line is printed when it changes, this far apart at least, and when
# nothing changes this often.
STATUS_CHANGE_S = 1.0
STATUS_EVERY_S = 10.0
# How long a STOP waits for a ROBOT_STATE with a higher stop number.
STOP_WAIT_S = 5.0
# Not attached: one attempt every ATTACH_EVERY_S, ATTACH_TRIES times, then
# only on the typed attach.
ATTACH_EVERY_S = 2.0
ATTACH_TRIES = 30
ATTACH_TIMEOUT_S = 2.0
# Real seconds between two looks at the clock, the carrier and the waits.
TICK_S = 0.05
# Longest wait, in real time, for one of the console's threads or its output.
_JOIN_S = 2.0
# Lines that may wait for a terminal that does not take them; more are dropped.
_LINES_WAITING = 10000
_SHOWN = 40  # how much of an unknown command is shown back

BANNER = (
    "operator console: type stop, reset, rest, open, release, shutdown, state, "
    "attach, help or quit; Ctrl+C is STOP"
)
HELP = (
    "stop      STOP: the arm holds where it is, motors on; ends a running task",
    "reset     clear a fault-stop; starts nothing",
    'rest      return the arm to rest; asks first for a typed "path is clear"',
    'open      open the jaws; asks first for a typed "yes"',
    "release   switch the motors off; the robot app does it only at measured rest",
    "shutdown  rest, motors off, end the robot app; asks first",
    "state     the robot connection and the newest ROBOT_STATE; sends nothing",
    "attach    try once more to attach to the robot link service",
    "help      this list",
    "quit      leave this console; nothing is sent and the robot app runs on",
    "Ctrl+C    takes back an open question, then STOP; it never ends the console",
)
QUIT_NOTE = (
    "the operator console ends; nothing was sent for that: the robot app runs on"
)
USE_THOR = "use the STOP key at the Thor terminal"
NOT_ATTACHED_HELP = (
    "not attached: stop (written raw, not confirmed), attach, help and quit work"
)

Say = Callable[[str], None]
Attach = Callable[..., LocalCarrier]


class _Station(NamedTuple):
    """The console's attachment: its carrier and the client on it."""

    carrier: LocalCarrier
    client: WorkstationClient


class _Pending(NamedTuple):
    """A STOP that was sent and that no ROBOT_STATE has shown yet."""

    before: RobotState | None  # the newest state when it was sent
    by: float  # console clock time at which it counts as not confirmed
    carrier: LocalCarrier


class Console:
    """Typed operator commands in, lines out; see the module text.

    run() serves the typed lines on the caller's thread. A writer thread prints,
    so a terminal that blocks holds up neither a STOP nor the receive thread;
    a stop thread turns each interrupt() into a STOP; a ticker calls tick().
    """

    def __init__(
        self,
        lines: Iterable[str],
        say: Say | None = None,
        *,
        socket_path: str | None = None,
        name: str = NAME,
        attach: Attach = connect_local,
        raw_stop: Callable[..., RawStop] = send_raw_stop,
        clock: Clock | None = None,
        flush_input: Callable[[], None] | None = None,
        tick_s: float | None = TICK_S,
    ) -> None:
        """lines yields the typed lines; say writes one line (None: print).

        flush_input drops what was typed and not yet read; it runs before a
        question is answered. tick_s None starts no ticker: the caller calls
        tick(), which is how a test moves a fake clock.
        """
        self._lines = iter(lines)
        self._write = _print_line if say is None else say
        self._path = socket_path
        self._name = name
        self._attach_fn = attach
        self._raw_stop = raw_stop
        self._clock: Clock = SystemClock() if clock is None else clock
        self._flush_input = flush_input
        self._tick_s = tick_s
        self._refusals = LineLimit(self._clock)
        # A line, a flush marker, or None to end the writer.
        self._out: queue.SimpleQueue[str | threading.Event | None] = queue.SimpleQueue()
        # Filled by interrupt(): SimpleQueue.put() is safe in a signal handler.
        self._interrupts: queue.SimpleQueue[str | None] = queue.SimpleQueue()
        self._lock = threading.RLock()  # guards everything below
        self._station: _Station | None = None
        self._welcome: Welcome | None = None
        self._state: RobotState | None = None
        self._state_at = 0.0  # console clock time the state arrived
        self._mine: dict[str, str] = {}  # command id -> the command typed
        self._pending: _Pending | None = None
        self._question = 0  # number of the open question; 0 when none is open
        self._asked = 0  # questions asked so far
        self._robot_up: bool | None = None  # of the newest LOCAL_STATUS seen
        self._notices: list[LocalNotice] = []  # of the carrier, as last seen
        self._tries_left = ATTACH_TRIES
        self._next_try = 0.0
        self._why = ""  # why the last attempt to attach failed
        self._status_key: object = None
        self._status_at: float | None = None
        self._attaching = threading.Lock()  # one attempt at a time
        self._ticking = threading.Lock()
        self._kept_line: str | None = None  # typed after a withdrawn question
        self._ended = threading.Event()
        self._stops_ended = False  # the stop thread was told to end
        self._started = False
        self._threads = [
            threading.Thread(target=self._write_loop, name="console-out", daemon=True),
            threading.Thread(target=self._stop_loop, name="console-stop", daemon=True),
        ]
        if tick_s is not None:
            self._threads.append(
                threading.Thread(
                    target=self._tick_loop, name="console-tick", daemon=True
                )
            )

    # Life cycle

    def run(self) -> int:
        """Serve the typed lines until quit or the end of the input; returns 0.

        Leaving sends nothing: the robot app runs on in the state it is in.
        """
        self._start()
        self._say(BANNER)
        try:
            self._attach(self._clock.monotonic())
            while (line := self._next_line()) is not None:
                if not self._serve(line):
                    break
        finally:
            self._end_stops()
            self._say(QUIT_NOTE)
            self.close()
        return EXIT_OK

    def close(self) -> None:
        """Detach and end the console's threads; the queued lines are written."""
        self._end_stops()
        if self._ended.is_set():
            return
        self._ended.set()
        self._detach()
        self._out.put(None)
        if self._started:
            for thread in self._threads:
                if thread is not threading.current_thread():
                    thread.join(_JOIN_S)

    def _end_stops(self) -> None:
        """End the stop thread once every Ctrl+C so far has become its STOP."""
        with self._lock:
            told, self._stops_ended = self._stops_ended, True
        if not told:
            self._interrupts.put(None)
            if self._started:
                self._threads[1].join(_JOIN_S)

    def interrupt(self) -> None:
        """What Ctrl+C does: take back an open question, then STOP.

        Safe in a signal handler: it takes no lock. It never ends the console.
        """
        self._interrupts.put(STOP_CTRL_C)

    def _start(self) -> None:
        if not self._started:
            self._started = True
            for thread in self._threads:
                thread.start()

    # What arrives

    def on_message(self, message: Message) -> None:
        """One link message of the robot app; runs on the client's receive thread."""
        if isinstance(message, RobotState):
            with self._lock:
                self._state = message
                self._state_at = self._clock.monotonic()
            self._check_stop()
        elif isinstance(message, Event):
            self._on_event(message)
        elif isinstance(message, Result):
            self._on_result(message)
        elif isinstance(message, Welcome):
            with self._lock:
                self._welcome = message
            self._say(welcome_line(message))

    def _on_event(self, event: Event) -> None:
        kind = printable(event.kind, MAX_WORD_CHARS)
        line = f"[{kind}] {printable(event.text, MAX_OUT_CHARS)}"
        self._say_repeats()
        if event.kind == KIND_REFUSED and not self._refusals.allow(line):
            return  # counted; said once its second is over
        self._say(BELL_CHAR + line if event.kind == KIND_BELL else line)
        if event.kind == KIND_HELP_NEEDED:
            with self._lock:
                state = self._state
            for text in help_block(event, state):
                self._say(text)

    def _on_result(self, result: Result) -> None:
        # The lock: a RESULT can arrive before its command id is written down.
        with self._lock:
            command = self._mine.get(result.command_id)
            if command is not None and result.status != ACCEPTED:
                del self._mine[result.command_id]
        text = printable(result_text(result), MAX_OUT_CHARS)
        if command is not None:
            self._say(f"{command}: {text}")
        elif result.status == "refused":
            line = f"(another client) {text}"
            self._say_repeats()
            if self._refusals.allow(line):
                self._say(line)

    def _say_repeats(self) -> None:
        for line, count in self._refusals.ended():
            if line is None:
                self._say(f"{count} more refusals of other kinds")
            else:
                self._say(f"{count} more like: {line}")

    # Time

    def tick(self) -> None:
        """Look at the clock, the carrier and what waits; the ticker calls it."""
        with self._ticking:
            now = self._clock.monotonic()
            self._watch_carrier(now)
            with self._lock:
                due = (
                    self._station is None
                    and self._tries_left > 0
                    and now >= self._next_try
                )
                if due:
                    self._tries_left -= 1
            if due:
                self._attach(now, again=True)
            self._check_stop()
            self._say_repeats()
            self._say_status(now)

    def _tick_loop(self) -> None:
        while not self._ended.wait(self._tick_s):
            try:
                self.tick()
            except Exception:
                _log.exception("the operator console's tick failed")

    def _watch_carrier(self, now: float) -> None:
        """Say what changed on the carrier: its death, the robot, the notices."""
        with self._lock:
            station = self._station
        if station is None:
            return
        carrier = station.carrier
        if not carrier.alive:
            why = printable(carrier.close_reason, MAX_OUT_CHARS)
            self._check_stop()  # a STOP that waits is said not confirmed
            self._detach()
            self._forget_commands("the robot link service is gone")
            self._say(
                f"{BELL_CHAR}ROBOT LINK SERVICE GONE: {why}; trying to attach "
                f"again. {NOT_ATTACHED_HELP}"
            )
            with self._lock:
                self._tries_left = ATTACH_TRIES
                self._next_try = now + ATTACH_EVERY_S
            return
        status = carrier.status
        up = None if status is None else status.robot_connected
        with self._lock:
            was, self._robot_up = self._robot_up, up
        if was and up is False:
            why = printable(status.close_reason or "not connected", MAX_OUT_CHARS)
            self._say(
                f"{BELL_CHAR}ROBOT CONNECTION LOST: {why}; the robot holds on its own"
            )
            self._forget_commands("the robot connection was lost")
        elif up and was is False:
            self._say("robot connection up again")
        for notice in self._new_notices(carrier):
            text = printable(notice.text, MAX_OUT_CHARS)
            with self._lock:
                undelivered = self._pending is not None and notice.text.startswith(
                    UNDELIVERED
                )
                if undelivered:
                    self._pending = None
            if undelivered:
                self._say(f"{BELL_CHAR}stop: {text}")
            else:
                self._say(f"robot link service: {text}")

    def _new_notices(self, carrier: LocalCarrier) -> list[LocalNotice]:
        """The carrier's notices this console has not said yet."""
        now = carrier.notices
        with self._lock:
            seen, self._notices = self._notices, now
        # The carrier keeps the newest ones only: find how far its list moved.
        for moved in range(len(seen) + 1):
            if seen[moved:] == now[: len(seen) - moved]:
                return now[len(seen) - moved :]
        return now

    def _forget_commands(self, why: str) -> None:
        with self._lock:
            waiting, self._mine = list(self._mine.values()), {}
        for command in waiting:
            self._say(f"{command}: no RESULT will come: {why}")

    def _say_status(self, now: float) -> None:
        with self._lock:
            station, state, welcome = self._station, self._state, self._welcome
            age = None if state is None else max(0.0, now - self._state_at)
            last_key, last_at = self._status_key, self._status_at
        status = None if station is None else station.carrier.status
        key = status_key(state, status)
        since = None if last_at is None else now - last_at
        changed = key != last_key and (since is None or since >= STATUS_CHANGE_S)
        if not changed and since is not None and since < STATUS_EVERY_S:
            return
        with self._lock:
            self._status_key, self._status_at = key, now
        self._say(status_line(welcome, state, status, age))

    # Attaching

    def _attach(self, now: float, again: bool = False) -> bool:
        """One attempt to attach; whether the console is attached after it.

        again marks one of the paced attempts: its failure is said only when
        the reason is a new one, and after the last of them.
        """
        with self._attaching:
            with self._lock:
                if self._station is not None:
                    return True
                self._next_try = now + ATTACH_EVERY_S
            try:
                carrier = self._attach_fn(
                    self._path, ROLE_CONSOLE, self._name, ATTACH_TIMEOUT_S
                )
            except OSError as problem:  # also LocalRefused and TimeoutError
                why = printable(problem, MAX_OUT_CHARS) or type(problem).__name__
                with self._lock:
                    told, self._why = self._why == why, why
                    left = self._tries_left
                if not (again and told):
                    self._say(
                        f"cannot attach to the robot link service: {why}. "
                        f"{NOT_ATTACHED_HELP}"
                    )
                if left == 0:
                    self._say("no further attempt by itself; type attach to try again")
                elif not told:
                    self._say(
                        f"trying again every {ATTACH_EVERY_S:g} s, {left} times at most"
                    )
                return False
            client = WorkstationClient(
                carrier, self._clock, name=self._name, listeners=[self.on_message]
            )
            with self._lock:
                self._station = _Station(carrier, client)
                self._robot_up, self._notices, self._why = None, [], ""
                self._tries_left = ATTACH_TRIES
            self._say(f"attached to the robot link service: {status_text(carrier)}")
            return True

    def _detach(self) -> None:
        with self._lock:
            station, self._station = self._station, None
            self._pending = None
            # What was shown belongs to that attachment: nothing quotes it later.
            self._state = self._welcome = None
        if station is not None:
            station.client.close()
            station.carrier.close()

    # STOP

    def _stop_loop(self) -> None:
        while (reason := self._interrupts.get()) is not None:
            try:
                with self._lock:
                    withdrawn, self._question = self._question, 0
                if withdrawn:
                    self._say("question taken back (Ctrl+C); nothing sent for it")
                self._stop(reason)
            except Exception:
                _log.exception("a Ctrl+C at the operator console failed")
                self._say(f"{BELL_CHAR}STOP FAILED HERE: {USE_THOR}")

    def _stop(self, reason: str) -> None:
        """One STOP: through the client when attached, else written raw."""
        with self._lock:
            station = self._station
            attached = station is not None and station.carrier.alive
            if attached:
                before = self._state if self._pending is None else self._pending.before
                by = self._clock.monotonic() + STOP_WAIT_S
                self._pending = _Pending(before, by, station.carrier)
        if attached:
            station.client.stop(reason, SENDER)
            self._say("stop: sent; waiting for the robot app to show it")
            return
        raw = self._raw_stop(self._path, reason=reason, sender=SENDER, name=self._name)
        text = raw_stop_text(raw.written, printable(raw.notice, MAX_OUT_CHARS))
        if not raw.written:
            text = f"{text}; {USE_THOR}"
        self._say(f"{BELL_CHAR}stop: {text}")

    def _check_stop(self) -> None:
        """Say what became of the STOP that waits, once that is known."""
        with self._lock:
            pending, state = self._pending, self._state
            if pending is None:
                return
            taken = (
                state is not None
                and state is not pending.before
                and stop_shows(pending.before, state)
            )
            gone = not pending.carrier.alive
            late = self._clock.monotonic() >= pending.by
            if not (taken or gone or late):
                return
            self._pending = None
        if taken:
            self._say(f"stop: taken. {state_text(state)}")
        elif gone:
            self._say(
                f"{BELL_CHAR}STOP NOT CONFIRMED: the robot link service is gone; "
                f"type stop again or {USE_THOR}"
            )
        else:
            self._say(
                f"{BELL_CHAR}STOP NOT CONFIRMED: no ROBOT_STATE showed a higher "
                f"stop number within {STOP_WAIT_S:g} s; {USE_THOR}"
            )

    # Typed lines

    def _next_line(self) -> str | None:
        """The next typed line; None once the input has ended."""
        kept, self._kept_line = self._kept_line, None
        if kept is not None:
            return kept
        try:
            return next(self._lines)
        except StopIteration:
            return None
        except (OSError, ValueError) as error:  # the terminal is gone
            _log.warning("the operator console cannot be read: %r", error)
            return None

    def _serve(self, line: str) -> bool:
        """Serve one typed line; False when it asks to leave the console."""
        word = " ".join(line.lower().split())
        if not word:
            return True
        if word == "quit":
            return False
        if word == "stop":
            self._stop(STOP_REASON)
        elif word == "help":
            for text in HELP:
                self._say(text)
        elif word == "state":
            self._show_state()
        elif word == "attach":
            with self._lock:
                attached = self._station is not None
            if attached:
                self._say("attach: attached already")
            else:
                self._attach(self._clock.monotonic())
        elif word == "reset" or word in CONFIRMED:
            self._command(word)
        else:
            shown = printable(line.strip(), _SHOWN)
            self._say(f"unknown command {shown!r}; type help")
        return True

    def _show_state(self) -> None:
        with self._lock:
            station, state = self._station, self._state
        if station is None:
            self._say(f"robot link service: {NOT_ATTACHED_HELP}")
        else:
            self._say(f"robot link service: {status_text(station.carrier)}")
        self._say(state_text(state) if state else "ROBOT_STATE: none has arrived")
        now = self._clock.monotonic()
        with self._lock:
            self._status_at = None  # the status line is due now
        self._say_status(now)

    def _command(self, command: str) -> None:
        """reset, rest, open, release or shutdown: ask, then console_cmd sends it."""
        with self._lock:
            station, state = self._station, self._state
        if station is None or not station.carrier.alive:
            self._say(f"{command}: {NOT_ATTACHED_HELP}; nothing sent")
            return
        status = station.carrier.status
        up = status is not None and status.robot_connected
        if up and state is None:
            state = self._first_state(station)
        if state is None or not up:
            self._say(
                f"{command}: no ROBOT_STATE to quote "
                f"({status_text(station.carrier)}); nothing sent"
            )
            return
        asked = question(command, state)
        typed = ""
        if asked is not None:
            answer = self._ask(asked[0], state)
            if answer is None:
                return
            if not confirms(answer, asked[1]):
                self._say(f"{command} {NOT_CONFIRMED}")
                return
            typed = answer
        with self._lock:
            if self._station is not station:
                self._say(
                    f"{command}: the console is no longer attached as it was when "
                    "it asked; nothing sent"
                )
                return
            try:
                # `state` is the one shown with the question: the robot app
                # refuses the command if the arm's owner or stop number moved on.
                command_id = send_command(
                    command, station.client, state, phrase_typed=typed
                )
            except NotConfirmed as problem:
                self._say(str(problem))
                return
            self._mine[command_id] = command
        self._say(f"{command}: sent; its RESULT is printed when it comes")

    def _first_state(self, station: _Station) -> RobotState | None:
        """Right after attaching: wait, in real time, for the first ROBOT_STATE."""
        for _ in range(int(STATE_WAIT_S / TICK_S)):
            with self._lock:
                state = self._state
            if state is not None or not station.carrier.alive:
                return state
            if self._ended.wait(TICK_S):
                break
        return None

    def _ask(self, prompt: str, state: RobotState) -> str | None:
        """Put a question; the next typed line, or None when it confirms nothing.

        None after a Ctrl+C, at the end of the input and for a typed stop,
        which is a STOP wherever it is typed. What was typed before the
        question was written is dropped.
        """
        with self._lock:
            self._asked += 1
            self._question = number = self._asked
        self._say(state_text(state))
        self._say(prompt.rstrip())
        self._flush()
        if self._flush_input is not None:
            self._flush_input()
        line = self._next_line()
        with self._lock:
            withdrawn = self._question != number
            self._question = 0
        if withdrawn:
            self._kept_line = line  # typed after the Ctrl+C: a new command
            return None
        if line is None:
            self._say(f"{NOT_CONFIRMED} (the input ended)")
            return None
        if confirms(line, "stop"):
            self._stop(STOP_REASON)
            self._say(NOT_CONFIRMED)
            return None
        return line

    # Output

    def _say(self, line: str) -> None:
        if not self._ended.is_set() and self._out.qsize() < _LINES_WAITING:
            self._out.put(line)

    def _flush(self, timeout_s: float = _JOIN_S) -> bool:
        """Wait in real time until the lines queued so far are written."""
        written = threading.Event()
        self._out.put(written)
        return written.wait(timeout_s)

    def _write_loop(self) -> None:
        failed = False
        while (item := self._out.get()) is not None:
            if isinstance(item, threading.Event):
                item.set()
                continue
            try:
                self._write(item)
            except Exception as error:
                if not failed:
                    _log.warning("the operator console cannot be written to: %r", error)
                failed = True
            else:
                failed = False


def _print_line(text: str) -> None:
    print(text, flush=True)


def _typed_lines() -> Iterator[str]:
    """The lines typed at standard input, until it ends."""
    while line := sys.stdin.readline():
        yield line


def _drop_typed_ahead() -> None:
    with contextlib.suppress(OSError, ValueError, termios.error):
        termios.tcflush(sys.stdin.fileno(), termios.TCIFLUSH)


def _unusable() -> str | None:
    """Why this process cannot be an operator's console; None when it can."""
    try:
        interactive = sys.stdin.isatty() and sys.stdout.isatty()
    except (AttributeError, ValueError):
        interactive = False
    if not interactive:
        return "it needs a terminal on standard input and output"
    marker = agent_environment()
    if marker is not None:
        return (
            "it is typed at by the operator, outside an agent environment "
            f"({marker} is set)"
        )
    return None


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m isaaccapture_examples.agentic_remote.console",
        description=(
            "The operator console: shows the robot connection, the robot's state "
            "and its events, and sends the typed commands stop, reset, rest, "
            "open, release and shutdown through the robot link service."
        ),
    )
    parser.add_argument(
        "--socket",
        default=None,
        metavar="PATH",
        help="the robot link service's local socket (default: its own default)",
    )
    parser.add_argument(
        "--name",
        default=NAME,
        metavar="NAME",
        help=f"this console's name at the robot link service (default: {NAME})",
    )
    parser.add_argument(
        "--log-level",
        default="warning",
        choices=LOG_LEVELS,
        type=str.lower,
        metavar="L",
        help=f"console log level, one of {', '.join(LOG_LEVELS)} (default: warning)",
    )
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    attach: Attach = connect_local,
    raw_stop: Callable[..., RawStop] = send_raw_stop,
    lines: Iterator[str] | None = None,
    say: Say | None = None,
    clock: Clock | None = None,
) -> int:
    """Run the operator console; returns the exit code.

    lines None reads the terminal: then standard input and output must be one,
    outside an agent's environment, and Ctrl+C is a STOP.
    """
    args = _parser().parse_args(argv)
    at_terminal = lines is None
    flush_input = None
    if at_terminal:
        why = _unusable()
        if why is not None:
            (say or _print_line)(f"{PROGRAM}: {why}; nothing sent")
            return EXIT_UNUSABLE
        lines, flush_input = _typed_lines(), _drop_typed_ahead
    set_console_level(args.log_level)
    console = Console(
        lines,
        say,
        socket_path=args.socket,
        name=args.name,
        attach=attach,
        raw_stop=raw_stop,
        clock=clock,
        flush_input=flush_input,
    )
    replaced = None
    if at_terminal:
        replaced = signal.signal(
            signal.SIGINT, lambda number, stack: console.interrupt()
        )
    try:
        return console.run()
    finally:
        if at_terminal:
            signal.signal(signal.SIGINT, replaced or signal.SIG_DFL)
