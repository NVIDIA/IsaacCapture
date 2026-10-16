# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TerminalKeySource against a pseudo-terminal: it restores the terminal on every exit path and
parses key sequences however the reads split them."""

import contextlib
import os
import pty
import select
import termios
import threading
import tty

import pytest

import keyboard_terminal_example
from keyboard_terminal_example import TerminalKeySource

KITTY_REPLY = (
    b"\x1b[?1u\x1b[?62c"  # kitty protocol flags, then the primary device attributes
)


@pytest.fixture
def terminal():
    """A pseudo-terminal: (master, slave) with the slave's original attributes."""
    master, slave = pty.openpty()
    before = termios.tcgetattr(slave)
    yield master, slave, before
    for fd in (master, slave):
        with contextlib.suppress(OSError):
            os.close(fd)


@pytest.fixture
def output_pipe():
    """A pipe standing in for stdout: (read end, write end)."""
    read_end, write_end = os.pipe()
    yield read_end, write_end
    for fd in (read_end, write_end):
        with contextlib.suppress(OSError):
            os.close(fd)


@contextlib.contextmanager
def _answering_probe(master):
    """Answer the kitty probe like a kitty-protocol terminal, once the query arrives.

    Answering ahead of time would race the cbreak switch, which flushes pending input.
    """

    def answer():
        seen = b""
        while b"\x1b[c" not in seen:
            if not select.select([master], [], [], 5.0)[0]:
                return
            seen += os.read(master, 1024)
        os.write(master, KITTY_REPLY)

    thread = threading.Thread(target=answer, daemon=True)
    thread.start()
    try:
        yield
    finally:
        thread.join(timeout=5.0)


def _fail(message):
    def fail(*args):
        raise RuntimeError(message)

    return fail


def _recording(source):
    events = []
    source.add_key_listener(
        lambda code, pressed: events.append((code, pressed)),
        lambda: events.append("focus lost"),
    )
    return events


def test_normal_exit_restores_the_terminal(terminal):
    master, slave, before = terminal
    with _answering_probe(master), TerminalKeySource(fd=slave, out_fd=slave) as source:
        assert source.kitty
        assert termios.tcgetattr(slave) != before
    assert termios.tcgetattr(slave) == before


def test_failing_listener_still_restores_the_terminal(terminal):
    _, slave, before = terminal
    source = TerminalKeySource(fd=slave, out_fd=slave)
    source.add_key_listener(lambda code, pressed: None, _fail("listener"))

    with pytest.raises(RuntimeError, match="listener"):
        with source:
            pass
    assert termios.tcgetattr(slave) == before


def test_closed_output_still_restores_the_terminal(terminal, output_pipe):
    _, slave, before = terminal
    read_end, write_end = output_pipe
    with TerminalKeySource(fd=slave, out_fd=write_end):
        os.close(read_end)  # the protocol resets on exit now fail to write
    assert termios.tcgetattr(slave) == before


def test_body_exception_survives_a_failing_reset(terminal, output_pipe):
    _, slave, before = terminal
    read_end, write_end = output_pipe
    with pytest.raises(ValueError, match="body"):
        with TerminalKeySource(fd=slave, out_fd=write_end):
            os.close(read_end)
            raise ValueError("body")
    assert termios.tcgetattr(slave) == before


def test_failed_setup_restores_the_terminal(terminal, output_pipe):
    _, slave, before = terminal
    read_end, write_end = output_pipe
    os.close(read_end)  # the first query cannot be written

    with pytest.raises(BrokenPipeError):
        TerminalKeySource(fd=slave, out_fd=write_end).__enter__()
    assert termios.tcgetattr(slave) == before


@pytest.fixture
def legacy(terminal, monkeypatch):
    """A TerminalKeySource on a cbreak pty, as on a terminal without the kitty protocol.

    Yields (source, master, events, clock); ``clock`` drives the lone-ESC timeout.
    """
    master, slave, _ = terminal
    clock = [0.0]
    monkeypatch.setattr(keyboard_terminal_example.time, "monotonic", lambda: clock[0])
    tty.setcbreak(slave)
    source = TerminalKeySource(fd=slave, out_fd=slave)
    yield source, master, _recording(source), clock


@pytest.fixture
def kitty(terminal, monkeypatch):
    """The same over a pty that answers the probe as a kitty-protocol terminal."""
    master, slave, _ = terminal
    clock = [0.0]
    with _answering_probe(master), TerminalKeySource(fd=slave, out_fd=slave) as source:
        assert source.kitty
        monkeypatch.setattr(
            keyboard_terminal_example.time, "monotonic", lambda: clock[0]
        )
        yield source, master, _recording(source), clock


def _feed(source, master, *parts):
    """Deliver ``parts`` one read at a time."""
    for part in parts:
        os.write(master, part)
        source.poll()


def test_unrecognized_reply_does_not_hold_back_keys(legacy):
    """A complete CSI sequence that is not a key (a late query reply) is skipped, not waited on."""
    source, master, events, _ = legacy
    _feed(source, master, b"\x1b[?62;22c" + b"\x1b[119;1:1u", b"k")

    # Without kitty mode each arrives as a tap.
    assert events == [("KeyW", True), ("KeyW", False), ("KeyK", True), ("KeyK", False)]


@pytest.mark.parametrize(
    "sequence",
    [b"\x1b[119;1:1u", b"\x1b[119;1:3u", b"\x1b[A", b"\x1b[O", b"\x1b[I"],
    ids=["press", "release", "legacy-arrow", "focus-out", "focus-in"],
)
def test_sequences_split_across_reads_parse_like_whole_ones(kitty, sequence):
    """Reads need not align with escape sequences: every split yields the unsplit events."""
    source, master, events, _ = kitty
    _feed(source, master, sequence)
    expected = list(events)

    for cut in range(1, len(sequence)):
        events.clear()
        _feed(source, master, sequence[:cut], sequence[cut:])
        assert events == expected, f"split at byte {cut}"


def test_kitty_release_split_by_an_empty_poll_still_releases(kitty):
    """With the kitty protocol a lone ESC is always a prefix, however long the rest takes."""
    source, master, events, clock = kitty
    _feed(source, master, b"\x1b[119;1:1u", b"\x1b")
    clock[0] += 1.0
    source.poll()  # nothing new arrived
    _feed(source, master, b"[119;1:3u")

    assert events == [("KeyW", True), ("KeyW", False)]


def test_legacy_sequence_split_by_an_empty_poll_is_not_escape(legacy):
    """On a legacy terminal the rest of a sequence arriving within the timeout still completes it."""
    source, master, events, clock = legacy
    _feed(source, master, b"\x1b")
    clock[0] += 0.05
    source.poll()  # nothing new arrived, but within the timeout
    _feed(source, master, b"[A")

    assert ("Escape", True) not in events
    assert events[:1] == [("ArrowUp", True)]


def test_lone_escape_is_the_escape_key(legacy):
    """On a legacy terminal a lone ESC that nothing follows for the timeout is the Escape key."""
    source, master, events, clock = legacy
    _feed(source, master, b"\x1b")
    assert events == []  # may be the start of a sequence split across reads

    clock[0] += 0.2
    source.poll()
    assert events == [("Escape", True), ("Escape", False)]


def test_kitty_ctrl_c_interrupts(kitty):
    """The kitty protocol reports Ctrl+C as a key rather than a signal."""
    source, master, _, _ = kitty
    with pytest.raises(KeyboardInterrupt):
        _feed(source, master, b"\x1b[99;5u")
