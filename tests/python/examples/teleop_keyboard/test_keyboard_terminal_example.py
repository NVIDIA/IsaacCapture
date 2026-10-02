# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TerminalKeySource restores the terminal on every exit path (against a pseudo-terminal)."""

import os
import pty
import termios
import tty

import pytest

from keyboard_terminal_example import TerminalKeySource


@pytest.fixture
def terminal():
    """A TerminalKeySource over a pty whose probe finds a kitty-capable terminal."""
    master, slave = pty.openpty()
    source = TerminalKeySource(fd=slave)
    source._write = lambda data: os.write(slave, data)  # keep escapes off the real tty
    source._read_until = lambda terminator, timeout_s: b"\x1b[?1u\x1b[?62c"
    before = termios.tcgetattr(slave)
    yield source, before, slave
    os.close(master)
    os.close(slave)


def _fail(message, exc=RuntimeError):
    def fail(*args, **kwargs):
        raise exc(message)

    return fail


def test_normal_exit_restores_the_terminal(terminal):
    source, before, fd = terminal
    with source:
        assert source.kitty
        assert termios.tcgetattr(fd) != before
    assert termios.tcgetattr(fd) == before


def test_failing_listener_still_restores_the_terminal(terminal):
    source, before, fd = terminal
    source.add_key_listener(lambda code, pressed: None, _fail("listener"))

    with pytest.raises(RuntimeError, match="listener"):
        with source:
            pass
    assert termios.tcgetattr(fd) == before


def test_closed_output_still_restores_the_terminal(terminal):
    source, before, fd = terminal
    with source:
        source._write = _fail("stdout closed", BrokenPipeError)
    assert termios.tcgetattr(fd) == before


def test_body_exception_survives_a_failing_reset(terminal):
    source, before, fd = terminal
    with pytest.raises(ValueError, match="body"):
        with source:
            source._write = _fail("stdout closed", BrokenPipeError)
            raise ValueError("body")
    assert termios.tcgetattr(fd) == before


def test_failed_probe_restores_the_terminal(terminal):
    source, before, fd = terminal
    source._read_until = _fail("probe")

    with pytest.raises(RuntimeError, match="probe"):
        source.__enter__()
    assert termios.tcgetattr(fd) == before


def test_unrecognized_reply_does_not_hold_back_keys():
    """A complete CSI sequence that is not a key (a late query reply) is skipped, not waited on."""
    source = TerminalKeySource(fd=0)
    keys = []
    source.add_key_listener(
        lambda code, pressed: keys.append((code, pressed)), lambda: None
    )

    source._buffer = b"\x1b[?62;22c" + b"\x1b[119;1:1u"
    source._parse()

    assert keys[:1] == [("KeyW", True)]  # without kitty mode it arrives as a tap
    assert source._buffer == b""


def _recording_source(fd=0):
    source = TerminalKeySource(fd=fd)
    events = []
    source.add_key_listener(
        lambda code, pressed: events.append((code, pressed)),
        lambda: events.append("focus lost"),
    )
    return source, events


@pytest.mark.parametrize(
    "sequence",
    [b"\x1b[119;1:1u", b"\x1b[119;1:3u", b"\x1b[A", b"\x1b[O", b"\x1b[I"],
    ids=["press", "release", "legacy-arrow", "focus-out", "focus-in"],
)
def test_sequences_split_across_reads_parse_like_whole_ones(sequence):
    """Reads need not align with escape sequences: every split yields the unsplit events."""
    whole, expected = _recording_source()
    whole._buffer = sequence
    whole._parse()

    for cut in range(1, len(sequence)):
        split, events = _recording_source()
        for part in (sequence[:cut], sequence[cut:]):
            split._buffer += part
            split._parse()
        assert events == expected, f"split at byte {cut}"
        assert split._buffer == b""


@pytest.fixture
def pty_source(monkeypatch):
    """A TerminalKeySource on a cbreak pseudo-terminal, its writer end, its events and a fake clock."""
    import keyboard_terminal_example

    clock = [0.0]
    monkeypatch.setattr(keyboard_terminal_example.time, "monotonic", lambda: clock[0])
    master, slave = pty.openpty()
    tty.setcbreak(slave)
    source, events = _recording_source(fd=slave)
    yield source, master, events, clock
    os.close(master)
    os.close(slave)


def test_kitty_release_split_by_an_empty_poll_still_releases(pty_source):
    """With the kitty protocol a lone ESC is always a prefix, however long the rest takes."""
    source, master, events, clock = pty_source
    source.kitty = True
    os.write(master, b"\x1b[119;1:1u")
    source.poll()
    os.write(master, b"\x1b")
    source.poll()
    clock[0] += 1.0
    source.poll()  # nothing new arrived
    os.write(master, b"[119;1:3u")
    source.poll()

    assert events == [("KeyW", True), ("KeyW", False)]


def test_legacy_sequence_split_by_an_empty_poll_is_not_escape(pty_source):
    """On a legacy terminal the rest of a sequence arriving within the timeout still completes it."""
    source, master, events, clock = pty_source
    os.write(master, b"\x1b")
    source.poll()
    clock[0] += 0.05
    source.poll()  # nothing new arrived, but within the timeout
    os.write(master, b"[A")
    source.poll()

    assert ("Escape", True) not in events
    assert events[:1] == [("ArrowUp", True)]


def test_lone_escape_is_the_escape_key(pty_source):
    """On a legacy terminal a lone ESC that nothing follows for the timeout is the Escape key."""
    source, master, events, clock = pty_source
    os.write(master, b"\x1b")
    source.poll()
    assert events == []  # may be the start of a sequence split across reads

    clock[0] += 0.2
    source.poll()
    assert events == [("Escape", True), ("Escape", False)]
