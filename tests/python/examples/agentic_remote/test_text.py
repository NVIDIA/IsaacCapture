# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for printable() and LineLimit in
isaaccapture_examples.agentic_remote.robot_app.text."""

from __future__ import annotations

import threading

import pytest
from isaaccapture_examples.agentic_remote.robot_app.text import (
    CUT_MARK,
    LINE_KINDS,
    LINE_WINDOW_S,
    MAX_ID_CHARS,
    MAX_OUT_CHARS,
    MAX_REASON_CHARS,
    MAX_WORD_CHARS,
    LineLimit,
    printable,
)

JOIN_S = 5.0  # real seconds a test gives another thread before it fails


def test_the_limits_are_the_ones_the_readme_names():
    assert (MAX_ID_CHARS, MAX_WORD_CHARS, MAX_REASON_CHARS) == (128, 64, 256)
    assert MAX_OUT_CHARS == 1024
    assert (LINE_WINDOW_S, LINE_KINDS) == (1.0, 16)


# printable()


@pytest.mark.parametrize("text", ["", "typed stop", "é ü 日本", "a" * 16])
def test_a_short_printable_text_is_left_as_it_is(text):
    assert printable(text, 16) == text


def test_a_long_text_is_cut_to_the_limit_and_says_so():
    cut = printable("a" * 17, 16)
    assert cut == "a" * 13 + CUT_MARK
    assert len(printable("é" * 5_000_000, MAX_REASON_CHARS)) == MAX_REASON_CHARS


@pytest.mark.parametrize(
    ("text", "shown"),
    [
        ("\x1b[2J\x1b[H", " [2J [H"),
        ("line\r\nbreak\ttab", "line  break tab"),
        ("bell\x07 nul\x00", "bell  nul "),
        ("lone \ud83d surrogate", "lone   surrogate"),
        ("right-to-left " + chr(0x202E) + " mark", "right-to-left   mark"),
    ],
)
def test_every_character_that_cannot_be_printed_becomes_a_space(text, shown):
    assert printable(text, 64) == shown
    assert printable(text, 64).isprintable()


def test_anything_is_shown_through_its_str():
    assert printable(ValueError("no good"), 64) == "no good"
    assert printable(None, 64) == "None"
    assert printable(12.5, 64) == "12.5"


@pytest.mark.parametrize("limit", [0, 1, 3])
def test_a_limit_with_no_room_for_the_cut_mark_still_holds(limit):
    assert printable("abcdef", limit) == "abcdef"[:limit]
    assert printable("abcdef", 4) == "a" + CUT_MARK


# LineLimit


def test_a_line_is_allowed_once_and_its_repeats_are_counted(clock):
    limit = LineLimit(clock)
    assert limit.allow("a") is True
    assert [limit.allow("a") for _ in range(3)] == [False] * 3
    assert limit.allow("b") is True
    # Nothing is told while the second lasts.
    clock.advance(0.75)
    assert limit.allow("a") is False
    assert limit.ended() == []
    clock.advance(0.25)
    assert limit.ended() == [("a", 4)]
    assert limit.ended() == []


def test_a_kind_is_allowed_again_in_the_next_second(clock):
    limit = LineLimit(clock)
    assert limit.allow("a") is True
    assert limit.allow("a") is False
    clock.advance(1.0)
    assert limit.allow("a") is True
    assert limit.ended() == [("a", 1)]
    # A kind that did not repeat is not told.
    clock.advance(1.0)
    assert limit.ended() == []


def test_the_second_begins_with_its_first_line(clock):
    limit = LineLimit(clock)
    clock.advance(10.0)
    assert limit.allow("a") is True
    clock.advance(0.9)
    assert limit.allow("a") is False
    clock.advance(0.9)  # 1.8 s after the first line: a new second
    assert limit.allow("a") is True


def test_kinds_beyond_the_bound_are_counted_together(clock):
    limit = LineLimit(clock, kinds=3)
    assert [limit.allow(kind) for kind in "abc"] == [True] * 3
    assert [limit.allow(kind) for kind in "defd"] == [False] * 4
    assert limit.allow("a") is False
    clock.advance(1.0)
    assert limit.ended() == [("a", 1), (None, 4)]
    # The next second tells kinds apart afresh.
    assert [limit.allow(kind) for kind in "def"] == [True] * 3


def test_seconds_nobody_asked_about_add_up_and_stay_bounded(clock):
    limit = LineLimit(clock, kinds=2)
    for second in range(50):
        for _ in range(3):
            limit.allow(f"kind-{second}")
        clock.advance(1.0)
    told = dict(limit.ended())
    assert sum(told.values()) == 100
    assert len(told) <= 4 and None in told


def test_a_window_of_another_length(clock):
    limit = LineLimit(clock, window_s=5.0)
    assert limit.allow("a") is True
    clock.advance(4.75)
    assert limit.allow("a") is False
    clock.advance(0.25)
    assert limit.allow("a") is True


def test_lines_from_several_threads_are_each_allowed_or_counted(clock):
    limit = LineLimit(clock)
    allowed: list[bool] = []
    start = threading.Barrier(4)

    def work() -> None:
        start.wait(JOIN_S)
        allowed.extend(limit.allow("a") for _ in range(500))

    threads = [threading.Thread(target=work) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(JOIN_S)
    assert allowed.count(True) == 1
    clock.advance(1.0)
    assert limit.ended() == [("a", 1999)]
