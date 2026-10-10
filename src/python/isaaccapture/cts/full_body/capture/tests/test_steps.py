# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The step loop, driven with a fake clock and fake record numbers."""

from __future__ import annotations

import pytest

from steps import KEYBOARD, TRIGGER, Phase, Press, Sound, Take

# label, hold seconds, spoken cue.
SCRIPT = [
    ("a_pose_still", 4.0, "Arms down. Stand still."),
    ("left_arm_raise", 3.0, "Left arm up, overhead."),
]
CUE_S = {"a_pose_still": 1.0, "left_arm_raise": 2.0}
END_TONE_S = 0.5


def take() -> Take:
    return Take(cue_seconds=CUE_S, end_tone_s=END_TONE_S, steps=SCRIPT)


def test_nothing_happens_before_the_first_frame():
    loop = take()
    assert loop.phase is Phase.IDLE
    assert loop.advance(0.0, 0) is None
    assert loop.press(0.0, 0, TRIGGER) is Press.IGNORED
    assert loop.windows == ()


def test_the_first_cue_waits_for_start():
    loop = take()
    assert loop.start(10.0) is Sound.CUE
    assert loop.phase is Phase.CUEING
    # Called twice -- a second valid frame must not re-speak the cue.
    assert loop.start(10.5) is None


def test_a_press_during_the_cue_only_cuts_it_short():
    loop = take()
    loop.start(0.0)
    assert loop.press(0.4, 7, TRIGGER) is Press.CUT_CUE
    assert loop.phase is Phase.WAITING
    assert loop.windows == (), "cutting the cue must not open a window"


def test_the_cue_gives_way_to_waiting_on_its_own_length():
    loop = take()
    loop.start(0.0)
    assert loop.advance(0.9, 3) is None
    assert loop.phase is Phase.CUEING
    assert loop.advance(1.0, 3) is None
    assert loop.phase is Phase.WAITING


def test_an_accepted_press_opens_the_window_on_its_own_frame():
    loop = take()
    loop.start(0.0)
    loop.advance(1.0, 3)
    assert loop.press(2.0, 40, TRIGGER) is Press.ACCEPTED
    assert loop.phase is Phase.HOLDING
    assert loop.remaining_s(2.0) == pytest.approx(4.0)
    loop.advance(6.0, 260)
    assert loop.windows[0].start_frame == 40
    assert loop.windows[0].end_frame == 260


@pytest.mark.parametrize("source", [TRIGGER, KEYBOARD])
def test_every_press_after_the_accepted_one_is_swallowed(source):
    """The hard requirement: no tone, no window, and the caller is told nothing."""
    loop = take()
    loop.start(0.0)
    loop.advance(1.0, 3)
    loop.press(2.0, 40, source)

    for when, frame in ((2.1, 45), (3.0, 100), (5.9, 250)):
        assert loop.press(when, frame, source) is Press.IGNORED
    assert loop.advance(6.0, 260) is Sound.END

    # Still swallowed while the end tone sounds, and the window already closed at 260.
    assert loop.phase is Phase.CLOSING
    assert loop.press(6.1, 265, source) is Press.IGNORED
    assert [(w.start_frame, w.end_frame) for w in loop.windows] == [(40, 260)]


def test_a_press_is_accepted_again_once_the_next_cue_has_been_spoken():
    loop = take()
    loop.start(0.0)
    loop.advance(1.0, 3)
    loop.press(2.0, 40, TRIGGER)
    loop.advance(6.0, 260)
    assert loop.advance(6.5, 290) is Sound.CUE
    assert (loop.phase, loop.index, loop.label) == (
        Phase.CUEING,
        1,
        "left_arm_raise",
    )
    loop.advance(8.5, 400)
    assert loop.phase is Phase.WAITING
    assert loop.press(9.0, 420, KEYBOARD) is Press.ACCEPTED


def test_the_source_of_each_boundary_is_recorded():
    """A trigger is in the recording's controllers channel; a key press is nowhere."""
    loop = take()
    loop.start(0.0)
    loop.advance(1.0, 3)
    loop.press(2.0, 40, TRIGGER)
    loop.advance(6.0, 260)
    loop.advance(6.5, 290)
    loop.advance(8.5, 400)
    loop.press(9.0, 420, KEYBOARD)
    loop.advance(12.0, 600)
    assert [window.source for window in loop.windows] == [TRIGGER, KEYBOARD]


def test_the_take_ends_on_the_last_window_not_on_a_tone():
    loop = take()
    loop.start(0.0)
    loop.advance(1.0, 3)
    loop.press(2.0, 40, TRIGGER)
    loop.advance(6.0, 260)
    loop.advance(6.5, 290)
    loop.advance(8.5, 400)
    loop.press(9.0, 420, TRIGGER)
    assert loop.advance(12.0, 600) is Sound.END
    assert loop.phase is Phase.DONE
    assert loop.advance(99.0, 999) is None
    assert len(loop.windows) == len(SCRIPT)


def test_still_windows_are_marked_from_the_script():
    loop = take()
    loop.start(0.0)
    loop.advance(1.0, 3)
    loop.press(2.0, 40, TRIGGER)
    loop.advance(6.0, 260)
    assert loop.windows[0].is_still_window


def test_a_step_with_no_spoken_length_is_refused():
    with pytest.raises(ValueError, match="left_arm_raise"):
        Take(cue_seconds={"a_pose_still": 1.0}, end_tone_s=END_TONE_S, steps=SCRIPT)


def test_the_hold_countdown_is_this_pose_not_the_next():
    loop = take()
    loop.start(0.0)
    loop.advance(1.0, 3)
    loop.press(2.0, 40, TRIGGER)
    assert loop.remaining_s(4.0) == pytest.approx(2.0)
    assert loop.elapsed_fraction(4.0) == pytest.approx(0.5)
    loop.advance(6.0, 260)
    assert loop.remaining_s(6.0) is None
