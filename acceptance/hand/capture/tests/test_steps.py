# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from hand_acceptance.labels import SCRIPT
from session import BEAT_S, COUNTDOWN_BEATS, STEPS
from steps import Phase, Sound, Take

MS = 1_000_000
CUE_S = 1.5
END_S = 0.1


def _take() -> Take:
    return Take(cue_seconds={label: CUE_S for label, _, _ in STEPS}, end_tone_s=END_S)


def _run(take: Take, tick_ms: int = 5) -> list[tuple[int, Sound]]:
    now = 1_000 * MS
    heard = [(now, take.start(now))]
    while take.phase is not Phase.DONE:
        now += tick_ms * MS
        sound = take.advance(now)
        if sound is not None:
            heard.append((now, sound))
        assert now < 10**12, "the take never finished"
    return heard


def test_full_take_opens_one_window_per_step_in_script_order():
    take = _take()
    _run(take)
    assert [w.label for w in take.windows] == list(SCRIPT)
    assert [w.index for w in take.windows] == list(range(len(SCRIPT)))
    assert all(w.source == "timer" for w in take.windows)
    assert [w.is_still_window for w in take.windows] == [
        label != "wrist_rotate" for label in SCRIPT
    ]


def test_windows_last_their_hold_and_never_overlap():
    take = _take()
    _run(take, tick_ms=1)
    holds = {label: hold for label, hold, _ in STEPS}
    for window in take.windows:
        assert window.end_ns - window.start_ns == pytest.approx(
            holds[window.label] * 1e9, abs=2 * MS
        )
    for earlier, later in zip(take.windows, take.windows[1:]):
        assert later.start_ns >= earlier.end_ns + int(
            (END_S + CUE_S + COUNTDOWN_BEATS * BEAT_S) * 1e9
        )


def test_every_step_is_cue_then_countdown_then_start_then_end():
    take = _take()
    heard = [sound for _, sound in _run(take)]
    one = [Sound.CUE, *[Sound.COUNT] * COUNTDOWN_BEATS, Sound.START, Sound.END]
    assert heard == one * len(STEPS)


def test_window_opens_only_after_the_last_beat():
    take = _take()
    heard = _run(take, tick_ms=1)
    last_count = max(t for t, s in heard[: COUNTDOWN_BEATS + 2] if s is Sound.COUNT)
    assert take.windows[0].start_ns - last_count == pytest.approx(BEAT_S * 1e9, abs=MS)


def test_nothing_happens_before_start_or_after_done():
    take = _take()
    assert take.advance(10**15) is None and take.phase is Phase.IDLE
    _run(take)
    assert take.advance(10**15) is None and take.start(10**15) is None


def test_missing_cue_length_is_refused():
    with pytest.raises(ValueError, match="no spoken length"):
        Take(cue_seconds={}, end_tone_s=END_S)


def test_session_steps_are_the_checker_script():
    assert tuple(label for label, _, _ in STEPS) == SCRIPT
