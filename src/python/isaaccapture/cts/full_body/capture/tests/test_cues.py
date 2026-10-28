# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Spoken cues without ``aplay``: nothing is played and the take keeps its pace."""

from __future__ import annotations

import subprocess

import pytest

import cues
import render
from session import STEPS
from steps import KEYBOARD, Phase, Press, Sound, Take

TICK_S = 0.01
TONES = {Sound.START: cues.START_TONE, Sound.END: cues.END_TONE}


@pytest.fixture
def spawned(monkeypatch) -> list:
    """Every command ``subprocess.Popen`` is asked to run during the test."""
    calls: list = []
    real = subprocess.Popen

    def spy(command, *args, **kwargs):
        calls.append(command)
        return real(command, *args, **kwargs)

    monkeypatch.setattr(subprocess, "Popen", spy)
    return calls


@pytest.fixture
def no_aplay(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("PATH", str(tmp_path))


def test_without_aplay_on_path_cues_are_off(no_aplay):
    assert cues.available() is False


def test_without_aplay_play_spawns_nothing(no_aplay, spawned):
    assert cues.play(cues.START_TONE) is None
    assert spawned == []


def test_with_aplay_on_path_play_spawns_it(monkeypatch, tmp_path, spawned):
    aplay = tmp_path / "aplay"
    aplay.write_text('#!/bin/sh\necho "$@" > "${0%/*}/played"\n')
    aplay.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path))
    assert cues.available() is True
    process = cues.play(cues.START_TONE)
    assert process is not None and process.wait(timeout=10) == 0
    assert spawned == [[str(aplay), "-q", str(cues.START_TONE)]]
    assert (tmp_path / "played").read_text().split() == ["-q", str(cues.START_TONE)]


def test_a_take_paced_by_the_wavs_runs_to_the_end_without_aplay(no_aplay, spawned):
    spoken = {label: cues.wav(label, text) for label, _, text in STEPS}
    take = Take(
        cue_seconds={label: cues.duration_of(path) for label, path in spoken.items()},
        end_tone_s=cues.duration_of(cues.END_TONE),
    )
    now, frame = 0.0, 0
    heard = [(take.label, take.start(now))]
    while take.phase is not Phase.DONE:
        now, frame = now + TICK_S, frame + 1
        heard.append((take.label, take.advance(now, frame)))
        if take.phase is Phase.WAITING:
            assert take.press(now, frame, KEYBOARD) is Press.ACCEPTED
            heard.append((take.label, Sound.START))
    for label, sound in heard:
        if sound is not None:
            path = spoken[label] if sound is Sound.CUE else TONES[sound]
            assert cues.play(path) is None
    assert cues.play(cues.wav("closing", cues.CLOSING)) is None
    assert spawned == []
    assert [w.label for w in take.windows] == [label for label, _, _ in STEPS]


def test_the_audio_off_notice_says_how_to_hear_the_cues():
    notice = render.audio_off()
    assert "<code>aplay</code> was not found" in notice
    assert "sudo apt install alsa-utils" in notice
    assert "restart the panel" in notice
    assert "on-screen cues" in notice
