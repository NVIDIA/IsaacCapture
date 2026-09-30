# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import synth

from hand_acceptance.checks import build_all
from hand_acceptance.frames import SCHEMA_NAME, SourceMetadata
from hand_acceptance.panel import status
from hand_acceptance.panel.track import HandTee, centre
from hand_acceptance.profile import WRIST
from hand_acceptance.report import run

PACKAGE = Path(__file__).resolve().parents[1] / "src" / "hand_acceptance"
RENDERER = PACKAGE / "panel" / "app.py"
IMPORTS_VISER = re.compile(r"^\s*(?:import viser|from viser)", re.MULTILINE)


class _Source:
    def __init__(self, frames):
        self._frames = frames
        self.metadata = SourceMetadata(
            schema_name=SCHEMA_NAME,
            schema_encoding="flatbuffer",
            schema_data=None,
            message_encoding="flatbuffer",
            topic="hands/left_hand",
            profile=None,
            channel_found=True,
        )

    def __iter__(self):
        return iter(self._frames)


def _interleaved(count: int):
    left = synth.take(count, "left")
    right = synth.take(count, "right")
    return [frame for pair in zip(left, right) for frame in pair]


def test_only_the_renderer_imports_viser():
    offenders = sorted(
        path.relative_to(PACKAGE).as_posix()
        for path in PACKAGE.rglob("*.py")
        if path != RENDERER and IMPORTS_VISER.search(path.read_text())
    )
    assert offenders == []


def test_the_checker_and_the_panel_arithmetic_import_without_viser():
    probe = (
        "import sys; sys.modules['viser'] = None;"
        "import hand_acceptance.cli, hand_acceptance.panel.track,"
        " hand_acceptance.panel.status"
    )
    done = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=PACKAGE.parent,
        capture_output=True,
        text=True,
    )
    assert done.returncode == 0, done.stderr


def test_tee_splits_the_hands_onto_one_time_axis():
    frames = _interleaved(30)
    tee = HandTee(_Source(frames))
    assert list(tee) == frames
    tracks = tee.tracks()
    assert set(tracks) == {"left", "right"}
    assert len(tracks["left"].samples) == len(tracks["right"].samples) == 30
    assert tracks["left"].times_s == tracks["right"].times_s
    assert tracks["left"].times_s[0] == 0.0


def test_centre_is_between_the_wrists():
    tracks = HandTee(_Source(_interleaved(3)))
    list(tracks)
    built = tracks.tracks()
    middle = centre(built)
    wrists = [t.samples[0].positions[WRIST] for t in built.values()]
    assert middle == tuple((a + b) / 2 for a, b in zip(*wrists))


def test_every_check_lands_in_a_titled_group():
    report = run(_Source(_interleaved(30)), build_all())
    shown = sum(len(group.results) for group in status.groups(report))
    assert shown == len(report.results)
