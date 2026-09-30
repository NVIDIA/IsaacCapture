# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The spoken cues and the three tones, played through ``aplay``.

The WAVs in ``cues/`` were rendered once by piper and committed; nothing is synthesised
at run time.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import wave
from pathlib import Path

HERE = Path(__file__).resolve().parent
CUES = HERE / "cues"

# Three sounds the performer must tell apart without looking: counting down, the window
# opening, the window closing.
COUNT_TONE = CUES / "count.wav"
START_TONE = CUES / "beep.wav"
END_TONE = CUES / "tick.wav"

CLOSING = "Done. You can stop now."


def wav(label: str, text: str) -> Path:
    """The WAV for one cue, refusing one rendered from other wording."""
    index = json.loads((CUES / "index.json").read_text())
    recorded = index["text_sha1"].get(label)
    current = hashlib.sha1(text.encode()).hexdigest()[:16]
    if recorded is None:
        raise SystemExit(f"{CUES}/index.json has no cue named {label!r}")
    if recorded != current:
        raise SystemExit(
            f"the wording of cue {label!r} changed, so {label}.wav no longer says it:\n"
            f"  now: {text!r}\n"
            f"re-render it with piper at voice {index['voice']} and length scale "
            f"{index['length_scale']}, then update index.json"
        )
    return CUES / f"{label}.wav"


def duration_of(path: Path) -> float:
    with wave.open(str(path)) as handle:
        return handle.getnframes() / handle.getframerate()


def play(path: Path) -> subprocess.Popen:
    return subprocess.Popen(
        ["aplay", "-q", str(path)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
