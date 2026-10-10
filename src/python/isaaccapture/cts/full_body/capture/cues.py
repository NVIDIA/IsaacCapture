# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The spoken cues and the two window tones, played through ``aplay``.

The WAVs in ``cues/`` are pre-rendered; nothing is synthesised at run time.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import wave
from pathlib import Path

HERE = Path(__file__).resolve().parent
CUES = HERE / "cues"

# Window start and end tones; they must sound different.
START_TONE = CUES / "beep.wav"
END_TONE = CUES / "tick.wav"

#: The one spoken line that is not a step, played once the file is closed.
CLOSING = "Done. You can stop now."


def _require_audio(path: Path) -> None:
    """Exit if ``path`` is a Git LFS pointer rather than audio."""
    with path.open("rb") as handle:
        if handle.read(24).startswith(b"version https://git-lfs"):
            raise SystemExit(
                f"{path} is a Git LFS pointer, not audio; run `git lfs pull`"
            )


def wav(label: str, text: str) -> Path:
    """The recorded WAV for one cue; exits if the cue text no longer matches ``cues/index.json``."""
    index = json.loads((CUES / "index.json").read_text())
    recorded = index["text_sha1"].get(label)
    current = hashlib.sha1(text.encode(), usedforsecurity=False).hexdigest()[:16]
    if recorded is None:
        raise SystemExit(f"{CUES}/index.json has no cue named {label!r}")
    if recorded != current:
        raise SystemExit(
            f"the wording of cue {label!r} changed, so {label}.wav no longer says it:\n"
            f"  now: {text!r}\n"
            f"re-render it with piper at voice {index['voice']} and length scale "
            f"{index['length_scale']}, then update index.json"
        )
    _require_audio(CUES / f"{label}.wav")
    return CUES / f"{label}.wav"


def duration_of(path: Path) -> float:
    _require_audio(path)
    with wave.open(str(path)) as handle:
        return handle.getnframes() / handle.getframerate()


def play(path: Path) -> subprocess.Popen:
    """Start a sound and return its process handle without waiting."""
    return subprocess.Popen(
        ["aplay", "-q", str(path)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
