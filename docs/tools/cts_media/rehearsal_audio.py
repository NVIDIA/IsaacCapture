# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Write the hand take's audio, timed as the capture panel plays it, as an MP3.

    python3 docs/tools/cts_media/rehearsal_audio.py docs/source/conformance/hand-rehearsal.mp3

Per step (hand/capture/steps.py): the cue, COUNTDOWN_BEATS count tones BEAT_S apart,
the beep, the hold, the tick; the next cue follows the tick's own length. After the last
tick come TAIL_S (capture_panel.py) and the closing cue. Prints the beep and tick times.
"""

from __future__ import annotations

import argparse
import ast
import subprocess
import tempfile
import wave
from pathlib import Path

HERE = Path(__file__).resolve().parent
CAPTURE = HERE.parents[2] / "src/python/isaaccapture/cts/hand/capture"
CUES = CAPTURE / "cues"
LEAD_S = 2.0  # silence before the first cue
BITRATE = "64k"


def constants(path: Path, *names: str) -> list:
    """Module-level literals from ``path``, read without importing it."""
    found = {}
    for node in ast.parse(path.read_text()).body:
        target = getattr(node, "target", None) or (
            node.targets[0] if isinstance(node, ast.Assign) else None
        )
        if getattr(target, "id", None) in names:
            found[target.id] = ast.literal_eval(node.value)
    missing = [n for n in names if n not in found]
    if missing:
        raise SystemExit(f"{path} has no {missing}")
    return [found[n] for n in names]


def read(name: str) -> tuple[bytes, int]:
    path = CUES / f"{name}.wav"
    if path.read_bytes()[:24].startswith(b"version https://git-lfs"):
        raise SystemExit(f"{path} is a Git LFS pointer, not audio; run `git lfs pull`")
    with wave.open(str(path)) as w:
        if (w.getnchannels(), w.getsampwidth()) != (1, 2):
            raise SystemExit(f"{path}: expected mono 16-bit PCM")
        return w.readframes(w.getnframes()), w.getframerate()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    steps, beats, beat_s = constants(
        CAPTURE / "session.py", "STEPS", "COUNTDOWN_BEATS", "BEAT_S"
    )
    (tail_s,) = constants(CAPTURE / "capture_panel.py", "TAIL_S")

    clips = {
        n: read(n) for n in ["count", "beep", "tick", "closing", *(s[0] for s in steps)]
    }
    rates = {rate for _, rate in clips.values()}
    if len(rates) != 1:
        raise SystemExit(f"cue WAVs differ in sample rate: {rates}")
    rate = rates.pop()
    seconds = {n: len(pcm) / 2 / rate for n, (pcm, _) in clips.items()}

    placed: list[tuple[float, str]] = []
    t = LEAD_S
    for i, (label, hold_s, _) in enumerate(steps):
        placed.append((t, label))
        t += seconds[label]
        for beat in range(beats):
            placed.append((t + beat * beat_s, "count"))
        t += beats * beat_s
        placed.append((t, "beep"))
        print(f"{i + 1:2d} {label:20s} beep {t:7.3f} s", end="")
        t += hold_s
        placed.append((t, "tick"))
        print(f"  tick {t:7.3f} s")
        if i + 1 < len(steps):
            t += seconds["tick"]
    t += tail_s
    placed.append((t, "closing"))
    end = t + seconds["closing"]

    track = bytearray(round(end * rate) * 2)
    for at, name in placed:
        pcm = clips[name][0]
        start = round(at * rate) * 2
        track[start : start + len(pcm)] = pcm
    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "track.wav"
        with wave.open(str(wav), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(rate)
            w.writeframes(bytes(track))
        args.out.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-y",
                "-i",
                str(wav),
                "-ac",
                "1",
                "-codec:a",
                "libmp3lame",
                "-b:a",
                BITRATE,
                "-map_metadata",
                "-1",
                "-fflags",
                "+bitexact",
                str(args.out),
            ],
            check=True,
        )
    print(f"closing {t:.3f} s; ends {end:.3f} s -> {args.out}")


if __name__ == "__main__":
    main()
