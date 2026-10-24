# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Writes the label sidecar from the timer windows and audits it against the take.

The audit reads the finished file: both hand channels present, every window inside the
recorded span with records from both hands, and the checker's "step performed" result.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from hand_cts.checks import Status, build
from hand_cts.labels import SIDECAR_SUFFIX, Step, StepTimeline
from hand_cts.mcap_source import McapFrameSource

from steps import Window

NOTE = "Step labels for the recording beside this file."
CLOCK_DOMAIN = "sample_time_local_common_clock (system monotonic, nanoseconds)"

Check = tuple[str, bool, str]


def _timeline(windows: Sequence[Window]) -> StepTimeline:
    return StepTimeline(
        steps=tuple(
            Step(w.index, w.label, w.start_ns, w.end_ns, w.is_still_window)
            for w in windows
        )
    )


def audit(recording: Path, windows: Sequence[Window]) -> tuple[list[Check], dict]:
    source = McapFrameSource(recording)
    samples: dict[str, list[int]] = {"left": [], "right": []}
    active: dict[str, int] = {"left": 0, "right": 0}
    checks_run = build(
        [
            "segmentation.label_alignment",
            "segmentation.labelled_step_actually_performed",
        ],
        _timeline(windows),
    )
    for frame in source:
        if frame.side in samples and frame.sample_time_ns is not None:
            samples[frame.side].append(frame.sample_time_ns)
            active[frame.side] += frame.has_payload
        for check in checks_run:
            check.update(frame)

    checks: list[Check] = [
        (
            "hand channels",
            all(samples.values()),
            f"left {len(samples['left'])} records ({active['left']} active), "
            f"right {len(samples['right'])} ({active['right']} active)",
        )
    ]
    stamped = sorted(samples["left"] + samples["right"])
    info = {"records": {side: len(v) for side, v in samples.items()}}
    if not stamped:
        return checks, info
    info["first_ns"] = stamped[0]
    span_s = (samples["left"][-1] - samples["left"][0]) / 1e9 if samples["left"] else 0
    info["rate_hz"] = round(len(samples["left"]) / span_s, 1) if span_s > 0 else None

    for check in checks_run:
        outcome = check.result()
        checks.append((check.name, outcome.status is Status.PASS, outcome.detail))
    return checks, info


def write(
    recording: Path, windows: Sequence[Window]
) -> tuple[Path | None, list[Check]]:
    """Writes the sidecar beside the take; returns it and the audit rows.

    No sidecar is written without windows.
    """
    checks, info = audit(recording, windows)
    if not windows or "first_ns" not in info:
        return None, checks
    first = info["first_ns"]
    payload = {
        "note": NOTE,
        "clock_domain": CLOCK_DOMAIN,
        "nominal_rate_hz": info.get("rate_hz"),
        "records": info["records"],
        "steps": [
            {
                "index": w.index,
                "label": w.label,
                "start_ns": w.start_ns,
                "end_ns": w.end_ns,
                "start_s_from_first_sample": round((w.start_ns - first) / 1e9, 6),
                "end_s_from_first_sample": round((w.end_ns - first) / 1e9, 6),
                "is_still_window": w.is_still_window,
                "boundary_source": w.source,
            }
            for w in windows
        ],
        "mcap": recording.name,
    }
    sidecar = recording.with_name(recording.stem + SIDECAR_SUFFIX)
    sidecar.write_text(json.dumps(payload, indent=2) + "\n")
    return sidecar, checks


def main(argv: list[str] | None = None) -> int:
    """Re-audit a take whose sidecar already exists: ``make_labels.py TAKE.mcap``."""
    import argparse

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("recording", type=Path)
    args = parser.parse_args(argv)
    timeline = StepTimeline.beside(args.recording)
    if timeline is None:
        parser.error("no sidecar beside the recording")
    windows = [
        Window(s.index, s.label, s.start_ns, s.end_ns, s.is_still_window)
        for s in timeline.steps
    ]
    checks, _ = audit(args.recording, windows)
    for label, ok, detail in checks:
        print(f"{'ok  ' if ok else 'BAD '} {label:<48} {detail}")
    return 0 if all(ok for _, ok, _ in checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
