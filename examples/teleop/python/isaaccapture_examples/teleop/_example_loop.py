# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Duration and cancellation shared by the interactive retargeting examples."""

import argparse
import math
import time
from collections.abc import Callable


def _duration_seconds(value: str) -> float:
    seconds = float(value)
    if not math.isfinite(seconds) or seconds < 0:
        raise argparse.ArgumentTypeError("duration must be finite and non-negative")
    return seconds


def add_duration_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--duration",
        type=_duration_seconds,
        default=0.0,
        metavar="SECONDS",
        help="Run for this many seconds after session startup; 0 runs until Ctrl+C (default).",
    )


def run_example_loop(step: Callable[[], None], duration: float) -> None:
    """Run inside the session context; let stream and teardown failures propagate."""
    if duration == 0:
        print("Running until Ctrl+C.", flush=True)
        deadline = None
    else:
        print(
            f"Running for {duration:g} seconds; press Ctrl+C to stop early.", flush=True
        )
        deadline = time.monotonic() + duration

    # Handle cancellation before unwinding the session, without hiding cleanup errors.
    try:
        while deadline is None or time.monotonic() < deadline:
            step()
            time.sleep(0.016)  # ~60 FPS
    except KeyboardInterrupt:
        print("\nStop requested (Ctrl+C). Closing session...", flush=True)
    else:
        print("\nDuration reached. Closing session...", flush=True)
