# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Race regression test for PairedFrameSource.

The bug: the original "consume both, drop if either is None" logic
threw away whichever eye published first whenever the consumer polled
between the two eyes' publishes. At ~1 kHz consumer poll vs ~45 fps
producer, that race hit roughly every cycle and halved the effective
fps with a very stuttery display.

The fix caches each eye's most recent frame and emits a pair on any
update. No GPU needed — pure Python with a controllable fake source.
"""

from __future__ import annotations

import time
from typing import Optional

from isaaccapture_examples.camera_viz.pipeline import Frame, FrameSource, SourceSpec
from isaaccapture_examples.camera_viz.sources._helpers import PairedFrameSource


class FakeSource(FrameSource):
    """One-shot mailbox; ``publish(image)`` arms the next ``latest()`` call."""

    def __init__(self, name: str, width: int = 32, height: int = 32) -> None:
        self._spec = SourceSpec(
            name=name, width=width, height=height, pixel_format="rgba8"
        )
        self._next: Optional[Frame] = None

    @property
    def spec(self) -> SourceSpec:
        return self._spec

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def latest(self) -> Optional[Frame]:
        f = self._next
        self._next = None
        return f

    def publish(self, image) -> None:
        self._next = Frame(
            image=image,
            timestamp_ns=time.monotonic_ns(),
            source_id=self._spec.name,
            stream=0,
        )


def test_paired_emits_first_pair_after_both_eyes_publish():
    """Bootstrap: no emit until both eyes have ever published."""
    left = FakeSource("L")
    right = FakeSource("R")
    paired = PairedFrameSource("pair", left, right)

    assert paired.latest() is None  # nothing published yet

    left.publish("L0")
    assert paired.latest() is None  # right still hasn't

    right.publish("R0")
    f = paired.latest()
    assert f is not None
    assert f.image == "L0" and f.image_right == "R0"


def test_paired_no_drop_at_full_rate():
    """The regression: at every producer cycle, publish L then poll
    then publish R then poll. Under the old code, every L was consumed
    and discarded (right.latest() returned None), so no pair ever made
    it to the consumer. The cache-and-emit fix must surface every
    (Lᵢ, Rᵢ) pair somewhere in the output stream."""
    left = FakeSource("L")
    right = FakeSource("R")
    paired = PairedFrameSource("pair", left, right)

    seen: set[tuple[str, str]] = set()
    N = 20
    for i in range(N):
        left.publish(f"L{i}")
        f = paired.latest()  # race window: old code returned None and lost L{i}
        if f is not None:
            seen.add((f.image, f.image_right))

        right.publish(f"R{i}")
        f = paired.latest()
        if f is not None:
            seen.add((f.image, f.image_right))

    # Every paired publish must appear at some point in the output.
    missing = [(f"L{i}", f"R{i}") for i in range(N) if (f"L{i}", f"R{i}") not in seen]
    assert not missing, f"lost pairs: {missing}"


def test_paired_returns_none_when_nothing_new():
    """latest() returns None when neither child has produced since the
    last call. Otherwise the consumer would keep submitting stale data
    and burning a mailbox slot per poll."""
    left = FakeSource("L")
    right = FakeSource("R")
    paired = PairedFrameSource("pair", left, right)

    left.publish("L0")
    right.publish("R0")
    assert paired.latest() is not None  # first pair emits

    # No new publish — must return None even though both caches are populated.
    assert paired.latest() is None
    assert paired.latest() is None


def test_paired_handles_one_eye_repeatedly_slower():
    """If one eye publishes more often than the other (real-world: RTP
    drops on one stream), the slow eye stays paired with its most-recent
    cached frame. No display freeze, mild inter-eye drift."""
    left = FakeSource("L")
    right = FakeSource("R")
    paired = PairedFrameSource("pair", left, right)

    left.publish("L0")
    right.publish("R0")
    assert paired.latest() is not None  # priming

    # Left advances three times, right doesn't move at all.
    last_image_right = None
    for i in range(1, 4):
        left.publish(f"L{i}")
        f = paired.latest()
        assert f is not None, (
            f"display should not freeze on right-eye stall (cycle {i})"
        )
        assert f.image == f"L{i}"
        assert f.image_right == "R0"  # cached right reused
        last_image_right = f.image_right
    assert last_image_right == "R0"
