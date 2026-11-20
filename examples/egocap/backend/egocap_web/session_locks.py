# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""One holder at a time for a recorded session's labels.

Two things rewrite or copy the labels of a session (or of a routed group): the end-of-day task correction, which
rewrites the episodes, ``config.yaml`` and the time-aligned copy, and the background aligner, which copies an
episode's labels into that aligned copy. Each takes the session's lock for the whole of its work, so a correction
never lands between the aligner's copy and its publication, and two corrections never interleave.

Taking a lock never waits. A caller that finds it held learns who holds it and decides for itself: the correction
refuses and the operator tries again, the aligner looks again shortly.
"""

from __future__ import annotations

import threading
from pathlib import Path

#: The two kinds of holder.
ALIGNER, CORRECTION = "aligner", "correction"


class SessionLocks:
    """Which sessions' labels are held, and by whom. Safe to use from the event loop and from worker threads."""

    def __init__(self) -> None:
        """Nothing is held at first."""
        self._guard = threading.Lock()
        # A session is listed only while it is held, so this never grows with the sessions on disk.
        self._held: dict[Path, str] = {}

    def acquire(self, directory: Path, holder: str) -> str | None:
        """Take the lock of ``directory`` for ``holder``. Returns None when taken, else who holds it now.

        ``directory`` is the resolved session or routed group directory; both callers resolve it the same
        way, off the event loop, so one session is never two keys.
        """
        with self._guard:
            current = self._held.get(directory)
            if current is None:
                self._held[directory] = holder
            return current

    def release(self, directory: Path) -> None:
        """Give the lock of ``directory`` back."""
        with self._guard:
            self._held.pop(directory, None)
