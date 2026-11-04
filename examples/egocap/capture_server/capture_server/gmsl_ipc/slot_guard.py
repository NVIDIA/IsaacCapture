# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Reader side of the receiver's slot ownership protocol (frame feed protocol v3).

Every ring slot has a guard file whose first eight bytes are the slot's current generation as a big-endian
unsigned integer. To overwrite a slot the publisher takes an exclusive lock and zeroes the generation; once the
new frame is complete it writes the new generation. A reader holds a shared lock for the whole device-to-host
copy, so the publisher cannot overwrite the slot mid-read, and compares the generation with the one in the frame
notice, so a slot that was reused after the notice was sent is never read as if it were the announced frame.

The file layout must match the publisher's, which lives in a different process and repository directory; it is
restated here instead of imported so this package depends on nothing outside ``capture_server``.
"""

from __future__ import annotations

import fcntl
import os
import struct
from collections.abc import Iterator
from contextlib import contextmanager

__all__ = ["read_slot"]

_GENERATION = struct.Struct("!Q")


@contextmanager
def read_slot(path: str, generation: int) -> Iterator[bool]:
    """Hold a shared lock on the slot's guard file and yield whether ``generation`` is still current.

    Yields False, without holding anything, when the guard file is gone, when the publisher holds the exclusive
    lock (it is writing the slot), or when the slot's generation is not ``generation``. Yields True only with
    the shared lock held, so the caller may copy the slot's bytes inside the ``with`` block and nowhere else.
    """
    try:
        guard = open(path, "rb", buffering=0)
    except FileNotFoundError:
        # The publisher restarted and removed its guard directory: every notice from before is stale.
        yield False
        return
    with guard:
        try:
            fcntl.flock(guard, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError:
            # Non-blocking on purpose: a busy slot means the publisher is overwriting it, so the frame is gone.
            yield False
            return
        current = os.pread(guard.fileno(), _GENERATION.size, 0)
        # Generation zero is "being written"; it never matches, even if a notice carried zero.
        yield generation > 0 and current == _GENERATION.pack(generation)
