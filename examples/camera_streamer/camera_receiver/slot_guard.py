# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cross-process slot ownership; locks cover GPU work, not just submission."""

import fcntl
import os
import struct
from contextlib import contextmanager

_GENERATION = struct.Struct("!Q")


@contextmanager
def write_slot(path, stop):
    """Invalidate old notifications before writing; wait interruptibly for readers."""
    with open(path, "r+b", buffering=0) as guard:
        while True:
            if stop.is_set():
                raise InterruptedError("CUDA IPC slot write cancelled")
            try:
                fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                stop.wait(0.01)
        guard.write(_GENERATION.pack(0))
        yield


def commit_slot(path, generation):
    if generation <= 0:
        raise ValueError("slot generation must be positive")
    with open(path, "r+b", buffering=0) as guard:
        fcntl.flock(guard, fcntl.LOCK_EX)
        guard.write(_GENERATION.pack(generation))


@contextmanager
def read_slot(path, generation):
    """Yield False for stale/busy slots; hold a shared lock through a valid read."""
    try:
        guard = open(path, "rb", buffering=0)
    except FileNotFoundError:
        yield False
        return
    with guard:
        try:
            fcntl.flock(guard, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError:
            yield False
            return
        current = os.pread(guard.fileno(), _GENERATION.size, 0)
        yield generation > 0 and current == _GENERATION.pack(generation)
