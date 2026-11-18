# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Test support for the real-backend service tests.

* A per-test watchdog (SIGALRM) turns a hang into a failure instead of blocking the run forever. It is only a
  last resort: it does not prove that a storage thread or encoder has settled.
* ``own`` is an explicit, opt-in cleanup: a test registers a service it creates, e.g.
  ``service = own(RecordingService(config))``, and when the test ends the service is closed and its CloseResult
  is checked. A failing test keeps its own assertion; a service that cannot release is reported as a separate
  teardown error, unless the test passes ``may_stay_open=True`` because leaving it unreleased is the behavior
  under test.
* After each test, any ffmpeg encoder still a direct child of this pytest process is reported (the test is failed
  naming the pids) and killed. Only children of this process are considered; nothing system-wide is touched.

Known risk (test code only, deliberately left simple): the upstream ffmpeg writer is a non-daemon thread that
blocks waiting for frames until its writer is released, and killing ffmpeg does not wake it. If a test leaks an
encoder, the failure above is reported but the pytest process may then never exit. Stop it by hand (Ctrl-C or
kill); no hard-exit hook is installed on purpose. This only happens when a test fails while holding an encoder,
for example a deliberately broken build in a mutation run, whose driver runs each case under a deadline.
"""

import os
import signal
from pathlib import Path

import pytest
import pytest_asyncio

WATCHDOG_SECONDS = 90


def _leaked_encoder_pids() -> list[int]:
    """PIDs of ffmpeg processes that are direct children of this process (read from /proc, no external tool)."""
    me, leaked = os.getpid(), []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            stat = (entry / "stat").read_text()
        except OSError:  # the process exited while we were looking
            continue
        # Format: pid (comm) state ppid ...; comm may contain spaces or parentheses, so split at the last ")".
        comm, rest = stat[stat.index("(") + 1 : stat.rindex(")")], stat[stat.rindex(")") + 2 :].split()
        if comm == "ffmpeg" and int(rest[1]) == me:
            leaked.append(int(entry.name))
    return leaked


@pytest.fixture(autouse=True)
def hang_watchdog_and_encoder_leak_check():
    """Fail a test that exceeds the watchdog, and fail (then kill) any encoder process the test leaked."""

    def on_timeout(_signum, _frame):
        raise TimeoutError(f"test exceeded the {WATCHDOG_SECONDS}s watchdog")

    previous = signal.signal(signal.SIGALRM, on_timeout)
    signal.alarm(WATCHDOG_SECONDS)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)
        leaked = _leaked_encoder_pids()
        for pid in leaked:
            os.kill(pid, signal.SIGKILL)
        if leaked:
            pytest.fail(f"test leaked ffmpeg encoder process(es) {leaked}; killed")


@pytest_asyncio.fixture
async def own(hang_watchdog_and_encoder_leak_check):
    """Register services a test creates; each is closed at teardown and must report release."""
    owned = []

    def register(service, *, may_stay_open: bool = False):
        owned.append((service, may_stay_open))
        return service

    yield register
    unreleased = []
    for service, may_stay_open in owned:
        closed = await service.close()
        if not closed.released and not may_stay_open:
            unreleased.append(closed)
    assert not unreleased, f"services did not release their resources at teardown: {unreleased}"
