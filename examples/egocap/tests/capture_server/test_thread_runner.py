"""Tests for ThreadRunner.close()'s bounded-wait behavior: a bounded
timeout must never cancel the underlying close_fn job itself, only this
coroutine's own wait for it, and must be structurally distinguishable
from close_fn itself raising a TimeoutError of its own -- see
capture_server.acquisition._thread_runner.ThreadRunner.close's
docstring for the full rationale.
"""

from __future__ import annotations

import asyncio
import threading

import pytest

from capture_server.acquisition._thread_runner import ThreadRunner

pytestmark = pytest.mark.asyncio


async def test_close_does_not_abandon_a_job_queued_behind_a_stuck_run_call():
    """A close_fn submitted while an earlier run() call is still blocked
    sits queued behind it on the single-worker executor. The bounded
    wait_for must time out (False, not confirmed) without cancelling
    that queued submission -- once the blocked run() call actually
    finishes, close_fn must still run (on the SAME worker thread, never
    a second one), and a later close() call must observe that genuine
    completion rather than racing a second submission against it."""
    entry_started = threading.Event()
    release_entry = threading.Event()
    run_thread_id: dict[str, int] = {}
    close_thread_ids: list[int] = []

    def _blocked_run() -> None:
        entry_started.set()
        run_thread_id["id"] = threading.get_ident()
        release_entry.wait()

    def _close_fn() -> None:
        close_thread_ids.append(threading.get_ident())

    runner = ThreadRunner(release_timeout_s=0.2)
    run_task = asyncio.ensure_future(runner.run(_blocked_run))
    await asyncio.wait_for(asyncio.to_thread(entry_started.wait, 5), timeout=5)

    # close()'s own _work job is now queued behind the still-blocked
    # run() call -- the 0.2s bound expires long before run() releases.
    first = await runner.close(_close_fn)
    assert first is False
    assert close_thread_ids == [], "close_fn must not have run yet -- still queued behind run()"

    release_entry.set()
    await run_task  # let the previously-stuck call actually finish

    # A second close() call re-awaits the SAME shielded future (never
    # cancelled by the earlier timeout) rather than submitting a new job.
    second = await runner.close(_close_fn)
    assert second is True
    assert len(close_thread_ids) == 1, "close_fn must run exactly once, never twice"
    assert close_thread_ids[0] == run_thread_id["id"], (
        "close_fn must run on the SAME worker thread as run(), never a second closer thread"
    )


async def test_close_retains_a_close_fn_failure_across_an_earlier_timeout():
    """A close_fn that fails must not have its exception silently
    discarded just because an earlier bounded wait already gave up on
    it -- a later call (or, equivalently, the future's own done callback)
    must still observe the real failure rather than reporting confirmed
    release for a release that actually failed."""
    release_entry = threading.Event()
    entry_started = threading.Event()

    def _slow_failing_close() -> None:
        entry_started.set()
        release_entry.wait()
        raise RuntimeError("native release genuinely failed")

    runner = ThreadRunner(release_timeout_s=0.2)
    close_task = asyncio.ensure_future(runner.close(_slow_failing_close))
    await asyncio.wait_for(asyncio.to_thread(entry_started.wait, 5), timeout=5)

    first = await close_task
    assert first is False  # timed out, not confirmed

    release_entry.set()
    with pytest.raises(RuntimeError, match="native release genuinely failed"):
        await runner.close(_slow_failing_close)


async def test_close_fn_raising_its_own_timeout_error_still_propagates():
    """close_fn raising a genuine TimeoutError of its own (e.g. a native
    call that timed out internally) must propagate as that real failure
    -- never get silently reinterpreted as "the bounded wait itself gave
    up" just because the exception happens to share a type with the
    wait-expiry case. A fast failure (well within the bound) proves the
    future resolved WITH this exception, not that close() merely stopped
    waiting for it."""

    def _fails_with_timeout_error() -> None:
        raise TimeoutError("native release call timed out internally")

    runner = ThreadRunner(release_timeout_s=5.0)
    with pytest.raises(TimeoutError, match="native release call timed out internally"):
        await runner.close(_fails_with_timeout_error)
