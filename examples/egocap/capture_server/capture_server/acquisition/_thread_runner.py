# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""A generic one-dedicated-worker-thread offload utility, shared by every
concrete acquisition adapter (:mod:`capture_server.acquisition.isaac_capture`,
:mod:`capture_server.acquisition.depthai`).

It is not an SDK abstraction at all -- just "run
blocking calls on one dedicated thread, never the event loop". The live
IC adapter owns context entry, synchronous stepping and context exit on
one dedicated thread; it must never block the WebSocket loop or
distribute native operations across arbitrary thread-pool workers.
Applies to every adapter including the mock: a real IC entry/step call, or
even just slow synthetic JPEG generation, must never be able to stall
Stop/keepalive/every other connection activity. One worker thread per
adapter instance (not a shared pool) keeps that guarantee exact.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
from typing import Any, Callable, TypeVar

__all__ = ["ThreadRunner"]

_T = TypeVar("_T")

_RELEASE_TIMEOUT_S = 5.0


class ThreadRunner:
    """Runs blocking calls on one dedicated worker thread."""

    def __init__(self, *, release_timeout_s: float = _RELEASE_TIMEOUT_S) -> None:
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self._release_timeout_s = release_timeout_s
        # Set the first time close() actually submits its job. A retried
        # close() call (made after an earlier bounded wait timed out)
        # awaits this SAME future rather than submitting a second one --
        # the single-worker executor already processes jobs strictly in
        # order, so a second submission would only queue a second
        # close_fn call behind the first, risking it running twice.
        self._close_future: asyncio.Future[None] | None = None

    async def run(self, func: Callable[..., _T], *args: Any) -> _T:
        return await asyncio.get_running_loop().run_in_executor(self._executor, func, *args)

    async def close(self, close_fn: Callable[[], None] | None) -> bool:
        """Release the adapter's resources, bounded, WITHOUT ever blocking
        the event loop. Returns whether release genuinely completed within
        that bound; a caller must never treat "we asked it to stop" as
        proof it did. A ``close_fn`` failure propagates as a real
        exception -- it is not swallowed, since a swallowed release
        failure would be indistinguishable from a genuine release,
        defeating ``_Session.cleanup_ok``'s whole purpose.

        ``close_fn`` runs on THIS runner's own dedicated thread -- the same
        one every other ``run()`` call already runs on -- never a second
        thread. The thread that owns a native resource is the only one
        ever allowed to touch/close it; a second "joiner" thread calling
        ``close_fn`` from outside would mean two threads able to touch the
        same native resource concurrently, which is worse than the
        blocking problem it would be trying to solve.

        Python cannot forcibly kill a thread: if ``close_fn`` itself never
        returns (a genuinely stuck native call), its thread keeps running
        in the background regardless of what this method does --
        ``asyncio.wait`` only bounds how long THIS coroutine waits for
        it, not the underlying thread. That is an acknowledged, unavoidable
        limitation, not something a second thread can paper over.

        Even with no ``close_fn`` at all, this still submits a (no-op) job
        to THIS runner's own executor and waits for it, bounded -- a
        single-worker executor processes jobs strictly in order, so this
        job cannot start until any EARLIER still-running call on this same
        thread (e.g. a stuck ``run()``) actually finishes. Returning
        ``True`` without that confirmation would mean reporting cleanup
        success merely because there was nothing new to close, while the
        same worker thread could still be stuck inside a prior call --
        exactly the "we stopped waiting" vs. "it actually finished"
        distinction this whole method exists to preserve.

        Calling this again after an earlier call returned ``False`` (the
        bound expired) is how a caller confirms a previously-stuck release
        eventually finished: it re-awaits the SAME underlying job, which
        kept running on the worker thread the whole time (see below for
        why an expired bound here never cancels it), rather than racing a
        second submission against it."""

        def _work() -> None:
            if close_fn is not None:
                close_fn()

        if self._close_future is None:
            self._close_future = asyncio.get_running_loop().run_in_executor(self._executor, _work)

            def _observe(fut: "asyncio.Future[None]") -> None:
                # Observation only -- no state stored here.
                # If close() times out and the caller never retries, no
                # one else will ever call .result() on this future; a
                # late close_fn failure would then be logged by asyncio as
                # an "exception never retrieved" warning instead of the
                # real diagnostic it is. Retrieving it here (even though
                # nothing is done with it) is enough to suppress that
                # warning without introducing a second owner of
                # completion state -- the future itself still is one.
                if not fut.cancelled():
                    fut.exception()

            self._close_future.add_done_callback(_observe)
            # wait=False: never block the event loop here even if _work is
            # still queued/running on the worker thread past the bound
            # below -- the executor itself just stops accepting NEW
            # submissions from this point on (there won't be any; this
            # branch only runs once).
            self._executor.shutdown(wait=False)

        # asyncio.wait (not wait_for) deliberately: it reports timeout
        # expiry structurally, via self._close_future being absent from
        # `done`, never by raising TimeoutError itself -- unlike wait_for,
        # whose own TimeoutError on expiry is indistinguishable from
        # close_fn's own code raising a TimeoutError of its own (a native
        # call timing out internally), which must still propagate as the
        # real failure it is, not get silently reinterpreted as "release
        # merely didn't finish in time". Also never cancels the awaited
        # future when this coroutine's own wait is itself cancelled from
        # outside -- the single-worker executor processes jobs strictly in
        # order, so cancelling the submission for a job still queued
        # behind an earlier stuck call would, per concurrent.futures
        # semantics, prevent it from ever running at all once the worker
        # dequeues it, permanently leaking whatever close_fn was meant to
        # release even after the earlier stuck call eventually finishes.
        done, _pending = await asyncio.wait({self._close_future}, timeout=self._release_timeout_s)
        if self._close_future not in done:
            return False
        # .result() re-raises close_fn's own exception if it failed (the
        # real point of this distinction), or returns None on success. A
        # retried close() call after this one still returns the same
        # outcome -- Future.result() is idempotent on an already-done
        # future, so the future itself is the one owner of completion
        # state; nothing else needs to track or re-derive it.
        self._close_future.result()
        return True
