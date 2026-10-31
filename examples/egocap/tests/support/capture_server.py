# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""The one shared way to start/stop the production CaptureServer for tests,
running it in its own built-in mock acquisition mode -- "mock" is a mode of
the one real server, not a separate implementation. Reused by
tests/backend/test_capture_status.py, tests/backend/test_preview_routes.py,
tests/capture_server/conftest.py and tests/capture_server/test_server.py so
no test package reimplements server startup/teardown. A test specifically
about CaptureServer's own run()/close() internals (not just using a running
server as a fixture for something else) may still construct CaptureServer
directly."""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
from collections.abc import Callable

from capture_server.launch import SourceFactory, build_mock_source_factory
from capture_server.server import CaptureServer, ServerConfig

# close()'s own documented worst case: up to _SOURCE_RELEASE_TIMEOUT_S(5) +
# _SEND_DEADLINE_S(2) + 1.0 margin waiting out an active session, THEN up to
# _CLOSE_TIMEOUT_S(2) waiting for the listener itself -- ~10s total. Bounding
# our own wait a little above that lets a genuine hang still fail loudly
# instead of hanging a test suite, without racing close()'s own budget.
_CLOSE_BOUND_S = 12.0
_RUN_TASK_JOIN_BOUND_S = 5.0


@dataclasses.dataclass
class RunningServer:
    """A CaptureServer together with the asyncio task actually running it,
    so teardown can wait for that task instead of discarding it."""

    server: CaptureServer
    run_task: asyncio.Task[None]


async def start_capture_server(
    port: int,
    *,
    source_factory: SourceFactory | None = None,
    ic_faults: set[str] | frozenset[str] | None = None,
    camera_faults: set[str] | frozenset[str] | None = None,
    config: ServerConfig | None = None,
) -> RunningServer:
    """Starts the actual CaptureServer -- with mock acquisition by default,
    or an explicit ``source_factory`` for a test that needs to inject one
    native failure -- waits for its listener to be ready, and returns it
    paired with its owned run() task. ``config`` carries any non-default
    ``ServerConfig`` fields a test needs (e.g. ``connect_delay``,
    ``known_cameras``); its ``port`` is always overridden with this call's
    own ``port``, so callers never need to keep the two in sync themselves.
    Cleans up the run task and any already-open listener, through the
    production ``close()`` path, on every exit path -- a startup failure, a
    readiness timeout, or this coroutine itself being cancelled by a
    caller's own timeout -- not only the timeout-reached path."""
    effective_config = dataclasses.replace(config, port=port) if config is not None else ServerConfig(port=port)
    server = CaptureServer(
        effective_config,
        source_factory=source_factory
        or build_mock_source_factory(
            scenario="demo",
            seed=1,
            ic_faults=ic_faults or set(),
            camera_faults=camera_faults or set(),
            # The SENSING composition shares this config with the server (its cached sensor description).
            config=effective_config,
        ),
    )
    run_task = asyncio.ensure_future(server.run())
    started = False
    try:
        for _ in range(200):
            if server._ws_server is not None:
                started = True
                return RunningServer(server, run_task)
            if run_task.done():
                run_task.result()  # surfaces a real startup failure instead of looping to the timeout below
            await asyncio.sleep(0.01)
        raise AssertionError("server did not start listening in time")
    finally:
        if not started:
            # Cancel/observe the run task BEFORE closing -- otherwise a
            # startup still in flight could finish acquiring the listener
            # just after the one close() attempt already ran and found
            # nothing open yet, leaking it. Nested finally: a failure or
            # timeout in either step must not skip the other.
            try:
                run_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await asyncio.wait_for(run_task, timeout=_RUN_TASK_JOIN_BOUND_S)
            finally:
                await asyncio.wait_for(server.close(), timeout=_CLOSE_BOUND_S)


async def stop_capture_server(running: RunningServer) -> None:
    """Requests the server stop and closes its listener through the
    production ``close()`` path, then waits (bounded) for its run() task to
    finish -- even if ``close()`` itself raised, so a close failure can
    never leave the run task unobserved."""
    try:
        running.server.request_stop()
        await asyncio.wait_for(running.server.close(), timeout=_CLOSE_BOUND_S)
    finally:
        await asyncio.wait_for(running.run_task, timeout=_RUN_TASK_JOIN_BOUND_S)


async def poll_until(predicate: Callable[[], bool], *, timeout_s: float = 5.0) -> None:
    """Bounded poll for a synchronous condition instead of a fixed sleep --
    callback-driven state can legitimately take a few event-loop turns."""
    for _ in range(int(timeout_s / 0.01)):
        if predicate():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("condition not met within timeout")
