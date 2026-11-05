# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Publisher-stamp-to-receipt delay of the offered LiveKit streams, measured by the backend's own subscription.

The browser's video element does not receive our per-frame metadata, so its page cannot compute this delay. The
backend can: its receiver sees every frame's publisher timestamp (``user_timestamp_us``) when it arrives. This is
one-way delay from the publisher's stamp to this process taking the frame from the SDK, for a Unix-time stamp; it
includes the publisher's clock offset from this host, so it is exact only as far as the two clocks agree, and it
leaves out the browser's own decode and render time. A monotonic stamp cannot be subtracted from this host's clock
and reports no delay.

The subscription is started by the first request and stopped after a short idle time, because it decodes every
frame. It is for preview, not recording.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections import deque
from collections.abc import Callable, Sequence

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from capture_server import protocol as p

from .livekit_receiver import LiveKitFrameReceiver

logger = logging.getLogger("isaaccapture.egocap.livekit_latency")

#: How long a subscription runs without anyone asking for its numbers before it is shut down.
IDLE_STOP_S = 15.0
#: Only samples from the last few seconds are summarized, so a stalled stream does not keep an old number.
WINDOW_S = 3.0
_MAX_SAMPLES = 600
#: A delay outside this range (a stamp from a different clock domain) is dropped, not averaged. A small negative
#: delay is kept: it means the publisher's clock is ahead by more than the true delay, which is a finding.
_MIN_PLAUSIBLE_MS = -1_000.0
_MAX_PLAUSIBLE_MS = 60_000.0


def summarize(samples_ms: Sequence[float]) -> dict[str, float | int] | None:
    """Median, 95th percentile and count of the samples, or None when there are none."""
    if not samples_ms:
        return None
    ordered = sorted(samples_ms)

    def at(q: float) -> float:
        return ordered[min(len(ordered) - 1, int(q * len(ordered)))]

    return {"median_ms": round(at(0.5), 1), "p95_ms": round(at(0.95), 1), "samples": len(ordered)}


class LiveKitLatencyProbe:
    """Starts on demand, measures each offered stream's stamp-to-receipt delay, and stops when idle."""

    def __init__(
        self,
        streams: Callable[[], Sequence[p.LiveKitCameraStream]],
        make_token: Callable[[str], str],
        *,
        clock_ns: Callable[[], int] = time.monotonic_ns,
        unix_ns: Callable[[], int] = time.time_ns,
    ) -> None:
        self._streams = streams
        self._make_token = make_token
        self._clock_ns = clock_ns
        self._unix_ns = unix_ns
        self._key: tuple[tuple[str, str, str], ...] | None = None
        self._receiver: LiveKitFrameReceiver | None = None
        self._tasks: list[asyncio.Task[None]] = []
        self._samples: dict[str, deque[tuple[int, float]]] = {}
        self._basis: dict[str, str] = {}
        self._last_request_ns = 0
        self._lock = asyncio.Lock()

    async def snapshot(self) -> dict[str, dict[str, object]]:
        """Per-stream delay summary for the last few seconds by stream name; starts the subscription if needed.

        A stream with no recent sample reports ``None`` for its summary, with the reason in ``note``.
        """
        self._last_request_ns = self._clock_ns()
        offered = [s for s in self._streams() if isinstance(s, p.LiveKitCameraStream)]
        async with self._lock:
            await self._ensure(offered)
        now = self._clock_ns()
        out: dict[str, dict[str, object]] = {}
        for stream in offered:
            recent = [ms for at, ms in self._samples.get(stream.name, ()) if now - at <= WINDOW_S * 1e9]
            basis = stream.timestamp_basis.name
            if stream.timestamp_basis is p.LiveKitTimestampBasis.PUBLISHER_MONOTONIC_US:
                note = "a monotonic publisher stamp cannot be compared with this host's clock"
            elif not recent:
                note = "no frames received recently"
            else:
                note = ""
            out[stream.name] = {"basis": basis, "summary": summarize(recent), "note": note}
        return out

    async def stop(self) -> None:
        async with self._lock:
            await self._stop_locked()

    async def _ensure(self, offered: Sequence[p.LiveKitCameraStream]) -> None:
        key = tuple(sorted((s.url, s.room, s.track) for s in offered))
        failed = self._receiver is not None and self._receiver.failed.is_set()
        if self._receiver is not None and key == self._key and not failed:
            return
        await self._stop_locked()
        if not offered:
            return
        self._key = key
        self._samples = {s.name: deque(maxlen=_MAX_SAMPLES) for s in offered}
        receiver = LiveKitFrameReceiver(offered, self._make_token(offered[0].room), queue_frames=2)
        try:
            await receiver.start()
        except Exception:
            logger.exception("LiveKit latency subscription failed to start")
            await receiver.stop()
            self._key = None
            return
        self._receiver = receiver
        self._tasks = [asyncio.get_running_loop().create_task(self._run(receiver, s)) for s in offered]
        self._tasks.append(asyncio.get_running_loop().create_task(self._stop_when_idle()))

    async def _stop_locked(self) -> None:
        tasks, receiver = self._tasks, self._receiver
        self._tasks, self._receiver, self._key = [], None, None
        current = asyncio.current_task()
        for task in tasks:
            if task is not current:
                task.cancel()
        for task in tasks:
            if task is not current:
                with contextlib.suppress(asyncio.CancelledError):
                    await task
        if receiver is not None:
            await receiver.stop()

    async def _stop_when_idle(self) -> None:
        while True:
            await asyncio.sleep(IDLE_STOP_S / 3)
            if self._clock_ns() - self._last_request_ns > IDLE_STOP_S * 1e9:
                logger.info("LiveKit latency subscription idle; stopping")
                asyncio.get_running_loop().create_task(self.stop())
                return

    async def _run(self, receiver: LiveKitFrameReceiver, stream: p.LiveKitCameraStream) -> None:
        """Record the stamp-to-receipt delay of every frame of one stream."""
        unix_basis = stream.timestamp_basis is not p.LiveKitTimestampBasis.PUBLISHER_MONOTONIC_US
        samples = self._samples[stream.name]
        while True:
            frame = await receiver.next_frame_or_lost(stream.stream_id)
            if frame is None:
                await asyncio.sleep(0.5)  # the track is gone; wait for it to come back
                continue
            if not unix_basis or frame.user_timestamp_us is None:
                continue
            delay_ms = (self._unix_ns() / 1000 - frame.user_timestamp_us) / 1000
            if _MIN_PLAUSIBLE_MS <= delay_ms <= _MAX_PLAUSIBLE_MS:
                samples.append((self._clock_ns(), delay_ms))


router = APIRouter()


@router.get("/api/livekit-latency")
async def get_livekit_latency(request: Request) -> JSONResponse:
    """Stamp-to-receipt delay of each offered LiveKit stream, from the backend's own subscription."""
    probe: LiveKitLatencyProbe = request.app.state.livekit_latency
    return JSONResponse({"streams": await probe.snapshot()}, headers={"Cache-Control": "no-store"})
