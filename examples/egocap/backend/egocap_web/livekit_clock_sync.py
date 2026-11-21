# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Sends clock-sync requests to a LiveKit publisher and turns its replies into a clock mapping.

Once a second this side stamps ``t1`` with its monotonic clock and sends a small request as a LiveKit data message
addressed to the publisher only; the publisher replies with its own monotonic ``t2`` (receipt) and ``t3`` (reply
send); this side stamps ``t4`` on the reply and hands the four times to ``ClockEstimator``. Requests ride the
existing room connection and never the video path. Messages are little-endian: a request is ``Q``, sequence, t1;
a reply is ``R``, sequence, t1, t2, t3 (nanoseconds, t1 on this side's clock, t2/t3 on the publisher's).

A reply that does not match an outstanding request (sequence and t1) is ignored, so a stale or forged message
cannot become a measurement. Missing replies leave the mapping to go stale, which is the caller's signal.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import struct
import time
from collections.abc import Callable

from .clock_exchange import ClockEstimator, ClockMapping, Exchange

try:
    from livekit import rtc
except ImportError:  # the optional "livekit" extra is not installed; annotations only, nothing runs without it
    rtc = None

logger = logging.getLogger("isaaccapture.egocap.livekit_clock_sync")

CLOCK_SYNC_TOPIC = "isaaccapture.clock-sync"
_REQUEST = struct.Struct("<cIq")
_REPLY = struct.Struct("<cIqqq")
#: Requests kept waiting for a reply; older ones are given up on.
_MAX_OUTSTANDING = 16


def encode_request(sequence: int, t1: int) -> bytes:
    return _REQUEST.pack(b"Q", sequence & 0xFFFFFFFF, t1)


def decode_reply(data: bytes) -> tuple[int, int, int, int] | None:
    """(sequence, t1, t2, t3) of a well-formed reply, else None."""
    if len(data) != _REPLY.size:
        return None
    kind, sequence, t1, t2, t3 = _REPLY.unpack(data)
    return (sequence, t1, t2, t3) if kind == b"R" else None


class ClockSyncClient:
    """Keeps a ``ClockEstimator`` fed by exchanges with one publisher, until stopped."""

    def __init__(
        self,
        room: rtc.Room,
        publisher_identity: str,
        *,
        interval_s: float = 1.0,
        monotonic_ns: Callable[[], int] = time.monotonic_ns,
        estimator: ClockEstimator | None = None,
    ) -> None:
        self._room = room
        self._publisher = publisher_identity
        self._interval_s = interval_s
        self._monotonic_ns = monotonic_ns
        self.estimator = estimator or ClockEstimator()
        self._outstanding: dict[int, int] = {}
        self._sequence = 0
        self._task: asyncio.Task[None] | None = None
        #: Replies rejected by the estimator (malformed, too slow), counted for diagnostics.
        self.rejected = 0

    def start(self) -> None:
        self._room.on("data_received", self._on_data)
        self._task = asyncio.get_running_loop().create_task(self._run())

    async def stop(self) -> None:
        task, self._task = self._task, None
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                _ = await task
        self._room.off("data_received", self._on_data)

    def mapping(self) -> ClockMapping | None:
        """The publisher-to-local clock mapping, or None when there is none yet or it is stale."""
        return self.estimator.mapping(self._monotonic_ns())

    @property
    def discontinuity(self) -> bool:
        return self.estimator.discontinuity

    def _on_data(self, packet: rtc.DataPacket) -> None:
        t4 = self._monotonic_ns()  # first, so it is as close to the receipt as software allows
        if packet.topic != CLOCK_SYNC_TOPIC or packet.participant is None:
            return
        if packet.participant.identity != self._publisher:
            return
        reply = decode_reply(packet.data)
        if reply is None:
            return
        sequence, t1, t2, t3 = reply
        if self._outstanding.get(sequence) != t1:
            return  # not an answer to a request of ours; a stray reply must not discard the real one's entry
        del self._outstanding[sequence]
        if self.estimator.add(Exchange(t1=t1, t2=t2, t3=t3, t4=t4)) is not None:
            self.rejected += 1

    async def _run(self) -> None:
        while True:
            self._sequence += 1
            sequence = self._sequence
            while len(self._outstanding) >= _MAX_OUTSTANDING:
                self._outstanding.pop(next(iter(self._outstanding)))
            try:
                # t1 is read as late as possible, immediately before the send.
                t1 = self._monotonic_ns()
                self._outstanding[sequence & 0xFFFFFFFF] = t1
                await self._room.local_participant.publish_data(
                    encode_request(sequence, t1),
                    reliable=False,
                    destination_identities=[self._publisher],
                    topic=CLOCK_SYNC_TOPIC,
                )
            except Exception:
                logger.exception("clock-sync request failed")
            await asyncio.sleep(self._interval_s)
