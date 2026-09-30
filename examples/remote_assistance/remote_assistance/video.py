# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A bounded video receiver, usable without CUDA or an XR runtime."""

import asyncio
import logging
import os
import threading
import time
from dataclasses import dataclass

from livekit import rtc

from .connection import connect, disconnect

logger = logging.getLogger("isaaccapture.remote_assistance.video")


@dataclass
class ReceivedFrame:
    frame: rtc.VideoFrame
    received_ns: int
    metadata: object


class VideoReceiver:
    """Receive one named track from one participant into a latest-frame mailbox."""

    def __init__(self, settings: dict, *, identity: str, publisher: str, track: str):
        decoder = settings.get("video_decoder", "auto")
        if decoder not in ("auto", "software"):
            raise ValueError("video_decoder must be auto or software")
        if decoder == "software":
            # SDK-wide option; set before creating media peer connections. Keep CUDA
            # available to Televiz: CUDA_VISIBLE_DEVICES would disable rendering too.
            os.environ["LK_DISABLE_NVDEC"] = "1"
        self.settings = settings
        self.identity = identity
        self.publisher = publisher
        self.track = track
        self.frames = 0
        self.overwritten = 0
        self.error = None
        self._latest = None
        self._lock = threading.Lock()
        self._ready = threading.Event()
        self._thread = None
        self._loop = None
        self._stop = None

    def latest(self):
        if self.error:
            raise RuntimeError("LiveKit video receiver failed") from self.error
        with self._lock:
            result, self._latest = self._latest, None
        return result

    def start(self):
        if self._thread:
            return
        self._ready.clear()
        self.error = None
        self._thread = threading.Thread(
            target=self._run, name="livekit-video", daemon=True
        )
        self._thread.start()
        if not self._ready.wait(20):
            self.stop()
            raise TimeoutError("LiveKit receiver did not connect within 20 seconds")
        if self.error:
            self.stop()
            raise RuntimeError("LiveKit receiver could not connect") from self.error

    def stop(self):
        if self._loop and self._stop and not self._loop.is_closed():
            self._loop.call_soon_threadsafe(self._stop.set)
        if self._thread:
            self._thread.join(10)
            if self._thread.is_alive():
                raise TimeoutError("LiveKit video receiver did not stop")
            self._thread = None
        with self._lock:
            self._latest = None

    def _run(self):
        try:
            asyncio.run(self._receive())
        except Exception as exc:
            self.error = exc
            logger.exception("Video receiver failed")
        finally:
            self._ready.set()

    async def _receive(self):
        self._loop = asyncio.get_running_loop()
        self._stop = asyncio.Event()
        room = rtc.Room()
        readers = {}

        def subscribe(publication, participant):
            if (
                participant.identity == self.publisher
                and publication.name == self.track
            ):
                publication.set_subscribed(True)

        async def read(track):
            stream = rtc.VideoStream(track, capacity=1, format=rtc.VideoBufferType.RGBA)
            try:
                async for event in stream:
                    with self._lock:
                        self.overwritten += self._latest is not None
                        self._latest = ReceivedFrame(
                            event.frame, time.monotonic_ns(), event.metadata
                        )
                        self.frames += 1
            finally:
                await stream.aclose()

        def subscribed(track, publication, participant):
            if participant.identity != self.publisher or publication.name != self.track:
                return
            if track.kind != rtc.TrackKind.KIND_VIDEO:
                return
            logger.info("Receiving %s / %s", participant.identity, publication.name)
            task = asyncio.create_task(read(track))
            previous = readers.get(publication.sid)
            if previous:
                previous.cancel()
            readers[publication.sid] = task

            def finished(task):
                if not task.cancelled() and task.exception():
                    self.error = task.exception()
                    self._stop.set()

            task.add_done_callback(finished)

        def unsubscribed(track, publication, participant):
            task = readers.pop(publication.sid, None)
            if task:
                task.cancel()
            with self._lock:
                self._latest = None

        room.on("track_published", subscribe)
        room.on("track_subscribed", subscribed)
        room.on("track_unsubscribed", unsubscribed)

        def disconnected(*_):
            if not self._stop.is_set():
                self.error = RuntimeError(
                    "LiveKit disconnected; restart the video receiver"
                )
            self._stop.set()

        room.on("disconnected", disconnected)
        try:
            await connect(room, self.settings, self.identity, publish=False)
            for participant in room.remote_participants.values():
                for publication in participant.track_publications.values():
                    subscribe(publication, participant)
            self._ready.set()
            await self._stop.wait()
        finally:
            for task in readers.values():
                task.cancel()
            await asyncio.gather(*readers.values(), return_exceptions=True)
            await disconnect(room)
