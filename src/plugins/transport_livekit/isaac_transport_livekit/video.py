# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A bounded video receiver, usable without CUDA or an XR runtime."""

import asyncio
import logging
import os
import threading
import time

from livekit import rtc

from .connection import connect, disconnect
from isaac_media_transport.stereo import StereoPairer
from isaac_media_transport.latency import get_recorder
from isaac_media_transport.transport import (
    FrameMetadata,
    VideoFrame,
    ReceivedFrame,
    ReceivedPair,
)

logger = logging.getLogger("isaaccapture.remote_assistance.video")


class VideoReceiver:
    """Receive a named mono track or matched stereo pair into a latest mailbox."""

    def __init__(
        self, settings: dict, *, identity: str, publisher: str, track, pairing=None
    ):
        decoder = settings.get("video_decoder", "auto")
        if decoder not in ("auto", "software", "hardware"):
            raise ValueError("video_decoder must be auto, software or hardware")
        if decoder == "hardware" and "LK_DISABLE_NVDEC" in os.environ:
            raise ValueError(
                "Hardware decoding requires a fresh viewer process with "
                "LK_DISABLE_NVDEC unset"
            )
        self.require_nvdec = decoder == "hardware"
        if decoder == "software":
            # SDK-wide option; set before creating media peer connections. Keep CUDA
            # available to Televiz: CUDA_VISIBLE_DEVICES would disable rendering too.
            os.environ["LK_DISABLE_NVDEC"] = "1"
        self.settings = settings
        self.timing = get_recorder(settings)
        self.identity = identity
        self.publisher = publisher
        self.track = track
        self.tracks = (track,) if isinstance(track, str) else tuple(track)
        if len(self.tracks) not in (1, 2) or len(set(self.tracks)) != len(self.tracks):
            raise ValueError("Receive one track or two distinct stereo tracks")
        self._pairer = StereoPairer(pairing) if len(self.tracks) == 2 else None
        self.frames = 0
        self.overwritten = 0
        self.error = None
        self._latest = None
        self._lock = threading.Lock()
        self._ready = threading.Event()
        self._thread = None
        self._loop = None
        self._stop = None
        self._pairing_changed = None
        self._capture_endpoint = None

    def _verify_decoder(self, stats, name):
        active = [
            item.inbound_rtp.inbound
            for item in stats
            if item.HasField("inbound_rtp")
            and item.inbound_rtp.inbound.frames_decoded > 0
        ]
        for inbound in active:
            implementation = inbound.decoder_implementation
            if implementation not in ("NVIDIA H264 Decoder", "NVIDIA H265 Decoder"):
                raise RuntimeError(
                    f"{name}: NVDEC required, but LiveKit reports decoder "
                    f"{implementation!r}; refusing an unverified measurement"
                )
            logger.info(
                "LiveKit NVDEC verified: track=%s decoder=%s frames_decoded=%d "
                "total_decode_seconds=%.6f; output=CPU RGBA",
                name,
                implementation,
                inbound.frames_decoded,
                inbound.total_decode_time,
            )
        return bool(active)

    async def _monitor_decoder(self, track, name, verified):
        deadline = time.monotonic() + 15
        while not self._stop.is_set():
            async with asyncio.timeout(5):
                stats = await track.get_stats()
            if self._verify_decoder(stats, name):
                verified.set()
                deadline = time.monotonic() + 15
            elif time.monotonic() >= deadline:
                raise RuntimeError(
                    f"{name}: no decoded NVDEC statistics within 15 seconds"
                )
            await asyncio.sleep(5 if verified.is_set() else 0.5)

    def _make_stream(self, track, name):
        return rtc.VideoStream(track, capacity=1, format=rtc.VideoBufferType.RGBA)

    def _received_frame(self, event):
        return ReceivedFrame(
            VideoFrame(event.frame.width, event.frame.height, event.frame.data),
            time.monotonic_ns(),
            FrameMetadata(event.metadata.frame_id, event.metadata.user_timestamp)
            if event.metadata is not None
            else None,
        )

    def latest(self):
        if self.error:
            raise RuntimeError("LiveKit video receiver failed") from self.error
        with self._lock:
            result, self._latest = self._latest, None
        return result

    def pairing_stats(self):
        with self._lock:
            return self._pairer.stats() if self._pairer else {}

    def _publish_pair_locked(self):
        pair = self._pairer.pop()
        if pair:
            for name, frame in zip(self.tracks, pair[:2]):
                if frame.metadata:
                    self.timing.record(
                        "pair_ready", name, frame.metadata.user_timestamp, at_ns=pair[2]
                    )
            self.overwritten += self._latest is not None
            self._latest = ReceivedPair(*pair)
            self.frames += 1

    def _accept_frame(self, name, frame):
        with self._lock:
            if self._pairer:
                self._pairer.push(self.tracks.index(name), frame)
                # Publish on the second eye's arrival, without a polling-timer hop.
                # Keep the pairer's original age/skew checks and frame ownership.
                self._publish_pair_locked()
            else:
                self.overwritten += self._latest is not None
                self._latest = frame
                self.frames += 1
        if self._pairer and self._pairing_changed is not None:
            self._pairing_changed.set()

    async def _coordinate(self):
        while not self._stop.is_set():
            self._pairing_changed.clear()
            with self._lock:
                self._publish_pair_locked()
                self._pairer.report(" / ".join(self.tracks))
                now = self._pairer.clock()
                deadlines = [self._pairer.last_report + 5_000_000_000]
                deadlines.extend(
                    queue[0][1].received_ns + self._pairer.max_wait_ms * 1e6 + 1
                    for queue in self._pairer.queues
                    if queue
                )
                timeout = max(0, (min(deadlines) - now) / 1e9)
            # Arrival and expiry run on the same event loop; no wakeup is lost
            # between clearing the event and awaiting it.
            try:
                async with asyncio.timeout(timeout):
                    await self._pairing_changed.wait()
            except TimeoutError:
                pass  # Expire unmatched frames even when a track stops sending.

    def start(self):
        if self._thread:
            return
        if self.settings.get("capture_socket"):
            from isaac_media_transport.capture import CaptureEndpoint

            self._capture_endpoint = CaptureEndpoint(self.settings["capture_socket"])
            for name in self.tracks:
                self._capture_endpoint.register(
                    name,
                    stage="receiver-decoded-before-resize-and-xr",
                    width=0,
                    height=0,
                    fps=float(self.settings.get("capture_fps", 30)),
                )
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
        if self._capture_endpoint:
            self._capture_endpoint.close()
            self._capture_endpoint = None
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
        self._pairing_changed = asyncio.Event()
        room = rtc.Room()
        readers = {}
        coordinator = None

        def subscribe(publication, participant):
            if (
                participant.identity == self.publisher
                and publication.name in self.tracks
            ):
                publication.set_subscribed(True)

        async def read(track, name):
            stream = self._make_stream(track, name)
            verified = asyncio.Event()
            monitor = None
            if self.require_nvdec:
                monitor = asyncio.create_task(
                    self._monitor_decoder(track, name, verified)
                )

                def monitor_finished(task):
                    if not task.cancelled() and task.exception():
                        self.error = task.exception()
                        logger.error(
                            "LiveKit hardware decoder verification failed: %s",
                            self.error,
                        )
                        self._stop.set()

                monitor.add_done_callback(monitor_finished)
            try:
                async for event in stream:
                    # Do not expose unverified startup frames to the measurement viewer.
                    if self.require_nvdec and not verified.is_set():
                        continue
                    if self._capture_endpoint:
                        meta = event.metadata
                        self._capture_endpoint.offer(
                            name,
                            event.frame.data.tobytes,
                            dict(
                                width=event.frame.width,
                                height=event.frame.height,
                                frame_id=meta.frame_id if meta is not None else None,
                                source_timestamp_ns=meta.user_timestamp
                                if meta is not None
                                else None,
                                received_monotonic_ns=time.monotonic_ns(),
                            ),
                        )
                    self._accept_frame(name, self._received_frame(event))
            finally:
                if monitor:
                    monitor.cancel()
                    await asyncio.gather(monitor, return_exceptions=True)
                await stream.aclose()

        def subscribed(track, publication, participant):
            if (
                participant.identity != self.publisher
                or publication.name not in self.tracks
            ):
                return
            if track.kind != rtc.TrackKind.KIND_VIDEO:
                return
            logger.info("Receiving %s / %s", participant.identity, publication.name)
            task = asyncio.create_task(read(track, publication.name))
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
            if (
                participant.identity != self.publisher
                or publication.name not in self.tracks
            ):
                return
            task = readers.pop(publication.sid, None)
            if task:
                task.cancel()
            with self._lock:
                self._latest = None
                if self._pairer:
                    self._pairer.reset()

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
            if self._pairer:
                logger.info(
                    "Stereo receiver pairing %s with bounded timestamp matching; no geometric rectification",
                    self.tracks,
                )
                coordinator = asyncio.create_task(self._coordinate())

                def coordination_finished(task):
                    if not task.cancelled() and task.exception():
                        self.error = task.exception()
                        self._stop.set()

                coordinator.add_done_callback(coordination_finished)
            self._ready.set()
            await self._stop.wait()
        finally:
            if coordinator:
                coordinator.cancel()
                await asyncio.gather(coordinator, return_exceptions=True)
            if self._pairer:
                with self._lock:
                    self._pairer.report(" / ".join(self.tracks), force=True)
                    self._pairer.reset()
            for task in readers.values():
                task.cancel()
            await asyncio.gather(*readers.values(), return_exceptions=True)
            await disconnect(room)
