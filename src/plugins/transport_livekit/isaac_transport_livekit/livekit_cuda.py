# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Opt-in LiveKit access-unit reception followed by GPU-only NVDEC decoding."""

import asyncio
from collections import OrderedDict
import ctypes
from importlib.metadata import version
import logging
import os
import threading
import time
from pathlib import Path
from types import SimpleNamespace

from google.protobuf import descriptor_pb2, descriptor_pool, message_factory
from livekit.rtc._ffi_client import FfiClient, FfiHandle
from livekit.rtc._proto import ffi_pb2

from isaac_media_transport.gpu_decode import CudaDecoder
from .video import VideoReceiver
from isaac_media_transport.latency import get_recorder
from .sdk_diagnostics import sample_stats

logger = logging.getLogger("isaaccapture.remote_assistance.livekit_cuda")


def encoded_event_type():
    # The extension adds field 4 without replacing the installed SDK's protobufs.
    file = descriptor_pb2.FileDescriptorProto(
        name="umb_encoded.proto", package="umb", syntax="proto2"
    )
    for name, fields in (
        ("Metadata", [("user_timestamp", 1, 4, None), ("frame_id", 2, 13, None)]),
        (
            "Frame",
            [
                ("payload", 1, 12, None),
                ("keyframe", 2, 8, None),
                ("metadata", 3, 11, ".umb.Metadata"),
                ("native_available_ns", 5, 4, None),
            ],
        ),
        ("Event", [("stream_handle", 1, 4, None), ("encoded", 4, 11, ".umb.Frame")]),
    ):
        msg = file.message_type.add(name=name)
        for field_name, number, kind, type_name in fields:
            field = msg.field.add(name=field_name, number=number, type=kind, label=1)
            if type_name:
                field.type_name = type_name
    pool = descriptor_pool.DescriptorPool()
    pool.Add(file)
    return message_factory.GetMessageClass(pool.FindMessageTypeByName("umb.Event"))


EncodedEvent = encoded_event_type()


def configure_native(settings):
    playout = settings.get("livekit_playout_mode", "default")
    if playout not in ("default", "immediate"):
        raise ValueError("livekit_playout_mode must be default or immediate")
    if version("livekit") != "1.1.20":
        raise RuntimeError("LiveKit CUDA receive adapter requires livekit==1.1.20")
    path = Path(settings.get("livekit_cuda_library", "")).expanduser()
    if not path.is_file():
        raise ValueError("Set livekit_cuda_library to the built UMB LiveKit extension")
    if FfiClient._instance is not None:
        raise RuntimeError("LiveKit CUDA receive requires a fresh receiver process")
    library = ctypes.CDLL(str(path.resolve()))
    try:
        abi = library.livekit_umb_encoded_receive_abi
    except AttributeError as exc:
        raise RuntimeError(
            "Native library lacks the UMB encoded receive extension"
        ) from exc
    abi.restype = ctypes.c_uint32
    if abi() != 1:
        raise RuntimeError("Unsupported UMB encoded receive ABI")
    if playout == "immediate":
        try:
            enable = library.livekit_umb_enable_zero_playout_delay
        except AttributeError as exc:
            raise RuntimeError(
                "Immediate playout requires a native library built with --playout-control"
            ) from exc
        enable.argtypes = []
        enable.restype = ctypes.c_int32
        if enable() != 0:
            raise RuntimeError(
                "Cannot enable immediate playout after WebRTC runtime initialization; "
                "start a fresh receiver process"
            )
    os.environ["LIVEKIT_LIB_PATH"] = str(path.resolve())
    os.environ["UMB_LIVEKIT_ENCODED_RECEIVE"] = "1"
    logger.info("LiveKit native receiver playout mode: %s", playout)


class CudaStream:
    def __init__(self, track, name, settings):
        self.name = name
        self.timing = get_recorder(settings)
        self.decoder = CudaDecoder({**settings, "latency_track": name})
        self.waiting = True
        self.previous = None
        self.frames = 0
        self.packets = asyncio.Queue(maxsize=64)
        self.handle = None
        self.closed = False
        self.decode_task = None
        self._callback_times = OrderedDict()
        self._callback_lock = threading.Lock()
        self.ffi = FfiClient.instance
        self.events = self.ffi.queue.subscribe(
            asyncio.get_running_loop(),
            filter_fn=self._filter_event,
        )
        try:
            req = ffi_pb2.FfiRequest()
            req.new_video_stream.track_handle = track._ffi_handle.handle
            req.new_video_stream.type = 0
            req.new_video_stream.queue_size_frames = 64
            self.handle = FfiHandle(
                self.ffi.request(req).new_video_stream.stream.handle.id
            )
        except BaseException:
            self.ffi.queue.unsubscribe(self.events)
            raise
        self.pump = asyncio.create_task(self._pump())
        self.stats_task = (
            asyncio.create_task(sample_stats(track, name, self.timing))
            if self.timing.enabled
            and settings.get("latency", {}).get("sdk_stats", False)
            else None
        )

    def _filter_event(self, event):
        matches = (
            event.HasField("video_stream_event")
            and self.handle is not None
            and event.video_stream_event.stream_handle == self.handle.handle
        )
        if matches and self.timing.enabled:
            # SDK 1.1.20 invokes this on its FFI callback thread, before scheduling
            # queue delivery. Retain the event to prevent id reuse after overflow.
            at = time.monotonic_ns()
            with self._callback_lock:
                self._callback_times[id(event)] = (event, at)
                if len(self._callback_times) > 128:
                    self._callback_times.popitem(last=False)
        return matches

    async def _pump(self):
        try:
            while True:
                ffi_event = await self.events.get()
                available_ns = time.monotonic_ns() if self.timing.enabled else None
                callback_ns = None
                if self.timing.enabled:
                    with self._callback_lock:
                        saved = self._callback_times.pop(id(ffi_event), None)
                    if saved is not None:
                        callback_ns = saved[1]
                event = ffi_event.video_stream_event
                if event.HasField("eos"):
                    raise RuntimeError(f"{self.name}: encoded stream ended")
                if event.HasField("frame_received"):
                    FfiHandle(event.frame_received.buffer.handle.id).dispose()
                    raise RuntimeError(
                        "LiveKit delivered a CPU frame; GPU receive must not fall back"
                    )
                packet = EncodedEvent.FromString(event.SerializeToString())
                if not packet.HasField("encoded"):
                    continue
                meta = packet.encoded.metadata
                if meta.HasField("user_timestamp"):
                    if callback_ns is not None:
                        self.timing.record(
                            "sdk_callback",
                            self.name,
                            meta.user_timestamp,
                            at_ns=callback_ns,
                        )
                    if packet.encoded.HasField("native_available_ns"):
                        self.timing.record(
                            "sdk_native_available",
                            self.name,
                            meta.user_timestamp,
                            at_ns=packet.encoded.native_available_ns,
                        )
                    self.timing.record(
                        "transport_available",
                        self.name,
                        meta.user_timestamp,
                        at_ns=available_ns,
                        details={
                            "frame_id": meta.frame_id,
                            "keyframe": packet.encoded.keyframe,
                            "queued_packets": self.packets.qsize(),
                        },
                    )
                if self.packets.full():
                    while not self.packets.empty():
                        self.packets.get_nowait()
                    logger.warning(
                        "LiveKit encoded queue overflow: %s; recovering at next IDR",
                        self.name,
                    )
                self.packets.put_nowait(packet.encoded)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            while not self.packets.empty():
                self.packets.get_nowait()
            self.packets.put_nowait(exc)

    def __aiter__(self):
        return self

    async def __anext__(self):
        while not self.closed:
            async with asyncio.timeout(15):
                packet = await self.packets.get()
            if isinstance(packet, Exception):
                raise packet
            meta = packet.metadata
            if self.waiting and not meta.HasField("frame_id"):
                continue
            if not meta.HasField("frame_id") or not meta.HasField("user_timestamp"):
                raise RuntimeError(
                    "GPU receive requires source frame IDs and timestamps"
                )
            index = meta.frame_id
            if self.previous is None or index != (self.previous + 1) % 2**32:
                self.waiting = True
                self.decoder.reset()
            self.previous = index
            if self.waiting:
                if not packet.keyframe:
                    continue
                self.waiting = False
            # Only compressed bytes cross the CPU boundary; decoded pixels stay CUDA.
            self.decode_task = asyncio.create_task(
                asyncio.to_thread(
                    self.decoder.decode,
                    packet.payload,
                    meta.user_timestamp // 1000,
                    index,
                )
            )
            frames = await asyncio.shield(self.decode_task)
            self.decode_task = None
            frame = frames[0]
            self.frames += 1
            if self.frames == 1 or self.frames % 150 == 0:
                logger.info(
                    "LiveKit GPU receive: track=%s decoded=%d output=CUDA RGBA; no decoded CPU staging",
                    self.name,
                    self.frames,
                )
            return SimpleNamespace(
                frame=frame.frame, metadata=frame.metadata, received=frame
            )
        raise StopAsyncIteration

    async def aclose(self):
        self.closed = True
        if self.stats_task:
            self.stats_task.cancel()
            await asyncio.gather(self.stats_task, return_exceptions=True)
        self.ffi.queue.unsubscribe(self.events)
        with self._callback_lock:
            self._callback_times.clear()
        self.handle.dispose()
        self.pump.cancel()
        await asyncio.gather(self.pump, return_exceptions=True)
        if self.decode_task:
            await asyncio.gather(self.decode_task, return_exceptions=True)
        self.decoder.reset()


class CudaVideoReceiver(VideoReceiver):
    def __init__(self, settings, **selection):
        if settings.get("codec", "h264").lower() != "h264":
            raise ValueError("LiveKit CUDA receive currently supports H.264 only")
        if settings.get("video_decoder") == "software":
            raise ValueError("CUDA receive requires hardware decoding")
        if settings.get("capture_socket"):
            raise ValueError(
                "Disable capture_socket for the LiveKit GPU-only measurement"
            )
        configure_native(settings)
        super().__init__({**settings, "video_decoder": "auto"}, **selection)

    def _make_stream(self, track, name):
        return CudaStream(track, name, self.settings)

    def _received_frame(self, event):
        return event.received
