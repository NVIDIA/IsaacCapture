# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Transport contracts and explicit backend selection; no native SDK imports."""

import asyncio
from dataclasses import dataclass
from typing import Callable, Protocol

MAX_MESSAGE = 128 * 1024


@dataclass(frozen=True)
class Capabilities:
    name: str
    encoded_codecs: tuple[str, ...]
    raw_codecs: tuple[str, ...]
    max_message_bytes: int
    topology: str
    decoded_format: str = "rgba"
    decoded_memory: str = "cpu"
    audio: bool = False
    decoded_memory_options: tuple[str, ...] = ("cpu",)


@dataclass(frozen=True)
class FrameMetadata:
    frame_id: int
    user_timestamp: int


@dataclass(frozen=True)
class VideoFrame:
    width: int
    height: int
    # CUDA data owns an immutable RGBA8 allocation; producer work is complete.
    # Retaining this object keeps pixels alive. CPU data remains a memoryview.
    data: object
    memory: str = "cpu"
    device_id: int | None = None

    def to_bytes(self):
        """Explicit diagnostic readback, never used by the normal display path."""
        if self.memory == "cuda":
            import cupy as cp

            with cp.cuda.Device(self.device_id):
                return cp.asnumpy(self.data).tobytes()
        return self.data.tobytes()


@dataclass
class ReceivedFrame:
    frame: VideoFrame
    received_ns: int
    metadata: FrameMetadata | None


@dataclass
class ReceivedPair:
    left: ReceivedFrame
    right: ReceivedFrame
    paired_ns: int

    @property
    def received_ns(self):
        return min(self.left.received_ns, self.right.received_ns)

    @property
    def frame(self):
        return self.left.frame

    @property
    def metadata(self):
        return self.left.metadata


@dataclass(frozen=True)
class RateControl:
    target_bitrate_bps: int
    framerate_fps: float


@dataclass(frozen=True)
class EncoderFeedback:
    keyframe_requested: bool = False
    rate_control: RateControl | None = None


class MessageSession(Protocol):
    closed: asyncio.Event
    error: Exception | None
    stats: dict

    @property
    def ready(self) -> bool: ...
    async def start(self) -> None: ...
    async def send(
        self, topic: str, payload: bytes, *, reliable: bool = False
    ) -> bool: ...
    async def stop(self) -> None: ...


class VideoPublication(Protocol):
    def capture_encoded(
        self, payload: bytes, keyframe: bool, timestamp_us: int, index: int
    ) -> bool: ...
    def capture_rgba(self, rgba, captured_ns: int, index: int) -> None: ...
    def feedback(self) -> EncoderFeedback: ...
    async def close(self) -> None: ...


class VideoSession(Protocol):
    async def start(self) -> None: ...
    async def publish(self, spec: dict, *, encoded: bool) -> VideoPublication: ...
    async def close(self) -> None: ...


class FrameReceiver(Protocol):
    overwritten: int

    def start(self) -> None: ...
    def latest(self) -> ReceivedFrame | ReceivedPair | None: ...
    def pairing_stats(self) -> dict: ...
    def stop(self) -> None: ...


class TransportProvisioner(Protocol):
    def generate(
        self,
        output,
        host_ip,
        session_template,
        camera_template,
        *,
        write_camera_profiles: Callable | None = None,
        **options,
    ): ...


class TransportAdapter(Protocol):
    capabilities: Capabilities

    def messages(
        self, settings: dict, identity: str, peer: str, topics, on_message: Callable
    ) -> MessageSession: ...
    def video_sender(self, settings: dict, stop: asyncio.Event) -> VideoSession: ...
    def video_receiver(self, settings: dict, **selection) -> FrameReceiver: ...
    def provisioner(self) -> TransportProvisioner: ...


def get_transport(settings=None) -> TransportAdapter:
    from .transport_plugins import load_transport

    name = (settings or {}).get("transport", "livekit")
    return load_transport(name)


def create_receiver(settings, **selection) -> FrameReceiver:
    """Choose local handoff independently of the network transport."""
    handoff = settings.get("decoded_handoff", "inprocess")
    if handoff == "ipc":
        from .gpu_ipc import IpcReceiver

        return IpcReceiver(settings, **selection)
    if handoff != "inprocess":
        raise ValueError("decoded_handoff must be inprocess or ipc")
    return get_transport(settings).video_receiver(settings, **selection)
