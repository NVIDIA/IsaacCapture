# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""LiveKit binding for the transport contracts and legacy configuration."""

from livekit import rtc

from .connection import connect, disconnect
from isaac_media_transport.transport import (
    Capabilities,
    EncoderFeedback,
    MAX_MESSAGE,
    RateControl,
)


class LiveKitTransport:
    capabilities = Capabilities(
        name="livekit",
        encoded_codecs=("h264", "h265"),
        raw_codecs=("h264", "h265", "vp8", "vp9", "av1"),
        max_message_bytes=MAX_MESSAGE,
        topology="sfu",
        decoded_memory_options=("cpu", "cuda"),
    )

    def messages(self, settings, identity, peer, topics, on_message):
        from .data import DataSession

        return DataSession(settings, identity, peer, topics, on_message)

    def video_sender(self, settings, stop):
        return LiveKitVideoSession(settings, stop)

    def video_receiver(self, settings, **selection):
        if settings.get("decoded_memory", "cpu") == "cuda":
            from .livekit_cuda import CudaVideoReceiver

            return CudaVideoReceiver(settings, **selection)
        if settings.get("decoded_memory", "cpu") != "cpu":
            raise ValueError("decoded_memory must be cpu or cuda")
        from .video import VideoReceiver

        return VideoReceiver(settings, **selection)

    def provisioner(self):
        return LiveKitProvisioner()


class LiveKitProvisioner:
    def generate(self, output, host_ip, session_template, camera_template, **options):
        from .network_setup import _generate_bundle

        return _generate_bundle(
            output, host_ip, session_template, camera_template, **options
        )


class LiveKitVideoSession:
    def __init__(self, settings, stop):
        self.settings = settings
        self.room = rtc.Room()
        self.room.on("disconnected", lambda *_: stop.set())

    async def start(self):
        await connect(
            self.room, self.settings, self.settings.get("publisher", "client-camera")
        )

    async def publish(self, spec, *, encoded):
        from .encoded_source import EncodedSource

        width, height = int(spec["width"]), int(spec["height"])
        codec = str(self.settings.get("codec", "h264")).upper()
        backend = (
            "PRE_ENCODED"
            if encoded
            else str(self.settings.get("video_encoder", "auto")).upper()
        )
        # Validate before allocating native handles; SDK/platform support is checked on publish.
        codec_value = rtc.VideoCodec.Value(codec)
        options = rtc.TrackPublishOptions(
            source=rtc.TrackSource.SOURCE_CAMERA,
            simulcast=False,
            video_codec=codec_value,
            video_encoding=rtc.VideoEncoding(
                max_framerate=float(spec["fps"]),
                max_bitrate=int(
                    float(self.settings.get("bitrate_mbps", 15)) * 1_000_000
                ),
            ),
            frame_metadata_features=[
                rtc.FrameMetadataFeature.FMF_FRAME_ID,
                rtc.FrameMetadataFeature.FMF_USER_TIMESTAMP,
            ],
        )
        options.video_encoder = (
            options.DESCRIPTOR.fields_by_name["video_encoder"]
            .enum_type.values_by_name["ENCODER_BACKEND_" + backend]
            .number
        )
        source = (
            EncodedSource(
                width, height, codec, settings=self.settings, track=spec["name"]
            )
            if encoded
            else rtc.VideoSource(width, height)
        )
        try:
            track = rtc.LocalVideoTrack.create_video_track(spec["name"], source)
            await self.room.local_participant.publish_track(track, options)
        except BaseException:
            await source.aclose()
            raise
        return LiveKitVideoPublication(self.room, track, source, settings=self.settings)

    async def close(self):
        await disconnect(self.room)


class LiveKitVideoPublication:
    def __init__(self, room, track, source, *, settings=None):
        self.room, self.track, self.source = room, track, source
        self.stats_task = None
        if (settings or {}).get("latency", {}).get("sdk_stats", False):
            import asyncio
            from isaac_media_transport.latency import get_recorder
            from .sdk_diagnostics import sample_stats

            timing = get_recorder(settings)
            if timing.enabled:
                self.stats_task = asyncio.create_task(
                    sample_stats(track, track.name, timing)
                )

    def capture_encoded(self, payload, keyframe, timestamp_us, index):
        return self.source.capture_encoded(payload, keyframe, timestamp_us, index)

    def capture_rgba(self, rgba, captured_ns, index):
        self.source.capture_frame(
            rtc.VideoFrame(
                rgba.shape[1], rgba.shape[0], rtc.VideoBufferType.RGBA, memoryview(rgba)
            ),
            timestamp_us=captured_ns // 1000,
            metadata=rtc.FrameMetadata(
                frame_id=index % (2**32), user_timestamp=captured_ns
            ),
        )

    def feedback(self):
        value = self.source.feedback()
        rate = value.rate_control if value.HasField("rate_control") else None
        return EncoderFeedback(
            keyframe_requested=value.keyframe_requested,
            rate_control=RateControl(rate.target_bitrate_bps, rate.framerate_fps)
            if rate is not None
            else None,
        )

    async def close(self):
        if self.stats_task:
            import asyncio

            self.stats_task.cancel()
            await asyncio.gather(self.stats_task, return_exceptions=True)
        try:
            if self.room.isconnected():
                await self.room.local_participant.unpublish_track(self.track.sid)
        finally:
            await self.source.aclose()
