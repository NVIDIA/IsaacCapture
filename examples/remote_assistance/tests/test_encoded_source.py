# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Queue recovery and optional native encoded-video round trips."""

import asyncio
import os
import subprocess
import time
import uuid

import numpy as np
import pytest
from livekit import rtc

from remote_assistance.connection import connect, disconnect
from remote_assistance.encoded_source import EncodedSource
from remote_assistance.gst_camera import AccessUnits, parameters
from remote_assistance.video import VideoReceiver


def test_access_units_recover_after_overflow_and_rejection():
    queue = AccessUnits(limit=2)
    delta = (b"delta", False, 1, 1)
    key = (b"key", True, 2, 2)
    queue.push(delta)
    assert queue.pop() is None
    queue.push(key)
    queue.push(delta)
    assert queue.push(delta)
    assert queue.pop() is None
    queue.push(key)
    assert queue.pop() == key
    queue.push(delta)
    queue.reset()
    queue.push(delta)
    assert queue.pop() is None
    queue.push(key)
    assert queue.pop() == key


def test_reject_unsupported_codec_and_capture():
    spec = dict(type="v4l2", width=1280, height=720, fps=30)
    with pytest.raises(ValueError, match="h264 or h265"):
        parameters({"codec": "av1"}, spec)
    with pytest.raises(ValueError, match="mono"):
        parameters({}, {**spec, "stereo": True})
    with pytest.raises(ValueError, match="dimensions"):
        parameters({}, {**spec, "width": 1279})


@pytest.mark.skipif(not os.getenv("LIVEKIT_TEST_URL"), reason="requires isolated SFU")
@pytest.mark.parametrize("codec", ["h264", "h265"])
def test_encoded_roundtrip_pixels(tmp_path, codec):
    if codec == "h265" and os.getenv("LIVEKIT_TEST_DECODER") != "auto":
        pytest.skip(
            "H.265 requires a separate GPU decoder process with LIVEKIT_TEST_DECODER=auto"
        )
    gi = pytest.importorskip("gi")
    gi.require_version("Gst", "1.0")
    from gi.repository import Gst

    Gst.init(None)
    path = tmp_path / f"test.{codec}"
    encoder = "libx264" if codec == "h264" else "libx265"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "nullsrc=s=320x180:r=30,geq=lum='if(lt(N,30),40,200)':cb=128:cr=128",
            "-frames:v",
            "60",
            "-c:v",
            encoder,
            "-preset",
            "ultrafast",
            "-tune",
            "zerolatency",
            "-g",
            "15",
            "-bf",
            "0",
            "-f",
            "hevc" if codec == "h265" else "h264",
            str(path),
        ],
        check=True,
        capture_output=True,
        timeout=30,
    )
    pipe = Gst.parse_launch(
        f"filesrc name=input ! {codec}parse ! video/x-{codec},stream-format=byte-stream,alignment=au ! appsink name=out sync=false"
    )
    pipe.get_by_name("input").set_property("location", str(path))
    sink = pipe.get_by_name("out")
    units = []
    pipe.set_state(Gst.State.PLAYING)
    try:
        while True:
            sample = sink.emit("try-pull-sample", 2 * Gst.SECOND)
            if sample is None:
                break
            buffer = sample.get_buffer()
            units.append(
                (
                    buffer.extract_dup(0, buffer.get_size()),
                    not buffer.has_flags(Gst.BufferFlags.DELTA_UNIT),
                )
            )
    finally:
        pipe.set_state(Gst.State.NULL)
    assert len(units) == 60

    async def scenario():
        cfg = dict(
            url=os.environ["LIVEKIT_TEST_URL"],
            development=True,
            room=f"encoded-{uuid.uuid4().hex}",
            video_decoder=os.getenv("LIVEKIT_TEST_DECODER", "software"),
        )
        receiver = VideoReceiver(
            cfg, identity="viewer", publisher="camera", track="cam"
        )
        source = EncodedSource(320, 180, codec)
        room = rtc.Room()
        try:
            await asyncio.to_thread(receiver.start)
            await connect(room, cfg, "camera")
            track = rtc.LocalVideoTrack.create_video_track("cam", source)
            opts = rtc.TrackPublishOptions(
                simulcast=False,
                video_codec=rtc.VideoCodec.Value(codec.upper()),
                video_encoding=rtc.VideoEncoding(
                    max_framerate=30, max_bitrate=2_000_000
                ),
                frame_metadata_features=[rtc.FrameMetadataFeature.FMF_FRAME_ID],
            )
            opts.video_encoder = (
                opts.DESCRIPTOR.fields_by_name["video_encoder"]
                .enum_type.values_by_name["ENCODER_BACKEND_PRE_ENCODED"]
                .number
            )
            await room.local_participant.publish_track(track, opts)
            await asyncio.sleep(1)
            levels = set()
            accepted = 0
            for index in range(180):
                payload, key = units[index % 60]
                accepted += source.capture_encoded(
                    payload, key, time.monotonic_ns() // 1000, index
                )
                await asyncio.sleep(1 / 30)
                frame = receiver.latest()
                if frame is not None:
                    assert frame.metadata is not None
                    pixels = np.frombuffer(frame.frame.data, dtype=np.uint8).reshape(
                        180, 320, 4
                    )
                    mean = pixels[:, :, 0].mean()
                    light = frame.metadata.frame_id % 60 >= 30
                    assert mean > 150 if light else mean < 60
                    levels.add(light)
            assert levels == {False, True}, (
                f"accepted={accepted}, received={receiver.frames}"
            )
        finally:
            await disconnect(room)
            await source.aclose()
            await asyncio.to_thread(receiver.stop)

    asyncio.run(scenario())
