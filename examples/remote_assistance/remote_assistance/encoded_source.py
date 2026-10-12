# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Version-pinned adapter for LiveKit's native pre-encoded interface."""

import ctypes
from importlib.metadata import version

from livekit import rtc
from livekit.rtc._ffi_client import FfiClient, FfiHandle
from livekit.rtc._proto import ffi_pb2 as ffi
from livekit.rtc._proto import video_frame_pb2 as video


class EncodedSource(rtc.VideoSource):
    def __init__(self, width, height, codec):
        if version("livekit") != "1.1.20":
            raise RuntimeError("Pre-encoded adapter requires livekit==1.1.20")
        self.width, self.height = width, height
        self.codec = video.VideoCodec.Value(codec.upper())
        req = ffi.FfiRequest()
        req.new_video_source.type = video.VIDEO_SOURCE_ENCODED
        req.new_video_source.resolution.width = width
        req.new_video_source.resolution.height = height
        self._info = FfiClient.instance.request(req).new_video_source.source
        self._ffi_handle = FfiHandle(self._info.handle.id)

    def capture_encoded(self, payload, keyframe, timestamp_us, index):
        # Native code copies this buffer synchronously before request() returns.
        data = ctypes.create_string_buffer(payload)
        req = ffi.FfiRequest()
        frame = req.capture_encoded_video_frame
        frame.source_handle = self._ffi_handle.handle
        frame.buffer.data_ptr = ctypes.addressof(data)
        frame.buffer.data_len = len(payload)
        frame.codec = self.codec
        frame.frame_type = (
            video.ENCODED_FRAME_KEY if keyframe else video.ENCODED_FRAME_DELTA
        )
        frame.width, frame.height = self.width, self.height
        frame.timestamp_us = timestamp_us
        frame.metadata.frame_id = index % (2**32)
        frame.metadata.user_timestamp = timestamp_us * 1000
        return FfiClient.instance.request(req).capture_encoded_video_frame.accepted

    def feedback(self):
        req = ffi.FfiRequest()
        req.take_encoded_video_source_feedback.source_handle = self._ffi_handle.handle
        return FfiClient.instance.request(req).take_encoded_video_source_feedback
