# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Live GMSL camera frames from the station's local frame feed (CUDA IPC rings published by the camera
receiver process), served through the same camera adapter as the OAK-D, mock and replay sources."""

from .feed import FeedClient, FeedError, FrameNotice, SlotMapper, SlotReader, StreamInfo, probe_feed

__all__ = ["FeedClient", "FeedError", "FrameNotice", "SlotMapper", "SlotReader", "StreamInfo", "probe_feed"]
