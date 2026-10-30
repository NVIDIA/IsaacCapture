# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Replay acquisition: play one recorded episode (Zarr tracking tables plus
MP4 camera files) back through the SAME production adapters and wire
protocol the mock and real sources use, with no hardware and no
IsaacCapture checkout.

This is the third source kind next to ``mock`` (synthetic motion) and
real acquisition: like the mock it feeds SDK-shaped objects into
:class:`~capture_server.acquisition.isaac_capture.IsaacCaptureAdapter`
and :class:`~capture_server.acquisition.depthai.DepthAIAdapter`, so
nothing downstream of :mod:`capture_server.launch` knows it exists.
Unlike the mock, every value it serves was recorded from real devices;
what it cannot derive from the recording it marks invalid rather than
inventing (see :mod:`capture_server.replay.tracking_map`).

Only :mod:`capture_server.launch` and the CLI import this package, and
only when replay is selected -- choosing mock or real acquisition never
loads it, and importing it never opens a file or a socket.
"""

from __future__ import annotations

from .episode import CameraTrack, Episode, EpisodeFormatError, TrackingTable, load_episode
from .playback import PlaybackClock

__all__ = [
    "CameraTrack",
    "Episode",
    "EpisodeFormatError",
    "PlaybackClock",
    "TrackingTable",
    "load_episode",
]
