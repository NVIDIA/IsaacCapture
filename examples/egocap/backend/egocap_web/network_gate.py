# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Whether CaptureServer's own camera delivery check allows a take to start.

The check belongs to CaptureServer, which reports it in its session status; this only reads the latest value the
capture monitor holds and says what a take must wait for. It decides admission and never ends a take that is
recording: a result ageing out mid-take is not a runtime failure.
"""

from __future__ import annotations

from capture_server import protocol as p
from capture_server.network_check import streams_generation

from .preview_protocol import NativeFacts


def tested_tracks(native: NativeFacts) -> dict[int, str]:
    """The LiveKit track each camera was measured on by the server's current check; empty when none reported."""
    status = native.status.status if native.status is not None else None
    check = status.network_check if status is not None else None
    if check is None:
        return {}
    return {c.stream_id: c.track_id for c in check.cameras if c.track_id is not None}


def network_check_block(native: NativeFacts) -> str | None:
    """Why the server's network check does not allow a take right now, or None when it does.

    None too when no LiveKit camera is offered or the server runs no check: a station opts in by enabling it there,
    so the absence of a verdict is not a refusal. A verdict for different cameras than the ones about to be
    recorded never counts.
    """
    started = native.started
    if started is None or native.stopped is not None:
        return None
    streams = [s for s in started.streams if isinstance(s, p.LiveKitCameraStream)]
    status = native.status.status if native.status is not None else None
    check = status.network_check if status is not None else None
    if not streams or check is None:
        return None
    # A verdict belongs to the capture session it was reached in; an old pass never admits a newer session.
    if check.session_id != native.status.session_id:
        return "the network check is for another capture session; wait for the new one"
    state = check.state
    if state == p.NetworkCheckState.CHECKING:
        return "the network check is still running"
    if state == p.NetworkCheckState.NOT_TESTED:
        return "the network check has not run yet"
    if state == p.NetworkCheckState.STALE:
        return "the last network check is out of date; run it again"
    if state == p.NetworkCheckState.UNSUITABLE:
        return check.reason or "the connection is unsuitable for recording"
    # READY, but only for the cameras and settings it measured.
    if check.generation != streams_generation(streams):
        return "the cameras changed since the network check; run it again"
    return None
