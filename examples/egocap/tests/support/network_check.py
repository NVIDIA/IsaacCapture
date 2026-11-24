# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Builders for the network check status values the egocap backend consumes."""

from __future__ import annotations

from types import SimpleNamespace

from capture_server import protocol as p
from capture_server.client import CaptureStatus, CaptureStatusType
from capture_server.network_check import streams_generation
from egocap_web.preview_protocol import NativeFacts


def livekit_stream(stream_id: int = 1, name: str = "cam0", **overrides) -> p.LiveKitCameraStream:
    fields = {
        "stream_id": stream_id,
        "name": name,
        "url": "wss://lk.example",
        "room": "room",
        "publisher": f"pub{stream_id}",
        "track": f"track{stream_id}",
        "width": 640,
        "height": 480,
        "fps": 30.0,
    }
    fields.update(overrides)
    return p.LiveKitCameraStream(**fields)


def check_status(
    state: p.NetworkCheckState = p.NetworkCheckState.READY,
    *,
    streams: list[p.LiveKitCameraStream] | None = None,
    generation: str | None = None,
    reason: str | None = None,
    session_id: str | None = "s1",
    cameras: tuple[p.NetworkCheckCamera, ...] = (),
) -> p.NetworkCheckStatus:
    if generation is None:
        generation = streams_generation(streams if streams is not None else [livekit_stream()])
    return p.NetworkCheckStatus(
        state=state,
        session_id=session_id,
        generation=generation,
        endpoint="wss://lk.example",
        legs_tested=("publisher-to-server",),
        legs_not_tested=(),
        window_s=10.0,
        valid_for_s=60.0,
        reason=reason,
        cameras=cameras,
    )


def status_snapshot(check: p.NetworkCheckStatus | None) -> p.StatusSnapshot:
    return p.StatusSnapshot(updated_at_ns=123, network_check=check)


def capture_status(snapshot: p.StatusSnapshot | None, session_id: str = "s1") -> CaptureStatus:
    return CaptureStatus(
        session_id=session_id,
        type=CaptureStatusType.STATUS_CHANGE,
        running=True,
        startup_stage=None,
        message=None,
        status=snapshot,
    )


def native_facts(
    check: p.NetworkCheckStatus | None,
    *,
    session_id: str = "s1",
    streams: list | None = None,
    stopped: object | None = None,
) -> NativeFacts:
    """A NativeFacts holding a started session offering ``streams`` and a status carrying ``check``.

    ``started`` is a stand-in: the gate reads only its ``streams``.
    """
    offered = [livekit_stream()] if streams is None else streams
    return NativeFacts(
        started=SimpleNamespace(streams=tuple(offered), session_id="s1", mode=p.CaptureMode.MOCK),
        status=capture_status(status_snapshot(check), session_id),
        stopped=stopped,
    )
