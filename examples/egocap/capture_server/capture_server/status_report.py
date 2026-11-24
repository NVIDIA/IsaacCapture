# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Prints CaptureServer's general status: ``python -m capture_server.status_report``.

Covers the server, its current session, the sources, devices and cameras it reports and, when the server runs
one, the camera delivery check. It reads the read-only query route, so it needs no capture session and joins no
media; the check being off or unavailable never hides the rest. The exit status is 0 when the server answered,
1 when it did not.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time

from . import protocol as p
from .client import CaptureClient, ServerStatusQueryError

_CHECK_STATES = {
    p.NetworkCheckState.NOT_TESTED: "not tested",
    p.NetworkCheckState.CHECKING: "checking",
    p.NetworkCheckState.READY: "passed",
    p.NetworkCheckState.UNSUITABLE: "failed",
    p.NetworkCheckState.STALE: "expired; run again before recording",
}


def _ms(value: float | None) -> str:
    """A measurement in milliseconds; unknown is not zero."""
    return "unknown" if value is None else f"{value:.0f} ms"


def _silence_ms(camera: p.NetworkCheckCamera) -> float | None:
    """The longest silence anywhere in the window: between arrivals and at its start and end."""
    known = [
        v for v in (camera.max_arrival_gap_ms, camera.start_silence_ms, camera.end_silence_ms) if v is not None
    ]
    return max(known) if known else None


def _network_lines(check: p.NetworkCheckStatus) -> list[str]:
    """The network check as readable lines: verdict, per-camera measurements, and what was and was not tested."""
    lines = [f"network check: {_CHECK_STATES[check.state]}"]
    if check.phase is not None:
        phase = (
            f"measuring, {check.progress_s or 0} of {check.window_s:g} seconds"
            if check.phase == "measuring"
            else "connecting to camera streams"
        )
        lines.append(f"  progress: {phase}")
    if check.reason is not None:
        lines.append(f"  reason: {check.reason}")
    if check.completed_at_unix_ns is not None:
        age = max(0.0, time.time() - check.completed_at_unix_ns / 1e9)
        lines.append(
            f"  test completed: {age:.0f} seconds ago; results expire after {check.valid_for_s:g} seconds"
        )
    lines.append(f"  media endpoint: {check.endpoint}")
    for camera in check.cameras:
        verdict = {None: "not measured", True: "passed", False: "failed"}[camera.passed]
        fps = "unknown" if camera.delivered_fps is None else f"{camera.delivered_fps:.1f}"
        missing = "unknown" if camera.missing_frames is None else camera.missing_frames
        lines.append(
            f"  {camera.name}: {verdict}; delivery rate: {fps} fps (configured: {camera.expected_fps:g} fps); "
            f"usable frames received: {camera.usable_frames if camera.usable_frames is not None else 'unknown'}; "
            f"missing frames: {missing}; "
            f"longest gap without usable frames: {_ms(_silence_ms(camera))}; "
            f"capture-to-receive delay: median {_ms(camera.age_median_ms)}, maximum {_ms(camera.age_max_ms)}"
        )
        lines.extend(f"    - {reason}" for reason in camera.reasons)
    lines.append("  measured path: " + ", ".join(check.legs_tested))
    lines.append("  outside this test: " + "; ".join(check.legs_not_tested))
    return lines


def _camera_line(camera: p.ServerCamera) -> str:
    """One camera: its source, configured geometry, whether the session has it, and its observed state."""
    if camera.transport == "livekit":
        source = f"LiveKit room {camera.room}, track {camera.track}"
    else:
        source = f"CaptureServer camera {camera.device_id}" if camera.device_id else "CaptureServer camera"
    size = f", {camera.width}x{camera.height}" if camera.width and camera.height else ""
    rate = f" @ {camera.fps:g} fps" if camera.fps else ""
    state = camera.state.name.lower() if camera.state is not None else "not observed"
    error = f" ({camera.error})" if camera.error else ""
    scope = "in session" if camera.selected else "configured"
    return f"camera {camera.name}: {source}{size}{rate}; {scope}; {state}{error}"


def format_report(reply: p.ServerStatusReply) -> str:
    """The whole status as readable text."""
    lines = [f"capture server: {reply.mode.name.lower()} mode"]
    lines.append(f"session: {reply.session_id}" if reply.session_id is not None else "session: none running")
    # The inventory already carries each camera's observed state, so session cameras are not listed a second time.
    for camera in reply.cameras:
        lines.append(_camera_line(camera))
    status = reply.status
    if status is not None:
        lines.extend(
            f"provider {x.name}: {x.status.name.lower()}" + (f" ({x.error})" if x.error else "")
            for x in status.providers
        )
        lines.extend(
            f"device {x.description or x.id}: {x.status.name.lower()}" + (f" ({x.error})" if x.error else "")
            for x in status.devices
        )
    lines.extend(
        _network_lines(reply.network_check)
        if reply.network_check is not None
        else ["network check: not configured"]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Query the server and print its status; returns the process exit status."""
    parser = argparse.ArgumentParser(prog="python -m capture_server.status_report", description=__doc__)
    parser.add_argument("--url", default="ws://127.0.0.1:8082", help="the CaptureServer WebSocket URL")
    parser.add_argument("--json", action="store_true", help="print the status as JSON")
    args = parser.parse_args(argv)
    try:
        reply = asyncio.run(CaptureClient(args.url).query_status())
    except ServerStatusQueryError as exc:
        print(f"status_report: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(reply.to_json(), indent=2) if args.json else format_report(reply))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by running the module, not pytest
    raise SystemExit(main())
