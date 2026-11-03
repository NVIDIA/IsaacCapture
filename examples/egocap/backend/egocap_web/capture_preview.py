# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Latest-sample preview cache: one tracking slot and at most three camera
slots, each holding its immutable callback value and backend-local receipt
time. Replacing a slot drops the superseded value -- there is no history,
replay queue or recording buffer here.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

# Initial application limits for this preview relay, not hardware-capacity
# claims. An oversized sample marks that preview unavailable; it never stops
# capture.
MAX_TRACKING_PAYLOAD_BYTES = 64 * 1024
MAX_CAMERA_JPEG_BYTES = 8 * 1024 * 1024
MAX_CAMERAS = 3


@dataclass(frozen=True)
class TrackingSlot:
    """One immutable latest-tracking-sample value, with its backend-local receipt time."""

    sequence: int
    capture_time_ns: int
    receive_time_ns: int
    payload: bytes
    received_monotonic: float


@dataclass(frozen=True)
class CameraSlot:
    """One immutable latest-camera-image value, with its backend-local receipt time."""

    sequence: int
    capture_time_ns: int
    receive_time_ns: int
    jpeg: bytes
    received_monotonic: float


_TRACKING_OVERSIZE_DIAGNOSTIC = "tracking sample exceeded the preview size limit and was dropped"
_CAMERA_OVERSIZE_DIAGNOSTIC = "camera image exceeded the preview size limit and was dropped"


class LatestPreviewCache:
    """Owned by one CaptureStatusMonitor attempt. Written only from
    CaptureClient callbacks (the owning event loop thread); read from the
    WebSocket/MJPEG routes on the same loop, so no lock is needed.

    An oversized sample does not keep the previous (now stale) slot visible
    as if it were still current: this clears that stream's slot and
    records an explicit diagnostic, so the preview is honestly reported as
    unavailable rather than silently serving old data behind a still-fresh
    ``age_ms``. This is a per-preview-stream condition, not a capture
    failure -- capture itself is unaffected."""

    def __init__(self) -> None:
        """Starts with an empty tracking slot and no camera slots."""
        self._tracking: TrackingSlot | None = None
        self._tracking_diagnostic: str | None = None
        self._cameras: dict[str, CameraSlot] = {}
        self._camera_diagnostics: dict[str, str] = {}

    def clear(self) -> None:
        """Drops every retained slot and diagnostic (e.g. on terminal completion)."""
        self._tracking = None
        self._tracking_diagnostic = None
        self._cameras = {}
        self._camera_diagnostics = {}

    def set_tracking(self, *, sequence: int, capture_time_ns: int, receive_time_ns: int, payload: bytes) -> None:
        """Replaces the tracking slot, or clears it with a diagnostic if oversized."""
        if len(payload) > MAX_TRACKING_PAYLOAD_BYTES:
            self._tracking = None
            self._tracking_diagnostic = _TRACKING_OVERSIZE_DIAGNOSTIC
            return
        self._tracking = TrackingSlot(
            sequence=sequence,
            capture_time_ns=capture_time_ns,
            receive_time_ns=receive_time_ns,
            payload=payload,
            received_monotonic=time.monotonic(),
        )
        self._tracking_diagnostic = None

    def tracking(self) -> TrackingSlot | None:
        """The latest tracking sample, or None if never set/cleared/oversized."""
        return self._tracking

    def tracking_diagnostic(self) -> str | None:
        """Why the tracking slot is empty/stale, if a diagnostic applies."""
        return self._tracking_diagnostic

    def set_camera(
        self, name: str, *, sequence: int, capture_time_ns: int, receive_time_ns: int, jpeg: bytes
    ) -> None:
        """Replaces one named camera's slot, or clears it with a diagnostic if oversized."""
        if len(jpeg) > MAX_CAMERA_JPEG_BYTES:
            self._cameras.pop(name, None)
            self._camera_diagnostics[name] = _CAMERA_OVERSIZE_DIAGNOSTIC
            return
        self._cameras[name] = CameraSlot(
            sequence=sequence,
            capture_time_ns=capture_time_ns,
            receive_time_ns=receive_time_ns,
            jpeg=jpeg,
            received_monotonic=time.monotonic(),
        )
        self._camera_diagnostics.pop(name, None)

    def camera(self, name: str) -> CameraSlot | None:
        """The latest image for one named camera, or None if never set/cleared/oversized."""
        return self._cameras.get(name)

    def camera_diagnostic(self, name: str) -> str | None:
        """Why one named camera's slot is empty/stale, if a diagnostic applies."""
        return self._camera_diagnostics.get(name)
