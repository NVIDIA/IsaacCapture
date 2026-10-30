"""WS /api/session snapshot adapter and the 32-byte browser binary tracking
header. This is a browser-facing envelope distinct from CaptureServer's own
wire protocol: SNAPSHOT=3/version=1 is a capture-preview projection of
native CaptureStarted/CaptureStatus/CaptureStopped facts, not a new hardware
schema, and the binary header's bytes are not the upstream GTC1 header.
"""

from __future__ import annotations

import os
import struct
from dataclasses import dataclass

from capture_server import protocol as p
from capture_server.client import CaptureStarted, CaptureStatus, CaptureStopped

from .capture_preview import CameraSlot, TrackingSlot


@dataclass(frozen=True)
class NativeFacts:
    """The one owner value for every native CaptureClient callback fact --
    CaptureStatusMonitor replaces this atomically per callback instead of
    hand-syncing three separate mutable attributes. Both the HTTP response
    (CaptureStatusMonitor.snapshot(), derived fresh from this value, not a
    second independently-updated copy) and this module's WS snapshot derive
    their native-fact fields from the same value; the transport-local facts
    each serves beyond these (HTTP's connecting/error_message, the WS
    snapshot's binary_generation) stay local to their own caller.

    ``latest_status`` is the last non-None ``CaptureStatus.status`` seen --
    a STARTING observation's ``status.status`` is None before any
    StatusSnapshot exists; once one has been observed, retain it even
    across a later observation that happens not to carry one, rather than
    letting a response flicker back to an empty status."""

    started: CaptureStarted | None = None
    status: CaptureStatus | None = None
    stopped: CaptureStopped | None = None
    latest_status: p.StatusSnapshot | None = None


PREVIEW_VERSION = 1
SNAPSHOT_TYPE = 3

# Browser binary envelope: <uint32 generation><uint8 kind><uint8 reserved>
# <uint16 stream_id><uint64 sequence><uint64 capture_time_ns><uint64 receive_time_ns>
BINARY_HEADER_SIZE = 32
BINARY_KIND_TRACKING = 1
BINARY_KIND_IMAGE = 2  # not emitted in this increment
_HEADER_STRUCT = struct.Struct("<IBBHQQQ")

# Initial application limits for this preview relay.
MAX_WS_VIEWERS = 4
MAX_MJPEG_VIEWERS_PER_CAMERA = 4
MAX_JSON_SNAPSHOT_BYTES = 128 * 1024
SEND_TIMEOUT_S = 2.0
SENDER_INTERVAL_S = 0.1
HEARTBEAT_INTERVAL_S = 1.0


def _sample_hz(env_name: str, default: float) -> float:
    """Read a positive sampling rate (Hz) from ``env_name``, falling back to ``default``.

    The browser preview is deliberately decimated below the capture rate (60 Hz cameras, 60+ Hz
    tracking) because every viewer pays for each sample over the WebSocket/MJPEG relay. A station
    that wants the page to show the native rate -- e.g. to verify a 60 Hz camera feed by eye --
    raises the rate at deployment time rather than by editing this module.
    """
    raw = os.environ.get(env_name)
    # Unset or blank keeps the conservative default; a non-positive or unparsable value is a
    # deployment typo and stops startup rather than silently running at the default.
    if raw is None or raw.strip() == "":
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{env_name} must be a number of Hz, got {raw!r}") from exc
    if value <= 0:
        raise ValueError(f"{env_name} must be > 0 Hz, got {raw!r}")
    return value


TRACKING_SAMPLE_HZ = _sample_hz("EGOCAP_TRACKING_SAMPLE_HZ", 10.0)
CAMERA_SAMPLE_HZ = _sample_hz("EGOCAP_CAMERA_SAMPLE_HZ", 10.0)

# Recording commands: a small, bounded JSON command from the browser gets a
# real accept/reject reply -- see recording_command.py for the actual
# parse/handling logic. The bound is tiny because every recognized action's
# payload is itself small (a batch/episode id, a task string); anything
# larger is rejected as a protocol violation rather than silently truncated.
MAX_RECORDING_COMMAND_BYTES = 8 * 1024
PENDING_REPLY_QUEUE_SIZE = 8


def pack_tracking_header(
    *, generation: int, stream_id: int, sequence: int, capture_time_ns: int, receive_time_ns: int
) -> bytes:
    """Packs the 32-byte browser binary envelope prefix for one tracking sample."""
    return _HEADER_STRUCT.pack(
        generation, BINARY_KIND_TRACKING, 0, stream_id, sequence, capture_time_ns, receive_time_ns
    )


def _capture_started_to_json(started: CaptureStarted) -> dict:
    """Boundary adapter for the snapshot's capture.started member."""
    return {
        "session_id": started.session_id,
        "running": started.running,
        "mode": int(started.mode),
        "effective_config": started.effective_config.to_json(),
        "clock": started.clock.to_json(),
        "streams": [s.to_json() for s in started.streams],
    }


def _capture_status_to_json(status: CaptureStatus) -> dict:
    """Boundary adapter for the snapshot's capture.status member."""
    return {
        "session_id": status.session_id,
        "type": int(status.type),
        "running": status.running,
        "startup_stage": None if status.startup_stage is None else int(status.startup_stage),
        "message": status.message,
        "status": None if status.status is None else _status_snapshot_to_json(status.status),
    }


def _capture_stopped_to_json(stopped: CaptureStopped) -> dict:
    """Boundary adapter for the snapshot's capture.stopped member, preserving
    the final native status rather than an earlier observation."""
    return {
        "session_id": stopped.session_id,
        "running": stopped.running,
        "reason": None if stopped.reason is None else int(stopped.reason),
        "error_message": stopped.error_message,
        "status": None if stopped.status is None else _status_snapshot_to_json(stopped.status),
        "streams": (None if stopped.streams is None else [s.to_json() for s in stopped.streams]),
    }


def _status_snapshot_to_json(status: p.StatusSnapshot) -> dict:
    """Boundary adapter for one native StatusSnapshot (providers/devices)."""
    # Same nanosecond-timestamp-as-decimal-string boundary rule as the
    # existing /api/capture-status response -- JS floats lose precision
    # above 2**53.
    data = status.to_json()
    data["updated_at_ns"] = str(data["updated_at_ns"])
    return data


def _age_ms(slot_monotonic: float, now_monotonic: float) -> int:
    """Backend-local age of a sample, from its receipt time to now."""
    return max(0, round((now_monotonic - slot_monotonic) * 1000))


def _tracking_descriptor(native: NativeFacts) -> p.TrackingStream | None:
    """The accepted tracking stream descriptor, if any."""
    if native.started is None:
        return None
    return next((s for s in native.started.streams if isinstance(s, p.TrackingStream)), None)


def _camera_descriptors(native: NativeFacts) -> tuple[p.CameraStream, ...]:
    """The accepted camera stream descriptors, if any."""
    if native.started is None:
        return ()
    return tuple(s for s in native.started.streams if isinstance(s, p.CameraStream))


def _sample_entry(slot: TrackingSlot | CameraSlot | None, diagnostic: str | None, now_monotonic: float) -> dict:
    """Boundary adapter for one samples.tracking/cameras[] entry."""
    if slot is None:
        return {"sequence": None, "age_ms": None, "diagnostic": diagnostic}
    return {
        "sequence": str(slot.sequence),
        "age_ms": _age_ms(slot.received_monotonic, now_monotonic),
        "diagnostic": diagnostic,
    }


def build_snapshot(
    *,
    backend_instance_id: str,
    revision: int,
    station_label: str,
    profile_label: str,
    connecting: bool,
    native: NativeFacts,
    error_message: str | None,
    binary_generation: int | None,
    tracking_slot: TrackingSlot | None,
    tracking_diagnostic: str | None,
    camera_slots_by_name: dict[str, CameraSlot],
    camera_diagnostics: dict[str, str],
    now_monotonic: float,
) -> dict:
    """Builds one SNAPSHOT=3 JSON message from one coherent native
    observation (``native``) plus the latest preview samples and the
    genuinely connection-specific ``binary_generation``. Everything else
    advertised (tracking/camera descriptors, the session ID in each camera
    URL) is derived from ``native.started`` here, not passed in separately
    by the caller. ``now_monotonic`` must come from the same monotonic
    clock used to record slot receipt times."""
    tracking_descriptor = _tracking_descriptor(native)
    camera_descriptors = _camera_descriptors(native)
    session_id_for_url = native.started.session_id if native.started is not None else ""

    samples_tracking = None
    if tracking_descriptor is not None:
        samples_tracking = _sample_entry(tracking_slot, tracking_diagnostic, now_monotonic)

    samples_cameras = []
    for cam in camera_descriptors:
        entry = _sample_entry(camera_slots_by_name.get(cam.name), camera_diagnostics.get(cam.name), now_monotonic)
        entry["stream_id"] = cam.stream_id
        entry["name"] = cam.name
        entry["url"] = (
            f"/stream{cam.stream_id}?backend_instance_id={backend_instance_id}&session_id={session_id_for_url}"
        )
        samples_cameras.append(entry)

    return {
        "version": PREVIEW_VERSION,
        "type": SNAPSHOT_TYPE,
        "backend_instance_id": backend_instance_id,
        "revision": revision,
        "station_label": station_label,
        "profile_label": profile_label,
        "capture": {
            "connecting": connecting,
            "started": None if native.started is None else _capture_started_to_json(native.started),
            "status": None if native.status is None else _capture_status_to_json(native.status),
            "stopped": None if native.stopped is None else _capture_stopped_to_json(native.stopped),
            "error_message": error_message,
            "binary_generation": binary_generation,
        },
        "samples": {
            "tracking": samples_tracking,
            "cameras": samples_cameras,
        },
    }
