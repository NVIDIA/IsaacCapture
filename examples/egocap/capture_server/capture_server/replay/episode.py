"""Read one recorded episode -- Zarr tracking tables plus MP4 camera files --
into the in-memory, time-aligned form the replay sources play back from.

A recorded episode directory holds one Zarr group per source. Tracking
sources (``pico_controller``: head/controller poses and inputs;
``manus_glove``: 25-node hand skeletons) are small tables indexed by
``capture_time``. A camera source is ``<name>.mp4`` plus a Zarr group
``<name>`` carrying one ``capture_time`` per frame and, when recorded, one
3x3 ``intrinsics`` matrix per frame. Every stream has its own
``capture_time`` base and row count, so playback aligns streams on one
shared timeline (seconds since the earliest recorded sample of any
stream), never by row index.

This module only knows the on-disk layout; it imports neither gr00t nor
the recording library nor the server protocol.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import zarr

__all__ = [
    "CONTROLLER_ARRAYS",
    "CONTROLLER_GROUP",
    "HAND_ARRAYS",
    "HAND_GROUP",
    "CameraTrack",
    "Episode",
    "EpisodeFormatError",
    "TrackingTable",
    "load_episode",
]

CONTROLLER_GROUP = "pico_controller"
HAND_GROUP = "manus_glove"
#: Per-row arrays a controller table must provide (besides ``capture_time``).
CONTROLLER_ARRAYS: tuple[str, ...] = (
    "head_pose",
    "left_pose",
    "right_pose",
    "left_trigger",
    "right_trigger",
    "a_button",
    "b_button",
    "x_button",
    "y_button",
)
#: Per-row arrays a hand table must provide (besides ``capture_time``).
HAND_ARRAYS: tuple[str, ...] = ("left_hand_pose", "right_hand_pose")
_TIME_ARRAY = "capture_time"
_INTRINSICS_ARRAY = "intrinsics"


class EpisodeFormatError(ValueError):
    """The directory is not a replayable episode: a stream is missing, malformed or unreadable."""


def _latest_index(times: np.ndarray, position: float) -> int:
    """Index of the last sample at or before *position* on a nondecreasing timeline, or -1 before the first."""
    return int(np.searchsorted(times, position, side="right")) - 1


@dataclass(frozen=True)
class TrackingTable:
    """One recorded numeric source: per-row arrays over the episode-relative timeline ``times``."""

    times: np.ndarray
    arrays: dict[str, np.ndarray]

    def latest_row(self, position: float) -> int:
        """Row shown at playback *position* (seconds since episode start): the last one recorded by then, or -1."""
        return _latest_index(self.times, position)


@dataclass(frozen=True)
class CameraTrack:
    """One recorded camera: its MP4 file, per-frame timeline and the stream geometry probed from the file."""

    name: str
    path: Path
    times: np.ndarray
    width: int
    height: int
    fps: float
    frame_count: int
    intrinsics: np.ndarray | None

    def latest_frame(self, position: float) -> int:
        """Frame shown at playback *position*: the last one captured by then (clamped to the frames the file
        actually holds, which can be fewer than the recorded timestamps), or -1 before the first frame."""
        return min(_latest_index(self.times, position), self.frame_count - 1)


@dataclass(frozen=True)
class Episode:
    """A loaded episode: its streams rebased onto one timeline starting at ``start_time`` and looping every
    ``period`` seconds (the last recorded sample of the longest stream plus one of its sample intervals, so the
    final sample is shown for a full interval before playback wraps)."""

    path: Path
    start_time: float
    period: float
    controllers: TrackingTable | None
    hands: TrackingTable | None
    cameras: dict[str, CameraTrack]


def _read_times(group: Any, name: str) -> np.ndarray:
    """The absolute ``capture_time`` column of *group*, validated as finite and nondecreasing (searchsorted
    needs sorted input, and a backwards jump would make playback show samples out of recorded order)."""
    if _TIME_ARRAY not in group:
        raise EpisodeFormatError(f"{name}: no '{_TIME_ARRAY}' array")
    times = np.asarray(group[_TIME_ARRAY][:], dtype=np.float64).reshape(-1)
    if times.size == 0 or not np.all(np.isfinite(times)):
        raise EpisodeFormatError(f"{name}: '{_TIME_ARRAY}' is empty or not finite")
    if np.any(np.diff(times) < 0):
        raise EpisodeFormatError(f"{name}: '{_TIME_ARRAY}' is not nondecreasing")
    return times


def _read_table(
    group: Any | None, name: str, required: tuple[str, ...]
) -> tuple[np.ndarray, dict[str, np.ndarray]] | None:
    """A tracking group's absolute times and its required arrays, fully loaded; None when the group is absent.

    Every array must have one row per timestamp -- a row-count mismatch would silently pair samples with the
    wrong time, so it is a format error, not something to truncate past.
    """
    if group is None:
        return None
    times = _read_times(group, name)
    arrays: dict[str, np.ndarray] = {}
    for array_name in required:
        if array_name not in group:
            raise EpisodeFormatError(f"{name}: no '{array_name}' array")
        values = np.asarray(group[array_name][:])
        if values.shape[:1] != times.shape:
            raise EpisodeFormatError(
                f"{name}: '{array_name}' has {values.shape[0]} rows but '{_TIME_ARRAY}' has {times.shape[0]}"
            )
        arrays[array_name] = values
    return times, arrays


def _probe_video(path: Path) -> tuple[int, int, float, int]:
    """Width, height, fps and frame count read from the container header -- no frame is decoded or seeked."""
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise EpisodeFormatError(f"{path.name}: could not open the video")
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()
    if width <= 0 or height <= 0 or not 0 < fps < float("inf") or frame_count <= 0:
        raise EpisodeFormatError(
            f"{path.name}: unusable video geometry {width}x{height}@{fps}, {frame_count} frames"
        )
    return width, height, fps, frame_count


def _read_intrinsics(group: Any, name: str, rows: int) -> np.ndarray | None:
    """Per-frame 3x3 intrinsics when the recorder wrote them, else None -- never a synthesized matrix."""
    if _INTRINSICS_ARRAY not in group:
        return None
    values = np.asarray(group[_INTRINSICS_ARRAY][:], dtype=np.float64)
    if values.shape != (rows, 3, 3):
        raise EpisodeFormatError(
            f"{name}: '{_INTRINSICS_ARRAY}' has shape {values.shape}, expected ({rows}, 3, 3)"
        )
    return values


def _sample_interval(times: np.ndarray, fallback: float) -> float:
    """The stream's typical sample spacing (median of its own gaps), or *fallback* for a single-sample stream."""
    if times.size < 2:
        return fallback
    return float(np.median(np.diff(times)))


def load_episode(path: str | Path) -> Episode:
    """Load the episode directory at *path* and rebase all of its streams onto one shared timeline.

    Raises :class:`EpisodeFormatError` for a directory that is not a replayable episode, including one with no
    streams at all; a missing directory raises ``FileNotFoundError``.
    """
    path = Path(path)
    if not path.is_dir():
        raise FileNotFoundError(f"episode directory not found: {path}")
    root = zarr.open_group(str(path), mode="r")

    def group(name: str) -> Any | None:
        """The named child group, or None. Looked up by name rather than by listing the directory: an episode
        also holds the MP4s and the recorder's marker files, which a listing would warn about one by one."""
        return root[name] if name in root else None

    controllers_raw = _read_table(group(CONTROLLER_GROUP), CONTROLLER_GROUP, CONTROLLER_ARRAYS)
    hands_raw = _read_table(group(HAND_GROUP), HAND_GROUP, HAND_ARRAYS)

    # Cameras are discovered from the MP4 files; each needs its Zarr group for frame times, since the
    # container's own timestamps are relative to the file, not to the other streams.
    cameras_raw: dict[str, tuple[np.ndarray, Path, tuple[int, int, float, int], np.ndarray | None]] = {}
    for mp4 in sorted(path.glob("*.mp4")):
        name = mp4.stem
        camera_group = group(name)
        if camera_group is None:
            raise EpisodeFormatError(f"{mp4.name}: no '{name}' Zarr group with its frame times")
        times = _read_times(camera_group, name)
        cameras_raw[name] = (times, mp4, _probe_video(mp4), _read_intrinsics(camera_group, name, times.size))

    all_times = [raw[0] for raw in (controllers_raw, hands_raw) if raw is not None]
    all_times += [raw[0] for raw in cameras_raw.values()]
    if not all_times:
        raise EpisodeFormatError(f"{path}: no replayable streams (no tracking groups and no *.mp4 files)")
    # One shared origin for every stream, so cross-stream alignment uses the recorded clock, not row indices.
    start_time = float(min(times[0] for times in all_times))

    controllers = None
    if controllers_raw is not None:
        controllers = TrackingTable(controllers_raw[0] - start_time, controllers_raw[1])
    hands = None
    if hands_raw is not None:
        hands = TrackingTable(hands_raw[0] - start_time, hands_raw[1])
    cameras = {
        name: CameraTrack(
            name=name,
            path=mp4,
            times=times - start_time,
            width=geometry[0],
            height=geometry[1],
            fps=geometry[2],
            frame_count=geometry[3],
            intrinsics=intrinsics,
        )
        for name, (times, mp4, geometry, intrinsics) in cameras_raw.items()
    }

    # The loop period covers the longest stream plus one of its own intervals, so its last sample is displayed
    # for a full interval before everything wraps together.
    ends = [table.times[-1] + _sample_interval(table.times, 0.0) for table in (controllers, hands) if table]
    ends += [track.times[-1] + _sample_interval(track.times, 1.0 / track.fps) for track in cameras.values()]
    period = float(max(ends))
    if period <= 0.0:
        # A single-sample, zero-length episode still needs a positive period for the modulo-based timeline.
        period = 1.0
    return Episode(
        path=path, start_time=start_time, period=period, controllers=controllers, hands=hands, cameras=cameras
    )
