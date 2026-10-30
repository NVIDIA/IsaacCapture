"""Read an already-recorded episode (Zarr arrays + MP4 files) as plain sample tuples.

This is a dataset reader, not an acquisition emulator: it hands back stored rows and decoded
frames in their original order with their original timestamps, so a caller can submit them to
a recorder through the same interface as any other producer. It never modifies the dataset.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import zarr

from .episode_store import TIMESTAMP_FIELDS, FieldSpec, NumericSourceSpec, VideoSourceSpec


@dataclass(frozen=True)
class NumericRow:
    """One stored numeric row: its timestamps and every non-timestamp field."""

    capture_time: float
    receive_time: float
    fields: dict[str, Any]


@dataclass(frozen=True)
class VideoFrame:
    """One decoded frame (RGB, uint8) with its timestamps and per-frame intrinsics."""

    image_rgb: np.ndarray
    capture_time: float
    receive_time: float
    intrinsics: np.ndarray | None


class EpisodeReader:
    """Read-only view of one recorded episode directory."""

    def __init__(self, path: Path) -> None:
        """Open ``path`` read-only; raises if it is not a Zarr group."""
        self.path = Path(path)
        self._group = zarr.open_group(str(self.path), mode="r")

    def source_names(self) -> list[str]:
        """Names of all source groups, sorted."""
        return sorted(name for name, _ in self._group.groups())

    def is_video(self, source: str) -> bool:
        """A source is a camera when an MP4 named after it sits beside the Zarr groups."""
        return (self.path / f"{source}.mp4").is_file()

    def row_count(self, source: str) -> int:
        """Number of rows (frames, for a camera) recorded for the source."""
        return int(self._group[source]["capture_time"].shape[0])

    def numeric_spec(self, source: str, fields: tuple[str, ...] | None = None) -> NumericSourceSpec:
        """Schema of a numeric source as stored (optionally restricted to ``fields``)."""
        group = self._group[source]
        names = [n for n, _ in group.arrays() if n not in TIMESTAMP_FIELDS and (fields is None or n in fields)]
        return NumericSourceSpec(
            source, tuple(FieldSpec(n, str(group[n].dtype), tuple(group[n].shape[1:])) for n in sorted(names))
        )

    def video_spec(self, source: str, encoder: str = "ffmpeg_h264") -> VideoSourceSpec:
        """Geometry and nominal fps of a camera, taken from its MP4."""
        capture = cv2.VideoCapture(str(self.path / f"{source}.mp4"))
        try:
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = float(capture.get(cv2.CAP_PROP_FPS))
        finally:
            capture.release()
        return VideoSourceSpec(source, width, height, fps, encoder, "intrinsics" in self._group[source])

    def numeric_rows(self, source: str, start: int = 0, stop: int | None = None) -> Iterator[NumericRow]:
        """Yield stored rows ``[start, stop)`` in recorded order."""
        group = self._group[source]
        total = self.row_count(source)
        stop = total if stop is None else min(stop, total)
        names = [n for n, _ in group.arrays() if n not in TIMESTAMP_FIELDS]
        capture, receive = group["capture_time"][start:stop], group["receive_time"][start:stop]
        data = {n: group[n][start:stop] for n in names}
        for i in range(stop - start):
            yield NumericRow(float(capture[i]), float(receive[i]), {n: data[n][i] for n in names})

    def video_frames(self, source: str, start: int = 0, stop: int | None = None) -> Iterator[VideoFrame]:
        """Yield decoded frames ``[start, stop)`` with their timestamps, converted from BGR to RGB once."""
        group = self._group[source]
        total = self.row_count(source)
        stop = total if stop is None else min(stop, total)
        capture = cv2.VideoCapture(str(self.path / f"{source}.mp4"))
        try:
            capture.set(cv2.CAP_PROP_POS_FRAMES, start)
            for i in range(start, stop):
                ok, bgr = capture.read()
                if not ok:
                    raise RuntimeError(f"{source}: could not decode frame {i}")
                intrinsics = group["intrinsics"][i] if "intrinsics" in group else None
                yield VideoFrame(
                    np.ascontiguousarray(bgr[..., ::-1]),
                    float(group["capture_time"][i]),
                    float(group["receive_time"][i]),
                    intrinsics,
                )
        finally:
            capture.release()
