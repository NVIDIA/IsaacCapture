"""Append-only on-disk episode storage: Zarr v3 numeric groups plus real MP4 video.

This is the concrete persistence backend used by the recording service. It writes
the same layout the recorders produce (one Zarr group per source holding one array
per field plus ``capture_time``/``receive_time``, and one ``<source>.mp4`` per
camera), but accepts each sample exactly once through :meth:`EpisodeStore.append_numeric`
and :meth:`EpisodeStore.append_video` instead of polling a clocked source buffer, so
nothing is resampled or dropped.

This module imports ``create_video_writer`` and ``resolve_video_encoder_options`` from this
package's own local ``video_writer_backends`` (``episode_layout`` imports only ``PENDING_SUFFIX``
from this package's own local ``constants``).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import zarr
from zarr.codecs import ZstdCodec

from .episode_layout import fsync_tree
from .video_writer_backends import create_video_writer, resolve_video_encoder_options

#: Per-source timestamp arrays every group carries (seconds, float64).
TIMESTAMP_FIELDS = ("capture_time", "receive_time")
#: Rows buffered in memory per array before they are appended to disk.
FLUSH_ROWS = 256


@dataclass(frozen=True)
class FieldSpec:
    """One numeric field of a source: a name, numpy dtype string and per-row shape."""

    name: str
    dtype: str
    shape: tuple[int, ...] = ()


@dataclass(frozen=True)
class NumericSourceSpec:
    """A numeric source (pose tracker, glove, ...) and the fields recorded per row."""

    name: str
    fields: tuple[FieldSpec, ...]


@dataclass(frozen=True)
class VideoSourceSpec:
    """A camera: frame geometry, nominal playback fps, encoder and optional per-frame intrinsics."""

    name: str
    width: int
    height: int
    fps: float
    encoder: str = "ffmpeg_h264"
    intrinsics: bool = True
    encoder_options: dict[str, Any] | None = None


def validate_source_specs(numeric: tuple[NumericSourceSpec, ...], video: tuple[VideoSourceSpec, ...]) -> None:
    """Raise ValueError, naming the offending source, for a schema the store could not write.

    Checked up front so a configuration typo fails before any episode is started instead of leaving a retained
    take that has to be discarded on every attempt: field names must be unique, safe array names that do not
    collide with the timestamp arrays; dtypes must be numeric or bool; shapes non-negative; and each camera's
    encoder options must be valid for its encoder.
    """
    for spec in numeric:
        names = [f.name for f in spec.fields]
        if len(set(names)) != len(names):
            raise ValueError(f"{spec.name}: duplicate field names {names}")
        for field in spec.fields:
            if not field.name or "/" in field.name or "\\" in field.name or field.name.startswith("."):
                raise ValueError(f"{spec.name}: field name {field.name!r} is not a safe array name")
            if field.name in TIMESTAMP_FIELDS:
                raise ValueError(f"{spec.name}: field name {field.name!r} collides with the timestamp arrays")
            try:
                kind = np.dtype(field.dtype).kind
            except TypeError as exc:
                raise ValueError(f"{spec.name}/{field.name}: invalid dtype {field.dtype!r}") from exc
            if kind not in "biuf":
                raise ValueError(f"{spec.name}/{field.name}: dtype {field.dtype!r} must be bool, integer or float")
            if any(not isinstance(n, int) or isinstance(n, bool) or n < 0 for n in field.shape):
                raise ValueError(f"{spec.name}/{field.name}: invalid shape {field.shape!r}")
    for vspec in video:
        resolve_video_encoder_options(vspec.encoder, vspec.encoder_options)  # ValueError on bad keys or values


class _SourceArrays:
    """Buffered, append-only Zarr arrays for one source group."""

    def __init__(self, group: zarr.Group, fields: tuple[FieldSpec, ...]) -> None:
        """Create one empty, row-appendable Zarr array per field inside ``group``."""
        self.rows = 0
        self._buffers: dict[str, list[np.ndarray]] = {}
        self._arrays: dict[str, zarr.Array] = {}
        for field in fields:
            self._arrays[field.name] = group.create_array(
                field.name,
                shape=(0, *field.shape),
                chunks=(FLUSH_ROWS, *field.shape),
                dtype=np.dtype(field.dtype),
                compressors=ZstdCodec(),
            )
            self._buffers[field.name] = []

    def prepare(self, row: Mapping[str, Any]) -> dict[str, np.ndarray]:
        """Validate and copy one complete row (declared fields, shapes and dtypes) without storing it."""
        if set(row) != set(self._arrays):
            raise ValueError(f"row fields {sorted(row)} != declared {sorted(self._arrays)}")
        converted = {}
        for name, array in self._arrays.items():
            value = np.asarray(row[name])
            if value.shape != array.shape[1:]:
                raise ValueError(f"field {name!r}: shape {value.shape} != declared {array.shape[1:]}")
            if value.dtype != array.dtype:
                raise ValueError(f"field {name!r}: dtype {value.dtype} != declared {array.dtype}")
            converted[name] = value.copy()
        return converted

    def commit(self, converted: dict[str, np.ndarray]) -> None:
        """Buffer a row returned by :meth:`prepare` and flush a full chunk to disk."""
        for name, value in converted.items():
            self._buffers[name].append(value)
        self.rows += 1
        if len(self._buffers[next(iter(self._buffers))]) >= FLUSH_ROWS:
            self.flush()

    def flush(self) -> None:
        """Append buffered rows to disk."""
        for name, buffered in self._buffers.items():
            if buffered:
                self._arrays[name].append(np.stack(buffered), axis=0)
                buffered.clear()


class EpisodeStore:
    """Writes one pending episode directory; each accepted sample is stored exactly once."""

    def __init__(
        self,
        pending_dir: Path,
        numeric: tuple[NumericSourceSpec, ...],
        video: tuple[VideoSourceSpec, ...],
    ) -> None:
        """Create ``pending_dir`` (which must not exist) and every Zarr group; no encoder is started yet.

        Encoders are started separately by :meth:`start_encoders` so the caller already owns this store (and can
        abort it) when a later encoder fails to open; there is deliberately no construct-and-start shortcut.
        """
        names = [s.name for s in numeric] + [s.name for s in video]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate source names: {names}")
        self.pending_dir = Path(pending_dir)
        self.pending_dir.mkdir(parents=True, exist_ok=False)
        self._root = zarr.open_group(str(self.pending_dir), mode="w", zarr_format=3)
        self._numeric: dict[str, _SourceArrays] = {}
        self._video: dict[str, _SourceArrays] = {}
        self._writers: dict[str, Any] = {}
        self._video_specs = {s.name: s for s in video}
        self._sealed = False
        self._accepting = True
        self._encoders_started = False
        #: Effective encoder options per camera (empty for encoders without options), for provenance.
        self.encoder_options: dict[str, dict[str, Any]] = {}
        self._release_failures: dict[str, str] | None = None
        timestamps = tuple(FieldSpec(n, "float64") for n in TIMESTAMP_FIELDS)
        for spec in numeric:
            self._numeric[spec.name] = _SourceArrays(self._root.create_group(spec.name), timestamps + spec.fields)
        for vspec in video:
            extra = (FieldSpec("intrinsics", "float64", (3, 3)),) if vspec.intrinsics else ()
            self._video[vspec.name] = _SourceArrays(self._root.create_group(vspec.name), timestamps + extra)

    def start_encoders(self) -> None:
        """Start one real encoder per camera, in order.

        If a later camera fails, the encoders already started are released and any release failure is
        recorded permanently (so :meth:`abort` reports unconfirmed release), then the primary failure is
        raised with those diagnostics. The store stays valid for the caller to abort.
        """
        self._require_open()
        if self._encoders_started:
            raise RuntimeError("encoders were already started for this store")
        self._encoders_started = True
        try:
            for vspec in self._video_specs.values():
                source_config = {"video_encoder_options": vspec.encoder_options} if vspec.encoder_options else None
                self.encoder_options[vspec.name] = resolve_video_encoder_options(
                    vspec.encoder, vspec.encoder_options
                )
                self._writers[vspec.name] = create_video_writer(
                    vspec.encoder,
                    str(self.pending_dir / f"{vspec.name}.mp4"),
                    vspec.width,
                    vspec.height,
                    vspec.fps,
                    source_config=source_config,
                )
        except BaseException as exc:
            self._accepting = False
            failures = self._release_writers()
            detail = f"; release of already-open encoders failed: {failures}" if failures else ""
            raise RuntimeError(f"video encoder startup failed: {exc}{detail}") from exc

    def _require_open(self) -> None:
        """Reject any append once sealing or aborting has begun."""
        if not self._accepting:
            raise RuntimeError("episode store is closed to new samples")

    def append_numeric(
        self, source: str, capture_time: float, receive_time: float, fields: Mapping[str, Any]
    ) -> None:
        """Store one numeric row (timestamps plus the declared fields)."""
        self._require_open()
        arrays = self._numeric[source]
        arrays.commit(arrays.prepare({"capture_time": capture_time, "receive_time": receive_time, **fields}))

    def append_video(
        self,
        source: str,
        frame_bgr: np.ndarray,
        capture_time: float,
        receive_time: float,
        intrinsics: np.ndarray | None = None,
    ) -> None:
        """Store one camera frame (uint8 BGR) and its timestamp row; one encoded frame per call.

        The row is validated, then the frame is handed to the encoder, and only then is the row
        counted, so a rejected row or a failed encoder write never leaves a count without its frame.
        """
        self._require_open()
        spec = self._video_specs[source]
        row: dict[str, Any] = {"capture_time": capture_time, "receive_time": receive_time}
        if spec.intrinsics:
            if intrinsics is None:
                raise ValueError(f"{source}: intrinsics are declared but missing")
            row["intrinsics"] = intrinsics
        elif intrinsics is not None:
            raise ValueError(f"{source}: intrinsics supplied but not declared")
        arrays = self._video[source]
        converted = arrays.prepare(row)
        self._writers[source].write(frame_bgr)
        arrays.commit(converted)

    def write_attributes(self, attributes: Mapping[str, Any]) -> None:
        """Set dataset-level attributes (task, identity, ...) on the episode root group."""
        self._require_open()
        self._root.attrs.update(dict(attributes))

    def counts(self) -> dict[str, int]:
        """Rows accepted so far per source (video: frames encoded)."""
        return {n: s.rows for n, s in {**self._numeric, **self._video}.items()}

    def _release_writers(self) -> dict[str, str]:
        """Release every encoder exactly once and remember each failure; later calls reuse the first outcome.

        An encoder that failed to release is never retried: upstream release is idempotent and a second
        call would wrongly look successful, so the failure stays recorded for seal and abort.
        """
        if self._release_failures is None:
            failures = {}
            for name, writer in self._writers.items():
                try:
                    writer.release()
                except Exception as exc:  # noqa: BLE001 - record every encoder failure, report them together
                    failures[name] = str(exc)
            self._release_failures = failures
        return self._release_failures

    def seal(self) -> dict[str, int]:
        """Flush arrays, finish video, and fsync; returns counts only if every step really succeeded.

        A failure is permanent for this store: a later seal raises again and never reports success.
        """
        self._accepting = False
        if self._sealed:
            return self.counts()
        for source in (*self._numeric.values(), *self._video.values()):
            source.flush()
        failures = self._release_writers()
        if failures:
            raise RuntimeError("video encoding failed: " + "; ".join(f"{n}: {e}" for n, e in failures.items()))
        # Payload durability must precede publishing a SEALED manifest.
        fsync_tree(self.pending_dir)
        self._sealed = True
        return self.counts()

    def abort(self) -> bool:
        """Stop the encoders; True only if every one confirmed release.

        The pending directory is left on disk for the caller to discard. When this returns False an
        encoder's release is unconfirmed and the caller must neither delete data nor report release.
        """
        self._accepting = False
        return not self._release_writers()
