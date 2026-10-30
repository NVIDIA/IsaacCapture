"""Value types for the recording service: configuration, samples, results and errors.

Everything here is recording-owned. Samples carry measured payloads and timestamps only;
nothing describes where a sample came from, how it was acquired, or whether it is
"live" -- the service treats every sample identically.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

from egocap.recording.capture_qc import CaptureQcConfig, QcDecision
from egocap.recording.dataset_check import CheckResult, Outcome, QualityReport
from egocap.recording.episode_store import FieldSpec, NumericSourceSpec, VideoSourceSpec
from egocap.recording.route import RoutingConfig

__all__ = [
    "CaptureQcConfig",
    "CheckResult",
    "CalibrationAsset",
    "CloseResult",
    "ConfigurationError",
    "Disposition",
    "EpisodeInfo",
    "EpisodeResult",
    "FieldSpec",
    "NumericSample",
    "NumericSourceSpec",
    "Outcome",
    "PendingEpisode",
    "PixelFormat",
    "QcConfig",
    "QcDecision",
    "QualityReport",
    "RecordingError",
    "RecordingPhase",
    "RecordingSnapshot",
    "RoutingConfig",
    "ServiceConfig",
    "SessionInfo",
    "SourceCounts",
    "StateError",
    "SubmissionResult",
    "SubmissionStatus",
    "ValidationError",
    "VideoSample",
    "VideoSourceSpec",
]


class RecordingError(Exception):
    """Base class for every error the recording service raises on purpose."""


class ConfigurationError(RecordingError):
    """The configuration cannot be recorded with; raised before any effect."""


class ValidationError(RecordingError):
    """A submitted sample does not match its declared source schema."""


class StateError(RecordingError):
    """The call is not allowed in the service's current state."""


class RecordingPhase(Enum):
    """Where the single recording lifecycle currently is."""

    IDLE = "IDLE"
    STARTING = "STARTING"
    RECORDING = "RECORDING"
    STOPPING = "STOPPING"
    FINALIZING = "FINALIZING"
    RECOVERY_REQUIRED = "RECOVERY_REQUIRED"
    CLOSED = "CLOSED"


class SubmissionStatus(Enum):
    """Ordinary outcomes of submitting a sample; invalid input raises instead."""

    ACCEPTED = "ACCEPTED"
    FULL = "FULL"


class Disposition(Enum):
    """What ultimately happened to an episode's data."""

    KEPT = "KEPT"
    REJECTED = "REJECTED"
    DISCARDED = "DISCARDED"
    RECOVERY_REQUIRED = "RECOVERY_REQUIRED"


class PixelFormat(Enum):
    """Channel order of a submitted 8-bit, 3-channel image."""

    RGB8 = "rgb8"
    BGR8 = "bgr8"


@dataclass(frozen=True)
class QcConfig:
    """Quality policy. ``required_sources`` must contain data (a failed integrity check rejects the take).

    ``capture`` optionally adds capture-quality checks over the sealed recording; ``None`` leaves them not run
    (recorded as SKIPPED, never as passed).
    """

    required_sources: tuple[str, ...] = ()
    capture: CaptureQcConfig | None = None


@dataclass(frozen=True)
class CalibrationAsset:
    """A caller-supplied calibration file copied into the session as ``name`` (never discovered or synthesized)."""

    name: str
    path: Path
    required: bool = True


@dataclass(frozen=True)
class ServiceConfig:
    """Everything the service needs to record: destination, source schemas, queue size, QC and deadlines."""

    output_root: Path
    numeric_sources: tuple[NumericSourceSpec, ...] = ()
    video_sources: tuple[VideoSourceSpec, ...] = ()
    qc: QcConfig = QcConfig()
    # Samples (not bytes) the single episode queue holds. Memory is bounded by queue_capacity times the largest
    # sample, e.g. 512 x one 1280x800 RGB frame (~3 MB) is about 1.5 GB. When the queue is full, submit_sample
    # returns FULL and consumes nothing: the recording and the service stay consistent, but a caller that ignores
    # FULL silently loses that sample (accepted == written still holds, so integrity checks cannot see the gap).
    # A caller must treat FULL as backpressure: wait and resubmit the same sample. If the encoder cannot keep up
    # at the submission rate, FULL repeats and a size-based or per-source budget would be needed before live use.
    queue_capacity: int = 512
    close_deadline_seconds: float = 30.0
    quarantine: bool = False
    calibration_assets: tuple[CalibrationAsset, ...] = ()
    latency: dict[str, float] | None = None
    # None keeps the single-session layout (<session>/episode_N). Set, kept episodes are routed to
    # <output_root>/<Studio>/<timestamp>/episode_<k> and each group carries native dataset-level totals.
    routing: RoutingConfig | None = None
    # This station's configured identity (e.g. "Station 04"), written into each new session's
    # config.yaml meta.egocap_station_name so analytics tools (count_daily_trajectories.py) can map
    # a recorded session back to the station that produced it. None skips the write entirely.
    station_label: str | None = None


@dataclass(frozen=True)
class NumericSample:
    """One row for a numeric source: named fields matching the source schema exactly, plus timestamps."""

    source: str
    capture_time: float
    receive_time: float
    fields: Mapping[str, Any]


@dataclass(frozen=True)
class VideoSample:
    """One decoded camera frame (uint8, height x width x 3) with timestamps and per-frame intrinsics."""

    source: str
    image: np.ndarray
    pixel_format: PixelFormat
    capture_time: float
    receive_time: float
    intrinsics: np.ndarray | None = None


@dataclass(frozen=True)
class SessionInfo:
    """An open session and where its data lives."""

    session_id: str
    output_root: Path
    session_path: Path


@dataclass(frozen=True)
class EpisodeInfo:
    """A started episode and its pending location."""

    session_id: str
    episode_id: str
    episode_index: int
    pending_path: Path


@dataclass(frozen=True)
class SubmissionResult:
    """Whether a sample was accepted (with its per-source admission index) or the queue was full."""

    status: SubmissionStatus
    source_name: str
    admission_index: int | None


@dataclass(frozen=True)
class SourceCounts:
    """Samples accepted into the queue versus rows/frames confirmed written, for one source."""

    source_name: str
    accepted: int
    written: int


@dataclass(frozen=True)
class EpisodeResult:
    """The real outcome of stopping, discarding or recovering an episode."""

    session_id: str
    episode_id: str
    episode_index: int
    disposition: Disposition
    final_path: Path | None
    pending_path: Path | None
    counts: tuple[SourceCounts, ...]
    duration_seconds: float
    quality: QualityReport | None
    issues: tuple[str, ...] = ()
    # What capture quality decided: None until a decision is made (a take discarded before the check, or not yet
    # checked); a take discarded or retained after the decision keeps it. ``capture_quality`` is the per-check
    # evidence behind it. Separate from ``quality`` (storage integrity) and from ``disposition``.
    qc_decision: QcDecision | None = None
    capture_quality: QualityReport | None = None
    # The kept output's slot in its routed group (None when not routed or not kept); ``episode_index`` stays the
    # attempt identity and never changes, so a rejected attempt leaves no gap in the output numbering.
    output_index: int | None = None


@dataclass(frozen=True)
class PendingEpisode:
    """An unresolved episode found on disk."""

    session_id: str
    episode_id: str
    episode_index: int
    pending_path: Path
    storage_phase: str
    recoverable: bool
    issues: tuple[str, ...] = ()


@dataclass(frozen=True)
class CloseResult:
    """Whether every resource was actually released, and what is still unfinished if not."""

    released: bool
    unfinished_episode_ids: tuple[str, ...] = ()
    issues: tuple[str, ...] = ()


@dataclass(frozen=True)
class RecordingSnapshot:
    """Immutable recording facts at one moment."""

    session: SessionInfo | None
    phase: RecordingPhase
    episode: EpisodeInfo | None
    counts: tuple[SourceCounts, ...] = ()
    last_result: EpisodeResult | None = None
    pending: tuple[PendingEpisode, ...] = ()
    issues: tuple[str, ...] = field(default=())
