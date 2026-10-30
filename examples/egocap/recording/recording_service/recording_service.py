"""RecordingService: turns submitted samples into a checked, finalized on-disk episode.

A caller supplies numeric samples and decoded camera images; the service validates and
copies them into one bounded queue, a single episode worker writes them through the real
Zarr/MP4 backend, and ``stop_episode`` drains, seals, quality-checks and finalizes the
take. The service never sees where a sample came from.

Ownership rules (all work that touches files runs on one storage thread per episode):

* Every task is started through ``_own`` and retained until it ends.
* Every blocking storage call goes through ``_call``: the call belongs to the service, so
  cancelling a waiting caller (or hitting a deadline) abandons only the wait. The take stays
  busy (``in_flight``) until the call really ends, and recovery/discard are refused meanwhile.
* ``close`` waits against one absolute deadline and reports resources as released only when
  they really are; the session lock is held until then.
"""

from __future__ import annotations

import asyncio
import copy
import datetime
import hashlib
import json
import logging
import os
import shutil
import tempfile
import time
import uuid
from collections import deque
from collections.abc import Awaitable, Callable, Coroutine
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from egocap.recording.capture_qc import (
    ALGORITHM_VERSION,
    QcDecision,
    check_capture_quality,
    decide,
    finalization_for,
    policy_to_dict,
    quality_attributes,
)
from egocap.recording.dataset_check import Outcome, check_episode, report_from_dict, report_to_dict
from egocap.recording.episode_layout import (
    CALIBRATION_RECORD,
    FINALIZATION_KEEP,
    FINALIZATION_REMOVE,
    FINALIZATION_RETAIN,
    IDENTITY_ATTR,
    MANIFEST_DIR,
    SESSION_CONFIG_FILE,
    UNRESOLVED_PHASES,
    EpisodeManifest,
    SessionLock,
    destination_path,
    episode_dir_names,
    finalize_episode_dir,
    fsync_dir,
    locate_episode,
    manifest_path,
    next_episode_index,
    remove_owned_episode,
    scan_manifests,
    write_episode_attributes,
    write_json_atomic,
    write_manifest,
    write_yaml_atomic,
)
from egocap.recording.episode_store import EpisodeStore, validate_source_specs
from egocap.recording.route import assign_output, refresh_group_totals, resolve_studio

from .models import (
    CloseResult,
    ConfigurationError,
    Disposition,
    EpisodeInfo,
    EpisodeResult,
    NumericSample,
    PendingEpisode,
    PixelFormat,
    RecordingPhase,
    RecordingSnapshot,
    ServiceConfig,
    SessionInfo,
    SourceCounts,
    StateError,
    SubmissionResult,
    SubmissionStatus,
    ValidationError,
    VideoSample,
)

logger = logging.getLogger(__name__)

_ENCODERS = ("opencv_mp4v", "ffmpeg_h264", "ffmpeg_h264_nvenc")


def _sha256(path: Path) -> str:
    """SHA-256 of a file's bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _copy_file_atomic(source: Path, dest: Path) -> None:
    """Durably copy *source* to *dest*: temp file in the same directory, flush, fsync, atomic replace,
    directory fsync, temp removed on any failure."""
    directory = dest.parent
    fd, tmp_name = tempfile.mkstemp(dir=directory, prefix=dest.stem + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(source.read_bytes())
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, dest)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise
    fsync_dir(directory)


def _layout_digest(config: ServiceConfig) -> str:
    """The configuration digest without the capture-QC policy: what the data layout alone depends on."""
    return _digest(replace(config, qc=replace(config.qc, capture=None)))


def _digest(config: ServiceConfig) -> str:
    """Stable digest of the parts of the configuration an episode's data layout depends on."""
    qc = asdict(config.qc)
    if config.qc.capture is None:
        del qc["capture"]  # absent, not null, so a take recorded before capture QC existed keeps its digest
    else:
        qc["capture_algorithm_version"] = ALGORITHM_VERSION  # a changed algorithm is a changed configuration
    canonical = {
        **({"routing": True} if config.routing is not None else {}),  # routed or not; each take freezes its studio
        "numeric": [asdict(s) for s in config.numeric_sources],
        "video": [asdict(s) for s in config.video_sources],
        "qc": qc,
        "latency": config.latency,
        "calibration": [a.name for a in config.calibration_assets],
    }
    return hashlib.sha256(json.dumps(canonical, sort_keys=True).encode()).hexdigest()


class _Take:
    """One episode's mutable bookkeeping: its manifest, store, queue and work ownership."""

    def __init__(self, manifest: EpisodeManifest, sources: list[str], store: EpisodeStore | None) -> None:
        """Create the bookkeeping for a take; ``store`` is None for an episode recovered from disk."""
        self.manifest = manifest
        self.store = store
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"episode-{manifest.episode_index}")
        self.queue: deque[NumericSample | VideoSample] = deque()
        self.wake = asyncio.Event()
        self.admission_closed = False
        self.accepted = dict.fromkeys(sources, 0)
        self.written = dict.fromkeys(sources, 0)
        self.span: dict[str, list[float]] = {}
        self.worker: asyncio.Task[Any] | None = None
        self.operation_task: asyncio.Task[Any] | None = None
        self.operation = ""
        self.error: str | None = None
        self.released = store is None
        # Work ownership: ``operating`` while a lifecycle operation owns the take,
        # ``in_flight`` while a storage call is running even if its waiter went away.
        self.operating = False
        self.in_flight = 0
        self.idle = asyncio.Event()
        self.idle.set()

    @property
    def episode_id(self) -> str:
        """The stable episode identity."""
        return self.manifest.episode_id

    def duration(self) -> float:
        """Maximum per-source capture-time span; never subtracts timestamps from different sources."""
        return max((last - first for first, last in self.span.values()), default=0.0)


class RecordingService:
    """Sample-to-files recorder; see the module docstring for the ownership rules."""

    def __init__(self, config: ServiceConfig) -> None:
        """Hold the configuration; nothing is opened and no task is started until ``open_session``."""
        # One detached snapshot: the caller may keep and mutate its own config (including nested dicts) freely.
        self._config = copy.deepcopy(config)
        if self._config.qc.capture is not None:  # concrete camera rates: stored policy and digest never hold None
            resolved = self._config.qc.capture.resolved(self._config.video_sources)
            self._config = replace(self._config, qc=replace(self._config.qc, capture=resolved))
        config = self._config
        self._phase = RecordingPhase.IDLE
        self._session: SessionInfo | None = None
        self._lock: SessionLock | None = None
        self._take: _Take | None = None
        self._pending: dict[str, PendingEpisode] = {}
        self._retained: dict[str, _Take] = {}
        self._last_result: EpisodeResult | None = None
        self._issues: list[str] = []
        self._tasks: set[asyncio.Task[Any]] = set()
        self._start_task: asyncio.Task[Any] | None = None
        self._blocking: list[str] = []
        self._closing = False
        self._close_task: asyncio.Task[CloseResult] | None = None
        self._fatal_error: str | None = None
        self._source_kinds = {s.name: "numeric" for s in config.numeric_sources}
        self._source_kinds.update({s.name: "video" for s in config.video_sources})
        self._numeric = {s.name: s for s in config.numeric_sources}
        self._video = {s.name: s for s in config.video_sources}

    # -- public API ---------------------------------------------------------------

    async def open_session(self, session_id: str | None = None) -> SessionInfo:
        """Validate the configuration, lock and scan the session, and return where it lives.

        With ``session_id`` an existing session directory is reopened so its pending
        episodes can be recovered or discarded; otherwise a new session is created.
        """
        return await self._run(self._open_impl(session_id))

    async def start_episode(self, task: str, operator: str = "") -> EpisodeInfo:
        """Allocate an episode, open real writers, and return only once samples can be admitted."""
        return await self._run(self._start_impl(task, operator))

    def submit_sample(self, episode_id: str, sample: NumericSample | VideoSample) -> SubmissionResult:
        """Validate, copy and enqueue one sample without waiting.

        FULL means the queue is at ``queue_capacity`` and nothing was consumed: the caller must wait and resubmit
        the same sample (see ``ServiceConfig.queue_capacity`` for the memory bound and what ignoring FULL costs).
        """
        take = self._take
        if take is None or take.episode_id != episode_id:
            raise StateError(f"no active episode {episode_id!r}")
        if self._phase is not RecordingPhase.RECORDING or take.admission_closed or self._closing:
            raise StateError(f"episode {episode_id!r} is not admitting samples ({self._phase.value})")
        self._validate(sample)
        if len(take.queue) >= self._config.queue_capacity:
            return SubmissionResult(SubmissionStatus.FULL, sample.source, None)
        copied = self._copy(sample)  # the only fallible step: finish it before any bookkeeping changes
        index = take.accepted[sample.source]
        take.accepted[sample.source] = index + 1
        take.queue.append(copied)
        take.wake.set()
        return SubmissionResult(SubmissionStatus.ACCEPTED, sample.source, index)

    async def stop_episode(self, episode_id: str) -> EpisodeResult:
        """Drain accepted samples, finish files, check quality, then finalize or retain; repeats join one stop."""
        last = self._last_result
        if self._take is None and last is not None and last.episode_id == episode_id:
            return last
        take = self._active(episode_id, "stop")
        if take.operation == "stop":
            return await self._join(take)
        if take.operation:
            raise StateError(f"episode {episode_id!r} already has a {take.operation} in progress")
        return await self._begin(take, "stop", self._stop_impl(take))

    async def discard_episode(self, episode_id: str) -> EpisodeResult:
        """Close admission, settle all work, and delete only this episode's unfinished output."""
        take = self._active(episode_id, "discard")
        if take.operation == "discard":
            return await self._join(take)
        if take.operation:
            raise StateError(f"episode {episode_id!r} already has a {take.operation} in progress")
        return await self._begin(take, "discard", self._discard_impl(take))

    async def interrupt_episode(self, episode_id: str, reason: str) -> EpisodeResult:
        """Close admission and RETAIN the take's partial output as recovery-required, with
        ``reason`` as its diagnostic -- for a failure in the external pipeline feeding
        ``submit_sample`` (lost/dropped samples, a conversion error) that the service itself
        has no way to detect, as opposed to a write failure the service already reports and
        retains on its own. Never deletes anything -- unlike ``discard_episode``, which is for
        a deliberate, explicit "throw this take away" decision, not a pipeline-caused failure
        whose data may still be partially usable. Repeated interruption for the same take joins
        its existing work; it cannot interrupt a replacement take or race an already-owned
        finish step."""
        take = self._active(episode_id, "interrupt")
        if take.operation == "interrupt":
            return await self._join(take)
        if take.operation:
            raise StateError(f"episode {episode_id!r} already has a {take.operation} in progress")
        return await self._begin(take, "interrupt", self._interrupt_impl(take, reason))

    def pending_episodes(self) -> tuple[PendingEpisode, ...]:
        """Unresolved episodes found on disk (or retained after a failure), in index order."""
        return tuple(sorted(self._pending.values(), key=lambda p: p.episode_index))

    async def recover_episode(self, episode_id: str) -> EpisodeResult:
        """Resume the safe disk step of a sealed/checked pending episode; ambiguous writes are refused."""
        take = self._pending_take(episode_id, "recovering")
        if take.manifest.storage_phase == "WRITING":
            raise StateError(
                f"episode {episode_id!r} is still marked WRITING, so its completeness is unknown and it cannot be "
                "recovered; discard it with discard_pending and record again"
            )
        if not self._same_configuration(take.manifest):
            raise StateError(f"episode {episode_id!r} was recorded with a different configuration")
        take.error = None
        return await self._begin(take, "recover", self._finish_impl(take))

    async def discard_pending(self, episode_id: str) -> EpisodeResult:
        """Remove one pending episode's unfinished output once nothing is working on it."""
        take = self._pending_take(episode_id, "discarding")
        return await self._begin(take, "discard", self._discard_impl(take))

    def snapshot(self) -> RecordingSnapshot:
        """Immutable facts about the session, current episode, counts, last result and pending data."""
        take = self._take
        info = None
        counts: tuple[SourceCounts, ...] = ()
        if take is not None:
            m = take.manifest
            info = EpisodeInfo(m.session_id, m.episode_id, m.episode_index, self._root(m.pending_path))
            counts = self._counts(take)
        issues = [*self._issues, *([self._fatal_error] if self._fatal_error else [])]
        return RecordingSnapshot(
            self._session, self._phase, info, counts, self._last_result, self.pending_episodes(), tuple(issues)
        )

    async def close(self) -> CloseResult:
        """Stop admitting samples and settle owned work within one deadline; report whether all was released."""
        if self._close_task is None:
            self._close_task = self._own(self._close_impl(), "close")
        task = self._close_task
        await asyncio.wait({task})
        return task.result()

    # -- validation and copying ---------------------------------------------------

    def _validate(self, sample: NumericSample | VideoSample) -> None:
        """Reject, with a specific error, any sample that does not match its declared source schema."""
        if not isinstance(sample, NumericSample | VideoSample):
            raise ValidationError(f"unsupported sample type {type(sample).__name__}")
        kind = self._source_kinds.get(sample.source)
        if kind is None:
            raise ValidationError(f"unknown source {sample.source!r}")
        for stamp in (sample.capture_time, sample.receive_time):
            if not isinstance(stamp, int | float) or not np.isfinite(stamp):
                raise ValidationError(f"{sample.source}: timestamps must be finite numbers")
        if isinstance(sample, NumericSample):
            if kind != "numeric":
                raise ValidationError(f"{sample.source} is a video source")
            declared = {f.name: f for f in self._numeric[sample.source].fields}
            if set(sample.fields) != set(declared):
                raise ValidationError(f"{sample.source}: fields {sorted(sample.fields)} != {sorted(declared)}")
            for name, field_spec in declared.items():
                value = np.asarray(sample.fields[name])
                if value.shape != field_spec.shape or value.dtype != np.dtype(field_spec.dtype):
                    raise ValidationError(
                        f"{sample.source}/{name}: {value.dtype}{value.shape} "
                        f"!= {field_spec.dtype}{field_spec.shape}"
                    )
        else:
            if kind != "video":
                raise ValidationError(f"{sample.source} is a numeric source")
            spec = self._video[sample.source]
            if not isinstance(sample.pixel_format, PixelFormat):
                raise ValidationError(f"{sample.source}: pixel_format must be a PixelFormat")
            image = sample.image
            if image.dtype != np.uint8 or image.shape != (spec.height, spec.width, 3):
                raise ValidationError(
                    f"{sample.source}: image {image.dtype}{image.shape} != uint8{(spec.height, spec.width, 3)}"
                )
            if spec.intrinsics:
                if sample.intrinsics is None:
                    raise ValidationError(f"{sample.source}: 3x3 intrinsics required")
                try:
                    matrix = np.asarray(sample.intrinsics, dtype=np.float64)
                except (TypeError, ValueError) as exc:
                    raise ValidationError(f"{sample.source}: intrinsics must be numeric: {exc}") from exc
                if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
                    raise ValidationError(f"{sample.source}: intrinsics must be a finite 3x3 matrix")
            elif sample.intrinsics is not None:
                raise ValidationError(f"{sample.source}: intrinsics not declared for this source")

    @staticmethod
    def _copy(sample: NumericSample | VideoSample) -> NumericSample | VideoSample:
        """Copy mutable arrays once so the caller may reuse its buffers after submission."""
        if isinstance(sample, NumericSample):
            return NumericSample(
                sample.source,
                float(sample.capture_time),
                float(sample.receive_time),
                {k: np.array(v, copy=True) for k, v in sample.fields.items()},
            )
        intrinsics = (
            None if sample.intrinsics is None else np.array(sample.intrinsics, dtype=np.float64, copy=True)
        )
        return VideoSample(
            sample.source,
            np.array(sample.image, copy=True),
            sample.pixel_format,
            float(sample.capture_time),
            float(sample.receive_time),
            intrinsics,
        )

    # -- task and call ownership --------------------------------------------------

    def _own(self, work: Coroutine[Any, Any, Any], name: str | None = None) -> asyncio.Task[Any]:
        """Start ``work`` as a task the service retains until it ends; an escaping exception is a defect."""
        try:
            task = asyncio.get_running_loop().create_task(work, name=name)
        except RuntimeError:
            work.close()
            raise
        self._tasks.add(task)
        task.add_done_callback(self._on_task_finished)
        return task

    def _on_task_finished(self, task: asyncio.Task[Any]) -> None:
        """Forget a finished task; an exception that escaped it is a service defect and stops admission."""
        self._tasks.discard(task)
        if not task.cancelled() and task.exception() is not None:
            error = task.exception()
            logger.error("recording service defect: %s", error, exc_info=error)
            self._fatal_error = f"service failure: {error}"
            self._closing = True

    async def _run(self, work: Coroutine[Any, Any, Any]) -> Any:
        """Run lifecycle ``work`` as an owned task; cancelling the caller does not cancel the work."""
        task = self._own(self._outcome(work))
        await asyncio.wait({task})
        value, error = task.result()
        if error is not None:
            raise error
        return value

    @staticmethod
    async def _outcome(work: Awaitable[Any]) -> tuple[Any, Exception | None]:
        """Return a failure as a value so the owning task never ends with an unobserved exception."""
        try:
            return await work, None
        except Exception as exc:  # noqa: BLE001 - re-raised to the waiting caller by the awaiting side
            return None, exc

    async def _begin(self, take: _Take, name: str, work: Coroutine[Any, Any, EpisodeResult]) -> EpisodeResult:
        """Start a lifecycle operation on the take as an owned task and wait for its result."""
        take.operation = name
        # The admission barrier is one synchronous step: no sample can be accepted after the command.
        take.admission_closed = True
        take.operating = True
        take.idle.clear()
        take.operation_task = self._own(self._operate(take, work), f"{name}-{take.episode_id}")
        return await self._join(take)

    async def _operate(
        self, take: _Take, work: Coroutine[Any, Any, EpisodeResult]
    ) -> tuple[Any, Exception | None]:
        """Run a lifecycle operation holding the take busy; return its result or error."""
        try:
            result, error = await self._outcome(work)
        finally:
            take.operating = False
            take.operation = ""
            self._refresh_idle(take)
        if error is None:
            self._last_result = result
        return result, error

    async def _join(self, take: _Take) -> EpisodeResult:
        """Wait for the take's lifecycle operation and return its result (the same one for every joiner)."""
        task = take.operation_task
        assert task is not None
        await asyncio.wait({task})
        value, error = task.result()
        if error is not None:
            raise error
        return value

    def _refresh_idle(self, take: _Take) -> None:
        """The take is idle exactly when no operation owns it and no storage call is still running."""
        if not take.operating and take.in_flight == 0:
            take.idle.set()

    def _call_finished(self, take: _Take, work: Coroutine[Any, Any, Any], call: asyncio.Task[Any]) -> None:
        """A storage call really ended: the take may become idle."""
        if call.cancelled():
            work.close()
        take.in_flight -= 1
        self._refresh_idle(take)

    async def _call(self, take: _Take, fn: Callable[..., Any], *args: Any) -> Any:
        """Run ``fn`` on the take's storage thread as a service-owned call; a cancelled waiter never cancels it."""
        loop = asyncio.get_running_loop()
        work = self._outcome(loop.run_in_executor(take.executor, fn, *args))
        take.in_flight += 1
        take.idle.clear()
        try:
            call = self._own(work)
        except BaseException:
            take.in_flight -= 1
            work.close()
            raise
        call.add_done_callback(lambda done: self._call_finished(take, work, done))
        await asyncio.wait({call})
        value, error = call.result()
        if error is not None:
            raise error
        return value

    # -- open and start -----------------------------------------------------------

    def _validate_config(self) -> None:
        """Refuse a configuration that cannot be recorded with, before any effect."""
        cfg = self._config
        if cfg.quarantine:
            raise ConfigurationError("quarantine routing is not supported")
        names = [s.name for s in cfg.numeric_sources] + [s.name for s in cfg.video_sources]
        if not names:
            raise ConfigurationError("at least one source must be configured")
        if len(set(names)) != len(names):
            raise ConfigurationError(f"duplicate source names: {names}")
        for name in names:
            if not name or "/" in name or "\\" in name or name.startswith("."):
                raise ConfigurationError(f"source name {name!r} is not a safe path component")
        missing = [s for s in cfg.qc.required_sources if s not in names]
        if missing:
            raise ConfigurationError(f"required sources not configured: {missing}")
        if cfg.routing is not None and (problems := cfg.routing.problems()):
            raise ConfigurationError("routing: " + "; ".join(problems))
        try:  # first, so the capture checks below only ever see well-formed source declarations
            validate_source_specs(cfg.numeric_sources, cfg.video_sources)
        except ValueError as exc:
            raise ConfigurationError(str(exc)) from exc
        if cfg.qc.capture is not None:
            problems = cfg.qc.capture.problems(cfg.numeric_sources, cfg.video_sources)
            # A pose stream that is not a required source may be empty, and an empty stream cannot freeze.
            problems += [
                f"pose stream source {source!r} must be a required source"
                for source in dict.fromkeys(s for s, _ in cfg.qc.capture.pose_streams)
                if source not in cfg.qc.required_sources
            ]
            if problems:
                raise ConfigurationError("capture QC: " + "; ".join(problems))
        for vspec in cfg.video_sources:
            if vspec.encoder not in _ENCODERS:
                raise ConfigurationError(f"{vspec.name}: unknown encoder {vspec.encoder!r}")
            if vspec.width <= 0 or vspec.height <= 0 or vspec.fps <= 0:
                raise ConfigurationError(f"{vspec.name}: width, height and fps must be positive")
        if cfg.queue_capacity <= 0 or cfg.close_deadline_seconds <= 0:
            raise ConfigurationError("queue_capacity and close_deadline_seconds must be positive")
        for source, seconds in (cfg.latency or {}).items():
            if source not in names:
                raise ConfigurationError(f"latency names an unknown source {source!r}")
            if (
                isinstance(seconds, bool)
                or not isinstance(seconds, int | float)
                or not 0 <= seconds < float("inf")
            ):
                raise ConfigurationError(f"latency for {source!r} must be a finite number of seconds >= 0")
        asset_names = [a.name for a in cfg.calibration_assets]
        if len(set(asset_names)) != len(asset_names):
            raise ConfigurationError(f"duplicate calibration asset names: {asset_names}")
        for asset in cfg.calibration_assets:
            if not asset.name or Path(asset.name).name != asset.name or asset.name.startswith("."):
                raise ConfigurationError(f"calibration asset name {asset.name!r} is not a safe file name")
            if asset.name in names or asset.name.startswith("episode_") or asset.name == SESSION_CONFIG_FILE:
                raise ConfigurationError(f"calibration asset name {asset.name!r} collides with recorded data")

    async def _open_impl(self, session_id: str | None) -> SessionInfo:
        """Lock the session directory, scan its manifests and publish the pending set."""
        if self._closing or self._phase is RecordingPhase.CLOSED:
            raise StateError("service is closed")
        if self._session is not None:
            raise StateError("a session is already open")
        self._validate_config()
        root = Path(self._config.output_root)
        root.mkdir(parents=True, exist_ok=True)
        is_new = session_id is None
        if is_new:
            for asset in self._config.calibration_assets:
                if asset.required and not Path(asset.path).is_file():
                    raise ConfigurationError(
                        f"required calibration asset {asset.name!r} not found at {asset.path}"
                    )
            session_id = f"session_{time.strftime('%Y%m%d-%H%M%S')}_{uuid.uuid4().hex[:6]}"
            path = root / session_id
            path.mkdir()
        else:
            path = root / session_id
            if "/" in session_id or session_id.startswith(".") or not path.is_dir():
                raise ConfigurationError(f"no session {session_id!r} under {root}")
        try:
            self._lock = SessionLock(path)
        except BlockingIOError as exc:
            raise StateError(f"session {session_id!r} is owned by another recorder") from exc
        self._session = SessionInfo(session_id, root, path)
        try:
            self._prepare_session_assets(path, is_new)
        except Exception:
            try:
                if is_new:
                    # Roll back only the directory this call created. New session ids are unique and random, so
                    # no other opener is using this path; removal starts while the lock is still held (rmtree
                    # itself unlinks the lock file partway, so the lock is not what protects the rollback).
                    shutil.rmtree(path, ignore_errors=True)
            finally:
                self._lock.release()
                self._lock, self._session = None, None
            raise
        manifests, problems = scan_manifests(path)
        self._issues.extend(problems)
        self._blocking = list(problems)
        for manifest in manifests:
            if manifest.storage_phase in UNRESOLVED_PHASES:
                self._add_pending(manifest)
        return self._session

    def _prepare_session_assets(self, session_path: Path, is_new: bool) -> None:
        """Preserve caller-supplied assets in a new session; verify the session's own copies on reopen.

        A new session copies each supplied asset in and records its hash. A reopened session never consults
        the original input paths: it requires the preserved copy to exist and match the recorded hash.
        """
        assets = self._config.calibration_assets
        record_path = session_path / MANIFEST_DIR / CALIBRATION_RECORD
        if is_new:
            record: dict[str, str | None] = {}
            for asset in assets:
                source = Path(asset.path)
                if source.is_file():
                    shutil.copy2(source, session_path / asset.name)
                    record[asset.name] = _sha256(session_path / asset.name)
                elif asset.required:  # it vanished after the up-front check: never downgrade it to optional
                    raise ConfigurationError(
                        f"required calibration asset {asset.name!r} is not available at {source}"
                    )
                else:
                    record[asset.name] = None
                    self._issues.append(f"optional calibration asset {asset.name!r} not found at {source}")
            write_json_atomic(record_path, record)
            if self._config.station_label:
                write_yaml_atomic(
                    session_path / SESSION_CONFIG_FILE,
                    {"meta": {"egocap_station_name": self._config.station_label}},
                )
            return
        record = json.loads(record_path.read_text()) if record_path.is_file() else {}
        for asset in assets:
            recorded, preserved = record.get(asset.name), session_path / asset.name
            if recorded is not None and preserved.is_file() and _sha256(preserved) == recorded:
                continue
            problem = f"calibration asset {asset.name!r}: session copy is missing, changed or was never preserved"
            if asset.required:
                raise ConfigurationError(problem)
            self._issues.append("optional " + problem)

    def _seed_group_assets(self, session_path: Path, group: str) -> None:
        """Copy the session's frozen config.yaml and preserved calibration assets into a routed group.

        Idempotent: a later take routed into the same group re-checks every file but copies nothing new.
        A group's existing config.yaml must come from the same station as this session, or this
        assignment is refused outright rather than finalizing under conflicting station metadata.
        Compares only the station key, not the whole structure: the EOD Episode Rename backfill
        (``update_episode_task_description.py``) legitimately rewrites a group's config.yaml meta
        after the fact (task description, studio), so a later take routed into that same group must
        not be refused over a difference the rename itself introduced.
        """
        group_dir = session_path.parent / group
        session_config = session_path / SESSION_CONFIG_FILE
        if session_config.is_file():
            session_data = yaml.safe_load(session_config.read_text()) or {}
            group_config = group_dir / SESSION_CONFIG_FILE
            if group_config.is_file():
                group_data = yaml.safe_load(group_config.read_text()) or {}
                session_station = session_data.get("meta", {}).get("egocap_station_name")
                group_station = group_data.get("meta", {}).get("egocap_station_name")
                if group_station != session_station:
                    raise ConfigurationError(
                        f"group {group!r}: existing config.yaml station name does not match this session's"
                    )
            else:
                write_yaml_atomic(group_config, session_data)
        record_path = session_path / MANIFEST_DIR / CALIBRATION_RECORD
        record: dict[str, str | None] = json.loads(record_path.read_text()) if record_path.is_file() else {}
        for asset in self._config.calibration_assets:
            recorded_hash = record.get(asset.name)
            if recorded_hash is None:
                continue  # recorded absent (or optional and not yet seen) -- never required at the group either
            session_copy = session_path / asset.name
            if not session_copy.is_file() or _sha256(session_copy) != recorded_hash:
                raise ConfigurationError(
                    f"session calibration asset {asset.name!r} was preserved but is now missing or changed"
                )
            group_copy = group_dir / asset.name
            if group_copy.is_file():
                if _sha256(group_copy) != recorded_hash:
                    raise ConfigurationError(
                        f"group {group!r}: existing {asset.name!r} does not match this session's"
                    )
                continue
            _copy_file_atomic(session_copy, group_copy)

    def _same_configuration(self, manifest: EpisodeManifest) -> bool:
        """Whether this service may finish the take.

        A SEALED take is still to be judged by the current capture policy, so the whole configuration must match.
        A CHECKED take already carries its decision and the policy that made it, so only the layout-affecting parts
        must match: a new capture policy or algorithm cannot strand it.
        """
        if manifest.storage_phase == "CHECKED":
            return (manifest.layout_digest or manifest.config_digest) == _layout_digest(self._config)
        return manifest.config_digest == _digest(self._config)

    def _add_pending(self, manifest: EpisodeManifest, issues: tuple[str, ...] = ()) -> None:
        """Record an unresolved episode and whether a safe recovery step exists for it."""
        same_config = self._same_configuration(manifest)
        problems = list(issues)
        if manifest.storage_phase == "WRITING":
            problems.append("interrupted while writing; completeness cannot be established")
        elif not same_config:
            problems.append("recorded with a different configuration")
        self._pending[manifest.episode_id] = PendingEpisode(
            manifest.session_id,
            manifest.episode_id,
            manifest.episode_index,
            self._root(manifest.pending_path),
            manifest.storage_phase,
            manifest.storage_phase in ("SEALED", "CHECKED") and same_config,
            tuple(problems),
        )

    def _root(self, relative: str) -> Path:
        """Absolute path of a session-relative manifest path (the session directory for an empty one)."""
        assert self._session is not None
        return self._session.session_path / relative

    def _source_declarations(self) -> list[dict[str, Any]]:
        """Ordered source declarations recorded in each manifest."""
        cfg = self._config
        return [{"kind": "numeric", **asdict(s)} for s in cfg.numeric_sources] + [
            {"kind": "video", **asdict(s)} for s in cfg.video_sources
        ]

    async def _start_impl(self, task: str, operator: str) -> EpisodeInfo:
        """Checkpoint a WRITING manifest, then open the real store; enter RECORDING only on success."""
        if self._session is None or self._closing:
            raise StateError("no open session")
        if self._phase is not RecordingPhase.IDLE:
            raise StateError(f"cannot start an episode while {self._phase.value}")
        if self._pending:
            raise StateError(f"unresolved pending episodes: {sorted(self._pending)}")
        if self._blocking:
            raise StateError(f"unreadable episode manifests must be resolved first: {self._blocking}")
        session = self._session
        index = next_episode_index(session.session_path)
        pending_name, final_name = episode_dir_names(index)
        manifest = EpisodeManifest(
            session_id=session.session_id,
            episode_id=f"{session.session_id}/episode_{index}",
            episode_index=index,
            config_digest=_digest(self._config),
            layout_digest=_layout_digest(self._config),
            task=task,
            operator=operator,
            sources=self._source_declarations(),
            pending_path=pending_name,
            final_path=final_name,
            # Frozen when the take is created, so interrupted checking resumes under this exact policy.
            capture_policy=policy_to_dict(self._config.qc.capture) if self._config.qc.capture else None,
            # The studio is decided now, so a later change of task cannot reroute a take waiting for QC.
            studio=resolve_studio(task, self._config.routing) if self._config.routing is not None else None,
        )
        take = _Take(manifest, list(self._source_kinds), None)
        self._phase = RecordingPhase.STARTING
        self._take = take
        self._start_task = asyncio.current_task()
        try:
            await self._call(take, write_manifest, session.session_path, manifest)
            take.released = False
            take.store = await self._call(
                take,
                EpisodeStore,
                session.session_path / pending_name,
                self._config.numeric_sources,
                self._config.video_sources,
            )
            await self._call(take, take.store.start_encoders)
            await self._call(
                take,
                take.store.write_attributes,
                {
                    IDENTITY_ATTR: manifest.episode_id,
                    "session_id": session.session_id,
                    "video_encoder_options": take.store.encoder_options,
                },
            )
        except Exception as exc:
            logger.warning("episode start failed: %s", exc)
            take.released = take.store is None
            if manifest_path(session.session_path, index).exists():
                await self._retain(take, f"start failed: {exc}")
            else:
                self._take = None
                self._phase = RecordingPhase.IDLE
            raise
        self._phase = RecordingPhase.RECORDING
        take.worker = self._own(self._worker(take), f"worker-{index}")
        return EpisodeInfo(session.session_id, manifest.episode_id, index, session.session_path / pending_name)

    # -- worker -------------------------------------------------------------------

    async def _worker(self, take: _Take) -> None:
        """Write accepted samples in admission order until admission closes and the queue is empty."""
        while True:
            if take.queue:
                sample = take.queue.popleft()
                try:
                    await self._call(take, self._write, take, sample)
                except Exception as exc:  # noqa: BLE001 - a write failure retains the take; stop reports it
                    logger.error("episode write failed: %s", exc)
                    take.error = f"write failed: {exc}"
                    take.admission_closed = True
                    return
                continue
            if take.admission_closed:
                return
            take.wake.clear()
            await take.wake.wait()

    @staticmethod
    def _write(take: _Take, sample: NumericSample | VideoSample) -> None:
        """Append one sample to the store (runs on the storage thread) and record its progress."""
        assert take.store is not None
        if isinstance(sample, NumericSample):
            take.store.append_numeric(sample.source, sample.capture_time, sample.receive_time, sample.fields)
        else:
            frame = sample.image
            if sample.pixel_format is PixelFormat.RGB8:
                frame = np.ascontiguousarray(frame[..., ::-1])
            take.store.append_video(
                sample.source, frame, sample.capture_time, sample.receive_time, sample.intrinsics
            )
        take.written[sample.source] += 1
        span = take.span.setdefault(sample.source, [sample.capture_time, sample.capture_time])
        span[1] = sample.capture_time

    # -- stop, finish, discard ----------------------------------------------------

    async def _stop_impl(self, take: _Take) -> EpisodeResult:
        """Close admission at one event-loop step, drain the queue, then run the sealed-episode chain."""
        self._phase = RecordingPhase.STOPPING
        take.admission_closed = True
        take.wake.set()
        if take.worker is not None:
            await asyncio.wait({take.worker})
        if take.error:
            return await self._retain(take, take.error)
        return await self._finish_impl(take)

    async def _finish_impl(self, take: _Take) -> EpisodeResult:
        """Seal, check and finalize (or reject) the take, resuming from whichever phase its manifest records."""
        m = take.manifest
        session_path = self._root("")
        try:
            if m.storage_phase == "WRITING":
                assert take.store is not None
                counts = await self._call(take, take.store.seal)
                take.released = True
                mismatch = [
                    f"{s}: accepted {take.accepted[s]}, sealed {counts.get(s)}"
                    for s in take.accepted
                    if counts.get(s) != take.accepted[s] or take.written[s] != take.accepted[s]
                ]
                if mismatch:
                    return await self._retain(take, "count mismatch: " + "; ".join(mismatch))
                try:
                    await self._checkpoint(
                        take,
                        counts={
                            s: {"accepted": take.accepted[s], "written": take.written[s]} for s in take.accepted
                        },
                        duration_seconds=take.duration(),
                        storage_phase="SEALED",
                    )
                except Exception as exc:  # noqa: BLE001 - reported as a clear, final failure of this take
                    # Not recoverable by design: the files are sealed but the take is still WRITING on disk.
                    return await self._retain(
                        take,
                        f"the take was written and sealed, but its SEALED manifest could not be saved ({exc}); "
                        "it cannot be resumed, so discard it with discard_pending and record again",
                    )
            if self._take is take:
                self._phase = RecordingPhase.FINALIZING
            # Establish where the episode really is (and that it is ours) before any check, write or rename.
            try:
                where = await self._call(take, locate_episode, session_path, m)
            except FileNotFoundError:
                # Only a take whose removal was already decided can have no data left: just its record remains.
                if not (m.storage_phase == "CHECKED" and m.finalization == FINALIZATION_REMOVE):
                    raise
                where = "removed"
            if m.storage_phase == "SEALED":
                if where != "pending":
                    raise FileNotFoundError("a sealed episode must still be in its pending directory")
                report = await self._call(
                    take,
                    check_episode,
                    session_path / m.pending_path,
                    self._config.numeric_sources,
                    self._config.video_sources,
                    self._config.qc.required_sources,
                    {s: c["accepted"] for s, c in m.counts.items()},
                )
                if report.outcome is Outcome.ERROR:
                    return await self._retain(take, "quality check errored: " + "; ".join(report.issues))
                capture = None
                if report.outcome is Outcome.PASSED and self._config.qc.capture is not None:
                    # Capture quality is judged only on integrity-valid data; the checker reports its own errors.
                    capture = await self._call(
                        take, check_capture_quality, session_path / m.pending_path, self._config.qc.capture
                    )
                decision = decide(capture)
                if decision is QcDecision.ERROR:
                    # Unlike a measured failure, an unmeasured gate says nothing about the take: keep it SEALED and
                    # retained so a recovery re-runs the checks (or the operator discards it), never as unflagged
                    # data.
                    assert capture is not None
                    return await self._retain(take, "capture quality check errored: " + "; ".join(capture.issues))
                await self._checkpoint(
                    take,
                    quality=report_to_dict(report),
                    capture_quality=report_to_dict(capture) if capture else None,
                    qc_decision=decision.value,
                    finalization=finalization_for(report.outcome, decision),
                    storage_phase="CHECKED",
                )
            if m.storage_phase == "CHECKED":
                if m.finalization == FINALIZATION_KEEP:
                    if m.studio is not None and m.assignment is None:
                        await self._assign_output(take)
                    # After a crash between the rename and the FINALIZED checkpoint the pending directory is
                    # gone and the final one already holds the data: never rewrite it, only verify its identity.
                    if where == "pending":
                        await self._call(
                            take, write_episode_attributes, session_path / m.pending_path, self._attributes(m)
                        )
                    final = await self._call(take, finalize_episode_dir, session_path, m)
                    await self._checkpoint(take, storage_phase="FINALIZED")
                    if m.assignment is not None:
                        # Best effort, after the take is durably FINALIZED: the totals are derived from the
                        # finalized episodes, so a failure here (an unreadable sibling, a damaged group stamp)
                        # must not strand a take that is already published. The next take finalized in the group
                        # recomputes them.
                        try:
                            await self._call(take, refresh_group_totals, final.parent)
                        except Exception as exc:  # noqa: BLE001 - reported on the kept result, never fatal
                            logger.error("refreshing totals of %s failed: %s", final.parent, exc)
                            m.issues.append(f"group totals were not refreshed: {exc}")
                elif m.finalization == FINALIZATION_REMOVE:
                    # Remove only this take's owned data; a failed removal retains the take (visible, recoverable).
                    if where == "pending":
                        await self._call(take, remove_owned_episode, session_path, m)
                    await self._checkpoint(take, storage_phase="REJECTED")
                elif m.finalization == FINALIZATION_RETAIN:
                    await self._checkpoint(take, storage_phase="REJECTED")  # the data stays for resolution
                else:
                    raise ValueError(f"unknown finalization {m.finalization!r}")
        except Exception as exc:  # noqa: BLE001 - any storage failure retains the take with its diagnostic
            logger.error("finishing episode %s failed: %s", m.episode_id, exc)
            return await self._retain(take, f"{m.storage_phase.lower()} step failed: {exc}")
        return self._conclude(take)

    async def _assign_output(self, take: _Take) -> None:
        """Freeze the kept take's group and slot durably, before anything is published to them."""
        m = take.manifest
        session_path = self._root("")

        def assign() -> tuple[str, int]:
            """Claim the group and slot on the take's worker thread, skipping slots other manifests hold."""
            manifests, _ = scan_manifests(session_path)
            # A discarded take never published its slot, so it no longer holds one; unresolved takes still do.
            reserved = [
                (x.assignment["group"], x.assignment["slot"])
                for x in manifests
                if x.assignment is not None and x.storage_phase != "DISCARDED"
            ]
            group, slot = assign_output(
                session_path, m.session_id, m.studio, m.task, m.operator, reserved, datetime.datetime.now()
            )
            self._seed_group_assets(session_path, group)
            return group, slot

        group, slot = await self._call(take, assign)
        await self._checkpoint(take, assignment={"group": group, "slot": slot})

    async def _checkpoint(self, take: _Take, **changes: Any) -> None:
        """Make ``changes`` durable in the manifest, and publish them in memory only after the write succeeded.

        The intended record is a detached copy; if the write fails the in-memory manifest keeps the last durable
        phase, so a retry in this process and a restart see the same state.
        """
        snapshot = replace(take.manifest, **changes)
        await self._call(take, write_manifest, self._root(""), snapshot)
        for name, value in changes.items():
            setattr(take.manifest, name, value)

    def _attributes(self, m: EpisodeManifest) -> dict[str, Any]:
        """Dataset-level metadata written to a kept episode, using the names existing readers consume.

        Only checks that actually ran are reported: the mandatory integrity check is recorded, and capture
        QC is either its real per-check results and decision or explicitly SKIPPED, never implied to have
        passed. ``bad_episode`` marks kept data the capture checks flagged, so a consumer never has to infer it.
        """
        quality = m.quality or {}
        decision = QcDecision(m.qc_decision) if m.qc_decision else QcDecision.SKIPPED  # older records ran none
        report = report_from_dict(m.capture_quality) if m.capture_quality else None
        rows = [
            c["written"]
            for s, c in m.counts.items()
            if any(d["name"] == s and d["kind"] == "video" for d in m.sources)
        ]
        rows = rows or [c["written"] for c in m.counts.values()]
        attributes = {
            "task_description": m.task,
            "instruction_language": m.task,
            "operator_name": m.operator,
            "episode_length": max(rows, default=0),
            "episode_duration": m.duration_seconds,
            # The time the episode is finalized, not the time recording started. A take
            # recovered after a crash is not resumed, so it carries its recovery date.
            "recorded_at": time.strftime("%Y-%m-%d; %H-%M-%S"),
            "integrity_check_passed": quality.get("outcome") == "PASSED",
            "integrity_check_issues": list(quality.get("issues", [])),
            **quality_attributes(decision, report, m.capture_policy),
            "config_digest": m.config_digest,
            "sources": m.sources,
        }
        if m.assignment is not None:
            attributes.update(
                {"studio": m.studio, "route_group": m.assignment["group"], "output_index": m.assignment["slot"]}
            )
        if self._config.latency:
            attributes["latency"] = dict(self._config.latency)
        if self._config.calibration_assets:
            session_path = self._root("")
            attributes["calibration_assets"] = {
                a.name: _sha256(session_path / a.name)
                for a in self._config.calibration_assets
                if (session_path / a.name).is_file()
            }
        return attributes

    async def _discard_impl(self, take: _Take) -> EpisodeResult:
        """Settle the worker and encoders, then remove only this episode's unfinished data."""
        take.admission_closed = True
        take.wake.set()
        if take.worker is not None:
            await asyncio.wait({take.worker})
        m = take.manifest
        session_path = self._root("")
        try:
            if take.store is not None and not take.released:
                take.released = await self._call(take, take.store.abort)
                if not take.released:
                    return await self._retain(take, "encoder release unconfirmed; data kept, restart the recorder")
            await self._call(take, remove_owned_episode, session_path, m)
            await self._checkpoint(take, storage_phase="DISCARDED")
        except Exception as exc:  # noqa: BLE001 - a failed removal leaves the take recoverable with its diagnostic
            logger.error("discarding episode %s failed: %s", m.episode_id, exc)
            return await self._retain(take, f"discard failed: {exc}")
        return self._conclude(take)

    async def _interrupt_impl(self, take: _Take, reason: str) -> EpisodeResult:
        """Close admission, let whatever was already accepted finish writing, then retain
        (never delete) the take's output as recovery-required -- the external-pipeline
        counterpart to ``_stop_impl``'s own write-failure path, which already calls
        ``_retain`` for a reason the service detected itself."""
        take.admission_closed = True
        take.wake.set()
        if take.worker is not None:
            await asyncio.wait({take.worker})
        return await self._retain(take, reason)

    async def _retain(self, take: _Take, message: str) -> EpisodeResult:
        """Keep the take's data and release its encoders; the episode now needs recovery or discard."""
        take.error = message
        m = take.manifest
        if take.store is not None and not take.released:
            try:
                take.released = await self._call(take, take.store.abort)
            except Exception as exc:  # noqa: BLE001 - release failure is reported; the take stays unreleased
                message += f"; encoder release failed: {exc}"
            if not take.released:
                message += "; encoder release unconfirmed"
        m.issues = [*m.issues, message]
        if m.storage_phase in UNRESOLVED_PHASES:
            try:
                await self._call(take, write_manifest, self._root(""), m)
            except Exception as exc:  # noqa: BLE001 - the in-memory record still reports the failure
                message += f"; manifest checkpoint failed: {exc}"
        self._add_pending(m, (message,))
        self._retained[m.episode_id] = take
        if self._take is take:
            self._phase = RecordingPhase.RECOVERY_REQUIRED
        return self._result(take, Disposition.RECOVERY_REQUIRED, (message,))

    def _conclude(self, take: _Take) -> EpisodeResult:
        """The take reached a terminal storage phase: clear pending state and return to idle."""
        m = take.manifest
        self._pending.pop(m.episode_id, None)
        self._retained.pop(m.episode_id, None)
        disposition = {
            "FINALIZED": Disposition.KEPT,
            "REJECTED": Disposition.REJECTED,
            "DISCARDED": Disposition.DISCARDED,
        }[m.storage_phase]
        result = self._result(take, disposition, tuple(m.issues))
        if self._take is take:
            self._take = None
            self._phase = RecordingPhase.IDLE
        take.executor.shutdown(wait=False)
        return result

    def _result(self, take: _Take, disposition: Disposition, issues: tuple[str, ...]) -> EpisodeResult:
        """Build the immutable result for the take's current state."""
        m = take.manifest
        # Built from the durable, worker-validated assignment: assembling a result does no filesystem I/O, so it
        # cannot fail after the take was already finalized.
        final = None
        if disposition is Disposition.KEPT:
            routed = self._root("").parent / destination_path(m.assignment) if m.assignment else None
            final = routed or self._root(m.final_path)
        # A resolved rejection has removed its data; an integrity rejection keeps its pending directory.
        removed = disposition is Disposition.DISCARDED or (
            disposition is Disposition.REJECTED and m.finalization == FINALIZATION_REMOVE
        )
        pending = None if disposition is Disposition.KEPT or removed else self._root(m.pending_path)
        quality = report_from_dict(m.quality) if m.quality else None
        return EpisodeResult(
            m.session_id,
            m.episode_id,
            m.episode_index,
            disposition,
            final,
            pending,
            self._counts(take),
            m.duration_seconds or take.duration(),
            quality,
            issues,
            QcDecision(m.qc_decision) if m.qc_decision else None,
            report_from_dict(m.capture_quality) if m.capture_quality else None,
            m.assignment["slot"] if disposition is Disposition.KEPT and m.assignment else None,
        )

    def _counts(self, take: _Take) -> tuple[SourceCounts, ...]:
        """Accepted/written counts per source in configured order."""
        return tuple(SourceCounts(s, take.accepted[s], take.written[s]) for s in take.accepted)

    # -- guards -------------------------------------------------------------------

    def _active(self, episode_id: str, verb: str) -> _Take:
        """The current take if it can be asked to ``verb`` now; otherwise a StateError."""
        take = self._take
        if take is None or take.episode_id != episode_id:
            raise StateError(f"no active episode {episode_id!r}")
        if take.operation:
            return take
        self._require_accepting()
        if self._phase is RecordingPhase.RECOVERY_REQUIRED:
            raise StateError(f"episode {episode_id!r} needs recovery; use recover_episode or discard_pending")
        if self._phase is not RecordingPhase.RECORDING:
            raise StateError(f"cannot {verb} while {self._phase.value}")
        return take

    def _require_accepting(self) -> None:
        """Refuse new lifecycle work once close has begun or the session is released (lock no longer held)."""
        if self._closing or self._phase is RecordingPhase.CLOSED or self._session is None:
            raise StateError("service is closing or closed; no new lifecycle commands are accepted")

    def _pending_take(self, episode_id: str, verb: str) -> _Take:
        """The take for an unresolved episode, once no work is running on it."""
        self._require_accepting()
        if episode_id not in self._pending:
            raise StateError(f"no pending episode {episode_id!r}")
        take = self._retained.get(episode_id)
        if take is not None:
            if not take.idle.is_set():
                raise StateError(
                    f"work on episode {episode_id!r} is still running; wait for it to finish before {verb}"
                )
            return take
        manifest = next((m for m in scan_manifests(self._root(""))[0] if m.episode_id == episode_id), None)
        if manifest is None:
            raise StateError(f"episode {episode_id!r} has no readable manifest")
        take = _Take(manifest, list(self._source_kinds), None)
        for source, counts in manifest.counts.items():
            take.accepted[source] = counts["accepted"]
            take.written[source] = counts["written"]
        self._retained[episode_id] = take
        return take

    # -- close --------------------------------------------------------------------

    async def _close_impl(self) -> CloseResult:
        """Discard an unsaved take, let an accepted stop finish, and settle everything by one deadline."""
        self._closing = True
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self._config.close_deadline_seconds
        issues: list[str] = []
        if self._start_task is not None and not self._start_task.done():
            await asyncio.wait({self._start_task}, timeout=max(0.0, deadline - loop.time()))
        take = self._take
        if take is not None and not take.operation and self._phase is RecordingPhase.RECORDING:
            take.operation = "discard"
            take.operating = True
            take.idle.clear()
            take.operation_task = self._own(self._operate(take, self._discard_impl(take)), "close-discard")
        for target in [t for t in (take, *self._retained.values()) if t is not None]:
            try:
                await asyncio.wait_for(target.idle.wait(), max(0.0, deadline - loop.time()))
            except TimeoutError:
                issues.append(f"{target.episode_id}: work did not settle within the close deadline")
        unfinished = self._unreleased()
        if unfinished:
            issues.append("resources still held; output lock retained")
            self._close_task = None
            self._issues.extend(issues)
            return CloseResult(False, tuple(unfinished), tuple(issues))
        for target in (take, *self._retained.values()):
            if target is not None:
                target.executor.shutdown(wait=False)
        if self._lock is not None:
            self._lock.release()
            self._lock = None
        self._phase = RecordingPhase.CLOSED
        return CloseResult(True, (), tuple(issues))

    def _unreleased(self) -> list[str]:
        """Episodes whose work is still running or whose encoders have not confirmed release."""
        held = []
        for take in (self._take, *self._retained.values()):
            if take is not None and (not take.idle.is_set() or not take.released):
                held.append(take.episode_id)
        return sorted(set(held))
