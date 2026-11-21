# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Manus glove calibration inside CaptureServer.

A :class:`CalibrationSession` occupies the same admission gate as a capture session, so the two are
mutually exclusive from acceptance through cleanup. It runs the standalone ``manus_glove_calibrator``
process through :mod:`isaaccapture_manus_calibration` and translates the worker's events into calibration
messages on the connection's single outbound writer. It emits control and status messages only; no
acquisition, OpenXR session or tracking source is created for it.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from isaaccapture_manus_calibration import protocol as wp
from isaaccapture_manus_calibration.worker import (
    AttemptEnded,
    AttemptError,
    CalibrationAttempt,
    CalibrationOutcome,
    CalibrationWorker,
    OutcomeKind,
)

from . import protocol as p

logger = logging.getLogger("isaaccapture.capture_server.calibration")

WORKER_NAME = "manus_glove_calibrator"

_SIDE_TO_WIRE = {wp.Side.LEFT: p.HandSide.LEFT, wp.Side.RIGHT: p.HandSide.RIGHT}
_SIDE_FROM_WIRE = {value: key for key, value in _SIDE_TO_WIRE.items()}
_PHASE_TO_WIRE = {
    wp.Phase.DISCOVERING: p.CalibrationPhase.CONNECTING,
    wp.Phase.AWAITING_STEP: p.CalibrationPhase.AWAITING_STEP,
    wp.Phase.CAPTURING_STEP: p.CalibrationPhase.CAPTURING_STEP,
    wp.Phase.SAVING: p.CalibrationPhase.SAVING,
    wp.Phase.STOPPING: p.CalibrationPhase.STOPPING,
}
_REASON_TO_WIRE = {
    OutcomeKind.COMPLETED: p.CalibrationStopReason.COMPLETED,
    OutcomeKind.CANCELLED: p.CalibrationStopReason.CANCELLED,
    OutcomeKind.FAILED: p.CalibrationStopReason.FAILED,
    OutcomeKind.UNCERTAIN: p.CalibrationStopReason.UNCERTAIN,
}


@dataclass(frozen=True)
class CalibrationConfig:
    """The MCAL directory on this host and, when one is available, the worker that generates files.

    Generated files are written into this directory as plain ``.mcal`` files. Calibration only writes there;
    it never reads or selects files back.
    """

    directory: Path
    worker_path: Path | None = None


def find_worker(override: str | os.PathLike[str] | None) -> Path | None:
    """The worker executable: the configured override, else the Manus install under ``ISAAC_ROOT``.

    ``None`` when no executable is found, so calibration is simply unavailable and capture is unaffected.
    """
    if override is not None:
        candidate = Path(override)
        return candidate if candidate.is_file() and os.access(candidate, os.X_OK) else None
    root = os.environ.get("ISAAC_ROOT")
    found = shutil.which(WORKER_NAME, path=str(Path(root) / "install" / "plugins" / "manus") if root else "")
    return Path(found) if found else None


class _Outbox(Protocol):
    """The part of the connection's single writer a calibration session uses."""

    closed: bool

    def bind_session(self, stream_ids: list[int], *, on_sent: Any, on_fatal: Any) -> None:
        """Attach this session's streams and its sent and fatal-error callbacks."""

    def publish_control(self, data: dict[str, Any]) -> bool:
        """Queue one control message; False when the writer can no longer take it."""

    def publish_status(self, data: dict[str, Any]) -> None:
        """Queue one status message, replacing an older unsent one."""

    async def quiesce(self) -> None:
        """Wait until everything queued so far has been sent."""


class ConfirmRefused(Exception):
    """A confirmation was not accepted; the active calibration is unchanged."""


class CalibrationSession:
    """One accepted calibration attempt; implements the admission surface a capture session has."""

    def __init__(
        self,
        request: p.StartCalibrationRequest,
        worker: CalibrationWorker,
        outbox: _Outbox,
        owner_ws: Any,
    ) -> None:
        self.session_id = request.session_id
        self.owner_ws = owner_ws
        self.request = request
        self.done = asyncio.Event()
        self.terminal: p.CalibrationStopped | None = None
        self._worker = worker
        self._outbox = outbox
        self._phase = p.CalibrationPhase.CONNECTING
        self._gloves: tuple[p.CalibrationGlove, ...] = ()
        self._step: p.CalibrationStep | None = None
        self._elapsed: float | None = None
        self._accepted: set[str] = set()
        self._tasks: set[asyncio.Task[Any]] = set()
        self._stop_requested = False
        # Taken synchronously, before any await, so the station is occupied the moment the request is accepted.
        self._attempt: CalibrationAttempt = worker.new_attempt(self._on_event)

    @property
    def cleanup_budget_s(self) -> float:
        """The longest a cancelled attempt can take to finish: cancel grace, then terminate, kill and reap."""
        w = self._worker
        return w.cancel_grace_s + w.exit_grace_s + w.term_grace_s + w.kill_wait_s + 2.0

    @property
    def cleanup_ok(self) -> bool:
        """False while the gloves may still be in use, so the station stays unavailable for every session.

        The supervisor sets the latch for a worker that could not be reaped and for one that exited after
        reporting the SDK was not released: a process exit alone is not evidence the gloves are free.
        """
        return self._worker.unreleased is None

    def request_stop(self, reason: Any = None, message: str | None = None) -> None:
        """Cancel the attempt; idempotent. Used for StopCalibration, owner loss and shutdown."""
        if self._stop_requested:
            return
        self._stop_requested = True
        self._spawn(self._attempt.cancel())

    async def run(self) -> None:
        """Run the attempt to its terminal outcome and publish the one terminal message after cleanup."""
        outcome: CalibrationOutcome | None = None
        try:
            self._outbox.bind_session([], on_sent=lambda _stream_id: None, on_fatal=self._on_fatal)
            self._outbox.publish_control(
                p.CalibrationStarted(session_id=self.session_id, sides=self.request.sides).to_json()
            )
            self._publish_snapshot()
            try:
                await self._attempt.start(tuple(_SIDE_FROM_WIRE[side] for side in self.request.sides))
            except AttemptError:
                # Stopped before the worker started: the attempt is already settled as cancelled.
                pass
            except Exception:
                # A failed start settled the attempt; its outcome below carries the reason.
                logger.exception("calibration start failed")
            outcome = await self._attempt.outcome()
        except asyncio.CancelledError:
            await self._attempt.cancel()
            outcome = await self._attempt.outcome()
            raise
        finally:
            await self._finalize(outcome)

    async def handle_confirm(self, token: str) -> None:
        """Start the offered step; the acceptance message itself is published from the worker's event, in order."""
        if token in self._accepted:
            # An accepted token is acknowledged again and never starts a second step.
            self._outbox.publish_control(p.CalibrationStepAccepted(self.session_id, token).to_json())
            return
        prompt = self._attempt.prompt
        if prompt is None or prompt.token != token:
            raise ConfirmRefused("that step is not awaiting confirmation")
        try:
            await self._attempt.confirm(token)
        except AttemptEnded:
            # The attempt ended while the confirm was in flight; the terminal message ends the caller's wait, so
            # no rejection is sent for a request that was not wrong.
            return
        except AttemptError as exc:
            raise ConfirmRefused(str(exc)) from None

    def spawn_confirm(self, token: str, on_refused: Any) -> None:
        """Run a confirmation without holding up the connection's read loop (a stop must still get through)."""

        async def _run() -> None:
            try:
                await self.handle_confirm(token)
            except ConfirmRefused as exc:
                on_refused(str(exc))

        self._spawn(_run())

    def replay(self) -> None:
        """Re-send acceptance and the current snapshot for an identical repeated start."""
        self._outbox.publish_control(p.CalibrationStarted(self.session_id, self.request.sides).to_json())
        self._publish_snapshot()

    def _on_event(self, event: wp.WorkerEvent) -> None:
        """Fold one worker event into the snapshot; called in worker event order."""
        if isinstance(event, wp.Status):
            self._phase = _PHASE_TO_WIRE[event.phase]
            self._gloves = tuple(p.CalibrationGlove(_SIDE_TO_WIRE[g.side], g.glove_id) for g in event.gloves)
            if self._phase in (
                p.CalibrationPhase.CONNECTING,
                p.CalibrationPhase.SAVING,
                p.CalibrationPhase.STOPPING,
            ):
                self._step = None
                self._elapsed = None
            # The prompt that follows carries the step; a snapshot without it would show a prompt-less wait.
            if self._phase is not p.CalibrationPhase.AWAITING_STEP:
                self._publish_snapshot()
        elif isinstance(event, wp.Prompt):
            self._phase = p.CalibrationPhase.AWAITING_STEP
            self._elapsed = None
            self._step = p.CalibrationStep(
                token=event.token,
                side=_SIDE_TO_WIRE[event.side],
                index=event.index,
                count=event.count,
                title=event.title,
                instruction=event.instruction,
                mode=p.CalibrationStepMode.MOTION
                if event.mode is wp.StepMode.MOTION
                else p.CalibrationStepMode.HOLD,
                duration_seconds=event.duration_s,
                continuous=event.continuous,
                estimated_seconds=event.estimated_s,
            )
            self._publish_snapshot()
        elif isinstance(event, wp.StepAccepted):
            self._accepted.add(event.token)
            # Published here, in the worker's own event order, so the next prompt cannot overtake it.
            self._outbox.publish_control(p.CalibrationStepAccepted(self.session_id, event.token).to_json())
        elif isinstance(event, wp.StepProgress):
            self._elapsed = event.elapsed_s
            self._publish_snapshot()

    def _publish_snapshot(self) -> None:
        """Coalesced: each status is a complete replacement, so only the latest needs to arrive."""
        self._outbox.publish_status(
            p.CalibrationStatus(
                session_id=self.session_id,
                phase=self._phase,
                gloves=self._gloves,
                step=self._step,
                step_elapsed_seconds=self._elapsed,
            ).to_json()
        )

    def _on_fatal(self, message: str) -> None:
        """The connection's writer died: nobody is listening, so cancel."""
        logger.warning("calibration connection lost: %s", message)
        self.request_stop()

    def _spawn(self, coro: Any) -> None:
        task = asyncio.ensure_future(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _finalize(self, outcome: CalibrationOutcome | None) -> None:
        """Quiesce the writer, publish the terminal message, and mark the session done."""
        try:
            with contextlib.suppress(Exception):
                await self._outbox.quiesce()
            self.terminal = self._terminal_of(outcome)
            self._outbox.publish_control(self.terminal.to_json())
        finally:
            for task in list(self._tasks):
                task.cancel()
            self.done.set()

    def _terminal_of(self, outcome: CalibrationOutcome | None) -> p.CalibrationStopped:
        """The server's honest account: files only for COMPLETED, never invented for any other outcome."""
        if outcome is None:
            return p.CalibrationStopped(
                self.session_id,
                p.CalibrationStopReason.UNCERTAIN,
                None,
                "the calibration ended without an outcome",
            )
        reason = _REASON_TO_WIRE[outcome.kind]
        if outcome.kind is OutcomeKind.COMPLETED:
            if not outcome.files:
                # A completed result is promised only with files; never leave the client without a terminal.
                return p.CalibrationStopped(
                    self.session_id,
                    p.CalibrationStopReason.UNCERTAIN,
                    None,
                    "the calibration reported completion but saved no files",
                )
            files = tuple(p.McalFile(_SIDE_TO_WIRE[f.side], f.file) for f in outcome.files)
            return p.CalibrationStopped(self.session_id, reason, p.CalibrationResult(files), None)
        message = (
            None if outcome.kind is OutcomeKind.CANCELLED else (outcome.message or "calibration did not complete")
        )
        return p.CalibrationStopped(self.session_id, reason, None, message)


def build_worker(config: CalibrationConfig) -> CalibrationWorker | None:
    """The supervisor for the configured executable and managed MCAL directory; None without a worker."""
    if config.worker_path is None:
        return None
    return CalibrationWorker(config.worker_path, config.directory)
