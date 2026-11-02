# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Presentation state and ownership rules for one calibration at a time, with no UI toolkit.

Every method runs on the asyncio loop that owns the worker. A UI renders
:meth:`CalibrationController.view`; nothing here (or in the UI) calls the SDK.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from enum import Enum

from . import protocol as proto
from .worker import (
    AttemptError,
    CalibrationAttempt,
    CalibrationOutcome,
    CalibrationWorker,
    OutcomeKind,
)


class Stage(str, Enum):
    """Coarse stage the page renders."""

    IDLE = "idle"
    STARTING = "starting"
    DISCOVERING = "discovering"
    AWAITING_CONFIRMATION = "awaiting_confirmation"
    SAMPLING = "sampling"
    SAVING = "saving"
    STOPPING = "stopping"
    FINISHED = "finished"


@dataclass(frozen=True)
class View:
    """Immutable snapshot of everything the page shows."""

    stage: Stage = Stage.IDLE
    owner: str | None = None
    sides: tuple[proto.Side, ...] = ()
    gloves: tuple[proto.Glove, ...] = ()
    prompt: proto.Prompt | None = None
    confirm_pending: bool = False
    elapsed_s: float | None = None
    notice: str = ""
    outcome: CalibrationOutcome | None = None


class NotOwner(Exception):
    """The caller is not the client that owns the active attempt."""


class CalibrationController:
    """Owns the single attempt, its owner, and the view derived from worker events."""

    def __init__(
        self,
        worker: CalibrationWorker,
        on_change: Callable[[View], None] | None = None,
    ) -> None:
        self._worker = worker
        self._on_change = on_change
        self._view = View()
        self._attempt: CalibrationAttempt | None = None
        self._watcher: asyncio.Task[None] | None = None

    def view(self) -> View:
        return self._view

    @property
    def active(self) -> bool:
        """True from the start request until the terminal outcome is recorded."""
        return self._attempt is not None

    async def start(self, client_id: str, sides: Sequence[proto.Side]) -> None:
        """Begin an attempt owned by ``client_id``; refused while another is active."""
        if self._attempt is not None:
            raise AttemptError("a calibration attempt is already active")
        try:
            attempt = self._worker.new_attempt(self._on_event)
        except AttemptError:
            # The worker is still winding down a previous attempt.
            self._notice("the previous attempt is still shutting down")
            raise
        self._attempt = attempt
        self._set(
            View(
                stage=Stage.STARTING,
                owner=client_id,
                sides=tuple(s for s in proto.Side if s in sides),
            )
        )
        self._watcher = asyncio.create_task(self._watch(attempt))
        try:
            await attempt.start(sides)
        except Exception:
            # start() settles worker-side failures itself; a rejected request never reached the worker, so
            # settle it here and wait for the watcher, leaving the controller free for the next start.
            await attempt.cancel()
            await self._watcher
            raise

    async def confirm(self, client_id: str, token: str) -> None:
        """Start the SDK step the browser is looking at; a click on a superseded prompt is refused."""
        attempt = self._require_owner(client_id)
        prompt = self._view.prompt
        if prompt is None or prompt.token != token:
            raise AttemptError("that prompt is no longer current")
        if self._view.confirm_pending:
            return
        self._set(replace(self._view, confirm_pending=True, notice=""))
        try:
            await attempt.confirm(prompt.token)
        except AttemptError as exc:
            self._notice(str(exc))
        finally:
            if self._attempt is attempt and self._view.confirm_pending:
                self._set(replace(self._view, confirm_pending=False))

    async def cancel(self, client_id: str) -> None:
        """Cancel the active attempt; only its owner may."""
        attempt = self._require_owner(client_id)
        self._set(replace(self._view, stage=Stage.STOPPING, notice="cancelling"))
        await attempt.cancel()

    def owner_attempt(self, client_id: str) -> CalibrationAttempt | None:
        """The active attempt ``client_id`` owns, if any; lets a caller bind a later cancel to it."""
        if self._attempt is not None and self._view.owner == client_id:
            return self._attempt
        return None

    async def client_disconnected(
        self, client_id: str, attempt: CalibrationAttempt | None = None
    ) -> None:
        """Owner loss cancels an active attempt; completed files are unaffected.

        With ``attempt`` given, only that attempt is cancelled, so a late timer cannot hit a newer one.
        """
        current = self._attempt
        if current is None or self._view.owner != client_id:
            return
        if attempt is not None and attempt is not current:
            return
        await current.cancel()

    async def aclose(self) -> None:
        """Cancel any active attempt and wait for its outcome, for orderly shutdown."""
        attempt = self._attempt
        if attempt is None:
            return
        await attempt.cancel()
        if self._watcher is not None:
            with contextlib.suppress(asyncio.CancelledError):
                await self._watcher

    def _require_owner(self, client_id: str) -> CalibrationAttempt:
        attempt = self._attempt
        if attempt is None:
            raise AttemptError("no calibration attempt is active")
        if self._view.owner != client_id:
            raise NotOwner(
                "only the browser that started this calibration can control it"
            )
        return attempt

    def _on_event(self, event: proto.WorkerEvent) -> None:
        """Fold one worker event into the view; the outcome arrives through :meth:`_watch`."""
        view = self._view
        if isinstance(event, proto.Status):
            self._set(
                replace(view, gloves=event.gloves, stage=_STAGE_OF_PHASE[event.phase])
            )
        elif isinstance(event, proto.Prompt):
            self._set(
                replace(
                    view,
                    stage=Stage.AWAITING_CONFIRMATION,
                    prompt=event,
                    confirm_pending=False,
                    elapsed_s=None,
                    notice="",
                )
            )
        elif isinstance(event, proto.StepAccepted):
            self._set(replace(view, stage=Stage.SAMPLING))
        elif isinstance(event, proto.StepProgress):
            self._set(replace(view, elapsed_s=event.elapsed_s))

    async def _watch(self, attempt: CalibrationAttempt) -> None:
        """Record the terminal outcome and release ownership so a new attempt can start."""
        outcome = await attempt.outcome()
        self._attempt = None
        self._set(
            View(
                stage=Stage.FINISHED,
                sides=outcome.sides,
                outcome=outcome,
                notice=_summary(outcome),
            )
        )

    def _notice(self, text: str) -> None:
        self._set(replace(self._view, notice=text))

    def _set(self, view: View) -> None:
        self._view = view
        if self._on_change is not None:
            self._on_change(view)


_STAGE_OF_PHASE = {
    proto.Phase.DISCOVERING: Stage.DISCOVERING,
    proto.Phase.AWAITING_STEP: Stage.AWAITING_CONFIRMATION,
    proto.Phase.CAPTURING_STEP: Stage.SAMPLING,
    proto.Phase.SAVING: Stage.SAVING,
    proto.Phase.STOPPING: Stage.STOPPING,
}


def _summary(outcome: CalibrationOutcome) -> str:
    """One line for the page; an uncertain outcome is never described as success."""
    if outcome.kind is OutcomeKind.COMPLETED:
        return "calibration completed"
    if outcome.kind is OutcomeKind.CANCELLED:
        return "calibration cancelled; no files were published"
    if outcome.kind is OutcomeKind.UNCERTAIN:
        return f"outcome uncertain, no files were published: {outcome.message}"
    return f"calibration failed: {outcome.message}"
