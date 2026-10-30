"""Owns the real ``RecordingService`` instance for this backend process:
one immutable committed/next-task context, at most one in-flight lifecycle
operation, a per-take recording ingress, and a small coordinator-owned
session status layered on top of the service's own real phase -- see
``recording/recording_service/README.md`` for the service's own lifecycle
contract.

This is NOT a second recording state machine: ``snapshot()`` always
derives from the real ``RecordingService.snapshot()`` plus this
coordinator's own "what operation, if any, is currently in flight" fact.
The frontend's own interaction FSM (idle/starting/recording/...) is
untouched; this module's job is to tell it the truth about what actually
happened.

A damaged take must NEVER report as a normal successful save, and its
data must NEVER be silently deleted either: if the per-take
``RecordingIngress`` ever actually fails (queue overflow, a FULL-retry
deadline, a conversion/submission exception) -- whether that is
discovered at stop time or happens mid-RECORDING -- this coordinator
settles that episode via the service's own ``interrupt_episode``, which
RETAINS the partial output and marks it recovery-required with the real
cause, exactly like a write failure the service detects itself. This is
deliberately never ``discard_episode`` (which deletes the take's data) --
storage/retention is the service's responsibility; this module's job is
only to report the real ingress-level cause honestly. A mid-recording
failure is settled as soon as possible via the ingress's own
``on_failure`` callback, which records the intent on retained coordinator
state (``_pending_interruption``) rather than a fire-and-forget task --
whichever operation next actually completes (including shutdown) is
guaranteed to be the thing that settles it, even if something unrelated
(e.g. a concurrent ``recover_episode``) was holding admission at the
moment of failure.

Every mutating operation's real work runs in an OWNED ``asyncio.Task``
(``_run_operation``), shielded from the calling coroutine's own
cancellation: a browser disconnect cancelling a WebSocket handler task
must not abandon real, possibly-shielded service work mid-flight (writer
startup, drain, a QC pass) -- the operation keeps running and this
coordinator keeps correctly tracking it regardless of whether anyone is
still awaiting the result. ``close()`` registers its own closing intent
synchronously (blocking further admission immediately), then joins
whatever operation is already in flight under one overall deadline, so
shutdown can always actually settle in-progress work rather than either
racing it or waiting forever.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, replace
from enum import Enum
from functools import partial
from typing import Any, TypeVar

from capture_server.client import CameraImage, CaptureFrame
from egocap_web.recording_adapter import RecordingIngress, validate_tracking_descriptor
from recording_service import (
    CloseResult,
    EpisodeInfo,
    EpisodeResult,
    PendingEpisode,
    RecordingPhase,
    ServiceConfig,
)
from recording_service import RecordingService as _RecordingService
from recording_service import RecordingSnapshot as ServiceSnapshot

_T = TypeVar("_T")

#: Overall bound on close()'s own settlement work (joining an in-flight
#: operation, draining the current ingress, closing the service) --
#: deliberately distinct from ServiceConfig.close_deadline_seconds (the
#: service's own internal bound for its close()), since this wraps that
#: call plus this coordinator's own additional settlement steps.
_CLOSE_DEADLINE_SECONDS = 35.0


class SessionStatus(Enum):
    """Coordinator-owned session lifecycle, layered on top of the
    service's own ``RecordingPhase`` -- distinct because "opening"/
    "closing" describe work this coordinator's own
    ``open()``/``close()`` calls are doing, not a phase the
    service itself reports."""

    CLOSED = "closed"
    OPENING = "opening"
    OPEN = "open"
    CLOSING = "closing"
    FAILED = "failed"


class OperationKind(Enum):
    """The one lifecycle operation that may be in flight at a time."""

    OPENING = "opening"
    STARTING = "starting"
    STOPPING = "stopping"
    DISCARDING = "discarding"
    RECOVERING = "recovering"
    DISCARDING_PENDING = "discarding_pending"
    CLOSING = "closing"
    NEW_BATCH = "new_batch"


class CoordinatorBusyError(Exception):
    """Raised when a caller tries to start an operation while a
    conflicting one is already in flight, or when a precondition (open
    session, idle phase, matching episode identity, ...) isn't met. The
    caller translates this into the existing business-rejection reply,
    never a protocol error -- this is an ordinary, expected outcome, not
    a bug."""


@dataclass(frozen=True)
class TaskContext:
    """The committed next-task context, validated/stored outside
    RECORDING; frozen into a take's own record at start. Deliberately
    narrow for this stage -- only what the existing inline task edit
    already provides: no task-catalog/mode fields yet, and nothing here
    waits for that to exist."""

    task_text: str
    operator: str = ""


@dataclass(frozen=True)
class _IngressRetirement:
    """The real outcome of one ``_retire_current_ingress`` attempt --
    ``settled`` is the load-bearing fact: an unsettled (still genuinely
    in-flight) ingress is NOT the same thing as a settled-but-damaged one,
    and callers must never conflate the two. ``failure``, if set, is a
    real diagnostic either way (the cause of non-settlement, or a data
    error discovered once settlement completed)."""

    settled: bool
    failure: str | None


@dataclass(frozen=True)
class CoordinatorSnapshot:
    """Everything the recording-command layer needs to publish an
    observation: the service's own real facts (``None`` before the first
    successful open), plus what operation (if any) this coordinator
    currently owns and the committed next-task context."""

    status: SessionStatus
    service: ServiceSnapshot | None
    active_operation: OperationKind | None
    context: TaskContext
    issue: str | None = None


#: Returns (ready, reason-if-not) from real, backend-local capture facts --
#: injected rather than owned here, since this coordinator must not import
#: CaptureStatusMonitor/CaptureClient: the backend owns exactly one
#: CaptureClient connection, made once at startup, and this coordinator
#: must not open a second one. REQUIRED (no default): an always-ready fallback
#: is a genuinely dangerous production default, not a convenience worth
#: keeping -- callers (including tests) must supply a real decision.
ReadinessCheck = Callable[[], tuple[bool, str | None]]


class RecordingSessionCoordinator:
    """Owns one ``RecordingService`` instance for this backend process's
    lifetime (one slot, ``self._service`` -- a close that fails to
    release keeps the SAME reference rather than tracking a second
    "unreleased" slot, so ownership is never ambiguous: a non-``None``
    ``self._service`` always means "this is the one instance that still
    needs releasing," whether or not it is currently healthy)."""

    def __init__(
        self,
        config_factory: Callable[[], ServiceConfig],
        *,
        queue_capacity: int,
        queue_bytes: int | None = None,
        readiness_check: ReadinessCheck,
        initial_context: TaskContext | None = None,
    ) -> None:
        self._config_factory = config_factory
        self._queue_capacity = queue_capacity
        self._queue_bytes = queue_bytes
        self._readiness_check = readiness_check
        self._service: _RecordingService | None = None
        self._status = SessionStatus.CLOSED
        self._operation: OperationKind | None = None
        self._operation_task: asyncio.Task[Any] | None = None
        # A configured deployment station YAML seeds this before the first snapshot can ever be
        # published (see app.py's lifespan) -- None (the untouched default) preserves today's
        # empty-task development start.
        self._context = initial_context if initial_context is not None else TaskContext(task_text="")
        # Scoped to exactly one take -- see RecordingIngress's own
        # docstring for why a new take always gets a new instance.
        self._current_ingress: RecordingIngress | None = None
        self._last_issue: str | None = None
        # Per-attempt descriptor-compatibility check (see on_capture_frame),
        # reset whenever a fresh RecordingService instance is opened.
        self._tracking_descriptor_checked = False
        self._tracking_supported = False
        # Registered synchronously at the top of close(), before any await
        # -- blocks further admission (open/set_task/start/.../new_batch)
        # immediately, and lets a second concurrent close() call join the
        # first rather than racing it.
        self._closing = False
        self._close_task: asyncio.Task[CloseResult] | None = None
        self._final_close = False
        # A recorded ingress-failure intent not yet settled -- set
        # synchronously by _schedule_failure_settlement (the ingress's own
        # on_failure callback), retried from _run_operation's own done
        # callback after EVERY operation completes and as an explicit
        # backstop in close() -- never a single fire-and-forget attempt.
        self._pending_interruption: tuple[str, str] | None = None
        self._settle_tasks: set[asyncio.Task[None]] = set()
        self._service_for_pending_interruption: _RecordingService | None = None
        # Public, settable hooks -- invoked synchronously, INSIDE the same
        # retained/shielded operation task this coordinator already owns
        # (never a second, independently-created task), right after the
        # real underlying event actually happens. This is how a caller
        # (RecordingContext) ties its own wire-protocol projection update
        # to the SAME task that survives a disconnecting requester's own
        # cancellation -- covering both a user-initiated command AND the
        # automatic mid-recording interruption path through one mechanism,
        # since both already run through _run_operation/these same impl
        # methods. None (the default) means no one is listening yet.
        self.on_episode_started: Callable[[], None] | None = None
        self.on_episode_result: Callable[[EpisodeResult], None] | None = None
        self.on_batch_rotated: Callable[[], None] | None = None

    @property
    def status(self) -> SessionStatus:
        return self._status

    # --- operation admission -------------------------------------------------

    async def _run_operation(self, kind: OperationKind, make_coro: Callable[[], Coroutine[Any, Any, _T]]) -> _T:
        """Registers ``kind`` as the one in-flight operation (a plain
        synchronous check -- so two callers can never both pass it, since
        nothing can interleave between the check and the registration on
        a single-threaded event loop), then runs the real work in an
        OWNED task, shielded from this caller's own cancellation. If the
        calling coroutine is itself cancelled (e.g. a browser disconnect),
        ``asyncio.shield`` re-raises that cancellation to the caller
        WITHOUT cancelling the owned task -- the real operation keeps
        running in the background and still correctly clears
        ``self._operation``/``self._operation_task`` once it actually
        finishes, via the done callback below, regardless of whether
        anyone is still awaiting it. If nobody ends up awaiting the task
        again (the caller already detached via cancellation) and it later
        raises a real exception, that exception is still surfaced into
        ``self._last_issue`` rather than only ever reaching an
        "exception was never retrieved" asyncio log line."""
        if self._closing:
            raise CoordinatorBusyError("recording session is closing")
        if self._operation is not None:
            raise CoordinatorBusyError(f"{self._operation.value} already in progress")
        self._operation = kind
        task: asyncio.Task[_T] = asyncio.create_task(make_coro())
        self._operation_task = task

        def _on_done(t: asyncio.Task[_T]) -> None:
            if self._operation_task is t:
                self._operation_task = None
                self._operation = None
            if not t.cancelled():
                exc = t.exception()
                if exc is not None:
                    self._last_issue = f"{kind.value} failed: {exc}"
            # Whichever operation just finished is the one responsible for
            # retrying a still-unsettled ingress-failure intent -- not only
            # the specific operation that happened to observe it.
            self._try_settle_pending_interruption()

        task.add_done_callback(_on_done)
        return await asyncio.shield(task)

    def _require_open(self) -> _RecordingService:
        if self._closing:
            raise CoordinatorBusyError("recording session is closing")
        if self._service is None or self._status is not SessionStatus.OPEN:
            raise CoordinatorBusyError("no open recording session")
        return self._service

    # --- session lifecycle ----------------------------------------------------

    async def open(self) -> None:
        """Constructs and opens a fresh ``RecordingService`` -- raises if
        a session already exists in ANY state (open/opening, or a
        previous one not yet confirmed released) or this coordinator is
        closing; a closed/failed coordinator with no retained service can
        otherwise always retry. An explicit, standalone open -- real
        usage never calls this directly; ``start_episode`` opens storage
        itself, lazily, the moment it actually needs it (see
        ``_open_storage_if_needed``)."""
        if self._closing:
            raise CoordinatorBusyError("recording session is closing")
        if self._service is not None:
            raise CoordinatorBusyError(
                f"a recording session ({self._status.value}) already exists; close it before opening a new one"
            )
        await self._run_operation(OperationKind.OPENING, self._open_impl)

    async def _open_impl(self) -> None:
        await self._open_storage_if_needed()

    async def _open_storage_if_needed(self) -> _RecordingService:
        """Returns the real storage session, opening one for real if none
        exists yet -- called from inside ``start_episode``'s own single
        retained operation (never a nested ``_run_operation``), so real
        storage (output directories, a frozen source schema) only ever
        gets created at the one point a committed Record attempt actually
        needs it, never on an earlier set_task/new_batch/stale command. A
        FAILED or not-yet-released previous session blocks this the same
        as ``open()`` itself -- never treated as if it were simply
        absent."""
        if self._service is not None:
            if self._status is not SessionStatus.OPEN:
                raise CoordinatorBusyError(
                    f"a recording session ({self._status.value}) already exists; close it before opening a new one"
                )
            return self._service
        self._status = SessionStatus.OPENING
        service = _RecordingService(self._config_factory())
        # Assigned BEFORE the await below: a failure or cancellation here
        # must not orphan a service instance that may have already
        # acquired real resources -- self._service is the one place close()
        # looks to find and release it.
        self._service = service
        self._tracking_descriptor_checked = False
        self._tracking_supported = False
        try:
            await service.open_session()
        except (Exception, asyncio.CancelledError) as exc:
            self._status = SessionStatus.FAILED
            self._last_issue = "open_session failed"
            if isinstance(exc, Exception):
                # A failed open leaves the service holding nothing worth
                # keeping: release it now so the next Record starts fresh
                # instead of waiting for a restart. If the release itself
                # does not finish, the instance stays owned (FAILED) and
                # New batch retries it.
                released = False
                with contextlib.suppress(Exception):
                    released = (await service.close()).released
                if released:
                    self._service = None
                    self._status = SessionStatus.CLOSED
            raise
        self._status = SessionStatus.OPEN
        self._last_issue = None
        return service

    def set_task(self, task_text: str) -> None:
        """Commits the next-task context -- backend-local state that
        requires no open storage session (and never opens one): only
        ``start_episode`` opens real storage, since freezing output
        directories/source schema on an earlier, possibly-stale or
        empty-text command would be a real filesystem side effect nothing
        asked for yet. Requires no operation already in flight and, if a
        session happens to already be open, that it isn't
        STARTING/RECORDING. A take already stopping/finalizing keeps its
        own frozen context; this only ever affects the NEXT take."""
        if self._closing:
            raise CoordinatorBusyError("recording session is closing")
        if self._operation is not None:
            raise CoordinatorBusyError(f"{self._operation.value} already in progress")
        if self._service is not None:
            phase = self._service.snapshot().phase
            if phase in (RecordingPhase.STARTING, RecordingPhase.RECORDING):
                raise CoordinatorBusyError("cannot edit task while recording")
        self._context = replace(self._context, task_text=task_text)

    async def start_episode(self, operator: str = "") -> EpisodeInfo:
        """The one place real storage is opened: requires current
        readiness (so the schema it opens with reflects actually accepted
        sources, not a guess made too early) and a committed task, then
        opens storage (if not already open) and starts the take as ONE
        retained operation -- see ``_start_episode_impl``. ``operator`` is
        the caller's already-validated, already-normalized name
        (attribution only, never authentication).

        Combined into a LOCAL frozen copy of ``self._context``, never
        written back to it: ``self._context`` is long-lived committed
        state shared with ``set_task``, and admission can still be
        rejected after this point (e.g. a duplicate start while already
        RECORDING, inside the real service's own state machine) -- so a
        rejected attempt must never leak its operator into shared state."""
        if self._closing:
            raise CoordinatorBusyError("recording session is closing")
        if self._service is not None and self._status is SessionStatus.FAILED:
            raise CoordinatorBusyError("recording session failed; use New batch before recording again")
        ready, reason = self._readiness_check()
        if not ready:
            raise CoordinatorBusyError(reason or "required sources are not ready")
        frozen_context = replace(self._context, operator=operator)
        return await self._run_operation(OperationKind.STARTING, partial(self._start_episode_impl, frozen_context))

    async def _start_episode_impl(self, frozen_context: TaskContext) -> EpisodeInfo:
        if self._current_ingress is not None:
            # A previous take's ingress is still the live one (its own
            # stop/discard/settle attempt found it unsettled) -- never
            # construct a new ingress over it; the old consumer thread
            # would be left with nothing owning it.
            raise CoordinatorBusyError("a previous take's ingress has not yet settled")
        service = await self._open_storage_if_needed()
        # Recheck readiness/closing intent right after storage finished
        # opening -- the same "don't trust state from before the last
        # await" pattern as the recheck below, just one await earlier.
        ready, _ = self._readiness_check()
        if not ready or self._closing:
            raise CoordinatorBusyError("start was superseded while storage was opening")
        episode = await service.start_episode(frozen_context.task_text, frozen_context.operator)
        # Recheck readiness AND the service's own phase after the await:
        # start_episode can succeed at the storage level after something
        # else has already superseded this attempt (a source going stale,
        # a shutdown intent). service.phase == RECORDING alone does not
        # prove capture is still actually available -- the readiness
        # check is real-time backend-local capture state, re-run here
        # rather than trusted from before the await.
        ready, _ = self._readiness_check()
        # Also recheck closing intent: a close() could have been requested
        # concurrently while this start was still in flight -- activating
        # a brand new ingress for a session that's already shutting down
        # would hand it work nothing is going to retire.
        if not ready or self._closing or service.snapshot().phase is not RecordingPhase.RECORDING:
            # A superseded storage-level start must be explicitly settled,
            # not merely raised past -- discard it so the service doesn't
            # retain an orphaned RECORDING take nothing will ever stop.
            with contextlib.suppress(Exception):
                await service.discard_episode(episode.episode_id)
            raise CoordinatorBusyError("start was superseded before activation")
        episode_id = episode.episode_id
        ingress = RecordingIngress(
            queue_capacity=self._queue_capacity,
            queue_bytes=self._queue_bytes,
            submit=partial(service.submit_sample, episode_id),
            on_failure=lambda message: self._schedule_failure_settlement(service, episode_id, message),
        )
        await ingress.start()
        self._current_ingress = ingress
        if self.on_episode_started is not None:
            self.on_episode_started()
        return episode

    def _schedule_failure_settlement(self, service: _RecordingService, episode_id: str, message: str) -> None:
        """Called synchronously (on the event loop thread) the moment the
        current take's ingress actually fails, including mid-RECORDING.
        Records the intent on retained coordinator state -- never only a
        fire-and-forget task -- so settlement is guaranteed to eventually
        happen via whichever operation next completes (see
        ``_try_settle_pending_interruption``), not merely "probably,
        because something else will likely observe it."""
        if self._pending_interruption is None:
            self._pending_interruption = (episode_id, message)
        self._service_for_pending_interruption = service
        self._try_settle_pending_interruption()

    def _try_settle_pending_interruption(self) -> None:
        """Attempts to settle a recorded ingress-failure intent right now.
        Called after EVERY operation completes (via ``_run_operation``'s
        own done callback) and explicitly from ``close()``, so whichever
        operation finishes next -- not necessarily the one whose own work
        observed the failure -- is what actually settles it. No-ops if
        there is nothing pending or admission is already owned; the state
        persists untouched for the next retry in that case."""
        if self._pending_interruption is None:
            return
        task = asyncio.create_task(self._attempt_settle_pending_interruption())
        self._settle_tasks.add(task)  # a held reference: the loop keeps only weak ones
        task.add_done_callback(self._settle_tasks.discard)

    async def _attempt_settle_pending_interruption(self) -> None:
        if self._pending_interruption is None:
            return
        service = self._service_for_pending_interruption
        if service is None:
            return
        episode_id, message = self._pending_interruption
        try:
            await self._run_operation(
                OperationKind.STOPPING, partial(self._settle_failed_episode_impl, service, episode_id, message)
            )
        except CoordinatorBusyError:
            # Admission is owned by something else right now -- left
            # pending for the NEXT operation's own completion to retry
            # (see _on_done above), never silently dropped.
            return
        if self._pending_interruption == (episode_id, message):
            self._pending_interruption = None

    async def _settle_failed_episode_impl(
        self, service: _RecordingService, episode_id: str, message: str
    ) -> EpisodeResult | None:
        current = service.snapshot().episode
        if current is None or current.episode_id != episode_id or self._current_ingress is None:
            return None  # already settled by something else
        retirement = await self._retire_current_ingress()
        if not retirement.settled:
            # Still genuinely in flight -- do not interrupt the service
            # out from under a consumer that's still actually running;
            # _pending_interruption stays set for the next retry (see
            # _attempt_settle_pending_interruption's own CoordinatorBusyError handling).
            raise CoordinatorBusyError(retirement.failure or "ingress did not settle yet")
        self._last_issue = f"recording pipeline failed during recording: {retirement.failure or message}"
        result = await service.interrupt_episode(episode_id, self._last_issue)
        self._emit_result(result)
        return result

    def _emit_result(self, result: EpisodeResult) -> None:
        """The one place ``on_episode_result`` is invoked -- covers the
        user-initiated stop/discard paths AND the automatic mid-recording
        interruption path, since all three already run inside this
        coordinator's own retained operation task."""
        if self.on_episode_result is not None:
            self.on_episode_result(result)

    async def stop_episode(self, episode_id: str) -> EpisodeResult:
        service = self._require_open()
        self._require_matching_active_episode(service, episode_id)
        return await self._run_operation(
            OperationKind.STOPPING, partial(self._stop_episode_impl, service, episode_id)
        )

    async def _stop_episode_impl(self, service: _RecordingService, episode_id: str) -> EpisodeResult:
        # A capture loss or stale source that no settlement has reached
        # yet must not let this Stop save a truncated take as KEPT -- but
        # only a pending interruption for THIS episode: a leftover from a
        # previous, already-replaced take must never turn a good Stop into
        # RECOVERY_REQUIRED.
        interruption: str | None = (
            self._pending_interruption[1]
            if self._pending_interruption is not None and self._pending_interruption[0] == episode_id
            else None
        )
        if interruption is None:
            ready, reason = self._readiness_check()
            if not ready:
                interruption = f"capture lost: {reason or 'a required source is no longer ready'}"
        retirement = await self._retire_current_ingress()
        if not retirement.settled:
            # Still genuinely in flight -- refuse this Stop rather than
            # act past an ingress that hasn't actually released yet; the
            # caller sees an ordinary busy rejection and may retry.
            raise CoordinatorBusyError(retirement.failure or "ingress did not settle yet")
        failure = retirement.failure or interruption
        if failure is not None:
            # A damaged take must NEVER report as a normal successful
            # save -- a real pipeline-level failure (queue overflow, a
            # conversion/submission exception, a FULL-retry timeout) can
            # drop a sample before it ever reaches the service's own
            # accepted-count, so the service's own accepted==written
            # integrity check cannot see it and would otherwise report a
            # false KEPT. Retain it as recovery-required instead -- never
            # discard_episode, which would delete the data outright; the
            # operator's own "stop" intent doesn't change the fact that
            # this specific take's data is not trustworthy as recorded.
            self._last_issue = f"recording pipeline failed before stop: {failure}"
            result = await service.interrupt_episode(episode_id, self._last_issue)
            self._emit_result(result)
            return result
        result = await service.stop_episode(episode_id)
        self._emit_result(result)
        return result

    async def discard_episode(self, episode_id: str) -> EpisodeResult:
        service = self._require_open()
        self._require_matching_active_episode(service, episode_id)
        return await self._run_operation(
            OperationKind.DISCARDING, partial(self._discard_episode_impl, service, episode_id)
        )

    async def _discard_episode_impl(self, service: _RecordingService, episode_id: str) -> EpisodeResult:
        # Being discarded anyway -- a settled ingress failure changes
        # nothing about the outcome, just retire it and move on. An
        # UNSETTLED ingress is different: its consumer/conversion work is
        # still genuinely running, so discarding the episode out from
        # under it now would leave that work with nothing left owning it
        # -- refuse until it's actually confirmed retired.
        retirement = await self._retire_current_ingress()
        if not retirement.settled:
            raise CoordinatorBusyError(retirement.failure or "ingress did not settle yet")
        result = await service.discard_episode(episode_id)
        self._emit_result(result)
        return result

    async def _retire_current_ingress(self) -> _IngressRetirement:
        """The ONE place a take's ingress is ever retired: ``ingress.stop()``
        is itself the single bounded retirement call (closes admission,
        waits under one overall budget for genuine settlement, then stops
        the consumer task) -- see its own docstring.

        Deliberately does NOT clear ``self._current_ingress`` unless
        ``stop()`` reports genuine settlement -- dropping the only
        reference on a mere timeout would mean a concurrent ``close()``
        has nothing left to find and retry; keeping it lets a later
        operation's own ``stop()`` call idempotently retry the SAME still-
        unresolved work. Every caller must check ``.settled`` before
        treating this take as retired -- an unsettled result means the
        old ingress is STILL the live one; nothing may replace, delete or
        otherwise act past it yet."""
        ingress = self._current_ingress
        if ingress is None:
            return _IngressRetirement(settled=True, failure=None)
        settled = await ingress.stop()
        if not settled:
            return _IngressRetirement(
                settled=False,
                failure=ingress.failure or "ingress did not settle (conversion work still unresolved)",
            )
        if self._current_ingress is ingress:
            self._current_ingress = None
        return _IngressRetirement(settled=True, failure=ingress.failure)

    def _require_matching_active_episode(self, service: _RecordingService, episode_id: str) -> None:
        """Validates the EXACT active episode identity BEFORE any
        gate/ingress mutation -- a stale/wrong target must have literally
        zero effect on whatever take is actually recording. Only once
        this passes does stop/discard touch the real ingress."""
        current = service.snapshot().episode
        if current is None or current.episode_id != episode_id:
            raise CoordinatorBusyError("no matching active episode")

    async def recover_episode(self, episode_id: str) -> EpisodeResult:
        service = self._require_open()
        return await self._run_operation(OperationKind.RECOVERING, partial(service.recover_episode, episode_id))

    async def discard_pending(self, episode_id: str) -> EpisodeResult:
        service = self._require_open()
        return await self._run_operation(
            OperationKind.DISCARDING_PENDING, partial(service.discard_pending, episode_id)
        )

    def pending_episodes(self) -> tuple[PendingEpisode, ...]:
        return self._service.pending_episodes() if self._service is not None else ()

    async def new_batch(self) -> None:
        """Releases the current storage session, if any, and rotates the
        batch -- it does NOT eagerly open a replacement.
        Real storage for the new batch is opened fresh, with the
        then-current valid profile, by the NEXT ``start_episode`` call
        (see ``_open_storage_if_needed``); opening it here would freeze a
        schema before anything has committed to actually recording with
        it. If no storage session exists yet (New batch used before any
        recording this process), this only rotates local batch/task
        facts -- there is nothing to close. It is also how an interrupted
        take (RECOVERY_REQUIRED) or a failed session is cleared: the old
        session is closed, the take stays on disk as pending, and the
        next ``start_episode`` opens a fresh session."""
        if self._closing:
            raise CoordinatorBusyError("recording session is closing")
        if self._current_ingress is not None:
            # A previous take's ingress is still the live one -- never
            # release/rotate storage out from under a consumer that's
            # still genuinely running.
            raise CoordinatorBusyError("a previous take's ingress has not yet settled")
        if self._service is not None:
            # IDLE is the normal case. RECOVERY_REQUIRED (an interrupted
            # take) and a FAILED session are exactly the states New batch
            # exists to clear: closing the service releases it and leaves
            # any retained take on disk as pending, never deleted.
            phase = self._service.snapshot().phase
            if phase not in (RecordingPhase.IDLE, RecordingPhase.RECOVERY_REQUIRED):
                raise CoordinatorBusyError("an episode is active")
        await self._run_operation(OperationKind.NEW_BATCH, self._new_batch_impl)

    async def _new_batch_impl(self) -> None:
        service = self._service
        if service is not None:
            close_result = await service.close()
            if not close_result.released:
                # self._service deliberately stays pointing at the
                # not-yet-released instance -- a later close()/new_batch
                # retry can still find and finish releasing it.
                self._status = SessionStatus.FAILED
                self._last_issue = "previous session did not fully release: " + "; ".join(close_result.issues)
                raise CoordinatorBusyError(self._last_issue)
            self._service = None
            self._status = SessionStatus.CLOSED
        if self.on_batch_rotated is not None:
            self.on_batch_rotated()

    async def close(self, *, final: bool = False) -> CloseResult:
        """Backend-owned shutdown -- not a browser Quit command. Registers
        its own closing intent SYNCHRONOUSLY (blocking further admission
        immediately, before any await), then joins whatever operation is
        already in flight and settles the current ingress/service under
        one overall deadline. A second concurrent ``close()`` call joins
        the first rather than racing it. Safe to call with nothing open.

        ``final=True`` is the application shutdown call: admission stays
        blocked once it returns, so nothing can start a take while the
        capture owner is still being stopped. Without it, admission
        reopens when the close finishes."""
        if self._closing and self._close_task is not None:
            return await asyncio.shield(self._close_task)
        self._closing = True
        self._final_close = final
        self._close_task = asyncio.create_task(self._close_impl())
        return await asyncio.shield(self._close_task)  # _close_impl clears _close_task/_closing itself

    async def _close_impl(self) -> CloseResult:
        try:
            async with asyncio.timeout(_CLOSE_DEADLINE_SECONDS):
                return await self._close_body()
        except TimeoutError:
            self._status = SessionStatus.FAILED
            self._last_issue = f"close did not complete within {_CLOSE_DEADLINE_SECONDS:.0f}s"
            return CloseResult(released=False, issues=(self._last_issue,))
        finally:
            self._closing = self._final_close
            self._close_task = None

    async def _close_body(self) -> CloseResult:
        if self._operation_task is not None:
            try:
                await asyncio.shield(self._operation_task)
            except asyncio.CancelledError:
                raise  # a real external cancellation of THIS close() call -- propagate, never swallow
            except Exception:  # noqa: BLE001 -- the operation itself failing doesn't block settling close
                pass
        # Explicit backstop: close() has exclusive ownership past this
        # point (self._closing already blocks any new admission), so if a
        # mid-recording failure is still pending (nothing else ever got a
        # turn to settle it), settle it directly here rather than leaving
        # it to a retry that will never come once the session is closed.
        if self._pending_interruption is not None and self._service_for_pending_interruption is not None:
            episode_id, message = self._pending_interruption
            self._pending_interruption = None
            with contextlib.suppress(Exception):
                await self._settle_failed_episode_impl(self._service_for_pending_interruption, episode_id, message)
        if self._current_ingress is not None:
            retirement = _IngressRetirement(settled=False, failure="ingress retirement raised during close")
            with contextlib.suppress(Exception):
                retirement = await self._retire_current_ingress()
            if not retirement.settled:
                # Genuinely still unresolved (e.g. a stuck conversion) --
                # retained, not cleared, so a later close() retry can
                # still find and rejoin this same ingress; reporting
                # released here would be a false claim of release.
                self._status = SessionStatus.FAILED
                issue = retirement.failure or "ingress did not settle before close"
                self._last_issue = issue
                return CloseResult(released=False, issues=(issue,))
        if self._service is None:
            self._status = SessionStatus.CLOSED
            return CloseResult(released=True)
        self._status = SessionStatus.CLOSING
        result = await self._service.close()
        if result.released:
            self._service = None
        self._status = SessionStatus.CLOSED if result.released else SessionStatus.FAILED
        if not result.released:
            self._last_issue = "close did not fully release: " + "; ".join(result.issues)
        return result

    def snapshot(self) -> CoordinatorSnapshot:
        return CoordinatorSnapshot(
            status=self._status,
            service=self._service.snapshot() if self._service is not None else None,
            active_operation=self._operation,
            context=self._context,
            issue=self._last_issue,
        )

    def readiness(self) -> tuple[bool, str | None]:
        """The real, current admission readiness -- the exact same check
        ``start_episode`` itself runs, exposed read-only so a caller
        publishing state (e.g. ``RecordingContext``) can report "not
        ready, because X" without duplicating or guessing at the
        underlying source/freshness logic."""
        return self._readiness_check()

    # --- capture callbacks (synchronous, called directly from CaptureStatusMonitor) ---

    def on_capture_frame(self, frame: CaptureFrame) -> None:
        """Offers one raw tracking frame to the currently-active take's
        ingress, if any -- never blocks, never converts/decodes here (see
        ``RecordingIngress``). Validates the descriptor once per attempt;
        a descriptor mismatch permanently disables recording tracking data
        for this attempt (the primary gate is the injected
        ``readiness_check`` at ``start_episode`` time, see
        ``_open_storage_if_needed``/``start_episode`` -- this is a
        defensive backstop for a descriptor that changes mid-attempt,
        not the only check).

        Also re-runs the SAME readiness check on every accepted frame
        while a take is active -- this native-callback stream is already
        firing continuously whenever tracking is alive, so reusing it
        here detects a required source (e.g. one camera) going stale or
        disappearing mid-take with no new timer/poll loop. See
        ``_check_still_ready_while_recording``."""
        if self._current_ingress is None:
            return
        self._check_still_ready_while_recording()
        if not self._tracking_descriptor_checked:
            self._tracking_descriptor_checked = True
            try:
                validate_tracking_descriptor(frame.description)
                self._tracking_supported = True
            except Exception as exc:  # noqa: BLE001 -- recorded, never raised into the capture callback
                self._tracking_supported = False
                self._last_issue = f"tracking descriptor unsupported for recording: {exc}"
        if not self._tracking_supported:
            return
        self._current_ingress.offer(frame)

    def on_capture_image(self, image: CameraImage) -> None:
        """Offers one raw camera image to the currently-active take's
        ingress, if any -- never blocks, never decodes here. Also
        re-checks readiness, same as ``on_capture_frame`` -- this is what
        catches tracking going stale/stalling while a camera keeps
        flowing (the symmetric case)."""
        if self._current_ingress is None:
            return
        self._check_still_ready_while_recording()
        self._current_ingress.offer(image)

    def _check_still_ready_while_recording(self) -> None:
        """Forwards a required-source readiness failure discovered WHILE a
        take is active into the same retained-interruption path as
        ``on_capture_lost``/an ingress pipeline failure -- a take must
        never silently keep recording (or admit a normal Stop) once a
        source it depends on is no longer actually ready."""
        if self._current_ingress is None:
            return
        ready, reason = self._readiness_check()
        if not ready:
            self.on_capture_lost(reason or "a required source is no longer ready")

    def on_capture_lost(self, reason: str) -> None:
        """Called when the underlying capture attempt has terminated for
        real (``CaptureStatusMonitor``'s own terminal callback) or a
        required source has gone stale mid-take (see
        ``_check_still_ready_while_recording``) -- forwarded into the
        SAME retained-interruption mechanism as an ingress pipeline
        failure, so an active take that loses its capture source
        mid-recording is retained as recovery-required, never silently
        admitted to a normal Stop as if nothing happened. A take that has
        already been deliberately cut off for a normal save (no current
        ingress, or the episode has already moved on) is left alone --
        ``_settle_failed_episode_impl``'s own existing identity/ingress
        check already enforces that boundary, exactly as it does for an
        ingress-reported failure."""
        if self._current_ingress is None or self._service is None or self._pending_interruption is not None:
            return
        episode = self._service.snapshot().episode
        if episode is None:
            return
        # Closes admission synchronously, right now -- the actual
        # settlement below still only runs via the retained-operation
        # retry path, but no further sample should be accepted into a
        # take already known to have lost its source.
        self._current_ingress.close()
        self._schedule_failure_settlement(self._service, episode.episode_id, f"capture lost: {reason}")
