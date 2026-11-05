# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Supervises one run of the calibration worker executable.

:class:`CalibrationWorker` holds the launch settings; :meth:`CalibrationWorker.new_attempt` returns
a :class:`CalibrationAttempt`, which runs the worker once and settles on exactly one
:class:`CalibrationOutcome`. Nothing here imports a UI toolkit.

Outcome rule: the worker's own terminal ``result`` event is the only evidence of what happened.
Without one (crash, kill, EOF, protocol violation, forced termination) the outcome is
``UNCERTAIN``, never ``COMPLETED``.
"""

from __future__ import annotations

import asyncio
import collections
import contextlib
import errno
import logging
import os
import secrets
import socket
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from . import protocol as proto

logger = logging.getLogger("isaaccapture.manus_calibration.worker")

_LOG_LINE_CHARS = 1000
_DRAIN_CHUNK = 4096
_PROGRESS_SLACK_S = 1e-6
_WATCHDOG_POLL_S = 0.25
_TASK_SETTLE_S = 2.0


class AttemptError(Exception):
    """The attempt API was used in a way the attempt's state does not allow."""


class ConfirmError(AttemptError):
    """A confirmation was refused, by this supervisor or by the worker."""


class AttemptEnded(AttemptError):
    """The attempt reached its terminal outcome before the awaited event."""


class AttemptState(str, Enum):
    NEW = "new"
    STARTING = "starting"
    RUNNING = "running"
    FINISHING = "finishing"
    DONE = "done"


class OutcomeKind(str, Enum):
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"
    UNCERTAIN = "uncertain"


@dataclass(frozen=True)
class CalibrationOutcome:
    """Terminal result of an attempt. ``files`` lists the exported files, only for ``COMPLETED``."""

    kind: OutcomeKind
    attempt: str
    sides: tuple[proto.Side, ...]
    message: str
    files: tuple[proto.ExportedFile, ...] = ()
    error: proto.WorkerError | None = None
    exit_code: int | None = None
    forced_termination: bool = False
    log_tail: tuple[str, ...] = ()


class CalibrationWorker:
    """Launch settings for the worker executable and the single-attempt-at-a-time guard."""

    def __init__(
        self,
        executable: str | Path,
        out_dir: str | Path,
        *,
        discovery_timeout_s: float = 15.0,
        ready_timeout_s: float = 10.0,
        cancel_grace_s: float = 10.0,
        exit_grace_s: float = 5.0,
        term_grace_s: float = 3.0,
        kill_wait_s: float = 2.0,
        log_tail_lines: int = 100,
        startup_timeout_s: float | None = None,
        confirm_timeout_s: float = 30.0,
        finish_timeout_s: float = 60.0,
        continuous_pacing: str = "operator",
    ) -> None:
        # Absolute, so a bare "./worker" is not re-read as a PATH lookup when the process is spawned.
        if continuous_pacing not in ("operator", "window"):
            raise ValueError("continuous_pacing must be 'operator' or 'window'")
        # How the worker paces a continuous step; the page words its instruction to match.
        self.continuous_pacing = continuous_pacing
        self.executable = Path(os.path.abspath(executable))
        # The plain folder the worker writes its .mcal files into.
        self.out_dir = Path(out_dir)
        self.discovery_timeout_s = discovery_timeout_s
        self.ready_timeout_s = ready_timeout_s
        self.cancel_grace_s = cancel_grace_s
        self.exit_grace_s = exit_grace_s
        self.term_grace_s = term_grace_s
        self.kill_wait_s = kill_wait_s
        self.log_tail_lines = log_tail_lines
        # Parent-side deadlines for answers only the worker can give. A worker blocked inside the SDK
        # cannot enforce its own, and operator think time is deliberately never bounded.
        self.startup_timeout_s = (
            discovery_timeout_s + 45.0
            if startup_timeout_s is None
            else startup_timeout_s
        )
        self.confirm_timeout_s = confirm_timeout_s
        self.finish_timeout_s = finish_timeout_s
        # Set when a worker could not be reaped: it may still hold the gloves, so nothing new may start.
        self.unreleased: str | None = None
        self._active: CalibrationAttempt | None = None

    def new_attempt(
        self, on_event: Callable[[proto.WorkerEvent], None] | None = None
    ) -> CalibrationAttempt:
        """Create the attempt for one run; the worker allows only one active attempt at a time."""
        # An unreaped worker may still own the dongles, so admission stays closed until this program restarts.
        if self.unreleased is not None:
            raise AttemptError(self.unreleased)
        if self._active is not None and self._active.state is not AttemptState.DONE:
            raise AttemptError("a calibration attempt is already active")
        attempt = CalibrationAttempt(self, on_event)
        self._active = attempt
        return attempt


class CalibrationAttempt:
    """One worker process, one attempt, one terminal outcome."""

    def __init__(
        self,
        owner: CalibrationWorker,
        on_event: Callable[[proto.WorkerEvent], None] | None,
    ) -> None:
        self.attempt_id = f"cal-{secrets.token_hex(6)}"
        self._owner = owner
        self._on_event = on_event
        self._state = AttemptState.NEW
        self._sides: tuple[proto.Side, ...] = ()
        self._proc: asyncio.subprocess.Process | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._tasks: list[asyncio.Task[Any]] = []
        self._tail: collections.deque[str] = collections.deque(
            maxlen=owner.log_tail_lines
        )
        self._outcome: asyncio.Future[CalibrationOutcome] = (
            asyncio.get_running_loop().create_future()
        )

        self._ready = asyncio.Event()
        self._cancel_requested = asyncio.Event()
        self._result_seen = asyncio.Event()
        self._start_sent = False
        # Why the worker refused ``start``, when it did; the worker stays idle and never reports a result.
        self._start_rejection: str | None = None
        self._user_cancel = False
        self._reader_task: asyncio.Task[None] | None = None
        self._abort_reason: str | None = None
        self._violation: str | None = None
        self._deadline_at: float | None = None
        self._deadline_span = 0.0
        self._deadline_what = ""
        self._unreaped = False

        self._result: proto.Result | None = None
        self._exports: dict[proto.Side, proto.Exported] = {}
        self._seen_tokens: set[str] = set()
        self._prompt: proto.Prompt | None = None
        self._pending_confirm: tuple[str, asyncio.Future[None]] | None = None
        self._accepted_token: str | None = None

    @property
    def state(self) -> AttemptState:
        return self._state

    @property
    def prompt(self) -> proto.Prompt | None:
        """The prompt awaiting confirmation, if any."""
        return self._prompt

    @property
    def sides(self) -> tuple[proto.Side, ...]:
        return self._sides

    async def start(self, sides: Sequence[proto.Side]) -> None:
        """Launch the worker and send ``start``. Worker-side failures settle :meth:`outcome`, not raise."""
        if self._state is not AttemptState.NEW:
            raise AttemptError("an attempt can be started only once")
        sides = tuple(sides)
        # Validates the sides before anything is created or launched.
        proto.encode_start(self.attempt_id, sides, "/")
        self._sides = tuple(s for s in proto.Side if s in sides)
        self._state = AttemptState.STARTING
        try:
            self._owner.out_dir.mkdir(parents=True, exist_ok=True)
            out_dir = self._owner.out_dir.resolve(strict=True)
            # An existing folder the worker cannot write would only surface as a rejected start much later.
            if not os.access(out_dir, os.W_OK | os.X_OK):
                raise PermissionError(errno.EACCES, "not a writable directory")
        except OSError as exc:
            message = f"cannot use {self._owner.out_dir}: {exc.strerror or exc}"
            self._finish(
                CalibrationOutcome(
                    OutcomeKind.FAILED, self.attempt_id, self._sides, message
                )
            )
            raise AttemptError(message) from None
        try:
            await self._launch()
        except OSError as exc:
            self._finish_failed(
                "spawn_failed",
                f"cannot launch {self._owner.executable}: {exc.strerror or exc}",
            )
            return
        self._tasks.append(
            asyncio.create_task(self._supervise(), name=f"{self.attempt_id}-supervise")
        )
        await self._wait_ready()
        if (
            self._state is AttemptState.DONE
            or self._abort_reason
            or self._cancel_requested.is_set()
        ):
            return
        await self._send(proto.encode_start(self.attempt_id, self._sides, str(out_dir)))
        self._start_sent = True
        self._state = AttemptState.RUNNING
        self._expect(
            "SDK initialisation, connection and discovery",
            self._owner.startup_timeout_s,
        )

    async def confirm(self, token: str) -> None:
        """Start the SDK step offered under ``token``; returns when the worker accepts it.

        A repeated confirmation of a token already accepted or in flight never reaches the worker twice.
        """
        if self._accepted_token == token:
            return
        if self._pending_confirm is not None and self._pending_confirm[0] == token:
            await asyncio.shield(self._pending_confirm[1])
            return
        if self._state is not AttemptState.RUNNING:
            raise AttemptEnded("the attempt is not running")
        if (
            self._prompt is None
            or self._prompt.token != token
            or self._pending_confirm is not None
        ):
            raise ConfirmError("that prompt is not awaiting confirmation")
        done: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        self._pending_confirm = (token, done)
        self._expect("the SDK to start the step", self._owner.confirm_timeout_s)
        await self._send(proto.encode_confirm(self.attempt_id, token))
        await asyncio.shield(done)

    async def cancel(self) -> None:
        """Ask the worker to cancel. Idempotent; a result already reported is not changed."""
        if self._state is AttemptState.DONE or self._cancel_requested.is_set():
            return
        if self._state is AttemptState.NEW:
            self._finish(
                CalibrationOutcome(
                    OutcomeKind.CANCELLED,
                    self.attempt_id,
                    self._sides,
                    "cancelled before start",
                )
            )
            return
        self._user_cancel = True
        self._deadline_at = None
        self._cancel_requested.set()
        if self._start_sent:
            await self._send(proto.encode_cancel(self.attempt_id))
        else:
            # The worker has not been told to start, so there is nothing to wind down.
            self._abort("cancelled before the worker was started")

    async def outcome(self) -> CalibrationOutcome:
        """Wait for the terminal outcome."""
        return await asyncio.shield(self._outcome)

    async def _launch(self) -> None:
        parent, child = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            self._proc = await asyncio.create_subprocess_exec(
                str(self._owner.executable),
                f"--ipc-fd={child.fileno()}",
                f"--discovery-timeout={self._owner.discovery_timeout_s}",
                f"--continuous-pacing={self._owner.continuous_pacing}",
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                pass_fds=(child.fileno(),),
            )
        except BaseException:
            parent.close()
            raise
        finally:
            child.close()
        # The line limit leaves room for the newline; longer lines surface as a protocol violation.
        reader, self._writer = await asyncio.open_unix_connection(
            sock=parent, limit=proto.MAX_LINE_BYTES + 1
        )
        assert self._proc.stdout is not None and self._proc.stderr is not None
        self._tasks.append(
            asyncio.create_task(self._drain(self._proc.stdout, "stdout"))
        )
        self._tasks.append(
            asyncio.create_task(self._drain(self._proc.stderr, "stderr"))
        )
        self._reader_task = asyncio.create_task(
            self._read_events(reader), name=f"{self.attempt_id}-reader"
        )
        self._tasks.append(self._reader_task)

    async def _wait_ready(self) -> None:
        """Wait for the worker's ``ready`` event; abort the worker if it does not arrive in time."""
        waiters = {
            asyncio.ensure_future(self._ready.wait()),
            asyncio.ensure_future(self._cancel_requested.wait()),
            self._outcome,
        }
        try:
            done, _ = await asyncio.wait(
                waiters,
                timeout=self._owner.ready_timeout_s,
                return_when=asyncio.FIRST_COMPLETED,
            )
        finally:
            for waiter in waiters:
                if waiter is not self._outcome:
                    waiter.cancel()
        if not done:
            self._abort(
                f"worker did not report ready within {self._owner.ready_timeout_s:g}s"
            )

    async def _send(self, line: bytes) -> None:
        writer = self._writer
        if writer is None or writer.is_closing():
            return
        try:
            writer.write(line)
            await writer.drain()
        except (ConnectionError, OSError):
            # A dead worker is reported by the supervisor through its missing result.
            logger.debug("worker connection closed while sending")

    async def _drain(self, stream: asyncio.StreamReader, name: str) -> None:
        """Keep a bounded tail of the worker's stdout/stderr and never let a full pipe block it."""
        partial = ""
        while chunk := await stream.read(_DRAIN_CHUNK):
            partial += chunk.decode("utf-8", errors="replace")
            *lines, partial = partial.split("\n")
            for line in lines:
                self._tail.append(f"{name}: {line[:_LOG_LINE_CHARS]}")
            partial = partial[-_LOG_LINE_CHARS:]
        if partial:
            self._tail.append(f"{name}: {partial[:_LOG_LINE_CHARS]}")

    async def _read_events(self, reader: asyncio.StreamReader) -> None:
        """Parse worker lines until EOF; any violation aborts the worker."""
        while True:
            try:
                line = await reader.readline()
            except (ValueError, asyncio.LimitOverrunError):
                self._violate(f"worker line exceeds {proto.MAX_LINE_BYTES} bytes")
                return
            except (ConnectionError, OSError):
                return
            if not line:
                return
            if not line.endswith(b"\n"):
                self._violate("worker line is not newline-terminated")
                return
            try:
                if self._result is not None:
                    raise proto.ProtocolError("worker sent a message after its result")
                event = proto.parse_event(line[:-1])
                self._handle(event)
            except proto.ProtocolError as exc:
                self._violate(str(exc))
                return

    def _violate(self, message: str) -> None:
        if self._violation is None:
            self._violation = message
            logger.warning("worker protocol violation: %s", message)
        self._abort(f"protocol violation: {message}")

    def _expect(self, what: str, span: float) -> None:
        """Start the clock on an answer only the worker can give."""
        self._deadline_what = what
        self._deadline_span = span
        self._deadline_at = asyncio.get_running_loop().time() + span

    def _touch(self) -> None:
        """Any worker event proves it is alive, so a running deadline restarts."""
        if self._deadline_at is not None:
            self._deadline_at = asyncio.get_running_loop().time() + self._deadline_span

    async def _watchdog(self) -> None:
        """Abort when the worker stays silent past an armed deadline."""
        loop = asyncio.get_running_loop()
        while True:
            await asyncio.sleep(_WATCHDOG_POLL_S)
            if self._deadline_at is not None and loop.time() >= self._deadline_at:
                self._abort(
                    f"the worker did not respond within {self._deadline_span:g} s while waiting for "
                    f"{self._deadline_what}; it may be blocked inside the SDK"
                )
                return

    def _abort(self, reason: str) -> None:
        """Stop waiting for the worker: the supervisor terminates it without a grace period."""
        if self._abort_reason is None:
            self._abort_reason = reason
        self._cancel_requested.set()

    def _handle(self, event: proto.WorkerEvent) -> None:
        """Check ``event`` against the attempt's state, then record it and notify the listener."""
        if not self._ready.is_set():
            if not isinstance(event, proto.Ready):
                raise proto.ProtocolError(
                    f"expected ready first, got {type(event).__name__}"
                )
            self._ready.set()
        elif isinstance(event, proto.Ready):
            raise proto.ProtocolError("ready sent twice")
        elif not self._start_sent and not self._is_idle_cancel_result(event):
            raise proto.ProtocolError(
                f"{type(event).__name__} before the attempt was started"
            )
        attempt = getattr(event, "attempt", self.attempt_id)
        if attempt not in (self.attempt_id, None):
            raise proto.ProtocolError(
                f"event for attempt {attempt!r}, expected {self.attempt_id!r}"
            )
        self._touch()
        if isinstance(event, proto.Prompt):
            self._on_prompt(event)
        elif isinstance(event, proto.StepAccepted):
            self._on_accepted(event)
        elif isinstance(event, proto.StepProgress):
            self._on_progress(event)
        elif isinstance(event, proto.Rejected):
            self._on_rejected(event)
        elif isinstance(event, proto.Exported):
            self._on_exported(event)
        elif isinstance(event, proto.Result):
            self._deadline_at = None
            self._result = event
            self._result_seen.set()
        self._notify(event)

    def _is_idle_cancel_result(self, event: proto.WorkerEvent) -> bool:
        """True for the clean result an idle worker sends when we cancelled it before ``start``."""
        return (
            self._user_cancel
            and isinstance(event, proto.Result)
            and event.attempt is None
            and event.outcome is proto.Outcome.CANCELLED
        )

    def _on_prompt(self, prompt: proto.Prompt) -> None:
        if self._prompt is not None and self._accepted_token != self._prompt.token:
            raise proto.ProtocolError(
                "prompt offered while the previous one is unconfirmed"
            )
        if prompt.token in self._seen_tokens:
            raise proto.ProtocolError("prompt token reused")
        if prompt.side not in self._sides:
            raise proto.ProtocolError("prompt for a side that was not requested")
        self._seen_tokens.add(prompt.token)
        self._prompt = prompt
        # The operator's think time is unbounded.
        self._deadline_at = None

    def _on_accepted(self, accepted: proto.StepAccepted) -> None:
        if self._pending_confirm is None or self._pending_confirm[0] != accepted.token:
            raise proto.ProtocolError("step accepted without a matching confirmation")
        done = self._pending_confirm[1]
        self._pending_confirm = None
        self._accepted_token = accepted.token
        if not done.done():
            done.set_result(None)
        # The next prompt follows at once, or the worker finishes and exports after the last step.
        window = self._prompt.pacing_s if self._prompt is not None else 0.0
        self._expect(
            "the next step or the result", self._owner.finish_timeout_s + window
        )

    def _on_progress(self, progress: proto.StepProgress) -> None:
        prompt = self._prompt
        if (
            prompt is None
            or progress.token != prompt.token
            or self._accepted_token != prompt.token
        ):
            raise proto.ProtocolError("progress for a step that is not running")
        if progress.elapsed_s > prompt.pacing_s + _PROGRESS_SLACK_S:
            raise proto.ProtocolError("progress exceeds the step duration")

    def _on_rejected(self, rejected: proto.Rejected) -> None:
        # A refused start means the worker never opened the SDK: end the attempt with its reason now.
        if rejected.command == "start":
            self._start_rejection = f"{rejected.reason.value}: {rejected.message}"
            self._abort(f"the worker rejected start ({self._start_rejection})")
            return
        pending = self._pending_confirm
        if (
            pending is not None
            and rejected.command == "confirm"
            and rejected.token in (None, pending[0])
        ):
            self._pending_confirm = None
            # The prompt stays offered, so we are back to waiting for the operator.
            self._deadline_at = None
            if not pending[1].done():
                pending[1].set_exception(
                    ConfirmError(f"{rejected.reason.value}: {rejected.message}")
                )

    def _on_exported(self, exported: proto.Exported) -> None:
        if exported.side not in self._sides or exported.side in self._exports:
            raise proto.ProtocolError(
                f"unexpected export for the {exported.side.value} side"
            )
        self._exports[exported.side] = exported

    def _notify(self, event: proto.WorkerEvent) -> None:
        if self._on_event is None:
            return
        try:
            self._on_event(event)
        except Exception:
            # A faulty listener must not stop supervision of the worker.
            logger.exception("calibration event listener raised")

    async def _supervise(self) -> None:
        """Run :meth:`_supervise_worker`, settling the outcome even if supervision is cancelled."""
        try:
            await self._supervise_worker()
        except asyncio.CancelledError:
            if self._proc is not None and self._proc.returncode is None:
                with contextlib.suppress(ProcessLookupError):
                    self._proc.kill()
            self._finish(
                self._outcome_of(
                    OutcomeKind.UNCERTAIN, "supervision was cancelled", None, True
                )
            )
            raise

    async def _supervise_worker(self) -> None:
        """Wait for the worker to end (or be stopped), reap it, and settle the outcome."""
        reader = self._reader_task
        assert reader is not None
        cancel = asyncio.create_task(self._cancel_requested.wait())
        result = asyncio.create_task(self._result_seen.wait())
        watchdog = asyncio.create_task(self._watchdog())
        try:
            await asyncio.wait(
                {reader, cancel, result}, return_when=asyncio.FIRST_COMPLETED
            )
            if not reader.done():
                if self._abort_reason:
                    grace = 0.0
                elif self._result_seen.is_set():
                    grace = self._owner.exit_grace_s
                else:
                    grace = self._owner.cancel_grace_s
                with contextlib.suppress(TimeoutError):
                    async with asyncio.timeout(grace):
                        await asyncio.shield(reader)
        finally:
            cancel.cancel()
            result.cancel()
            watchdog.cancel()
        self._state = AttemptState.FINISHING
        forced = await self._stop_process()
        # Exit not established (even SIGKILL did not reap it): the process may still hold the gloves.
        self._unreaped = self._proc is not None and self._proc.returncode is None
        if not reader.done():
            reader.cancel()
        await self._settle_tasks()
        await self._finalize(forced)

    async def _settle_tasks(self) -> None:
        """Wait a bounded time for the drain and reader tasks; a surviving process can hold their pipes open."""
        others = [t for t in self._tasks if t is not asyncio.current_task()]
        if not others:
            return
        _done, pending = await asyncio.wait(others, timeout=_TASK_SETTLE_S)
        for task in pending:
            task.cancel()

    async def _stop_process(self) -> bool:
        """Reap the worker, escalating to SIGTERM then SIGKILL. Returns whether it had to be forced."""
        proc = self._proc
        assert proc is not None
        grace = 0.0 if self._abort_reason else self._owner.exit_grace_s
        with contextlib.suppress(TimeoutError):
            async with asyncio.timeout(grace):
                await proc.wait()
        if proc.returncode is not None:
            return False
        with contextlib.suppress(ProcessLookupError):
            proc.terminate()
        try:
            async with asyncio.timeout(self._owner.term_grace_s):
                await proc.wait()
            return True
        except TimeoutError:
            pass
        with contextlib.suppress(ProcessLookupError):
            proc.kill()
        with contextlib.suppress(TimeoutError):
            async with asyncio.timeout(self._owner.kill_wait_s):
                await proc.wait()
        return True

    async def _finalize(self, forced: bool) -> None:
        """Turn the evidence collected into the one terminal outcome, publishing files only on success."""
        proc = self._proc
        code = proc.returncode if proc is not None else None
        try:
            outcome = await self._decide(code, forced)
        except Exception as exc:
            logger.exception("calibration finalization failed")
            outcome = self._outcome_of(
                OutcomeKind.UNCERTAIN,
                f"internal error finishing the attempt: {exc}",
                code,
                forced,
            )
        self._finish(outcome)

    async def _decide(self, code: int | None, forced: bool) -> CalibrationOutcome:
        if self._unreaped:
            pid = self._proc.pid if self._proc is not None else "?"
            message = (
                f"the worker (pid {pid}) could not be stopped and may still be using the gloves; "
                "restart this program once it has exited"
            )
            self._owner.unreleased = message
            return self._outcome_of(
                OutcomeKind.UNCERTAIN,
                message,
                code,
                forced,
                error=proto.WorkerError("worker_unreaped", message),
            )
        if self._start_rejection is not None:
            # Nothing was started, so this is a clean failure rather than an unknown SDK state.
            return self._outcome_of(
                OutcomeKind.FAILED,
                f"the worker rejected start: {self._start_rejection}",
                code,
                forced,
                error=proto.WorkerError("start_rejected", self._start_rejection),
            )
        result = self._result
        if result is not None and not result.released:
            # The process exited, but it reported that the SDK was not released: the gloves may still be in use.
            detail = f": {result.error.message}" if result.error is not None else ""
            message = (
                f"the worker could not release the SDK{detail}; restart this program once the gloves are known "
                "to be free"
            )
            self._owner.unreleased = message
            return self._outcome_of(
                OutcomeKind.UNCERTAIN,
                message,
                code,
                forced,
                error=proto.WorkerError("worker_unreleased", message),
            )
        if self._violation is not None:
            # Before `start` the worker has not opened the SDK, so only a started worker leaves it unknown.
            kind = OutcomeKind.UNCERTAIN if self._start_sent else OutcomeKind.FAILED
            return self._outcome_of(
                kind,
                f"worker broke the protocol ({self._violation})",
                code,
                forced,
                error=proto.WorkerError("protocol_error", self._violation),
            )
        if result is None:
            if not self._start_sent:
                if self._user_cancel:
                    return self._outcome_of(
                        OutcomeKind.CANCELLED,
                        "cancelled before the worker started",
                        code,
                        forced,
                    )
                return self._outcome_of(
                    OutcomeKind.FAILED,
                    self._abort_reason
                    or f"worker exited before it was started (exit code {code})",
                    code,
                    forced,
                    error=proto.WorkerError(
                        "worker_exit",
                        self._abort_reason or "worker exited before start",
                    ),
                )
            reason = (
                self._abort_reason
                or f"worker ended without reporting a result (exit code {code})"
            )
            return self._outcome_of(
                OutcomeKind.UNCERTAIN,
                f"{reason}; the worker was stopped and its SDK state is unknown",
                code,
                forced,
                error=proto.WorkerError("worker_exit", f"no result; exit code {code}"),
            )
        if forced:
            return self._outcome_of(
                OutcomeKind.UNCERTAIN,
                f"worker reported {result.outcome.value} but had to be terminated afterwards",
                code,
                forced,
            )
        if result.outcome is proto.Outcome.FAILED:
            assert result.error is not None
            return self._outcome_of(
                OutcomeKind.FAILED,
                result.error.message,
                code,
                forced,
                error=result.error,
            )
        if code != 0:
            return self._outcome_of(
                OutcomeKind.UNCERTAIN,
                f"worker reported {result.outcome.value} but exited with code {code}",
                code,
                forced,
            )
        if result.outcome is proto.Outcome.CANCELLED:
            return self._outcome_of(OutcomeKind.CANCELLED, "cancelled", code, forced)
        return self._completed(result, code, forced)

    def _completed(
        self, result: proto.Result, code: int | None, forced: bool
    ) -> CalibrationOutcome:
        """A completed result, once its files match what the worker reported exporting."""
        reported = {f.side: f for f in result.files}
        exported = {
            side: proto.ExportedFile(side, e.file, e.bytes, e.sha256)
            for side, e in self._exports.items()
        }
        if reported != exported or set(reported) != set(self._sides):
            return self._outcome_of(
                OutcomeKind.UNCERTAIN,
                "worker's completed result does not match the files it exported",
                code,
                forced,
                error=proto.WorkerError(
                    "protocol_error", "result files differ from exported files"
                ),
            )
        return self._outcome_of(
            OutcomeKind.COMPLETED,
            "calibration saved",
            code,
            forced,
            files=tuple(exported[side] for side in self._sides),
        )

    def _outcome_of(
        self,
        kind: OutcomeKind,
        message: str,
        code: int | None,
        forced: bool,
        *,
        error: proto.WorkerError | None = None,
        files: tuple[proto.ExportedFile, ...] = (),
    ) -> CalibrationOutcome:
        return CalibrationOutcome(
            kind=kind,
            attempt=self.attempt_id,
            sides=self._sides,
            message=message,
            files=files,
            error=error,
            exit_code=code,
            forced_termination=forced,
            log_tail=tuple(self._tail),
        )

    def _remove_exports(self) -> None:
        """Delete everything this attempt wrote into the folder, including a half-written file.

        The attempt id is random and prefixes every name the worker writes, so this matches no other file.
        """
        for path in self._owner.out_dir.glob(f"{self.attempt_id}_*"):
            with contextlib.suppress(OSError):
                path.unlink()

    def _finish_failed(self, code: str, message: str) -> None:
        self._finish(
            self._outcome_of(
                OutcomeKind.FAILED,
                message,
                None,
                False,
                error=proto.WorkerError(code, message),
            )
        )

    def _finish(self, outcome: CalibrationOutcome) -> None:
        """Settle the attempt exactly once: drop partial files, fail waiters, record the outcome."""
        if self._outcome.done():
            return
        # Only a completed attempt keeps files. A worker that may still be running can still write, so
        # nothing is removed while it is unreaped.
        if outcome.kind is not OutcomeKind.COMPLETED and not self._unreaped:
            self._remove_exports()
        self._state = AttemptState.DONE
        self._prompt = None
        if self._pending_confirm is not None:
            pending = self._pending_confirm[1]
            self._pending_confirm = None
            if not pending.done():
                pending.set_exception(
                    AttemptEnded("the attempt ended before the step was accepted")
                )
        self._outcome.set_result(outcome)
