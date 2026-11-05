# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Single-worker pipelined TeleopSession step runner."""

from __future__ import annotations

import copy
from collections import deque
from dataclasses import dataclass, replace
import threading
import time
from collections.abc import Callable
from typing import Any, Dict

from isaaccapture.retargeting_engine.interface.retargeter_core_types import (
    ComputeContext,
    GraphTime,
    RetargeterIO,
)
from isaaccapture.retargeting_engine.interface.execution_events import (
    ExecutionEvents,
    ExecutionState,
)

from .config import RetargetingExecutionConfig


StepExecutor = Callable[["StepRequest"], tuple[RetargeterIO, ComputeContext]]


@dataclass(frozen=True)
class StepRequest:
    """One application-frame request for the sync path or worker."""

    frame_id: int
    external_inputs: Dict[str, RetargeterIO] | None
    graph_time: GraphTime | None
    execution_events: ExecutionEvents | None
    submitted_time_s: float
    reset_requested: bool = False


@dataclass(frozen=True)
class RetargetFrame:
    """Completed retarget result stored as the latest reusable frame."""

    frame_id: int
    outputs: RetargeterIO
    context: ComputeContext
    submitted_time_s: float
    started_time_s: float
    completed_time_s: float
    compute_duration_s: float
    reset_generation: int = 0


class AsyncRetargetWorkerError(RuntimeError):
    """Raised on the application thread after the worker fails a step request."""


class AsyncRetargetRunnerStopped(RuntimeError):
    """Raised when the retarget runner cannot accept or run more work."""


def snapshot_retargeter_io(io: RetargeterIO) -> RetargeterIO:
    """Return an owned copy of a returned retargeter I/O dictionary.

    Pipelined mode can return the same completed frame on more than one
    application step when the worker has not published a newer result. Copying
    the returned value prevents caller mutation from corrupting the cached
    frame. If a value cannot be copied, pipelined mode fails with a clear error
    instead of silently aliasing mutable state across frames; such values
    should implement ``create_snapshot()`` or run in sync mode.
    """

    return {name: _snapshot_value(group) for name, group in io.items()}


def snapshot_compute_context(context: ComputeContext) -> ComputeContext:
    """Return an owned copy of a compute context crossing an async boundary.

    ``last_context`` is public mutable state. Copying keeps user edits to the
    public object from corrupting the cached frame that pipelined mode may
    return again as the latest completed result.
    """

    return ComputeContext(
        graph_time=GraphTime(
            sim_time_ns=context.graph_time.sim_time_ns,
            real_time_ns=context.graph_time.real_time_ns,
        ),
        execution_events=ExecutionEvents(
            reset=bool(context.execution_events.reset),
            execution_state=ExecutionState(context.execution_events.execution_state),
        ),
    )


def _snapshot_value(value: Any) -> Any:
    """Copy one value using TensorGroup hooks first, then deepcopy.

    External inputs cross the application/worker thread boundary and returned
    outputs may be reused when the same completed frame is returned again. Both
    paths need owned data; a failed copy is a correctness problem, not an
    optimization miss.
    """
    create_snapshot = getattr(value, "create_snapshot", None)
    try:
        if create_snapshot is not None:
            return create_snapshot()
        return copy.deepcopy(value)
    except (TypeError, copy.Error) as exc:
        raise TypeError(
            "Pipelined retargeting requires external inputs and returned outputs "
            "to be snapshot-copyable. Implement create_snapshot() for this value "
            "or use RetargetingExecutionConfig(mode='sync')."
        ) from exc


def snapshot_pipeline_inputs(
    inputs: Dict[str, RetargeterIO],
) -> Dict[str, RetargeterIO]:
    """Return strict owned copies of caller-provided external inputs.

    Only explicit ``external_inputs`` cross into the worker. DeviceIO polling
    happens inside the worker, which is why async mode does not need broad
    schema/message-channel snapshotting. If an external value cannot be copied,
    fail early rather than racing caller-owned state.
    """

    return {
        name: {
            input_name: _snapshot_value(group) for input_name, group in values.items()
        }
        for name, values in inputs.items()
    }


class AsyncRetargetRunner:
    """Run whole TeleopSession steps on one background worker.

    The runner accepts prepared ``StepRequest`` objects from the application
    thread and executes the session's normal sync step serially on the worker.
    It intentionally has only one running request and one pending request so
    stateful retargeters are never called concurrently and backlog cannot grow
    without bound.
    """

    def __init__(self, step_fn: StepExecutor, cfg: RetargetingExecutionConfig):
        self._cfg = cfg
        pacing = cfg.pacing
        self._step_fn = step_fn
        self._cond = threading.Condition()
        self._pending: StepRequest | None = None
        self._published: RetargetFrame | None = None
        self._reset_generation = 0
        self._acknowledged_reset_generation = 0
        self._dropped_submissions = 0
        self._last_submission_time_s: float | None = None
        self._submission_count = 0
        startup = pacing.startup_state()
        self._submit_period_s = startup.submit_period_s
        self._compute_duration_s = startup.compute_duration_s
        self._compute_duration_samples = deque(maxlen=startup.compute_sample_window)
        self._stop = False
        self._closed = False
        self._exception: BaseException | None = None
        self._exception_reported = False
        self._thread: threading.Thread | None = None

    @property
    def dropped_submissions(self) -> int:
        """Total number of unstarted requests replaced by newer requests."""

        with self._cond:
            return self._dropped_submissions

    def start(self) -> None:
        """Start the worker if it is not already running.

        Runners are intentionally one-shot. ``TeleopSession`` creates a new
        runner for each context-manager entry, which avoids reusing cached
        frames or pacing estimates across independent DeviceIO/OpenXR sessions.
        """

        with self._cond:
            if self._thread is not None:
                return
            if self._closed:
                raise AsyncRetargetRunnerStopped(
                    "Async retarget runner cannot be restarted after stop"
                )
            self._stop = False
            self._exception = None
            self._exception_reported = False
            self._thread = threading.Thread(
                target=self._run,
                name="IsaacTeleopAsyncRetarget",
                daemon=True,
            )
            self._thread.start()

    def stop(self, timeout_s: float | None = None) -> bool:
        """Stop accepting pending work and wait for the active step to finish.

        ``TeleopSession.__exit__`` calls this before releasing DeviceIO/OpenXR
        resources. The running request is not interrupted because retargeters
        and DeviceIO updates generally are not cancellation-safe. Stopping
        rejects new submissions, clears unstarted pending work, wakes
        ``wait_for_frame()`` observers, and lets any running worker step leave
        cleanly.

        Returns:
            ``True`` when no worker thread remains alive. ``False`` means the
            caller supplied a timeout and the active step is still finishing;
            call ``stop()`` again after the step can complete.
        """

        with self._cond:
            self._stop = True
            self._closed = True
            self._pending = None
            self._cond.notify_all()
            thread = self._thread

        if thread is not None:
            thread.join(timeout=timeout_s)

        with self._cond:
            stopped = thread is None or not thread.is_alive()
            if self._thread is thread and stopped:
                self._thread = None
            return stopped

    def raise_if_failed(self, *, only_unreported: bool = False) -> None:
        """Re-raise a captured worker failure on the application thread.

        Worker exceptions cannot be raised directly where they occur, so
        ``TeleopSession.step()`` and ``__exit__`` call this to surface failures
        at deterministic public API boundaries.
        """

        with self._cond:
            self._raise_if_failed_locked(only_unreported=only_unreported)

    def publish_seed(self, frame: RetargetFrame) -> None:
        """Publish the synchronous first frame before background work starts.

        The first pipelined ``step()`` has no completed output to return, so
        ``TeleopSession`` runs it synchronously, starts the worker, then seeds
        the runner's latest-result slot with that frame. The seed also anchors
        pacing estimates so the first background request can be scheduled
        toward the next application frame.
        """

        with self._cond:
            self._published = frame
            self._record_submission_locked(frame.submitted_time_s)
            self._record_compute_duration_locked(frame.compute_duration_s)
            self._cond.notify_all()

    def submit(self, request: StepRequest) -> int:
        """Submit an ordinary step request.

        If a previous request is pending but has not started, it is replaced by
        this newer request. That keeps latency bounded: the worker either
        finishes the currently running step or executes the most recent pending
        step, but it never drains a superseded backlog.

        Returns:
            Number of unstarted pending requests dropped by this submission.
        """

        with self._cond:
            self._raise_if_failed_locked()
            self._raise_if_stopped_locked(
                "Async retarget runner stopped before submission was accepted"
            )
            dropped = 0
            if self._pending is not None:
                dropped = 1
                self._dropped_submissions += 1
                if self._pending.reset_requested or (
                    self._pending.execution_events is not None
                    and self._pending.execution_events.reset
                ):
                    # Reset is an idempotent edge, not replaceable latest state.
                    # Do not override the new request's live control decoding.
                    request = replace(request, reset_requested=True)
            self._pending = request
            self._record_submission_locked(request.submitted_time_s)
            self._cond.notify_all()
            return dropped

    def latest(self) -> RetargetFrame | None:
        """Return the latest published frame without waiting.

        This raises any worker failure instead of returning an older frame.
        """

        with self._cond:
            self._raise_if_failed_locked()
            if self._published is None:
                return None
            context = snapshot_compute_context(self._published.context)
            context.execution_events.reset = (
                self._reset_generation > self._acknowledged_reset_generation
            )
            return replace(
                self._published,
                context=context,
                reset_generation=self._reset_generation,
            )

    def acknowledge_reset(self, generation: int) -> None:
        """Acknowledge delivered resets, without clearing a concurrently newer edge.

        Latest state may replace completed outputs. Pending reset pulses collapse
        to one idempotent reset, but remain visible until the application has
        successfully copied the returned frame. Observers do not acknowledge it.
        """
        with self._cond:
            if not 0 <= generation <= self._reset_generation:
                raise ValueError("Unknown reset generation")
            self._acknowledged_reset_generation = max(
                self._acknowledged_reset_generation, generation
            )

    def wait_for_frame(
        self,
        frame_id: int,
        timeout_s: float | None = None,
    ) -> RetargetFrame | None:
        """Wait until at least ``frame_id`` has been published.

        Tests and internal observers use this to detect worker progress.
        Returning a newer frame is acceptable because it still proves the worker
        reached the requested frame id.
        """

        deadline = None if timeout_s is None else time.monotonic() + timeout_s
        with self._cond:
            while True:
                self._raise_if_failed_locked()
                if self._published is not None and self._published.frame_id >= frame_id:
                    return self._published
                if self._stop:
                    return None
                if deadline is None:
                    self._cond.wait()
                else:
                    wait_s = deadline - time.monotonic()
                    if wait_s <= 0:
                        return None
                    self._cond.wait(timeout=wait_s)

    def _run(self) -> None:
        """Worker loop: execute one ready request at a time and publish results.

        This is the only place the step executor runs. That invariant is what
        keeps stateful retargeters and DeviceIO polling serialized even though
        the application thread keeps calling ``TeleopSession.step()``.
        """
        while True:
            with self._cond:
                request = self._take_ready_request_locked()
            if request is None:
                return

            try:
                started = time.monotonic()
                outputs, context = self._step_fn(request)
                completed = time.monotonic()
                frame = RetargetFrame(
                    frame_id=request.frame_id,
                    outputs=snapshot_retargeter_io(outputs),
                    context=snapshot_compute_context(context),
                    submitted_time_s=request.submitted_time_s,
                    started_time_s=started,
                    completed_time_s=completed,
                    compute_duration_s=completed - started,
                )
            except BaseException as exc:
                with self._cond:
                    self._exception = exc
                    self._exception_reported = False
                    self._cond.notify_all()
                return

            with self._cond:
                self._record_compute_duration_locked(frame.compute_duration_s)
                if frame.context.execution_events.reset:
                    self._reset_generation += 1
                self._published = frame
                self._cond.notify_all()

    def _take_ready_request_locked(self) -> StepRequest | None:
        """Return the latest unstarted request once its pacing delay has elapsed.

        Pacing waits keep the request in ``_pending`` until it is ready to run.
        That lets a newer application frame replace delayed work instead of
        spending time on a superseded full step.
        """
        while True:
            while self._pending is None and not self._stop:
                self._cond.wait()
            if self._stop:
                return None

            request = self._pending
            assert request is not None
            if not self._wait_until_ready_locked(request):
                continue
            if self._pending is not request:
                continue
            self._pending = None
            self._cond.notify_all()
            return request

    def _raise_if_failed_locked(self, *, only_unreported: bool = False) -> None:
        """Raise the stored worker exception while the condition lock is held."""
        if self._exception is not None:
            if only_unreported and self._exception_reported:
                return
            self._exception_reported = True
            raise AsyncRetargetWorkerError(
                "Async retarget worker failed"
            ) from self._exception

    def _raise_if_stopped_locked(self, message: str) -> None:
        """Reject new or still-blocked submissions after teardown starts."""
        if self._stop:
            raise AsyncRetargetRunnerStopped(message)

    def _record_submission_locked(self, submitted_time_s: float) -> None:
        """Update the application step-period estimate from submit cadence."""
        if self._last_submission_time_s is not None:
            period_s = submitted_time_s - self._last_submission_time_s
            pacing = self._cfg.pacing
            self._submit_period_s = pacing.update_submit_period_s(
                self._submit_period_s,
                period_s,
            )
        self._last_submission_time_s = submitted_time_s
        self._submission_count += 1

    def _record_compute_duration_locked(self, duration_s: float) -> None:
        """Update the retarget compute-duration estimate."""
        pacing = self._cfg.pacing
        self._compute_duration_s = pacing.update_compute_duration_s(
            self._compute_duration_s,
            duration_s,
        )
        self._compute_duration_samples.append(duration_s)

    def _wait_until_ready_locked(self, request: StepRequest) -> bool:
        """Wait for ``request`` pacing while letting newer submissions replace it.

        Returns:
            ``True`` when this request is still pending and ready to execute.
        """
        sleep_s = self._compute_pacing_sleep_s(request, time.monotonic())
        if sleep_s <= 0.0:
            return True

        deadline = time.monotonic() + sleep_s
        while not self._stop and self._pending is request:
            remaining_s = deadline - time.monotonic()
            if remaining_s <= 0.0:
                return self._pending is request
            self._cond.wait(timeout=remaining_s)
        return False

    def _compute_pacing_sleep_s(self, request: StepRequest, now_s: float) -> float:
        """Return how long the worker should wait before executing ``request``."""
        pacing = self._cfg.pacing
        return pacing.compute_delay_s(
            submitted_time_s=request.submitted_time_s,
            now_s=now_s,
            submission_count=self._submission_count,
            submit_period_s=self._submit_period_s,
            compute_duration_s=self._compute_duration_s,
            compute_duration_samples=self._compute_duration_samples,
        )
