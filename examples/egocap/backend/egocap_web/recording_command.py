"""Working command handlers for the browser's recording-command channel.

This module owns the structural payload parsing/validation and the small
batch-identity/elapsed-time/saved-count bookkeeping the frontend's wire
protocol needs on top of the real ``RecordingService`` -- ``RecordingContext``
is a thin wrapper around ``RecordingSessionCoordinator``
(``recording_session.py``), which owns the actual service instance. Real
episodes are written, checked and finalized through it; this module's own
job is just projecting its real facts into the existing wire snapshot
shape and translating a real rejection into the existing reply protocol.

A recording "batch" (the group of episodes sharing one batch_id) is a
concept scoped entirely to this module and its frontend counterpart -- it
is distinct from the capture server's own session (``/api/session``,
``CaptureStatusMonitor``) AND from ``RecordingService``'s own internal
session identity, which this module never touches or renames. There is no
``open_session``/``close_session`` action here -- the live recording strip
only ever rotates to a fresh batch via ``new_batch``, it never opens/
closes anything.
"""

from __future__ import annotations

import logging
import re
import secrets
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from recording_service import ConfigurationError, Disposition, EpisodeResult, StateError

from .recording_session import (
    CoordinatorBusyError,
    CoordinatorSnapshot,
    RecordingSessionCoordinator,
    SessionStatus,
)

#: recording_service.Disposition values the existing frontend wire
#: contract can actually represent as a last_result.disposition (see
#: useCaptureSession.ts's DISPOSITIONS set, unchanged -- any backend/
#: protocol change to extend it is explicitly out of scope here and needs
#: separate approval). RECOVERY_REQUIRED is a real, valid disposition now
#: that interrupt_episode exists, but is NOT one of these -- see
#: RecordingContext._apply_result for how that case is handled without
#: ever serializing an unsupported value.
_WIRE_COMPATIBLE_DISPOSITIONS = frozenset({Disposition.KEPT, Disposition.REJECTED, Disposition.DISCARDED})


_logger = logging.getLogger(__name__)


class RecordingCommandError(ValueError):
    """Raised for a malformed recording command -- a real protocol violation
    (wrong shape/type), not a business rejection."""


@dataclass(frozen=True)
class RecordingCommand:
    """One parsed, structurally valid recording command from the browser.
    ``operator_token`` is intentionally untyped/unvalidated here -- a
    missing, blank or malformed operator name is an ordinary business
    rejection (see ``_normalize_operator``), not a protocol violation that
    should close the connection. ``control_token`` is likewise
    untyped/unvalidated -- apply()'s own check decides whether it actually
    matches the server's secret, not this parse step. ``repr=False`` so a
    logged/printed ``RecordingCommand`` never displays it."""

    request_id: str
    action: str
    payload: dict[str, Any]
    operator_token: object = None
    control_token: object = field(default=None, repr=False)


#: Bound for a normalized operator name (see _normalize_operator and
#: apply()'s own gate).
_MAX_OPERATOR_LENGTH = 128


def _normalize_operator(raw_value: object) -> str | None:
    """Normalizes the operator attribution name by trimming and replacing
    internal whitespace with a single underscore. Returns ``None``
    for anything that isn't a reasonably-sized string, or that normalizes
    to empty -- both are ordinary rejections, not protocol errors."""
    if not isinstance(raw_value, str) or len(raw_value) > _MAX_OPERATOR_LENGTH:
        return None
    normalized = re.sub(r"\s+", "_", raw_value.strip())
    return normalized or None


def _require_str(payload: dict[str, Any], field_name: str) -> None:
    """Raises unless ``payload[field_name]`` is a nonempty string."""
    value = payload.get(field_name)
    if not isinstance(value, str) or value == "":
        raise RecordingCommandError(f"payload.{field_name} must be a nonempty string")


def _require_str_type(payload: dict[str, Any], field_name: str) -> None:
    """Raises unless ``payload[field_name]`` is a string -- empty is a
    structurally valid value here, left to business-rule validation, not a
    protocol violation (an empty task text is an ordinary rejected
    outcome, not a reason to tear down the connection)."""
    if not isinstance(payload.get(field_name), str):
        raise RecordingCommandError(f"payload.{field_name} must be a string")


def _validate_set_task(payload: dict[str, Any]) -> None:
    """Requires a batch id and a string next-task instruction (any string,
    including empty -- whether empty text is acceptable is a business rule
    the handler decides, not a shape check). This component only ever
    edits that text, never a mode/catalog/metadata object."""
    _require_str(payload, "batch_id")
    _require_str_type(payload, "task")


def _validate_batch_target(payload: dict[str, Any]) -> None:
    """Requires a batch id -- used by actions that target the batch alone (start/new_batch)."""
    _require_str(payload, "batch_id")


def _validate_episode_target(payload: dict[str, Any]) -> None:
    """Requires a batch id and an episode id -- used by actions that target one episode (stop/discard)."""
    _require_str(payload, "batch_id")
    _require_str(payload, "episode_id")


#: Per-action payload validators -- each checks exactly the fields that
#: action's real frontend caller sends (see recordingTypes.ts's
#: RecordingActions/RecordingBatchIdentity/RecordingEpisodeIdentity), not
#: a generic schema system.
_PAYLOAD_VALIDATORS: dict[str, Any] = {
    "set_task": _validate_set_task,
    "start_episode": _validate_batch_target,
    "stop_episode": _validate_episode_target,
    "discard_episode": _validate_episode_target,
    "new_batch": _validate_batch_target,
}


def parse_recording_command(raw: dict[str, Any]) -> RecordingCommand:
    """Validates the minimal required shape of a recording command,
    including the specific fields each action's real payload needs.

    Raises :class:`RecordingCommandError` for any missing/wrong-typed field
    or an unrecognized action -- the caller treats this the same as any
    other protocol violation on this socket."""
    request_id = raw.get("request_id")
    action = raw.get("action")
    payload = raw.get("payload")
    if not isinstance(request_id, str) or request_id == "":
        raise RecordingCommandError("request_id must be a nonempty string")
    if not isinstance(action, str) or action not in _PAYLOAD_VALIDATORS:
        # Deliberately doesn't echo the raw `action` value: a client could
        # put an arbitrary string (including a real control_token, hoping
        # it leaks back out through a log line) in this field instead of
        # the actual top-level control_token field.
        raise RecordingCommandError("unknown recording action")
    if not isinstance(payload, dict):
        raise RecordingCommandError("payload must be an object")
    _PAYLOAD_VALIDATORS[action](payload)
    return RecordingCommand(
        request_id=request_id,
        action=action,
        payload=payload,
        operator_token=raw.get("operator_token"),
        control_token=raw.get("control_token"),
    )


def _new_batch_id() -> str:
    """A fresh batch identity, distinct from the capture server's own session id."""
    return f"egocap-batch-{uuid.uuid4()}"


def _control_token_valid(candidate: object, expected: str) -> bool:
    """Timing-safe comparison against the server's one generated secret.
    Rejects a non-``str``, wrong-length, or non-ASCII candidate before
    ``compare_digest`` itself, which raises ``TypeError`` on non-ASCII
    ``str`` input rather than just returning ``False``."""
    if not isinstance(candidate, str) or len(candidate) != len(expected) or not candidate.isascii():
        return False
    return secrets.compare_digest(candidate, expected)


@dataclass
class RecordingContext:
    """Thin wrapper around one ``RecordingSessionCoordinator``: owns the
    web-protocol-only facts the coordinator itself has no reason to know
    about (the batch identity, confirmed-kept count, and the current
    episode's live elapsed-time basis), and projects the coordinator's
    real facts into the existing wire snapshot shape.

    Owned once for the whole backend process lifetime (same pattern as
    ``CaptureStatusMonitor``): constructed in app.py's lifespan, shared by
    every WebSocket connection via ``app.state.recording_context``. This
    keeps the active episode/task/totals intact across a reconnect and
    gives every viewer the same episode identity space. There is no
    terminal/closed state: ``new_batch`` rotates ``batch_id`` in place
    within this same owner rather than closing and requiring a restart.
    """

    coordinator: RecordingSessionCoordinator
    #: The one secret generated for this backend process (see app.py's
    #: lifespan) -- every mutating command must present it. repr=False so
    #: it never appears in a logged/printed RecordingContext.
    control_token: str = field(repr=False)
    #: Whether the frontend should prefill the operator entry field's DRAFT text from
    #: coordinator_snapshot.context.operator -- off by default (see station_config.py's
    #: auto_fill_operator_name). Never a signal to commit an operator automatically: that
    #: still always requires the operator's own explicit "Set operator" action.
    auto_fill_operator_name: bool = False
    batch_id: str = field(default_factory=_new_batch_id)
    saved_count: int = 0
    last_result: dict[str, Any] | None = None
    #: Monotonic clock basis (time.monotonic()) the current episode started
    #: at, or None when idle -- process-local, never serialized directly;
    #: used only to compute current_episode_elapsed_ms fresh at snapshot
    #: time. Set once the coordinator's own on_episode_started fires (not
    #: re-derived from the service, which has no live elapsed-time concept
    #: of its own -- only a finished EpisodeResult's duration_seconds).
    _episode_started_monotonic: float | None = field(default=None, repr=False)
    #: Bumped exactly once a real projection change has actually happened
    #: -- a plain counter, not a derived/hashed value. Two sources feed it:
    #: (1) the coordinator's own on_episode_started/on_episode_result/
    #: on_batch_rotated callbacks, or synchronously for set_task (see
    #: __post_init__ for why these are driven by the coordinator's own
    #: retained operation task rather than this wrapper's own code after
    #: an await); (2) snapshot()'s own plain equality check against
    #: ``_last_decision_state`` (phase/operation/readiness), which is what
    #: makes a mid-operation transition (STARTING/STOPPING/...) or a
    #: readiness change visible even though no discrete callback fires for
    #: either -- not a hash, not a timer, just a comparison run each time
    #: _observe() runs (each viewer's sender loop calls refresh() on every
    #: iteration; see preview_routes.py).
    revision: int = field(default=0, repr=False)
    #: The last (status, phase, active_operation, ready, reason) tuple
    #: snapshot() observed -- compared on every call so a real change with
    #: no dedicated callback still advances ``revision``. None before the
    #: first snapshot() call, so that call always counts as a real first
    #: observation (not implicitly "no change").
    _last_decision_state: tuple[Any, ...] | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        """Registers this wrapper's own projection-update methods as the
        coordinator's callbacks -- see their own docstrings for why this
        is what makes these updates survive a disconnecting requester AND
        cover the automatic mid-recording interruption path, which never
        goes through ``apply()`` at all: both paths already run inside the
        coordinator's own retained/shielded operation task, and these
        callbacks are invoked synchronously from within that same task,
        not a second independently-created one."""
        self.coordinator.on_episode_started = self._on_episode_started
        self.coordinator.on_episode_result = self._on_episode_result
        self.coordinator.on_batch_rotated = self._on_batch_rotated

    def _on_episode_started(self) -> None:
        self._episode_started_monotonic = time.monotonic()
        self.revision += 1

    def _on_episode_result(self, result: EpisodeResult) -> None:
        self._episode_started_monotonic = None
        if result.disposition not in _WIRE_COMPATIBLE_DISPOSITIONS:
            # The interrupted take has no wire form (see below); keep the
            # previous take's outcome from standing in for it.
            self.last_result = None
        if result.disposition in _WIRE_COMPATIBLE_DISPOSITIONS:
            self.last_result = {
                "episode_id": result.episode_id,
                "disposition": result.disposition.value,
                "issues": list(result.issues),
                "duration_ms": int(result.duration_seconds * 1000),
            }
            if result.disposition is Disposition.KEPT:
                self.saved_count += 1
        # else: RECOVERY_REQUIRED (from interrupt_episode -- a real
        # pipeline failure, whether operator-stopped or automatically
        # interrupted mid-recording) is NOT serialized into last_result,
        # since the existing frontend wire contract's EpisodeDisposition
        # parser only accepts KEPT/REJECTED/DISCARDED and would reject the
        # WHOLE snapshot as malformed on an unrecognized value. The real
        # recovery state is still visible through episode_phase
        # (RECOVERY_REQUIRED is an already-supported phase value), but the
        # specific failure reason text has no existing place to go in the
        # current wire contract -- a real, acknowledged protocol gap, not
        # silently patched around here; extending the frontend's
        # EpisodeDisposition/DISPOSITIONS set (or adding a dedicated
        # surfaced-issue field) is a deliberate follow-up decision, not
        # made unilaterally in this backend-only change.
        self.revision += 1

    def _on_batch_rotated(self) -> None:
        self.batch_id = _new_batch_id()
        self.saved_count = 0
        self.last_result = None
        self._episode_started_monotonic = None
        self.revision += 1

    def _decision_state(
        self, coordinator_snapshot: CoordinatorSnapshot, ready: bool, reason: str | None
    ) -> tuple[Any, ...]:
        """The small, stable set of real facts that can change ready_to_record
        or the published phase/operation WITHOUT any discrete callback
        firing for the transition -- compared against the last-observed
        tuple on every ``snapshot()`` call (see ``revision``'s own
        docstring)."""
        service_snapshot = coordinator_snapshot.service
        phase_value = service_snapshot.phase.value if service_snapshot is not None else "IDLE"
        return (coordinator_snapshot.status, phase_value, coordinator_snapshot.active_operation, ready, reason)

    def _observe(self) -> tuple[CoordinatorSnapshot, tuple[Any, ...]]:
        """Reads the coordinator and advances ``revision`` if the decision state changed since the last look."""
        coordinator_snapshot = self.coordinator.snapshot()
        ready, reason = self.coordinator.readiness()
        decision_state = self._decision_state(coordinator_snapshot, ready, reason)
        if decision_state != self._last_decision_state:
            self._last_decision_state = decision_state
            self.revision += 1
        return coordinator_snapshot, decision_state

    def refresh(self) -> None:
        """Notices a readiness or in-progress change that no callback announces. Each viewer's sender calls this
        every iteration, before it compares ``revision``, so such a change is pushed without any command."""
        self._observe()

    def snapshot(self) -> dict[str, Any]:
        """The current recording facts, in the wire shape the frontend's RecordingSnapshot parser expects."""
        coordinator_snapshot, decision_state = self._observe()
        status, phase_value, active_operation, ready, reason = decision_state
        service_snapshot = coordinator_snapshot.service
        current_episode_id = None
        if service_snapshot is not None and service_snapshot.episode is not None:
            current_episode_id = service_snapshot.episode.episode_id
        elapsed_ms = self._elapsed_ms_now() if self._episode_started_monotonic is not None else None
        return {
            "batch_id": self.batch_id,
            "episode_phase": phase_value,
            "current_episode_id": current_episode_id,
            "current_episode_elapsed_ms": elapsed_ms,
            "task_text": coordinator_snapshot.context.task_text,
            # The committed context's operator, exposed only as a PREFILL hint for the entry
            # field -- never used to admit a command: start_episode always requires the
            # browser's own operator_token and never reads this value. null when unconfigured
            # (empty string), matching the frontend's nullable-string convention.
            "default_operator": coordinator_snapshot.context.operator or None,
            # Whether the frontend may use default_operator to prefill the entry field's draft
            # text at all -- off by default. Even when true, the frontend must still require an
            # explicit "Set operator" action; this never means auto-commit.
            "auto_fill_operator_name": self.auto_fill_operator_name,
            "saved_count": self.saved_count,
            "last_result": self.last_result,
            # Real command-admission readiness: the same phase==IDLE/no-
            # active-operation/not-closing conditions start_episode itself
            # enforces, PLUS the real injected readiness_check result (so
            # "no camera selected"/stale-input cases are reflected here
            # too, not just a loose phase-only proxy).
            "ready_to_record": (
                phase_value == "IDLE"
                and active_operation is None
                and status not in (SessionStatus.CLOSING, SessionStatus.FAILED)
                and ready
                and coordinator_snapshot.context.task_text.strip() != ""
            ),
            # Lets a receiver drop a snapshot older than one it already
            # applied; restarts from 0 with the process, so a receiver
            # compares it only within one connection.
            "revision": self.revision,
        }

    def snapshot_message(self) -> dict[str, Any]:
        """The proactive-push wire message carrying this context's current
        facts -- distinct from a ``recording_reply`` (which also carries a
        ``request_id``/outcome correlated to one command). Sent by each
        viewer's own sender loop on initial connect and whenever
        ``revision`` has advanced since that viewer's own last send,
        including a change made by a DIFFERENT viewer's command."""
        return {"type": "recording_snapshot", "snapshot": self.snapshot()}

    def _elapsed_ms_now(self) -> int:
        """Elapsed ms since the current episode's own monotonic start, as of
        right now -- 0 if none was ever started (defensive; both call sites
        only reach this with an active episode)."""
        if self._episode_started_monotonic is None:
            return 0
        return int((time.monotonic() - self._episode_started_monotonic) * 1000)

    async def apply(self, command: RecordingCommand) -> tuple[str, str | None]:
        """Applies one command's real effect through the coordinator.

        Returns ``(outcome, reason)`` where ``outcome`` is ``"ok"`` or
        ``"rejected"`` and ``reason`` is a short human-readable string
        (always present when rejected, always ``None`` when accepted). A
        real rejection from the coordinator/service is translated into the
        same ordinary "rejected" outcome every other business rejection
        already uses here -- never a protocol error, never an unhandled
        exception.

        Checked first, before any other validation: a missing or wrong
        ``control_token`` is rejected before admission, for every one of
        the five actions reaching this point (no per-action allowlist to
        keep in sync -- every admitted command already mutates).

        This method itself does NOT update any projection state (no
        ``last_result``/``saved_count``/``batch_id``/elapsed-basis writes
        here) -- that happens via the coordinator's own callbacks (see
        ``__post_init__``), invoked from INSIDE the coordinator's own
        retained/shielded operation task. So even if THIS coroutine (the
        one a disconnecting requester's own task is awaiting) never
        resumes past its own ``await``, the projection update still
        happens, because it isn't this coroutine doing it.

        There is no frontend-visible "open session" command, but a
        committed batch/task is NOT the same thing as a live storage
        session: ``set_task`` and ``new_batch`` never require (or open)
        real storage -- only ``start_episode`` (Record) does, as the one
        place that actually needs current accepted sources and a
        committed instruction. See ``RecordingSessionCoordinator``'s own
        ``start_episode``/``set_task``/``new_batch`` docstrings."""
        if not _control_token_valid(command.control_token, self.control_token):
            return "rejected", "Enter the current recording control token"
        if command.payload.get("batch_id") != self.batch_id:
            return "rejected", "no matching batch"
        # Attribution, not authentication -- the Origin check in
        # preview_routes.py is what restricts which pages can reach this channel.
        # Every action reaching this point already mutates (see
        # _PAYLOAD_VALIDATORS/parse_recording_command), so this is
        # unconditional, not a per-action list to keep in sync.
        operator = _normalize_operator(command.operator_token)
        if operator is None:
            return "rejected", "Enter an operator name before using recording controls"
        try:
            if command.action == "set_task":
                # An ordinary rejected business outcome -- parse_recording_command
                # already confirmed this is a real string; empty content is
                # not a protocol violation.
                task = command.payload["task"]
                # Whitespace-only text is not a real instruction either.
                if task.strip() == "":
                    return "rejected", "task text cannot be empty"
                self.coordinator.set_task(task)
                self.revision += 1
                return "ok", None
            if command.action == "start_episode":
                if self.coordinator.snapshot().context.task_text.strip() == "":
                    return "rejected", "set a task before starting an episode"
                await self.coordinator.start_episode(operator)
                return "ok", None
            if command.action == "stop_episode":
                await self.coordinator.stop_episode(command.payload["episode_id"])
                return "ok", None
            if command.action == "discard_episode":
                await self.coordinator.discard_episode(command.payload["episode_id"])
                return "ok", None
            if command.action == "new_batch":
                # A genuinely active take must be explicitly stopped/
                # discarded first, never implicitly discarded by this
                # command -- but idle, recovery-required and failed are
                # all safe points to rotate from (enforced for real by
                # the coordinator itself).
                await self.coordinator.new_batch()
                return "ok", None
        except (CoordinatorBusyError, StateError, ConfigurationError) as exc:
            return "rejected", str(exc)
        except Exception as exc:  # noqa: BLE001 -- e.g. a full disk or an unwritable output directory
            # Anything else must still reach the viewer as a reply; letting it escape would end the receiver
            # and close the socket with no answer.
            _logger.exception("recording command %s failed", command.action)
            return "rejected", f"recording failed: {exc}"
        raise AssertionError(f"unhandled recording action: {command.action}")


async def handle_recording_command(command: RecordingCommand, context: RecordingContext) -> dict[str, Any]:
    """Builds the real reply for one parsed command: applies its effect to
    ``context`` (and, through it, the real ``RecordingService``) and
    returns a correlated reply carrying the real outcome, rejection reason
    (if any), and the context's current, honest snapshot."""
    outcome, reason = await context.apply(command)
    return {
        "type": "recording_reply",
        "request_id": command.request_id,
        "outcome": outcome,
        "reason": reason,
        "snapshot": context.snapshot(),
    }
