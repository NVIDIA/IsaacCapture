# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed events and commands of the calibration worker's wire protocol.

The contract is `src/plugins/manus/calibration/README.md`. Parsing is strict: any line that is not
exactly a documented message raises :class:`ProtocolError`, so a misbehaving worker is a visible
failure rather than a skipped line.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any

PROTOCOL_VERSION = 1
MAX_LINE_BYTES = 65536

_ATTEMPT_RE = re.compile(r"[A-Za-z0-9_-]{1,64}")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_FILE_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")


class ProtocolError(Exception):
    """A worker line (or a command we were asked to build) violates the contract."""


class Side(str, Enum):
    LEFT = "left"
    RIGHT = "right"


class Phase(str, Enum):
    DISCOVERING = "discovering"
    AWAITING_STEP = "awaiting_step"
    CAPTURING_STEP = "capturing_step"
    SAVING = "saving"
    STOPPING = "stopping"


class StepMode(str, Enum):
    HOLD = "hold"
    MOTION = "motion"


class Outcome(str, Enum):
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


class RejectReason(str, Enum):
    INVALID = "invalid"
    UNKNOWN_ATTEMPT = "unknown_attempt"
    STALE_TOKEN = "stale_token"
    UNEXPECTED = "unexpected"
    BUSY = "busy"


_ERROR_CODES = frozenset(
    {
        "sdk_init_failed",
        "no_gloves",
        "glove_missing",
        "glove_replaced",
        "sdk_error",
        "step_failed",
        "export_failed",
        "protocol_error",
        "internal",
    }
)


@dataclass(frozen=True)
class Glove:
    side: Side
    glove_id: int


@dataclass(frozen=True)
class ExportedFile:
    """One MCAL file as the worker reports it: its basename, size and SHA-256."""

    side: Side
    file: str
    bytes: int
    sha256: str


@dataclass(frozen=True)
class WorkerError:
    code: str
    message: str
    side: Side | None = None
    step: int | None = None


@dataclass(frozen=True)
class Ready:
    worker_version: str
    sdk_version: str | None


@dataclass(frozen=True)
class Status:
    attempt: str
    phase: Phase
    gloves: tuple[Glove, ...]


@dataclass(frozen=True)
class Prompt:
    attempt: str
    token: str
    side: Side
    glove_id: int
    index: int
    count: int
    title: str
    instruction: str
    mode: StepMode
    duration_s: float
    # A continuous SDK step: operator-paced like a hold, with the SDK's estimate of its length as guidance.
    continuous: bool = False
    estimated_s: float = 0.0


@dataclass(frozen=True)
class StepAccepted:
    attempt: str
    token: str


@dataclass(frozen=True)
class StepProgress:
    attempt: str
    token: str
    elapsed_s: float


@dataclass(frozen=True)
class Rejected:
    attempt: str | None
    command: str
    reason: RejectReason
    message: str
    token: str | None = None


@dataclass(frozen=True)
class Exported:
    attempt: str
    side: Side
    glove_id: int
    file: str
    bytes: int
    sha256: str


@dataclass(frozen=True)
class Result:
    attempt: str | None
    outcome: Outcome
    files: tuple[ExportedFile, ...] = ()
    error: WorkerError | None = None
    # False when the worker could not stop the SDK calibration or shut the SDK down, so the gloves may still
    # be in use whatever the outcome says.
    released: bool = True


WorkerEvent = (
    Ready | Status | Prompt | StepAccepted | StepProgress | Rejected | Exported | Result
)


def _fail(context: str, problem: str) -> ProtocolError:
    return ProtocolError(f"{context}: {problem}")


def _keys(
    data: dict[str, Any], required: set[str], optional: set[str], context: str
) -> None:
    present = set(data)
    missing = required - present
    unknown = present - required - optional
    if missing:
        raise _fail(context, f"missing key(s) {sorted(missing)}")
    if unknown:
        raise _fail(context, f"unknown key(s) {sorted(unknown)}")


def _str(
    data: dict[str, Any], key: str, context: str, *, nonempty: bool = False
) -> str:
    value = data[key]
    if not isinstance(value, str) or (nonempty and not value):
        raise _fail(
            context, f"{key} must be {'a nonempty ' if nonempty else 'a '}string"
        )
    return value


def _opt_str(data: dict[str, Any], key: str, context: str) -> str | None:
    return None if data[key] is None else _str(data, key, context)


def _uint(data: dict[str, Any], key: str, context: str) -> int:
    value = data[key]
    # bool is an int subclass; a JSON boolean must not pass as a count.
    if type(value) is not int or value < 0:
        raise _fail(context, f"{key} must be a nonnegative integer")
    return value


def _seconds(data: dict[str, Any], key: str, context: str) -> float:
    value = data[key]
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        raise _fail(context, f"{key} must be a finite nonnegative number")
    return float(value)


def _enum(enum_cls: type[Enum], value: Any, context: str) -> Any:
    try:
        return enum_cls(value)
    except ValueError:
        raise _fail(
            context, f"{value!r} is not one of {[m.value for m in enum_cls]}"
        ) from None


def _attempt(
    data: dict[str, Any], context: str, *, nullable: bool = False
) -> str | None:
    value = data["attempt"]
    if value is None and nullable:
        return None
    if not isinstance(value, str) or not _ATTEMPT_RE.fullmatch(value):
        raise _fail(context, "attempt must be 1-64 characters of [A-Za-z0-9_-]")
    return value


def _file_name(value: Any, context: str) -> str:
    # Basename only: the supervisor joins this to its own output directory.
    if not isinstance(value, str) or not _FILE_RE.fullmatch(value) or ".." in value:
        raise _fail(context, f"file must be a plain basename, got {value!r}")
    return value


def _sha(value: Any, context: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise _fail(context, "sha256 must be 64 lowercase hex characters")
    return value


def _gloves(value: Any, context: str) -> tuple[Glove, ...]:
    if not isinstance(value, list):
        raise _fail(context, "gloves must be an array")
    gloves = []
    for item in value:
        if not isinstance(item, dict):
            raise _fail(context, "each glove must be an object")
        _keys(item, {"side", "glove_id"}, set(), f"{context}.gloves[]")
        gloves.append(
            Glove(
                _enum(Side, item["side"], f"{context}.gloves[].side"),
                _uint(item, "glove_id", f"{context}.gloves[]"),
            )
        )
    sides = [g.side for g in gloves]
    if len(sides) != len(set(sides)):
        raise _fail(context, "gloves lists a side twice")
    return tuple(gloves)


def _exported_files(value: Any, context: str) -> tuple[ExportedFile, ...]:
    if not isinstance(value, list):
        raise _fail(context, "files must be an array")
    files = []
    for item in value:
        if not isinstance(item, dict):
            raise _fail(context, "each file must be an object")
        _keys(item, {"side", "file", "bytes", "sha256"}, set(), f"{context}.files[]")
        files.append(
            ExportedFile(
                side=_enum(Side, item["side"], f"{context}.files[].side"),
                file=_file_name(item["file"], f"{context}.files[]"),
                bytes=_uint(item, "bytes", f"{context}.files[]"),
                sha256=_sha(item["sha256"], f"{context}.files[]"),
            )
        )
    return tuple(files)


def _error(value: Any, context: str) -> WorkerError:
    if not isinstance(value, dict):
        raise _fail(context, "error must be an object")
    _keys(value, {"code", "message"}, {"side", "step"}, f"{context}.error")
    code = value["code"]
    # isinstance first: an unhashable JSON value (a list or object) would raise TypeError in the lookup.
    if not isinstance(code, str) or code not in _ERROR_CODES:
        raise _fail(context, f"unknown error code {code!r}")
    side = value.get("side")
    step = value.get("step")
    if step is not None and (type(step) is not int or step < 0):
        raise _fail(context, "error.step must be a nonnegative integer")
    return WorkerError(
        code=code,
        message=_str(value, "message", f"{context}.error"),
        side=None if side is None else _enum(Side, side, f"{context}.error.side"),
        step=step,
    )


def _parse_ready(d: dict[str, Any]) -> Ready:
    _keys(d, {"v", "type", "worker_version", "sdk_version"}, set(), "ready")
    return Ready(
        _str(d, "worker_version", "ready"), _opt_str(d, "sdk_version", "ready")
    )


def _parse_status(d: dict[str, Any]) -> Status:
    _keys(d, {"v", "type", "attempt", "phase", "gloves"}, set(), "status")
    return Status(
        _attempt(d, "status"),
        _enum(Phase, d["phase"], "status.phase"),
        _gloves(d["gloves"], "status"),
    )


def _parse_prompt(d: dict[str, Any]) -> Prompt:
    keys = {
        "v",
        "type",
        "attempt",
        "token",
        "side",
        "glove_id",
        "index",
        "count",
        "title",
        "instruction",
        "mode",
        "duration_s",
        "continuous",
        "estimated_s",
    }
    _keys(d, keys, set(), "prompt")
    count = _uint(d, "count", "prompt")
    index = _uint(d, "index", "prompt")
    if count == 0 or index >= count:
        raise _fail("prompt", "index must be below a positive count")
    mode = _enum(StepMode, d["mode"], "prompt.mode")
    duration = _seconds(d, "duration_s", "prompt")
    # Hold steps carry no sampling window; motion steps must have one.
    if (mode is StepMode.HOLD) != (duration == 0):
        raise _fail("prompt", "duration_s must be 0 for hold and positive for motion")
    continuous = d["continuous"]
    if type(continuous) is not bool:
        raise _fail("prompt", "continuous must be a boolean")
    estimated = _seconds(d, "estimated_s", "prompt")
    # Only a hold-paced continuous step carries an estimate; a timed motion already has its window.
    if continuous and (mode is not StepMode.HOLD or estimated == 0):
        raise _fail("prompt", "a continuous step is a hold with a positive estimated_s")
    if not continuous and estimated != 0:
        raise _fail("prompt", "estimated_s must be 0 unless the step is continuous")
    return Prompt(
        attempt=_attempt(d, "prompt"),
        token=_str(d, "token", "prompt", nonempty=True),
        side=_enum(Side, d["side"], "prompt.side"),
        glove_id=_uint(d, "glove_id", "prompt"),
        index=index,
        count=count,
        title=_str(d, "title", "prompt"),
        # The worker forwards the SDK description verbatim, so it may be empty.
        instruction=_str(d, "instruction", "prompt"),
        mode=mode,
        duration_s=duration,
        continuous=continuous,
        estimated_s=estimated,
    )


def _parse_step_accepted(d: dict[str, Any]) -> StepAccepted:
    _keys(d, {"v", "type", "attempt", "token"}, set(), "step_accepted")
    return StepAccepted(
        _attempt(d, "step_accepted"), _str(d, "token", "step_accepted", nonempty=True)
    )


def _parse_step_progress(d: dict[str, Any]) -> StepProgress:
    _keys(d, {"v", "type", "attempt", "token", "elapsed_s"}, set(), "step_progress")
    return StepProgress(
        _attempt(d, "step_progress"),
        _str(d, "token", "step_progress", nonempty=True),
        _seconds(d, "elapsed_s", "step_progress"),
    )


def _parse_rejected(d: dict[str, Any]) -> Rejected:
    _keys(
        d,
        {"v", "type", "attempt", "command", "reason", "message"},
        {"token"},
        "rejected",
    )
    token = d.get("token")
    if token is not None and (not isinstance(token, str) or not token):
        raise _fail("rejected", "token must be a nonempty string")
    return Rejected(
        attempt=_attempt(d, "rejected", nullable=True),
        command=_str(d, "command", "rejected", nonempty=True),
        reason=_enum(RejectReason, d["reason"], "rejected.reason"),
        message=_str(d, "message", "rejected"),
        token=token,
    )


def _parse_exported(d: dict[str, Any]) -> Exported:
    _keys(
        d,
        {"v", "type", "attempt", "side", "glove_id", "file", "bytes", "sha256"},
        set(),
        "exported",
    )
    return Exported(
        attempt=_attempt(d, "exported"),
        side=_enum(Side, d["side"], "exported.side"),
        glove_id=_uint(d, "glove_id", "exported"),
        file=_file_name(d["file"], "exported"),
        bytes=_uint(d, "bytes", "exported"),
        sha256=_sha(d["sha256"], "exported"),
    )


def _parse_result(d: dict[str, Any]) -> Result:
    _keys(
        d, {"v", "type", "attempt", "outcome", "released"}, {"files", "error"}, "result"
    )
    outcome = _enum(Outcome, d["outcome"], "result.outcome")
    if type(d["released"]) is not bool:
        raise _fail("result", "released must be a boolean")
    has_files = "files" in d
    has_error = "error" in d
    # `files` belongs to completed results only and `error` to failed ones.
    if has_files != (outcome is Outcome.COMPLETED):
        raise _fail("result", "files must be present exactly for a completed result")
    if has_error != (outcome is Outcome.FAILED):
        raise _fail("result", "error must be present exactly for a failed result")
    return Result(
        attempt=_attempt(d, "result", nullable=True),
        outcome=outcome,
        files=_exported_files(d["files"], "result") if has_files else (),
        error=_error(d["error"], "result") if has_error else None,
        released=d["released"],
    )


_PARSERS = {
    "ready": _parse_ready,
    "status": _parse_status,
    "prompt": _parse_prompt,
    "step_accepted": _parse_step_accepted,
    "step_progress": _parse_step_progress,
    "rejected": _parse_rejected,
    "exported": _parse_exported,
    "result": _parse_result,
}


def parse_event(line: bytes | str) -> WorkerEvent:
    """Parse one worker line into its typed event, or raise :class:`ProtocolError`."""
    if isinstance(line, bytes):
        if len(line) > MAX_LINE_BYTES:
            raise ProtocolError(f"line exceeds {MAX_LINE_BYTES} bytes")
        try:
            line = line.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ProtocolError(f"line is not UTF-8: {exc}") from None
    try:
        data = json.loads(line)
    except ValueError as exc:
        raise ProtocolError(f"line is not JSON: {exc}") from None
    if not isinstance(data, dict):
        raise ProtocolError("line must be a JSON object")
    if type(data.get("v")) is not int or data["v"] != PROTOCOL_VERSION:
        raise ProtocolError(f"expected protocol version {PROTOCOL_VERSION}")
    message_type = data.get("type")
    # Checked before the lookup: an unhashable JSON value would raise TypeError instead of ProtocolError.
    parser = _PARSERS.get(message_type) if isinstance(message_type, str) else None
    if parser is None:
        raise ProtocolError(f"unknown message type {message_type!r}")
    return parser(data)


def encode_start(attempt: str, sides: tuple[Side, ...], out_dir: str) -> bytes:
    """Build the `start` command line."""
    if not _ATTEMPT_RE.fullmatch(attempt):
        raise ProtocolError("attempt must be 1-64 characters of [A-Za-z0-9_-]")
    if not sides or len(set(sides)) != len(sides):
        raise ProtocolError("sides must be a nonempty set of unique sides")
    # The worker processes left before right regardless of request order.
    ordered = [s.value for s in Side if s in sides]
    return _line(
        {
            "v": PROTOCOL_VERSION,
            "type": "start",
            "attempt": attempt,
            "sides": ordered,
            "out_dir": out_dir,
        }
    )


def encode_confirm(attempt: str, token: str) -> bytes:
    return _line(
        {"v": PROTOCOL_VERSION, "type": "confirm", "attempt": attempt, "token": token}
    )


def encode_cancel(attempt: str) -> bytes:
    return _line({"v": PROTOCOL_VERSION, "type": "cancel", "attempt": attempt})


def _line(message: dict[str, Any]) -> bytes:
    return (json.dumps(message, separators=(",", ":")) + "\n").encode("utf-8")
