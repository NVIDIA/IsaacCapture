# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""HTTP surface for the end-of-day report: today's takes for the station in use, saving the report, and
correcting a task.

The report is read from the recording service's manifests (``egocap.analytics.day_report``), so it shows what
became of every take, with the station's time from the uptime log. Everything here needs the run's control
token: the report names operators and tasks, and the correction rewrites recorded metadata.

A correction is confined to a recorded session inside the recording output, refused for the session open for
recording, while a take is being written and while that session's take is being time-aligned, and replaces
each metadata file whole. It holds the session's lock (session_locks.py) from its first write to its last, so
neither the aligner nor a second correction can come between them. It changes labels only; recorded video and
tracking stay as captured, and nothing is ever moved.
"""

from __future__ import annotations

import asyncio
import datetime
import json
import logging
import re
import time
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from egocap.analytics.day_report import build_day_report, recorded_seconds
from egocap.data_processing.update_episode_task_description import (
    EpisodeRangeError,
    update_episode_task_description,
)
from egocap.recording.episode_layout import MANIFEST_DIR, ROUTE_ATTR

from .recording_command import MAX_TASK_DETAILS_CHARS, control_token_valid, interrupted_take
from .request_body import read_authorized_json
from .session_locks import ALIGNER, CORRECTION, SessionLocks

logger = logging.getLogger("isaaccapture.egocap.day_report_routes")

router = APIRouter()

_NO_STORE = {"Cache-Control": "no-store"}
_MAX_BODY_BYTES = 16 * 1024
#: Longer than any session path under the output; a longer one is refused before the filesystem is asked.
_MAX_SESSION_CHARS = 512
#: Room for a range of two long numbers; a longer value is not a take or a range.
_MAX_EPISODES_CHARS = 32
_NOT_A_SESSION = "that directory is not a recorded session"
_EPISODE_DIR = re.compile(r"episode_\d+")


class DayReportError(ValueError):
    """A refused correction, with a public, path-free reason."""


def _reply(body: dict[str, Any], status: int = 200) -> JSONResponse:
    """A non-cacheable JSON reply: the day changes with every take."""
    return JSONResponse(body, status_code=status, headers=_NO_STORE)


def _state(request: Request) -> tuple[Any, Any, Any, Path] | None:
    """The recording context, station service, uptime log and recording output, or None before startup."""
    state = request.app.state
    names = ("recording_context", "station_service", "station_uptime", "recording_output_root")
    found = tuple(getattr(state, name, None) for name in names)
    return None if any(item is None for item in found) else found  # type: ignore[return-value]


def _active_episode_id(context: Any) -> str | None:
    """The take being recorded now, so the report does not call it interrupted. A take that was interrupted
    is not one: it is still the service's episode, and the report shows it as pending."""
    service = context.coordinator.snapshot().service
    if service is None or service.episode is None or interrupted_take(service):
        return None
    return service.episode.episode_id


def _gather(state: tuple[Any, Any, Any, Path]) -> dict[str, Any]:
    """Today's report for the station in use, with its time; runs in a worker thread, since it walks the disk."""
    context, stations, uptime, root = state
    today = datetime.date.fromtimestamp(time.time()).isoformat()
    station = stations.label()
    # A station renamed today recorded its earlier takes under the old name; they are still its day.
    report = build_day_report(
        root, today, stations=uptime.names(station), active_episode_id=_active_episode_id(context)
    )
    return {
        "report": {**report, "station": station},
        "uptime": uptime.report(station),
        "today": today,
        "station_in_use": station,
    }


@router.get("/api/day-report")
async def read_day_report(request: Request) -> JSONResponse:
    """Today's takes for the station in use, by session, with totals per disposition and the station's time."""
    state = _state(request)
    if state is None:
        return _reply({"detail": "the report is not available yet"}, 503)
    if not control_token_valid(request.headers.get("x-control-token"), state[0].control_token):
        return _reply({"detail": "control token rejected"}, 403)
    return _reply(await asyncio.to_thread(_gather, state))


def _save(state: tuple[Any, Any, Any, Path]) -> dict[str, Any]:
    """Write the report beside the daily log and refresh that log's recorded time; in a worker thread."""
    _context, _stations, uptime, _root = state
    gathered = _gather(state)
    station = gathered["station_in_use"]
    saved = {**gathered, "generated_at": datetime.datetime.now().isoformat(timespec="seconds")}
    name = uptime.save_report(station, saved)
    # The uptime log cannot measure what was recorded; the report gives the daily log that figure.
    total, kept = recorded_seconds(gathered["report"])
    activity_log = uptime.write_log(station, data_seconds=total, valid_seconds=kept)
    return {**gathered, "saved": {"report": name, "activity_log": activity_log}}


@router.post("/api/day-report/save")
async def save_day_report(request: Request) -> JSONResponse:
    """Save today's report as JSON on this machine and bring the daily log up to date."""
    state = _state(request)
    if state is None:
        return _reply({"detail": "the report is not available yet"}, 503)
    body, refused = await read_authorized_json(request, state[0].control_token, _MAX_BODY_BYTES)
    if body is None:
        return refused  # type: ignore[return-value]
    try:
        return _reply(await asyncio.to_thread(_save, state))
    except OSError:
        logger.exception("Could not save the day report")
        return _reply({"detail": "could not save the report"}, 503)


def _route_stamp(directory: Path) -> str | None:
    """The stamp the recording service leaves on a routed group, naming the session that fills it, or None
    when the directory carries none it can read."""
    try:
        stamp = json.loads((directory / "zarr.json").read_text()).get("attributes", {}).get(ROUTE_ATTR)
    except (OSError, ValueError, AttributeError):
        return None
    return stamp if isinstance(stamp, str) else None


def _confined_session(root: Path, raw: Any) -> Path:
    """A recorded session named relative to the recording output, refused unless it really is one inside it.

    Only two shapes hold recorded episodes: a recording-service session directly under the output, and a
    studio's group one level below. Anything else would have metadata written into a place that has none.
    """
    if not isinstance(raw, str) or raw.strip() == "":
        raise DayReportError("session must be a path under the recording output")
    # A NUL or an over-long name makes the path calls raise instead of answering.
    if "\x00" in raw or len(raw) > _MAX_SESSION_CHARS:
        raise DayReportError(_NOT_A_SESSION)
    try:
        return _recorded_session(root.resolve(), raw)
    except OSError:
        # A name the filesystem will not even look up is not a session; the cause is not the caller's to read.
        raise DayReportError(_NOT_A_SESSION) from None


def _recorded_session(base: Path, raw: str) -> Path:
    """The recorded session ``raw`` names under the resolved output ``base``; refused otherwise."""
    target = (base / raw).resolve()
    # Resolved first, so a traversal, an absolute path or a link out of the output are all caught here.
    if base not in target.parents:
        raise DayReportError("the session is outside the recording output")
    parts = target.relative_to(base).parts
    if any(part.startswith(".") for part in parts) or not target.is_dir():
        raise DayReportError(_NOT_A_SESSION)
    # A time-aligned copy and a single episode are corrected with the session they belong to, never alone.
    if any(part.endswith("_aligned") for part in parts) or _EPISODE_DIR.fullmatch(target.name):
        raise DayReportError(_NOT_A_SESSION)
    session = len(parts) == 1 and (target / MANIFEST_DIR).is_dir()
    # A routed group is known by the recording service's stamp, not by what the directory happens to lack.
    group = len(parts) == 2 and _route_stamp(target) is not None
    if not (session or group):
        raise DayReportError(_NOT_A_SESSION)
    # The time-aligned copy beside a session is rewritten with it, so it must be inside the output too.
    aligned = target.with_name(f"{target.name}_aligned")
    if aligned.is_symlink() or (aligned.exists() and base not in aligned.resolve().parents):
        raise DayReportError("the session's aligned copy is outside the recording output")
    return target


def _owned_by(session: Any, directory: Path) -> bool:
    """Whether a directory is the open session itself or a routed group stamped as that session's."""
    try:
        if directory.samefile(session.session_path):
            return True
    except OSError:
        # A session path that cannot be compared is not this directory; the route stamp below still decides.
        pass
    stamp = _route_stamp(directory)
    return stamp is not None and stamp.startswith(f"{session.session_id}|")


def _refuse_while_recording(context: Any, target: Path) -> None:
    """Refuse a correction while a take is being written, or to the session still open for recording."""
    snapshot = context.coordinator.snapshot()
    service = snapshot.service
    # An interrupted take is not being written; its session is still open, which the check below refuses.
    writing = service is not None and service.episode is not None and not interrupted_take(service)
    if snapshot.active_operation is not None or writing:
        raise DayReportError("finish or discard the current take before correcting a session")
    if service is not None and service.session is not None and _owned_by(service.session, target):
        raise DayReportError("this session is still open for recording; start a new batch first")


#: Why a correction is refused while the session's labels are held, by who holds them.
_HELD = {
    # The aligner copies a take's labels before it publishes the copy; a correction made in between would
    # reach the recording and miss the copy. An alignment can take minutes, so the correction does not wait.
    ALIGNER: "time alignment of this take is still running; try again in a moment",
    # Two corrections interleaved would leave the episodes, config.yaml and the aligned copy disagreeing.
    CORRECTION: "another correction of this session is still being written; try again in a moment",
}


def _correct(state: tuple[Any, Any, Any, Path], body: dict[str, Any], locks: SessionLocks) -> dict[str, Any]:
    """Apply one task correction; runs in a worker thread. ``locks`` is shared with the background aligner."""
    context, _stations, _uptime, root = state
    target = _confined_session(root, body.get("session"))
    _refuse_while_recording(context, target)
    task, episodes = body.get("task"), body.get("episodes")
    if not isinstance(task, str) or task.strip() == "" or len(task) > MAX_TASK_DETAILS_CHARS:
        raise DayReportError(f"task must be text of 1 to {MAX_TASK_DETAILS_CHARS} characters")
    if not isinstance(episodes, str) or len(episodes) > _MAX_EPISODES_CHARS:
        raise DayReportError("episodes must be a take or a range, such as 3 or 2-7")
    # Held across every file the correction writes; the aligner takes the same lock before it copies labels.
    holder = locks.acquire(target, CORRECTION)
    if holder is not None:
        raise DayReportError(_HELD[holder])
    try:
        result = update_episode_task_description(target, episodes, task_description=task.strip())
    except EpisodeRangeError as exc:
        raise DayReportError(f"episodes: {exc}") from None
    except (OSError, ValueError):
        # The library's messages carry paths; the cause goes to the log and the operator gets the outcome.
        logger.exception("A task correction could not be written")
        raise DayReportError("the session's metadata could not be updated") from None
    finally:
        locks.release(target)
    return {
        "session": result.zarr_root.relative_to(root.resolve()).as_posix(),
        "updated": result.updated_episodes,
        "missing": result.missing_episodes,
        "config_updated": result.config_yaml_updated,
    }


@router.post("/api/day-report/correct")
async def correct_session(request: Request) -> JSONResponse:
    """Correct the task of one take or a range of a session's takes."""
    state = _state(request)
    locks = getattr(request.app.state, "session_locks", None)
    if state is None or locks is None:
        return _reply({"detail": "the report is not available yet"}, 503)
    body, refused = await read_authorized_json(request, state[0].control_token, _MAX_BODY_BYTES)
    if body is None:
        return refused  # type: ignore[return-value]
    try:
        return _reply(await asyncio.to_thread(_correct, state, body, locks))
    except DayReportError as exc:
        return _reply({"detail": str(exc)}, 400)
