# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""HTTP surface for the end-of-day report: the day's takes, saving the report, and correcting a task.

The report is read from the recording service's manifests (``egocap.analytics.day_report``), so it shows what
became of every take. Everything here needs the run's control token: the report names operators and tasks, and
the correction rewrites recorded metadata.

A correction is confined to the recording output, refused for the session open for recording and while a take is
being written, and replaces each metadata file whole. It changes labels only; recorded video and tracking stay as
captured.
"""

from __future__ import annotations

import asyncio
import datetime
import json
import logging
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

from .recording_command import MAX_TASK_DETAILS_CHARS, control_token_valid
from .request_body import read_authorized_json

logger = logging.getLogger("isaaccapture.egocap.day_report_routes")

router = APIRouter()

_NO_STORE = {"Cache-Control": "no-store"}
_MAX_BODY_BYTES = 16 * 1024
#: The ``station`` value that asks for every station instead of the one in use.
ALL_STATIONS = "*"


class DayReportError(ValueError):
    """A refused report request or correction, with a public, path-free reason."""


def _reply(body: dict[str, Any], status: int = 200) -> JSONResponse:
    """A non-cacheable JSON reply: the day changes with every take."""
    return JSONResponse(body, status_code=status, headers=_NO_STORE)


def _state(request: Request) -> tuple[Any, Any, Any, Path] | None:
    """The recording context, station service, activity log and recording output, or None before startup."""
    state = request.app.state
    names = ("recording_context", "station_service", "station_activity", "recording_output_root")
    found = tuple(getattr(state, name, None) for name in names)
    return None if any(item is None for item in found) else found  # type: ignore[return-value]


def _parse_date(raw: Any, today: str) -> str:
    """A ``YYYY-MM-DD`` day, defaulting to today."""
    if raw is None or raw == "":
        return today
    try:
        return datetime.date.fromisoformat(raw).isoformat()
    except (TypeError, ValueError):
        raise DayReportError("date must be a day as YYYY-MM-DD") from None


def _active_episode_id(context: Any) -> str | None:
    """The take being recorded now, so the report does not call it interrupted."""
    service = context.coordinator.snapshot().service
    return service.episode.episode_id if service is not None and service.episode is not None else None


def _gather(state: tuple[Any, Any, Any, Path], date_raw: Any, station_raw: Any) -> dict[str, Any]:
    """The day's report with the station's activity; runs in a worker thread, since it walks the disk."""
    context, stations, activity, root = state
    today = datetime.date.fromtimestamp(time.time()).isoformat()
    date = _parse_date(date_raw, today)
    if station_raw is not None and not isinstance(station_raw, str):
        raise DayReportError("station must be a station name")
    every_station = station_raw == ALL_STATIONS
    station = None if every_station else (station_raw or stations.label())
    report = build_day_report(root, date, station=station, active_episode_id=_active_episode_id(context))
    return {
        "report": report,
        # Activity is kept per station and for today only; another day or every station has none to show.
        "activity": activity.report(station) if station is not None and date == today else None,
        "today": today,
        "station_in_use": stations.label(),
    }


@router.get("/api/day-report")
async def read_day_report(request: Request, date: str | None = None, station: str | None = None) -> JSONResponse:
    """A day's takes by session with totals per disposition, for the station in use unless ``station`` names
    another or ``*`` for all."""
    state = _state(request)
    if state is None:
        return _reply({"detail": "the report is not available yet"}, 503)
    if not control_token_valid(request.headers.get("x-control-token"), state[0].control_token):
        return _reply({"detail": "control token rejected"}, 403)
    try:
        return _reply(await asyncio.to_thread(_gather, state, date, station))
    except DayReportError as exc:
        return _reply({"detail": str(exc)}, 400)


def _save(state: tuple[Any, Any, Any, Path], body: dict[str, Any]) -> dict[str, Any]:
    """Write the report beside the activity log and refresh that log's recorded time; in a worker thread."""
    _context, _stations, activity, _root = state
    gathered = _gather(state, body.get("date"), body.get("station"))
    report = gathered["report"]
    station = report["station"]
    saved = {**gathered, "generated_at": datetime.datetime.now().isoformat(timespec="seconds")}
    name = activity.save_report(station if station is not None else "all stations", report["date"], saved)
    activity_log = None
    # The activity log is today's and one station's; the report gives it the recorded time it cannot measure.
    if gathered["activity"] is not None:
        total, kept = recorded_seconds(report)
        activity_log = activity.write_log(station, data_seconds=total, valid_seconds=kept)
    return {**gathered, "saved": {"report": name, "activity_log": activity_log}}


@router.post("/api/day-report/save")
async def save_day_report(request: Request) -> JSONResponse:
    """Save the day's report as JSON on this machine and bring the daily activity log up to date."""
    state = _state(request)
    if state is None:
        return _reply({"detail": "the report is not available yet"}, 503)
    body, refused = await read_authorized_json(request, state[0].control_token, _MAX_BODY_BYTES)
    if body is None:
        return refused  # type: ignore[return-value]
    try:
        return _reply(await asyncio.to_thread(_save, state, body))
    except DayReportError as exc:
        return _reply({"detail": str(exc)}, 400)
    except OSError:
        logger.exception("Could not save the day report")
        return _reply({"detail": "could not save the report"}, 503)


def _confined_session(root: Path, raw: Any) -> Path:
    """A session directory named relative to the recording output, refused unless it is really inside it."""
    if not isinstance(raw, str) or raw.strip() == "":
        raise DayReportError("session must be a path under the recording output")
    base = root.resolve()
    target = (base / raw).resolve()
    # Resolved first, so a traversal, an absolute path or a link out of the output are all caught here.
    if base not in target.parents:
        raise DayReportError("the session is outside the recording output")
    if any(part.startswith(".") for part in target.relative_to(base).parts):
        raise DayReportError("that directory is not a recorded session")
    if not target.is_dir():
        raise DayReportError("no such session")
    return target


def _owned_by(session_id: str, directory: Path) -> bool:
    """Whether a directory is the open session itself or a routed group stamped as that session's."""
    if directory.name == session_id and (directory / MANIFEST_DIR).is_dir():
        return True
    try:
        stamp = json.loads((directory / "zarr.json").read_text()).get("attributes", {}).get(ROUTE_ATTR)
    except (OSError, ValueError, AttributeError):
        return False
    return isinstance(stamp, str) and stamp.startswith(f"{session_id}|")


def _refuse_while_recording(context: Any, target: Path) -> None:
    """Refuse a correction while a take is being written, or to the session still open for recording."""
    snapshot = context.coordinator.snapshot()
    service = snapshot.service
    if snapshot.active_operation is not None or (service is not None and service.episode is not None):
        raise DayReportError("finish or discard the current take before correcting a session")
    if service is not None and service.session is not None and _owned_by(service.session.session_id, target):
        raise DayReportError("this session is still open for recording; start a new batch first")


def _text(body: dict[str, Any], key: str, *, required: bool = False) -> str:
    """A bounded text field of a correction."""
    value = body.get(key, "")
    if not isinstance(value, str) or len(value) > MAX_TASK_DETAILS_CHARS:
        raise DayReportError(f"{key} must be text of at most {MAX_TASK_DETAILS_CHARS} characters")
    if required and value.strip() == "":
        raise DayReportError(f"{key} cannot be empty")
    return value.strip()


def _correct(state: tuple[Any, Any, Any, Path], body: dict[str, Any]) -> dict[str, Any]:
    """Apply one task correction; runs in a worker thread."""
    context, _stations, _activity, root = state
    target = _confined_session(root, body.get("session"))
    _refuse_while_recording(context, target)
    task, task_name, studio = _text(body, "task", required=True), _text(body, "task_name"), _text(body, "studio")
    episodes = body.get("episodes")
    relocate = body.get("relocate", False)
    if not isinstance(episodes, str) or not isinstance(relocate, bool):
        raise DayReportError("episodes must be a range such as 3 or 2-7, and relocate true or false")
    # A relocation moves the session to <output>/<studio>/<name>; only a session already filed under a studio
    # folder keeps that destination inside the recording output.
    if relocate and target.parent.parent != root.resolve():
        raise DayReportError("only a session filed under a studio folder can be relocated")
    try:
        result = update_episode_task_description(
            target, episodes, task_description=task, task_name=task_name, studio=studio, relocate=relocate
        )
    except EpisodeRangeError as exc:
        raise DayReportError(f"episodes: {exc}") from None
    except ValueError:
        raise DayReportError("a relocation needs a studio") from None
    except OSError as exc:
        # The library's messages carry paths; say which rule refused it without them.
        fixed = "fixed by its episode manifests" in str(exc)
        logger.warning("Session correction refused: %s", exc)
        raise DayReportError(
            "this recording's location is fixed by its manifests; correct its labels without relocating it"
            if fixed
            else "the session could not be relocated: a session of that name is already under the studio"
        ) from None
    return {
        "session": result.zarr_root.relative_to(root.resolve()).as_posix(),
        "updated": result.updated_episodes,
        "missing": result.missing_episodes,
        "studio": result.studio_stored or None,
        "config_updated": result.config_yaml_updated,
    }


@router.post("/api/day-report/correct")
async def correct_session(request: Request) -> JSONResponse:
    """Correct the task of a range of a session's episodes, and optionally its studio label and location."""
    state = _state(request)
    if state is None:
        return _reply({"detail": "the report is not available yet"}, 503)
    body, refused = await read_authorized_json(request, state[0].control_token, _MAX_BODY_BYTES)
    if body is None:
        return refused  # type: ignore[return-value]
    try:
        return _reply(await asyncio.to_thread(_correct, state, body))
    except DayReportError as exc:
        return _reply({"detail": str(exc)}, 400)
