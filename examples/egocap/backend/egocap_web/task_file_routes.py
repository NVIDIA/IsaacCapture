# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""The task-catalog endpoints: the shipped catalogs plus the operator's imported ones.

Two roots feed one listing. ``app.state.task_file_root`` holds the catalogs shipped with the application
(read-only, see data/taskfiles/README.md); ``app.state.imported_task_file_root`` holds the CSV files an operator
imported from the Settings page, which can also be removed again. An imported catalog with a shipped catalog's
name shadows it, so an operator can replace a shipped list without touching the installation. Imports and
removals carry the run's control token, like recording commands. The filesystem work runs in a worker thread so
it never blocks the event loop.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from . import task_files
from .operator_settings import OperatorSettingsError
from .recording_command import control_token_valid
from .request_body import read_authorized_json

router = APIRouter()

_NO_STORE = {"Cache-Control": "no-store"}
# A bad name is a client error (400) and an absent catalog a 404. A catalog on the server that is oversized or
# malformed is a server-side problem (500, with the reason in ``detail``), and unreadable storage is 503.
_STATUS = {
    task_files.InvalidTaskFileName: 400,
    task_files.TaskFileNotFound: 404,
    task_files.TaskFileTooLarge: 500,
    task_files.MalformedTaskFile: 500,
    task_files.TaskFilesUnavailable: 503,
}
# The same file arriving from an operator is their mistake, not the server's: too large or malformed is a 400.
_IMPORT_STATUS = {**_STATUS, task_files.TaskFileTooLarge: 400, task_files.MalformedTaskFile: 400}


def _failure(exc: task_files.TaskFileError, status: dict[type, int] = _STATUS) -> JSONResponse:
    """The error response for a task-file error: its status, a concise reason, never a path."""
    return JSONResponse({"detail": exc.public_message}, status_code=status[type(exc)], headers=_NO_STORE)


def _detail(message: str, status: int) -> JSONResponse:
    """Expose only a deliberately selected operator-facing message."""
    return JSONResponse({"detail": message}, status_code=status, headers=_NO_STORE)


def _imported_root(request: Request) -> Path | None:
    """Find the configured writable catalog root, if imports are enabled."""
    return getattr(request.app.state, "imported_task_file_root", None)


def _listing(shipped_root: Path, imported_root: Path | None) -> dict[str, list[str]]:
    """Every catalog name once, plus which of them are imports; a not-yet-created imported root is empty."""
    shipped = task_files.list_task_files(shipped_root)
    imported = (
        task_files.list_task_files(imported_root) if imported_root is not None and imported_root.is_dir() else []
    )
    return {"files": sorted(set(shipped) | set(imported)), "imported": imported}


@router.get("/api/task-files")
async def list_files(request: Request) -> JSONResponse:
    """The sorted catalog file names, and which of them the operator imported."""
    try:
        listing = await asyncio.to_thread(_listing, request.app.state.task_file_root, _imported_root(request))
    except task_files.TaskFileError as exc:
        return _failure(exc)
    return JSONResponse(listing, headers=_NO_STORE)


@router.get("/api/task-files/{name}")
async def read_file(name: str, request: Request) -> JSONResponse:
    """One catalog's task definitions in file order, from the imported root first."""
    try:
        contents = await asyncio.to_thread(
            task_files.read_available_task_file, request.app.state.task_file_root, _imported_root(request), name
        )
    except task_files.TaskFileError as exc:
        return _failure(exc)
    return JSONResponse(contents.to_json(), headers=_NO_STORE)


async def _authorised_body(request: Request) -> tuple[dict[str, Any] | None, JSONResponse | None]:
    """The JSON body once its control token is accepted, or the response that refuses it."""
    recording_context = getattr(request.app.state, "recording_context", None)
    if recording_context is None:
        return None, _detail("task file imports are not available yet", 503)
    # JSON escaping can expand a valid CSV byte into six bytes; parsing still enforces the CSV limit.
    return await read_authorized_json(
        request, recording_context.control_token, 6 * task_files.MAX_FILE_BYTES + 1024
    )


@router.post("/api/task-files/import")
async def import_file(request: Request) -> JSONResponse:
    """Import browser-uploaded ``name`` and ``content`` after validating the catalog."""
    body, refused = await _authorised_body(request)
    if body is None:
        return refused  # type: ignore[return-value]
    imported_root = _imported_root(request)
    if imported_root is None:
        return _detail("task file imports are not available yet", 503)
    name, content = body.get("name"), body.get("content")
    if "path" in body:
        return _detail("server path imports are disabled; choose a CSV file to upload", 400)
    if not isinstance(name, str) or not isinstance(content, str):
        return _detail("give name and content", 400)
    raw = content.encode("utf-8")
    store = getattr(request.app.state, "operator_settings_store", None)
    if store is None:
        return _detail("operator settings are not available yet", 503)

    def save_import() -> dict[str, Any] | None:
        """Prevent an import from changing a selected catalog underneath its task selection."""
        with store.lock:
            if store.load().task_file == name:
                return None
            contents = task_files.import_task_file(imported_root, name, raw)
            return {"file": contents.to_json(), **_listing(request.app.state.task_file_root, imported_root)}

    try:
        result = await asyncio.to_thread(save_import)
    except OperatorSettingsError:
        return _detail("repair operator settings before importing catalogs", 409)
    except OSError:
        return _detail("operator settings storage is unavailable", 503)
    except task_files.TaskFileError as exc:
        return _failure(exc, _IMPORT_STATUS)
    if result is None:
        return _detail("choose another catalog or free text before replacing the selected catalog", 409)
    return JSONResponse(result, headers=_NO_STORE)


@router.delete("/api/task-files/{name}")
async def remove_file(name: str, request: Request) -> JSONResponse:
    """Remove an unselected import, authenticating with the X-Control-Token header."""
    context = getattr(request.app.state, "recording_context", None)
    if context is None:
        return _detail("task file imports are not available yet", 503)
    if not control_token_valid(request.headers.get("x-control-token"), context.control_token):
        return _detail("control token rejected", 403)
    imported_root = _imported_root(request)
    if imported_root is None or not imported_root.is_dir():
        return _detail(f"no imported task file named {name}", 404)
    store = getattr(request.app.state, "operator_settings_store", None)
    if store is None:
        return _detail("operator settings are not available yet", 503)

    def remove() -> dict[str, list[str]] | None:
        """Hold the settings lock so selection cannot race a catalog removal."""
        with store.lock:
            if store.load().task_file == name:
                return None
            task_files.remove_task_file(imported_root, name)
            return _listing(request.app.state.task_file_root, imported_root)

    try:
        listing = await asyncio.to_thread(remove)
    except OperatorSettingsError:
        return _detail("repair operator settings before removing catalogs", 409)
    except OSError:
        return _detail("operator settings storage is unavailable", 503)
    except task_files.TaskFileError as exc:
        return _failure(exc)
    if listing is None:
        return _detail("choose another catalog or free text before removing the selected catalog", 409)
    return JSONResponse(listing, headers=_NO_STORE)
