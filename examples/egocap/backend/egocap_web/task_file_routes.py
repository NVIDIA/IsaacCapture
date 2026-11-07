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
from .recording_command import _control_token_valid

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
    return JSONResponse({"detail": str(exc)}, status_code=status[type(exc)], headers=_NO_STORE)


def _detail(message: str, status: int) -> JSONResponse:
    return JSONResponse({"detail": message}, status_code=status, headers=_NO_STORE)


def _imported_root(request: Request) -> Path | None:
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
    imported_root = _imported_root(request)

    def read() -> task_files.TaskFileContents:
        if (
            imported_root is not None
            and imported_root.is_dir()
            and name in task_files.list_task_files(imported_root)
        ):
            return task_files.read_task_file(imported_root, name)
        return task_files.read_task_file(request.app.state.task_file_root, name)

    try:
        contents = await asyncio.to_thread(read)
    except task_files.TaskFileError as exc:
        return _failure(exc)
    return JSONResponse(contents.to_json(), headers=_NO_STORE)


async def _authorised_body(request: Request) -> tuple[dict[str, Any] | None, JSONResponse | None]:
    """The JSON body once its control token is accepted, or the response that refuses it."""
    recording_context = getattr(request.app.state, "recording_context", None)
    if recording_context is None:
        return None, _detail("task file imports are not available yet", 503)
    try:
        body = await request.json()
    except ValueError:
        return None, _detail("body must be JSON", 400)
    if not isinstance(body, dict):
        return None, _detail("body must be an object", 400)
    if not _control_token_valid(body.get("control_token"), recording_context.control_token):
        return None, _detail("control token rejected", 403)
    return body, None


def _read_local_file(path_text: str) -> tuple[str, bytes]:
    """A CSV the operator named by path on the backend's machine: its file name and bytes (bounded)."""
    path = Path(path_text).expanduser()
    with path.open("rb") as handle:
        return path.name, handle.read(task_files.MAX_FILE_BYTES + 1)


@router.post("/api/task-files/import")
async def import_file(request: Request) -> JSONResponse:
    """Import a catalog from ``path`` (a file on the backend's machine, for example in the operator's Downloads)
    or from ``name`` + ``content`` (the file's text, uploaded by the browser). The file must parse as a catalog
    before it is stored; the response carries its tasks and the new listing."""
    body, refused = await _authorised_body(request)
    if body is None:
        return refused  # type: ignore[return-value]
    imported_root = _imported_root(request)
    if imported_root is None:
        return _detail("task file imports are not available yet", 503)
    path_text, name, content = body.get("path"), body.get("name"), body.get("content")
    try:
        if isinstance(path_text, str) and path_text.strip():
            try:
                name, raw = await asyncio.to_thread(_read_local_file, path_text.strip())
            except FileNotFoundError:
                return _detail("no file at that path on the backend's machine", 404)
            except (IsADirectoryError, PermissionError, OSError) as exc:
                return _detail(f"the file could not be read: {exc.__class__.__name__}", 400)
        elif isinstance(name, str) and isinstance(content, str):
            raw = content.encode("utf-8")
        else:
            return _detail("give either path, or name and content", 400)
        contents = await asyncio.to_thread(task_files.import_task_file, imported_root, name, raw)
        listing = await asyncio.to_thread(_listing, request.app.state.task_file_root, imported_root)
    except task_files.TaskFileError as exc:
        return _failure(exc, _IMPORT_STATUS)
    return JSONResponse({"file": contents.to_json(), **listing}, headers=_NO_STORE)


@router.delete("/api/task-files/{name}")
async def remove_file(name: str, request: Request) -> JSONResponse:
    """Remove an imported catalog (never a shipped one); the body carries ``control_token``."""
    body, refused = await _authorised_body(request)
    if body is None:
        return refused  # type: ignore[return-value]
    imported_root = _imported_root(request)
    if imported_root is None or not imported_root.is_dir():
        return _detail(f"no imported task file named {name}", 404)
    try:
        await asyncio.to_thread(task_files.remove_task_file, imported_root, name)
        listing = await asyncio.to_thread(_listing, request.app.state.task_file_root, imported_root)
    except task_files.TaskFileError as exc:
        return _failure(exc)
    return JSONResponse(listing, headers=_NO_STORE)
