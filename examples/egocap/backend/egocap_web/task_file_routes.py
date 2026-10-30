"""GET /api/task-files and GET /api/task-files/{name}: the read-only catalog endpoints.

A thin HTTP adapter over :mod:`task_files`; the filesystem work runs in a worker thread so it never blocks the
event loop. The catalog root is ``app.state.task_file_root``, set once by the application lifespan.
"""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from . import task_files

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


def _failure(exc: task_files.TaskFileError) -> JSONResponse:
    """The error response for a task-file error: its status, a concise reason, never a path."""
    return JSONResponse({"detail": str(exc)}, status_code=_STATUS[type(exc)], headers=_NO_STORE)


@router.get("/api/task-files")
async def list_files(request: Request) -> JSONResponse:
    """The sorted catalog file names."""
    try:
        names = await asyncio.to_thread(task_files.list_task_files, request.app.state.task_file_root)
    except task_files.TaskFileError as exc:
        return _failure(exc)
    return JSONResponse({"files": names}, headers=_NO_STORE)


@router.get("/api/task-files/{name}")
async def read_file(name: str, request: Request) -> JSONResponse:
    """One catalog's task definitions in file order."""
    try:
        contents = await asyncio.to_thread(task_files.read_task_file, request.app.state.task_file_root, name)
    except task_files.TaskFileError as exc:
        return _failure(exc)
    return JSONResponse(contents.to_json(), headers=_NO_STORE)
