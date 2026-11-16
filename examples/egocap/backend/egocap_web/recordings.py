# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""PoC demo interface for browsing and downloading recorded files -- not
part of the final system design, and not a RecordingService responsibility
of its own. Self-contained: its own JSON listing endpoint (scans the
recording output directory fresh on every request, so new subfolders/files
written by a session in progress show up without a server restart), its
own file-serving route, and its own page -- built independently by
../../recordings_frontend (its own package.json/tsconfig/vite.config.ts,
no dependency on the main frontend's toolchain). app.py's only involvement
is mounting `router`; everything the recordings feature needs lives here.

Off by default: set EGOCAP_RECORDINGS_ENABLED=1 to turn it on. Unlike
EGOCAP_VISER_ENABLED (opt-out, on by default), this feature is opt-in --
unset, `router` has no routes at all.

Turned on, the listing and every file need the run's control token, the
same credential that admits recording commands. A caller presents it in
``X-Control-Token``; the page signs in once with it and is given a
same-site, HTTP-only cookie, since a download link cannot carry a header.
A request a browser marks as coming from another site is refused whatever
it carries. The page's own static files stay public: they hold no
recording, and they are what shows the sign-in.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import secrets
import stat as stat_module
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles

from .preview_routes import configured_allowed_origins, trusted_hostnames
from .recording_command import control_token_valid

router = APIRouter()

#: The cookie the page is given once it has presented the control token.
SESSION_COOKIE = "egocap_recordings"

_FRONTEND_DIST = Path(__file__).resolve().parents[2] / "recordings_frontend" / "dist"


def _hidden(name: str) -> bool:
    """True for a name that must never be listed or downloaded directly: a
    session in progress holds `.recording/` (manifests, session.lock, temp
    files -- see recording/egocap/recording/episode_layout.py) and
    `episode_N.pending/` (growing MP4s appended while a take records).
    FileResponse sets Content-Length from one stat then reads to EOF, so
    downloading a growing file either aborts mid-transfer or serves
    something unplayable; a dot-prefixed name is also just not meant for
    browsing. Finalized episodes still appear live, which is the point."""
    return name.startswith(".") or name.endswith(".pending")


def enabled() -> bool:
    """Whether the whole feature is turned on -- see the module docstring's
    off-by-default rationale."""
    return os.environ.get("EGOCAP_RECORDINGS_ENABLED", "").lower() in ("1", "true", "yes")


def _recording_output_root(request: Request) -> Path:
    """The SAME ``Path`` the real recording coordinator writes to -- read from
    ``request.app.state.recording_output_root``, resolved exactly once at backend startup
    (app.py's ``lifespan``) from the station YAML/env precedence, never reread per request: a
    YAML file edited mid-process must never point this browsing UI somewhere the actual recorder
    isn't writing to.

    Falls back to the same fixed development default app.py's own helper uses when that app
    state isn't set -- this module's own standalone tests mount just ``router`` on a bare
    ``FastAPI()``, without running the real app's lifespan; see the module docstring's "no
    dependency on app.py" design."""
    configured = getattr(request.app.state, "recording_output_root", None)
    if configured is not None:
        return configured
    return Path("./recordings").expanduser()


def _resolve(base: Path, path: str) -> Path | None:
    """Resolves ``path`` under ``base``, or ``None`` if it would escape that
    root -- the only thing standing between this and the rest of the
    filesystem being requestable by anyone who can reach the app.
    ``resolve()`` follows symlinks, so a symlink inside the root pointing
    outside it is caught the same way a literal ``..`` is."""
    target = (base / path).resolve()
    if target != base and base not in target.parents:
        return None
    return target


def _hidden_path(base: Path, target: Path) -> bool:
    """True if ``target`` itself, or any directory between it and ``base``,
    is ``_hidden()`` -- covers both a hidden entry nested under an ordinary
    directory (e.g. a file under ``.recording/``) and the requested path
    itself being hidden (e.g. listing ``.recording/`` or ``episode_0.pending/``
    directly). ``target == base`` (the root itself) yields no parts, so this
    naturally returns ``False`` without a separate root case. Both callers
    only ever pass a ``target`` already inside ``base`` (``_resolve`` ran
    first), but ``relative_to`` raising on a genuinely outside path -- rather
    than silently walking ``.parent`` forever -- is the safe failure mode if
    that precondition were ever violated."""
    return any(_hidden(part) for part in target.relative_to(base).parts)


_NO_STORE_HEADERS = {"Cache-Control": "no-store"}


def _session_value(control_token: str) -> str:
    """The cookie value that proves the control token was presented, without being the token: a cookie is
    stored by the browser, and the token itself must not be readable from there."""
    return hmac.new(control_token.encode(), b"egocap recordings browser", hashlib.sha256).hexdigest()


def _same_site(request: Request) -> bool:
    """Whether a request comes from this backend's own page, a configured origin, or no page at all.

    A browser states where a request comes from; one that says another site sent it is refused. A request
    with neither header is a navigation or a non-browser client, which the credential alone decides.
    """
    site = request.headers.get("sec-fetch-site")
    if site is not None and site not in ("same-origin", "none"):
        return False
    origin = request.headers.get("origin")
    if origin is None or origin in configured_allowed_origins():
        return True
    # The same rule the live session socket applies: the backend's own origin, and only for a trusted host.
    trusted = trusted_hostnames()
    if trusted is not None and (request.url.hostname or "") not in trusted:
        return False
    return origin == f"{request.url.scheme}://{request.headers.get('host', '')}"


def _refusal(request: Request, *, cookie_accepted: bool = True) -> JSONResponse | None:
    """Why a request may not read recordings, as the reply to send; None when it may."""
    context = getattr(request.app.state, "recording_context", None)
    token = getattr(context, "control_token", None)
    if token is None:
        return JSONResponse({"detail": "recordings are not available yet"}, 503, headers=_NO_STORE_HEADERS)
    if not _same_site(request):
        return JSONResponse({"detail": "origin not allowed"}, 403, headers=_NO_STORE_HEADERS)
    if control_token_valid(request.headers.get("x-control-token"), token):
        return None
    cookie = request.cookies.get(SESSION_COOKIE)
    # compare_digest refuses non-ASCII text outright, so that is ruled out first.
    signed_in = cookie is not None and cookie.isascii() and secrets.compare_digest(cookie, _session_value(token))
    if cookie_accepted and signed_in:
        return None
    return JSONResponse({"detail": "enter the control token to browse recordings"}, 401, headers=_NO_STORE_HEADERS)


if enabled():

    @router.post("/recordings/api/session", response_model=None)
    def sign_in(request: Request) -> Response:
        """Exchange the control token, sent in ``X-Control-Token``, for the cookie the page's listing calls
        and download links carry. The cookie is HTTP-only and same-site, and is scoped to these routes."""
        refused = _refusal(request, cookie_accepted=False)
        if refused is not None:
            return refused
        response = Response(status_code=204, headers=_NO_STORE_HEADERS)
        response.set_cookie(
            SESSION_COOKIE,
            _session_value(request.app.state.recording_context.control_token),
            path="/recordings",
            httponly=True,
            samesite="strict",
            secure=request.url.scheme == "https",
        )
        return response

    @router.get("/recordings/api/list", response_model=None)
    @router.get("/recordings/api/list/{path:path}", response_model=None)
    def list_recordings(request: Request, path: str = "") -> JSONResponse:
        """Scans ``path`` (relative to the recording root) fresh on every
        call -- new subfolders/files from a session still being written
        need to appear without caching -- and returns name/is_dir/size/
        mtime per entry. A plain (not async) def: FastAPI runs it in its
        threadpool, since the filesystem work below is synchronous and
        would otherwise block the event loop every other route shares."""
        refused = _refusal(request)
        if refused is not None:
            return refused
        base = _recording_output_root(request).resolve()
        target = _resolve(base, path)
        if target is None or _hidden_path(base, target):
            return JSONResponse({"detail": "Not Found"}, status_code=404, headers=_NO_STORE_HEADERS)
        if not target.is_dir():
            # The root not existing yet just means nothing has been
            # recorded on this deployment -- a normal, expected
            # first-time state, not an error -- so it's an empty
            # listing, not a 404. A missing SUBPATH (a stale URL, or one
            # renamed away mid-scan by finalization) is still a real
            # not-found.
            if path == "":
                return JSONResponse({"path": path, "entries": []}, headers=_NO_STORE_HEADERS)
            return JSONResponse({"detail": "Not Found"}, status_code=404, headers=_NO_STORE_HEADERS)
        entries = []
        try:
            raw_entries = list(target.iterdir())
        except OSError:
            # The directory itself vanished/renamed between is_dir() and
            # iterdir() (finalization), or a real permission error -- an
            # ordinary not-found either way, not a 500.
            return JSONResponse({"detail": "Not Found"}, status_code=404, headers=_NO_STORE_HEADERS)
        for entry in raw_entries:
            if _hidden(entry.name):
                continue
            try:
                entry.name.encode("utf-8")
            except UnicodeEncodeError:
                # A non-UTF-8 name would otherwise 500 the whole listing
                # at JSON encoding time (JSONResponse uses
                # ensure_ascii=False); skip just that one entry.
                continue
            try:
                info = entry.stat()
            except OSError:
                # Vanished/renamed mid-scan, or unreadable -- skip rather
                # than fail the whole listing over one entry.
                continue
            is_dir = stat_module.S_ISDIR(info.st_mode)
            entries.append(
                {
                    "name": entry.name,
                    "is_dir": is_dir,
                    "size": None if is_dir else info.st_size,
                    "mtime": info.st_mtime,
                }
            )
        entries.sort(key=lambda e: (not e["is_dir"], e["name"]))
        return JSONResponse({"path": path, "entries": entries}, headers=_NO_STORE_HEADERS)

    @router.get("/recordings/files/{path:path}", response_model=None)
    def get_recording_file(request: Request, path: str) -> FileResponse | JSONResponse:
        """Serves one recorded file directly, for links the page rendered
        by the ``_FRONTEND_DIST`` mount below builds from the listing
        above. A plain def, same reason as list_recordings: _resolve and
        is_file are synchronous filesystem work. ``filename=`` makes the
        download go out as an attachment (Content-Disposition), so a
        stray ``.html``/``.svg`` recorded alongside a take can't execute
        as script on this origin if opened directly."""
        refused = _refusal(request)
        if refused is not None:
            return refused
        base = _recording_output_root(request).resolve()
        target = _resolve(base, path)
        if target is None or not target.is_file() or _hidden_path(base, target):
            return JSONResponse({"detail": "Not Found"}, status_code=404)
        return FileResponse(target, filename=target.name)

    # Mounted under router so app.py can register everything this module
    # needs with one `app.include_router(recordings.router)` call. Only
    # mounted if ../../recordings_frontend has actually been built (a
    # source checkout before `npm run build` there can still run the
    # backend standalone).
    if _FRONTEND_DIST.is_dir():

        @router.get("/recordings", response_model=None)
        async def recordings_root_redirect() -> RedirectResponse:
            """StaticFiles' mount only resolves index.html under a trailing
            slash; without this, the bare /recordings URL 404s."""
            return RedirectResponse(url="/recordings/")

        router.mount("/recordings", StaticFiles(directory=_FRONTEND_DIST, html=True), name="recordings_frontend")
