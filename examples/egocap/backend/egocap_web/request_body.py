# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Bound JSON mutation requests before parsing or performing filesystem work."""

import json
from typing import Any

from fastapi import Request
from fastapi.responses import JSONResponse

from .recording_command import control_token_valid


def _refusal(message: str, status: int) -> tuple[None, JSONResponse]:
    """Return a non-cacheable refusal without disclosing request or server details."""
    return None, JSONResponse({"detail": message}, status_code=status, headers={"Cache-Control": "no-store"})


async def read_authorized_json(
    request: Request, expected_token: str, max_bytes: int
) -> tuple[dict[str, Any] | None, JSONResponse | None]:
    """Authenticate header tokens before reading; retain bounded legacy body-token support."""
    # Header authentication avoids spending body memory on callers with a rejected credential.
    token = request.headers.get("x-control-token")
    if token is not None and not control_token_valid(token, expected_token):
        return _refusal("control token rejected", 403)
    declared = request.headers.get("content-length")
    if declared is not None:
        try:
            length = int(declared)
        except ValueError:
            return _refusal("invalid content length", 400)
        if length < 0:
            return _refusal("invalid content length", 400)
        if length > max_bytes:
            return _refusal("request body is too large", 413)

    # Content-Length can be absent or dishonest, so the stream itself is the authority.
    raw = bytearray()
    async for chunk in request.stream():
        if len(chunk) > max_bytes - len(raw):
            return _refusal("request body is too large", 413)
        raw.extend(chunk)
    try:
        body = json.loads(raw)
    except (ValueError, UnicodeError):
        return _refusal("body must be JSON", 400)
    if not isinstance(body, dict):
        return _refusal("body must be an object", 400)
    # Older clients send the token in the body; an explicit header always takes precedence.
    if token is None and not control_token_valid(body.get("control_token"), expected_token):
        return _refusal("control token rejected", 403)
    return body, None
