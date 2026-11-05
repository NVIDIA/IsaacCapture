# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Direct LiveKit camera preview: the station's room/track mapping, a subscribe-only token, and one endpoint.

The camera host publishes already-encoded H.264 into a LiveKit room. The browser subscribes to those tracks
itself, so the backend never decodes, re-encodes or relays video and does not take part in recording. This module
only tells the browser where the room is and hands it a short-lived token that can subscribe and nothing else.

``LIVEKIT_API_KEY`` and ``LIVEKIT_API_SECRET`` belong to the LiveKit server and are read from the backend's
environment only; they never appear in the station YAML, a response or a log line.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import time
from dataclasses import dataclass

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

API_KEY_ENV = "LIVEKIT_API_KEY"
API_SECRET_ENV = "LIVEKIT_API_SECRET"
#: A token only has to outlive the page's connect; LiveKit keeps an established connection going afterwards.
TOKEN_TTL_S = 3600

router = APIRouter()


@dataclass(frozen=True)
class LiveKitTrack:
    """One camera tile: the operator-facing ``name`` and the published video track ``track`` it shows."""

    name: str
    track: str


@dataclass(frozen=True)
class LiveKitPreview:
    """Where the camera host publishes: signalling ``url``, ``room``, ``publisher`` identity and its tracks.

    A track name is only unique within one participant, so the browser selects by publisher plus track.
    """

    url: str
    room: str
    publisher: str
    tracks: tuple[LiveKitTrack, ...]


def _b64url(data: bytes) -> str:
    """URL-safe base64 without padding, as JWT segments require."""
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def subscriber_token(*, api_key: str, api_secret: str, room: str, now: int, ttl_s: int = TOKEN_TTL_S) -> str:
    """A signed LiveKit access token that may join ``room`` and subscribe, but not publish."""
    header = {"alg": "HS256", "typ": "JWT"}
    claims = {
        "iss": api_key,
        # Each browser gets its own identity: LiveKit disconnects an existing participant that reuses one.
        "sub": f"egocap-viewer-{secrets.token_hex(4)}",
        "nbf": now,
        "exp": now + ttl_s,
        "video": {
            "room": room,
            "roomJoin": True,
            "canSubscribe": True,
            "canPublish": False,
            "canPublishData": False,
        },
    }
    # Compact JSON: the signature covers exactly these bytes, so both sides must agree on them.
    signing_input = ".".join(
        _b64url(json.dumps(part, separators=(",", ":")).encode()) for part in (header, claims)
    )
    signature = hmac.new(api_secret.encode(), signing_input.encode("ascii"), hashlib.sha256).digest()
    return f"{signing_input}.{_b64url(signature)}"


@router.get("/api/livekit-preview")
async def get_livekit_preview(request: Request) -> JSONResponse:
    """The room, tracks and a fresh subscribe-only token for the browser, or why preview is unavailable.

    Like the camera preview routes this is unauthenticated: anyone who can reach the backend can watch the
    cameras, which is why the token cannot publish.
    """
    headers = {"Cache-Control": "no-store"}
    preview: LiveKitPreview | None = request.app.state.livekit_preview
    # Not configured is a normal state for a station that previews through CaptureServer instead.
    if preview is None:
        return JSONResponse({"detail": "LiveKit preview is not configured"}, status_code=404, headers=headers)
    api_key = os.environ.get(API_KEY_ENV, "")
    api_secret = os.environ.get(API_SECRET_ENV, "")
    # Configured but unable to sign: say what to set, without echoing any value.
    if not api_key or not api_secret:
        detail = f"{API_KEY_ENV} and {API_SECRET_ENV} must be set for the backend"
        return JSONResponse({"detail": detail}, status_code=503, headers=headers)
    token = subscriber_token(api_key=api_key, api_secret=api_secret, room=preview.room, now=int(time.time()))
    body = {
        "url": preview.url,
        "room": preview.room,
        "publisher": preview.publisher,
        "token": token,
        "tracks": [{"name": t.name, "track": t.track} for t in preview.tracks],
    }
    return JSONResponse(body, headers=headers)
