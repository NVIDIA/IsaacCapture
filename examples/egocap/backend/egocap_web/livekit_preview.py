# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Viewer tokens for the LiveKit video CaptureServer offers.

CaptureServer says where the video is (URL, room, publisher, track) in the LiveKit camera entries of the
session's ``streams``. This
module only issues the credential a consumer needs to join that room and subscribe, so a missing signing key can
stop video but never hides the stream list. The backend does not decode or relay video.

``LIVEKIT_API_KEY`` and ``LIVEKIT_API_SECRET`` belong to the LiveKit server and are read from the backend's
environment only; they never appear in a response or a log line.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import time

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from capture_server import protocol as p

API_KEY_ENV = "LIVEKIT_API_KEY"
API_SECRET_ENV = "LIVEKIT_API_SECRET"
#: A token only has to outlive the page's connect; LiveKit keeps an established connection going afterwards.
TOKEN_TTL_S = 3600

router = APIRouter()


def _b64url(data: bytes) -> str:
    """URL-safe base64 without padding, as JWT segments require."""
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def subscriber_token(*, api_key: str, signing_key: str, room: str, now: int, ttl_s: int = TOKEN_TTL_S) -> str:
    """A signed LiveKit token that may join ``room``, subscribe and send data messages, but not publish media."""
    header = {"alg": "HS256", "typ": "JWT"}
    claims = {
        "iss": api_key,
        # Each consumer gets its own identity: LiveKit disconnects an existing participant that reuses one.
        "sub": f"egocap-viewer-{secrets.token_hex(4)}",
        "nbf": now,
        "exp": now + ttl_s,
        "video": {
            "room": room,
            "roomJoin": True,
            "canSubscribe": True,
            "canPublish": False,
            # Data messages only, for the clock-sync exchange with the publisher. Any holder of this token can send
            # data messages into the room, so the publisher must act only on well-formed clock-sync requests.
            "canPublishData": True,
        },
    }
    # Compact JSON: the signature covers exactly these bytes, so both sides must agree on them.
    signing_input = ".".join(
        _b64url(json.dumps(part, separators=(",", ":")).encode()) for part in (header, claims)
    )
    signature = hmac.new(signing_key.encode(), signing_input.encode("ascii"), hashlib.sha256).digest()
    return f"{signing_input}.{_b64url(signature)}"


@router.get("/api/livekit-token")
async def get_livekit_token(request: Request, room: str = "") -> JSONResponse:
    """A fresh viewer token for ``room``, which must be one the current session's LiveKit cameras offer.

    Like the camera preview routes this is unauthenticated: anyone who can reach the backend can watch the
    cameras and send data messages into the room. The token cannot publish media, and it is only issued for
    rooms CaptureServer offers.
    """
    headers = {"Cache-Control": "no-store"}
    started = request.app.state.monitor.native.started
    offered = (
        {s.room for s in started.streams if isinstance(s, p.LiveKitCameraStream)} if started is not None else set()
    )
    # A token for an arbitrary room would let a caller probe or join rooms this station never advertised.
    if room not in offered:
        return JSONResponse({"detail": "room is not offered by CaptureServer"}, status_code=404, headers=headers)
    api_key = os.environ.get(API_KEY_ENV, "")
    signing_key = os.environ.get(API_SECRET_ENV, "")
    # Offered but unable to sign: say what to set, without echoing any value.
    if not api_key or not signing_key:
        detail = f"{API_KEY_ENV} and {API_SECRET_ENV} must be set for the backend"
        return JSONResponse({"detail": detail}, status_code=503, headers=headers)
    token = subscriber_token(api_key=api_key, signing_key=signing_key, room=room, now=int(time.time()))
    return JSONResponse({"token": token}, headers=headers)
