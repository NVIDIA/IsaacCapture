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

import os
import time

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from capture_server import protocol as p
from capture_server.livekit_token import API_KEY_ENV, API_SECRET_ENV, subscriber_token

router = APIRouter()


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
