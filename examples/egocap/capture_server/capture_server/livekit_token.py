# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Viewer tokens for joining the LiveKit room a station's cameras are published in.

Every consumer of the offered video (the browser preview, the recorder and the server's own delivery check)
joins with a token that can subscribe and send data messages but cannot publish media. ``LIVEKIT_API_KEY`` and
``LIVEKIT_API_SECRET`` belong to the LiveKit server and are read from the process environment only; they never
appear in a response or a log line.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets

API_KEY_ENV = "LIVEKIT_API_KEY"
API_SECRET_ENV = "LIVEKIT_API_SECRET"
#: A token only has to outlive the page's connect; LiveKit keeps an established connection going afterwards.
TOKEN_TTL_S = 3600


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
