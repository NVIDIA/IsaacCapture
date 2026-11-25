# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Room credentials and connection lifecycle, shared by media and data."""

import asyncio
import logging
import os
from datetime import timedelta
from urllib.parse import urlparse

from livekit import api, rtc

logger = logging.getLogger("isaaccapture.remote_assistance.connection")
CONNECT_TIMEOUT_S = 15
_pending_cleanup = set()


def token_for(settings: dict, identity: str, *, publish: bool = True) -> str:
    if settings.get("development", False):
        if urlparse(settings["url"]).hostname not in ("localhost", "127.0.0.1", "::1"):
            raise ValueError("development credentials are restricted to a loopback URL")
        return (
            api.AccessToken("devkey", "secret")
            .with_identity(identity)
            .with_ttl(timedelta(hours=6))
            .with_grants(
                api.VideoGrants(
                    room_join=True,
                    room=settings["room"],
                    can_publish=publish,
                    can_subscribe=True,
                    can_publish_data=True,
                )
            )
            .to_jwt()
        )
    if urlparse(settings["url"]).scheme not in ("wss", "https"):
        raise ValueError("Production credentials require a wss:// or https:// URL")
    env = settings.get("token_env", "LIVEKIT_TOKEN")
    token = os.environ.get(env)
    if not token:
        raise ValueError(f"Set {env} to a room-scoped token for {identity}")
    return token


async def connect(room: rtc.Room, settings: dict, identity: str, *, publish=True):
    token = token_for(settings, identity, publish=publish)
    joining = asyncio.create_task(
        room.connect(
            settings["url"],
            token,
            options=rtc.RoomOptions(auto_subscribe=False, dynacast=False),
        )
    )
    try:
        await asyncio.wait_for(asyncio.shield(joining), CONNECT_TIMEOUT_S)
    except BaseException:
        # The SDK assigns its handle late; cancellation cannot safely reclaim the room.
        cleanup = asyncio.create_task(_disconnect_late_room(joining, room))
        _pending_cleanup.add(cleanup)
        cleanup.add_done_callback(_pending_cleanup.discard)
        raise
    if room.local_participant.identity != identity or room.name != settings["room"]:
        await room.disconnect()
        raise ValueError("Token identity/room does not match the configured endpoint")
    logger.info("Joined room %s as %s", room.name, identity)


async def _disconnect_late_room(joining, room):
    try:
        await joining
        await disconnect(room)
    except Exception:
        logger.debug("Late LiveKit connection cleanup failed", exc_info=True)


async def disconnect(room: rtc.Room):
    try:
        async with asyncio.timeout(5):
            await room.disconnect()
    except TimeoutError:
        logger.warning("LiveKit disconnect timed out")
