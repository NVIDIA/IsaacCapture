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
    env = settings.get("token_env", "LIVEKIT_TOKEN")
    token = os.environ.get(env)
    if not token:
        raise ValueError(f"Set {env} to a room-scoped token for {identity}")
    return token


async def connect(room: rtc.Room, settings: dict, identity: str, *, publish=True):
    token = token_for(settings, identity, publish=publish)
    async with asyncio.timeout(15):
        await room.connect(
            settings["url"],
            token,
            options=rtc.RoomOptions(auto_subscribe=False, dynacast=False),
        )
    if room.local_participant.identity != identity or room.name != settings["room"]:
        await room.disconnect()
        raise ValueError("Token identity/room does not match the configured endpoint")
    logger.info("Joined room %s as %s", room.name, identity)


async def disconnect(room: rtc.Room):
    try:
        async with asyncio.timeout(5):
            await room.disconnect()
    except TimeoutError:
        logger.warning("LiveKit disconnect timed out")
