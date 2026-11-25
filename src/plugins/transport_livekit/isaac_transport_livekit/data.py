# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fresh, bounded application messages over LiveKit data packets."""

import asyncio
import logging
import time

from livekit import rtc

from .connection import connect, disconnect
from isaac_media_transport.transport import MAX_MESSAGE as MAX_MESSAGE

from isaac_media_transport.message_protocol import (
    LEASE_TOPIC,
    Reassembler as Reassembler,
    fragments as fragments,
    CHUNK_BYTES as CHUNK_BYTES,
    HEADER as HEADER,
)


logger = logging.getLogger("isaaccapture.remote_assistance.data")


class DataSession:
    """One authenticated peer, topic allowlist, and receiver-clock freshness leases."""

    def __init__(self, settings, identity, peer, topics, on_message):
        if identity == peer:
            raise ValueError("Local and peer identities must differ")
        self.settings = settings
        self.identity = identity
        self.peer = peer
        self.topics = set(topics)
        self.on_message = on_message
        self.room = rtc.Room()
        self.receiver = Reassembler(float(settings.get("max_age_ms", 300)) / 1000)
        self.stats = self.receiver.stats
        self._peer_lease = None
        self._peer_lease_at = 0
        self._sequence = 0
        self._heartbeat = None
        self._send_lock = asyncio.Lock()
        self.closed = asyncio.Event()
        self.error = None
        self.room.on("data_received", self._received)
        self.room.on("participant_disconnected", self._peer_left)
        self.room.on("reconnecting", self._reset)
        self.room.on("disconnected", self._disconnected)

    @property
    def ready(self):
        return (
            not self.closed.is_set()
            and self.room.isconnected()
            and self._peer_lease is not None
            and time.monotonic() - self._peer_lease_at < self.receiver.max_age_s
        )

    def _reset(self, *_):
        self._peer_lease = None
        self.receiver.reset()

    def _peer_left(self, participant):
        if participant.identity == self.peer:
            self._reset()

    def _disconnected(self, *_):
        self._reset()
        self.closed.set()

    def _received(self, packet):
        if self.closed.is_set():
            return
        if packet.participant is None or packet.participant.identity != self.peer:
            self.stats["wrong_peer"] += 1
            return
        if packet.topic == LEASE_TOPIC:
            if len(packet.data) == 16:
                self._peer_lease = bytes(packet.data)
                self._peer_lease_at = time.monotonic()
            return
        if packet.topic not in self.topics:
            self.stats["wrong_topic"] += 1
            return
        payload = self.receiver.accept(packet.topic, packet.data)
        if payload is not None:
            try:
                self.on_message(packet.topic, payload)
            except Exception as exc:
                self.error = exc
                self.closed.set()

    async def start(self):
        await connect(self.room, self.settings, self.identity, publish=False)
        self._heartbeat = asyncio.create_task(self._leases())

    async def _leases(self):
        try:
            while not self.closed.is_set():
                if (
                    self.room.isconnected()
                    and self.peer in self.room.remote_participants
                ):
                    lease = self.receiver.issue_lease()
                    await self.room.local_participant.publish_data(
                        lease,
                        reliable=False,
                        destination_identities=[self.peer],
                        topic=LEASE_TOPIC,
                    )
                await asyncio.sleep(self.receiver.max_age_s / 3)
        except Exception as exc:
            self.error = exc
            self.closed.set()

    async def send(self, topic, payload, *, reliable=False):
        if not topic.startswith("ra/v1/") or topic == LEASE_TOPIC:
            raise ValueError(
                "Application topics must start with ra/v1/ and not use lease"
            )
        async with self._send_lock:
            if not self.ready:
                self.stats["not_ready"] += 1
                return False
            self._sequence += 1
            lease = self._peer_lease
            async with asyncio.timeout(self.receiver.max_age_s):
                for packet in fragments(payload, lease, self._sequence):
                    await self.room.local_participant.publish_data(
                        packet,
                        reliable=reliable,
                        destination_identities=[self.peer],
                        topic=topic,
                    )
            self.stats["sent"] += 1
            return True

    async def stop(self):
        self.closed.set()
        if self._heartbeat:
            self._heartbeat.cancel()
            await asyncio.gather(self._heartbeat, return_exceptions=True)
        self._reset()
        await disconnect(self.room)
