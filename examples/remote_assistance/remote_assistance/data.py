# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fresh, bounded application messages over LiveKit data packets."""

import asyncio
import logging
import secrets
import struct
import time
from collections import Counter

from livekit import rtc

from .connection import connect, disconnect

logger = logging.getLogger("isaaccapture.remote_assistance.data")
LEASE_TOPIC = "ra/v1/lease"
HEADER = struct.Struct("!4s16sQHHI")
CHUNK_BYTES = 1100
MAX_MESSAGE = 128 * 1024
MAX_PENDING = 32


class Reassembler:
    """Validate leases, size and sequence before accepting a complete message."""

    def __init__(self, max_age_s=0.3, clock=time.monotonic):
        if not 0.05 <= max_age_s <= 2:
            raise ValueError("max_age_s must be between 0.05 and 2 seconds")
        self.max_age_s = max_age_s
        self.clock = clock
        self.leases = {}
        self.pending = {}
        self.last_seq = {}
        self.stats = Counter()

    def reset(self):
        self.leases.clear()
        self.pending.clear()
        self.last_seq.clear()

    def issue_lease(self):
        now = self.clock()
        self.leases = {k: v for k, v in self.leases.items() if now - v < self.max_age_s}
        self.pending = {k: v for k, v in self.pending.items() if v[0] in self.leases}
        lease = secrets.token_bytes(16)
        self.leases[lease] = now
        return lease

    def accept(self, topic, packet):
        if len(packet) < HEADER.size or len(packet) > HEADER.size + CHUNK_BYTES:
            self.stats["malformed"] += 1
            return None
        magic, lease, seq, index, count, size = HEADER.unpack_from(packet)
        if (
            magic != b"RA01"
            or not 0 < size <= MAX_MESSAGE
            or count != (size + CHUNK_BYTES - 1) // CHUNK_BYTES
            or index >= count
        ):
            self.stats["malformed"] += 1
            return None
        if (
            lease not in self.leases
            or self.clock() - self.leases[lease] >= self.max_age_s
        ):
            self.stats["expired"] += 1
            return None
        if seq <= self.last_seq.get(topic, -1):
            self.stats["old_sequence"] += 1
            return None
        chunk = packet[HEADER.size :]
        if len(chunk) != min(CHUNK_BYTES, size - index * CHUNK_BYTES):
            self.stats["malformed"] += 1
            return None
        key = (topic, seq)
        if key not in self.pending:
            if len(self.pending) >= MAX_PENDING:
                self.pending.pop(next(iter(self.pending)))
                self.stats["evicted"] += 1
            self.pending[key] = (lease, size, {})
        saved_lease, saved_size, parts = self.pending[key]
        if saved_lease != lease or saved_size != size:
            self.stats["malformed"] += 1
            return None
        parts[index] = chunk
        if len(parts) != count:
            return None
        result = b"".join(parts[i] for i in range(count))
        self.last_seq[topic] = seq
        self.pending = {
            k: v for k, v in self.pending.items() if k[0] != topic or k[1] > seq
        }
        self.stats["received"] += 1
        return result


def fragments(payload, lease, sequence):
    if len(lease) != 16 or not 0 < len(payload) <= MAX_MESSAGE:
        raise ValueError("Invalid lease or payload size (limit: 128 KiB)")
    count = (len(payload) + CHUNK_BYTES - 1) // CHUNK_BYTES
    for index in range(count):
        yield (
            HEADER.pack(b"RA01", lease, sequence, index, count, len(payload))
            + payload[index * CHUNK_BYTES : (index + 1) * CHUNK_BYTES]
        )


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
            self.room.isconnected()
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
