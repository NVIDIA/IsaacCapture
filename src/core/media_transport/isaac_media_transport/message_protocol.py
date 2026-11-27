# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Receiver-clock freshness framing shared across transports."""

import secrets
import struct
import time
from collections import Counter

from .transport import MAX_MESSAGE

LEASE_TOPIC = "ra/v1/lease"
HEADER = struct.Struct("!4s16sQHHI")
CHUNK_BYTES = 1100
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
