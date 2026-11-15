# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The link messages the robot app has taken off the link and not yet handled."""

from __future__ import annotations

from collections import OrderedDict

from ..protocol.messages import NEWEST_WINS_TYPES

# The most link messages, and bytes of payload, that wait at one time. The count
# is this high so that a burst is taken off the link whole, which is how a STOP
# behind it is found: what finds no room is refused one by one ahead of the STOP.
INBOX_MESSAGES = 65536
INBOX_BYTES = 8 * 1024 * 1024


class Inbox:
    """A bounded backlog of (type number, payload), kept in the order of arrival.

    As on the link, a newest-wins link message replaces a waiting one of its
    type and takes the newer place. Used by the dispatcher thread only.
    """

    def __init__(
        self, max_messages: int = INBOX_MESSAGES, max_bytes: int = INBOX_BYTES
    ) -> None:
        self.max_messages = int(max_messages)
        self.max_bytes = int(max_bytes)
        # Arrival number -> (type number, payload), oldest first.
        self._waiting: OrderedDict[int, tuple[int, bytes]] = OrderedDict()
        # Type number -> arrival number of its waiting newest-wins link message.
        self._newest: dict[int, int] = {}
        self._arrivals = 0
        self._bytes = 0

    def __len__(self) -> int:
        return len(self._waiting)

    def add(self, msg_type: int, payload: bytes) -> bool:
        """Keep a link message for later; False, and it is not kept, when full."""
        replaced = None
        if msg_type in NEWEST_WINS_TYPES:
            replaced = self._newest.get(msg_type)
        freed = 0 if replaced is None else len(self._waiting[replaced][1])
        count = len(self._waiting) - (replaced is not None)
        if count >= self.max_messages or (
            self._bytes - freed + len(payload) > self.max_bytes
        ):
            return False
        if replaced is not None:
            del self._waiting[replaced]
            self._bytes -= freed
        if msg_type in NEWEST_WINS_TYPES:
            self._newest[msg_type] = self._arrivals
        self._waiting[self._arrivals] = (msg_type, payload)
        self._bytes += len(payload)
        self._arrivals += 1
        return True

    def take(self) -> tuple[int, bytes] | None:
        """The oldest waiting link message, or None when none waits."""
        if not self._waiting:
            return None
        arrival, taken = self._waiting.popitem(last=False)
        if self._newest.get(taken[0]) == arrival:
            del self._newest[taken[0]]
        self._bytes -= len(taken[1])
        return taken
