# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The three calls of a Carrier over a real Isaac Capture link.

Isaac Capture is imported only where a link is made, so the package runs without
it on loopback links.
"""

from __future__ import annotations

import math
from typing import Any

from .messages import MAX_PAYLOAD_BYTES, NEWEST_WINS_TYPES, ProtocolError

Buffer = bytes | bytearray | memoryview

# Of each of these types the peer keeps only the newest unread link message.
LATEST_TYPES = NEWEST_WINS_TYPES
# Type numbers of the link's own HEARTBEAT, PING and PONG.
_LINK_OWN_TYPES = frozenset({11, 13, 14})
# The longest one receive() waits, in s; a longer timeout is cut to it.
_LONGEST_RECEIVE_S = 3600.0


class LinkCarrier:
    """One end of a robot connection: a Carrier over isaaccapture.remote_link.Link.

    A link that died still hands out what arrived before; with nothing left
    receive() waits out its timeout. Sending on a dead link is ignored.
    """

    def __init__(self, link: Any) -> None:
        """link is a started isaaccapture.remote_link.Link; the carrier owns it."""
        self._link = link

    @classmethod
    def connect(
        cls, endpoint: str, timeout_s: float, *, name: str = "robot"
    ) -> LinkCarrier:
        """Dial endpoint ("host:port") and start the link.

        Raises ValueError for a bad endpoint or timeout_s, TimeoutError when
        timeout_s ran out and RuntimeError when the connection cannot be made.
        """
        if not (math.isfinite(timeout_s) and timeout_s > 0.0):
            raise ValueError(f"timeout_s must be finite and above 0, got {timeout_s!r}")
        from isaaccapture.remote_link import Link

        timeout_ms = max(1, math.ceil(timeout_s * 1000.0))
        return cls(Link.connect(endpoint, timeout_ms, name=name, **_link_options()))

    @classmethod
    def adopt(cls, fd: int, *, name: str = "robot") -> LinkCarrier:
        """Start the link on an accepted, connected socket descriptor.

        fd must be the caller's alone to give (socket.detach()). After ValueError
        or TypeError the caller still owns it; otherwise the link does.
        """
        from isaaccapture.remote_link import Link

        return cls(Link.adopt(fd, name=name, **_link_options()))

    @property
    def link(self) -> Any:
        """The link itself, for its counters (stats())."""
        return self._link

    def send_reliable(self, msg_type: int, payload: Buffer) -> None:
        """Queue a link message for delivery in order; raises ProtocolError."""
        self._link.send_reliable(*_checked(msg_type, payload))

    def send_latest(self, msg_type: int, payload: Buffer) -> None:
        """Send newest-wins; raises ProtocolError as send_reliable()."""
        self._link.send_latest(*_checked(msg_type, payload))

    def receive(self, timeout_s: float) -> tuple[int, bytes] | None:
        """The next link message as (type number, payload), or None after timeout_s."""
        timeout_ms = 0
        if timeout_s > 0.0:  # also False for NaN; the link takes below 0 as no limit
            timeout_ms = max(1, math.ceil(min(timeout_s, _LONGEST_RECEIVE_S) * 1000.0))
        received = self._link.receive(timeout_ms)
        if received is None:
            return None
        return int(received.type), bytes(received.payload)

    @property
    def alive(self) -> bool:
        """False once the link died or was closed."""
        return bool(self._link.alive)

    @property
    def close_reason(self) -> str:
        """Why the link went down; "" while it is alive."""
        return str(self._link.close_reason)

    def clock_offset(self) -> tuple[int, bool, bool]:
        """(offset_ns, two_way, valid): this machine's monotonic clock minus the peer's.

        two_way says that ping round trips made the estimate; valid that there
        is one at all.
        """
        estimate = self._link.clock()
        return int(estimate.offset_ns), bool(estimate.two_way), bool(estimate.valid)

    def close(self) -> None:
        """End the link; what is queued as reliable still goes out for a moment."""
        self._link.close()


def _link_options() -> dict[str, Any]:
    """What connect() and adopt() ask of the link beyond its defaults.

    The heartbeat period and the peer timeout stay the link's own: 0.5 s and 3 s.
    Both ends send pings, so each has a two-way clock estimate.
    """
    return {"send_pings": True, "latest_types": LATEST_TYPES}


def _checked(msg_type: int, payload: Buffer) -> tuple[int, bytes]:
    """The type number and the payload as the link takes them; raises ProtocolError."""
    usable = type(msg_type) is int and 0 <= msg_type <= 255
    if not usable or msg_type in _LINK_OWN_TYPES:
        raise ProtocolError(f"{msg_type!r} is not a type number a link carries")
    if len(payload) > MAX_PAYLOAD_BYTES:
        raise ProtocolError(
            f"{len(payload)} bytes is over the limit of {MAX_PAYLOAD_BYTES} bytes"
        )
    return msg_type, bytes(payload)
