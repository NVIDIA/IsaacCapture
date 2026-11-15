# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The three calls everything above a link uses, and an in-process link for tests."""

from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from typing import Protocol, runtime_checkable

from .messages import (
    Message,
    ProtocolError,
    Stop,
    decode,
    is_reliable,
)

_log = logging.getLogger("isaaccapture.examples.agentic_remote.carrier")


@runtime_checkable
class Carrier(Protocol):
    """One end of a link. A dead link stays dead; sending on it is ignored."""

    def send_reliable(self, msg_type: int, payload: bytes) -> None:
        """Queue a link message for delivery in order."""

    def send_latest(self, msg_type: int, payload: bytes) -> None:
        """Send newest-wins: replaces an unsent link message of the same type."""

    def receive(self, timeout_s: float) -> tuple[int, bytes] | None:
        """The next link message as (type number, payload), or None after timeout_s."""

    @property
    def alive(self) -> bool:
        """False once the link died or was closed."""

    def close(self) -> None:
        """End the link."""


class _LoopbackLink:
    """What the two ends of a loopback link share."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.alive = True


class LoopbackCarrier:
    """One end of an in-process link; loopback_pair() builds both ends.

    Link messages arrive in the order sent. A newest-wins one that replaces an
    unreceived one of its type takes the newer place in that order, so it never
    passes a reliable link message sent before it. On a dead link receive() still
    waits out its timeout, so a loop paced by it keeps its pace.
    """

    def __init__(self, link: _LoopbackLink) -> None:
        self._link = link
        self._peer = self  # loopback_pair() sets the other end
        self._arrived = threading.Condition(link.lock)
        # Arrival number -> (type number, payload), oldest first.
        self._inbox: OrderedDict[int, tuple[int, bytes]] = OrderedDict()
        # Type number -> arrival number of its unreceived newest-wins link message.
        self._newest: dict[int, int] = {}
        self._arrivals = 0

    def send_reliable(self, msg_type: int, payload: bytes) -> None:
        self._send(msg_type, payload, newest_wins=False)

    def send_latest(self, msg_type: int, payload: bytes) -> None:
        self._send(msg_type, payload, newest_wins=True)

    def receive(self, timeout_s: float) -> tuple[int, bytes] | None:
        with self._arrived:
            if not self._inbox and timeout_s > 0:
                self._arrived.wait_for(lambda: bool(self._inbox), timeout_s)
            if not self._inbox:
                return None
            arrival, received = self._inbox.popitem(last=False)
            if self._newest.get(received[0]) == arrival:
                del self._newest[received[0]]
            return received

    @property
    def alive(self) -> bool:
        with self._link.lock:
            return self._link.alive

    def close(self) -> None:
        """End the link; the peer can still receive what was sent before."""
        with self._link.lock:
            self._link.alive = False
            self._inbox.clear()
            self._newest.clear()

    def break_link(self) -> None:
        """Cut the link at once: both ends are dead and nothing under way arrives."""
        with self._link.lock:
            self._link.alive = False
            for end in (self, self._peer):
                end._inbox.clear()
                end._newest.clear()

    def _send(self, msg_type: int, payload: bytes, newest_wins: bool) -> None:
        peer = self._peer
        with self._link.lock:
            if not self._link.alive:
                return
            if newest_wins:
                replaced = peer._newest.pop(msg_type, None)
                if replaced is not None:
                    del peer._inbox[replaced]
                peer._newest[msg_type] = peer._arrivals
            peer._inbox[peer._arrivals] = (msg_type, bytes(payload))
            peer._arrivals += 1
            peer._arrived.notify()


def loopback_pair() -> tuple[LoopbackCarrier, LoopbackCarrier]:
    """The two ends of one in-process, thread-safe link."""
    link = _LoopbackLink()
    a, b = LoopbackCarrier(link), LoopbackCarrier(link)
    a._peer, b._peer = b, a
    return a, b


def send(carrier: Carrier, message: Message) -> None:
    """Send a link message reliably or newest-wins, as its type number says."""
    payload = message.to_payload()
    if is_reliable(message.TYPE):
        carrier.send_reliable(message.TYPE, payload)
    else:
        carrier.send_latest(message.TYPE, payload)


def receive_message(carrier: Carrier, timeout_s: float) -> Message | None:
    """The next link message decoded, or None when none arrived in timeout_s.

    A link message that fails to decode is logged and dropped, also as None, so a
    bad one from the peer cannot end the caller's receive loop. A STOP is never
    dropped: one with a bad payload is returned as a STOP that says so.
    """
    received = carrier.receive(timeout_s)
    if received is None:
        return None
    msg_type, payload = received
    try:
        return decode(msg_type, payload)
    except ProtocolError as error:
        if msg_type == Stop.TYPE:
            _log.warning("STOP with a bad payload is still a STOP: %s", error)
            return Stop(reason="STOP with a bad payload", sender="unknown")
        _log.warning("dropped a link message of type %s: %s", msg_type, error)
        return None
