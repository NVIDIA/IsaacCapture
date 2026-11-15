# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Where the robot link service waits for the robot app to dial in.

Isaac Capture is imported only where the listener is made.
"""

from __future__ import annotations

import logging
import math
import os
import threading
from typing import Any

from ..protocol.link_carrier import LinkCarrier
from .service import Incoming

_log = logging.getLogger("isaaccapture.examples.agentic_remote.listen")

LINK_NAME = "robot-link-service"
_MAPPED = "::ffff:"  # how a listener on "::" reports an IPv4 peer


def plain_peer(peer: str) -> str:
    """A peer address with an IPv4 host written as IPv4.

    "[::ffff:10.0.0.2]:5000" becomes "10.0.0.2:5000"; anything else is kept.
    """
    if peer.startswith("[" + _MAPPED):
        host, bracket, port = peer[1 + len(_MAPPED) :].partition("]")
        if bracket and "." in host and ":" not in host:
            return host + port
    return peer


class RobotListener:
    """A listening TCP socket whose connections become LinkCarriers.

    accept() is the `accept` a RobotLinkService takes. The carriers are safe to
    send on from several threads, as the service does: a Link's sends only queue
    under the link's own lock, and its own thread writes to the socket.
    """

    def __init__(self, host: str, port: int) -> None:
        """Bind and listen; port 0 picks a free port.

        Raises ImportError without Isaac Capture and RuntimeError when the
        address cannot be listened on.
        """
        from isaaccapture.remote_link import LinkListener

        self._listener: Any = LinkListener(host, int(port))
        self._host = host
        self._closed = threading.Event()

    @property
    def port(self) -> int:
        """The bound port, also after close()."""
        return int(self._listener.port)

    @property
    def endpoint(self) -> str:
        """What a robot app on this machine dials: "host:port"."""
        return f"{self._host}:{self.port}"

    def accept(self, timeout_s: float) -> Incoming | None:
        """The next connection as an Incoming; None after timeout_s or once closed."""
        if self._closed.is_set():
            return None
        timeout_ms = 0
        if timeout_s > 0.0:
            timeout_ms = max(1, math.ceil(timeout_s * 1000.0))
        try:
            got = self._listener.accept(timeout_ms)
        except RuntimeError:
            if self._closed.is_set():
                return None
            raise
        if got is None:
            return None
        fd, peer = got

        def open_link() -> LinkCarrier:
            try:
                return LinkCarrier.adopt(fd, name=LINK_NAME)
            except (ValueError, TypeError):
                os.close(fd)  # the link did not take the descriptor
                raise

        return Incoming(plain_peer(str(peer)), open_link, lambda: os.close(fd))

    def close(self) -> None:
        """Stop listening; connections not accepted yet are reset. Idempotent."""
        self._closed.set()
        self._listener.close()
