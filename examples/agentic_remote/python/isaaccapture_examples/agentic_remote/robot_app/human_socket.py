# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The robot app's local socket for a human program on the Thor.

The human program is the program through which a person drives the arm with
the headset's right controller. It attaches to a Unix socket that speaks the
local API (link_service/local.py): at the workstation the robot link service's,
on the Thor, the machine on the robot, this one. Here a robot link service of
this process serves the socket, to clients of the role "human" only, and what
stands for its robot connection is an in-process link whose other end the
robot app reads. A STOP of the human program then passes neither the robot
link service of the workstation nor the robot connection, the link between
the two machines, and reaches the robot app with that connection down. The
button behind it, right B, still arrives at the human program over the teleop
connection from the workstation, Isaac Capture's remote session.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from ..link_service.service import (
    Incoming,
    RobotLinkService,
    incoming_carrier,
)
from ..protocol.carrier import loopback_pair
from ..protocol.local_messages import ROLE_HUMAN, ROLES, LocalHello, decode_local
from ..protocol.messages import ProtocolError

if TYPE_CHECKING:
    from ..protocol.carrier import Carrier

_log = logging.getLogger("isaaccapture.examples.agentic_remote.human_socket")

# What a client of another role is told.
ONLY_HUMAN = "this socket takes the human program only: the role human"
# Sockets served at one time: the human program's, and a few that are turned
# away or bring only a STOP.
MAX_CLIENTS = 4
# The address text of the in-process link, for the service's check and its log.
_PEER = "this-process"


class _HumanOnlyService(RobotLinkService):
    """A robot link service that takes clients of the role "human" alone."""

    def _hello(self, client: Any, payload: bytes) -> bool:
        try:
            role = decode_local(LocalHello.TYPE, payload).role
        except ProtocolError:
            role = None  # refused below, with the service's own reason
        if role in ROLES and role != ROLE_HUMAN:
            self._bad(client, ONLY_HUMAN)
            return False
        return super()._hello(client, payload)


class HumanSocket:
    """A Unix socket of the robot app that one human program attaches to.

    The socket speaks the local API as the robot link service's does, with its
    checks: a second human program and a client of another role are refused,
    a client that does not read its socket is cut off, and a STOP goes on
    whoever sent it. `carrier` is the robot app's end: it receives what the
    human program sent, and what is sent on it goes to the human program.
    Reading the socket and writing to it happen on threads of this object;
    the robot app's end never waits for the socket.
    """

    def __init__(self, path: str, **service_options: Any) -> None:
        """path is the socket's file; service_options go to RobotLinkService."""
        if not path:
            raise ValueError("the human program's socket needs a path")
        self._app_end, self._service_end = loopback_pair()
        self._handed_over = False
        options = {"max_clients": MAX_CLIENTS, **service_options}
        self._service = _HumanOnlyService(
            self._accept, socket_path=path, allowed_peers=(_PEER,), **options
        )

    @property
    def carrier(self) -> Carrier:
        """The robot app's end; alive until close()."""
        return self._app_end

    @property
    def path(self) -> str:
        """The socket's file."""
        return self._service.socket_path

    def attached(self) -> bool:
        """Whether a human program is attached now."""
        return self._service.roles().get(ROLE_HUMAN, 0) > 0

    def start(self) -> None:
        """Make the socket and serve it.

        Raises LocalSocketInUse when a live program has the socket and
        LocalSocketError when it cannot be made; both are OSErrors.
        """
        self._service.start()
        _log.info("the human program's socket listens on %s", self.path)

    def close(self) -> None:
        """Close the socket, remove its file and end the threads."""
        self._service.close()
        self._app_end.close()

    def _accept(self, timeout_s: float) -> Incoming | None:
        """The service's one robot connection: the in-process link, once."""
        if self._handed_over:
            return None
        self._handed_over = True
        return incoming_carrier(_PEER, self._service_end)
