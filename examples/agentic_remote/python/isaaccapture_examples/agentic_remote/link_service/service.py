# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The robot link service: holds the robot connection and serves the local API.

See sections 5 and 6 of docs/design/claude_agentic_remote_execution.md. The
service forwards what its clients send, unchanged, and never originates motion:
the one link message it makes itself is the "agent program gone" help request.
"""

from __future__ import annotations

import contextlib
import itertools
import logging
import re
import socket
import threading
import time
import uuid
from collections import OrderedDict
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..protocol.local_messages import (
    LOCAL_MESSAGE_TYPES,
    MAX_LOCAL_PAYLOAD_BYTES,
    ROLE_AGENT,
    ROLE_CONSOLE,
    ROLE_TOOL,
    ROLE_VIEWER,
    ROLES,
    LocalHello,
    LocalNotice,
    LocalStatus,
    decode_local,
    is_local,
)
from ..protocol.messages import (
    COMMAND_TYPES,
    MAX_PAYLOAD_BYTES,
    MESSAGE_TYPES,
    PROTOCOL_VERSION,
    CheckerState,
    Event,
    Gripper,
    Move,
    Open,
    OwnerRequest,
    Permit,
    ProtocolError,
    Release,
    Reset,
    Result,
    RobotState,
    Shutdown,
    Stop,
    ViewStatus,
    Welcome,
    command_id_in,
    decode,
    is_reliable,
)
from ..robot_app.text import (
    MAX_ID_CHARS,
    MAX_REASON_CHARS,
    MAX_WORD_CHARS,
    LineLimit,
    printable,
)
from .local import (
    HEADER,
    SEND_TIMEOUT_S,
    LocalListener,
    RecordQueue,
    Stalled,
    pack_record,
    read_exact,
)

if TYPE_CHECKING:
    from ..protocol.carrier import Carrier
    from ..protocol.messages import Message

_log = logging.getLogger("isaaccapture.examples.agentic_remote.link_service")

# Reasons the service itself gives.
LOST = "robot connection lost"
AGENT_GONE = "agent program gone"
BAD_STOP = "STOP with a bad payload"
STOP_UNDELIVERED = (
    "STOP was not delivered: there is no robot connection; the robot holds on its "
    "own when the robot connection is lost"
)

# Which link messages each role may send; everything else is refused.
ROLE_TABLE: dict[str, frozenset[int]] = {
    ROLE_AGENT: frozenset(
        cls.TYPE for cls in (Move, Gripper, Permit, CheckerState, OwnerRequest, Stop)
    ),
    ROLE_CONSOLE: frozenset(
        cls.TYPE for cls in (Stop, OwnerRequest, Reset, Release, Open, Shutdown)
    ),
    ROLE_VIEWER: frozenset(cls.TYPE for cls in (ViewStatus, Stop)),
    ROLE_TOOL: frozenset({Stop.TYPE}),
}
# Which OWNER_REQUESTs a role may send.
OWNER_REQUESTS: dict[str, frozenset[str]] = {
    ROLE_AGENT: frozenset({"agent", "help_needed", "nobody"}),
    ROLE_CONSOLE: frozenset({"rest", "nobody"}),
}
# What the Thor sends; anything else from it is dropped.
_FROM_THOR = frozenset(cls.TYPE for cls in (Welcome, RobotState, Result, Event))

# The longest payload a client may send: a MOVE, and anything else.
SMALL_PAYLOAD_BYTES = 64 * 1024
# How long the payload of a STOP may take to follow its header before the STOP
# is forwarded without it.
STOP_PAYLOAD_WAIT_S = 0.2
# How long a client there is no room for has to send a STOP, and as long again
# to take the notice that tells it why it is closed.
TURN_AWAY_S = 0.2
# Threads that turn clients away; with all of them busy the accept thread does.
TURN_AWAY_HELPERS = 4
# Commands whose sender is remembered for their RESULTs.
COMMANDS_KEPT = 1024

_WORD_START = re.compile(r"(?<!^)(?=[A-Z])")


def link_message_name(msg_type: int) -> str:
    """The name a type number has in the design, such as OWNER_REQUEST."""
    cls = MESSAGE_TYPES.get(msg_type) or LOCAL_MESSAGE_TYPES.get(msg_type)
    if cls is None:
        return f"type {msg_type}"
    return _WORD_START.sub("_", cls.__name__).upper()


def peer_host(address: str) -> str:
    """The host of a peer address: "10.0.0.2:5000" and "[::1]:5000" lose the port."""
    address = str(address).strip()
    if address.startswith("["):
        return address[1:].partition("]")[0]
    host, colon, port = address.rpartition(":")
    if colon and port.isdigit() and ":" not in host:
        return host
    return address


@dataclass(frozen=True)
class Incoming:
    """A robot connection that waits to be taken or refused.

    address is the peer's address text. open() makes the Carrier (and may
    raise); refuse() ends the connection without one.
    """

    address: str
    open: Callable[[], Carrier]
    refuse: Callable[[], None]


# accept(timeout_s): the next Incoming, or None when none came in timeout_s.
Accept = Callable[[float], "Incoming | None"]


def incoming_carrier(address: str, carrier: Carrier) -> Incoming:
    """An Incoming for a Carrier that already exists; refusing it closes it."""
    return Incoming(address, lambda: carrier, carrier.close)


class _Client:
    """One local client: its socket, its role and its bounded outbound queue."""

    def __init__(
        self,
        number: int,
        sock: socket.socket,
        send_timeout_s: float,
        queue_messages: int,
        queue_bytes: int,
    ) -> None:
        self.number = number
        self.sock = sock
        self.role: str | None = None  # None until its LOCAL_HELLO was taken
        self.name = ""
        self.why = ""  # why it was dropped
        sock.settimeout(send_timeout_s)
        self._cond = threading.Condition()
        self._queue = RecordQueue(queue_messages, queue_bytes)
        self._closed = False
        self._finishing = False
        self.writer = threading.Thread(
            target=self._write_loop, name=f"link-service-out-{number}", daemon=True
        )

    @property
    def label(self) -> str:
        who = f"{self.role} client" if self.role else "client without a role"
        return f"{who} {self.number}" + (f" ({self.name})" if self.name else "")

    @property
    def closed(self) -> bool:
        with self._cond:
            return self._closed

    def offer(self, msg_type: int, payload: bytes) -> None:
        """Queue a record for the client; a client whose queue is full is dropped."""
        with self._cond:
            if self._closed or self._finishing:
                return
            if self._queue.add(msg_type, payload):
                self._cond.notify()
                return
        self.kill("its outbound queue is full")

    def tell(self, message: Message) -> None:
        self.offer(message.TYPE, message.to_payload())

    def finish(self) -> None:
        """Send what is queued, then close."""
        with self._cond:
            self._finishing = True
            self._cond.notify()

    def kill(self, why: str) -> None:
        """Close at once; what is queued is dropped. The first reason is kept."""
        with self._cond:
            if self._closed:
                return
            self._closed = True
            self.why = why
            self._queue.clear()
            self._cond.notify()
        with contextlib.suppress(OSError):
            self.sock.shutdown(socket.SHUT_RDWR)

    def _write_loop(self) -> None:
        while True:
            with self._cond:
                self._cond.wait_for(
                    lambda: self._closed or self._finishing or len(self._queue)
                )
                if self._closed:
                    return
                record = self._queue.take()
            if record is None:  # finishing, and everything is out
                break
            try:
                self.sock.sendall(pack_record(*record))
            except TimeoutError:
                self.kill("it does not read its socket")
                return
            except (OSError, ProtocolError) as error:
                self.kill(f"sending to it failed: {error}")
                return
        self.kill("closed by the service")


@dataclass
class _Sent:
    """A forwarded command: who sent it, and whether its final RESULT came."""

    client: int | None  # the client's number; None for the service's own
    final: bool = False


class RobotLinkService:
    """Holds the workstation end of the robot connection and serves local clients.

    accept(timeout_s) gives the next robot connection as an Incoming, or None.
    The Carriers it opens must be safe to use from several threads. start()
    binds the local socket; close() ends everything and does not stop the
    robot: the Thor holds when the robot connection is lost.
    """

    def __init__(
        self,
        accept: Accept,
        *,
        socket_path: str | None = None,
        allowed_peers: Iterable[str] = ("127.0.0.1",),
        max_clients: int = 16,
        hello_timeout_s: float = 2.0,
        send_timeout_s: float = SEND_TIMEOUT_S,
        queue_messages: int = 256,
        queue_bytes: int = 8 * 1024 * 1024,
        status_period_s: float = 1.0,
        poll_s: float = 0.05,
    ) -> None:
        self._accept = accept
        self._socket_path = socket_path
        self._allowed = frozenset(peer_host(peer) for peer in allowed_peers)
        self._max_clients = int(max_clients)
        self._hello_timeout_s = float(hello_timeout_s)
        self._send_timeout_s = float(send_timeout_s)
        self._queue_messages = int(queue_messages)
        self._queue_bytes = int(queue_bytes)
        self._status_period_s = float(status_period_s)
        self._poll_s = float(poll_s)

        self._lock = threading.Lock()
        self._closed = threading.Event()
        self._started = False
        self._listener: LocalListener | None = None
        self._threads: list[threading.Thread] = []
        self._lines = LineLimit(time)
        self._ids = itertools.count(1)
        self._id_prefix = f"link-service-{uuid.uuid4().hex[:8]}"
        self._numbers = itertools.count(1)
        self._turn_away_helpers = threading.BoundedSemaphore(TURN_AWAY_HELPERS)

        # Under the lock, which is never held across a send to the Thor.
        self._carrier: Carrier | None = None
        self._close_reason = ""
        self._welcome: Welcome | None = None
        self._welcome_payload: bytes | None = None
        self._state: RobotState | None = None
        self._state_payload: bytes | None = None
        # (session id, stop number) of the newest ROBOT_STATE of this connection.
        self._stop_key: tuple[str | None, int] = (None, -1)
        # Set by a STOP on its way out to the _stop_key of that moment; lifted
        # by a ROBOT_STATE with a higher stop number or another session id.
        self._gate: tuple[str | None, int] | None = None
        self._guests: dict[int, _Client] = {}  # before their LOCAL_HELLO
        self._clients: dict[int, _Client] = {}
        self._agent: _Client | None = None
        self._commands: OrderedDict[str, _Sent] = OrderedDict()
        # (session id, owner number) the last help request was sent for.
        self._helped: tuple[str, int] | None = None
        self._agent_left = False  # an agent client was attached and is gone
        self._last_status: bytes | None = None

    def __enter__(self) -> RobotLinkService:
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def start(self) -> None:
        """Bind the local socket and start serving.

        Raises LocalSocketInUse when a live service has the socket, and
        LocalSocketError when the socket cannot be made.
        """
        if self._started:
            raise RuntimeError("the robot link service was started before")
        self._listener = LocalListener(self._socket_path)
        self._started = True
        for name, target in (
            ("link-service-robot-accept", self._robot_accept_loop),
            ("link-service-robot", self._robot_loop),
            ("link-service-local-accept", self._local_accept_loop),
        ):
            thread = threading.Thread(target=target, name=name, daemon=True)
            self._threads.append(thread)
            thread.start()
        _log.info("the local API listens on %s", self._listener.path)

    def close(self) -> None:
        """End the robot connection and every local connection."""
        if self._closed.is_set():
            return
        self._closed.set()
        if self._listener is not None:
            self._listener.close()
        with self._lock:
            carrier, self._carrier = self._carrier, None
            clients = [*self._clients.values(), *self._guests.values()]
        if carrier is not None:
            _quietly(carrier.close)
        for client in clients:
            client.kill("the service is closing")
        for thread in list(self._threads):
            if thread is not threading.current_thread():
                thread.join(self._send_timeout_s + 2.0)

    # What the service knows

    @property
    def socket_path(self) -> str:
        """The local socket's path; the configured one before start()."""
        if self._listener is not None:
            return self._listener.path
        return self._socket_path or ""

    @property
    def robot_connected(self) -> bool:
        carrier = self._carrier
        return carrier is not None and carrier.alive

    @property
    def welcome(self) -> Welcome | None:
        """The newest WELCOME; kept when the robot connection is lost."""
        with self._lock:
            return self._welcome

    @property
    def robot_state(self) -> RobotState | None:
        """The newest ROBOT_STATE of the robot connection that is up, else None."""
        with self._lock:
            return self._state

    @property
    def clock_offset(self) -> tuple[int, bool] | None:
        """(Thor monotonic clock minus this machine's in ns, two-way), if known."""
        with self._lock:
            return self._clock_offset()

    def roles(self) -> dict[str, int]:
        """How many clients are attached in each role that has one."""
        with self._lock:
            return self._roles()

    def status(self) -> LocalStatus:
        """What a LOCAL_STATUS sent now says."""
        with self._lock:
            return self._status()

    # The robot connection

    def _robot_accept_loop(self) -> None:
        while not self._closed.is_set():
            began = time.monotonic()
            try:
                incoming = self._accept(self._poll_s)
            except Exception as error:
                self._note(
                    logging.ERROR, f"accepting a robot connection failed: {error}"
                )
                self._closed.wait(max(self._poll_s, 0.2))
                continue
            if incoming is None:
                # An accept that does not wait must not make this loop spin.
                if time.monotonic() - began < self._poll_s / 2:
                    self._closed.wait(self._poll_s)
                continue
            self._take_incoming(incoming)

    def _take_incoming(self, incoming: Incoming) -> None:
        address = printable(incoming.address, MAX_WORD_CHARS)
        if peer_host(incoming.address) not in self._allowed:
            _quietly(incoming.refuse)
            self._note(
                logging.WARNING,
                f"refused a robot connection from {address}: not an allowed address",
            )
            return
        old = self._carrier
        if old is not None and old.alive:
            _quietly(incoming.refuse)
            self._note(
                logging.WARNING,
                f"refused a second robot connection from {address}: one is alive",
            )
            return
        try:
            carrier = incoming.open()
        except Exception as error:
            self._note(
                logging.WARNING,
                f"the robot connection from {address} could not be opened: {error}",
            )
            return
        if old is not None:
            self._lost(old)
        with self._lock:
            if self._closed.is_set():
                _quietly(carrier.close)
                return
            self._carrier = carrier
            self._close_reason = ""
            self._status_changed()
        _log.info("robot connection from %s", address)

    def _robot_loop(self) -> None:
        status_at = time.monotonic()
        while not self._closed.is_set():
            carrier = self._carrier
            if carrier is None:
                self._closed.wait(self._poll_s)
            else:
                try:
                    received = carrier.receive(self._poll_s)
                except Exception as error:
                    self._note(logging.ERROR, f"the robot connection failed: {error}")
                    _quietly(carrier.close)
                    received = None
                if received is not None:
                    self._from_robot(carrier, *received)
                elif not carrier.alive:
                    self._lost(carrier)
            now = time.monotonic()
            if now - status_at >= self._status_period_s:
                status_at = now
                with self._lock:
                    self._status_changed()
                for kind, count in self._lines.ended():
                    _log.warning("%d more lines like: %s", count, kind)

    def _lost(self, carrier: Carrier) -> None:
        """Handle the loss of the robot connection, once per carrier."""
        with self._lock:
            if self._carrier is not carrier:
                return
            self._carrier = None
            reason = getattr(carrier, "close_reason", "") or LOST
            self._close_reason = printable(reason, MAX_REASON_CHARS)
            self._state = self._state_payload = None
            self._stop_key = (None, -1)
            consoles = self._consoles()
            for command_id, sent in self._commands.items():
                if sent.final:
                    continue
                payload = Result(command_id, "failed", LOST).to_payload()
                for client in self._with_sender(consoles, sent.client):
                    client.offer(Result.TYPE, payload)
            self._commands.clear()
            self._status_changed()
        _quietly(carrier.close)
        _log.warning("%s: %s", LOST, self._close_reason)

    def _from_robot(self, carrier: Carrier, msg_type: int, payload: bytes) -> None:
        name = link_message_name(msg_type)
        if msg_type not in _FROM_THOR:
            self._note(logging.WARNING, f"dropped a {name} from the Thor")
            return
        try:
            message = decode(msg_type, payload)
        except ProtocolError as error:
            problem = printable(error, MAX_REASON_CHARS)
            self._note(logging.WARNING, f"dropped a {name} from the Thor: {problem}")
            return
        help_request = None
        with self._lock:
            if carrier is not self._carrier:
                return
            if isinstance(message, Welcome):
                self._welcome, self._welcome_payload = message, payload
                self._to_all(msg_type, payload)
                self._status_changed()
            elif isinstance(message, RobotState):
                self._state, self._state_payload = message, payload
                self._stop_key = (message.session_id, message.stop_number)
                self._lift_gate()
                self._to_all(msg_type, payload)
                help_request = self._agent_gone(once=True)
            elif isinstance(message, Result):
                self._route_result(message, payload)
            else:
                self._to_all(msg_type, payload)
        if help_request is not None:
            self._send(carrier, OwnerRequest.TYPE, help_request)

    def _route_result(self, result: Result, payload: bytes) -> None:
        """RESULT: to the client that sent the command and to every console."""
        sent = self._commands.get(result.command_id)
        if sent is not None and result.status != "accepted":
            sent.final = True
        sender = None if sent is None else sent.client
        for client in self._with_sender(self._consoles(), sender):
            client.offer(Result.TYPE, payload)

    def _agent_gone(self, once: bool) -> bytes | None:
        """The help request to send when the agent owns the arm with no agent client.

        once: only after an agent client left, for an ownership that shows late,
        and not again for the owner number a help request was sent for. Called
        with the lock held; the caller sends the payload after releasing it.
        """
        carrier, state = self._carrier, self._state
        if self._agent is not None or carrier is None or not carrier.alive:
            return None
        if state is None or state.owner != ROLE_AGENT:
            return None
        key = (state.session_id, state.owner_number)
        if once and (key == self._helped or not self._agent_left):
            return None
        self._helped = key
        request = OwnerRequest(
            command_id=f"{self._id_prefix}-{next(self._ids)}",
            session_id=state.session_id,
            owner_number=state.owner_number,
            stop_number=state.stop_number,
            request="help_needed",
            reason=AGENT_GONE,
        )
        self._remember(request.command_id, None)
        _log.warning(
            "the agent owns the arm and no agent client is attached: asking for "
            "help (%s)",
            AGENT_GONE,
        )
        return request.to_payload()

    def _send(self, carrier: Carrier, msg_type: int, payload: bytes) -> bool:
        try:
            if is_reliable(msg_type):
                carrier.send_reliable(msg_type, payload)
            else:
                carrier.send_latest(msg_type, payload)
        except Exception as error:
            name = link_message_name(msg_type)
            self._note(logging.ERROR, f"sending a {name} to the Thor failed: {error}")
            return False
        return True

    # Local clients

    def _local_accept_loop(self) -> None:
        listener = self._listener
        while not self._closed.is_set():
            sock = listener.accept(self._poll_s)
            if sock is None:
                continue
            self._threads = [t for t in self._threads if t.is_alive()]
            client = _Client(
                next(self._numbers),
                sock,
                self._send_timeout_s,
                self._queue_messages,
                self._queue_bytes,
            )
            with self._lock:
                full = len(self._clients) + len(self._guests) >= self._max_clients
                taken = not full and not self._closed.is_set()
                if taken:
                    self._guests[client.number] = client
            if not taken:
                self._note(
                    logging.WARNING, "refused a local client: too many are attached"
                )
                if not self._turn_away_helpers.acquire(blocking=False):
                    self._turn_away(client)
                    continue
                helper = threading.Thread(
                    target=self._turn_away,
                    args=(client, self._turn_away_helpers),
                    name=f"link-service-away-{client.number}",
                    daemon=True,
                )
                self._threads.append(helper)
                helper.start()
                continue
            reader = threading.Thread(
                target=self._serve,
                args=(client,),
                name=f"link-service-in-{client.number}",
                daemon=True,
            )
            self._threads.append(reader)
            client.writer.start()
            reader.start()

    def _turn_away(
        self, client: _Client, helper: threading.BoundedSemaphore | None = None
    ) -> None:
        """Close a client there is no room for, within about twice TURN_AWAY_S.

        A STOP it begins with, alone or right after its LOCAL_HELLO, still goes
        out. helper is the slot to give back when a helper thread runs this.
        """
        sock = client.sock
        try:
            by = time.monotonic() + TURN_AWAY_S
            msg_type, length = HEADER.unpack(read_exact(sock, HEADER.size, by))
            if msg_type == LocalHello.TYPE and length <= MAX_LOCAL_PAYLOAD_BYTES:
                read_exact(sock, length, by)
                msg_type, length = HEADER.unpack(read_exact(sock, HEADER.size, by))
            if msg_type == Stop.TYPE:
                self._stop_from(client, self._stop_payload(sock, length, by))
        except (Stalled, EOFError):
            pass
        except Exception:
            _log.exception("turning a local client away failed")
        finally:
            notice = LocalNotice(text="too many local clients are attached")
            with contextlib.suppress(OSError):
                sock.settimeout(TURN_AWAY_S)
                sock.sendall(pack_record(notice.TYPE, notice.to_payload()))
            sock.close()
            if helper is not None:
                helper.release()

    @staticmethod
    def _stop_payload(
        sock: socket.socket, length: int, by: float | None = None
    ) -> bytes | None:
        """The payload of a STOP whose header was read; None if it does not come.

        It has until `by` (time.monotonic()), else STOP_PAYLOAD_WAIT_S from now.
        """
        if length > SMALL_PAYLOAD_BYTES:
            return None
        if by is None:
            by = time.monotonic() + STOP_PAYLOAD_WAIT_S
        try:
            return read_exact(sock, length, by)
        except (Stalled, EOFError):
            return None

    def _serve(self, client: _Client) -> None:
        """Read one client's records until it closes or is dropped."""
        try:
            self._read_records(client)
        except EOFError:
            pass
        except Stalled:
            self._bad(client, "no LOCAL_HELLO in time")
        except Exception:
            _log.exception("serving a local client failed")
        finally:
            self._detach(client)
            client.finish()
            client.writer.join(self._send_timeout_s + 0.5)
            client.kill("closed")
            client.sock.close()

    def _read_records(self, client: _Client) -> None:
        hello_by = time.monotonic() + self._hello_timeout_s
        while True:
            deadline = hello_by if client.role is None else None
            header = read_exact(client.sock, HEADER.size, deadline)
            msg_type, length = HEADER.unpack(header)
            if msg_type == Stop.TYPE:
                # A STOP goes out whatever its payload and whoever sent it.
                payload = self._stop_payload(client.sock, length)
                self._stop_from(client, payload)
                if payload is None:
                    self._bad(client, "a STOP whose payload did not arrive whole")
                    return
                continue
            name = link_message_name(msg_type)
            if client.role is None:
                limit = MAX_LOCAL_PAYLOAD_BYTES
            elif msg_type == Move.TYPE:
                limit = MAX_PAYLOAD_BYTES
            else:
                limit = SMALL_PAYLOAD_BYTES
            if length > limit:
                self._bad(client, f"a {name} of {length} bytes is over the limit")
                return
            payload = read_exact(client.sock, length, deadline)
            if client.role is None:
                if msg_type != LocalHello.TYPE:
                    self._bad(client, f"sent a {name} before its LOCAL_HELLO")
                    return
                if not self._hello(client, payload):
                    return
            elif not self._from_client(client, msg_type, payload):
                return

    def _hello(self, client: _Client, payload: bytes) -> bool:
        """Take a client's LOCAL_HELLO; False when the client is refused."""
        try:
            hello = decode_local(LocalHello.TYPE, payload)
        except ProtocolError as error:
            self._bad(client, f"its LOCAL_HELLO could not be read: {error}")
            return False
        if hello.role not in ROLES:
            role = printable(hello.role, MAX_WORD_CHARS)
            self._bad(
                client, f"unknown role {role!r}; the roles are {', '.join(ROLES)}"
            )
            return False
        if hello.protocol_version != PROTOCOL_VERSION:
            self._bad(
                client,
                f"protocol version {hello.protocol_version}; this service speaks "
                f"version {PROTOCOL_VERSION}",
            )
            return False
        with self._lock:
            taken = hello.role == ROLE_AGENT and self._agent is not None
            if not taken and not client.closed:
                client.role = hello.role
                client.name = printable(hello.name, MAX_WORD_CHARS)
                del self._guests[client.number]
                self._clients[client.number] = client
                if client.role == ROLE_AGENT:
                    self._agent = client
                self._status_changed()
                if self._carrier is not None:
                    if self._welcome_payload is not None:
                        client.offer(Welcome.TYPE, self._welcome_payload)
                    if self._state_payload is not None:
                        client.offer(RobotState.TYPE, self._state_payload)
        if taken:
            self._bad(client, "an agent client is attached already; one at a time")
            return False
        _log.info("attached: %s", client.label)
        return True

    def _detach(self, client: _Client) -> None:
        help_request = None
        with self._lock:
            self._guests.pop(client.number, None)
            attached = self._clients.pop(client.number, None) is not None
            if attached:
                if self._agent is client:
                    self._agent = None
                    self._agent_left = True
                    help_request = self._agent_gone(once=False)
                self._status_changed()
            carrier = self._carrier
        if attached:
            _log.info(
                "detached: %s%s", client.label, f": {client.why}" * bool(client.why)
            )
        if help_request is not None and carrier is not None:
            self._send(carrier, OwnerRequest.TYPE, help_request)

    def _bad(self, client: _Client, problem: str) -> None:
        """Tell a client why it is dropped; the caller then stops serving it."""
        problem = printable(problem, MAX_REASON_CHARS)
        client.tell(LocalNotice(text=problem))
        self._note(
            logging.WARNING, f"dropped a local client ({client.label}): {problem}"
        )

    def _stop_from(self, client: _Client, payload: bytes | None) -> None:
        """Forward a client's STOP at once; it waits behind no send to the Thor.

        The gate is set before the STOP goes out, so a PERMIT, MOVE or GRIPPER
        checked from then on stays here. One that is past its check when the
        STOP arrives can follow it to the Thor: the Thor's stop-number check is
        the guarantee for that one. A STOP whose send fails leaves the gate set.
        """
        out = payload
        if out is not None:
            try:
                decode(Stop.TYPE, out)
            except ProtocolError:
                out = None
        if out is None:
            sender = client.role or "unknown"
            out = Stop(reason=BAD_STOP, sender=sender).to_payload()
        with self._lock:
            carrier = self._carrier
            up = carrier is not None and carrier.alive
            # A STOP sent before any ROBOT_STATE does not replace a gate that
            # names a stop number.
            if up and (self._stop_key[0] is not None or self._gate is None):
                self._gate = self._stop_key
                self._status_changed()
        sent = False
        if up:
            try:
                carrier.send_reliable(Stop.TYPE, out)
                sent = True
            except Exception as error:
                self._note(logging.ERROR, f"sending a STOP to the Thor failed: {error}")
        if sent:
            _log.warning("STOP from the %s forwarded", client.label)
        else:
            client.tell(LocalNotice(text=STOP_UNDELIVERED, about_type=Stop.TYPE))
            _log.warning("STOP from the %s not delivered: %s", client.label, LOST)

    def _lift_gate(self) -> None:
        """End the gate once a ROBOT_STATE shows that the STOP took effect.

        A gate set before any ROBOT_STATE takes the first one's stop number
        instead: that ROBOT_STATE may be older than the STOP.
        """
        gate, state = self._gate, self._state
        if gate is None or state is None:
            return
        if gate[0] is None:
            self._gate = (state.session_id, state.stop_number)
        elif state.session_id != gate[0] or state.stop_number > gate[1]:
            self._gate = None
            self._status_changed()

    def _from_client(self, client: _Client, msg_type: int, payload: bytes) -> bool:
        """Forward or refuse one link message of a client; False drops the client."""
        name = link_message_name(msg_type)
        if is_local(msg_type) or msg_type not in MESSAGE_TYPES:
            self._bad(client, f"sent a {name}, which no client sends")
            return False
        role = client.role
        if msg_type not in ROLE_TABLE[role]:
            self._refuse(
                client, msg_type, payload, f"the {role} role may not send {name}"
            )
            return True
        try:
            message = decode(msg_type, payload)
        except ProtocolError as error:
            problem = printable(error, MAX_REASON_CHARS)
            self._refuse(
                client, msg_type, payload, f"the {name} is not valid: {problem}"
            )
            return True
        if isinstance(message, OwnerRequest):
            if message.request not in OWNER_REQUESTS[role]:
                request = printable(message.request, MAX_WORD_CHARS)
                self._refuse(
                    client,
                    msg_type,
                    payload,
                    f'the {role} role may not send {name} "{request}"',
                )
                return True
        command_id = None
        if msg_type in COMMAND_TYPES:
            command_id = message.command_id
            if not _usable_id(command_id):
                self._refuse(
                    client, msg_type, payload, f"the {name} has no usable command id"
                )
                return True
        with self._lock:
            carrier = self._carrier
            if carrier is None or not carrier.alive:
                # PERMIT and VIEW_STATUS have no answer; they are dropped.
                if command_id is not None:
                    client.tell(Result(command_id, "refused", LOST))
                return True
            # The last check before the send; see _stop_from.
            gate = self._gate
            if gate is not None and msg_type == Permit.TYPE:
                return True
            if gate is not None and msg_type in (Move.TYPE, Gripper.TYPE):
                if message.stop_number <= gate[1]:
                    reason = (
                        f"a STOP was sent after stop number {gate[1]}; the {name} "
                        f"quotes stop number {message.stop_number}"
                    )
                    client.tell(Result(command_id, "refused", reason))
                    return True
            if command_id is not None:
                reason = self._remember(command_id, client.number)
                if reason:
                    client.tell(Result(command_id, "refused", reason))
                    return True
        self._send(carrier, msg_type, payload)
        return True

    def _refuse(self, client: _Client, msg_type: int, payload: bytes, why: str) -> None:
        """Answer the sender alone: a RESULT "refused", or a LOCAL_NOTICE."""
        why = printable(why, MAX_REASON_CHARS)
        found = command_id_in(payload) if msg_type in COMMAND_TYPES else None
        if found is not None and found[1] and _usable_id(found[0]):
            client.tell(Result(found[0], "refused", why))
        else:
            client.tell(LocalNotice(text=why, about_type=msg_type))
        self._note(logging.INFO, f"refused ({client.label}): {why}")

    def _remember(self, command_id: str, client: int | None) -> str:
        """Note who sent a command; the reason when it may not be forwarded."""
        sent = self._commands.get(command_id)
        if sent is not None and not sent.final and sent.client != client:
            return (
                "command id in use: another sender's command with this id has no "
                "final RESULT yet"
            )
        self._commands.pop(command_id, None)
        if len(self._commands) >= COMMANDS_KEPT:
            done = next((k for k, sent in self._commands.items() if sent.final), None)
            if done is None:
                return "too many commands wait for their RESULT"
            del self._commands[done]
        self._commands[command_id] = _Sent(client)
        return ""

    # Shared pieces; the lock is held

    def _consoles(self) -> list[_Client]:
        return [c for c in self._clients.values() if c.role == ROLE_CONSOLE]

    def _with_sender(
        self, consoles: list[_Client], sender: int | None
    ) -> list[_Client]:
        """The consoles, and the sending client if it is attached and not one."""
        client = None if sender is None else self._clients.get(sender)
        if client is None or client in consoles:
            return consoles
        return [*consoles, client]

    def _to_all(self, msg_type: int, payload: bytes) -> None:
        for client in self._clients.values():
            client.offer(msg_type, payload)

    def _roles(self) -> dict[str, int]:
        roles: dict[str, int] = {}
        for client in self._clients.values():
            roles[client.role] = roles.get(client.role, 0) + 1
        return roles

    def _clock_offset(self) -> tuple[int, bool] | None:
        read = getattr(self._carrier, "clock_offset", None)
        if read is None:
            return None
        try:
            offset_ns, two_way, valid = read()
        except Exception as error:
            self._note(logging.WARNING, f"reading the clock offset failed: {error}")
            return None
        return (int(offset_ns), bool(two_way)) if valid else None

    def _status(self) -> LocalStatus:
        carrier = self._carrier
        up = carrier is not None and carrier.alive
        offset = self._clock_offset() if up else None
        return LocalStatus(
            robot_connected=up,
            close_reason="" if up else self._close_reason,
            session_id="" if self._welcome is None else self._welcome.session_id,
            clock_offset_ns=None if offset is None else offset[0],
            clock_two_way=offset is not None and offset[1],
            roles=self._roles(),
            stop_unconfirmed=self._gate is not None,
        )

    def _status_changed(self) -> None:
        """Send LOCAL_STATUS to every client, unless it says what the last one said."""
        payload = self._status().to_payload()
        if payload == self._last_status:
            return
        self._last_status = payload
        self._to_all(LocalStatus.TYPE, payload)

    def _note(self, level: int, text: str) -> None:
        """Log a line that a peer can cause; a repeated one is written once a second."""
        if self._lines.allow(text):
            _log.log(level, "%s", text)


def _usable_id(command_id: str) -> bool:
    """Whether a command id can be kept and shown as it is."""
    if not 0 < len(command_id) <= MAX_ID_CHARS:
        return False
    return printable(command_id, MAX_ID_CHARS) == command_id


def _quietly(call: Callable[[], object]) -> None:
    try:
        call()
    except Exception:
        _log.exception("closing a connection failed")
