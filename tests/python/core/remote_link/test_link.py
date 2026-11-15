# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Links over loopback, in one process.

Link.connect() dials a LinkListener and Link.adopt() takes what it accepts. Where a test
must hold back what a Link sends, or needs a peer that stays silent, a plain socket is the
peer.
"""

import functools
import gc
import os
import re
import signal
import socket
import struct
import subprocess
import sys
import threading
import time
from contextlib import ExitStack, contextmanager

import pytest

from isaaccapture.remote_link import (
    MAX_PAYLOAD_SIZE,
    Link,
    LinkListener,
    LinkPeerClosedError,
)

pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason="links need POSIX sockets"
)

# Upper bound on a wait that is expected to end sooner.
WAIT_MS = 10_000
WAIT_S = WAIT_MS / 1000
# How long a call blocks in the tests that show it releases the GIL.
BLOCK_MS = 1000
# No HEARTBEAT and no peer timeout within a test: only what the test sends travels.
QUIET = {"heartbeat_period_ms": 60_000, "peer_timeout_ms": 60_000}
TRANSPORTS = pytest.mark.parametrize("websocket", [False, True], ids=["tcp", "ws"])
# The header of a link message on the wire: magic, type, version, channel, seq, payload size, send time.
HEADER = struct.Struct("<IBBHIIq")
MAGIC = 0x30535452


def _endpoint(listener, websocket=False):
    return f"{'ws' if websocket else 'tcp'}://127.0.0.1:{listener.port}"


def _wait_for(predicate, timeout_s=WAIT_S):
    deadline = time.monotonic() + timeout_s
    while not predicate():
        if time.monotonic() > deadline:
            return False
        time.sleep(0.002)
    return True


def _wait_for_queued(link, count):
    return _wait_for(lambda: link.stats().received_queued == count)


def _wait_for_replaced(link, count):
    return _wait_for(lambda: link.stats().received_replaced == count)


def _wait_for_arrivals(link):
    return _wait_for(lambda: link.stats().received_queued > 0)


def _connect(listener, websocket=False, client=None, server=None):
    """(client, server): Link.connect() dials `listener` on a thread, this one accepts and adopts.

    Over ws:// connect() returns only once the adopted Link answered the upgrade, hence the thread.
    """
    dialled = []

    def dial():
        try:
            options = {**QUIET, **(client or {})}
            dialled.append(
                Link.connect(_endpoint(listener, websocket), WAIT_MS, **options)
            )
        except Exception as e:  # raised below, on the test's thread
            dialled.append(e)

    dialer = threading.Thread(target=dial)
    dialer.start()
    accepted = listener.accept(WAIT_MS)
    assert accepted is not None, "nothing connected"
    adopted = Link.adopt(
        accepted[0], websocket=websocket, **{**QUIET, **(server or {})}
    )
    dialer.join(2 * WAIT_S)
    assert not dialer.is_alive()
    if isinstance(dialled[0], Exception):
        adopted.close()
        raise dialled[0]
    return dialled[0], adopted


@contextmanager
def _pair(websocket=False, client=None, server=None):
    """A connected (client, server), closed on exit; `client` and `server` are keyword options."""
    with LinkListener("127.0.0.1", 0) as listener:
        client_link, server_link = _connect(listener, websocket, client, server)
    with client_link, server_link:
        yield client_link, server_link


@contextmanager
def _plain_connection():
    """(fd, sock): a descriptor for Link.adopt(), and the plain socket that dialled it.

    The socket is read only when the test reads it, and its receive buffer is small: a large
    link message stays in the Link's sender until then.
    """
    with LinkListener("127.0.0.1", 0) as listener, socket.socket() as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 16)
        sock.settimeout(WAIT_S)
        sock.connect(("127.0.0.1", listener.port))
        accepted = listener.accept(WAIT_MS)
        assert accepted is not None, "nothing connected"
        yield accepted[0], sock


@contextmanager
def _plain_peer(**options):
    """(link, sock): a Link whose peer is the plain socket of _plain_connection()."""
    with _plain_connection() as (fd, sock):
        with Link.adopt(fd, **{**QUIET, **options}) as link:
            yield link, sock


def _read_exactly(sock, size):
    data = bytearray()
    while len(data) < size:
        chunk = sock.recv(min(size - len(data), 1 << 20))
        if not chunk:
            break
        data += chunk
    return bytes(data)


def _read_link_message(sock):
    """(type, payload) of the next link message on a plain socket; None at the end of the stream."""
    header = _read_exactly(sock, HEADER.size)
    if not header:
        return None
    magic, number, _version, _channel, _seq, size, _sent = HEADER.unpack(header)
    assert magic == MAGIC
    return number, _read_exactly(sock, size)


def _receive(link):
    message = link.receive(WAIT_MS)
    assert message is not None, (
        f"nothing received: alive={link.alive}, {link.close_reason!r}"
    )
    return message


def _unread(link):
    """What waits in `link`'s inbox now, taken out of it, as (type, payload) pairs."""
    return [(m.type, m.payload) for m in iter(lambda: link.receive(0), None)]


def _pattern(size):
    """`size` bytes that repeat every 251, so a shifted or repeated block shows."""
    return (bytes(range(251)) * (size // 251 + 1))[:size]


@TRANSPORTS
def test_connect_and_adopt_exchange_no_link_message(websocket):
    with _pair(websocket) as (client, server):
        for link in (client, server):
            assert link.alive is True
            assert link.close_reason == ""
            assert link.peer_closed_before_any_message is False
        client.send_reliable(64, b"from the client")
        server.send_reliable(65, b"from the server")
        # The first thing each end receives is what the test sent: no HELLO went ahead of it.
        first = _receive(server)
        assert (first.type, first.payload) == (64, b"from the client")
        first = _receive(client)
        assert (first.type, first.payload) == (65, b"from the server")
        for link in (client, server):
            assert link.stats().messages_received == 1
            assert link.receive(0) is None


def test_connect_takes_host_and_port_without_a_scheme_and_no_time_limit():
    with LinkListener("127.0.0.1", 0) as listener:
        # Over TCP connect() returns once the kernel took the connection: no second thread is needed.
        with Link.connect(f"127.0.0.1:{listener.port}", -1, name="client") as client:
            fd, _peer = listener.accept(-1)
            with Link.adopt(fd, name="server") as server:
                client.send_reliable(64, b"x")
                assert _receive(server).payload == b"x"


@TRANSPORTS
def test_reliable_link_messages_arrive_in_order_with_their_bytes(websocket):
    payloads = [
        b"",
        b"\x00",
        _pattern(1 << 20),
        _pattern(MAX_PAYLOAD_SIZE),
        b"after the largest",
    ]
    payloads += [b"%d" % i for i in range(300)]
    with _pair(websocket) as (client, server):
        for sender, receiver in ((client, server), (server, client)):
            for index, payload in enumerate(payloads):
                sender.send_reliable(64 + index % 32, payload)
            for index, payload in enumerate(payloads):
                message = _receive(receiver)
                assert message.type == 64 + index % 32
                assert len(message.payload) == len(payload)
                # Not compared inside the assert: a 16 MiB difference would be printed.
                intact = message.payload == payload
                assert intact, f"link message {index} arrived with other bytes"
            assert receiver.receive(0) is None
        assert client.alive
        assert server.alive


@TRANSPORTS
def test_type_numbers_pass_both_ways_as_plain_ints(websocket):
    numbers = (0, 1, 12, 34, 64, 95, 96, 127, 255)
    outside_the_session_protocol = (0, 64, 95, 96, 127, 255)
    with _pair(websocket) as (client, server):
        for sender, receiver in ((client, server), (server, client)):
            for number in numbers:
                for send in (sender.send_reliable, sender.send_latest):
                    payload = b"%d by %s" % (number, send.__name__.encode())
                    send(number, payload)
                    message = _receive(receiver)
                    assert type(message.type) is int
                    assert (message.type, message.payload) == (number, payload)
            stats = receiver.stats()
            assert stats.messages_received == 2 * len(numbers)
            assert stats.unknown_received == 2 * len(outside_the_session_protocol)


def test_a_received_link_message_carries_both_clocks():
    with _pair() as (client, server):
        # One machine: the peer's clock is this machine's CLOCK_MONOTONIC too.
        before = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
        client.send_reliable(64, b"x")
        message = _receive(server)
        after = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
        assert before <= message.remote_send_ns <= message.local_receive_ns <= after


def test_send_latest_keeps_one_unsent_link_message_of_a_type_and_the_newest_wins():
    with _plain_peer() as (link, peer):
        # The peer is not reading, so this stays in the link's sender and all that follows stays queued.
        blocker = bytes(4 << 20)
        link.send_reliable(64, blocker)
        for i in range(200):
            link.send_latest(96, b"%d" % i)
            link.send_latest(97, b"x%d" % i)
        stats = link.stats()
        # The first of a type waits unsent, and every later one replaces it.
        assert stats.latest_replaced == 2 * 199
        assert stats.messages_sent == 0

        # Neither newest-wins link message passes the reliable one sent before it.
        number, payload = _read_link_message(peer)
        assert (number, len(payload)) == (64, len(blocker))
        intact = payload == blocker
        assert intact
        assert sorted(_read_link_message(peer) for _ in range(2)) == [
            (96, b"199"),
            (97, b"x199"),
        ]
        assert _wait_for(lambda: link.stats().messages_sent == 3)
        link.close()
        assert _read_link_message(peer) is None
        assert link.stats().latest_replaced == 2 * 199


def test_a_newest_wins_link_message_never_passes_a_reliable_one_sent_before_it():
    count = 200
    with _pair() as (client, server):
        for i in range(count):
            client.send_reliable(64, b"%d" % i)
            client.send_latest(96, b"%d" % i)
        reliable_seen = -1
        latest_seen = -1
        while (reliable_seen, latest_seen) != (count - 1, count - 1):
            message = _receive(server)
            index = int(message.payload)
            if message.type == 64:
                assert index == reliable_seen + 1
                reliable_seen = index
            else:
                # Newer than the last one, and its reliable companion is already here.
                assert latest_seen < index <= reliable_seen
                latest_seen = index
        assert server.receive(0) is None


def test_latest_types_keep_only_the_newest_unread_link_message_of_each_type():
    count = 200
    with _pair(server={"latest_types": range(96, 128)}) as (client, server):
        client.send_reliable(64, b"first")
        for i in range(count):
            client.send_latest(96, b"%d" % i)
            client.send_latest(127, b"x%d" % i)
        client.send_reliable(65, b"last")

        # Every send_latest() but the last of its type is replaced: in the sender while unsent, or in the
        # inbox of the receiver, which is not reading.
        def settled():
            replaced = client.stats().latest_replaced + server.stats().received_replaced
            return replaced == 2 * (count - 1) and server.stats().received_queued == 4

        assert _wait_for(settled)
        unread = _unread(server)
        assert unread[0] == (64, b"first")
        assert sorted(unread[1:]) == [
            (65, b"last"),
            (96, b"%d" % (count - 1)),
            (127, b"x%d" % (count - 1)),
        ]
        assert server.stats().received_queued == 0


def test_without_latest_types_the_inbox_keeps_every_link_message():
    with _pair() as (client, server):
        for i in range(3):
            client.send_latest(96, b"%d" % i)
            # It arrives before the next one is sent, so the sender replaces nothing either.
            assert _wait_for_queued(server, i + 1)
        assert (client.stats().latest_replaced, server.stats().received_replaced) == (
            0,
            0,
        )
        assert _unread(server) == [(96, b"0"), (96, b"1"), (96, b"2")]


def test_a_newer_link_message_of_a_latest_type_takes_the_newest_place_in_the_inbox():
    with _pair(server={"latest_types": [96]}) as (client, server):
        client.send_latest(96, b"older")
        assert _wait_for_queued(server, 1)
        client.send_reliable(64, b"between")
        assert _wait_for_queued(server, 2)
        client.send_latest(96, b"newer")
        assert _wait_for_replaced(server, 1)
        assert server.stats().received_queued == 2
        assert _unread(server) == [(64, b"between"), (96, b"newer")]


def test_stats_count_what_travelled():
    with _pair() as (client, server):
        client.send_reliable(64, b"a")  # outside the session protocol's numbers
        client.send_reliable(34, b"b")  # one of them
        client.send_latest(96, b"c")
        assert _wait_for_queued(server, 3)
        assert _wait_for(lambda: client.stats().messages_sent == 3)
        sent = client.stats()
        assert (sent.messages_sent, sent.messages_received) == (3, 0)
        assert (sent.latest_replaced, sent.received_queued) == (0, 0)
        received = server.stats()
        assert (received.messages_sent, received.messages_received) == (0, 3)
        assert (received.unknown_received, received.received_replaced) == (2, 0)
        assert received.received_queued == 3
        assert server.receive(0) is not None
        assert server.stats().received_queued == 2
        assert server.stats().messages_received == 3


def test_receive_with_timeout_0_does_not_wait():
    with _pair() as (client, server):
        waits = []
        for _ in range(5):
            start = time.monotonic()
            assert server.receive(0) is None
            waits.append(time.monotonic() - start)
        # The quickest of five, so that a busy machine does not count: one wait slice is 100 ms.
        assert min(waits) < 0.05
        client.send_reliable(64, b"x")
        assert _wait_for_queued(server, 1)
        assert server.receive(0).payload == b"x"


def test_receive_waits_out_a_positive_timeout():
    with _pair() as (_client, server):
        start = time.monotonic()
        assert server.receive(250) is None
        assert time.monotonic() - start >= 0.249


@pytest.mark.parametrize("timeout_ms", [WAIT_MS, -1], ids=["bounded", "no limit"])
def test_receive_returns_as_soon_as_a_link_message_comes(timeout_ms):
    with _pair() as (client, server):
        start = time.monotonic()
        sender = threading.Timer(0.3, client.send_reliable, (64, b"late"))
        sender.start()
        message = server.receive(timeout_ms)
        waited = time.monotonic() - start
        sender.join()
        assert message is not None
        assert message.payload == b"late"
        # It waited for the link message, and no longer than for it.
        assert 0.299 <= waited < WAIT_S / 2


def _assert_releases_the_gil(call):
    """`call` blocks another thread for BLOCK_MS while this one keeps running."""

    def target():
        try:
            call()
        except (RuntimeError, TimeoutError):
            pass

    thread = threading.Thread(target=target)
    # Each step of the loop needs the GIL (start() included), so a call that holds it shows up as one long gap.
    longest = 0.0
    start = last = time.monotonic()
    thread.start()
    while thread.is_alive():
        time.sleep(0.01)
        now = time.monotonic()
        longest = max(longest, now - last)
        last = now
    thread.join()
    assert last - start >= BLOCK_MS / 1000 - 0.001, "the call did not block"
    assert longest < BLOCK_MS / 2000


@contextmanager
def _unanswered_endpoint():
    """A loopback endpoint that never answers a TCP connect, like an unreachable host.

    Its listener's accept queue is full, so the kernel drops further connection attempts.
    """
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen(0)
        port = listener.getsockname()[1]
        fillers = []
        try:
            while True:
                if len(fillers) == 16:
                    pytest.skip("the kernel answers connects to a full accept queue")
                filler = socket.socket()
                fillers.append(filler)
                filler.settimeout(0.2)
                try:
                    filler.connect(("127.0.0.1", port))
                except TimeoutError:
                    break
            yield f"tcp://127.0.0.1:{port}"
        finally:
            for filler in fillers:
                filler.close()


@contextmanager
def _silent_listener():
    """The port of a listener that never accepts: a connection waits in its backlog, unanswered."""
    with socket.socket() as silent:
        silent.bind(("127.0.0.1", 0))
        silent.listen(1)
        yield silent.getsockname()[1]


def test_accept_releases_the_gil():
    with LinkListener("127.0.0.1", 0) as listener:
        _assert_releases_the_gil(lambda: listener.accept(BLOCK_MS))


def test_receive_releases_the_gil():
    with _pair() as (_client, server):
        _assert_releases_the_gil(lambda: server.receive(BLOCK_MS))


def test_a_tcp_connect_releases_the_gil():
    with _unanswered_endpoint() as endpoint:
        _assert_releases_the_gil(lambda: Link.connect(endpoint, BLOCK_MS))


def test_the_wait_for_a_websocket_upgrade_releases_the_gil():
    with _silent_listener() as port:
        _assert_releases_the_gil(
            lambda: Link.connect(f"ws://127.0.0.1:{port}", BLOCK_MS)
        )


@pytest.mark.parametrize("how", ["close", "__exit__", "the last reference going"])
def test_closing_a_link_releases_the_gil(how):
    with _plain_connection() as (fd, _peer):
        # The peer is not reading: closing gives this link message close_drain_ms to go out.
        links = [Link.adopt(fd, **QUIET, close_drain_ms=BLOCK_MS)]
        links[0].send_reliable(64, bytes(4 << 20))
        if how == "close":
            _assert_releases_the_gil(links[0].close)
        elif how == "__exit__":
            _assert_releases_the_gil(lambda: links[0].__exit__(None, None, None))
        else:
            _assert_releases_the_gil(links.clear)


def test_close_with_linger_releases_the_gil():
    with _plain_peer() as (link, _peer):
        # The peer keeps its end open, so close() waits out the linger.
        _assert_releases_the_gil(lambda: link.close(BLOCK_MS))


def test_link_messages_travel_while_python_holds_the_gil():
    """The link's own threads send and queue link messages; neither needs the GIL."""
    with _pair() as (client, server):
        interval = sys.getswitchinterval()
        # No other Python thread gets the GIL from this one until it blocks or the interval is back.
        sys.setswitchinterval(1000)
        try:
            client.send_reliable(64, b"x")
            hold_until = time.monotonic() + 0.3
            while time.monotonic() < hold_until:
                pass
            held_until_ns = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
            message = server.receive(0)
        finally:
            sys.setswitchinterval(interval)
        assert message is not None
        assert message.local_receive_ns < held_until_ns


@contextmanager
def _sigint_after(delay_s, then=None):
    """Sends this process SIGINT after `delay_s` and then calls `then`; yields a list that gets the send time."""
    sent = []

    def interrupt():
        sent.append(time.monotonic())
        os.kill(os.getpid(), signal.SIGINT)
        if then is not None:
            then()

    previous = signal.signal(signal.SIGINT, signal.default_int_handler)
    timer = threading.Timer(delay_s, interrupt)
    timer.start()
    try:
        yield sent
    finally:
        timer.cancel()
        timer.join()
        signal.signal(signal.SIGINT, previous)


@pytest.mark.parametrize("timeout_ms", [WAIT_MS, -1], ids=["bounded", "no limit"])
def test_ctrl_c_interrupts_receive(timeout_ms):
    with _pair() as (client, server):
        with _sigint_after(0.3) as sent, pytest.raises(KeyboardInterrupt):
            server.receive(timeout_ms)
        assert time.monotonic() - sent[0] < 0.5
        # The link is as it was.
        client.send_reliable(64, b"after")
        assert _receive(server).payload == b"after"


def test_ctrl_c_that_comes_with_a_link_message_leaves_it_in_the_inbox():
    with _pair() as (client, server):
        # The link message ends the wait when the signal is already pending: receive() has it in hand
        # when KeyboardInterrupt is raised, and the caller never gets what receive() returns.
        send = functools.partial(client.send_reliable, 64, b"kept")
        with _sigint_after(0.3, then=send), pytest.raises(KeyboardInterrupt):
            server.receive(WAIT_MS)
        assert _receive(server).payload == b"kept"
        assert server.receive(0) is None


@pytest.mark.parametrize("timeout_ms", [WAIT_MS, -1], ids=["bounded", "no limit"])
def test_ctrl_c_interrupts_accept(timeout_ms):
    with LinkListener("127.0.0.1", 0) as listener:
        with _sigint_after(0.3) as sent, pytest.raises(KeyboardInterrupt):
            listener.accept(timeout_ms)
        assert time.monotonic() - sent[0] < 0.5
        # The listener is as it was.
        with socket.create_connection(("127.0.0.1", listener.port)):
            fd, _peer = listener.accept(WAIT_MS)
            os.close(fd)


def test_ctrl_c_during_a_tcp_connect_raises_keyboard_interrupt_alone():
    with _unanswered_endpoint() as endpoint:
        with _sigint_after(0.5) as sent, pytest.raises(KeyboardInterrupt) as raised:
            Link.connect(endpoint, 1000)
        # It replaces the connect error instead of interrupting its handling.
        assert raised.value.__context__ is None
        # The signal does not restart the attempt's wait: the attempt ends with timeout_ms.
        assert time.monotonic() - sent[0] < 1.0


def test_ctrl_c_interrupts_an_unbounded_tcp_connect():
    with _unanswered_endpoint() as endpoint:
        with _sigint_after(0.3) as sent, pytest.raises(KeyboardInterrupt):
            Link.connect(endpoint, -1)
        # Within one connect attempt (2 s).
        assert time.monotonic() - sent[0] < 2.5


def test_ctrl_c_interrupts_an_unbounded_wait_for_a_websocket_upgrade():
    with _silent_listener() as port:
        with _sigint_after(0.3) as sent, pytest.raises(KeyboardInterrupt):
            Link.connect(f"ws://127.0.0.1:{port}", -1)
        assert time.monotonic() - sent[0] < 0.5


def test_a_refused_connect_raises_runtime_error_without_waiting():
    # Bound but not listening: the kernel refuses the connect.
    with socket.socket() as bound:
        bound.bind(("127.0.0.1", 0))
        port = bound.getsockname()[1]
        start = time.monotonic()
        with pytest.raises(
            RuntimeError, match=rf"^cannot connect to 127\.0\.0\.1:{port}: "
        ) as raised:
            Link.connect(f"tcp://127.0.0.1:{port}", WAIT_MS)
        assert time.monotonic() - start < WAIT_S / 2
    assert type(raised.value) is RuntimeError


def test_a_tcp_connect_nothing_answers_raises_timeout_error_after_timeout_ms():
    with _unanswered_endpoint() as endpoint:
        start = time.monotonic()
        # Longer than one connect attempt (2 s), so a second one runs.
        with pytest.raises(
            TimeoutError,
            match=r"^cannot connect to 127\.0\.0\.1:\d+: connect timed out$",
        ):
            Link.connect(endpoint, 2500)
        assert 2.499 <= time.monotonic() - start < 2.5 + WAIT_S / 2


@contextmanager
def _closing_listener():
    """A listener that closes every connection unanswered once the peer sent something; yields its port."""
    stop = threading.Event()
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(4)
    listener.settimeout(0.05)

    def serve():
        while not stop.is_set():
            try:
                connection, _ = listener.accept()
            except TimeoutError:
                continue
            with connection:
                connection.settimeout(WAIT_S)
                try:
                    connection.recv(4096)
                except OSError:
                    pass

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    try:
        yield listener.getsockname()[1]
    finally:
        stop.set()
        thread.join(timeout=WAIT_S)
        listener.close()


def test_a_websocket_peer_that_closes_unanswered_raises_link_peer_closed_error():
    with _closing_listener() as port:
        with pytest.raises(LinkPeerClosedError) as raised:
            Link.connect(f"ws://127.0.0.1:{port}", WAIT_MS)
        assert isinstance(raised.value, RuntimeError)
        assert str(raised.value) == (
            f"WebSocket upgrade to 127.0.0.1:{port}/: connection closed before the answer"
        )


def test_a_websocket_upgrade_nothing_answers_raises_timeout_error_after_timeout_ms():
    with _silent_listener() as port:
        start = time.monotonic()
        with pytest.raises(
            TimeoutError,
            match=rf"^cannot connect to 127\.0\.0\.1:{port}: no answer to the WebSocket upgrade$",
        ):
            Link.connect(f"ws://127.0.0.1:{port}", 400)
        assert 0.399 <= time.monotonic() - start < 0.4 + WAIT_S / 2


def test_a_websocket_connect_to_a_tcp_link_raises_runtime_error():
    with LinkListener("127.0.0.1", 0) as listener:
        adopted = []

        def adopt():
            accepted = listener.accept(WAIT_MS)
            if accepted is not None:
                adopted.append(Link.adopt(accepted[0], **QUIET))

        adopter = threading.Thread(target=adopt)
        adopter.start()
        with pytest.raises(
            RuntimeError, match=r"^WebSocket upgrade to 127\.0\.0\.1:\d+/ refused: "
        ) as raised:
            Link.connect(_endpoint(listener, websocket=True), WAIT_MS)
        adopter.join(WAIT_S)
        assert type(raised.value) is RuntimeError
        with adopted[0] as link:
            assert _wait_for(lambda: not link.alive)
            assert link.close_reason == "receive failed: malformed data"


@TRANSPORTS
def test_the_peer_closing_ends_the_link_once_what_it_sent_is_handed_out(websocket):
    with _pair(websocket, client={"close_drain_ms": WAIT_MS}) as (client, server):
        client.send_reliable(64, b"one")
        client.send_reliable(65, b"two")
        client.close(WAIT_MS)
        assert client.alive is False
        assert client.close_reason == "closed"
        assert client.peer_closed_before_any_message is False

        assert _wait_for(lambda: not server.alive)
        assert server.close_reason == "receive failed: connection closed"
        assert server.peer_closed_before_any_message is False
        # What arrived before the death is handed out, whatever the timeout.
        assert server.receive(0).payload == b"one"
        assert server.receive(-1).payload == b"two"
        # Then nothing. Without a limit receive() returns too; a positive timeout is waited out.
        assert server.receive(0) is None
        assert server.receive(-1) is None
        start = time.monotonic()
        assert server.receive(200) is None
        assert time.monotonic() - start >= 0.199

        # Sends on a dead link do nothing and raise nothing.
        sent = server.stats().messages_sent
        server.send_reliable(64, b"to nobody")
        server.send_latest(96, b"to nobody")
        assert server.stats().messages_sent == sent
        assert server.stats().latest_replaced == 0
        server.close()
        # The first reason stays.
        assert server.close_reason == "receive failed: connection closed"


def test_a_peer_that_closes_before_any_link_message():
    # Seen from the end that adopted the connection ...
    with _plain_peer() as (link, peer):
        peer.close()
        assert _wait_for(lambda: not link.alive)
        assert link.close_reason == "receive failed: connection closed"
        assert link.peer_closed_before_any_message is True
        assert link.receive(-1) is None
    # ... and from the end that dialled.
    with LinkListener("127.0.0.1", 0) as listener:
        with Link.connect(_endpoint(listener), WAIT_MS, **QUIET) as link:
            fd, _peer = listener.accept(WAIT_MS)
            os.close(fd)
            assert _wait_for(lambda: not link.alive)
            assert link.close_reason == "receive failed: connection closed"
            assert link.peer_closed_before_any_message is True


def test_a_peer_that_goes_silent_ends_the_link_after_peer_timeout_ms():
    # The client's quiet options send no HEARTBEAT: after its one link message it is silent.
    with _pair(server={"peer_timeout_ms": 300}) as (client, server):
        start = time.monotonic()
        client.send_reliable(64, b"before the silence")
        assert _wait_for(lambda: not server.alive)
        assert time.monotonic() - start >= 0.3
        silence = re.fullmatch(r"no messages for (\d+) ms", server.close_reason)
        assert silence is not None, server.close_reason
        assert int(silence[1]) >= 300
        assert server.peer_closed_before_any_message is False
        assert server.receive(-1).payload == b"before the silence"
        assert server.receive(-1) is None
        # The peer learns of it: the dead link closed the connection.
        assert _wait_for(lambda: not client.alive)


def test_a_peer_that_never_sends_ends_the_link_after_peer_timeout_ms():
    with _plain_connection() as (fd, _peer):
        start = time.monotonic()
        with Link.adopt(fd, peer_timeout_ms=300) as link:
            assert _wait_for(lambda: not link.alive)
            assert time.monotonic() - start >= 0.3
            assert re.fullmatch(r"no messages for \d+ ms", link.close_reason)
            # Silence is not a connection that broke.
            assert link.peer_closed_before_any_message is False


def test_peer_timeout_from_first_message_starts_the_peer_timeout_with_it():
    options = {"peer_timeout_ms": 200, "peer_timeout_from_first_message": True}
    with _plain_connection() as (fd, peer), Link.adopt(fd, **options) as link:
        # Three peer timeouts of silence do not end it.
        assert not _wait_for(lambda: not link.alive, timeout_s=0.6)
        # A link message made by hand: its send time is whatever the peer says.
        peer.sendall(HEADER.pack(MAGIC, 64, 0, 0, 0, 5, 123_456_789) + b"first")
        message = _receive(link)
        assert (message.type, message.payload) == (64, b"first")
        assert message.remote_send_ns == 123_456_789
        # From here on the peer timeout applies.
        assert _wait_for(lambda: not link.alive)
        assert re.fullmatch(r"no messages for \d+ ms", link.close_reason)


def test_unsent_reliable_bytes_above_max_queued_reliable_bytes_end_the_link():
    with _pair(client={"max_queued_reliable_bytes": 1000}) as (client, server):
        client.send_reliable(64, bytes(900))
        assert len(_receive(server).payload) == 900
        # With its header this one is above the limit; the send raises nothing.
        client.send_reliable(64, bytes(1000))
        assert client.alive is False
        assert client.close_reason == (
            "more than 1000 bytes of reliable messages unsent: the peer is not reading"
        )
        assert client.peer_closed_before_any_message is False
        assert _wait_for(lambda: not server.alive)
        assert server.receive(0) is None


@pytest.mark.parametrize("timeout_ms", [-1, 700], ids=["no limit", "bounded"])
def test_close_during_a_receive_on_another_thread(timeout_ms):
    with _pair() as (_client, server):
        outcome = []

        def receive():
            start = time.monotonic()
            outcome.append((server.receive(timeout_ms), time.monotonic() - start))

        receiver = threading.Thread(target=receive)
        receiver.start()
        # receive() is waiting by now; one that starts after the close ends the same way.
        time.sleep(0.1)
        closed = time.monotonic()
        server.close()
        receiver.join(WAIT_S)
        assert not receiver.is_alive()
        message, waited = outcome[0]
        assert message is None
        if timeout_ms < 0:
            assert time.monotonic() - closed < 2.0
        else:
            # A positive timeout is waited out, as on any dead link.
            assert waited >= timeout_ms / 1000 - 0.001
        assert server.alive is False
        assert server.close_reason == "closed"
        assert server.peer_closed_before_any_message is False


def test_a_closed_link_still_hands_out_what_arrived():
    with _pair() as (client, server):
        for i in range(3):
            client.send_reliable(64, b"%d" % i)
        assert _wait_for_queued(server, 3)
        server.close()
        assert server.alive is False
        assert server.stats().received_queued == 3
        assert [server.receive(-1).payload for _ in range(3)] == [b"0", b"1", b"2"]
        assert server.receive(0) is None
        assert server.receive(-1) is None
        # The other end learns of the close, and its sends then do nothing.
        assert _wait_for(lambda: not client.alive)
        client.send_reliable(64, b"to nobody")
        client.send_latest(96, b"to nobody")


def test_a_new_link_to_the_same_listener_gets_nothing_from_the_old_one():
    with LinkListener("127.0.0.1", 0) as listener:
        old_client, old_server = _connect(listener)
        with old_client, old_server:
            old_client.send_reliable(64, b"old and unread")
            assert _wait_for_queued(old_server, 1)
            old_server.close()
            assert _wait_for(lambda: not old_client.alive)
            old_client.send_reliable(64, b"old and never sent")
            old_client.send_latest(96, b"old and never sent")

            new_client, new_server = _connect(listener)
            with new_client, new_server:
                # A dead link stays dead: the new connection is another Link.
                assert not old_client.alive
                assert not old_server.alive
                new_client.send_reliable(65, b"new")
                new_server.send_reliable(66, b"new too")
                first = _receive(new_server)
                assert (first.type, first.payload) == (65, b"new")
                first = _receive(new_client)
                assert (first.type, first.payload) == (66, b"new too")
                # Nothing else came: nothing is replayed.
                for link in (new_client, new_server):
                    assert link.stats().messages_received == 1
                    assert link.receive(0) is None
            # The old link kept what it had.
            assert _unread(old_server) == [(64, b"old and unread")]


def test_too_many_unread_link_messages_end_the_link():
    with _pair(server={"max_received_messages": 8}) as (client, server):
        for i in range(8):
            client.send_reliable(64, b"%d" % i)
        assert _wait_for_queued(server, 8)
        assert server.alive  # at the limit
        client.send_reliable(64, b"8")
        assert _wait_for(lambda: not server.alive)
        assert (
            server.close_reason
            == "receive failed: more than 8 received link messages unread"
        )
        assert server.peer_closed_before_any_message is False
        # What was queued stays readable; the one too many is not among it.
        assert _unread(server) == [(64, b"%d" % i) for i in range(8)]
        assert _wait_for(lambda: not client.alive)


def test_too_many_unread_bytes_end_the_link():
    with _pair(server={"max_received_bytes": 1000}) as (client, server):
        client.send_reliable(64, bytes(600))
        client.send_reliable(64, bytes(400))
        assert _wait_for_queued(server, 2)
        assert server.alive  # at the limit
        client.send_reliable(64, b"x")
        assert _wait_for(lambda: not server.alive)
        assert (
            server.close_reason
            == "receive failed: more than 1000 bytes of received link messages unread"
        )
        assert [len(payload) for _type, payload in _unread(server)] == [600, 400]


def test_a_reader_that_keeps_up_never_overflows_the_inbox():
    with _pair(server={"max_received_messages": 2, "max_received_bytes": 100}) as (
        client,
        server,
    ):
        for i in range(50):
            client.send_reliable(64, bytes(80) + b"%2d" % i)
            assert _receive(server).payload[80:] == b"%2d" % i
        assert server.alive


def test_a_replaced_link_message_does_not_count_against_the_inbox_limit():
    options = {"max_received_messages": 2, "latest_types": [96]}
    with _pair(server=options) as (client, server):
        client.send_reliable(64, b"reliable")
        client.send_latest(96, b"0")
        assert _wait_for_queued(server, 2)  # at the limit
        for i in range(1, 10):
            client.send_latest(96, b"%d" % i)
            # It arrives and replaces its predecessor before the next one is sent.
            assert _wait_for_replaced(server, i)
        assert server.alive
        assert server.stats().received_queued == 2
        assert _unread(server) == [(64, b"reliable"), (96, b"9")]


@TRANSPORTS
def test_close_with_linger_lets_the_peer_read_everything(websocket):
    payloads = [_pattern(200_000 + i) for i in range(40)]
    with _pair(websocket, client={"close_drain_ms": WAIT_MS}) as (client, server):
        for payload in payloads:
            client.send_reliable(64, payload)
        client.close(WAIT_MS)
        assert client.close_reason == "closed"
        # close() came back because the peer read to the end of the stream, not because the linger ran out.
        assert server.alive is False
        assert server.close_reason == "receive failed: connection closed"
        received = [payload for _type, payload in _unread(server)]
        assert len(received) == len(payloads)
        intact = received == payloads
        assert intact


def test_close_with_linger_waits_for_the_peer_to_close():
    with _plain_peer(close_drain_ms=WAIT_MS) as (link, peer):
        link.send_reliable(64, b"sent before the close")
        start = time.monotonic()
        link.close(300)  # the peer keeps its end open
        assert time.monotonic() - start >= 0.299
        assert link.close_reason == "closed"
        assert link.peer_closed_before_any_message is False
        assert _read_link_message(peer) == (64, b"sent before the close")
        assert _read_link_message(peer) is None


def test_close_does_not_linger_once_close_drain_ms_ran_out():
    with _plain_peer(close_drain_ms=300) as (link, peer):
        # The peer is not reading: most of this is still unsent when close_drain_ms is over.
        link.send_reliable(64, bytes(4 << 20))
        start = time.monotonic()
        link.close(WAIT_MS)
        waited = time.monotonic() - start
        # The connection is closed then, without the linger, and what is unsent is dropped.
        assert 0.299 <= waited < WAIT_S / 2
        assert link.close_reason == "closed"
        assert len(_read_exactly(peer, HEADER.size + (4 << 20))) < 4 << 20


def test_ctrl_c_during_close_is_raised_once_the_link_is_closed():
    with _plain_peer() as (link, _peer):
        start = time.monotonic()
        with _sigint_after(0.1), pytest.raises(KeyboardInterrupt):
            link.close(500)  # the peer keeps its end open, so the linger is waited out
        # close() is not interrupted.
        assert time.monotonic() - start >= 0.499
        assert link.alive is False
        assert link.close_reason == "closed"


def test_heartbeats_keep_an_idle_link_alive_beyond_peer_timeout_ms():
    options = {"heartbeat_period_ms": 50, "peer_timeout_ms": 500}
    with _pair(client=options, server=options) as (client, server):
        # Three peer timeouts with nothing sent by the test.
        died = _wait_for(lambda: not (client.alive and server.alive), timeout_s=1.5)
        assert not died, (client.close_reason, server.close_reason)
        for link in (client, server):
            # About thirty HEARTBEATs arrived, and none reached receive().
            assert link.stats().messages_received >= 10
            assert link.stats().received_queued == 0
            assert link.receive(0) is None


def test_clock_becomes_valid_with_the_first_received_link_message():
    with _pair() as (client, server):
        clock = server.clock()
        assert (clock.valid, clock.two_way, clock.offset_ns, clock.rtt_min_ns) == (
            False,
            False,
            0,
            -1,
        )
        client.send_reliable(64, b"x")
        _receive(server)
        clock = server.clock()
        assert (clock.valid, clock.two_way, clock.rtt_min_ns) == (True, False, -1)
        # One machine, one clock: the one-way estimate is the delivery time.
        assert 0 <= clock.offset_ns < WAIT_MS * 1_000_000
        assert client.clock().valid is False


def test_send_pings_makes_the_clock_estimate_two_way():
    with _pair(client={"send_pings": True}) as (client, server):
        assert _wait_for(lambda: client.clock().two_way)
        clock = client.clock()
        assert clock.valid is True
        assert clock.rtt_min_ns >= 0
        # One machine, one clock: the estimate is off by at most the round trip.
        assert abs(clock.offset_ns) <= clock.rtt_min_ns
        # The end that does not ping keeps a one-way estimate.
        assert _wait_for(lambda: server.clock().valid)
        assert server.clock().two_way is False
        # PING and PONG never reach receive().
        assert client.receive(0) is None
        assert server.receive(0) is None


def test_many_links_at_once_end_clean():
    pairs = 20
    count = 200
    errors = []

    def exchange(sender, receiver, tag):
        """Sends `count` link messages of each kind on `sender` and checks them on `receiver`."""
        try:
            for i in range(count):
                sender.send_reliable(64, b"%s %d" % (tag, i))
                sender.send_latest(96, b"%d" % i)
            reliable_seen = -1
            latest_seen = -1
            while (reliable_seen, latest_seen) != (count - 1, count - 1):
                message = _receive(receiver)
                if message.type == 64:
                    reliable_seen += 1
                    assert message.payload == b"%s %d" % (tag, reliable_seen)
                else:
                    assert latest_seen < int(message.payload) <= reliable_seen
                    latest_seen = int(message.payload)
        except Exception as e:  # reported below, on the test's thread
            errors.append(e)

    with ExitStack() as stack:
        links = [stack.enter_context(_pair()) for _ in range(pairs)]
        threads = []
        for client, server in links:
            threads.append(
                threading.Thread(target=exchange, args=(client, server, b"up"))
            )
            threads.append(
                threading.Thread(target=exchange, args=(server, client, b"down"))
            )
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(3 * WAIT_S)
        assert not any(thread.is_alive() for thread in threads)
        assert errors == []
        for client, server in links:
            assert client.alive
            assert server.alive
            assert client.receive(0) is None
            assert server.receive(0) is None
    assert not any(client.alive or server.alive for client, server in links)


def test_one_link_used_and_closed_from_many_threads():
    with _pair(server={"latest_types": [96]}) as (client, server):
        stop = threading.Event()
        errors = []

        def repeat(call):
            def run():
                try:
                    while not stop.is_set():
                        call()
                except Exception as e:  # reported below, on the test's thread
                    errors.append(e)

            return threading.Thread(target=run)

        users = [
            repeat(lambda: client.send_reliable(64, b"reliable")),
            repeat(lambda: client.send_latest(96, b"latest")),
            repeat(lambda: server.send_latest(97, b"back")),
            repeat(lambda: server.receive(5)),
            repeat(lambda: server.receive(0)),
            repeat(lambda: server.receive(-1) or time.sleep(0.001)),
            repeat(lambda: client.receive(20)),
            repeat(lambda: (server.alive, server.close_reason, server.stats())),
            repeat(lambda: (client.clock(), client.peer_closed_before_any_message)),
        ]
        for user in users:
            user.start()
        time.sleep(0.2)
        closers = [threading.Thread(target=server.close) for _ in range(4)]
        closers += [threading.Thread(target=client.close, args=(50,)) for _ in range(2)]
        for closer in closers:
            closer.start()
        for closer in closers:
            closer.join(WAIT_S)
        # The users go on for a while on the closed links.
        time.sleep(0.1)
        stop.set()
        for user in users:
            user.join(WAIT_S)
        assert not any(thread.is_alive() for thread in users + closers)
        assert errors == []
        assert not client.alive
        assert not server.alive


def test_dropping_the_last_reference_to_a_live_link_closes_its_connection():
    with LinkListener("127.0.0.1", 0) as listener:
        client, server = _connect(listener)
        with server:
            client.send_reliable(64, b"x")
            assert _receive(server).payload == b"x"
            del client
            gc.collect()
            assert _wait_for(lambda: not server.alive)
            assert server.close_reason == "receive failed: connection closed"


def test_dropping_a_link_while_link_messages_arrive_does_not_hang_or_crash():
    # The inbox outlives the threads that fill it only if the link is torn down in the right order.
    unlimited = {"max_received_messages": 1 << 30, "max_received_bytes": 1 << 40}
    with LinkListener("127.0.0.1", 0) as listener:
        for _ in range(20):
            client, server = _connect(listener, client=unlimited)
            with server:

                def flood(link=server):
                    while link.alive:
                        link.send_reliable(64, bytes(1000))

                flooder = threading.Thread(target=flood)
                flooder.start()
                assert _wait_for_arrivals(client)
                del client
                flooder.join(WAIT_S)
                assert not flooder.is_alive()
                assert not server.alive


@pytest.mark.skipif(
    not os.path.isdir("/proc/self/fd"), reason="reads the open descriptors from /proc"
)
def test_links_and_listeners_give_their_descriptors_back():
    resource = pytest.importorskip("resource")

    def use_links(websocket, how):
        with LinkListener("127.0.0.1", 0) as listener:
            client, server = _connect(listener, websocket)
            client.send_reliable(64, b"x")
            assert _receive(server).payload == b"x"
            if how == "close":
                client.close()
                server.close()
            elif how == "with":
                with client, server:
                    pass
            # Otherwise the last references go with this frame, on live links.

    # Room for 32 descriptors above the highest one open now: far fewer than the uses below, so a kind of
    # use that kept one descriptor each time would run out of it. Not a count before and after: the first
    # log record of the extension opens a connection to this process's log socket that stays.
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    highest = max(int(name) for name in os.listdir("/proc/self/fd"))
    resource.setrlimit(resource.RLIMIT_NOFILE, (highest + 33, hard))
    try:
        for websocket in (False, True):
            for how in ("close", "with", "drop"):
                for _ in range(40):
                    use_links(websocket, how)
    finally:
        resource.setrlimit(resource.RLIMIT_NOFILE, (soft, hard))


# What every exit scenario starts from: a connected pair, and a finalization that lasts longer than a
# wait slice of the bindings (100 ms), so that a thread waiting in them wakes up during it.
_EXIT_PRELUDE = """
import os
import sys
import threading
import time
import types

from isaaccapture.remote_link import Link, LinkListener

listener = LinkListener("127.0.0.1", 0)
client = Link.connect(f"127.0.0.1:{listener.port}", 10000)
server = Link.adopt(listener.accept(10000)[0])
client.send_reliable(64, b"x")
assert server.receive(10000).payload == b"x"


class Slow:
    def __del__(self, _sleep=time.sleep, _write=os.write):
        _sleep(0.3)
        _write(1, b"finalized\\n")


# Not in __main__, whose globals a daemon thread's frame keeps alive.
holder = types.ModuleType("slow_finalization")
holder.slow = Slow()
sys.modules["slow_finalization"] = holder
del holder
"""

# Scenario: (what the child does before it ends, its exit code).
_EXIT_SCENARIOS = {
    "live links": ("", 0),
    "sys.exit": ("print('ready', flush=True)\nsys.exit(3)", 3),
    "a link in a reference cycle": (
        """
class Holder:
    pass


holder = Holder()
holder.itself = holder
holder.links = (client, server)
del holder, client, server
""",
        0,
    ),
    "a daemon thread in receive": (
        """
threading.Thread(target=server.receive, args=(-1,), daemon=True).start()
time.sleep(0.05)
""",
        0,
    ),
    "a daemon thread polling receive": (
        """
def poll():
    while True:
        server.receive(30)


threading.Thread(target=poll, daemon=True).start()
time.sleep(0.05)
""",
        0,
    ),
    "a daemon thread in accept": (
        """
threading.Thread(target=listener.accept, args=(-1,), daemon=True).start()
time.sleep(0.05)
""",
        0,
    ),
}


# Every scenario in a child that leads its own logging session, and one in a child that forwards its
# log records to this process, as a child of a running isaaccapture process does.
_EXIT_CASES = [(scenario, "leader") for scenario in _EXIT_SCENARIOS]
_EXIT_CASES.append(("a daemon thread in receive", "forwarder"))


def _assert_exits(script, exit_code=0, log_role="leader"):
    """Runs `script` after _EXIT_PRELUDE in a child interpreter, which must end with `exit_code`."""
    environment = dict(os.environ)
    if log_role == "leader":
        # The forwarding socket of this process is what would make the child a forwarder.
        environment.pop("ISAACCAPTURE_LOG_SOCKET", None)
    done = subprocess.run(
        [sys.executable, "-c", _EXIT_PRELUDE + script],
        capture_output=True,
        text=True,
        timeout=60,
        env=environment,
        check=False,
    )
    assert done.returncode == exit_code, (done.stdout, done.stderr[-3000:])
    assert done.stdout == "ready\nfinalized\n", done.stderr[-3000:]


@pytest.mark.parametrize("scenario, log_role", _EXIT_CASES)
def test_interpreter_exit_with_a_live_link_does_not_hang_or_crash(scenario, log_role):
    script, exit_code = _EXIT_SCENARIOS[scenario]
    _assert_exits(script + "\nprint('ready', flush=True)\n", exit_code, log_role)


# The calls of the bindings that give up the GIL, each as one daemon thread of the child makes it, with
# keyword arguments where it takes any: those are what the thread still holds when it is ended.
_EXIT_WAITS = {
    "receive": "server.receive(timeout_ms=10_000)",
    "accept": "listener.accept(timeout_ms=10_000)",
    "the wait for a websocket upgrade": "Link.connect(silent, 10_000, name='waits')",
    "close with linger": "unread[0].close(linger_ms=10_000)",
    "the last reference going": "unread.clear()",
}

# What the child of such a scenario runs after the prelude. Finalization begins while the thread waits
# for the GIL at the end of its call: up to Python 3.13 the interpreter then ends the thread there.
_EXIT_WAIT_SCRIPT = """
import atexit
import socket

# A Link with a reliable link message that cannot go out: its peer neither reads nor closes.
mine, peer = socket.socketpair()
unread = [Link.adopt(mine.detach(), close_drain_ms=10_000)]
unread[0].send_reliable(64, bytes(4 << 20))
# A listener that never accepts: a ws:// connect to it waits for the answer to its upgrade.
unanswered = socket.socket()
unanswered.bind(("127.0.0.1", 0))
unanswered.listen(1)
silent = f"ws://127.0.0.1:{unanswered.getsockname()[1]}"

threading.Thread(target=lambda: THE_CALL, daemon=True).start()
time.sleep(0.1)
print("ready", flush=True)


class HoldsTheGil:
    def __del__(self):
        # No other thread gets the GIL from here until this one blocks: in Slow.__del__, after finalization
        # began. Meanwhile the thread's wait slice ends (the peer's close ends a close()), and it waits for
        # the GIL.
        sys.setswitchinterval(1000)
        peer.close()
        until = time.monotonic() + 0.3
        while time.monotonic() < until:
            pass


# The arguments of the atexit callbacks are freed when all callbacks ran, those of the last one registered
# last: nothing lets go of the GIL between this hold and the start of finalization.
atexit.register(lambda held: None, HoldsTheGil())
"""


@pytest.mark.parametrize("wait", list(_EXIT_WAITS))
def test_interpreter_exit_with_a_daemon_thread_waiting_for_the_gil_does_not_crash(wait):
    _assert_exits(_EXIT_WAIT_SCRIPT.replace("THE_CALL", _EXIT_WAITS[wait]))


def test_listener_binds_a_free_port_and_keeps_reporting_it():
    with LinkListener("127.0.0.1", 0) as listener:
        port = listener.port
        assert type(port) is int
        assert 0 < port < 65536
        with pytest.raises(AttributeError):
            listener.port = 1
        with pytest.raises(
            RuntimeError, match=rf"^cannot listen on 127\.0\.0\.1:{port}: "
        ):
            LinkListener("127.0.0.1", port)
    # Closed by the context manager: the port is free again, and still reported.
    assert listener.port == port
    with LinkListener("127.0.0.1", port) as again:
        assert again.port == port


def test_accept_hands_out_the_descriptor_and_the_peers_address():
    with LinkListener("127.0.0.1", 0) as listener:
        with socket.create_connection(("127.0.0.1", listener.port)) as dialer:
            accepted = listener.accept(WAIT_MS)
            assert type(accepted) is tuple
            fd, peer = accepted
            assert peer == f"127.0.0.1:{dialer.getsockname()[1]}"
            # The descriptor is the caller's: here a plain socket object takes it.
            with socket.socket(fileno=fd) as taken:
                dialer.sendall(b"ping")
                assert taken.recv(4) == b"ping"
            assert listener.accept(0) is None


def test_accept_returns_none_after_its_timeout():
    with LinkListener("127.0.0.1", 0) as listener:
        waits = []
        for _ in range(5):
            start = time.monotonic()
            assert listener.accept(0) is None
            waits.append(time.monotonic() - start)
        # The quickest of five, so that a busy machine does not count: one wait slice is 100 ms.
        assert min(waits) < 0.05
        start = time.monotonic()
        assert listener.accept(150) is None
        assert time.monotonic() - start >= 0.149


def test_a_closed_listener_refuses_accept():
    listener = LinkListener("127.0.0.1", 0)
    listener.close()
    listener.close()  # idempotent
    for timeout_ms in (0, 100, -1):
        with pytest.raises(RuntimeError, match=r"^Listener is closed$"):
            listener.accept(timeout_ms)


def test_close_ends_an_accept_on_another_thread():
    listener = LinkListener("127.0.0.1", 0)
    outcome = []

    def accept():
        try:
            outcome.append(listener.accept(-1))
        except RuntimeError as e:
            outcome.append(e)

    acceptor = threading.Thread(target=accept)
    acceptor.start()
    # accept() is waiting by now; one that starts after the close ends the same way.
    time.sleep(0.1)
    start = time.monotonic()
    listener.close()
    acceptor.join(WAIT_S)
    assert not acceptor.is_alive()
    assert time.monotonic() - start < 2.0
    assert isinstance(outcome[0], RuntimeError)
    assert str(outcome[0]) == "Listener is closed"
