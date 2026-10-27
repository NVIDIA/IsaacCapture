# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A RemoteSessionHost and a RemoteSession over loopback, in one process.

The server runs on a thread. Its KeyboardTracker never touches the OpenXR session handles, so
placeholder handles stand in for a runtime.
"""

import json
import os
import signal
import socket
import sys
import threading
import time
from contextlib import contextmanager

import pytest

from isaaccapture import oxr
from isaaccapture.deviceio_session import (
    RemotePeerClosedError,
    RemoteSession,
    RemoteSessionHost,
    RemoteSessionListener,
    VendorConfig,
)
from isaaccapture.deviceio_trackers import EvdevKeyCode, KeyboardTracker

pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason="remote sessions need POSIX sockets"
)

WAIT_MS = 5000
KEY_W = int(EvdevKeyCode.KeyW)
HELLO_ACK = {"session_id": "test", "server_version": "0"}
READY = {"stage": "ready", "message": ""}
TRANSPORTS = pytest.mark.parametrize("websocket", [False, True], ids=["tcp", "ws"])


def _endpoint(listener, websocket):
    return f"{'ws' if websocket else 'tcp'}://127.0.0.1:{listener.port}"


def _wait_for(predicate, timeout_s=5.0):
    deadline = time.monotonic() + timeout_s
    while not predicate():
        if time.monotonic() > deadline:
            return False
        time.sleep(0.002)
    return True


class _Server:
    """Accepts one robot and streams a keyboard that holds W, on its own thread.

    Sends READY at once unless `ready` is False; on stop() sends `final_error` (if any) and closes.
    """

    def __init__(self, websocket=False, ready=True, final_error=None):
        self.listener = RemoteSessionListener("127.0.0.1", 0)
        self.websocket = websocket
        self.endpoint = _endpoint(self.listener, websocket)
        self.hello = None
        self.error = None
        self._ready = ready
        self._final_error = final_error
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        try:
            accepted = self.listener.accept(WAIT_MS)
            assert accepted is not None, "no robot connected"
            with RemoteSessionHost.adopt(accepted[0], self.websocket) as host:
                self.hello = json.loads(host.wait_hello(WAIT_MS))
                host.send_hello_ack(json.dumps(HELLO_ACK))
                keyboard = KeyboardTracker()
                host.start_session(
                    [(keyboard, "0")],
                    oxr.OpenXRSessionHandles(1, 1, 1, 1),
                    VendorConfig(),
                    self.hello["topics"],
                )
                provider = keyboard.create_provider()
                provider.key_down(KEY_W)
                if self._ready:
                    host.send_ready(json.dumps(READY))
                while not self._stop.is_set() and host.alive:
                    host.tick()
                    time.sleep(0.005)
                if self._final_error is not None:
                    host.send_error(self._final_error)
        except Exception as e:  # reported by stop()
            self.error = e

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=10)
        assert not self._thread.is_alive()
        if self.error is not None:
            raise self.error


def _robot():
    keyboard = KeyboardTracker()
    return keyboard, RemoteSession.create([(keyboard, "0")])


def _hello(session):
    return json.dumps({"protocol": 1, "topics": session.subscribed_topics()})


def _holds_w(keyboard, session):
    session.update()
    data = keyboard.get_keyboard_data(session)
    return data is not None and list(data.held_keys) == [KEY_W]


@TRANSPORTS
def test_robot_reads_the_server_keyboard(websocket):
    keyboard, session = _robot()
    server = _Server(websocket)
    with session:
        session.connect(server.endpoint, _hello(session), WAIT_MS)
        assert json.loads(session.hello_ack()) == HELLO_ACK
        assert session.wait_ready(WAIT_MS)
        assert json.loads(session.latest_status()) == READY

        assert _wait_for(lambda: _holds_w(keyboard, session))
        assert session.is_fresh
        assert session.tick_age_ns is not None
        assert abs(session.tick_age_ns) < 250_000_000
    server.stop()
    assert server.hello["topics"] == ["0/keyboard"]
    with pytest.raises(RuntimeError, match="closed"):
        session.update()


def test_wait_ready_is_false_until_ready():
    keyboard, session = _robot()
    server = _Server(ready=False)
    with session:
        session.connect(server.endpoint, _hello(session), WAIT_MS)
        start = time.monotonic()
        assert session.wait_ready(100) is False
        assert time.monotonic() - start >= 0.1
        assert session.latest_status() is None
        # SAMPLES flow before READY.
        assert _wait_for(lambda: _holds_w(keyboard, session))
    server.stop()


def _update_error(session):
    try:
        session.update()
    except RuntimeError as e:
        return str(e)
    return None


def test_update_raises_once_the_server_closes():
    keyboard, session = _robot()
    server = _Server()
    with session:
        session.connect(server.endpoint, _hello(session), WAIT_MS)
        assert session.wait_ready(WAIT_MS)
        assert _wait_for(lambda: _holds_w(keyboard, session))
        server.stop()
        assert _wait_for(lambda: _update_error(session) is not None)
        assert _update_error(session).startswith("Remote session lost: ")
        with pytest.raises(RuntimeError, match="Remote session lost"):
            session.wait_ready(0)


@TRANSPORTS
def test_update_raises_the_server_error_text(websocket):
    keyboard, session = _robot()
    server = _Server(websocket, final_error="headset lost")
    with session:
        session.connect(server.endpoint, _hello(session), WAIT_MS)
        assert _wait_for(lambda: _holds_w(keyboard, session))
        server.stop()
        assert _wait_for(lambda: _update_error(session) is not None)
        assert _update_error(session) == "headset lost"


@TRANSPORTS
def test_connect_raises_the_text_of_a_refusal(websocket):
    listener = RemoteSessionListener("127.0.0.1", 0)
    busy = "busy: another robot app is connected to this Teleop Session Server"

    def refuse():
        accepted = listener.accept(WAIT_MS)
        if accepted is not None:
            RemoteSessionListener.reject(accepted[0], websocket, busy)

    refuser = threading.Thread(target=refuse)
    refuser.start()
    _, session = _robot()
    with session:
        with pytest.raises(RuntimeError) as raised:
            session.connect(_endpoint(listener, websocket), _hello(session), WAIT_MS)
    refuser.join()
    assert str(raised.value) == busy


def test_connect_times_out_without_hello_ack():
    listener = RemoteSessionListener("127.0.0.1", 0)
    accepted = []
    acceptor = threading.Thread(
        target=lambda: accepted.append(listener.accept(WAIT_MS))
    )
    acceptor.start()
    _, session = _robot()
    try:
        with session:
            with pytest.raises(TimeoutError, match="HELLO_ACK"):
                session.connect(_endpoint(listener, False), _hello(session), 300)
    finally:
        acceptor.join()
        if accepted and accepted[0] is not None:
            os.close(accepted[0][0])


def test_close_ends_a_wait_ready_on_another_thread():
    _, session = _robot()
    server = _Server(ready=False)
    session.connect(server.endpoint, _hello(session), WAIT_MS)
    outcome = []

    def wait():
        try:
            outcome.append(session.wait_ready(WAIT_MS))
        except RuntimeError as e:
            outcome.append(e)

    waiter = threading.Thread(target=wait)
    waiter.start()
    time.sleep(0.1)
    start = time.monotonic()
    session.close()
    waiter.join()
    assert time.monotonic() - start < 2.0
    assert isinstance(outcome[0], RuntimeError)
    server.stop()


def test_host_refuses_work_after_close():
    listener = RemoteSessionListener("127.0.0.1", 0)
    with socket.create_connection(("127.0.0.1", listener.port)) as robot:
        fd, peer = listener.accept(WAIT_MS)
        assert peer == f"127.0.0.1:{robot.getsockname()[1]}"
        host = RemoteSessionHost.adopt(fd, False)
        with pytest.raises(RuntimeError, match="start_session"):
            host.tick()
        assert host.alive
        assert host.close_reason == ""
        robot.close()  # lets close() skip its linger
        host.close()
        assert not host.alive
        assert host.close_reason
        with pytest.raises(RuntimeError, match="closed"):
            host.tick()
        with pytest.raises(RuntimeError, match="closed"):
            host.start_session(
                [(KeyboardTracker(), "0")],
                oxr.OpenXRSessionHandles(1, 1, 1, 1),
                VendorConfig(),
                [],
            )
        host.close()  # idempotent


def _runs_alongside(call):
    """True when this thread keeps running while `call` blocks for about 1.5 s on another thread."""

    def target():
        try:
            call()
        except (RuntimeError, TimeoutError):
            pass

    thread = threading.Thread(target=target)
    # Each step here needs the GIL (start() included), so a call that holds it shows up as one long gap.
    longest = 0.0
    last = time.monotonic()
    thread.start()
    while thread.is_alive():
        time.sleep(0.01)
        now = time.monotonic()
        longest = max(longest, now - last)
        last = now
    thread.join()
    return longest < 0.75


def test_blocking_calls_release_the_gil():
    listener = RemoteSessionListener("127.0.0.1", 0)
    assert _runs_alongside(lambda: listener.accept(1500))

    # A robot that never sends HELLO.
    with socket.create_connection(("127.0.0.1", listener.port)):
        fd, _ = listener.accept(WAIT_MS)
        with RemoteSessionHost.adopt(fd, False) as host:
            assert _runs_alongside(lambda: host.wait_hello(1500))

    # A server that never answers HELLO.
    accepted = []
    acceptor = threading.Thread(
        target=lambda: accepted.append(listener.accept(WAIT_MS))
    )
    acceptor.start()
    _, session = _robot()
    with session:
        assert _runs_alongside(
            lambda: session.connect(_endpoint(listener, False), "{}", 1500)
        )
    acceptor.join()
    os.close(accepted[0][0])

    _, session = _robot()
    server = _Server(ready=False)
    with session:
        session.connect(server.endpoint, _hello(session), WAIT_MS)
        assert _runs_alongside(lambda: session.wait_ready(1500))
    server.stop()


def test_dropping_an_open_host_releases_the_gil():
    listener = RemoteSessionListener("127.0.0.1", 0)
    # The robot stays connected, so the host's teardown lingers for it to close.
    with socket.create_connection(("127.0.0.1", listener.port)):
        fd, _ = listener.accept(WAIT_MS)
        hosts = [RemoteSessionHost.adopt(fd, False)]
        start = time.monotonic()
        assert _runs_alongside(hosts.clear)
        assert time.monotonic() - start >= 0.5


def test_start_session_waits_for_a_closing_host_without_the_gil():
    listener = RemoteSessionListener("127.0.0.1", 0)
    with socket.create_connection(("127.0.0.1", listener.port)) as robot:
        fd, _ = listener.accept(WAIT_MS)
        host = RemoteSessionHost.adopt(fd, False)
        closer = threading.Thread(target=host.close)
        closer.start()
        # End of stream: close() has half-closed and lingers for the robot, holding the host.
        robot.settimeout(WAIT_MS / 1000)
        while robot.recv(4096):
            pass
        assert _runs_alongside(
            lambda: host.start_session(
                [(KeyboardTracker(), "0")],
                oxr.OpenXRSessionHandles(1, 1, 1, 1),
                VendorConfig(),
                [],
            )
        )
        closer.join()


def test_dropping_an_open_robot_session_disconnects_it():
    _, session = _robot()
    server = _Server()
    session.connect(server.endpoint, _hello(session), WAIT_MS)
    assert session.wait_ready(WAIT_MS)
    del session
    # The server's loop ends once its link is down.
    assert _wait_for(lambda: not server._thread.is_alive())
    server.stop()


@TRANSPORTS
def test_start_then_wait_hello_ack_joins_a_server(websocket):
    _, session = _robot()
    server = _Server(websocket=websocket)
    with session:
        session.start(server.endpoint, _hello(session), WAIT_MS)
        assert session.wait_hello_ack(WAIT_MS)
        assert json.loads(session.hello_ack()) == HELLO_ACK
        assert session.wait_ready(WAIT_MS)
    server.stop()


@TRANSPORTS
def test_start_then_wait_ready_joins_a_server(websocket):
    _, session = _robot()
    server = _Server(websocket=websocket)
    with session:
        session.start(server.endpoint, _hello(session), WAIT_MS)
        if websocket:
            # Nothing completes the WebSocket upgrade, so update() cannot report progress.
            with pytest.raises(
                RuntimeError, match=r"call wait_hello_ack\(\) or wait_ready\(\)"
            ):
                session.update()
        assert session.wait_ready(WAIT_MS)
        assert json.loads(session.hello_ack()) == HELLO_ACK
        session.update()
    server.stop()


def test_start_racing_close_on_another_thread_leaves_no_connection():
    """A start() that runs while close() releases the GIL must not connect the closed session."""
    with socket.socket() as busy, socket.socket() as other:
        for listener in (busy, other):
            listener.bind(("127.0.0.1", 0))
            listener.listen(8)
        _, session = _robot()
        # Nothing reads this HELLO, so close() spends a while draining it without the GIL.
        session.start(
            f"tcp://127.0.0.1:{busy.getsockname()[1]}", "x" * 15_000_000, WAIT_MS
        )
        other_endpoint = f"tcp://127.0.0.1:{other.getsockname()[1]}"
        connected = []
        done = threading.Event()

        def race():
            while not done.is_set():
                try:
                    session.start(other_endpoint, "{}", 1000)
                    connected.append(True)
                except RuntimeError:
                    pass  # already connected, or closed

        racer = threading.Thread(target=race)
        racer.start()
        time.sleep(0.05)
        session.close()
        done.set()
        racer.join()
        assert connected == []


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


def test_a_tcp_connect_nothing_answers_times_out_after_connect_timeout_ms():
    with _unanswered_endpoint() as endpoint:
        _, session = _robot()
        with session:
            start = time.monotonic()
            with pytest.raises(RuntimeError, match=r"^cannot connect to .*timed out"):
                session.start(endpoint, _hello(session), 2500)
            assert 2.4 < time.monotonic() - start < 3.5


def test_ctrl_c_during_a_tcp_connect_raises_keyboard_interrupt_alone():
    with _unanswered_endpoint() as endpoint:
        _, session = _robot()
        with session:
            with (
                _sigint_after(0.5) as sent,
                pytest.raises(KeyboardInterrupt) as raised,
            ):
                session.start(endpoint, _hello(session), 1000)
            # It replaces the connect error instead of interrupting its handling.
            assert raised.value.__context__ is None
            # The signal does not restart the attempt's wait.
            assert time.monotonic() - sent[0] < 0.85
            with pytest.raises(RuntimeError, match="not connected"):
                session.update()


def test_ctrl_c_interrupts_an_unbounded_tcp_connect():
    with _unanswered_endpoint() as endpoint:
        _, session = _robot()
        with session:
            with _sigint_after(0.3) as sent, pytest.raises(KeyboardInterrupt):
                session.connect(endpoint, _hello(session), -1)
            # Within one connect attempt (2 s).
            assert time.monotonic() - sent[0] < 2.5


def test_start_may_be_called_again_after_nothing_accepted():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    _, session = _robot()
    server = None
    with session:
        with pytest.raises(RuntimeError, match=rf"^cannot connect to 127.0.0.1:{port}"):
            session.start(f"tcp://127.0.0.1:{port}", _hello(session), 1000)
        server = _Server()
        session.start(server.endpoint, _hello(session), WAIT_MS)
        assert session.wait_hello_ack(WAIT_MS)
    server.stop()


@contextmanager
def _closing_listener():
    """A listener that closes every connection unanswered once the robot sent something; yields its port."""
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
                connection.settimeout(WAIT_MS / 1000)
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
        thread.join(timeout=5)
        listener.close()


@TRANSPORTS
@pytest.mark.parametrize("call", ["wait_hello_ack", "wait_ready", "connect"])
def test_a_server_that_closes_unanswered_raises_remote_peer_closed_error(
    call, websocket
):
    scheme = "ws" if websocket else "tcp"
    with _closing_listener() as port:
        endpoint = f"{scheme}://127.0.0.1:{port}"
        _, session = _robot()
        with session:
            with pytest.raises(RemotePeerClosedError) as raised:
                if call == "connect":
                    session.connect(endpoint, _hello(session), WAIT_MS)
                else:
                    session.start(endpoint, _hello(session), WAIT_MS)
                    getattr(session, call)(WAIT_MS)
            assert isinstance(raised.value, RuntimeError)
            if websocket:
                assert str(raised.value) == (
                    f"WebSocket upgrade to 127.0.0.1:{port}/: connection closed before "
                    "the answer"
                )
            else:
                lost = (
                    "Remote session lost"
                    if call == "wait_ready"
                    else ("Remote session ended before HELLO_ACK")
                )
                assert str(raised.value) == f"{lost}: receive failed: connection closed"
            # The session is unconnected: start() may connect again.
            with pytest.raises(RuntimeError, match="not connected"):
                session.update()
            server = _Server(websocket=websocket)
            try:
                session.start(server.endpoint, _hello(session), WAIT_MS)
                assert session.wait_hello_ack(WAIT_MS)
            finally:
                session.close()
                server.stop()


def test_a_refusal_is_no_remote_peer_closed_error():
    listener = RemoteSessionListener("127.0.0.1", 0)
    refusal = "busy: another robot app is connected to this Teleop Session Server"

    def refuse():
        accepted = listener.accept(WAIT_MS)
        if accepted is not None:
            RemoteSessionListener.reject(accepted[0], False, refusal)

    thread = threading.Thread(target=refuse)
    thread.start()
    _, session = _robot()
    with session:
        session.start(_endpoint(listener, False), _hello(session), WAIT_MS)
        with pytest.raises(RuntimeError) as raised:
            session.wait_hello_ack(WAIT_MS)
    thread.join(timeout=10)
    assert type(raised.value) is RuntimeError
    assert str(raised.value) == refusal


@contextmanager
def _sigint_after(delay_s):
    """Sends this process SIGINT after `delay_s`; yields a list that gets the send time."""
    sent = []

    def interrupt():
        sent.append(time.monotonic())
        os.kill(os.getpid(), signal.SIGINT)

    previous = signal.signal(signal.SIGINT, signal.default_int_handler)
    timer = threading.Timer(delay_s, interrupt)
    timer.start()
    try:
        yield sent
    finally:
        timer.cancel()
        timer.join()
        signal.signal(signal.SIGINT, previous)


@pytest.mark.parametrize("websocket", [False, True], ids=["tcp", "ws"])
@pytest.mark.parametrize("call", ["connect", "wait_hello_ack"])
def test_ctrl_c_interrupts_an_unbounded_wait_for_hello_ack(call, websocket):
    """The listener never accepts: the connection waits in its backlog, unanswered."""
    with socket.socket() as silent:
        silent.bind(("127.0.0.1", 0))
        silent.listen(1)
        endpoint = (
            f"{'ws' if websocket else 'tcp'}://127.0.0.1:{silent.getsockname()[1]}"
        )
        _, session = _robot()
        with session:
            with _sigint_after(0.3) as sent, pytest.raises(KeyboardInterrupt):
                if call == "connect":
                    session.connect(endpoint, _hello(session), -1)
                else:
                    session.start(endpoint, _hello(session), WAIT_MS)
                    session.wait_hello_ack(-1)
            assert time.monotonic() - sent[0] < 0.5
            if call == "connect":
                # A failed connect leaves the session unconnected.
                with pytest.raises(RuntimeError, match="not connected"):
                    session.update()


def test_ctrl_c_interrupts_an_unbounded_wait_for_ready():
    _, session = _robot()
    server = _Server(ready=False)
    with session:
        session.connect(server.endpoint, _hello(session), WAIT_MS)
        with _sigint_after(0.3) as sent, pytest.raises(KeyboardInterrupt):
            session.wait_ready(-1)
        assert time.monotonic() - sent[0] < 0.5
    server.stop()
