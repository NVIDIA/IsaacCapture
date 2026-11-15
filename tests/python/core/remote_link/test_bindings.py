# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The isaaccapture.remote_link bindings, exercised without a network.

Where a test needs a connection, a socketpair joins two Links.
"""

import gc
import os
import re
import socket
import subprocess
import sys
import time

import pytest

from isaaccapture import deviceio_session, remote_link
from isaaccapture.remote_link import (
    MAX_PAYLOAD_SIZE,
    ClockEstimate,
    Link,
    LinkListener,
    LinkPeerClosedError,
    LinkStats,
    ReceivedLinkMessage,
)

pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason="links need POSIX sockets"
)

WAIT_MS = 10_000
PUBLIC_NAMES = {
    "MAX_PAYLOAD_SIZE",
    "ClockEstimate",
    "Link",
    "LinkListener",
    "LinkPeerClosedError",
    "LinkStats",
    "ReceivedLinkMessage",
}
# No HEARTBEAT and no peer timeout within a test: only what the test sends travels.
QUIET = {"heartbeat_period_ms": 60_000, "peer_timeout_ms": 60_000}
# The link's own HEARTBEAT, PING and PONG.
OWN_TYPES = (11, 13, 14)
# Each keyword option of connect() and adopt() with its default, as the signature shows it.
OPTION_DEFAULTS = (
    "name: str = 'link'",
    "heartbeat_period_ms: int = 500",
    "peer_timeout_ms: int = 3000",
    "close_drain_ms: int = 500",
    f"max_queued_reliable_bytes: int = {64 << 20}",
    "send_pings: bool = False",
    "peer_timeout_from_first_message: bool = False",
    "max_received_messages: int = 4096",
    f"max_received_bytes: int = {64 << 20}",
)
BAD_OPTIONS = (
    ("heartbeat_period_ms", 0),
    ("peer_timeout_ms", 0),
    ("close_drain_ms", -1),
    # Not "no limit": the link would die with its first send.
    ("max_queued_reliable_bytes", 0),
    ("max_received_messages", 0),
    ("max_received_bytes", 0),
    ("latest_types", [256]),
    ("latest_types", [96, -1]),
)


def _joined_links(**options):
    """Two Links joined by a socketpair."""
    one, other = socket.socketpair()
    return (
        Link.adopt(one.detach(), **{**QUIET, **options}),
        Link.adopt(other.detach(), **{**QUIET, **options}),
    )


def _receive(link):
    message = link.receive(WAIT_MS)
    assert message is not None, f"nothing received: {link.close_reason!r}"
    return message


def _wait_for(predicate, timeout_s=WAIT_MS / 1000):
    deadline = time.monotonic() + timeout_s
    while not predicate():
        if time.monotonic() > deadline:
            return False
        time.sleep(0.002)
    return True


def test_the_package_exports_exactly_its_public_names():
    assert set(remote_link.__all__) == PUBLIC_NAMES
    assert len(remote_link.__all__) == len(PUBLIC_NAMES)
    assert {n for n in vars(remote_link) if not n.startswith("_")} == PUBLIC_NAMES
    # The extension binds nothing more: no enum of type numbers, no transport class.
    extension = remote_link._remote_link
    assert {n for n in vars(extension) if not n.startswith("_")} == PUBLIC_NAMES
    for name in PUBLIC_NAMES:
        assert getattr(remote_link, name) is getattr(extension, name)


def test_max_payload_size_is_16_mib():
    assert type(MAX_PAYLOAD_SIZE) is int
    assert MAX_PAYLOAD_SIZE == 16 * 1024 * 1024


def test_link_peer_closed_error_is_a_runtime_error_of_this_module():
    assert issubclass(LinkPeerClosedError, RuntimeError)
    assert "closed or reset the connection" in LinkPeerClosedError.__doc__
    assert LinkPeerClosedError is not deviceio_session.RemotePeerClosedError
    assert not issubclass(LinkPeerClosedError, deviceio_session.RemotePeerClosedError)


def test_the_listener_is_a_class_of_this_module():
    assert LinkListener is not deviceio_session.RemoteSessionListener
    assert not issubclass(LinkListener, deviceio_session.RemoteSessionListener)
    assert LinkListener.__module__ == "isaaccapture.remote_link._remote_link"


# What every import order must leave working: both modules' listener and exception classes, and a Link.
_BOTH_MODULES_WORK = """
import socket
from isaaccapture import deviceio_session, remote_link

assert remote_link.LinkListener is not deviceio_session.RemoteSessionListener
assert remote_link.LinkPeerClosedError is not deviceio_session.RemotePeerClosedError
for listener in (remote_link.LinkListener, deviceio_session.RemoteSessionListener):
    try:
        listener("127.0.0.1", 70000)
    except ValueError as e:
        assert "port" in str(e), e
    else:
        raise AssertionError(f"{listener.__name__} took port 70000")
one, other = socket.socketpair()
with remote_link.Link.adopt(one.detach()) as sender:
    with remote_link.Link.adopt(other.detach()) as receiver:
        sender.send_reliable(64, b"x")
        assert receiver.receive(10000).payload == b"x"
print("both modules work")
"""

_IMPORT_ORDERS = {
    "remote_link first": """
import isaaccapture.remote_link
import isaaccapture.deviceio_session
""",
    "deviceio_session first": """
import isaaccapture.deviceio_session
import isaaccapture.remote_link
""",
    # `import isaaccapture` loads _deviceio_session before anything else can ask for _remote_link, so only
    # loading the extension by its file puts it first.
    "extension before the package": """
import importlib.util
import pathlib
import sys

package = pathlib.Path(importlib.util.find_spec("isaaccapture").origin).parent
(path,) = (package / "remote_link").glob("_remote_link*.so")
name = "isaaccapture.remote_link._remote_link"
spec = importlib.util.spec_from_file_location(name, path)
extension = importlib.util.module_from_spec(spec)
spec.loader.exec_module(extension)
assert "isaaccapture" not in sys.modules
sys.modules[name] = extension

import isaaccapture.remote_link
assert isaaccapture.remote_link.Link is extension.Link
""",
}


def _run_python(script):
    """Runs `script` in a new interpreter; its output when it ended with exit code 0."""
    done = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert done.returncode == 0, done.stderr[-3000:]
    return done.stdout


@pytest.mark.parametrize("order", list(_IMPORT_ORDERS))
def test_remote_link_and_deviceio_session_load_in_either_order(order):
    output = _run_python(_IMPORT_ORDERS[order] + _BOTH_MODULES_WORK)
    assert output.endswith("both modules work\n")


def test_only_an_explicit_import_loads_remote_link_and_it_opens_nothing():
    output = _run_python(
        """
import os
import sys

import isaaccapture

assert "isaaccapture.remote_link" not in sys.modules
counted = [d for d in ("/proc/self/fd", "/proc/self/task") if os.path.isdir(d)]
before = [len(os.listdir(d)) for d in counted]
import isaaccapture.remote_link

# No descriptor (socket) and no thread.
assert [len(os.listdir(d)) for d in counted] == before
print("imported")
"""
    )
    assert output.endswith("imported\n")


def test_connect_and_adopt_show_their_keyword_options_and_defaults():
    connect = Link.connect.__doc__
    adopt = Link.adopt.__doc__
    assert "connect(endpoint: str, timeout_ms: int, *, name: str = 'link'" in connect
    assert "adopt(fd: int, *, websocket: bool = False, name: str = 'link'" in adopt
    for option in OPTION_DEFAULTS:
        assert option in connect, option
        assert option in adopt, option
    for signature in (connect, adopt):
        assert signature.split("\n", 1)[0].count("latest_types: ") == 1
        # The type the generated stub gets takes its own default.
        assert "latest_types: Iterable[int] = ()) -> " in signature
    assert "linger_ms: int = 0" in Link.close.__doc__


def test_link_and_the_result_classes_have_no_constructor():
    for cls in (Link, ReceivedLinkMessage, ClockEstimate, LinkStats):
        with pytest.raises(TypeError, match="No constructor defined"):
            cls()


@pytest.mark.parametrize(
    "endpoint, text",
    [
        ("", r"^missing host in endpoint ''$"),
        (":47600", r"^missing host in endpoint ':47600'$"),
        ("127.0.0.1:99999", r"^invalid port in endpoint '127\.0\.0\.1:99999'$"),
        ("127.0.0.1:port", r"^invalid port in endpoint '127\.0\.0\.1:port'$"),
        ("[::1:47600", r"^missing '\]' in endpoint '\[::1:47600'$"),
        ("tcp://127.0.0.1:47600/path", r"^a TCP endpoint takes no path: "),
        ("udp://127.0.0.1:47600", r"^unsupported scheme in 'udp://127\.0\.0\.1:47600'"),
        (
            "wss://127.0.0.1:47601",
            r"^wss:// is not supported yet; use ws:// through an encrypted tunnel "
            r"\(VPN, SSH or stunnel\) between the two machines$",
        ),
    ],
)
def test_connect_rejects_a_malformed_endpoint(endpoint, text):
    with pytest.raises(ValueError, match=text):
        Link.connect(endpoint, 1000)


@pytest.mark.parametrize(
    "endpoint",
    [
        "127.0.0.1",
        "localhost",
        "tcp://127.0.0.1",
        "ws://127.0.0.1/path",
        "[::1]",
        "::1",
    ],
)
def test_connect_rejects_an_endpoint_without_a_port(endpoint):
    # Not the Teleop Session Server's port, which such an endpoint means to a RemoteSession: a link to
    # the wrong service would look alive.
    with pytest.raises(
        ValueError,
        match=rf"^Link\.connect: endpoint '{re.escape(endpoint)}' names no port$",
    ):
        Link.connect(endpoint, 1000)


@pytest.mark.parametrize("option, value", BAD_OPTIONS)
def test_connect_rejects_a_bad_option_before_it_connects(option, value):
    # Had it dialled port 1 first, the error would be the RuntimeError of a refused connect.
    with pytest.raises(ValueError, match=rf"^Link: {option} must be "):
        Link.connect("127.0.0.1:1", 1000, **{option: value})


def test_connect_and_adopt_take_their_options_by_keyword_only():
    with pytest.raises(TypeError):
        Link.connect("127.0.0.1:1", 1000, "name")
    with pytest.raises(TypeError):
        Link.connect("127.0.0.1:1", 1000, unknown_option=1)
    with pytest.raises(TypeError):
        Link.connect("127.0.0.1:1")
    with pytest.raises(TypeError):
        Link.connect(None, 1000)
    with pytest.raises(TypeError):
        Link.connect("127.0.0.1:1", "soon")
    one, other = socket.socketpair()
    with one, other:
        with pytest.raises(TypeError):
            Link.adopt(one.fileno(), True)
        with pytest.raises(TypeError):
            Link.adopt(one.fileno(), unknown_option=1)
        with pytest.raises(TypeError):
            Link.adopt(str(one.fileno()))
        with pytest.raises(TypeError):
            Link.adopt(None)
        with pytest.raises(TypeError):
            Link.adopt()


@pytest.mark.parametrize("websocket", [False, True], ids=["tcp", "ws"])
def test_adopt_rejects_a_negative_descriptor(websocket):
    with pytest.raises(ValueError, match="descriptor"):
        Link.adopt(-1, websocket=websocket)


@pytest.mark.parametrize("option, value", BAD_OPTIONS)
def test_adopt_rejects_a_bad_option_and_leaves_the_descriptor_with_the_caller(
    option, value
):
    one, other = socket.socketpair()
    with one, other:
        with pytest.raises(ValueError, match=rf"^Link: {option} must be "):
            Link.adopt(one.fileno(), **{option: value})
        # Still open, and still the caller's to use.
        one.sendall(b"mine")
        assert other.recv(4) == b"mine"


def test_adopt_rejects_a_descriptor_that_is_not_a_socket(tmp_path):
    read_end, write_end = os.pipe()
    try:
        with open(tmp_path / "file", "w+b") as file:
            for fd in (file.fileno(), read_end, write_end):
                with pytest.raises(
                    ValueError,
                    match=rf"^Link\.adopt: fd {fd} is not an open socket descriptor$",
                ):
                    Link.adopt(fd)
            # Each is still open, and still the caller's to use.
            file.write(b"mine")
            os.write(write_end, b"mine")
            assert os.read(read_end, 4) == b"mine"
    finally:
        os.close(read_end)
        os.close(write_end)


def test_adopt_rejects_a_descriptor_that_a_link_already_owns():
    one, other = socket.socketpair()
    fd = one.detach()
    first = Link.adopt(fd, **QUIET)
    owned = rf"^Link\.adopt: fd {fd} already belongs to a Link$"
    with Link.adopt(other.detach(), **QUIET) as peer:
        with pytest.raises(ValueError, match=owned):
            Link.adopt(fd)
        # The link that owns it is as it was.
        peer.send_reliable(64, b"x")
        assert _receive(first).payload == b"x"
        first.close()
        # Closed, it owns the descriptor until the Link object goes.
        with pytest.raises(ValueError, match=owned):
            Link.adopt(fd)
    del first
    gc.collect()
    # The number is free again, here or wherever the next socket gets it.
    for _ in range(4):
        sender, receiver = _joined_links()
        with sender, receiver:
            assert sender.alive


def test_adopt_puts_the_descriptor_in_blocking_mode():
    one, other = socket.socketpair()
    # A socket with a timeout is in non-blocking mode, where a send fails once the socket's buffer is full.
    one.settimeout(5)
    fd = one.detach()
    assert os.get_blocking(fd) is False
    with Link.adopt(fd, **QUIET) as sender:
        with Link.adopt(other.detach(), **QUIET) as receiver:
            assert os.get_blocking(fd) is True
            for _ in range(4):
                sender.send_reliable(64, bytes(1 << 20))
            for _ in range(4):
                assert len(_receive(receiver).payload) == 1 << 20
            assert sender.alive


def test_latest_types_takes_any_iterable_of_type_numbers():
    for latest_types in (
        (),
        [96],
        (0, 255),
        range(96, 128),
        {96, 97},
        (number for number in (96, 97)),
    ):
        sender, receiver = _joined_links(latest_types=latest_types)
        with sender, receiver:
            assert sender.alive
    one, other = socket.socketpair()
    with one, other:
        for not_type_numbers in (96, ["96"], [96.0], "96", [None]):
            with pytest.raises(TypeError):
                Link.adopt(one.fileno(), latest_types=not_type_numbers)
        with pytest.raises(ValueError, match=r"^Link: latest_types must be in "):
            Link.adopt(one.fileno(), latest_types=[2**70])
        # Still open, and still the caller's to use.
        one.sendall(b"mine")
        assert other.recv(4) == b"mine"


def test_send_rejects_a_type_number_outside_0_to_255():
    sender, receiver = _joined_links()
    with sender, receiver:
        for send in (sender.send_reliable, sender.send_latest):
            for number in (-1, 256, 1000, -(2**40)):
                with pytest.raises(
                    ValueError, match=r"^Link: type must be in \[0, 255\]"
                ):
                    send(number, b"")
            for not_an_int in ("64", 64.0, None, b"@"):
                with pytest.raises(TypeError):
                    send(not_an_int, b"")
        assert sender.alive


def test_send_rejects_the_links_own_types_and_takes_every_other_number():
    sender, receiver = _joined_links()
    with sender, receiver:
        for send in (sender.send_reliable, sender.send_latest):
            for number in OWN_TYPES:
                with pytest.raises(ValueError, match="HEARTBEAT, PING, PONG"):
                    send(number, b"")
        others = [number for number in range(256) if number not in OWN_TYPES]
        for number in others:
            sender.send_reliable(number, bytes([number]))
        for number in others:
            message = _receive(receiver)
            assert type(message.type) is int
            assert (message.type, message.payload) == (number, bytes([number]))
        # None of the refused ones travelled.
        assert receiver.stats().messages_received == len(others)
        assert receiver.receive(0) is None


def test_send_rejects_a_payload_above_max_payload_size():
    sender, receiver = _joined_links()
    with sender, receiver:
        too_long = bytes(MAX_PAYLOAD_SIZE + 1)
        for send in (sender.send_reliable, sender.send_latest):
            with pytest.raises(
                ValueError,
                match=rf"^Link: payload of {MAX_PAYLOAD_SIZE + 1} bytes exceeds MAX_PAYLOAD_SIZE$",
            ):
                send(64, too_long)
        assert sender.alive
        sender.send_reliable(64, b"fits")
        assert _receive(receiver).payload == b"fits"
        assert receiver.stats().messages_received == 1


def test_send_takes_bytes_only():
    sender, receiver = _joined_links()
    with sender, receiver:
        for send in (sender.send_reliable, sender.send_latest):
            for not_bytes in (
                "text",
                bytearray(b"x"),
                memoryview(b"x"),
                [1, 2],
                5,
                None,
            ):
                with pytest.raises(TypeError):
                    send(64, not_bytes)
            with pytest.raises(TypeError):
                send(64)
        sender.send_reliable(64, b"bytes")
        assert _receive(receiver).payload == b"bytes"
        assert receiver.stats().messages_received == 1


def test_send_latest_takes_no_merge_callback():
    sender, receiver = _joined_links()
    with sender, receiver:
        with pytest.raises(TypeError):
            sender.send_latest(96, b"", lambda queued, incoming: incoming)
        with pytest.raises(TypeError):
            sender.send_latest(96, b"", merge=lambda queued, incoming: incoming)


def test_receive_and_close_reject_bad_arguments():
    link, _other = _joined_links()
    with link, _other:
        with pytest.raises(TypeError):
            link.receive()
        for not_an_int in (None, 1.5, "100"):
            with pytest.raises(TypeError):
                link.receive(not_an_int)
        for beyond_64_bits in (2**63, -(2**63) - 1):
            with pytest.raises(TypeError):
                link.receive(beyond_64_bits)
        with pytest.raises(ValueError, match="linger_ms"):
            link.close(-1)
        with pytest.raises(TypeError):
            link.close(None)
        assert link.alive
        assert link.receive(0) is None


def test_listener_rejects_a_bad_port():
    for port in (-1, 65536, 70000):
        with pytest.raises(
            ValueError, match=r"^LinkListener: port must be in \[0, 65535\]$"
        ):
            LinkListener("127.0.0.1", port)
    with pytest.raises(TypeError):
        LinkListener("127.0.0.1")
    with pytest.raises(TypeError):
        LinkListener("127.0.0.1", "47600")
    with pytest.raises(TypeError):
        LinkListener(None, 0)


def test_received_link_message_is_read_only():
    sender, receiver = _joined_links()
    with sender, receiver:
        sender.send_reliable(64, b"\x00payload\xff")
        message = _receive(receiver)
    assert type(message) is ReceivedLinkMessage
    assert (type(message.type), message.type) == (int, 64)
    assert (type(message.payload), message.payload) == (bytes, b"\x00payload\xff")
    assert message.payload is message.payload
    assert type(message.remote_send_ns) is int
    assert type(message.local_receive_ns) is int
    for name in ("type", "payload", "remote_send_ns", "local_receive_ns"):
        with pytest.raises(AttributeError):
            setattr(message, name, getattr(message, name))


def test_clock_estimate_and_link_stats_are_read_only():
    link, _other = _joined_links()
    with link, _other:
        clock = link.clock()
        stats = link.stats()
    assert type(clock) is ClockEstimate
    # Nothing was received: no sample yet.
    assert (clock.offset_ns, clock.two_way, clock.valid, clock.rtt_min_ns) == (
        0,
        False,
        False,
        -1,
    )
    for name in ("offset_ns", "two_way", "valid", "rtt_min_ns"):
        with pytest.raises(AttributeError):
            setattr(clock, name, getattr(clock, name))
    assert type(stats) is LinkStats
    counters = (
        "messages_sent",
        "messages_received",
        "latest_replaced",
        "unknown_received",
        "received_queued",
        "received_replaced",
    )
    for name in counters:
        assert (type(getattr(stats, name)), getattr(stats, name)) == (int, 0), name
        with pytest.raises(AttributeError):
            setattr(stats, name, 1)


def test_link_properties_are_read_only():
    link, _other = _joined_links()
    with link, _other:
        assert link.alive is True
        assert link.close_reason == ""
        assert link.peer_closed_before_any_message is False
        for name in ("alive", "close_reason", "peer_closed_before_any_message"):
            with pytest.raises(AttributeError):
                setattr(link, name, getattr(link, name))


def test_the_context_manager_returns_the_link_and_closes_it():
    link, other = _joined_links()
    with other:
        with link as entered:
            assert entered is link
            assert link.alive
        assert link.alive is False
        assert link.close_reason == "closed"
        with pytest.raises(KeyError):
            with link:
                raise KeyError("passes through __exit__")


def test_every_method_works_on_a_closed_link():
    link, other = _joined_links()
    with other:
        other.send_reliable(64, b"queued")
        assert _wait_for(lambda: link.stats().received_queued == 1)
        link.close()
        link.close()  # idempotent
        link.close(50)
        assert link.alive is False
        assert link.close_reason == "closed"
        assert link.peer_closed_before_any_message is False
        # Sends do nothing and raise nothing; argument errors are still raised.
        sent = link.stats().messages_sent
        link.send_reliable(64, b"ignored")
        link.send_latest(96, b"ignored")
        with pytest.raises(ValueError, match="type must be"):
            link.send_reliable(256, b"")
        assert link.stats().messages_sent == sent
        assert link.stats().latest_replaced == 0
        # receive() hands out what is queued, then None.
        assert link.stats().received_queued == 1
        assert link.receive(-1).payload == b"queued"
        assert link.receive(0) is None
        assert link.receive(-1) is None
        assert link.receive(20) is None
        assert link.clock().valid
        with link as entered:
            assert entered is link


@pytest.mark.skipif(not hasattr(os, "fork"), reason="needs os.fork()")
def test_a_forked_child_leaves_the_links_of_its_parent_alone():
    # The child has the Link objects but not the links' threads: closing a link there would wait for them
    # for good, and shut down the connection, which is the parent's.
    output = _run_python(
        """
import gc
import os
import socket
import sys
import time
import warnings

from isaaccapture.remote_link import Link

one, other = socket.socketpair()
sender = Link.adopt(one.detach())
receiver = Link.adopt(other.detach())
sender.send_reliable(64, b"before")
assert receiver.receive(10000).payload == b"before"

warnings.simplefilter("ignore", DeprecationWarning)  # for forking with threads
child = os.fork()
if child == 0:
    # Closes one link, drops the other, and finalizes what is left as it exits.
    sender.close()
    del receiver
    gc.collect()
    sys.exit(0)

exit_code = None
deadline = time.monotonic() + 20
while exit_code is None and time.monotonic() < deadline:
    pid, status = os.waitpid(child, os.WNOHANG)
    if pid:
        exit_code = os.waitstatus_to_exitcode(status)
    else:
        time.sleep(0.01)
if exit_code is None:
    os.kill(child, 9)
    os.waitpid(child, 0)
    raise SystemExit("the forked child hangs")
assert exit_code == 0, exit_code
assert sender.alive and receiver.alive, (sender.close_reason, receiver.close_reason)
sender.send_reliable(64, b"after")
assert receiver.receive(10000).payload == b"after"
print("the links of the parent live")
"""
    )
    assert output.endswith("the links of the parent live\n")


def test_a_link_that_cannot_start_its_threads_fails_without_holding_the_gil():
    pytest.importorskip("resource")
    output = _run_python(
        """
import resource
import socket
import threading
import time

from isaaccapture.remote_link import Link

gaps = []


def tick():
    last = time.monotonic()
    while True:
        time.sleep(0.001)
        now = time.monotonic()
        gaps.append(now - last)
        last = now


threading.Thread(target=tick, daemon=True).start()
one, other = socket.socketpair()
soft, hard = resource.getrlimit(resource.RLIMIT_NPROC)
# No thread starts from here on, unless the limit does not bind this user.
resource.setrlimit(resource.RLIMIT_NPROC, (1, hard))
gaps.clear()
start = time.monotonic()
try:
    Link.adopt(one.detach(), close_drain_ms=1000)
    print("started")
except RuntimeError:
    waited = time.monotonic() - start
    time.sleep(0.05)
    print(f"failed after {waited:.3f} s, longest stall {max(gaps):.3f} s")
finally:
    resource.setrlimit(resource.RLIMIT_NPROC, (soft, hard))
"""
    )
    if output.endswith("started\n"):
        pytest.skip("the limit on threads does not bind this user")
    failed = re.search(r"failed after (\S+) s, longest stall (\S+) s\n$", output)
    assert failed is not None, output
    # The link gave its sender, which never ran, close_drain_ms to finish ...
    assert float(failed[1]) >= 0.999
    # ... and the other Python threads ran meanwhile.
    assert float(failed[2]) < 0.5
