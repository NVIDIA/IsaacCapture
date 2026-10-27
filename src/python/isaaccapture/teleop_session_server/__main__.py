# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Teleop Session Server: CloudXR, the listeners, one OpenXR session and one robot app at a time.

The OpenXR session lives as long as the server; each robot app's trackers read through it while
that robot app is connected. Plugins are started on this machine by hand or with a script.
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import signal
import sys
import threading
import time
from collections.abc import Callable, Sequence
from contextlib import ExitStack
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

from isaaccapture import oxr
from isaaccapture.cloudxr import CloudXRLauncher
from isaaccapture.deviceio_session import RemoteSessionHost, RemoteSessionListener
from isaaccapture.logging_config._native_api import capture_native_output

from ._common import (
    BUSY_MESSAGE,
    DEFAULT_TCP_PORT,
    DEFAULT_WS_PORT,
    SHUTDOWN_MESSAGE,
    configure_logging,
)
from ._session import (
    UNTRACKED_HEADSET,
    RobotConnection,
    headset_status,
    served_extensions,
)

logger = logging.getLogger("isaaccapture.teleop_session_server")

PROG = "python -m isaaccapture.teleop_session_server"
OPENXR_APP_NAME = "TeleopSessionServer"
ACCEPT_POLL_MS = 200
HEALTH_CHECK_PERIOD_S = 1.0
OPENXR_RETRY_S = 1.0
# How long a robot app that connects while another is served waits for that one to leave. Above
# the link's 3 s peer timeout, so a robot app reconnecting after its own link dropped is served.
PENDING_GRACE_S = 4.0
# Refusals run beside the loop for up to about 3 s each; past this many, a connection is closed
# unanswered, which a robot app retries.
MAX_REFUSALS = 8
PLACEHOLDER_HANDLES = (1, 1, 1, 1)
_STOP_SIGNALS = (signal.SIGINT, signal.SIGTERM)
# The one xrGetSystem failure that means "no headset yet"; OpenXRSession's own wait retries only it.
_NO_HEADSET_ERROR = "Failed to get OpenXR system: XR_ERROR_FORM_FACTOR_UNAVAILABLE"


def _address_type(scheme: str, default_port: int) -> Callable[[str], tuple[str, int]]:
    """argparse type for ``[scheme://]HOST[:PORT]``: ``(host, port)``."""

    def parse(text: str) -> tuple[str, int]:
        url = text if "://" in text else f"{scheme}://{text}"
        try:
            parts = urlsplit(url)
            port = parts.port
        except ValueError as error:
            raise argparse.ArgumentTypeError(f"{text!r}: {error}") from None
        if (
            parts.scheme != scheme
            or not parts.hostname
            or parts.path not in ("", "/")
            or parts.query
            or parts.fragment
            or parts.username is not None
        ):
            raise argparse.ArgumentTypeError(
                f"{text!r} is not a {scheme}://HOST[:PORT] address"
            )
        return parts.hostname, default_port if port is None else port

    return parse


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=PROG,
        description=(
            "Serve this machine's headset, hand, controller and plugin data to one remote "
            "robot app at a time (TeleopSession in SessionMode.REMOTE). Start the plugins "
            "on this machine by hand or with a script."
        ),
        epilog=(
            "Robot apps are not authenticated and the link is not encrypted: run the "
            "server on a trusted network or VPN, and open its ports only toward the "
            "robot machine."
        ),
    )
    parser.add_argument(
        "--listen",
        type=_address_type("tcp", DEFAULT_TCP_PORT),
        default=f"tcp://0.0.0.0:{DEFAULT_TCP_PORT}",
        metavar="tcp://HOST:PORT",
        help=(
            "TCP address robot apps connect to with tcp://HOST:PORT or HOST:PORT "
            "(default: %(default)s); port 0 picks one."
        ),
    )
    parser.add_argument(
        "--ws-listen",
        type=_address_type("ws", DEFAULT_WS_PORT),
        default=None,
        metavar="ws://HOST:PORT",
        help=(
            f"Also accept WebSocket connections, e.g. ws://0.0.0.0:{DEFAULT_WS_PORT} "
            "(default: off); robot apps connect to it with ws://HOST:PORT. They cannot "
            "connect with wss:// yet; to encrypt the link, tunnel it between the two "
            "machines (VPN, SSH or stunnel)."
        ),
    )
    parser.add_argument("--verbose", action="store_true", help="Log debug output.")
    cloudxr = parser.add_argument_group("CloudXR")
    CloudXRLauncher.add_cloudxr_install_dir_argument(cloudxr)
    CloudXRLauncher.add_cloudxr_device_profile_argument(cloudxr)
    CloudXRLauncher.add_cloudxr_env_config_argument(cloudxr)
    CloudXRLauncher.add_accept_eula_argument(cloudxr)
    CloudXRLauncher.add_host_client_argument(cloudxr)
    CloudXRLauncher.add_launch_cloudxr_runtime_argument(cloudxr)
    # Tests only: no OpenXR session, placeholder handles, READY at once; only KeyboardTracker.
    parser.add_argument(
        "--placeholder-openxr", action="store_true", help=argparse.SUPPRESS
    )
    return parser


@dataclass
class _Listening:
    listener: Any
    websocket: bool
    url: str


def _open_listeners(args: argparse.Namespace) -> list[_Listening]:
    """Bind every requested address; raises RuntimeError when one cannot be bound."""
    requested = [("tcp", args.listen, False)]
    if args.ws_listen is not None:
        requested.append(("ws", args.ws_listen, True))
    listening = []
    for scheme, (host, port), websocket in requested:
        listener = RemoteSessionListener(host, port)
        shown = f"[{host}]" if ":" in host else host
        listening.append(
            _Listening(listener, websocket, f"{scheme}://{shown}:{listener.port}")
        )
    return listening


class _StopRequest:
    """SIGINT/SIGTERM handler: the server stops at its next check, within about 0.2 s."""

    def __init__(self) -> None:
        self.signal: int | None = None

    def __call__(self, signum: int, frame: Any) -> None:
        self.signal = signum


class _ServerFailure(Exception):
    """Stops the server with exit code 1; ``robot_text`` goes to the robot app as ERROR."""

    def __init__(self, text: str, robot_text: str | None = None) -> None:
        super().__init__(text)
        self.robot_text = text if robot_text is None else robot_text


@dataclass
class _Pending:
    """A robot app that connected while another is served; its connection is not read yet."""

    listening: _Listening
    fd: int
    peer: str
    deadline: float


def _reject(fd: int, websocket: bool, peer: str, text: str) -> None:
    """Refuse a connection with ``text``; closes ``fd``."""
    try:
        RemoteSessionListener.reject(fd, websocket, text)
    except RuntimeError as error:
        logger.warning("Could not send the refusal to %s: %s", peer, error)


class _Server:
    """The loop: accepts robot apps, keeps the OpenXR session and steps the robot app it serves."""

    def __init__(
        self,
        listening: Sequence[_Listening],
        launcher: Any,
        stop: _StopRequest,
        *,
        placeholder: bool = False,
    ) -> None:
        self._listening = list(listening)
        self._launcher = launcher
        self._stop = stop
        self._placeholder = placeholder
        self._extensions = None if placeholder else served_extensions()
        self._openxr: Any = None
        # The OpenXR session's, once it exists.
        self._handles = (
            oxr.OpenXRSessionHandles(*PLACEHOLDER_HANDLES) if placeholder else None
        )
        self._robot: RobotConnection | None = None
        # Only while a robot app is served.
        self._pending: _Pending | None = None
        self._refusals: list[threading.Thread] = []
        self._next_health_check = -math.inf
        self._next_openxr_attempt = -math.inf
        self._waiting_for_headset = False

    def run(self) -> int:
        """Serve until stopped by a signal (0), or until CloudXR or OpenXR fails (1)."""
        robot_error, code = SHUTDOWN_MESSAGE, 0
        try:
            while self._stop.signal is None:
                self._check_health()
                self._open_openxr()
                self._step_robot()
                self._wait(self._next_due())
            logger.info(
                "Received %s; stopping the Teleop Session Server",
                signal.Signals(self._stop.signal).name,
            )
        except _ServerFailure as failure:
            logger.error("%s; stopping the Teleop Session Server", failure)
            robot_error, code = failure.robot_text, 1
        finally:
            self._shut_down(robot_error)
        return code

    def _check_health(self) -> None:
        """Every HEALTH_CHECK_PERIOD_S: CloudXR, and the OpenXR session between ticks."""
        now = time.monotonic()
        if now < self._next_health_check:
            return
        self._next_health_check = now + HEALTH_CHECK_PERIOD_S
        try:
            self._launcher.health_check()
        except RuntimeError as error:
            raise _ServerFailure(str(error), SHUTDOWN_MESSAGE) from error
        if self._openxr is not None:
            # Also takes in the runtime's events while no robot app ticks.
            self._headset()

    def _open_openxr(self) -> None:
        """Open the OpenXR session, retried every OPENXR_RETRY_S while no headset is connected."""
        now = time.monotonic()
        if self._handles is not None or now < self._next_openxr_attempt:
            return
        self._next_openxr_attempt = now + OPENXR_RETRY_S
        try:
            session = oxr.OpenXRSession(
                OPENXR_APP_NAME, self._extensions, wait_for_system=False
            )
        except RuntimeError as error:
            if not str(error).startswith(_NO_HEADSET_ERROR):
                raise _ServerFailure(
                    f"The Teleop Session Server could not open an OpenXR session: {error}"
                ) from error
            if not self._waiting_for_headset:
                self._waiting_for_headset = True
                logger.info("No headset is connected to CloudXR; waiting for one")
            return
        self._openxr = session
        self._handles = session.get_handles()
        logger.info("Opened the OpenXR session")
        logger.debug("OpenXR extensions: %s", self._extensions)

    def _headset(self) -> tuple[dict, bool]:
        """The STATUS headset entry and whether the headset is connected."""
        if self._openxr is None:
            return UNTRACKED_HEADSET, False
        snapshot = self._openxr.get_provider_snapshot()
        if snapshot.state == oxr.OpenXRProviderState.FAILED:
            raise _ServerFailure(
                "The OpenXR session on the Teleop Session Server failed "
                f"({snapshot.reason.name}): {snapshot.error}"
            )
        connected = snapshot.headset_state == oxr.OpenXRHeadsetState.CONNECTED
        return headset_status(snapshot), connected

    def _step_robot(self) -> None:
        """Step the robot app when due; serve or refuse the pending one."""
        robot = self._robot
        pending = self._pending
        now = time.monotonic()
        # At the pending deadline a robot app whose link is down is stepped, so it ends now.
        if robot is not None and (
            now >= robot.next_step
            or (
                pending is not None and now >= pending.deadline and not robot.host.alive
            )
        ):
            try:
                robot.step(self._handles, self._headset)
            except _ServerFailure:
                raise
            except Exception as error:
                robot.fail(error)
            if robot.ended:
                self._robot = None
                robot.close()
        if pending is None:
            return
        if self._robot is None:
            self._pending = None
            self._serve(pending.listening, pending.fd, pending.peer)
        elif time.monotonic() >= pending.deadline:
            self._pending = None
            logger.warning(
                "Refused the robot app at %s: the one at %s is connected",
                pending.peer,
                self._robot.peer,
            )
            self._refuse(pending.listening, pending.fd, pending.peer, BUSY_MESSAGE)

    def _next_due(self) -> float:
        due = [self._next_health_check]
        if self._handles is None:
            due.append(self._next_openxr_attempt)
        if self._robot is not None:
            due.append(self._robot.next_step)
        if self._pending is not None:
            due.append(self._pending.deadline)
        return min(due)

    def _wait(self, until: float) -> None:
        """Accept connections until ``until`` (time.monotonic()), for ACCEPT_POLL_MS at most."""
        wait_ms = min(ACCEPT_POLL_MS, max(0.0, (until - time.monotonic()) * 1000))
        share_ms = int(wait_ms // len(self._listening))
        for listening in self._listening:
            accepted = listening.listener.accept(share_ms)
            if accepted is not None:
                self._admit(listening, *accepted)
        if share_ms == 0:
            # Under a millisecond per listener, which accept() cannot wait.
            time.sleep(wait_ms / 1000)

    def _admit(self, listening: _Listening, fd: int, peer: str) -> None:
        if self._robot is None:
            self._serve(listening, fd, peer)
        elif self._pending is None:
            self._pending = _Pending(
                listening, fd, peer, time.monotonic() + PENDING_GRACE_S
            )
            logger.info(
                "Robot app at %s waits up to %g s for the one at %s to leave",
                peer,
                PENDING_GRACE_S,
                self._robot.peer,
            )
        else:
            logger.warning(
                "Refused the robot app at %s: the one at %s is connected",
                peer,
                self._robot.peer,
            )
            self._refuse(listening, fd, peer, BUSY_MESSAGE)

    def _serve(self, listening: _Listening, fd: int, peer: str) -> None:
        host = RemoteSessionHost.adopt(fd, listening.websocket)
        self._robot = RobotConnection(
            host, peer, placeholder=self._placeholder, extensions=self._extensions
        )
        logger.info("Robot app connected from %s on %s", peer, listening.url)

    def _refuse(self, listening: _Listening, fd: int, peer: str, text: str) -> None:
        """Refuse a connection with ``text`` beside the loop, which keeps ticking meanwhile."""
        self._refusals = [thread for thread in self._refusals if thread.is_alive()]
        if len(self._refusals) >= MAX_REFUSALS:
            logger.warning(
                "Closed the connection from %s unanswered: %d refusals are in progress",
                peer,
                len(self._refusals),
            )
            os.close(fd)
            return
        thread = threading.Thread(
            target=_reject,
            args=(fd, listening.websocket, peer, text),
            name=f"refuse {peer}",
            daemon=True,
        )
        thread.start()
        self._refusals.append(thread)

    def _refuse_stopping(self, listening: _Listening, fd: int, peer: str) -> None:
        logger.info(
            "Refused the robot app at %s: the Teleop Session Server is stopping", peer
        )
        self._refuse(listening, fd, peer, SHUTDOWN_MESSAGE)

    def _shut_down(self, robot_error: str) -> None:
        """Refuse the waiting robot apps, end the served one, close the OpenXR session.

        Listening stops, so a robot app that connects afterwards is refused by the kernel and
        keeps retrying until a restarted server serves it.
        """
        with ExitStack() as stack:
            stack.callback(self._join_refusals)
            stack.callback(self._close_openxr)
            robot, self._robot = self._robot, None
            if robot is not None:
                # The trackers read through the OpenXR session, so they close first.
                stack.callback(robot.close, None if robot.ended else robot_error)
            pending, self._pending = self._pending, None
            if pending is not None:
                self._refuse_stopping(pending.listening, pending.fd, pending.peer)
            for listening in self._listening:
                while (accepted := listening.listener.accept(0)) is not None:
                    self._refuse_stopping(listening, *accepted)
                listening.listener.close()

    def _close_openxr(self) -> None:
        openxr, self._openxr = self._openxr, None
        if openxr is not None:
            openxr.close()

    def _join_refusals(self) -> None:
        for thread in self._refusals:
            thread.join()
        self._refusals = []


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    configure_logging(args.verbose)
    if not sys.platform.startswith("linux"):
        logger.error("The Teleop Session Server runs on Linux only")
        return 2
    try:
        listening = _open_listeners(args)
    except RuntimeError as error:
        logger.error("%s", error)
        return 1
    if args.placeholder_openxr:
        logger.warning(
            "--placeholder-openxr: no OpenXR session; only KeyboardTracker works"
        )

    try:
        launcher = CloudXRLauncher.launch_context(args)
    except (RuntimeError, ValueError) as error:
        logger.error("CloudXR is not available: %s", error)
        for item in listening:
            item.listener.close()
        return 1
    stop = _StopRequest()
    previous = {sig: signal.signal(sig, stop) for sig in _STOP_SIGNALS}
    try:
        with launcher:
            server = _Server(
                listening, launcher, stop, placeholder=args.placeholder_openxr
            )
            for item in listening:
                print(f"Teleop Session Server listening on {item.url}", flush=True)
            # This process is ours, so the OpenXR runtime's raw output goes to the capture file
            # for as long as the server runs; Python's own output stays on the terminal.
            with capture_native_output():
                return server.run()
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


if __name__ == "__main__":
    sys.exit(main())
