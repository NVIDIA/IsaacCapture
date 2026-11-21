# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Teleop Session Server: CloudXR, the listeners, one OpenXR session and one robot app at a time.

Each robot app's trackers read through the OpenXR session while that robot app is connected.
While CloudXR or the OpenXR session is missing the server serves robot apps without headset
data and opens a new OpenXR session by itself. Plugins are started on this machine by hand or
with a script.
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
from isaaccapture.robot_state import RobotStateStore

from . import _cloudxr, _session
from ._common import (
    BUSY_MESSAGE,
    DEFAULT_TCP_PORT,
    DEFAULT_WS_PORT,
    RETRY_S,
    SHUTDOWN_MESSAGE,
    Backoff,
    configure_logging,
)
from ._session import (
    UNTRACKED_HEADSET,
    WAITING_FOR_HEADSET,
    HelloError,
    RobotConnection,
    TrackerUpdateError,
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
# One robot per reader socket path: the ERROR for a robot app that offers robot state while
# another program serves the path.
ROBOT_STATE_TAKEN_MESSAGE = (
    "Another program on the Teleop Session Server's machine already serves a robot's state "
    "at {path}. One robot per machine is served; stop that program, or connect this robot "
    "app without a robot description."
)
# The STATUS message of a server started with --no-headset.
NO_HEADSET_MESSAGE = "this Teleop Session Server runs without a headset (--no-headset)"
# Follows the reason in the warning about a headset side that cannot be used.
_KEEPS_SERVING = (
    "Robot apps are served without headset data until this is fixed; the Teleop Session "
    "Server keeps checking and needs no restart"
)


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
    parser.add_argument(
        "--robot-state-socket",
        default=None,
        metavar="PATH",
        help=(
            "Local socket on which reader programs on this machine (isaaccapture.robot_state) "
            "get the robot state a robot app sends; created when the first robot app sends "
            "some (default: robot_state.sock in this user's isaaccapture runtime directory). "
            "Its directory must belong to this user and be closed to group and others."
        ),
    )
    parser.add_argument(
        "--no-headset",
        action="store_true",
        help=(
            "Serve robot apps and their robot state without a headset: CloudXR is not "
            "started or attached and OpenXR is never called, so robot apps get no head, "
            "hand, controller or plugin data."
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
    """The loop: accepts robot apps, keeps the OpenXR session and steps the robot app it serves.

    ``launcher`` is CloudXR: anything with ``health_check()``, which raises while CloudXR cannot
    be used; the warning names its ``start_command`` when it has one. With ``headset=False`` it
    may be None: CloudXR and OpenXR are never called.
    """

    def __init__(
        self,
        listening: Sequence[_Listening],
        launcher: Any,
        stop: _StopRequest,
        *,
        placeholder: bool = False,
        robot_state_socket: str | None = None,
        headset: bool = True,
    ) -> None:
        self._listening = list(listening)
        self._launcher = launcher
        self._stop = stop
        self._placeholder = placeholder
        # False with --no-headset.
        self._headset_enabled = headset
        # Robot state of the robot app served, kept across robot apps for the server's life.
        self._robot_state = RobotStateStore()
        _session.set_hosted_robot_state(self._robot_state)
        # None: the default path. The service exists from the first robot app that sends robot state.
        self._robot_state_socket = robot_state_socket
        self._robot_state_service: Any = None
        # True from the first robot app that agreed robot state: the socket is then kept open.
        self._robot_state_wanted = False
        # The error text of the last warning; None after the socket was opened.
        self._robot_state_error: str | None = None
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
        # Why there is no OpenXR session, as robot apps read it in STATUS: "" before the first
        # try, None while the session is open.
        self._no_headset: str | None = "" if headset else NO_HEADSET_MESSAGE
        # Why CloudXR cannot be used, from the last check; None while it can.
        self._cloudxr_error: str | None = None
        self._backoff = Backoff()

    def run(self) -> int:
        """Serve until stopped by a signal; returns 0. A failing CloudXR or OpenXR does not stop it."""
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
        finally:
            self._shut_down(SHUTDOWN_MESSAGE)
        return 0

    def _check_health(self) -> None:
        """Every HEALTH_CHECK_PERIOD_S: CloudXR, the OpenXR session between ticks, and the
        reader socket while it could not be opened."""
        now = time.monotonic()
        if now < self._next_health_check:
            return
        self._next_health_check = now + HEALTH_CHECK_PERIOD_S
        if self._headset_enabled:
            self._check_cloudxr()
            if self._openxr is not None:
                # Also takes in the runtime's events while no robot app ticks.
                self._headset()
        if self._robot_state_wanted:
            self._serve_robot_state(retry=True)

    def _check_cloudxr(self) -> None:
        """Ask whether CloudXR can be used; take the headset side down when it stops."""
        try:
            self._launcher.health_check()
            error = None
        except Exception as failure:
            error = str(failure) or type(failure).__name__
        if error == self._cloudxr_error:
            return
        self._cloudxr_error = error
        if self._placeholder:
            # Its trackers read through no OpenXR session.
            if error is not None:
                logger.warning(
                    "%s; --placeholder-openxr serves on without CloudXR", error
                )
        elif error is not None:
            # The check's result changed, so this stop is a new one.
            self._headset_down(error, log_always=True)
        else:
            logger.info("CloudXR is available again")
            self._next_openxr_attempt = time.monotonic()

    def _open_openxr(self) -> None:
        """Open the OpenXR session when a try is due; a failure sets the reason and the next try.

        Tried every OPENXR_RETRY_S while no headset is connected, and not at all while CloudXR
        cannot be used.
        """
        now = time.monotonic()
        if (
            not self._headset_enabled
            or self._handles is not None
            or self._cloudxr_error is not None
            or now < self._next_openxr_attempt
        ):
            return
        session = None
        try:
            session = oxr.OpenXRSession(
                OPENXR_APP_NAME, self._extensions, wait_for_system=False
            )
            handles = session.get_handles()
        except Exception as error:
            if session is not None:
                self._openxr = session
                self._close_openxr()
            if isinstance(error, RuntimeError) and str(error).startswith(
                _NO_HEADSET_ERROR
            ):
                self._next_openxr_attempt = now + OPENXR_RETRY_S
                self._report(WAITING_FOR_HEADSET)
            else:
                reason = f"The Teleop Session Server could not open an OpenXR session: {error}"
                self._next_openxr_attempt = now + self._backoff.delay(reason, now)
                self._report(reason)
            return
        self._openxr = session
        self._handles = handles
        self._report(None)
        logger.debug("OpenXR extensions: %s", self._extensions)

    def _report(self, reason: str | None, *, log: bool = True) -> None:
        """Record why there is no OpenXR session (None: it is open); log each change once."""
        if reason == self._no_headset:
            return
        self._no_headset = reason
        if not log:
            return
        if reason is None:
            logger.info("Opened the OpenXR session")
        elif reason == WAITING_FOR_HEADSET:
            logger.info("No headset is connected to CloudXR; waiting for one")
        else:
            hint = ""
            # The command helps only while CloudXR is what is missing.
            if self._cloudxr_error is not None:
                command = getattr(
                    self._launcher, "start_command", _cloudxr.START_COMMAND
                )
                hint = f". Start CloudXR with: {command}"
            logger.warning("%s. %s%s", reason.rstrip("."), _KEEPS_SERVING, hint)

    def _headset(self) -> tuple[dict, bool]:
        """The STATUS headset entry and whether the headset is connected.

        An OpenXR session that reports a failure is taken down; never raises.
        """
        if self._openxr is None:
            return UNTRACKED_HEADSET, False
        try:
            snapshot = self._openxr.get_provider_snapshot()
        except Exception as error:
            self._openxr_lost(
                f"The OpenXR session on the Teleop Session Server failed: {error}"
            )
            return UNTRACKED_HEADSET, False
        if snapshot.state == oxr.OpenXRProviderState.FAILED:
            self._openxr_lost(
                "The OpenXR session on the Teleop Session Server failed "
                f"({snapshot.reason.name}): {snapshot.error}"
            )
            return UNTRACKED_HEADSET, False
        connected = snapshot.headset_state == oxr.OpenXRHeadsetState.CONNECTED
        return headset_status(snapshot), connected

    def _headset_down(self, reason: str, *, log_always: bool = False) -> None:
        """Stop the robot app's trackers, then close the OpenXR session they read through.

        The trackers call into the OpenXR runtime's library, which is unloaded when the session
        closes, so they go first. The robot app stays connected and is told ``reason`` in
        STATUS. Never raises.
        """
        robot = self._robot
        if robot is not None:
            try:
                robot.stop_trackers()
            except Exception as error:
                logger.error(
                    "Robot app at %s: the trackers could not be stopped",
                    robot.peer,
                    exc_info=error,
                )
                robot.ended = True
            if robot.ended:
                # Its trackers could not be stopped; closing the connection ends them.
                self._robot = None
                try:
                    robot.close()
                except Exception as error:
                    logger.warning(
                        "Closing the robot app at %s failed: %s", robot.peer, error
                    )
        self._handles = None
        self._close_openxr()
        now = time.monotonic()
        delay = self._backoff.delay(reason, now)
        self._next_openxr_attempt = now + delay
        # A failure that repeats is logged the first time only; a stop of CloudXR every time.
        self._report(reason, log=log_always or delay == RETRY_S)

    def _openxr_lost(self, reason: str) -> None:
        """The OpenXR session failed or a tick through it did: take the headset side down.

        A stopped CloudXR is the usual cause and the clearer reason, so it is checked first.
        """
        self._check_cloudxr()
        if self._openxr is not None:
            self._headset_down(reason)

    def _trackers_lost(self, robot: RobotConnection, text: str) -> None:
        """A tick failed and the robot app's trackers are stopped."""
        if self._openxr is None:
            # --placeholder-openxr: only the trackers are started again.
            logger.error(
                "Robot app at %s: %s. It stays connected without tracker data; the "
                "server starts the trackers again by itself",
                robot.peer,
                text,
            )
        else:
            self._openxr_lost(text)

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
                robot.step(
                    self._handles,
                    self._headset,
                    self._no_headset or WAITING_FOR_HEADSET,
                )
            except TrackerUpdateError as failure:
                self._trackers_lost(robot, str(failure))
            except Exception as error:
                robot.fail(error)
            # A robot app whose trackers could not be stopped is closed in _headset_down().
            if robot.ended and self._robot is robot:
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
        if (
            self._headset_enabled
            and self._handles is None
            and self._cloudxr_error is None
        ):
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
            host,
            peer,
            placeholder=self._placeholder,
            extensions=self._extensions,
            robot_state=self._robot_state,
            on_robot_state=self._serve_robot_state,
        )
        logger.info("Robot app connected from %s on %s", peer, listening.url)

    def _serve_robot_state(self, retry: bool = False) -> None:
        """Open the local socket for reader programs, for a robot app whose HELLO offers robot
        state.

        HelloError when another program serves robot state at the socket's path. With ``retry``
        (the health check, while the socket is not served) that is a warning like any other
        failure.
        """
        service = self._robot_state_service
        if service is not None:
            if service.serving:
                return
            self._robot_state_service = None
            service.close()
        from isaaccapture.robot_state import RobotStateLocalService
        from isaaccapture.robot_state._local import ReaderSocketTaken

        service = RobotStateLocalService(self._robot_state, self._robot_state_socket)
        try:
            service.start()
        except (OSError, RuntimeError) as error:
            if isinstance(error, ReaderSocketTaken) and not retry:
                raise HelloError(
                    ROBOT_STATE_TAKEN_MESSAGE.format(path=service.path)
                ) from error
            # One warning per reason, however often the socket is tried.
            if str(error) != self._robot_state_error:
                self._robot_state_error = str(error)
                logger.warning(
                    "Reader programs cannot get robot state from this Teleop Session Server: %s",
                    error,
                )
        else:
            self._robot_state_error = None
            self._robot_state_service = service
        # Not refused: the socket is served, or tried again by the health check, from here on.
        self._robot_state_wanted = True

    def _close_robot_state(self) -> None:
        service, self._robot_state_service = self._robot_state_service, None
        if service is not None:
            service.close()
        if _session._hosted_robot_state is self._robot_state:
            _session.set_hosted_robot_state(None)

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
            # After the robot app's connection, which ends its robot state.
            stack.callback(self._close_robot_state)
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
        if openxr is None:
            return
        try:
            openxr.close()
        except Exception as error:
            logger.warning("Closing the OpenXR session failed: %s", error)

    def _join_refusals(self) -> None:
        for thread in self._refusals:
            thread.join()
        self._refusals = []


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.no_headset and args.placeholder_openxr:
        parser.error("--no-headset cannot be combined with --placeholder-openxr")
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

    if args.no_headset:
        logger.info(
            "--no-headset: no headset is served; CloudXR and OpenXR are not used"
        )
        cloudxr = None
    else:
        # A CloudXR that cannot be started does not stop the server.
        cloudxr = _cloudxr.start(args)
    stop = _StopRequest()
    previous = {sig: signal.signal(sig, stop) for sig in _STOP_SIGNALS}
    try:
        with ExitStack() as stack:
            if cloudxr is not None:
                stack.enter_context(cloudxr)
            server = _Server(
                listening,
                cloudxr,
                stop,
                placeholder=args.placeholder_openxr,
                robot_state_socket=args.robot_state_socket,
                headset=not args.no_headset,
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
