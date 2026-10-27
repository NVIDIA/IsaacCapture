# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Teleop Session Server main process: CloudXR, the listeners, and one session process at a time.

It never opens OpenXR itself; each connected robot app gets a session process (``_session.py``)
for that, in its own process group so that plugins it leaves behind can be killed with it.
"""

from __future__ import annotations

import argparse
import contextlib
import logging
import math
import os
import select
import signal
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from isaaccapture.cloudxr import CloudXRLauncher
from isaaccapture.deviceio_session import RemoteSessionListener
from isaaccapture.teleop_session_manager import BUNDLED_PLUGINS_DIR

from ._common import (
    BUSY_MESSAGE,
    DEFAULT_TCP_PORT,
    DEFAULT_WS_PORT,
    SHUTDOWN_MESSAGE,
    configure_logging,
)

logger = logging.getLogger("isaaccapture.teleop_session_server")

PROG = "python -m isaaccapture.teleop_session_server"
ACCEPT_POLL_MS = 200
HEALTH_CHECK_PERIOD_S = 1.0
# Longest a wait for a session process goes without checking for a stop signal.
WAIT_SLICE_S = 0.1
# After SIGTERM, at least. A session serving a robot app reports how long its teardown may take
# (each plugin's shutdown_timeout_seconds, default 2 s, plus the host's close) and gets that long
# when it is more.
SESSION_STOP_TIMEOUT_S = 10.0
# How long a new robot app waits for a session process that is starting (its HELLO wait is
# 5 s) or ending; an ending one gets its reported teardown time when that is more.
HANDOVER_TIMEOUT_S = 10.0
# How long a session serving a robot app has to report it gone before a new one is refused:
# one tick at 1 Hz, or one OpenXR retry.
BUSY_GRACE_S = 1.5
_SESSION_MODULE = "isaaccapture.teleop_session_server._session"
_STOP_SIGNALS = (signal.SIGINT, signal.SIGTERM)


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
            "robot app at a time (TeleopSession in SessionMode.REMOTE)."
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
        "--plugin-dir",
        type=Path,
        action="append",
        default=[],
        metavar="DIR",
        help=(
            "Directory of plugins robot apps may start by name, searched after the bundled "
            "plugins and ISAAC_TELEOP_PLUGIN_PATH. Repeatable; for a plugin name found in "
            "several directories the last one wins."
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


def plugin_dirs(extra: Sequence[Path]) -> list[Path]:
    """Bundled plugins, ISAAC_TELEOP_PLUGIN_PATH, then ``extra``; a later directory wins."""
    from_env = [
        Path(entry)
        for entry in os.environ.get("ISAAC_TELEOP_PLUGIN_PATH", "").split(os.pathsep)
        if entry.strip()
    ]
    dirs = [
        Path(path).expanduser().resolve()
        for path in (BUNDLED_PLUGINS_DIR, *from_env, *extra)
    ]
    # Keep each directory at its last position, which is what decides the winner.
    return list(reversed(dict.fromkeys(reversed(dirs))))


class _StopRequest:
    """SIGINT/SIGTERM handler: the server stops at its next check.

    That comes within about 0.2 s, or once a refusal in progress ends (about 3 s at most). A
    further signal while the server waits for its session process to stop kills that process.
    Python runs the handler once for signals that arrive during one native call, so ``count``
    also counts the bytes each delivery writes to the signal wakeup fd, when there is one.
    """

    def __init__(self, wakeup_fd: int | None = None) -> None:
        self.signal: int | None = None
        self._wakeup_fd = wakeup_fd
        self._handled = 0
        self._delivered = 0

    def __call__(self, signum: int, frame: Any) -> None:
        self.signal = signum
        self._handled += 1

    @property
    def count(self) -> int:
        """Stop signals received so far."""
        if self._wakeup_fd is not None:
            with contextlib.suppress(BlockingIOError):
                while delivered := os.read(self._wakeup_fd, 64):
                    self._delivered += sum(sig in _STOP_SIGNALS for sig in delivered)
        return max(self._handled, self._delivered)


def _signal_process(pid: int, sig: int) -> None:
    try:
        os.kill(pid, sig)
    except ProcessLookupError:
        pass


def _signal_group(pgid: int, sig: int) -> None:
    try:
        os.killpg(pgid, sig)
    except (ProcessLookupError, PermissionError):
        pass


def _reject(fd: int, websocket: bool, peer: str, text: str) -> None:
    """Refuse a connection with ``text``; closes ``fd``."""
    try:
        RemoteSessionListener.reject(fd, websocket, text)
    except RuntimeError as error:
        logger.warning("Could not send the refusal to %s: %s", peer, error)


@dataclass
class _SessionProcess:
    """A session process and the pipe it reports through (see ``_session._ServingPipe``)."""

    popen: subprocess.Popen
    peer: str
    # Read end: the teardown time once the session serves its robot app; EOF once it is ending.
    serving_fd: int
    serving: bool = False
    ending: bool = False
    teardown_s: float = 0.0

    def read_reports(self, timeout_s: float) -> None:
        """Take in what the session process reported, waiting up to ``timeout_s`` for news."""
        poller = select.poll()
        poller.register(self.serving_fd, select.POLLIN)
        if poller.poll(max(0, round(timeout_s * 1000))):
            report = os.read(self.serving_fd, 64)
            if report:
                self.serving = True
                with contextlib.suppress(ValueError):
                    self.teardown_s = float(report)
            else:
                self.ending = True

    @property
    def stop_timeout_s(self) -> float:
        return max(SESSION_STOP_TIMEOUT_S, self.teardown_s)

    @property
    def handover_timeout_s(self) -> float:
        return max(HANDOVER_TIMEOUT_S, self.teardown_s)


class _Server:
    """The accept loop; one session process at a time."""

    def __init__(
        self,
        listening: Sequence[_Listening],
        session_command: Sequence[str],
        launcher: Any,
        stop: _StopRequest,
    ) -> None:
        self._listening = list(listening)
        self._session_command = list(session_command)
        self._launcher = launcher
        self._stop = stop
        self._session: _SessionProcess | None = None
        # Set once the server is stopping.
        self._exit_code: int | None = None
        # Stop signals that do not kill a stopping session process: the one that stopped the server.
        self._stop_signals_spent = 0
        self._next_health_check = -math.inf

    def run(self) -> int:
        """Serve until stopped by a signal (0) or until CloudXR stops (1)."""
        try:
            return self._serve()
        finally:
            try:
                self._close_listeners()
            finally:
                self._end_session()

    def _serve(self) -> int:
        accept_ms = max(1, ACCEPT_POLL_MS // len(self._listening))
        while not self._stopping():
            self._reap()
            for listening in self._listening:
                accepted = listening.listener.accept(accept_ms)
                if accepted is not None:
                    self._admit(listening, *accepted)
        return self._exit_code

    def _stopping(self) -> bool:
        """True once a stop signal arrived or CloudXR stopped; checks CloudXR when it is due."""
        if self._exit_code is not None:
            return True
        if self._stop.signal is not None:
            logger.info(
                "Received %s; stopping the Teleop Session Server",
                signal.Signals(self._stop.signal).name,
            )
            self._exit_code = 0
            self._stop_signals_spent = 1
        elif time.monotonic() >= self._next_health_check:
            self._next_health_check = time.monotonic() + HEALTH_CHECK_PERIOD_S
            try:
                self._launcher.health_check()
            except RuntimeError as error:
                logger.error("%s; stopping the Teleop Session Server", error)
                self._exit_code = 1
        return self._exit_code is not None

    def _close_listeners(self) -> None:
        """Refuse the robot apps still in the listen backlog, then stop listening.

        A robot app that connects afterwards is refused by the kernel, so it keeps retrying
        and a restarted server serves it.
        """
        for listening in self._listening:
            while (accepted := listening.listener.accept(0)) is not None:
                self._refuse_stopping(listening, *accepted)
            listening.listener.close()

    def _refuse_stopping(self, listening: _Listening, fd: int, peer: str) -> None:
        logger.info(
            "Refused the robot app at %s: the Teleop Session Server is stopping", peer
        )
        _reject(fd, listening.websocket, peer, SHUTDOWN_MESSAGE)

    def _admit(self, listening: _Listening, fd: int, peer: str) -> None:
        if not self._make_room(peer):
            if self._exit_code is not None:
                self._refuse_stopping(listening, fd, peer)
                return
            logger.warning(
                "Refused the robot app at %s: the one at %s is connected",
                peer,
                self._session.peer,
            )
            _reject(fd, listening.websocket, peer, BUSY_MESSAGE)
            return
        serving_read, serving_write = os.pipe()
        command = [
            *self._session_command,
            "--fd",
            str(fd),
            "--peer",
            peer,
            "--serving-fd",
            str(serving_write),
        ]
        if listening.websocket:
            command.append("--websocket")
        try:
            # From the main thread: PR_SET_PDEATHSIG fires when the forking thread exits.
            popen = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                pass_fds=(fd, serving_write),
                start_new_session=True,
            )
        except OSError as error:
            os.close(serving_read)
            logger.error("Could not start a session process for %s: %s", peer, error)
            _reject(
                fd,
                listening.websocket,
                peer,
                f"The Teleop Session Server could not start a session: {error}",
            )
            return
        finally:
            os.close(serving_write)
        os.close(fd)
        self._session = _SessionProcess(popen, peer, serving_read)
        logger.info(
            "Robot app connected from %s on %s; session process %d",
            peer,
            listening.url,
            popen.pid,
        )

    def _make_room(self, peer: str) -> bool:
        """End the session process unless it serves a robot app; ``peer`` is the one waiting.

        False if it does, or once the server is stopping, which the wait checks every
        WAIT_SLICE_S.
        """
        start = time.monotonic()
        announced_s: float | None = None
        while not self._stopping():
            session = self._session
            if session is None:
                return True
            elapsed = time.monotonic() - start
            if session.popen.poll() is not None:
                self._end_session()
                continue
            if announced_s is None:
                session.read_reports(0)
            # The bound holds whether the wait ends served or refused as busy. It grows, and is
            # announced again, when a session that was still starting reports a longer teardown.
            if announced_s is None or session.handover_timeout_s > announced_s:
                announced_s = session.handover_timeout_s
                logger.info(
                    "Robot app at %s waits up to %g s for the previous session (pid %d) to end",
                    peer,
                    announced_s,
                    session.popen.pid,
                )
            if session.ending:
                limit = session.handover_timeout_s
                if elapsed >= limit:
                    logger.warning(
                        "Session process %d is still running after its robot app left; "
                        "killing it",
                        session.popen.pid,
                    )
                    self._end_session(kill=True)
                else:
                    with contextlib.suppress(subprocess.TimeoutExpired):
                        session.popen.wait(timeout=min(WAIT_SLICE_S, limit - elapsed))
            else:
                limit = BUSY_GRACE_S if session.serving else HANDOVER_TIMEOUT_S
                if elapsed >= limit:
                    return False
                session.read_reports(min(WAIT_SLICE_S, limit - elapsed))
        return False

    def _reap(self) -> None:
        if self._session is not None and self._session.popen.poll() is not None:
            self._end_session()

    def _end_session(self, kill: bool = False) -> None:
        """Stop the session process and anything left in its process group.

        SIGTERM goes to the session process alone: its teardown stops the plugins as
        TeleopSession does (SIGINT, then SIGKILL after the plugin's shutdown timeout). The group
        gets SIGKILL after the session's stop timeout or a further stop signal, or at once with
        ``kill``.
        """
        session, self._session = self._session, None
        if session is None:
            return
        popen = session.popen
        if popen.poll() is None and not kill:
            # A report not read yet holds the teardown time.
            session.read_reports(0)
            _signal_process(popen.pid, signal.SIGTERM)
            self._wait_for_stop(session)
        if popen.poll() is None:
            _signal_group(popen.pid, signal.SIGKILL)
            popen.wait()
        # Plugins of a session process that died abruptly are still in its process group.
        _signal_group(popen.pid, signal.SIGKILL)
        os.close(session.serving_fd)
        logger.log(
            logging.INFO if popen.returncode == 0 else logging.WARNING,
            "Session process %d for %s exited with code %d",
            popen.pid,
            session.peer,
            popen.returncode,
        )

    def _wait_for_stop(self, session: _SessionProcess) -> None:
        """Wait in slices for the session process to exit, up to its stop timeout."""
        popen, timeout_s = session.popen, session.stop_timeout_s
        if timeout_s > SESSION_STOP_TIMEOUT_S:
            logger.info(
                "Waiting up to %g s for session process %d to stop its plugins; "
                "another stop signal kills it now",
                timeout_s,
                popen.pid,
            )
        deadline = time.monotonic() + timeout_s
        while popen.poll() is None:
            if self._stop.count > self._stop_signals_spent:
                logger.warning(
                    "Received another stop signal; killing session process %d",
                    popen.pid,
                )
                return
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                logger.warning(
                    "Session process %d did not stop within %g s; killing it",
                    popen.pid,
                    timeout_s,
                )
                return
            with contextlib.suppress(subprocess.TimeoutExpired):
                popen.wait(timeout=min(WAIT_SLICE_S, remaining))


def _session_command(args: argparse.Namespace, dirs: Sequence[Path]) -> list[str]:
    command = [sys.executable, "-m", _SESSION_MODULE, "--parent-pid", str(os.getpid())]
    for path in dirs:
        command += ["--plugin-dir", str(path)]
    if args.placeholder_openxr:
        command.append("--placeholder-openxr")
    if args.verbose:
        command.append("--verbose")
    return command


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

    dirs = plugin_dirs(args.plugin_dir)
    for path in args.plugin_dir:
        if not path.is_dir():
            logger.warning("--plugin-dir %s is not a directory", path)
    logger.info(
        "Plugin directories, a later one winning a plugin name: %s",
        [str(d) for d in dirs],
    )
    if args.placeholder_openxr:
        logger.warning(
            "--placeholder-openxr: no OpenXR session; only KeyboardTracker works"
        )

    try:
        launcher = CloudXRLauncher.launch_context(args)
    except (RuntimeError, ValueError) as error:
        logger.error("CloudXR is not available: %s", error)
        return 1
    wakeup_read, wakeup_write = os.pipe2(os.O_NONBLOCK | os.O_CLOEXEC)
    stop = _StopRequest(wakeup_read)
    previous = {sig: signal.signal(sig, stop) for sig in _STOP_SIGNALS}
    previous_wakeup_fd = signal.set_wakeup_fd(wakeup_write, warn_on_full_buffer=False)
    try:
        with launcher:
            for item in listening:
                print(f"Teleop Session Server listening on {item.url}", flush=True)
            server = _Server(listening, _session_command(args, dirs), launcher, stop)
            return server.run()
    finally:
        signal.set_wakeup_fd(previous_wakeup_fd)
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        os.close(wakeup_read)
        os.close(wakeup_write)


if __name__ == "__main__":
    sys.exit(main())
