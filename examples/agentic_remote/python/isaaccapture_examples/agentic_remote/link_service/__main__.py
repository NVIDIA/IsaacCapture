# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""python -m isaaccapture_examples.agentic_remote.link_service --config PATH.

The robot link service as a program: it waits for the robot app to dial in and
serves the local API until SIGINT or SIGTERM.
"""

from __future__ import annotations

import argparse
import importlib
import logging
import queue
import signal
import sys
import threading
from collections.abc import Callable, Sequence

from ..console_log import set_console_level
from .config import LOG_LEVELS, ConfigError, LinkServiceConfig, load_link_service_config
from .listen import RobotListener
from .local import LocalSocketError
from .service import Incoming, RobotLinkService

_log = logging.getLogger("isaaccapture.examples.agentic_remote.link_service_main")

EXIT_OK = 0
EXIT_FAILED = 1  # the local socket or the listening address cannot be had
EXIT_NOT_RUN = 2  # the arguments, the file or the installation cannot be used

PROGRAM = "robot link service"
NO_LINK = (
    "Isaac Capture's remote link cannot be imported (isaaccapture.remote_link on "
    "the Python path)"
)

Say = Callable[[str], None]


def _say(text: str) -> None:
    print(text, flush=True)


def _complain(text: str) -> None:
    print(f"{PROGRAM}: {text}", file=sys.stderr, flush=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m isaaccapture_examples.agentic_remote.link_service",
        description=(
            "The robot link service of the agentic remote execution example: the "
            "workstation end of the robot connection, and the local API."
        ),
    )
    parser.add_argument(
        "--config", required=True, metavar="PATH", help="the service's YAML file"
    )
    parser.add_argument(
        "--log-level",
        default=None,
        type=str.lower,
        choices=LOG_LEVELS,
        metavar="L",
        help=f"console log level, one of {', '.join(LOG_LEVELS)} (default: the file's)",
    )
    return parser


class LinkServiceProgram:
    """A robot link service on a listening TCP address, started and closed as one."""

    def __init__(self, config: LinkServiceConfig) -> None:
        self._config = config
        self._listener: RobotListener | None = None
        self.service = RobotLinkService(self._accept, **config.service_options())

    def start(self) -> None:
        """Bind the local socket, then listen for the robot app.

        Raises LocalSocketError (LocalSocketInUse when a live service has the
        socket) and RuntimeError when the address cannot be listened on. After
        an error nothing is left open.
        """
        self.service.start()
        try:
            config = self._config
            self._listener = RobotListener(config.listen_address, config.listen_port)
        except BaseException:
            self.service.close()
            raise

    @property
    def port(self) -> int:
        """The TCP port the robot app dials; 0 before start()."""
        return 0 if self._listener is None else self._listener.port

    @property
    def endpoint(self) -> str:
        """The listening address as "host:port"."""
        return f"{self._config.listen_address}:{self.port}"

    def close(self) -> None:
        """End the service, then stop listening. The robot is not stopped."""
        self.service.close()
        if self._listener is not None:
            self._listener.close()

    def _accept(self, timeout_s: float) -> Incoming | None:
        listener = self._listener
        return None if listener is None else listener.accept(timeout_s)


def _wait_for_signal() -> str:
    """Wait on the main thread for SIGINT or SIGTERM; the name of the one that came."""
    # The handler only queues: SimpleQueue.put() is reentrant.
    came: queue.SimpleQueue[int] = queue.SimpleQueue()
    numbers = (signal.SIGINT, signal.SIGTERM)
    replaced = {
        n: signal.signal(n, lambda number, _: came.put(number)) for n in numbers
    }
    try:
        return signal.Signals(came.get()).name
    finally:
        for number, handler in replaced.items():
            signal.signal(number, signal.SIG_DFL if handler is None else handler)


def main(
    argv: Sequence[str] | None = None,
    *,
    wait: Callable[[LinkServiceProgram], str] | None = None,
    say: Say = _say,
) -> int:
    """Run the robot link service's command line; returns the exit code.

    wait(program) blocks until the service is to close and returns why; None
    waits for SIGINT or SIGTERM, which only the main thread can.
    """
    try:
        args = _parser().parse_args(argv)
    except SystemExit as exit_:  # argparse has printed the usage or the help
        code = exit_.code
        return code if isinstance(code, int) else EXIT_NOT_RUN
    try:
        config = load_link_service_config(args.config)
    except ConfigError as error:
        _complain(f"the configuration cannot be used: {error}")
        return EXIT_NOT_RUN
    if wait is None and threading.current_thread() is not threading.main_thread():
        _complain("signals can be waited for on the main thread only")
        return EXIT_NOT_RUN
    try:
        importlib.import_module("isaaccapture.remote_link")
    except ImportError as error:
        _complain(f"{NO_LINK}: {error}")
        return EXIT_NOT_RUN
    set_console_level(args.log_level or config.log_level)
    program = LinkServiceProgram(config)
    try:
        program.start()
    except LocalSocketError as error:
        _complain(str(error))
        return EXIT_FAILED
    except RuntimeError as error:
        _complain(f"cannot listen on {config.listen_endpoint}: {error}")
        return EXIT_FAILED
    try:
        peers = ", ".join(config.allowed_peers)
        say(
            f"{PROGRAM}: waits for the robot app on {program.endpoint} (taken from "
            f"{peers}); local API at {program.service.socket_path}"
        )
        why = _wait_for_signal() if wait is None else wait(program)
        say(f"{PROGRAM}: {why}; closing. The robot holds on its own.")
    finally:
        program.close()
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
