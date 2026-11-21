# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CloudXR for the Teleop Session Server: started or attached to once, then only checked.

The server runs without CloudXR. A CloudXR runtime that a person starts later is attached to by
reading its environment file.
"""

from __future__ import annotations

import argparse
import logging
import os
import shlex
import sys
from pathlib import Path
from typing import Any

from isaaccapture.cloudxr import CloudXRLauncher
from isaaccapture.cloudxr.env_config import ENV_FILE_NAME, read_exported_env
from isaaccapture.cloudxr.runtime import is_runtime_live

logger = logging.getLogger("isaaccapture.teleop_session_server.cloudxr")

# What a person runs to start CloudXR in the default install folder.
START_COMMAND = "python -m isaaccapture.cloudxr.service start"
# The install folder of the CloudXR service and the launcher when none is given.
DEFAULT_INSTALL_DIR = "~/.cloudxr"
EULA_NOT_ACCEPTED = "CloudXR is not started: the NVIDIA CloudXR EULA is not accepted"


class CloudXR:
    """CloudXR as the server's loop checks it.

    ``launcher`` is what ``CloudXRLauncher.launch_context()`` gave, None when it failed;
    ``unavailable`` is then the reason. ``run_dir`` is the run folder of the CloudXR install, None
    with ``--no-launch-cloudxr-runtime``: the OpenXR environment is then the caller's and nothing
    is checked. Nothing here starts CloudXR, asks a question or waits.
    """

    def __init__(
        self, launcher: Any, run_dir: str | None, unavailable: str | None = None
    ) -> None:
        self._launcher = launcher
        self._run_dir = run_dir
        self._unavailable = unavailable or "CloudXR is not available"
        # False while the environment of the runtime serving run_dir is still to be read.
        self._attached = launcher is not None

    @property
    def start_command(self) -> str:
        """What a person runs to start the CloudXR this server attaches to."""
        default_run = os.path.join(os.path.expanduser(DEFAULT_INSTALL_DIR), "run")
        if self._run_dir is None or os.path.abspath(self._run_dir) == os.path.abspath(
            default_run
        ):
            return START_COMMAND
        # The service's own default is another folder, where this server does not look.
        install_dir = os.path.dirname(os.path.abspath(self._run_dir))
        return f"{START_COMMAND} --cloudxr-install-dir {shlex.quote(install_dir)}"

    def __enter__(self) -> CloudXR:
        if self._launcher is not None:
            self._launcher.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool | None:
        if self._launcher is not None:
            return self._launcher.__exit__(exc_type, exc_val, exc_tb)
        return None

    def health_check(self) -> None:
        """Raise RuntimeError with the reason while CloudXR cannot be used.

        The first check that passes after one that failed attaches to the runtime serving then.
        """
        if self._run_dir is None:
            return
        try:
            if self._launcher is not None:
                self._launcher.health_check()
            elif not is_runtime_live(self._run_dir):
                raise RuntimeError(self._unavailable)
        except Exception:
            self._attached = False
            raise
        if not self._attached:
            self._attach()

    def _attach(self) -> None:
        """Point OpenXR at the runtime serving ``run_dir``, as CloudXRLauncher does at start."""
        env = read_exported_env(Path(self._run_dir) / ENV_FILE_NAME)
        if not env:
            raise RuntimeError(
                f"A CloudXR runtime is serving {self._run_dir}, but its environment file "
                "is missing or unreadable, so OpenXR cannot be pointed at it"
            )
        # Only what differs: setting the environment is not safe beside threads that read it.
        for key, value in env.items():
            if os.environ.get(key) != value:
                os.environ[key] = value
        self._attached = True
        self._unavailable = f"The CloudXR runtime serving {self._run_dir} has stopped"
        logger.info("Attached to the CloudXR runtime serving %s", self._run_dir)


def _at_terminal() -> bool:
    """Whether a person can answer on standard input."""
    try:
        return sys.stdin is not None and sys.stdin.isatty()
    except ValueError:
        # Standard input is closed.
        return False


def start(args: argparse.Namespace) -> CloudXR:
    """Start or attach to CloudXR; a failure gives a CloudXR that is not available yet.

    SystemExit goes on to the caller when a person declined at a terminal (the EULA question)
    or pressed a key at the launcher's notice.
    """
    run_dir = (
        os.path.join(os.path.expanduser(args.cloudxr_install_dir), "run")
        if args.launch_cloudxr_runtime
        else None
    )
    try:
        launcher = CloudXRLauncher.launch_context(args)
    except SystemExit as stopped:
        # The EULA check exits with a number, the launcher's notice with a text.
        if isinstance(stopped.code, str) or _at_terminal():
            raise
        # The EULA question is on standard output without a line end; what the server prints
        # next starts a line of its own.
        print(flush=True)
        return CloudXR(None, run_dir, EULA_NOT_ACCEPTED)
    except Exception as error:
        return CloudXR(None, run_dir, f"CloudXR is not available: {error}")
    return CloudXR(launcher, run_dir)
