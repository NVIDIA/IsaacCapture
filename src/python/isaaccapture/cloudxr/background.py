# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the CloudXR service detached from the calling terminal.

``start_new_session=True`` is ``setsid(2)``: the child leads its own session,
so it has no controlling terminal and neither a hangup nor a signal aimed at
the shell's process group can reach it.  Works the same on a host, in a
container, and under CI — nothing here needs systemd.

What it does not do is supervise.  A crashed service stays dead until someone
starts it again.
"""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path

try:  # POSIX only; without it concurrent starts are not serialised.
    import fcntl
except ImportError:  # pragma: no cover - Windows
    fcntl = None

PID_FILE = "service.pid"
LOG_FILE = "service.log"
LOCK_FILE = "service.lock"

#: How long :func:`terminate` waits for a SIGTERM to be honoured.
STOP_TIMEOUT_SEC = 15.0

_MODULE = "isaaccapture.cloudxr.service"

#: Recognised, never spawned: a detached service outlives the shell, so an upgrade
#: meets a /proc cmdline no import alias rewrites.  TODO(1.9): drop with isaacteleop.
_LEGACY_MODULES = ("isaacteleop.cloudxr.service",)


#: Refusal shown wherever a start meets a runtime that is already serving.
ALREADY_SERVING = (
    "A CloudXR runtime is already serving {run_dir}.  "
    "Use `service status`, or stop it with `service stop`."
)


class AlreadyServingError(RuntimeError):
    """A runtime was already serving the run directory, so nothing was spawned.

    Distinct from the other startup failures because it is not one for some
    callers: a launcher that only wants a runtime can attach to that one.
    """


def process_exists(pid: object) -> bool:
    """Probe a positive POSIX process ID without sending a signal."""
    if os.name != "posix" or type(pid) is not int or pid <= 0:
        return False
    try:
        # POSIX signal 0 is a probe; on Windows os.kill would terminate instead.
        os.kill(pid, 0)  # NOSONAR
    except (OSError, ValueError, OverflowError):
        return False
    return True


def pid_path(run_dir: str) -> Path:
    """Path of the detached service's pid file."""
    return Path(run_dir) / PID_FILE


def log_path(logs_dir: Path) -> Path:
    """Path the detached service's stdout and stderr are appended to."""
    return Path(logs_dir) / LOG_FILE


def read_pid(run_dir: str) -> int | None:
    """Return the recorded pid, or ``None`` if it is absent or no longer ours.

    A pid file outlives the process that wrote it and pids get reused, so the
    recorded process is checked against its own command line before any caller
    is allowed to signal it.
    """
    try:
        pid = int(pid_path(run_dir).read_text(encoding="utf-8").strip())
    except (FileNotFoundError, ValueError):
        return None
    return pid if _is_our_service(pid) else None


def _is_our_service(pid: int) -> bool:
    """Whether *pid* is alive and running the service module.

    ``read_pid`` gates ``terminate`` and ``read_run_flags`` on this, so failing to
    recognise a service makes ``stop`` report success while it still holds the GPU.
    """
    try:
        cmdline = Path(f"/proc/{pid}/cmdline").read_bytes()
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return False
    # A whole argument, not a substring of the blob: pids are reused, and anything
    # that merely names the module -- a log path, a `tail`, a `pkill` pattern --
    # would otherwise be mistaken for the service and survive `stop`.
    argv = cmdline.decode(errors="replace").split("\0")
    return any(arg in (_MODULE, *_LEGACY_MODULES) for arg in argv)


def read_run_flags(run_dir: str) -> list[str]:
    """Return the flags the detached service was started with.

    Recovered from its command line, so a later ``status`` reports the session
    that is actually running rather than the defaults.  Empty when there is no
    detached service (started in the foreground, or by something else).
    """
    pid = read_pid(run_dir)
    if pid is None:
        return []
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes().decode()
    except OSError:
        return []
    # cmdline NUL-*terminates* every argument, so a plain split leaves a
    # trailing empty string that argparse rejects as an unknown argument.
    argv = [arg for arg in raw.split("\0") if arg]
    return argv[argv.index("run") + 1 :] if "run" in argv else []


def spawn(
    run_args: list[str],
    run_dir: str,
    logs_dir: Path,
    extra_env: dict[str, str] | None = None,
) -> tuple[int, Path]:
    """Start a detached ``service run`` and record its pid.

    *extra_env* is merged into the child's environment; :class:`EnvConfig`
    reads settings like ``NV_DEVICE_PROFILE`` from there, which is how a
    caller configures the service without every setting becoming a flag.

    Returns the pid and the log file its output is appended to.
    """
    os.makedirs(run_dir, mode=0o700, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    log = log_path(logs_dir)

    # print() to a file is block-buffered, so without PYTHONUNBUFFERED the
    # startup banner sits in the buffer and `tail -f` looks like a hang.
    env = {**os.environ, "PYTHONUNBUFFERED": "1", **(extra_env or {})}
    # Not inherited: a process that sees this variable gets a forwarding handler
    # *and nothing else*, on the assumption that the publisher outlives it. This
    # service is started to outlive its launcher, so dropping the variable makes
    # it its own session leader. The C++ half reads the same one.
    env.pop("ISAACCAPTURE_LOG_SOCKET", None)

    with open(log, "a", encoding="utf-8") as handle:
        proc = subprocess.Popen(
            [sys.executable, "-m", _MODULE, "run", *run_args],
            start_new_session=True,
            stdin=subprocess.DEVNULL,
            stdout=handle,
            stderr=subprocess.STDOUT,
            env=env,
        )

    pid_path(run_dir).write_text(f"{proc.pid}\n", encoding="utf-8")
    return proc.pid, log


@contextlib.contextmanager
def _start_lock(run_dir: str):
    """Serialise service starts for *run_dir*.

    Two callers that both saw no runtime would otherwise spawn one each, and
    two runtimes racing for the same IPC socket is worse than a wait.

    The lock lives on the open fd, so the kernel drops it however the process
    dies, SIGKILL included.  A leftover lock file is therefore not stale
    state; do not unlink it, because another process may hold the lock on
    that inode.
    """
    if fcntl is None:
        yield
        return
    os.makedirs(run_dir, mode=0o700, exist_ok=True)
    with open(os.path.join(run_dir, LOCK_FILE), "w", encoding="utf-8") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield


def start_and_wait(
    run_args: list[str],
    run_dir: str,
    logs_dir: Path,
    extra_env: dict[str, str] | None = None,
    *,
    before_spawn: Callable[[], None] | None = None,
) -> tuple[int, Path]:
    """Spawn a detached service and wait until its runtime is serving *run_dir*.

    *before_spawn* runs under the start lock after the liveness check. A
    failure prevents a detached child from being left behind.

    Raises:
        AlreadyServingError: If a runtime is serving *run_dir* by the time
            this caller holds the start lock.
        RuntimeError: If the service exits during startup, or never comes up.
    """
    from .runtime import RUNTIME_STARTUP_TIMEOUT_SEC, is_runtime_live  # noqa: PLC0415

    with _start_lock(run_dir):
        # Re-check under the lock.  A caller that queued here behind another
        # start would otherwise spawn a service that refuses and exits, and
        # its pid file would already have replaced the pid of the service
        # actually serving -- leaving that one unreachable by `service stop`.
        if is_runtime_live(run_dir):
            raise AlreadyServingError(ALREADY_SERVING.format(run_dir=run_dir))
        if before_spawn is not None:
            before_spawn()
        pid, log = spawn(run_args, run_dir, logs_dir, extra_env)
        deadline = time.monotonic() + RUNTIME_STARTUP_TIMEOUT_SEC
        while time.monotonic() < deadline:
            if is_runtime_live(run_dir):
                return pid, log
            if read_pid(run_dir) is None:
                raise RuntimeError(
                    f"The CloudXR service exited during startup.  See {log}"
                )
            time.sleep(0.2)
    raise RuntimeError(
        f"The CloudXR service did not come up within {RUNTIME_STARTUP_TIMEOUT_SEC}s "
        f"(pid {pid} is still running).  See {log}"
    )


def terminate(run_dir: str, timeout_sec: float = STOP_TIMEOUT_SEC) -> bool:
    """Ask the detached service to stop; return whether it did.

    SIGTERM only, never SIGKILL: the runtime subprocess leads its own session,
    so killing the service outright would orphan a process holding the GPU.
    The service's own handler is what tears that down.
    """
    pid = read_pid(run_dir)
    if pid is None:
        pid_path(run_dir).unlink(missing_ok=True)
        return False

    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        # Exited between the pid check and the signal; that is a stop.
        pid_path(run_dir).unlink(missing_ok=True)
        return True
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        if not _is_our_service(pid):
            pid_path(run_dir).unlink(missing_ok=True)
            return True
        time.sleep(0.1)
    return False
