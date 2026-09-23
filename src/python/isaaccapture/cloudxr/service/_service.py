# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Owner of the CloudXR runtime process and the WSS TLS proxy."""

from __future__ import annotations

import asyncio
import atexit
import concurrent.futures

# asyncio resolves the listen address through run_in_executor, which imports
# concurrent.futures.thread on first use.  That import calls
# threading._register_atexit, which raises "can't register atexit after
# shutdown" once interpreter finalization has begun — so import it here, while
# the main thread still can, rather than from the proxy thread mid-teardown.
import concurrent.futures.thread  # noqa: F401
from collections import deque
import logging
import json
import uuid
import os
import signal
import subprocess
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path

from ..env_config import DEFAULT_DEVICE_PROFILE, ENV_FILE_NAME, EnvConfig
from ...logging_config._core import logging_enabled
from ...logging_config._native_api import native_capture_fd, native_capture_path
from ..runtime import (
    RUNTIME_STARTUP_TIMEOUT_SEC,
    RUNTIME_TERMINATE_TIMEOUT_SEC,
    WSS_STARTUP_TIMEOUT_SEC,
    get_sdk_path,
    is_runtime_live,
    resolve_cloudxr_runtime_module,
    check_eula,
    wait_for_runtime_ready_sync,
)

logger = logging.getLogger(__name__)

_RUNTIME_ALREADY_SERVING = """\
A CloudXR runtime is already serving {run_dir}, and starting a second one would
drop the live session.

  To use the running one, point OpenXR at it:
    source {env_file}
  Applications that embed CloudXRLauncher do this for themselves.

  To replace it, stop the running service first:
    python -m isaaccapture.cloudxr.service stop
  (Ctrl+C in its terminal if it is running in the foreground.)"""

#: Runtime worker stderr, kept apart from the runtime process's own -- see
#: _gather_diagnostic_logs, which reads both.
_WORKER_STDERR_LOG = "runtime_worker_stderr.log"


def _tail_text(path: Path, limit: int) -> str:
    """Last *limit* bytes of *path*, decoded leniently.

    Seeks rather than reading the whole file: one of the files this is pointed
    at is the session's native capture file, which may be a full 10 MiB
    generation even though this diagnostic needs only its last few KiB.
    """
    with open(path, "rb") as handle:
        handle.seek(0, os.SEEK_END)
        size = handle.tell()
        handle.seek(max(0, size - limit))
        chunk = handle.read(limit)
    text = chunk.decode(errors="replace").strip()
    if text and size > limit:
        return "...\n" + text
    return text


_RUNTIME_WORKER_CODE = """\
import sys, os
if sys.platform.startswith("linux"):
    # After exec: preexec_fn can deadlock once the log receiver thread exists.
    import ctypes, signal
    ctypes.CDLL(None).prctl(1, signal.SIGTERM, 0, 0, 0)  # PR_SET_PDEATHSIG
    if os.getppid() != {parent_pid}:
        raise SystemExit(1)
sys.path = [p for p in sys.path if p]
from {runtime_mod}.runtime import run
run()
"""


class CloudXRService:
    """Owns a CloudXR runtime process and its WSS TLS proxy.

    Both start on construction; use :meth:`stop` or the context manager
    protocol to shut them down.  The runtime runs as an isolated subprocess
    to avoid CUDA context conflicts with host applications like Isaac Sim
    that have already initialized GPU resources.

    Example::

        with CloudXRService() as service:
            # runtime + WSS proxy are running and owned by this process
            ...
    """

    def __init__(
        self,
        install_dir: str = "~/.cloudxr",
        env_config: str | Path | None = None,
        device_profile: str = DEFAULT_DEVICE_PROFILE,
        accept_eula: bool = False,
        setup_oob: bool = False,
        usb_local: bool = False,
        host_client: bool = False,
        recovery_config=None,
    ) -> None:
        """Start the CloudXR runtime and the WSS proxy.

        Configures the environment, spawns the runtime subprocess, and starts
        the WSS TLS proxy.  Blocks until the runtime signals readiness (up to
        :data:`~isaaccapture.cloudxr.runtime.RUNTIME_STARTUP_TIMEOUT_SEC`) and
        the proxy is listening (up to
        :data:`~isaaccapture.cloudxr.runtime.WSS_STARTUP_TIMEOUT_SEC`), or
        raises :class:`RuntimeError` on failure.

        Args:
            install_dir: CloudXR install directory.
            env_config: Optional path to a KEY=value env file for
                CloudXR env-var overrides.
            device_profile: CloudXR ``NV_DEVICE_PROFILE`` when not set in
                *env_config* or the process environment.
            accept_eula: Accept the NVIDIA CloudXR EULA
                non-interactively.  When ``False`` and the EULA marker
                does not exist, the user is prompted on stdin.
            setup_oob: Enable the OOB teleop control hub and USB
                adb automation in the WSS proxy.
            usb_local: Route teleop traffic over USB headset loopback via
                ``adb reverse`` (requires *setup_oob*); also starts coturn
                for WebRTC ICE relay and serves ``/client/`` on the WSS
                proxy (same assets as *host_client*).  Ports are overridable
                via ``PROXY_PORT`` / ``USB_BACKEND_PORT`` / ``USB_TURN_PORT``.
            host_client: Serve the web client at ``/client/`` on the WSS
                proxy port.  Assets are fetched once from GitHub Pages into
                ``TELEOP_WEB_CLIENT_STATIC_DIR`` or ``~/.cloudxr/static-client``.

        Raises:
            RuntimeError: If the EULA is not accepted, another runtime is
                already serving *install_dir*, or the runtime or WSS proxy
                fails to start within its timeout.
        """
        self._install_dir = install_dir
        self._env_config = str(env_config) if env_config is not None else None
        self._device_profile = device_profile
        self._accept_eula = accept_eula
        self._setup_oob = setup_oob
        self._usb_local = usb_local
        self._host_client = host_client

        if self._usb_local or self._host_client:
            from ..oob_teleop_env import require_web_client_static_dir  # noqa: PLC0415

            require_web_client_static_dir(
                require_health_probe=setup_oob and not os.getenv("TELEOP_OOB_HUB_ONLY")
            )

        self._runtime_proc: subprocess.Popen | None = None
        self._wss_thread: threading.Thread | None = None
        self._wss_loop: asyncio.AbstractEventLoop | None = None
        self._wss_stop_future: asyncio.Future | None = None
        self._wss_log_path: Path | None = None
        self._atexit_registered = False
        self._stopping = False
        self._stop_lock = threading.RLock()
        self._oob_lock = threading.Lock()
        self._oob_snapshot: dict | None = None
        self._oob_updates: deque[dict] = deque(maxlen=128)
        self._oob_session_id = uuid.uuid4().hex
        self._fatal_error: Exception | None = None
        self._fatal_supervisor: threading.Thread | None = None
        # sig -> (previous handler, service-installed wrapper)
        self._prev_signal_handlers: dict[int, tuple[object, object]] = {}

        # Before EnvConfig: resolving rewrites cloudxr.env and mutates
        # os.environ, which must not happen when the start is about to be
        # refused because another runtime owns this directory.
        run_dir = os.path.join(
            os.path.abspath(os.path.expanduser(self._install_dir)), "run"
        )
        self._oob_status_path = Path(run_dir) / "oob_status.json"
        from ..oob_teleop_env import resolve_oob_recovery_config  # noqa: PLC0415

        self._recovery_config = (
            recovery_config or resolve_oob_recovery_config() if setup_oob else None
        )
        self._cleanup_stale_runtime(run_dir)
        if setup_oob:
            # Embedded launchers do not pass through the service CLI preflight.
            # Validate host dependencies before starting the CloudXR worker;
            # headset-dependent readiness remains the lifecycle's job.
            from ..oob_teleop_adb import (  # noqa: PLC0415
                require_adb_on_path,
                require_coturn_available,
                require_turn_port_free,
            )
            from ..oob_teleop_env import (  # noqa: PLC0415
                resolve_lan_host_for_oob,
                usb_turn_port,
            )

            if usb_local or not os.getenv("TELEOP_OOB_HUB_ONLY"):
                require_adb_on_path()
            if usb_local:
                require_coturn_available()
                require_turn_port_free(usb_turn_port())
            else:
                resolve_lan_host_for_oob()
        try:
            self._oob_status_path.unlink(missing_ok=True)
        except OSError as exc:
            logger.warning(
                "Cannot clear stale OOB status %s: %s", self._oob_status_path, exc
            )

        env_cfg = EnvConfig.from_args(
            self._install_dir,
            self._env_config,
            launcher_defaults={"NV_DEVICE_PROFILE": self._device_profile},
        )
        try:
            check_eula(accept_eula=self._accept_eula or None)
        except SystemExit as exc:
            raise RuntimeError(
                "CloudXR EULA was not accepted; cannot start the runtime"
            ) from exc
        logs_dir_path = env_cfg.ensure_logs_dir()

        # The worker imports asyncio (via isaaccapture.cloudxr.runtime), which imports
        # Python's ssl and loads the SYSTEM OpenSSL before the native stack dlopens the
        # bundled one. Two OpenSSL builds in one process crash (SIGSEGV) inside
        # SSL_CTX_use_certificate when the DTLS transport comes up on client connect.
        # LD_PRELOAD the bundled libraries so every OpenSSL symbol in the worker
        # resolves to the version libNvStreamServer.so was built against.
        worker_env = os.environ.copy()
        runtime_mod = resolve_cloudxr_runtime_module()
        sdk_dir = get_sdk_path()
        logger.info("CloudXR Runtime: module=%s sdk_path=%s", runtime_mod, sdk_dir)
        bundled_ssl = [
            os.path.join(sdk_dir, lib)
            for lib in ("libcrypto_nvst.so.3", "libssl_nvst.so.3")
        ]
        if all(os.path.isfile(lib) for lib in bundled_ssl):
            preload = " ".join(bundled_ssl)
            prev = worker_env.get("LD_PRELOAD")
            worker_env["LD_PRELOAD"] = f"{preload} {prev}" if prev else preload

        # To a file, not a pipe: the runtime is long-lived and nothing here
        # would drain a pipe while it runs, so past the pipe buffer (64 KiB)
        # its next write to stderr would block the runtime with no diagnostic.
        # Truncated per start so a failure report shows only this one.
        worker_stderr = logs_dir_path / _WORKER_STDERR_LOG
        # stdout explicitly, not inherited: the runtime's startup banner and the
        # native stack's chatter would otherwise land on the host application's
        # terminal. This is a process this library launched, so its descriptors
        # are ours to set. None when logging is off or the capture file could not
        # be opened, in which case inheriting is the right fallback.
        capture_fd = native_capture_fd()
        with open(worker_stderr, "w", encoding="utf-8") as stderr_file:
            self._runtime_proc = subprocess.Popen(
                [
                    sys.executable,
                    "-c",
                    _RUNTIME_WORKER_CODE.format(
                        runtime_mod=runtime_mod, parent_pid=os.getpid()
                    ),
                ],
                env=worker_env,
                stdout=capture_fd,
                stderr=stderr_file,
                start_new_session=True,
            )
        logger.info("CloudXR runtime process started (pid=%s)", self._runtime_proc.pid)

        if not self._atexit_registered:
            atexit.register(self.stop)
            self._atexit_registered = True
        # SIGTERM/SIGINT do not run atexit; stop the session-scoped runtime first.
        self._install_signal_handlers()

        if not wait_for_runtime_ready_sync(is_process_alive=self._is_runtime_alive):
            detail = self._collect_startup_failure_detail(logs_dir_path)
            self.stop()
            raise RuntimeError(
                "CloudXR runtime failed to start within "
                f"{RUNTIME_STARTUP_TIMEOUT_SEC}s.  {detail}"
            )
        logger.info("CloudXR runtime ready")

        wss_ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
        wss_log_path = logs_dir_path / f"wss.{wss_ts}.log"
        self._wss_log_path = wss_log_path
        try:
            self._start_wss_proxy_thread(wss_log_path)
        except RuntimeError:
            # The runtime is already up by now; a service that cannot serve it
            # must not leave it behind holding the run directory.
            self.stop()
            raise
        logger.info("CloudXR WSS proxy listening (log=%s)", wss_log_path)

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> CloudXRService:
        """Return the service for use in a ``with`` block."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Stop the service on exiting the ``with`` block."""
        self.stop()

    def stop(self) -> None:
        """Shut down the WSS proxy and terminate the runtime process.

        Safe to call multiple times or when nothing is running.

        Raises:
            RuntimeError: If the runtime process could not be terminated.
                The process handle is retained so callers can retry or
                inspect the still-running process.
        """
        # Restore handlers only after teardown; _stopping blocks re-entrant stop().
        with self._stop_lock:
            if self._stopping:
                return
            self._stopping = True
            try:
                self._stop_wss_proxy()

                if self._runtime_proc is not None:
                    try:
                        self._terminate_runtime()
                    except RuntimeError:
                        logger.warning(
                            "Failed to cleanly terminate CloudXR runtime process "
                            "(pid=%s); handle retained for later cleanup",
                            self._runtime_proc.pid,
                        )
                        raise
                    self._runtime_proc = None
                    logger.info("CloudXR runtime process stopped")
                if self._fatal_error is None and self._oob_snapshot is not None:
                    try:
                        current = json.loads(
                            self._oob_status_path.read_text(encoding="utf-8")
                        )
                        if current.get("sessionId") == self._oob_session_id:
                            self._oob_status_path.unlink(missing_ok=True)
                    except FileNotFoundError:
                        # Another shutdown path may already have removed the file.
                        pass
                    except (OSError, ValueError) as exc:
                        logger.warning(
                            "Cannot clean OOB status %s: %s", self._oob_status_path, exc
                        )
            finally:
                self._restore_signal_handlers()
                self._stopping = False

    def _publish_oob_status(self, snapshot: dict) -> None:
        """Atomically hand off a redacted lifecycle snapshot across threads."""
        payload = {
            **snapshot,
            "writerPid": os.getpid(),
            "runtimePid": self._runtime_proc.pid if self._runtime_proc else None,
            "sessionId": self._oob_session_id,
        }
        with self._oob_lock:
            self._oob_snapshot = payload
            self._oob_updates.append(payload)
            try:
                self._oob_status_path.parent.mkdir(parents=True, exist_ok=True)
                temporary = self._oob_status_path.with_suffix(".json.tmp")
                temporary.write_text(
                    json.dumps(payload, sort_keys=True), encoding="utf-8"
                )
                os.replace(temporary, self._oob_status_path)
            except OSError as exc:
                logger.warning(
                    "Cannot publish OOB status %s: %s", self._oob_status_path, exc
                )

    def oob_status(self) -> dict | None:
        """Return the latest lifecycle snapshot without touching the WSS event loop."""
        with self._oob_lock:
            return dict(self._oob_snapshot) if self._oob_snapshot else None

    def drain_oob_updates(self) -> list[dict]:
        """Return transition snapshots for the foreground CLI, in order."""
        with self._oob_lock:
            updates = list(self._oob_updates)
            self._oob_updates.clear()
        return updates

    def _on_oob_fatal(self, error: Exception) -> None:
        """Transfer fatal teardown to a thread that can safely join WSS."""
        with self._oob_lock:
            if self._fatal_error is not None:
                return
            self._fatal_error = error
        self._fatal_supervisor = threading.Thread(
            target=self.stop, name="cloudxr-oob-fatal-supervisor", daemon=True
        )
        self._fatal_supervisor.start()

    def health_check(self) -> None:
        """Verify that the runtime process and WSS proxy are healthy.

        Returns immediately when the runtime is running and the WSS proxy
        thread, once started, is alive.  Raises :class:`RuntimeError` with
        diagnostic details when any monitored component has stopped
        unexpectedly, allowing supervisors to perform a controlled teardown.

        Raises:
            RuntimeError: If the service has not been started, or if
                the runtime process or the WSS proxy has stopped.
        """
        if self._fatal_error is not None:
            raise RuntimeError(
                f"{self._fatal_error}; see {self._oob_status_path}"
            ) from self._fatal_error
        if self._runtime_proc is None:
            raise RuntimeError("CloudXR service is not running")

        exit_code = self._runtime_proc.poll()
        if exit_code is not None:
            raise RuntimeError(
                f"CloudXR runtime process exited unexpectedly (exit code {exit_code})"
            )

        if self._wss_thread is not None and not self._wss_thread.is_alive():
            raise RuntimeError(
                f"CloudXR WSS proxy thread stopped unexpectedly: {getattr(self, '_wss_error', None)}"
            )

    @property
    def wss_log_path(self) -> Path | None:
        """Path to the WSS proxy log file, or ``None`` if not yet started."""
        return self._wss_log_path

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _install_signal_handlers(self) -> None:
        """Install SIGTERM/SIGINT handlers that propagate to the prior handler.

        Propagating first lets with-block __exit__ teardown run in order so
        inner contexts (e.g. an OpenXR session) close before stop() kills the
        runtime.  stop() is reached via __exit__ and atexit.
        """
        if threading.current_thread() is not threading.main_thread():
            return

        self._restore_signal_handlers()

        def _make_handler(prev):
            def _handler(signum, frame):
                if callable(prev):
                    prev(signum, frame)
                elif prev == signal.SIG_DFL:
                    # Raise SystemExit so __exit__ and atexit handlers run.
                    raise SystemExit(0)
                # SIG_IGN: preserve the "ignore" disposition — no-op.

            return _handler

        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                prev = signal.getsignal(sig)
                handler = _make_handler(prev)
                signal.signal(sig, handler)
            except (ValueError, OSError):
                continue
            self._prev_signal_handlers[sig] = (prev, handler)

    def _restore_signal_handlers(self) -> None:
        """Restore handlers from :meth:`_install_signal_handlers` if still ours.

        Skip overwrite when a host replaced the wrapper. Keep the entry if
        restore fails (e.g. non-main thread) for a later attempt.
        """
        for sig in list(self._prev_signal_handlers):
            prev, ours = self._prev_signal_handlers[sig]
            try:
                if signal.getsignal(sig) is not ours:
                    del self._prev_signal_handlers[sig]
                    continue
                signal.signal(sig, prev)
            except (ValueError, OSError):
                continue
            del self._prev_signal_handlers[sig]

    @staticmethod
    def _cleanup_stale_runtime(run_dir: str) -> None:
        """Refuse to start over a live runtime; clear stale sentinels otherwise.

        A run directory holds one runtime.  Liveness is decided by connecting
        to the IPC socket, not by its existence — the file routinely outlives
        the process that made it.

        Raises:
            RuntimeError: If a runtime is already serving the run directory.
        """
        if is_runtime_live(run_dir):
            raise RuntimeError(
                _RUNTIME_ALREADY_SERVING.format(
                    run_dir=run_dir,
                    env_file=os.path.join(run_dir, ENV_FILE_NAME),
                )
            )

        for name in ("ipc_cloudxr", "runtime_started", "monado.pid", "cloudxr.pid"):
            path = os.path.join(run_dir, name)
            try:
                os.remove(path)
            except FileNotFoundError:
                continue
            logger.warning("Removed stale CloudXR runtime file \033[90m%s\033[0m", path)

    def _collect_startup_failure_detail(self, logs_dir: Path) -> str:
        """Build a diagnostic string after a failed runtime startup.

        Captures the process exit code, the worker's stderr, this session's
        native-fd capture file (where the worker's own fd 1 goes), and the
        most recent CloudXR native server log. See
        :meth:`_gather_diagnostic_logs`.
        """
        _MAX_LOG_BYTES = 4096
        parts: list[str] = []
        proc = self._runtime_proc
        if proc is not None:
            exit_code = proc.poll()
            if exit_code is not None:
                parts.append(f"Process exited with code {exit_code}.")
            else:
                parts.append("Process is still running but did not signal readiness.")

        for log_path in self._gather_diagnostic_logs(logs_dir):
            try:
                content = _tail_text(log_path, _MAX_LOG_BYTES)
                if not content:
                    continue
                parts.append(f"{log_path.name}:\n{content}")
            except Exception:
                pass

        parts.append(f"Check logs under {logs_dir} for details.")
        return "  ".join(parts)

    @staticmethod
    def _gather_diagnostic_logs(logs_dir: Path) -> list[Path]:
        """Return log files useful for diagnosing a startup failure."""
        result: list[Path] = []

        worker_stderr = logs_dir / _WORKER_STDERR_LOG
        if worker_stderr.is_file():
            result.append(worker_stderr)

        # With logging off, runtime.run() redirects its own stderr here.
        runtime_stderr = logs_dir / "runtime_stderr.log"
        if not logging_enabled() and runtime_stderr.is_file():
            result.append(runtime_stderr)

        # The worker's own fd 1 lands in *this* process's capture file, not
        # under `logs_dir` and not in the worker's own (which stays empty --
        # nothing rebinds that process's descriptors onto it): it is launched
        # with stdout=native_capture_fd(), a descriptor onto this one.
        capture = native_capture_path()
        if capture is not None and capture.is_file():
            result.append(capture)

        cxr_logs = sorted(logs_dir.glob("cxr_server.*.log"))
        if cxr_logs:
            result.append(cxr_logs[-1])

        return result

    def _is_runtime_alive(self) -> bool:
        """Return whether the runtime subprocess is still running."""
        return self._runtime_proc is not None and self._runtime_proc.poll() is None

    def _terminate_runtime(self) -> None:
        """Terminate the runtime subprocess and all its children.

        On POSIX, the subprocess is launched with ``start_new_session=True``
        so it leads its own process group; ``killpg`` tears down Monado and
        other children.  Windows is not supported (see
        :meth:`_terminate_runtime_windows`).
        """
        proc = self._runtime_proc
        if proc is None or proc.poll() is not None:
            return

        if sys.platform == "win32":
            self._terminate_runtime_windows(proc)
            return

        try:
            pgid = os.getpgid(proc.pid)
        except ProcessLookupError:
            return

        try:
            os.killpg(pgid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            proc.wait(timeout=RUNTIME_TERMINATE_TIMEOUT_SEC)
        except subprocess.TimeoutExpired:
            pass

        if proc.poll() is None:
            try:
                os.killpg(pgid, signal.SIGKILL)
            except ProcessLookupError:
                return
            try:
                proc.wait(timeout=RUNTIME_TERMINATE_TIMEOUT_SEC)
            except subprocess.TimeoutExpired:
                pass

        if proc.poll() is None:
            raise RuntimeError("Failed to terminate or kill runtime process group")

    @staticmethod
    def _terminate_runtime_windows(_proc: subprocess.Popen) -> None:
        """Windows runtime termination is not supported."""
        raise RuntimeError(
            "CloudXR runtime process termination is not supported on Windows"
        )

    # ------------------------------------------------------------------
    # WSS proxy (background thread with its own event loop)
    # ------------------------------------------------------------------

    def _start_wss_proxy_thread(
        self, log_path: Path, timeout_sec: float = WSS_STARTUP_TIMEOUT_SEC
    ) -> None:
        """Launch the WSS proxy in a daemon thread and wait for it to listen.

        Returning before the socket is bound would report a proxy that is not
        yet serving, and would leave a bind failure (a taken port, a bad cert
        pair) visible only to a later :meth:`health_check`.  A caller that
        exits straight after construction would also be finalizing the
        interpreter while the proxy is still starting up.

        The thread stays a daemon: it parks on *stop_future*, which only
        :meth:`stop` resolves, and that runs from ``atexit`` — after
        ``threading._shutdown`` would have joined a non-daemon thread.

        Raises:
            RuntimeError: If the proxy fails or does not listen within
                *timeout_sec*.
        """
        from ..wss import run as wss_run  # noqa: PLC0415

        loop = asyncio.new_event_loop()
        self._wss_loop = loop
        stop_future = loop.create_future()
        self._wss_stop_future = stop_future
        listening: concurrent.futures.Future = concurrent.futures.Future()

        setup_oob = self._setup_oob
        usb_local = self._usb_local
        host_client = self._host_client

        def _run_wss() -> None:
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(
                    wss_run(
                        log_file_path=log_path,
                        stop_future=stop_future,
                        setup_oob=setup_oob,
                        usb_local=usb_local,
                        host_client=host_client,
                        on_listening=lambda: listening.set_result(None),
                        recovery_config=self._recovery_config,
                        on_oob_status=self._publish_oob_status,
                        on_oob_fatal=self._on_oob_fatal,
                    )
                )
            except Exception as exc:
                self._wss_error = exc
                if not listening.done():
                    listening.set_exception(exc)
                logger.exception("WSS proxy thread exited with error")
            finally:
                # A thread that ended without listening leaves nobody to wait
                # for; unblock the caller rather than hold it to the timeout.
                if not listening.done():
                    listening.set_result(None)
                loop.close()

        self._wss_thread = threading.Thread(
            target=_run_wss, name="cloudxr-wss-proxy", daemon=True
        )
        self._wss_thread.start()

        try:
            listening.result(timeout=timeout_sec)
        except concurrent.futures.TimeoutError as exc:
            raise RuntimeError(
                f"CloudXR WSS proxy did not start listening within "
                f"{timeout_sec}s (log: {log_path})"
            ) from exc
        except Exception as exc:
            raise RuntimeError(
                f"CloudXR WSS proxy failed to start (log: {log_path}): {exc}"
            ) from exc

    def _stop_wss_proxy(self) -> None:
        """Signal the WSS proxy to shut down and wait for the thread."""
        if self._wss_loop is not None and self._wss_stop_future is not None:
            loop = self._wss_loop
            future = self._wss_stop_future

            def _set_result() -> None:
                if not future.done():
                    future.set_result(None)

            if not loop.is_closed():
                try:
                    loop.call_soon_threadsafe(_set_result)
                except RuntimeError:
                    logger.debug(
                        "WSS event loop closed before stop signal; "
                        "proxy already shut down"
                    )

        if self._wss_thread is not None:
            self._wss_thread.join(timeout=15 if self._setup_oob else 5)
            if self._wss_thread.is_alive():
                logger.warning("WSS proxy thread did not exit cleanly")

        self._wss_thread = None
        self._wss_loop = None
        self._wss_stop_future = None
