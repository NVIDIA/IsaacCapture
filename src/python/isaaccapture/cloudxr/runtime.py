# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import ctypes
import glob
import importlib.util
import multiprocessing
import os
import shutil
import signal
import socket
import sys
import tempfile
import threading
import time
from collections.abc import Callable
from pathlib import Path

from ..logging_config._core import logging_enabled
from .env_config import get_env_config


_EULA_URL = (
    "https://github.com/NVIDIA/IsaacTeleop/blob/main/deps/cloudxr/CLOUDXR_LICENSE"
)

RUNTIME_STARTUP_TIMEOUT_SEC: float = 30
"""Maximum time [s] to wait for the runtime ``runtime_started`` sentinel."""

WSS_STARTUP_TIMEOUT_SEC: float = 30
"""Maximum time [s] to wait for the WSS proxy to bind its listening socket."""

RUNTIME_TERMINATE_TIMEOUT_SEC: float = 10
"""Timeout [s] for each escalation step (SIGTERM, then SIGKILL) when stopping the runtime."""

RUNTIME_POLL_INTERVAL_SEC: float = 0.5
"""Polling interval [s] used by :func:`wait_for_runtime_ready_sync`."""

IPC_PROBE_TIMEOUT_SEC: float = 0.5
"""Connect timeout [s] for :func:`is_runtime_live`."""

_CLOUDXR_EXP_ENV = "ISAAC_TELEOP_CLOUDXR_EXP"
_CLOUDXR_JOIN_MAIN_ENV = "ISAAC_TELEOP_CLOUDXR_JOIN_MAIN"
_CLOUDXR_MODULE = "isaaccapture.cloudxr"
_CLOUDXR_EXP_MODULE = "isaaccapture.cloudxr_exp"
_CLOUDXR_NATIVE_SUBDIR = "native"
_CLOUDXR_RUNTIME_LIB = "libcloudxr.so"


_TEGRA_T234_CHIP_ID = 0x23
_TEGRA_T234_JEP106_ID = "jep106:036b:0234"  # Arm SMCCC SoC ID: NVIDIA JEDEC ID + part
_SOC_DEVICES = Path("/sys/bus/soc/devices")
_DEVICE_TREE_COMPATIBLE = Path("/proc/device-tree/compatible")


def _read_text(path: Path) -> str:
    """Return the file contents, or "" if it is missing or unreadable."""
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""


def _is_tegra_t234() -> bool:
    """Return True on Jetson Orin-class platforms (T234 / CHIPID 0x23).

    Only the sysfs SoC devices are visible inside Docker, which masks
    /sys/firmware (hence /proc/device-tree). The ``socN`` index depends on probe
    order, so scan them all.
    """
    for soc in sorted(_SOC_DEVICES.glob("soc*")):
        family = _read_text(soc / "family").strip()
        soc_id = _read_text(soc / "soc_id").strip()
        if (
            family == "Tegra"
            and soc_id.isdigit()
            and int(soc_id) == _TEGRA_T234_CHIP_ID
        ):
            return True
        if soc_id == _TEGRA_T234_JEP106_ID:
            return True

    return "tegra234" in _read_text(_DEVICE_TREE_COMPATIBLE)


def _should_use_exp() -> bool:
    """Return True when the experimental CloudXR runtime should be used."""
    raw = os.environ.get(_CLOUDXR_EXP_ENV, "").strip().lower()
    if not raw:
        return _is_tegra_t234()
    return raw in ("1", "true", "yes", "on")


def _should_join_main() -> bool:
    """Return True when ``nv_cxr_service_join`` should run on the main thread.

    Default is auto-on for Jetson Orin (T234). Override with
    ``ISAAC_TELEOP_CLOUDXR_JOIN_MAIN`` (``1``/``0``). See the join-main
    branch in :func:`run` for why new Python threads are forbidden there.
    """
    raw = os.environ.get(_CLOUDXR_JOIN_MAIN_ENV, "").strip().lower()
    if not raw:
        return _is_tegra_t234()
    return raw in ("1", "true", "yes", "on")


def _package_search_roots(package: str) -> list[str]:
    """Return every filesystem root of ``package``'s ``__path__``, or [] if unlocatable."""
    # Bare package name only: find_spec imports the parent of a dotted name, so
    # probing "<pkg>.runtime" would execute the package as a side effect.
    try:
        spec = importlib.util.find_spec(package)
    except ImportError:
        return []
    except ValueError:  # a parent sits in sys.modules with __spec__ = None
        return []
    if spec is None:
        return []
    return list(spec.submodule_search_locations or ())


def _find_native_runtime_dir(package: str) -> str | None:
    """Return the ``native/`` dir holding the CloudXR runtime, or None."""
    # Scan every root, never [0]: scikit-build-core builds them from a set, so
    # order is unstable, and under a split install only one carries the runtime.
    # Test the file, not the dir: a missing SDK tarball leaves native/ empty.
    for root in _package_search_roots(package):
        native_dir = os.path.join(root, _CLOUDXR_NATIVE_SUBDIR)
        if os.path.isfile(os.path.join(native_dir, _CLOUDXR_RUNTIME_LIB)):
            return native_dir
    return None


def _is_exp_available() -> bool:
    """Return True when the experimental package ships the CloudXR runtime."""
    # Probe the artifact, not the module: cloudxr_exp is authored Python that
    # ships in every wheel, so importability says nothing about the bundle.
    return _find_native_runtime_dir(_CLOUDXR_EXP_MODULE) is not None


def _format_roots(roots: list[str]) -> str:
    return ", ".join(roots) if roots else "<package not importable>"


def resolve_cloudxr_runtime_module() -> str:
    """Return the module prefix whose ``runtime.run`` should be spawned."""
    if _should_use_exp():
        if not _is_exp_available():
            roots = _format_roots(_package_search_roots(_CLOUDXR_EXP_MODULE))
            raise RuntimeError(
                f"Experimental CloudXR runtime package {_CLOUDXR_EXP_MODULE} does not "
                f"bundle {_CLOUDXR_NATIVE_SUBDIR}/{_CLOUDXR_RUNTIME_LIB}; searched "
                f"{roots}. Rebuild with -DENABLE_CLOUDXR_EXP_BUNDLE=ON to bundle it, "
                f"or set {_CLOUDXR_EXP_ENV}=0 to use the stable runtime."
            )
        return _CLOUDXR_EXP_MODULE
    return _CLOUDXR_MODULE


def get_sdk_path() -> str:
    """Return ``native/`` for the resolved stable or experimental runtime package."""
    mod = resolve_cloudxr_runtime_module()
    sdk_dir = _find_native_runtime_dir(mod)
    if sdk_dir is None:
        roots = _format_roots(_package_search_roots(mod))
        raise RuntimeError(
            f"CloudXR SDK missing {_CLOUDXR_NATIVE_SUBDIR}/{_CLOUDXR_RUNTIME_LIB} "
            f"under every {mod} root; searched {roots}."
        )
    return sdk_dir


def _write_eula_marker(marker: str) -> None:
    run_dir = os.path.dirname(marker)
    os.makedirs(run_dir, mode=0o700, exist_ok=True)
    with open(marker, "w") as f:
        f.write("accepted\n")


def eula_marker(run_dir: str) -> str:
    """Path of the EULA acceptance marker under *run_dir*."""
    return os.path.join(run_dir, "eula_accepted")


def check_eula(*, accept_eula: bool | None = None, run_dir: str | None = None) -> None:
    """Require CloudXR EULA to be accepted; exits the process if not. Call from main process before spawning runtime.

    Args:
        accept_eula: If True, accept and write marker. If None, check marker then prompt interactively.
        run_dir: Where the marker lives.  Pass it to prompt before a service
            exists — resolving it from :class:`EnvConfig` would rewrite the
            env file of a runtime that may already be running.
    """
    marker = eula_marker(run_dir or get_env_config().openxr_run_dir())
    if os.path.isfile(marker):
        return

    if accept_eula is True:
        _write_eula_marker(marker)
        return

    print(
        "\nNVIDIA CloudXR EULA must be accepted to run. View: " + _EULA_URL,
        file=sys.stderr,
    )
    try:
        reply = input("\nAccept NVIDIA CloudXR EULA? [y/N]: ").strip().lower()
    except EOFError:
        reply = ""
    if reply not in ("y", "yes"):
        print("EULA not accepted. Exiting.", file=sys.stderr)
        sys.exit(1)

    _write_eula_marker(marker)


async def wait_for_runtime_ready(
    is_process_alive: Callable[[], bool],
    timeout_sec: float = RUNTIME_STARTUP_TIMEOUT_SEC,
    poll_interval_sec: float = RUNTIME_POLL_INTERVAL_SEC,
) -> bool:
    """Poll for the ``runtime_started`` sentinel file.

    This is the canonical implementation used by both the async callers
    and :func:`wait_for_runtime_ready_sync`.

    Args:
        is_process_alive: Callable returning ``True`` while the
            runtime process is still running.
        timeout_sec: Maximum time to wait [s].
        poll_interval_sec: Polling interval [s].

    Returns:
        ``True`` when the runtime is ready, ``False`` on timeout or
        if the process exits early.
    """
    lock_file = os.path.join(get_env_config().openxr_run_dir(), "runtime_started")
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_sec

    while loop.time() < deadline:
        if not is_process_alive():
            return False
        if os.path.isfile(lock_file):
            return True
        await asyncio.sleep(poll_interval_sec)

    return False


def wait_for_runtime_ready_sync(
    is_process_alive: Callable[[], bool],
    timeout_sec: float = RUNTIME_STARTUP_TIMEOUT_SEC,
    poll_interval_sec: float = RUNTIME_POLL_INTERVAL_SEC,
) -> bool:
    """Synchronous poll for the ``runtime_started`` sentinel file.

    Unlike :func:`wait_for_runtime_ready`, this implementation uses
    :func:`time.monotonic` and :func:`time.sleep` so it is safe to call
    from threads or processes that already have a running asyncio event
    loop (e.g. Omniverse Kit / Isaac Sim).

    Args:
        is_process_alive: Callable returning ``True`` while the
            runtime process is still running.
        timeout_sec: Maximum time to wait [s].
        poll_interval_sec: Polling interval [s].

    Returns:
        ``True`` when the runtime is ready, ``False`` on timeout or
        if the process exits early.
    """
    lock_file = os.path.join(get_env_config().openxr_run_dir(), "runtime_started")
    deadline = time.monotonic() + timeout_sec

    while time.monotonic() < deadline:
        if not is_process_alive():
            return False
        if os.path.isfile(lock_file):
            return True
        time.sleep(poll_interval_sec)

    return False


def is_runtime_live(run_dir: str) -> bool:
    """Return whether a CloudXR runtime is currently serving ``run_dir``.

    The socket file outliving its process is the common case, so existence
    proves nothing; only a successful ``connect()`` does. Ambiguous errors
    (permissions, timeout) count as live: refusing to start is recoverable,
    tearing down someone else's session is not.
    """
    path = os.path.join(run_dir, "ipc_cloudxr")
    if not os.path.exists(path):
        return False
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        sock.settimeout(IPC_PROBE_TIMEOUT_SEC)
        sock.connect(path)
        return True
    except (ConnectionRefusedError, FileNotFoundError):
        return False
    except OSError:
        return True
    finally:
        sock.close()


def _load_libcloudxr(sdk_path: str) -> ctypes.CDLL:
    """Load libcloudxr.so with RTLD_DEEPBIND so the native stack resolves symbols from its own packaged libraries."""
    deepbind = getattr(os, "RTLD_DEEPBIND", 0)
    return ctypes.CDLL(os.path.join(sdk_path, _CLOUDXR_RUNTIME_LIB), mode=deepbind)


def runtime_version() -> str:
    """Query the selected CloudXR runtime version (major.minor.patch)."""
    sdk_path = get_sdk_path()
    lib = _load_libcloudxr(sdk_path)
    if not hasattr(lib, "nv_cxr_get_runtime_version"):
        return "unknown"
    major, minor, patch = ctypes.c_uint32(), ctypes.c_uint32(), ctypes.c_uint32()
    lib.nv_cxr_get_runtime_version(
        ctypes.byref(major), ctypes.byref(minor), ctypes.byref(patch)
    )
    return f"{major.value}.{minor.value}.{patch.value}"


def latest_runtime_log(logs_dir: Path | None = None) -> str | None:
    """Return the path to the most recent cxr_server log file, or None if not found.

    *logs_dir* is required outside the service's own process: resolving it from
    :class:`EnvConfig` would rewrite the running service's ``cloudxr.env``.
    """
    logs_dir = logs_dir or get_env_config().ensure_logs_dir()
    candidates = sorted(glob.glob(str(logs_dir / "cxr_server.*.log")))
    return candidates[-1] if candidates else None


def latest_wss_log(logs_dir: Path | None = None) -> str | None:
    """Return the path to the most recent WSS proxy log file, or None.

    The service holds this path while it runs; other processes have to find it,
    and must pass *logs_dir* rather than re-resolving :class:`EnvConfig`.
    """
    logs_dir = logs_dir or get_env_config().ensure_logs_dir()
    candidates = sorted(glob.glob(str(logs_dir / "wss.*.log")))
    return candidates[-1] if candidates else None


def terminate_or_kill_runtime(process: multiprocessing.Process) -> None:
    """Terminate or kill a :class:`multiprocessing.Process` runtime."""
    if process.is_alive():
        process.terminate()
        process.join(timeout=RUNTIME_TERMINATE_TIMEOUT_SEC)
    if process.is_alive():
        process.kill()
        process.join(timeout=RUNTIME_TERMINATE_TIMEOUT_SEC)
    if process.is_alive():
        raise RuntimeError("Failed to terminate or kill runtime process")


def _setup_openxr_dir(sdk_path: str, run_dir: str) -> str:
    """Create run dir and copy OpenXR runtime lib + json; return path to openxr dir (parent of run)."""
    openxr_dir = os.path.dirname(run_dir)
    os.makedirs(run_dir, mode=0o755, exist_ok=True)

    lib_src = os.path.join(sdk_path, "libopenxr_cloudxr.so")
    json_src = os.path.join(sdk_path, "openxr_cloudxr.json")
    for name, src in (
        ("libopenxr_cloudxr.so", lib_src),
        ("openxr_cloudxr.json", json_src),
    ):
        if not os.path.isfile(src):
            raise RuntimeError(f"CloudXR SDK missing {name} at {src}. ")
        dst = os.path.join(openxr_dir, name)
        # Replace through the directory so legacy read-only files do not block upgrades.
        with tempfile.NamedTemporaryFile(
            prefix=f".{name}.", dir=openxr_dir, delete=False
        ) as temp:
            temp_path = temp.name
        try:
            shutil.copy2(src, temp_path)
            os.replace(temp_path, dst)
        finally:
            try:
                os.remove(temp_path)
            except FileNotFoundError:
                pass

    for stale in ("ipc_cloudxr", "runtime_started", "monado.pid", "cloudxr.pid"):
        p = os.path.join(run_dir, stale)
        if os.path.exists(p):
            os.remove(p)

    return openxr_dir


def run() -> None:
    """Run the CloudXR runtime service until SIGINT/SIGTERM. Blocks until shutdown."""
    cfg = get_env_config()
    sdk_path = get_sdk_path()
    run_dir = cfg.openxr_run_dir()
    openxr_dir = _setup_openxr_dir(sdk_path, run_dir)

    expected_json = os.path.join(openxr_dir, "openxr_cloudxr.json")
    for var, expected in (
        ("XR_RUNTIME_JSON", expected_json),
        ("NV_CXR_RUNTIME_DIR", run_dir),
    ):
        actual = os.environ.get(var)
        if actual is None:
            raise RuntimeError(
                f"{var} is not set. Source setup_cloudxr_env.sh before running."
            )
        if os.path.abspath(actual) != os.path.abspath(expected):
            raise RuntimeError(
                f"{var} mismatch: environment has {actual!r}, expected {expected!r}"
            )

    prev_ld = os.environ.get("LD_LIBRARY_PATH", "")
    os.environ["LD_LIBRARY_PATH"] = sdk_path + (f":{prev_ld}" if prev_ld else "")

    # With logging on, fd 1/fd 2 -- the native library's console banner and its
    # Vulkan-loader / GPU-init diagnostics -- are deliberately left alone. This
    # process exists only to host that native stack, and its launcher
    # (cloudxr/service/_service.py) already pointed its descriptors where the
    # output belongs: the session's capture file for stdout, the worker's own
    # stderr log for stderr. Redirecting here would only fight it.
    if not logging_enabled():
        # The native library logs in detail to NV_CXR_OUTPUT_DIR: drop its console
        # banner and keep stderr (GPU-init errors, tracebacks) in runtime_stderr.log.
        _file_logging = os.environ.get("NV_CXR_FILE_LOGGING", "yes")
        if _file_logging and _file_logging.lower() not in (
            "false",
            "off",
            "no",
            "n",
            "f",
            "0",
        ):
            logs_dir = cfg.ensure_logs_dir()
            stderr_log = os.path.join(str(logs_dir), "runtime_stderr.log")
            devnull_fd = os.open(os.devnull, os.O_WRONLY)
            stderr_fd = os.open(
                stderr_log, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644
            )
            os.dup2(devnull_fd, sys.stdout.fileno())
            os.dup2(stderr_fd, sys.stderr.fileno())
            os.close(devnull_fd)
            os.close(stderr_fd)

    lib = _load_libcloudxr(sdk_path)
    svc = ctypes.c_void_p()
    # Signal handler must only call stop() after create() has run; avoid calling with null svc.
    state = {"service_created": False, "interrupted": False}

    def stop(sig: int, frame: object) -> None:
        if not state["service_created"]:
            return
        state["interrupted"] = sig == signal.SIGINT
        lib.nv_cxr_service_stop(svc)

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    lib.nv_cxr_service_create(ctypes.byref(svc))
    state["service_created"] = True
    lib.nv_cxr_service_start(svc)

    if _should_join_main():
        # Orin (T234) workaround: after nv_cxr_service_create/start, starting
        # any additional Python thread aborts with:
        #   _PyGILState_NoteThreadState: Couldn't create autoTSSkey mapping
        # That includes both the historical join worker thread and a later
        # attempt to recover Ctrl+C via a sigwait helper thread. Stay on the
        # main thread for join+destroy only.
        #
        # Tradeoff: while join blocks, Python signal handlers on this process
        # do not run. Shutdown is the launcher tearing down this worker
        # process (SIGTERM/SIGINT + atexit -> nv_cxr_service_stop), not an
        # in-process signal path during join.
        lib.nv_cxr_service_join(svc)
        lib.nv_cxr_service_destroy(svc)
    else:
        # Non-Orin: join in a worker so the main thread can run signal handlers
        # (Ctrl+C) while blocked in native nv_cxr_service_join().
        def join_then_destroy() -> None:
            lib.nv_cxr_service_join(svc)
            lib.nv_cxr_service_destroy(svc)

        worker = threading.Thread(target=join_then_destroy, daemon=False)
        worker.start()
        worker.join()

    if state["interrupted"]:
        raise KeyboardInterrupt()
