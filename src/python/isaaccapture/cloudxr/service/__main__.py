# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CLI for the CloudXR service: run it in the foreground, or detached."""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable

from ._service import CloudXRService


def _add_install_dir_argument(parser: argparse.ArgumentParser) -> None:
    """Register ``--cloudxr-install-dir``, which every command needs."""
    parser.add_argument(
        "--cloudxr-install-dir",
        type=str,
        default=os.path.expanduser("~/.cloudxr"),
        metavar="PATH",
        help="CloudXR install directory (default: ~/.cloudxr)",
    )


def _add_run_arguments(parser: argparse.ArgumentParser) -> None:
    """Register the flags that shape a running service.

    Shared by ``run`` and ``start``: whatever ``start`` accepts here it passes
    through to the detached ``run``.
    """
    _add_install_dir_argument(parser)
    parser.add_argument(
        "--cloudxr-env-config",
        type=str,
        default=None,
        metavar="PATH",
        help="Optional env file (KEY=value per line) to override default CloudXR env vars",
    )
    parser.add_argument(
        "--accept-eula",
        action="store_true",
        help="Accept the NVIDIA CloudXR EULA non-interactively (e.g. for CI or containers).",
    )
    parser.add_argument(
        "--setup-oob",
        action="store_true",
        default=False,
        help=(
            "Enable OOB teleop control hub, open the teleop page on the headset via USB adb, "
            "and auto-click CONNECT via CDP (Chrome DevTools Protocol). "
            "The headset must be connected via USB cable (for adb) and on WiFi (for streaming). "
            'See docs: "Out-of-band teleop control".'
        ),
    )
    parser.add_argument(
        "--usb-local",
        action="store_true",
        default=False,
        help=(
            "Route teleop traffic over the USB cable on headset loopback "
            "(127.0.0.1) via adb reverse.  Requires --setup-oob.  Requires "
            "`coturn` and `adb` on PATH.  Serves /client/ on the WSS proxy "
            "port (same path as --host-client; does not set that flag)."
        ),
    )
    parser.add_argument(
        "--host-client",
        action="store_true",
        default=False,
        help=(
            "Serve the web client at /client/ on the WSS proxy port (default 48322), "
            "fetched once from the matching versioned release into "
            "TELEOP_WEB_CLIENT_STATIC_DIR or ~/.cloudxr/static-client."
        ),
    )


def _run_flags(args: argparse.Namespace) -> list[str]:
    """Re-serialise the run flags in *args* that differ from their defaults.

    ``start`` passes these through to the detached ``run``.  ``--accept-eula``
    is not among them: acceptance is recorded as a marker file by ``start``
    itself, in the process where the operator actually consented.
    """
    flags: list[str] = []
    if args.cloudxr_install_dir != os.path.expanduser("~/.cloudxr"):
        flags += ["--cloudxr-install-dir", args.cloudxr_install_dir]
    if args.cloudxr_env_config:
        flags += ["--cloudxr-env-config", args.cloudxr_env_config]
    for name in ("setup_oob", "usb_local", "host_client"):
        if getattr(args, name):
            flags.append("--" + name.replace("_", "-"))
    return flags


_ANSI = re.compile(r"\033\[[0-9;]*m")


def _out(text: str = "") -> None:
    """Print to stdout, dropping colour when it is not a terminal.

    ``service start`` redirects this into ``service.log``, where escape codes
    are noise rather than emphasis.
    """
    print(_ANSI.sub("", text) if not sys.stdout.isatty() else text)


def _out_interactive(text: str = "") -> None:
    """Print only where a person could act on it.

    Instructions about the terminal are false in ``service.log``: a detached
    service has none, and nobody is there to press Ctrl+C.
    """
    if sys.stdout.isatty():
        _out(text)


def _fail(message: str) -> None:
    """Print *message* in red on stderr and exit 1."""
    print(f"\n\033[31m{message}\033[0m\n", file=sys.stderr)
    raise SystemExit(1)


def _oob_preflight(args: argparse.Namespace) -> str | None:
    """Check host prerequisites; headset readiness belongs to the lifecycle.

    Valid flag combinations:

    ==============================  ==================================================
    (none)                          headset navigates to the GitHub Pages URL over WiFi
    ``--host-client``               client served at ``https://<lan>:<port>/client/``
    ``--setup-oob``                 OOB hub + CDP automation; GitHub Pages URL
    ``--setup-oob --host-client``   OOB hub + CDP; client on the WSS proxy
    ``--setup-oob --usb-local``     OOB hub + CDP; adb-reverse + coturn; /client/ on WSS
    ==============================  ==================================================
    """
    from ..oob_teleop_adb import (  # noqa: PLC0415
        OobAdbError,
        require_adb_on_path,
        require_coturn_available,
        require_turn_port_free,
    )
    from ..oob_teleop_env import (  # noqa: PLC0415
        oob_progress,
        print_host_preflight_warnings,
        resolve_lan_host_for_oob,
        resolve_oob_recovery_config,
        usb_turn_port,
    )

    if args.setup_oob:
        try:
            config = resolve_oob_recovery_config()
        except ValueError as exc:
            _fail(str(exc))
        args._oob_recovery_config = config
        oob_progress(
            "setup-oob",
            f"recovery episode {config.timeout_sec:g}s, retry/observe {config.interval_sec:g}s",
        )

    if args.usb_local:
        oob_progress(
            "usb-local",
            "host preflight: adb, coturn, ports, assets ...",
        )
        require_adb_on_path()
        try:
            require_coturn_available()
            require_turn_port_free(usb_turn_port())
        except OobAdbError as exc:
            _fail(str(exc))
        try:
            print_host_preflight_warnings(usb_local=True)
        except RuntimeError as exc:
            _fail(str(exc))
        oob_progress("usb-local", "host preflight OK; waiting for headset after listen")
        return None

    if not args.setup_oob:
        return None

    # TELEOP_OOB_HUB_ONLY skips every adb step — the hub starts, but the
    # operator opens the teleop page on the headset themselves.
    hub_only = bool(os.getenv("TELEOP_OOB_HUB_ONLY"))
    if hub_only:
        oob_progress(
            "setup-oob", "hub-only mode (TELEOP_OOB_HUB_ONLY) — skipping adb preflight"
        )
    else:
        oob_progress("setup-oob", "host preflight: adb and network configuration ...")
        require_adb_on_path()
    lan_host = resolve_lan_host_for_oob()
    try:
        print_host_preflight_warnings(usb_local=False)
    except RuntimeError as exc:
        _fail(str(exc))
    oob_progress("setup-oob", "host preflight OK; headset automation may be pending")
    return lan_host


def _env_file_value(run_dir: str, key: str) -> str | None:
    """Read *key* out of the env file the running service wrote.

    ``cloudxr.env`` is shell format — ``export KEY='value'`` — so it cannot be
    fed back through ``EnvConfig``'s reader, and re-resolving it here would
    overwrite the running service's own file.
    """
    import shlex  # noqa: PLC0415

    try:
        text = Path(run_dir, "cloudxr.env").read_text(encoding="utf-8")
    except OSError:
        return None
    for line in text.splitlines():
        name, _, value = line.removeprefix("export ").partition("=")
        if name.strip() == key:
            return next(iter(shlex.split(value)), None)
    return None


def _print_service_summary(
    args: argparse.Namespace,
    *,
    wss_log: object,
    device_profile: str | None,
    env_file: str,
    logs_dir: Path | None = None,
    oob_lan_host: str | None = None,
    include_oob: bool = False,
) -> None:
    """Print the operator-facing summary of a running service.

    ``include_oob`` prints the full OOB hub block, which only the foreground
    ``run`` does; ``start`` and ``status`` show the connection URL instead.
    """
    from isaaccapture import __version__ as isaaccapture_version  # noqa: PLC0415
    from ..oob_teleop_env import (  # noqa: PLC0415
        USB_HOST,
        guess_lan_ipv4,
        print_hosted_client_line,
        print_oob_hub_startup_banner,
        versioned_web_client_url,
        wss_proxy_port,
    )
    from ..runtime import latest_runtime_log, runtime_version  # noqa: PLC0415

    _out(
        f"Running Isaac Teleop \033[36m{isaaccapture_version}\033[0m, "
        f"CloudXR Runtime \033[36m{runtime_version()}\033[0m"
    )

    cxr_log = latest_runtime_log(logs_dir) or "(none yet)"
    _out(
        f"CloudXR runtime:   \033[36mrunning\033[0m, log file: \033[90m{cxr_log}\033[0m"
    )
    _out(
        f"CloudXR WSS proxy: \033[36mrunning\033[0m, log file: \033[90m{wss_log}\033[0m"
    )
    # A profile that does not match the connecting device is the usual cause of
    # XR_ERROR_FORM_FACTOR_UNAVAILABLE (-35) in clients.
    _out(
        f"device profile:    \033[36m{device_profile}\033[0m  "
        "\033[90m(NV_DEVICE_PROFILE)\033[0m"
    )

    if args.usb_local:
        # Bookmark uses localhost; summary uses 127.0.0.1 (same listener).
        hosted_client_url = f"https://127.0.0.1:{wss_proxy_port()}/client/"
    elif args.host_client:
        hosted_client_url = (
            f"https://{guess_lan_ipv4() or 'localhost'}:{wss_proxy_port()}/client/"
        )
    else:
        hosted_client_url = None

    if include_oob and args.setup_oob:
        if args.usb_local:
            _out(
                "        oob:       \033[32menabled\033[0m  "
                "(hub + USB-local: adb reverse + coturn)"
            )
            print_oob_hub_startup_banner(lan_host=USB_HOST, usb_local=True)
        else:
            suffix = " + host-client" if args.host_client else ""
            _out(
                f"        oob:       \033[32menabled\033[0m  (hub + USB adb "
                f"automation{suffix} — see OOB TELEOP block)"
            )
            print_oob_hub_startup_banner(
                lan_host=oob_lan_host, web_client_base=hosted_client_url
            )
    elif hosted_client_url is not None:
        print_hosted_client_line(hosted_client_url)
    else:
        _out(
            f"web client:        \033[36m{versioned_web_client_url(isaaccapture_version)}\033[0m"
        )

    _out(
        "Activate CloudXR environment in another terminal: "
        f"\033[1;32msource {env_file}\033[0m"
    )


class _OobConsoleReporter:
    """Turn lifecycle snapshots into sparse, actionable launcher messages."""

    def __init__(self, clock: Callable[[], float] = time.monotonic):
        self._clock = clock
        self._last_emitted: dict[str, float] = {}
        self._adb_ready = False
        self._ever_ready = False
        self._last_health: str | None = None
        self._last_stage: tuple | None = None
        self._last_ignored: tuple | None = None

    def observe(self, snapshot: dict) -> list[str]:
        """Return messages for a new snapshot, with repeated states limited."""
        health = snapshot.get("health")
        if health == "starting":
            return []
        state = snapshot.get("state")
        reason = snapshot.get("reason") or "unknown reason"
        serial = snapshot.get("selectedSerial")
        ready = bool(snapshot.get("adbReady"))
        stage = (state, snapshot.get("episodeStartedAt"), snapshot.get("generation"))
        entered_stage = stage != self._last_stage
        messages: list[str] = []
        ignored = tuple(sorted(snapshot.get("ignoredSerials") or ()))
        ignored_signature = (ignored, ready, reason if not ready else None)

        def emit(key: str, message: str, interval: float = 30.0) -> None:
            now = self._clock()
            if now - self._last_emitted.get(key, float("-inf")) >= interval:
                self._last_emitted[key] = now
                messages.append(f"[setup-oob] {message}")

        if ready and not self._adb_ready:
            verb = "reconnected" if self._ever_ready else "detected"
            emit(
                "adb-ready",
                f"USB-attached HMD {verb} (serial {serial}); continuing OOB setup.",
                0,
            )
            self._ever_ready = True
        elif not ready and self._adb_ready:
            emit(
                "adb-lost",
                "USB-attached HMD connection lost; waiting for the selected headset.",
                0,
            )
        self._adb_ready = ready

        if ignored and ignored_signature != self._last_ignored:
            names = ", ".join(ignored)
            if ready:
                messages.append(
                    f"[setup-oob] Ignored ADB device(s) {names}; continuing with selected headset {serial}."
                )
            else:
                state_detail = (
                    " (offline)"
                    if "offline" in reason.lower()
                    else " (unauthorized)"
                    if "unauthorized" in reason.lower()
                    else ""
                )
                messages.append(
                    f"[setup-oob] Ignored ADB device(s) {names}; waiting for selected headset {serial}{state_detail} to reconnect."
                )
        self._last_ignored = ignored_signature

        if state == "WAITING_FOR_ADB" and not ready:
            if serial is None and not any(
                word in reason.lower()
                for word in ("unauthorized", "offline", "multiple")
            ):
                emit(
                    "no-hmd",
                    "No USB-attached HMD detected; connect a headset to continue.",
                    60,
                )
            else:
                emit(f"adb-wait:{reason}", f"Waiting for USB-attached HMD: {reason}")
        elif state == "PREPARING_DEVICE" and ready and entered_stage:
            emit("prepare", f"Preparing headset {serial} for OOB setup.", 0)
        elif state == "REBUILDING_USB" and entered_stage:
            emit("usb", "Configuring USB reverse rules and TURN.", 0)
        elif state == "AUTOMATING_BROWSER" and entered_stage:
            emit("browser", "Opening headset browser and connecting OOB client.", 0)
        elif state == "VERIFYING_BROWSER" and entered_stage:
            emit("verify", "Waiting for headset browser health report.", 0)
        elif state == "DEGRADED":
            emit(f"failure:{reason}", f"OOB recovery delayed: {reason}")
        elif health == "fatal":
            emit("fatal", f"OOB stopped: {reason}", 0)

        if health == "browser_ready" and self._last_health != health:
            emit("browser-ready", "OOB browser ready; waiting for OpenXR stream.")
        elif health == "active" and self._last_health != health:
            emit("active", "OOB stream active with fresh client metrics.")
        self._last_health = health
        self._last_stage = stage
        return messages


def _cmd_run(args: argparse.Namespace) -> int:
    """Run the service in the foreground until interrupted."""
    if args.usb_local and not args.setup_oob:
        _fail("--usb-local requires --setup-oob.")
    if args.usb_local and os.getenv("TELEOP_OOB_HUB_ONLY"):
        _fail(
            "TELEOP_OOB_HUB_ONLY is not compatible with --usb-local "
            "(hub-only mode supports WiFi setup only)."
        )

    oob_lan_host = _oob_preflight(args)

    try:
        service = CloudXRService(
            install_dir=args.cloudxr_install_dir,
            env_config=args.cloudxr_env_config,
            accept_eula=args.accept_eula,
            setup_oob=args.setup_oob,
            usb_local=args.usb_local,
            host_client=args.host_client,
            recovery_config=getattr(args, "_oob_recovery_config", None),
        )
    except RuntimeError as exc:
        # Operator-facing conditions (a live runtime, a rejected EULA); the
        # message is the whole point, so don't bury it in a traceback.
        _fail(str(exc))

    with service:
        run_dir, _ = _resolve_dirs(args)
        _print_service_summary(
            args,
            wss_log=service.wss_log_path,
            device_profile=_env_file_value(run_dir, "NV_DEVICE_PROFILE"),
            env_file=os.path.join(run_dir, "cloudxr.env"),
            oob_lan_host=oob_lan_host,
            include_oob=True,
        )
        _out_interactive("\033[33mKeep this terminal open, Ctrl+C to terminate.\033[0m")
        reporter = (
            _OobConsoleReporter()
            if args.setup_oob and not os.getenv("TELEOP_OOB_HUB_ONLY")
            else None
        )

        def report_oob_updates() -> None:
            if reporter is not None:
                for snapshot in service.drain_oob_updates():
                    for message in reporter.observe(snapshot):
                        _out(message)

        report_oob_updates()

        stop = False

        def on_signal(sig, frame):
            """Set the stop flag on SIGINT/SIGTERM."""
            nonlocal stop
            stop = True

        signal.signal(signal.SIGINT, on_signal)
        signal.signal(signal.SIGTERM, on_signal)

        while not stop:
            report_oob_updates()
            service.health_check()
            time.sleep(0.1)

    _out("Stopped.")
    return 0


def _resolve_dirs(args: argparse.Namespace) -> tuple[str, Path]:
    """Return the run and logs directories for *args*' install dir."""
    install_dir = Path(os.path.expanduser(args.cloudxr_install_dir))
    return str(install_dir / "run"), install_dir / "logs"


def _require_eula(args: argparse.Namespace, run_dir: str) -> None:
    """Record EULA acceptance, or explain why the detached service cannot ask."""
    from ..runtime import _EULA_URL, _write_eula_marker, eula_marker  # noqa: PLC0415

    marker = eula_marker(run_dir)
    if os.path.isfile(marker):
        return
    if not args.accept_eula:
        _fail(
            "The NVIDIA CloudXR EULA has not been accepted.  A detached "
            "service has no terminal to prompt on, so accept it here:\n"
            "  python -m isaaccapture.cloudxr.service start --accept-eula\n"
            "Review it first: " + _EULA_URL
        )
    os.makedirs(run_dir, mode=0o700, exist_ok=True)
    _write_eula_marker(marker)
    _out(f"Recorded EULA acceptance: {marker}")


def _print_summary_for(args: argparse.Namespace, run_dir: str, logs_dir: Path) -> None:
    """Summarise a service running in another process."""
    from ..runtime import latest_wss_log  # noqa: PLC0415

    _print_service_summary(
        args,
        wss_log=latest_wss_log(logs_dir) or "(none yet)",
        device_profile=_env_file_value(run_dir, "NV_DEVICE_PROFILE"),
        env_file=os.path.join(run_dir, "cloudxr.env"),
        logs_dir=logs_dir,
    )


def _cmd_start(args: argparse.Namespace) -> int:
    """Start the service detached from this terminal."""
    from .. import background  # noqa: PLC0415
    from ..runtime import is_runtime_live  # noqa: PLC0415

    run_dir, logs_dir = _resolve_dirs(args)

    if is_runtime_live(run_dir):
        _fail(background.ALREADY_SERVING.format(run_dir=run_dir))

    _require_eula(args, run_dir)

    try:
        # AlreadyServingError -- lost the race between the check above and the
        # start lock -- carries that same refusal, so it needs no case here.
        pid, log = background.start_and_wait(_run_flags(args), run_dir, logs_dir)
    except RuntimeError as exc:
        _fail(str(exc))

    _print_summary_for(args, run_dir, logs_dir)
    print(
        f"CloudXR service: \033[32mdetached\033[0m (pid {pid}), "
        f"log file: \033[90m{log}\033[0m"
    )
    print("\033[33mStop it with: python -m isaaccapture.cloudxr.service stop\033[0m")
    return 0


def _cmd_stop(args: argparse.Namespace) -> int:
    """Stop the detached service."""
    from .. import background  # noqa: PLC0415

    run_dir, _ = _resolve_dirs(args)
    if background.read_pid(run_dir) is None:
        print("No detached CloudXR service is running.")
        return 0
    if background.terminate(run_dir):
        print("CloudXR service stopped.")
        return 0
    _fail(
        "The service did not stop within "
        f"{background.STOP_TIMEOUT_SEC:.0f}s.  It is not killed outright "
        "because that would orphan the runtime process holding the GPU; "
        "check the log and signal it yourself if it is wedged."
    )
    return 1


def _cmd_status(args: argparse.Namespace) -> int:
    """Report whether a runtime is serving this install dir."""
    from .. import background  # noqa: PLC0415
    from ..runtime import is_runtime_live  # noqa: PLC0415

    run_dir, logs_dir = _resolve_dirs(args)
    pid = background.read_pid(run_dir)
    status_path = Path(run_dir) / "oob_status.json"
    try:
        oob = json.loads(status_path.read_text(encoding="utf-8"))
        if not isinstance(oob, dict) or oob.get("schemaVersion") != 1:
            oob = None
    except (OSError, ValueError):
        oob = None

    if not is_runtime_live(run_dir):
        print(
            f"CloudXR runtime:   \033[31mnot running\033[0m  \033[90m({run_dir})\033[0m"
        )
        if oob and oob.get("health") == "fatal":
            print(
                f"CloudXR OOB:       fatal: {oob.get('reason', 'unknown')} ({status_path})"
            )
        return 1

    # Report the session that is actually running, not this command's defaults.
    # Its flags come from another process, so they may be from a build that
    # knows options this one does not; status must still say what it can.
    try:
        running = _build_parser().parse_args(
            ["run", *background.read_run_flags(run_dir)]
        )
    except SystemExit:
        running = _build_parser().parse_args(["run"])
    running.cloudxr_install_dir = args.cloudxr_install_dir
    _print_summary_for(running, run_dir, logs_dir)

    if pid is not None:
        print(
            f"CloudXR service: \033[32mdetached\033[0m (pid {pid}), "
            f"log file: \033[90m{background.log_path(logs_dir)}\033[0m"
        )
    else:
        print(
            "CloudXR service: \033[36mforeground\033[0m  \033[90m(started by "
            "`service run`, a container entrypoint, or run_embedded)\033[0m"
        )
    if oob:
        writer_pid = oob.get("writerPid")
        runtime_pid = oob.get("runtimePid")
        try:
            if not isinstance(writer_pid, int) or not isinstance(runtime_pid, int):
                raise ProcessLookupError
            os.kill(writer_pid, 0)
            os.kill(runtime_pid, 0)
        except (ProcessLookupError, PermissionError):
            print("CloudXR OOB:       status unavailable (stale writer)")
        else:
            print(
                f"CloudXR OOB:       {oob.get('health', 'unknown')}: {oob.get('reason', '')}"
            )
    return 0


def _cmd_logs(args: argparse.Namespace) -> int:
    """Show the detached service's log."""
    from .. import background  # noqa: PLC0415

    _, logs_dir = _resolve_dirs(args)
    log = background.log_path(logs_dir)
    if not log.is_file():
        print(f"No log yet at {log}", file=sys.stderr)
        return 1
    cmd = ["tail", "-n", str(args.lines)] + (["-f"] if args.follow else []) + [str(log)]
    return subprocess.call(cmd)


def _build_parser() -> argparse.ArgumentParser:
    """Build the ``isaaccapture.cloudxr.service`` argument parser."""
    parser = argparse.ArgumentParser(
        prog="python -m isaaccapture.cloudxr.service",
        description="Run the CloudXR service in the foreground, or detached.",
    )
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    run = sub.add_parser("run", help="run the service in the foreground")
    _add_run_arguments(run)
    run.set_defaults(func=_cmd_run)

    start = sub.add_parser(
        "start", help="start the service detached from this terminal"
    )
    _add_run_arguments(start)
    start.set_defaults(func=_cmd_start)

    stop = sub.add_parser("stop", help="stop the detached service")
    _add_install_dir_argument(stop)
    stop.set_defaults(func=_cmd_stop)

    status = sub.add_parser("status", help="report whether a runtime is running")
    _add_install_dir_argument(status)
    status.set_defaults(func=_cmd_status)

    logs = sub.add_parser("logs", help="show the detached service's log")
    _add_install_dir_argument(logs)
    logs.add_argument("-n", "--lines", type=int, default=50, help="lines to show")
    logs.add_argument("-f", "--follow", action="store_true", help="follow the log")
    logs.set_defaults(func=_cmd_logs)

    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.  Returns a process exit code."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "func", None) is None:
        parser.print_help()
        return 0
    from ..oob_teleop_adb import OobAdbError  # noqa: PLC0415

    try:
        return args.func(args)
    except OobAdbError as exc:
        print(f"\n{exc}\n", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
