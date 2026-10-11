# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Programmatic access to a running CloudXR runtime and WSS proxy.

:class:`~isaaccapture.cloudxr.service.CloudXRService` owns them; this is the
API embedding applications (e.g. Isaac Lab Teleop) use to reach one, and the
CLI plumbing the examples share.
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import select
import sys
import time
import warnings
from pathlib import Path
from urllib.parse import urlencode

try:  # POSIX only; without it the mismatch notice does not pause.
    import termios
    import tty
except ImportError:  # pragma: no cover - Windows
    termios = None
    tty = None

from . import background
from .env_config import (
    DEFAULT_DEVICE_PROFILE,
    ENV_FILE_NAME,
    EnvConfig,
    read_exported_env,
)
from .runtime import check_eula, is_runtime_live, latest_wss_log
from .service import CloudXRService

logger = logging.getLogger(__name__)

_STARTED_SERVICE = """\
\033[33mNo CloudXR service was running — started one (pid {pid}).\033[0m
  logs: \033[90m{log}\033[0m
  It outlives this script.  Stop it with:
    \033[1;32mpython -m isaaccapture.cloudxr.service stop\033[0m"""

_STARTED_HOST_CLIENT_PREFIX = "  web client: "

_ENV_CONFIG_IGNORED = """\
\033[33m{path} is ignored: the CloudXR runtime already serving this host was \
started with its own configuration.\033[0m
{rows}
  Restart the service to apply it:
    \033[1;32mpython -m isaaccapture.cloudxr.service stop\033[0m
    \033[1;32mpython -m isaaccapture.cloudxr.service start --cloudxr-env-config \
{path}\033[0m"""

_HOST_CLIENT_IGNORED = """\
\033[33m--{requested} is ignored: the CloudXR service already serving this \
host was started {running}.\033[0m
  Stop that service, then rerun this application with the same arguments:
    \033[1;32m{stop}\033[0m"""

# No colour: this one is raised, not printed, so it lands in logs and captured
# output as often as on a terminal.
_RUN_EMBEDDED_REFUSED = """\
A CloudXR runtime is already serving {run_dir}, so run_embedded cannot own one.

  To own the runtime here, stop the running service first:
    python -m isaaccapture.cloudxr.service stop

  To use the running one, construct the launcher without run_embedded.
  It then attaches, and setup_oob / usb_local / host_client belong to the
  service that started the runtime — its WSS proxy is not this process's to
  configure."""

_ENV_CONFIG_ROW = (
    "  {key}: \033[33m{requested}\033[0m requested, \033[36m{running}\033[0m in effect"
)

#: How long the mismatch notice holds the terminal before the caller continues.
ENV_CONFIG_PAUSE_SEC = 5.0

#: Redraw interval of the countdown, and so how fast the dots move.
_PAUSE_TICK_SEC = 0.25

# A live runtime cannot prove a headset stream from an old lifecycle snapshot.
_OOB_STATUS_MAX_AGE_SEC = 30.0
_OOB_METRICS_MAX_AGE_SEC = 5.0

_ENV_CONFIG_PAUSE = (
    "  \033[33mContinuing with the running configuration in {seconds}s{dots} — "
    "press any key to abort.\033[0m"
)


def _countdown(remaining: float, *, dots: int, redraw: bool = False) -> None:
    """Draw one frame of the pause countdown on stderr.

    Dots are padded to a fixed width so the text after them does not jitter
    from frame to frame.
    """
    line = _ENV_CONFIG_PAUSE.format(
        seconds=math.ceil(remaining), dots=("." * dots).ljust(3)
    )
    print(f"\r\033[K{line}" if redraw else line, end="", file=sys.stderr, flush=True)


class NoopContext:
    """Skip CloudXR start/attach; leave the process OpenXR environment alone.

    Duck-typed subset of :class:`CloudXRLauncher` for ``with`` blocks:
    ``owns_runtime``, ``wss_log_path``, ``stop``, ``health_check``.
    """

    @property
    def owns_runtime(self) -> bool:
        return False

    @property
    def wss_log_path(self) -> Path | None:
        return None

    def stop(self) -> None:
        pass

    def health_check(self) -> None:
        pass

    def __enter__(self) -> NoopContext:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        pass


class CloudXRLauncher:
    """Attaches to the CloudXR runtime and WSS proxy a service is running.

    Owns nothing by default: it attaches to the runtime
    :class:`~isaaccapture.cloudxr.service.CloudXRService` is serving, adopting
    its environment so OpenXR resolves to it, and leaves it running on exit.
    With no service running it starts a detached one — announced, because that
    outlives this process.  ``run_embedded`` is the only way to own a runtime,
    and it refuses to run beside one.

    Example::

        with CloudXRLauncher() as launcher:
            # attached to `service start`'s runtime; still running afterwards
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
        host_client: bool = True,
        run_embedded: bool = False,
        start_wss_proxy: bool | None = None,
    ) -> None:
        """Attach to the running runtime, or own one when *run_embedded*.

        Args:
            run_embedded: Run a :class:`CloudXRService` inside this process
                instead of starting a detached one.  This process then owns the
                runtime and stops it on exit.  Refused where a runtime is
                already serving *install_dir*, rather than quietly attaching
                to one this process cannot configure.
            start_wss_proxy: Deprecated no-op; the proxy always starts with
                the runtime.

        Every other argument is forwarded to :class:`CloudXRService` and only
        applies when this process owns it.  When attaching they describe a
        runtime that already exists.  Requested OOB/USB mode must match the
        detached service; other mismatches are reported rather than applied.
        ``host_client`` is recovered from the detached service's command line
        when that is available.

        Raises:
            RuntimeError: If the runtime fails to start or come up, if
                *run_embedded* is set while a runtime is already serving, or
                if the requested OOB/USB mode cannot be verified or matched.
        """
        if start_wss_proxy is not None:
            self._warn_start_wss_proxy_deprecated()

        self._run_dir = os.path.join(os.path.expanduser(install_dir), "run")
        self._logs_dir = Path(os.path.expanduser(install_dir)) / "logs"
        self._service: CloudXRService | None = None
        self._observed_oob_session: tuple[str, int, int] | None = None

        if run_embedded:
            self._refuse_beside_live_runtime()
            self._service = CloudXRService(
                install_dir=install_dir,
                env_config=env_config,
                device_profile=device_profile,
                accept_eula=accept_eula,
                setup_oob=setup_oob,
                usb_local=usb_local,
                host_client=host_client,
            )
            return

        if is_runtime_live(self._run_dir):
            self._validate_requested_mode(setup_oob, usb_local)
            self._attach(device_profile, env_config, host_client)
            self.oob_status()
            return

        started = self._start_service(
            install_dir,
            env_config,
            device_profile,
            accept_eula,
            setup_oob,
            usb_local,
            host_client,
        )
        # Skip mismatch reports for settings this call just applied.  A
        # runtime that beat us to it did not get them, so that one is reported
        # like any other attach.
        if not started:
            self._validate_requested_mode(setup_oob, usb_local)
        self._attach(
            device_profile,
            None if started else env_config,
            None if started else host_client,
        )
        self.oob_status()
        # After attach so wss_proxy_port() sees PROXY_PORT from cloudxr.env,
        # not a stale caller environment.
        if started and (host_client or usb_local):
            self._announce_hosted_client(usb_local=usb_local)

    def _refuse_beside_live_runtime(self) -> None:
        """Reject ``run_embedded`` where a runtime is already serving.

        Attaching would answer a request to own with a runtime this process
        cannot configure.  The WSS proxy options (``setup_oob``,
        ``usb_local``, ``host_client``) belong to the service that started
        the runtime; attaching can report a mismatch but cannot apply them.

        Raises:
            RuntimeError: If a runtime is already serving the run directory.
        """
        if not is_runtime_live(self._run_dir):
            return
        raise RuntimeError(_RUN_EMBEDDED_REFUSED.format(run_dir=self._run_dir))

    def _start_service(
        self,
        install_dir: str,
        env_config: str | Path | None,
        device_profile: str,
        accept_eula: bool,
        setup_oob: bool,
        usb_local: bool,
        host_client: bool,
    ) -> bool:
        """Start a detached service, then leave it running for the next caller.

        Announced rather than silent: it outlives this process, so a caller who
        did not ask for one still needs to know it is there and how to stop it.

        Returns whether this call is the one that started it.
        """
        # Accept here, where a terminal exists: the detached service inherits
        # /dev/null on stdin and could not prompt.
        check_eula(accept_eula=accept_eula or None, run_dir=self._run_dir)

        before_spawn = None
        if host_client or usb_local:
            from .oob_teleop_env import (  # noqa: PLC0415
                require_web_client_static_dir,
            )

            before_spawn = require_web_client_static_dir

        flags = [
            *(
                ["--cloudxr-install-dir", install_dir]
                if install_dir != "~/.cloudxr"
                else []
            ),
            *(["--cloudxr-env-config", str(env_config)] if env_config else []),
            *(["--setup-oob"] if setup_oob else []),
            *(["--usb-local"] if usb_local else []),
            *(["--host-client"] if host_client else []),
        ]
        # EnvConfig reads NV_DEVICE_PROFILE from the process environment, and
        # an env file still overrides it — so the profile needs no CLI flag.
        extra_env = (
            {"NV_DEVICE_PROFILE": device_profile}
            if device_profile != DEFAULT_DEVICE_PROFILE
            else None
        )
        try:
            pid, log = background.start_and_wait(
                flags,
                self._run_dir,
                self._logs_dir,
                extra_env,
                before_spawn=before_spawn,
            )
        except background.AlreadyServingError:
            # Another caller won the race and is serving; attaching to it is
            # what this path wanted.  Its configuration is not ours, though.
            return False
        print(_STARTED_SERVICE.format(pid=pid, log=log), file=sys.stderr)
        if setup_oob:
            print(
                "  Headset setup is retrying until stream confirmation; "
                "service startup does not confirm streaming.",
                file=sys.stderr,
            )
        return True

    def _announce_hosted_client(self, *, usb_local: bool = False) -> None:
        """Print the hosted ``/client/`` URL using the attached service env.

        Call only after :meth:`_attach` so :func:`wss_proxy_port` reads
        ``PROXY_PORT`` from the service's ``cloudxr.env``. USB-local clients
        reach the proxy through headset loopback via ``adb reverse``.
        """
        from .oob_teleop_env import (  # noqa: PLC0415
            guess_lan_ipv4,
            print_hosted_client_line,
            wss_proxy_port,
        )

        configured_host = os.environ.get("TELEOP_PROXY_HOST", "").strip()
        host = (
            "127.0.0.1"
            if usb_local
            else configured_host or guess_lan_ipv4() or "localhost"
        )
        port = wss_proxy_port()
        query = urlencode({"serverIP": host, "port": port})
        url = f"https://{host}:{port}/client/?{query}"
        print_hosted_client_line(
            url, prefix=_STARTED_HOST_CLIENT_PREFIX, file=sys.stderr
        )

    def _validate_requested_mode(self, setup_oob: bool, usb_local: bool) -> None:
        """Require a verifiable matching mode when attaching for OOB teleop."""
        if not setup_oob and not usb_local:
            return
        stop = self._service_stop_invocation()
        if background.read_pid(self._run_dir) is None:
            raise RuntimeError(
                "Cannot verify the existing CloudXR runtime's OOB/USB mode: "
                "its detached service flags are unavailable. "
                f"Stop it with `{stop}`, then start it with the requested mode."
            )
        flags = background.read_run_flags(self._run_dir)
        running_oob = "--setup-oob" in flags
        running_usb = "--usb-local" in flags
        if (setup_oob and not running_oob) or (running_usb != usb_local):
            requested = "USB-local OOB" if usb_local else "OOB without USB-local"
            running = (
                "USB-local OOB"
                if running_usb
                else "OOB without USB-local"
                if running_oob
                else "without OOB"
            )
            raise RuntimeError(
                f"Requested {requested} mode, but the existing CloudXR runtime "
                f"is running {running}. Stop it with `{stop}`, then start it "
                "with the requested mode."
            )

    def _attach(
        self,
        device_profile: str,
        env_config: str | Path | None,
        host_client: bool | None = None,
    ) -> None:
        """Adopt the running runtime's environment and report any mismatch.

        The env file is applied, never re-resolved: resolving it would rewrite
        the file out from under the service that owns it.  ``host_client`` is
        ``None`` when this call just started the service, so there is nothing
        to compare.
        """
        env = read_exported_env(self.env_file)
        if not env:
            raise RuntimeError(
                f"A CloudXR runtime is serving {self._run_dir}, but its "
                "environment file is missing or unreadable, so OpenXR cannot "
                "be pointed at it.  Restart the service."
            )
        os.environ.update(env)
        logger.info("Attached to the CloudXR runtime serving %s", self._run_dir)

        # These configure a runtime at start-up; attaching cannot apply them,
        # and a silently ignored device profile is the usual cause of a client
        # failing with XR_ERROR_FORM_FACTOR_UNAVAILABLE (-35).
        running = env.get("NV_DEVICE_PROFILE")
        if running and running != device_profile:
            logger.warning(
                "Attached to a runtime started with NV_DEVICE_PROFILE=%s; the "
                "requested %s is ignored.  Restart the service to change it.",
                running,
                device_profile,
            )
        if env_config is not None:
            self._warn_env_config_ignored(env_config, env)
        if host_client is not None:
            self._warn_host_client_mismatch(host_client)

    def _warn_host_client_mismatch(self, host_client: bool) -> None:
        """Report when the running service's hosted-client mode differs.

        Recovered from the detached service command line.  ``--host-client``
        and ``--usb-local`` both serve ``/client/`` on the WSS port, so either
        counts as hosted.  Foreground services leave no pid file, so their
        flags cannot be recovered here and the comparison stays quiet rather
        than guessing.
        """
        if background.read_pid(self._run_dir) is None:
            return
        flags = background.read_run_flags(self._run_dir)
        running = "--host-client" in flags or "--usb-local" in flags
        if running == host_client:
            return
        if running:
            running_desc = (
                "with --usb-local" if "--usb-local" in flags else "with --host-client"
            )
        else:
            running_desc = "without a hosted /client/"
        print(
            _HOST_CLIENT_IGNORED.format(
                requested="host-client" if host_client else "no-host-client",
                running=running_desc,
                stop=self._service_stop_invocation(),
            ).rstrip(),
            file=sys.stderr,
        )

    def _service_stop_invocation(self) -> str:
        """CLI that stops the service attached to this launcher's install dir."""
        cmd = "python -m isaaccapture.cloudxr.service stop"
        default_run = os.path.join(os.path.expanduser("~/.cloudxr"), "run")
        if os.path.abspath(self._run_dir) == os.path.abspath(default_run):
            return cmd
        return f"{cmd} --cloudxr-install-dir {os.path.dirname(self._run_dir)}"

    @staticmethod
    def _warn_env_config_ignored(
        env_config: str | Path, running: dict[str, str]
    ) -> None:
        """Report the settings a requested env config would have changed.

        Only keys whose requested value differs from the runtime's resolved
        environment are named: passing the same config on every run is how
        this gets scripted, and warning when nothing conflicts trains people
        to ignore the one case that matters.
        """
        try:
            requested = EnvConfig._load_env_file(env_config)
        except (OSError, RuntimeError) as exc:
            logger.warning("%s is ignored, and could not be read: %s", env_config, exc)
            return

        rows = [
            _ENV_CONFIG_ROW.format(
                key=key,
                requested=value,
                running=running.get(key) or "(not set)",
            )
            for key, value in requested.items()
            if running.get(key) != value
        ]
        if not rows:
            return
        print(
            _ENV_CONFIG_IGNORED.format(path=env_config, rows="\n".join(rows)),
            file=sys.stderr,
        )
        CloudXRLauncher._pause_for_abort()

    @staticmethod
    def _pause_for_abort(seconds: float = ENV_CONFIG_PAUSE_SEC) -> None:
        """Hold the terminal briefly so the notice above is not scrolled past.

        Continues on its own, and is skipped entirely when nobody could be
        watching: this runs from container entrypoints and CI too, where
        waiting for a keypress would hang the run instead of warning it.

        Raises:
            SystemExit: If a key is pressed before the deadline.
        """
        if termios is None or os.environ.get("CI") or not sys.stdin.isatty():
            return
        try:
            fd = sys.stdin.fileno()
            saved = termios.tcgetattr(fd)
        except (OSError, ValueError):
            # isatty() can be true for a stdin with no usable descriptor
            # (notebooks, embedded consoles).  A cosmetic pause is never
            # worth raising over.
            return

        # Redraw in place only on a terminal; into a redirected stderr the
        # carriage returns would just pile up as one unreadable line.
        animate = sys.stderr.isatty()
        pressed = False
        _countdown(seconds, dots=0)
        try:
            # cbreak so a bare keypress registers; a newline-terminated read
            # would make "press any key" a lie.
            tty.setcbreak(fd)
            deadline = time.monotonic() + seconds
            tick = 0
            while True:
                # Poll before testing the deadline, so a key already waiting
                # aborts even when there is no time left to wait.
                left = deadline - time.monotonic()
                wait = max(0.0, min(_PAUSE_TICK_SEC, left))
                if select.select([sys.stdin], [], [], wait)[0]:
                    sys.stdin.read(1)
                    pressed = True
                    break
                if left <= 0:
                    break
                tick += 1
                if animate:
                    _countdown(left, dots=1 + tick % 3, redraw=True)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, saved)

        print("\r\033[K" if animate else "", file=sys.stderr)

        if pressed:
            raise SystemExit(
                "Aborted: start the CloudXR service with this configuration, or "
                "rerun without --cloudxr-env-config to use the running one."
            )

    @property
    def run_dir(self) -> Path:
        """The run directory of the runtime this is attached to."""
        return Path(self._run_dir)

    @property
    def env_file(self) -> Path:
        """The env file describing the running runtime.

        Sourcing it points OpenXR at that runtime, which is how processes the
        launcher cannot reach — native binaries, other shells — attach to it.
        """
        return Path(self._run_dir) / ENV_FILE_NAME

    @property
    def owns_runtime(self) -> bool:
        """Whether this process started the runtime, and will stop it."""
        return self._service is not None

    # TODO(1.7): drop start_wss_proxy, --launch-wss-proxy and this helper.
    @staticmethod
    def _warn_start_wss_proxy_deprecated() -> None:
        """Announce that the ``start_wss_proxy`` no-op is on its way out."""
        message = (
            "start_wss_proxy is deprecated and does nothing; the WSS proxy "
            "always starts with the runtime.  It is removed in Isaac Teleop 1.7."
        )
        warnings.warn(message, DeprecationWarning, stacklevel=3)
        # Python drops DeprecationWarning raised outside __main__, which is
        # every --launch-wss-proxy run, so log it as well.
        logger.warning(message)

    # ------------------------------------------------------------------
    # CLI helpers for embedding applications and examples
    # ------------------------------------------------------------------

    @staticmethod
    def add_cloudxr_install_dir_argument(parser: argparse.ArgumentParser) -> None:
        """Register ``--cloudxr-install-dir`` on ``parser`` (default ``~/.cloudxr``)."""
        parser.add_argument(
            "--cloudxr-install-dir",
            type=str,
            default=os.path.expanduser("~/.cloudxr"),
            metavar="PATH",
            help="CloudXR install directory (default: ~/.cloudxr)",
        )

    @staticmethod
    def add_launch_cloudxr_runtime_argument(parser: argparse.ArgumentParser) -> None:
        """Register ``--launch-cloudxr-runtime`` on ``parser``.

        Uses :class:`argparse.BooleanOptionalAction`, so callers may pass
        ``--no-launch-cloudxr-runtime`` to use an already-configured OpenXR
        runtime (system or sourced env) without attaching to or starting CloudXR.
        """
        parser.add_argument(
            "--launch-cloudxr-runtime",
            action=argparse.BooleanOptionalAction,
            default=True,
            help=(
                "Attach to or start the CloudXR runtime before running "
                "(default: true). Pass --no-launch-cloudxr-runtime to use the "
                "OpenXR runtime already configured in this process "
                "(e.g. XR_RUNTIME_JSON for a system runtime)."
            ),
        )

    @staticmethod
    def add_cloudxr_device_profile_argument(parser: argparse.ArgumentParser) -> None:
        """Register ``--cloudxr-device-profile`` on ``parser`` (default Quest3)."""
        parser.add_argument(
            "--cloudxr-device-profile",
            type=str,
            default=DEFAULT_DEVICE_PROFILE,
            metavar="PROFILE",
            help=(
                "CloudXR NV_DEVICE_PROFILE for the runtime "
                f"(default: {DEFAULT_DEVICE_PROFILE}). "
                "Examples: Quest3, auto-webrtc, auto-native, AppleVisionPro. "
                "Overridden by --cloudxr-env-config or NV_DEVICE_PROFILE in the environment."
            ),
        )

    @staticmethod
    def add_cloudxr_env_config_argument(parser: argparse.ArgumentParser) -> None:
        """Register ``--cloudxr-env-config`` on ``parser`` (default: none).

        Points the launcher at a KEY=value env file of CloudXR runtime
        overrides (see the ``env_config`` argument of
        :meth:`CloudXRService.__init__`).
        """
        parser.add_argument(
            "--cloudxr-env-config",
            type=str,
            default=None,
            metavar="PATH",
            help=(
                "Path to a KEY=value env file of CloudXR runtime overrides "
                "(default: none). Reserved keys (XR_RUNTIME_JSON, "
                "NV_CXR_RUNTIME_DIR, ...) are always computed and ignored if set."
            ),
        )

    @staticmethod
    def add_accept_eula_argument(parser: argparse.ArgumentParser) -> None:
        """Register ``--accept-eula`` on ``parser`` (default: false).

        When omitted and no acceptance marker exists, the service prompts
        on stdin before starting the runtime.
        """
        parser.add_argument(
            "--accept-eula",
            action="store_true",
            help=(
                "Accept the NVIDIA CloudXR EULA non-interactively "
                "(e.g. for CI or containers)."
            ),
        )

    @staticmethod
    def add_host_client_argument(parser: argparse.ArgumentParser) -> None:
        """Register ``--host-client`` / ``--no-host-client`` on ``parser``."""
        parser.add_argument(
            "--host-client",
            action=argparse.BooleanOptionalAction,
            default=True,
            help=(
                "Serve the web client at /client/ on the WSS proxy port "
                "(default 48322) when this starts a CloudXR service. "
                "Attaching to an existing service leaves its hosting unchanged. "
                "Fetched once from the matching versioned release into "
                "TELEOP_WEB_CLIENT_STATIC_DIR or ~/.cloudxr/static-client."
            ),
        )

    @staticmethod
    def add_launch_wss_proxy_argument(parser: argparse.ArgumentParser) -> None:
        """Register the deprecated no-op ``--launch-wss-proxy`` on ``parser``.

        Defaults to ``None`` so an explicit flag is distinguishable from an
        absent one and only the former warns.
        """
        parser.add_argument(
            "--launch-wss-proxy",
            action=argparse.BooleanOptionalAction,
            default=None,
            help=(
                "Deprecated no-op, removed in 1.7: the WSS TLS proxy always "
                "starts with the runtime."
            ),
        )

    @staticmethod
    def add_launcher_arguments(parser: argparse.ArgumentParser) -> None:
        """Register CloudXR launcher CLI arguments on ``parser``."""
        CloudXRLauncher.add_cloudxr_install_dir_argument(parser)
        CloudXRLauncher.add_cloudxr_device_profile_argument(parser)
        CloudXRLauncher.add_cloudxr_env_config_argument(parser)
        CloudXRLauncher.add_accept_eula_argument(parser)
        CloudXRLauncher.add_host_client_argument(parser)
        CloudXRLauncher.add_launch_cloudxr_runtime_argument(parser)
        CloudXRLauncher.add_launch_wss_proxy_argument(parser)

    @staticmethod
    def _resolve_install_dir(
        args: argparse.Namespace,
        install_dir: str | None = None,
    ) -> str:
        """Return ``install_dir`` or ``args.cloudxr_install_dir`` when registered."""
        if install_dir is not None:
            return install_dir
        return getattr(args, "cloudxr_install_dir", "~/.cloudxr")

    @staticmethod
    def _resolve_device_profile(
        args: argparse.Namespace,
        device_profile: str | None = None,
    ) -> str:
        """Return ``device_profile`` or ``args.cloudxr_device_profile`` when registered."""
        if device_profile is not None:
            return device_profile
        return getattr(args, "cloudxr_device_profile", DEFAULT_DEVICE_PROFILE)

    @staticmethod
    def _resolve_env_config(
        args: argparse.Namespace,
        env_config: str | Path | None = None,
    ) -> str | Path | None:
        """Return ``env_config`` or ``args.cloudxr_env_config`` when registered."""
        if env_config is not None:
            return env_config
        return getattr(args, "cloudxr_env_config", None)

    @staticmethod
    def _resolve_accept_eula(
        args: argparse.Namespace,
        accept_eula: bool | None = None,
    ) -> bool:
        """Return ``accept_eula`` or ``args.accept_eula`` when registered.

        ``None`` means no override (fall back to ``args``); an explicit ``False``
        disables EULA acceptance even when ``args.accept_eula`` is true.
        """
        if accept_eula is not None:
            return accept_eula
        return bool(getattr(args, "accept_eula", False))

    @staticmethod
    def _resolve_host_client(
        args: argparse.Namespace,
        host_client: bool | None = None,
    ) -> bool:
        """Return ``host_client`` or ``args.host_client`` when registered."""
        if host_client is not None:
            return host_client
        return bool(getattr(args, "host_client", True))

    @staticmethod
    def launch_context(
        args: argparse.Namespace,
        *,
        install_dir: str | None = None,
        env_config: str | Path | None = None,
        device_profile: str | None = None,
        accept_eula: bool | None = None,
        setup_oob: bool = False,
        usb_local: bool = False,
        host_client: bool | None = None,
        run_embedded: bool = False,
        start_wss_proxy: bool | None = None,
    ) -> CloudXRLauncher | NoopContext:
        """Build a launcher context from parsed arguments.

        Returns :class:`NoopContext` when ``args.launch_cloudxr_runtime`` is
        false so callers can always ``with CloudXRLauncher.launch_context(args):``.

        ``install_dir``, ``env_config``, ``device_profile``, ``accept_eula``,
        and ``host_client`` default to the values registered by
        :meth:`add_launcher_arguments` (``args.cloudxr_install_dir`` etc.); pass an
        explicit keyword only to override what came in on the command line.
        ``run_embedded`` is forwarded to :class:`CloudXRLauncher`.
        ``start_wss_proxy`` is a deprecated no-op removed in 1.7.
        """
        if (
            start_wss_proxy is not None
            or getattr(args, "launch_wss_proxy", None) is not None
        ):
            CloudXRLauncher._warn_start_wss_proxy_deprecated()
        if not getattr(args, "launch_cloudxr_runtime", True):
            ignored: list[str] = []
            if run_embedded:
                ignored.append("run_embedded")
            if setup_oob:
                ignored.append("setup_oob")
            if usb_local:
                ignored.append("usb_local")
            if host_client:
                ignored.append("host_client")
            if ignored:
                logger.warning(
                    "--no-launch-cloudxr-runtime: ignoring CloudXR launcher "
                    "options %s (no runtime is started or attached).",
                    ", ".join(ignored),
                )
            return NoopContext()
        return CloudXRLauncher(
            install_dir=CloudXRLauncher._resolve_install_dir(args, install_dir),
            env_config=CloudXRLauncher._resolve_env_config(args, env_config),
            device_profile=CloudXRLauncher._resolve_device_profile(
                args, device_profile
            ),
            accept_eula=CloudXRLauncher._resolve_accept_eula(args, accept_eula),
            setup_oob=setup_oob,
            usb_local=usb_local,
            host_client=CloudXRLauncher._resolve_host_client(args, host_client),
            run_embedded=run_embedded,
        )

    # ------------------------------------------------------------------
    # Lifecycle — acts on the service only when this process owns it
    # ------------------------------------------------------------------

    def __enter__(self) -> CloudXRLauncher:
        """Return the launcher for use in a ``with`` block."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Stop the runtime on exit, if this process started it."""
        self.stop()

    def stop(self) -> None:
        """Stop the runtime and WSS proxy this process started.

        A no-op when attached: the service that owns the runtime outlives
        every script that uses it.
        """
        if self._service is not None:
            self._service.stop()

    def health_check(self) -> None:
        """Raise :class:`RuntimeError` if the runtime is no longer available.

        Raises:
            RuntimeError: If the owned runtime or WSS proxy has stopped, or
                the runtime this attached to is gone.
        """
        if self._service is not None:
            self._service.health_check()
            return
        if not is_runtime_live(self._run_dir):
            raise RuntimeError(
                f"The CloudXR runtime serving {self._run_dir} has stopped"
            )

    def wait_for_oob_stage(
        self, stage: str = "streamConfirmed", *, timeout_sec: float = 90.0
    ) -> dict:
        """Wait for live OOB evidence, or raise with the last recovery reason."""
        stages = {
            "socketBound",
            "transportReady",
            "clientLoaded",
            "connectDispatched",
            "streamConfirmed",
        }
        if stage not in stages:
            raise ValueError(f"Unknown OOB readiness stage: {stage}")
        if not math.isfinite(timeout_sec) or timeout_sec <= 0:
            raise ValueError("timeout_sec must be positive and finite")
        deadline = time.monotonic() + timeout_sec
        last_status: dict | None = None
        while True:
            last_status = self.oob_status()
            if last_status and last_status.get("health") == "fatal":
                raise RuntimeError(
                    f"OOB startup failed: {last_status.get('reason', 'unknown reason')}"
                )
            self.health_check()
            updated_at = (last_status or {}).get("updatedAt")
            age = (
                time.time() - updated_at
                if isinstance(updated_at, (int, float))
                and not isinstance(updated_at, bool)
                and math.isfinite(updated_at)
                else float("inf")
            )
            metrics_at = (last_status or {}).get("lastMetricsAt")
            metrics_age = (
                time.time() - metrics_at / 1000
                if isinstance(metrics_at, (int, float))
                and not isinstance(metrics_at, bool)
                and math.isfinite(metrics_at)
                else float("inf")
            )
            if (
                last_status
                and last_status.get(stage) is True
                and 0 <= age <= _OOB_STATUS_MAX_AGE_SEC
                and (
                    stage != "streamConfirmed"
                    or (
                        age < _OOB_METRICS_MAX_AGE_SEC
                        and 0 <= metrics_age < _OOB_METRICS_MAX_AGE_SEC
                    )
                )
            ):
                return last_status
            if time.monotonic() >= deadline:
                current = (last_status or {}).get("readinessStage", "unavailable")
                reason = (last_status or {}).get("reason", "no OOB status published")
                raise TimeoutError(
                    f"OOB startup did not reach {stage} within {timeout_sec:g}s "
                    f"(stage={current}, reason={reason})"
                )
            time.sleep(min(0.2, max(0, deadline - time.monotonic())))

    def oob_status(self) -> dict | None:
        """Return the OOB lifecycle status for owned or attached services."""
        if self._service is not None:
            return self._service.oob_status()
        import json  # noqa: PLC0415

        try:
            status = json.loads(
                (Path(self._run_dir) / "oob_status.json").read_text(encoding="utf-8")
            )
        except (OSError, ValueError):
            return None
        if not isinstance(status, dict) or status.get("schemaVersion") != 1:
            return None
        runtime_live = is_runtime_live(self._run_dir)
        writer_pid = status.get("writerPid")
        runtime_pid = status.get("runtimePid")
        if not isinstance(writer_pid, int) or not isinstance(runtime_pid, int):
            return None
        session_id = status.get("sessionId")
        identity = (session_id, writer_pid, runtime_pid)
        if not runtime_live:
            if status.get("health") == "fatal" and identity == getattr(
                self, "_observed_oob_session", None
            ):
                return status
            return None
        try:
            os.kill(writer_pid, 0)
            os.kill(runtime_pid, 0)
        except (OSError, ValueError):
            return None
        if isinstance(session_id, str) and session_id:
            self._observed_oob_session = identity
        return status

    @property
    def wss_log_path(self) -> Path | None:
        """Path to the WSS proxy log file, or ``None`` if there is none."""
        if self._service is not None:
            return self._service.wss_log_path
        found = latest_wss_log(self._logs_dir)
        return Path(found) if found else None
