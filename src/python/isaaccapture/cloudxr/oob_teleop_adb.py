# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ADB automation for OOB teleop (``--setup-oob``): open the headset bookmark URL via USB adb.

Default mode (WiFi streaming):
    The headset is connected via USB cable for adb commands only.  Streaming and
    web-page access use WiFi.  ``adb forward`` is used temporarily for CDP
    automation (DevTools socket).

USB-local mode (``--usb-local``):
    Teleop signalling and streaming travel over USB via ``adb reverse`` on the
    headset's loopback.  The headset URL uses ``serverIP=127.0.0.1`` and loads
    the web client from ``https://localhost:<PROXY_PORT>/client/`` on the WSS
    proxy (same ``/client/`` path as ``--host-client``).  coturn runs locally
    and is reachable from the headset through adb reverse for WebRTC ICE relay.
    Note: WebRTC requires a non-loopback interface on the headset, so WiFi must
    remain connected (no traffic traverses it).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shlex
import shutil
import socket
import stat
import subprocess
import sys
import time
import urllib.request
from contextvars import ContextVar
from collections.abc import Awaitable, Callable
from inspect import isawaitable
from dataclasses import dataclass
from enum import Enum
from .oob_teleop_env import (
    default_web_client_origin,
    parse_env_port,
    build_headset_bookmark_url,
    client_ui_fields_from_env,
    oob_progress,
    redact_control_token,
    resolve_lan_host_for_oob,
    web_client_base_override_from_env,
)

log = logging.getLogger("isaaccapture.cloudxr.oob_teleop_adb")


class OobAdbError(Exception):
    """``--setup-oob`` adb step failed; ``str(exception)`` is formatted for users (print without traceback)."""


SELECTED_ADB_SERIAL: ContextVar[str | None] = ContextVar(
    "selected_adb_serial", default=None
)


def _adb_run(args: list[str], **kwargs):
    """Apply the lifecycle's pinned serial to commands, including ``to_thread`` calls."""
    serial = SELECTED_ADB_SERIAL.get()
    if serial and args[0] == "adb" and args[1] != "devices" and "-s" not in args:
        args = ["adb", "-s", serial, *args[1:]]
    return subprocess.run(args, check=kwargs.pop("check", False), **kwargs)


@dataclass(frozen=True)
class AdbDevices:
    devices: tuple[tuple[str, str], ...]
    diagnostic: str = ""

    @property
    def ready(self) -> tuple[str, ...]:
        return tuple(serial for serial, state in self.devices if state == "device")


def enumerate_adb_devices() -> AdbDevices:
    """Enumerate every transport so selection can distinguish replacement."""
    try:
        proc = _adb_run(
            ["adb", "devices"], capture_output=True, text=True, timeout=5, check=False
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return AdbDevices((), str(exc))
    if proc.returncode:
        return AdbDevices((), _adb_output_text(proc))
    devices = []
    for line in (proc.stdout or "").splitlines()[1:]:
        parts = line.split()
        if len(parts) >= 2:
            devices.append((parts[0], parts[1]))
    return AdbDevices(tuple(devices))


def _adb_output_text(proc: subprocess.CompletedProcess[str]) -> str:
    return (proc.stderr or proc.stdout or "").strip()


def adb_automation_failure_hint(diagnostic: str) -> str:
    """Human-readable next steps for common ``adb`` failures."""
    d = diagnostic.lower()
    if "unauthorized" in d:
        return (
            "Device is unauthorized: unlock the headset, confirm the USB debugging (RSA) prompt, "
            "and run `adb devices` until the device shows `device` not `unauthorized`. "
            "If this persists, try `adb kill-server` and reconnect the cable."
        )
    if (
        "no devices/emulators" in d
        or "no devices found" in d
        or "device not found" in d
    ):
        return (
            "No adb device: plug in the USB cable, enable USB debugging on the headset, "
            "and check `adb devices`."
        )
    if "more than one device" in d:
        return "Multiple adb devices: unplug extras so only one headset shows in `adb devices`."
    if "offline" in d:
        return "Device offline: reconnect the USB cable and confirm USB debugging on the headset."
    return ""


def oob_adb_automation_message(rc: int, detail: str, hint: str) -> str:
    d = detail.strip() if detail else "(no output from adb)"
    lines = [
        f"OOB adb automation failed (adb exit code {rc}).",
        "",
        d,
    ]
    if hint.strip():
        lines.extend(["", hint])
    lines.extend(["", MANUAL_FALLBACK_HINT])
    return "\n".join(lines)


# Remediation appended to adb failures. Deliberately names no CLI flag: both
# the launcher (--setup-oob) and the standalone opener raise these, and manual
# navigation is the correct fallback for either.
MANUAL_FALLBACK_HINT = "Or open the teleop URL on the headset yourself."


def require_adb_on_path() -> None:
    """Raise :exc:`OobAdbError` if ``adb`` is missing."""
    if shutil.which("adb"):
        return
    raise OobAdbError(
        "`adb` was not found on PATH.\n\n"
        "Install Android Platform Tools and ensure `adb` is available. "
        f"{MANUAL_FALLBACK_HINT}"
    )


def _run_adb(label: str, args: list[str], *, timeout: float = 5.0) -> str | None:
    """Run *args* (an adb invocation); return stdout on rc==0, else log.warning + None.

    The label fronts every warning so a tail of the log identifies which
    helper failed without needing the raw command.
    """
    try:
        proc = _adb_run(
            args,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        log.warning("%s: %s", label, exc)
        return None
    if proc.returncode != 0:
        log.warning(
            "%s: rc=%d %s",
            label,
            proc.returncode,
            (proc.stderr or proc.stdout or "").strip(),
        )
        return None
    return proc.stdout or ""


class HeadsetNetworkState(Enum):
    """Result of a headset network probe over ADB."""

    ADB_UNAVAILABLE = "adb_unavailable"
    NO_NETWORK = "no_network"
    NETWORK_PRESENT = "network_present"


@dataclass(frozen=True)
class HeadsetNetworkProbe:
    state: HeadsetNetworkState
    interfaces: tuple[tuple[str, str], ...] = ()
    diagnostic: str = ""


def probe_headset_network(*, serial: str | None = None) -> HeadsetNetworkProbe:
    """Keep an ADB transport failure distinct from an empty IP listing."""
    command = [
        "adb",
        *(["-s", serial] if serial else []),
        "shell",
        "ip",
        "-o",
        "-4",
        "addr",
        "show",
    ]
    try:
        proc = _adb_run(command, capture_output=True, text=True, timeout=5, check=False)
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return HeadsetNetworkProbe(
            HeadsetNetworkState.ADB_UNAVAILABLE, diagnostic=str(exc)
        )
    if proc.returncode:
        return HeadsetNetworkProbe(
            HeadsetNetworkState.ADB_UNAVAILABLE,
            diagnostic=(
                proc.stderr or proc.stdout or f"adb exited {proc.returncode}"
            ).strip(),
        )
    interfaces = _parse_non_loopback_interfaces(proc.stdout or "")
    state = (
        HeadsetNetworkState.NETWORK_PRESENT
        if interfaces
        else HeadsetNetworkState.NO_NETWORK
    )
    return HeadsetNetworkProbe(state, tuple(interfaces))


def _parse_non_loopback_interfaces(text: str) -> list[tuple[str, str]]:
    """Parse non-loopback IPv4 addresses from ``ip -o -4 addr show``."""
    out: list[tuple[str, str]] = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 4 or parts[1] == "lo":
            continue
        try:
            idx = parts.index("inet")
        except ValueError:
            continue
        if idx + 1 < len(parts):
            out.append((parts[1], parts[idx + 1].split("/")[0]))
    return out


def headset_non_loopback_interfaces() -> list[tuple[str, str]]:
    """Return ``(iface, ipv4)`` for each non-loopback interface with an address.

    Uses ``adb shell ip -o -4 addr show`` on the connected headset.  Returns
    an empty list when the command fails (no device, adb broken, etc.) — the
    caller decides whether that's fatal.
    """
    return list(probe_headset_network().interfaces)


def require_headset_non_loopback_network() -> None:
    """Fail fast when the headset has no non-loopback IP (USB-local blocker).

    Chromium's WebRTC ``rtc::NetworkManager`` excludes loopback interfaces
    when enumerating networks for ICE.  Without at least one non-loopback
    interface with an IP, ICE gathering hangs forever (``iceGatheringState``
    stuck at ``gathering``, no candidates, no errors), and the teleop
    session fails with "No local connection candidates" (0xC0F2220F).

    The packets don't actually traverse the reported interface in USB-local
    mode — the kernel short-circuits loopback regardless of source — but
    the interface must *exist* for WebRTC's enumeration to be non-empty.
    """
    probe = probe_headset_network()
    if probe.state is HeadsetNetworkState.ADB_UNAVAILABLE:
        raise OobAdbError(
            f"ADB unavailable while checking headset network: {probe.diagnostic}"
        )
    ifaces = probe.interfaces
    if not ifaces:
        raise OobAdbError(
            "--usb-local requires Wi-Fi associated on the headset throughout the session.\n\n"
            "No teleop traffic actually flows over Wi-Fi — every byte goes over the USB\n"
            "cable via adb reverse. But Chromium's WebRTC excludes loopback interfaces\n"
            "during ICE candidate enumeration, so a non-loopback IP must exist on the\n"
            "headset or ICE hangs and the session errors out with\n"
            '"No local connection candidates" (0xC0F2220F).\n\n'
            "Fix: associate the headset with any Wi-Fi network and retry. Internet is\n"
            "NOT required — a phone hotspot with no SIM works, an open AP you never\n"
            "authenticate to works. The interface just needs to exist with an IP."
        )
    log.info(
        "USB-local: headset has %d non-loopback interface(s): %s",
        len(ifaces),
        ", ".join(f"{i}={ip}" for i, ip in ifaces),
    )


async def monitor_headset_wifi(*, poll_seconds: float = 5.0) -> None:
    """Log a one-shot warning when the headset loses its non-loopback interface.

    WebRTC needs at least one non-loopback interface on the headset side; if
    WiFi drops mid-session, ICE silently stops gathering candidates. Polling
    ``ip addr show`` over adb is cheap and lets us call out the cause without
    waiting for the user to puzzle out an opaque connection-stuck error.
    """
    # headset_non_loopback_interfaces() shells out to `adb`, which is sync;
    # run off-loop so the event loop isn't blocked for the duration of the
    # subprocess (up to a few hundred ms).
    previous = (await asyncio.to_thread(probe_headset_network)).state
    while True:
        try:
            await asyncio.sleep(poll_seconds)
        except asyncio.CancelledError:
            return
        current = (await asyncio.to_thread(probe_headset_network)).state
        if current is HeadsetNetworkState.ADB_UNAVAILABLE:
            continue
        if (
            previous is HeadsetNetworkState.NETWORK_PRESENT
            and current is HeadsetNetworkState.NO_NETWORK
        ):
            log.warning(
                "Headset Wi-Fi dropped — required even in USB-local mode. "
                "Chromium's WebRTC needs a non-loopback interface for ICE; "
                "reconnect any network (no internet needed)."
            )
        previous = current


def headset_wakefulness() -> str:
    """Return ``mWakefulness`` from ``adb shell dumpsys power``, or ``""`` on failure.

    Typical values: ``Awake`` | ``Asleep`` | ``Dreaming`` | ``Dozing``.
    """
    text = _run_adb("dumpsys power", ["adb", "shell", "dumpsys", "power"])
    if text is None:
        return ""
    m = re.search(r"mWakefulness=(\w+)", text)
    return m.group(1) if m else ""


def assert_headset_awake(*, timeout: float = 15.0, require_awake: bool = False) -> None:
    """Check wakefulness before OOB automation, sending one wake key if asleep.

    A sleeping Quest/PICO may accept ``am start`` yet sleep before CONNECT,
    preventing WebXR session entry.

    Args:
        timeout: Seconds to poll after ``KEYCODE_WAKEUP``; zero skips retry polling.
        require_awake: Raise ``OobAdbError`` if still not Awake; otherwise warn and return.
    """
    wake = headset_wakefulness()
    if wake == "Awake":
        return

    try:
        _adb_run(
            ["adb", "shell", "input", "keyevent", "KEYCODE_WAKEUP"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    log.warning(
        "Headset appears asleep (wakefulness=%s). Put it on or cover the "
        "proximity sensor; waiting up to %.0fs for it to wake.",
        wake or "?",
        timeout,
    )

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        time.sleep(1.0)
        wake = headset_wakefulness()
        if wake == "Awake":
            log.info("Headset is awake (wakefulness=%s)", wake)
            return

    log.warning(
        "Headset still appears asleep after %.0fs (wakefulness=%s); %s.",
        timeout,
        wake or "?",
        "aborting USB transfer" if require_awake else "continuing anyway",
    )
    if require_awake:
        raise OobAdbError("Headset wakefulness was not confirmed as Awake")


def adb_device_state() -> str:
    """Return ``adb get-state`` (lowercased), or ``""`` if adb is unreachable."""
    try:
        proc = _adb_run(
            ["adb", "get-state"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return ""
    out = proc.stdout if proc.returncode == 0 else (proc.stderr or proc.stdout or "")
    return out.strip().lower()


def assert_adb_device_online() -> None:
    """Raise :exc:`OobAdbError` if the headset isn't in ``device`` state right now.

    Run before adb operations to convert a stale-preflight failure (USB
    jiggle, revoked debugging, headset reboot) into an actionable error.
    Auto-retries once via ``adb reconnect`` for transient ``offline`` —
    the most common cause is a brief USB renumeration that the daemon
    sorts out on its own when nudged.
    """
    state = adb_device_state()
    if state == "device":
        return
    # Single auto-recovery attempt for transient `offline` (USB renumeration).
    # `adb reconnect` is fast (~ms) and preserves existing reverse rules.
    if "offline" in state:
        log.warning("adb device offline — attempting `adb reconnect`")
        try:
            _adb_run(
                ["adb", "reconnect"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        time.sleep(0.5)
        state = adb_device_state()
        if state == "device":
            log.info("adb device recovered after reconnect")
            return
    if not state:
        raise OobAdbError(
            "adb is not responding. Try `adb kill-server`, then reconnect the USB cable."
        )
    if "unauthorized" in state:
        raise OobAdbError(
            "Headset adb is unauthorized. Unlock the headset and accept the "
            "USB-debugging RSA prompt; verify with `adb devices`."
        )
    if "offline" in state:
        raise OobAdbError(
            "Headset is `offline` to adb (reconnect attempted). "
            "Reconnect the USB cable; if that doesn't help, "
            "`adb kill-server && adb start-server`."
        )
    raise OobAdbError(
        f"adb state `{state}`, expected `device`. Reconnect the USB cable."
    )


def assert_exactly_one_adb_device() -> str:
    """Return the selected ready serial for explicit ``-s`` commands.

    Resolution order:

    1. ``ANDROID_SERIAL`` (the standard adb env var) names the serial to
       use. We confirm it is currently in ``device`` state. Callers carry
       the returned serial into every device command via ``-s``.
    2. No env var: exactly one device must be in ``device`` state. More
       than one is fatal — the operator must either unplug the extras or
       set ``ANDROID_SERIAL=<serial>`` to disambiguate.
    """
    try:
        proc = _adb_run(
            ["adb", "devices"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except FileNotFoundError as e:
        raise OobAdbError(
            "`adb` was not found on PATH.\n\n"
            "Install Android Platform Tools and ensure `adb` is available. "
            f"{MANUAL_FALLBACK_HINT}"
        ) from e
    except subprocess.TimeoutExpired as e:
        raise OobAdbError(
            "adb command timed out; ensure Android Platform Tools are installed and adb is callable.\n\n"
            f"Try `adb kill-server` and reconnect the USB cable. {MANUAL_FALLBACK_HINT}"
        ) from e
    if proc.returncode != 0:
        diag = _adb_output_text(proc)
        raise OobAdbError(
            f"adb devices failed (exit code {proc.returncode}).\n\n"
            f"{diag}\n\n"
            "Check your adb installation and USB connection."
        )
    text = (proc.stdout or "") + "\n" + (proc.stderr or "")
    ready: list[str] = []
    for line in text.strip().splitlines()[1:]:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) >= 2 and parts[-1] == "device":
            ready.append(parts[0])
    if len(ready) == 0:
        raise OobAdbError(
            "No adb device found.\n\n"
            "Plug in the USB cable, enable USB debugging on the headset, and check `adb devices`. "
            f"{MANUAL_FALLBACK_HINT}"
        )

    # If the operator pinned a specific device, validate it is ready and stop.
    requested = os.environ.get("ANDROID_SERIAL", "").strip()
    if requested:
        if requested not in ready:
            listed = ", ".join(ready) if ready else "(none ready)"
            raise OobAdbError(
                f"ANDROID_SERIAL={requested!r} is not currently in `device` state.\n\n"
                f"Devices ready right now: {listed}\n\n"
                f"Either unset ANDROID_SERIAL (to use the single-device default), "
                f"or set it to one of the serials above."
            )
        log.info("adb device pinned via ANDROID_SERIAL=%s", requested)
        return requested

    if len(ready) > 1:
        listed = ", ".join(ready)
        raise OobAdbError(
            "Too many adb devices.\n\n"
            f"Currently connected: {listed}\n\n"
            "Unplug extras so only one headset is connected, OR set "
            "ANDROID_SERIAL=<serial> to pin the one you want, then retry. "
            f"({MANUAL_FALLBACK_HINT})"
        )
    return ready[0]


def build_teleop_url(
    *,
    resolved_port: int,
    usb_local: bool = False,
    host_client: bool = False,
    web_client_base: str | None = None,
    oob_enable: bool = True,
) -> str:
    """Build the headset teleop bookmark URL for ``am start`` and CDP automation.

    *web_client_base* overrides the WebXR client origin and takes precedence
    over :envvar:`TELEOP_WEB_CLIENT_BASE` (an explicit argument outranks the
    environment); the streaming query params are computed the same way either
    way.

    *oob_enable* forwards to :func:`~.oob_teleop_env.build_headset_bookmark_url`;
    pass ``False`` only when the OOB control hub is not running.
    """
    env_port = os.environ.get("TELEOP_STREAM_PORT", "").strip()
    signaling_port = (
        parse_env_port("TELEOP_STREAM_PORT", env_port) if env_port else resolved_port
    )

    if usb_local:
        from .oob_teleop_env import (  # noqa: PLC0415
            USB_HOST,
            USB_TURN_USER,
            USB_TURN_CREDENTIAL,
            usb_turn_port,
        )

        stream_cfg: dict = {
            "serverIP": USB_HOST,
            "port": signaling_port,
            # No mediaAddress: it's a NAT-override that bypasses ICE and would
            # short-circuit the TURN-relayed media path. Let the SDK discover
            # the media endpoint through ICE via coturn.
            "turnServer": f"turn:{USB_HOST}:{usb_turn_port()}?transport=tcp",
            "turnUsername": USB_TURN_USER,
            "turnCredential": USB_TURN_CREDENTIAL,
            "iceRelayOnly": True,
            **client_ui_fields_from_env(),
        }
        ovr = web_client_base or web_client_base_override_from_env()
        web_base = ovr if ovr else f"https://localhost:{resolved_port}/client"
    else:
        lan_host = resolve_lan_host_for_oob()
        stream_cfg = {
            "serverIP": lan_host,
            "port": signaling_port,
            **client_ui_fields_from_env(),
        }
        ovr = web_client_base or web_client_base_override_from_env()
        if host_client:
            default_base = f"https://{lan_host}:{resolved_port}/client"
        else:
            default_base = default_web_client_origin()
        web_base = ovr if ovr else default_base

    token = os.environ.get("CONTROL_TOKEN") or None
    return build_headset_bookmark_url(
        web_client_base=web_base,
        stream_config=stream_cfg,
        control_token=token,
        oob_enable=oob_enable,
    )


def _adb_getprop(prop: str) -> str:
    """Read an Android system property via adb. Returns "" on failure."""
    text = _run_adb(f"getprop {prop}", ["adb", "shell", "getprop", prop])
    return text.strip() if text is not None else ""


def _adb_pkg_installed(package: str) -> bool:
    """Return ``True`` iff *package* is installed on the connected headset.

    Uses ``pm list packages <pkg>`` (a prefix filter) and matches the exact
    ``package:<pkg>`` line so a query for ``com.pico.browser`` doesn't
    accidentally report success when only ``com.pico.browser.overseas`` is
    present (or vice versa).
    """
    if not package:
        return False
    text = _run_adb(
        f"pm list packages {package}",
        ["adb", "shell", "pm", "list", "packages", package],
    )
    if text is None:
        return False
    target = f"package:{package}"
    return any(line.strip() == target for line in text.splitlines())


def _first_installed_pkg(candidates: tuple[str, ...]) -> str | None:
    """Return the first package from *candidates* that's installed, or ``None``."""
    for pkg in candidates:
        if _adb_pkg_installed(pkg):
            return pkg
    return None


def headset_browser_package() -> str | None:
    """Return the Android package of the full-fat WebXR browser on this headset.

    WebLayer (the default VIEW-intent handler on Meta Quest + PICO) ships a
    minimal Chromium that accepts ``navigator.xr.requestSession`` but does
    not fully plumb controller input sources through to ``@react-three/xr``.
    Forcing the real vendor browser fixes controller rays and clicks.

    Resolution order:

    1. ``TELEOP_HEADSET_BROWSER_PACKAGE`` env var (explicit override; not
       validated against ``pm list packages`` — caller is trusted).
    2. Vendor map based on ``ro.product.manufacturer`` / ``ro.product.brand``,
       then probed against ``pm list packages`` so we never return a package
       that isn't actually installed:

       * Meta / Oculus → ``com.oculus.browser``.
       * PICO         → ``com.pico.browser.overseas`` (global firmware) if
         present, else ``com.pico.browser`` (domestic / China firmware).
    3. ``None`` (fall back to the generic VIEW intent).

    On current PICO firmware both browser variants are thin shells over the
    system WebLayer (same ``@weblayer_devtools_remote`` socket), so forcing
    the package does not actually escape WebLayer today.  We still target
    them for correctness and forward-compatibility with any future PICO
    build that ships an independent Chromium.
    """
    override = os.environ.get("TELEOP_HEADSET_BROWSER_PACKAGE", "").strip()
    if override:
        return override
    vendor = (
        _adb_getprop("ro.product.manufacturer") + " " + _adb_getprop("ro.product.brand")
    ).lower()
    if "meta" in vendor or "oculus" in vendor:
        # Full-fat Chromium, distinct from WebLayer — forcing it fixes WebXR
        # controller plumbing that WebLayer handles incompletely.
        return "com.oculus.browser"
    if "pico" in vendor:
        # Prefer the overseas/global package when both are installed; the
        # global firmware tends to ship the more capable Chromium build.
        # Probe via `pm list packages` so we don't try to `am start` into a
        # missing package on the wrong-region firmware.
        return _first_installed_pkg(("com.pico.browser.overseas", "com.pico.browser"))
    return None


def open_url_on_headset(url: str) -> tuple[int, str]:
    """Open *url* on the connected headset via ``am start``.

    When a known vendor browser is detected (Meta / PICO), launches into it
    explicitly via ``-p <package>`` so the URL opens in the full Chromium
    (with working WebXR controller input), not Android WebLayer.  Falls
    back to the generic VIEW intent on unknown vendors.

    Takes the URL as an argument rather than deriving it, so callers that
    already have one (``python -m isaaccapture.cloudxr.webclient``) share the
    vendor-browser selection and token redaction with the OOB launcher path.

    Returns ``(exit_code, diagnostic)``.
    """
    try:
        assert_adb_device_online()
    except OobAdbError as exc:
        return 99, str(exc)

    package = headset_browser_package()
    if package:
        log.info("ADB automation: launching into %s (bypass WebLayer)", package)
        shell_cmd = (
            f"am start -p {shlex.quote(package)} "
            f"-a android.intent.action.VIEW -d {shlex.quote(url)}"
        )
    else:
        shell_cmd = "am start -a android.intent.action.VIEW -d " + shlex.quote(url)
    full = ["adb", "shell", shell_cmd]
    log.info(
        "ADB automation: %s",
        redact_control_token(" ".join(shlex.quote(c) for c in full)),
    )
    try:
        proc = _adb_run(full, capture_output=True, text=True, timeout=30, check=False)
    except subprocess.TimeoutExpired as e:
        partial = (
            (e.stderr or e.stdout or b"")
            if isinstance(e.stderr or e.stdout, bytes)
            else (e.stderr or e.stdout or "")
        )
        if isinstance(partial, bytes):
            partial = partial.decode(errors="replace")
        diag = f"adb shell timed out after 30s. {partial}".strip()
        return 1, diag
    if proc.returncode != 0:
        diag = _adb_output_text(proc)
        return proc.returncode, diag
    log.info("ADB automation: am start completed")
    return 0, ""


def run_adb_headset_bookmark(
    *, resolved_port: int, usb_local: bool = False, host_client: bool = False
) -> tuple[int, str]:
    """Open the computed teleop bookmark URL on the headset (browser not yet running).

    Thin wrapper: :func:`build_teleop_url` for the URL,
    :func:`open_url_on_headset` for the ``am start``.

    Returns ``(exit_code, diagnostic)``.
    """
    return open_url_on_headset(
        build_teleop_url(
            resolved_port=resolved_port, usb_local=usb_local, host_client=host_client
        )
    )


# ---------------------------------------------------------------------------
# USB-local mode: adb reverse port-forwarding + coturn TURN relay
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AdbReverseProbe:
    adb_available: bool
    missing_ports: tuple[int, ...]


def probe_adb_reverse_rules(expected_ports: list[int]) -> AdbReverseProbe:
    """Distinguish ADB failure from an online device with missing rules."""
    output = _run_adb("adb reverse --list", ["adb", "reverse", "--list"])
    if output is None:
        return AdbReverseProbe(False, tuple(expected_ports))
    listed: set[tuple[str, str]] = set()
    for line in output.splitlines():
        parts = line.split()
        if len(parts) >= 3:
            listed.add((parts[1], parts[2]))
    return AdbReverseProbe(
        True,
        tuple(
            port
            for port in expected_ports
            if (f"tcp:{port}", f"tcp:{port}") not in listed
        ),
    )


def verify_adb_reverse_rules(expected_ports: list[int]) -> list[int]:
    """Return ports from *expected_ports* that are not in ``adb reverse --list``.

    rc=0 from ``adb reverse`` doesn't guarantee the rule survived — a competing
    adbd or transient ``offline`` can evict it moments later. If adb is itself
    unreachable, treat all expected as missing so the warning fires.
    """
    return list(probe_adb_reverse_rules(expected_ports).missing_ports)


def setup_adb_reverse_ports(proxy_port: int | None = None) -> None:
    """Set up ``adb reverse`` for the USB-local TCP ports.

    Reverse-maps headset loopback ports to the PC so the headset can reach
    the WSS proxy (including ``/client/``) and CloudXR backend over USB.

    Ports reversed: *proxy_port* (or :func:`~.oob_teleop_env.wss_proxy_port`
    when omitted) and the CloudXR backend port (resolved via
    :func:`~.oob_teleop_env.usb_backend_port`, default 49100; override via
    the ``USB_BACKEND_PORT`` env var).

    Raises:
        OobAdbError: device offline / unauthorized, or an ``adb reverse`` call failed.
    """
    from .oob_teleop_env import usb_backend_port, wss_proxy_port  # noqa: PLC0415

    assert_adb_device_online()
    resolved_proxy_port = wss_proxy_port() if proxy_port is None else proxy_port
    ports = [resolved_proxy_port, usb_backend_port()]
    for port in ports:
        try:
            _adb_run(
                ["adb", "reverse", f"tcp:{port}", f"tcp:{port}"],
                capture_output=True,
                text=True,
                timeout=10,
                check=True,
            )
        except subprocess.CalledProcessError as exc:
            detail = (exc.stderr or exc.stdout or "").strip() or "(no adb output)"
            raise OobAdbError(
                f"adb reverse tcp:{port} failed: {detail}. "
                "Reconnect the USB cable and verify `adb devices`."
            ) from exc
        log.info("adb reverse tcp:%d -> tcp:%d (PC)", port, port)


def teardown_adb_reverse_ports(proxy_port: int | None = None) -> None:
    """Remove the ``adb reverse`` rules set by :func:`setup_adb_reverse_ports`."""
    from .oob_teleop_env import usb_backend_port, wss_proxy_port  # noqa: PLC0415

    resolved_proxy_port = wss_proxy_port() if proxy_port is None else proxy_port
    ports = [resolved_proxy_port, usb_backend_port()]
    for port in ports:
        _adb_run(
            ["adb", "reverse", "--remove", f"tcp:{port}"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        log.info("adb reverse removed tcp:%d", port)


def setup_adb_reverse_turn(turn_port: int) -> None:
    """Set up ``adb reverse`` for the TURN server port.

    Maps ``headset tcp:turn_port`` → ``PC tcp:turn_port`` so the headset
    browser can reach the coturn TURN server at ``127.0.0.1:turn_port``
    without WiFi.

    Raises:
        OobAdbError: device offline / unauthorized, or the ``adb reverse`` call failed.
    """
    assert_adb_device_online()
    try:
        _adb_run(
            ["adb", "reverse", f"tcp:{turn_port}", f"tcp:{turn_port}"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "").strip() or "(no adb output)"
        raise OobAdbError(
            f"adb reverse tcp:{turn_port} (TURN) failed: {detail}. "
            "Reconnect the USB cable and verify `adb devices`."
        ) from exc
    log.info("adb reverse tcp:%d (TURN) -> tcp:%d (PC coturn)", turn_port, turn_port)


def teardown_adb_reverse_turn(turn_port: int) -> None:
    """Remove the TURN ``adb reverse`` rule."""
    _adb_run(
        ["adb", "reverse", "--remove", f"tcp:{turn_port}"],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    log.info("adb reverse removed tcp:%d (TURN)", turn_port)


def coturn_binary_path() -> str | None:
    """Return the path to the coturn TURN-server binary, or ``None`` if not found."""
    # Debian's `coturn` package installs `turnserver`, but some containers
    # and source builds expose `coturn` — probe both.
    for name in ("turnserver", "coturn"):
        path = shutil.which(name) or (
            f"/usr/bin/{name}" if os.path.exists(f"/usr/bin/{name}") else None
        )
        if path:
            return path
    return None


def require_coturn_available() -> None:
    """Fail fast if coturn is not installed.

    Raises :class:`OobAdbError` with install instructions when neither
    ``turnserver`` nor ``coturn`` is on PATH and not at the default
    Debian/Ubuntu location.  Call this early (before starting the
    launcher) in ``--usb-local`` mode so the user gets a clear error
    instead of a silent WebRTC-gather-timeout later.
    """
    if coturn_binary_path() is not None:
        return
    raise OobAdbError(
        "--usb-local requires coturn (TURN server) but neither `turnserver` "
        "nor `coturn` was found on PATH or in /usr/bin.\n\n"
        "Install it with:\n"
        "    sudo apt-get install -y coturn\n\n"
        "coturn runs locally (no systemd service needed — the launcher starts its "
        "own instance on port 3478 and shuts it down on exit)."
    )


def require_turn_port_free(port: int) -> None:
    """Fail fast if anything else is already listening on TCP/UDP *port*.

    Why: a system-wide ``coturn`` (or any other service) bound to
    ``0.0.0.0:port`` can coexist with our ``127.0.0.1:port`` bind via
    ``SO_REUSEADDR``, so the standard bind-conflict probe in
    :func:`~.oob_teleop_env.print_host_preflight_warnings` misses it.
    But the system listener has different credentials and possibly an
    overlapping relay-port range, so headset traffic over adb-reverse
    can still end up at the wrong daemon (or our daemon's relay
    allocations collide with theirs). Cleanest is to refuse to start.
    Skipped silently when ``ss`` is unavailable.
    """
    from .oob_teleop_env import ss_listeners_on_port  # noqa: PLC0415

    listeners = ss_listeners_on_port(port)
    if not listeners:
        return
    formatted = "\n  ".join(listeners)
    raise OobAdbError(
        f"--usb-local: TCP/UDP port {port} is already in use:\n  {formatted}\n\n"
        "A pre-existing TURN server (or any service) on this port can coexist "
        "with our 127.0.0.1 bind via SO_REUSEADDR but will collide on the "
        "relay-port range and credentials, so coturn won't be the authoritative "
        "listener for the adb-reverse path.\n\n"
        "Stop the conflicting service (e.g. `sudo systemctl stop coturn`) "
        "or set USB_TURN_PORT to a different unused port and retry."
    )


def _open_private(path: str) -> int:
    """Create *path* fresh and owner-only, reusing nothing already there.

    These are predictable ``/tmp`` names, so another local user can get there
    first. ``O_NOFOLLOW`` alone only refuses a symlink; a plain file they created
    would be written through, which for the config file means handing them the
    TURN credential it carries. Unlinking first is what makes the ``O_EXCL``
    create meaningful, and under ``/tmp``'s sticky bit that unlink fails on a
    file owned by someone else, so a planted path is refused rather than reused.

    Raises:
        OSError: if the path cannot be cleared or created safely.
    """
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass
    return os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)


def start_coturn(turn_port: int, user: str, credential: str) -> subprocess.Popen | None:
    """Start a coturn TURN server for USB-local ICE relay.

    coturn listens on ``127.0.0.1:turn_port`` (TCP + UDP).  ``adb reverse``
    exposes this port to the headset so WebRTC can obtain TURN relay
    candidates.  ``--allow-loopback-peers`` lets coturn relay between the
    headset (via adb reverse) and the CloudXR backend (UDP on PC loopback).

    Args:
        turn_port: TCP/UDP port for coturn (resolved via
            :func:`~.oob_teleop_env.usb_turn_port`, default 3478; override via
            the ``USB_TURN_PORT`` env var).
        user: TURN username.
        credential: TURN credential (password).

    Returns:
        :class:`subprocess.Popen` handle, or ``None`` if coturn failed to
        start.  Callers should treat ``None`` as non-fatal (TURN-less
        streaming may still work on LAN) but warn the operator prominently.
    """
    coturn_bin = coturn_binary_path()
    if coturn_bin is None:
        log.warning(
            "coturn: neither `turnserver` nor `coturn` on PATH — "
            "install with `sudo apt-get install coturn`"
        )
        return None

    # Write a config file — easier to maintain than a long arg list and avoids
    # shell quoting issues with special characters in credentials.
    conf_path = f"/tmp/turnserver-cloudxr-{turn_port}.conf"
    log_path = f"/tmp/coturn-cloudxr-{turn_port}.log"
    conf_content = f"""\
listening-port={turn_port}
listening-ip=127.0.0.1
external-ip=127.0.0.1
min-port=49152
max-port=49200
lt-cred-mech
fingerprint
user={user}:{credential}
realm=cloudxr
allow-loopback-peers
cli-password=cloudxr-internal
no-tls
no-dtls
no-stdout-log
log-file={log_path}
simple-log
"""
    # 0600 and O_EXCL/O_NOFOLLOW: this file carries the TURN credential and its
    # name is predictable under a world-writable /tmp.
    try:
        with os.fdopen(_open_private(conf_path), "w", encoding="utf-8") as f:
            f.write(conf_content)
    except OSError as exc:
        log.warning("coturn: failed to write config file %s: %s", conf_path, exc)
        return None

    # Truncate the log so operators only see lines from this run. Not fatal: the
    # relay is still worth starting without it, and coturn opens the path itself
    # from `log-file=` above.
    try:
        os.close(_open_private(log_path))
    except OSError as exc:
        log.warning("coturn: failed to prepare log file %s: %s", log_path, exc)

    try:
        proc = subprocess.Popen(
            [coturn_bin, "-c", conf_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except OSError as exc:
        log.warning("coturn failed to start (%s): %s", coturn_bin, exc)
        return None

    # Give coturn a moment to start (or exit with a config error)
    time.sleep(0.5)
    if proc.poll() is not None:
        log.warning(
            "coturn exited immediately (exit code %d). Tail of %s:\n%s",
            proc.returncode,
            log_path,
            _tail_file(log_path, 10),
        )
        return None

    log.info(
        "coturn TURN server started (pid=%d) at 127.0.0.1:%d (log: %s)",
        proc.pid,
        turn_port,
        log_path,
    )
    return proc


def verify_coturn_listening(turn_port: int, *, timeout: float = 1.0) -> bool:
    """Confirm coturn is accepting TCP on ``127.0.0.1:turn_port``.

    ``start_coturn`` checks the process is alive 0.5 s after fork — that
    is not the same as "listener bound". A short TCP connect closes the gap.
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(timeout)
            s.connect(("127.0.0.1", turn_port))
        return True
    except OSError:
        return False


#: Bytes read back from the end of a coturn log to find its last few lines.
#: Generous for the ten to twenty lines the callers ask for, and a fixed bound
#: on what a long-running relay's log can make this read.
_TAIL_WINDOW = 16 * 1024


def _tail_file(path: str, lines: int) -> str:
    """Return the last *lines* lines of *path* (empty string on read failure).

    Seeks instead of reading the whole file: coturn writes this log for as long
    as the session runs and nothing truncates it mid-run, so ``readlines()``
    pulled an unbounded file into memory to print twenty lines of it. The
    regular-file check is not decoration -- a FIFO left at this predictable
    ``/tmp`` name would make the read block.
    """
    try:
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        with os.fdopen(os.open(path, flags), "rb") as f:
            if not stat.S_ISREG(os.fstat(f.fileno()).st_mode):
                return ""
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - _TAIL_WINDOW))
            chunk = f.read(_TAIL_WINDOW)
    except OSError:
        return ""
    text = chunk.decode("utf-8", errors="replace")
    return "\n".join(text.splitlines()[-lines:]).rstrip()


async def watch_coturn(
    proc_box: list[subprocess.Popen | None],
    *,
    turn_port: int,
    user: str,
    credential: str,
    poll_seconds: float = 2.0,
) -> None:
    """Restart coturn once if it dies mid-session; surface its log on the way out.

    Stores the (possibly replaced) handle back in ``proc_box[0]`` so the
    caller's cleanup teardown stops the live process. After one restart a
    second death is left for ``stop_coturn`` to log — chasing further
    restarts usually masks a config / port-binding bug we want visible.
    """
    restarted = False
    while True:
        try:
            await asyncio.sleep(poll_seconds)
        except asyncio.CancelledError:
            return
        proc = proc_box[0]
        if proc is None or proc.poll() is None:
            continue
        rc = proc.returncode
        log_path = f"/tmp/coturn-cloudxr-{turn_port}.log"
        log.warning(
            "coturn died (rc=%d). Tail of %s:\n%s",
            rc,
            log_path,
            _tail_file(log_path, 20),
        )
        if restarted:
            log.error("coturn died again — leaving down. Inspect %s", log_path)
            return
        restarted = True
        new_proc = start_coturn(turn_port, user, credential)
        proc_box[0] = new_proc
        if new_proc is None:
            log.error(
                "coturn died and could not be restarted — WebRTC has no relay candidates"
            )
            return
        log.info("coturn restarted (pid=%d)", new_proc.pid)


def stop_coturn(proc: subprocess.Popen | None) -> None:
    """Terminate the coturn process started by :func:`start_coturn`."""
    if proc is None:
        return
    try:
        proc.terminate()
        proc.wait(timeout=5)
    except (subprocess.TimeoutExpired, OSError):
        try:
            proc.kill()
        except OSError:
            pass
    log.info("coturn TURN server stopped")


# ---------------------------------------------------------------------------
# CDP automation — click the CONNECT button via Chrome DevTools Protocol
# ---------------------------------------------------------------------------

_CDP_LOCAL_PORT = 9223  # avoid clashing with any pre-existing 9222 forward


_DEVTOOLS_SOCKET_RE = re.compile(r"@([A-Za-z0-9._+-]*_devtools_remote(?:_\d+)?)")


def _discover_devtools_socket() -> str | None:
    """Return the bare name of the browser's DevTools abstract socket, or None.

    Chromium-based Android browsers expose a Unix abstract socket matching
    ``@<prefix>_devtools_remote[_<pid>]`` in ``/proc/net/unix``.  Known
    prefixes in the wild:

    * ``weblayer_devtools_remote_<pid>`` — WebLayer (Meta Quest / Pico
      default VIEW handler for some OS versions)
    * ``chrome_devtools_remote`` — full Chrome builds
    * ``com.oculus.browser_devtools_remote`` — Meta Quest Browser
    * ``<package>_devtools_remote`` — custom Chromium embedders

    This matcher accepts any of them.  If multiple candidates exist,
    WebLayer / Quest Browser sockets are preferred over generic ones (the
    teleop page is most likely to live there rather than an unrelated
    WebView from another app).
    """
    text = _run_adb(
        "/proc/net/unix scan",
        ["adb", "shell", "cat", "/proc/net/unix"],
        timeout=10.0,
    )
    if text is None:
        return None
    candidates: list[str] = []
    for line in text.splitlines():
        for token in line.split():
            m = _DEVTOOLS_SOCKET_RE.fullmatch(token)
            if m:
                candidates.append(m.group(1))
    if not candidates:
        return None
    # Prefer teleop-relevant prefixes in this order, otherwise fall back to
    # the first discovered socket.
    priority = ("weblayer", "com.oculus.browser", "chrome", "webview")
    for prefix in priority:
        for cand in candidates:
            if cand.startswith(prefix):
                return cand
    return candidates[0]


def _adb_forward_cdp(socket_name: str, local_port: int) -> None:
    assert_adb_device_online()
    _adb_run(
        ["adb", "forward", f"tcp:{local_port}", f"localabstract:{socket_name}"],
        capture_output=True,
        text=True,
        timeout=10,
        check=True,
    )
    log.info("CDP: forwarded tcp:%d -> @%s", local_port, socket_name)


def _adb_forward_remove(local_port: int) -> None:
    _adb_run(
        ["adb", "forward", "--remove", f"tcp:{local_port}"],
        capture_output=True,
        text=True,
        timeout=2,
        check=False,
    )


def teardown_adb_forward_cdp() -> None:
    """Remove the ``adb forward`` rule used for CDP, if present.

    The adb server (a long-lived daemon) owns forward rules — they
    survive a hard kill of *our* process. Calling this at startup
    clears any stale rule from a previous run that didn't get to do
    its own cleanup. ``--remove`` is a no-op when the rule doesn't
    exist, so the call is safe regardless of state.
    """
    _adb_forward_remove(_CDP_LOCAL_PORT)


def _cdp_list_tabs(local_port: int) -> list[dict]:
    try:
        with urllib.request.urlopen(
            f"http://localhost:{local_port}/json", timeout=3
        ) as resp:
            return json.loads(resp.read())
    except Exception as exc:
        log.debug("CDP: failed to list tabs on port %d: %s", local_port, exc)
        return []


def _close_stale_teleop_tabs() -> int:
    """Close any pre-existing teleop tabs (matched on ``oobEnable=``) before opening a new one.

    Why: an errored prior session can leave a tab holding XR resources,
    silently blocking ``requestSession()`` in the next tab. Best-effort —
    no-op if the browser isn't running. Returns the number closed.
    """
    socket_name = _discover_devtools_socket()
    if not socket_name:
        return 0
    try:
        _adb_forward_cdp(socket_name, _CDP_LOCAL_PORT)
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "").strip() or "(no adb output)"
        log.warning(
            "stale-tab cleanup: adb forward rc=%d: %s",
            exc.returncode,
            detail,
        )
        return 0
    except OobAdbError as exc:
        log.warning("stale-tab cleanup: adb forward failed: %s", exc)
        return 0
    closed = 0
    try:
        for tab in _cdp_list_tabs(_CDP_LOCAL_PORT):
            url = tab.get("url") or ""
            tab_id = tab.get("id")
            if not tab_id or "oobEnable=" not in url:
                continue
            try:
                with urllib.request.urlopen(
                    f"http://localhost:{_CDP_LOCAL_PORT}/json/close/{tab_id}",
                    timeout=3,
                ) as resp:
                    resp.read()
                closed += 1
                log.info("stale-tab cleanup: closed tab id=%s url=%s", tab_id, url)
            except Exception as exc:
                log.warning(
                    "stale-tab cleanup: failed to close tab id=%s: %s", tab_id, exc
                )
    finally:
        _adb_forward_remove(_CDP_LOCAL_PORT)
    return closed


async def _cdp_clear_origin_storage(ws_url: str, origins: list[str]) -> int:
    """Clear localStorage / IndexedDB / cookies / cache for *origins* via CDP.

    Why: stale ``general.iceTransportPolicy`` / ``general.turnInfo`` /
    ``cxr.isaac.*`` in localStorage can silently override fresh URL config.
    Per-origin (not everything) so unrelated headset state isn't nuked.
    Best-effort. Returns number of origins cleared.
    """
    from websockets.asyncio.client import connect as ws_connect  # noqa: PLC0415

    _seq = 0

    async def send(ws, method: str, params: dict | None = None) -> dict:
        nonlocal _seq
        _seq += 1
        req_id = _seq
        await ws.send(
            json.dumps({"id": req_id, "method": method, "params": params or {}})
        )
        while True:
            msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=5.0))
            if msg.get("id") == req_id:
                return msg.get("result", {})

    cleared = 0
    try:
        async with ws_connect(ws_url) as ws:
            # Storage.clearDataForOrigin with storageTypes="all" already covers
            # cookies + appcache + cache_storage + service_workers for the
            # given origin, so we don't follow up with browser-wide
            # Network.clearBrowserCache / clearBrowserCookies (those would
            # nuke unrelated tabs the operator may have open).
            for origin in origins:
                try:
                    await send(
                        ws,
                        "Storage.clearDataForOrigin",
                        {"origin": origin, "storageTypes": "all"},
                    )
                    cleared += 1
                    log.info("cache clear: cleared all storage for %s", origin)
                except Exception as exc:
                    log.warning("cache clear: %s failed (%s)", origin, exc)
    except Exception as exc:
        log.warning("cache clear: CDP session failed (%s)", exc)
    return cleared


def clear_headset_browser_cache(*, usb_local: bool) -> int:
    """Sync wrapper around :func:`_cdp_clear_origin_storage` for the teleop UI origin.

    USB-local clears both ``https://localhost:<PROXY_PORT>`` (the bookmark host
    used by ``build_teleop_url``) and ``https://127.0.0.1:<PROXY_PORT>`` (the
    same listener but a different Chromium origin); WiFi clears the
    published client origin (or ``TELEOP_WEB_CLIENT_BASE`` override).
    Returns 0 if the browser isn't running. Never raises.
    """
    from .oob_teleop_env import (  # noqa: PLC0415
        default_web_client_origin,
        web_client_base_override_from_env,
        wss_proxy_port,
    )

    socket_name = _discover_devtools_socket()
    if not socket_name:
        log.info("cache clear: headset browser not running, nothing to clear")
        return 0
    try:
        _adb_forward_cdp(socket_name, _CDP_LOCAL_PORT)
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "").strip() or "(no adb output)"
        log.warning("cache clear: adb forward rc=%d: %s", exc.returncode, detail)
        return 0
    except OobAdbError as exc:
        log.warning("cache clear: adb forward failed: %s", exc)
        return 0

    try:
        tabs = _cdp_list_tabs(_CDP_LOCAL_PORT)
        ws_url = next(
            (t["webSocketDebuggerUrl"] for t in tabs if t.get("webSocketDebuggerUrl")),
            None,
        )
        if not ws_url:
            log.warning("cache clear: no debuggable tab to talk to")
            return 0

        # Origins we own. Trim trailing slash because Storage.clearDataForOrigin
        # is strict about exact origin (scheme://host[:port], no path). Chromium
        # treats ``localhost`` and ``127.0.0.1`` as separate origins, so clear
        # both: the bookmark uses ``localhost`` (per build_teleop_url) but
        # ``127.0.0.1`` may have stale state from earlier development.
        origins: list[str] = []
        if usb_local:
            proxy_port = wss_proxy_port()
            origins.append(f"https://localhost:{proxy_port}")
            origins.append(f"https://127.0.0.1:{proxy_port}")
        else:
            origin_base = (
                web_client_base_override_from_env() or default_web_client_origin()
            )
            # The client base has a trailing slash + path; reduce to origin.
            from urllib.parse import urlparse  # noqa: PLC0415

            parsed = urlparse(origin_base)
            if parsed.scheme and parsed.netloc:
                origins.append(f"{parsed.scheme}://{parsed.netloc}")

        if not origins:
            log.warning("cache clear: no origins to clear (override misconfigured?)")
            return 0

        return asyncio.run(_cdp_clear_origin_storage(ws_url, origins))
    finally:
        _adb_forward_remove(_CDP_LOCAL_PORT)


# Must match CloudXRUI.tsx's RESET_PANEL_KEY / the keydown listener's key check.
_RESET_PANEL_KEY = "r"
_RESET_PANEL_CODE = (
    "KeyR"  # DOM KeyboardEvent.code convention, not the same string as .key
)
_RESET_PANEL_VIRTUAL_KEY_CODE = 82  # VK_R


async def _cdp_send_reset_panel_key(ws_url: str) -> None:
    """Synthesize the reset-panel keypress (CloudXRUI.tsx's window keydown listener) over CDP.

    Recovers a "missing panel" (hidden, or dragged out of reach) by fixing it rather than
    detecting it first: the client-side handler un-hides the panel and repositions it in front
    of the current head pose, regardless of where it ended up or why. Best-effort - exceptions
    propagate; callers treat this as non-fatal.
    """
    from websockets.asyncio.client import connect as ws_connect  # noqa: PLC0415

    _seq = 0

    async def send(ws, method: str, params: dict | None = None) -> None:
        nonlocal _seq
        _seq += 1
        req_id = _seq
        await ws.send(
            json.dumps({"id": req_id, "method": method, "params": params or {}})
        )
        while True:
            msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=5.0))
            if msg.get("id") == req_id:
                return

    key_params = {
        "key": _RESET_PANEL_KEY,
        "code": _RESET_PANEL_CODE,
        "windowsVirtualKeyCode": _RESET_PANEL_VIRTUAL_KEY_CODE,
        "nativeVirtualKeyCode": _RESET_PANEL_VIRTUAL_KEY_CODE,
    }
    async with ws_connect(ws_url) as ws:
        await send(ws, "Input.dispatchKeyEvent", {"type": "rawKeyDown", **key_params})
        await send(ws, "Input.dispatchKeyEvent", {"type": "keyUp", **key_params})
    log.info("CDP: sent reset-panel key (%s)", _RESET_PANEL_KEY)


async def _cdp_session_click_connect(
    ws_url: str,
    *,
    refresh_static_assets: bool = False,
    allow_navigation: bool = True,
    clear_stale_error: bool = False,
    on_dispatched: Callable[[], Awaitable[None] | None] | None = None,
    on_client_loaded: Callable[[], Awaitable[None] | None] | None = None,
) -> None:
    """Open a single CDP session and click the CONNECT button.

    Handles the self-signed cert interstitial before looking for the button:

    * Primary path — ``Security.setIgnoreCertificateErrors`` + ``Page.navigate``
      (re-loads the page with cert checking disabled).
    * Fallback — DOM click-through: ``details-button`` → ``proceed-link``
      (standard Chromium cert-warning IDs).
    """
    from websockets.asyncio.client import connect as ws_connect  # already a dep

    _seq = 0

    async def send(ws, method, params=None):
        nonlocal _seq
        _seq += 1
        req_id = _seq
        await ws.send(
            json.dumps({"id": req_id, "method": method, "params": params or {}})
        )
        while True:
            msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=10.0))
            if msg.get("id") == req_id:
                return msg.get("result", {})

    async with ws_connect(ws_url) as ws:
        # ---- cert warning handling ----------------------------------------
        cert_suppressed = False
        try:
            await send(ws, "Security.setIgnoreCertificateErrors", {"ignore": True})
            cert_suppressed = True
            log.info("CDP: cert errors suppressed")
        except Exception as exc:
            log.debug(
                "CDP: Security domain unavailable (%s), will try DOM fallback", exc
            )

        # Detect interstitial: Chromium cert warning pages have #details-button
        r = await send(
            ws,
            "Runtime.evaluate",
            {
                "expression": "!!document.getElementById('details-button')",
                "returnByValue": True,
            },
        )
        on_interstitial = r.get("result", {}).get("value", False)

        if on_interstitial:
            if not allow_navigation:
                raise OobAdbError(
                    "CDP: surviving teleop tab is on a certificate interstitial"
                )
            log.info("CDP: cert interstitial detected")
            navigated = False
            if cert_suppressed:
                r2 = await send(
                    ws,
                    "Runtime.evaluate",
                    {
                        "expression": "window.location.href",
                        "returnByValue": True,
                    },
                )
                current_url = r2.get("result", {}).get("value", "")
                if current_url and not current_url.startswith("chrome-error"):
                    log.info("CDP: re-navigating to %s", current_url)
                    await send(ws, "Page.navigate", {"url": current_url})
                    await asyncio.sleep(3.0)
                    navigated = True
                else:
                    log.warning(
                        "CDP: interstitial URL is %r, falling back to DOM click-through",
                        current_url,
                    )

            if not navigated:
                await send(
                    ws,
                    "Runtime.evaluate",
                    {
                        "expression": "document.getElementById('details-button')?.click()",
                    },
                )
                await asyncio.sleep(1.5)
                await send(
                    ws,
                    "Runtime.evaluate",
                    {
                        "expression": "document.getElementById('proceed-link')?.click()",
                    },
                )
                await asyncio.sleep(3.0)

        if refresh_static_assets:
            # The source build can replace bundle.js at the same URL. Chromium
            # may reuse its old HTTP cache even after Storage.clearDataForOrigin;
            # reload this tab with caching disabled before clicking CONNECT.
            try:
                await send(ws, "Network.enable")
                await send(ws, "Network.setCacheDisabled", {"cacheDisabled": True})
                await send(ws, "Page.reload", {"ignoreCache": True})
                log.info("CDP: reloaded local WebXR page bypassing HTTP cache")
                await asyncio.sleep(1.0)
            except Exception as exc:
                log.warning(
                    "CDP: cache-bypassing reload unavailable (%s); browser health "
                    "probe will still verify the client",
                    exc,
                )

        if clear_stale_error:
            # A terminal CloudXR error remains visible after the button returns to
            # CONNECT. Hide that old banner before polling the new attempt so it
            # cannot be mistaken for a failure produced by this click.
            await send(
                ws,
                "Runtime.evaluate",
                {
                    "expression": """(function() {
                    const box = document.getElementById('errorMessageBox');
                    const text = document.getElementById('errorMessageText');
                    box?.classList.remove('show');
                    if (text) text.textContent = '';
                })()"""
                },
            )

        # ---- bring tab to foreground so WebXR requestSession() succeeds ------
        # WebXR requires the page to be visible; Page.bringToFront activates the tab.
        try:
            await send(ws, "Page.bringToFront")
            log.info("CDP: tab brought to foreground")
        except Exception as exc:
            log.debug("CDP: Page.bringToFront failed (%s), continuing", exc)

        # ---- wait for #startButton to become actionable ----------------------
        # State machine returned each poll:
        #   {state: 'loading'}       — document / button not ready yet
        #   {state: 'initializing'}  — button exists but disabled (IWER +
        #                              capability checks still running)
        #   {state: 'failed', text}  — capability check set a "failed" label;
        #                              we will never be able to click, error out
        #   {state: 'ready', x, y, text, disabled}
        #                            — text === 'CONNECT' and not disabled
        _READINESS_TIMEOUT = 30.0  # capability/IWER checks can be slow
        loop = asyncio.get_running_loop()
        deadline_ready = loop.time() + _READINESS_TIMEOUT
        start_ready = loop.time()
        val: dict | None = None
        last_state: str | None = None
        while loop.time() < deadline_ready:
            r = await send(
                ws,
                "Runtime.evaluate",
                {
                    "expression": """(function() {
                    if (document.readyState !== 'complete') return {state: 'loading'};
                    const btn = document.getElementById('startButton');
                    if (!btn) return {state: 'loading'};
                    const text = btn.textContent?.trim() || '';
                    const disabled = !!btn.disabled;
                    if (text.toUpperCase().includes('FAIL')) {
                        return {state: 'failed', text, disabled};
                    }
                    if (disabled || text.toUpperCase() !== 'CONNECT') {
                        return {state: 'initializing', text, disabled};
                    }
                    const rc = btn.getBoundingClientRect();
                    return {
                        state: 'ready',
                        text, disabled,
                        x: rc.left + rc.width / 2,
                        y: rc.top + rc.height / 2,
                    };
                })()""",
                    "returnByValue": True,
                },
            )
            val = (r.get("result") or {}).get("value") or {"state": "loading"}
            state = val.get("state")
            if state != last_state:
                log.info(
                    "CDP: page state=%s text=%r disabled=%s",
                    state,
                    val.get("text"),
                    val.get("disabled"),
                )
                last_state = state
            if state == "ready":
                break
            if state == "failed":
                raise OobAdbError(
                    f"CDP: startButton marked failed (text={val.get('text')!r}). "
                    "The web client's capability check failed — inspect the headset."
                )
            await asyncio.sleep(0.5)

        if val is None or val.get("state") != "ready":
            raise OobAdbError(
                f"CDP: startButton not actionable within {_READINESS_TIMEOUT:.0f}s "
                f"(state={val.get('state') if val else 'unknown'!r}, "
                f"text={val.get('text') if val else None!r}). "
                "The page may still be initializing — check the headset."
            )

        log.info("CDP: page ready in %.1fs", loop.time() - start_ready)
        if on_client_loaded is not None:
            result = on_client_loaded()
            if isawaitable(result):
                await result

        # Extra grace period: the CONNECT button can become enabled before the
        # React <XR> store is fully mounted, which causes "XR is not available"
        # errors if clicked immediately.
        await asyncio.sleep(2.0)

        # Click in two phases:
        #
        # 1. Input.dispatchMouseEvent (mousePressed + mouseReleased) — this
        #    is a *trusted* input event, so the browser grants a user-
        #    activation token (required for navigator.xr.requestSession()).
        #    PICO Browser's onClick also fires from this path, so on PICO
        #    phase 2 is effectively a no-op.
        # 2. element.click() — programmatic follow-up needed on Meta Quest
        #    Browser, where the synthesized mouse event grants activation
        #    but does not fire React's onClick handler (touch-first routing).
        #    By running inside the user-activation window opened in phase 1,
        #    requestSession() still sees activation when onClick runs.
        x, y = val["x"], val["y"]
        log.info("CDP: clicking CONNECT at (%.0f, %.0f)", x, y)
        for event_type in ("mousePressed", "mouseReleased"):
            await send(
                ws,
                "Input.dispatchMouseEvent",
                {
                    "type": event_type,
                    "x": x,
                    "y": y,
                    "button": "left",
                    "clickCount": 1,
                },
            )
        dispatched = on_dispatched() if on_dispatched is not None else None
        # Keep the Quest fallback inside the user-activation window; status
        # persistence may block, so await it only after the DOM click.
        try:
            await send(
                ws,
                "Runtime.evaluate",
                {
                    "expression": "document.getElementById('startButton')?.click()",
                },
            )
        finally:
            if isawaitable(dispatched):
                await dispatched
        log.info("CDP: CONNECT click dispatched (mouse + DOM)")

        # ---- monitor connection outcome -------------------------------------
        # DOM facts:
        #   - Button:     id="startButton", textContent "CONNECT" when idle,
        #                 changes to "DISCONNECT" / other while streaming.
        #   - Error text: id="errorMessageText" (child of the error box).
        #                 textContent is used (not innerText) so the text is
        #                 readable even when the box has display:none.
        _CONNECT_TIMEOUT = 30.0
        loop = asyncio.get_running_loop()
        deadline = loop.time() + _CONNECT_TIMEOUT
        while loop.time() < deadline:
            await asyncio.sleep(1.0)
            r = await send(
                ws,
                "Runtime.evaluate",
                {
                    "expression": """(function() {
                    const btn = document.getElementById('startButton');
                    const btnText = btn?.textContent?.trim()?.toUpperCase() || null;
                    const box = document.getElementById('errorMessageBox');
                    // Only treat as error when the box is shown with type 'error'
                    // (not 'success' or 'info' which are non-fatal status messages).
                    const isError = box?.classList?.contains('show') &&
                                   !box?.classList?.contains('success') &&
                                   !box?.classList?.contains('info');
                    const errorText = isError
                        ? (document.getElementById('errorMessageText')?.textContent?.trim() || null)
                        : null;
                    return {btnText, errorText};
                })()""",
                    "returnByValue": True,
                },
            )
            state = (r.get("result") or {}).get("value") or {}
            btn_text = state.get("btnText")
            error_text = state.get("errorText") or None

            if error_text:
                raise OobAdbError(f"Teleop connection failed: {error_text}")
            if btn_text is not None and btn_text != "CONNECT":
                log.info("CDP: start button changed to %r — session active", btn_text)
                return

        log.warning(
            "CDP: connection state unknown after %.0fs — check headset",
            _CONNECT_TIMEOUT,
        )


async def attach_existing_oob_tab(
    *,
    resolved_port: int,
    click_connect: bool = False,
    on_dispatched: Callable[[], Awaitable[None] | None] | None = None,
    on_client_loaded: Callable[[], Awaitable[None] | None] | None = None,
    usb_local: bool = False,
    host_client: bool = False,
    on_terminal_error: Callable[[str], None] | None = None,
) -> asyncio.Task:
    """Attach CDP monitoring to a surviving OOB tab without navigating it.

    When *click_connect* is true, dispatch one trusted CONNECT click in the
    existing tab, then unconditionally reset the panel (see
    :func:`_cdp_send_reset_panel_key`) — this is the one reconnect path the
    client's own automatic initial-placement reset never covers, since no new
    XR session starts (the page was never reloaded). Best-effort: a reset
    failure is logged and swallowed, not fatal to the reattach. This path
    never closes tabs, invokes ``am start``, reloads, or creates a new
    browser page. *on_terminal_error* is forwarded to the monitor — see
    :func:`_monitor_teleop_error_banner`'s doc comment for why it only
    reports, never acts.
    """
    socket_name = await asyncio.to_thread(_discover_devtools_socket)
    if not socket_name:
        raise OobAdbError("CDP: no browser DevTools socket after USB repair")
    # Shielded: a cancellation landing right after asyncio.to_thread() returns (but before
    # this coroutine resumes) wouldn't stop the underlying subprocess call anyway - shielding
    # just makes sure *this* coroutine also waits for it to settle, so the forward's actual
    # install state is known before the outer except below decides whether to remove it.
    forward_task = asyncio.create_task(
        asyncio.to_thread(_adb_forward_cdp, socket_name, _CDP_LOCAL_PORT)
    )
    try:
        try:
            await asyncio.shield(forward_task)
        except asyncio.CancelledError:
            try:
                await forward_task
            except Exception:
                # Only draining forward_task to settle it before the `raise` below
                # propagates the real cancellation - a failure here must not replace it.
                pass
            raise
        tabs = await asyncio.to_thread(_cdp_list_tabs, _CDP_LOCAL_PORT)
        tab = next(
            (
                item
                for item in tabs
                if item.get("webSocketDebuggerUrl")
                and "oobEnable=" in (item.get("url") or "")
            ),
            None,
        )
        if tab is None:
            raise OobAdbError("CDP: no surviving teleop tab after USB repair")
        ws_url = tab["webSocketDebuggerUrl"]
        if click_connect:
            await _cdp_session_click_connect(
                ws_url,
                refresh_static_assets=False,
                allow_navigation=False,
                clear_stale_error=True,
                on_dispatched=on_dispatched,
                on_client_loaded=on_client_loaded,
            )
            try:
                await _cdp_send_reset_panel_key(ws_url)
            except Exception as exc:
                log.warning("CDP: reset-panel key failed after reattach: %s", exc)
        return asyncio.create_task(
            _monitor_teleop_error_banner(
                ws_url,
                _CDP_LOCAL_PORT,
                on_terminal_error=on_terminal_error,
            ),
            name="cloudxr-oob-error-monitor",
        )
    except BaseException:
        await asyncio.to_thread(_adb_forward_remove, _CDP_LOCAL_PORT)
        raise


async def _find_and_click_teleop_tab(
    *,
    resolved_port: int,
    deadline: float,
    timeout: float,
    usb_local: bool = False,
    host_client: bool = False,
    refresh_static_assets: bool = False,
    on_dispatched: Callable[[], Awaitable[None] | None] | None = None,
    on_client_loaded: Callable[[], Awaitable[None] | None] | None = None,
) -> str:
    """Find the teleop tab (freshly opened or navigated by ``am start``) via the CDP forward
    already up on ``_CDP_LOCAL_PORT``, then click CONNECT on it.

    Re-fires ``am start`` once, at *timeout*'s midpoint (measured from entry to this
    function, not from *deadline*), if the tab hasn't appeared yet — cold-launch can
    swallow the first VIEW intent while the browser wakes up, and a second intent
    reliably navigates an already-warm tab.

    Called by :func:`run_oob_connect` right after its own ``am start`` + forward setup, so
    *deadline* may already be partly consumed by that setup. Both a genuinely fresh launch
    and :class:`OobLifecycle`'s own full-bootstrap recovery (``_automate()``) go through
    :func:`run_oob_connect`, so this one caller covers both.

    Returns the tab's ``webSocketDebuggerUrl``. Raises :exc:`OobAdbError` if no matching
    tab appears before *deadline*.
    """

    # Teleop URL substrings match on the happy path.  We also accept
    # ``chrome-error://`` URLs because Chromium parks the tab there when
    # the self-signed cert is blocked — the cert-bypass in
    # ``_cdp_session_click_connect`` recovers from that state.
    def _is_candidate_tab(tab: dict) -> bool:
        url = tab.get("url") or ""
        if not tab.get("webSocketDebuggerUrl") or not url:
            return False
        return (
            "oobEnable" in url
            or "localhost" in url
            or "IsaacTeleop" in url
            or url.startswith("chrome-error://")
        )

    # Snapshot {id → url} BEFORE we look for changes so we can detect both
    # new tabs and existing tabs that were navigated to the new URL by am start.
    tabs_url_before = {
        t["id"]: (t.get("url") or "")
        for t in await asyncio.to_thread(_cdp_list_tabs, _CDP_LOCAL_PORT)
        if "id" in t
    }
    log.info("CDP: %d tab(s) before navigation", len(tabs_url_before))

    ws_url: str | None = None
    am_start_retried = False
    # Re-fire am start once if we don't see the tab in the first half
    # of the deadline. Cold-launch can swallow the first VIEW intent
    # (browser was being woken up), and a second intent reliably
    # navigates an already-warm tab.
    retry_at = time.monotonic() + (timeout / 2)
    while ws_url is None and time.monotonic() < deadline:
        await asyncio.sleep(1.0)
        for tab in await asyncio.to_thread(_cdp_list_tabs, _CDP_LOCAL_PORT):
            if "id" not in tab or not tab.get("webSocketDebuggerUrl"):
                continue
            old_url = tabs_url_before.get(tab["id"])
            current_url = tab.get("url") or ""
            # Case A: brand-new tab — accept only if it looks like our page
            # (happy path) or is a cert-error page (recoverable).
            if old_url is None:
                if not _is_candidate_tab(tab):
                    continue
                ws_url = tab["webSocketDebuggerUrl"]
                log.info("CDP: new tab %r url=%s", tab.get("title"), current_url)
                break
            # Case B: existing tab whose URL changed after am start — this
            # is ours (the VIEW intent just navigated it).  Trust the diff
            # even if the new URL is chrome-error://.
            if old_url != current_url:
                ws_url = tab["webSocketDebuggerUrl"]
                log.info(
                    "CDP: navigated tab %r url=%s (was %s)",
                    tab.get("title"),
                    current_url,
                    old_url or "<new>",
                )
                break
            # Case C: existing tab whose URL was already our teleop URL
            # at snapshot time and hasn't changed since. Happens when
            # the browser navigated between cleanup and snapshot — am
            # start fired, the tab landed on our URL, /json/list returned,
            # and now there's no diff to detect. The ``oobEnable=`` query
            # param is unique to URLs we generate, so matching it here
            # won't grab an unrelated tab.
            if "oobEnable=" in current_url:
                ws_url = tab["webSocketDebuggerUrl"]
                log.info(
                    "CDP: existing teleop tab %r url=%s (snapshot already current)",
                    tab.get("title"),
                    current_url,
                )
                break

        if ws_url is not None:
            break

        if not am_start_retried and time.monotonic() >= retry_at:
            am_start_retried = True
            log.warning(
                "CDP: tab not found after %.1fs — re-firing am start (cold-launch race)",
                timeout / 2,
            )
            oob_progress(
                "setup-oob",
                "tab not found yet — re-firing am start (browser may have "
                "swallowed the first intent on cold launch) ...",
            )
            try:
                rc, diag = await asyncio.to_thread(
                    run_adb_headset_bookmark,
                    resolved_port=resolved_port,
                    usb_local=usb_local,
                    host_client=host_client,
                )
                if rc != 0:
                    log.warning("CDP: am start re-fire rc=%d: %s", rc, diag)
            except Exception as exc:
                log.warning("CDP: am start re-fire raised: %s", exc)

    if ws_url is None:
        raise OobAdbError(
            "CDP: browser tab for the teleop page not found within timeout "
            "(am start was re-fired once mid-way and still no match).\n"
            "The page may not have loaded — open the teleop URL on the headset manually "
            "and tap CONNECT."
        )

    oob_progress(
        "setup-oob",
        "teleop tab found — accepting cert, waiting for CONNECT button, "
        "then auto-clicking it ...",
    )

    # cert interstitial + bring to front + readiness + click
    # _cdp_session_click_connect polls the DOM for document.readyState +
    # #startButton (up to 10s) so no fixed page-init sleep is needed here.
    await _cdp_session_click_connect(
        ws_url,
        refresh_static_assets=refresh_static_assets,
        on_dispatched=on_dispatched,
        on_client_loaded=on_client_loaded,
    )
    return ws_url


async def run_oob_connect(
    *,
    resolved_port: int,
    timeout: float = 60.0,
    usb_local: bool = False,
    host_client: bool = False,
    on_dispatched: Callable[[], Awaitable[None] | None] | None = None,
    on_client_loaded: Callable[[], Awaitable[None] | None] | None = None,
    on_terminal_error: Callable[[str], None] | None = None,
) -> asyncio.Task | None:
    """Open the teleop page on the headset via ``am start`` and click CONNECT via CDP.

    Flow:
      1. Launch the teleop URL on the headset via ``adb shell am start``.
      2. Wait for the browser's DevTools abstract socket to appear and forward it
         (WebLayer / Meta Quest Browser / Chrome all supported).
      3. Find the teleop tab in ``/json`` (by URL content or recent navigation).
      4. Bring the tab to the foreground (required by WebXR ``requestSession``).
      5. Handle the self-signed cert interstitial if present.
      6. Find the CONNECT button and click it via ``Input.dispatchMouseEvent``.
      7. Start a background monitor that forwards mid-stream errors from the
         web client's ``errorMessageBox`` into the server log, and reports
         terminal ones via *on_terminal_error* - diagnostic only, see
         :func:`_monitor_teleop_error_banner`'s doc comment.

    No reset-panel keypress here (unlike :func:`attach_existing_oob_tab`): this
    path always navigates to a fresh page, and ``CloudXRUI.tsx``'s own
    first-XR-frame placement already resets the panel unconditionally on every
    new session — sending the key too would just be a redundant second write
    to the same position.

    Args:
        resolved_port: WSS proxy port used for signalling.
        timeout: Maximum seconds to wait for the browser/tab to appear.
        usb_local: When ``True``, the headset URL uses ``serverIP=127.0.0.1``
            and the local HTTPS static server on the USB loopback port.
        host_client: When ``True`` (and not usb_local), the headset URL uses
            ``https://<lan>:<wss_port>/client/`` instead of the versioned
            GitHub Pages origin.
        on_dispatched: Called after the trusted CONNECT mouse click, before
            the Quest DOM fallback and connection polling.

    Returns:
        A running :class:`asyncio.Task` that monitors the headset's error
        banner and keeps the ``adb forward`` alive.  Callers should cancel
        it at shutdown (``task.cancel()``).  Returns ``None`` if the click
        phase succeeded but the monitor could not be spawned.

    Raises :exc:`OobAdbError` on any unrecoverable failure during the click
    phase; callers should treat this as non-fatal and ask the user to tap
    CONNECT manually.
    """
    deadline = time.monotonic() + timeout

    # --- Step 0: close any stale teleop tabs from a prior session -----------
    # An errored WebXR session in an old tab can hold XR resources and
    # block the new tab's requestSession(). Cleaning up first gives us a
    # known-good starting state.
    closed = await asyncio.to_thread(_close_stale_teleop_tabs)
    if closed:
        oob_progress(
            "setup-oob",
            f"closed {closed} stale teleop tab(s) from a prior session",
        )
        # Brief pause so the browser actually releases WebXR / media
        # resources before the new page tries to acquire them.
        await asyncio.sleep(1.0)

    # --- Step 1: launch browser with the teleop URL --------------------------
    rc, diag = await asyncio.to_thread(
        run_adb_headset_bookmark,
        resolved_port=resolved_port,
        usb_local=usb_local,
        host_client=host_client,
    )
    if rc != 0:
        hint = adb_automation_failure_hint(diag)
        raise OobAdbError(oob_adb_automation_message(rc, diag, hint))
    log.info("ADB: am start completed")
    oob_progress(
        "setup-oob",
        "headset browser launched — discovering DevTools socket ...",
    )

    # --- Step 2: wait for DevTools socket ------------------------------------
    socket_name = None
    while time.monotonic() < deadline:
        socket_name = await asyncio.to_thread(_discover_devtools_socket)
        if socket_name:
            break
        log.info("CDP: waiting for browser DevTools socket...")
        await asyncio.sleep(2.0)

    if not socket_name:
        raise OobAdbError(
            "CDP: no *_devtools_remote abstract socket found on the headset "
            "after opening the teleop URL.\n\n"
            "Chromium-based headset browsers (WebLayer, Meta Quest Browser, "
            "Chrome) expose one when remote debugging is enabled. Check that "
            "USB debugging is authorized, the browser actually launched from "
            "`am start`, and `adb shell cat /proc/net/unix | grep devtools_remote` "
            "lists a socket."
        )
    log.info("CDP: found socket @%s", socket_name)

    forward_task = asyncio.create_task(
        asyncio.to_thread(_adb_forward_cdp, socket_name, _CDP_LOCAL_PORT)
    )
    try:
        await asyncio.shield(forward_task)
    except asyncio.CancelledError:
        # Finish the mutation before removing the forward on cancellation.
        try:
            await forward_task
        except Exception:
            # Keep cancellation as the outcome even if forwarding failed.
            pass
        await asyncio.to_thread(_adb_forward_remove, _CDP_LOCAL_PORT)
        raise
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "").strip() or "(no adb output)"
        raise OobAdbError(
            f"CDP: adb forward tcp:{_CDP_LOCAL_PORT} -> @{socket_name} "
            f"failed (rc={exc.returncode}): {detail}"
        ) from exc

    try:
        # --- Steps 3+4: find the teleop tab and click CONNECT ------------------
        ws_url = await _find_and_click_teleop_tab(
            resolved_port=resolved_port,
            deadline=deadline,
            timeout=timeout,
            usb_local=usb_local,
            host_client=host_client,
            refresh_static_assets=usb_local or host_client,
            on_dispatched=on_dispatched,
            on_client_loaded=on_client_loaded,
        )

        # --- Step 5: background monitor for mid-stream error banners ---------
        # Keep the adb forward alive; the monitor tears it down on exit.
        monitor_task = asyncio.create_task(
            _monitor_teleop_error_banner(
                ws_url,
                _CDP_LOCAL_PORT,
                on_terminal_error=on_terminal_error,
            ),
            name="cloudxr-oob-error-monitor",
        )
        return monitor_task
    except BaseException:
        # Any failure after the forward is set up but before we hand ownership
        # of it to the monitor task must clean the forward up here.
        await asyncio.to_thread(_adb_forward_remove, _CDP_LOCAL_PORT)
        raise


def _teleop_error_hint(banner: str) -> str:
    """Map known headset error banners to actionable host-side hints, or ``""``."""
    b = banner.lower()
    if "0xc0f2220f" in b or "no local connection candidates" in b:
        return (
            "no ICE candidates: typically TURN unreachable, headset WiFi off, "
            "or host firewall. Verify coturn is running and `ufw allow` the proxy port."
        )
    if "wss" in b and ("close" in b or "1006" in b):
        return "WSS dropped: check the proxy port is reachable from the headset."
    return ""


async def _monitor_teleop_error_banner(
    ws_url: str,
    local_port: int,
    *,
    on_terminal_error: Callable[[str], None] | None = None,
) -> None:
    """Forward ``errorMessageBox`` content from the web client into the server log, and
    report a terminal (non-info/non-success) client error via *on_terminal_error*.

    Polls the DOM once per second over its own CDP session. Only class ``error`` banners
    are reported — CloudXRComponent's bounded mid-retry "Reconnecting (n/max)" status is
    class ``info``, so this only fires once the client has given up retrying. De-dupes
    identical messages so a banner that remains displayed reports once.

    Diagnostic only: this function never closes, navigates, or relaunches the tab itself.
    Recovery is entirely owned by the caller (:class:`OobLifecycle`), which decides
    whether/how to act on *on_terminal_error* and owns tearing this monitor down via task
    cancellation when it does.

    Runs until the task is cancelled (normal shutdown) or the WebSocket drops - e.g.
    because the caller closed the tab as part of its own recovery. Always tears down the
    ``adb forward`` on exit.
    """
    from websockets.asyncio.client import connect as ws_connect  # noqa: PLC0415

    _seq = 0

    async def send(ws, method: str, params: dict | None = None) -> dict:
        nonlocal _seq
        _seq += 1
        req_id = _seq
        await ws.send(
            json.dumps({"id": req_id, "method": method, "params": params or {}})
        )
        while True:
            msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=10.0))
            if msg.get("id") == req_id:
                return msg.get("result", {})

    try:
        async with ws_connect(ws_url) as ws:
            # Keep errors suppressed so the tab never stops rendering because of
            # a stray cert hiccup on a later navigation.
            try:
                await send(ws, "Security.setIgnoreCertificateErrors", {"ignore": True})
            except Exception as exc:
                log.debug("monitor: Security domain unavailable (%s)", exc)
            log.info("monitor: tracking errorMessageBox on the teleop page")
            last_banner = ""
            while True:
                await asyncio.sleep(1.0)
                r = await send(
                    ws,
                    "Runtime.evaluate",
                    {
                        "expression": """(function() {
                            const box = document.getElementById('errorMessageBox');
                            if (!box || !box.classList.contains('show')) return '';
                            if (box.classList.contains('success') ||
                                box.classList.contains('info')) return '';
                            return document.getElementById('errorMessageText')
                                       ?.textContent?.trim() || '';
                        })()""",
                        "returnByValue": True,
                    },
                )
                banner = (r.get("result") or {}).get("value") or ""
                if banner and banner != last_banner:
                    log.warning("Teleop client error: %s", banner)
                    extra = _teleop_error_hint(banner)
                    # Mirror to stderr so the operator sees mid-stream errors
                    # in the console, not only in the server log file.
                    print(
                        f"\n\033[33mTeleop client error: {banner}\033[0m\n"
                        + (f"\033[33m  → {extra}\033[0m\n" if extra else ""),
                        file=sys.stderr,
                        flush=True,
                    )
                    if on_terminal_error is not None:
                        on_terminal_error(banner)
                last_banner = banner
    except asyncio.CancelledError:
        log.info("monitor: cancelled")
        raise
    except Exception as exc:
        # WS drop, CDP error, etc. — expected at tab close; log and exit quietly.
        log.info("monitor: exiting (%s)", exc)
    finally:
        await asyncio.to_thread(_adb_forward_remove, local_port)
