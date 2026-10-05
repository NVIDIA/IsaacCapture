# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OOB teleop environment: proxy port, LAN detection, stream defaults, headset bookmark URLs, startup banner."""

from __future__ import annotations

import logging
import math
import os
import re
import socket
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from urllib.error import URLError
from urllib.parse import urlencode, urljoin
from urllib.request import Request, urlopen

from .oob_teleop_hub import OOB_WS_PATH

log = logging.getLogger("isaaccapture.cloudxr.oob_teleop_env")

WSS_PROXY_DEFAULT_PORT = 48322

# GitHub Pages WebXR client root. The published client lives under a per-ref
# slug (``main``, ``release-1.3.x``, ``v1.2.3``, ...); the docs build emits one
# slug per ref it builds. :func:`default_web_client_origin` resolves the slug
# for the installed version so OOB opens the matching client.
WEB_CLIENT_BASE = "https://nvidia.github.io/IsaacCapture/client/"

# Origin used when the installed version can't be resolved (dev trees, tests).
FALLBACK_WEB_CLIENT_ORIGIN = urljoin(WEB_CLIENT_BASE, "main/")

CLIENT_RECONNECT_DEFAULT_ENABLED = True
CLIENT_RECONNECT_DEFAULT_MAX_ATTEMPTS = 10
CLIENT_RECONNECT_DEFAULT_DELAY_MS = 3000


def client_reconnect_config_from_env() -> dict:
    """Return validated WebXR retry settings shared by the URL and host lifecycle."""

    def boolean(name: str, default: bool) -> bool:
        raw = os.environ.get(name)
        if raw is None:
            return default
        value = raw.strip().lower()
        if value in {"1", "true", "yes", "on"}:
            return True
        if value in {"0", "false", "no", "off"}:
            return False
        raise ValueError(f"{name} must be a boolean")

    def nonnegative_int(name: str, default: int) -> int:
        raw = os.environ.get(name)
        try:
            value = default if raw is None else int(raw)
        except ValueError as exc:
            raise ValueError(f"{name} must be a non-negative integer") from exc
        if value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
        return value

    return {
        "reconnectEnabled": boolean(
            "TELEOP_CLIENT_RECONNECT_ENABLED", CLIENT_RECONNECT_DEFAULT_ENABLED
        ),
        "reconnectMaxAttempts": nonnegative_int(
            "TELEOP_CLIENT_RECONNECT_MAX_ATTEMPTS",
            CLIENT_RECONNECT_DEFAULT_MAX_ATTEMPTS,
        ),
        "reconnectDelayMs": nonnegative_int(
            "TELEOP_CLIENT_RECONNECT_DELAY_MS", CLIENT_RECONNECT_DEFAULT_DELAY_MS
        ),
    }


def resolve_oob_recovery_config():
    """Resolve and validate the bounded recovery cadence once at startup."""
    from .oob_teleop_lifecycle import RecoveryConfig  # noqa: PLC0415

    def positive(name: str, default: float) -> float:
        raw = os.environ.get(name)
        try:
            value = default if raw is None else float(raw)
        except ValueError as exc:
            raise ValueError(f"{name} must be a positive finite number") from exc
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be a positive finite number")
        return value

    reconnect = client_reconnect_config_from_env()
    return RecoveryConfig(
        timeout_sec=positive("TELEOP_OOB_RECOVERY_TIMEOUT_SEC", 60.0),
        interval_sec=positive("TELEOP_OOB_RETRY_INTERVAL_SEC", 5.0),
        client_reconnect_enabled=reconnect["reconnectEnabled"],
        client_reconnect_max_attempts=reconnect["reconnectMaxAttempts"],
        client_reconnect_delay_ms=reconnect["reconnectDelayMs"],
    )


def versioned_web_client_url(version: str) -> str:
    """GitHub Pages WebXR client URL matching *version*.

    A clean ``MAJOR.MINOR.PATCH`` release (a tag build) maps to the per-tag
    client ``client/vMAJOR.MINOR.PATCH/``. Pre-release / dev builds (``1.2.4rc1``,
    ``1.3.0.dev5``, ...) and any other version with a leading MAJOR.MINOR map to
    the release line ``client/release-MAJOR.MINOR.x/``. Versions with no
    parseable MAJOR.MINOR fall back to the generic ``client/`` URL, which the
    site redirects to the latest stable tag. The same helper backs the
    standalone "web client:" line printed in non-OOB mode, so every path
    agrees on which client to open.
    """
    v = version.strip()
    if re.fullmatch(r"\d+\.\d+\.\d+", v):
        return urljoin(WEB_CLIENT_BASE, f"v{v}/")
    m = re.match(r"(\d+)\.(\d+)", v)
    if m:
        return urljoin(WEB_CLIENT_BASE, f"release-{m.group(1)}.{m.group(2)}.x/")
    return WEB_CLIENT_BASE


def default_web_client_origin() -> str:
    """Versioned WebXR client origin for the installed ``isaaccapture`` version.

    Reads the installed distribution version (namespace-independent, so it works
    under the test package alias too) and maps it via
    :func:`versioned_web_client_url`. Falls back to
    :data:`FALLBACK_WEB_CLIENT_ORIGIN` when the version can't be determined.
    """
    try:
        return versioned_web_client_url(version("isaaccapture"))
    except PackageNotFoundError:
        return FALLBACK_WEB_CLIENT_ORIGIN


# Upper bound for downloaded client assets (supply-chain / accident guard).
_USB_LOCAL_ASSET_MAX_BYTES = 32 * 1024 * 1024

TELEOP_WEB_CLIENT_BASE_ENV = "TELEOP_WEB_CLIENT_BASE"

# Hash-router fragment appended to the bookmark URL (no fragment by default).
# The WebXR client uses HashRouter, so the fragment selects the route. Set
# ``TELEOP_CLIENT_ROUTE`` to a path like ``/real/gear/dexmate`` to land the
# headset on a specific route; leave unset (or empty) for the client default.
TELEOP_CLIENT_ROUTE_ENV = "TELEOP_CLIENT_ROUTE"
DEFAULT_TELEOP_CLIENT_ROUTE = ""

# Directory with prebuilt WebXR assets (``index.html``, ``bundle.js``,
# ``bundle.emulator.js``). Optional for ``--usb-local``: defaults to
# ``~/.cloudxr/static-client``; missing files are fetched from published URLs.
TELEOP_WEB_CLIENT_STATIC_DIR_ENV = "TELEOP_WEB_CLIENT_STATIC_DIR"

CHROME_INSPECT_DEVICES_URL = "chrome://inspect/#devices"

# ---------------------------------------------------------------------------
# USB-local mode constants
#
# "USB-local" means: the headset reaches the PC over loopback (127.0.0.1) via
# ``adb reverse``.  Static assets live under ``TELEOP_WEB_CLIENT_STATIC_DIR`` or
# default ``~/.cloudxr/static-client`` (downloaded from NVIDIA GitHub Pages if missing).
# The WSS proxy serves these files from ``/client/`` on its existing TLS port.
# USB-local reaches that same origin through the PROXY_PORT reverse rule.
# ---------------------------------------------------------------------------

USB_HOST = "127.0.0.1"  # serverIP seen by the headset (its own localhost)
USB_BACKEND_DEFAULT_PORT = 49100  # CloudXR backend (webrtc client direct connection)
USB_TURN_DEFAULT_PORT = 3478  # coturn TURN server port (adb reverse'd to headset)
USB_TURN_USER = "cloudxr"  # TURN username
USB_TURN_CREDENTIAL = "cloudxrpass"  # TURN credential


def default_web_client_static_dir() -> Path:
    """Default directory for web client static files (under ``~/.cloudxr``)."""
    return Path.home() / ".cloudxr" / "static-client"


def resolve_web_client_static_dir() -> Path:
    """Directory for web client assets: :envvar:`TELEOP_WEB_CLIENT_STATIC_DIR` or default.

    Raises:
        RuntimeError: If ``TELEOP_WEB_CLIENT_STATIC_DIR`` points at a non-directory path.
    """
    raw = os.environ.get(TELEOP_WEB_CLIENT_STATIC_DIR_ENV, "").strip()
    if raw:
        p = Path(os.path.expanduser(raw)).resolve()
        if p.exists() and not p.is_dir():
            raise RuntimeError(
                f"{TELEOP_WEB_CLIENT_STATIC_DIR_ENV} is not a directory: {p}"
            )
        return p
    return default_web_client_static_dir()


def _fetch_url_bytes(url: str, *, timeout: float = 120.0) -> bytes:
    """Download *url* into memory (WebXR client assets only; size-capped)."""
    req = Request(url, headers={"User-Agent": "isaacteleop-cloudxr"})
    try:
        with urlopen(req, timeout=timeout) as resp:
            cl = resp.headers.get("Content-Length")
            if cl is not None:
                try:
                    n = int(cl)
                except ValueError:
                    pass
                else:
                    if n > _USB_LOCAL_ASSET_MAX_BYTES:
                        raise RuntimeError(
                            f"Refusing download larger than {_USB_LOCAL_ASSET_MAX_BYTES} bytes "
                            f"(Content-Length={n}): {url}"
                        )
            data = resp.read(_USB_LOCAL_ASSET_MAX_BYTES + 1)
    except URLError as exc:
        raise RuntimeError(f"Could not download {url}: {exc}") from exc
    if len(data) > _USB_LOCAL_ASSET_MAX_BYTES:
        raise RuntimeError(
            f"Download exceeded {_USB_LOCAL_ASSET_MAX_BYTES} bytes (no trusted Content-Length): {url}"
        )
    return data


def _write_atomic_bytes(dest: Path, data: bytes) -> None:
    """Write *data* to *dest* atomically via a ``.part`` temp file."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    try:
        tmp.write_bytes(data)
        tmp.replace(dest)
    except OSError:
        if tmp.is_file():
            try:
                tmp.unlink()
            except OSError:
                pass
        raise


_REQUIRED_WEB_CLIENT_ASSETS = ("index.html", "bundle.js")
# Hardcoded — any new async chunk emitted by webpack must be added here manually.
_OPTIONAL_WEB_CLIENT_ASSETS = ("bundle.emulator.js",)


def require_web_client_static_dir(*, require_health_probe: bool = False) -> Path:
    """Ensure web client static assets exist under :func:`resolve_web_client_static_dir`.

    Creates the directory if needed. If ``index.html``, ``bundle.js``, or
    ``bundle.emulator.js`` is missing or empty, downloads from the published
    Isaac Teleop client URLs (emulator bundle is optional on older releases).

    Idempotent: safe to call from both :class:`~.launcher.CloudXRLauncher` and ``wss.run``
    (second call skips network when files are present). OOB automation requires
    the current browser health protocol; a non-empty older cache is not a
    compatible substitute.

    Raises:
        RuntimeError: If the path is invalid or downloads/final validation fail.
    """
    p = resolve_web_client_static_dir()

    try:
        p.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise RuntimeError(
            f"Cannot create web client static directory {p}: {exc}"
        ) from exc

    client_origin = default_web_client_origin()
    assets = [
        (name, urljoin(client_origin, name))
        for name in (*_REQUIRED_WEB_CLIENT_ASSETS, *_OPTIONAL_WEB_CLIENT_ASSETS)
    ]
    for name, url in assets:
        dest = p / name
        if dest.is_file() and dest.stat().st_size > 0:
            continue
        log.info("web client: fetching %s → %s", url, dest)
        try:
            data = _fetch_url_bytes(url)
        except RuntimeError as exc:
            if name in _OPTIONAL_WEB_CLIENT_ASSETS:
                log.warning(
                    "web client: optional asset %s not available: %s", name, exc
                )
                continue
            raise
        if not data:
            raise RuntimeError(f"Downloaded empty body from {url}")
        try:
            _write_atomic_bytes(dest, data)
        except OSError as exc:
            raise RuntimeError(f"Failed to write {dest}: {exc}") from exc

    for name in _REQUIRED_WEB_CLIENT_ASSETS:
        fp = p / name
        if not fp.is_file() or fp.stat().st_size == 0:
            raise RuntimeError(f"Web client file missing or empty after fetch: {fp}")
    if require_health_probe:
        bundle = p / "bundle.js"
        try:
            bundle_bytes = bundle.read_bytes()
        except OSError as exc:
            raise RuntimeError(f"Cannot read WebXR bundle {bundle}: {exc}") from exc
        if b"healthProbe" not in bundle_bytes or b"healthReport" not in bundle_bytes:
            raise RuntimeError(
                f"WebXR bundle {bundle} lacks the OOB healthProbe/healthReport "
                "protocol. Build deps/cloudxr/webxr_client with `npm run build` "
                "and set TELEOP_WEB_CLIENT_STATIC_DIR to its build directory. "
                "An older non-empty static cache is not replaced automatically."
            )
    return p


def web_client_base_override_from_env() -> str | None:
    """Return the stripped ``TELEOP_WEB_CLIENT_BASE`` value, or ``None`` if unset/blank."""
    v = os.environ.get(TELEOP_WEB_CLIENT_BASE_ENV, "").strip()
    return v or None


def parse_env_port(env_var: str, raw: str) -> int:
    """Parse and validate a port string from an environment variable."""
    try:
        port = int(raw)
    except ValueError:
        raise ValueError(
            f"{env_var}={raw!r} is not a valid integer; "
            f"set it to a port number (1–65535) or unset it to use the default."
        ) from None
    if not 1 <= port <= 65535:
        raise ValueError(f"{env_var}={port} is out of range; must be 1–65535.")
    return port


def print_hosted_client_line(
    url: str,
    *,
    prefix: str = "web client:        ",
    file=None,
) -> None:
    """Print ``prefix`` + *url* (cyan when *file* is a TTY)."""
    out = sys.stdout if file is None else file
    use_color = bool(getattr(out, "isatty", lambda: False)())
    left = f"{prefix}\033[36m{url}\033[0m" if use_color else f"{prefix}{url}"
    print(left, file=out)


def wss_proxy_port() -> int:
    """TCP port for the WSS proxy (``PROXY_PORT`` environment variable if set, else ``48322``)."""
    raw = os.environ.get("PROXY_PORT", "").strip()
    if raw:
        return parse_env_port("PROXY_PORT", raw)
    return WSS_PROXY_DEFAULT_PORT


def usb_backend_port() -> int:
    """TCP port for the USB-local CloudXR backend (native client direct connection).

    Reads the ``USB_BACKEND_PORT`` environment variable if set, else falls
    back to :data:`USB_BACKEND_DEFAULT_PORT` (49100).  This port is exposed
    to the headset via ``adb reverse``; override only when a host process
    already owns 49100.
    """
    raw = os.environ.get("USB_BACKEND_PORT", "").strip()
    if raw:
        return parse_env_port("USB_BACKEND_PORT", raw)
    return USB_BACKEND_DEFAULT_PORT


def usb_turn_port() -> int:
    """TCP/UDP port for the USB-local coturn TURN server.

    Reads the ``USB_TURN_PORT`` environment variable if set, else falls back
    to :data:`USB_TURN_DEFAULT_PORT` (3478).  coturn binds this on
    127.0.0.1 and ``adb reverse`` exposes it to the headset for WebRTC
    ICE relay.  Override only when 3478 is occupied (e.g. by a system
    coturn that wasn't masked).
    """
    raw = os.environ.get("USB_TURN_PORT", "").strip()
    if raw:
        return parse_env_port("USB_TURN_PORT", raw)
    return USB_TURN_DEFAULT_PORT


def guess_lan_ipv4() -> str | None:
    """Best-effort LAN IPv4 for operator URLs when headsets reach the PC by IP."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.settimeout(0.25)
            s.connect(("192.0.2.1", 1))
            addr, _ = s.getsockname()
    except OSError:
        return None
    if not addr or addr == "127.0.0.1":
        return None
    return addr


def default_initial_stream_config(resolved_proxy_port: int) -> dict:
    """Default hub stream config from env and LAN guess (same host as proxy port by default)."""
    env_ip = os.environ.get("TELEOP_STREAM_SERVER_IP", "").strip()
    env_port = os.environ.get("TELEOP_STREAM_PORT", "").strip()
    server_ip = env_ip or guess_lan_ipv4() or "127.0.0.1"
    port = (
        parse_env_port("TELEOP_STREAM_PORT", env_port)
        if env_port
        else resolved_proxy_port
    )
    return {"serverIP": server_ip, "port": port}


def client_ui_fields_from_env() -> dict:
    """Optional WebXR client UI defaults merged into hub ``config`` and bookmarks.

    Keys match query params the WebXR client reads on page load, including
    codec, panel visibility, and bounded stream-reconnect policy.
    """
    out: dict = {}
    out.update(client_reconnect_config_from_env())
    codec = os.environ.get("TELEOP_CLIENT_CODEC", "").strip()
    if codec:
        out["codec"] = codec
    ph = os.environ.get("TELEOP_CLIENT_PANEL_HIDDEN_AT_START", "").strip().lower()
    if ph in ("1", "true", "yes", "on"):
        out["panelHiddenAtStart"] = True
    elif ph in ("0", "false", "no", "off"):
        out["panelHiddenAtStart"] = False
    return out


def teleop_client_route_from_env() -> str:
    """Return the HashRouter fragment to append to the headset bookmark URL.

    Default: :data:`DEFAULT_TELEOP_CLIENT_ROUTE` (empty — no fragment, the
    WebXR client picks its own landing route). Set ``TELEOP_CLIENT_ROUTE``
    to a path like ``/real/gear/dexmate`` to force a specific route; an
    explicit empty value also suppresses the fragment. A leading ``#`` in
    the override is stripped (the URL builder always emits exactly one).
    Returns ``""`` to mean "no fragment".
    """
    raw = os.environ.get(TELEOP_CLIENT_ROUTE_ENV)
    if raw is None:
        return DEFAULT_TELEOP_CLIENT_ROUTE
    val = raw.strip()
    if not val:
        return ""
    return val.lstrip("#")


def build_headset_bookmark_url(
    *,
    web_client_base: str,
    stream_config: dict | None = None,
    control_token: str | None = None,
    oob_enable: bool = True,
) -> str:
    """Full WebXR page URL with stream query params, and OOB params when enabled.

    The client derives ``wss://{serverIP}:{port}/oob/v1/ws`` from ``serverIP`` + ``port`` in the query
    when ``oobEnable=1``.

    Set *oob_enable* to ``False`` to omit ``oobEnable`` (and the control token,
    which only authenticates hub operations).  The remaining params —
    ``serverIP``, ``port``, ``codec``, ``panelHiddenAtStart`` — are plain form
    overrides the client honours either way, so the headset still lands with
    the right streaming target pre-filled.  Callers must not disable OOB while
    the control hub is down: without a hub, ``/oob/v1/ws`` is proxied to the
    CloudXR streaming backend rather than refused.

    A HashRouter fragment is appended at the end when ``TELEOP_CLIENT_ROUTE``
    is set (e.g. ``#/real/gear/dexmate``); by default no fragment is added
    and the WebXR client picks its own landing route.
    """
    cfg = stream_config or {}
    if not cfg.get("serverIP") or cfg.get("port") is None:
        raise ValueError(
            "build_headset_bookmark_url requires stream_config with serverIP and port"
        )
    params: dict[str, str] = {}
    if oob_enable:
        params["oobEnable"] = "1"
        if control_token:
            params["controlToken"] = control_token
    params["serverIP"] = str(cfg["serverIP"])
    params["port"] = str(int(cfg["port"]))
    v = cfg.get("codec")
    if v is not None and str(v).strip() != "":
        params["codec"] = str(v).strip()
    v = cfg.get("panelHiddenAtStart")
    if isinstance(v, bool):
        params["panelHiddenAtStart"] = "true" if v else "false"
    v = cfg.get("reconnectEnabled")
    if isinstance(v, bool):
        params["reconnectEnabled"] = "true" if v else "false"
    for key in ("reconnectMaxAttempts", "reconnectDelayMs"):
        v = cfg.get(key)
        if isinstance(v, int) and not isinstance(v, bool) and v >= 0:
            params[key] = str(v)
    v = cfg.get("turnServer")
    if v is not None and str(v).strip() != "":
        params["turnServer"] = str(v).strip()
    v = cfg.get("turnUsername")
    if v is not None and str(v).strip() != "":
        params["turnUsername"] = str(v).strip()
    v = cfg.get("turnCredential")
    if v is not None and str(v).strip() != "":
        params["turnCredential"] = str(v).strip()
    if cfg.get("iceRelayOnly"):
        params["iceRelayOnly"] = "1"
    q = urlencode(params)
    base = web_client_base.rstrip("/")
    sep = "&" if "?" in base else "?"
    url = f"{base}{sep}{q}"
    route = teleop_client_route_from_env()
    if route:
        url = f"{url}#{route}"
    return url


def redact_control_token(text: str) -> str:
    """Mask ``controlToken`` and ``token`` query values in display or log text.

    Shared by every path that surfaces a credential-bearing URL, including
    the OOB WSS failure log and service status diagnostic.
    """
    return re.sub(
        r"(?<![\w])((?:controlToken|token)=)[^&\s'\"#]+",
        r"\1<REDACTED>",
        text,
        flags=re.IGNORECASE,
    )


def resolve_lan_host_for_oob() -> str:
    """PC LAN address the headset uses for ``wss://…:PROXY_PORT`` over WiFi."""
    h = os.environ.get("TELEOP_PROXY_HOST", "").strip() or guess_lan_ipv4()
    if not h:
        raise RuntimeError(
            "--setup-oob needs this PC's LAN IP for WebXR/WSS URLs. "
            "Set TELEOP_PROXY_HOST to an address the headset can reach over WiFi "
            "(or fix routing so guess_lan_ipv4() works)."
        )
    return h


def oob_progress(stage: str, msg: str) -> None:
    """One-line progress update for ``--setup-oob`` / ``--usb-local`` steps.

    Goes to stderr in dim cyan so the operator can see *where* the launcher
    is in its sequence of steps without these lines competing with the
    success banner (stdout) or error prints (red).

    This stays a print so progress remains visible when the console log level
    is set to warning.
    """
    print(f"\033[36m[{stage}]\033[0m {msg}", file=sys.stderr, flush=True)


def print_oob_hub_startup_banner(
    *,
    lan_host: str | None = None,
    usb_local: bool = False,
    web_client_base: str | None = None,
) -> None:
    """Print operator instructions for OOB + USB adb automation.

    Args:
        lan_host: PC LAN address (WiFi mode) or ``"127.0.0.1"`` (USB-local mode).
        usb_local: When ``True``, adjust the banner to describe the USB-local
            topology: everything reachable from the headset via ``adb reverse``
            on loopback; WebXR UI from ``TELEOP_WEB_CLIENT_STATIC_DIR`` (HTTPS, same PEM as WSS).
        web_client_base: Override the WebXR client base URL in the bookmark.
            When ``None`` (default), uses the versioned GitHub Pages client
            (WiFi mode) or the USB-local HTTPS origin (USB-local mode).
            ``TELEOP_WEB_CLIENT_BASE`` env var still takes precedence over this.
    """
    port = wss_proxy_port()
    backend_port = usb_backend_port()
    turn_port = usb_turn_port()
    token = os.environ.get("CONTROL_TOKEN") or None

    if not lan_host:
        lan_host = resolve_lan_host_for_oob() if not usb_local else "127.0.0.1"
    primary_host = lan_host

    if usb_local:
        web_base = (
            os.environ.get("TELEOP_WEB_CLIENT_BASE", "").strip()
            or f"https://localhost:{port}/client/"
        )
    elif web_client_base is not None:
        web_base = web_client_base
    else:
        web_base = default_web_client_origin()

    stream_cfg: dict = {"serverIP": primary_host, "port": port}
    if usb_local:
        # No mediaAddress: it's a NAT-override that bypasses ICE and would
        # short-circuit the TURN-relayed media path. Let the SDK discover
        # the media endpoint through ICE via coturn.
        stream_cfg["turnServer"] = f"turn:{USB_HOST}:{turn_port}?transport=tcp"
        stream_cfg["turnUsername"] = USB_TURN_USER
        stream_cfg["turnCredential"] = USB_TURN_CREDENTIAL
        stream_cfg["iceRelayOnly"] = True

    web_client_base_override = web_client_base_override_from_env()
    if web_client_base_override:
        web_base = web_client_base_override

    stream_cfg = {**stream_cfg, **client_ui_fields_from_env()}

    primary_base = f"https://{primary_host}:{port}"
    bookmark_display = build_headset_bookmark_url(
        web_client_base=web_base,
        stream_config=stream_cfg,
        control_token=None,
    )
    if token:
        bookmark_display = redact_control_token(
            f"{bookmark_display}&controlToken={token}"
        )
    wss_primary = f"wss://{primary_host}:{port}{OOB_WS_PATH}"

    bar = "=" * 72
    print(bar)
    if usb_local:
        print("OOB TELEOP (USB-local) — headset reaches PC on loopback via adb reverse")
    else:
        print(
            "OOB TELEOP — enabled (out-of-band control hub is running in this WSS proxy)"
        )
    print(bar)
    print()
    print(
        f"  The hub shares the CloudXR proxy TLS port {port} on this machine "
        f"(control WebSocket: {wss_primary})."
    )
    if usb_local:
        print(
            "  USB-local mode: adb reverse active for ports "
            f"{port}/tcp (WSS + WebXR UI), "
            f"{backend_port}/tcp (backend), "
            f"{turn_port}/tcp (TURN relay — coturn)."
        )
        print(
            "  The launcher serves the WebXR UI on the WSS proxy and starts coturn automatically "
            "(see coturn-cloudxr-3478.log if CONNECT fails)."
        )
    else:
        print(
            "  Same steps as docs: references/oob_teleop_control.rst — "
            '"End-to-end workflow (the usual path)".'
        )
    print()
    if usb_local:
        print("  adb: USB cable required — headset reaches this PC via adb reverse.")
    else:
        print(
            "  adb: USB cable — headset connected via USB for adb; streaming and web page over WiFi."
        )
    print()
    print("  Step 1 — Open teleop page on headset (adb)")
    print(
        '           After "WSS proxy listening on port …", `--setup-oob` runs '
        "`adb` to open the page on the headset. If that fails, open this URL manually:"
    )
    print(f"           {bookmark_display}")
    if web_client_base_override:
        print(
            f"           ({TELEOP_WEB_CLIENT_BASE_ENV} overrides the WebXR origin; "
            "query still targets this streaming host.)"
        )
    route = teleop_client_route_from_env()
    route_src = (
        f"from {TELEOP_CLIENT_ROUTE_ENV}"
        if os.environ.get(TELEOP_CLIENT_ROUTE_ENV) is not None
        else "default"
    )
    if route:
        print(
            f"           Client route: \033[36m#{route}\033[0m  "
            f"({route_src}; change via {TELEOP_CLIENT_ROUTE_ENV}, "
            f"set empty to suppress)"
        )
    else:
        print(
            f"           Client route: \033[36m<none>\033[0m  "
            f"(default; set {TELEOP_CLIENT_ROUTE_ENV}=/real/gear/dexmate "
            "to land on a specific route)"
        )
    print()
    print("  Step 2 — Accept cert + click CONNECT (CDP automation)")
    print("           CDP automation will accept the self-signed certificate and click")
    print("           CONNECT automatically via Chrome DevTools Protocol.")
    print(
        "           If it fails, fall back to manual: open "
        f"{CHROME_INSPECT_DEVICES_URL},"
    )
    print("           inspect the headset tab, and click CONNECT in DevTools.")
    print()
    print("-" * 72)
    print("OOB HTTP (optional — operators / curl / scripts on this PC)")
    print("-" * 72)
    cfg_q = urlencode(
        {
            "serverIP": str(stream_cfg["serverIP"]),
            "port": str(int(stream_cfg["port"])),
        }
    )
    print(f"  State:  {primary_base}/api/oob/v1/state")
    print(f"  Config: {primary_base}/api/oob/v1/config?{cfg_q}")
    if token:
        print()
        print(
            "  CONTROL_TOKEN is set: add ?token=... or header X-Control-Token on OOB HTTP requests."
        )
    print(bar)
    print()


# ---------------------------------------------------------------------------
# Host preflight: best-effort ufw + port-bindability warnings.
# Never fatal — operators routinely run with ufw inactive or with non-default
# ports. Goal is to surface "you forgot ufw allow" before the headset times
# out, with the exact remediation command.
# ---------------------------------------------------------------------------


def _ufw_unallowed_ports(ports: list[int]) -> list[int] | None:
    """Return the subset of *ports* that ufw appears not to allow.

    ``None`` if ufw is unavailable or inactive (caller should skip the warning).
    """
    try:
        proc = subprocess.run(
            ["ufw", "status"],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    out = proc.stdout or ""
    if "inactive" in out.lower() or "Status: active" not in out:
        return None
    return [
        p
        for p in ports
        if not re.search(rf"^\s*{p}(?:/(?:tcp|udp))?\b.*ALLOW", out, re.MULTILINE)
    ]


def _port_in_use(port: int, host: str) -> bool:
    """True if a TCP listener already owns ``(host, port)``."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind((host, port))
        except OSError:
            return True
    return False


def ss_listeners_on_port(port: int) -> list[str]:
    """Return one trimmed ``ss -tulpn`` line per TCP/UDP listener on *port*.

    Why: a wildcard listener (``0.0.0.0:port``) can coexist with a loopback
    bind via ``SO_REUSEADDR``, so :func:`_port_in_use` misses it. Parsing
    ``ss`` output catches every listener regardless of bound address.
    Empty list when ``ss`` is unavailable (minimal containers, BSD, etc.) —
    callers must treat that as "couldn't ask", not "definitely free".
    """
    try:
        proc = subprocess.run(
            ["ss", "-tulpn"],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []
    if proc.returncode != 0:
        return []
    out: list[str] = []
    for line in (proc.stdout or "").splitlines():
        cols = line.split()
        if len(cols) < 5:
            continue
        # Local-Address:Port is column 5 (index 4); bracketed IPv6 like ``[::]:3478``
        # also ends in ``:<port>``.
        if cols[4].endswith(f":{port}"):
            out.append(line.strip())
    return out


def print_host_preflight_warnings(*, usb_local: bool) -> None:
    """Best-effort host preflight (port conflicts + ufw).

    In ``--usb-local`` mode this is fail-fast: a port conflict on any of
    the three required loopback ports (WSS / backend / TURN) raises
    :class:`RuntimeError` so the launcher exits before sinking time into
    a setup that can't possibly stream. In WiFi mode the same port-busy
    case stays warn-only — the WSS bind will fail loudly on its own with
    ``EADDRINUSE`` and there's no usb-local-style multi-port surface.

    ufw findings are always warn-only (best-effort: ``status`` parsing
    can yield false positives on permissive rulesets), and ufw is only
    checked in WiFi mode (USB-local is loopback, not firewalled).

    The WSS proxy binds the wildcard address (``host=""`` in
    :mod:`websockets`, i.e. ``0.0.0.0``), but we probe via loopback
    here: a holder that already owns the wildcard will still cause our
    loopback probe to fail with ``EADDRINUSE`` on Linux, so coverage of
    real conflicts is preserved while keeping the literal ``0.0.0.0``
    out of the source (security-scanner false positive about binding to
    all interfaces). The probe socket is closed immediately on success
    — nothing is actually exposed.
    """
    if usb_local:
        targets: list[tuple[int, str]] = [
            (wss_proxy_port(), "127.0.0.1"),
            (usb_backend_port(), "127.0.0.1"),
            (usb_turn_port(), "127.0.0.1"),
        ]
    else:
        targets = [(wss_proxy_port(), "127.0.0.1")]
    busy = [p for p, host in targets if _port_in_use(p, host)]
    if busy:
        ports_re = "|".join(f":{p}" for p in busy)
        if usb_local:
            log.error("preflight: port(s) already in use: %s", busy)
            raise RuntimeError(
                f"USB-local: required port(s) {busy} already in use — cannot proceed.\n"
                f"Kill the holder (`ss -tulpn | grep -E '{ports_re}'`) or override "
                "via PROXY_PORT / USB_BACKEND_PORT / USB_TURN_PORT, "
                "then retry."
            )
        log.warning("preflight: port(s) already in use: %s", busy)
        print(
            f"\n\033[33m[preflight] port(s) {busy} already in use; kill the "
            f"holder (`ss -tulpn | grep -E '{ports_re}'`) or override via "
            "PROXY_PORT.\033[0m\n"
        )
    # ufw is only meaningful in WiFi mode (USB-local is loopback, not firewalled).
    if not usb_local:
        unallowed = _ufw_unallowed_ports([wss_proxy_port()])
        if unallowed:
            cmd = "; ".join(f"sudo ufw allow {p}/tcp" for p in unallowed)
            log.warning("preflight: ufw blocks port(s) %s", unallowed)
            print(
                f"\n\033[33m[preflight] ufw is active and may block port(s) "
                f"{unallowed} from reaching the headset. Allow with: {cmd}"
                "\033[0m\n"
            )
