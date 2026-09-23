# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Open the Isaac Teleop WebXR client on a USB-connected headset.

``python -m isaaccapture.cloudxr.webclient [CLIENT_URL]``

Re-opens the teleop page on the headset without restarting the launcher —
useful when the headset browser was closed, navigated away, or needs to point
at a different client build.

URL building, adb preflight, vendor-browser selection and token redaction are
all reused from :mod:`~.oob_teleop_adb` / :mod:`~.oob_teleop_env`; this module
only adds argument handling and the summary block.

Scope is deliberately narrow: it opens a page over WiFi with this host's
streaming target pre-filled.  Out-of-band control, USB-local and host-client
topologies stay with the launcher, which owns the servers they need — pass a
URL explicitly to target one of them.
"""

from __future__ import annotations

import argparse
import os
import sys
from urllib.parse import parse_qs, urlsplit

from .oob_teleop_adb import (
    OobAdbError,
    SELECTED_ADB_SERIAL,
    adb_automation_failure_hint,
    assert_exactly_one_adb_device,
    assert_headset_awake,
    build_teleop_url,
    headset_browser_package,
    oob_adb_automation_message,
    open_url_on_headset,
    require_adb_on_path,
)
from .oob_teleop_env import (
    TELEOP_CLIENT_ROUTE_ENV,
    redact_control_token,
    teleop_client_route_from_env,
    wss_proxy_port,
)

# Query param the WebXR client keys off for out-of-band control. Its presence
# in a user-supplied URL means that URL is already a full bookmark.
_OOB_MARKER = "oobEnable="


def resolve_client_url(override: str | None) -> tuple[str, str]:
    """Return ``(url, source)`` for the page to open on the headset.

    *source* is a short human-readable label for the summary block.

    Three cases:

    * *override* is ``None`` — the computed default: the versioned client with
      ``serverIP`` / ``port`` pre-filled.
    * *override* already contains ``oobEnable=`` — opened verbatim.  This is
      the paste-back case: copy the URL the launcher's banner printed, tweak a
      param, re-open it.  Appending our own params would duplicate the
      existing ones, and it is how you reach an OOB / USB-local / host-client
      session from here.
    * *override* is anything else — treated as the WebXR client *base*, with
      the query params appended.  Same semantics as
      :envvar:`TELEOP_WEB_CLIENT_BASE`, which the argument outranks.

    The computed URL never sets ``oobEnable``: the control hub only exists
    when the launcher runs with ``--setup-oob``, and without one the client's
    control WebSocket would reach the CloudXR streaming backend instead.
    """
    if override is not None and _OOB_MARKER in override:
        return override, "verbatim"

    url = build_teleop_url(
        resolved_port=wss_proxy_port(),
        web_client_base=override,
        oob_enable=False,
    )
    return url, ("client base override" if override else "computed default")


def stream_target(url: str) -> str:
    """``serverIP:port`` parsed out of *url*, or ``"<unknown>"``.

    Read back from the URL rather than recomputed, so the summary always
    describes the page actually being opened — a verbatim URL may well point
    somewhere the local defaults would not.
    """
    query = parse_qs(urlsplit(url).query)
    server_ip = (query.get("serverIP") or [""])[0]
    port = (query.get("port") or [""])[0]
    return f"{server_ip}:{port}" if server_ip and port else "<unknown>"


def _paint(out, code: str, text: str) -> str:
    """Wrap *text* in ANSI *code*, or return it bare for non-TTY / ``NO_COLOR``."""
    if os.environ.get("NO_COLOR") or not getattr(out, "isatty", lambda: False)():
        return text
    return f"\033[{code}m{text}\033[0m"


def _field(out, label: str, value: str, note: str = "") -> str:
    """Format one aligned ``label   value  (note)`` row of the summary block."""
    row = f"  {_paint(out, '90', label.ljust(7))} {value}"
    return f"{row}  {_paint(out, '90', f'({note})')}" if note else row


def print_summary(*, url: str, source: str, probe_device: bool, stream=None) -> None:
    """Print the resolved-configuration block and the URL.

    *probe_device* controls whether the browser row is filled in from adb;
    ``--print-only`` skips it so the command works with no headset attached.
    """
    out = stream if stream is not None else sys.stdout
    bar = "━" * 64

    print(bar, file=out)
    print(f" {_paint(out, '1', 'Isaac Teleop')} · web client → headset", file=out)
    print(bar, file=out)
    print(file=out)

    if probe_device:
        # Which browser matters: WebLayer accepts requestSession() but does not
        # fully plumb controller input through to the client.
        print(
            _field(out, "browser", headset_browser_package() or "system default"),
            file=out,
        )

    # Parsed from the URL, so a verbatim URL reports its own target.
    print(_field(out, "stream", stream_target(url)), file=out)

    # Only worth a row when set — the client picks its own landing route otherwise.
    route = teleop_client_route_from_env()
    if route:
        print(_field(out, "route", f"#{route}", TELEOP_CLIENT_ROUTE_ENV), file=out)

    print(_field(out, "source", source), file=out)
    print(file=out)
    # Unwrapped and on its own line so terminal double-click / drag copies it whole.
    print(f"  {_paint(out, '36', redact_control_token(url))}", file=out)
    print(file=out)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the web client opener."""
    parser = argparse.ArgumentParser(
        prog="python -m isaaccapture.cloudxr.webclient",
        description=(
            "Open the Isaac Teleop WebXR client on a USB-connected headset. "
            "With no argument, opens the versioned client with this host's "
            "streaming target pre-filled."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "CLIENT_URL handling:\n"
            "  omitted                     the computed default URL\n"
            "  contains 'oobEnable='       opened verbatim (paste back a printed URL)\n"
            "  anything else               used as the client base; params appended\n"
            "                              (outranks TELEOP_WEB_CLIENT_BASE)\n"
            "\n"
            "To open an OOB, USB-local or host-client session, pass the URL the\n"
            "launcher printed for it.\n"
        ),
    )
    parser.add_argument(
        "client_url",
        nargs="?",
        default=None,
        metavar="CLIENT_URL",
        help="Optional WebXR client URL, or client base URL, to open.",
    )
    parser.add_argument(
        "--print-only",
        action="store_true",
        help="Resolve and print the URL without touching adb (no headset required).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Resolve the client URL and open it on the headset. Returns an exit code."""
    args = _parse_args(argv)
    token = None

    try:
        url, source = resolve_client_url(args.client_url)
        # Preflight before the summary so its browser row describes a headset
        # we have actually validated, and so failures surface without a banner.
        if not args.print_only:
            require_adb_on_path()
            token = SELECTED_ADB_SERIAL.set(assert_exactly_one_adb_device())
            assert_headset_awake()
    except (OobAdbError, RuntimeError, ValueError) as exc:
        if token is not None:
            SELECTED_ADB_SERIAL.reset(token)
        print(f"\n{exc}\n", file=sys.stderr)
        return 1

    try:
        print_summary(url=url, source=source, probe_device=not args.print_only)

        if args.print_only:
            return 0

        rc, diag = open_url_on_headset(url)
        if rc != 0:
            print(f"  {_paint(sys.stdout, '31', '✖')} adb failed\n")
            print(
                oob_adb_automation_message(rc, diag, adb_automation_failure_hint(diag)),
                file=sys.stderr,
            )
            return 1

        print(f"  {_paint(sys.stdout, '32', '✔')} opened on headset\n")
        return 0
    finally:
        if token is not None:
            SELECTED_ADB_SERIAL.reset(token)


if __name__ == "__main__":
    raise SystemExit(main())
