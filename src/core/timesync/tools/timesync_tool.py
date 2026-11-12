#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Runs and deploys the chrony setup that the timesync library reads: the camera host serves its clock and the PC measures it.

    timesync_tool.py ensure-local
    timesync_tool.py configure-remote user@host --host-lan-ip IP [--timeout S] [--dry-run]
    timesync_tool.py follow --server HOST [--cmd-port N]
    timesync_tool.py status --remote HOST [--cmd-port N] [--wait S]

``ensure-local`` installs chrony on this machine if its ``chronyd`` is missing (needs sudo). ``configure-remote`` installs
chrony on the camera host if it is missing, lets this machine sync from it and read its state, and makes sure it is
running (it needs passwordless sudo and the network there). ``follow`` runs a second, unprivileged chronyd on this
machine in the foreground until interrupted: it measures the camera host's clock and never adjusts the system clock, so
it needs no root, leaves the system chronyd alone and needs no inbound path to this machine. ``status`` is read-only: it
succeeds when that instance has selected the camera host and is settled. POSIX only.
"""

from __future__ import annotations

import argparse
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

DEFAULT_CMD_PORT = 12301
DROP_IN = "/etc/chrony/conf.d/camera-agx.conf"
TRACKING_FIELDS = 14


class ToolError(Exception):
    """A failure with a message worth showing as is."""


@dataclass(frozen=True)
class Tracking:
    """One ``chronyc -c tracking`` line; seconds and ppm, as chrony prints them."""

    reference_id: str
    reference_name: str
    stratum: int
    ref_time: float
    system_time: float
    root_delay: float
    root_dispersion: float
    skew_ppm: float
    update_interval: float
    leap_status: str

    @property
    def synchronised(self) -> bool:
        return (
            self.leap_status in ("Normal", "Insert second", "Delete second")
            and 1 <= self.stratum <= 15
            and self.reference_id not in ("", "00000000")
        )


def parse_tracking(line: str) -> Tracking:
    """Parse one chrony 4.x CSV tracking line; anything unexpected is an error."""
    fields = line.strip().split(",")
    if len(fields) != TRACKING_FIELDS:
        raise ToolError(
            f"expected {TRACKING_FIELDS} tracking fields, got {len(fields)}: {line.strip()!r}"
        )
    try:
        return Tracking(
            reference_id=fields[0].strip().upper(),
            reference_name=fields[1].strip(),
            stratum=int(fields[2]),
            ref_time=float(fields[3]),
            system_time=float(fields[4]),
            root_delay=float(fields[10]),
            root_dispersion=float(fields[11]),
            skew_ppm=float(fields[9]),
            update_interval=float(fields[12]),
            leap_status=fields[13].strip(),
        )
    except ValueError as error:
        raise ToolError(
            f"unparseable tracking line {line.strip()!r}: {error}"
        ) from error


def reference_id_for(ip: str) -> str:
    """The reference id chrony shows for an NTP source at an IPv4 address: its four bytes in upper-case hex."""
    parts = ip.split(".")
    if len(parts) != 4 or not all(p.isdigit() and 0 <= int(p) <= 255 for p in parts):
        raise ToolError(f"{ip!r} is not an IPv4 address")
    return "".join(f"{int(p):02X}" for p in parts)


def lan_address_toward(host: str) -> str:
    """This machine's address on the route to ``host`` (a UDP connect sends nothing)."""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.connect((host, 9))
        return sock.getsockname()[0]


def follower_config(server: str, server_port: int, cmd_port: int, workdir: Path) -> str:
    """The unprivileged follower instance: it measures ``server``'s clock, serves no NTP and adjusts nothing."""
    return (
        f"server {server} port {server_port} iburst minpoll 0 maxpoll 3\n"
        "port 0\n"
        f"cmdport {cmd_port}\n"
        "bindcmdaddress 127.0.0.1\n"
        f"pidfile {workdir / 'chronyd.pid'}\n"
        f"driftfile {workdir / 'chronyd.drift'}\n"
        f"dumpdir {workdir}\n"
    )


def remote_drop_in(host_lan_ip: str) -> str:
    """The camera host's chrony drop-in: serve this PC and let it read this chronyd's state."""
    return f"allow {host_lan_ip}\nbindcmdaddress 0.0.0.0\ncmdallow {host_lan_ip}\n"


def remote_script(host_lan_ip: str) -> str:
    """The bash that ``configure-remote`` runs on the camera host."""
    wanted = remote_drop_in(host_lan_ip)
    return f"""set -eu
sudo -n true || {{ echo "passwordless sudo is required on the target" >&2; exit 1; }}
if ! command -v chronyd >/dev/null; then
    echo "chrony is not installed: installing it"
    sudo -n apt-get update -qq
    sudo -n env DEBIAN_FRONTEND=noninteractive apt-get install -y chrony
fi
grep -Eqs '^[[:space:]]*(confdir|include)[[:space:]].*conf\\.d' /etc/chrony/chrony.conf \\
    || {{ echo "/etc/chrony/chrony.conf does not include conf.d" >&2; exit 1; }}
wanted={shlex.quote(wanted.rstrip())}
current=$(sudo -n cat {DROP_IN} 2>/dev/null || true)
if [[ "$current" != "$wanted" ]]; then
    printf '%s\\n' "$wanted" | sudo -n tee {DROP_IN} >/dev/null
    echo "wrote {DROP_IN}; restarting chrony"
    sudo -n systemctl restart chrony
elif ! sudo -n systemctl is-active --quiet chrony; then
    echo "chrony is not running: starting it"
    sudo -n systemctl start chrony
else
    echo "{DROP_IN} is already as wanted"
fi
sudo -n systemctl enable chrony >/dev/null 2>&1 || true
"""


def chronyc_tracking(
    chronyc: str, host: str | None, port: int | None, timeout: float = 5.0
) -> Tracking:
    """Ask a chronyd for its tracking report: a remote one over UDP when ``host`` is given, else the system's."""
    command = [chronyc, "-c"]
    if host:
        command += ["-h", host]
    if port:
        command += ["-p", str(port)]
    command.append("tracking")
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=timeout, check=False
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise ToolError(f"{shlex.join(command)} failed: {error}") from error
    if result.returncode != 0:
        raise ToolError(
            f"{shlex.join(command)} exited {result.returncode}: {result.stdout.strip() or result.stderr.strip()}"
        )
    return parse_tracking(result.stdout)


def judge(local: Tracking, expected_id: str, max_dispersion_s: float) -> list[str]:
    """What keeps this machine's measuring instance from being usable, one message each; empty when it is usable."""
    problems = []
    if not local.synchronised:
        problems.append(
            f"this machine is not synchronised (stratum {local.stratum}, {local.leap_status})"
        )
    if local.reference_id != expected_id:
        problems.append(
            f"this machine follows {local.reference_id} ({local.reference_name or 'unnamed'}), "
            f"not the camera host ({expected_id})"
        )
    if local.root_dispersion > max_dispersion_s:
        problems.append(
            f"this machine's root dispersion is {local.root_dispersion * 1e3:.1f} ms, "
            f"above {max_dispersion_s * 1e3:.1f} ms; it is still settling"
        )
    return problems


def run_chronyd(args: argparse.Namespace, make_config, what: str) -> int:
    """Run an unprivileged, clock-neutral chronyd in the foreground until interrupted."""
    chronyd = args.chronyd or shutil.which("chronyd") or "/usr/sbin/chronyd"
    if not Path(chronyd).is_file():
        raise ToolError(f"{chronyd} not found; install chrony")
    workdir = (
        Path(args.workdir)
        if args.workdir
        else Path(tempfile.mkdtemp(prefix="timesync-"))
    )
    workdir.mkdir(parents=True, exist_ok=True)
    # A copy of the binary: the distribution's AppArmor profile is tied to /usr/sbin/chronyd and would stop it reading a
    # configuration outside /etc/chrony.
    binary = workdir / "chronyd"
    shutil.copy2(chronyd, binary)
    config = workdir / "chrony.conf"
    config.write_text(make_config(workdir))
    command = [str(binary), "-x", "-U", "-d", "-f", str(config)]
    print(what, shlex.join(command), flush=True)
    process = subprocess.Popen(command)

    def stop(*_: object) -> None:
        process.send_signal(signal.SIGINT)

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    return process.wait()


def cmd_follow(args: argparse.Namespace) -> int:
    reference_id_for(args.server)
    return run_chronyd(
        args,
        lambda workdir: follower_config(
            args.server, args.server_ntp_port, args.cmd_port, workdir
        ),
        f"measuring {args.server}'s clock:",
    )


def cmd_ensure_local(args: argparse.Namespace) -> int:
    """Install chrony here when ``chronyd`` is missing; the follower and the PC's chronyc need it."""
    if shutil.which("chronyd") or Path("/usr/sbin/chronyd").is_file():
        print("chrony is already installed")
        return 0
    print("chrony is not installed: installing it (sudo)", flush=True)
    for command in (
        ["sudo", "apt-get", "update", "-qq"],
        [
            "sudo",
            "env",
            "DEBIAN_FRONTEND=noninteractive",
            "apt-get",
            "install",
            "-y",
            "chrony",
        ],
    ):
        try:
            returncode = subprocess.run(
                command, check=False, timeout=args.timeout
            ).returncode
        except subprocess.TimeoutExpired as error:
            raise ToolError(
                f"{shlex.join(command)} did not finish within {args.timeout:.0f} s"
            ) from error
        if returncode != 0:
            raise ToolError(f"{shlex.join(command)} exited {returncode}")
    return 0


def cmd_configure_remote(args: argparse.Namespace) -> int:
    reference_id_for(args.host_lan_ip)
    script = remote_script(args.host_lan_ip)
    command = [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=10",
        args.target,
        "bash -s",
    ]
    if args.dry_run:
        print("DRY-RUN", shlex.join(command), "<<<")
        print(script)
        return 0
    try:
        return subprocess.run(
            command, input=script, text=True, check=False, timeout=args.timeout
        ).returncode
    except subprocess.TimeoutExpired as error:
        raise ToolError(
            f"configuring {args.target} did not finish within {args.timeout:.0f} s"
        ) from error


def cmd_status(args: argparse.Namespace) -> int:
    """Succeeds when this machine's measuring instance follows the camera host and has settled."""
    expected = reference_id_for(args.remote)
    deadline = time.monotonic() + args.wait
    while True:
        problems: list[str] = []
        try:
            local = chronyc_tracking(args.chronyc, "127.0.0.1", args.cmd_port)
        except ToolError as error:
            local = None
            problems.append(
                f"this machine's chronyd on command port {args.cmd_port} cannot be read: {error}"
            )
        try:
            remote = chronyc_tracking(args.chronyc, args.remote, None)
        except ToolError:
            remote = None
        if remote is None:
            problems.append(
                "the camera host's chronyd cannot be read (not installed, not running, or it does not cmdallow this machine)"
            )
        elif not remote.synchronised:
            problems.append(
                f"the camera host is not synchronised (stratum {remote.stratum}, {remote.leap_status})"
            )
        if local is not None:
            problems += judge(local, expected, args.max_dispersion_ms / 1e3)
        if not problems:
            assert local is not None
            print(
                f"ok: this machine follows {local.reference_id} (stratum {local.stratum}), "
                f"root delay {local.root_delay * 1e3:.3f} ms, root dispersion {local.root_dispersion * 1e3:.3f} ms, "
                f"System time {abs(local.system_time) * 1e6:.1f} us"
            )
            return 0
        if time.monotonic() >= deadline:
            for problem in problems:
                print("not ready:", problem, file=sys.stderr)
            return 1
        time.sleep(2.0)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    ensure = sub.add_parser(
        "ensure-local", help="install chrony on this machine if it is missing"
    )
    ensure.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="seconds each install step may take",
    )
    follow = sub.add_parser(
        "follow",
        help="run the unprivileged instance that measures the camera host's clock, in the foreground",
    )
    follow.add_argument(
        "--server", required=True, help="the camera host's IPv4 address"
    )
    follow.add_argument("--server-ntp-port", type=int, default=123)
    follow.add_argument("--cmd-port", type=int, default=DEFAULT_CMD_PORT)
    follow.add_argument("--workdir")
    follow.add_argument("--chronyd")
    configure = sub.add_parser(
        "configure-remote",
        help="install and start chrony on the camera host and let this machine measure it",
    )
    configure.add_argument("target", help="user@host of the camera host")
    configure.add_argument(
        "--host-lan-ip", required=True, help="this machine's address on the camera LAN"
    )
    configure.add_argument(
        "--timeout", type=float, default=90.0, help="seconds the whole setup may take"
    )
    configure.add_argument("--dry-run", action="store_true")
    status = sub.add_parser(
        "status",
        help="read-only: does this machine follow the camera host, and is it settled",
    )
    status.add_argument("--remote", required=True, help="the camera host's address")
    status.add_argument("--cmd-port", type=int, default=DEFAULT_CMD_PORT)
    status.add_argument(
        "--wait",
        type=float,
        default=0.0,
        help="keep checking this many seconds before giving up",
    )
    status.add_argument("--max-dispersion-ms", type=float, default=2.0)
    status.add_argument("--chronyc", default=shutil.which("chronyc") or "chronyc")
    args = parser.parse_args(argv)
    commands = {
        "ensure-local": cmd_ensure_local,
        "follow": cmd_follow,
        "configure-remote": cmd_configure_remote,
        "status": cmd_status,
    }
    try:
        return commands[args.command](args)
    except ToolError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
