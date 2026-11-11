#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build and try argus_sender on a Jetson from this checkout, without activating it.

Run on the desktop:

    deploy_sender.py user@orin check                       read-only: release, camera processes, disk
    deploy_sender.py user@orin discover                    read-only: platform files, tuning files, camera units
    deploy_sender.py user@orin deploy [--with-sipl] [--name NAME]
    deploy_sender.py user@orin run --name NAME --config FILE [--seconds N]
    deploy_sender.py user@orin stop                        stop the senders that `run` started
    deploy_sender.py user@orin linktest [--mb N]           TCP throughput from the target to this machine

`deploy` copies the committed source of argus_sender (git archive of HEAD) into a new release directory under
~/camera-streamer/releases/ on the target, configures and builds it there and runs the unit tests that need no camera.
It never stops, replaces or starts a service and never opens a camera; a release directory that already exists is
refused rather than overwritten. `run` starts a built release for a bounded time with a config from this machine and
opens the cameras; it prints a note, and does not stop anything, if another camera process is running.
"""

from __future__ import annotations

import argparse
import re
import shlex
import subprocess
import sys
from pathlib import Path

RELEASES = "~/camera-streamer/releases"
SIPL_ROOT = "/usr/src/jetson_sipl_api/sipl"
SOURCE_ROOT = "examples/camera_streamer/argus_sender"
NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


class DeployError(Exception):
    """A failure with a message worth showing as is."""


def ssh_command(target: str, remote: str) -> list[str]:
    return ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", target, remote]


def remote(target: str, script: str, dry_run: bool) -> None:
    """Run a bash script on the target; its output goes straight to this terminal."""
    command = ssh_command(target, "bash -lc " + shlex.quote(script))
    if dry_run:
        print("DRY-RUN", shlex.join(command))
        return
    result = subprocess.run(command, check=False)
    if result.returncode != 0:
        raise DeployError(f"remote command failed with exit code {result.returncode}")


def repo_root() -> Path:
    out = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
        check=False,
    )
    if out.returncode != 0:
        raise DeployError("run this inside the IsaacCapture checkout")
    return Path(out.stdout.strip())


def git_output(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args], capture_output=True, text=True, check=True
    ).stdout.strip()


def cmd_check(target: str, dry_run: bool) -> None:
    remote(
        target,
        r"""
set -u
echo "host: $(hostname)  kernel: $(uname -r)"
head -1 /etc/nv_tegra_release 2>/dev/null || echo "no /etc/nv_tegra_release: not a Jetson"
echo "--- processes that may own the cameras"
ps -eo pid,etime,cmd | grep -Ei 'nvargus|argus_sender|sipl|camera' | grep -v grep | cut -c1-160 || true
echo "--- SIPL and multimedia packages"
for path in /usr/src/jetson_sipl_api /usr/src/jetson_multimedia_api; do [ -d "$path" ] && echo "present: $path" || echo "missing: $path"; done
echo "--- disk"
df -h ~ | tail -1
""",
        dry_run,
    )


def cmd_discover(target: str, dry_run: bool) -> None:
    remote(
        target,
        r"""
set -u
echo "--- SIPL platform configuration candidates"
find /opt /usr/share /var/nvidia ~ -maxdepth 6 \( -iname 'shw5g*.json' -o -iname 'camera-platform*.json' \) 2>/dev/null | head -20
echo "--- NITO tuning files"
ls -la /var/nvidia/nvcam/settings/sipl 2>/dev/null | head -20 || echo "none"
echo "--- camera units (read-only)"
systemctl list-unit-files 2>/dev/null | grep -Ei 'camera|argus|sipl' || echo "none"
systemctl cat camera-argus-sender.service 2>/dev/null | head -30
echo "--- video devices and driver"
ls /dev/video* 2>/dev/null | head; lsmod 2>/dev/null | grep -Ei 'tegra_camera|nvhost_vi|max9296|max96712|sg8a' | head
""",
        dry_run,
    )


def cmd_linktest(target: str, megabytes: int, dry_run: bool) -> None:
    """Measure TCP throughput from the target to this machine, connecting the way the camera stream does.

    The target listens and sends; this machine connects and counts, like camera_receiver and argus_sender.
    """
    import socket
    import time

    port = 5611
    host = target.rsplit("@", 1)[-1]
    sender = (
        "import socket;l=socket.socket();l.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1);"
        "l.bind(('0.0.0.0',%d));l.listen(1);l.settimeout(20);c,_=l.accept();b=bytes(1<<20)\n"
        "for _ in range(%d): c.sendall(b)\n"
        "c.close()" % (port, megabytes)
    )
    command = ssh_command(target, f"python3 -c {shlex.quote(sender)}")
    if dry_run:
        print("DRY-RUN", shlex.join(command))
        return
    process = subprocess.Popen(command)
    try:
        connection = None
        for _ in range(40):
            try:
                connection = socket.create_connection((host, port), timeout=2)
                break
            except OSError:
                time.sleep(0.5)
        if connection is None:
            raise DeployError(f"could not connect to {host}:{port}")
        connection.settimeout(60)
        total, start = 0, time.monotonic()
        while True:
            chunk = connection.recv(1 << 20)
            if not chunk:
                break
            total += len(chunk)
        seconds = time.monotonic() - start
    finally:
        process.wait(timeout=30)
    print(
        f"{host} -> this machine: {total / 1e6:.1f} MB in {seconds:.2f} s = {total * 8 / seconds / 1e6:.1f} Mbit/s"
    )


def cmd_stop(target: str, dry_run: bool) -> None:
    """Ask the senders that `run` started (./build/argus_sender --config run-config.yaml) to stop (SIGINT, so they drain and free the cameras)."""
    remote(
        target,
        r"""
pids=$(pgrep -f "^\./build/argus_sender --config run-config.yaml" || true)
if [ -z "$pids" ]; then echo "no sender from ~/camera-streamer/releases is running"; exit 0; fi
echo "signalling: $pids"
kill -INT $pids
for _ in $(seq 1 20); do
    pgrep -f "^\./build/argus_sender --config run-config.yaml" >/dev/null || { echo "stopped"; exit 0; }
    sleep 1
done
echo "still running after 20 s; leaving it alone" >&2
exit 1
""",
        dry_run,
    )


def cmd_run(target: str, name: str, config: Path, seconds: int, dry_run: bool) -> None:
    """Run a deployed release for a bounded time, in the foreground, with a config from this machine."""
    if not NAME.fullmatch(name):
        raise DeployError(f"release name {name!r} is not a plain file name")
    if not config.is_file():
        raise DeployError(f"{config} is not a file")
    directory = f"{RELEASES}/{name}"
    guard = f"""
set -e
d={directory}
[ -x $d/build/argus_sender ] || {{ echo "no built sender in $d" >&2; exit 1; }}
# Taking the cameras is intended; say who else may hold them, but do not stop anything.
if systemctl is-active --quiet camera-argus-sender.service 2>/dev/null; then
    echo "note: camera-argus-sender.service is active and may hold the cameras" >&2
fi
if pgrep -f '/argus_sender' >/dev/null; then
    echo "note: another camera process is running" >&2
fi
"""
    upload = ssh_command(target, f"cat > {directory}/run-config.yaml")
    if dry_run:
        print("DRY-RUN", shlex.join(upload), "<", config)
    else:
        with config.open("rb") as handle:
            if subprocess.run(upload, stdin=handle, check=False).returncode != 0:
                raise DeployError("uploading the config failed")
    remote(
        target,
        guard
        + f"""
cd $d
# SIGINT first: the sender drains its encoder and returns every camera buffer on SIGINT/SIGTERM.
timeout --signal=INT --kill-after=15 {seconds} ./build/argus_sender --config run-config.yaml || echo "sender exited with status $?"
""",
        dry_run,
    )


def cmd_deploy(target: str, with_sipl: bool, name: str | None, dry_run: bool) -> None:
    root = repo_root()
    if git_output(root, "status", "--porcelain", "--", SOURCE_ROOT):
        raise DeployError(
            f"{SOURCE_ROOT} has uncommitted changes; commit them (the release is a git archive of HEAD)"
        )
    commit = git_output(root, "rev-parse", "--short=9", "HEAD")
    release = name or commit
    if not NAME.fullmatch(release):
        raise DeployError(f"release name {release!r} is not a plain file name")
    directory = f"{RELEASES}/{release}"

    print(f"==> [1/3] preparing {directory} on {target} (commit {commit})", flush=True)
    remote(
        target,
        f'set -e; d={directory}; [ ! -e $d ] || {{ echo "release $d already exists" >&2; exit 1; }}; mkdir -p $d/src',
        dry_run,
    )

    print("==> [2/3] copying the committed source", flush=True)
    archive = ["git", "-C", str(root), "archive", "--format=tar", f"HEAD:{SOURCE_ROOT}"]
    extract = ssh_command(target, f"tar -x -C {directory}/src")
    if dry_run:
        print("DRY-RUN", shlex.join(archive), "|", shlex.join(extract))
    else:
        producer = subprocess.Popen(archive, stdout=subprocess.PIPE)
        consumer = subprocess.run(extract, stdin=producer.stdout, check=False)
        producer.stdout.close()
        if producer.wait() != 0 or consumer.returncode != 0:
            raise DeployError("copying the source failed")

    print("==> [3/3] configuring, building and testing on the target", flush=True)
    options = "-DCMAKE_BUILD_TYPE=Release -DDISABLE_MULTIPROCESS=ON"
    if with_sipl:
        options += f" -DWITH_SIPL=ON -DSIPL_ROOT={SIPL_ROOT}"
    remote(
        target,
        f"""
set -e
cd {directory}
cmake -S src -B build {options} -DBUILD_TESTING=ON
cmake --build build -j "$(nproc)"
# Only the tests that need no camera; ctest does not open any device.
(cd build && ctest --output-on-failure) || echo "some tests failed (see above); the build itself succeeded"
./build/argus_sender --help >/dev/null 2>&1 || true
echo "built: {directory}/build/argus_sender  (not activated, no camera opened)"
""",
        dry_run,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("target", help="user@host of the Jetson")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("check", help="read-only look at the target")
    sub.add_parser("discover", help="read-only: SIPL platform files and existing units")
    sub.add_parser(
        "stop", help="stop the senders started from ~/camera-streamer/releases"
    )
    link = sub.add_parser("linktest", help="TCP throughput from the target to this PC")
    link.add_argument("--mb", type=int, default=32, help="megabytes to send")
    run = sub.add_parser("run", help="run a deployed release for a bounded time")
    run.add_argument("--name", required=True, help="release directory name")
    run.add_argument(
        "--config", required=True, type=Path, help="sender config to upload"
    )
    run.add_argument("--seconds", type=int, default=60, help="stop after this long")
    deploy = sub.add_parser("deploy", help="copy, build and test; never activates")
    deploy.add_argument(
        "--with-sipl", action="store_true", help="also build the SIPL backend"
    )
    deploy.add_argument("--name", help="release directory name (default: commit)")
    parser.add_argument(
        "--dry-run", action="store_true", help="print the commands instead of running"
    )
    args = parser.parse_args(argv)
    try:
        if args.command == "check":
            cmd_check(args.target, args.dry_run)
        elif args.command == "stop":
            cmd_stop(args.target, args.dry_run)
        elif args.command == "linktest":
            cmd_linktest(args.target, args.mb, args.dry_run)
        elif args.command == "run":
            cmd_run(args.target, args.name, args.config, args.seconds, args.dry_run)
        elif args.command == "discover":
            cmd_discover(args.target, args.dry_run)
        else:
            cmd_deploy(args.target, args.with_sipl, args.name, args.dry_run)
    except DeployError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
