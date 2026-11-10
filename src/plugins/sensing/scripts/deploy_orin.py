#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build the camera producer on an Orin from this checkout, package it, and run it as a service.

Run on the desktop:

    deploy_orin.py user@orin --config orin.json

`--install-prereqs` and `--check-camera` run the Orin's prerequisite install and camera check instead of deploying
(neither needs a config).

The config is JSON with the producer's own options, nothing else:

    {"args": ["--add-stream=sensor=0,livekit=sensing-left", "--add-stream=sensor=1,livekit=sensing-right"]}

The token file stays on the Orin, outside the release; only its path is passed. With "token_source" (a local file)
the deploy first uploads it to "token_file" on the Orin with mode 0600. Nothing is built, stopped or
replaced until the target passes its checks. A build or transfer failure leaves the running producer alone.
The same file runs on the target (`deploy_orin.py remote ...`) from the source snapshot it copied there.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import os
import posixpath
import pwd
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path

SERVICE = "isaaccapture-producer.service"
BIN_REL = "plugins/sensing_camera/camera_plugin_sensing"
UNIT_TEMPLATE = Path(__file__).with_name("isaaccapture-producer.service.in")
CACHE = "~/.cache/isaaccapture-deploy"
MODES = ("direct", "livekit")
CONFIG_KEYS = {
    "mode",
    "livekit_url",
    "token_file",
    "token_source",
    "args",
    "env",
    "cameras",
}
# With "cameras": "auto" the deploy finds the cameras and supplies these, so the config must not.
DETECTED_ARGS = ("--platform-config", "--config-name", "--link-masks")
PLATFORM_FILE = (
    "camera-platform.json"  # under the install prefix; the detected platform config
)
PLATFORM_CONFIG_NAME = "SHW5G_2"
ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
RESERVED_ENV = ("LD_LIBRARY_PATH",)
# The producer's connection options come from the config's own fields, not from `args`.
PREFIX = "/opt/isaaccapture"
SERVICE_GROUPS = "video"  # device access for the cameras
READY_TIMEOUT_S = 60
REBOOT_TIMEOUT_S = (
    600  # a reboot to load the device-tree overlay, until ssh answers again
)
REBOOT_START_S = (
    120  # how long to wait for the Orin to go down once setup asks for a reboot
)
NEEDS_REBOOT = 2  # orin_setup.sh exit status: reboot or log in again, then re-run
SDKM_DOWNLOADS = "~/Downloads/nvidia/sdkm_downloads"
RESERVED_ARGS = ("--livekit-url", "--livekit-token-file")
# The Orin-side setup and camera-check scripts, and the platform config they stream with.
TOOL_SCRIPTS = ("orin_setup.sh", "setup.sh", "common.sh", "verify.sh", "validate.sh")
TOOL_CONFIG = "shw5g.json"
SETUP_HINT = f"bash {CACHE}/tools/scripts/orin_setup.sh"


def tool_files() -> list[tuple[Path, str]]:
    """(local file, path under the tools directory) for the Orin-side scripts.

    They come from the checkout holding this helper, never from the deployed source, so any source can be deployed.
    On the Orin the helper sits beside the `tools/` directory the desktop uploaded.
    """
    here = Path(__file__).resolve().parent
    if (here / "tools").is_dir():
        scripts, configs = here / "tools" / "scripts", here / "tools" / "configs"
    else:
        scripts, configs = here, here.parent / "configs"
    return [(scripts / name, f"scripts/{name}") for name in TOOL_SCRIPTS] + [
        (configs / TOOL_CONFIG, f"configs/{TOOL_CONFIG}")
    ]


STEPS = 5


def step(number: int, text: str) -> None:
    print(f"==> [{number}/{STEPS}] {text}", flush=True)


class Progress:
    """A text progress bar: redrawn in place on a terminal, one line per 10% otherwise."""

    def __init__(self, label: str, total: int) -> None:
        self.label, self.total = label, max(total, 1)
        self._last = -1

    def update(self, done: int) -> None:
        percent = min(100, done * 100 // self.total)
        if sys.stdout.isatty():
            filled = percent * 30 // 100
            bar = "#" * filled + "-" * (30 - filled)
            print(
                f"\r    {self.label} [{bar}] {percent:3d}% ({done}/{self.total})",
                end="",
                flush=True,
            )
        elif percent // 10 != self._last // 10:
            print(f"    {self.label} {percent}%", flush=True)
        self._last = percent

    def finish(self) -> None:
        self.update(self.total)
        if sys.stdout.isatty():
            print()


class DeployError(Exception):
    """A failure with a message worth showing as is."""


def run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    result = subprocess.run(cmd, check=False, **kwargs)
    if result.returncode != 0:
        raise DeployError(
            f"{shlex.join(cmd[:3])} ... failed with exit code {result.returncode}"
        )
    return result


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Config and service unit
# ---------------------------------------------------------------------------


def configured_streams(config: dict) -> list[tuple[int, bool]]:
    """(sensor, has a direct destination) for each --add-stream=sensor=N,... argument of the config."""
    streams = []
    for arg in config.get("args", []):
        if not arg.startswith("--add-stream="):
            continue
        # The argument is a comma-separated list of key=value pairs, in any order.
        pairs = dict(
            item.partition("=")[::2]
            for item in arg.split("=", 1)[1].split(",")
            if "=" in item
        )
        if pairs.get("sensor", "").isdigit():
            streams.append((int(pairs["sensor"]), bool(pairs.get("direct"))))
    return streams


def configured_sensors(config: dict) -> list[int]:
    """The sensor numbers the config asks the producer to publish."""
    return [sensor for sensor, _ in configured_streams(config)]


def load_config(path: Path) -> dict:
    try:
        config = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise DeployError(f"cannot read the config {path}: {exc}") from exc
    if not isinstance(config, dict) or set(config) - CONFIG_KEYS:
        raise DeployError(
            f"{path}: expected an object with args and, optionally, livekit_url and token_file"
        )
    env = config.get("env", {})
    if not isinstance(env, dict) or not all(
        isinstance(k, str)
        and ENV_NAME.fullmatch(k)
        and isinstance(v, str)
        and "\n" not in v
        for k, v in env.items()
    ):
        raise DeployError(
            f"{path}: env must map variable names to single-line string values"
        )
    if any(k in RESERVED_ENV for k in env):
        raise DeployError(
            f"{path}: env may not set {', '.join(RESERVED_ENV)}; the release sets it"
        )
    mode = config.get("mode", "direct")
    if mode not in MODES:
        raise DeployError(f'{path}: "mode" must be one of {", ".join(MODES)}')
    has_livekit = any(
        k in config for k in ("livekit_url", "token_file", "token_source")
    )
    if mode == "direct" and has_livekit:
        raise DeployError(
            f'{path}: livekit_url, token_file and token_source belong to "mode": "livekit"; a direct producer has no LiveKit settings'
        )
    if mode == "livekit" and not ("livekit_url" in config and "token_file" in config):
        raise DeployError(f'{path}: "mode": "livekit" needs livekit_url and token_file')
    args = config.get("args", [])
    if not isinstance(args, list) or not all(isinstance(a, str) for a in args):
        raise DeployError(f"{path}: args must be a list of strings")
    if any(a.split("=", 1)[0] in RESERVED_ARGS for a in args):
        raise DeployError(
            f"{path}: give the URL and token file as livekit_url and token_file, not in args"
        )
    if not any(a.startswith("--add-stream=") for a in args):
        raise DeployError(
            f"{path}: args needs at least one --add-stream=sensor=N,livekit=NAME"
        )
    if "cameras" in config:
        if config["cameras"] != "auto":
            raise DeployError(f'{path}: "cameras" must be "auto"')
        if any(a.split("=", 1)[0] in DETECTED_ARGS for a in args):
            raise DeployError(
                f'{path}: with "cameras": "auto" the deploy supplies {", ".join(DETECTED_ARGS)}; remove them from args'
            )
    if "token_file" in config:
        if not all(
            isinstance(config[k], str) and config[k]
            for k in ("livekit_url", "token_file")
        ):
            raise DeployError(
                f"{path}: livekit_url and token_file must be non-empty strings"
            )
        if not config["token_file"].startswith("/"):
            raise DeployError(
                f"{path}: token_file must be an absolute path on the Orin"
            )
    if "token_source" in config:
        source = config["token_source"]
        if not isinstance(source, str) or not source:
            raise DeployError(f"{path}: token_source must be a non-empty string")
        # A relative source is relative to the config file, so a config and its token can move together.
        local = Path(source).expanduser()
        if not local.is_absolute():
            local = path.parent / local
        if not local.is_file():
            raise DeployError(f"{path}: token_source {local} is not a file")
        if not local.read_text().strip():
            raise DeployError(f"{path}: token_source {local} is empty")
        config["token_source"] = str(local.resolve())
    return config


def install_token(target: "Target", local: Path, remote: str) -> None:
    """Upload a local token file to ``remote`` on the Orin, readable only by the ssh user, replacing any older one.

    The token is never part of an argument list, the release or the output: it travels by scp into a staging file
    created with mode 0600, and moves into place only once complete, so the service never reads half a token.
    """
    staging = f"{remote}.new"
    quoted_remote, quoted_staging = shlex.quote(remote), shlex.quote(staging)
    # Create the staging file private before any bytes arrive; scp keeps an existing file's mode.
    target.capture(
        f"mkdir -p {shlex.quote(posixpath.dirname(remote))} && umask 077 && rm -f {quoted_staging} && : > {quoted_staging}"
    )
    target.put(local, staging)
    target.capture(
        f"chmod 600 {quoted_staging} && mv -f {quoted_staging} {quoted_remote}"
    )
    print(f"  uploaded the LiveKit token to {remote} (mode 0600)")


def systemd_quote(arg: str) -> str:
    """One ExecStart word: systemd expands % and $ and splits on whitespace, so escape and quote them."""
    arg = (
        arg.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("%", "%%")
        .replace("$", "$$")
    )
    return f'"{arg}"' if not arg or re.search(r"\s|[\"']", arg) else arg


def render_unit(config: dict, prefix: str, user: str, groups: str) -> str:
    words = [
        f"{prefix}/current/{BIN_REL}",
        *(
            [
                f"--livekit-url={config['livekit_url']}",
                f"--livekit-token-file={config['token_file']}",
            ]
            if config.get("mode") == "livekit"
            else []
        ),
        *config.get("args", []),
    ]
    exec_start = " ".join(systemd_quote(w) for w in words)
    fields = {
        "SERVICE_USER": user,
        "SERVICE_GROUPS": groups,
        "PREFIX": prefix,
        "EXEC_START": exec_start,
        # The producer's own environment variables (for example a diagnostic switch), one Environment= line each.
        "ENVIRONMENT": "\n".join(
            f"Environment={systemd_quote(f'{name}={value}')}"
            for name, value in config.get("env", {}).items()
        ),
    }
    text = UNIT_TEMPLATE.read_text()
    for key, value in fields.items():
        text = text.replace(f"@{key}@", value)
    return text


# ---------------------------------------------------------------------------
# Desktop side
# ---------------------------------------------------------------------------


def repo_root() -> Path:
    out = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=Path(__file__).parent,
        capture_output=True,
        text=True,
        check=True,
    )
    return Path(out.stdout.strip())


DEPLOY_DIR = ".deploy"


def tracked_files(root: Path) -> list[str]:
    """The tracked files that exist on disk, so tracked local edits are included and untracked files never are."""
    listing = subprocess.run(
        ["git", "ls-files", "-z", "--recurse-submodules"],
        cwd=root,
        capture_output=True,
        check=True,
    ).stdout
    return sorted(f for f in listing.decode().split("\0") if f and (root / f).is_file())


def hash_files(root: Path, files: list[str]) -> dict[str, str]:
    """Content hash per file, with the executable bit, which the Orin's copy must also have."""
    bar = Progress("hashing", len(files))
    entries = {}
    for done, name in enumerate(files, 1):
        path = root / name
        executable = "x" if os.access(path, os.X_OK) else "-"
        entries[name] = f"{sha256_file(path)}{executable}"
        if done % 200 == 0:
            bar.update(done)
    bar.finish()
    return entries


def diff_manifests(
    local: dict[str, str], remote: dict[str, str]
) -> tuple[list[str], list[str]]:
    """(files to send, files to remove on the Orin): one comparison of the two per-file manifests."""
    send = sorted(n for n, h in local.items() if remote.get(n) != h)
    removed = sorted(n for n in remote if n not in local)
    return send, removed


def read_remote_manifest(target: "Target", cache: str) -> dict[str, str]:
    """What the Orin's source copy holds, or nothing when it has no copy or the record is unreadable."""
    text = target.capture(
        f"test -d {cache}/src && cat {cache}/src.manifest.json || true"
    )
    try:
        entries = json.loads(text) if text else {}
    except ValueError:
        return {}
    return entries if isinstance(entries, dict) else {}


def git_state(root: Path) -> dict:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    return {"base_revision": head, "dirty": bool(dirty)}


def build_delta(
    root: Path,
    send: list[str],
    removed: list[str],
    manifest: dict[str, str],
    dest: Path,
) -> None:
    """Archive only the files to send, with the new full manifest and the list of files to remove."""
    bar = Progress("packing", len(send))
    with tarfile.open(dest, "w:gz") as tar:
        for done, name in enumerate(send, 1):
            tar.add(root / name, arcname=name, recursive=False)
            if done % 200 == 0:
                bar.update(done)
        for name, payload in (("manifest.json", manifest), ("removed.json", removed)):
            data = json.dumps(payload).encode()
            info = tarfile.TarInfo(f"{DEPLOY_DIR}/{name}")
            info.size = len(data)
            info.mtime = int(time.time())
            tar.addfile(info, io.BytesIO(data))
    bar.finish()


PASSWORD_ENV = "ORIN_PASSWORD"
USER_ENV = "ORIN_USER"


def target_host(target: str) -> str:
    """`host` becomes `$ORIN_USER@host` when ORIN_USER is set; `user@host` is used as given."""
    user = os.environ.get(USER_ENV)
    return f"{user}@{target}" if user and "@" not in target else target


class Target:
    """The Orin over ssh/scp.

    With ORIN_PASSWORD set, ssh gets it through SSH_ASKPASS and sudo reads it from stdin (`sudo -S`), so it never
    appears in an argument list or a log; without it, ssh and sudo prompt as usual. One ssh connection is reused for
    every step.
    """

    def __init__(self, host: str) -> None:
        self.host = host
        self.password = os.environ.get(PASSWORD_ENV)
        self._dir = Path(tempfile.mkdtemp(prefix="deploy-orin-"))
        self._env = dict(os.environ)
        if self.password:
            askpass = self._dir / "askpass"
            askpass.write_text(f'#!/bin/sh\nprintf "%s\\n" "${PASSWORD_ENV}"\n')
            askpass.chmod(0o700)
            self._env.update(SSH_ASKPASS=str(askpass), SSH_ASKPASS_REQUIRE="force")
        self._opts = [
            "-o",
            "ControlMaster=auto",
            "-o",
            f"ControlPath={self._dir}/cm",
            "-o",
            "ControlPersist=300",
        ]

    def _run(self, tool: str, args: list[str], **kwargs) -> subprocess.CompletedProcess:
        return subprocess.run(
            [tool, *self._opts, *args], env=self._env, check=False, **kwargs
        )

    def ssh(self, command: str, *, tty: bool = False) -> subprocess.CompletedProcess:
        result = self._run("ssh", [*(["-t"] if tty else []), self.host, command])
        if result.returncode != 0:
            raise DeployError(
                f"ssh {self.host} failed with exit code {result.returncode}"
            )
        return result

    def status(
        self,
        command: str,
        *,
        tty: bool = False,
        timeout: int = 0,
        stdin: str | None = None,
    ) -> int:
        """Run a command and return its exit status instead of raising (255 when ssh itself fails)."""
        opts = ["-o", f"ConnectTimeout={timeout}"] if timeout else []
        extra: dict = {"capture_output": True} if timeout else {}
        if stdin is not None:
            extra.update(input=stdin, text=True)
        return self._run(
            "ssh", [*opts, *(["-t"] if tty else []), self.host, command], **extra
        ).returncode

    def output(self, command: str, *, stdin: str | None = None) -> tuple[int, str]:
        """Run a command; return (exit status, stdout). Its stderr still goes to this terminal."""
        extra: dict = (
            {"input": stdin, "text": True} if stdin is not None else {"text": True}
        )
        result = self._run("ssh", [self.host, command], stdout=subprocess.PIPE, **extra)
        return result.returncode, result.stdout

    def sudo(self, command: str) -> None:
        """Run a command as root: the password from the environment on stdin, else an interactive prompt."""
        if self.password:
            result = self._run(
                "ssh",
                [self.host, f"sudo -S -p '' {command}"],
                input=self.password + "\n",
                text=True,
            )
            if result.returncode != 0:
                raise DeployError(
                    f"sudo on {self.host} failed with exit code {result.returncode}"
                )
        else:
            self.ssh(f"sudo {command}", tty=True)

    def validate_connection(self) -> None:
        """Fail early, with the reason, when the Orin cannot be reached or lacks what the deployment runs on it."""
        probe = self._run(
            "ssh",
            [
                "-o",
                "ConnectTimeout=10",
                self.host,
                "command -v python3 >/dev/null && uname -m",
            ],
            capture_output=True,
            text=True,
        )
        if probe.returncode != 0:
            reason = probe.stderr.strip() or "python3 is not installed on the target"
            raise DeployError(f"cannot use {self.host} over ssh: {reason}")
        if probe.stdout.strip() != "aarch64":
            raise DeployError(
                f"{self.host} is {probe.stdout.strip() or 'unknown'}, not an aarch64 Jetson"
            )

    def capture(self, command: str) -> str:
        out = self._run("ssh", [self.host, command], capture_output=True, text=True)
        if out.returncode != 0:
            raise DeployError(
                f"ssh {self.host} {command!r} failed: {out.stderr.strip()}"
            )
        return out.stdout.strip()

    def readable(self, path: str) -> bool:
        return self._run("ssh", [self.host, "test", "-r", path]).returncode == 0

    @staticmethod
    def _scp_quiet() -> list[str]:
        """scp shows its own transfer meter on a terminal; stay quiet when output is piped."""
        return [] if sys.stdout.isatty() else ["-q"]

    def put(self, local: Path, remote: str) -> None:
        if (
            self._run(
                "scp", [*self._scp_quiet(), str(local), f"{self.host}:{remote}"]
            ).returncode
            != 0
        ):
            raise DeployError(f"copying {local.name} to {self.host} failed")

    def close(self) -> None:
        self._run("ssh", ["-O", "exit", self.host], capture_output=True)
        shutil.rmtree(self._dir, ignore_errors=True)


def upload_tools(target: "Target", cache: str) -> None:
    """Put the setup and camera-check scripts, with the config they stream with, under the Orin's cache."""
    target.capture(f"mkdir -p {cache}/tools/scripts {cache}/tools/configs")
    for local_file, relative in tool_files():
        target.put(local_file, f"{cache}/tools/{relative}")


def show_logs(target: "Target") -> None:
    """Print the producer service's state and recent journal; nothing on the Orin changes."""
    target.validate_connection()
    target.sudo(f"systemctl status {SERVICE} --no-pager -l")
    target.sudo(f"journalctl -u {SERVICE} -n 120 --no-pager")


def run_tools(args: argparse.Namespace, target: "Target") -> None:
    """Install the Orin's prerequisites and/or check its cameras with the uploaded scripts; deploy nothing.

    The output shows live. Given ORIN_PASSWORD the Orin's sudo is answered for you; otherwise it prompts in the
    terminal.
    """
    if args.logs:
        show_logs(target)
        return
    step(1, f"Connecting to {target.host}")
    target.validate_connection()
    cache = target.capture(f"mkdir -p {CACHE} && cd {CACHE} && pwd")
    step(2, "Upload the setup and camera-check scripts")
    upload_tools(target, cache)
    check = args.check_camera or args.scan_cameras
    flags = []
    if args.install_prereqs and not check:
        flags.append("--prereqs-only")
    elif not args.install_prereqs:
        flags.append("--check-only")
    if args.scan_cameras:
        flags.append("--scan")
    label = " and ".join(
        part
        for part, wanted in (
            ("Install prerequisites", args.install_prereqs),
            (
                "Scan the camera links" if args.scan_cameras else "Check the cameras",
                check,
            ),
        )
        if wanted
    )
    step(3, label)
    code = run_orin_setup(target, cache, " ".join(flags))
    if code != 0:
        raise DeployError(f"{label.lower()} failed on {target.host} (exit {code})")


def detect_cameras(target: "Target", cache: str, rescan: bool = False) -> dict:
    """Find where cameras are connected, on any of the carrier's 8 inputs, and write the platform config for them.

    The inputs found are remembered on the Orin and tried first next time; all 8 are scanned again only when a
    remembered camera does not answer, or with rescan. The Orin's producer service holds the cameras, so it is
    stopped first. Returns the JSON orin_setup.sh prints:
    the platform config's path on the Orin, the link masks, and how many cameras were found and used.
    """
    print("==> Finding the cameras (any of the carrier's 8 inputs)", flush=True)
    target.sudo(f"sh -c {shlex.quote(f'systemctl stop {SERVICE} || true')}")
    code, text = target.output(
        f"bash {cache}/tools/scripts/orin_setup.sh --detect {cache}/cameras"
        + (" --rescan" if rescan else "")
    )
    if code != 0:
        raise DeployError(
            "no camera answers on any of the 8 inputs; check the carrier's 12 V supply and each cable, "
            "or run --scan-cameras"
        )
    try:
        found = json.loads(text.strip().splitlines()[-1])
        found["config"], found["masks"], found["used"]
    except (ValueError, IndexError, KeyError) as exc:
        raise DeployError(
            f"camera detection printed no usable result: {text!r}"
        ) from exc
    for place in found.get("positions", []):
        print(f"    camera: {place}")
    return found


def apply_detected(config: dict, found: dict) -> dict:
    """The deploy config with the detected cameras: their platform config and masks, and streams for them only.

    Sensors are numbered from 0 in the order of the detected cameras, so a stream for a sensor that was not found
    is dropped. All streams dropped is an error.
    """
    used = found["used"]
    kept, dropped = [], []
    for arg in config["args"]:
        if arg.startswith("--add-stream="):
            sensors = configured_sensors({"args": [arg]})
            if sensors and sensors[0] >= used:
                dropped.append(arg)
                continue
        kept.append(arg)
    if not configured_sensors({"args": kept}):
        raise DeployError(
            f"the config streams sensors the detected cameras do not have ({used} found)"
        )
    if dropped:
        print(
            f"    {used} camera(s) found; dropping {len(dropped)} stream(s) for sensors that are not there",
            flush=True,
        )
    return {
        **config,
        "args": [
            *kept,
            f"--platform-config={PREFIX}/{PLATFORM_FILE}",
            f"--config-name={PLATFORM_CONFIG_NAME}",
            f"--link-masks={found['masks']}",
        ],
    }


def find_sipl_package(explicit: str | None) -> Path | None:
    """The L4T SIPL API headers archive on this desktop: the named file, else the newest SDK Manager download."""
    if explicit:
        path = Path(explicit).expanduser()
        if not path.is_file():
            raise DeployError(f"--sipl-api {explicit}: no such file")
        return path
    found = sorted(
        Path(SDKM_DOWNLOADS).expanduser().glob("Jetson_SIPL_API_R*_aarch64.tbz2"),
        key=lambda p: p.stat().st_mtime,
    )
    return found[-1] if found else None


def wait_for_reboot(target: "Target") -> None:
    """Block until the Orin has gone down and answered ssh again; fail when it never reboots or never returns."""
    print("==> Waiting for the Orin to reboot", flush=True)
    start = time.monotonic()
    went_down = False
    while time.monotonic() - start < REBOOT_TIMEOUT_S:
        up = target.status("true", timeout=5) == 0
        if not up:
            went_down = True
        elif went_down:
            return
        elif time.monotonic() - start > REBOOT_START_S:
            break
        time.sleep(5)
    raise DeployError(
        "the Orin did not reboot and come back; reboot it yourself and run this command again"
        if not went_down
        else f"the Orin did not answer ssh within {REBOOT_TIMEOUT_S} s of rebooting"
    )


def prerequisites_ready(target: "Target", cache: str, sdk_root: str) -> bool:
    """The Orin passes its own preflight and has the drivers, overlay and SIPL headers the cameras need."""
    remote = f"python3 {cache}/deploy_orin.py remote"
    if target.status(f"{remote} preflight --sdk-root {shlex.quote(sdk_root)}") != 0:
        return False
    check = (
        f"bash {cache}/tools/scripts/orin_setup.sh --prereqs-only --no-apt --no-drivers"
    )
    return target.status(f"{check} --sdk-root {shlex.quote(sdk_root or '/')}") == 0


def run_orin_setup(target: "Target", cache: str, flags: str) -> int:
    """Run orin_setup.sh on the Orin and return its exit status.

    With the password in the environment nothing prompts: it goes to the script on stdin, for its sudo calls.
    Without one the script runs in a terminal and sudo prompts there.
    """
    command = f"bash {cache}/tools/scripts/orin_setup.sh {flags}".rstrip()
    if target.password:
        return target.status(
            f"{command} --sudo-password-stdin", stdin=target.password + "\n"
        )
    return target.status(command, tty=True)


def install_prerequisites(
    args: argparse.Namespace, target: "Target", cache: str
) -> None:
    """Install what the preflight found missing: packages, uv, SIPL headers, the SENSING drivers and overlay.

    The drivers step selects the overlay through Jetson-IO's command line and reboots the Orin; the deploy waits out
    the reboot and carries on. With no ORIN_PASSWORD it runs in a terminal and sudo prompts.
    """
    print("==> Installing the Orin's missing prerequisites", flush=True)
    flags = "--prereqs-only"
    if args.sdk_root:
        flags += f" --sdk-root {shlex.quote(args.sdk_root)}"
    sipl = find_sipl_package(args.sipl_api)
    if sipl:
        target.put(sipl, f"{cache}/{sipl.name}")
        flags += f" --sipl-api {cache}/{sipl.name}"

    def run_setup() -> int:
        return run_orin_setup(target, cache, flags)

    code = run_setup()
    if code in (
        NEEDS_REBOOT,
        255,
    ):  # 255: ssh dropped because Jetson-IO rebooted the Orin
        wait_for_reboot(target)
        target.validate_connection()
        code = run_setup()
    if code != 0:
        raise DeployError(
            f"prerequisite setup did not finish (exit {code}); fix what it reported and run this command again"
        )
    if not prerequisites_ready(target, cache, args.sdk_root):
        raise DeployError("the Orin still fails the preflight after setup")


def deploy(args: argparse.Namespace) -> None:
    tools_only = (
        args.install_prereqs or args.check_camera or args.scan_cameras or args.logs
    )
    if not tools_only and not args.config:
        raise DeployError("--config is required to deploy")
    config = None if tools_only else load_config(Path(args.config))
    target = Target(target_host(args.target))
    try:
        if tools_only:
            run_tools(args, target)
        else:
            run_deploy(args, config, target)
    finally:
        target.close()


def git_toplevel(folder: Path) -> Path:
    found = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=folder,
        capture_output=True,
        text=True,
        check=False,
    )
    if found.returncode != 0:
        raise DeployError(f"{folder} is not inside a git checkout")
    return Path(found.stdout.strip())


@contextlib.contextmanager
def open_source(args: argparse.Namespace):
    """The source to deploy as (root, files, git state).

    By default the tracked files of the working tree of the checkout holding this script, with local edits. `--source`
    names another checkout (a worktree, say) to deploy instead. `--branch` deploys that branch's committed files from
    that checkout, ignoring the working tree, exported to a temporary directory.
    """
    repo = git_toplevel(Path(args.source)) if args.source else repo_root()
    if not args.branch:
        yield repo, tracked_files(repo), git_state(repo)
        return
    found = subprocess.run(
        ["git", "rev-parse", "--verify", "--quiet", f"{args.branch}^{{commit}}"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    if found.returncode != 0:
        raise DeployError(f"branch {args.branch!r} not found in {repo}")
    revision = found.stdout.strip()
    with tempfile.TemporaryDirectory() as tmp:
        export = Path(tmp) / "tree"
        export.mkdir()
        archive = subprocess.Popen(
            ["git", "archive", "--format=tar", revision],
            cwd=repo,
            stdout=subprocess.PIPE,
        )
        with tarfile.open(fileobj=archive.stdout, mode="r|") as tar:
            tar.extractall(export, filter="data")
        if archive.wait() != 0:
            raise DeployError(f"git archive of {args.branch!r} failed")
        files = sorted(
            p.relative_to(export).as_posix() for p in export.rglob("*") if p.is_file()
        )
        yield export, files, {"base_revision": revision, "dirty": False}


def run_deploy(args: argparse.Namespace, config: dict, target: Target) -> None:
    with open_source(args) as (root, files, state):
        deploy_source(args, config, target, root, files, state)


def deploy_source(
    args: argparse.Namespace,
    config: dict,
    target: Target,
    root: Path,
    files: list[str],
    state: dict,
) -> None:
    step(1, f"Connecting to {target.host}")
    target.validate_connection()
    user = args.service_user or target.capture("id -un")

    step(2, f"Preflight on {target.host}")
    cache = target.capture(f"mkdir -p {CACHE} && cd {CACHE} && pwd")
    if config.get("token_source"):
        # The file is owned by the ssh user with mode 0600, so only a service running as that user can read it.
        if args.service_user and args.service_user != target.capture("id -un"):
            raise DeployError(
                "token_source uploads the token readable by the ssh user only; "
                "with a different --service-user, place the token file yourself"
            )
        install_token(target, Path(config["token_source"]), config["token_file"])
    if config.get("mode") == "livekit" and not target.readable(config["token_file"]):
        raise DeployError(
            f"the token file {config['token_file']} is not readable on the Orin"
        )
    # The helper always travels alone and fresh, so the Orin-side steps never depend on an older source copy.
    helper = f"{cache}/deploy_orin.py"
    target.put(Path(__file__).resolve(), helper)
    # The setup and camera-check scripts go up before the preflight, so a missing prerequisite can be fixed with them.
    upload_tools(target, cache)
    remote = f"python3 {helper} remote"
    if not prerequisites_ready(target, cache, args.sdk_root):
        if args.no_setup:
            raise DeployError(
                f"the Orin fails the preflight; run {SETUP_HINT} --prereqs-only or drop --no-setup"
            )
        install_prerequisites(args, target, cache)

    platform_file = ""
    if config.get("cameras") == "auto":
        found = detect_cameras(target, cache, args.rescan_cameras)
        config = apply_detected(config, found)
        platform_file = found["config"]
    unit = render_unit(config, PREFIX, user, SERVICE_GROUPS)

    with tempfile.TemporaryDirectory() as tmp:
        step(3, "Compare the source with the Orin's and upload what differs")
        meta = dict(state)
        local = hash_files(root, files)
        meta["manifest_sha256"] = hashlib.sha256(
            json.dumps(local, sort_keys=True).encode()
        ).hexdigest()
        release = f"{time.strftime('%Y%m%d-%H%M%S')}-{meta['base_revision'][:9]}{'-dirty' if meta['dirty'] else ''}"
        print(
            f"    base {meta['base_revision'][:12]}, dirty={meta['dirty']}, {len(files)} tracked files"
        )
        # One fixed source path, so the persistent build directory's CMake cache keeps matching it; files that are
        # unchanged keep their modification times and are not rebuilt.
        src = f"{cache}/src"
        remote_manifest = read_remote_manifest(target, cache)
        send, removed = diff_manifests(local, remote_manifest)
        size = sum((root / n).stat().st_size for n in send)
        print(
            f"    {len(local) - len(send)} identical, {len(send)} to send ({size / 1e6:.1f} MB), "
            f"{len(removed)} to remove"
        )
        if send or removed:
            delta = Path(tmp) / "delta.tar.gz"
            build_delta(root, send, removed, local, delta)
            target.put(delta, f"{cache}/delta.tar.gz")
            target.ssh(
                f"{remote} apply --archive {cache}/delta.tar.gz --src {src} --manifest {cache}/src.manifest.json"
                + ("" if remote_manifest else " --fresh")
            )
        else:
            print("    the Orin's source is identical; nothing to send")
        step(4, "Build and package on the Orin (the running producer is untouched)")
        stage, archive = (
            f"{cache}/stage-{release}",
            f"{cache}/producer-{release}.tar.gz",
        )
        target.ssh(
            f"{remote} build --src {src} --stage {stage} --jobs {args.jobs} "
            f"--sdk-root {shlex.quote(args.sdk_root)}"
            + "".join(f" --cmake-arg={shlex.quote(a)}" for a in args.cmake_arg)
        )
        manifest = {
            **meta,
            "release": release,
            "arch": "aarch64",
            "config": args.config,
        }
        manifest_path = Path(tmp) / "MANIFEST.json"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        target.put(manifest_path, f"{stage}/MANIFEST.json")
        target.ssh(f"{remote} package --stage {stage} --archive {archive}")
        unit_path = Path(tmp) / SERVICE
        unit_path.write_text(unit)
        target.put(unit_path, f"{cache}/{SERVICE}")

    if args.no_activate:
        print(f"==> Built release {release}; not activated (--no-activate)")
        return
    step(5, "Install and replace the running producer (needs sudo on the Orin)")
    activate = (
        f"{remote} activate --archive {archive} --prefix {PREFIX} --release {release} "
        f"--unit {cache}/{SERVICE} --owner {user} --ready-timeout {READY_TIMEOUT_S} "
        f"--sensors={','.join(str(n) for n in configured_sensors(config))} "
        f"--direct-sensors={','.join(str(n) for n, direct in configured_streams(config) if direct)} "
        f"--mode={config.get('mode', 'direct')}"
        + (f" --platform-file {shlex.quote(platform_file)}" if platform_file else "")
        + (" --replace-unmanaged" if args.replace_unmanaged else "")
    )
    target.sudo(activate)
    print(f"==> Deployed {release}; logs: ssh {target.host} journalctl -u {SERVICE} -f")


# ---------------------------------------------------------------------------
# Target side
# ---------------------------------------------------------------------------


def tail(cmd: list[str]) -> str:
    out = subprocess.run(cmd, capture_output=True, text=True, check=False)
    return (out.stdout + out.stderr).strip()


def sdk_dirs(sdk_root: str) -> tuple[Path, Path]:
    """The Multimedia API and SIPL header trees: under --sdk-root when given, else the system locations.

    Only where the headers live is decided here; whether a given source tree needs them is its own CMake's concern.
    """
    base = Path(sdk_root) / "usr/src" if sdk_root else Path("/usr/src")
    return base / "jetson_multimedia_api", base / "jetson_sipl_api"


def build_path() -> str:
    """PATH for the build: an ssh command runs without the login profile, which is where uv and CUDA are added."""
    extra = [str(Path.home() / ".local/bin"), "/usr/local/cuda/bin"]
    return ":".join([*extra, os.environ.get("PATH", "")])


def remote_preflight(args: argparse.Namespace) -> None:
    if os.uname().machine != "aarch64":
        raise DeployError(f"the target is {os.uname().machine}, not aarch64")
    release_file = Path("/etc/nv_tegra_release")
    if not release_file.exists():
        raise DeployError("no /etc/nv_tegra_release: this is not a Jetson L4T system")
    print(f"  L4T: {release_file.read_text().splitlines()[0]}")
    on_path = {**os.environ, "PATH": build_path()}
    for tool in ("cmake", "g++", "systemctl", "uv"):
        found = subprocess.run(
            ["which", tool], capture_output=True, check=False, env=on_path
        )
        if found.returncode != 0:
            raise DeployError(
                f"{tool} is not installed; run {SETUP_HINT} --prereqs-only"
            )
    if (
        not Path("/usr/local/cuda/bin/nvcc").exists()
        and subprocess.run(
            ["which", "nvcc"], capture_output=True, check=False
        ).returncode
    ):
        raise DeployError(
            f"nvcc not found (CUDA toolkit); run {SETUP_HINT} --prereqs-only"
        )
    stat = os.statvfs(Path.home())
    if stat.f_bavail * stat.f_frsize < 4 << 30:
        raise DeployError("less than 4 GB free in the home directory")
    mmapi, _ = sdk_dirs(args.sdk_root)
    if not mmapi.is_dir():
        raise DeployError(
            f"the Jetson Multimedia API is missing at {mmapi}; run {SETUP_HINT} --prereqs-only "
            "or pass --sdk-root"
        )
    print("  preflight ok")


def cache_value(build_dir: Path, name: str) -> str:
    """One entry of the build's CMakeCache.txt, or empty."""
    for line in (build_dir / "CMakeCache.txt").read_text().splitlines():
        if line.startswith(f"{name}:"):
            return line.split("=", 1)[1]
    return ""


def livekit_notice(build_dir: Path, name: str) -> Path:
    """The LiveKit SDK's LICENSE or NOTICE: where the build recorded it, else where the fetch or a local SDK put it."""
    sdk = cache_value(build_dir, "LIVEKIT_SDK_DIR")
    candidates = [
        cache_value(build_dir, f"SENSING_LIVEKIT_{name}_FILE"),
        str(build_dir / "_deps" / f"livekit_sdk_{name.lower()}-src" / name),
        f"{sdk}/{name}" if sdk else "",
        f"{sdk}/../{name}" if sdk else "",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return Path(candidate)
    raise DeployError(f"the LiveKit SDK's {name} file was not found for the release")


def stage_producer(build_dir: Path, stage: Path) -> None:
    """Assemble the release from the build tree itself, so it does not depend on the source's install rules.

    The release is the binary, the SIPL platform configs the build puts beside it, the setup and camera-check
    scripts, the LiveKit runtime libraries the binary resolves, and the LiveKit licence files. The service sets LD_LIBRARY_PATH to the release directory,
    so the copied libraries win over the build tree's RUNPATH.
    """
    binaries = [p for p in build_dir.rglob("camera_plugin_sensing") if p.is_file()]
    if len(binaries) != 1:
        raise DeployError(
            f"expected one camera_plugin_sensing in the build, found {len(binaries)}"
        )
    built = binaries[0]
    out = stage / Path(BIN_REL).parent
    out.mkdir(parents=True)
    binary = stage / BIN_REL
    shutil.copy2(built, binary)
    configs = built.parent / "configs"
    if not configs.is_dir():
        raise DeployError(f"the build staged no platform configs beside {built}")
    shutil.copytree(configs, out / "configs")
    # The setup and camera-check scripts resolve the platform config as ../configs from their own directory.
    (out / "scripts").mkdir()
    for local_file, relative in tool_files():
        if relative.startswith("scripts/"):
            shutil.copy2(local_file, out / relative)
    # The libraries are found by name among what the binary resolves in the build, not by the SDK's layout.
    for line in tail(["ldd", str(built)]).splitlines():
        name, _, rest = line.strip().partition(" => ")
        if name.startswith("liblivekit") and rest and not rest.startswith("not found"):
            shutil.copy2(rest.split(" (")[0], out / name)
    # A producer that sends only over its direct links links no LiveKit; one that does carries its libraries and the
    # SDK's licence files.
    if list(out.glob("liblivekit*.so*")):
        notices = out / "licenses" / "livekit-sdk"
        notices.mkdir(parents=True)
        for name in ("LICENSE", "NOTICE"):
            shutil.copy2(livekit_notice(build_dir, name), notices / name)
    # Everything must resolve from the release or the system: a library found only in the build tree would vanish
    # with the next clean build.
    env = {**os.environ, "LD_LIBRARY_PATH": str(out)}
    resolved = subprocess.run(
        ["ldd", str(binary)], capture_output=True, text=True, check=False, env=env
    )
    listing = (resolved.stdout + resolved.stderr).splitlines()
    missing = [ln for ln in listing if "not found" in ln]
    if missing:
        raise DeployError(
            "the producer has unresolved shared libraries on this target:\n"
            + "\n".join(missing)
        )
    in_build = [ln for ln in listing if str(build_dir) in ln]
    if in_build:
        raise DeployError(
            "the producer still resolves libraries from the build tree:\n"
            + "\n".join(in_build)
        )


def remote_build(args: argparse.Namespace) -> None:
    """Configure and build only the producer, then stage the release from the build tree.

    The build directory persists between deployments and is discarded when the toolchain, L4T release, source
    path or CMake arguments change.
    """
    cache = Path(os.path.expanduser(CACHE))
    build_dir = cache / "build"
    stamp = "|".join(
        [
            Path("/etc/nv_tegra_release").read_text().splitlines()[0],
            tail(["cmake", "--version"]).splitlines()[0],
            tail(["g++", "-dumpfullversion"]),
            args.src,
            *args.cmake_arg,
        ]
    )
    stamp_file = build_dir / ".stamp"
    if not stamp_file.exists() or stamp_file.read_text() != stamp:
        print(
            "  new toolchain, release, source or CMake arguments: clean build directory"
        )
        run(["rm", "-rf", str(build_dir)])
        build_dir.mkdir(parents=True)
        stamp_file.write_text(stamp)
    mmapi, sipl_api = sdk_dirs(args.sdk_root)
    env = {**os.environ, "PATH": build_path()}
    sdk_flags = [f"-DJETSON_MMAPI_DIR={mmapi}"]
    if sipl_api.is_dir():
        sdk_flags.append(f"-DSIPL_API_DIR={sipl_api}")
    # The CUDA driver library is not on the default link path on L4T, and is dropped by --as-needed.
    gpu_libcuda = Path("/opt/nvidia/l4t-gpu-libs/nvgpu/libcuda.so.1")
    link_flags = (
        [f"-DCMAKE_EXE_LINKER_FLAGS=-Wl,--no-as-needed {gpu_libcuda}"]
        if gpu_libcuda.exists()
        else []
    )
    run(
        [
            "cmake",
            "-S",
            args.src,
            "-B",
            str(build_dir),
            "-DCMAKE_BUILD_TYPE=Release",
            "-DCMAKE_CUDA_ARCHITECTURES=87",
            "-DBUILD_PLUGIN_SENSING=ON",
            "-DBUILD_TESTING=OFF",
            "-DBUILD_EXAMPLES=OFF",
            "-DBUILD_PYTHON_BINDINGS=OFF",
            "-DBUILD_VIZ=OFF",
            "-DENABLE_CLANG_FORMAT_CHECK=OFF",
            "-DCMAKE_CXX_FLAGS=-I/usr/local/cuda/include",
            *sdk_flags,
            *link_flags,
            # Options only some source trees define (a capture backend, say) go last so they can override.
            *args.cmake_arg,
        ],
        env=env,
    )
    run(
        [
            "cmake",
            "--build",
            str(build_dir),
            "--target",
            "camera_plugin_sensing",
            "--parallel",
            str(args.jobs),
        ],
        env=env,
    )
    run(["rm", "-rf", args.stage])
    stage = Path(args.stage)
    stage_producer(build_dir, stage)
    print(f"  staged {sum(len(files) for _, _, files in os.walk(stage))} files")


def remote_apply(args: argparse.Namespace) -> None:
    """Apply a source delta: extract the sent files, delete the removed ones, verify, record the new manifest."""
    src = Path(args.src)
    if args.fresh:
        # No trustworthy record of what is there: start from nothing so no stale file can reach the build.
        shutil.rmtree(src, ignore_errors=True)
    src.mkdir(parents=True, exist_ok=True)
    with tarfile.open(args.archive) as tar:
        members = tar.getmembers()
        meta = {m.name: m for m in members if m.name.startswith(f"{DEPLOY_DIR}/")}
        manifest = json.loads(
            tar.extractfile(meta[f"{DEPLOY_DIR}/manifest.json"]).read()
        )
        removed = json.loads(tar.extractfile(meta[f"{DEPLOY_DIR}/removed.json"]).read())
        sent = [m for m in members if m.name not in meta]
        tar.extractall(src, members=sent, filter="data")
    for name in removed:
        target = (src / name).resolve()
        if src.resolve() not in target.parents:
            raise DeployError(
                f"refusing to remove {name}: outside the source directory"
            )
        target.unlink(missing_ok=True)
        for parent in target.parents:
            if parent == src.resolve() or any(parent.iterdir()):
                break
            parent.rmdir()
    for member in sent:
        expected = manifest[member.name]
        if (
            f"{sha256_file(src / member.name)}{'x' if os.access(src / member.name, os.X_OK) else '-'}"
            != expected
        ):
            raise DeployError(
                f"{member.name} does not match its recorded hash after extraction"
            )
    Path(args.manifest).write_text(json.dumps(manifest))
    print(f"  applied {len(sent)} files, removed {len(removed)}")


def remote_package(args: argparse.Namespace) -> None:
    stage = Path(args.stage)
    sums = []
    for path in sorted(
        p for p in stage.rglob("*") if p.is_file() and p.name != "SHA256SUMS"
    ):
        sums.append(f"{sha256_file(path)}  ./{path.relative_to(stage)}")
    (stage / "SHA256SUMS").write_text("\n".join(sums) + "\n")
    with tarfile.open(args.archive, "w:gz") as tar:
        tar.add(stage, arcname=".")
    print(f"  packaged {args.archive}")


def unmanaged_producers() -> list[tuple[int, str]]:
    """Running camera_plugin_sensing processes the service does not own, as (pid, owner)."""
    main_pid = tail(["systemctl", "show", "-p", "MainPID", "--value", SERVICE])
    found = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit() or proc.name == main_pid:
            continue
        try:
            exe = os.readlink(proc / "exe")
            owner = proc.stat().st_uid
        except OSError:
            continue
        if Path(exe).name == "camera_plugin_sensing":
            found.append((int(proc.name), pwd.getpwuid(owner).pw_name))
    return sorted(found)


def stop_unmanaged(procs: list[tuple[int, str]], timeout_s: int = 20) -> None:
    """Interrupt exactly these PIDs (the producer stops cleanly on SIGINT) and wait for them to exit."""
    for pid, _ in procs:
        os.kill(pid, signal.SIGINT)
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if not any(Path(f"/proc/{pid}").exists() for pid, _ in procs):
            return
        time.sleep(0.5)
    raise DeployError(
        f"unmanaged producer PIDs did not exit after SIGINT: {[p for p, _ in procs]}"
    )


def remote_activate(args: argparse.Namespace) -> None:
    """Install the release, then stop the service, switch the active link and start it. Runs as root."""
    prefix = Path(args.prefix)
    unmanaged = unmanaged_producers()
    if unmanaged and not args.replace_unmanaged:
        listing = ", ".join(f"pid {pid} (user {owner})" for pid, owner in unmanaged)
        raise DeployError(
            f"a camera_plugin_sensing is already running outside the service: {listing}. "
            "Stop it, or pass --replace-unmanaged to stop exactly these processes."
        )
    dest = prefix / "releases" / args.release
    if dest.exists():
        raise DeployError(
            f"release {args.release} already exists under {prefix}/releases"
        )
    dest.mkdir(parents=True)
    try:
        print("  unpacking the release", flush=True)
        with tarfile.open(args.archive) as tar:
            tar.extractall(dest, filter="data")
        sums = (dest / "SHA256SUMS").read_text().splitlines()
        bar = Progress("verifying files", len(sums))
        for done, line in enumerate(sums, 1):
            digest, name = line.split("  ", 1)
            if sha256_file(dest / name) != digest:
                raise DeployError(
                    f"checksum mismatch for {name} in release {args.release}"
                )
            bar.update(done)
        bar.finish()
    except BaseException:
        shutil.rmtree(dest, ignore_errors=True)
        raise
    if args.platform_file:
        # The detected platform config outlives the release: the service reads it from a fixed path.
        shutil.copy2(args.platform_file, prefix / PLATFORM_FILE)
    run(["chown", "-R", f"{args.owner}:", str(prefix)])
    print(f"  installed release {args.release}")

    unit_target = Path("/etc/systemd/system") / SERVICE
    unit_text = Path(args.unit).read_text()
    if not unit_target.exists() or unit_target.read_text() != unit_text:
        unit_target.write_text(unit_text)
        run(["systemctl", "daemon-reload"])
        run(["systemctl", "enable", SERVICE], capture_output=True)
        print("  service unit updated")
    print("  stopping the running producer", flush=True)
    run(["systemctl", "stop", SERVICE])
    if unmanaged:
        print(f"  stopping unmanaged producer(s): {[pid for pid, _ in unmanaged]}")
        stop_unmanaged(unmanaged)
    link, staging_link = prefix / "current", prefix / "current.new"
    if staging_link.is_symlink():
        staging_link.unlink()
    staging_link.symlink_to(Path("releases") / args.release)
    os.replace(staging_link, link)
    since = time.strftime("%Y-%m-%d %H:%M:%S")
    print("  starting the new producer; waiting for the cameras", flush=True)
    run(["systemctl", "start", SERVICE])
    summary = wait_ready(
        since,
        args.ready_timeout,
        [int(n) for n in args.sensors.split(",") if n],
        mode=args.mode,
        direct_sensors=[int(n) for n in args.direct_sensors.split(",") if n],
    )
    for line in summary:
        print(f"  {line}")
    print(f"  producer is up (release {args.release})")


# Every line the capture driver logs carries "Severity N :"; at any severity they bury the lines that explain a
# failure. DRIVER_FAULT picks the few that do explain one, and BENIGN_DRIVER the ones that look like errors but are
# the driver noting a library or power driver it does not need.
DRIVER_NOISE = re.compile(r"Severity \d+ :")
DRIVER_FAULT = re.compile(r"Failed to|Error processing|HSL I2C error|status: \d+")
BENIGN_DRIVER = re.compile(r"driverCatalog|uddf_discover|No module power driver")


# The deserializer's own report that a link did not lock: SIPL names the link by letter, A to D for links 0 to 3.
DEAD_LINK = re.compile(r'Sequence: "enable_link_([A-D])"')


def dead_link_lines(lines: list[str]) -> list[str]:
    """One plain line per camera link the deserializer could not lock, in the order they failed."""
    found = []
    for line in lines:
        match = DEAD_LINK.search(line)
        if match and match.group(1) not in found:
            found.append(match.group(1))
    return [
        f"link {ord(letter) - ord('A')} (COAX{ord(letter) - ord('A')} on CN1) did not lock: no camera answers there. "
        "Check that camera's cable and its harness lead."
        for letter in found
    ]


def failure_report(since: str, sensors: list[int]) -> str:
    """What to read when the producer did not become ready.

    The service's state, how much of its log is driver chatter, the first and last real lines of what remains, the
    producer's own fault lines, which sensors never published, and where the complete log was saved.
    """
    state = tail(
        [
            "systemctl",
            "show",
            SERVICE,
            "-p",
            "ActiveState",
            "-p",
            "SubState",
            "-p",
            "Result",
        ]
        + ["-p", "ExecMainStatus", "-p", "NRestarts"]
    ).replace("\n", ", ")
    log = tail(
        ["journalctl", "-u", SERVICE, "--since", since, "--no-pager", "-o", "cat"]
    )
    saved = Path(os.path.expanduser(CACHE)) / "last-start-failure.log"
    saved.parent.mkdir(parents=True, exist_ok=True)
    saved.write_text(log + "\n")
    lines = log.splitlines()
    quiet = [ln for ln in lines if not DRIVER_NOISE.search(ln)]
    faults = [
        ln
        for ln in quiet
        if re.search(
            r"camera_|startup_failed|stream_failed|publish_failed|check each camera|driver log:",
            ln,
        )
    ]
    driver_faults = []
    for line in lines:
        if (
            DRIVER_NOISE.search(line)
            and DRIVER_FAULT.search(line)
            and not BENIGN_DRIVER.search(line)
        ):
            text = line.split(" : ", 1)[-1].strip()
            if text not in driver_faults:
                driver_faults.append(text)
    out = [
        f"service: {state}",
        *dead_link_lines(lines),
        f"log: {len(lines)} lines since the start, {len(lines) - len(quiet)} of them from the camera driver; "
        f"complete log saved on the Orin at {saved}",
    ]
    if driver_faults:
        out.append("camera driver errors:")
        out += [f"  {ln}" for ln in driver_faults[:4]]
    if sensors:
        out.append(f"sensors expected to publish: {', '.join(map(str, sensors))}")
    if faults:
        out.append("producer fault lines:")
        out += [f"  {ln}" for ln in faults[:8]]
    out.append("first lines (driver chatter removed):")
    first, rest = quiet[:15], quiet[15:]
    out += [f"  {ln}" for ln in first]
    last = rest[-15:]
    if last:
        hidden = len(rest) - len(last)
        out.append(
            f"last lines ({hidden} not shown between):"
            if hidden
            else "remaining lines:"
        )
        out += [f"  {ln}" for ln in last]
    return "\n".join(out)


def current_session(log: str, mode: str) -> str:
    """The journal text of the producer's current session: what follows its last start marker.

    A producer that restarts its cameras inside one process (it retries after a fault) logs the marker each time,
    so progress from an earlier attempt never counts toward the current one.
    """
    marker = "Capturing." if mode == "direct" else "Joined LiveKit room"
    return log.rsplit(marker, 1)[1] if marker in log else ""


def readiness(
    log: str, mode: str, sensors: list[int], direct_sensors: list[int]
) -> tuple[list[str], list[str]]:
    """(what is ready, what is not) for every configured sensor, judged on the latest status line of the session.

    direct mode: a sensor is up when its latest line shows frames and, if the config gave it a direct destination,
    that link connected; a stream with no destination is reported as capture-only. livekit mode (an older producer
    that publishes to LiveKit itself): the latest line shows a positive "livekit N published".
    """
    current = current_session(log, mode)
    ready, problems = [], []
    if not current:
        marker = "capturing" if mode == "direct" else "joined a LiveKit room"
        return ready, [f"the producer has not reported {marker} yet"]
    for n in sensors:
        if mode == "livekit":
            found = re.findall(rf"sensor {n}: [^\n]*livekit (\d+) published", current)
            if found and int(found[-1]) > 0:
                ready.append(f"sensor {n}: publishing to LiveKit")
            else:
                problems.append(f"sensor {n}: not publishing to LiveKit")
            continue
        lines = re.findall(rf"sensor {n}: (\d+) frames([^\n]*)", current)
        if not lines or int(lines[-1][0]) == 0:
            problems.append(f"sensor {n}: no frames captured")
            continue
        rest = lines[-1][1]
        if n not in direct_sensors:
            ready.append(f"sensor {n}: capturing (no direct destination configured)")
        elif re.search(r"direct: \S+ connected", rest) and "NOT connected" not in rest:
            ready.append(f"sensor {n}: capturing and connected to its destination")
        else:
            reason = re.search(r"direct: (\S+ NOT connected[^\n]*)", rest)
            problems.append(
                f"sensor {n}: capturing but {reason.group(1) if reason else 'its direct link is not connected'}"
            )
    return ready, problems


def wait_ready(
    since: str,
    timeout: int,
    sensors: list[int],
    mode: str = "direct",
    direct_sensors: list[int] | None = None,
) -> list[str]:
    """Wait until the current session is up for every configured sensor; return one line per sensor.

    "Up" is the producer's own current output (see readiness()), never a pid and never progress left over from an
    earlier attempt. A producer that is not up in time raises, with the sensors that were not.
    """
    direct_sensors = direct_sensors or []
    start = time.monotonic()
    deadline = start + timeout
    bar = Progress("waiting for cameras", timeout)
    log = ""
    while time.monotonic() < deadline:
        if (
            subprocess.run(
                ["systemctl", "is-active", "--quiet", SERVICE], check=False
            ).returncode
            != 0
        ):
            break
        log = tail(
            ["journalctl", "-u", SERVICE, "--since", since, "--no-pager", "-o", "cat"]
        )
        ready, problems = readiness(log, mode, sensors, direct_sensors)
        if not problems:
            bar.finish()
            return ready
        bar.label = f"waiting for cameras ({len(ready)}/{len(sensors)} up)"
        bar.update(int(time.monotonic() - start))
        time.sleep(2)
    if sys.stdout.isatty():
        print()
    print(failure_report(since, sensors), file=sys.stderr)
    # Judged on the log the last check saw, so the error names what that check found missing.
    _, problems = readiness(log, mode, sensors, direct_sensors)
    detail = f"; {'; '.join(problems)}" if problems else ""
    raise DeployError(
        f"the producer did not become ready{detail}; logs: journalctl -u {SERVICE}"
    )


# ---------------------------------------------------------------------------


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "target", help=f"host or user@host of the Orin (a bare host uses ${USER_ENV})"
    )
    ap.add_argument(
        "--config",
        help="producer options (JSON); required unless only setting up or checking",
    )
    ap.add_argument(
        "--install-prereqs",
        action="store_true",
        help="install the Orin's build prerequisites (asks for its sudo password), then stop without deploying",
    )
    ap.add_argument(
        "--logs",
        action="store_true",
        help="show the producer service's status and recent journal, then stop",
    )
    ap.add_argument(
        "--rescan-cameras",
        action="store_true",
        help='with "cameras": "auto", scan all 8 inputs instead of trying the cameras found last time first',
    )
    ap.add_argument(
        "--scan-cameras",
        action="store_true",
        help="try each camera link alone and report which has a camera (stops the producer service), then stop "
        "without deploying",
    )
    ap.add_argument(
        "--check-camera",
        action="store_true",
        help="stream both cameras for a few seconds and report, then stop without deploying; "
        "with --install-prereqs the prerequisites go first",
    )
    ap.add_argument(
        "--jobs", type=int, default=2, help="parallel compile jobs on the Orin"
    )
    ap.add_argument(
        "--service-user", help="account the service runs as (default: the ssh user)"
    )
    ap.add_argument(
        "--sdk-root",
        default="",
        help="directory on the Orin holding usr/src/jetson_multimedia_api and jetson_sipl_api (default: /)",
    )
    ap.add_argument(
        "--replace-unmanaged",
        action="store_true",
        help="stop a camera_plugin_sensing the service does not own (its exact PIDs are shown) instead of refusing",
    )
    ap.add_argument(
        "--source",
        metavar="DIR",
        help="deploy this git checkout (a worktree, say) instead of the one holding this script",
    )
    ap.add_argument(
        "--branch",
        metavar="NAME",
        help="deploy this branch's committed files (from --source, or this checkout), ignoring the working tree",
    )
    ap.add_argument(
        "--cmake-arg",
        action="append",
        default=[],
        metavar="ARG",
        help="extra CMake argument for the producer build, repeatable (for options only some source trees have)",
    )
    ap.add_argument("--no-activate", action="store_true", help="build and package only")
    ap.add_argument(
        "--no-setup",
        action="store_true",
        help="fail when the Orin lacks a prerequisite instead of installing it",
    )
    ap.add_argument(
        "--sipl-api",
        metavar="FILE",
        help="Jetson_SIPL_API_R<release>_aarch64.tbz2 to unpack on the Orin when its SIPL headers are missing "
        f"(default: the newest in {SDKM_DOWNLOADS})",
    )
    return ap


def remote_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="deploy_orin.py remote")
    sub = ap.add_subparsers(dest="step", required=True)
    pre = sub.add_parser("preflight")
    pre.add_argument("--sdk-root", default="")
    build = sub.add_parser("build")
    build.add_argument("--src", required=True)
    build.add_argument("--cmake-arg", action="append", default=[])
    build.add_argument("--stage", required=True)
    build.add_argument("--jobs", type=int, default=2)
    build.add_argument("--sdk-root", default="")
    apply_ = sub.add_parser("apply")
    for name in ("archive", "src", "manifest"):
        apply_.add_argument(f"--{name}", required=True)
    apply_.add_argument("--fresh", action="store_true")
    pack = sub.add_parser("package")
    pack.add_argument("--stage", required=True)
    pack.add_argument("--archive", required=True)
    act = sub.add_parser("activate")
    for name in ("archive", "prefix", "release", "unit", "owner"):
        act.add_argument(f"--{name}", required=True)
    act.add_argument("--ready-timeout", type=int, default=60)
    act.add_argument("--sensors", default="")
    act.add_argument("--direct-sensors", default="")
    act.add_argument("--mode", choices=MODES, default="direct")
    act.add_argument("--replace-unmanaged", action="store_true")
    act.add_argument("--platform-file", default="")
    return ap


def main(argv: list[str]) -> int:
    try:
        if argv and argv[0] == "remote":
            args = remote_parser().parse_args(argv[1:])
            {
                "preflight": remote_preflight,
                "build": remote_build,
                "package": remote_package,
                "apply": remote_apply,
                "activate": remote_activate,
            }[args.step](args)
        else:
            deploy(parser().parse_args(argv))
    except DeployError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
