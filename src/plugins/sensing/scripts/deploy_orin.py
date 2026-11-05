#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build the camera producer on an Orin from this checkout, package it, and run it as a service.

Run on the desktop:

    deploy_orin.py user@orin --backend sipl --config orin.json

The config is JSON with the producer's own options, nothing else:

    {"livekit_url": "wss://livekit.example.org",
     "token_file": "/etc/isaaccapture/join.token",
     "args": ["--add-stream=sensor=0,livekit=sensing-left", "--add-stream=sensor=1,livekit=sensing-right"]}

The token file stays on the Orin, outside the release; only its path is passed. Nothing is built, stopped or
replaced until the target passes its checks. A build or transfer failure leaves the running producer alone.
The same file runs on the target (`deploy_orin.py remote ...`) from the source snapshot it copied there.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
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
SCRIPT_REL = "src/plugins/sensing/scripts/deploy_orin.py"
UNIT_TEMPLATE = Path(__file__).with_name("isaaccapture-producer.service.in")
CACHE = "~/.cache/isaaccapture-deploy"
CONFIG_KEYS = {"livekit_url", "token_file", "args"}
# The producer's connection options come from the config's own fields, not from `args`.
RESERVED_ARGS = ("--livekit-url", "--livekit-token-file")


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


def load_config(path: Path) -> dict:
    try:
        config = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise DeployError(f"cannot read the config {path}: {exc}") from exc
    if (
        not isinstance(config, dict)
        or set(config) - CONFIG_KEYS
        or not {"livekit_url", "token_file"} <= set(config)
    ):
        raise DeployError(
            f"{path}: expected an object with livekit_url, token_file and optional args only"
        )
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
    if not all(
        isinstance(config[k], str) and config[k] for k in ("livekit_url", "token_file")
    ):
        raise DeployError(
            f"{path}: livekit_url and token_file must be non-empty strings"
        )
    if not config["token_file"].startswith("/"):
        raise DeployError(f"{path}: token_file must be an absolute path on the Orin")
    return config


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
        f"--livekit-url={config['livekit_url']}",
        f"--livekit-token-file={config['token_file']}",
        *config.get("args", []),
    ]
    exec_start = " ".join(systemd_quote(w) for w in words)
    fields = {
        "SERVICE_USER": user,
        "SERVICE_GROUPS": groups,
        "PREFIX": prefix,
        "EXEC_START": exec_start,
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


def make_snapshot(root: Path, dest: Path) -> dict:
    """Archive the tracked files as they are on disk, including tracked local edits; nothing untracked."""
    listing = subprocess.run(
        ["git", "ls-files", "-z", "--recurse-submodules"],
        cwd=root,
        capture_output=True,
        check=True,
    ).stdout
    files = sorted(
        f for f in listing.decode().split("\0") if f and (root / f).is_file()
    )
    bar = Progress("packing", len(files))
    with tarfile.open(dest, "w:gz") as tar:
        for done, name in enumerate(files, 1):
            tar.add(root / name, arcname=name, recursive=False)
            if done % 200 == 0:
                bar.update(done)
    bar.finish()
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    )
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    return {
        "base_revision": head.stdout.strip(),
        "dirty": bool(dirty),
        "sha256": sha256_file(dest),
    }


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

    def get(self, remote: str, local: Path) -> None:
        if (
            self._run(
                "scp", [*self._scp_quiet(), f"{self.host}:{remote}", str(local)]
            ).returncode
            != 0
        ):
            raise DeployError(f"copying {remote} from {self.host} failed")

    def close(self) -> None:
        self._run("ssh", ["-O", "exit", self.host], capture_output=True)
        shutil.rmtree(self._dir, ignore_errors=True)


def deploy(args: argparse.Namespace) -> None:
    config = load_config(Path(args.config))
    target = Target(target_host(args.target))
    try:
        run_deploy(args, config, target)
    finally:
        target.close()


def run_deploy(args: argparse.Namespace, config: dict, target: Target) -> None:
    root = repo_root()
    step(1, f"Connecting to {target.host}")
    target.validate_connection()
    user = args.service_user or target.capture("id -un")
    unit = render_unit(config, args.prefix, user, args.service_groups)

    step(2, f"Preflight on {target.host}")
    cache = target.capture(f"mkdir -p {CACHE} && cd {CACHE} && pwd")
    if not target.readable(config["token_file"]):
        raise DeployError(
            f"the token file {config['token_file']} is not readable on the Orin"
        )

    with tempfile.TemporaryDirectory() as tmp:
        snapshot_path = Path(tmp) / "snapshot.tar.gz"
        step(3, "Snapshot and upload")
        meta = make_snapshot(root, snapshot_path)
        release = f"{time.strftime('%Y%m%d-%H%M%S')}-{meta['base_revision'][:9]}{'-dirty' if meta['dirty'] else ''}"
        print(
            f"    base {meta['base_revision'][:12]}, dirty={meta['dirty']}, sha256 {meta['sha256'][:16]}"
        )
        src = f"{cache}/src-{meta['sha256'][:16]}"
        target.put(snapshot_path, f"{cache}/snapshot.tar.gz")
        target.ssh(
            f"rm -rf {src} && mkdir -p {src} && tar -xzf {cache}/snapshot.tar.gz -C {src}"
        )
        remote = f"python3 {src}/{SCRIPT_REL} remote"
        target.ssh(
            f"{remote} preflight --backend {args.backend} --sdk-root {shlex.quote(args.sdk_root)}"
        )

        step(4, "Build and package on the Orin (the running producer is untouched)")
        stage, archive = (
            f"{cache}/stage-{release}",
            f"{cache}/producer-{release}.tar.gz",
        )
        target.ssh(
            f"{remote} build --src {src} --backend {args.backend} --stage {stage} --jobs {args.jobs} "
            f"--sdk-root {shlex.quote(args.sdk_root)}"
        )
        manifest = {
            **meta,
            "backend": args.backend,
            "release": release,
            "arch": "aarch64",
            "config": args.config,
        }
        manifest_path = Path(tmp) / "MANIFEST.json"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        target.put(manifest_path, f"{stage}/MANIFEST.json")
        target.ssh(f"{remote} package --stage {stage} --archive {archive}")
        if args.fetch_archive:
            Path(args.fetch_archive).mkdir(parents=True, exist_ok=True)
            target.get(archive, Path(args.fetch_archive) / Path(archive).name)

        unit_path = Path(tmp) / SERVICE
        unit_path.write_text(unit)
        target.put(unit_path, f"{cache}/{SERVICE}")

    if args.no_activate:
        print(f"==> Built release {release}; not activated (--no-activate)")
        return
    step(5, "Install and replace the running producer (needs sudo on the Orin)")
    activate = (
        f"{remote} activate --archive {archive} --prefix {args.prefix} --release {release} "
        f"--unit {cache}/{SERVICE} --owner {user} --ready-timeout {args.ready_timeout}"
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
    """The Multimedia API and SIPL header trees: under --sdk-root when given, else the system locations."""
    base = Path(sdk_root) / "usr/src" if sdk_root else Path("/usr/src")
    return base / "jetson_multimedia_api", base / "jetson_sipl_api"


def remote_preflight(args: argparse.Namespace) -> None:
    if os.uname().machine != "aarch64":
        raise DeployError(f"the target is {os.uname().machine}, not aarch64")
    release_file = Path("/etc/nv_tegra_release")
    if not release_file.exists():
        raise DeployError("no /etc/nv_tegra_release: this is not a Jetson L4T system")
    print(f"  L4T: {release_file.read_text().splitlines()[0]}")
    for tool in ("cmake", "g++", "systemctl"):
        if (
            subprocess.run(["which", tool], capture_output=True, check=False).returncode
            != 0
        ):
            raise DeployError(f"{tool} is not installed")
    if (
        not Path("/usr/local/cuda/bin/nvcc").exists()
        and subprocess.run(
            ["which", "nvcc"], capture_output=True, check=False
        ).returncode
    ):
        raise DeployError("nvcc not found (CUDA toolkit)")
    stat = os.statvfs(Path.home())
    if stat.f_bavail * stat.f_frsize < 4 << 30:
        raise DeployError("less than 4 GB free in the home directory")
    mmapi, sipl_api = sdk_dirs(args.sdk_root)
    if not mmapi.is_dir():
        raise DeployError(
            f"the Jetson Multimedia API is missing at {mmapi}; run src/plugins/sensing/scripts/setup.sh "
            "or pass --sdk-root"
        )
    if args.backend == "sipl" and not sipl_api.is_dir():
        raise DeployError(
            f"SIPL headers not found at {sipl_api}; run src/plugins/sensing/scripts/setup.sh or pass --sdk-root"
        )
    if args.backend == "argus" and not (mmapi / "argus").is_dir():
        raise DeployError(f"Argus headers not found under {mmapi}/argus")
    print(f"  preflight ok for backend {args.backend}")


def remote_build(args: argparse.Namespace) -> None:
    """Configure and build only the producer, then stage its install component.

    The build directory persists between deployments and is discarded when the toolchain, L4T release or backend
    changes.
    """
    cache = Path(os.path.expanduser(CACHE))
    build_dir = cache / f"build-{args.backend}"
    stamp = "|".join(
        [
            Path("/etc/nv_tegra_release").read_text().splitlines()[0],
            tail(["cmake", "--version"]).splitlines()[0],
            tail(["g++", "-dumpfullversion"]),
        ]
    )
    stamp_file = build_dir / ".stamp"
    if not stamp_file.exists() or stamp_file.read_text() != stamp:
        print("  new toolchain, release or backend: clean build directory")
        run(["rm", "-rf", str(build_dir)])
        build_dir.mkdir(parents=True)
        stamp_file.write_text(stamp)
    sipl, argus = ("ON", "OFF") if args.backend == "sipl" else ("OFF", "ON")
    mmapi, sipl_api = sdk_dirs(args.sdk_root)
    env = {**os.environ, "PATH": f"/usr/local/cuda/bin:{os.environ.get('PATH', '')}"}
    sdk_flags = [f"-DJETSON_MMAPI_DIR={mmapi}"]
    if args.backend == "sipl":
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
            f"-DSENSING_WITH_SIPL={sipl}",
            f"-DSENSING_WITH_ARGUS={argus}",
            "-DBUILD_TESTING=OFF",
            "-DBUILD_EXAMPLES=OFF",
            "-DBUILD_PYTHON_BINDINGS=OFF",
            "-DBUILD_VIZ=OFF",
            "-DENABLE_CLANG_FORMAT_CHECK=OFF",
            "-DCMAKE_CXX_FLAGS=-I/usr/local/cuda/include",
            *sdk_flags,
            *link_flags,
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
    run(
        [
            "cmake",
            "--install",
            str(build_dir),
            "--component",
            "producer",
            "--prefix",
            args.stage,
        ]
    )
    stage = Path(args.stage)
    binary = stage / BIN_REL
    if not binary.exists():
        raise DeployError(f"the staged tree has no {BIN_REL}")
    missing = [
        ln for ln in tail(["ldd", str(binary)]).splitlines() if "not found" in ln
    ]
    if missing:
        raise DeployError(
            "the producer has unresolved shared libraries on this target:\n"
            + "\n".join(missing)
        )
    print(f"  staged {sum(1 for p in stage.rglob('*') if p.is_file())} files")


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
        with tarfile.open(args.archive) as tar:
            tar.extractall(dest, filter="data")
        for line in (dest / "SHA256SUMS").read_text().splitlines():
            digest, name = line.split("  ", 1)
            if sha256_file(dest / name) != digest:
                raise DeployError(
                    f"checksum mismatch for {name} in release {args.release}"
                )
    except BaseException:
        shutil.rmtree(dest, ignore_errors=True)
        raise
    run(["chown", "-R", f"{args.owner}:", str(prefix)])
    print(f"  installed release {args.release}")

    unit_target = Path("/etc/systemd/system") / SERVICE
    unit_text = Path(args.unit).read_text()
    if not unit_target.exists() or unit_target.read_text() != unit_text:
        unit_target.write_text(unit_text)
        run(["systemctl", "daemon-reload"])
        run(["systemctl", "enable", SERVICE], capture_output=True)
        print("  service unit updated")
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
    run(["systemctl", "start", SERVICE])
    wait_ready(since, args.ready_timeout)
    print(f"  producer is publishing (release {args.release})")


def wait_ready(since: str, timeout: int) -> None:
    """Ready means the producer joined the room and has published frames, as its own log says.

    The two patterns come from the LiveKit session's join message and the periodic stats line in
    sensing_camera.cpp; a pid alone is not readiness.
    """
    deadline = time.monotonic() + timeout
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
        if "Joined LiveKit room" in log and re.search(
            r"livekit [1-9][0-9]* published", log
        ):
            return
        time.sleep(2)
    print(
        tail(["journalctl", "-u", SERVICE, "--since", since, "--no-pager", "-n", "40"]),
        file=sys.stderr,
    )
    raise DeployError(
        f"the producer did not become ready; logs: journalctl -u {SERVICE}"
    )


# ---------------------------------------------------------------------------


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "target", help=f"host or user@host of the Orin (a bare host uses ${USER_ENV})"
    )
    ap.add_argument("--backend", choices=("sipl", "argus"), required=True)
    ap.add_argument("--config", required=True, help="producer options (JSON)")
    ap.add_argument("--prefix", default="/opt/isaaccapture")
    ap.add_argument(
        "--jobs", type=int, default=2, help="parallel compile jobs on the Orin"
    )
    ap.add_argument(
        "--service-user", help="account the service runs as (default: the ssh user)"
    )
    ap.add_argument(
        "--service-groups",
        default="video",
        help="supplementary groups for device access",
    )
    ap.add_argument("--ready-timeout", type=int, default=60)
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
    ap.add_argument("--no-activate", action="store_true", help="build and package only")
    ap.add_argument(
        "--fetch-archive",
        metavar="DIR",
        help="also copy the package back to this directory",
    )
    return ap


def remote_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="deploy_orin.py remote")
    sub = ap.add_subparsers(dest="step", required=True)
    pre = sub.add_parser("preflight")
    pre.add_argument("--backend", required=True)
    pre.add_argument("--sdk-root", default="")
    build = sub.add_parser("build")
    build.add_argument("--src", required=True)
    build.add_argument("--backend", required=True)
    build.add_argument("--stage", required=True)
    build.add_argument("--jobs", type=int, default=2)
    build.add_argument("--sdk-root", default="")
    pack = sub.add_parser("package")
    pack.add_argument("--stage", required=True)
    pack.add_argument("--archive", required=True)
    act = sub.add_parser("activate")
    for name in ("archive", "prefix", "release", "unit", "owner"):
        act.add_argument(f"--{name}", required=True)
    act.add_argument("--ready-timeout", type=int, default=60)
    act.add_argument("--replace-unmanaged", action="store_true")
    return ap


def main(argv: list[str]) -> int:
    try:
        if argv and argv[0] == "remote":
            args = remote_parser().parse_args(argv[1:])
            {
                "preflight": remote_preflight,
                "build": remote_build,
                "package": remote_package,
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
