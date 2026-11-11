#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Set up compiling the SENSING camera producer for a Jetson Orin on an x86 PC, without root.

    setup_orin_crosscompile.py doctor      what is present and what is missing
    setup_orin_crosscompile.py fetch       build the Orin sysroot and download the aarch64 compiler
    setup_orin_crosscompile.py syntax      type-check every producer source against the Orin headers (fast)
    setup_orin_crosscompile.py toolchain   write the CMake toolchain file
    setup_orin_crosscompile.py build       configure and build camera_plugin_sensing for aarch64

Everything lands under one directory (``--root``, default ``~/orin-cross-<L4T release>``); nothing outside it
changes and no step needs sudo. The sysroot is unpacked from NVIDIA's Jetson Linux package and the Multimedia and
SIPL API packages, which ``fetch`` takes from the SDK Manager download directory (``--sdk-downloads``) when they
are there and otherwise from NVIDIA's apt repository, checking the repository's SHA-256 for each. The compiler is
Arm's GNU toolchain, checked against the SHA-256 Arm publishes beside it.

``syntax`` needs only clang and the fetched headers, and catches the compile errors a build on the Orin would.
``build`` also needs the CUDA cross toolchain for aarch64 (``cuda-cross-sbsa``), CMake and network access for
the dependencies CMake fetches.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import shutil
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path

RELEASE = "39.2.1"
L4T_REPO = "https://repo.download.nvidia.com/jetson/common"
L4T_SUITE = "r39.2"
ARM_GCC_VERSION = "13.3.rel1"
ARM_GCC_NAME = f"arm-gnu-toolchain-{ARM_GCC_VERSION}-x86_64-aarch64-none-linux-gnu"
ARM_GCC_URL = f"https://developer.arm.com/-/media/Files/downloads/gnu/{ARM_GCC_VERSION}/binrel/{ARM_GCC_NAME}.tar.xz"
JETSON_LINUX = f"Jetson_Linux_R{RELEASE}_aarch64.tbz2"
SIPL_PACKAGE = f"Jetson_SIPL_API_R{RELEASE}_aarch64.tbz2"
MMAPI_PACKAGE = "nvidia-l4t-jetson-multimedia-api"
# The L4T packages inside the Jetson Linux archive that hold the libraries and headers the producer links against.
L4T_DEBS = (
    "nvidia-l4t-core",
    "nvidia-l4t-camera",
    "nvidia-l4t-multimedia-utils",
    "nvidia-l4t-multimedia-nvgpu",
    "nvidia-l4t-nvsci",
    "nvidia-l4t-3d-core",
    "nvidia-l4t-cuda-nvgpu",
    "nvidia-l4t-cuda-utils",
    "nvidia-l4t-wayland",
)
# Ubuntu's own arm64 packages for the C library and the libraries the build finds by name; the Orin runs Ubuntu 24.04,
# so these match its glibc. Indexes are read over https and each file is checked against its index's SHA-256.
UBUNTU_PORTS = "https://ports.ubuntu.com/ubuntu-ports"
UBUNTU_SUITES = ("noble-updates", "noble-security", "noble")
UBUNTU_DEBS = (
    "libc6",
    "libc6-dev",
    "linux-libc-dev",
    "libgcc-s1",
    "libstdc++6",
    "libegl1",
    "libegl-dev",
    "libglvnd0",
    "libv4l-0t64",
    "libv4l-dev",
    "libv4lconvert0t64",
)
# What the build needs from the sysroot: (name shown, glob under the sysroot).
SYSROOT_LIBS = (
    ("libnvsipl", "usr/lib/aarch64-linux-gnu/nvidia/libnvsipl.so"),
    ("libnvsipl_query", "usr/lib/aarch64-linux-gnu/nvidia/libnvsipl_query.so"),
    ("libnvscibuf", "usr/lib/aarch64-linux-gnu/nvidia/libnvscibuf.so*"),
    ("libnvscisync", "usr/lib/aarch64-linux-gnu/nvidia/libnvscisync.so*"),
    ("libnvbufsurface", "usr/lib/aarch64-linux-gnu/nvidia/libnvbufsurface.so"),
    (
        "libnvbufsurface_nvsci",
        "usr/lib/aarch64-linux-gnu/nvidia/libnvbufsurface_nvsci.so",
    ),
    (
        "libnvargus_socketclient",
        "usr/lib/aarch64-linux-gnu/nvidia/libnvargus_socketclient.so",
    ),
    ("libcuda", "opt/nvidia/l4t-gpu-libs/nvgpu/libcuda.so"),
    ("libc (crt1.o)", "usr/lib/aarch64-linux-gnu/crt1.o"),
    ("libEGL", "usr/lib/aarch64-linux-gnu/libEGL.so"),
    ("libv4l2", "usr/lib/aarch64-linux-gnu/libv4l2.so*"),
)
SYSROOT_HEADERS = (
    (
        "NvSIPLCamera.hpp (SIPL API)",
        "usr/src/jetson_sipl_api/sipl/include/NvSIPLCamera.hpp",
    ),
    (
        "nvbufsurface.h (Multimedia API)",
        "usr/src/jetson_multimedia_api/include/nvbufsurface.h",
    ),
    (
        "Argus/Argus.h (Multimedia API)",
        "usr/src/jetson_multimedia_api/argus/include/Argus/Argus.h",
    ),
)
CUDA_CROSS_ROOT = Path("/usr/local/cuda/targets/sbsa-linux")


class SetupError(Exception):
    """A step that cannot continue; the message says what to do."""


def say(message: str = "") -> None:
    """Progress and diagnostics go to stderr, like the other scripts here."""
    print(message, file=sys.stderr, flush=True)


def repo_root() -> Path:
    """The checkout this script lives in."""
    return Path(__file__).resolve().parents[4]


def sensing_dir() -> Path:
    return repo_root() / "src" / "plugins" / "sensing"


class Layout:
    """Where everything lives under the chosen root."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.sysroot = root / "sysroot"
        self.toolchain_dir = root / ARM_GCC_NAME
        self.cache = root / "downloads"
        self.cmake_file = root / "orin-aarch64.cmake"
        self.build_dir = root / "build"

    @property
    def gxx(self) -> Path:
        return self.toolchain_dir / "bin" / "aarch64-none-linux-gnu-g++"

    @property
    def gcc(self) -> Path:
        return self.toolchain_dir / "bin" / "aarch64-none-linux-gnu-gcc"

    @property
    def spdlog(self) -> Path:
        return self.root / "spdlog"


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, target: Path, expected_sha256: str | None) -> Path:
    """Fetch ``url`` to ``target`` (kept between runs) and verify its checksum; a bad file is never left in place."""
    if target.exists() and expected_sha256 and sha256_of(target) == expected_sha256:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    say(f"  downloading {url}")
    with urllib.request.urlopen(url, timeout=60) as response, partial.open("wb") as out:  # noqa: S310 - fixed https URLs
        shutil.copyfileobj(response, out, 1 << 20)
    if expected_sha256 and sha256_of(partial) != expected_sha256:
        partial.unlink()
        raise SetupError(
            f"{url}: the SHA-256 does not match the published {expected_sha256}"
        )
    partial.replace(target)
    return target


def fetch_text(url: str) -> str:
    with urllib.request.urlopen(url, timeout=60) as response:  # noqa: S310 - fixed https URLs
        return response.read().decode()


def apt_index() -> dict[str, dict[str, str]]:
    """NVIDIA's package index for this release: name -> {Version, Filename, SHA256} (the newest entry wins)."""
    import gzip

    raw = urllib.request.urlopen(
        f"{L4T_REPO}/dists/{L4T_SUITE}/main/binary-arm64/Packages.gz", timeout=60
    ).read()  # noqa: S310
    packages: dict[str, dict[str, str]] = {}
    for stanza in gzip.decompress(raw).decode().split("\n\n"):
        fields = dict(
            line.split(": ", 1)
            for line in stanza.splitlines()
            if ": " in line and not line.startswith(" ")
        )
        if "Package" in fields:
            packages.setdefault(fields["Package"], fields)
            if fields.get("Version", "") > packages[fields["Package"]].get(
                "Version", ""
            ):
                packages[fields["Package"]] = fields
    return packages


def ubuntu_index() -> dict[str, dict[str, str]]:
    """Ubuntu 24.04's arm64 package index: name -> fields, taking the entry from the first suite that lists it."""
    import gzip

    packages: dict[str, dict[str, str]] = {}
    for suite in UBUNTU_SUITES:
        for component in ("main", "universe"):
            url = f"{UBUNTU_PORTS}/dists/{suite}/{component}/binary-arm64/Packages.gz"
            raw = gzip.decompress(
                urllib.request.urlopen(url, timeout=120).read()
            ).decode()  # noqa: S310
            for stanza in raw.split("\n\n"):
                fields = dict(
                    line.split(": ", 1)
                    for line in stanza.splitlines()
                    if ": " in line and not line.startswith(" ")
                )
                if (
                    fields.get("Package") in UBUNTU_DEBS
                    and fields["Package"] not in packages
                ):
                    packages[fields["Package"]] = fields
    return packages


def extract_deb(deb: Path, destination: Path) -> None:
    """Unpack a .deb's files under ``destination`` without root (dpkg-deb reads it; nothing is installed)."""
    if not shutil.which("dpkg-deb"):
        raise SetupError(
            "dpkg-deb is needed to unpack the packages (it ships with every Debian or Ubuntu host)"
        )
    destination.mkdir(parents=True, exist_ok=True)
    subprocess.run(["dpkg-deb", "-x", str(deb), str(destination)], check=True)


def find_sdk_file(args: argparse.Namespace, pattern: str) -> Path | None:
    """The first file in the SDK Manager download directory whose name matches ``pattern``."""
    directory = Path(args.sdk_downloads).expanduser()
    matches = sorted(directory.glob(pattern)) if directory.is_dir() else []
    return matches[-1] if matches else None


def fetch_sysroot(args: argparse.Namespace, layout: Layout) -> None:
    """Unpack the L4T libraries, the Multimedia API and the SIPL API into the sysroot."""
    layout.sysroot.mkdir(parents=True, exist_ok=True)
    marker = layout.sysroot / ".fetched"
    if marker.exists() and not args.force:
        say(
            f"sysroot already built ({marker.read_text().strip()}); --force rebuilds it"
        )
        return
    # 1. The L4T libraries, from the Jetson Linux archive.
    archive = find_sdk_file(args, JETSON_LINUX)
    if archive is None:
        raise SetupError(
            f"{JETSON_LINUX} is not in {args.sdk_downloads}. Download it with NVIDIA SDK Manager (or from the Jetson "
            "Linux release page) and pass its folder as --sdk-downloads."
        )
    say(
        f"unpacking L4T libraries from {archive.name} (the archive is large; this takes a minute)"
    )
    staging = layout.cache / "l4t_debs"
    staging.mkdir(parents=True, exist_ok=True)
    wanted = {f"{name}_{RELEASE}-": name for name in L4T_DEBS}
    with tarfile.open(archive, "r:bz2") as tar:
        for member in tar:
            base = os.path.basename(member.name)
            if "l4t_deb_packages" in member.name and any(
                base.startswith(prefix) for prefix in wanted
            ):
                member.name = base
                tar.extract(member, staging)
    found = sorted(staging.glob("*.deb"))
    missing = [
        name
        for prefix, name in wanted.items()
        if not any(d.name.startswith(prefix) for d in found)
    ]
    if missing:
        raise SetupError(f"not in {archive.name}: {', '.join(missing)}")
    for deb in found:
        extract_deb(deb, layout.sysroot)
    # 1b. The C library and the libraries found by name, from Ubuntu 24.04 (what the Orin runs).
    index = ubuntu_index()
    absent = [name for name in UBUNTU_DEBS if name not in index]
    if absent:
        raise SetupError(f"not in Ubuntu's arm64 index: {', '.join(absent)}")
    for name in UBUNTU_DEBS:
        entry = index[name]
        deb = download(
            f"{UBUNTU_PORTS}/{entry['Filename']}",
            layout.cache / Path(entry["Filename"]).name,
            entry["SHA256"],
        )
        extract_deb(deb, layout.sysroot)
    # 2. The Multimedia API: the SDK Manager copy when present, else NVIDIA's apt repository, checksum verified.
    mmapi = find_sdk_file(args, f"{MMAPI_PACKAGE}_{RELEASE}-*_arm64.deb")
    if mmapi is None:
        entry = apt_index().get(MMAPI_PACKAGE)
        if not entry or not entry["Version"].startswith(RELEASE):
            raise SetupError(
                f"{MMAPI_PACKAGE} {RELEASE} is not in the {L4T_SUITE} repository"
            )
        mmapi = download(
            f"{L4T_REPO}/{entry['Filename']}",
            layout.cache / Path(entry["Filename"]).name,
            entry["SHA256"],
        )
    say(f"unpacking the Multimedia API from {mmapi.name}")
    extract_deb(mmapi, layout.sysroot)
    # 3. The SIPL API headers.
    sipl = find_sdk_file(args, SIPL_PACKAGE)
    if sipl is None:
        raise SetupError(
            f"{SIPL_PACKAGE} is not in {args.sdk_downloads}. Download it with NVIDIA SDK Manager or from the Jetson "
            "Linux release page."
        )
    say(f"unpacking the SIPL API from {sipl.name}")
    with tarfile.open(sipl, "r:bz2") as tar:
        tar.extractall(layout.sysroot, filter="data")
    fix_sysroot_layout(layout.sysroot)
    marker.write_text(f"L4T {RELEASE}\n")


def fix_sysroot_layout(sysroot: Path) -> None:
    """Ubuntu's libraries sit under usr/ with /lib a link to usr/lib; unpacking debs alone does not make that link."""
    link = sysroot / "lib"
    if not link.exists() and not link.is_symlink():
        link.symlink_to("usr/lib")


def fetch_compiler(layout: Layout, force: bool) -> None:
    """Download Arm's aarch64 GNU toolchain next to the sysroot and check it against Arm's published SHA-256."""
    if layout.gxx.exists() and not force:
        say(f"compiler already present: {layout.gxx}")
        return
    checksum_text = fetch_text(ARM_GCC_URL + ".sha256asc")
    match = re.search(r"\b([0-9a-f]{64})\b", checksum_text)
    if not match:
        raise SetupError(f"could not read the SHA-256 published for {ARM_GCC_NAME}")
    archive = download(
        ARM_GCC_URL, layout.cache / f"{ARM_GCC_NAME}.tar.xz", match.group(1)
    )
    say(f"unpacking {archive.name}")
    with tarfile.open(archive, "r:xz") as tar:
        tar.extractall(layout.root, filter="data")


def pinned_spdlog_commit() -> str:
    """The spdlog revision the repository's own build fetches, so the type check sees the same headers."""
    text = (repo_root() / "deps" / "third_party" / "CMakeLists.txt").read_text()
    match = re.search(r"spdlog\.git\s+GIT_TAG\s+([0-9a-f]{40})", text)
    if not match:
        raise SetupError(
            "could not read the pinned spdlog revision from deps/third_party/CMakeLists.txt"
        )
    return match.group(1)


def fetch_spdlog(layout: Layout, force: bool) -> None:
    """Fetch spdlog at the pinned commit with git, which checks the object hashes, and confirm the checkout."""
    commit = pinned_spdlog_commit()
    if (layout.spdlog / "include" / "spdlog" / "spdlog.h").exists() and not force:
        say(f"spdlog already present: {layout.spdlog}")
        return
    if not shutil.which("git"):
        raise SetupError("git is needed to fetch spdlog")
    say(f"fetching spdlog {commit[:12]}")
    shutil.rmtree(layout.spdlog, ignore_errors=True)
    layout.spdlog.mkdir(parents=True)
    for step in (
        ["git", "init", "-q"],
        [
            "git",
            "fetch",
            "-q",
            "--depth",
            "1",
            "https://github.com/gabime/spdlog.git",
            commit,
        ],
        ["git", "checkout", "-q", "FETCH_HEAD"],
    ):
        subprocess.run(step, cwd=layout.spdlog, check=True)
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=layout.spdlog,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if head != commit:
        raise SetupError(f"spdlog checked out {head}, not the pinned {commit}")


def cmd_fetch(args: argparse.Namespace) -> int:
    layout = Layout(Path(args.root).expanduser())
    layout.root.mkdir(parents=True, exist_ok=True)
    fetch_sysroot(args, layout)
    fetch_spdlog(layout, args.force)
    if not args.no_compiler:
        fetch_compiler(layout, args.force)
    say("done; run `doctor` to see what is still missing")
    return 0


def check_rows(layout: Layout) -> list[tuple[str, bool, str]]:
    """Every prerequisite as (name, present, hint)."""
    rows: list[tuple[str, bool, str]] = []
    for tool, hint in (
        ("clang++", "needed by `syntax`: sudo apt install clang"),
        ("cmake", "needed by `build`: sudo apt install cmake"),
        ("nvcc", "needed by `build`: install the CUDA toolkit"),
        ("dpkg-deb", "needed by `fetch`"),
    ):
        rows.append((tool, shutil.which(tool) is not None, hint))
    rows.append(
        (
            "aarch64 g++ (Arm toolchain)",
            layout.gxx.exists(),
            "run `fetch` (downloads Arm's toolchain)",
        )
    )
    rows.append(
        (
            "spdlog headers",
            (layout.spdlog / "include" / "spdlog" / "spdlog.h").exists(),
            "run `fetch`",
        )
    )
    rows.append(
        (
            "CUDA aarch64 cross target",
            (CUDA_CROSS_ROOT / "include" / "cuda_runtime.h").exists(),
            "needed by `build`: install cuda-cross-sbsa for your CUDA version",
        )
    )
    for label, pattern in SYSROOT_HEADERS + SYSROOT_LIBS:
        present = bool(list(layout.sysroot.glob(pattern)))
        rows.append((label, present, "run `fetch`" if not present else ""))
    return rows


def cmd_doctor(args: argparse.Namespace) -> int:
    layout = Layout(Path(args.root).expanduser())
    say(f"root: {layout.root}")
    rows = check_rows(layout)
    for name, present, hint in rows:
        say(
            f"  [{'ok' if present else '--'}] {name}"
            + ("" if present or not hint else f"  ({hint})")
        )
    return 0 if all(present for _, present, _ in rows) else 1


def include_dirs(layout: Layout, args: argparse.Namespace) -> list[Path]:
    """The include path the producer's sources need, in the order the build uses."""
    sensing = sensing_dir()
    sysroot = layout.sysroot
    dirs = [
        sensing / "core" / "inc",
        sensing / "livekit" / "inc",
        sensing / "third_party" / "jetson_multimedia_api" / "include",
        sysroot / "usr/src/jetson_multimedia_api/include",
        sysroot / "usr/src/jetson_multimedia_api/argus/include",
        sysroot / "usr/src/jetson_sipl_api/sipl/include",
        sysroot / "usr/src/jetson_sipl_api/sipl/include/query/include",
        sysroot / "usr/src/jetson_sipl_api/sipl/include/nvsci",
        repo_root() / "src/core/log_bridge/cpp/inc",
        Path(args.cuda).expanduser() / "include",
    ]
    if args.spdlog_include:
        dirs.append(Path(args.spdlog_include).expanduser())
    return dirs


def find_spdlog(explicit: str | None, layout: Layout) -> str | None:
    """A spdlog include directory: the one given, else the copy `fetch` made."""
    if explicit:
        return explicit
    fetched = layout.spdlog / "include"
    return str(fetched) if (fetched / "spdlog" / "spdlog.h").exists() else None


def cmd_syntax(args: argparse.Namespace) -> int:
    """Type-check every producer C++ source with clang against the sysroot's headers; no linking, no CUDA."""
    layout = Layout(Path(args.root).expanduser())
    clang = shutil.which("clang++")
    if not clang:
        raise SetupError("clang++ is not installed (sudo apt install clang)")
    if not all(list(layout.sysroot.glob(pattern)) for _, pattern in SYSROOT_HEADERS):
        raise SetupError("the Orin headers are missing; run `fetch` first")
    args.spdlog_include = find_spdlog(args.spdlog_include, layout)
    if not args.spdlog_include:
        raise SetupError(
            "no spdlog headers found; run `fetch`, or pass --spdlog-include DIR"
        )
    sensing = sensing_dir()
    sources = sorted(
        [
            *sensing.glob("core/*.cpp"),
            *sensing.glob("app/*.cpp"),
            *sensing.glob("third_party/**/classes/*.cpp"),
        ]
    )
    flags = ["-std=c++17", "-fsyntax-only", "-Wno-everything", "-DSENSING_HAS_SIPL"]
    if (sensing / "core" / "argus_capture.cpp").exists():
        flags.append("-DSENSING_HAS_ARGUS")
    for directory in include_dirs(layout, args):
        flags += ["-I", str(directory)]
    failed = 0
    for source in sources:
        result = subprocess.run(
            [clang, *flags, str(source)], capture_output=True, text=True, check=False
        )
        errors = [
            line
            for line in result.stderr.splitlines()
            if "error" in line or "fatal" in line
        ]
        if result.returncode != 0:
            failed += 1
            say(f"FAIL {source.relative_to(repo_root())}")
            for line in errors[:8]:
                say(f"     {line}")
        elif args.verbose:
            say(f"ok   {source.relative_to(repo_root())}")
    say(f"{len(sources) - failed} of {len(sources)} sources type-check")
    return 1 if failed else 0


def cmd_toolchain(args: argparse.Namespace) -> int:
    layout = Layout(Path(args.root).expanduser())
    if not layout.gxx.exists():
        raise SetupError("the aarch64 compiler is missing; run `fetch` first")
    cuda_cross = Path(args.cuda).expanduser()
    # CMake finds a cross target's device library through <sysroot>/usr/lib/cuda (Debian's cross layout).
    cuda_link = layout.sysroot / "usr" / "lib" / "cuda"
    if not cuda_link.exists() and not cuda_link.is_symlink():
        cuda_link.symlink_to(cuda_cross.resolve())
    # The nvcc must come from the same toolkit as the cross target; a distribution's nvcc on PATH is another version.
    toolkit_nvcc = cuda_cross.resolve().parents[1] / "bin" / "nvcc"
    cuda_nvcc = (
        str(toolkit_nvcc) if toolkit_nvcc.exists() else (shutil.which("nvcc") or "nvcc")
    )
    extra_roots = ""
    layout.cmake_file.write_text(
        f"""# Generated by setup_orin_crosscompile.py; regenerate rather than edit.
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)
set(CMAKE_C_COMPILER   "{layout.gcc}")
set(CMAKE_CXX_COMPILER "{layout.gxx}")
set(CMAKE_SYSROOT "{layout.sysroot}")
# The Tegra libraries sit beside their own dependencies; shared libraries' own needs are not re-resolved on the host.
set(_orin_libs "{layout.sysroot}/usr/lib/aarch64-linux-gnu/nvidia" "{layout.sysroot}/usr/lib/aarch64-linux-gnu" "{layout.sysroot}/opt/nvidia/l4t-gpu-libs/nvgpu")
foreach(_dir IN LISTS _orin_libs)
  string(APPEND CMAKE_EXE_LINKER_FLAGS_INIT " -Wl,-rpath-link,${{_dir}} -L${{_dir}}")
endforeach()
string(APPEND CMAKE_EXE_LINKER_FLAGS_INIT " -Wl,--allow-shlib-undefined")
# Arm's compiler looks for the C runtime and glibc's architecture headers in the plain usr/lib and usr/include; Ubuntu
# keeps them in the aarch64-linux-gnu subdirectories.
foreach(_lang C CXX)
  string(APPEND CMAKE_${{_lang}}_FLAGS_INIT " -B{layout.sysroot}/usr/lib/aarch64-linux-gnu -isystem {layout.sysroot}/usr/include/aarch64-linux-gnu")
endforeach()
string(APPEND CMAKE_EXE_LINKER_FLAGS_INIT " -B{layout.sysroot}/usr/lib/aarch64-linux-gnu")
# The CUDA link rule does not add --sysroot on its own, and glibc's linker scripts name /lib/... absolutely.
string(APPEND CMAKE_EXE_LINKER_FLAGS_INIT " --sysroot={layout.sysroot}")
# Libraries and headers come from the Orin sysroot; build tools (cmake, python) from this host.
set(CMAKE_FIND_ROOT_PATH "{layout.sysroot}" "{cuda_cross}"{extra_roots})
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
# The fetched LiveKit SDK sits in the build tree, outside the sysroot; its config files are searched by explicit path.
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE BOTH)
# nvcc compiles the device code for the Orin's GPU and hands the host code to the aarch64 compiler.
set(CMAKE_CUDA_COMPILER "{cuda_nvcc}")
set(CMAKE_CUDA_HOST_COMPILER "{layout.gxx}")
set(CMAKE_CUDA_FLAGS_INIT "--target-directory sbsa-linux -isystem {layout.sysroot}/usr/include/aarch64-linux-gnu")
set(CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES "{cuda_cross}/include")
set(CMAKE_CUDA_ARCHITECTURES 87)
set(CUDAToolkit_ROOT "{cuda_cross}")
"""
    )
    say(f"wrote {layout.cmake_file}")
    return 0


def write_project(layout: Layout) -> Path:
    """A small top-level project that builds only the producer.

    The repository's own top-level CMake pulls in the OpenXR SDK, Python and the viewer, which need an aarch64
    desktop stack that has no place here; the producer needs only spdlog and the logging bridge from outside its
    own directory.
    """
    project = layout.root / "project"
    project.mkdir(parents=True, exist_ok=True)
    (project / "CMakeLists.txt").write_text(
        f"""# Generated by setup_orin_crosscompile.py; regenerate rather than edit.
cmake_minimum_required(VERSION 3.24)
project(sensing_cross LANGUAGES C CXX)
include(FetchContent)
FetchContent_Declare(spdlog SOURCE_DIR "{layout.spdlog}")
set(SPDLOG_BUILD_EXAMPLE OFF CACHE BOOL "" FORCE)
set(SPDLOG_BUILD_TESTS OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(spdlog)
add_subdirectory("{repo_root() / "src/core/log_bridge/cpp"}" log_bridge)
add_subdirectory("{sensing_dir()}" sensing)
"""
    )
    return project


def cmd_build(args: argparse.Namespace) -> int:
    """Configure and build ``camera_plugin_sensing`` for aarch64 into ``<root>/build``."""
    layout = Layout(Path(args.root).expanduser())
    for tool in ("cmake", "nvcc"):
        if not shutil.which(tool):
            raise SetupError(f"{tool} is not installed")
    if not (CUDA_CROSS_ROOT / "include" / "cuda_runtime.h").exists():
        raise SetupError(
            "the CUDA aarch64 cross target is missing (install cuda-cross-sbsa for your CUDA version)"
        )
    cmd_toolchain(args)
    configure = [
        "cmake",
        "-S",
        str(write_project(layout)),
        "-B",
        str(layout.build_dir),
        f"-DCMAKE_TOOLCHAIN_FILE={layout.cmake_file}",
        "-DCMAKE_BUILD_TYPE=Release",
        f"-DSIPL_API_DIR={layout.sysroot}/usr/src/jetson_sipl_api",
        f"-DJETSON_MMAPI_DIR={layout.sysroot}/usr/src/jetson_multimedia_api",
        *args.cmake_arg,
    ]
    say("configuring: " + " ".join(configure))
    if subprocess.run(configure, check=False).returncode != 0:
        return 1
    build = [
        "cmake",
        "--build",
        str(layout.build_dir),
        "--target",
        args.target,
        "-j",
        str(args.jobs),
    ]
    say("building: " + " ".join(build))
    return subprocess.run(build, check=False).returncode


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--root",
        default=f"~/orin-cross-{RELEASE}",
        help="directory holding the sysroot and compiler",
    )
    ap.add_argument(
        "--sdk-downloads",
        default="~/Downloads/nvidia/sdkm_downloads",
        help="folder with the Jetson Linux and SIPL API archives from SDK Manager",
    )
    ap.add_argument(
        "--cuda", default=str(CUDA_CROSS_ROOT), help="the CUDA aarch64 target directory"
    )
    ap.add_argument(
        "--spdlog-include",
        help="spdlog include directory (default: the copy `fetch` makes)",
    )
    sub = ap.add_subparsers(dest="command", required=True)
    sub.add_parser(
        "doctor", help="show what is present and what is missing"
    ).set_defaults(run=cmd_doctor)
    fetch = sub.add_parser(
        "fetch", help="build the sysroot and download the aarch64 compiler"
    )
    fetch.add_argument(
        "--force", action="store_true", help="redo steps that look finished"
    )
    fetch.add_argument(
        "--no-compiler", action="store_true", help="sysroot only (enough for `syntax`)"
    )
    fetch.set_defaults(run=cmd_fetch)
    syntax = sub.add_parser(
        "syntax", help="type-check the producer's sources against the Orin headers"
    )
    syntax.add_argument(
        "-v", "--verbose", action="store_true", help="also list the files that pass"
    )
    syntax.set_defaults(run=cmd_syntax)
    sub.add_parser("toolchain", help="write the CMake toolchain file").set_defaults(
        run=cmd_toolchain
    )
    build = sub.add_parser("build", help="cross-build camera_plugin_sensing")
    build.add_argument("--target", default="camera_plugin_sensing")
    build.add_argument("--jobs", type=int, default=os.cpu_count() or 4)
    build.add_argument(
        "--cmake-arg",
        action="append",
        default=[],
        help="extra CMake argument, repeatable",
    )
    build.set_defaults(run=cmd_build)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        return args.run(args)
    except SetupError as exc:
        say(f"error: {exc}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
