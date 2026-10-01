# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A synthetic project, build tree and wheel for the SBOM collector's tests.

The fixture has to be a build tree: a file API reply, fetched source trees, an
SDK tarball, a staged package directory. Generated rather than built, because the
gates need inputs a working build cannot produce -- a member from nowhere, a
component with no license text, a tampered wheel -- and because the suite must
run with no compiler, no network and no configured CMake.

Keep every reader anchored to something real as well, or a fixture the collector
agrees with proves nothing: the codemodel reader against this repo's own build
tree, the ELF reader against a system library, the matcher against the fetched
SPDX corpus.
"""

from __future__ import annotations

import base64
import csv
import gzip
import hashlib
import io
import os
import json
import struct
import subprocess
import tarfile
import zipfile
from pathlib import Path

CONFIG = "Release"
WHEEL_NAME = "isaaccapture-0.4.0-cp312-cp312-manylinux_2_35_x86_64.whl"
DIST_INFO = "isaaccapture-0.4.0.dist-info"
SDK_ARCHIVE = "deps/vendor/demosdk-1.2.0-linux-amd64.tar.gz"
SDK_DEMO_BUILD_ID = "1111111111111111111111111111111111111111"

# ==============================================================================
# ELF
# ==============================================================================

_PT_LOAD = 1
_PT_DYNAMIC = 2
_PT_NOTE = 4
_NT_GNU_BUILD_ID = 3
_DT_NULL = 0
_DT_NEEDED = 1
_DT_STRTAB = 5
_DT_STRSZ = 10
_DT_SONAME = 14
_EM_X86_64 = 62
_EHDR_SIZE = 64
_PHDR_SIZE = 56
_BASE_VADDR = 0x400000


def _build_id_note(build_id: str) -> bytes:
    """A .note.gnu.build-id in the form the linker writes it."""
    name = b"GNU\0"
    desc = bytes.fromhex(build_id)
    padding = b"\0" * (-len(desc) % 4)
    return (
        struct.pack("<III", len(name), len(desc), _NT_GNU_BUILD_ID)
        + name
        + desc
        + padding
    )


def elf_shared_object(
    soname: str,
    needed: tuple[str, ...] = (),
    padding: bytes = b"",
    build_id: str | None = None,
) -> bytes:
    """A 64-bit little-endian ELF carrying a readable dynamic segment.

    With `build_id`, it also carries the note the linker writes -- which is what
    survives a repair tool rewriting SONAME and RPATH.
    """
    strings = [b"\0"]  # index 0 is the empty string, per the ELF string-table format
    offsets: dict[str, int] = {}
    cursor = 1
    for value in (soname, *needed):
        if value in offsets:
            continue
        offsets[value] = cursor
        encoded = value.encode("utf-8") + b"\0"
        strings.append(encoded)
        cursor += len(encoded)
    strtab = b"".join(strings)

    entries = [(_DT_SONAME, offsets[soname])]
    entries += [(_DT_NEEDED, offsets[name]) for name in needed]

    note = _build_id_note(build_id) if build_id else b""
    segments = 3 if note else 2
    header_size = _EHDR_SIZE + segments * _PHDR_SIZE
    dynamic_offset = header_size
    dynamic_size = (len(entries) + 3) * 16
    strtab_offset = dynamic_offset + dynamic_size
    note_offset = strtab_offset + len(strtab)
    total = note_offset + len(note) + len(padding)

    entries.append((_DT_STRTAB, _BASE_VADDR + strtab_offset))
    entries.append((_DT_STRSZ, len(strtab)))
    entries.append((_DT_NULL, 0))

    image = bytearray(total)
    image[0:4] = b"\x7fELF"
    image[4] = 2  # ELFCLASS64
    image[5] = 1  # ELFDATA2LSB
    image[6] = 1  # EV_CURRENT
    struct.pack_into("<H", image, 0x10, 3)  # ET_DYN
    struct.pack_into("<H", image, 0x12, _EM_X86_64)
    struct.pack_into("<I", image, 0x14, 1)
    struct.pack_into("<Q", image, 0x20, _EHDR_SIZE)
    struct.pack_into("<H", image, 0x34, _EHDR_SIZE)
    struct.pack_into("<H", image, 0x36, _PHDR_SIZE)
    struct.pack_into("<H", image, 0x38, segments)

    def phdr(index: int, kind: int, offset: int, size: int) -> None:
        base = _EHDR_SIZE + index * _PHDR_SIZE
        struct.pack_into("<I", image, base, kind)
        struct.pack_into("<I", image, base + 4, 4)  # PF_R
        struct.pack_into(
            "<QQQ", image, base + 8, offset, _BASE_VADDR + offset, _BASE_VADDR + offset
        )
        struct.pack_into("<QQ", image, base + 32, size, size)
        struct.pack_into("<Q", image, base + 48, 8)

    phdr(0, _PT_LOAD, 0, total)
    phdr(1, _PT_DYNAMIC, dynamic_offset, len(entries) * 16)
    if note:
        phdr(2, _PT_NOTE, note_offset, len(note))

    for index, (tag, value) in enumerate(entries):
        struct.pack_into("<qQ", image, dynamic_offset + index * 16, tag, value)
    image[strtab_offset : strtab_offset + len(strtab)] = strtab
    if note:
        image[note_offset : note_offset + len(note)] = note
    if padding:
        image[note_offset + len(note) :] = padding
    return bytes(image)


# ==============================================================================
# License texts, taken from the same corpus the collector matches against
# ==============================================================================


_LICENSE_DATA: Path | None = None

# Enough of the SPDX list for the fixtures; the rest is copied on demand so the
# synthetic build tree stays small.
FIXTURE_LICENSES = (
    "Apache-2.0",
    "BSD-2-Clause",
    "BSD-3-Clause",
    "BSL-1.0",
    "CC-BY-4.0",
    "ECL-2.0",
    "MIT",
    "MIT-0",
    "Zlib",
)


def set_license_data(path: Path) -> None:
    """Point the fixtures at the SPDX json/ directory the build materialized."""
    global _LICENSE_DATA
    _LICENSE_DATA = path


def license_data() -> Path:
    if _LICENSE_DATA is None:
        raise RuntimeError("set_license_data() has not been called")
    return _LICENSE_DATA


def spdx_text(license_id: str) -> str:
    """The canonical text of a license, from the data the collector reads."""
    payload = json.loads(
        (license_data() / "details" / f"{license_id}.json").read_text(encoding="utf-8")
    )
    return payload["licenseText"]


PROPRIETARY_LICENSE = """\
DEMO CORP SOFTWARE LICENSE AGREEMENT

This agreement governs your use of the Demo SDK. You may redistribute the
runtime libraries with your application provided this notice accompanies them.
The software is provided as is, without warranty of any kind.
"""


# ==============================================================================
# Workspace
# ==============================================================================

_DEPS = {
    # key: (remote, license file name, SPDX id whose canonical text it ships)
    "alpha": ("https://github.com/example-org/alpha.git", "LICENSE", "MIT"),
    "beta": ("https://github.com/example-org/beta.git", "LICENSE", "Apache-2.0"),
    "gamma": ("https://github.com/example-org/gamma.git", "LICENSE", "Zlib"),
    "delta": (
        "https://github.com/example-org/delta.git",
        "BSD-LICENSE",
        "BSD-3-Clause",
    ),
    "omega": ("https://github.com/example-org/omega.git", "LICENSE.txt", "BSL-1.0"),
    "vendorpy": (
        "https://github.com/example-org/vendorpy.git",
        "LICENSE",
        "Apache-2.0",
    ),
}

_CACHE = """\
# This is the CMakeCache file.
CMAKE_BUILD_TYPE:STRING=Release
CMAKE_SYSTEM_PROCESSOR:STRING=x86_64
CMAKE_CXX_COMPILER:FILEPATH=/usr/bin/c++
ISAAC_TELEOP_PYTHON_VERSION:STRING=3.12
BUILD_VIZ:BOOL=ON
BUILD_PYTHON_BINDINGS:BOOL=ON
"""

_AUTHORED_INIT = """\
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

__version__ = "0.4.0"
"""
_GENERATED = "# generated at configure time\nTRACKERS = ()\n"
_VENDOR_MODULE = """\
# SPDX-FileCopyrightText: Copyright (c) 2026 Example Org
# SPDX-License-Identifier: Apache-2.0

def retarget():
    return None
"""
# Checked in rather than fetched, and says whose it is the way upstream files
# usually do: a plain copyright line, no REUSE tag.
_VENDORED_HEADER = """\
// Copyright 2024, Upstream Widgets Ltd.
// SPDX-License-Identifier: BSL-1.0
#pragma once
"""

_PROJECT_PYPROJECT = """\
# SPDX-FileCopyrightText: Copyright (c) 2026 Example Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

[project]
name = "isaaccapture"
version = "0.4.0"
authors = [{ name = "NVIDIA" }]
"""

_VENDOR_ASSET_SOURCE = (
    '<mujoco model="demo"><asset><mesh file="hand.stl"/></asset></mujoco>\n'
)
_VENDOR_ASSET_STAGED = '<mujoco model="demo"></mujoco>\n'


class Workspace:
    """Paths into a generated project, and the wheel its build would produce."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.build = root / "build"
        self.staged = self.build / "python_package" / CONFIG
        self.wheel = root / "dist" / WHEEL_NAME


def _git_checkout(root: Path, initialise: bool = True) -> None:
    """Make the generated project a real checkout.

    The collector separates repository source from build output by what git
    tracks, because an install prefix carries no marker saying it is output.
    `build/` and `dist/` are ignored here for the same reason they are in the
    real repository.
    """
    _write(root / ".gitignore", "build/\ndist/\n")
    # Fixed identity *and* clock: a commit hashes its timestamp, the collector
    # records the commit, and the wheel carries that record -- so a drifting
    # clock makes two builds of the same project differ.
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "synth",
        "GIT_AUTHOR_EMAIL": "synth@example.invalid",
        "GIT_AUTHOR_DATE": "2026-01-01T00:00:00+00:00",
        "GIT_COMMITTER_NAME": "synth",
        "GIT_COMMITTER_EMAIL": "synth@example.invalid",
        "GIT_COMMITTER_DATE": "2026-01-01T00:00:00+00:00",
    }
    run = lambda *args: subprocess.run(  # noqa: E731 - local shorthand
        ["git", "-C", str(root), *args], check=True, capture_output=True, env=env
    )
    if initialise:
        run("init", "-q", "-b", "main")
    run("add", "-A")
    run("commit", "-q", "--allow-empty", "-m", "synthetic project")


def track(root: Path) -> None:
    """Commit whatever a test added after the project was generated."""
    _git_checkout(root, initialise=False)


def _write(path: Path, data: bytes | str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(data, str):
        path.write_text(data, encoding="utf-8")
    else:
        path.write_bytes(data)


def _stage_license_data(target: Path) -> None:
    """Mirror what deps/third_party leaves in _deps, with just the licenses used here."""
    source = license_data()
    _write(
        target / "licenses.json", (source / "licenses.json").read_text(encoding="utf-8")
    )
    for license_id in FIXTURE_LICENSES:
        detail = source / "details" / f"{license_id}.json"
        if detail.is_file():
            _write(target / "details" / detail.name, detail.read_text(encoding="utf-8"))


def _gitclone_stub(deps: Path, key: str, remote: str) -> None:
    tmp = deps / f"{key}-subbuild" / f"{key}-populate-prefix" / "tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    _write(
        tmp / f"{key}-populate-gitclone.cmake",
        "  clone --no-checkout --depth 1 --no-single-branch "
        f'--config "advice.detachedHead=false" "{remote}" "{key}-src"\n'
        '  checkout "v1.2.3" --\n',
    )


def _sdk_tarball(path: Path) -> dict[str, bytes]:
    """An SDK archive whose libraries are redistributed, one of them patched."""
    members = {
        "LICENSE.txt": PROPRIETARY_LICENSE.encode("utf-8"),
        "VERSION": b"1.2.0\n",
        "libdemo.so": elf_shared_object(
            "libdemo.so", ("libc.so.6",), build_id=SDK_DEMO_BUILD_ID
        ),
        "libextra.so": elf_shared_object("libextra.so", ("libssl.so.3", "libc.so.6")),
        "include/demo.h": b"#pragma once\n",
    }
    # Fixed mtimes, on the members and on the gzip stream: "w:gz" stamps the
    # current time into both, and two archives written a second apart are not
    # the same input.
    path.parent.mkdir(parents=True, exist_ok=True)
    with (
        gzip.GzipFile(filename=path, mode="wb", mtime=0) as compressed,
        tarfile.open(fileobj=compressed, mode="w") as archive,
    ):
        for name, data in sorted(members.items()):
            info = tarfile.TarInfo(f"./{name}")
            info.size = len(data)
            info.mtime = 0
            archive.addfile(info, io.BytesIO(data))
    return members


def create(root: Path) -> Workspace:
    """Generate the project, its build tree, and the staged package."""
    workspace = Workspace(root)
    build = workspace.build
    deps = build / "_deps"

    _write(root / "VERSION", "0.4.x\n")
    _write(root / "CMakeLists.txt", "cmake_minimum_required(VERSION 3.24)\n")
    _write(root / "LICENSE.md", spdx_text("Apache-2.0"))
    for pool_id in ("Apache-2.0", "BSL-1.0"):
        _write(root / "LICENSES" / f"{pool_id}.txt", spdx_text(pool_id))
    _write(root / "src/python/isaaccapture/__init__.py", _AUTHORED_INIT)
    _write(root / "src/vendor/upstream_helper.h", _VENDORED_HEADER)
    _write(root / "pyproject.toml", _PROJECT_PYPROJECT)
    _write(build / "CMakeCache.txt", _CACHE)

    _stage_license_data(deps / "license-list-data-src" / "json")

    for key, (remote, license_name, license_id) in _DEPS.items():
        source = deps / f"{key}-src"
        _write(source / license_name, spdx_text(license_id))
        _write(source / f"{key}.cpp", f"// {key}\n")
        _gitclone_stub(deps, key, remote)

    # vendorpy ships Python that the build stages into the wheel.
    _write(deps / "vendorpy-src/pkg/mod.py", _VENDOR_MODULE)
    _write(deps / "vendorpy-src/pkg/assets/hand.xml", _VENDOR_ASSET_SOURCE)

    sdk_members = _sdk_tarball(root / SDK_ARCHIVE)

    # --- build outputs -------------------------------------------------------
    extension = "isaaccapture/_ext.cpython-312-x86_64-linux-gnu.so"
    extension_bytes = elf_shared_object(
        "_ext.cpython-312-x86_64-linux-gnu.so", ("libstdc++.so.6", "libc.so.6")
    )
    gamma_bytes = elf_shared_object("libgamma_renamed.so", ("libc.so.6",))
    _write(workspace.staged / extension, extension_bytes)
    _write(build / "lib/libgamma_renamed.so", gamma_bytes)
    _write(build / "lib/libdelta.a", b"!<arch>\ndelta\n")
    _write(build / "generated/_generated.py", _GENERATED)
    # Empty, like the real PEP 561 marker: RECORD states size 0 for it, and the
    # bytes explain nothing, so it also stands in for any zero-length member.
    _write(build / "generated/stubs/isaaccapture/py.typed", b"")

    # Staged tree: what the wheel is built from.
    _write(workspace.staged / "isaaccapture/__init__.py", _AUTHORED_INIT)
    _write(workspace.staged / "isaaccapture/_generated.py", _GENERATED)
    _write(workspace.staged / "isaaccapture/py.typed", b"")
    _write(workspace.staged / "isaaccapture/viz/libgamma_renamed.so", gamma_bytes)
    _write(workspace.staged / "isaaccapture/viz/GAMMA_LICENSE", spdx_text("Zlib"))
    _write(
        workspace.staged / "isaaccapture/sdk/native/libdemo.so",
        sdk_members["libdemo.so"],
    )
    # patchelf drops a NEEDED entry, so the shipped bytes differ from the archive's.
    _write(
        workspace.staged / "isaaccapture/sdk/native/libextra.so",
        elf_shared_object("libextra.so", ("libc.so.6",)),
    )
    _write(workspace.staged / "vendorpy/mod.py", _VENDOR_MODULE)
    _write(workspace.staged / "vendorpy/assets/hand.xml", _VENDOR_ASSET_STAGED)

    _file_api_reply(
        root, build, (workspace.staged / extension).relative_to(build).as_posix()
    )
    _git_checkout(root)
    _write(build / "tests/unit_tests", b"\x7fELF-not-really")
    return workspace


def _file_api_reply(root: Path, build: Path, extension: str) -> None:
    """Write the codemodel a `cmake_file_api()` query leaves behind.

    The collector reads CMake's file API rather than scraping generator output,
    so the fixture has to speak the same thing a real configure produces.
    """
    targets = [
        {
            "name": "ext",
            "type": "MODULE_LIBRARY",
            # The project's own source, compiled against beta's headers, with
            # alpha's source compiled straight in.
            "artifacts": [extension],
            "sources": ["src/cpp/module.cpp", "build/_deps/alpha-src/alpha.cpp"],
            "includes": [
                "build/_deps/beta-src/include",
                "build/_deps/alpha-src",
                "src/vendor",
            ],
            "dependencies": [],
        },
        {
            "name": "gamma",
            "type": "SHARED_LIBRARY",
            "artifacts": ["lib/libgamma_renamed.so"],
            "sources": ["build/_deps/gamma-src/gamma.cpp"],
            "includes": ["build/_deps/gamma-src"],
            "dependencies": ["delta"],
        },
        {
            "name": "delta",
            "type": "STATIC_LIBRARY",
            "artifacts": ["lib/libdelta.a"],
            "sources": ["build/_deps/delta-src/delta.cpp"],
            "includes": ["build/_deps/delta-src"],
            "dependencies": [],
        },
        {
            # Tests link omega; nothing they produce reaches a wheel.
            "name": "unit_tests",
            "type": "EXECUTABLE",
            "artifacts": ["tests/unit_tests"],
            "sources": ["tests/cpp/test_main.cpp"],
            "includes": ["build/_deps/omega-src/include"],
            "dependencies": [],
        },
    ]

    reply = build / ".cmake" / "api" / "v1" / "reply"
    index = []
    for spec in targets:
        name = spec["name"]
        payload = {
            "name": name,
            "type": spec["type"],
            "artifacts": [{"path": item} for item in spec["artifacts"]],
            "sources": [{"path": item} for item in spec["sources"]],
            "compileGroups": [
                {"includes": [{"path": item} for item in spec["includes"]]}
            ],
            "dependencies": [{"id": dep} for dep in spec["dependencies"]],
        }
        _write(reply / f"target-{name}-Release.json", json.dumps(payload, indent=1))
        index.append(
            {"id": name, "name": name, "jsonFile": f"target-{name}-Release.json"}
        )

    codemodel = {
        "kind": "codemodel",
        "version": {"major": 2, "minor": 6},
        "paths": {"source": str(root), "build": str(build)},
        "configurations": [{"name": CONFIG, "targets": index}],
    }
    _write(reply / "codemodel-v2-fixture.json", json.dumps(codemodel, indent=1))


# ==============================================================================
# Wheel
# ==============================================================================

METADATA = "\n".join(
    [
        "Metadata-Version: 2.4",
        "Name: isaaccapture",
        "Version: 0.4.0",
        "Summary: Isaac Capture",
        "Author: Example Org",
        "Project-URL: Homepage, https://example.com/isaaccapture",
        "License-Expression: Apache-2.0",
        "License-File: LICENSE.md",
        "Requires-Python: >=3.11",
        # The transition wheel requires this one and this one requires it back;
        # a cycle is the shape the real pair has.
        "Requires-Dist: isaacteleop>=0.4.0",
        "Requires-Dist: numpy>=1.23.0",
        "Requires-Dist: pyyaml>=6.0.3",
        "",
        "Isaac Capture long description.",
        "",
    ]
)


AUDITWHEEL_SBOM = "auditwheel.cdx.json"
AUDITWHEEL_CDX = json.dumps(
    {
        "bomFormat": "CycloneDX",
        "specVersion": "1.4",
        "version": 1,
        "metadata": {"component": {"type": "library", "name": "isaaccapture"}},
        "components": [],
    }
).encode("utf-8")


def wheel_payload(workspace: Workspace) -> dict[str, bytes]:
    """The wheel setuptools would produce from the staged tree."""
    payload: dict[str, bytes] = {}
    for path in sorted(workspace.staged.rglob("*")):
        if path.is_file():
            payload[path.relative_to(workspace.staged).as_posix()] = path.read_bytes()

    payload[f"{DIST_INFO}/METADATA"] = METADATA.encode("utf-8")
    payload[f"{DIST_INFO}/WHEEL"] = (
        b"Wheel-Version: 1.0\nGenerator: setuptools\nRoot-Is-Purelib: false\n"
    )
    payload[f"{DIST_INFO}/licenses/LICENSE.md"] = (
        workspace.root / "LICENSE.md"
    ).read_bytes()
    payload[f"{DIST_INFO}/top_level.txt"] = b"isaaccapture\nvendorpy\n"
    # auditwheel runs before the collector and writes its own PEP 770 document
    # for the libraries it vendored. Every fixture carries it so the suite works
    # on the shape a released wheel actually has.
    payload[f"{DIST_INFO}/sboms/{AUDITWHEEL_SBOM}"] = AUDITWHEEL_CDX
    return payload


def _record_hash(data: bytes) -> str:
    digest = hashlib.sha256(data).digest()
    return "sha256=" + base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")


def write_wheel(path: Path, payload: dict[str, bytes]) -> Path:
    """Write a wheel with a correct RECORD, the way a real build leaves one."""
    dist_info = next(
        name.split("/", 1)[0]
        for name in payload
        if name.endswith(".dist-info/METADATA")
    )
    record_name = f"{dist_info}/RECORD"

    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    for name in sorted(payload):
        writer.writerow((name, _record_hash(payload[name]), str(len(payload[name]))))
    writer.writerow((record_name, "", ""))

    # Fixed stamps: writestr would otherwise record wall-clock, and two wheels
    # written either side of a second boundary are not the same input.
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in [*sorted(payload), record_name]:
            data = (
                payload[name] if name in payload else buffer.getvalue().encode("utf-8")
            )
            member = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            member.external_attr = 0o644 << 16
            member.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(member, data)
    return path
