# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate synthetic build trees and wheels, including malformed gate inputs.

Check reader assumptions separately against real CMake replies, ELF files,
and the SPDX corpus.
"""

from __future__ import annotations

import base64
import contextlib
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

from isaaccapture_sbom import build as build_module

CONFIG = "Release"
WHEEL_NAME = "isaaccapture-0.4.0-cp312-cp312-manylinux_2_35_x86_64.whl"
STEM = WHEEL_NAME.removesuffix(".whl")
DIST_INFO = "isaaccapture-0.4.0.dist-info"
SDK_ARCHIVE = "deps/vendor/demosdk-1.2.0-linux-amd64.tar.gz"
SDK_DEMO_BUILD_ID = "1111111111111111111111111111111111111111"

# ELF

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
    """Build a little-endian ELF64 dynamic segment with an optional GNU build-ID note."""
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


# License texts


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
    "GPL-3.0-only",
    "GPL-3.0-or-later",
    "MIT",
    "MIT-0",
    "Zlib",
)
# Newer than the grammar the collector ships, which is what the tests need.
FIXTURE_EXCEPTIONS = ("rsync-linking-exception",)


def set_license_data(path: Path) -> None:
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


# Workspace

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

# Upstream's header names no identifier; this repository declares it through
# REUSE instead, as it does for Chromium's key table.
UNTAGGED_HEADER = """\
// Copyright 2013 The Keys Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
"""
UNTAGGED_KEY = "vendored:the-keys-authors"

_PROJECT_PYPROJECT = """\
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


def git(root: Path, *args: str) -> str:
    # Fix identity and timestamps for reproducible synthetic commits and wheel provenance.
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "synth",
        "GIT_AUTHOR_EMAIL": "synth@example.invalid",
        "GIT_AUTHOR_DATE": "2026-01-01T00:00:00+00:00",
        "GIT_COMMITTER_NAME": "synth",
        "GIT_COMMITTER_EMAIL": "synth@example.invalid",
        "GIT_COMMITTER_DATE": "2026-01-01T00:00:00+00:00",
    }
    return subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
        env=env,
        text=True,
    ).stdout.strip()


def _commit_all(root: Path, message: str) -> None:
    git(root, "add", "-A")
    git(root, "commit", "-q", "--allow-empty", "-m", message)


def clone_of(root: Path, remote: str, tag: str, git_dir: Path | None = None) -> str:
    """Create a dependency checkout and return its commit ID.

    git_dir uses a separate repository behind a .git file, as in a submodule.
    """
    root.mkdir(parents=True, exist_ok=True)
    separate = []
    if git_dir:
        git_dir.parent.mkdir(parents=True, exist_ok=True)
        separate = ["--separate-git-dir", str(git_dir)]
    git(root, "init", "-q", "-b", "main", *separate)
    _commit_all(root, "upstream")
    git(root, "tag", tag)
    git(root, "remote", "add", "origin", remote)
    return git(root, "rev-parse", "HEAD")


def _git_checkout(root: Path, initialise: bool = True) -> None:
    """Create a real checkout to distinguish tracked sources from build output."""
    write(root / ".gitignore", "build/\ndist/\n")
    if initialise:
        git(root, "init", "-q", "-b", "main")
    _commit_all(root, "synthetic project")


def annotation(
    path: str,
    holder: str | None = None,
    *,
    licence: str | None = "MIT",
    precedence: str = "closest",
    year: int = 2020,
) -> str:
    """One REUSE.toml annotation; it states no copyright where `holder` is None."""
    lines = ["[[annotations]]", f'path = "{path}"', f'precedence = "{precedence}"']
    if holder is not None:
        lines.append(f'SPDX-FileCopyrightText = "Copyright {year} {holder}"')
    if licence is not None:
        lines.append("SPDX-" + f'License-Identifier = "{licence}"')
    return "\n".join(lines) + "\n"


def reuse_toml(*annotations: str) -> str:
    return "version = 1\n\n" + "\n".join(annotations)


def track(root: Path) -> None:
    _git_checkout(root, initialise=False)


def write(path: Path, data: bytes | str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(data, str):
        path.write_text(data, encoding="utf-8")
    else:
        path.write_bytes(data)


def _stage_license_data(target: Path) -> None:
    """Mirror what deps/third_party leaves in _deps, with just the licenses used here."""
    source = license_data()
    for index in ("licenses.json", "exceptions.json"):
        write(target / index, (source / index).read_text(encoding="utf-8"))
    for license_id in FIXTURE_LICENSES:
        detail = source / "details" / f"{license_id}.json"
        if detail.is_file():
            write(target / "details" / detail.name, detail.read_text(encoding="utf-8"))
    for exception_id in FIXTURE_EXCEPTIONS:
        detail = source / "exceptions" / f"{exception_id}.json"
        write(target / "exceptions" / detail.name, detail.read_text(encoding="utf-8"))


def declare(
    deps: Path,
    key: str,
    *,
    urls: tuple[str, ...] = (),
    digest: str | None = None,
    repository: str | None = None,
    tag: str | None = None,
    direct: bool = False,
    source: str | None = None,
) -> None:
    """Write the stamps a FetchContent declaration leaves in the build tree.

    Laid out as the populate sub-build does, or, with `direct`, as CMP0168 NEW
    does. `source` names the tree, `<key>-src` by default.
    """
    source_dir = deps / (source or f"{key}-src")
    if direct:
        files = deps.parent / "CMakeFiles"
        stamps, scripts, prefix = files / "fc-stamp" / key, files / "fc-tmp" / key, key
    else:
        prefix = f"{key}-populate"
        scripts = deps / f"{key}-subbuild" / f"{prefix}-prefix" / "tmp"
        stamps = scripts.parent / "src" / f"{prefix}-stamp"

    def stamp(kind: str, **fields: str) -> None:
        body = "".join(f"{name}={value}\n" for name, value in fields.items())
        write(
            stamps / f"{prefix}-{kind}info.txt",
            "# This is a generated file and its contents are an internal "
            f"implementation detail.\n\nmethod={kind}\n{body}",
        )

    if urls:
        stamp(
            "url",
            source_dir=str(source_dir),
            **{"url(s)": ";".join(urls)},
            hash=digest or "",
        )
    if repository:
        clone = scripts / f"{prefix}-gitclone.cmake"
        write(
            clone,
            # The URL straight after `--no-checkout`, which is not a checkout.
            f'  clone --no-checkout "{repository}" "{source_dir.name}"\n'
            + (f'  checkout "{tag}" --\n' if tag else ""),
        )
        stamp(
            "git",
            command=f"/usr/bin/cmake;-P;{clone}",
            source_dir=str(source_dir),
            repository=repository,
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
    # Fix member and gzip timestamps for reproducible archive bytes.
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
    workspace = Workspace(root)
    build = workspace.build
    deps = build / "_deps"

    write(root / "VERSION", "0.4.x\n")
    write(root / "CMakeLists.txt", "cmake_minimum_required(VERSION 3.24)\n")
    write(root / "LICENSE.md", spdx_text("Apache-2.0"))
    for pool_id in ("Apache-2.0", "BSL-1.0"):
        write(root / "LICENSES" / f"{pool_id}.txt", spdx_text(pool_id))
    write(root / "src/python/isaaccapture/__init__.py", _AUTHORED_INIT)
    write(root / "src/vendor/upstream_helper.h", _VENDORED_HEADER)
    write(root / "pyproject.toml", _PROJECT_PYPROJECT)
    write(build / "CMakeCache.txt", _CACHE)

    _stage_license_data(deps / "license-list-data-src" / "json")

    for key, (remote, license_name, license_id) in _DEPS.items():
        source = deps / f"{key}-src"
        write(source / license_name, spdx_text(license_id))
        write(source / f"{key}.cpp", f"// {key}\n")
        declare(deps, key, repository=remote, tag="v1.2.3")

    write(deps / "vendorpy-src/pkg/mod.py", _VENDOR_MODULE)
    write(deps / "vendorpy-src/pkg/assets/hand.xml", _VENDOR_ASSET_SOURCE)

    sdk_members = _sdk_tarball(root / SDK_ARCHIVE)

    # Build outputs
    extension = "isaaccapture/_ext.cpython-312-x86_64-linux-gnu.so"
    extension_bytes = elf_shared_object(
        "_ext.cpython-312-x86_64-linux-gnu.so", ("libstdc++.so.6", "libc.so.6")
    )
    gamma_bytes = elf_shared_object("libgamma_renamed.so", ("libc.so.6",))
    write(workspace.staged / extension, extension_bytes)
    write(build / "lib/libgamma_renamed.so", gamma_bytes)
    write(build / "lib/libdelta.a", b"!<arch>\ndelta\n")
    write(build / "generated/_generated.py", _GENERATED)
    # Empty marker exercises RECORD size 0 and path-only attribution.
    write(build / "generated/stubs/isaaccapture/py.typed", b"")

    # Model a package-manager header installed into the build tree.
    write(
        build / "vcpkg_installed/x64-linux/include/eigen/Core.h",
        b"// Copyright (C) 2008 Gael Guennebaud <gael.guennebaud@inria.fr>\n",
    )

    write(workspace.staged / "isaaccapture/__init__.py", _AUTHORED_INIT)
    write(workspace.staged / "isaaccapture/_generated.py", _GENERATED)
    write(workspace.staged / "isaaccapture/py.typed", b"")
    write(workspace.staged / "isaaccapture/viz/libgamma_renamed.so", gamma_bytes)
    write(workspace.staged / "isaaccapture/viz/GAMMA_LICENSE", spdx_text("Zlib"))
    write(
        workspace.staged / "isaaccapture/sdk/native/libdemo.so",
        sdk_members["libdemo.so"],
    )
    # patchelf drops a NEEDED entry, so the shipped bytes differ from the archive's.
    write(
        workspace.staged / "isaaccapture/sdk/native/libextra.so",
        elf_shared_object("libextra.so", ("libc.so.6",)),
    )
    write(workspace.staged / "vendorpy/mod.py", _VENDOR_MODULE)
    write(workspace.staged / "vendorpy/assets/hand.xml", _VENDOR_ASSET_STAGED)

    _file_api_reply(
        root, build, (workspace.staged / extension).relative_to(build).as_posix()
    )
    _git_checkout(root)
    write(build / "tests/unit_tests", b"\x7fELF-not-really")
    return workspace


def _file_api_reply(root: Path, build: Path, extension: str) -> None:
    """Write a synthetic CMake file API reply."""
    targets = [
        {
            "name": "ext",
            "type": "MODULE_LIBRARY",
            # The project's own source, compiled against beta's headers, with
            # alpha's source compiled straight in.
            "artifacts": [extension],
            # List a package-manager header as a source to exercise build-tree exclusion.
            "sources": [
                "src/cpp/module.cpp",
                "build/_deps/alpha-src/alpha.cpp",
                "build/vcpkg_installed/x64-linux/include/eigen/Core.h",
            ],
            "includes": [
                "build/_deps/beta-src/include",
                "build/_deps/alpha-src",
                "build/vcpkg_installed/x64-linux/include",
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

    reply = _reply(build)
    index = []
    for spec in targets:
        entry = _target_entry(spec["name"])
        payload = _target_payload(**spec)
        write(reply / entry["jsonFile"], json.dumps(payload, indent=1))
        index.append(entry)

    codemodel = {
        "kind": "codemodel",
        "version": {"major": 2, "minor": 6},
        "paths": {"source": str(root), "build": str(build)},
        "configurations": [{"name": CONFIG, "targets": index}],
    }
    write(reply / "codemodel-v2-fixture.json", json.dumps(codemodel, indent=1))
    write(
        reply / "index-2026-01-01T00-00-00-0000.json",
        json.dumps(
            {
                "objects": [
                    {
                        "kind": "codemodel",
                        "version": {"major": 2, "minor": 6},
                        "jsonFile": "codemodel-v2-fixture.json",
                    }
                ]
            },
            indent=1,
        ),
    )


def _source(path: str) -> dict:
    """A codemodel source entry; one CMake compiles names its compile group."""
    compiled = path.endswith((".c", ".cpp"))
    return {"path": path, **({"compileGroupIndex": 0} if compiled else {})}


def _target_payload(
    name: str,
    type: str,  # noqa: A002 - the codemodel's own key
    artifacts: list[str],
    sources: list[str],
    includes: list[str] | None,
    dependencies: list[str],
) -> dict:
    """A codemodel target; one with no compile groups compiles nothing."""
    payload = {
        "name": name,
        "type": type,
        "artifacts": [{"path": item} for item in artifacts],
        "sources": [_source(item) for item in sources],
        "dependencies": [{"id": dep} for dep in dependencies],
    }
    if includes is not None:
        payload["compileGroups"] = [{"includes": [{"path": item} for item in includes]}]
    return payload


def _target_entry(name: str) -> dict:
    return {"id": name, "name": name, "jsonFile": f"target-{name}-{CONFIG}.json"}


@contextlib.contextmanager
def _edit_json(path: Path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    yield payload
    path.write_text(json.dumps(payload, indent=1), encoding="utf-8")


def _reply(build: Path) -> Path:
    return build / ".cmake" / "api" / "v1" / "reply"


def _edit_target(build: Path, target: str):
    return _edit_json(_reply(build) / f"target-{target}-{CONFIG}.json")


def _edit_index(build: Path):
    return _edit_json(next(_reply(build).glob("index-*.json")))


def link_line(
    build: Path,
    target: str,
    fragments: tuple[str | tuple[str, str], ...],
    *,
    dependencies: tuple[str, ...] = (),
    directory: str | None = None,
    sysroot: Path | None = None,
) -> None:
    """Give a fixture target the link line CMake records for a linkable target.

    A fragment is its text, filed under `libraries`, or (text, role).
    `directory` is the target's build directory, which is what a relative
    fragment is relative to under the Makefile generators. `sysroot` is
    CMAKE_SYSROOT, as the reply records it.
    """
    with _edit_target(build, target) as payload:
        payload["link"] = {
            "language": "CXX",
            "commandFragments": [
                {"fragment": item, "role": "libraries"}
                if isinstance(item, str)
                else {"fragment": item[0], "role": item[1]}
                for item in fragments
            ],
        }
        if sysroot is not None:
            payload["link"]["sysroot"] = {"path": str(sysroot)}
        payload["dependencies"] += [{"id": item} for item in dependencies]
    if directory is not None:
        set_directory(build, target, directory)


def set_toolchains(
    build: Path,
    link_directories: dict[str, list[str] | None],
    compilers: dict[str, str] | None = None,
) -> None:
    """A toolchains reply: each language's compiler and implicit link directories.

    None for a language's directories is what CMake writes for a language the
    top-level project does not enable.
    """
    compilers = compilers or {}
    toolchains = []
    for language, directories in link_directories.items():
        compiler: dict = {
            "implicit": {} if directories is None else {"linkDirectories": directories}
        }
        if language in compilers:
            compiler["path"] = compilers[language]
        toolchains.append({"language": language, "compiler": compiler})
    write(
        _reply(build) / "toolchains-v1-fixture.json",
        json.dumps(
            {
                "kind": "toolchains",
                "version": {"major": 1, "minor": 0},
                "toolchains": toolchains,
            }
        ),
    )
    with _edit_index(build) as index:
        index["objects"].append(
            {
                "kind": "toolchains",
                "version": {"major": 1, "minor": 0},
                "jsonFile": "toolchains-v1-fixture.json",
            }
        )


def add_include(build: Path, target: str, directory: Path) -> None:
    with _edit_target(build, target) as payload:
        payload["compileGroups"][0]["includes"].append({"path": str(directory)})


def add_source(build: Path, target: str, path: Path) -> None:
    with _edit_target(build, target) as payload:
        payload["sources"].append(_source(str(path)))


def depfile(
    build: Path,
    target: str,
    source: Path | str,
    read: tuple[Path | str, ...] = (),
    directory: str = ".",
) -> None:
    """The depfile GCC leaves beside one of a target's objects under the
    Makefile generators: the object, its source, every header it read, and
    `-MP`'s empty rule per header. `directory` is the target's build directory.
    """
    obj = f"CMakeFiles/{target}.dir/{Path(source).name}.o"
    rule = " \\\n ".join([f"{obj}: {source}", *(str(item) for item in read)])
    write(
        build / directory / f"{obj}.d",
        rule + "\n" + "".join(f"\n{item}:\n" for item in read),
    )


def compiler_depend(
    build: Path, target: str, rules: dict[str, tuple[str, ...]], directory: str = "."
) -> None:
    """compiler_depend.make as CMake consolidates a target's depfiles: one rule
    per object, paths inside the build tree relative to its top. No rules is
    the placeholder CMake writes before the first build."""
    path = build / directory / "CMakeFiles" / f"{target}.dir" / "compiler_depend.make"
    if not rules:
        write(
            path,
            f"# Empty compiler generated dependencies file for {target}.\n"
            "# This may be replaced when dependencies are built.\n",
        )
        return
    write(
        path,
        "# CMAKE generated file: DO NOT EDIT!\n\n"
        + "\n".join(
            " \\\n  ".join([f"{obj}: {prerequisites[0]}", *prerequisites[1:]])
            for obj, prerequisites in rules.items()
        )
        + "\n",
    )


def add_target(
    build: Path, name: str, kind: str, sources: tuple[str, ...], dependent: str
) -> None:
    """One more target, which `dependent` depends on, producing nothing shipped."""
    entry = _target_entry(name)
    payload = _target_payload(
        name=name,
        type=kind,
        artifacts=[f"tools/{name}"] if kind == "EXECUTABLE" else [],
        sources=list(sources),
        includes=None if kind == "UTILITY" else [],
        dependencies=[],
    )
    write(_reply(build) / entry["jsonFile"], json.dumps(payload))
    with _edit_json(_reply(build) / "codemodel-v2-fixture.json") as codemodel:
        codemodel["configurations"][0]["targets"].append(entry)
    with _edit_target(build, dependent) as target:
        target["dependencies"].append({"id": name})


def set_directory(build: Path, target: str, directory: str) -> None:
    with _edit_target(build, target) as payload:
        payload["paths"] = {"build": directory, "source": directory}


def set_type(build: Path, target: str, kind: str) -> None:
    with _edit_target(build, target) as payload:
        payload["type"] = kind


def set_generator(build: Path, name: str) -> None:
    with _edit_index(build) as index:
        index["cmake"] = {"generator": {"name": name, "multiConfig": False}}


def vcpkg_port(
    installed: Path,
    port: str,
    version: str,
    files: dict[str, bytes],
    licence: str,
    source: str | tuple[str, ...],
    triplet: str = "x64-linux",
) -> None:
    """A port as vcpkg installs it: its files, its install list and its SBOM.

    Shaped after what vcpkg 2026-04-08 writes for fmt 11.0.2#1.
    """
    base = version.partition("#")[0]
    for relative, data in files.items():
        write(installed / triplet / relative, data)
    share = installed / triplet / "share" / port
    write(share / "copyright", licence)
    write(
        share / "vcpkg.spdx.json",
        json.dumps(
            {
                "spdxVersion": "SPDX-2.2",
                "SPDXID": "SPDXRef-DOCUMENT",
                "packages": [
                    {
                        "name": port,
                        "SPDXID": "SPDXRef-port",
                        "versionInfo": version,
                        "downloadLocation": "git+https://github.com/Microsoft/vcpkg@07a73a7565e5de9eb42e90c16c133bdfdfebbcda",
                        "homepage": f"https://github.com/{port}lib/{port}",
                    },
                    {
                        "name": f"{port}:{triplet}",
                        "SPDXID": "SPDXRef-binary",
                        "downloadLocation": "NONE",
                    },
                    *(
                        {
                            "SPDXID": f"SPDXRef-resource-{index}",
                            "name": item,
                            "downloadLocation": f"git+https://github.com/{item}@{base}",
                        }
                        for index, item in enumerate(
                            (source,) if isinstance(source, str) else source
                        )
                    ),
                ],
            }
        ),
    )
    listed = sorted(
        {f"{triplet}/{item}" for item in files}
        | {
            f"{triplet}/share/{port}/copyright",
            f"{triplet}/share/{port}/vcpkg.spdx.json",
        }
    )
    write(
        installed / "vcpkg" / "info" / f"{port}_{base}_{triplet}.list",
        "".join(f"{line}\n" for line in [f"{triplet}/", *listed]),
    )


def stub_compiler(directory: Path, libraries: Path, sysroot: Path, search: str) -> Path:
    """A compiler driver that answers the questions the collector asks one.

    Its linker lists `search` as a SEARCH_DIR under the sysroot, `=` and all.
    """
    linker = directory / "ld"
    write(
        linker,
        f'#!/bin/sh\necho \'SEARCH_DIR("={search}"); SEARCH_DIR("/nonexistent")\'\n',
    )
    compiler = directory / "cc"
    write(
        compiler,
        "#!/bin/sh\n"
        'case "$1" in\n'
        f'  -print-search-dirs) echo "libraries: ={libraries}" ;;\n'
        f'  -print-prog-name=ld) echo "{linker}" ;;\n'
        f'  -print-sysroot) echo "{sysroot}" ;;\n'
        "esac\n",
    )
    for path in (linker, compiler):
        path.chmod(0o755)
    return compiler


def elf_object() -> bytes:
    """A relocatable ELF, the kind a compiler writes and an archive holds."""
    image = bytearray(elf_shared_object("object.o"))
    struct.pack_into("<H", image, 0x10, 1)  # ET_REL
    return bytes(image)


def ar_header(name: bytes, size: int) -> bytes:
    """One `ar` member header, its name written as given."""
    return (
        name.ljust(16)
        + b"0".ljust(12)
        + b"0".ljust(6)
        + b"0".ljust(6)
        + b"644".ljust(8)
        + str(size).encode().ljust(10)
        + b"`\n"
    )


def ar_archive(members: dict[str, bytes]) -> bytes:
    """A static archive in the GNU `ar` format, holding `members`."""
    image = bytearray(b"!<arch>\n")
    for name, data in members.items():
        image += ar_header(f"{name}/".encode(), len(data))
        image += data + b"\n" * (len(data) % 2)
    return bytes(image)


# Wheel

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
        # Model the cyclic dependency between the main and transition wheels.
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
    # Include auditwheel's PEP 770 document to test coexistence.
    payload[f"{DIST_INFO}/sboms/{AUDITWHEEL_SBOM}"] = AUDITWHEEL_CDX
    return payload


def collect(workspace: Workspace, **options) -> dict:
    out_dir = workspace.root / "sbom"
    manifest = build_module.build(
        workspace.root, workspace.build, workspace.wheel, out_dir, **options
    )
    with zipfile.ZipFile(workspace.wheel) as archive:
        notices = archive.read(f"{DIST_INFO}/licenses/THIRD-PARTY-NOTICES.md")
    return {
        "out_dir": out_dir,
        "manifest": manifest,
        **{
            kind: json.loads((out_dir / f"{STEM}.{kind}.json").read_text("utf-8"))
            for kind in ("spdx", "licenses", "build-evidence")
        },
        "md": (out_dir / f"{STEM}.licenses.md").read_text("utf-8"),
        "notices": notices.decode("utf-8"),
    }


def collect_and_bind(workspace: Workspace, **options) -> dict:
    """`collect`, with the manifest written beside it and the paths tests read."""
    collected = collect(workspace, **options)
    out_dir = collected["out_dir"]
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(collected["manifest"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        **collected,
        "workspace": workspace,
        "root": workspace.root,
        "wheel": workspace.wheel,
        "manifest_path": manifest_path,
        "evidence_path": out_dir / f"{STEM}.build-evidence.json",
        "spdx_path": out_dir / f"{STEM}.spdx.json",
    }


def members(wheel: Path) -> dict[str, bytes]:
    """A wheel's members, RECORD left out so `write_wheel` can restate it."""
    with zipfile.ZipFile(wheel) as archive:
        return {
            name: archive.read(name)
            for name in archive.namelist()
            if not name.endswith(".dist-info/RECORD")
        }


def drop_member(wheel: Path, name: str) -> None:
    """Remove one member after the fact, leaving RECORD as it was."""
    with zipfile.ZipFile(wheel) as archive:
        payload = {
            item: archive.read(item) for item in archive.namelist() if item != name
        }
    with zipfile.ZipFile(wheel, "w", zipfile.ZIP_DEFLATED) as archive:
        for item in sorted(payload):
            archive.writestr(item, payload[item])


def replace_member(wheel: Path, name: str, data: bytes) -> None:
    """Change one member after the fact, leaving RECORD as it was."""
    with zipfile.ZipFile(wheel) as archive:
        payload = {item: archive.read(item) for item in archive.namelist()}
    payload[name] = data
    with zipfile.ZipFile(wheel, "w", zipfile.ZIP_DEFLATED) as archive:
        for item in sorted(payload):
            archive.writestr(item, payload[item])


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

    # Fix ZIP timestamps for reproducible wheel bytes.
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
