# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SONAME, NEEDED and the GNU build-id for the shared libraries a wheel carries.

Parsing is pyelftools' job, not ours -- it is the same library auditwheel uses
to repair these wheels, so the tool that wrote the ELF and the tool that reads
it agree by construction.

Only the build path needs any of this. Consumer-side verification re-hashes
files and never parses a binary, which keeps attacker-controlled input out of a
parser on that path entirely.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

from elftools.common.exceptions import ELFError
from elftools.elf.elffile import ELFFile

ELF_MAGIC = b"\x7fELF"

# pyelftools' arch names, mapped to what a wheel's platform tag calls them.
_ARCH = {"x64": "x86_64", "x86": "i386", "AArch64": "aarch64", "ARM": "arm"}


@dataclass(frozen=True)
class DynamicInfo:
    soname: str | None
    needed: tuple[str, ...]
    machine: str
    # Identifies the bytes the linker produced, and patchelf leaves it alone.
    # So it still matches after a repair tool rewrites SONAME and RPATH, which
    # is what makes it the only link back to a library's pristine origin.
    build_id: str | None = None


class NotAnElf(Exception):
    """The blob does not start with the ELF magic."""


def is_elf(data: bytes) -> bool:
    return data[:4] == ELF_MAGIC


def _build_id_from(elffile: ELFFile) -> str | None:
    # Segment, not section: a stripped library keeps its program headers.
    for segment in elffile.iter_segments(type="PT_NOTE"):
        for note in segment.iter_notes():
            if note["n_type"] == "NT_GNU_BUILD_ID":
                return note["n_desc"]
    return None


def read_build_id(path: Path) -> str | None:
    """The build-id of a file on disk, read without loading it.

    Scanning a machine's libraries means touching hundreds of files, some of
    them hundreds of megabytes. pyelftools seeks, so only the headers and the
    note are ever read.
    """
    try:
        with path.open("rb") as handle:
            if handle.read(4) != ELF_MAGIC:
                return None
            handle.seek(0)
            return _build_id_from(ELFFile(handle))
    except (OSError, ELFError):
        # Scanning a machine's libraries meets files that only look like ELF.
        # Anything else raising here is a bug in this parser, not bad input.
        return None


def read_dynamic(data: bytes) -> DynamicInfo:
    """Parse SONAME, NEEDED and the build-id out of an in-memory ELF image."""
    if not is_elf(data):
        raise NotAnElf("missing ELF magic")

    elffile = ELFFile(io.BytesIO(data))
    machine = elffile.get_machine_arch()

    soname: str | None = None
    needed: list[str] = []

    # The dynamic segment, not the section: a stripped library keeps its program
    # headers and loses its section headers, and the loader reads the segment.
    for segment in elffile.iter_segments(type="PT_DYNAMIC"):
        for tag in segment.iter_tags():
            kind = tag.entry.d_tag
            if kind == "DT_SONAME":
                soname = tag.soname
            elif kind == "DT_NEEDED":
                needed.append(tag.needed)

    return DynamicInfo(
        soname=soname,
        needed=tuple(needed),
        machine=_ARCH.get(machine, machine),
        build_id=_build_id_from(elffile),
    )
