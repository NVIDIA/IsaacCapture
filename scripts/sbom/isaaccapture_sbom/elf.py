# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read SONAME, NEEDED, and GNU build IDs with pyelftools.

Only collection parses ELF; consumer verification hashes bytes without
invoking a binary parser.
"""

from __future__ import annotations

import io
import struct
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
    # The linker's build ID survives patchelf SONAME/RPATH rewrites.
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


def build_id_of(data: bytes) -> str | None:
    """Read a build ID from an ELF prefix; return None if the note lies beyond it."""
    if not is_elf(data):
        return None
    try:
        return _build_id_from(ELFFile(io.BytesIO(data)))
    except (ELFError, OSError, struct.error):
        return None


def _from_file(path: Path, read):
    """Apply read to an ELFFile without loading the whole binary.

    Unreadable or truncated ELF returns None, including parser struct.error failures.
    """
    try:
        with path.open("rb") as handle:
            if handle.read(4) != ELF_MAGIC:
                return None
            handle.seek(0)
            return read(ELFFile(handle))
    except (OSError, ELFError, struct.error):
        return None


def read_build_id(path: Path) -> str | None:
    """The build-id of a file on disk, read without loading it."""
    return _from_file(path, _build_id_from)


def read_dynamic(data: bytes) -> DynamicInfo:
    """Parse SONAME, NEEDED and the build-id out of an in-memory ELF image."""
    if not is_elf(data):
        raise NotAnElf("missing ELF magic")

    elffile = ELFFile(io.BytesIO(data))
    machine = elffile.get_machine_arch()

    soname: str | None = None
    needed: list[str] = []

    # Read the dynamic segment: stripped libraries can lack section headers.
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


def file_type(path: Path) -> str | None:
    """A file's ELF `e_type` -- `ET_REL` for an object, `ET_DYN` for a shared
    library -- or None if it is not an ELF file."""
    return _from_file(path, lambda elffile: elffile.header["e_type"])
