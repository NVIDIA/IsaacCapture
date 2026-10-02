# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The ELF reader is how bundled libraries are told apart from external ones."""

from __future__ import annotations

import pytest
import synth
from sbom import elf


def test_reads_soname_and_needed():
    image = synth.elf_shared_object("libcloudxr.so", ("libPoco.so", "libcudart.so.12"))
    info = elf.read_dynamic(image)

    assert info.soname == "libcloudxr.so"
    assert info.needed == ("libPoco.so", "libcudart.so.12")
    assert info.machine == "x86_64"


def test_reads_a_library_with_no_dependencies():
    info = elf.read_dynamic(synth.elf_shared_object("libisaacteleop_mujoco.so"))

    assert info.soname == "libisaacteleop_mujoco.so"
    assert info.needed == ()


def test_rejects_non_elf_payloads():
    assert not elf.is_elf(b"# not an elf\n")
    with pytest.raises(elf.NotAnElf):
        elf.read_dynamic(b"PK\x03\x04 zip, not elf")


def test_reads_a_real_system_library():
    """Guards the reader against the synthesizer agreeing with itself."""
    import ctypes.util

    name = ctypes.util.find_library("c")
    if name is None:
        pytest.skip("no system libc to read")
    for candidate in (
        "/lib/x86_64-linux-gnu",
        "/lib/aarch64-linux-gnu",
        "/usr/lib",
        "/lib",
    ):
        path = f"{candidate}/{name}"
        try:
            data = open(path, "rb").read()  # noqa: SIM115
        except OSError:
            continue
        if elf.is_elf(data):
            assert elf.read_dynamic(data).soname == name
            return
    pytest.skip("could not locate the system libc on disk")


def test_verification_does_not_parse_binaries(built):
    """A consumer checking a downloaded wheel should not run a binary parser
    over it; scanning only reads the dynamic section when asked to."""
    from sbom import wheelfile

    plain = wheelfile.scan(built["wheel"])
    analyzed = wheelfile.scan(built["wheel"], analyze_elf=True)

    assert not any(entry.is_elf for entry in plain.entries)
    assert any(entry.is_elf for entry in analyzed.entries)
    # The digests verification relies on are identical either way.
    assert {e.name: e.sha256 for e in plain.entries} == {
        e.name: e.sha256 for e in analyzed.entries
    }
