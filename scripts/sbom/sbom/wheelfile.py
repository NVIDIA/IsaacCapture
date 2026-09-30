# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read a wheel's contents and rewrite it with the evidence it must carry.

RECORD and METADATA are specified formats, so their own tooling parses them:
``installer`` (PyPA) owns RECORD's hashing and escaping rules, and ``packaging``
owns core metadata. Only the zip rewriting is ours, because no library offers it.
"""

from __future__ import annotations

import base64
import csv
import hashlib
import io
import os
import time
import zipfile
from dataclasses import dataclass
from email import message_from_bytes
from email.generator import BytesGenerator
from email.policy import compat32
from installer.records import Hash, RecordEntry, parse_record_file
from packaging.metadata import RawMetadata, parse_email
from packaging.version import Version
from pathlib import Path

from . import elf

# Fixed member timestamp for files this tool adds, so two runs over the same
# inputs produce byte-identical wheels.
_ADDED_TIMESTAMP = (1980, 1, 1, 0, 0, 0)


def _reproducible_timestamp() -> tuple[int, int, int, int, int, int] | None:
    """SOURCE_DATE_EPOCH as a zip timestamp, when a reproducible build asked for one.

    Zip records a local timestamp per member, so a wheel carries whatever clock
    its builder had. Under SOURCE_DATE_EPOCH every member gets the same stamp --
    including ones copied through untouched -- or the rewrite would be
    reproducible only for an input wheel that already was.
    """
    epoch = os.environ.get("SOURCE_DATE_EPOCH")
    if not epoch or not epoch.isdigit():
        return None
    moment = time.gmtime(int(epoch))
    if moment.tm_year < 1980:  # zip cannot record anything earlier
        return _ADDED_TIMESTAMP
    return (
        moment.tm_year,
        moment.tm_mon,
        moment.tm_mday,
        moment.tm_hour,
        moment.tm_min,
        moment.tm_sec,
    )


class WheelError(Exception):
    """The wheel is not shaped the way the packaging specs require."""


@dataclass(frozen=True)
class Entry:
    """One member of the wheel's zip archive."""

    name: str
    size: int
    sha256: str
    sha1: str
    soname: str | None = None
    needed: tuple[str, ...] = ()
    machine: str | None = None
    # Enough to find a REUSE tag, kept so nothing re-opens the archive for it.
    head: bytes = b""

    @property
    def is_elf(self) -> bool:
        return self.machine is not None


@dataclass(frozen=True)
class WheelInfo:
    path: Path
    distribution: str
    version: str
    dist_info: str
    entries: tuple[Entry, ...]


def record_hash(data: bytes) -> str:
    """RECORD's hash field, as PEP 376 spells it."""
    return str(Hash.parse(f"sha256={_urlsafe_b64(data)}"))


def record_hash_from_sha256(digest: str) -> str:
    """The same field, from a digest already computed."""
    encoded = (
        base64.urlsafe_b64encode(bytes.fromhex(digest)).rstrip(b"=").decode("ascii")
    )
    return str(Hash.parse(f"sha256={encoded}"))


def _urlsafe_b64(data: bytes) -> str:
    return (
        base64.urlsafe_b64encode(hashlib.sha256(data).digest())
        .rstrip(b"=")
        .decode("ascii")
    )


def _dist_info(names: list[str]) -> str:
    candidates = sorted(
        {
            name.split("/", 1)[0]
            for name in names
            if name.endswith(".dist-info/METADATA")
        }
    )
    if len(candidates) != 1:
        raise WheelError(
            f"expected exactly one .dist-info with METADATA, found {candidates}"
        )
    return candidates[0]


def scan(wheel_path: Path, analyze_elf: bool = False) -> WheelInfo:
    """Hash every member, and read the dynamic section of ELF members on request.

    `analyze_elf` is what the inventory needs and what verification does not, so
    a consumer checking a downloaded wheel never runs a binary parser over it.
    """
    with zipfile.ZipFile(wheel_path) as archive:
        names = archive.namelist()
        dist_info = _dist_info(names)
        entries: list[Entry] = []
        for info in archive.infolist():
            if info.is_dir():
                continue
            data = archive.read(info.filename)
            soname = needed = machine = None
            if analyze_elf and elf.is_elf(data[:4]):
                dynamic = elf.read_dynamic(data)
                soname, needed, machine = (
                    dynamic.soname,
                    dynamic.needed,
                    dynamic.machine,
                )
            entries.append(
                Entry(
                    name=info.filename,
                    size=len(data),
                    sha256=hashlib.sha256(data).hexdigest(),
                    sha1=hashlib.sha1(data).hexdigest(),  # noqa: S324 - SPDX 2.3 requires SHA1
                    soname=soname,
                    needed=needed or (),
                    machine=machine,
                    head=data[:4096],
                )
            )

    stem = dist_info.removesuffix(".dist-info")
    distribution, _, version = stem.partition("-")
    if not version:
        raise WheelError(f"cannot read distribution and version from {dist_info!r}")
    return WheelInfo(
        path=wheel_path,
        distribution=distribution,
        version=version,
        dist_info=dist_info,
        entries=tuple(entries),
    )


def read_metadata(wheel_path: Path, dist_info: str) -> tuple[bytes, RawMetadata]:
    """Core metadata, parsed by packaging rather than as loose email headers."""
    with zipfile.ZipFile(wheel_path) as archive:
        raw = archive.read(f"{dist_info}/METADATA")
    return raw, parse_metadata(raw)


def parse_metadata(raw: bytes) -> RawMetadata:
    parsed, _ = parse_email(raw)
    return parsed


def add_license_files(raw_metadata: bytes, relative_paths: list[str]) -> bytes:
    """Declare packaged license files, which PEP 639 requires of Metadata 2.4.

    Older metadata cannot express License-File, so the files still ship but go
    undeclared; the caller reports that rather than emitting invalid metadata.
    """
    metadata = parse_metadata(raw_metadata)
    # Compare as a version: "2.10" sorts before "2.4" as a string.
    if Version(metadata.get("metadata_version") or "0") < Version("2.4"):
        return raw_metadata

    existing = set(metadata.get("license_files") or [])
    additions = [path for path in relative_paths if path not in existing]
    if not additions:
        return raw_metadata

    # email owns RFC 822 header insertion, including keeping the payload (the
    # long description) intact.
    message = message_from_bytes(raw_metadata, policy=compat32)
    for path in additions:
        message.add_header("License-File", path)

    # maxheaderlen=0 disables RFC 822 folding. A folded value is still valid and
    # packaging unfolds it, but no other tool wraps paths in METADATA, and a
    # continuation line defeats anything reading it more simply.
    buffer = io.BytesIO()
    BytesGenerator(buffer, policy=compat32, maxheaderlen=0).flatten(message)
    return buffer.getvalue()


def read_record(wheel_path: Path, dist_info: str) -> dict[str, tuple[str, str]]:
    """Parse RECORD with installer, which owns the format's escaping rules."""
    with zipfile.ZipFile(wheel_path) as archive:
        lines = archive.read(f"{dist_info}/RECORD").decode("utf-8").splitlines()
    entries = (RecordEntry.from_elements(*row) for row in parse_record_file(lines))
    return {
        entry.path: (str(entry.hash_) if entry.hash_ else "", str(entry.size or ""))
        for entry in entries
    }


def rewrite(
    wheel_path: Path,
    output_path: Path,
    *,
    additions: dict[str, bytes],
    replacements: dict[str, bytes],
    dist_info: str,
) -> None:
    """Copy the wheel applying replacements and additions, then rebuild RECORD.

    RECORD is regenerated from what the new archive actually holds, so a wrong
    digest cannot survive the step that writes it.
    """
    record_name = f"{dist_info}/RECORD"
    if record_name in additions or record_name in replacements:
        raise WheelError("RECORD is regenerated here; do not pass it in")

    pinned = _reproducible_timestamp()
    rows: list[tuple[str, str, str]] = []
    with (
        zipfile.ZipFile(wheel_path) as source,
        zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as target,
    ):
        for info in source.infolist():
            if info.is_dir() or info.filename == record_name:
                continue
            data = replacements.get(info.filename)
            if data is None:
                data = source.read(info.filename)
            member = zipfile.ZipInfo(info.filename, date_time=pinned or info.date_time)
            member.external_attr = info.external_attr
            member.compress_type = zipfile.ZIP_DEFLATED
            target.writestr(member, data)
            rows.append((info.filename, record_hash(data), str(len(data))))

        for name in sorted(additions):
            data = additions[name]
            member = zipfile.ZipInfo(name, date_time=pinned or _ADDED_TIMESTAMP)
            member.external_attr = 0o644 << 16
            member.compress_type = zipfile.ZIP_DEFLATED
            target.writestr(member, data)
            rows.append((name, record_hash(data), str(len(data))))

        buffer = io.StringIO()
        writer = csv.writer(buffer, lineterminator="\n")
        for row in sorted(rows):
            writer.writerow(row)
        writer.writerow((record_name, "", ""))
        member = zipfile.ZipInfo(record_name, date_time=pinned or _ADDED_TIMESTAMP)
        member.external_attr = 0o644 << 16
        member.compress_type = zipfile.ZIP_DEFLATED
        target.writestr(member, buffer.getvalue())
