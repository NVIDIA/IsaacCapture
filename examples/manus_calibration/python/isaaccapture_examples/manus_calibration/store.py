# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Transactional storage for exported MCAL files on the server host.

A calibration set is selectable only through its manifest. Files are verified in a private staging
directory, hard-linked to unique final names (a link never overwrites), and the manifest is written
last through a temporary file and a no-overwrite link. A set that fails at any point leaves no
manifest, so a partial pair can never be selected later.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import secrets
import shutil
import stat
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .protocol import ExportedFile, Side

logger = logging.getLogger("isaaccapture.examples.manus_calibration.store")

MANIFEST_VERSION = 1
_ATTEMPT_RE = re.compile(r"[A-Za-z0-9_-]{1,64}")
_FILE_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\.mcal")
_SET_ID_RE = re.compile(r"[0-9]{8}T[0-9]{6}Z-[0-9a-f]{8}")
_READ_CHUNK = 1 << 20
_SET_ID_ATTEMPTS = 8


class StoreError(Exception):
    """A staged export could not be verified or published; nothing was made selectable."""


@dataclass(frozen=True)
class PublishedFile:
    side: Side
    file: str
    bytes: int
    sha256: str


@dataclass(frozen=True)
class PublishedSet:
    set_id: str
    attempt: str
    created: str
    files: tuple[PublishedFile, ...]
    manifest: Path

    def path_of(self, side: Side) -> Path:
        """Absolute path of this set's file for ``side``, in the manifest's directory tree."""
        name = next(f.file for f in self.files if f.side is side)
        return self.manifest.parent.parent / name


def _sha256_of(path: Path) -> tuple[int, str]:
    """Size and digest of a regular file, refusing symlinks."""
    flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise StoreError(f"cannot open {path.name}: {exc.strerror or exc}") from None
    # Checked on the descriptor, before fdopen: a directory opens read-only and fdopen then raises.
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise StoreError(f"{path.name} is not a regular file")
        handle = os.fdopen(fd, "rb")
    except BaseException:
        os.close(fd)
        raise
    with handle:
        digest = hashlib.sha256()
        size = 0
        while chunk := handle.read(_READ_CHUNK):
            digest.update(chunk)
            size += len(chunk)
    return size, digest.hexdigest()


def _fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


class McalStore:
    """The configured calibration directory and the only code that publishes into it."""

    def __init__(self, directory: str | os.PathLike[str]) -> None:
        self._directory = Path(directory).expanduser().resolve()

    @property
    def directory(self) -> Path:
        return self._directory

    @property
    def _staging_root(self) -> Path:
        return self._directory / ".staging"

    @property
    def _manifest_dir(self) -> Path:
        return self._directory / "manifests"

    def create_staging(self, attempt: str) -> Path:
        """Create the private (0700) staging directory the worker writes into for ``attempt``."""
        if not _ATTEMPT_RE.fullmatch(attempt):
            raise StoreError("attempt id must be 1-64 characters of [A-Za-z0-9_-]")
        try:
            self._directory.mkdir(mode=0o700, parents=True, exist_ok=True)
            self._staging_root.mkdir(mode=0o700, exist_ok=True)
            staging = self._staging_root / attempt
            staging.mkdir(mode=0o700)
        except OSError as exc:
            raise StoreError(
                f"cannot create staging for {attempt}: {exc.strerror or exc}"
            ) from None
        return staging

    def discard_staging(self, attempt: str) -> None:
        """Remove ``attempt``'s staging directory. Safe to call when it is absent."""
        staging = self._staging_root / attempt
        # Only a real directory directly under the staging root is ever removed.
        if (
            not _ATTEMPT_RE.fullmatch(attempt)
            or staging.is_symlink()
            or not staging.is_dir()
        ):
            return
        shutil.rmtree(staging, ignore_errors=True)

    def publish(
        self,
        attempt: str,
        requested: Sequence[Side],
        exports: Mapping[Side, ExportedFile],
    ) -> PublishedSet:
        """Verify every requested side's staged file and publish the set, or raise :class:`StoreError`."""
        staging = self._staging_root / attempt
        if set(exports) != set(requested):
            raise StoreError(
                f"exports {sorted(s.value for s in exports)} do not match requested sides"
            )
        verified = {side: self._verify(staging, exports[side]) for side in requested}
        created: list[Path] = []
        try:
            set_id, finals = self._link_set(staging, verified, created)
            manifest = self._write_manifest(set_id, attempt, verified, finals)
        except BaseException:
            # No manifest exists yet, so these links were never selectable; remove them.
            for path in created:
                path.unlink(missing_ok=True)
            raise
        return PublishedSet(
            set_id=set_id,
            attempt=attempt,
            created=_created_of(set_id),
            files=tuple(
                PublishedFile(
                    side, finals[side], verified[side].bytes, verified[side].sha256
                )
                for side in requested
            ),
            manifest=manifest,
        )

    def list_published(self) -> list[PublishedSet]:
        """Every manifest whose files all still exist with the recorded size and digest."""
        published = []
        if not self._manifest_dir.is_dir():
            return published
        for manifest in sorted(self._manifest_dir.glob("*.json")):
            if manifest.name.startswith("."):
                continue
            try:
                published.append(self._load(manifest))
            except (StoreError, OSError, ValueError, KeyError, TypeError) as exc:
                logger.warning(
                    "ignoring calibration manifest %s: %s", manifest.name, exc
                )
        return published

    def _verify(self, staging: Path, export: ExportedFile) -> ExportedFile:
        """Check a staged file against the worker's reported size and digest."""
        if not _FILE_RE.fullmatch(export.file):
            raise StoreError(f"unexpected export name {export.file!r}")
        path = staging / export.file
        size, digest = _sha256_of(path)
        if size == 0 or export.bytes == 0:
            raise StoreError(f"{export.file} is empty")
        if size != export.bytes or digest != export.sha256:
            raise StoreError(
                f"{export.file} does not match the size/digest the worker reported"
            )
        return export

    def _link_set(
        self,
        staging: Path,
        verified: Mapping[Side, ExportedFile],
        created: list[Path],
    ) -> tuple[str, dict[Side, str]]:
        """Hard-link the staged files to a fresh set id; a link never replaces an existing file."""
        for _ in range(_SET_ID_ATTEMPTS):
            set_id = (
                datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                + "-"
                + secrets.token_hex(4)
            )
            finals = {side: f"{set_id}_{side.value}.mcal" for side in verified}
            try:
                for side, export in verified.items():
                    os.link(staging / export.file, self._directory / finals[side])
                    created.append(self._directory / finals[side])
            except FileExistsError:
                # Id collision with an earlier set: undo this round and draw another id.
                for path in created:
                    path.unlink(missing_ok=True)
                created.clear()
                continue
            except OSError as exc:
                raise StoreError(
                    f"cannot publish calibration files: {exc.strerror or exc}"
                ) from None
            return set_id, finals
        raise StoreError("could not allocate a unique calibration set id")

    def _write_manifest(
        self,
        set_id: str,
        attempt: str,
        verified: Mapping[Side, ExportedFile],
        finals: Mapping[Side, str],
    ) -> Path:
        """Write the manifest through a temporary file and a no-overwrite link."""
        body = {
            "version": MANIFEST_VERSION,
            "set_id": set_id,
            "attempt": attempt,
            "created": _created_of(set_id),
            "files": [
                {
                    "side": side.value,
                    "file": finals[side],
                    "bytes": export.bytes,
                    "sha256": export.sha256,
                }
                for side, export in verified.items()
            ],
        }
        final = self._manifest_dir / f"{set_id}.json"
        temporary = self._manifest_dir / f".{set_id}.tmp"
        linked = False
        try:
            self._manifest_dir.mkdir(mode=0o700, exist_ok=True)
            fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(body, handle, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            try:
                os.link(temporary, final)
                linked = True
            finally:
                temporary.unlink(missing_ok=True)
            _fsync_dir(self._manifest_dir)
            _fsync_dir(self._directory)
        except OSError as exc:
            # Remove only a manifest this call created; an existing one is never touched.
            if linked:
                final.unlink(missing_ok=True)
            raise StoreError(
                f"cannot write calibration manifest: {exc.strerror or exc}"
            ) from None
        return final

    def _load(self, manifest: Path) -> PublishedSet:
        """Parse and verify one manifest; raise if anything about the set is not as recorded."""
        data = json.loads(manifest.read_text(encoding="utf-8"))
        if data["version"] != MANIFEST_VERSION or not _SET_ID_RE.fullmatch(
            data["set_id"]
        ):
            raise StoreError("unsupported manifest")
        if manifest.name != f"{data['set_id']}.json":
            raise StoreError("manifest name does not match its set id")
        files = []
        for item in data["files"]:
            if not _FILE_RE.fullmatch(item["file"]):
                raise StoreError(f"unexpected file name {item['file']!r}")
            size, digest = _sha256_of(self._directory / item["file"])
            if size != item["bytes"] or digest != item["sha256"] or size == 0:
                raise StoreError(f"{item['file']} no longer matches its manifest")
            files.append(PublishedFile(Side(item["side"]), item["file"], size, digest))
        if not files or len({f.side for f in files}) != len(files):
            raise StoreError("manifest lists no files or a side twice")
        return PublishedSet(
            data["set_id"], data["attempt"], data["created"], tuple(files), manifest
        )


def _created_of(set_id: str) -> str:
    """ISO-8601 UTC timestamp encoded in a set id."""
    return (
        datetime.strptime(set_id[:16], "%Y%m%dT%H%M%SZ")
        .replace(tzinfo=timezone.utc)
        .isoformat()
    )
