#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 RealHand. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fetch pinned P7 and RealHand robot assets from the official Hugging Face release."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any


REPO_ID = "realhandinc/realhand-teleop"
RELEASE_NAME = "v0.2.0"
REVISION = "e20d1f7dfe27d84784270012f291ae16f4762b0c"
REMOTE_ASSET_ROOT = "examples/isaac_lab/p7_realhand_bimanual/assets/robots"
MODEL_ASSET_DIRS = {"l6": "p7_l6", "o6": "p7_o6", "l20": "p7_l20"}
MODEL_URDFS = {
    "l6": "P7_l6_bimanual.urdf",
    "o6": "P7_o6_bimanual.urdf",
    "l20": "P7_L20_bimanual.urdf",
}
URDF_SHA256 = {
    "l6": "5a504b7fb055a2401c0e28e8be1e2f9d642b28a00c4d9e38021072ed4c4a72d2",
    "o6": "1d5750bc6ee0b0d8a90ac7ea2a747d42d2a5d73cadd0244493e4815ddd370b32",
    "l20": "6194e9dafd98067d5c08f7200ec6c61aefa31578d453da2f858fbe10ad7dc99a",
}
LICENSE_FILES = {
    "LICENSE",
    "NOTICE",
    "LICENSES/Apache-2.0.txt",
    "LICENSES/BSD-3-Clause.txt",
}


@dataclass(frozen=True)
class RemoteFile:
    path: str
    size: int
    sha256: str | None


def _example_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_output_dir() -> Path:
    return _example_root() / "assets" / "realhand"


def _request_json(url: str) -> dict[str, Any]:
    request = urllib.request.Request(
        url, headers={"User-Agent": "IsaacCapture-RealHand-assets/1.0"}
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.load(response)
    except (urllib.error.URLError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Failed to read Hugging Face metadata from {url}: {exc}"
        ) from exc


def _repository_metadata() -> dict[str, Any]:
    quoted_repo = urllib.parse.quote(REPO_ID, safe="/")
    quoted_revision = urllib.parse.quote(REVISION, safe="")
    metadata = _request_json(
        f"https://huggingface.co/api/models/{quoted_repo}/revision/{quoted_revision}?blobs=true"
    )
    actual_revision = metadata.get("sha")
    if actual_revision != REVISION:
        raise RuntimeError(
            f"Hugging Face resolved revision {REVISION} to {actual_revision!r}; refusing an unpinned download"
        )
    return metadata


def _normalize_remote_path(path: str) -> str:
    normalized = PurePosixPath(path)
    if normalized.is_absolute() or ".." in normalized.parts:
        raise ValueError(f"Unsafe repository path: {path!r}")
    return normalized.as_posix()


def _selected_files(metadata: dict[str, Any], models: set[str]) -> list[RemoteFile]:
    prefixes = {f"{REMOTE_ASSET_ROOT}/{MODEL_ASSET_DIRS[model]}/" for model in models}
    selected: list[RemoteFile] = []
    for sibling in metadata.get("siblings", []):
        path = _normalize_remote_path(str(sibling.get("rfilename", "")))
        if path not in LICENSE_FILES and not any(
            path.startswith(prefix) for prefix in prefixes
        ):
            continue
        size = sibling.get("size")
        if not isinstance(size, int) or size < 0:
            raise RuntimeError(f"Missing file size in Hugging Face metadata for {path}")
        lfs = sibling.get("lfs")
        sha256 = lfs.get("sha256") if isinstance(lfs, dict) else None
        if sha256 is not None and (not isinstance(sha256, str) or len(sha256) != 64):
            raise RuntimeError(
                f"Invalid LFS SHA-256 in Hugging Face metadata for {path}"
            )
        selected.append(RemoteFile(path=path, size=size, sha256=sha256))

    required_prefixes = {
        prefix
        for prefix in prefixes
        if not any(item.path.startswith(prefix) for item in selected)
    }
    missing_licenses = LICENSE_FILES - {item.path for item in selected}
    if required_prefixes or missing_licenses:
        missing = sorted(required_prefixes | missing_licenses)
        raise RuntimeError(
            f"Pinned RealHand release is missing required paths: {', '.join(missing)}"
        )
    return sorted(selected, key=lambda item: item.path)


def _local_path(remote: RemoteFile, output_dir: Path) -> Path:
    prefix = f"{REMOTE_ASSET_ROOT}/"
    relative = (
        remote.path[len(prefix) :] if remote.path.startswith(prefix) else remote.path
    )
    destination = (output_dir / Path(relative)).resolve()
    output_root = output_dir.resolve()
    if destination != output_root and output_root not in destination.parents:
        raise ValueError(f"Unsafe output path for {remote.path!r}")
    return destination


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_complete(path: Path, remote: RemoteFile) -> bool:
    if not path.is_file() or path.stat().st_size != remote.size:
        return False
    return remote.sha256 is None or _sha256(path) == remote.sha256


def _download(remote: RemoteFile, output_dir: Path, force: bool) -> tuple[Path, bool]:
    destination = _local_path(remote, output_dir)
    if not force and _is_complete(destination, remote):
        return destination, False

    quoted_repo = urllib.parse.quote(REPO_ID, safe="/")
    quoted_revision = urllib.parse.quote(REVISION, safe="")
    quoted_path = urllib.parse.quote(remote.path, safe="/")
    url = (
        f"https://huggingface.co/{quoted_repo}/resolve/{quoted_revision}/{quoted_path}"
    )
    request = urllib.request.Request(
        url, headers={"User-Agent": "IsaacCapture-RealHand-assets/1.0"}
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.part-{os.getpid()}")
    digest = hashlib.sha256()
    size = 0
    try:
        with (
            urllib.request.urlopen(request, timeout=120) as response,
            temporary.open("wb") as stream,
        ):
            while chunk := response.read(1024 * 1024):
                stream.write(chunk)
                digest.update(chunk)
                size += len(chunk)
        if size != remote.size:
            raise RuntimeError(
                f"Downloaded {remote.path} has size {size}, expected {remote.size}"
            )
        if remote.sha256 is not None and digest.hexdigest() != remote.sha256:
            raise RuntimeError(
                f"Downloaded {remote.path} has SHA-256 {digest.hexdigest()}, expected {remote.sha256}"
            )
        temporary.replace(destination)
    except (OSError, urllib.error.URLError) as exc:
        raise RuntimeError(f"Failed to download {remote.path}: {exc}") from exc
    finally:
        temporary.unlink(missing_ok=True)
    return destination, True


def _validate_model(output_dir: Path, model: str) -> Path:
    asset_dir = output_dir / MODEL_ASSET_DIRS[model]
    urdf_path = asset_dir / MODEL_URDFS[model]
    if not urdf_path.is_file():
        raise FileNotFoundError(
            f"Downloaded {model.upper()} URDF is missing: {urdf_path}"
        )
    actual_sha256 = _sha256(urdf_path)
    if actual_sha256 != URDF_SHA256[model]:
        raise RuntimeError(
            f"{urdf_path} has SHA-256 {actual_sha256}, expected {URDF_SHA256[model]}"
        )

    root = ET.parse(urdf_path).getroot()
    if root.tag != "robot":
        raise ValueError(f"Expected a URDF robot root in {urdf_path}")
    missing_meshes: list[str] = []
    for mesh in root.findall(".//mesh"):
        filename = mesh.attrib.get("filename")
        if not filename:
            raise ValueError(f"URDF mesh without a filename in {urdf_path}")
        mesh_path = PurePosixPath(filename)
        if mesh_path.is_absolute() or ".." in mesh_path.parts or "://" in filename:
            raise ValueError(f"Unsupported mesh path {filename!r} in {urdf_path}")
        if not (asset_dir / Path(mesh_path.as_posix())).is_file():
            missing_meshes.append(filename)
    if missing_meshes:
        preview = ", ".join(sorted(set(missing_meshes))[:8])
        raise FileNotFoundError(f"{urdf_path} references missing meshes: {preview}")
    return urdf_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hand-model",
        choices=("l6", "o6", "l20", "all"),
        default="all",
        help="Robot-hand assembly to download (default: all).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_default_output_dir(),
        help="Destination containing p7_l6, p7_o6, and p7_l20 directories.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Download files that already pass validation.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List the pinned files without downloading them.",
    )
    parser.add_argument(
        "--workers", type=int, default=6, help="Parallel downloads (default: 6)."
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    models = set(MODEL_ASSET_DIRS) if args.hand_model == "all" else {args.hand_model}
    metadata = _repository_metadata()
    files = _selected_files(metadata, models)
    total_bytes = sum(item.size for item in files)
    print(
        f"RealHand assets {RELEASE_NAME} ({REVISION}) from {REPO_ID}: "
        f"{len(files)} files, {total_bytes / (1024 * 1024):.1f} MiB"
    )
    if args.dry_run:
        for item in files:
            print(item.path)
        return 0
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")

    downloaded = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(_download, item, args.output_dir, args.force)
            for item in files
        ]
        for index, future in enumerate(
            concurrent.futures.as_completed(futures), start=1
        ):
            path, changed = future.result()
            downloaded += int(changed)
            print(
                f"[{index}/{len(files)}] {'Downloaded' if changed else 'Verified'} {path}"
            )

    for model in sorted(models):
        print(f"Validated {model.upper()}: {_validate_model(args.output_dir, model)}")
    print(f"Ready: downloaded {downloaded}, reused {len(files) - downloaded}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
