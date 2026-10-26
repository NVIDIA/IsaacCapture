# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One-snapshot local packages, published with a single directory replacement."""

from __future__ import annotations

from collections.abc import Mapping
import csv
from dataclasses import dataclass
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import stat
import tempfile

from .checks import (
    CheckResult,
    check_device,
    check_summary,
    checks_exit_code,
    declared_source_indices,
)
from .document import metadata_payload, output_taxels, raw_values
from .domain import FrozenMap, Issue, Snapshot, thaw


@dataclass(frozen=True)
class ExportOptions:
    out: str | Path
    format: str = "json"


@dataclass(frozen=True)
class ExportResult:
    path: Path
    revision: int
    checks: tuple[CheckResult, ...]
    manifest: FrozenMap
    report: FrozenMap


class ExportFailure(ValueError):
    def __init__(self, issues, checks=(), *, exit_code=1):
        self.issues = tuple(issues)
        self.checks = tuple(checks)
        self.exit_code = exit_code
        super().__init__("; ".join(issue.message for issue in self.issues))


def _target_available(path):
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError:
        return
    if stat.S_ISLNK(mode):
        raise ValueError("Export target must not be a symbolic link")
    if not stat.S_ISDIR(mode):
        raise ValueError("Export target must be a new or empty directory")
    if any(path.iterdir()):
        raise ValueError("Export target is not empty; choose a new directory")


def _target_path(value):
    path = Path(value).expanduser()
    # Do not resolve the final component before rejecting symbolic links.
    _target_available(path)
    return path.absolute()


def _publish_directory(staging, target):
    if staging.parent.resolve() != target.parent.resolve():
        raise ValueError("Package staging must share the target's parent")
    _target_available(target)
    os.replace(staging, target)


def _write_bytes(path, value):
    with path.open("xb") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())


def _write_json(path, value):
    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    _write_bytes(path, text.encode("utf-8"))


def _file_info(path):
    payload = path.read_bytes()
    return {
        "name": path.name,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }


def _csv_bytes(device, source_indices):
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(("taxel_index", "taxel_id", "input_taxel_index"))
    for taxel in output_taxels(device):
        writer.writerow((
            int(taxel.taxel_index), taxel.id, source_indices.get(taxel.id)
        ))
    return stream.getvalue().encode("utf-8")


def _selected_devices(snapshot, device_ids):
    identifiers = tuple(device_ids)
    if not identifiers or len(set(identifiers)) != len(identifiers):
        raise ExportFailure((Issue(
            "error", "export_selection", "Select distinct devices for export",
            field="devices",
        ),))
    issues, devices, sides, collections = [], [], set(), set()
    for identifier in identifiers:
        try:
            device = snapshot.document.device(identifier)
        except KeyError:
            issues.append(Issue(
                "error", "device_missing", "Export device does not exist",
                identifier, field="devices",
            ))
            continue
        devices.append(device)
        if device.side not in ("left", "right") or device.side in sides:
            issues.append(Issue(
                "error", "export_side", "Export needs distinct left/right sides",
                device.id, field="side",
            ))
        sides.add(device.side)
        binding = device.application_binding
        collection = (
            binding.get("tactile_collection_id")
            if isinstance(binding, Mapping) else None
        )
        if isinstance(collection, str) and collection:
            if collection in collections:
                issues.append(Issue(
                    "error", "collection_duplicate",
                    "Export collection IDs must be distinct when present",
                    device.id, field="application_binding.tactile_collection_id",
                ))
            collections.add(collection)
        _, binding_issues = declared_source_indices(device)
        issues.extend(binding_issues)
    if issues:
        raise ExportFailure(issues)
    return tuple(devices)


def export_package(
    snapshot: Snapshot, device_ids, options: ExportOptions, model=None
) -> ExportResult:
    if options.format not in ("json", "sdk"):
        raise ValueError("Export format must be json or sdk")
    target = _target_path(options.out)
    devices = _selected_devices(snapshot, device_ids)
    use_sdk = options.format == "sdk"
    results = tuple(
        check_device(snapshot, device.id, model, use_sdk=use_sdk)
        for device in devices
    )
    code = checks_exit_code(results, use_sdk=use_sdk)
    if code:
        raise ExportFailure(
            (issue for result in results for issue in result.issues),
            results, exit_code=code,
        )
    report = {
        "format": "isaaccapture.tactile-report",
        "version": 1,
        "revision": snapshot.revision,
        "devices": [check_summary(result) for result in results],
    }
    manifest = {
        "format": "isaaccapture.tactile-package",
        "version": 1,
        "revision": snapshot.revision,
        "devices": [],
        "files": [],
    }
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
    try:
        for device, checked in zip(devices, results):
            stem = device.side
            names = {
                "metadata": f"{stem}.metadata.json",
                "values": f"{stem}.values.json",
                "taxels": f"{stem}.taxels.csv",
            }
            # Compiled values prove validity; only Document supplies saved numbers.
            metadata = metadata_payload(
                snapshot, device.id, path=target / names["metadata"],
                values=checked.compiled.values(),
            )
            _write_json(staging / names["metadata"], metadata)
            _write_json(
                staging / names["values"], raw_values(device, output_order=True)
            )
            source_indices, _ = declared_source_indices(device)
            _write_bytes(
                staging / names["taxels"], _csv_bytes(device, source_indices)
            )
            if use_sdk:
                encoded = checked.sdk_result.encoded
                if not encoded or len(encoded) != checked.sdk_result.report.get(
                    "encoded_size"
                ):
                    raise ValueError("SDK binary does not match its checked report")
                names["binary"] = f"{stem}.metadata.bin"
                _write_bytes(staging / names["binary"], encoded)
            binding = device.application_binding
            binding = binding if isinstance(binding, Mapping) else {}
            manifest["devices"].append({
                "device_id": device.id,
                "side": device.side,
                "collection_id": binding.get("tactile_collection_id") or None,
                "hand_stream_ref": thaw(binding.get("hand_source")),
                "files": names,
            })
            manifest["files"].extend(
                _file_info(staging / name) for name in names.values()
            )
        _write_json(staging / "report.json", report)
        manifest["files"].append(_file_info(staging / "report.json"))
        _write_json(staging / "manifest.json", manifest)
        _publish_directory(staging, target)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return ExportResult(
        target, snapshot.revision, results, FrozenMap(manifest), FrozenMap(report)
    )
