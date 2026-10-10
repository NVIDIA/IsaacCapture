# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One zip archive holding a take and everything needed to judge it again."""

from __future__ import annotations

import hashlib
import io
import json
import subprocess
import zipfile
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from ..labels import SIDECAR_SUFFIX, StepTimeline
from ..report import Report
from . import status
from .track import Track

CHUNK = 1 << 20

# Companion file suffixes beside the recording: capture provenance and the recorder log.
PROVENANCE_SUFFIXES = (".json", ".log")


def build(
    report: Report,
    track: Track,
    recording: Path | str,
    timeline: StepTimeline | None = None,
    on_progress: Callable[[float], None] | None = None,
) -> tuple[str, bytes]:
    """The archive's filename and its bytes. ``on_progress`` is called with 0.0–1.0."""
    recording = Path(recording)
    present, missing = companions(recording, timeline)
    packed = [recording, *present]
    total = sum(path.stat().st_size for path in packed) or 1
    root = archive_stem(recording, report)

    buffer = io.BytesIO()
    done = 0
    files: list[dict[str, Any]] = []
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in packed:
            digest = hashlib.sha256()
            size = 0
            with archive.open(f"{root}/{path.name}", "w") as member:
                with path.open("rb") as source:
                    while chunk := source.read(CHUNK):
                        member.write(chunk)
                        digest.update(chunk)
                        size += len(chunk)
                        done += len(chunk)
                        if on_progress is not None:
                            on_progress(min(done / total, 1.0))
            files.append(
                {"name": path.name, "bytes": size, "sha256": digest.hexdigest()}
            )
        payload = manifest(report, track, timeline, files, missing)
        archive.writestr(
            f"{root}/report.json", json.dumps(payload, indent=2, allow_nan=False)
        )
        archive.writestr(f"{root}/report.txt", report.to_text())
    if on_progress is not None:
        on_progress(1.0)
    return (f"{root}.zip", buffer.getvalue())


def manifest(
    report: Report,
    track: Track,
    timeline: StepTimeline | None,
    files: Sequence[dict[str, Any]],
    missing: Iterable[str],
) -> dict[str, Any]:
    """``report.to_dict()`` plus inputs, groups and series; the packed results are a cache and the recording wins."""
    return {
        "tool": tool(len(report.results)),
        "inputs": {
            "files": list(files),
            "labels": _labels(timeline),
            "missing": list(missing),
        },
        **report.to_dict(),
        "groups": [
            {
                "code": group.code,
                "title": group.title,
                "checks": [result.name for result in group.results],
            }
            for group in status.groups(report)
        ],
        "series": series(track),
    }


def series(track: Track) -> dict[str, list[Any]]:
    """Frame rate and valid-joint count against time, columnar.

    A missing rate is ``null``, never NaN.
    """
    samples = track.samples
    return {
        "t_ms": [round(sample.t_s * 1000.0, 1) for sample in samples],
        "rate_hz": [
            None if sample.rate_hz != sample.rate_hz else round(sample.rate_hz, 2)
            for sample in samples
        ],
        "valid": [sample.valid_count for sample in samples],
    }


def archive_stem(recording: Path, report: Report) -> str:
    """``<recording stem>.<verdict>``."""
    return f"{_filename_safe(recording.stem)}.{report.verdict}"


def companions(
    recording: Path, timeline: StepTimeline | None = None
) -> tuple[list[Path], list[str]]:
    """Which files beside the recording exist, and the names of those that do not."""
    labels = (
        Path(timeline.source)
        if timeline is not None and timeline.source
        else recording.with_name(recording.stem + SIDECAR_SUFFIX)
    )
    wanted = [labels] + [
        recording.with_name(recording.stem + suffix) for suffix in PROVENANCE_SUFFIXES
    ]
    return (
        [path for path in wanted if path.is_file()],
        [path.name for path in wanted if not path.is_file()],
    )


def tool(checks: int) -> dict[str, Any]:
    """The checker's git commit, whether the package is dirty, and the check count.

    ``commit`` and ``dirty`` are null where git cannot answer.
    """
    package = Path(__file__).resolve().parent.parent
    changed = _git(package, "status", "--porcelain", "--", str(package))
    return {
        "commit": _git(package, "rev-parse", "HEAD"),
        "dirty": None if changed is None else bool(changed),
        "checks": checks,
    }


def _filename_safe(text: str) -> str:
    """``text`` with anything other than alphanumerics and ``-._`` replaced by ``_``."""
    return "".join(c if c.isalnum() or c in "-._" else "_" for c in text).strip("_.")


def _labels(timeline: StepTimeline | None) -> dict[str, Any]:
    """Label summary with the same keys whether or not labels are present."""
    if timeline is None:
        return {"present": False, "steps": None, "source": None}
    return {
        "present": True,
        "steps": len(timeline.steps),
        "source": Path(timeline.source).name if timeline.source else None,
    }


def _git(cwd: Path, *args: str) -> str | None:
    try:
        done = subprocess.run(
            ("git", "-C", str(cwd), *args),
            check=False,
            capture_output=True,
            text=True,
            timeout=5.0,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout.strip() if done.returncode == 0 else None
