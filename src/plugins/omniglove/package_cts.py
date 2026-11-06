#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 Agibot Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Check and package a hand take with the official CTS viewer's bundle builder."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import zipfile

from hand_cts.labels import StepTimeline
from hand_cts.mcap_source import McapFrameSource
from hand_cts.panel import bundle
from hand_cts.panel.track import HandTee
from hand_cts.report import run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    recording = args.recording.resolve()
    required = [
        recording,
        *(recording.with_suffix(s) for s in (".labels.json", ".json", ".log")),
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        parser.error("missing take files: " + ", ".join(missing))

    before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in required}
    timeline = StepTimeline.beside(recording)
    source = HandTee(McapFrameSource(recording), timeline)
    report = run(source, timeline=timeline)
    name, data = bundle.build(report, source.tracks(), recording, timeline)
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("CTS archive failed its integrity check")
        root = name.removesuffix(".zip")
        manifest = json.loads(archive.read(f"{root}/report.json"))
        for item in manifest["inputs"]["files"]:
            path = recording.parent / item["name"]
            if item["sha256"] != before[path]:
                raise RuntimeError(f"take changed during packaging: {path}")
    if any(
        hashlib.sha256(path.read_bytes()).hexdigest() != digest
        for path, digest in before.items()
    ):
        raise RuntimeError(
            "take changed during packaging; finish recording before packaging"
        )

    output_dir = args.output_dir or recording.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / name
    with output.open("xb") as destination:
        destination.write(data)
    print(
        json.dumps(
            {
                "archive": str(output),
                "verdict": report.verdict,
                "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
                "checker": manifest["tool"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
