# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""The acceptance replay tool, run as a real subprocess against a real recorded episode."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from real import FIXTURE, needs_real_episode

TOOL = Path(__file__).resolve().parents[2] / "recording" / "tools" / "replay_episode.py"


def _run(tmp_path: Path, *extra: str) -> subprocess.CompletedProcess:
    """Run the tool from a neutral working directory with the test process's environment."""
    return subprocess.run(
        [sys.executable, str(TOOL), *extra],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        env=os.environ.copy(),
        timeout=600,
        check=False,
    )


@needs_real_episode
def test_replay_passes_on_real_data_and_reports_identity_and_exact_counts(tmp_path):
    """A bounded real replay passes every acceptance check, including camera identity and the swapped control."""
    report_path = tmp_path / "report.json"
    done = _run(
        tmp_path,
        "--input", FIXTURE, "--manifest", str(tmp_path / "m.json"), "--write-manifest",
        "--output-root", str(tmp_path / "out"), "--rows", "40", "--report", str(report_path),
    )  # fmt: skip
    assert done.returncode == 0, done.stdout[-2000:] + done.stderr[-2000:]
    report = json.loads(report_path.read_text())
    assert report["passed"] and all(report["checks"].values()), report["checks"]
    assert report["swapped_camera_control"]["detected"] is True
    assert all(f["passed"] for f in report["camera_identity"].values()) and len(report["camera_identity"]) == 3
    assert all(a == w == 40 for a, w in report["counts"].values())


def _base(manifest: Path, output_root: Path) -> list[str]:
    """Common arguments: the real input, a manifest path and an output root."""
    return ["--input", FIXTURE, "--manifest", str(manifest), "--output-root", str(output_root)]


@needs_real_episode
def test_replay_refuses_bad_rows_changed_inputs_and_overlapping_output(tmp_path):
    """Non-positive --rows, a manifest mismatch and an output root inside the input are all refused."""
    manifest = tmp_path / "m.json"
    ok = _run(tmp_path, *_base(manifest, tmp_path / "o1"), "--write-manifest", "--rows", "5")
    assert ok.returncode == 0, ok.stdout[-1500:]
    bad_rows = _run(tmp_path, *_base(manifest, tmp_path / "o2"), "--rows", "0")
    assert bad_rows.returncode == 2 and "--rows" in bad_rows.stderr
    data = json.loads(manifest.read_text())
    data["recorded_at.txt"]["sha256"] = "0" * 64
    manifest.write_text(json.dumps(data))
    changed = _run(tmp_path, *_base(manifest, tmp_path / "o3"))
    assert changed.returncode == 1 and "changed: recorded_at.txt" in changed.stdout
    nested = _run(tmp_path, *_base(tmp_path / "m2.json", Path(FIXTURE) / "out"), "--write-manifest")
    assert nested.returncode == 1


def test_missing_input_is_reported_as_skipped_not_passed(tmp_path):
    """A missing input directory exits 2 and says SKIPPED."""
    args = [
        "--input",
        str(tmp_path / "nope"),
        "--manifest",
        str(tmp_path / "m.json"),
        "--output-root",
        str(tmp_path),
    ]
    done = _run(tmp_path, *args)
    assert done.returncode == 2 and "SKIPPED" in done.stdout


def test_manifest_helpers_detect_each_kind_of_difference(tmp_path):
    """Missing, unexpected and changed files are all reported by the manifest verifier."""
    sys.path.insert(0, str(TOOL.parent))
    pytest.importorskip("egocap")
    import replay_episode as tool  # noqa: PLC0415

    (tmp_path / "a.txt").write_text("a")
    (tmp_path / "b.txt").write_text("b")
    manifest = tool.build_manifest(tmp_path)
    assert tool.verify_manifest(tmp_path, manifest) == []
    (tmp_path / "a.txt").write_text("changed")
    (tmp_path / "b.txt").unlink()
    (tmp_path / "c.txt").write_text("new")
    assert sorted(tool.verify_manifest(tmp_path, manifest)) == [
        "changed: a.txt",
        "missing: b.txt",
        "unexpected: c.txt",
    ]
