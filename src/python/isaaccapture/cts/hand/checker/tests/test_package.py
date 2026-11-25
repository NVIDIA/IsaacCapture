# SPDX-FileCopyrightText: Copyright (c) 2026 Agibot Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Headless packaging preserves the take and rejects incomplete or changing inputs."""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import pytest
import synth

from hand_cts import package


@pytest.fixture
def take(tmp_path: Path) -> Path:
    recording = synth.write_recording(tmp_path / "take-hand.mcap", synth.both_hands(60))
    recording.with_suffix(".labels.json").write_text('{"steps": []}')
    recording.with_suffix(".json").write_text('{"device": "example"}')
    recording.with_suffix(".log").write_text("capture log\n")
    return recording


def test_headless_archive_preserves_all_inputs(take, tmp_path, capsys):
    paths = [take, *(take.with_suffix(s) for s in (".labels.json", ".json", ".log"))]
    before = {path.name: path.read_bytes() for path in paths}
    package.main([str(take), "--output-dir", str(tmp_path / "output")])
    result = json.loads(capsys.readouterr().out)
    archive = Path(result["archive"])
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == result["sha256"]
    with zipfile.ZipFile(archive) as bundle:
        assert bundle.testzip() is None
        root = archive.stem
        report = json.loads(bundle.read(f"{root}/report.json"))
        assert report["verdict"] == result["verdict"]
        for item in report["inputs"]["files"]:
            original = before[item["name"]]
            assert bundle.read(f"{root}/{item['name']}") == original
            assert item["sha256"] == hashlib.sha256(original).hexdigest()
    assert {path.name: path.read_bytes() for path in paths} == before


def test_headless_archive_never_overwrites(take, capsys):
    package.main([str(take)])
    output = Path(json.loads(capsys.readouterr().out)["archive"])
    original = output.read_bytes()
    with pytest.raises(FileExistsError):
        package.main([str(take)])
    assert output.read_bytes() == original


def test_headless_packaging_rejects_incomplete_takes(take):
    take.with_suffix(".labels.json").unlink()
    with pytest.raises(SystemExit) as exited:
        package.main([str(take)])
    assert exited.value.code == 2
    assert not list(take.parent.glob("*.zip"))


def test_headless_packaging_rejects_a_concurrent_input_change(take, monkeypatch):
    original = package.bundle.build

    def changing_input(*args, **kwargs):
        archive = original(*args, **kwargs)
        take.with_suffix(".log").write_text("modified during packaging\n")
        return archive

    monkeypatch.setattr(package.bundle, "build", changing_input)
    with pytest.raises(RuntimeError, match="take changed during packaging"):
        package.main([str(take)])
    assert not list(take.parent.glob("*.zip"))
