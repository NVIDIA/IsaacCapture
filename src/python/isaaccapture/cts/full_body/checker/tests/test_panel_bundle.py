# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The submission bundle."""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from pathlib import Path

import pytest
import synth
from full_body_cts.labels import StepTimeline
from full_body_cts.panel import bundle
from full_body_cts.panel.track import TeeSource
from full_body_cts.report import run

LABELS = {
    "nominal_rate_hz": 60.0,
    "steps": [
        {"index": 0, "label": "a_pose_still", "start_ns": 0, "end_ns": 1_000_000_000},
        {
            "index": 1,
            "label": "t_pose_hold_open",
            "start_ns": 1_000_000_000,
            "end_ns": 2_000_000_000,
        },
    ],
}


CAPTURE = {
    "recorded_at": "2026-01-01T00:00:00.000000+00:00",
    "device": "example",
    "repo_commit": "0" * 40,
}

# The take name, which `record.sh` gives to the directory and to every file in it.
STEM = "example_2026-01-01_000000-body"


@pytest.fixture
def take(tmp_path: Path) -> Path:
    """A recording laid out and companioned the way `record.sh` leaves one."""
    folder = tmp_path / "example_2026-01-01_000000"
    folder.mkdir(parents=True)
    recording = synth.write_recording(folder / f"{STEM}.mcap", synth.frames(120))
    recording.with_name(f"{STEM}.labels.json").write_text(json.dumps(LABELS))
    recording.with_name(f"{STEM}.json").write_text(json.dumps(CAPTURE))
    recording.with_name(f"{STEM}.log").write_text(f"[record] writing {STEM}.mcap\n")
    return recording


def packaged(recording: Path, timeline: StepTimeline | None = ..., **kwargs):
    """Runs the checks over frames in memory and packages them beside ``recording``."""
    if timeline is ...:
        timeline = StepTimeline.beside(recording)
    source = TeeSource(synth.StubSource(synth.frames(120)), timeline)
    report = run(source, None, timeline)
    name, data = bundle.build(report, source.track(), recording, timeline, **kwargs)
    return (name, data, report)


def members(data: bytes) -> dict[str, bytes]:
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        return {name: archive.read(name) for name in archive.namelist()}


def test_the_archive_holds_the_take_its_companions_and_the_report(take):
    name, data, report = packaged(take)
    root = f"{STEM}.{report.verdict}"
    assert name == f"{root}.zip"
    assert set(members(data)) == {
        f"{root}/{STEM}.mcap",
        f"{root}/{STEM}.labels.json",
        f"{root}/{STEM}.json",
        f"{root}/{STEM}.log",
        f"{root}/report.json",
        f"{root}/report.txt",
    }


def test_the_name_is_the_take_and_the_verdict(take):
    name, _, report = packaged(take)
    assert name == f"{STEM}.{report.verdict}.zip"


def test_the_name_does_not_depend_on_the_capture_sidecar(take):
    take.with_name(f"{STEM}.json").unlink()
    name, _, report = packaged(take)
    assert name == f"{STEM}.{report.verdict}.zip"


def test_a_take_from_nowhere_in_particular_still_gets_a_name(tmp_path):
    """A copy outside the capture tree must not crash or produce a bare suffix."""
    loose = synth.write_recording(tmp_path / "hand-copied.mcap", synth.frames(10))
    name, _, report = packaged(loose, timeline=None)
    assert name.endswith(f"hand-copied.{report.verdict}.zip")


def test_a_stem_that_would_escape_the_filename_cannot(tmp_path):
    """The device name reaches the stem from a shell argument, through record.sh."""
    folder = tmp_path / "odd"
    folder.mkdir()
    odd = synth.write_recording(folder / "..pico 4_2026-01-01_000000-body.mcap", [])
    name, _, _ = packaged(odd, timeline=None)
    assert name.startswith("pico_4_2026-01-01_000000-body.")
    assert "/" not in name and " " not in name


def test_the_packed_recording_is_byte_for_byte_the_source(take):
    _, data, report = packaged(take)
    inside = members(data)
    packed = inside[f"{STEM}.{report.verdict}/{STEM}.mcap"]
    assert packed == take.read_bytes()

    payload = json.loads(inside[f"{STEM}.{report.verdict}/report.json"])
    entry = next(
        item for item in payload["inputs"]["files"] if item["name"].endswith(".mcap")
    )
    assert entry["sha256"] == hashlib.sha256(packed).hexdigest()
    assert entry["bytes"] == len(packed)


def test_report_json_survives_a_strict_parser(take):
    """A bare NaN is legal to `json` and fatal to `JSON.parse`, which reads this."""

    def refuse(constant: str) -> None:
        raise AssertionError(f"non-finite {constant} would break JSON.parse")

    _, data, report = packaged(take)
    raw = members(data)[f"{STEM}.{report.verdict}/report.json"]
    json.loads(raw, parse_constant=refuse)


def test_every_packed_mark_is_the_checkers_own(take):
    _, data, report = packaged(take)
    payload = json.loads(members(data)[f"{STEM}.{report.verdict}/report.json"])
    assert {check["name"]: check["mark"] for check in payload["checks"]} == {
        result.name: str(result.mark) for result in report.results
    }


def test_the_groups_name_every_packed_check_once(take):
    _, data, report = packaged(take)
    payload = json.loads(members(data)[f"{STEM}.{report.verdict}/report.json"])
    grouped = [name for group in payload["groups"] for name in group["checks"]]
    assert sorted(grouped) == sorted(check["name"] for check in payload["checks"])


def test_the_series_carries_every_frame_and_nulls_a_rate_it_cannot_state(take):
    _, data, report = packaged(take)
    payload = json.loads(members(data)[f"{STEM}.{report.verdict}/report.json"])
    series = payload["series"]
    assert len(series["t_ms"]) == len(series["rate_hz"]) == report.frames
    # The first frame has no interval to divide, so it has no rate.
    assert series["rate_hz"][0] is None
    assert series["rate_hz"][1] is not None


def test_a_take_with_no_labels_says_which_input_is_missing(take):
    take.with_name(f"{STEM}.labels.json").unlink()
    _, data, report = packaged(take, timeline=None)
    payload = json.loads(members(data)[f"{STEM}.{report.verdict}/report.json"])
    assert payload["inputs"]["missing"] == [f"{STEM}.labels.json"]
    assert payload["inputs"]["labels"] == {
        "present": False,
        "steps": None,
        "source": None,
    }


def test_labels_read_from_elsewhere_are_packed_from_where_they_were_read(
    take, tmp_path
):
    take.with_name(f"{STEM}.labels.json").unlink()
    elsewhere = tmp_path / "moved" / "session.labels.json"
    elsewhere.parent.mkdir()
    elsewhere.write_text(json.dumps(LABELS))

    _, data, report = packaged(take, timeline=StepTimeline.load(elsewhere))
    root = f"{STEM}.{report.verdict}"
    assert f"{root}/session.labels.json" in members(data)
    payload = json.loads(members(data)[f"{root}/report.json"])
    assert payload["inputs"]["labels"] == {
        "present": True,
        "steps": 2,
        "source": "session.labels.json",
    }


def test_progress_runs_forward_and_finishes(take):
    seen: list[float] = []
    packaged(take, on_progress=seen.append)
    assert seen == sorted(seen)
    assert seen[-1] == 1.0
    assert min(seen) > 0.0


def test_two_builds_of_one_take_agree(take):
    _, first, report = packaged(take)
    _, second, _ = packaged(take)
    name = f"{STEM}.{report.verdict}/report.json"
    # The zip bytes differ by member mtime; the contents must not.
    assert members(first)[name] == members(second)[name]


def test_the_tool_block_names_the_checker_that_ran(take):
    _, data, report = packaged(take)
    payload = json.loads(members(data)[f"{STEM}.{report.verdict}/report.json"])
    tool = payload["tool"]
    assert tool["checks"] == len(report.results)
    # Null where git cannot answer, which is a submitter working from an archive.
    assert tool["commit"] is None or len(tool["commit"]) == 40
    assert tool["dirty"] in (True, False, None)


def test_a_report_with_no_results_still_packages(take):
    """An empty report has no groups and no worst mark; neither may raise."""
    source = TeeSource(synth.StubSource([]))
    empty = run(source, [], None)
    _, data = bundle.build(empty, source.track(), take)
    payload = json.loads(members(data)[f"{STEM}.{empty.verdict}/report.json"])
    assert payload["groups"] == []
    assert payload["series"]["t_ms"] == []
