# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The hand submission bundle."""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from pathlib import Path

import pytest
import synth

from hand_cts.labels import StepTimeline
from hand_cts.panel import bundle
from hand_cts.panel.track import HandTee
from hand_cts.report import run

LABELS = {
    "nominal_rate_hz": 60.0,
    "steps": [
        {
            "index": 0,
            "label": "flat_on_table_open",
            "start_ns": 1_000_000_000,
            "end_ns": 1_500_000_000,
        },
        {
            "index": 1,
            "label": "fist",
            "start_ns": 1_500_000_000,
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
STEM = "example_2026-01-01_000000-hand"

# Frames per hand.
FRAMES = 60

# 2.56 MB, more than one 1 MiB read chunk, so progress moves during the recording.
RECORDING = bytes(range(256)) * 10_000


@pytest.fixture
def take(tmp_path: Path) -> Path:
    """A recording laid out and companioned the way `record.sh` leaves one."""
    folder = tmp_path / "example_2026-01-01_000000"
    folder.mkdir(parents=True)
    recording = folder / f"{STEM}.mcap"
    recording.write_bytes(RECORDING)
    recording.with_name(f"{STEM}.labels.json").write_text(json.dumps(LABELS))
    recording.with_name(f"{STEM}.json").write_text(json.dumps(CAPTURE))
    recording.with_name(f"{STEM}.log").write_text(f"[record] writing {STEM}.mcap\n")
    return recording


def packaged(
    recording: Path,
    timeline: StepTimeline | None = ...,
    frames: list | None = None,
    **kwargs,
):
    """Runs the checks over frames in memory and packages them beside ``recording``."""
    if timeline is ...:
        timeline = StepTimeline.beside(recording)
    source = HandTee(
        synth.StubSource(synth.both_hands(FRAMES) if frames is None else frames),
        timeline,
    )
    report = run(source, None, timeline)
    name, data = bundle.build(report, source.tracks(), recording, timeline, **kwargs)
    return (name, data, report)


def members(data: bytes) -> dict[str, bytes]:
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        return {name: archive.read(name) for name in archive.namelist()}


def report_json(data: bytes, report) -> dict:
    return json.loads(members(data)[f"{STEM}.{report.verdict}/report.json"])


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
    loose = tmp_path / "hand-copied.mcap"
    loose.write_bytes(RECORDING)
    name, _, report = packaged(loose, timeline=None)
    assert name.endswith(f"hand-copied.{report.verdict}.zip")


def test_a_stem_that_would_escape_the_filename_cannot(tmp_path):
    """The device name reaches the stem from a shell argument, through record.sh."""
    folder = tmp_path / "odd"
    folder.mkdir()
    odd = folder / "..manus 2_2026-01-01_000000-hand.mcap"
    odd.write_bytes(b"")
    name, _, _ = packaged(odd, timeline=None, frames=[])
    assert name.startswith("manus_2_2026-01-01_000000-hand.")
    assert "/" not in name and " " not in name


def test_the_packed_recording_is_byte_for_byte_the_source(take):
    _, data, report = packaged(take)
    inside = members(data)
    packed = inside[f"{STEM}.{report.verdict}/{STEM}.mcap"]
    assert packed == take.read_bytes()

    payload = report_json(data, report)
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
    payload = report_json(data, report)
    assert {check["name"]: check["mark"] for check in payload["checks"]} == {
        result.name: str(result.mark) for result in report.results
    }


def test_the_groups_are_the_hand_groups_and_name_every_packed_check_once(take):
    _, data, report = packaged(take)
    payload = report_json(data, report)
    assert [group["code"] for group in payload["groups"]] == [
        "signal",
        "shape",
        "placement",
        "windows",
    ]
    grouped = [name for group in payload["groups"] for name in group["checks"]]
    assert sorted(grouped) == sorted(check["name"] for check in payload["checks"])


def test_the_series_carries_every_frame_of_each_hand_on_one_time_axis(take):
    _, data, report = packaged(take)
    series = report_json(data, report)["series"]
    assert set(series) == {"left", "right"}
    for side in ("left", "right"):
        hand = series[side]
        assert len(hand["t_ms"]) == len(hand["rate_hz"]) == len(hand["valid"])
        assert len(hand["t_ms"]) == FRAMES
        # The first frame has no interval to divide, so it has no rate.
        assert hand["rate_hz"][0] is None
        assert hand["rate_hz"][1] is not None
    assert series["left"]["t_ms"] == series["right"]["t_ms"]
    assert series["left"]["t_ms"][0] == 0.0


def test_a_hand_with_no_channel_packs_an_empty_series(take):
    _, data, report = packaged(take, frames=synth.take(FRAMES, "right"))
    series = report_json(data, report)["series"]
    assert series["left"] == {"t_ms": [], "rate_hz": [], "valid": []}
    assert len(series["right"]["t_ms"]) == FRAMES


def test_a_take_with_no_labels_says_which_input_is_missing(take):
    take.with_name(f"{STEM}.labels.json").unlink()
    _, data, report = packaged(take, timeline=None)
    payload = report_json(data, report)
    assert payload["inputs"]["missing"] == [f"{STEM}.labels.json"]
    assert payload["inputs"]["labels"] == {
        "present": False,
        "steps": None,
        "source": None,
    }


def test_missing_companions_are_named_not_raised(take):
    take.with_name(f"{STEM}.json").unlink()
    take.with_name(f"{STEM}.log").unlink()
    _, data, report = packaged(take)
    payload = report_json(data, report)
    assert payload["inputs"]["missing"] == [f"{STEM}.json", f"{STEM}.log"]
    assert [item["name"] for item in payload["inputs"]["files"]] == [
        f"{STEM}.mcap",
        f"{STEM}.labels.json",
    ]


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
    assert any(0.0 < fraction < 0.9 for fraction in seen)


def test_two_builds_of_one_take_agree(take):
    _, first, report = packaged(take)
    _, second, _ = packaged(take)
    name = f"{STEM}.{report.verdict}/report.json"
    # The zip bytes differ by member mtime; the contents must not.
    assert members(first)[name] == members(second)[name]


def test_the_tool_block_names_the_checker_that_ran(take):
    _, data, report = packaged(take)
    tool = report_json(data, report)["tool"]
    assert tool["checks"] == len(report.results)
    # Null where git cannot answer, which is a submitter working from an archive.
    assert tool["commit"] is None or len(tool["commit"]) == 40
    assert tool["dirty"] in (True, False, None)


def test_a_report_with_no_results_still_packages(take):
    """An empty report has no groups and no worst mark; neither may raise."""
    source = HandTee(synth.StubSource([]))
    empty = run(source, [], None)
    _, data = bundle.build(empty, source.tracks(), take)
    payload = json.loads(members(data)[f"{STEM}.{empty.verdict}/report.json"])
    assert payload["groups"] == []
    assert payload["series"] == {
        "left": {"t_ms": [], "rate_hz": [], "valid": []},
        "right": {"t_ms": [], "rate_hz": [], "valid": []},
    }
