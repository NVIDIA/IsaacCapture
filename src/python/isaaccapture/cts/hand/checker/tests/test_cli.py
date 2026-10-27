# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest
import synth
from conftest import DEFAULT_FIXTURES, index_entries

from hand_cts import cli
from hand_cts.cli import EXIT_STATUS, main
from hand_cts.labels import SIDECAR_SUFFIX

ENTRIES = {Path(e["filename"]).stem: e for e in index_entries()}

# Sidecar contents that ``StepTimeline.load`` cannot turn into step windows.
MALFORMED_LABELS = {
    "not_json": b"not json at all",
    "not_utf8": b"\xff\xfe\x00garbage",
    "a_list": b"[]",
    "no_steps": b"{}",
    "steps_not_a_list": b'{"steps": 5}',
    "step_without_start_ns": b'{"steps": [{"index": 0, "label": "x", "end_ns": 2}]}',
    "time_not_an_integer": (
        b'{"steps": [{"index": 0, "label": "x", "start_ns": "soon", "end_ns": 2}]}'
    ),
}

# Payloads ``decode_record`` cannot unpack, made from a valid encoded record.
MALFORMED_PAYLOADS = {
    "empty": lambda valid: b"",
    "short": lambda valid: valid[:3],
    "joints_past_the_end": lambda valid: valid[: len(valid) // 2],
}


def _cannot_run(argv: list[str], capsys) -> str:
    """Runs the CLI expecting exit status 4, the check could not run; returns its one stderr line."""
    with pytest.raises(SystemExit) as exited:
        main(argv)
    assert exited.value.code == 4
    err = capsys.readouterr().err
    assert len(err.splitlines()) == 1, err
    return err


def _recording(tmp_path: Path) -> Path:
    return synth.write_recording(tmp_path / "take-hand.mcap", synth.both_hands(60))


def test_list_checks_needs_no_recording(capsys):
    assert main(["--list-checks"]) == 0
    assert "pinch.finger_identity" in capsys.readouterr().out


def test_help_exits_zero(capsys):
    with pytest.raises(SystemExit) as exited:
        main(["--help"])
    assert exited.value.code == 0
    assert "--list-checks" in capsys.readouterr().out


def test_no_recording_cannot_run(capsys):
    assert "a recording is required" in _cannot_run([], capsys)


def test_a_bad_flag_cannot_run(capsys):
    assert "--no-such-flag" in _cannot_run(["--no-such-flag", "take.mcap"], capsys)


def test_an_unknown_check_cannot_run(tmp_path, capsys):
    argv = [str(_recording(tmp_path)), "--check", "no.such_check"]
    assert "unknown checks: ['no.such_check']" in _cannot_run(argv, capsys)


def test_a_missing_recording_cannot_run(tmp_path, capsys):
    err = _cannot_run([str(tmp_path / "absent.mcap")], capsys)
    assert "No such file or directory" in err


def test_a_file_that_is_not_mcap_cannot_run(tmp_path, capsys):
    garbage = tmp_path / "garbage.mcap"
    garbage.write_bytes(bytes(range(256)) * 16)
    assert "not a readable MCAP" in _cannot_run([str(garbage)], capsys)


@pytest.mark.parametrize(
    "keep", [lambda size: size // 2, lambda size: size - 10], ids=["half", "footer"]
)
def test_a_truncated_recording_cannot_run(keep, tmp_path, capsys):
    whole = _recording(tmp_path)
    assert main([str(whole)]) in EXIT_STATUS.values()
    data = whole.read_bytes()
    truncated = tmp_path / "truncated-hand.mcap"
    truncated.write_bytes(data[: keep(len(data))])
    capsys.readouterr()
    assert "ends part way through a record" in _cannot_run([str(truncated)], capsys)


@pytest.mark.parametrize("cut", MALFORMED_PAYLOADS.values(), ids=MALFORMED_PAYLOADS)
def test_an_undecodable_payload_cannot_run(cut, tmp_path, monkeypatch, capsys):
    encode = synth.encode
    monkeypatch.setattr(synth, "encode", lambda frame: cut(encode(frame)))
    recording = synth.write_recording(tmp_path / "take-hand.mcap", synth.both_hands(2))
    assert "is not a decodable" in _cannot_run([str(recording)], capsys)


def test_a_missing_labels_file_cannot_run(tmp_path, capsys):
    argv = [str(_recording(tmp_path)), "--labels", str(tmp_path / "absent.labels.json")]
    assert "No such file or directory" in _cannot_run(argv, capsys)


def test_a_labels_directory_cannot_run(tmp_path, capsys):
    folder = tmp_path / "folder.labels.json"
    folder.mkdir()
    argv = [str(_recording(tmp_path)), "--labels", str(folder)]
    assert "Is a directory" in _cannot_run(argv, capsys)


@pytest.mark.parametrize("content", MALFORMED_LABELS.values(), ids=MALFORMED_LABELS)
def test_malformed_labels_cannot_run(content, tmp_path, capsys):
    labels = tmp_path / "given.labels.json"
    labels.write_bytes(content)
    argv = [str(_recording(tmp_path)), "--labels", str(labels)]
    assert "is not a label sidecar" in _cannot_run(argv, capsys)


@pytest.mark.parametrize("content", MALFORMED_LABELS.values(), ids=MALFORMED_LABELS)
def test_a_malformed_sidecar_beside_the_recording_cannot_run(content, tmp_path, capsys):
    recording = _recording(tmp_path)
    recording.with_name(recording.stem + SIDECAR_SUFFIX).write_bytes(content)
    assert "is not a label sidecar" in _cannot_run([str(recording)], capsys)


def test_a_missing_sidecar_is_still_supported(tmp_path, capsys):
    assert main([str(_recording(tmp_path))]) in EXIT_STATUS.values()
    assert "no motion-step labels" in capsys.readouterr().out


def test_labels_with_no_steps_are_well_formed(tmp_path):
    labels = tmp_path / "given.labels.json"
    labels.write_text('{"steps": []}')
    argv = [str(_recording(tmp_path)), "--labels", str(labels)]
    assert main(argv) in EXIT_STATUS.values()


@pytest.mark.parametrize("fault", [struct.error, OSError, KeyError])
def test_a_fault_inside_a_check_is_not_swallowed(fault, tmp_path, monkeypatch):
    def broken(source, checks, timeline):
        for _ in source:
            raise fault("a bug in a check")

    monkeypatch.setattr(cli, "run", broken)
    with pytest.raises(fault):
        main([str(_recording(tmp_path))])


@pytest.mark.skipif(not ENTRIES, reason=f"run {DEFAULT_FIXTURES}/generate.sh")
@pytest.mark.parametrize(
    "stem, status",
    [("golden_full_script", 0), ("defect_centimetres", 1), ("retake_shallow_fist", 2)],
)
def test_exit_status_follows_the_verdict(stem, status, fixture_dir, capsys):
    path = fixture_dir / ENTRIES[stem]["filename"]
    assert main([str(path), "--json"]) == status
    assert json.loads(capsys.readouterr().out)["verdict"] in ("pass", "fail", "retake")


@pytest.mark.skipif(not ENTRIES, reason=f"run {DEFAULT_FIXTURES}/generate.sh")
def test_without_labels_the_verdict_is_still_reached(fixture_dir, tmp_path, capsys):
    source = fixture_dir / ENTRIES["golden_full_script"]["filename"]
    copy = tmp_path / "unlabelled.mcap"
    copy.write_bytes(source.read_bytes())
    assert main([str(copy)]) == 0
    assert "no motion-step labels" in capsys.readouterr().out
