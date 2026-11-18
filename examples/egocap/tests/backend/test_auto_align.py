# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Background alignment of kept episodes: what is queued, scanned, aligned in a child process, and refused."""

import asyncio
import json
import logging
import os
import time
from pathlib import Path

import pytest
import yaml
import zarr

from egocap.recording.episode_layout import (
    FINALIZATION_KEEP,
    FINALIZATION_RETAIN,
    EpisodeManifest,
    episode_dir_names,
    write_manifest,
)
from egocap_web import auto_align
from egocap_web.auto_align import AutoAligner
from recording_service import Disposition, EpisodeResult
from tests.recording.unit.test_align_kept_episode import INDEX, build_raw, make_manifest

STUB = """
import json, os, sys, time
from pathlib import Path

argv = sys.argv[1:]
with open(os.environ["STUB_LOG"], "a") as log:
    log.write(json.dumps(argv) + "\\n")
group = Path(argv[0])
slot = argv[argv.index("--episode") + 1]
if "hang" in group.name:
    Path(os.environ["STUB_LOG"] + ".pid").write_text(str(os.getpid()))
    time.sleep(60)
if "bad" in group.name:
    print("no qualified latency", file=sys.stderr)
    sys.exit(1)
if "crash" in group.name:
    sys.exit(3)
(group.with_name(group.name + "_aligned") / f"episode_{slot}").mkdir(parents=True)
"""


def _result(index=3, disposition=Disposition.KEPT, final_path="default", session_id="session_a"):
    if final_path == "default":
        final_path = Path("/routed/group") / "episode_000007"
    return EpisodeResult(
        session_id=session_id,
        episode_id=f"ep{index}",
        episode_index=index,
        disposition=disposition,
        final_path=final_path,
        pending_path=None,
        counts=(),
        duration_seconds=1.0,
        quality=None,
    )


@pytest.fixture
def stub_tool(tmp_path, monkeypatch):
    tool = tmp_path / "stub_tool" / "align_stub.py"
    tool.parent.mkdir()
    tool.write_text(STUB, encoding="utf-8")
    log = tmp_path / "stub_tool" / "calls.log"
    monkeypatch.setattr(auto_align, "_ALIGN_TOOL", tool)
    monkeypatch.setenv("STUB_LOG", str(log))
    return log


def _stub_calls(log):
    return [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []


@pytest.mark.parametrize(
    "result",
    [
        _result(disposition=Disposition.DISCARDED),
        _result(final_path=None),
    ],
)
def test_results_that_are_not_a_kept_episode_with_a_path_are_ignored(result):
    aligner = AutoAligner(Path("/out"), "head")
    aligner.submit(result)
    assert aligner._queue.empty()


def test_kept_results_are_queued_before_any_session_has_reported_its_sources():
    aligner = AutoAligner(Path("/out"), "head")
    aligner.submit(_result())
    assert aligner._queue.qsize() == 1


def test_kept_results_are_queued_when_the_index_source_and_every_recorded_camera_are_qualified():
    aligner = AutoAligner(Path("/out"), "head")
    aligner.note_video_sources(frozenset({"head", "wrist"}), frozenset({"head", "wrist", "other"}))
    aligner.submit(_result())
    assert aligner._queue.qsize() == 1


def test_kept_results_are_not_queued_when_the_index_source_is_recorded_without_a_sensor_capture_time():
    aligner = AutoAligner(Path("/out"), "head")
    aligner.note_video_sources(frozenset({"head"}), frozenset())
    aligner.submit(_result())
    assert aligner._queue.empty()


def test_kept_results_are_not_queued_when_the_index_source_is_not_recorded():
    aligner = AutoAligner(Path("/out"), "head")
    aligner.note_video_sources(frozenset({"wrist"}), frozenset({"wrist"}))
    aligner.submit(_result())
    assert aligner._queue.empty()


def test_kept_results_are_not_queued_when_another_recorded_camera_is_not_qualified():
    aligner = AutoAligner(Path("/out"), "head")
    aligner.note_video_sources(frozenset({"head", "wrist"}), frozenset({"head"}))
    aligner.submit(_result())
    assert aligner._queue.empty()


def test_the_not_alignable_warning_is_logged_once_per_transition(caplog):
    aligner = AutoAligner(Path("/out"), "head")
    with caplog.at_level(logging.WARNING, logger="isaaccapture"):
        aligner.note_video_sources(frozenset({"wrist"}), frozenset({"wrist"}))
        aligner.note_video_sources(frozenset(), frozenset())
        assert len(caplog.records) == 1
        assert "'head'" in caplog.text and "sensor capture time" in caplog.text
        aligner.note_video_sources(frozenset({"head"}), frozenset({"head"}))
        aligner.note_video_sources(frozenset({"wrist"}), frozenset())
    assert len(caplog.records) == 2


def test_queueing_resumes_when_a_later_session_is_alignable():
    aligner = AutoAligner(Path("/out"), "head")
    aligner.note_video_sources(frozenset({"head"}), frozenset())
    aligner.note_video_sources(frozenset({"head"}), frozenset({"head"}))
    aligner.submit(_result())
    assert aligner._queue.qsize() == 1


async def test_cancelling_an_idle_worker_ends_it_cleanly(tmp_path):
    task = asyncio.create_task(AutoAligner(tmp_path / "out", "head").run())
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        _ = await task
    assert task.cancelled()


def _write_evidence(group_dir, slot, *, qualified):
    group_dir.mkdir(parents=True, exist_ok=True)
    (group_dir / "config.yaml").write_text(yaml.safe_dump({"video_sources": [{"name": "head"}]}), encoding="utf-8")
    episode = zarr.open_group(str(group_dir / f"episode_{slot}"), mode="w")
    episode.attrs["timing"] = {"event_time": {"head": {"qualified": qualified}}}


def _write_session_manifest(output_root, session, index, *, aligned=False, qualified=True, **overrides):
    pending, final = episode_dir_names(index)
    fields = {
        "session_id": session,
        "episode_id": f"e{index}",
        "episode_index": index,
        "config_digest": "d",
        "task": "t",
        "operator": "o",
        "sources": [],
        "pending_path": pending,
        "final_path": final,
        "storage_phase": "FINALIZED",
        "finalization": FINALIZATION_KEEP,
    }
    fields.update(overrides)
    (output_root / session).mkdir(parents=True, exist_ok=True)
    write_manifest(output_root / session, EpisodeManifest(**fields))
    assignment = fields.get("assignment")
    group = output_root / assignment["group"] if assignment else output_root / session
    slot = assignment["slot"] if assignment else index
    if qualified is not None:
        _write_evidence(group, slot, qualified=qualified)
    if aligned:
        (group.with_name(f"{group.name}_aligned") / f"episode_{slot}").mkdir(parents=True)
    return final


def test_scan_finds_only_kept_episodes_without_an_aligned_copy(tmp_path):
    _write_session_manifest(tmp_path, "s1", 0)
    _write_session_manifest(tmp_path, "s1", 1, aligned=True)
    _write_session_manifest(tmp_path, "s1", 2, finalization=FINALIZATION_RETAIN)
    _write_session_manifest(tmp_path, "s1", 3, storage_phase="WRITING")
    aligner = AutoAligner(tmp_path, "head")
    jobs = aligner.scan()
    assert jobs == [auto_align._Job("s1", 0, tmp_path / "s1" / episode_dir_names(0)[1])]


def test_scan_finds_older_sessions_work_even_when_the_newest_session_is_empty(tmp_path):
    _write_session_manifest(tmp_path, "s1", 0)
    _write_session_manifest(tmp_path, "s2", 0)
    (tmp_path / "s3" / auto_align.MANIFEST_DIR).mkdir(parents=True)
    aligner = AutoAligner(tmp_path, "head")
    jobs = aligner.scan()
    assert [job.session_id for job in jobs] == ["s1", "s2"]


def test_scan_skips_an_episode_whose_timing_is_not_qualified(tmp_path):
    _write_session_manifest(tmp_path, "s1", 0, qualified=False)
    _write_session_manifest(tmp_path, "s1", 1)
    aligner = AutoAligner(tmp_path, "head")
    jobs = aligner.scan()
    assert [job.episode_index for job in jobs] == [1]


def test_scan_skips_an_episode_with_no_readable_evidence(tmp_path):
    _write_session_manifest(tmp_path, "s1", 0, qualified=None)
    aligner = AutoAligner(tmp_path, "head")
    assert aligner.scan() == []


def test_scan_skips_an_unreadable_manifest_and_keeps_the_rest(tmp_path):
    _write_session_manifest(tmp_path, "s1", 1)
    (tmp_path / "s1" / auto_align.MANIFEST_DIR / "episode_0.json").write_text("{not json")
    aligner = AutoAligner(tmp_path, "head")
    jobs = aligner.scan()
    assert [job.episode_index for job in jobs] == [1]


def test_scan_uses_the_routed_destination_for_a_routed_episode(tmp_path):
    _write_session_manifest(tmp_path, "s1", 0, studio="Lab", assignment={"group": "Lab/20260101", "slot": 4})
    aligner = AutoAligner(tmp_path, "head")
    jobs = aligner.scan()
    assert jobs == [auto_align._Job("s1", 0, tmp_path / "Lab/20260101/episode_4")]


def test_scan_skips_a_routed_episode_that_is_already_aligned(tmp_path):
    _write_session_manifest(
        tmp_path, "s1", 0, aligned=True, studio="Lab", assignment={"group": "Lab/20260101", "slot": 4}
    )
    aligner = AutoAligner(tmp_path, "head")
    jobs = aligner.scan()
    assert jobs == []


@pytest.mark.parametrize("create_root", [False, True])
def test_scan_with_no_sessions_finds_nothing(tmp_path, create_root):
    root = tmp_path / "out"
    if create_root:
        root.mkdir()
    aligner = AutoAligner(root, "head")
    assert aligner.scan() == []


def test_scan_is_not_vetoed_by_a_declared_source_the_episode_did_not_record(tmp_path):
    _write_session_manifest(tmp_path, "s1", 0)
    (tmp_path / "s1" / "config.yaml").write_text(
        yaml.safe_dump({"video_sources": [{"name": "head"}, {"name": "optional_camera"}]}), encoding="utf-8"
    )
    aligner = AutoAligner(tmp_path, "head")
    jobs = aligner.scan()
    assert [job.episode_index for job in jobs] == [0]


def test_the_logger_is_in_the_isaaccapture_tree():
    assert auto_align.logger.name == "isaaccapture.egocap_web.auto_align"


async def test_a_kept_episode_is_aligned_by_a_child_process_with_its_slot_and_manifest(tmp_path, stub_tool):
    aligner = AutoAligner(tmp_path / "out", "head")
    aligner.submit(_result(index=3, final_path=tmp_path / "out" / "Lab" / "g" / "episode_000007"))
    task = asyncio.create_task(aligner.run())
    await aligner.close(task, 30)
    (argv,) = _stub_calls(stub_tool)
    assert argv == [
        str(tmp_path / "out" / "Lab" / "g"),
        "--episode",
        "7",
        "--index-source",
        "head",
        "--manifest-session",
        str(tmp_path / "out" / "session_a"),
        "--manifest-episode",
        "3",
    ]
    assert (tmp_path / "out" / "Lab" / "g_aligned" / "episode_7").is_dir()


async def test_an_episode_that_already_has_an_aligned_copy_is_not_aligned_again(tmp_path, stub_tool):
    (tmp_path / "group_aligned" / "episode_7").mkdir(parents=True)
    aligner = AutoAligner(tmp_path, "head")
    aligner.submit(_result(final_path=tmp_path / "group" / "episode_7"))
    task = asyncio.create_task(aligner.run())
    await aligner.close(task, 30)
    assert _stub_calls(stub_tool) == []


async def test_many_submits_are_all_aligned_in_order_and_close_drains_them(tmp_path, stub_tool):
    aligner = AutoAligner(tmp_path, "head")
    for index in range(6):
        aligner.submit(_result(index=index, final_path=tmp_path / f"g{index}" / f"episode_{index}"))
    task = asyncio.create_task(aligner.run())
    await aligner.close(task, 60)
    assert [call[0] for call in _stub_calls(stub_tool)] == [str(tmp_path / f"g{i}") for i in range(6)]
    assert aligner._queue.empty() and task.cancelled()


async def test_a_refusal_writes_a_marker_and_the_next_episode_is_still_aligned(tmp_path, stub_tool, caplog):
    aligner = AutoAligner(tmp_path, "head")
    aligner.submit(_result(index=1, final_path=tmp_path / "bad" / "episode_1"))
    aligner.submit(_result(index=2, final_path=tmp_path / "good" / "episode_2"))
    task = asyncio.create_task(aligner.run())
    with caplog.at_level(logging.INFO, logger=auto_align.logger.name):
        await aligner.close(task, 30)
    assert (tmp_path / "bad_aligned" / ".episode_1.refused").read_text() == "no qualified latency\n"
    assert (tmp_path / "good_aligned" / "episode_2").is_dir()
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1 and "session_a episode 1" in warnings[0] and "no qualified latency" in warnings[0]
    assert any("session_a episode 2 was time-aligned" in r.getMessage() for r in caplog.records)


async def test_a_crash_is_logged_without_a_marker_and_the_next_episode_is_still_aligned(
    tmp_path, stub_tool, caplog
):
    aligner = AutoAligner(tmp_path, "head")
    aligner.submit(_result(index=1, final_path=tmp_path / "crash" / "episode_1"))
    aligner.submit(_result(index=2, final_path=tmp_path / "good" / "episode_2"))
    task = asyncio.create_task(aligner.run())
    with caplog.at_level(logging.WARNING, logger=auto_align.logger.name):
        await aligner.close(task, 30)
    assert not (tmp_path / "crash_aligned").exists()
    assert "exited with status 3" in caplog.text
    assert (tmp_path / "good_aligned" / "episode_2").is_dir()


async def test_a_refused_episode_is_not_queued_again_until_its_marker_is_deleted(tmp_path, stub_tool):
    _write_session_manifest(tmp_path, "bad", 0)
    aligner = AutoAligner(tmp_path, "head")
    (job,) = aligner.scan()
    aligner.submit(_result(index=0, session_id="bad", final_path=job.final))
    task = asyncio.create_task(aligner.run())
    await aligner.close(task, 30)
    assert job.refused_marker.is_file() and aligner.scan() == []
    job.refused_marker.unlink()
    assert aligner.scan() == [job]


async def test_the_worker_aligns_what_the_startup_scan_finds_without_a_submit(tmp_path, stub_tool):
    _write_session_manifest(tmp_path, "s1", 0)
    _write_session_manifest(tmp_path, "s1", 1, aligned=True)
    aligner = AutoAligner(tmp_path, "head")
    task = asyncio.create_task(aligner.run())
    for _ in range(300):
        if (tmp_path / "s1_aligned" / "episode_0").is_dir():
            break
        await asyncio.sleep(0.05)
    await aligner.close(task, 30)
    assert len(_stub_calls(stub_tool)) == 1


async def test_a_failing_scan_is_logged_and_the_worker_still_aligns_submitted_episodes(
    tmp_path, stub_tool, monkeypatch, caplog
):
    def broken_scan():
        raise OSError("disk unavailable")

    aligner = AutoAligner(tmp_path, "head")
    monkeypatch.setattr(aligner, "scan", broken_scan)
    aligner.submit(_result(index=2, final_path=tmp_path / "good" / "episode_2"))
    task = asyncio.create_task(aligner.run())
    with caplog.at_level(logging.WARNING, logger=auto_align.logger.name):
        await aligner.close(task, 30)
    assert "recovery failed: disk unavailable" in caplog.text
    assert (tmp_path / "good_aligned" / "episode_2").is_dir()


async def test_close_kills_an_alignment_still_running_at_the_deadline_and_returns(tmp_path, stub_tool, caplog):
    aligner = AutoAligner(tmp_path, "head")
    aligner.submit(_result(index=1, final_path=tmp_path / "hang" / "episode_1"))
    task = asyncio.create_task(aligner.run())
    pid_file = Path(str(stub_tool) + ".pid")
    for _ in range(300):
        if pid_file.exists() and pid_file.read_text():
            break
        await asyncio.sleep(0.05)
    pid = int(pid_file.read_text())
    with caplog.at_level(logging.WARNING, logger=auto_align.logger.name):
        await asyncio.wait_for(aligner.close(task, 0.2), 10)
    assert "still has queued episodes" in caplog.text
    assert task.cancelled() and aligner._process is None
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)
    assert not (tmp_path / "hang_aligned").exists()


async def test_a_kept_episode_is_aligned_for_real_through_the_command_line_tool(tmp_path):
    session = tmp_path / "out" / "s1"
    build_raw(session)
    write_manifest(session, make_manifest(session, session_id="s1"))
    aligner = AutoAligner(tmp_path / "out", INDEX)
    assert [job.episode_index for job in aligner.scan()] == [0]
    task = asyncio.create_task(aligner.run())
    for _ in range(1200):
        if (tmp_path / "out" / "s1_aligned" / "episode_0").is_dir():
            break
        await asyncio.sleep(0.1)
    await aligner.close(task, 120)
    assert (tmp_path / "out" / "s1_aligned" / "episode_0").is_dir()
    assert aligner.scan() == []


async def test_an_episode_the_tool_refuses_is_marked_with_its_reason_and_left_unaligned(tmp_path):
    session = tmp_path / "out" / "s1"
    build_raw(session)
    write_manifest(session, make_manifest(session, session_id="s1"))
    aligner = AutoAligner(tmp_path / "out", "not_a_camera")
    aligner.submit(_result(index=0, session_id="s1", final_path=session / "episode_0"))
    task = asyncio.create_task(aligner.run())
    await aligner.close(task, 120)
    marker = tmp_path / "out" / "s1_aligned" / ".episode_0.refused"
    assert marker.read_text().strip() and not (tmp_path / "out" / "s1_aligned" / "episode_0").exists()


async def test_an_unreadable_manifest_leaves_no_marker_and_a_later_scan_queues_the_episode_again(tmp_path):
    session = tmp_path / "out" / "s1"
    build_raw(session)
    write_manifest(session, make_manifest(session, session_id="s1"))
    aligner = AutoAligner(tmp_path / "out", INDEX)
    (job,) = aligner.scan()
    manifest = session / auto_align.MANIFEST_DIR / "episode_0.json"
    original = manifest.read_text()
    aligner.submit(_result(index=0, session_id="s1", final_path=job.final))
    manifest.write_text("{not json")
    task = asyncio.create_task(aligner.run())
    await aligner.close(task, 120)
    assert not job.refused_marker.exists() and not job.aligned.exists()
    manifest.write_text(original)
    assert aligner.scan() == [job]


async def test_a_missing_manifest_exits_2_without_a_marker(tmp_path):
    session = tmp_path / "out" / "s1"
    build_raw(session)
    aligner = AutoAligner(tmp_path / "out", INDEX)
    aligner.submit(_result(index=0, session_id="s1", final_path=session / "episode_0"))
    task = asyncio.create_task(aligner.run())
    await aligner.close(task, 120)
    assert not (tmp_path / "out" / "s1_aligned").exists()


def _old(path, days=2):
    stamp = time.time() - days * 86400
    os.utime(path, (stamp, stamp))


def test_scan_removes_old_staging_directories_beside_a_group_and_keeps_fresh_ones(tmp_path):
    _write_session_manifest(tmp_path, "s1", 0)
    stale = tmp_path / ".s1_aligned.staging-abc"
    fresh = tmp_path / ".s1_aligned.staging-def"
    unrelated = tmp_path / ".other_aligned.staging-xyz"
    for directory in (stale, fresh, unrelated):
        (directory / "episode_0").mkdir(parents=True)
    _old(stale)
    _old(unrelated)
    AutoAligner(tmp_path, "head").scan()
    assert not stale.exists() and fresh.is_dir() and unrelated.is_dir()
