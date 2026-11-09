# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Background alignment of kept episodes: which results are queued, what is aligned, and worker resilience."""

import asyncio
import logging
import threading
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
def calls(monkeypatch):
    recorded = []
    monkeypatch.setattr(auto_align, "manifest_path", lambda root, index: Path(root) / f"manifest_{index}.json")
    monkeypatch.setattr(auto_align, "read_manifest", lambda path: f"manifest:{path}")
    monkeypatch.setattr(auto_align, "align_kept_episode", lambda *args: recorded.append(args))
    return recorded


async def _drain(aligner):
    task = asyncio.create_task(aligner.run())
    for _ in range(200):
        if aligner._queue.empty():
            break
        await asyncio.sleep(0.01)
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


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


async def test_a_kept_episode_is_aligned_from_its_manifest_and_slot(calls):
    aligner = AutoAligner(Path("/out"), "head")
    aligner.submit(_result())
    await _drain(aligner)
    assert calls == [(Path("/routed/group"), 7, "head", "manifest:/out/session_a/manifest_3.json")]


async def test_a_failed_alignment_is_logged_and_the_next_episode_is_still_aligned(calls, monkeypatch, caplog):
    def refuse_first(*args):
        if not calls:
            calls.append(args)
            raise ValueError("no qualified latency")
        calls.append(args)

    monkeypatch.setattr(auto_align, "align_kept_episode", refuse_first)
    aligner = AutoAligner(Path("/out"), "head")
    aligner.submit(_result(index=1))
    aligner.submit(_result(index=2))
    with caplog.at_level(logging.INFO, logger=auto_align.logger.name):
        await _drain(aligner)
    assert len(calls) == 2
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert (
        len(warnings) == 1
        and "session_a episode 1" in warnings[0].getMessage()
        and "no qualified latency" in warnings[0].getMessage()
    )
    assert any("session_a episode 2 was time-aligned" in r.getMessage() for r in caplog.records)


async def test_cancelling_an_idle_worker_ends_it_cleanly():
    task = asyncio.create_task(AutoAligner(Path("/out"), "head").run())
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
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


def _queued(aligner):
    jobs = []
    while not aligner._queue.empty():
        jobs.append(aligner._queue.get_nowait())
    return jobs


def test_recover_queues_only_kept_episodes_without_an_aligned_copy(tmp_path):
    _write_session_manifest(tmp_path, "s1", 0)
    _write_session_manifest(tmp_path, "s1", 1, aligned=True)
    _write_session_manifest(tmp_path, "s1", 2, finalization=FINALIZATION_RETAIN)
    _write_session_manifest(tmp_path, "s1", 3, storage_phase="WRITING")
    aligner = AutoAligner(tmp_path, "head")
    aligner.recover()
    assert _queued(aligner) == [auto_align._Job("s1", 0, tmp_path / "s1" / episode_dir_names(0)[1])]


def test_recover_queues_older_sessions_work_even_when_the_newest_session_is_empty(tmp_path):
    _write_session_manifest(tmp_path, "s1", 0)
    _write_session_manifest(tmp_path, "s2", 0)
    (tmp_path / "s3" / auto_align.MANIFEST_DIR).mkdir(parents=True)
    aligner = AutoAligner(tmp_path, "head")
    aligner.recover()
    assert [job.session_id for job in _queued(aligner)] == ["s1", "s2"]


def test_recover_skips_an_episode_whose_timing_is_not_qualified(tmp_path):
    _write_session_manifest(tmp_path, "s1", 0, qualified=False)
    _write_session_manifest(tmp_path, "s1", 1)
    aligner = AutoAligner(tmp_path, "head")
    aligner.recover()
    assert [job.episode_index for job in _queued(aligner)] == [1]


def test_recover_skips_an_episode_with_no_readable_evidence(tmp_path):
    _write_session_manifest(tmp_path, "s1", 0, qualified=None)
    aligner = AutoAligner(tmp_path, "head")
    aligner.recover()
    assert aligner._queue.empty()


def test_recover_skips_an_unreadable_manifest_and_keeps_the_rest(tmp_path):
    _write_session_manifest(tmp_path, "s1", 1)
    (tmp_path / "s1" / auto_align.MANIFEST_DIR / "episode_0.json").write_text("{not json")
    aligner = AutoAligner(tmp_path, "head")
    aligner.recover()
    assert [job.episode_index for job in _queued(aligner)] == [1]


def test_recover_uses_the_routed_destination_for_a_routed_episode(tmp_path):
    _write_session_manifest(tmp_path, "s1", 0, studio="Lab", assignment={"group": "Lab/20260101", "slot": 4})
    aligner = AutoAligner(tmp_path, "head")
    aligner.recover()
    assert _queued(aligner) == [auto_align._Job("s1", 0, tmp_path / "Lab/20260101/episode_4")]


def test_recover_skips_a_routed_episode_that_is_already_aligned(tmp_path):
    _write_session_manifest(
        tmp_path, "s1", 0, aligned=True, studio="Lab", assignment={"group": "Lab/20260101", "slot": 4}
    )
    aligner = AutoAligner(tmp_path, "head")
    aligner.recover()
    assert _queued(aligner) == []


@pytest.mark.parametrize("create_root", [False, True])
def test_recover_with_no_sessions_queues_nothing(tmp_path, create_root):
    root = tmp_path / "out"
    if create_root:
        root.mkdir()
    aligner = AutoAligner(root, "head")
    aligner.recover()
    assert aligner._queue.empty()


def test_recover_is_not_vetoed_by_a_declared_source_the_episode_did_not_record(tmp_path):
    _write_session_manifest(tmp_path, "s1", 0)
    (tmp_path / "s1" / "config.yaml").write_text(
        yaml.safe_dump({"video_sources": [{"name": "head"}, {"name": "optional_camera"}]}), encoding="utf-8"
    )
    aligner = AutoAligner(tmp_path, "head")
    aligner.recover()
    assert [job.episode_index for job in _queued(aligner)] == [0]


async def test_more_recovered_episodes_than_the_queue_holds_are_all_aligned_in_order(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(auto_align, "align_kept_episode", lambda *args: calls.append(args))
    monkeypatch.setattr(auto_align, "_QUEUE_LIMIT", 2)
    for index in range(5):
        _write_session_manifest(tmp_path, "s1", index)
    aligner = AutoAligner(tmp_path, "head")
    aligner.recover()
    assert aligner._queue.qsize() == 2 and len(aligner._backlog) == 3
    task = asyncio.create_task(aligner.run())
    await aligner.close(task, 5)
    assert [args[1] for args in calls] == [0, 1, 2, 3, 4]
    assert not aligner._backlog


async def test_a_live_submit_while_the_queue_is_full_is_kept_and_aligned_last(calls, monkeypatch):
    monkeypatch.setattr(auto_align, "_QUEUE_LIMIT", 1)
    aligner = AutoAligner(Path("/out"), "head")
    for index in (1, 2, 3):
        aligner.submit(_result(index=index, final_path=Path("/g") / f"episode_{index}"))
    task = asyncio.create_task(aligner.run())
    await aligner.close(task, 5)
    assert [args[1] for args in calls] == [1, 2, 3]


async def test_close_waits_for_the_backlog_too(calls, monkeypatch):
    monkeypatch.setattr(auto_align, "_QUEUE_LIMIT", 1)
    aligner = AutoAligner(Path("/out"), "head")
    for index in range(4):
        aligner.submit(_result(index=index))
    assert len(aligner._backlog) == 3
    task = asyncio.create_task(aligner.run())
    await aligner.close(task, 5)
    assert len(calls) == 4 and not aligner._backlog and task.cancelled()


def test_the_logger_is_in_the_isaaccapture_tree():
    assert auto_align.logger.name == "isaaccapture.egocap_web.auto_align"


async def test_an_episode_that_already_has_an_aligned_copy_is_not_aligned_again(calls, tmp_path):
    aligned = tmp_path / "group_aligned" / "episode_7"
    aligned.mkdir(parents=True)
    aligner = AutoAligner(tmp_path, "head")
    aligner.submit(_result(final_path=tmp_path / "group" / "episode_7"))
    await _drain(aligner)
    assert calls == []


async def test_a_recovered_episode_is_aligned_for_real_and_its_aligned_copy_appears(tmp_path):
    session = tmp_path / "out" / "s1"
    build_raw(session)
    write_manifest(session, make_manifest(session, session_id="s1"))
    aligner = AutoAligner(tmp_path / "out", INDEX)
    aligner.recover()
    assert aligner._queue.qsize() == 1
    task = asyncio.create_task(aligner.run())
    await aligner.close(task, 60)
    assert (tmp_path / "out" / "s1_aligned" / "episode_0").is_dir()
    assert task.cancelled()


async def test_close_waits_for_queued_episodes_before_stopping_the_worker(calls):
    aligner = AutoAligner(Path("/out"), "head")
    aligner.submit(_result(index=1))
    aligner.submit(_result(index=2))
    task = asyncio.create_task(aligner.run())
    await aligner.close(task, 5)
    assert len(calls) == 2 and task.cancelled()


async def test_close_with_a_blocked_job_warns_after_the_timeout_and_cancels_the_worker(calls, monkeypatch, caplog):
    release = threading.Event()
    monkeypatch.setattr(auto_align, "align_kept_episode", lambda *args: release.wait(5))
    aligner = AutoAligner(Path("/out"), "head")
    aligner.submit(_result())
    task = asyncio.create_task(aligner.run())
    await asyncio.sleep(0.05)
    try:
        with caplog.at_level(logging.WARNING, logger=auto_align.logger.name):
            await aligner.close(task, 0.1)
    finally:
        release.set()
    assert task.cancelled()
    assert "still has queued episodes" in caplog.text
