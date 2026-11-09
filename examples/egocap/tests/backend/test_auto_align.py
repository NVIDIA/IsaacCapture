# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Background alignment of kept episodes: which results are queued, what is aligned, and worker resilience."""

import asyncio
import logging
from pathlib import Path

import pytest

from egocap_web import auto_align
from egocap_web.auto_align import AutoAligner
from recording_service import Disposition, EpisodeResult


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
        and "ep1" in warnings[0].getMessage()
        and "no qualified latency" in warnings[0].getMessage()
    )
    assert any("ep2 was time-aligned" in r.getMessage() for r in caplog.records)


async def test_cancelling_an_idle_worker_ends_it_cleanly():
    task = asyncio.create_task(AutoAligner(Path("/out"), "head").run())
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert task.cancelled()
