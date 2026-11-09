# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Time-aligns each kept episode in the background, right after its take is finalized.

One worker handles episodes in the order they were kept, so aligning never competes with itself or with the
recording in progress for more than one episode's worth of CPU. An episode that cannot be aligned (for example a
source with no qualified time) is logged with the reason and left as recorded; the raw episode is never modified,
and ``recording/tools/align_episode.py`` aligns it later by hand. Output is published by one rename, so an
alignment cut short by a shutdown or crash leaves no partial result, and :meth:`AutoAligner.recover` picks the
unfinished episodes up again at the next start.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from dataclasses import dataclass
from pathlib import Path

from egocap.recording.aligned_writer import align_kept_episode, episode_qualification
from egocap.recording.episode_layout import (
    FINALIZATION_KEEP,
    MANIFEST_DIR,
    destination_path,
    manifest_path,
    read_manifest,
)
from recording_service import Disposition, EpisodeResult

logger = logging.getLogger("isaaccapture.egocap_web.auto_align")


@dataclass(frozen=True)
class _Job:
    """One kept episode to align: its recording session, its index there, and where it was kept."""

    session_id: str
    episode_index: int
    final: Path

    @property
    def aligned(self) -> Path:
        """Where this episode's aligned copy goes, beside its group."""
        return self.final.parent.with_name(f"{self.final.parent.name}_aligned") / self.final.name


class AutoAligner:
    """Aligns kept episodes one at a time on its own worker task."""

    def __init__(self, output_root: Path, index_source: str) -> None:
        """``output_root`` holds the recording sessions; ``index_source`` is the camera that sets the timeline."""
        self._output_root = output_root
        self._index_source = index_source
        # Unbounded: a job is three small fields, and the number waiting is the number of kept episodes.
        self._queue: asyncio.Queue[_Job] = asyncio.Queue()

    def submit(self, result: EpisodeResult) -> None:
        """Queue a finished take for alignment; anything other than a kept episode is ignored."""
        if result.disposition is Disposition.KEPT and result.final_path is not None:
            self._queue.put_nowait(_Job(result.session_id, result.episode_index, result.final_path))

    def recover(self) -> None:
        """Queue every kept episode, in any session, that has no aligned copy and whose evidence allows one.

        Picks up alignments a restart cut short. An episode recorded without qualified timing evidence (an older
        recording, or a station with no latency set) is skipped, so history is not retried at every start.
        """
        for path in sorted(self._output_root.glob(f"*/{MANIFEST_DIR}/episode_*.json")):
            try:
                manifest = read_manifest(path)
                session = path.parent.parent
                final = (
                    self._output_root / destination_path(manifest.assignment)
                    if manifest.assignment
                    else session / manifest.final_path
                )
                job = _Job(manifest.session_id, manifest.episode_index, final)
                eligible = (
                    manifest.storage_phase == "FINALIZED"
                    and manifest.finalization == FINALIZATION_KEEP
                    and not job.aligned.exists()
                    and all(
                        episode_qualification(
                            final.parent, int(final.name.removeprefix("episode_")), self._index_source
                        ).values()
                    )
                )
            except Exception:  # noqa: BLE001, S112 - an unreadable record is simply not eligible
                continue
            if eligible:
                self._queue.put_nowait(job)

    async def run(self) -> None:
        """Align queued episodes until cancelled; a failed alignment is logged and never stops the worker."""
        while True:
            job = await self._queue.get()
            try:
                # Blocking file and numeric work, kept off the event loop that serves recording and preview.
                await asyncio.to_thread(self._align, job)
            except Exception as exc:  # noqa: BLE001 - one episode's failure must not end the worker
                logger.warning("%s episode %d was not time-aligned: %s", job.session_id, job.episode_index, exc)
            finally:
                self._queue.task_done()

    async def close(self, task: asyncio.Task[None], timeout: float) -> None:
        """Let the queued episodes finish for up to ``timeout`` seconds, then stop the worker ``task``.

        Call after the recorder has closed, so the last kept episode is already queued. A thread still writing
        when the deadline passes cannot be interrupted; it only ever publishes by one rename, and the episode is
        recovered at the next start if it did not finish.
        """
        try:
            await asyncio.wait_for(self._queue.join(), timeout)
        except asyncio.TimeoutError:
            logger.warning("time alignment still has queued episodes at shutdown; they are retried at next start")
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    def _align(self, job: _Job) -> None:
        """Align one kept episode from its stored manifest; raises with the reason when it is refused."""
        if job.aligned.exists():
            return
        # The manifest stays in the recording session; a routed episode sits elsewhere, under its slot.
        manifest = read_manifest(manifest_path(self._output_root / job.session_id, job.episode_index))
        slot = int(job.final.name.removeprefix("episode_"))
        align_kept_episode(job.final.parent, slot, self._index_source, manifest)
        logger.info("%s episode %d was time-aligned", job.session_id, job.episode_index)
