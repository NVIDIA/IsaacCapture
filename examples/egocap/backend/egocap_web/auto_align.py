# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Time-aligns each kept episode in the background, right after its take is finalized.

One worker handles episodes in the order they were kept, so aligning never competes with itself or with the
recording in progress for more than one episode's worth of CPU. An episode that cannot be aligned (for example a
source with no qualified time or latency) is logged with the reason and left as recorded; the raw episode is never
modified, and ``recording/tools/align_episode.py`` aligns it later by hand.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path

from egocap.recording.aligned_writer import align_kept_episode
from egocap.recording.episode_layout import (
    FINALIZATION_KEEP,
    MANIFEST_DIR,
    destination_path,
    manifest_path,
    read_manifest,
)
from recording_service import Disposition, EpisodeResult

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _Job:
    """One kept episode to align: its recording session, its index there, and where it was kept."""

    session_id: str
    episode_index: int
    final: Path


class AutoAligner:
    """Aligns kept episodes one at a time on its own worker task."""

    def __init__(self, output_root: Path, index_source: str) -> None:
        """``output_root`` holds the recording sessions; ``index_source`` is the camera that sets the timeline."""
        self._output_root = output_root
        self._index_source = index_source
        self._queue: asyncio.Queue[_Job] = asyncio.Queue()

    def submit(self, result: EpisodeResult) -> None:
        """Queue a finished take for alignment; anything other than a kept episode is ignored."""
        if result.disposition is Disposition.KEPT and result.final_path is not None:
            self._queue.put_nowait(_Job(result.session_id, result.episode_index, result.final_path))

    def recover(self) -> None:
        """Queue the latest recording session's kept episodes that have no aligned copy yet.

        An alignment cut short by a restart is picked up here. Only the latest session is looked at, so a
        station with a long history does not re-attempt every old episode (many of which cannot be aligned).
        """
        sessions = sorted(p for p in self._output_root.glob("*") if (p / MANIFEST_DIR).is_dir())
        for path in sorted((sessions[-1] / MANIFEST_DIR).glob("episode_*.json")) if sessions else []:
            try:
                manifest = read_manifest(path)
            except (OSError, ValueError):
                continue
            if manifest.storage_phase != "FINALIZED" or manifest.finalization != FINALIZATION_KEEP:
                continue
            session = path.parent.parent
            final = (
                self._output_root / destination_path(manifest.assignment)
                if manifest.assignment
                else session / manifest.final_path
            )
            aligned = final.parent.with_name(f"{final.parent.name}_aligned") / final.name
            if not aligned.exists():
                self._queue.put_nowait(_Job(manifest.session_id, manifest.episode_index, final))

    async def run(self) -> None:
        """Align queued episodes until cancelled; a failed alignment is logged and never stops the worker."""
        while True:
            job = await self._queue.get()
            try:
                # Blocking file and numeric work, kept off the event loop that serves recording and preview.
                await asyncio.to_thread(self._align, job)
            except Exception as exc:  # noqa: BLE001 - one episode's failure must not end the worker
                logger.warning("%s episode %d was not time-aligned: %s", job.session_id, job.episode_index, exc)

    def _align(self, job: _Job) -> None:
        """Align one kept episode from its stored manifest; raises with the reason when it is refused."""
        # The manifest stays in the recording session; a routed episode sits elsewhere, under its slot.
        manifest = read_manifest(manifest_path(self._output_root / job.session_id, job.episode_index))
        slot = int(job.final.name.removeprefix("episode_"))
        align_kept_episode(job.final.parent, slot, self._index_source, manifest)
        logger.info("%s episode %d was time-aligned", job.session_id, job.episode_index)
