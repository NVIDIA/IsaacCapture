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
from pathlib import Path

from egocap.recording.aligned_writer import align_kept_episode
from egocap.recording.episode_layout import manifest_path, read_manifest
from recording_service import Disposition, EpisodeResult

logger = logging.getLogger(__name__)


class AutoAligner:
    """Aligns kept episodes one at a time on its own worker task."""

    def __init__(self, output_root: Path, index_source: str) -> None:
        """``output_root`` holds the recording sessions; ``index_source`` is the camera that sets the timeline."""
        self._output_root = output_root
        self._index_source = index_source
        self._queue: asyncio.Queue[EpisodeResult] = asyncio.Queue()

    def submit(self, result: EpisodeResult) -> None:
        """Queue a finished take for alignment; anything other than a kept episode is ignored."""
        if result.disposition is Disposition.KEPT and result.final_path is not None:
            self._queue.put_nowait(result)

    async def run(self) -> None:
        """Align queued episodes until cancelled; a failed alignment is logged and never stops the worker."""
        while True:
            result = await self._queue.get()
            try:
                # Blocking file and numeric work, kept off the event loop that serves recording and preview.
                await asyncio.to_thread(self._align, result)
            except Exception as exc:  # noqa: BLE001 - one episode's failure must not end the worker
                logger.warning("episode %s was not time-aligned: %s", result.episode_id, exc)

    def _align(self, result: EpisodeResult) -> None:
        """Align one kept episode from its stored manifest; raises with the reason when it is refused."""
        final = result.final_path
        assert final is not None
        # The manifest stays in the recording session; a routed episode sits elsewhere, under its slot.
        manifest = read_manifest(manifest_path(self._output_root / result.session_id, result.episode_index))
        slot = int(final.name.removeprefix("episode_"))
        align_kept_episode(final.parent, slot, self._index_source, manifest)
        logger.info("episode %s was time-aligned", result.episode_id)
