# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Time-aligns each kept episode in the background, right after its take is finalized.

One worker handles episodes in the order they were kept. Each alignment runs as a low-priority child process (the
same ``recording/tools/align_episode.py`` an operator can run by hand), so its CPU and memory never compete with a
live take inside the recorder's own process, and a shutdown can stop it. An episode that cannot be aligned is
logged with the reason and left as recorded; the raw episode is never modified. Output is published by one rename,
so an alignment that is killed leaves no partial result. At startup the worker re-queues the kept episodes that
still have no aligned copy.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from egocap.recording.aligned_writer import episode_qualification
from egocap.recording.episode_layout import (
    FINALIZATION_KEEP,
    MANIFEST_DIR,
    destination_path,
    read_manifest,
)
from recording_service import Disposition, EpisodeResult

logger = logging.getLogger("isaaccapture.egocap_web.auto_align")

#: The command-line aligner, run once per episode.
_ALIGN_TOOL = Path(__file__).resolve().parents[2] / "recording" / "tools" / "align_episode.py"
#: A staging directory untouched this long belongs to a run that is no longer going.
_STALE_STAGING_SECONDS = 24 * 3600


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

    @property
    def refused_marker(self) -> Path:
        """A hidden file in the aligned session recording that this episode was refused; deleting it retries."""
        return self.aligned.parent / f".{self.final.name}.refused"


class AutoAligner:
    """Aligns kept episodes one at a time on its own worker task."""

    def __init__(self, output_root: Path, index_source: str) -> None:
        """``output_root`` holds the recording sessions; ``index_source`` is the camera that sets the timeline."""
        self._output_root = output_root
        self._index_source = index_source
        # Unbounded: a job is three small fields, and the number waiting is the number of kept episodes.
        self._queue: asyncio.Queue[_Job] = asyncio.Queue()
        self._process: asyncio.subprocess.Process | None = None
        # False while the open session cannot be aligned; set once per session open.
        self._session_alignable = True

    def note_video_sources(self, recorded: frozenset[str], qualified: frozenset[str]) -> None:
        """Note whether the session being opened can be aligned; if not, its episodes are not queued.

        It can when the index camera is among the cameras recorded with a sensor capture time and every recorded
        camera has one, since an episode with any other camera is refused whole.
        """
        alignable = self._index_source in qualified and recorded <= qualified
        if not alignable and self._session_alignable:
            logger.warning(
                "automatic time alignment is off: index source %r and every recorded camera need a sensor "
                "capture time (recorded: %s; with one: %s)",
                self._index_source,
                sorted(recorded),
                sorted(qualified),
            )
        self._session_alignable = alignable

    def submit(self, result: EpisodeResult) -> None:
        """Queue a finished take for alignment; anything other than a kept episode is ignored."""
        if result.disposition is Disposition.KEPT and result.final_path is not None and self._session_alignable:
            self._queue.put_nowait(_Job(result.session_id, result.episode_index, result.final_path))

    def scan(self) -> list[_Job]:
        """Every kept episode, in any session, that has no aligned copy and whose evidence allows one.

        Skips episodes already refused (see ``_Job.refused_marker``), and an episode recorded without qualified
        timing evidence (an older recording, or a station with no tracking delay set), so history is not retried at
        every start. Also removes staging directories left by a run that was killed long ago. Blocking file work.
        """
        jobs: list[_Job] = []
        groups: set[Path] = set()
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
                groups.add(final.parent)
                eligible = (
                    manifest.storage_phase == "FINALIZED"
                    and manifest.finalization == FINALIZATION_KEEP
                    and not job.aligned.exists()
                    and not job.refused_marker.exists()
                    and all(
                        episode_qualification(
                            final.parent, int(final.name.removeprefix("episode_")), self._index_source
                        ).values()
                    )
                )
            except Exception:  # noqa: BLE001, S112 - an unreadable record is simply not eligible
                continue
            if eligible:
                jobs.append(job)
        for group in groups:
            for stale in group.parent.glob(f".{group.name}_aligned.staging-*"):
                with contextlib.suppress(OSError):
                    if time.time() - stale.stat().st_mtime > _STALE_STAGING_SECONDS:
                        shutil.rmtree(stale, ignore_errors=True)
        return jobs

    async def run(self) -> None:
        """Re-queue unfinished work, then align queued episodes until cancelled; failures never stop the worker."""
        try:
            for job in await asyncio.to_thread(self.scan):
                self._queue.put_nowait(job)
        except Exception as exc:  # noqa: BLE001 - recovery is best effort; live episodes are still aligned
            logger.warning("time-alignment recovery failed: %s", exc)
        while True:
            job = await self._queue.get()
            try:
                await self._align(job)
            except Exception as exc:  # noqa: BLE001 - one episode's failure must not end the worker
                logger.warning("%s episode %d was not time-aligned: %s", job.session_id, job.episode_index, exc)
            finally:
                self._queue.task_done()

    async def close(self, task: asyncio.Task[None], timeout: float) -> None:
        """Let the queued episodes finish for up to ``timeout`` seconds, then stop the worker ``task``.

        Call after the recorder has closed, so the last kept episode is already queued. An alignment still running
        at the deadline is killed; it publishes only by one rename, so nothing partial appears, and the episode is
        picked up again at the next start.
        """
        try:
            await asyncio.wait_for(self._queue.join(), timeout)
        except asyncio.TimeoutError:
            logger.warning("time alignment still has queued episodes at shutdown; they are retried at next start")
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            _ = await task

    async def _align(self, job: _Job) -> None:
        """Align one kept episode in a child process; raises with the reason when it is refused or fails."""
        if job.aligned.exists():
            return
        # The manifest stays in the recording session; a routed episode sits elsewhere, under its slot.
        command = [
            "nice",
            "-n",
            "10",
            sys.executable,
            str(_ALIGN_TOOL),
            str(job.final.parent),
            "--episode",
            str(int(job.final.name.removeprefix("episode_"))),
            "--index-source",
            self._index_source,
            "--manifest-session",
            str(self._output_root / job.session_id),
            "--manifest-episode",
            str(job.episode_index),
        ]
        process = await asyncio.create_subprocess_exec(
            *command, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE
        )
        self._process = process
        try:
            _, stderr = await process.communicate()
        except asyncio.CancelledError:
            # Cancelled at shutdown: stop the child rather than leave it running past the recorder.
            process.kill()
            await process.wait()
            raise
        finally:
            self._process = None
        reason = stderr.decode(errors="replace").strip()
        if process.returncode == 1:
            # Exit 1 is a refusal, which is the same every time, so remember it instead of retrying at every start;
            # any other failure (an unreadable manifest, a killed child) stays retryable.
            job.refused_marker.parent.mkdir(parents=True, exist_ok=True)
            job.refused_marker.write_text(reason + "\n", encoding="utf-8")
        if process.returncode != 0:
            raise RuntimeError(reason or f"aligner exited with status {process.returncode}")
        logger.info("%s episode %d was time-aligned", job.session_id, job.episode_index)
