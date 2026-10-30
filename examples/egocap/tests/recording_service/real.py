# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Helpers that feed a real recorded episode through the public RecordingService API.

Nothing here simulates a device: samples are the stored rows and decoded frames of an actual
episode (``EGOCAP_REAL_EPISODE``), submitted in deterministic round-robin order over the
frozen source order, retrying the same sample when the service reports FULL.
"""

import asyncio
import os
import subprocess
import sys
from pathlib import Path

import pytest

from egocap.recording.dataset_reader import EpisodeReader
from recording_service import (
    NumericSample,
    PixelFormat,
    QcConfig,
    RecordingService,
    ServiceConfig,
    SubmissionStatus,
    VideoSample,
)

FIXTURE = os.environ.get("EGOCAP_REAL_EPISODE")
needs_real_episode = pytest.mark.skipif(not FIXTURE, reason="EGOCAP_REAL_EPISODE not set (real-data test SKIPPED)")
NUMERIC = {
    "pico_controller": ("head_pose", "left_pose", "right_pose", "a_button", "left_trigger"),
    "manus_glove": None,
}
CAMERA = "head_human_camera"


def reader() -> EpisodeReader:
    """Open the real episode named by the environment."""
    return EpisodeReader(Path(FIXTURE))


def make_config(output_root: Path, **overrides) -> ServiceConfig:
    """A configuration whose source schemas come from the real episode's own stored schema."""
    r = reader()
    numeric = tuple(r.numeric_spec(name, fields) for name, fields in NUMERIC.items())
    video = (r.video_spec(CAMERA),)
    qc = overrides.pop("qc", QcConfig(required_sources=("pico_controller", CAMERA)))
    return ServiceConfig(output_root=output_root, numeric_sources=numeric, video_sources=video, qc=qc, **overrides)


def samples(rows: int):
    """Round-robin sample list over the frozen source order: rows from each source, in recorded order."""
    r = reader()
    streams = []
    for name in NUMERIC:
        spec_fields = {f.name for f in r.numeric_spec(name, NUMERIC[name]).fields}
        streams.append(
            [
                NumericSample(
                    name,
                    row.capture_time,
                    row.receive_time,
                    {k: v for k, v in row.fields.items() if k in spec_fields},
                )
                for row in r.numeric_rows(name, 0, rows)
            ]
        )
    streams.append(
        [
            VideoSample(CAMERA, f.image_rgb, PixelFormat.RGB8, f.capture_time, f.receive_time, f.intrinsics)
            for f in r.video_frames(CAMERA, 0, rows)
        ]
    )
    ordered = []
    for i in range(rows):
        ordered.extend(stream[i] for stream in streams if i < len(stream))
    return ordered


async def submit_all(service: RecordingService, episode_id: str, items) -> None:
    """Submit every sample, yielding to the writer and retrying the same sample whenever the queue is FULL."""
    for item in items:
        while service.submit_sample(episode_id, item).status is SubmissionStatus.FULL:
            await asyncio.sleep(0)


CHILD = Path(__file__).parent / "crash_child.py"


def run_child(root: Path, mode: str, session_id: str | None = None) -> subprocess.Popen:
    """Start the crash child with the same environment (fixture path) as this test process."""
    args = [sys.executable, str(CHILD), str(root), mode] + ([session_id] if session_id else [])
    return subprocess.Popen(args, stdout=subprocess.PIPE, text=True, env=os.environ.copy())


async def kept_episode(config, rows: int = 10):
    """Record one real take to completion and return (session, result)."""
    service = RecordingService(config)
    try:
        session = await service.open_session()
        episode = await service.start_episode("kept")
        await submit_all(service, episode.episode_id, samples(rows))
        result = await service.stop_episode(episode.episode_id)
    finally:
        closed = await service.close()
    assert closed.released
    return session, result
