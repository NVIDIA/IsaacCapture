# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Timing evidence attached to a take: written with the kept episode's attributes, validated, in memory only.

Uses a small synthetic take, so no recorded-episode fixture is needed.
"""

import asyncio
import json

import numpy as np
import pytest
import zarr

from recording_service import (
    FieldSpec,
    NumericSample,
    NumericSourceSpec,
    PixelFormat,
    QcConfig,
    RecordingService,
    ServiceConfig,
    StateError,
    VideoSample,
    VideoSourceSpec,
)

ROWS = 5
TIMING = {
    "clock_mapping": {"method": "m", "mappings": [{"epoch": 1, "offset_ns": -7}]},
}


def _config(tmp_path) -> ServiceConfig:
    return ServiceConfig(
        output_root=tmp_path / "out",
        numeric_sources=(NumericSourceSpec("pose", (FieldSpec("v", "float64"),)),),
        video_sources=(VideoSourceSpec("cam", 64, 48, 30.0),),
        qc=QcConfig(required_sources=("pose", "cam")),
    )


async def _submit(service: RecordingService, episode_id: str) -> None:
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    for i in range(ROWS):
        t = 1.0 + i / 30
        for sample in (
            NumericSample("pose", t, t, {"v": float(i)}),
            VideoSample("cam", frame, PixelFormat.RGB8, t, t, np.eye(3)),
        ):
            while service.submit_sample(episode_id, sample).status.name == "FULL":
                await asyncio.sleep(0)


async def _take(service: RecordingService, timing):
    episode = await service.start_episode("t")
    if timing is not None:
        service.set_episode_timing(episode.episode_id, timing)
    await _submit(service, episode.episode_id)
    result = await service.stop_episode(episode.episode_id)
    return dict(zarr.open_group(str(result.final_path), mode="r").attrs)


async def test_timing_set_before_stop_is_written_to_the_kept_attributes_and_not_to_the_next_take(tmp_path, own):
    service = own(RecordingService(_config(tmp_path)))
    await service.open_session()
    assert (await _take(service, TIMING))["timing"] == TIMING
    assert "timing" not in await _take(service, None)


async def test_a_take_without_timing_has_no_timing_attribute(tmp_path, own):
    service = own(RecordingService(_config(tmp_path)))
    await service.open_session()
    assert "timing" not in await _take(service, None)


async def test_timing_for_an_unknown_or_finished_episode_is_refused(tmp_path, own):
    service = own(RecordingService(_config(tmp_path)))
    await service.open_session()
    with pytest.raises(StateError, match="no active episode"):
        service.set_episode_timing("not-an-episode", TIMING)
    episode = await service.start_episode("t")
    await _submit(service, episode.episode_id)
    await service.stop_episode(episode.episode_id)
    with pytest.raises(StateError, match="no active episode"):
        service.set_episode_timing(episode.episode_id, TIMING)


async def test_a_value_that_cannot_be_serialized_is_refused_and_nothing_is_stored(tmp_path, own):
    service = own(RecordingService(_config(tmp_path)))
    await service.open_session()
    episode = await service.start_episode("t")
    with pytest.raises(TypeError):
        service.set_episode_timing(episode.episode_id, {"bad": object()})
    await _submit(service, episode.episode_id)
    result = await service.stop_episode(episode.episode_id)
    assert "timing" not in dict(zarr.open_group(str(result.final_path), mode="r").attrs)


async def test_timing_is_not_in_the_recovery_manifest(tmp_path, own):
    """It is held in memory only, so a take recovered from its manifest cannot carry it."""
    service = own(RecordingService(_config(tmp_path)))
    session = await service.open_session()
    episode = await service.start_episode("t")
    service.set_episode_timing(episode.episode_id, TIMING)
    manifest = (session.session_path / ".recording" / "episode_0.json").read_text()
    assert "timing" not in json.loads(manifest) and "host_poll" not in manifest
    await service.discard_episode(episode.episode_id)
