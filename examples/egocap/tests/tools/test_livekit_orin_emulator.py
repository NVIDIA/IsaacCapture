# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

# tools/ is a plain scripts directory, not a package.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
import livekit_orin_emulator as emu  # noqa: E402


@pytest.fixture
def token_file(tmp_path: Path) -> Path:
    path = tmp_path / "join.token"
    path.write_text("secret-token\n")
    return path


def argv(token_file: Path, *extra: str) -> list[str]:
    return ["--livekit-url=ws://x", f"--livekit-token-file={token_file}", *extra]


def test_parse_valid_streams_and_defaults(token_file):
    args = emu.parse_args(
        argv(token_file, "--add-stream=sensor=0,livekit=left", "--add-stream=sensor=1,livekit=right")
    )
    assert args.streams == [emu.StreamSpec(0, "left"), emu.StreamSpec(1, "right")]
    assert args.token == "secret-token"
    assert (args.width, args.height, args.fps, args.timestamp_basis) == (1280, 720, 30.0, "capture_unix_us")


@pytest.mark.parametrize(
    "bad",
    ["sensor=0", "livekit=a", "sensor=x,livekit=a", "sensor=0,livekit=", "sensor=0,livekit=a,extra=1", "nonsense"],
)
def test_invalid_stream_is_rejected(token_file, bad):
    with pytest.raises(SystemExit):
        emu.parse_args(argv(token_file, f"--add-stream={bad}"))


@pytest.mark.parametrize(
    "streams",
    [["sensor=0,livekit=a", "sensor=0,livekit=b"], ["sensor=0,livekit=a", "sensor=1,livekit=a"]],
)
def test_duplicate_sensor_or_track_is_rejected(token_file, streams):
    with pytest.raises(SystemExit):
        emu.parse_args(argv(token_file, *[f"--add-stream={s}" for s in streams]))


def test_missing_or_empty_token_file_is_rejected(tmp_path, token_file):
    stream = "--add-stream=sensor=0,livekit=a"
    with pytest.raises(SystemExit):
        emu.parse_args(argv(tmp_path / "absent", stream))
    empty = tmp_path / "empty"
    empty.write_text("\n")
    with pytest.raises(SystemExit):
        emu.parse_args(argv(empty, stream))


@pytest.mark.parametrize("basis", ["capture_unix_us", "processing_unix_us", "monotonic_us"])
def test_each_timestamp_basis_choice_is_accepted(token_file, basis):
    args = emu.parse_args(argv(token_file, "--add-stream=sensor=0,livekit=a", f"--timestamp-basis={basis}"))
    assert args.timestamp_basis == basis


@pytest.mark.parametrize(
    "extra", ["--timestamp-basis=unix_us", "--timestamp-basis=bogus", "--fps=0", "--width=1", "--height=0"]
)
def test_bad_options_are_rejected(token_file, extra):
    with pytest.raises(SystemExit):
        emu.parse_args(argv(token_file, "--add-stream=sensor=0,livekit=a", extra))


def test_user_timestamp_basis(monkeypatch):
    monkeypatch.setattr(emu.time, "time_ns", lambda: 5_000_000)
    monkeypatch.setattr(emu.time, "monotonic_ns", lambda: 9_000_000)
    assert emu.user_timestamp_us("capture_unix_us") == 5_000
    assert emu.user_timestamp_us("processing_unix_us") == 5_000
    assert emu.user_timestamp_us("monotonic_us") == 9_000


def test_metadata_fields_and_little_endian_user_data():
    meta = emu.build_metadata(7, 0x0102030405060708, 123)
    assert (meta.user_timestamp, meta.frame_id) == (123, 7)
    assert bytes(meta.user_data) == bytes([8, 7, 6, 5, 4, 3, 2, 1])


def test_pattern_shape_dtype_and_change_over_time():
    first = emu.make_pattern(64, 48, 0, 1)
    assert first.shape == (48, 64, 4) and first.dtype == np.uint8
    assert not np.array_equal(first, emu.make_pattern(64, 48, 0, 2))
    assert not np.array_equal(first, emu.make_pattern(64, 48, 1, 1))


class FakeSource:
    """Records captured frames and stops the run after a fixed number."""

    def __init__(self, stop, limit=None):
        self.stop, self.limit, self.calls, self.closed = stop, limit, [], False

    def capture_frame(self, frame, *, timestamp_us, metadata):
        self.calls.append((frame, timestamp_us, metadata))
        if self.limit and len(self.calls) >= self.limit:
            self.stop.set()

    async def aclose(self):
        self.closed = True


@pytest.fixture
def fake_rtc(monkeypatch):
    """Replaces the SDK with recorders so no server is needed."""
    log = SimpleNamespace(published=[], options=[], unpublished=[], disconnected=False, sources=[], stop=None)

    class Participant:
        async def publish_track(self, track, options):
            log.published.append(track)
            log.options.append(options)
            return SimpleNamespace(sid=f"sid-{track[0]}")

        async def unpublish_track(self, sid):
            log.unpublished.append(sid)

    class Room:
        local_participant = Participant()

        async def connect(self, url, token):
            log.connected = (url, token)

        async def disconnect(self):
            log.disconnected = True

    def make_source(width, height):
        src = FakeSource(log.stop, limit=3)
        log.sources.append(src)
        return src

    fake = SimpleNamespace(
        Room=Room,
        VideoSource=make_source,
        LocalVideoTrack=SimpleNamespace(create_video_track=lambda name, source: (name, source)),
        VideoFrame=lambda w, h, t, data: SimpleNamespace(width=w, height=h, data=data),
        VideoBufferType=SimpleNamespace(RGBA=1),
        TrackPublishOptions=lambda **kw: kw,
        TrackSource=SimpleNamespace(SOURCE_CAMERA=2),
        FrameMetadataFeature=SimpleNamespace(FMF_USER_TIMESTAMP=1, FMF_FRAME_ID=2, FMF_USER_DATA=3),
        FrameMetadata=emu.rtc.FrameMetadata,
    )
    monkeypatch.setattr(emu, "rtc", fake)
    return log


async def test_run_publishes_paces_and_shuts_down_cleanly(token_file, fake_rtc):
    args = emu.parse_args(
        argv(token_file, "--add-stream=sensor=0,livekit=a", "--add-stream=sensor=1,livekit=b", "--fps=200")
    )
    args.width, args.height = 48, 16
    stop = asyncio.Event()
    fake_rtc.stop = stop
    await asyncio.wait_for(emu.run(args, stop), timeout=5)
    # Every track is published by name and each source stops at its own limit or on shutdown.
    assert [t[0] for t in fake_rtc.published] == ["a", "b"]
    assert fake_rtc.connected == ("ws://x", "secret-token")
    ids = [m.frame_id for _, _, m in fake_rtc.sources[0].calls]
    assert ids == list(range(1, len(ids) + 1)) and len(ids) >= 1
    # Cleanup happened: unpublished, closed, disconnected.
    assert fake_rtc.unpublished == ["sid-a", "sid-b"]
    assert all(s.closed for s in fake_rtc.sources) and fake_rtc.disconnected


async def test_every_track_requests_all_frame_metadata_features(token_file, fake_rtc):
    args = emu.parse_args(
        argv(token_file, "--add-stream=sensor=0,livekit=a", "--add-stream=sensor=1,livekit=b", "--fps=200")
    )
    args.width, args.height = 48, 16
    stop = asyncio.Event()
    fake_rtc.stop = stop
    await asyncio.wait_for(emu.run(args, stop), timeout=5)
    features = emu.rtc.FrameMetadataFeature
    wanted = [features.FMF_USER_TIMESTAMP, features.FMF_FRAME_ID, features.FMF_USER_DATA]
    assert len(fake_rtc.options) == 2
    assert all(o["frame_metadata_features"] == wanted for o in fake_rtc.options)
    assert all(o["source"] == emu.rtc.TrackSource.SOURCE_CAMERA for o in fake_rtc.options)


async def test_publish_stream_is_paced_by_fps(token_file, fake_rtc):
    args = emu.parse_args(argv(token_file, "--add-stream=sensor=0,livekit=a", "--fps=50"))
    args.width, args.height = 48, 16
    stop = asyncio.Event()
    source = FakeSource(stop, limit=6)
    counters = emu.Counters({"a": 0})
    loop = asyncio.get_running_loop()
    start = loop.time()
    await emu.publish_stream(source, emu.StreamSpec(0, "a"), args, stop, counters)
    elapsed = loop.time() - start
    # Six frames at 50 fps leave five 20 ms gaps before the stop is noticed.
    assert elapsed >= 0.09
    assert counters.sent["a"] == 6


async def test_stats_print_until_stopped(capsys, monkeypatch):
    monkeypatch.setattr(emu, "STATS_INTERVAL_S", 0.01)
    stop = asyncio.Event()
    counters = emu.Counters({"a": 3})
    task = asyncio.create_task(emu.report_stats(stop, counters))
    await asyncio.sleep(0.05)
    stop.set()
    await task
    assert "a=3" in capsys.readouterr().out


def test_banner_never_shows_token(token_file, capsys):
    emu.print_banner(emu.parse_args(argv(token_file, "--add-stream=sensor=0,livekit=a")))
    out = capsys.readouterr().out
    assert "secret-token" not in out and "a" in out
