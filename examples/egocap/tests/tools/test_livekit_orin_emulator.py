# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import struct
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
    log = SimpleNamespace(
        published=[], options=[], unpublished=[], disconnected=False, sources=[], stop=None, handlers={}
    )

    class Participant:
        async def publish_track(self, track, options):
            log.published.append(track)
            log.options.append(options)
            return SimpleNamespace(sid=f"sid-{track[0]}")

        async def unpublish_track(self, sid):
            log.unpublished.append(sid)

    class Room:
        local_participant = Participant()

        def on(self, event, callback=None):
            log.handlers[event] = callback

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
    assert "data_received" in fake_rtc.handlers
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
    _ = await task
    assert "a=3" in capsys.readouterr().out


def test_banner_never_shows_token(token_file, capsys):
    emu.print_banner(emu.parse_args(argv(token_file, "--add-stream=sensor=0,livekit=a")))
    out = capsys.readouterr().out
    assert "secret-token" not in out and "a" in out


async def test_stamps_are_taken_together_before_the_frame_is_synthesized(token_file, fake_rtc, monkeypatch):
    args = emu.parse_args(argv(token_file, "--add-stream=sensor=0,livekit=a", "--fps=200"))
    args.width, args.height = 4, 4
    now_us = [1_000]

    def slow_pattern(width, height, sensor, frame_id):
        now_us[0] += 500  # synthesizing a frame takes time
        return np.zeros((height, width, 4), np.uint8)

    monkeypatch.setattr(emu, "make_pattern", slow_pattern)
    monkeypatch.setattr(emu, "user_timestamp_us", lambda basis: now_us[0])
    stop = asyncio.Event()
    source = FakeSource(stop, limit=1)
    await emu.publish_stream(
        source, emu.StreamSpec(0, "a"), args, stop, emu.Counters({"a": 0}), capture_clock=lambda: now_us[0] * 1000
    )
    _, timestamp_us, metadata = source.calls[0]
    # The same stamp goes to the frame and the metadata, and excludes the synthesis time.
    assert timestamp_us == metadata.user_timestamp == 1_000
    assert int.from_bytes(bytes(metadata.user_data), "little") == 1_000_000


@pytest.mark.parametrize(
    ("basis", "expected_us"),
    [("capture_unix_us", 1_000), ("monotonic_us", 1_000), ("processing_unix_us", 1_500)],
)
async def test_only_the_processing_basis_is_stamped_after_synthesis(
    token_file, fake_rtc, monkeypatch, basis, expected_us
):
    args = emu.parse_args(
        argv(token_file, "--add-stream=sensor=0,livekit=a", "--fps=200", f"--timestamp-basis={basis}")
    )
    args.width, args.height = 4, 4
    now_us = [1_000]

    def slow_pattern(width, height, sensor, frame_id):
        now_us[0] += 500
        return np.zeros((height, width, 4), np.uint8)

    monkeypatch.setattr(emu, "make_pattern", slow_pattern)
    monkeypatch.setattr(emu, "user_timestamp_us", lambda _basis: now_us[0])
    stop = asyncio.Event()
    source = FakeSource(stop, limit=1)
    await emu.publish_stream(
        source, emu.StreamSpec(0, "a"), args, stop, emu.Counters({"a": 0}), capture_clock=lambda: now_us[0] * 1000
    )
    _, timestamp_us, metadata = source.calls[0]
    assert timestamp_us == metadata.user_timestamp == expected_us
    # The raw capture counter is always read before synthesis.
    assert int.from_bytes(bytes(metadata.user_data), "little") == 1_000_000


# --- ClockModel ----------------------------------------------------------------


def _model(now_ns, **kw):
    """A model whose run-time clock is the one-element list ``now_ns`` (read at call time)."""
    return emu.ClockModel(started_ns=0, monotonic_ns=lambda: now_ns[0], **kw)


def test_offset_only_applies_to_unix_bases_at_any_run_time():
    now_ns = [0]
    model = _model(now_ns, offset_us=250_000)
    for now_ns[0] in (0, 10_000_000_000):
        assert model.adjustment_us("capture_unix_us", at_capture=True) == 250_000
        assert model.adjustment_us("processing_unix_us", at_capture=False) == 250_000


def test_drift_grows_linearly_with_injected_run_time():
    now_ns = [0]
    model = _model(now_ns, drift_ppm=15.0)
    seen = []
    for now_ns[0] in (0, 10_000_000_000, 20_000_000_000):
        seen.append(model.adjustment_us("capture_unix_us", at_capture=True))
    assert seen == [0, 150, 300]


def test_pipeline_delay_applies_at_capture_only_and_monotonic_follows_its_own_offset():
    now_ns = [10_000_000_000]
    model = _model(
        now_ns, offset_us=1_000, drift_ppm=15.0, pipeline_delay_us=40_000, monotonic_offset_ns=3_000_000_000
    )
    assert model.adjustment_us("capture_unix_us", at_capture=True) == 1_000 + 150 - 40_000
    assert model.adjustment_us("processing_unix_us", at_capture=False) == 1_000 + 150
    # The Unix offset does not leak into the monotonic basis.
    assert model.adjustment_us("monotonic_us", at_capture=True) == 3_000_000 + 150 - 40_000
    assert model.adjustment_us("monotonic_us", at_capture=False) == 3_000_000 + 150


def test_orin_monotonic_is_host_plus_offset_plus_drift():
    now_ns = [0]
    model = _model(now_ns, monotonic_offset_ns=3_000_000_000, drift_ppm=15.0)
    assert model.orin_monotonic_ns() == 3_000_000_000
    now_ns[0] = 10_000_000_000
    assert model.orin_monotonic_ns() == 10_000_000_000 + 3_000_000_000 + 150_000


def test_from_args_reads_the_command_line_units():
    args = SimpleNamespace(
        clock_offset_ms=-1.5, clock_drift_ppm=2.0, pipeline_delay_ms=30.0, monotonic_offset_s=3.0
    )
    model = emu.ClockModel.from_args(args, monotonic_ns=lambda: 42)
    assert (
        model.offset_us,
        model.drift_ppm,
        model.pipeline_delay_us,
        model.monotonic_offset_ns,
        model.started_ns,
    ) == (-1_500, 2.0, 30_000, 3_000_000_000, 42)


# --- Clock-sync replies --------------------------------------------------------


def _request(seq=7, t1=123_456_789, kind=b"Q"):
    return struct.pack("<cIq", kind, seq, t1)


def test_build_sync_reply_round_trips_and_echoes_the_request():
    reply = emu.build_sync_reply(_request(), 1_000, 2_000)
    assert struct.unpack("<cIqqq", reply) == (b"R", 7, 123_456_789, 1_000, 2_000)


@pytest.mark.parametrize("request_bytes", [b"", _request()[:-1], _request() + b"\0", _request(kind=b"R")])
def test_build_sync_reply_rejects_malformed_requests(request_bytes):
    assert emu.build_sync_reply(request_bytes, 1, 2) is None


class FakeRoom:
    """A room whose local participant records every publish_data call."""

    def __init__(self):
        self.sent = []
        self.local_participant = self

    async def publish_data(self, payload, **kwargs):
        self.sent.append((payload, kwargs))


async def test_answer_clock_sync_replies_to_the_requester_only_with_t3_after_t2():
    now_ns = [5_000_000_000]
    clock = _model(now_ns, monotonic_offset_ns=3_000_000_000)
    t2 = clock.orin_monotonic_ns()
    now_ns[0] += 40_000
    room = FakeRoom()
    await emu.answer_clock_sync(room, "pc", _request(), t2, clock)
    [(payload, kwargs)] = room.sent
    assert kwargs == {"reliable": False, "destination_identities": ["pc"], "topic": emu.CLOCK_SYNC_TOPIC}
    kind, seq, t1, got_t2, t3 = struct.unpack("<cIqqq", payload)
    assert (kind, seq, t1, got_t2) == (b"R", 7, 123_456_789, t2)
    assert t3 == t2 + 40_000


async def test_answer_clock_sync_sends_nothing_for_a_malformed_request():
    room = FakeRoom()
    await emu.answer_clock_sync(room, "pc", b"junk", 1, _model([0]))
    assert room.sent == []


def test_exchange_with_the_emulator_reply_recovers_the_monotonic_offset():
    exchange_cls = pytest.importorskip("egocap_web.clock_exchange").Exchange
    now_ns = [0]
    clock = _model(now_ns, monotonic_offset_ns=3_000_000_000)
    one_way, t1 = 5_000_000, 100_000_000_000
    # The PC's clock is this host's; the emulator reads host + 3 s and replies on arrival.
    now_ns[0] = t1 + one_way
    t2 = t3 = clock.orin_monotonic_ns()
    _, _, echoed_t1, r2, r3 = struct.unpack("<cIqqq", emu.build_sync_reply(_request(t1=t1), t2, t3))
    exchange = exchange_cls(t1=echoed_t1, t2=r2, t3=r3, t4=t1 + 2 * one_way)
    assert abs(exchange.offset_ns - 3_000_000_000) <= 1_000
    assert exchange.rtt_ns == 10_000_000


async def _publish_one(token_file, monkeypatch, sensor, clock, *extra):
    """Publishes one frame of a track at a frozen host time of 5 s Unix and returns the captured call."""
    name = f"t{sensor}"
    args = emu.parse_args(argv(token_file, f"--add-stream=sensor={sensor},livekit={name}", *extra))
    args.width, args.height = 4, 4
    monkeypatch.setattr(emu, "user_timestamp_us", lambda _basis: 5_000_000)
    stop = asyncio.Event()
    source = FakeSource(stop, limit=1)
    await emu.publish_stream(
        source, emu.StreamSpec(sensor, name), args, stop, emu.Counters({name: 0}), lambda: 7_000_000, clock
    )
    return source.calls[0]


async def test_tracks_sharing_one_model_get_equal_stamps_and_keep_their_identity(token_file, monkeypatch):
    clock = _model([10_000_000_000], offset_us=2_000, drift_ppm=15.0)
    first = await _publish_one(token_file, monkeypatch, 0, clock)
    second = await _publish_one(token_file, monkeypatch, 1, clock)
    expected = 5_000_000 + 2_000 + 150
    for _, timestamp_us, metadata in (first, second):
        assert timestamp_us == metadata.user_timestamp == expected
        assert metadata.frame_id == 1
        assert int.from_bytes(bytes(metadata.user_data), "little") == 7_000_000


async def test_all_zero_model_reproduces_this_hosts_readings(token_file, monkeypatch):
    model = emu.ClockModel()
    for basis in ("capture_unix_us", "processing_unix_us", "monotonic_us"):
        assert model.adjustment_us(basis, at_capture=True) == model.adjustment_us(basis, at_capture=False) == 0
    _, timestamp_us, metadata = await _publish_one(token_file, monkeypatch, 0, model)
    assert timestamp_us == metadata.user_timestamp == 5_000_000
