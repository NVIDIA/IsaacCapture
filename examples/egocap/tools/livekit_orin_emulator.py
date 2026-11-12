# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Stands in for the Orin's LiveKit publisher so the PC side can be developed without a Jetson.

It joins a room with a token file and publishes one test-pattern video track per ``--add-stream``, with the
per-frame metadata the real ``camera_plugin_sensing`` sender attaches: ``user_timestamp`` (microseconds),
``frame_id`` (from 1 per track) and 8 bytes of little-endian ``user_data`` holding the sensor's raw capture
counter. Video goes through the SDK's software encoder, so this exercises the PC side and the metadata contract,
not the Jetson hardware encoder, its H.264 profile or the real sender's pacing.

The top half of the file is pure (argument parsing, metadata, pattern); the bottom half publishes.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import signal
import struct
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    from livekit import rtc
except ImportError as exc:  # the optional "livekit" extra is not installed
    raise SystemExit("the emulator needs the livekit package: install the egocap 'livekit' extra") from exc

#: Seconds between periodic stats lines.
STATS_INTERVAL_S = 5.0
#: Bases on the emulated publisher's monotonic clock, and bases that stamp the capture event (not processing time).
_MONOTONIC_BASES = ("capture_monotonic_us", "monotonic_us")
_CAPTURE_BASES = ("capture_monotonic_us", "capture_unix_us")
#: Width of the frame-counter strip, in bits (one coloured block per bit).
_COUNTER_BITS = 24


@dataclass(frozen=True)
class StreamSpec:
    """One published track: which sensor it imitates and the track name consumers look for."""

    sensor: int
    name: str


# --- Pure helpers -------------------------------------------------------------------------------------------


def parse_stream(text: str) -> StreamSpec:
    """Parses ``sensor=N,livekit=NAME`` (the native sender's ``--add-stream`` spelling)."""
    # Split into key=value pairs; anything malformed is a usage error, not a crash.
    try:
        fields = dict(part.split("=", 1) for part in text.split(","))
        sensor = int(fields.pop("sensor"))
        name = fields.pop("livekit")
    except (ValueError, KeyError):
        raise argparse.ArgumentTypeError(f"expected sensor=N,livekit=NAME, got {text!r}") from None
    # Unknown keys or empty names would silently publish the wrong thing.
    if fields or not name or sensor < 0:
        raise argparse.ArgumentTypeError(f"expected sensor=N,livekit=NAME, got {text!r}")
    return StreamSpec(sensor, name)


def build_parser() -> argparse.ArgumentParser:
    """Builds the command line, mirroring the native sender's option spelling."""
    parser = argparse.ArgumentParser(description="Emulate the Orin's LiveKit camera publisher.")
    parser.add_argument("--livekit-url", required=True, help="LiveKit server URL (ws:// or wss://)")
    parser.add_argument("--livekit-token-file", required=True, type=Path, help="file holding the join token")
    parser.add_argument(
        "--add-stream",
        action="append",
        required=True,
        type=parse_stream,
        dest="streams",
        metavar="sensor=N,livekit=NAME",
        help="publish a track named NAME imitating sensor N; repeatable",
    )
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument(
        "--timestamp-basis",
        choices=("capture_monotonic_us", "capture_unix_us", "processing_unix_us", "monotonic_us"),
        default="monotonic_us",
        help="what user_timestamp carries: monotonic_us is the emulated Orin's monotonic time when the frame is "
        "ready (what the native sender stamps by default; recordable, mapped by clock-sync); "
        "capture_monotonic_us is the same clock declared as the sensor capture event (also recordable); "
        "capture_unix_us and processing_unix_us are Unix-time stamps, which cannot be recorded",
    )
    parser.add_argument(
        "--clock-offset-ms",
        type=float,
        default=0.0,
        help="emulate an Orin whose Unix clock reads this many milliseconds ahead of this host's (negative: "
        "behind); applies to the Unix bases, since monotonic clocks of two machines are unrelated",
    )
    parser.add_argument(
        "--clock-drift-ppm",
        type=float,
        default=0.0,
        help="the emulated Orin clock gains this many microseconds per second of run time "
        "(every basis, and the clock-sync replies)",
    )
    parser.add_argument(
        "--pipeline-delay-ms",
        type=float,
        default=0.0,
        help="a capture-basis stamp names the capture event, which happened this many milliseconds before the "
        "frame is published (exposure, readout, ISP and encode); the processing basis is unaffected",
    )
    parser.add_argument(
        "--monotonic-offset-s",
        type=float,
        default=0.0,
        help="emulate an Orin whose monotonic clock reads this many seconds ahead of this host's (a different "
        "boot time); applies to the monotonic basis and to the clock-sync replies",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parses and validates the command line; exits with a usage error on any invalid combination."""
    parser = build_parser()
    args = parser.parse_args(argv)
    # A sensor or track name published twice would collide in the room.
    sensors = [s.sensor for s in args.streams]
    names = [s.name for s in args.streams]
    if len(set(sensors)) != len(sensors):
        parser.error("duplicate sensor in --add-stream")
    if len(set(names)) != len(names):
        parser.error("duplicate livekit track name in --add-stream")
    # Sizes and rate must be positive or the pacing loop and the pattern are meaningless.
    if args.width < _COUNTER_BITS or args.height < 1 or args.fps <= 0:
        parser.error(f"--width must be at least {_COUNTER_BITS}, --height at least 1 and --fps positive")
    # Fail now, not after connecting, on a token file that cannot be read; the token itself is never printed.
    try:
        args.token = args.livekit_token_file.read_text().strip()
    except OSError as exc:
        parser.error(f"cannot read --livekit-token-file: {exc.strerror}")
    if not args.token:
        parser.error("--livekit-token-file is empty")
    return args


def user_timestamp_us(basis: str) -> int:
    """Returns now in microseconds on the chosen clock (a Unix basis, or a monotonic one)."""
    ns = time.monotonic_ns() if basis in _MONOTONIC_BASES else time.time_ns()
    return ns // 1000


@dataclass(frozen=True)
class ClockModel:
    """How the emulated Orin's stamps differ from this host's clock: an offset, a slow drift, and the delay between
    a capture event and its publication. All zero reproduces this host's own readings."""

    offset_us: int = 0
    drift_ppm: float = 0.0
    pipeline_delay_us: int = 0
    monotonic_offset_ns: int = 0
    started_ns: int = 0
    #: The run-time clock the drift is measured on; injectable so a test can step it deterministically.
    monotonic_ns: Callable[[], int] = time.monotonic_ns

    @classmethod
    def from_args(
        cls, args: argparse.Namespace, monotonic_ns: Callable[[], int] = time.monotonic_ns
    ) -> ClockModel:
        return cls(
            offset_us=round(args.clock_offset_ms * 1000),
            drift_ppm=args.clock_drift_ppm,
            pipeline_delay_us=round(args.pipeline_delay_ms * 1000),
            monotonic_offset_ns=round(args.monotonic_offset_s * 1e9),
            started_ns=monotonic_ns(),
            monotonic_ns=monotonic_ns,
        )

    def _drift_us(self) -> float:
        return (self.monotonic_ns() - self.started_ns) / 1e9 * self.drift_ppm

    def orin_monotonic_ns(self) -> int:
        """The emulated Orin's monotonic clock: this host's, a different boot time, and the same slow drift."""
        return self.monotonic_ns() + self.monotonic_offset_ns + round(self._drift_us() * 1000)

    def adjustment_us(self, basis: str, *, at_capture: bool) -> int:
        """Microseconds to add to this host's reading of ``basis`` to get the emulated Orin's stamp."""
        offset_us = self.monotonic_offset_ns // 1000 if basis in _MONOTONIC_BASES else self.offset_us
        delay_us = self.pipeline_delay_us if at_capture else 0
        return round(offset_us + self._drift_us() - delay_us)


#: A clock-sync request from the PC (``Q``, sequence, t1) and this publisher's reply (``R``, sequence, t1, t2, t3):
#: little-endian, t1 on the PC's monotonic clock and t2/t3 on this publisher's, in nanoseconds.
CLOCK_SYNC_TOPIC = "isaaccapture.clock-sync"
_SYNC_REQUEST = struct.Struct("<cIq")
_SYNC_REPLY = struct.Struct("<cIqqq")


def build_sync_reply(request: bytes, t2: int, t3: int) -> bytes | None:
    """The reply to a well-formed clock-sync request, or None for anything else."""
    if len(request) != _SYNC_REQUEST.size:
        return None
    kind, sequence, t1 = _SYNC_REQUEST.unpack(request)
    if kind != b"Q":
        return None
    return _SYNC_REPLY.pack(b"R", sequence, t1, t2, t3)


async def answer_clock_sync(room: rtc.Room, identity: str, request: bytes, t2: int, clock: ClockModel) -> None:
    """Answers one clock-sync request, stamping t3 just before sending; anything malformed is ignored."""
    # Stamped here, after t2, and as late as possible: the PC removes (t3 - t2) from its round trip.
    reply = build_sync_reply(request, t2, clock.orin_monotonic_ns())
    if reply is None:
        return
    await room.local_participant.publish_data(
        reply, reliable=False, destination_identities=[identity], topic=CLOCK_SYNC_TOPIC
    )


def build_metadata(frame_id: int, capture_counter_ns: int, timestamp_us: int) -> rtc.FrameMetadata:
    """Builds the per-frame metadata exactly as the native sender does.

    ``user_data`` is the capture counter as 8 little-endian bytes (``frame_id`` and ``user_timestamp`` are plain
    integers).
    """
    return rtc.FrameMetadata(
        user_timestamp=timestamp_us,
        frame_id=frame_id,
        user_data=capture_counter_ns.to_bytes(8, "little"),
    )


def make_pattern(width: int, height: int, sensor: int, frame_id: int) -> np.ndarray:
    """Returns an RGBA frame: a per-sensor colour ramp, a moving bar, and the frame id as bit blocks on top.

    The bit strip lets a human or a script read the frame id off the video; the moving bar makes freezes visible.
    """
    frame = np.empty((height, width, 4), np.uint8)
    # A horizontal ramp tinted by sensor so different cameras are told apart at a glance.
    ramp = np.linspace(40, 200, width, dtype=np.uint8)
    frame[..., 0] = ramp
    frame[..., 1] = (sensor * 70) % 200 + 40
    frame[..., 2] = ramp[::-1]
    frame[..., 3] = 255
    # The bar advances every frame, so consecutive frames always differ.
    bar_x = (frame_id * 8) % width
    frame[:, bar_x : bar_x + 8, :3] = 255
    # Frame id bits, most significant first, as white/black blocks across the top.
    block = width // _COUNTER_BITS
    strip = max(1, min(height, height // 8))
    for i in range(_COUNTER_BITS):
        bit = (frame_id >> (_COUNTER_BITS - 1 - i)) & 1
        frame[:strip, i * block : (i + 1) * block, :3] = 255 if bit else 0
    return frame


# --- Publishing ---------------------------------------------------------------------------------------------


@dataclass
class Counters:
    """Frames sent per track name, for the stats line."""

    sent: dict[str, int]


async def publish_stream(
    source: rtc.VideoSource,
    spec: StreamSpec,
    args: argparse.Namespace,
    stop: asyncio.Event,
    counters: Counters,
    capture_clock: Callable[[], int] = time.monotonic_ns,
    clock: ClockModel | None = None,
) -> None:
    """Sends frames for one track at ``--fps`` until ``stop`` is set.

    Deadlines are start + n*period rather than sleep(period), so encode time and loop jitter do not accumulate
    into drift. After a stall longer than a period the schedule restarts from now instead of bursting to catch up.
    ``clock`` is the emulated Orin's clock, shared by every track of one publisher so their stamps drift together.
    """
    period = 1.0 / args.fps
    clock = clock or ClockModel.from_args(args)
    loop = asyncio.get_running_loop()
    deadline = loop.time()
    frame_id = 0
    while not stop.is_set():
        frame_id += 1
        # The counter and a capture-time stamp are taken before the frame is synthesized, as a camera stamps at
        # capture; a processing-time stamp is taken once the frame is ready, as the real sender's is.
        capture_ns = capture_clock()
        at_capture = args.timestamp_basis in _CAPTURE_BASES
        timestamp_us = (
            user_timestamp_us(args.timestamp_basis) + clock.adjustment_us(args.timestamp_basis, at_capture=True)
            if at_capture
            else 0
        )
        pixels = make_pattern(args.width, args.height, spec.sensor, frame_id)
        frame = rtc.VideoFrame(args.width, args.height, rtc.VideoBufferType.RGBA, pixels.tobytes())
        if not at_capture:
            timestamp_us = user_timestamp_us(args.timestamp_basis) + clock.adjustment_us(
                args.timestamp_basis, at_capture=False
            )
        source.capture_frame(
            frame,
            timestamp_us=timestamp_us,
            metadata=build_metadata(frame_id, capture_ns, timestamp_us),
        )
        counters.sent[spec.name] = frame_id
        # Wait for the next slot, but wake at once on shutdown.
        deadline += period
        delay = deadline - loop.time()
        if delay < -period:
            deadline = loop.time()
            delay = 0.0
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(stop.wait(), timeout=max(delay, 0.0))


async def report_stats(stop: asyncio.Event, counters: Counters) -> None:
    """Prints frames sent per track every ``STATS_INTERVAL_S`` until ``stop`` is set."""
    while not stop.is_set():
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(stop.wait(), timeout=STATS_INTERVAL_S)
        # Counters are cumulative, so a stalled track is visible as a number that stops moving.
        print("sent: " + ", ".join(f"{n}={c}" for n, c in counters.sent.items()), flush=True)


async def run(args: argparse.Namespace, stop: asyncio.Event) -> None:
    """Joins the room, publishes every stream until ``stop`` is set, then unpublishes and disconnects."""
    room = rtc.Room()
    await room.connect(args.livekit_url, args.token)
    counters = Counters({s.name: 0 for s in args.streams})
    sources: list[rtc.VideoSource] = []
    publications = []
    tasks: list[asyncio.Task] = []
    try:
        # One emulated Orin clock for every track, so their stamps share one offset and drift origin.
        clock = ClockModel.from_args(args)
        replies: set[asyncio.Task] = set()

        def on_data(packet: rtc.DataPacket) -> None:
            # t2 is read first, before anything else runs, so it is as close to the receipt as software allows.
            t2 = clock.orin_monotonic_ns()
            if packet.topic != CLOCK_SYNC_TOPIC or packet.participant is None:
                return
            task = asyncio.get_running_loop().create_task(
                answer_clock_sync(room, packet.participant.identity, packet.data, t2, clock)
            )
            replies.add(task)
            task.add_done_callback(replies.discard)

        room.on("data_received", on_data)
        # One source and one published track per stream, named as the PC side expects.
        for spec in args.streams:
            source = rtc.VideoSource(args.width, args.height)
            sources.append(source)
            track = rtc.LocalVideoTrack.create_video_track(spec.name, source)
            # As the native sender does, request all three per-frame metadata fields, or none reach the receiver.
            options = rtc.TrackPublishOptions(
                source=rtc.TrackSource.SOURCE_CAMERA,
                frame_metadata_features=[
                    rtc.FrameMetadataFeature.FMF_USER_TIMESTAMP,
                    rtc.FrameMetadataFeature.FMF_FRAME_ID,
                    rtc.FrameMetadataFeature.FMF_USER_DATA,
                ],
            )
            publications.append(await room.local_participant.publish_track(track, options))
            tasks.append(asyncio.create_task(publish_stream(source, spec, args, stop, counters, clock=clock)))
        tasks.append(asyncio.create_task(report_stats(stop, counters)))
        # A failed task would otherwise be noticed only at shutdown; stop everything as soon as one ends.
        for task in tasks:
            task.add_done_callback(lambda _t: stop.set())
        await stop.wait()
    finally:
        # Cleanup runs on normal stop and on error, and must not leave the room half-joined.
        stop.set()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for publication in publications:
            await room.local_participant.unpublish_track(publication.sid)
        for source in sources:
            await source.aclose()
        await room.disconnect()
        for result in results:
            if isinstance(result, Exception):
                raise result


def print_banner(args: argparse.Namespace) -> None:
    """Prints what is being emulated, without the token."""
    streams = ", ".join(f"sensor {s.sensor} -> {s.name}" for s in args.streams)
    print(
        f"Orin emulator: {args.livekit_url}  {args.width}x{args.height}@{args.fps:g}  "
        f"timestamp basis {args.timestamp_basis}\n  streams: {streams}\n  software encoder; Ctrl-C to stop",
        flush=True,
    )


async def amain(args: argparse.Namespace) -> None:
    """Runs the emulator with SIGINT/SIGTERM wired to a clean shutdown."""
    stop = asyncio.Event()
    # Signals set the event so run() can unpublish and disconnect instead of being killed mid-session.
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)
    print_banner(args)
    await run(args, stop)


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point."""
    asyncio.run(amain(parse_args(argv)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
