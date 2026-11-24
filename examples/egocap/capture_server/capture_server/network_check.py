# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""The server's own test of whether its LiveKit cameras deliver well enough to record.

The server joins the room as an ordinary subscriber, so a pass reflects real frames crossing the whole path from
the sensor to a decoder in this process. A ping, a published track or a healthy control connection cannot pass it.
Rates are unique usable frames over the measured local time, never averaged frame intervals, and arrival silence at
the start and end of the window counts: a burst that follows a stall would otherwise look healthy.

The result describes the conditions that were observed; it does not promise they will last. It is reported to
clients in the session status (``StatusSnapshot.network_check``) and ages out to STALE. The limits are proposals
that no healthy saved take has validated yet, so a station opts in by naming them (``livekit.network_check`` in the
``--livekit-streams`` file); without that this check does not run and nothing is reported or gated.
"""

from __future__ import annotations

import asyncio
import hashlib
import ipaddress
import json
import logging
import math
import statistics
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, fields, replace
from typing import Any
from urllib.parse import urlparse

from . import protocol as p
from .frame_timing import frame_age_is_plausible, uncertainty_budget_ns
from .livekit_receiver import LiveKitFrameReceiver

logger = logging.getLogger("isaaccapture.capture_server.network_check")

#: Frame-rate fraction below which a one-second window counts as low: the recorder's own average rejection floor.
LOW_WINDOW_FRACTION = 0.86
#: Longest startup allowance plus window the policy may name, so a check always ends and cancellation is prompt.
MAX_TOTAL_S = 300.0
#: How long leaving the room may take before the check moves on and lets the leave finish in the background.
_LEAVE_TIMEOUT_S = 10.0

#: The hops a pass exercised, and the ones it did not, so a reader cannot take it for more than it is.
LEGS_TESTED = (
    "sensor capture and sender encode",
    "camera network to the PC",
    "receiver decode",
    "bridge encode",
    "LiveKit server",
    "decode in the capture server",
)
LEGS_NOT_TESTED = (
    "LiveKit delivery to the recording host",
    "recorder CPU, decode load and disk throughput",
    "the browser preview",
    "delivery after the check ended: freshness is not monitored, so a pass is only a short-lived sample",
)


def _finite_number(value: Any, name: str) -> float:
    """``value`` as a finite number; a bool, text, NaN or infinity is a configuration error."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"network_check.{name} must be a finite number")
    return float(value)


@dataclass(frozen=True)
class NetworkCheckPolicy:
    """What delivery each camera must show. Station policy; every limit is an unvalidated starting value."""

    #: Time allowed to join the room and receive a first frame and a usable clock mapping from every camera.
    startup_s: float = 15.0
    #: Length of the measured window.
    window_s: float = 30.0
    #: Fewest usable frames, as a fraction of the configured rate over the window.
    min_delivery_fraction: float = LOW_WINDOW_FRACTION
    #: Longest gap allowed between consecutive capture times of usable frames.
    max_capture_gap_s: float = 0.25
    #: Longest silence allowed in arrival, between frames and at the start and end of the window.
    max_arrival_gap_s: float = 0.25
    #: One-second windows allowed below the low-rate floor; a window with no frame at all is never allowed.
    max_low_windows: int = 2
    #: Most the frame age may grow over the window, comparing its last third with its first third.
    max_age_growth_s: float = 0.25
    #: How long a pass counts before the status turns STALE and a new check is needed.
    valid_for_s: float = 120.0

    def __post_init__(self) -> None:
        """Reject a policy that could never be measured, never end, or would admit anything."""
        for name in (
            "startup_s",
            "window_s",
            "max_capture_gap_s",
            "max_arrival_gap_s",
            "max_age_growth_s",
            "valid_for_s",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"network_check.{name} must be a finite number")
            if not value > 0:
                raise ValueError(f"network_check.{name} must be positive")
        fraction = self.min_delivery_fraction
        if isinstance(fraction, bool) or not isinstance(fraction, (int, float)) or not 0 < fraction <= 1:
            raise ValueError("network_check.min_delivery_fraction must be a number in (0, 1]")
        if type(self.max_low_windows) is not int or self.max_low_windows < 0:
            raise ValueError("network_check.max_low_windows must be a nonnegative integer")
        # The window is measured in whole nanoseconds; one that rounds to none would divide every rate by zero.
        if int(self.window_s * 1e9) < 1:
            raise ValueError("network_check.window_s must be at least one nanosecond")
        if self.startup_s + self.window_s > MAX_TOTAL_S:
            raise ValueError(f"network_check.startup_s plus window_s must not exceed {MAX_TOTAL_S:g} s")

    @classmethod
    def from_mapping(cls, raw: Any) -> NetworkCheckPolicy:
        """The policy a station file names; an unknown key, a non-number or a fractional count is an error."""
        if not isinstance(raw, Mapping):
            raise ValueError("network_check must be a mapping")
        known = {f.name for f in fields(cls)}
        unknown = set(raw) - known
        if unknown:
            # Keys from YAML can be any type; sorting their text keeps the error a ValueError, never a TypeError.
            raise ValueError(f"network_check has unknown key(s): {sorted(map(str, unknown))}")
        values: dict[str, Any] = {}
        for name, value in raw.items():
            if name == "max_low_windows":
                if type(value) is not int:
                    raise ValueError("network_check.max_low_windows must be an integer")
                values[name] = value
            else:
                values[name] = _finite_number(value, name)
        return cls(**values)


def is_loopback_url(url: str) -> bool:
    """Whether ``url`` names this machine: ``localhost`` or a loopback IP literal."""
    host = urlparse(url).hostname or ""
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def require_secure_livekit_urls(streams: Sequence[p.LiveKitCameraStream]) -> None:
    """The check sends a viewer token to the room's URL, so only a loopback host may be reached without TLS."""
    for url in sorted({s.url for s in streams}):
        if urlparse(url).scheme == "ws" and not is_loopback_url(url):
            raise ValueError(f"network_check needs a wss:// LiveKit URL for a non-loopback host, got {url!r}")


def streams_generation(streams: Sequence[p.LiveKitCameraStream]) -> str:
    """A name for the camera identities, publication and configuration a result was measured on.

    A consumer compares this with the streams it is about to use: a different value means the result is about
    different cameras, geometry or rates and must not be relied on.
    """
    identity = sorted(
        (
            s.stream_id,
            s.name,
            s.url,
            s.room,
            s.publisher,
            s.track,
            int(s.timestamp_basis),
            s.width,
            s.height,
            s.fps,
        )
        for s in streams
    )
    return hashlib.sha256(json.dumps(identity, default=str).encode()).hexdigest()[:16]


def _endpoint(streams: Sequence[p.LiveKitCameraStream]) -> str:
    """The media endpoint a result describes."""
    first = streams[0]
    return f"LiveKit {first.url}, room {first.room}, publisher {first.publisher}"


@dataclass
class StreamWindow:
    """One camera's deliveries inside the measured window, judged against the policy.

    The window is fixed by the caller (``start_ns`` to ``end_ns`` on the local monotonic clock), shared by every
    camera, so a camera that was silent at either end is judged against the same instants as the others. A frame
    received outside it, such as queued backlog from before it began, is not part of it.
    """

    stream: p.LiveKitCameraStream
    policy: NetworkCheckPolicy
    start_ns: int
    end_ns: int
    #: Receiver counters when the window began, so only what happened inside it is reported.
    overflow_at_start: int = 0
    publication_changes_at_start: int = 0
    missing_at_start: int = 0
    unique: int = 0
    valid: int = 0
    missing_timestamp: int = 0
    mapping_problems: int = 0
    implausible_age: int = 0
    wrong_size: int = 0
    lost: bool = False
    overflow: int = 0
    publication_changes: int = 0
    missing_frames: int | None = None
    _saw_frame_ids: bool = field(default=False, repr=False)
    _last_id: int | None = field(default=None, repr=False)
    _receive_ns: list[int] = field(default_factory=list, repr=False)
    _capture_s: list[float] = field(default_factory=list, repr=False)
    _ages_s: list[float] = field(default_factory=list, repr=False)

    @property
    def elapsed_s(self) -> float:
        """The measured window's length, which every rate is taken over."""
        return (self.end_ns - self.start_ns) / 1e9

    def add_frame(
        self,
        *,
        frame_id: int | None,
        capture_s: float | None,
        receive_ns: int,
        age_s: float | None,
        mapping_ok: bool,
        size_ok: bool = True,
    ) -> None:
        """Record one received frame. ``capture_s`` and ``age_s`` are None when the frame has no usable timing."""
        if not self.start_ns <= receive_ns <= self.end_ns:
            return
        # A repeated or backwards id is a restarted or duplicated frame, not a delivery.
        if frame_id is not None:
            self._saw_frame_ids = True
            if self._last_id is not None and frame_id <= self._last_id:
                return
            self._last_id = frame_id
        self.unique += 1
        if age_s is not None:
            self._ages_s.append(age_s)
        if not size_ok:
            # A camera that changes size mid-window is not the stream that was offered.
            self.wrong_size += 1
        elif capture_s is None:
            self.missing_timestamp += 1
        elif not mapping_ok:
            self.mapping_problems += 1
        elif age_s is None or not frame_age_is_plausible(age_s):
            # The runtime consumer rejects these frames; counting them would pass what it cannot use.
            self.implausible_age += 1
        else:
            self.valid += 1
            self._receive_ns.append(receive_ns)
            self._capture_s.append(capture_s)

    def finish(self, counters: Any) -> None:
        """Take the receiver's loss counters for the window once it has ended."""
        self.overflow = counters.queue_overflow - self.overflow_at_start
        self.publication_changes = counters.publication_changes - self.publication_changes_at_start
        # Frame ids that never arrived are only knowable when the publisher numbers its frames.
        self.missing_frames = counters.missing_frames - self.missing_at_start if self._saw_frame_ids else None

    def _arrival_gaps_s(self) -> tuple[float, float, float]:
        """The longest gap between usable arrivals, and the silence at the start and at the end of the window."""
        if not self._receive_ns:
            # Nothing usable arrived: the whole window was silence at both ends.
            return 0.0, self.elapsed_s, self.elapsed_s
        times = self._receive_ns
        between = max((b - a) / 1e9 for a, b in zip(times, times[1:], strict=False)) if len(times) > 1 else 0.0
        return between, (times[0] - self.start_ns) / 1e9, (self.end_ns - times[-1]) / 1e9

    def _second_counts(self) -> list[int]:
        """Usable frames per whole second counted from the window's start; a partial last second is ignored."""
        counts = [0] * int(self.elapsed_s)
        for t in self._receive_ns:
            index = int((t - self.start_ns) / 1e9)
            if index < len(counts):
                counts[index] += 1
        return counts

    def _age_summary_ms(self) -> tuple[float | None, float | None, float | None]:
        """Median age, largest age and how much the age grew from the first third of the window to the last."""
        ages = self._ages_s
        if not ages:
            return None, None, None
        third = max(1, len(ages) // 3)
        growth = statistics.median(ages[-third:]) - statistics.median(ages[:third])
        return statistics.median(ages) * 1e3, max(ages) * 1e3, growth * 1e3

    def result(self) -> p.NetworkCheckCamera:
        """The camera's measurements and, in plain words, every way its delivery fell short of the policy."""
        policy, elapsed = self.policy, self.elapsed_s
        fps = float(self.stream.fps or 0)
        between, start_silence, end_silence = self._arrival_gaps_s()
        capture = self._capture_s
        capture_gap = max((b - a for a, b in zip(capture, capture[1:], strict=False)), default=0.0)
        median_age, max_age, growth = self._age_summary_ms()
        reasons: list[str] = []
        if self.lost:
            reasons.append("its LiveKit track was lost")
        if self.overflow:
            reasons.append(f"the server's receiver fell behind and dropped {self.overflow} frame(s)")
        if self.publication_changes:
            reasons.append("its publication was replaced")
        if self.missing_timestamp:
            reasons.append(f"{self.missing_timestamp} frame(s) arrived without a capture time")
        if self.wrong_size:
            reasons.append(
                f"{self.wrong_size} frame(s) were not the offered {self.stream.width}x{self.stream.height}"
            )
        if self.mapping_problems:
            reasons.append(f"the clock mapping was missing or too uncertain for {self.mapping_problems} frame(s)")
        if self.implausible_age:
            reasons.append(f"{self.implausible_age} frame(s) had an implausible age (clocks disagree)")
        if self.valid < policy.min_delivery_fraction * fps * elapsed:
            reasons.append(f"{self.valid / elapsed:.1f} usable frames/s arrived against {fps:g} configured")
        if capture_gap > policy.max_capture_gap_s:
            reasons.append(f"a {capture_gap * 1e3:.0f} ms gap between capture times")
        if between > policy.max_arrival_gap_s:
            reasons.append(f"a {between * 1e3:.0f} ms silence between arrivals")
        if start_silence > policy.max_arrival_gap_s:
            reasons.append(f"nothing arrived for the first {start_silence * 1e3:.0f} ms of the window")
        if end_silence > policy.max_arrival_gap_s:
            reasons.append(f"nothing arrived for the last {end_silence * 1e3:.0f} ms of the window")
        counts = self._second_counts()
        empty = sum(1 for c in counts if c == 0)
        low = sum(1 for c in counts if c < LOW_WINDOW_FRACTION * fps)
        if empty:
            reasons.append(f"No usable frames received in {empty} of {len(counts)} one-second samples")
        if low > policy.max_low_windows:
            reasons.append(
                f"Low frame count in {low} of {len(counts)} one-second samples: "
                f"fewer than {math.ceil(LOW_WINDOW_FRACTION * fps)} usable frames per sample "
                f"(lowest count: {min(counts)} frames)"
            )
        if growth is not None and growth / 1e3 > policy.max_age_growth_s:
            reasons.append(f"frame age grew by {growth / 1e3:.2f} s over the window")
        return p.NetworkCheckCamera(
            stream_id=self.stream.stream_id,
            name=self.stream.name,
            passed=not reasons,
            expected_fps=fps,
            usable_frames=self.valid,
            delivered_fps=self.valid / elapsed,
            missing_frames=self.missing_frames,
            max_capture_gap_ms=capture_gap * 1e3 if capture else None,
            max_arrival_gap_ms=between * 1e3 if self._receive_ns else None,
            start_silence_ms=start_silence * 1e3,
            end_silence_ms=end_silence * 1e3,
            age_median_ms=median_age,
            age_max_ms=max_age,
            age_growth_ms=growth,
            reasons=tuple(reasons),
        )


class _CheckFailed(Exception):
    """The check ended before or during measurement with a reason a person can act on."""

    def __init__(self, reason: str, cameras: tuple[p.NetworkCheckCamera, ...] = ()) -> None:
        super().__init__(reason)
        self.reason = reason
        self.cameras = cameras


class NetworkCheckService:
    """Owns the check: its lifecycle, deadline, cancellation, teardown and the status it reports.

    One check runs at a time and joins the room only for its duration, so nothing keeps decoding the cameras
    afterwards. A pass turns STALE once ``valid_for_s`` has elapsed, judged here so every consumer agrees.
    """

    def __init__(
        self,
        streams: Sequence[p.LiveKitCameraStream],
        policy: NetworkCheckPolicy,
        make_token: Callable[[str], str],
        *,
        receiver_factory: Callable[..., LiveKitFrameReceiver] = LiveKitFrameReceiver,
        monotonic_ns: Callable[[], int] = time.monotonic_ns,
        unix_ns: Callable[[], int] = time.time_ns,
    ) -> None:
        if not streams:
            raise ValueError("the network check needs at least one LiveKit camera stream")
        if any(None in (s.width, s.height, s.fps) for s in streams):
            raise ValueError("the network check needs every LiveKit stream's width, height and fps")
        self._streams = tuple(streams)
        self._policy = policy
        self._make_token = make_token
        self._receiver_factory = receiver_factory
        self._now_ns = monotonic_ns
        self._unix_ns = unix_ns
        self._generation = streams_generation(self._streams)
        self._endpoint = _endpoint(self._streams)
        self._state = p.NetworkCheckState.NOT_TESTED
        self._phase: str | None = None
        self._phase_started_ns = 0
        self._reason: str | None = None
        self._cameras: tuple[p.NetworkCheckCamera, ...] = ()
        self._completed_ns: int | None = None
        self._completed_unix_ns: int | None = None
        self._task: asyncio.Task[None] | None = None
        self._retiring: asyncio.Task[None] | None = None
        # Advances whenever a session begins or ends, so a teardown that awaited can tell a newer check is running.
        self._generation = 0
        # The capture session this verdict belongs to; None between sessions, when nothing is reported as passed.
        self._session_id: str | None = None
        # Joins that outlived their deadline; held so they are not garbage-collected mid-flight.
        self._background: set[asyncio.Future[Any]] = set()

    @property
    def generation(self) -> str:
        """Names the cameras and configuration this service tests; see :func:`streams_generation`."""
        return self._generation

    @property
    def running(self) -> bool:
        """Whether a check is in progress."""
        return self._task is not None and not self._task.done()

    def snapshot(self) -> p.NetworkCheckStatus:
        """The status now. A pass older than the policy allows reads STALE, whichever consumer asks."""
        state = self._state
        if (
            state == p.NetworkCheckState.READY
            and self._now_ns() - (self._completed_ns or 0) > self._policy.valid_for_s * 1e9
        ):
            state = p.NetworkCheckState.STALE
        checking = state == p.NetworkCheckState.CHECKING
        cameras = self._cameras or tuple(
            p.NetworkCheckCamera(stream_id=s.stream_id, name=s.name, passed=None, expected_fps=float(s.fps or 0))
            for s in self._streams
        )
        return p.NetworkCheckStatus(
            state=state,
            session_id=self._session_id,
            generation=self._generation,
            endpoint=self._endpoint,
            legs_tested=LEGS_TESTED,
            legs_not_tested=LEGS_NOT_TESTED,
            window_s=self._policy.window_s,
            valid_for_s=self._policy.valid_for_s,
            phase=self._phase if checking else None,
            progress_s=int((self._now_ns() - self._phase_started_ns) / 1e9) if checking else None,
            completed_at_unix_ns=self._completed_unix_ns,
            reason=self._reason,
            cameras=cameras,
        )

    def begin_session(self, session_id: str) -> None:
        """A new capture session has started its sources: drop the previous verdict and check afresh.

        Runs in the background so the session stays responsive; the status reads CHECKING until it ends.
        """
        self._cancel_running()
        self._session_id = session_id
        self._generation += 1
        self._begin()

    def start(self) -> bool:
        """Run the check again in the current session; False when no session is active or one is running."""
        if self._session_id is None or self.running:
            return False
        self._begin()
        return True

    async def end_session(self, session_id: str) -> None:
        """The session ended: cancel its check and leave nothing reported as passed."""
        if self._session_id != session_id:
            return
        self._session_id = None
        self._generation += 1
        generation = self._generation
        await self.stop()
        # A newer session may have begun while this one awaited; its verdict is not this teardown's to reset.
        if self._generation == generation:
            self._state, self._reason, self._cameras = p.NetworkCheckState.NOT_TESTED, None, ()
            self._completed_ns = self._completed_unix_ns = None

    def _cancel_running(self) -> None:
        task, self._task = self._task, None
        if task is not None and not task.done():
            task.cancel()
            # The next check waits for this one to finish retiring its room before it may connect.
            self._retiring = task

    def _begin(self) -> None:
        """Invalidate the previous result at once, so an old verdict cannot show while this one runs."""
        self._state = p.NetworkCheckState.CHECKING
        self._reason = None
        self._cameras = ()
        self._completed_ns = self._completed_unix_ns = None
        self._set_phase("starting")
        self._task = asyncio.ensure_future(self._run())

    async def stop(self) -> None:
        """Cancel a running check and leave the room; no verdict is reported for a check that never finished."""
        generation = self._generation
        task, self._task = self._task, None
        if task is not None and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        if self._state == p.NetworkCheckState.CHECKING and self._generation == generation:
            self._state, self._phase = p.NetworkCheckState.NOT_TESTED, None
        # A leave that outlived its deadline gets the same bound here, so shutdown cannot wait on a stuck room.
        if self._background:
            await asyncio.wait(set(self._background), timeout=_LEAVE_TIMEOUT_S)

    def _set_phase(self, phase: str) -> None:
        self._phase = phase
        self._phase_started_ns = self._now_ns()

    def _finish(
        self, state: p.NetworkCheckState, reason: str | None, cameras: tuple[p.NetworkCheckCamera, ...]
    ) -> None:
        """Record the verdict and when it was reached."""
        self._state, self._reason, self._cameras, self._phase = state, reason, cameras, None
        self._completed_ns, self._completed_unix_ns = self._now_ns(), self._unix_ns()

    async def _run(self) -> None:
        """One whole check, ending in a verdict whatever goes wrong; only cancellation escapes."""
        receiver = join = None
        try:
            # Two observers must never overlap: wait for a previous check to retire, then for its room to close.
            if self._retiring is not None and not self._retiring.done():
                await asyncio.wait({self._retiring}, timeout=_LEAVE_TIMEOUT_S)
            if self._background:
                await asyncio.wait(set(self._background), timeout=_LEAVE_TIMEOUT_S)
                if self._background:
                    raise _CheckFailed("the previous check's connection to LiveKit has not closed yet")
            receiver = self._receiver_factory(
                self._streams, self._make_token(self._streams[0].room), clock_sync=True
            )
            # Owned here, not by the startup wait, so a cancelled or late join is retired exactly once below.
            join = asyncio.ensure_future(receiver.start())
            budget_ns = uncertainty_budget_ns()
            await self._startup(receiver, join, budget_ns)
            cameras = await self._measure(receiver, budget_ns)
        except asyncio.CancelledError:
            raise
        except _CheckFailed as exc:
            self._finish(p.NetworkCheckState.UNSUITABLE, exc.reason, exc.cameras)
        except Exception as exc:  # noqa: BLE001 -- the verdict must say the check could not run, not vanish
            logger.warning("network check could not run: %s", exc)
            self._finish(p.NetworkCheckState.UNSUITABLE, f"the check could not run: {exc}", ())
        else:
            failing = [c for c in cameras if not c.passed]
            if failing:
                reason = " | ".join(f"{c.name}: " + "; ".join(c.reasons) for c in failing)
                self._finish(
                    p.NetworkCheckState.UNSUITABLE,
                    f"the connection is unsuitable for recording ({reason})",
                    cameras,
                )
            else:
                self._finish(p.NetworkCheckState.READY, None, cameras)
        finally:
            if receiver is not None:
                await self._retire(receiver, join)

    async def _retire(self, receiver: LiveKitFrameReceiver, join: asyncio.Future[None] | None) -> None:
        """Leave the room once, within a deadline; a join still in flight is left as soon as it ends.

        Cancelling ``Room.connect`` can strand the SDK's room-event handshake, so an unfinished join keeps
        running and is the only thing that leaves, never the caller racing it.
        """
        if join is not None and not join.done():
            self._track(join)
            join.add_done_callback(lambda _done: self._track(asyncio.ensure_future(receiver.stop())))
            return
        leave = asyncio.ensure_future(receiver.stop())
        self._track(leave)
        await asyncio.wait({leave}, timeout=_LEAVE_TIMEOUT_S)

    def _track(self, task: asyncio.Future[Any]) -> None:
        """Keep a background join or leave referenced until it ends, so it is not collected mid-flight."""
        self._background.add(task)
        task.add_done_callback(self._background.discard)

    async def _startup(self, receiver: LiveKitFrameReceiver, join: asyncio.Future[None], budget_ns: int) -> None:
        """Wait for the join, then for a usable first frame from every camera and a clock mapping in budget."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self._policy.startup_s
        await asyncio.wait({join}, timeout=self._policy.startup_s)
        if not join.done():
            raise _CheckFailed(f"joining the LiveKit room took longer than {self._policy.startup_s:g} s")
        if join.exception() is not None:
            raise _CheckFailed(f"could not join the LiveKit room: {join.exception()}")
        problems = await asyncio.gather(*(self._first_frame(receiver, s, deadline) for s in self._streams))
        failed = tuple(
            p.NetworkCheckCamera(
                stream_id=s.stream_id,
                name=s.name,
                passed=False,
                expected_fps=float(s.fps or 0),
                reasons=(problem,),
            )
            for s, problem in zip(self._streams, problems, strict=True)
            if problem is not None
        )
        if failed:
            raise _CheckFailed("; ".join(f"{c.name}: {c.reasons[0]}" for c in failed), failed)
        while (problem := self._mapping_problem(receiver, budget_ns)) is not None:
            if loop.time() >= deadline:
                raise _CheckFailed(problem)
            await asyncio.sleep(0.05)

    async def _first_frame(
        self, receiver: LiveKitFrameReceiver, stream: p.LiveKitCameraStream, deadline: float
    ) -> str | None:
        """Why this camera has no usable first frame by the deadline, or None once it has one.

        An untimed frame is normal while the bridge's clock settles, so reading continues until a usable frame
        arrives, the track is lost or the startup allowance ends; the last problem seen is the one reported.
        """
        loop = asyncio.get_running_loop()
        problem = f"no frame arrived within the {self._policy.startup_s:g} s startup allowance"
        while (remaining := deadline - loop.time()) > 0:
            try:
                frame = await receiver.next_frame_or_lost(stream.stream_id, timeout=remaining)
            except asyncio.TimeoutError:
                return problem
            if frame is None:
                return "its LiveKit track was lost before the check started"
            if frame.user_timestamp_us is None:
                problem = "no frame carried a capture time (the bridge publishes none while time sync is not up)"
            elif frame.image.shape[:2] != (stream.height, stream.width):
                got = f"{frame.image.shape[1]}x{frame.image.shape[0]}"
                problem = f"frames are {got}, expected {stream.width}x{stream.height}"
            else:
                return None
        return problem

    @staticmethod
    def _mapping_problem(receiver: LiveKitFrameReceiver, budget_ns: int) -> str | None:
        """Why the publisher's clock mapping cannot place frame times right now, or None when it can."""
        clock_sync = receiver.clock_sync
        mapping = None if clock_sync is None else clock_sync.mapping()
        if clock_sync is None or mapping is None:
            return "the publisher did not answer clock-sync requests, so its frame times cannot be placed"
        if clock_sync.discontinuity:
            return "the publisher's clock mapping jumped"
        if mapping.uncertainty_ns > budget_ns:
            return (
                f"the clock mapping is uncertain by {mapping.uncertainty_ns / 1e6:.2f} ms, "
                f"over the {budget_ns / 1e6:.2f} ms budget"
            )
        return None

    async def _measure(self, receiver: LiveKitFrameReceiver, budget_ns: int) -> tuple[p.NetworkCheckCamera, ...]:
        """Observe every camera concurrently over one shared window and judge each."""
        # Backlog from the startup phase is not part of the window.
        receiver.clear_queued()
        self._set_phase("measuring")
        start_ns = self._now_ns()
        end_ns = start_ns + int(self._policy.window_s * 1e9)
        # The result is about the publication measured when the window began, not whichever one is current later.
        start_tracks = receiver.track_ids()
        windows = {
            s.stream_id: StreamWindow(
                s,
                self._policy,
                start_ns,
                end_ns,
                overflow_at_start=receiver.counters[s.stream_id].queue_overflow,
                publication_changes_at_start=receiver.counters[s.stream_id].publication_changes,
                missing_at_start=receiver.counters[s.stream_id].missing_frames,
            )
            for s in self._streams
        }
        await asyncio.gather(*(self._observe(receiver, windows[s.stream_id], budget_ns) for s in self._streams))
        clock_sync = receiver.clock_sync
        end_tracks = receiver.track_ids()
        for stream_id, window in windows.items():
            # A camera whose track is gone or replaced by the end is not the one whose frames were measured; the
            # receiver's own counter misses a replacement that follows a detach.
            if end_tracks.get(stream_id) != start_tracks.get(stream_id):
                window.publication_changes += 1
            # A mapping that jumped during the window is a fault of every camera placed on it.
            if clock_sync is not None and clock_sync.discontinuity:
                window.mapping_problems += 1
        return tuple(replace(w.result(), track_id=start_tracks.get(w.stream.stream_id)) for w in windows.values())

    async def _observe(self, receiver: LiveKitFrameReceiver, window: StreamWindow, budget_ns: int) -> None:
        """Feed one camera's frames to its window until the window's end instant or its track's loss."""
        stream_id = window.stream.stream_id
        try:
            await self._feed_window(receiver, window, budget_ns)
        finally:
            # Read when this camera's window ends: what the receiver loses afterwards is not part of it.
            window.finish(receiver.counters[stream_id])

    async def _feed_window(self, receiver: LiveKitFrameReceiver, window: StreamWindow, budget_ns: int) -> None:
        """Read frames into the window until its end instant or its track's loss."""
        stream_id = window.stream.stream_id
        while (remaining := (window.end_ns - self._now_ns()) / 1e9) > 0:
            try:
                frame = await receiver.next_frame_or_lost(stream_id, timeout=remaining)
            except asyncio.TimeoutError:
                return
            if frame is None:
                window.lost = True
                return
            clock_sync = receiver.clock_sync
            mapping = None if clock_sync is None else clock_sync.mapping()
            capture_s = age_s = None
            if frame.user_timestamp_us is not None and mapping is not None:
                capture_s = mapping.to_local_ns(frame.user_timestamp_us * 1000) / 1e9
                age_s = frame.receive_time_ns / 1e9 - capture_s
            window.add_frame(
                frame_id=frame.frame_id,
                capture_s=capture_s,
                receive_ns=frame.receive_time_ns,
                age_s=age_s,
                mapping_ok=mapping is not None and mapping.uncertainty_ns <= budget_ns,
                size_ok=frame.image.shape[:2] == (window.stream.height, window.stream.width),
            )
