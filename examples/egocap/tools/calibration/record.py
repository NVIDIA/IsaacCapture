# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Record one calibration take from a running CaptureServer, from the terminal.

Tracking frames go through the same conversion and recording service as any Ego Capture recording, so the take
is an ordinary episode holding ``pico_controller`` and ``manus_glove``. No page, camera or LiveKit is involved.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

import calib_common as common
from websockets.exceptions import WebSocketException

from capture_server import protocol
from capture_server.client import (
    CaptureClient,
    CaptureFrame,
    CaptureStopped,
    ManusPluginConfig,
    RequestRejectedError,
)
from egocap.core.data_root import calibration_dir
from egocap_web.recording_adapter import (
    MANUS_GLOVE_SPEC,
    PICO_CONTROLLER_SPEC,
    RecordingAdapterError,
    convert_controller_aim_pose,
    convert_hand_pose,
    convert_sensor_pose,
    validate_tracking_descriptor,
)
from recording_service import (
    Disposition,
    FieldSpec,
    NumericSample,
    NumericSourceSpec,
    QcConfig,
    RecordingService,
    ServiceConfig,
    SubmissionStatus,
)

#: A device counts as live while its last row is younger than this.
LIVE_S = 1.0
_GLOVE, _CONTROLLER, _SENSORS = MANUS_GLOVE_SPEC.name, PICO_CONTROLLER_SPEC.name, "sensors"
#: What to tell the operator while something the take needs is not live.
_WAITING = {
    _GLOVE: "Waiting for the {side} glove. Turn it on and check it is connected.",
    _SENSORS: "Waiting for the {side} glove's fingertip sensors. " + common.SENSOR_HINT,
    _CONTROLLER: "Waiting for the {side} controller. Turn it on and keep it where the headset can see it.",
}


def _specs(side: str, controllers: bool) -> tuple[NumericSourceSpec, ...]:
    """The take's sources: the fields of the calibrated side only, named and typed as a full recording's.

    A full recording writes a row only while both hands, or the headset and both controllers, are valid. A take
    holds one side, so the other glove and controller can stay off.
    """
    glove = NumericSourceSpec(
        _GLOVE,
        (
            FieldSpec(f"{side}_hand_pose", "float64", (25, 4, 4)),
            FieldSpec(f"{side}_sensor_pose", "float64", (5, 4, 4)),
        ),
        MANUS_GLOVE_SPEC.frame_type,
    )
    controller = NumericSourceSpec(
        _CONTROLLER, (FieldSpec(f"{side}_pose", "float64", (4, 4)),), PICO_CONTROLLER_SPEC.frame_type
    )
    return (glove, controller) if controllers else (glove,)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """The arguments every calibration tool takes: which hand, where the take comes from, where results go."""
    parser.add_argument("--side", choices=("left", "right"), required=True, help="the hand to calibrate")
    parser.add_argument(
        "episode", type=Path, nargs="?", help="a recording to solve again; leave out to record a new take"
    )
    parser.add_argument("--capture-server", default="ws://127.0.0.1:8082", help="CaptureServer to record from")
    parser.add_argument(
        "--results",
        type=Path,
        default=calibration_dir() / common.RESULTS_NAME,
        help="file a passed result is saved in",
    )
    parser.add_argument("--takes-dir", type=Path, default=Path("calibration_takes"), help="where a take is kept")


class _Feed:
    """Turns tracking frames into the take's rows and remembers which devices are live."""

    def __init__(self, service: RecordingService, side: str, controllers: bool) -> None:
        """Feed ``service`` with the ``side`` glove and, with ``controllers``, that side's controller."""
        self._service = service
        self._side = side
        self._controllers = controllers
        self._seen: dict[str, float] = {}
        self._layout_checked = False
        self.episode_id: str | None = None
        self.problem: str | None = None
        #: Set once the take is over, so that our own stop is not reported as a failure.
        self.finished = False

    def on_frame(self, frame: CaptureFrame) -> None:
        """Convert one frame; submit its rows while an episode is open."""
        # The layout is checked once: a mismatch would decode every later frame wrongly.
        if not self._layout_checked:
            self._layout_checked = True
            try:
                validate_tracking_descriptor(frame.description)
            except RecordingAdapterError as exc:
                self.problem = f"CaptureServer's tracking layout is not the one this checkout records: {exc}"
        if self.problem is not None:
            return
        tracking = protocol.decode_tracking_frame(frame.payload)
        side, now, rows = self._side, time.monotonic(), {}
        # A row is written only for a valid hand or controller, never a placeholder.
        hand = convert_hand_pose(tracking, side)
        if hand is not None:
            rows[_GLOVE] = {f"{side}_hand_pose": hand, f"{side}_sensor_pose": convert_sensor_pose(tracking, side)}
            if tracking[f"manus_{side}.sensor_present"]:
                self._seen[_SENSORS] = now
        aim = convert_controller_aim_pose(tracking, side) if self._controllers else None
        if aim is not None:
            rows[_CONTROLLER] = {f"{side}_pose": aim}
        stamp, received = frame.capture_time_ns / 1e9, frame.receive_time_ns / 1e9
        for source, fields in rows.items():
            self._seen[source] = now
            if self.episode_id is not None:
                sample = NumericSample(source, stamp, received, fields, ("host_clock", frame.session_id))
                # A full queue would leave a hole in the take, which the solver cannot see.
                if self._service.submit_sample(self.episode_id, sample).status is not SubmissionStatus.ACCEPTED:
                    self.problem = "The recorder could not keep up and lost data. Record again."

    def on_stopped(self, stopped: CaptureStopped) -> None:
        """Remember why CaptureServer ended the session, unless the take had already finished."""
        if not self.finished:
            self.problem = f"CaptureServer stopped: {stopped.error_message or stopped.reason}"

    def missing(self, needed: tuple[str, ...]) -> list[str]:
        """What the take needs that is not live."""
        now = time.monotonic()
        return [name for name in needed if now - self._seen.get(name, -LIVE_S) >= LIVE_S]


async def _wait_until_live(feed: _Feed, needed: tuple[str, ...], side: str) -> None:
    """Return once every needed stream is live, telling the operator what is still missing as it changes."""
    told: list[str] = []
    while feed.problem is None:
        # Checked after a pause, so that the first frames have had time to arrive.
        await asyncio.sleep(LIVE_S / 2)
        missing = feed.missing(needed)
        if not missing:
            return
        if missing != told:
            print("\n".join(_WAITING[name].format(side=side) for name in missing), flush=True)
            told = missing


async def _enter(prompt: str) -> None:
    """Wait for the operator to press Enter.

    Read through the event loop, not a thread blocked in ``input``: that thread would keep Ctrl-C from ending
    the tool until Enter is pressed.
    """
    loop = asyncio.get_running_loop()
    pressed = loop.create_future()
    print(prompt, end="", flush=True)
    loop.add_reader(sys.stdin, lambda: pressed.done() or pressed.set_result(sys.stdin.readline()))
    try:
        await pressed
    finally:
        loop.remove_reader(sys.stdin)


async def record_take(url: str, takes_dir: Path, task: str, side: str, controllers: bool, procedure: str) -> Path:
    """Guide the operator through one take and return its kept episode directory.

    ``controllers`` says whether the take needs controller rows; ``procedure`` is shown before recording.
    """
    specs = _specs(side, controllers)
    sources = tuple(spec.name for spec in specs)
    service = RecordingService(ServiceConfig(takes_dir, specs, qc=QcConfig(required_sources=sources)))
    feed = _Feed(service, side, controllers)
    client = CaptureClient(url)
    await service.open_session()
    try:
        try:
            await client.start(
                # Every source, also for a glove-only take: a hands-only session delivers no glove data.
                sources=tuple(protocol.SourceType),
                plugins=(ManusPluginConfig(),),
                on_capture_frame=feed.on_frame,
                on_stopped=feed.on_stopped,
            )
        except (RequestRejectedError, OSError, WebSocketException, protocol.ProtocolError) as exc:
            # No server, a server that is serving another client, or a proxy in the way of a local address.
            raise SystemExit(
                f"Cannot reach CaptureServer at {url} ({exc or type(exc).__name__}). Check that it is running "
                "and that nothing else is using it. For a local address, unset the proxy variables."
            ) from exc
        print(procedure, flush=True)
        await _wait_until_live(feed, (*sources, _SENSORS), side)
        if feed.problem is None:
            await _enter("Ready. Press Enter to start recording. ")
            episode = await service.start_episode(task)
            feed.episode_id = episode.episode_id
            await _enter("Recording. Press Enter when you are done. ")
            feed.episode_id = None
            result = await service.stop_episode(episode.episode_id)
    finally:
        feed.finished = True
        await client.stop()
        await service.close()
    if feed.problem is not None:
        raise SystemExit(feed.problem)
    if result.disposition is not Disposition.KEPT:
        raise SystemExit(f"The take could not be saved: {'; '.join(result.issues)}")
    print(f"Recorded {result.duration_seconds:.0f} s to {result.final_path}")
    return result.final_path


def episode_for(args: argparse.Namespace, task: str, controllers: bool, procedure: str) -> Path:
    """The episode to solve: the one named on the command line, or a take recorded now."""
    if args.episode is not None:
        return common.resolve_episode(args.episode)
    return asyncio.run(record_take(args.capture_server, args.takes_dir, task, args.side, controllers, procedure))
