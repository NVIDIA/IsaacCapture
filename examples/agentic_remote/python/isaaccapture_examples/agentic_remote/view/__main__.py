# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""python -m isaaccapture_examples.agentic_remote.view --config PATH.

The live view: it attaches to the robot link service in the role "viewer",
serves a camera picture and the robot's status to a browser on this machine
(127.0.0.1 only) and tells the Thor, with VIEW_STATUS, how fresh the picture on
that browser's screen is. VIEW_STATUS is the only link message it sends.

The camera pictures come from the camera receiver through a camera frame bridge
of the view's own, so the view also runs while the agent program is down.
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import math
import os
import queue
import signal
import sys
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import yaml

from ..agent.camera.frame_source import (
    BridgeSource,
    CameraError,
    PairSource,
    RecordedPairSource,
)
from ..agent.config import ConfigError, expanded_path, user_path
from ..console_log import set_console_level
from ..link_service.local import LocalCarrier, connect_local
from ..protocol.carrier import receive_message, send
from ..protocol.local_messages import ROLE_VIEWER
from ..protocol.messages import ViewStatus
from .model import AGE_WHEN_NOT_LIVE_S, ViewModel
from .server import LOOPBACK, make_server

_log = logging.getLogger("isaaccapture.examples.agentic_remote.view")

EXIT_OK = 0
EXIT_FAILED = 1  # no robot link service, a second live view, or the port is taken
EXIT_NOT_RUN = 2  # the arguments or the file cannot be used

PROGRAM = "live view"
SECOND_VIEW = "another live view is attached"
LOG_LEVELS = ("debug", "info", "warning", "error")
SOURCE_LINK = "link"
SOURCE_NONE = "none"
SOURCE_RECORDED = "recorded:"
# A camera source that does not open is tried this often, this far apart.
OPEN_ATTEMPTS = 12
OPEN_RETRY_S = 5.0
# How long one request for a stereo pair may wait.
PAIR_WAIT_S = 1.0
ATTACH_TIMEOUT_S = 5.0
_RECEIVE_S = 0.2

Say = Callable[[str], None]


class ViewConfigError(ValueError):
    """A setting is missing, unknown, of the wrong kind or out of range."""


class Clock(Protocol):
    def monotonic(self) -> float: ...


class _MachineClock:
    def monotonic(self) -> float:
        return time.monotonic()


@dataclass(frozen=True)
class ViewSettings:
    """The settings of the live view, as view.yaml holds them."""

    socket_path: str | None = None  # None: the robot link service's default
    log_level: str = "info"
    state_dir: Path = Path(".state")  # the bridge's log goes into <state_dir>/view
    http_address: str = LOOPBACK
    http_port: int = 47620
    bridge_python: str = ""
    camera_pr: str = ""  # "" takes AGENTIC_REMOTE_CAMERA_PR
    receiver_socket: str = "/tmp/camera_receiver_avs1_cuda_ipc.sock"
    streams: tuple[str, str] = ("head.left", "head.right")
    max_skew_ms: float = 5.0
    width_px: int = 960
    fps: float = 5.0
    jpeg_quality: int = 80
    period_s: float = 0.2
    ack_timeout_s: float = 1.0
    state_max_age_s: float = 1.0


# Block -> its keys; "" is the top of the file.
_KEYS: dict[str, tuple[str, ...]] = {
    "": ("socket_path", "log_level", "state_dir"),
    "http": ("address", "port"),
    "camera": (
        "bridge_python",
        "camera_pr",
        "receiver_socket",
        "streams",
        "max_skew_ms",
    ),
    "picture": ("width_px", "fps", "jpeg_quality"),
    "status": ("period_s", "ack_timeout_s", "state_max_age_s"),
}
# Setting -> (lowest, highest); a number of a setting not named here is above 0.
_WHOLE = {"port": (0, 65535), "width_px": (64, 4096), "jpeg_quality": (1, 100)}
_AT_MOST = {
    "max_skew_ms": 50.0,
    "fps": 15.0,
    "period_s": 1.0,
    "ack_timeout_s": 10.0,
    "state_max_age_s": 10.0,
}
_PATHS = ("bridge_python", "camera_pr", "receiver_socket")


def load_view_settings(path: str | os.PathLike[str]) -> ViewSettings:
    """The settings in a YAML file; ViewConfigError names the one that is wrong.

    A setting the file leaves out keeps its first value; one it does not know
    is an error. A relative path is taken from the file's folder.
    """
    file = _user_path("--config", os.fspath(path))
    try:
        with open(file, encoding="utf-8") as stream:
            data = yaml.safe_load(stream)
    except (OSError, yaml.YAMLError) as error:
        raise ViewConfigError(f"cannot read {file}: {error}") from error
    return view_settings(data, file.resolve().parent)


def view_settings(data: object, folder: Path) -> ViewSettings:
    """The settings in a mapping as the YAML file holds it."""
    top = _block("the file", {} if data is None else data)
    blocks = {name: _block(name, top.pop(name, {})) for name in _KEYS if name}
    blocks[""] = top
    values: dict[str, Any] = {}
    for name, keys in _KEYS.items():
        block = blocks[name]
        where = f"{name}." if name else ""
        unknown = sorted(str(key) for key in block if key not in keys)
        if unknown:
            raise ViewConfigError(
                "unknown setting(s): " + ", ".join(where + key for key in unknown)
            )
        for key, value in block.items():
            values[_FIELDS.get((name, key), key)] = _checked(where + key, key, value)
    if values.get("socket_path"):
        values["socket_path"] = _path("socket_path", values["socket_path"], folder)
    else:
        values.pop("socket_path", None)
    if values.get("http_address", LOOPBACK) != LOOPBACK:
        raise ViewConfigError(
            f"http.address must be {LOOPBACK}, got {values['http_address']!r}: the "
            "page is for a browser on this machine"
        )
    for key in _PATHS:
        if values.get(key):
            values[key] = _path(f"camera.{key}", values[key], folder)
    state_dir = _path("state_dir", values.get("state_dir") or "../.state", folder)
    values["state_dir"] = Path(state_dir)
    return ViewSettings(**values)


_FIELDS = {("http", "address"): "http_address", ("http", "port"): "http_port"}


def _block(where: str, value: object) -> dict[Any, Any]:
    if not isinstance(value, dict):
        raise ViewConfigError(f"{where} must be a mapping, got {value!r}")
    return dict(value)


def _checked(where: str, key: str, value: Any) -> Any:
    if key == "streams":
        if (
            not isinstance(value, (list, tuple))
            or len(value) != 2
            or not all(isinstance(name, str) and name.strip() for name in value)
            or value[0] == value[1]
        ):
            raise ViewConfigError(f"{where} must be two different names, got {value!r}")
        return (value[0], value[1])
    if key == "log_level":
        if value not in LOG_LEVELS:
            raise ViewConfigError(
                f"{where} must be one of {', '.join(LOG_LEVELS)}, got {value!r}"
            )
        return value
    if key in _WHOLE:
        low, high = _WHOLE[key]
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise ViewConfigError(f"{where} must be a whole number, got {value!r}")
        if not low <= value <= high:
            raise ViewConfigError(f"{where} must be from {low} to {high}, got {value}")
        return int(value)
    if key in _AT_MOST:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ViewConfigError(f"{where} must be a number, got {value!r}")
        lowest_ok = value >= 0.0 if key == "max_skew_ms" else value > 0.0
        if not (math.isfinite(value) and lowest_ok and value <= _AT_MOST[key]):
            raise ViewConfigError(
                f"{where} must be above 0 and at most {_AT_MOST[key]:g}, got {value!r}"
            )
        return float(value)
    if not isinstance(value, str):
        raise ViewConfigError(f"{where} must be text, got {value!r}")
    return value


def _path(where: str, text: str, folder: Path) -> str:
    # The agent program's rule: its three ${NAME}, and no other "$".
    try:
        expanded = expanded_path(where, text)
    except ConfigError as problem:
        raise ViewConfigError(str(problem)) from None
    path = _user_path(where, expanded)
    return str(path if path.is_absolute() else (folder / path).resolve())


def _user_path(where: str, text: str) -> Path:
    """The agent program's user_path(), with this program's error."""
    try:
        return user_path(where, text)
    except ConfigError as problem:
        raise ViewConfigError(str(problem)) from None


def encode_picture(nv12: np.ndarray, width_px: int, quality: int) -> bytes:
    """One eye's NV12 camera frame as a JPEG at most width_px wide."""
    import cv2

    rows, width = nv12.shape
    if rows % 3 or width % 2:
        raise ValueError(f"not an NV12 image: shape {nv12.shape}")
    bgr = cv2.cvtColor(np.ascontiguousarray(nv12), cv2.COLOR_YUV2BGR_NV12)
    height = rows * 2 // 3
    if width > width_px:
        size = (width_px, max(1, round(height * width_px / width)))
        bgr = cv2.resize(bgr, size, interpolation=cv2.INTER_AREA)
    done, jpeg = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not done:
        raise ValueError("OpenCV could not pack the JPEG")
    return jpeg.tobytes()


class PictureLoop:
    """Reads stereo pairs from a source and hands the left eye to the model.

    pause(s) waits s seconds and returns True when the loop is to end. A source
    that does not open, or that ends before it gave one picture, is tried again
    after retry_s; after `attempts` such failures in a row the loop ends and
    the view runs without pictures.
    """

    def __init__(
        self,
        model: ViewModel,
        source: PairSource,
        *,
        invented: bool,
        settings: ViewSettings,
        clock: Clock,
        pause: Callable[[float], bool],
        say: Say,
        attempts: int | None = None,
        retry_s: float | None = None,
    ) -> None:
        self._model = model
        self._source = source
        self._invented = invented
        self._settings = settings
        self._clock = clock
        self._pause = pause
        self._say = say
        self._attempts = OPEN_ATTEMPTS if attempts is None else attempts
        self._retry_s = OPEN_RETRY_S if retry_s is None else retry_s
        self._seq = 0
        self._ended = False

    def run(self) -> None:
        failures = 0
        while not self._ended:
            before = self._seq
            try:
                self._source.open()
            except (CameraError, OSError) as problem:
                why = f"the camera source did not open: {problem}"
            else:
                self._tell("")
                try:
                    why = self._show_pairs()
                finally:
                    self._source.close()
            if self._ended or self._pause(0.0):
                return
            failures = 1 if self._seq > before else failures + 1
            if failures >= self._attempts:
                self._tell(
                    f"no camera pictures: {why} ({failures} attempts, "
                    f"{self._retry_s:g} s apart); the view shows the status only"
                )
                return
            self._tell(f"{why}; trying again in {self._retry_s:g} s")
            if self._pause(self._retry_s):
                return

    def _show_pairs(self) -> str:
        """Show pairs until the source fails; why it did ("" when told to end)."""
        settings = self._settings
        period_s = 1.0 / settings.fps
        while True:
            started = self._clock.monotonic()
            try:
                pair = self._source.next_pair(PAIR_WAIT_S)
                if pair is not None:
                    jpeg = encode_picture(
                        pair.left, settings.width_px, settings.jpeg_quality
                    )
            except CameraError as problem:
                return f"the camera source ended: {problem}"
            except Exception as problem:
                return f"a camera frame could not be made a picture: {problem}"
            if pair is not None:
                self._seq += 1
                invented = self._invented or bool(pair.check.get("invented_times"))
                self._model.put_picture(
                    self._seq,
                    jpeg,
                    min(pair.sof_wall_ns) / 1e9,
                    pair.receiver_rx_mono_s,
                    invented,
                )
            left = period_s - (self._clock.monotonic() - started)
            if self._pause(max(0.0, left)):
                self._ended = True
                return ""

    def _tell(self, note: str) -> None:
        self._model.set_camera_note(note)
        if note:
            self._say(f"{PROGRAM}: {note}")


class ViewProgram:
    """The live view's threads on an attached carrier, started and closed as one."""

    def __init__(
        self,
        settings: ViewSettings,
        carrier: LocalCarrier,
        model: ViewModel,
        source: PairSource | None,
        invented: bool,
        clock: Clock,
        say: Say,
    ) -> None:
        self.settings = settings
        self.carrier = carrier
        self.model = model
        self._source = source
        self._clock = clock
        self._end = threading.Event()
        self._threads: list[threading.Thread] = []
        self._pictures = (
            None
            if source is None
            else PictureLoop(
                model,
                source,
                invented=invented,
                settings=settings,
                clock=clock,
                pause=self._end.wait,
                say=say,
            )
        )
        # OSError when the port cannot be had.
        self._server = make_server(
            model,
            settings.http_address,
            settings.http_port,
            monotonic=clock.monotonic,
            picture_period_s=1.0 / settings.fps,
        )

    @property
    def port(self) -> int:
        """The TCP port the page is served on."""
        return self._server.server_address[1]

    @property
    def url(self) -> str:
        return f"http://{LOOPBACK}:{self.port}/"

    def start(self) -> None:
        work: list[tuple[str, Callable[[], None]]] = [
            ("view-receive", self._receive),
            ("view-status", self._send_status),
            ("view-http", lambda: self._server.serve_forever(0.1)),
        ]
        if self._pictures is not None:
            work.append(("view-pictures", self._run_pictures))
        for name, target in work:
            thread = threading.Thread(target=target, name=name, daemon=True)
            self._threads.append(thread)
            thread.start()

    def close(self) -> None:
        """End the threads and the camera source; tell the Thor the view is over."""
        self._end.set()
        if self._threads:
            self._server.shutdown()
        self._server.server_close()
        if self._source is not None:
            self._source.close()
        for thread in self._threads:
            thread.join(PAIR_WAIT_S + 4.0)
        send(self.carrier, ViewStatus(live=False, age_s=AGE_WHEN_NOT_LIVE_S))

    def _receive(self) -> None:
        while not self._end.is_set():
            try:
                message = receive_message(self.carrier, _RECEIVE_S)
            except Exception:
                _log.exception("receiving a link message failed")
                self._end.wait(_RECEIVE_S)
                continue
            if message is not None:
                self.model.on_message(message, self._clock.monotonic())

    def _send_status(self) -> None:
        """VIEW_STATUS every status.period_s, also while the view is not live."""
        while True:
            try:
                self.model.on_local(self.carrier.status, self.carrier.alive)
                now = self._clock.monotonic()
                status = self.model.view_status(now, self.model.thor_wall_now(now))
                send(self.carrier, status)
                self.model.note_sent(status)
            except Exception:
                _log.exception("sending VIEW_STATUS failed")
            if self._end.wait(self.settings.period_s):
                return

    def _run_pictures(self) -> None:
        try:
            self._pictures.run()
        except Exception as problem:
            _log.exception("the picture loop failed")
            self.model.set_camera_note(f"no camera pictures: {problem}")


def make_source(
    settings: ViewSettings, kind: str, model: ViewModel, clock: Clock
) -> tuple[PairSource | None, bool]:
    """The camera source --source names, and whether its times are invented."""
    if kind == SOURCE_NONE:
        return None, False
    if kind.startswith(SOURCE_RECORDED):
        folder = _user_path("--source", kind.removeprefix(SOURCE_RECORDED))
        if not folder.is_dir():
            raise ViewConfigError(f"--source: {folder} is not a folder")

        def thor_wall_now() -> float:
            known = model.thor_wall_now(clock.monotonic())
            return time.time() if known is None else known

        source = RecordedPairSource(
            folder,
            period_s=1.0 / settings.fps,
            thor_wall_now=thor_wall_now,
            monotonic=clock.monotonic,
        )
        return source, True
    if kind != SOURCE_LINK:
        raise ViewConfigError(
            f"--source must be {SOURCE_LINK}, {SOURCE_RECORDED}<dir> or "
            f"{SOURCE_NONE}, got {kind!r}"
        )
    if not settings.bridge_python:
        raise ViewConfigError(
            "camera.bridge_python (the camera consumer's interpreter) is needed for "
            f"--source {SOURCE_LINK}; --source {SOURCE_NONE} shows the status only"
        )
    source = BridgeSource(
        python=settings.bridge_python,
        receiver_socket=settings.receiver_socket,
        run_dir=settings.state_dir / "view",
        camera_pr=settings.camera_pr or None,
        streams=settings.streams,
        max_skew_ms=settings.max_skew_ms,
    )
    return source, False


def _say(text: str) -> None:
    print(text, flush=True)


def _complain(text: str) -> None:
    print(f"{PROGRAM}: {text}", file=sys.stderr, flush=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m isaaccapture_examples.agentic_remote.view",
        description=(
            "The live view of the agentic remote execution example: a camera "
            "picture and the robot's status for a browser on this machine. It "
            "only looks; it sends the robot no command."
        ),
    )
    parser.add_argument(
        "--config", required=True, metavar="PATH", help="the view's YAML file"
    )
    parser.add_argument(
        "--source",
        default=SOURCE_LINK,
        metavar="SOURCE",
        help=(
            f"{SOURCE_LINK} (default): the camera receiver; {SOURCE_RECORDED}<dir>: "
            f"raw pairs from a folder, never live; {SOURCE_NONE}: the status only"
        ),
    )
    parser.add_argument(
        "--port",
        default=None,
        type=int,
        metavar="N",
        help="the page's TCP port on 127.0.0.1 (default: the file's http.port)",
    )
    parser.add_argument(
        "--log-level",
        default=None,
        type=str.lower,
        choices=LOG_LEVELS,
        metavar="L",
        help=f"console log level, one of {', '.join(LOG_LEVELS)} (default: the file's)",
    )
    return parser


def _wait_for_end(program: ViewProgram) -> str:
    """Wait on the main thread for SIGINT or SIGTERM, or for the service's end."""
    # The handler only queues: SimpleQueue.put() is reentrant.
    came: queue.SimpleQueue[int] = queue.SimpleQueue()
    numbers = (signal.SIGINT, signal.SIGTERM)
    replaced = {
        n: signal.signal(n, lambda number, _: came.put(number)) for n in numbers
    }
    try:
        while program.carrier.alive:
            try:
                return signal.Signals(came.get(timeout=0.5)).name
            except queue.Empty:
                continue
        return "the robot link service is gone"
    finally:
        for number, handler in replaced.items():
            signal.signal(number, signal.SIG_DFL if handler is None else handler)


def main(
    argv: Sequence[str] | None = None,
    *,
    attach: Callable[..., LocalCarrier] = connect_local,
    source: PairSource | None = None,
    clock: Clock | None = None,
    wait: Callable[[ViewProgram], str] | None = None,
    say: Say = _say,
) -> int:
    """Run the live view's command line; returns the exit code.

    attach(path, role, name, timeout_s) gives the carrier of the local API.
    source stands in for the camera frame bridge of --source link.
    wait(program) blocks until the view is to end and returns why; None waits
    for SIGINT or SIGTERM, which only the main thread can.
    """
    try:
        args = _parser().parse_args(argv)
    except SystemExit as exit_:  # argparse has printed the usage or the help
        code = exit_.code
        return code if isinstance(code, int) else EXIT_NOT_RUN
    clock = clock or _MachineClock()
    try:
        settings = load_view_settings(args.config)
        if args.port is not None:
            if not 0 <= args.port <= 65535:
                raise ViewConfigError(
                    f"--port must be from 0 to 65535, got {args.port}"
                )
            settings = dataclasses.replace(settings, http_port=args.port)
        model = ViewModel(
            ack_timeout_s=settings.ack_timeout_s,
            state_max_age_s=settings.state_max_age_s,
        )
        if source is not None and args.source == SOURCE_LINK:
            made, invented = source, False
        else:
            made, invented = make_source(settings, args.source, model, clock)
    except ViewConfigError as error:
        _complain(f"the settings cannot be used: {error}")
        return EXIT_NOT_RUN
    if wait is None and threading.current_thread() is not threading.main_thread():
        _complain("signals can be waited for on the main thread only")
        return EXIT_NOT_RUN
    set_console_level(args.log_level or settings.log_level)
    try:
        carrier = attach(settings.socket_path, ROLE_VIEWER, "view", ATTACH_TIMEOUT_S)
    except OSError as problem:  # also LocalRefused and TimeoutError
        _complain(f"cannot attach to the robot link service: {problem}")
        return EXIT_FAILED
    try:
        status = carrier.status
        if status is not None and status.roles.get(ROLE_VIEWER, 0) > 1:
            # Two views would send the Thor two different VIEW_STATUS in turn.
            _complain(f"{SECOND_VIEW}; this one ends")
            return EXIT_FAILED
        try:
            program = ViewProgram(settings, carrier, model, made, invented, clock, say)
        except OSError as problem:
            _complain(
                f"cannot serve on {settings.http_address}:{settings.http_port}: "
                f"{problem}"
            )
            return EXIT_FAILED
        try:
            program.start()
            if made is None:
                model.set_camera_note(f"no camera (--source {SOURCE_NONE})")
            say(
                f"{PROGRAM}: open {program.url} in a browser on this machine; "
                f"camera pictures: {_source_text(args.source, made)}"
            )
            why = _wait_for_end(program) if wait is None else wait(program)
            gone = not carrier.alive
            say(f"{PROGRAM}: {why}; closing. The robot is not touched.")
        finally:
            program.close()
        if gone:
            _complain(f"the robot link service is gone ({carrier.close_reason})")
            return EXIT_FAILED
        return EXIT_OK
    finally:
        carrier.close()


def _source_text(kind: str, source: PairSource | None) -> str:
    if source is None:
        return "none; the page shows the status only"
    if kind.startswith(SOURCE_RECORDED):
        return f"recorded pairs from {kind.removeprefix(SOURCE_RECORDED)}, never live"
    return "the camera receiver, through the view's own camera frame bridge"


if __name__ == "__main__":
    raise SystemExit(main())
