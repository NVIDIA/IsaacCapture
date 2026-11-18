# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""python -m isaaccapture_examples.agentic_remote.viewer_live [OPTIONS] [the viewer's own].

Runs the experiment's viewer (thor_viewer.py of the experiment folder) in this
process, as it is, and adds the camera's live picture to its page: the newest
picture of the camera receiver, through a camera frame bridge of its own. The
picture is shown and never stored. Without a camera the viewer serves the
recorded runs as it does alone.

The page with its live picture showing is the live view: the viewer attaches
to the robot link service in the role "viewer" and tells the Thor, with
VIEW_STATUS, how fresh the picture on that page is. Without a robot link
service it serves everything and attaches once one runs.
"""

from __future__ import annotations

import argparse
import importlib
import os
import shutil
import signal
import sys
import tempfile
import threading
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType

from ..link_service.local import connect_local
from .live import (
    DEFAULT_FPS,
    DEFAULT_WIDTH_PX,
    FPS_RANGE,
    JPEG_QUALITY,
    PROGRAM,
    RECEIVER_SOCKET,
    STREAM_PATH,
    WIDTH_RANGE_PX,
    LivePicture,
    Say,
    SourceFactory,
    bridge_source_factory,
    live_handler,
    use_one_encoder_thread,
)
from .view_link import Attach, ViewLink

EXIT_OK = 0
EXIT_NOT_RUN = 2  # the arguments or the experiment folder cannot be used

EXPERIMENT_DIR_ENV = "AGENTIC_REMOTE_EXPERIMENT_DIR"
VIEWER_MODULE = "thor_viewer"
# The signals that end the viewer as Ctrl+C does; SIGHUP is a closed terminal.
_END_SIGNALS = (signal.SIGTERM, signal.SIGHUP)


def _say(text: str) -> None:
    print(text, flush=True)


def _complain(text: str) -> None:
    print(f"{PROGRAM}: {text}", file=sys.stderr, flush=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m isaaccapture_examples.agentic_remote.viewer_live",
        description=(
            "The experiment's viewer with the camera's live picture on its page. "
            "Every other argument goes to the viewer as it is."
        ),
        add_help=False,
        allow_abbrev=False,
    )
    parser.add_argument("-h", "--help", action="store_true", help="this text")
    parser.add_argument(
        "--experiment-dir",
        default=os.environ.get(EXPERIMENT_DIR_ENV),
        metavar="DIR",
        help=f"the experiment folder (default: ${EXPERIMENT_DIR_ENV})",
    )
    parser.add_argument(
        "--live-fps",
        type=float,
        default=DEFAULT_FPS,
        metavar="N",
        help=(
            f"the most live pictures a second, {FPS_RANGE[0]:g} to {FPS_RANGE[1]:g} "
            f"(default: {DEFAULT_FPS:g})"
        ),
    )
    parser.add_argument(
        "--live-width",
        type=int,
        default=DEFAULT_WIDTH_PX,
        metavar="PIXELS",
        help=(
            f"the live picture's greatest width, {WIDTH_RANGE_PX[0]} to "
            f"{WIDTH_RANGE_PX[1]} (default: {DEFAULT_WIDTH_PX})"
        ),
    )
    parser.add_argument(
        "--receiver-socket",
        default=RECEIVER_SOCKET,
        metavar="PATH",
        help=f"the camera receiver's socket (default: {RECEIVER_SOCKET})",
    )
    parser.add_argument(
        "--link-socket",
        default=None,
        metavar="PATH",
        help=(
            "the robot link service's local socket, for the live view (default: "
            "the one the console attaches to)"
        ),
    )
    return parser


def _load_viewer(folder: Path) -> ModuleType:
    """The experiment's viewer module, with its folder first on the import path.

    That is where Python puts the folder when it runs thor_viewer.py as a script.
    """
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))
    return importlib.import_module(VIEWER_MODULE)


def main(
    argv: Sequence[str] | None = None,
    *,
    make_source: SourceFactory | None = None,
    attach: Attach = connect_local,
    say: Say = _say,
) -> int:
    """Run the viewer with the live picture; returns the exit code.

    make_source() stands in for the camera frame bridge.
    attach(path, role, name, timeout_s) gives the carrier of the local API.
    """
    args, rest = _parser().parse_known_args(argv)
    if args.help:
        _parser().print_help()
        say("\nThe viewer's own arguments:")
        rest = [*rest, "--help"]
    if not FPS_RANGE[0] <= args.live_fps <= FPS_RANGE[1]:
        _complain(
            f"--live-fps must be from {FPS_RANGE[0]:g} to {FPS_RANGE[1]:g}, got "
            f"{args.live_fps:g}"
        )
        return EXIT_NOT_RUN
    if not WIDTH_RANGE_PX[0] <= args.live_width <= WIDTH_RANGE_PX[1]:
        _complain(
            f"--live-width must be from {WIDTH_RANGE_PX[0]} to {WIDTH_RANGE_PX[1]}, "
            f"got {args.live_width}"
        )
        return EXIT_NOT_RUN
    folder = Path(args.experiment_dir or "").resolve()
    if not args.experiment_dir or not (folder / f"{VIEWER_MODULE}.py").is_file():
        _complain(
            f"--experiment-dir or {EXPERIMENT_DIR_ENV} must name the folder that "
            f"holds {VIEWER_MODULE}.py (it is '{args.experiment_dir or ''}')"
        )
        return EXIT_NOT_RUN
    try:
        viewer = _load_viewer(folder)
    except ImportError as problem:
        _complain(f"the experiment's viewer cannot be imported: {problem}")
        return EXIT_NOT_RUN

    use_one_encoder_thread()
    run_dir: list[str] = []  # the bridge's log and socket, once the viewer serves
    live: list[LivePicture] = []
    links: list[ViewLink] = []
    own_handler = viewer.handler

    def handler(runs: object, token: str | None) -> type:
        """The viewer's handler factory; it starts the live picture and the live view."""
        base = own_handler(runs, token)
        run_dir.append(tempfile.mkdtemp(prefix="agentic_remote_viewer_live_"))
        source = make_source or bridge_source_factory(args.receiver_socket, run_dir[0])
        picture = LivePicture(
            source, fps=args.live_fps, width_px=args.live_width, say=say
        )
        live.append(picture)
        say(
            f"{PROGRAM}: live picture: on the page and at {STREAM_PATH}, at most "
            f"{args.live_fps:g} a second, at most {args.live_width} px wide, JPEG "
            f"quality {JPEG_QUALITY}; shown, not stored. Camera receiver: "
            f"{args.receiver_socket}"
        )
        picture.start()
        link = ViewLink(picture.model, args.link_socket, attach=attach, say=say)
        links.append(link)
        say(
            f"{PROGRAM}: live view: this page with its live picture showing, in a "
            "visible browser tab; the robot is told its age through the robot "
            f"link service at {link.socket_path}"
        )
        link.start()
        return live_handler(base, token, picture, viewer.authorized, link.state)

    # Signal handlers can be set on the main thread only.
    signals = (
        (*_END_SIGNALS, signal.SIGINT)
        if threading.current_thread() is threading.main_thread()
        else ()
    )
    before = {number: signal.getsignal(number) for number in signals}

    def end(number: int, frame: object) -> None:
        raise KeyboardInterrupt

    for number in signals[:-1]:
        signal.signal(number, end)
    argv_before = sys.argv
    sys.argv = [str(folder / f"{VIEWER_MODULE}.py"), *rest]
    viewer.handler = handler
    try:
        viewer.main()
    except KeyboardInterrupt:  # before the viewer served; it takes a later one itself
        pass
    finally:
        # No signal may cut the bridge's end short.
        for number in signals:
            signal.signal(number, signal.SIG_IGN)
        viewer.handler = own_handler
        sys.argv = argv_before
        for link in links:  # the robot is told that the view is over
            link.close()
        for picture in live:
            picture.close()
        for made in run_dir:
            shutil.rmtree(made, ignore_errors=True)
        for number, handler_before in before.items():
            signal.signal(
                number, signal.SIG_DFL if handler_before is None else handler_before
            )
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
