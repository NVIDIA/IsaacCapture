# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Command-line entry point for the tactile shape editor and visualizer."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import viser

from isaaccapture.cloudxr import CloudXRLauncher

from .app import TactileVisualizer
from .model import ShapeDocument, default_openxr_shape
from .sources import MockSource, SessionSource, SessionSourceConfig
from .urdf_config import load_urdf_config


def _default_plugin_paths() -> tuple[Path, ...]:
    tree = Path(__file__).resolve().parents[5]
    return tuple(
        path
        for path in (tree / "plugins", tree / "install" / "plugins")
        if (path / "sharpa_avatar" / "avatar_hand_plugin").is_file()
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", type=Path, required=True, help="Glove URDF")
    parser.add_argument(
        "--urdf-config",
        type=Path,
        help="YAML/JSON with scale, offset_cm, rotation_xyzw, and joint_map",
    )
    parser.add_argument("--shape", type=Path, help="TactileShape FlatBuffer to load")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--side", choices=("left", "right"), default="right")
    parser.add_argument(
        "--taxels", type=int, default=200, help="Taxel count for --mock (default: 200)"
    )
    parser.add_argument("--taxel-size-mm", type=float, default=4.0)
    parser.add_argument("--liveness-timeout", type=float, default=1.0)
    parser.add_argument(
        "--allow-server-write-root",
        type=Path,
        help="Enable atomic local exports only below this directory",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--live", action="store_true", help="Use live DeviceIO streams")
    mode.add_argument("--mock", action="store_true", help="Use fist/palm mock data")
    mode.add_argument("--replay", type=Path, help="Replay an MCAP recording")
    parser.add_argument("--left-tactile-collection", default="tactile_left")
    parser.add_argument("--right-tactile-collection", default="tactile_right")
    parser.add_argument("--left-tactile-channel", default="tactile_left")
    parser.add_argument("--right-tactile-channel", default="tactile_right")
    parser.add_argument("--launch-plugin", action="store_true")
    parser.add_argument("--plugin-search-path", type=Path, action="append")
    parser.add_argument("--sdk-config", type=Path)
    parser.add_argument("--datasets", default="human,raw,robot,haptic")
    CloudXRLauncher.add_launcher_arguments(parser)
    return parser


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    for path, label in (
        (args.urdf, "--urdf"),
        (args.shape, "--shape"),
        (args.replay, "--replay"),
    ):
        if path is not None and not path.is_file():
            parser.error(f"{label} does not exist: {path}")
    if args.taxels < 0:
        parser.error("--taxels must be nonnegative")
    for value, label in (
        (args.taxel_size_mm, "--taxel-size-mm"),
        (args.liveness_timeout, "--liveness-timeout"),
    ):
        if not math.isfinite(value) or value <= 0:
            parser.error(f"{label} must be finite and positive")
    if args.launch_plugin and not args.live:
        parser.error("--launch-plugin requires --live")
    paths = args.plugin_search_path or _default_plugin_paths()
    if args.launch_plugin and not any(
        (path / "sharpa_avatar" / "avatar_hand_plugin").is_file() for path in paths
    ):
        parser.error(
            "avatar_hand_plugin not found; build/install the Sharpa plugin or pass "
            "--plugin-search-path"
        )


def _make_source(args: argparse.Namespace):
    if args.mock:
        return MockSource(args.taxels)
    if not args.live and args.replay is None:
        return None
    config = SessionSourceConfig(
        tactile_collections={
            "left": args.left_tactile_collection,
            "right": args.right_tactile_collection,
        },
        tactile_channels={
            "left": args.left_tactile_channel,
            "right": args.right_tactile_channel,
        },
        launch_plugin=args.launch_plugin,
        plugin_search_paths=tuple(args.plugin_search_path or _default_plugin_paths()),
        sdk_config=args.sdk_config,
        datasets=args.datasets,
        replay_path=args.replay,
    )
    return SessionSource(config, args)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _validate_args(parser, args)
    urdf_config = load_urdf_config(args.urdf_config)
    shape_error = ""
    if args.shape is None:
        document = default_openxr_shape()
    else:
        try:
            document = ShapeDocument.from_bytes(args.shape.read_bytes())
        except Exception as error:
            document = default_openxr_shape()
            shape_error = (
                f"Initial shape `{args.shape}` was rejected: {error}. "
                "Started with an empty default shape."
            )
    if args.allow_server_write_root is not None:
        args.allow_server_write_root.mkdir(parents=True, exist_ok=True)
    server = viser.ViserServer(host=args.host, port=args.port, label="TactileShape")
    app = TactileVisualizer(
        server=server,
        urdf_path=args.urdf,
        urdf_config_path=args.urdf_config,
        urdf_config=urdf_config,
        document=document,
        document_name=(
            str(args.shape)
            if args.shape is not None and not shape_error
            else "untitled"
        ),
        source=_make_source(args),
        side=args.side,
        liveness_timeout_s=args.liveness_timeout,
        taxel_size_m=args.taxel_size_mm / 1000.0,
        write_root=args.allow_server_write_root,
    )
    if shape_error:
        app._report_error(shape_error)
    print(f"Tactile visualizer: http://{args.host}:{args.port}")
    print("Ctrl+C stops the local server.")
    app.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
