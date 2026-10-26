# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Local GUI, checking and atomic package export through shared module APIs."""

from __future__ import annotations

import argparse
from importlib import metadata
import json
import logging
import os
from pathlib import Path
import re
import sys
import tempfile

from .checks import check_device, check_summary, checks_exit_code
from .document import MetadataFormatError, load_metadata, merge_documents
from .domain import Snapshot
from .export import ExportFailure, ExportOptions, export_package

LOGGER = logging.getLogger("isaaccapture.tools.tactile_metadata.cli")


class ArgumentError(ValueError):
    pass


def _port(value):
    number = int(value)
    if not 1 <= number <= 65535:
        raise argparse.ArgumentTypeError("Port must be between 1 and 65535")
    return number


def build_parser():
    parser = argparse.ArgumentParser(description="Tactile Metadata Editor")
    commands = parser.add_subparsers(dest="command", required=True)
    gui = commands.add_parser("gui", help="Open the local editor")
    gui.add_argument("--urdf", type=Path)
    gui.add_argument("--sides", choices=("left", "right", "both"))
    gui.add_argument("--metadata", type=Path)
    gui.add_argument("--left-metadata", type=Path)
    gui.add_argument("--right-metadata", type=Path)
    gui.add_argument("--port", type=_port, default=8080)
    gui.add_argument(
        "--demo", action="store_true",
        help="Use synthetic interaction data; this does not verify SDK reception",
    )
    check = commands.add_parser("check", help="Check one metadata JSON")
    check.add_argument("metadata", type=Path)
    check.add_argument("--sides", choices=("left", "right"),
                       help="Optionally assign the input's hand side")
    check.add_argument("--sdk", action="store_true",
                       help="Construct and encode through the installed SDK")
    check.add_argument("--report", type=Path)
    export = commands.add_parser("export", help="Export a new local data package")
    export.add_argument("metadata", type=Path, nargs="*")
    export.add_argument("--left-metadata", type=Path)
    export.add_argument("--right-metadata", type=Path)
    export.add_argument("--sides", choices=("left", "right"),
                        help="Assign a side to one positional metadata input")
    export.add_argument("--format", choices=("json", "sdk"), default="json")
    export.add_argument("--out", type=Path, required=True)
    return parser


def _write_report(path, value):
    path = path.expanduser().absolute()
    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def _run_check(args):
    if args.report is not None and args.report.resolve() == args.metadata.resolve():
        raise ArgumentError("The check report must use a different path from metadata")
    document = load_metadata(args.metadata, side=args.sides)
    snapshot = Snapshot(document, 0)
    results = tuple(
        check_device(snapshot, device.id, use_sdk=args.sdk)
        for device in document.devices
    )
    report = {
        "format": "isaaccapture.tactile-report",
        "version": 1,
        "revision": snapshot.revision,
        "devices": [check_summary(result) for result in results],
    }
    if args.report is not None:
        _write_report(args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    return checks_exit_code(results, use_sdk=args.sdk)


def _export_document(args):
    positional = list(args.metadata)
    assigned = [
        (path, side) for path, side in (
            (args.left_metadata, "left"), (args.right_metadata, "right")
        ) if path is not None
    ]
    if args.sides is not None and (len(positional) != 1 or assigned):
        raise ArgumentError("--sides assigns exactly one positional metadata input")
    inputs = [(path, args.sides) for path in positional] + assigned
    if not inputs:
        raise ArgumentError("Provide at least one metadata JSON to export")
    documents = tuple(load_metadata(path, side=side) for path, side in inputs)
    for document in documents:
        if any(device.side not in ("left", "right") for device in document.devices):
            raise ArgumentError(
                "Export requires explicit hand sides; use --sides or "
                "--left-metadata/--right-metadata"
            )
    return merge_documents(documents)


def _run_export(args):
    document = _export_document(args)
    result = export_package(
        Snapshot(document, 0), tuple(device.id for device in document.devices),
        ExportOptions(args.out, args.format),
    )
    print(json.dumps({
        "path": str(result.path),
        "revision": result.revision,
        "devices": [check_summary(check) for check in result.checks],
    }, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


def _gui_dependencies():
    exact = {
        "viser": ("1.1.1", "GUI events and the browser bridge"),
        "yourdfpy": ("0.0.60", "URDF graph and forward kinematics"),
        "trimesh": ("5.1.1", "mesh instances, normals and ray queries"),
    }
    reasons = []
    for name in (*exact, "numpy", "rtree"):
        try:
            version = metadata.version(name)
        except metadata.PackageNotFoundError:
            reasons.append(f"{name} is not installed in {sys.executable}")
            continue
        if name in exact:
            required, purpose = exact[name]
            if version != required:
                reasons.append(
                    f"{purpose} requires {name}=={required}; found {version}"
                )
        else:
            match = re.fullmatch(
                r"(\d+)\.(\d+)\.(\d+)(?:\.post\d+)?",
                version.split("+", 1)[0],
            )
            release = tuple(map(int, match.groups())) if match else ()
            valid = release and (
                release[0] == 2 if name == "numpy"
                else (1, 3, 0) <= release < (2, 0, 0)
            )
            if not valid:
                required = ">=2,<3" if name == "numpy" else ">=1.3,<2"
                reasons.append(f"{name}{required} is required; found {version}")
    if reasons:
        raise RuntimeError("; ".join(reasons))


def _run_gui(args):
    if args.urdf is not None and args.sides is None:
        raise ArgumentError("--urdf requires an explicit --sides declaration")
    if args.metadata is not None and (
        args.left_metadata is not None or args.right_metadata is not None
    ):
        raise ArgumentError("Use --metadata or the explicitly assigned side inputs")
    if args.sides == "both" and args.urdf is None:
        raise ArgumentError("--sides both requires a URDF containing both hands")
    _gui_dependencies()
    from .ui import run_gui

    return run_gui(
        urdf=args.urdf, sides=args.sides, metadata=args.metadata,
        left_metadata=args.left_metadata, right_metadata=args.right_metadata,
        port=args.port, demo=args.demo,
    )


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if not (3, 11) <= sys.version_info[:2] < (3, 14):
        LOGGER.error(
            "Python >=3.11,<3.14 is required; current interpreter is %s (%s)",
            sys.executable, sys.version.split()[0],
        )
        return 2
    try:
        if args.command == "check":
            return _run_check(args)
        if args.command == "export":
            return _run_export(args)
        return _run_gui(args)
    except (ArgumentError, MetadataFormatError) as error:
        LOGGER.error("%s", error)
        return 2
    except ExportFailure as error:
        LOGGER.error("Export failed: %s", error)
        return error.exit_code
    except (OSError, ValueError, RuntimeError, ImportError) as error:
        LOGGER.error("%s", error)
        return 1


if __name__ == "__main__":
    # This standalone process imports the SDK only in local worker processes.
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    raise SystemExit(main())
