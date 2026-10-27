# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import storage
from .document import Document, State


def main():
    parser = argparse.ArgumentParser(description="Local tactile metadata editor")
    commands = parser.add_subparsers(dest="command", required=True)
    gui = commands.add_parser("gui")
    gui.add_argument("--urdf")
    gui.add_argument("--sides", choices=("left", "right", "both"))
    gui.add_argument("--metadata")
    gui.add_argument("--left-metadata")
    gui.add_argument("--right-metadata")
    gui.add_argument("--port", type=int, default=8080)
    gui.add_argument("--demo", action="store_true")
    check = commands.add_parser("check")
    check.add_argument("metadata")
    check.add_argument("--sdk", action="store_true")
    check.add_argument("--report")
    export = commands.add_parser("export")
    export.add_argument("metadata", nargs="+")
    export.add_argument("--sides", choices=("left", "right", "both"))
    export.add_argument("--format", choices=("json", "sdk"), default="json")
    export.add_argument("--out", required=True)
    args = parser.parse_args()
    try:
        if args.command == "check":
            d = storage.load(args.metadata)
            result = dict(local=d.check(), sdk="not checked")
            failed = any(v["severity"] == "error" for v in result["local"])
            if args.sdk:
                from .sdk import check

                try:
                    result["sdk"] = check(d)
                except (ValueError, RuntimeError) as exc:
                    result["sdk"], failed = dict(error=str(exc)), True
            if args.report:
                storage.write(args.report, result)
            print(json.dumps(result, indent=2))
            return int(failed)
        sides = (
            ["left", "right"]
            if args.sides == "both"
            else [args.sides]
            if args.sides
            else []
        )
        inputs = (
            [
                (p, sides[i] if len(sides) == len(args.metadata or []) else None)
                for i, p in enumerate(args.metadata)
            ]
            if args.command == "export"
            else (
                [(args.metadata, sides[0] if len(sides) == 1 else None)]
                if args.metadata
                else []
            )
        )
        if args.command == "gui":
            inputs += [
                (getattr(args, s + "_metadata"), s)
                for s in ("left", "right")
                if getattr(args, s + "_metadata")
            ]
        documents, model, paths = {}, {}, {}
        for path, side in inputs:
            d = storage.load(path, side)
            if not d.side:
                if args.command == "gui":
                    continue
                raise ValueError("Assign JSON sides with editor.side or --sides")
            if d.side in documents:
                raise ValueError("Duplicate metadata side")
            cfg = d.editor.get("model", {})
            if model.get("urdf") and cfg.get("urdf") and model != cfg:
                raise ValueError("Conflicting URDF configurations")
            if cfg.get("urdf"):
                model = cfg
            documents[d.side], paths[d.side] = d, str(Path(path).resolve())
        if args.command == "export":
            print(
                json.dumps(
                    storage.export(documents, model, 0, args.out, args.format == "sdk"),
                    indent=2,
                )
            )
        else:
            from .ui import Editor

            for side in sides:
                documents.setdefault(side, Document(side))
            if args.urdf:
                model["urdf"] = str(Path(args.urdf).resolve())
            state = State(documents, model)
            state.paths = paths
            state.saved = {
                s: state.fingerprint(s)
                for s in paths
                if documents[s].editor.get("side") == s
                and documents[s].editor.get("model", {}) == state.model
            }
            Editor(state, args).run()
        return 0
    except (ValueError, OSError, RuntimeError, ImportError, TypeError) as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
