# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import storage
from .document import State


def main() -> int:
    parser = argparse.ArgumentParser(description="Local tactile metadata editor")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--metadata", help="Metadata JSON path")
    parser.add_argument("--side", choices=("left", "right"), default="left")
    parser.add_argument("--urdf", help="URDF path")
    args = parser.parse_args()
    try:
        from .ui import Editor

        state = State()
        state.side = args.side
        if args.metadata:
            path = Path(args.metadata).expanduser().resolve()
            state.replace(storage.load(path, args.side), str(path))
        Editor(state, args).run()
        return 0
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
