# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse
import sys

from .document import State


def main() -> int:
    parser = argparse.ArgumentParser(description="Local tactile metadata editor")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    try:
        from .ui import Editor

        Editor(State(), args).run()
        return 0
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
