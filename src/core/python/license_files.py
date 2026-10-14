# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Print the root pyproject.toml's project.license-files for CMake.

The first line is the TOML array, verbatim, for pyproject.toml.in; each line
after it is one file those patterns match, relative to the repository root.
"""

import json
import sys
import tomllib
from pathlib import Path


def main(root: Path) -> int:
    with (root / "pyproject.toml").open("rb") as handle:
        patterns = tomllib.load(handle)["project"]["license-files"]
    print(json.dumps(patterns))
    for pattern in patterns:
        # PEP 639: a pattern that matches nothing is an error, not an empty set.
        matches = sorted(path for path in root.glob(pattern) if path.is_file())
        if not matches:
            print(
                f"project.license-files: {pattern!r} matches no file", file=sys.stderr
            )
            return 1
        for path in matches:
            print(path.relative_to(root).as_posix())
    return 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1])))
