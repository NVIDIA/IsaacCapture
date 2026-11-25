# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build only selected transport plugins, optionally installing into an interpreter."""

import argparse
import os
from pathlib import Path
import subprocess
import tempfile
import tomllib

GROUP = "isaac_media_bridge.transports.v1"


def projects(repo, extra=()):
    result = {}
    candidates = sorted((repo / "src/plugins").glob("*/pyproject.toml"))
    candidates += [Path(p).expanduser().resolve() / "pyproject.toml" for p in extra]
    for manifest in candidates:
        data = tomllib.loads(manifest.read_text())
        entries = data.get("project", {}).get("entry-points", {}).get(GROUP, {})
        for name in entries:
            path = manifest.parent
            if name in result and result[name] != path:
                raise ValueError(f"Duplicate transport plugin: {name}")
            result[name] = path
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transport", action="append", required=True)
    parser.add_argument(
        "--plugin-project",
        type=Path,
        action="append",
        default=[],
        help="Additional plugin project directory; selected by its registered transport name",
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="Empty wheel directory"
    )
    parser.add_argument("--python", help="Install into this interpreter after building")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    try:
        catalog = projects(repo, args.plugin_project)
        names = list(dict.fromkeys(args.transport))
        unknown = set(names) - catalog.keys()
        if unknown:
            raise ValueError(
                f"Unknown transport(s): {sorted(unknown)}; available: {sorted(catalog)}"
            )
        output = args.output.expanduser().resolve()
        if output.exists() and (not output.is_dir() or any(output.iterdir())):
            raise ValueError("Choose an empty wheel directory")
        output.mkdir(parents=True, exist_ok=True, mode=0o700)
        info = output.stat()
        if info.st_uid != os.getuid() or info.st_mode & 0o022:
            raise ValueError(
                "Wheel directory must be owned by you and not writable by others"
            )
        wheels = []
        for project in [
            repo / "src/core/media_transport",
            *(catalog[name] for name in names),
        ]:
            with tempfile.TemporaryDirectory(prefix="transport-build-") as temp:
                subprocess.run(
                    ["uv", "build", "--wheel", "--out-dir", temp, str(project)],
                    check=True,
                )
                for wheel in Path(temp).glob("*.whl"):
                    target = output / wheel.name
                    with target.open("xb") as destination:
                        destination.write(wheel.read_bytes())
                    wheels.append(str(target))
        if args.python:
            subprocess.run(
                ["uv", "pip", "install", "--python", args.python, *wheels], check=True
            )
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        parser.error(str(exc))
    print(f"Transport wheels: {output}")
    print(
        "Optional native receiver libraries are built separately for the target architecture."
    )


if __name__ == "__main__":
    main()
