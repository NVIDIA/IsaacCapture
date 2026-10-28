# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Locate the install tree that the plugin scripts here run against."""

import argparse
from pathlib import Path


def plugins_dir() -> Path:
    """Return ``install/plugins`` of this checkout, or of the CloudXR test image.

    The checkout root is the nearest ancestor with ``VERSION`` and ``CMakeLists.txt``,
    as in ``tests/python/repo_paths.py``. The image has neither and puts these
    scripts in ``/app/tests``, beside ``/app/install``.
    """
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "VERSION").is_file() and (parent / "CMakeLists.txt").is_file():
            return parent / "install" / "plugins"
    return here.parents[1] / "install" / "plugins"


def add_plugin_root_argument(parser: argparse.ArgumentParser) -> None:
    """Register ``--plugin-root``, defaulting to :func:`plugins_dir`."""
    default = plugins_dir()
    parser.add_argument(
        "--plugin-root",
        type=Path,
        default=default,
        help=f"Plugin search directory, e.g. <install prefix>/plugins (default: {default})",
    )
