# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Isaac Teleop - Teleoperation Core Library

This package provides Python bindings for teleoperation with Device I/O.
"""

import os


def _reserve_closed_standard_fds() -> list[int]:
    """Prevent eager imports from retaining a host-closed standard fd."""
    reserved = []
    for expected in (0, 1, 2):
        try:
            os.fstat(expected)
            continue
        except OSError:
            pass
        try:
            actual = os.open(os.devnull, os.O_RDWR)
        except OSError:
            continue
        if actual == expected:
            reserved.append(actual)
        else:
            os.close(actual)
    return reserved


_reserved_standard_fds = _reserve_closed_standard_fds()
try:
    from importlib.metadata import PackageNotFoundError, distribution, version

    # The transition distribution owns no code; legacy wheels still own isaacteleop/.
    # Wheel installers do not execute this import-time check.
    try:
        _legacy_files = distribution("isaacteleop").files or ()
    except PackageNotFoundError:
        _legacy_files = ()
    if any(path.parts[0] == "isaacteleop" for path in _legacy_files):
        raise ImportError(
            "The legacy 'isaacteleop' distribution is still installed in this environment. "
            "Run `python -m pip install --upgrade isaaccapture` "
            "with your intended version and extras to replace it."
        )

    try:
        __version__ = version("isaaccapture")
    except PackageNotFoundError:
        # Fallback for local source-tree usage before wheel/package installation.
        __version__ = "0+unknown"

    # Import submodules.
    from . import (
        cloudxr,
        deviceio,
        deviceio_session,
        deviceio_trackers,
        logging_config,
        oxr,
        ovxr_session_manager,
        plugin_manager,
        schema,
        teleop_session_manager,
    )

    # Build the Python handlers and publish the C++ forwarding socket.
    logging_config.install()
finally:
    while _reserved_standard_fds:
        os.close(_reserved_standard_fds.pop())

__all__ = [
    "logging_config",
    "deviceio_trackers",
    "deviceio_session",
    "deviceio",
    "oxr",
    "ovxr_session_manager",
    "plugin_manager",
    "schema",
    "teleop_session_manager",
    "cloudxr",
    "__version__",
]
