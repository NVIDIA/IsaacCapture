# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Locate the install tree that the plugin scripts here run against."""

from pathlib import Path


def plugins_dir() -> Path:
    """Return the nearest ancestor's ``install/plugins``, else that relative path.

    That is ``<checkout>/install/plugins`` after ``cmake --install build``, and
    ``/app/install/plugins`` in the CloudXR test image.
    """
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "install" / "plugins"
        if candidate.is_dir():
            return candidate
    return Path("install/plugins")
