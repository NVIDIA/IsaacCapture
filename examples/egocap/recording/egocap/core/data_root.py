# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Where a station keeps its recordings and calibration, configurable per machine."""

import os
from pathlib import Path

DATA_ROOT_ENV = "EGOCAP_DATA_DIR"
CALIBRATION_DIR_ENV = "EGOCAP_CALIBRATION_DIR"
_DEFAULT_DATA_ROOT = "~/egocap_data"


def data_root() -> Path:
    """Return the root directory recordings are written under.

    ``EGOCAP_DATA_DIR`` overrides the neutral ``~/egocap_data`` default so no
    machine-specific layout has to be baked into shipped configs. It only seeds
    the generated default config; an existing config's ``data_dir`` is used as written.
    """
    # An empty value is treated as unset: a blank export must not make the
    # root resolve to the current working directory.
    return Path(os.environ.get(DATA_ROOT_ENV) or _DEFAULT_DATA_ROOT).expanduser()


def calibration_dir() -> Path:
    """Return the directory holding this station's head-camera calibration JSONs.

    ``EGOCAP_CALIBRATION_DIR`` overrides the default ``<data root>/calibration``;
    calibration is per-station, so it is never read from inside the repo.
    """
    # Same empty-means-unset rule as data_root().
    override = os.environ.get(CALIBRATION_DIR_ENV)
    return Path(override).expanduser() if override else data_root() / "calibration"
