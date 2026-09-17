# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 RealHand. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pytest configuration and fixtures for isaaccapture.retargeting_engine tests."""

import pytest
import numpy as np
import yaml


@pytest.fixture
def realhand_calibration_dir(tmp_path):
    """Create deterministic synthetic FFG calibration files for RealHand tests."""
    calibration_dir = tmp_path / "realhand_ffg_glove"
    calibration_dir.mkdir()
    opened = np.zeros(21, dtype=np.float64)
    fist = np.linspace(0.8, 1.2, 21, dtype=np.float64)
    thumb_curl = opened.copy()
    thumb_curl[:5] = (0.25, 0.55, 0.85, 1.05, 0.70)
    touch_anchors = {
        "index": (0.30, 0.45, 0.60, 0.75, 0.90),
        "middle": (0.45, 0.60, 0.75, 0.90, 0.55),
        "ring": (0.60, 0.75, 0.90, 0.55, 0.70),
        "pinky": (0.75, 0.90, 0.55, 0.70, 0.85),
    }

    for model in ("l6", "o6", "l20"):
        data = {"model": model, "format": "realhand-ffg-glove-raw-calibration-v2"}
        for suffix in ("l", "r"):
            data[f"jointangleoriginal_{suffix}"] = opened.tolist()
            data[f"jointanglefist_{suffix}"] = fist.tolist()
            data[f"jointanglethumb_curl_{suffix}"] = thumb_curl.tolist()
            for finger, values in touch_anchors.items():
                pose = opened.copy()
                pose[:5] = values
                data[f"jointangleopose_{finger}_{suffix}"] = pose.tolist()
        with (calibration_dir / f"{model}.yml").open("w", encoding="utf-8") as stream:
            yaml.safe_dump(data, stream, sort_keys=False)

    return calibration_dir


@pytest.fixture
def float32_array():
    """Create a simple float32 numpy array."""
    return np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)


@pytest.fixture
def int32_array():
    """Create a simple int32 numpy array."""
    return np.array([1, 2, 3, 4, 5], dtype=np.int32)


@pytest.fixture
def uint8_array():
    """Create a simple uint8 numpy array."""
    return np.array([0, 127, 255], dtype=np.uint8)


@pytest.fixture
def matrix_3x3():
    """Create a 3x3 float32 matrix."""
    return np.array(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=np.float32
    )
