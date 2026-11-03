# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pytest configuration and fixtures for isaaccapture.retargeting_engine tests."""

import pytest
import numpy as np


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
