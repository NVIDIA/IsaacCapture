# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A failed CUDA producer must report its error to the consumer."""

import sys
from unittest.mock import MagicMock

import pytest

from sources.synthetic import SyntheticSource, SyntheticStereoSource


@pytest.mark.parametrize("source_type", [SyntheticSource, SyntheticStereoSource])
def test_cuda_failure_reaches_latest(monkeypatch, source_type):
    failure = RuntimeError("CUDA_ERROR_NO_BINARY_FOR_GPU")
    cp = MagicMock()
    cp.zeros.return_value.device.id = 0
    cp.arange.side_effect = failure
    monkeypatch.setitem(sys.modules, "cupy", cp)
    source = source_type("test-camera", 8, 4)
    try:
        source.start()
        source._thread.join(timeout=2)
        assert not source._thread.is_alive(), "producer did not terminate"
        with pytest.raises(RuntimeError, match="CUDA_ERROR_NO_BINARY_FOR_GPU") as error:
            source.latest()
        assert error.value.__cause__ is failure
    finally:
        source.stop()
