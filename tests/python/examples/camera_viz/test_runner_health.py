# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Runtime-health monitoring tests for VizRunner."""

from __future__ import annotations

import pytest

from pipeline import VizRunner


class _AliveThread:
    def is_alive(self) -> bool:
        return True

    def join(self, timeout: float) -> None:
        assert timeout == 0.1


def test_wait_propagates_health_check_failure() -> None:
    runner = VizRunner(object(), [], [])
    runner._render_thread = _AliveThread()
    checks = 0

    def health_check() -> None:
        nonlocal checks
        checks += 1
        raise RuntimeError("CloudXR runtime exited")

    with pytest.raises(RuntimeError, match="CloudXR runtime exited"):
        runner.wait(health_check=health_check)

    assert checks == 1
