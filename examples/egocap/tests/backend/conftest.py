# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Shared fixture for tests that need a real, built frontend ``dist/`` on
disk (as opposed to the API-only tests, which are the majority and need
nothing here)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# build_support.py lives at egocap/ root, one level above tests/ -- not a
# package, so it needs its directory on sys.path to import by name.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from build_support import RECORDINGS_FRONTEND_DIR, build_frontends  # noqa: E402


@pytest.fixture(autouse=True)
def _viser_off_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """The 3D view is on by default for a real deployment, but a test that doesn't care about it
    shouldn't pay for a real ViserServer -- set the opt-out unless the test itself overrides this
    env var (e.g. to exercise the view turned on)."""
    monkeypatch.setenv("EGOCAP_VISER_ENABLED", "0")


@pytest.fixture(scope="session")
def built_recordings_frontend_dist() -> Path:
    """Builds only ``recordings_frontend/`` via its real npm scripts once
    per test session, into its real (gitignored) ``dist/`` -- the same path
    ``recordings.py`` already looks for. Deliberately builds just this one
    frontend, not both: the only test that needs it doesn't touch the main
    ``frontend/``, so building that one too would make this test needlessly
    slower and fragile to a build failure in unrelated code. Only tests
    that actually serve a built page should request this; it requires
    Node/npm and costs real wall-clock time, so it must stay opt-in rather
    than running for every backend test."""
    (dist_dir,) = build_frontends((RECORDINGS_FRONTEND_DIR,))
    return dist_dir
