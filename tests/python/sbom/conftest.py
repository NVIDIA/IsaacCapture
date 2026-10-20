# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fixtures for the SBOM collector tests."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import synth
from sbom import licensing


repo_root = synth.load(
    "repo_paths", Path(__file__).resolve().parents[1] / "repo_paths.py"
).repo_root

LICENSE_DATA_ENV = "ISAACCAPTURE_SBOM_LICENSE_DATA"
BUILD_DIR_ENV = "ISAACCAPTURE_SBOM_BUILD_DIR"


@pytest.fixture(scope="session")
def license_data():
    """The SPDX license list deps/third_party fetched into this build's _deps.

    CTest points the environment variable at it. Outside that, fall back to a
    default build directory; if neither exists the license tests skip rather
    than run against data that is not what the build pinned.
    """
    candidates = []
    if os.environ.get(LICENSE_DATA_ENV):
        candidates.append(Path(os.environ[LICENSE_DATA_ENV]))
    candidates.append(
        repo_root() / "build" / "_deps" / "license-list-data-src" / "json"
    )

    for candidate in candidates:
        try:
            licensing.load_corpus(candidate)
            return candidate
        except licensing.CorpusError:
            continue

    pytest.skip(
        "SPDX license list data not found; configure and build so deps/third_party "
        "materializes it, or set " + LICENSE_DATA_ENV
    )


@pytest.fixture(autouse=True)
def _license_data_loaded(license_data):
    """Load the reference texts before each test.

    Reloading matters: a `build()` call points the collector at its own build
    tree, which for the fixtures holds only a handful of licenses. Without this
    the next test would inherit that subset.
    """
    licensing.load_corpus(license_data)
    synth.set_license_data(license_data)


@pytest.fixture(scope="session")
def checkout():
    """This repository, as the collector would be pointed at it."""
    return repo_root()


@pytest.fixture(scope="session")
def live_build_dir():
    """This repository's own configured build tree, when there is one.

    The synthetic workspaces are written to a model of what CMake emits. This is
    the anchor to what it actually emits, so a change to the codemodel's shape
    fails here instead of passing everywhere.
    """
    candidates = []
    if os.environ.get(BUILD_DIR_ENV):
        candidates.append(Path(os.environ[BUILD_DIR_ENV]))
    candidates.append(repo_root() / "build")

    for candidate in candidates:
        if sorted(
            (candidate / ".cmake" / "api" / "v1" / "reply").glob("codemodel-v2-*.json")
        ):
            return candidate

    pytest.skip("no configured build tree to read; configure this project first")


@pytest.fixture
def workspace(tmp_path):
    """A generated project, its build tree, and the wheel that build produced."""
    space = synth.create(tmp_path / "project")
    synth.write_wheel(space.wheel, synth.wheel_payload(space))
    return space


@pytest.fixture
def built(workspace):
    """The workspace after the collector has packaged evidence into the wheel."""
    return synth.collect_and_bind(workspace)
