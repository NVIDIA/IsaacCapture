# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fixtures for the SBOM collector tests."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import synth
from isaaccapture_sbom import discovery, licensing

LICENSE_DATA_ENV = "ISAACCAPTURE_SBOM_LICENSE_DATA"
BUILD_DIR_ENV = "ISAACCAPTURE_SBOM_BUILD_DIR"


def repo_root() -> Path | None:
    """The source tree these tests sit in: the first ancestor with VERSION and CMakeLists.txt."""
    for parent in Path(__file__).resolve().parents:
        if (parent / "VERSION").is_file() and (parent / "CMakeLists.txt").is_file():
            return parent
    return None


def _candidates(env: str, *in_checkout: str):
    """The environment override, then the checkout's default, asked lazily."""
    if os.environ.get(env):
        yield Path(os.environ[env])
    root = repo_root()
    if root is not None:
        yield root.joinpath(*in_checkout)


@pytest.fixture(scope="session")
def license_data():
    """The SPDX license list deps/third_party fetched into this build's _deps.

    CTest points the environment variable at it. Outside that, fall back to a
    default build directory; if neither exists the license tests skip rather
    than run against data that is not what the build pinned.
    """
    for candidate in _candidates(
        LICENSE_DATA_ENV, "build", "_deps", "license-list-data-src", "json"
    ):
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


@pytest.fixture(autouse=True)
def _fresh_checkout_answers():
    """Each test builds its own checkouts and fakes its own host; a cached
    answer is a previous test's."""
    discovery._is_own_checkout.cache_clear()
    discovery._apt_releases.cache_clear()
    discovery._apt_origin.cache_clear()


@pytest.fixture(scope="session")
def checkout():
    """This repository, as the collector would be pointed at it."""
    root = repo_root()
    if root is None:
        pytest.skip("not inside this repository's source tree")
    return root


@pytest.fixture(scope="session")
def live_build_dir():
    """This repository's own configured build tree, when there is one.

    The synthetic workspaces are written to a model of what CMake emits. This is
    the anchor to what it actually emits, so a change to the codemodel's shape
    fails here instead of passing everywhere.
    """
    for candidate in _candidates(BUILD_DIR_ENV, "build"):
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
