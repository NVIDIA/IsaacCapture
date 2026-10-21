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


def _located(env: str, usable, missing: str, *in_checkout: str) -> Path:
    """The path `env` names, else the checkout's default, if `usable` accepts it.

    A path the environment names is one CTest promised, so its absence fails
    every test; without one, the tests skip, since nothing was promised.
    """
    if os.environ.get(env):
        named = Path(os.environ[env])
        if usable(named):
            return named
        pytest.fail(f"{env}={named}: {missing}")
    root = repo_root()
    if root is not None and usable(root.joinpath(*in_checkout)):
        return root.joinpath(*in_checkout)
    pytest.skip(f"{missing}; configure this project, or set {env}")


def _corpus_loads(path: Path) -> bool:
    try:
        licensing.load_corpus(path)
    except licensing.CorpusError:
        return False
    return True


def _has_codemodel(path: Path) -> bool:
    return any((path / ".cmake" / "api" / "v1" / "reply").glob("codemodel-v2-*.json"))


@pytest.fixture(scope="session")
def license_data():
    """The SPDX license list deps/third_party fetched into this build's _deps."""
    return _located(
        LICENSE_DATA_ENV,
        _corpus_loads,
        "no SPDX license list data there",
        "build",
        "_deps",
        "license-list-data-src",
        "json",
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
    return _located(
        BUILD_DIR_ENV, _has_codemodel, "no CMake file API codemodel there", "build"
    )


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
