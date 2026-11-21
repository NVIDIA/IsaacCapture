# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fixtures for the SBOM collector tests."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import synth
from isaaccapture_sbom import build, discovery, licensing

BUILD_DIR_ENV = "ISAACCAPTURE_SBOM_BUILD_DIR"


def repo_root() -> Path | None:
    for parent in Path(__file__).resolve().parents:
        if (parent / "VERSION").is_file() and (parent / "CMakeLists.txt").is_file():
            return parent
    return None


def _located(usable, missing: str, within=lambda build_dir: build_dir) -> Path:
    """Use the selected build tree or default build/.

    Missing explicit inputs fail; missing defaults skip.
    """
    if os.environ.get(BUILD_DIR_ENV):
        named = within(Path(os.environ[BUILD_DIR_ENV]).resolve())
        if usable(named):
            return named
        pytest.fail(f"{BUILD_DIR_ENV}: {missing} at {named}")
    root = repo_root()
    if root is not None and usable(found := within(root / "build")):
        return found
    pytest.skip(f"{missing}; configure this project, or set {BUILD_DIR_ENV}")


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
        _corpus_loads, "no SPDX license list data", build.default_license_data
    )


@pytest.fixture(autouse=True)
def _license_data_loaded(license_data):
    """Restore the full corpus after tests that load a fixture subset."""
    licensing.load_corpus(license_data)
    synth.set_license_data(license_data)


@pytest.fixture(autouse=True)
def _fresh_checkout_answers():
    """Clear cached checkout and host answers between synthetic workspaces."""
    discovery._is_own_checkout.cache_clear()
    discovery._apt_releases.cache_clear()
    discovery._apt_origin.cache_clear()


@pytest.fixture(scope="session")
def checkout():
    root = repo_root()
    if root is None:
        pytest.skip("not inside this repository's source tree")
    return root


@pytest.fixture(scope="session")
def live_build_dir():
    """Use real CMake output to validate the synthetic codemodel assumptions."""
    return _located(_has_codemodel, "no CMake file API codemodel")


@pytest.fixture
def workspace(tmp_path):
    """A generated project, its build tree, and the wheel that build produced."""
    space = synth.create(tmp_path / "project")
    synth.write_wheel(space.wheel, synth.wheel_payload(space))
    return space


@pytest.fixture
def built(workspace):
    return synth.collect_and_bind(workspace)
