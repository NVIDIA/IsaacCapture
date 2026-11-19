# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The OVXR pin is declared for wheels, for the source tree and in the guide."""

import tomllib
from importlib import metadata
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[4]
_GUIDE = _ROOT / "docs/source/getting_started/ovxr.rst"


def _wheel_extra() -> list[str]:
    text = (_ROOT / "src/core/python/requirements-ovxr.txt").read_text()
    lines = (line.strip() for line in text.splitlines())
    return sorted(line for line in lines if line and not line.startswith("#"))


def _pinned(requirements: list[str], name: str) -> str:
    (version,) = (
        requirement.split("==")[1].split(";")[0].strip()
        for requirement in requirements
        if requirement.split("==")[0].strip() == name
    )
    return version


def test_ovxr_extra_matches_between_wheel_and_source_tree():
    project = tomllib.loads((_ROOT / "pyproject.toml").read_text())["project"]
    assert _wheel_extra() == sorted(project["optional-dependencies"]["ovxr"])


def test_guide_names_the_pinned_ovxr():
    assert f"``{_pinned(_wheel_extra(), 'ovxr')}``" in _GUIDE.read_text()


def test_guide_names_the_providers_the_pinned_ovxr_requires():
    pin = _pinned(_wheel_extra(), "ovxr")
    try:
        installed = metadata.version("ovxr")
    except metadata.PackageNotFoundError:
        pytest.skip("OVXR is not installed")
    assert installed == pin
    guide = _GUIDE.read_text()
    requirements = metadata.requires("ovxr") or []
    for provider in ("ovrtx", "ovstage"):
        assert f"``{_pinned(requirements, provider)}``" in guide
