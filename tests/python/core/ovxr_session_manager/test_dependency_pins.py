# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dependencies are declared for wheels and for the source tree; pins also in the guide."""

import tomllib
from importlib import metadata
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[4]
_WHEEL = _ROOT / "src/core/python"
_GUIDE = _ROOT / "docs/source/getting_started/ovxr.rst"
_PROJECT = tomllib.loads((_ROOT / "pyproject.toml").read_text())["project"]


def _normalized(requirements) -> list[str]:
    return sorted(r.replace('"', "'").replace(" ", "") for r in requirements)


def _wheel_requirements(extra: str | None) -> list[str]:
    name = f"requirements-{extra}.txt" if extra else "requirements.txt"
    lines = (line.strip() for line in (_WHEEL / name).read_text().splitlines())
    return _normalized(line for line in lines if line and not line.startswith("#"))


def _pinned(requirements: list[str], name: str) -> str:
    (version,) = (
        requirement.split("==")[1].split(";")[0].strip()
        for requirement in requirements
        if requirement.split("==")[0].strip() == name
    )
    return version


def test_source_tree_and_wheel_declare_the_same_extras():
    wheel = {
        path.stem.removeprefix("requirements-")
        for path in _WHEEL.glob("requirements-*.txt")
    }
    assert wheel == set(_PROJECT["optional-dependencies"])


@pytest.mark.parametrize("extra", [None, *_PROJECT["optional-dependencies"]])
def test_source_tree_and_wheel_declare_the_same_requirements(extra):
    source = (
        _PROJECT["optional-dependencies"][extra] if extra else _PROJECT["dependencies"]
    )
    assert _normalized(source) == _wheel_requirements(extra)


def test_guide_names_the_pinned_ovxr_and_ovui():
    guide = _GUIDE.read_text()
    assert f"``{_pinned(_wheel_requirements('ovxr'), 'ovxr')}``" in guide
    assert f"OVUI {_pinned(_wheel_requirements('desktop-mirror'), 'ovui')} " in guide


def test_guide_names_the_providers_the_pinned_ovxr_requires():
    pin = _pinned(_wheel_requirements("ovxr"), "ovxr")
    try:
        installed = metadata.version("ovxr")
    except metadata.PackageNotFoundError:
        pytest.skip("OVXR is not installed")
    assert installed == pin
    guide = _GUIDE.read_text()
    requirements = metadata.requires("ovxr") or []
    for provider in ("ovrtx", "ovstage"):
        assert f"``{_pinned(requirements, provider)}``" in guide
