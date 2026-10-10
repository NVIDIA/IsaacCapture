# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The implementation wheel owns both import names; the transition owns no code.

setuptools treats a package-data or `include` key that matches nothing as
contributing nothing, so a renamed key that stopped matching drops files -- or
the whole tree -- from the wheel with no error at all.
"""

from __future__ import annotations

import os
import zipfile
from email.parser import BytesParser
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from repo_paths import repo_root

DIST = "isaaccapture"
ALIAS = "isaacteleop"  # the shim from src/compat; TODO(1.9): drop it
VENDORED = "robotic_grounding"  # curated V2D package, vendored and not renamed


def _wheel(dist: str = DIST) -> Path:
    wheels_dir = os.environ.get("ISAAC_TELEOP_WHEELS_DIR")
    if not wheels_dir:
        pytest.skip("ISAAC_TELEOP_WHEELS_DIR is unset; run this through ctest")
    found = list(Path(wheels_dir).glob(f"{dist}-*.whl"))
    assert found, f"no {dist} wheel in {wheels_dir}"
    # Newest, not "exactly one": nothing cleans wheels/, so in CI's dev shape --
    # where the version carries the commit count -- every commit leaves another
    # wheel in an incremental build dir.
    return max(found, key=lambda wheel: wheel.stat().st_mtime)


def _top_level_packages(wheel: Path) -> set[str]:
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
    return {
        name.split("/", 1)[0]
        for name in names
        if "/" in name and not name.split("/", 1)[0].endswith((".dist-info", ".data"))
    }


def test_wheel_ships_exactly_the_expected_packages() -> None:
    wheel = _wheel()
    assert _top_level_packages(wheel) == {DIST, VENDORED}


def test_package_tree_is_not_empty() -> None:
    wheel = _wheel()
    with zipfile.ZipFile(wheel) as archive:
        owned = [n for n in archive.namelist() if n.startswith(f"{DIST}/")]
    assert f"{DIST}/__init__.py" in owned
    assert any(n.endswith((".so", ".pyd")) for n in owned), (
        "no extension module in the wheel: the staged tree or a package-data key "
        "stopped matching"
    )


def test_chromium_notice_ships_beside_the_key_table() -> None:
    """_deviceio_trackers compiles in Chromium's BSD-3-Clause key table; its notice must ship."""
    vendored = (repo_root() / "deps/third_party/chromium/LICENSE").read_bytes()
    with zipfile.ZipFile(_wheel()) as archive:
        shipped = archive.read(f"{DIST}/deviceio_trackers/CHROMIUM_LICENSE")
    assert shipped == vendored


def test_wheel_does_not_ship_cts() -> None:
    wheel = _wheel()
    with zipfile.ZipFile(wheel) as archive:
        shipped = [n for n in archive.namelist() if n.startswith(f"{DIST}/cts/")]
    assert shipped == []


def test_the_shim_ships_beside_the_real_package() -> None:
    wheel = _wheel()
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
    assert f"{ALIAS}.py" in names
    assert not any(n.startswith(f"{ALIAS}/") for n in names)


def test_transition_replaces_the_legacy_distribution_without_owning_code() -> None:
    metadata = {}
    for name in (DIST, ALIAS):
        with zipfile.ZipFile(_wheel(name)) as archive:
            paths = archive.namelist()
            entry = next(p for p in paths if p.endswith(".dist-info/METADATA"))
            metadata[name] = BytesParser().parsebytes(archive.read(entry))
            if name == ALIAS:
                assert all(".dist-info/" in p for p in paths)

    for name, dependency in ((DIST, ALIAS), (ALIAS, DIST)):
        requirement = next(
            r
            for r in map(Requirement, metadata[name].get_all("Requires-Dist", []))
            if r.name == dependency and r.marker is None
        )
        assert metadata[dependency]["Version"] in requirement.specifier
        assert "1.5.0" not in requirement.specifier
        if name == DIST:
            assert "1.6.999a1" not in requirement.specifier

    assert set(metadata[ALIAS].get_all("Provides-Extra", [])) == set(
        metadata[DIST].get_all("Provides-Extra", [])
    )
