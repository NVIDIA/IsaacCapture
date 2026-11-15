# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""When conftest.py skips the tests marked "apai", and when it lets them fail."""

from __future__ import annotations

import sys
import types

import pytest

import conftest

SRC = "/somewhere/apai/src"


@pytest.fixture
def checkout(monkeypatch) -> None:
    """An APAI checkout that imports, with every package of the apai extra."""
    monkeypatch.setattr(conftest, "_apai_src", SRC)
    for name in ("apai", *conftest.APAI_EXTRA_MODULES):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))


def test_the_apai_tests_run_with_a_checkout_and_the_extra(checkout) -> None:
    assert conftest._apai_unavailable() is None


def test_without_the_variable_the_apai_tests_are_skipped(monkeypatch) -> None:
    monkeypatch.setattr(conftest, "_apai_src", None)
    assert conftest._apai_unavailable() == "AGENTIC_REMOTE_APAI_SRC is not set"


def test_the_extra_names_what_the_apai_tests_import_besides_apai() -> None:
    assert conftest.APAI_EXTRA_MODULES == ("pinocchio",)


@pytest.mark.parametrize("name", ["pinocchio"])
def test_without_a_package_of_the_extra_the_apai_tests_are_skipped(
    checkout, monkeypatch, name
) -> None:
    monkeypatch.setitem(sys.modules, name, None)  # not importable
    assert conftest._apai_unavailable() == (
        f"the apai extra of this folder's pyproject.toml is not installed: import "
        f"of {name} halted; None in sys.modules"
    )


def test_a_folder_without_apai_skips_the_apai_tests(checkout, monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "apai", None)
    assert conftest._apai_unavailable() == (
        f"AGENTIC_REMOTE_APAI_SRC={SRC} has no package apai"
    )


@pytest.mark.parametrize("broken", ["apai", "pinocchio"])
def test_an_import_that_fails_another_way_is_not_a_skip(
    checkout, monkeypatch, broken
) -> None:
    def fails(name: str):
        if name == broken:
            raise ImportError("libsomething.so: cannot open shared object file")
        return sys.modules[name]

    monkeypatch.setattr(conftest.importlib, "import_module", fails)
    # The tests marked "apai" then run and show the error.
    assert conftest._apai_unavailable() is None


def test_a_missing_module_that_apai_itself_imports_is_not_a_skip(
    checkout, monkeypatch
) -> None:
    def fails(name: str):
        if name == "apai":
            raise ModuleNotFoundError("No module named 'apai.types'", name="apai.types")
        return sys.modules[name]

    monkeypatch.setattr(conftest.importlib, "import_module", fails)
    assert conftest._apai_unavailable() is None
