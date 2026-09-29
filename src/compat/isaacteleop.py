# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compatibility alias: ``isaacteleop`` is ``isaaccapture``.

Ships inside the isaaccapture wheel, so the two are never installed apart. Every
``isaacteleop.X`` is the ``isaaccapture.X`` module object itself, so pybind11 type
registrations are not duplicated. See docs/source/references/migration.rst.
TODO(1.9): delete this module.
"""

from __future__ import annotations

import copy
import importlib
import importlib.abc
import importlib.machinery
import importlib.util
import sys
import warnings
from types import CodeType, ModuleType
from typing import Any

_OLD_PACKAGE = "isaacteleop"
_NEW_PACKAGE = "isaaccapture"

_DEPRECATION = (
    "The 'isaacteleop' import package was renamed to 'isaaccapture' in Isaac Teleop "
    "1.6; import 'isaaccapture' instead. This alias ships inside the isaaccapture "
    "wheel and will be removed in 1.9. "
    "See https://nvidia.github.io/IsaacCapture/main/references/migration.html"
)


def _renamed(fullname: str) -> str:
    return _NEW_PACKAGE + fullname[len(_OLD_PACKAGE) :]


def __getattr__(name: str) -> Any:
    # Defined before the body runs: a holder of this module pre-rebind lands here.
    return getattr(importlib.import_module(_NEW_PACKAGE), name)


class _AliasLoader(importlib.abc.Loader):
    """Binds the old name to the module object the new name already resolves to."""

    def __init__(self, real: importlib.machinery.ModuleSpec) -> None:
        self._real = real

    def exec_module(self, module: ModuleType) -> None:
        # _bootstrap._load re-reads sys.modules[spec.name] after exec_module.
        sys.modules[module.__name__] = importlib.import_module(
            _renamed(module.__name__)
        )

    def get_code(self, fullname: str) -> CodeType | None:
        # runpy asks the loader, not the module, for `python -m isaacteleop.X`.
        return self._real.loader.get_code(_renamed(fullname))


class _AliasFinder(importlib.abc.MetaPathFinder):
    """Serves every isaacteleop.* name from the isaaccapture spec."""

    def find_spec(self, fullname, path=None, target=None):
        # Exact-or-dotted: `isaacteleop_examples` would otherwise prefix-match.
        if fullname != _OLD_PACKAGE and not fullname.startswith(_OLD_PACKAGE + "."):
            return None
        real = importlib.util.find_spec(_renamed(fullname))
        if real is None:
            return None
        # copy.copy carries submodule_search_locations, which runpy reads to find
        # X.__main__ and cloudxr.runtime reads to find the vendored libcloudxr.so.
        spec = copy.copy(real)
        spec.name = fullname
        spec.loader = _AliasLoader(real)
        return spec


def _install() -> None:
    # Before PathFinder: it would otherwise reach the aliased parent's real __path__
    # and load the pybind11 extensions twice, aborting on a duplicate registration.
    sys.meta_path.insert(0, _AliasFinder())


# Warn before installing the finder so -W error cannot leave an alias behind.
warnings.warn(_DEPRECATION, DeprecationWarning, stacklevel=2)
_install()
# Without this rebind, sys.modules keeps the shim under the old name.
sys.modules[__name__] = importlib.import_module(_NEW_PACKAGE)
