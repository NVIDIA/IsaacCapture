# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-OpenMDW-1.1

# Python 3.12 / NumPy 1.24+ compatibility shims for chumpy.
#
# chumpy (pinned at its last 2020 release) has two module-level loads that break on
# modern environments, both triggered the first time something imports chumpy:
#   1. chumpy/ch.py uses `inspect.getargspec` — removed in Python 3.11+.
#   2. chumpy/__init__.py does `from numpy import bool, int, float, complex, object, unicode, str`
#      — all of those were deprecated aliases for Python built-ins and removed in NumPy 1.24.
# All egohand entry points (inference pipeline, eval callbacks) import from egohand.* before any
# chumpy use, so patching here applies before the first chumpy load.
import builtins as _builtins
import inspect as _inspect

import numpy as _np

if not hasattr(_inspect, "getargspec"):
    _inspect.getargspec = _inspect.getfullargspec

for _attr, _val in (
    ("bool", _builtins.bool),
    ("int", _builtins.int),
    ("float", _builtins.float),
    ("complex", _builtins.complex),
    ("object", _builtins.object),
    ("str", _builtins.str),
    ("unicode", _builtins.str),
):
    if not hasattr(_np, _attr):
        setattr(_np, _attr, _val)
