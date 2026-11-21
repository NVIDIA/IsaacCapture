# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The take's wrist-source label, read from the recorded plugins' environment.

A glove plugin that picks its wrist source from the environment reads
``NAME_WRIST_SOURCE``: its ``plugin.yaml`` name upper-cased, without a trailing
``_plugin``. The label only records that choice.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping


def variable(plugin: str) -> str:
    """``wuji_glove_plugin`` -> ``WUJI_GLOVE_WRIST_SOURCE``."""
    return f"{plugin.removesuffix('_plugin').upper()}_WRIST_SOURCE"


def read(plugins: Iterable[str], environ: Mapping[str, str]) -> dict[str, str | None]:
    """Each recorded plugin's variable and its value, None when unset."""
    return {variable(name): environ.get(variable(name)) for name in plugins}


def label(env: Mapping[str, str | None]) -> str | None:
    """The one wrist source the plugins were given; ValueError when they disagree."""
    found = {name: value for name, value in env.items() if value}
    if len(set(found.values())) > 1:
        given = ", ".join(f"{name}={value}" for name, value in found.items())
        raise ValueError(f"the plugins' wrist sources disagree ({given})")
    return next(iter(found.values()), None)
