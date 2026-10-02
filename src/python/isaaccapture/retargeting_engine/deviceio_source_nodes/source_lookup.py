# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Locate source nodes inside a pipeline the caller did not build."""

from __future__ import annotations

from typing import Any, List, Type, TypeVar

T = TypeVar("T")


def find_sources(pipeline: Any, source_type: Type[T]) -> List[T]:
    """Return every leaf of ``pipeline`` that is a ``source_type``, once each, in leaf order.

    Lets a host reach, for example, the ``KeyboardSource`` inside a pipeline returned by a
    config's builder in order to attach its input surface. Only leaves reachable from the
    pipeline's outputs are found, the same set ``TeleopSession`` discovers.
    """
    found: List[T] = []
    seen: set[int] = set()
    for node in pipeline.get_leaf_nodes():
        if isinstance(node, source_type) and id(node) not in seen:
            seen.add(id(node))
            found.append(node)
    return found
