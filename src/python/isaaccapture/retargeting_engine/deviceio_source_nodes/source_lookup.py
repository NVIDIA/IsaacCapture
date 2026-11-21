# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Locate source nodes inside a pipeline the caller did not build."""

from __future__ import annotations

from typing import Any, List, Type, TypeVar

T = TypeVar("T")


def find_sources(pipeline: Any, source_type: Type[T]) -> List[T]:
    """Return every source of ``pipeline`` that is a ``source_type``, once each, in leaf order.

    Lets a host reach, for example, the ``KeyboardSource`` inside a pipeline returned by a
    config's builder in order to attach its input surface. Searches the sources
    ``TeleopSession`` discovers: the leaves reachable from the pipeline's outputs.
    """
    # Imported here: teleop_session_manager imports this package.
    from isaaccapture.teleop_session_manager.helpers import _get_sources_from_pipeline

    found: List[T] = []
    for source in _get_sources_from_pipeline(pipeline):
        if isinstance(source, source_type) and not any(f is source for f in found):
            found.append(source)
    return found
