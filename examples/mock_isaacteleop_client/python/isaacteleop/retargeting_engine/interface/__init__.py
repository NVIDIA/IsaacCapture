# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``OutputCombiner``/``BaseRetargeter`` stand-ins -- see module docstring in
``deviceio_source_nodes.py`` for why these carry no real compute-graph machinery.
"""


class OutputSelector:
    """What ``SourceNode.output(name)`` returns: just enough for ``OutputCombiner`` to learn the
    pipeline's output names (and, for ``JointStateSource``, its joint count) from.
    """

    def __init__(self, name: str, *, joint_count: int | None = None) -> None:
        self.name = name
        self.joint_count = joint_count


class OutputCombiner:
    """Holds the ``outputs`` dict's keys/selectors -- all ``TeleopSession`` needs from a pipeline."""

    def __init__(self, outputs: dict[str, OutputSelector]) -> None:
        self.output_names = list(outputs.keys())
        self.joint_counts = {
            key: selector.joint_count
            for key, selector in outputs.items()
            if selector.joint_count is not None
        }


class BaseRetargeter:
    """Import-surface placeholder; Guman imports this name but never subclasses or calls it."""
