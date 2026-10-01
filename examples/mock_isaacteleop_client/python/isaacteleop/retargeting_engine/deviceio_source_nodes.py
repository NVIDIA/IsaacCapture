# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Source-node stand-ins a mock consumer constructs and wires into ``OutputCombiner``.

Per ``GumanIsaacTeleopUsage.md`` §4: a real consumer never touches these nodes' internals
(``IDeviceIOSource``/``ITracker``/``poll_tracker``/``_compute_fn``) -- it only constructs them
and calls ``.output(name)`` to get something to put in an ``OutputCombiner`` dict. So these are
deliberately thin: hold a name (and, for ``JointStateSource``, a joint count), and hand back an
``OutputSelector`` naming which synthetic shape ``TeleopSession.step()`` should produce for them.
"""

from .interface import OutputSelector


class _SourceNode:
    def __init__(self, name: str) -> None:
        self.name = name

    def output(self, output_name: str) -> OutputSelector:
        return OutputSelector(output_name)


class HeadSource(_SourceNode):
    def output(self, output_name: str = "head") -> OutputSelector:  # noqa: D102 -- trivial override
        return OutputSelector(output_name)


class ControllersSource(_SourceNode):
    LEFT = "controller_left"
    RIGHT = "controller_right"


class HandsSource(_SourceNode):
    LEFT = "hand_left"
    RIGHT = "hand_right"


class JointStateSource(_SourceNode):
    JOINTS = "joints"

    def __init__(self, name: str, collection_id: str, joint_names: list[str]) -> None:
        super().__init__(name)
        self.collection_id = collection_id
        self.joint_names = list(joint_names)

    def output(self, output_name: str = JOINTS) -> OutputSelector:  # noqa: D102
        # The collection_id, not "joints", is the OutputCombiner key Guman actually uses
        # (GumanIsaacTeleopUsage.md §3.2: outputs[collection_id] = source.output(JOINTS)) --
        # name the selector by collection_id so SyntheticPoseGenerator's joint_counts table
        # (built from every JointStateSource in the pipeline) lines up with it.
        return OutputSelector(self.collection_id, joint_count=len(self.joint_names))


class HapticSink(_SourceNode):
    """Reverse-channel stub -- see ``haptic_devices.glove`` for ``set_tactile``."""
