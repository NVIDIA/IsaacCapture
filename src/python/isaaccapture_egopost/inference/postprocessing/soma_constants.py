# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-OpenMDW-1.1

"""The SOMAHand 25-joint skeleton: one definition, for the whole package.

Matches ``soma.hand.SOMAHandLayer`` from the PyPI distribution ``py-soma-x``
(version pinned in ``requirements.txt``). The layer exposes the topology
as ``joint_parent_ids`` and the joint order as ``rig_data["joint_names"]``.
These constants are a static copy of that 25-joint contract so standalone QC
can run without importing SOMA/PyTorch or loading the model assets.

Deliberately dependency-free -- no numpy, no OpenCV, no torch -- so any module
can import the topology without inheriting the imports of whatever else needs
it. Consumers build their own arrays:

    BONE_CHILD = np.arange(1, SOMA25_N_JOINTS)
    BONE_PARENT = np.asarray(SOMA25_PARENT)[BONE_CHILD]
"""

#: Parent of each joint; the wrist root is normalized to ``-1``.
#: Order: wrist, thumb 1-4, then index / middle / ring / pinky, each as
#: metacarpal + three phalanges + tip.
SOMA25_PARENT = [
    -1,
    0,
    1,
    2,
    3,
    0,
    5,
    6,
    7,
    8,
    0,
    10,
    11,
    12,
    13,
    0,
    15,
    16,
    17,
    18,
    0,
    20,
    21,
    22,
    23,
]

SOMA25_N_JOINTS = len(SOMA25_PARENT)

#: ``(child, parent)`` for all 24 bones: every joint but the wrist.
SOMA25_BONES = tuple((j, SOMA25_PARENT[j]) for j in range(1, SOMA25_N_JOINTS))

#: Named joints. The five ``*_MCP`` names follow SOMAHandLayer's labels
#: (``Thumb1``, ``Index1``, ...): for the four fingers these sit at the
#: metacarpal, one joint proximal of where MANO puts its knuckle.
SOMA25_WRIST = 0
SOMA25_THUMB_MCP = 1
SOMA25_INDEX_MCP = 5
SOMA25_MIDDLE_MCP = 10
SOMA25_RING_MCP = 15
SOMA25_LITTLE_MCP = 20

#: The five in palm order, for anything that spans the whole hand.
SOMA25_MCP = [
    SOMA25_THUMB_MCP,
    SOMA25_INDEX_MCP,
    SOMA25_MIDDLE_MCP,
    SOMA25_RING_MCP,
    SOMA25_LITTLE_MCP,
]

#: Leaf of each finger: no child to point at, so direction comes from the parent.
SOMA25_FINGERTIPS = {4, 9, 14, 19, 24}
