# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The 26-joint hand layout, in ``core.HandJoint`` / OpenXR ``XrHandJointEXT`` order.

Only 21 joints are judged: the layout the Wuji, Sharpa and dex-retargeting retargeters
consume. PALM and the four finger METACARPALs may be invalid.
"""

from __future__ import annotations

JOINT_NAMES: tuple[str, ...] = (
    "PALM",
    "WRIST",
    "THUMB_METACARPAL",
    "THUMB_PROXIMAL",
    "THUMB_DISTAL",
    "THUMB_TIP",
    "INDEX_METACARPAL",
    "INDEX_PROXIMAL",
    "INDEX_INTERMEDIATE",
    "INDEX_DISTAL",
    "INDEX_TIP",
    "MIDDLE_METACARPAL",
    "MIDDLE_PROXIMAL",
    "MIDDLE_INTERMEDIATE",
    "MIDDLE_DISTAL",
    "MIDDLE_TIP",
    "RING_METACARPAL",
    "RING_PROXIMAL",
    "RING_INTERMEDIATE",
    "RING_DISTAL",
    "RING_TIP",
    "LITTLE_METACARPAL",
    "LITTLE_PROXIMAL",
    "LITTLE_INTERMEDIATE",
    "LITTLE_DISTAL",
    "LITTLE_TIP",
)
INDEX = {name: i for i, name in enumerate(JOINT_NAMES)}

PALM = 0
WRIST = 1

FINGERS: tuple[str, ...] = ("index", "middle", "ring", "little")

# Judged joints of each digit, root to tip. The thumb's metacarpal is judged; the
# fingers' are not.
CHAINS: dict[str, tuple[int, ...]] = {
    "thumb": (2, 3, 4, 5),
    "index": (7, 8, 9, 10),
    "middle": (12, 13, 14, 15),
    "ring": (17, 18, 19, 20),
    "little": (22, 23, 24, 25),
}
METACARPALS: dict[str, int] = {"index": 6, "middle": 11, "ring": 16, "little": 21}
PROXIMAL = {finger: chain[0] for finger, chain in CHAINS.items()}
TIP = {digit: chain[-1] for digit, chain in CHAINS.items()}

OPTIONAL: frozenset[int] = frozenset((PALM, *METACARPALS.values()))
REQUIRED: tuple[int, ...] = tuple(i for i in range(26) if i not in OPTIONAL)


def judged_bones() -> tuple[tuple[int, int], ...]:
    """Parent/child pairs between judged joints, wrist outward."""
    bones = []
    for chain in CHAINS.values():
        bones.append((WRIST, chain[0]))
        bones.extend(zip(chain, chain[1:]))
    return tuple(bones)


def phalanges(finger: str) -> tuple[tuple[int, int], ...]:
    chain = CHAINS[finger]
    return tuple(zip(chain, chain[1:]))


class HandProfile:
    """What the shared skeleton renderer reads."""

    joint_names = JOINT_NAMES

    @staticmethod
    def bones() -> tuple[tuple[str, str], ...]:
        pairs = list(judged_bones())
        for finger, metacarpal in METACARPALS.items():
            pairs.remove((WRIST, PROXIMAL[finger]))
            pairs += [(WRIST, metacarpal), (metacarpal, PROXIMAL[finger])]
        return tuple((JOINT_NAMES[a], JOINT_NAMES[b]) for a, b in pairs)


HAND = HandProfile()
