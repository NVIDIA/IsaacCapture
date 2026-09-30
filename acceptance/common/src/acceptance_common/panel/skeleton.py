# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Draws one skeleton in a viser scene. The only module here that imports viser."""

from __future__ import annotations

from typing import Protocol

import numpy as np
import viser

from .sample import Sample

LIVE_JOINT_COLOUR = (86, 196, 138)
HELD_JOINT_COLOUR = (214, 74, 62)
BONE_COLOUR = (150, 158, 172)


class Topology(Protocol):
    @property
    def joint_names(self) -> tuple[str, ...]: ...

    def bones(self) -> tuple[tuple[int, int], ...]: ...


class Skeleton:
    """Joint cloud, bones, and a name floating beside every joint that is not live.

    Takes the topology rather than a recorded track so a live panel, which has no track
    and never will, can draw the same skeleton from the same code. ``root`` is the
    scene path, so two skeletons (a pair of hands) can share one scene.
    """

    def __init__(
        self,
        server: viser.ViserServer,
        profile: Topology,
        root: str = "/body",
        point_size: float = 0.022,
    ) -> None:
        self._names = profile.joint_names
        self._bones = profile.bones()
        self._points = server.scene.add_point_cloud(
            f"{root}/joints",
            points=np.zeros((0, 3), np.float32),
            colors=np.zeros((0, 3), np.uint8),
            point_size=point_size,
            point_shape="circle",
        )
        self._lines = server.scene.add_line_segments(
            f"{root}/bones",
            points=np.zeros((0, 2, 3), np.float32),
            colors=np.zeros((0, 2, 3), np.uint8),
            thickness=2.5,
            thickness_units="screen",
        )
        self._labels = [
            server.scene.add_label(f"{root}/name/{name}", name, visible=False)
            for name in self._names
        ]
        self._named: set[int] = set()

    def draw(self, sample: Sample, name_held: bool) -> None:
        points: list[tuple[float, float, float]] = []
        colours: list[tuple[int, int, int]] = []
        for index, position in enumerate(sample.positions):
            if position is None:
                continue
            points.append(position)
            colours.append(
                LIVE_JOINT_COLOUR if sample.valid[index] else HELD_JOINT_COLOUR
            )
        self._points.points = np.asarray(points, np.float32).reshape(-1, 3)
        self._points.colors = np.asarray(colours, np.uint8).reshape(-1, 3)

        segments: list[tuple] = []
        segment_colours: list[tuple] = []
        for parent, child in self._bones:
            start, end = sample.positions[parent], sample.positions[child]
            if start is None or end is None:
                continue
            live = sample.valid[parent] and sample.valid[child]
            colour = BONE_COLOUR if live else HELD_JOINT_COLOUR
            segments.append((start, end))
            segment_colours.append((colour, colour))
        self._lines.points = np.asarray(segments, np.float32).reshape(-1, 2, 3)
        self._lines.colors = np.asarray(segment_colours, np.uint8).reshape(-1, 2, 3)

        wanted = (
            {
                index
                for index, position in enumerate(sample.positions)
                if position is not None and not sample.valid[index]
            }
            if name_held
            else set()
        )
        for index in wanted:
            position = sample.positions[index]
            assert position is not None
            self._labels[index].position = position
        for index in wanted - self._named:
            self._labels[index].visible = True
        for index in self._named - wanted:
            self._labels[index].visible = False
        self._named = wanted

    def held_names(self, sample: Sample) -> list[str]:
        return [
            self._names[index]
            for index, position in enumerate(sample.positions)
            if position is not None and not sample.valid[index]
        ]
