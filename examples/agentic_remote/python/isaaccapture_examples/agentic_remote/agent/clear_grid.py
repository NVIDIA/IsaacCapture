# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The clear grid: a stand-in for the grid bridge that reports nothing near.

A grid bridge is the program that keeps the obstacle grid and answers APAI's
obstacle map. With experiment.grid: clear the launcher puts ClearGrid in place
of APAI's client of that bridge while APAI builds its runtime. APAI's obstacle
map, the check ahead and the PERMITs then run as with a bridge, but no move is
checked against obstacles: for the one-machine demo on a simulated or mock arm
only (config.check_robot_driver refuses any other driver).

numpy and APAI are imported inside the methods, so the package imports without
them.
"""

from __future__ import annotations

BACKEND = "clear"  # what the probe names as the bridge's backend
# The grid: CELLS cubes of VOXEL_M along each axis, from ORIGIN_M on each axis.
VOXEL_M = 0.05
CELLS = 80
ORIGIN_M = -2.0
# The distance from every cube to the nearest obstacle.
DISTANCE_M = 1.0


class ClearGrid:
    """In place of APAI's OccupancyClient: a distance grid with nothing near."""

    def __init__(self, host: str = "", port: int = 0, timeout_ms: int = 0) -> None:
        """Takes what APAI gives its client; nothing is dialed."""

    def request(self, payload: dict, timeout_ms: int | None = None) -> dict:
        """The answer a bridge with an empty map gives to this action.

        APAI's OccupancyError for an action it does not know, as an old bridge
        answers: APAI then falls back to an action it knows.
        """
        import numpy as np

        action = payload["action"]
        if action == "probe":
            return {
                "ok": True,
                "backend": BACKEND,
                "device": "cpu",
                "voxel": VOXEL_M,
                "esdf": True,
            }
        if action == "query":
            # A new grid for each query: no reader can change the next answer.
            return {
                "points": np.empty((0, 3), dtype=np.float32),
                "grid": np.full((CELLS, CELLS, CELLS), DISTANCE_M, dtype=np.float32),
                "origin": np.full(3, ORIGIN_M),
                "voxel": VOXEL_M,
            }
        if action in ("integrate", "integrate_depth", "clear"):
            return {}
        from apai.perception.occupancy import OccupancyError

        raise OccupancyError(f"occupancy bridge error: unknown action {action!r}")

    def probe(self, timeout_ms: int = 300) -> dict:
        return self.request({"action": "probe"})

    def close(self) -> None:
        """Nothing is open."""
