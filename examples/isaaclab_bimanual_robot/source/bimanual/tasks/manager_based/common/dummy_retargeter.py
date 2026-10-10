# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import dataclass

import torch

import isaaclab.sim as sim_utils
from isaaclab.devices.retargeter_base import RetargeterBase, RetargeterCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg


class DummyRetargeter(RetargeterBase):
    """Dummy retargeter that maps motion controller inputs to robot commands.
    """

    def __init__(self, cfg: DummyRetargeterCfg):
        """Initialize the retargeter."""
        super().__init__(cfg)
        self._sim_device = cfg.sim_device
        self._hand_joint_names = cfg.hand_joint_names
        self._enable_visualization = cfg.enable_visualization

        if cfg.hand_joint_names is None:
            raise ValueError("hand_joint_names must be provided")

        # Initialize visualization if enabled
        if self._enable_visualization:
            marker_cfg = VisualizationMarkersCfg(
                prim_path="/Visuals/g1_controller_markers",
                markers={
                    "joint": sim_utils.SphereCfg(
                        radius=0.01,
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
                    ),
                },
            )
            self._markers = VisualizationMarkers(marker_cfg)

    def retarget(self, data: dict) -> torch.Tensor:
        """Convert controller inputs to robot commands.
        """
        pass

    def get_requirements(self) -> list[RetargeterBase.Requirement]:
        return [RetargeterBase.Requirement.MOTION_CONTROLLER]


@dataclass
class DummyRetargeterCfg(RetargeterCfg):
    """Configuration for the Dummy retargeter."""

    enable_visualization: bool = False
    hand_joint_names: list[str] | None = None  # List of robot hand joint names
    retargeter_type: type[RetargeterBase] = DummyRetargeter
