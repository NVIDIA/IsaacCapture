# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A teleoperated robot's digital twin, and the affordances that depend on it.

Not a general scene-graph or viewer API. The scene backend sits behind
:class:`RobotTwin`, and only :mod:`.scene` and :mod:`.frames` reach it; everything else
uses numpy and duck typing, with the CUDA guide loaded only when rendering. The scene
backend is Linux-only (its OpenGL context is EGL), so they resolve lazily -- this package
still imports on a Windows Televiz build, and asking for :class:`SceneTwin` there raises.
"""

from typing import TYPE_CHECKING

from .clutch_phase import ClutchPhase
from .harness import InterventionMonitor
from .operator_frame import OperatorFrame
from .engage_gate import EngageGate, EngageGateConfig, GateVerdict
from .frame_info import head_pose
from .session import VIEW_COUNT, XrTwinSession
from .twin import RobotTwinPublisher

# Only for type checkers and IDEs, which cannot see through __getattr__ and otherwise
# report every lazy export as undefined. False at runtime, so the laziness below -- and
# with it the Windows import -- is unaffected.
if TYPE_CHECKING:
    from . import assets, clutch_preview, frames, so101_ghost
    from .clutch_preview import ClutchPreview
    from .preview_arm import PREVIEW_ARMS, PreviewArm, PreviewArmProfile
    from .scene import SceneTwin

# Resolved on first access: everything below reaches `frames` or `scene`, and those need
# the compiled `_robot_twin` a Windows Televiz build has no copy of.
_LAZY = {
    "ClutchPreview": ".clutch_preview",
    "PREVIEW_ARMS": ".preview_arm",
    "PreviewArm": ".preview_arm",
    "PreviewArmProfile": ".preview_arm",
    "SceneTwin": ".scene",
    "assets": ".assets",
    "clutch_preview": ".clutch_preview",
    "frames": ".frames",
    "preview_arm": ".preview_arm",
    "scene": ".scene",
    "so101_ghost": ".so101_ghost",
}
#: Which of those names are the module itself rather than something inside it. Spelled
#: out rather than inferred from the case: `deflection` is a lowercase function.
_LAZY_MODULES = frozenset(
    {"assets", "clutch_preview", "frames", "preview_arm", "scene", "so101_ghost"}
)


def __getattr__(name: str):
    module_path = _LAZY.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    try:
        module = importlib.import_module(module_path, __name__)
    except ImportError as error:
        # Only a missing backend gets re-labelled; a typo in a relative import or a
        # broken sibling must surface as itself.
        if "_robot_twin" not in str(error):
            raise
        raise ImportError(
            f"isaacteleop.viz.robot.{name} needs the compiled scene backend, which this "
            "build does not have. It ships with Televiz on Linux: build with "
            "-DBUILD_VIZ=ON."
        ) from error
    value = module if name in _LAZY_MODULES else getattr(module, name)
    globals()[name] = value
    return value


# The supported surface: what `examples/robot_viz`, `teleop_session_manager.twin_runner`
# and LeRobot's `isaac_teleop_to_so101` import. Everything else this package defines is
# an implementation detail -- still reachable from its own module, but not a promise.
__all__ = [
    "VIEW_COUNT",
    "ClutchPhase",
    "ClutchPreview",
    "EngageGate",
    "EngageGateConfig",
    "GateVerdict",
    "InterventionMonitor",
    "OperatorFrame",
    "PREVIEW_ARMS",
    "PreviewArm",
    "PreviewArmProfile",
    "RobotTwinPublisher",
    "SceneTwin",
    "XrTwinSession",
    "assets",
    "clutch_preview",
    "frames",
    "head_pose",
    "so101_ghost",
]
