# SPDX-FileCopyrightText: Copyright (c) 2026 RealHand. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""RealHand P7 arm and L6/O6/L20 hand retargeters."""

import importlib

_EXPORT_MODULES = {
    "arm": (
        "P7ControllerPoseRetargeter",
        "P7HandPoseRetargeter",
        "P7WorkspacePoseConfig",
    ),
    "hand": (
        "ControllerTriggerRealHandRetargeter",
        "ControllerTriggerRealHandRetargeterConfig",
    ),
    "profiles": (
        "L6_PROFILE",
        "L20_PROFILE",
        "O6_PROFILE",
        "REALHAND_PROFILES",
        "RealHandJointSpec",
        "RealHandProfile",
        "get_realhand_profile",
    ),
}


def __getattr__(name: str):
    for module_name, exports in _EXPORT_MODULES.items():
        if name in exports:
            try:
                module = importlib.import_module(f".{module_name}", __name__)
            except ModuleNotFoundError as error:
                if (
                    module_name == "arm"
                    and error.name
                    and error.name.startswith("scipy")
                ):
                    raise ModuleNotFoundError(
                        f"{name} requires scipy. Install it with: pip install 'isaaccapture[retargeters-lite]'"
                    ) from error
                raise
            value = getattr(module, name)
            globals()[name] = value
            return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ControllerTriggerRealHandRetargeter",
    "ControllerTriggerRealHandRetargeterConfig",
    "L6_PROFILE",
    "L20_PROFILE",
    "O6_PROFILE",
    "P7ControllerPoseRetargeter",
    "P7HandPoseRetargeter",
    "P7WorkspacePoseConfig",
    "REALHAND_PROFILES",
    "RealHandJointSpec",
    "RealHandProfile",
    "get_realhand_profile",
]
