# SPDX-FileCopyrightText: Copyright (c) 2026 Chris von Csefalvay (HCLTech)
# SPDX-License-Identifier: Apache-2.0

"""dVRK Patient Side Manipulator retargeters."""

import importlib as _importlib

_RETARGETER_EXPORTS = {
    "DVRKPSMClutchConfig",
    "DVRKPSMClutchRetargeter",
    "DVRKPSMGripperConfig",
    "DVRKPSMGripperRetargeter",
}


def __getattr__(name: str):
    """Load engine-dependent retargeter wrappers only when requested."""
    if name not in _RETARGETER_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(_importlib.import_module(".psm_retargeter", __package__), name)
    globals()[name] = value
    return value


__all__ = [
    "DVRKPSMClutchConfig",
    "DVRKPSMClutchRetargeter",
    "DVRKPSMGripperConfig",
    "DVRKPSMGripperRetargeter",
]
