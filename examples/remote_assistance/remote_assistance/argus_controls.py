# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate camera tuning independently of GStreamer and sensor drivers."""

import math

RANGES = {
    "exposure_time_us": "exposuretimerange",
    "gain_range": "gainrange",
    "isp_digital_gain_range": "ispdigitalgainrange",
}
SCALARS = {
    "exposure_compensation": ("exposurecompensation", -2, 2),
    "saturation": ("saturation", 0, 2),
    "noise_reduction_strength": ("tnr-strength", -1, 1),
    "edge_enhancement_strength": ("ee-strength", -1, 1),
}
ENUMS = {
    "white_balance": (
        "wbmode",
        {
            name: i
            for i, name in enumerate(
                (
                    "off",
                    "auto",
                    "incandescent",
                    "fluorescent",
                    "warm-fluorescent",
                    "daylight",
                    "cloudy-daylight",
                    "twilight",
                    "shade",
                )
            )
        },
    ),
    "antibanding": ("aeantibanding", {"off": 0, "auto": 1, "50hz": 2, "60hz": 3}),
    "noise_reduction": ("tnr-mode", {"off": 0, "fast": 1, "high-quality": 2}),
    "edge_enhancement": ("ee-mode", {"off": 0, "fast": 1, "high-quality": 2}),
}
LOCKS = {"ae_lock": "aelock", "awb_lock": "awblock"}


def number(value, name):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return value


def argus_properties(controls, *, fps=None):
    """Null means no override; exposure uses microseconds in YAML, nanoseconds in Argus."""
    if not isinstance(controls, dict):
        raise ValueError("Argus controls must be a mapping")
    unknown = controls.keys() - (
        RANGES.keys() | SCALARS.keys() | ENUMS.keys() | LOCKS.keys()
    )
    if unknown:
        raise ValueError(f"Unknown Argus controls: {sorted(map(str, unknown))}")
    result = {}
    for name, value in controls.items():
        if value is None:
            continue
        if name in RANGES:
            if not isinstance(value, (list, tuple)) or len(value) != 2:
                raise ValueError(f"{name} must be [minimum, maximum]")
            low, high = (number(v, name) for v in value)
            if low <= 0 or high < low or (name != "exposure_time_us" and low < 1):
                raise ValueError(
                    f"{name} requires an ordered positive range (gains >= 1)"
                )
            if name == "exposure_time_us":
                if fps is not None and high > 1_000_000 / float(fps):
                    raise ValueError(
                        f"{name} maximum exceeds the frame period at {fps} fps"
                    )
                low, high = round(low * 1000), round(high * 1000)
                if low < 1:
                    raise ValueError(
                        "exposure_time_us must represent at least one nanosecond"
                    )
                result[RANGES[name]] = f"{low} {high}"
            else:
                result[RANGES[name]] = f"{low:g} {high:g}"
        elif name in SCALARS:
            prop, low, high = SCALARS[name]
            if not low <= number(value, name) <= high:
                raise ValueError(f"{name} must be in {low}..{high}")
            result[prop] = float(value)
        elif name in ENUMS:
            prop, choices = ENUMS[name]
            if not isinstance(value, str) or value not in choices:
                raise ValueError(f"{name} must be one of {', '.join(choices)}")
            result[prop] = choices[value]
        else:
            if type(value) is not bool:
                raise ValueError(f"{name} must be true or false")
            result[LOCKS[name]] = value
    return result
