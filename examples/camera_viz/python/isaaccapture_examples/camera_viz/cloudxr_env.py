# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CloudXR runtime settings, declared in the YAML instead of the process env.

The runtime takes its configuration from environment variables, and there are
three ways to set one and be silently wrong:

* a name nothing reads -- ``NV_CXR_DEVICE_PROFILE`` looks like its neighbours
  but the runtime reads ``NV_DEVICE_PROFILE``;
* a stale shell export -- the launcher writes ``~/.cloudxr/run/cloudxr.env``
  and tells you to source it, after which ``os.environ.setdefault`` here and
  even ``--cloudxr-device-profile`` both quietly lose to the sourced value;
* a bool spelled ``False``, which the runtime's exact-match parser does not
  recognise and therefore reads as **true**.

None of the three produces an error. So the settings go through the one tier
that outranks the process environment: a ``--cloudxr-env-config`` file, whose
entries beat both a shell export and the launcher's own defaults. Verified
against isaaccapture.cloudxr.env_config._load_resolve_and_apply, which merges
the file last.

Each of the three has a guard here: an unknown name warns with a suggestion,
the env file outranks the shell, and booleans are written in the one spelling
the parser reads as false.
"""

from __future__ import annotations

import difflib
import sys
from pathlib import Path
from typing import Dict

# Computed by the launcher; an env file that sets them earns a UserWarning and
# is ignored, so reject them here where the message can name the YAML key.
RESERVED_KEYS = frozenset(
    {"XR_RUNTIME_JSON", "XRT_NO_STDIN", "NV_CXR_RUNTIME_DIR", "NV_CXR_OUTPUT_DIR"}
)

# camera_viz's own defaults, overridable per deployment from the YAML.
DEFAULT_ENV: Dict[str, str] = {
    # The runtime otherwise blocks each server frame until a fresh client pose
    # arrives; the app is not a head-tracked renderer, so it need not wait.
    "NV_ENABLE_POSE_WAIT": "false",
    # Runtime-side fixed foveation: the composited image is warped before
    # encoding, so peripheral pixels cost less bandwidth. Off in the runtime by
    # default, and it applies to the layers fast path camera_viz uses.
    "NV_CXR_RUNTIME_FOVEATION": "true",
}

# Names the runtime actually reads, so a typo can be caught before launch
# rather than by silently doing nothing. Not exhaustive -- an unlisted name
# warns and is still passed through, because the runtime has many more knobs
# than camera_viz has opinions about.
KNOWN_KEYS = frozenset(
    {
        "NV_ENABLE_POSE_WAIT",
        "NV_MAX_POSE_WAIT_DURATION_MS",
        "NV_CXR_RUNTIME_FOVEATION",
        "NV_CXR_RUNTIME_FOVEATION_BLUR",
        "NV_CXR_RUNTIME_FOVEATION_WARPED_WIDTH",
        "NV_CXR_RUNTIME_FOVEATION_UNWARPED_WIDTH",
        "NV_CXR_RUNTIME_FOVEATION_INSET",
        "NV_CXR_APP_PQW_INSET",
        "NV_DEVICE_PROFILE",
        "NV_DISABLE_DEPTH_DILATION",
        "NV_IMMEDIATE_COMPOSITOR",
        "NV_MAX_FPS",
        "NV_SKIP_FRAME_ON_SPIKE",
        "XRT_PRINT_OPTIONS",
    }
)


def env_from_yaml(display: dict) -> Dict[str, str]:
    """``display.cloudxr`` merged over :data:`DEFAULT_ENV`.

    Values are stringified here rather than at the call site because the
    conversion is where a bug hides: YAML ``false`` reaches Python as ``False``
    and ``str()`` would spell it ``"False"``, which the runtime's parser does
    not recognise -- and an unrecognised value reads as *true*, i.e. the exact
    opposite of what the config says.
    """
    spec = display.get("cloudxr") or {}
    if not isinstance(spec, dict):
        raise ValueError(
            f"camera_viz: display.cloudxr must be a mapping of "
            f"NAME: value, got {type(spec).__name__}"
        )
    env = dict(DEFAULT_ENV)
    for key, value in spec.items():
        name = str(key)
        if name in RESERVED_KEYS:
            raise ValueError(
                f"camera_viz: display.cloudxr.{name} is computed by the "
                "CloudXR launcher and cannot be set here"
            )
        _warn_if_unknown(name)
        if value is None:  # explicit null = "drop camera_viz's default"
            env.pop(name, None)
            continue
        env[name] = _as_env_value(value)
    return env


def _warn_if_unknown(name: str) -> None:
    """A name nothing reads is the failure mode this module exists for:
    ``NV_CXR_DEVICE_PROFILE`` sits between two variables that are spelled that
    way, but the runtime reads ``NV_DEVICE_PROFILE`` and ignores the other
    without a word."""
    if name in KNOWN_KEYS:
        return
    hint = difflib.get_close_matches(name, KNOWN_KEYS, n=1)
    suggestion = f" (did you mean {hint[0]!r}?)" if hint else ""
    print(
        f"camera_viz: warning: display.cloudxr.{name} is not a CloudXR "
        f"variable camera_viz knows{suggestion} — passing it through anyway",
        file=sys.stderr,
        flush=True,
    )


def _as_env_value(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def write_env_file(env: Dict[str, str], path: Path) -> Path:
    """Write ``env`` as the launcher's ``KEY=value`` format.

    Deliberately not ``export KEY=value``: that is the shape the launcher
    *writes* for humans to source, but its own parser splits on the first
    ``=`` without stripping a prefix, so an exported line would arrive as the
    key ``"export KEY"`` and be ignored.
    """
    lines = ["# Generated by camera_viz from display.cloudxr — do not edit."]
    lines += [f"{key}={value}" for key, value in sorted(env.items())]
    path.write_text("\n".join(lines) + "\n")
    return path
