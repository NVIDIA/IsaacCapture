# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``TeleopSession`` stand-in: the one class real consumer code actually drives.

Per ``GumanIsaacTeleopUsage.md`` §3: construct with a ``TeleopSessionConfig``, use as a context
manager, call ``step()`` once per tick with no args, get back ``dict[str, TensorGroup]`` keyed
exactly by the pipeline's ``OutputCombiner`` output names. §10's recommended shape, implemented
directly: discover output names from ``config.pipeline``, synthesize per step, never touch
DeviceIO/OpenXR/plugins for real.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ._synthetic import SyntheticPoseGenerator, TensorGroup


@dataclass
class PluginConfig:
    """Accepted, not simulated -- see ``GumanIsaacTeleopUsage.md`` §7."""

    plugin_name: str
    plugin_root_id: str
    search_paths: list[Path] = field(default_factory=list)
    enabled: bool = True
    plugin_args: list[str] = field(default_factory=list)
    required: bool = False


@dataclass
class TeleopSessionConfig:
    app_name: str
    pipeline: Any
    plugins: list[PluginConfig] = field(default_factory=list)
    sinks: list[Any] = field(default_factory=list)


class TeleopSession:
    """Mirrors the real class's lifecycle: ``__enter__``/``step()``/``__exit__``, nothing else
    Guman's backend exercises (no ``REPLAY`` mode, ``mcap_config``, ``oxr_handles`` -- see
    ``GumanIsaacTeleopUsage.md`` §10's "what this mock does not need to support").
    """

    def __init__(self, config: TeleopSessionConfig) -> None:
        self.config = config
        self._generator = SyntheticPoseGenerator(config.pipeline.joint_counts)

    def __enter__(self) -> "TeleopSession":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        pass

    def step(self, **_kwargs: Any) -> dict[str, TensorGroup]:
        return {
            name: self._generator.sample(name)
            for name in self.config.pipeline.output_names
        }
