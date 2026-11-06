# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lifetime of the CloudXR runtime launched by an OVXR session."""

from __future__ import annotations
from collections.abc import Callable
from pathlib import Path
from typing import Protocol
from .interfaces import OvxrSessionConfig


class CloudXRRuntime(Protocol):
    """Owned runtime returned by a session's launcher factory."""

    def stop(self) -> None: ...


def _create_cloudxr_launcher(env_config: str, *, accept_eula: bool = False):
    from isaaccapture.cloudxr import CloudXRLauncher

    return CloudXRLauncher(
        install_dir=str(Path.home() / ".cloudxr"),
        env_config=env_config,
        accept_eula=accept_eula,
        run_embedded=True,
    )


class CloudXRRuntimeOwner:
    """Stop only the runtime this owner launched; preserve failed stops for retry."""

    def __init__(
        self,
        config: OvxrSessionConfig,
        factory: Callable[..., CloudXRRuntime] | None = None,
    ) -> None:
        self._config = config
        self._factory = factory or _create_cloudxr_launcher
        self._launcher: CloudXRRuntime | None = None

    def start(self) -> None:
        if self._config.cloudxr_env_file and self._config.auto_launch_cloudxr:
            self._launcher = self._factory(
                self._config.cloudxr_env_file, accept_eula=self._config.accept_eula
            )

    def close(self) -> None:
        if self._launcher is not None:
            self._launcher.stop()
            self._launcher = None
