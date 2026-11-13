# SPDX-FileCopyrightText: Copyright (c) 2026 j-mex. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""How this example launches ``jmex_plugin`` and reads what it publishes."""

from __future__ import annotations

from pathlib import Path

from isaaccapture.deviceio import TrackerVendor
from isaaccapture.retargeting_engine.deviceio_source_nodes import FullBodySource
from isaaccapture.teleop_session_manager import PluginConfig

PLUGIN_NAME = "jmex"
PLUGIN_ROOT_ID = "jmex"
VENDOR_ID = "body.jmex"

# jmex_plugin's own defaults.
DEFAULT_COLLECTION_ID = "jmex_full_body"
DEFAULT_CHANNEL = 1


def plugin_search_paths() -> list[Path]:
    """This checkout's ``install/plugins`` once ``cmake --install`` has made it, else nothing."""
    # The repository root: this file sits in examples/jmex/python/isaaccapture_examples/jmex/.
    installed = Path(__file__).resolve().parents[5] / "install" / "plugins"
    return [installed] if installed.exists() else []


def plugin_config(channel: int, collection_id: str, *, otherwise: str) -> PluginConfig:
    """The installed ``jmex_plugin``, launched on ``channel`` and publishing on ``collection_id``.

    Raises ``RuntimeError`` ending in ``otherwise`` when this checkout has no installed plugin.
    """
    search_paths = plugin_search_paths()
    if not search_paths:
        raise RuntimeError(
            "jmex plugin directory not found. Configure with -DBUILD_PLUGIN_JMEX=ON and run "
            f"`cmake --install build`, or {otherwise}"
        )
    return PluginConfig(
        plugin_name=PLUGIN_NAME,
        plugin_root_id=PLUGIN_ROOT_ID,
        search_paths=search_paths,
        # Appended after plugin.yaml's args; the last value of a flag wins.
        plugin_args=[f"--channel={channel}", f"--collection-id={collection_id}"],
        # A build without BUILD_PLUGIN_JMEX or without the SDK installs the other plugins but
        # not this one.
        required=True,
    )


def full_body_source(collection_id: str) -> FullBodySource:
    """The ``full_body`` source node, reading the plugin's body pose from ``collection_id``."""
    return FullBodySource(
        name="full_body",
        vendor=TrackerVendor(VENDOR_ID, {"collection_id": collection_id}),
    )


__all__ = [
    "DEFAULT_CHANNEL",
    "DEFAULT_COLLECTION_ID",
    "PLUGIN_NAME",
    "PLUGIN_ROOT_ID",
    "VENDOR_ID",
    "full_body_source",
    "plugin_config",
    "plugin_search_paths",
]
