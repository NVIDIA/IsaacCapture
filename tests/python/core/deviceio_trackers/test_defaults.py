# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check generated tracker defaults through the real Python bindings."""

import runpy

import pytest
from repo_paths import repo_root

from isaaccapture import deviceio_trackers

_core = repo_root() / "src" / "core"
_manifest = runpy.run_path(str(_core / "codegen" / "manifest.py"))
_entries = _manifest["load_manifest"](
    _core / "deviceio_trackers" / "trackers.toml",
    _core / "deviceio_trackers" / "defaults.toml",
)


@pytest.mark.parametrize(
    "entry",
    [entry for entry in _entries if entry["direction"] == "pull"],
    ids=lambda entry: entry["class"],
)
def test_default_flatbuffer_size_matches_manifest(entry):
    tracker_type = getattr(deviceio_trackers, entry["class"])
    size = tracker_type.DEFAULT_MAX_FLATBUFFER_SIZE
    assert isinstance(size, int)
    assert size == entry["max_flatbuffer_size"]
