# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for episode_layout.py's write_yaml_atomic: the happy path and the failure-cleanup path
(no orphaned temp file, the real exception still propagates)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from egocap.recording.episode_layout import write_yaml_atomic


def test_write_yaml_atomic_writes_readable_yaml(tmp_path: Path):
    path = tmp_path / "config.yaml"
    write_yaml_atomic(path, {"meta": {"egocap_station_name": "Station 01"}})
    assert yaml.safe_load(path.read_text()) == {"meta": {"egocap_station_name": "Station 01"}}


def test_write_yaml_atomic_creates_a_missing_parent_directory(tmp_path: Path):
    path = tmp_path / "nested" / "config.yaml"
    write_yaml_atomic(path, {"a": 1})
    assert yaml.safe_load(path.read_text()) == {"a": 1}


def test_write_yaml_atomic_cleans_up_the_temp_file_and_reraises_on_failure(tmp_path: Path):
    path = tmp_path / "config.yaml"

    class _Unserializable:
        """yaml.safe_dump has no representer for this -- a real failure, not a mock."""

    with pytest.raises(yaml.representer.RepresenterError):
        write_yaml_atomic(path, {"bad": _Unserializable()})
    assert not path.exists()
    assert list(tmp_path.iterdir()) == []  # no orphaned .tmp file left behind
