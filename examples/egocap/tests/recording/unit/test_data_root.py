# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from egocap.core.config_io import ConfigIO
from egocap.core.data_root import calibration_dir, data_root


def test_data_root_defaults_to_neutral_home_directory(monkeypatch):
    monkeypatch.delenv("EGOCAP_DATA_DIR", raising=False)
    assert data_root() == Path.home() / "egocap_data"


def test_data_root_env_override_expands_user(monkeypatch, tmp_path):
    monkeypatch.setenv("EGOCAP_DATA_DIR", str(tmp_path / "root"))
    assert data_root() == tmp_path / "root"


def test_empty_data_root_env_is_treated_as_unset(monkeypatch):
    monkeypatch.setenv("EGOCAP_DATA_DIR", "")
    assert data_root() == Path.home() / "egocap_data"


def test_calibration_dir_follows_data_root(monkeypatch, tmp_path):
    monkeypatch.delenv("EGOCAP_CALIBRATION_DIR", raising=False)
    monkeypatch.setenv("EGOCAP_DATA_DIR", str(tmp_path))
    assert calibration_dir() == tmp_path / "calibration"


def test_calibration_dir_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("EGOCAP_CALIBRATION_DIR", str(tmp_path / "calib"))
    assert calibration_dir() == tmp_path / "calib"


def test_default_config_data_dir_uses_data_root(monkeypatch, tmp_path):
    monkeypatch.setenv("EGOCAP_DATA_DIR", str(tmp_path))
    default = ConfigIO._get_default_config()
    assert default["recording"]["data_dir"] == str(tmp_path / "recordings")


def test_empty_calibration_env_is_treated_as_unset(monkeypatch, tmp_path):
    monkeypatch.setenv("EGOCAP_CALIBRATION_DIR", "")
    monkeypatch.setenv("EGOCAP_DATA_DIR", str(tmp_path))
    assert calibration_dir() == tmp_path / "calibration"
