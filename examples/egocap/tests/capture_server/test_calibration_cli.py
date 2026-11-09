# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""The ``--manus-calibration-dir`` / ``--manus-calibrator`` options and worker discovery.

The directory is always kept when it is given; only a missing worker disables generating a calibration.
An explicit worker path that cannot run is a startup error.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from capture_server import server as server_module
from capture_server.calibration import WORKER_NAME, CalibrationConfig, find_worker
from capture_server.server import _build_arg_parser, _calibration_config, main


def _args(*flags: str):
    return _build_arg_parser().parse_args(["--mock", *flags])


def _executable(path: Path) -> Path:
    path.write_text("#!/bin/sh\nexit 0\n")
    path.chmod(0o755)
    return path


def _install_worker(isaac_root: Path, *, executable: bool = True) -> Path:
    """Where the Manus component installs the worker under ISAAC_ROOT."""
    directory = isaac_root / "install" / "plugins" / "manus"
    directory.mkdir(parents=True)
    path = directory / WORKER_NAME
    path.write_text("#!/bin/sh\nexit 0\n")
    path.chmod(0o755 if executable else 0o644)
    return path


# ---------------------------------------------------------------------------
# Option parsing
# ---------------------------------------------------------------------------


def test_the_calibration_options_default_to_off():
    args = _args()

    assert args.manus_calibration_dir is None
    assert args.manus_calibrator is None


def test_the_calibration_options_round_trip(tmp_path):
    args = _args("--manus-calibration-dir", str(tmp_path), "--manus-calibrator", "/opt/worker")

    assert args.manus_calibration_dir == str(tmp_path)
    assert args.manus_calibrator == "/opt/worker"


# ---------------------------------------------------------------------------
# _calibration_config
# ---------------------------------------------------------------------------


def test_no_calibration_options_leave_calibration_off():
    assert _calibration_config(_args()) is None


def test_a_directory_without_a_findable_worker_is_kept_and_only_generation_is_disabled(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr(server_module, "find_worker", lambda override: None)

    config = _calibration_config(_args("--manus-calibration-dir", str(tmp_path)))

    assert config == CalibrationConfig(directory=tmp_path, worker_path=None)
    err = capsys.readouterr().err
    assert "generating new calibrations is unavailable" in err


def test_a_directory_with_a_found_worker_enables_both_without_a_note(tmp_path, monkeypatch, capsys):
    worker = _executable(tmp_path / "found-worker")
    monkeypatch.setattr(server_module, "find_worker", lambda override: worker)

    config = _calibration_config(_args("--manus-calibration-dir", str(tmp_path / "mcal")))

    assert config == CalibrationConfig(directory=tmp_path / "mcal", worker_path=worker)
    assert capsys.readouterr().err == ""


def test_an_explicit_worker_and_a_directory_set_both(tmp_path):
    worker = _executable(tmp_path / "my-worker")

    config = _calibration_config(
        _args("--manus-calibration-dir", str(tmp_path / "mcal"), "--manus-calibrator", str(worker))
    )

    assert config == CalibrationConfig(directory=tmp_path / "mcal", worker_path=worker)


def test_an_explicit_worker_that_does_not_exist_is_an_error_not_a_silent_fallback(tmp_path, monkeypatch):
    # A worker installed under ISAAC_ROOT must not hide a wrong explicit path.
    _install_worker(tmp_path / "root")
    monkeypatch.setenv("ISAAC_ROOT", str(tmp_path / "root"))

    with pytest.raises(ValueError, match="is not an executable file"):
        _calibration_config(
            _args("--manus-calibration-dir", str(tmp_path / "mcal"), "--manus-calibrator", "/nonexistent/worker")
        )


def test_an_explicit_worker_that_is_not_executable_is_an_error(tmp_path):
    plain = tmp_path / "not-executable"
    plain.write_text("x")
    plain.chmod(0o644)

    with pytest.raises(ValueError, match="is not an executable file"):
        _calibration_config(
            _args("--manus-calibration-dir", str(tmp_path / "mcal"), "--manus-calibrator", str(plain))
        )


def test_a_worker_without_a_directory_is_an_error(tmp_path):
    worker = _executable(tmp_path / "my-worker")

    with pytest.raises(ValueError, match="needs --manus-calibration-dir"):
        _calibration_config(_args("--manus-calibrator", str(worker)))


@pytest.mark.parametrize(
    "flags",
    [
        ["--manus-calibrator", "/nonexistent/worker"],
        ["--manus-calibration-dir", "{dir}", "--manus-calibrator", "/nonexistent/worker"],
    ],
    ids=["worker-without-directory", "missing-explicit-worker"],
)
def test_a_bad_calibration_option_exits_with_status_two_and_the_reason(tmp_path, capsys, flags):
    flags = [flag.replace("{dir}", str(tmp_path)) for flag in flags]

    assert main(["--mock", *flags]) == 2

    err = capsys.readouterr().err
    assert err.startswith("capture-server: ")
    assert "--manus-calibrator" in err


# ---------------------------------------------------------------------------
# find_worker
# ---------------------------------------------------------------------------


def test_find_worker_prefers_an_explicit_executable_over_the_install(tmp_path, monkeypatch):
    _install_worker(tmp_path / "root")
    monkeypatch.setenv("ISAAC_ROOT", str(tmp_path / "root"))
    explicit = _executable(tmp_path / "explicit-worker")

    assert find_worker(explicit) == explicit


def test_find_worker_does_not_fall_back_when_the_explicit_path_is_unusable(tmp_path, monkeypatch):
    _install_worker(tmp_path / "root")
    monkeypatch.setenv("ISAAC_ROOT", str(tmp_path / "root"))
    plain = tmp_path / "not-executable"
    plain.write_text("x")
    plain.chmod(0o644)

    assert find_worker(plain) is None
    assert find_worker(tmp_path / "missing") is None


def test_find_worker_falls_back_to_the_manus_install_under_isaac_root(tmp_path, monkeypatch):
    installed = _install_worker(tmp_path / "root")
    monkeypatch.setenv("ISAAC_ROOT", str(tmp_path / "root"))

    assert find_worker(None) == installed


def test_find_worker_ignores_a_non_executable_file_in_the_install(tmp_path, monkeypatch):
    _install_worker(tmp_path / "root", executable=False)
    monkeypatch.setenv("ISAAC_ROOT", str(tmp_path / "root"))

    assert find_worker(None) is None


def test_find_worker_without_isaac_root_finds_nothing_even_if_the_worker_is_on_path(tmp_path, monkeypatch):
    path_dir = tmp_path / "bin"
    path_dir.mkdir()
    _executable(path_dir / WORKER_NAME)
    monkeypatch.delenv("ISAAC_ROOT", raising=False)
    monkeypatch.setenv("PATH", str(path_dir))

    assert find_worker(None) is None
