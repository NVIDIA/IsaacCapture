# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Names, levels, the line format and the log directory both halves resolve."""

from __future__ import annotations

import logging
import os
import stat
from pathlib import Path

import pytest
from conftest import LINE_RE

from isaaccapture import logging_config
from isaaccapture.logging_config import _core

not_root = pytest.mark.skipif(
    os.getuid() == 0, reason="root is not refused by directory permissions"
)


class TestLevelVocabulary:
    """One operator value must not mean two thresholds across the two halves."""

    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("trace", 5),
            ("debug", logging.DEBUG),
            ("INFO", logging.INFO),
            ("Warning", logging.WARNING),
            ("error", logging.ERROR),
            ("critical", logging.CRITICAL),
        ],
    )
    def test_resolve_level_accepts_the_six_names(self, name, expected):
        assert logging_config._core.resolve_level(name) == expected

    @pytest.mark.parametrize("name", ["warn", "err", "banana", ""])
    def test_resolve_level_rejects_anything_else(self, name):
        # "warn"/"err" are spdlog's spellings, deliberately absent here.
        with pytest.raises(ValueError):
            _core.resolve_level(name)

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("debug", logging.DEBUG),
            (" WARNING ", logging.WARNING),
            ("40", logging.ERROR),
            ("+30", logging.WARNING),
            ("-5", -5),
            ("warn", logging.INFO),
            ("err", logging.INFO),
            ("banana", logging.INFO),
            ("", logging.INFO),
        ],
    )
    def test_env_console_level_never_raises(self, monkeypatch, raw, expected):
        monkeypatch.setenv("ISAACCAPTURE_LOG_LEVEL", raw)
        assert _core.env_console_level() == expected

    def test_env_console_level_defaults_to_info(self, monkeypatch):
        monkeypatch.delenv("ISAACCAPTURE_LOG_LEVEL", raising=False)
        assert _core.env_console_level() == logging.INFO

    def test_trace_sits_below_debug_and_has_a_name(self):
        assert _core.TRACE == 5
        assert _core.TRACE < logging.DEBUG
        assert logging.getLevelName(_core.TRACE) == "TRACE"


class TestLineFormat:
    """The shape log_bridge's kFilePattern is written to reproduce."""

    def _render(self, level: int, message: str, args) -> str:
        record = logging.LogRecord(
            "isaaccapture.core.Probe", level, __file__, 1, message, args, None
        )
        formatter = logging.Formatter(_core.LINE_FORMAT, datefmt=_core.DATE_FORMAT)
        return formatter.format(record)

    def test_renders_the_documented_fields(self):
        match = LINE_RE.fullmatch(self._render(logging.WARNING, "hello %s", ("you",)))
        assert match is not None
        assert match["level"] == "WARNING"
        assert match["name"] == "isaaccapture.core.Probe"
        assert match["pid"] == str(os.getpid())
        assert match["message"] == "hello you"

    def test_pads_short_level_names_to_five(self):
        assert LINE_RE.fullmatch(self._render(logging.INFO, "x", None))["level"] == (
            "INFO "
        )

    def test_renders_trace(self):
        rendered = LINE_RE.fullmatch(self._render(_core.TRACE, "x", None))
        assert rendered["level"] == "TRACE"


class TestLogDir:
    """Where both halves put files, and what they publish about it."""

    def test_defaults_to_a_per_uid_directory(self, monkeypatch):
        # sink_config.cpp builds this same string; a shared /tmp/isaaccapture
        # would belong to whichever user created it first.
        monkeypatch.delenv("ISAACCAPTURE_LOG_DIR", raising=False)
        assert _core.log_dir() == Path(f"/tmp/isaaccapture-{os.getuid()}/logs")

    def test_republishes_an_absolute_path(self, monkeypatch, tmp_path):
        # A child may chdir before its first C++ logger resolves the variable.
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("ISAACCAPTURE_LOG_DIR", "logs")
        resolved = _core.log_dir()
        assert resolved == tmp_path / "logs"
        assert os.environ["ISAACCAPTURE_LOG_DIR"] == str(resolved)


class TestEnsurePrivateDir:
    def test_narrows_every_component_it_created(self, tmp_path):
        base = tmp_path / "shared"
        base.mkdir()
        base.chmod(0o755)
        leaf = base / "runtime" / "logs"

        assert _core.ensure_private_dir(leaf) == leaf
        # The runtime directory is where the log socket lives, so the leaf alone
        # is not enough.
        assert stat.S_IMODE((base / "runtime").stat().st_mode) == 0o700
        assert stat.S_IMODE(leaf.stat().st_mode) == 0o700
        assert stat.S_IMODE(base.stat().st_mode) == 0o755

    def test_leaves_a_directory_the_operator_chose_alone(self, tmp_path):
        existing = tmp_path / "operator"
        existing.mkdir()
        existing.chmod(0o755)
        assert _core.ensure_private_dir(existing) == existing
        assert stat.S_IMODE(existing.stat().st_mode) == 0o755

    @not_root
    def test_raises_rather_than_returning_an_unusable_directory(self, tmp_path):
        locked = tmp_path / "locked"
        locked.mkdir()
        locked.chmod(0o500)
        try:
            with pytest.raises(OSError):
                _core.ensure_private_dir(locked / "logs")
        finally:
            locked.chmod(0o700)


def test_move_above_std_leaves_a_high_descriptor_alone():
    # The relocation itself needs a free fd below 3; test_install covers that
    # in a process started with fd 1 closed.
    fd = os.open(os.devnull, os.O_RDONLY)
    try:
        assert _core._move_above_std(fd) == fd
    finally:
        os.close(fd)
