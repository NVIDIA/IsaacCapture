# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Names, levels, the line format and the log directory both halves resolve."""

from __future__ import annotations

import errno
import logging
import os
import stat
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

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
        assert (
            _core.log_dir()
            == Path("/tmp").resolve() / f"isaaccapture-{os.getuid()}" / "logs"
        )

    @pytest.mark.parametrize("channel", ["logs", "socket"])
    def test_default_root_alias_does_not_bypass_child_validation(
        self, monkeypatch, tmp_path, channel
    ):
        from isaaccapture.logging_config import _forwarding

        root = tmp_path / "platform-temp"
        root.mkdir()
        alias = tmp_path / "tmp-alias"
        alias.symlink_to(root, target_is_directory=True)
        resolve = Path.resolve

        def resolve_root(path, *args, **kwargs):
            return resolve(alias if path == Path("/tmp") else path, *args, **kwargs)

        monkeypatch.setattr(Path, "resolve", resolve_root)
        monkeypatch.delenv("ISAACCAPTURE_LOG_DIR", raising=False)
        monkeypatch.delenv("XDG_RUNTIME_DIR", raising=False)
        directory = _core.log_dir() if channel == "logs" else _forwarding._runtime_dir()
        expected = root / f"isaaccapture-{os.getuid()}"
        assert directory == (expected / "logs" if channel == "logs" else expected)
        assert _core.ensure_private_dir(directory) == directory

        # A symlink below that canonical root must still be refused.
        link = root / "operator-link"
        link.symlink_to(expected, target_is_directory=True)
        monkeypatch.setenv("ISAACCAPTURE_LOG_DIR", str(link / "logs"))
        selected = _core.log_dir()
        assert selected == link / "logs"
        with pytest.raises(OSError) as failure:
            _core.ensure_private_dir(selected)
        assert failure.value.errno in (errno.ENOTDIR, errno.ELOOP)

    @pytest.mark.parametrize("failure", [OSError, RuntimeError])
    def test_unusable_default_root_keeps_the_fallback_path(self, monkeypatch, failure):
        with patch.object(Path, "resolve", side_effect=failure("unusable root")):
            assert _core.default_runtime_dir() == Path(
                f"/tmp/isaaccapture-{os.getuid()}"
            )

    def test_republishes_an_absolute_path(self, monkeypatch, tmp_path):
        # A child may chdir before its first C++ logger resolves the variable.
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("ISAACCAPTURE_LOG_DIR", "logs")
        resolved = _core.log_dir()
        assert resolved == tmp_path / "logs"
        assert os.environ["ISAACCAPTURE_LOG_DIR"] == str(resolved)


class TestEnsurePrivateDir:
    @pytest.mark.parametrize(("mode", "foreign"), [(0o777, False), (0o755, True)])
    def test_rejects_an_untrusted_root_before_creating_children(
        self, tmp_path, mode, foreign
    ):
        info = SimpleNamespace(
            st_mode=stat.S_IFDIR | mode,
            st_uid=os.geteuid() + 1 if foreign else os.geteuid(),
        )
        with (
            patch.object(_core.os, "fstat", return_value=info),
            patch.object(_core.os, "mkdir") as mkdir,
        ):
            with pytest.raises(PermissionError):
                _core.ensure_private_dir(tmp_path / "logs")
            mkdir.assert_not_called()

    @pytest.mark.parametrize("effective_owner", [True, False])
    def test_ownership_uses_the_effective_user(self, effective_owner):
        real_uid, effective_uid = 12345, 12346
        info = SimpleNamespace(
            st_mode=stat.S_IFDIR | 0o700,
            st_uid=effective_uid if effective_owner else real_uid,
        )
        with (
            patch.object(_core.os, "getuid", return_value=real_uid),
            patch.object(_core.os, "geteuid", return_value=effective_uid),
            patch.object(_core.os, "fstat", return_value=info),
        ):
            if effective_owner:
                _core._validate_directory(123, Path("/owned"), target=True)
            else:
                with pytest.raises(PermissionError, match="foreign-owned"):
                    _core._validate_directory(123, Path("/foreign"), target=True)

    def test_failure_names_the_replaceable_ancestor(self, tmp_path):
        ancestor = tmp_path / "shared"
        ancestor.mkdir()
        ancestor.chmod(0o777)
        with pytest.raises(PermissionError) as failure:
            _core.ensure_private_dir(ancestor / "not-created" / "logs")
        assert str(ancestor) in str(failure.value)
        assert "not-created" not in str(failure.value)
        assert not (ancestor / "not-created").exists()

    @not_root
    @pytest.mark.skipif(not hasattr(os, "O_PATH"), reason="Linux traversal descriptors")
    @pytest.mark.parametrize("existing", [False, True])
    def test_searchable_ancestor_does_not_need_read_permission(
        self, tmp_path, existing
    ):
        ancestor = tmp_path / "searchable"
        ancestor.mkdir()
        leaf = ancestor / "logs"
        if existing:
            leaf.mkdir(mode=0o700)
        ancestor.chmod(0o300)
        try:
            assert _core.ensure_private_dir(leaf) == leaf
            (leaf / "probe").write_text("usable")
        finally:
            ancestor.chmod(0o700)

    def test_validation_only_never_creates_missing_components(self, tmp_path):
        with patch.object(_core.os, "mkdir") as mkdir:
            assert not _core.is_private_dir(tmp_path / "missing" / "runtime")
            mkdir.assert_not_called()
        assert not (tmp_path / "missing").exists()
        assert _core.is_private_dir(tmp_path)

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

    def test_rejects_a_symlink_at_the_target(self, tmp_path):
        private = tmp_path / "private"
        private.mkdir()
        link = tmp_path / "logs"
        link.symlink_to(private, target_is_directory=True)

        with pytest.raises(OSError) as failure:
            _core.ensure_private_dir(link)
        assert failure.value.errno in (errno.ENOTDIR, errno.ELOOP)

    def test_does_not_create_through_a_symlinked_parent(self, tmp_path):
        destination = tmp_path / "destination"
        destination.mkdir()
        link = tmp_path / "runtime"
        link.symlink_to(destination, target_is_directory=True)

        with pytest.raises(OSError) as failure:
            _core.ensure_private_dir(link / "logs")
        assert failure.value.errno in (errno.ENOTDIR, errno.ELOOP)
        assert not (destination / "logs").exists()

    def test_rejects_a_replaceable_path_component(self, tmp_path):
        shared = tmp_path / "shared"
        shared.mkdir()
        shared.chmod(0o777)

        with pytest.raises(PermissionError, match="replaceable"):
            _core.ensure_private_dir(shared / "logs")

    def test_sticky_ancestor_is_allowed_but_sticky_target_is_not(self, tmp_path):
        shared = tmp_path / "shared"
        shared.mkdir()
        shared.chmod(0o1777)
        assert _core.ensure_private_dir(shared / "logs") == shared / "logs"
        with pytest.raises(PermissionError, match="replaceable"):
            _core.ensure_private_dir(shared)

    def test_rejects_parent_traversal_before_creating_anything(self, tmp_path):
        with pytest.raises(PermissionError, match="ambiguous"):
            _core.ensure_private_dir(tmp_path / "new" / ".." / "logs")
        assert not (tmp_path / "new").exists()

    def test_creation_is_private_even_with_a_permissive_umask(self, tmp_path):
        previous = os.umask(0)
        try:
            leaf = _core.ensure_private_dir(tmp_path / "runtime" / "logs")
        finally:
            os.umask(previous)
        assert stat.S_IMODE(leaf.stat().st_mode) == 0o700
        assert stat.S_IMODE(leaf.parent.stat().st_mode) == 0o700

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
