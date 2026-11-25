# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Vive discovery shares the publisher's directory policy without creating paths."""

import os
from pathlib import Path
import stat
from types import SimpleNamespace

import pytest

from isaaccapture.logging_config import _core
from isaaccapture_examples.mcap_record_replay import record_se3_vive as recorder

pytestmark = pytest.mark.skipif(os.name != "posix", reason="POSIX Vive discovery")


@pytest.fixture
def advertisement(tmp_path, monkeypatch):
    monkeypatch.delenv("VIVE_SE3_COLLECTIONS_FILE", raising=False)
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    path = tmp_path / "vive_se3_collections.txt"
    path.write_text("one\ntwo\none\n\n")
    path.chmod(0o600)
    return path


def test_discovery_follows_environment_changes_after_import(
    advertisement, tmp_path, monkeypatch
):
    assert recorder.discover_collections() == ["one", "two"]
    second = tmp_path / "second"
    second.mkdir()
    path = second / advertisement.name
    path.write_text("three\n")
    path.chmod(0o600)
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(second))
    assert recorder.discover_collections() == ["three"]


@pytest.mark.parametrize("kind", ["missing", "symlink", "writable"])
def test_untrusted_xdg_is_not_created_and_falls_back(tmp_path, monkeypatch, kind):
    monkeypatch.delenv("VIVE_SE3_COLLECTIONS_FILE", raising=False)
    xdg = tmp_path / "xdg"
    if kind == "symlink":
        xdg.symlink_to(tmp_path, target_is_directory=True)
    elif kind == "writable":
        xdg.mkdir()
        xdg.chmod(0o777)
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(xdg))
    fallback = Path("/tmp").resolve() / f"vive_se3_tracker-{os.geteuid()}"
    validate = _core.is_private_dir
    checked = []

    def validate_selection(path):
        checked.append(path)
        # Do not inspect or create the running user's real application directory.
        return True if path == fallback else validate(path)

    monkeypatch.setattr(_core, "is_private_dir", validate_selection)
    assert recorder._collections_file() == str(fallback / "vive_se3_collections.txt")
    assert checked == [xdg, fallback]
    if kind == "missing":
        assert not xdg.exists()


def test_no_trusted_directory_disables_discovery(monkeypatch):
    monkeypatch.delenv("VIVE_SE3_COLLECTIONS_FILE", raising=False)
    monkeypatch.setenv("XDG_RUNTIME_DIR", "/unavailable/runtime")
    monkeypatch.setattr(_core, "is_private_dir", lambda _path: False)
    assert recorder._collections_file() is None
    assert recorder.discover_collections() == []


def test_fallback_name_uses_effective_uid(monkeypatch):
    monkeypatch.delenv("VIVE_SE3_COLLECTIONS_FILE", raising=False)
    monkeypatch.delenv("XDG_RUNTIME_DIR", raising=False)
    monkeypatch.setattr(
        recorder,
        "os",
        SimpleNamespace(
            **{**vars(os), "getuid": lambda: 12345, "geteuid": lambda: 12346}
        ),
    )
    checked = []
    monkeypatch.setattr(
        _core, "is_private_dir", lambda path: checked.append(path) or True
    )
    fallback = Path("/tmp").resolve() / "vive_se3_tracker-12346"
    assert recorder._collections_file() == str(fallback / "vive_se3_collections.txt")
    assert checked == [fallback]


@pytest.mark.parametrize(
    "kind", ["symlink", "fifo", "directory", "writable", "group-writable", "foreign"]
)
def test_default_metadata_refuses_unsafe_opened_files(advertisement, monkeypatch, kind):
    if kind in ("symlink", "fifo", "directory"):
        advertisement.unlink()
    if kind == "symlink":
        target = advertisement.with_name("target")
        target.write_text("untrusted\n")
        advertisement.symlink_to(target)
    elif kind == "fifo":
        os.mkfifo(advertisement, 0o600)
    elif kind == "directory":
        advertisement.mkdir()
    elif kind in ("writable", "group-writable"):
        advertisement.chmod(0o666 if kind == "writable" else 0o664)
    else:
        fstat = os.fstat

        def foreign_file(fd):
            info = fstat(fd)
            if stat.S_ISREG(info.st_mode):
                return SimpleNamespace(st_mode=info.st_mode, st_uid=os.geteuid() + 1)
            return info

        monkeypatch.setattr(
            recorder, "os", SimpleNamespace(**{**vars(os), "fstat": foreign_file})
        )
    assert recorder.discover_collections() == []


def test_automatic_read_does_not_switch_policy_during_selection(
    advertisement, monkeypatch
):
    target = advertisement.with_name("target")
    advertisement.rename(target)
    advertisement.symlink_to(target)
    validate = _core.is_private_dir

    def changed_environment(path):
        valid = validate(path)
        monkeypatch.setenv("VIVE_SE3_COLLECTIONS_FILE", str(advertisement))
        return valid

    monkeypatch.setattr(_core, "is_private_dir", changed_environment)
    assert recorder.discover_collections() == []


def test_missing_advertisement_is_not_created(advertisement):
    advertisement.unlink()
    assert recorder.discover_collections() == []
    assert not advertisement.exists()


def test_explicit_override_retains_operator_selected_symlinks(
    advertisement, monkeypatch
):
    link = advertisement.with_name("explicit")
    link.symlink_to(advertisement)
    monkeypatch.setenv("VIVE_SE3_COLLECTIONS_FILE", str(link))

    def unexpected_validation(_path):
        pytest.fail("an explicit override unexpectedly used automatic discovery")

    monkeypatch.setattr(_core, "is_private_dir", unexpected_validation)
    assert recorder.discover_collections() == ["one", "two"]


def test_help_does_not_load_native_runtime(capsys):
    with pytest.raises(SystemExit) as result:
        recorder.main(["record-se3-vive", "--help"])
    assert result.value.code == 0
    assert "--collections" in capsys.readouterr().out


def test_default_temp_alias_preserves_child_symlink_refusal(tmp_path, monkeypatch):
    monkeypatch.delenv("VIVE_SE3_COLLECTIONS_FILE", raising=False)
    monkeypatch.delenv("XDG_RUNTIME_DIR", raising=False)
    root = tmp_path / "canonical-temp"
    root.mkdir()
    alias = tmp_path / "temp-alias"
    alias.symlink_to(root, target_is_directory=True)
    resolve = Path.resolve

    def resolve_root(path, *args, **kwargs):
        return resolve(alias if path == Path("/tmp") else path, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", resolve_root)
    directory = root / f"vive_se3_tracker-{os.geteuid()}"
    directory.mkdir()
    path = directory / "vive_se3_collections.txt"
    path.write_text("trusted\n")
    path.chmod(0o600)
    assert recorder.discover_collections() == ["trusted"]
    target = root / "other"
    directory.rename(target)
    directory.symlink_to(target, target_is_directory=True)
    assert recorder.discover_collections() == []
