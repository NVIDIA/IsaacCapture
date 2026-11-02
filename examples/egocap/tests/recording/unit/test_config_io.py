# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import yaml

import egocap.core.config_io as config_io_mod
from egocap.core.config_io import ConfigIO


def test_init_sets_config_path():
    cio = ConfigIO("/tmp/x/config.yaml")
    assert cio.config_path == "/tmp/x/config.yaml"


def test_load_missing_file_returns_default(tmp_path):
    cio = ConfigIO(str(tmp_path / "missing.yaml"))
    config = cio.load()
    assert config["video_sources"] == []
    assert config["numeric_sources"] == []
    assert config["recording"]["record_fps"] == 60


def test_load_empty_file_returns_default(tmp_path):
    path = tmp_path / "empty.yaml"
    path.write_text("", encoding="utf-8")
    cio = ConfigIO(str(path))
    assert cio.load()["recording"]["record_fps"] == 60


def test_load_valid_file_returns_contents(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("recording:\n  record_fps: 30\n", encoding="utf-8")
    cio = ConfigIO(str(path))
    assert cio.load() == {"recording": {"record_fps": 30}}


def test_load_malformed_yaml_raises(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("recording: [unbalanced\n", encoding="utf-8")
    cio = ConfigIO(str(path))
    try:
        cio.load()
        raise AssertionError("expected a YAML parse error")
    except yaml.YAMLError:
        pass


def test_save_creates_directory_and_writes_file(tmp_path):
    path = tmp_path / "nested" / "config.yaml"
    cio = ConfigIO(str(path))
    cio.save({"recording": {"record_fps": 60}})
    assert path.is_file()
    assert cio.load()["recording"]["record_fps"] == 60


def test_save_no_parent_dir_component(tmp_path, monkeypatch):
    # dirname("") is falsy -- exercises the "no directory to create" branch.
    monkeypatch.chdir(tmp_path)
    cio = ConfigIO("bare.yaml")
    cio.save({"a": 1})
    assert (tmp_path / "bare.yaml").is_file()


def test_dump_yaml_inlines_short_numeric_lists(tmp_path):
    path = tmp_path / "config.yaml"
    cio = ConfigIO(str(path))
    cio.save({"camera_resolution": [640, 480]})
    text = path.read_text(encoding="utf-8")
    assert "[640, 480]" in text


def test_dump_yaml_does_not_inline_long_or_nonnumeric_lists(tmp_path):
    path = tmp_path / "config.yaml"
    cio = ConfigIO(str(path))
    cio.save({"many": [1, 2, 3, 4, 5], "names": ["a", "b"]})
    text = path.read_text(encoding="utf-8")
    assert "[1, 2, 3, 4, 5]" not in text
    assert "[a, b]" not in text


def test_dump_yaml_quotes_path_like_strings(tmp_path):
    path = tmp_path / "config.yaml"
    cio = ConfigIO(str(path))
    cio.save({"data_dir": "~/egocap_data/recordings", "name": "plain"})
    text = path.read_text(encoding="utf-8")
    assert '"~/egocap_data/recordings"' in text
    assert "name: plain" in text


def test_load_with_comments_missing_file_returns_default(tmp_path):
    cio = ConfigIO(str(tmp_path / "missing.yaml"))
    assert cio.load_with_comments()["video_sources"] == []


def test_load_with_comments_ruamel_available(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("recording:\n  record_fps: 30  # a comment\n", encoding="utf-8")
    cio = ConfigIO(str(path))
    config = cio.load_with_comments()
    assert config["recording"]["record_fps"] == 30


def test_load_with_comments_ruamel_available_empty_file_returns_default(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("", encoding="utf-8")
    cio = ConfigIO(str(path))
    assert cio.load_with_comments()["video_sources"] == []


def test_save_with_comments_ruamel_available_round_trips(tmp_path):
    path = tmp_path / "config.yaml"
    cio = ConfigIO(str(path))
    cio.save_with_comments({"recording": {"record_fps": 45}})
    assert cio.load_with_comments()["recording"]["record_fps"] == 45


def test_save_with_comments_repeated_value_writes_no_yaml_anchor(tmp_path):
    # Regression: ignore_aliases was previously assigned a two-argument
    # lambda, which ruamel calls as `ignore_aliases(data)` with no implicit
    # `self` -- every save raised TypeError as soon as any value repeated
    # (ruamel's dumper always probes ignore_aliases, not just when it would
    # otherwise emit an anchor). A one-argument lambda fixed that; this
    # locks it in by saving the same list object under two keys (ruamel
    # would otherwise emit it once with an `&id001` anchor and reference it
    # with `*id001`) and asserting the save succeeds and emits neither.
    path = tmp_path / "config.yaml"
    cio = ConfigIO(str(path))
    shared = [1, 2, 3]
    cio.save_with_comments({"a": shared, "b": shared})
    text = path.read_text(encoding="utf-8")
    assert "&id" not in text
    assert "*id" not in text


def test_load_with_comments_ruamel_unavailable_falls_back(tmp_path, monkeypatch):
    monkeypatch.setattr(config_io_mod, "RUAMEL_AVAILABLE", False)
    path = tmp_path / "config.yaml"
    path.write_text("recording:\n  record_fps: 30\n", encoding="utf-8")
    cio = ConfigIO(str(path))
    assert cio.load_with_comments()["recording"]["record_fps"] == 30


def test_save_with_comments_ruamel_unavailable_falls_back(tmp_path, monkeypatch):
    monkeypatch.setattr(config_io_mod, "RUAMEL_AVAILABLE", False)
    path = tmp_path / "config.yaml"
    cio = ConfigIO(str(path))
    cio.save_with_comments({"recording": {"record_fps": 45}})
    assert cio.load()["recording"]["record_fps"] == 45


def test_save_with_comments_no_parent_dir_component(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cio = ConfigIO("bare.yaml")
    cio.save_with_comments({"a": 1})
    assert (tmp_path / "bare.yaml").is_file()


def test_module_handles_ruamel_import_failure(monkeypatch):
    # Exercises the `except ImportError: RUAMEL_AVAILABLE = False` branch at
    # module scope by forcing ruamel's import to fail and reimporting fresh.
    import builtins
    import importlib
    import sys

    import egocap.core as core_pkg

    real_import = builtins.__import__

    def blocked_import(name, *args, **kwargs):
        if name == "ruamel.yaml" or name.startswith("ruamel."):
            raise ImportError("blocked for test")
        return real_import(name, *args, **kwargs)

    # monkeypatch restores both the sys.modules entry and the package
    # attribute that a fresh import rebinds, so later tests keep using the
    # original module object rather than the ruamel-less reload.
    monkeypatch.delitem(sys.modules, "egocap.core.config_io")
    monkeypatch.setattr(core_pkg, "config_io", core_pkg.config_io)
    monkeypatch.setattr(builtins, "__import__", blocked_import)
    reloaded = importlib.import_module("egocap.core.config_io")
    assert reloaded.RUAMEL_AVAILABLE is False


def test_get_default_config_shape():
    default = ConfigIO._get_default_config()
    assert set(default.keys()) == {"recording", "video_sources", "numeric_sources"}
