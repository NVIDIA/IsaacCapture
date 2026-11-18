# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CloudXR runtime settings: YAML -> env file -> verification.

Every case here is one of the three ways a runtime setting was silently
ignored before this module existed.
"""

from __future__ import annotations

import pytest

from isaaccapture_examples.camera_viz import cloudxr_env


# ── YAML -> env ───────────────────────────────────────────────────────


def test_defaults_apply_without_any_yaml():
    env = cloudxr_env.env_from_yaml({})
    assert env["NV_ENABLE_POSE_WAIT"] == "false"
    assert env["NV_CXR_RUNTIME_FOVEATION"] == "true"


def test_yaml_overrides_the_default():
    env = cloudxr_env.env_from_yaml({"cloudxr": {"NV_CXR_RUNTIME_FOVEATION": False}})
    assert env["NV_CXR_RUNTIME_FOVEATION"] == "false"


def test_booleans_are_lowercased():
    """The runtime matches "false" by exact strcmp and reads anything it does
    not recognise as TRUE -- so Python's "False" would silently mean true, the
    exact opposite of the config."""
    env = cloudxr_env.env_from_yaml({"cloudxr": {"NV_ENABLE_POSE_WAIT": False}})
    assert env["NV_ENABLE_POSE_WAIT"] == "false"
    assert "False" not in env.values()


def test_null_drops_a_camera_viz_default():
    env = cloudxr_env.env_from_yaml({"cloudxr": {"NV_ENABLE_POSE_WAIT": None}})
    assert "NV_ENABLE_POSE_WAIT" not in env


def test_numbers_survive_as_strings():
    env = cloudxr_env.env_from_yaml({"cloudxr": {"NV_MAX_FPS": 72}})
    assert env["NV_MAX_FPS"] == "72"


def test_a_launcher_computed_key_is_refused():
    """The launcher ignores these with a UserWarning buried in the output;
    refusing here names the YAML key instead."""
    with pytest.raises(ValueError, match="computed by the CloudXR launcher"):
        cloudxr_env.env_from_yaml({"cloudxr": {"XR_RUNTIME_JSON": "/tmp/x.json"}})


def test_a_non_mapping_block_is_refused():
    with pytest.raises(ValueError, match="must be a mapping"):
        cloudxr_env.env_from_yaml({"cloudxr": ["NV_MAX_FPS=72"]})


# ── env file ──────────────────────────────────────────────────────────


def test_env_file_is_written_without_export(tmp_path):
    """The launcher writes `export KEY=value` for humans to source, but its own
    parser splits on the first `=` without stripping the prefix -- an exported
    line arrives as the key "export KEY" and is dropped."""
    path = cloudxr_env.write_env_file({"NV_MAX_FPS": "72"}, tmp_path / "cloudxr.env")
    body = path.read_text()
    assert "NV_MAX_FPS=72" in body
    assert "export" not in body


def test_the_launcher_can_parse_what_we_write(tmp_path):
    """Round-trip through the real parser rather than trusting the format."""
    from isaaccapture.cloudxr.env_config import EnvConfig

    env = {"NV_DEVICE_PROFILE": "apple-vision-pro", "NV_ENABLE_POSE_WAIT": "false"}
    path = cloudxr_env.write_env_file(env, tmp_path / "cloudxr.env")
    assert EnvConfig._load_env_file(path) == env


def test_an_env_file_entry_beats_a_stale_shell_export(tmp_path, monkeypatch):
    """The whole reason this module exists: a sourced ~/.cloudxr/run/cloudxr.env
    leaves NV_DEVICE_PROFILE in the shell, which beats both os.environ
    .setdefault and --cloudxr-device-profile. Only the env file outranks it."""
    from isaaccapture.cloudxr.env_config import EnvConfig

    monkeypatch.setenv("NV_DEVICE_PROFILE", "Quest3")
    path = cloudxr_env.write_env_file(
        {"NV_DEVICE_PROFILE": "apple-vision-pro"}, tmp_path / "cloudxr.env"
    )
    EnvConfig._instance = None  # singleton: do not inherit another test's state
    EnvConfig.from_args(
        str(tmp_path / "install"),
        path,
        launcher_defaults={"NV_DEVICE_PROFILE": "Quest3"},
    )
    assert EnvConfig._instance.resolved("NV_DEVICE_PROFILE") == "apple-vision-pro"
    EnvConfig._instance = None


# ── a name nothing reads ──────────────────────────────────────────────


def test_an_unknown_name_warns_with_a_suggestion(capsys):
    """The failure this module exists for: NV_CXR_DEVICE_PROFILE sits between
    two variables spelled that way, and the runtime ignores it in silence."""
    env = cloudxr_env.env_from_yaml(
        {"cloudxr": {"NV_CXR_DEVICE_PROFILE": "apple-vision-pro"}}
    )
    err = capsys.readouterr().err
    assert "NV_CXR_DEVICE_PROFILE" in err
    assert "did you mean 'NV_DEVICE_PROFILE'?" in err
    # Warned, not dropped: the runtime has more knobs than camera_viz lists.
    assert env["NV_CXR_DEVICE_PROFILE"] == "apple-vision-pro"


def test_a_known_name_is_silent(capsys):
    cloudxr_env.env_from_yaml({"cloudxr": {"NV_DEVICE_PROFILE": "quest3"}})
    assert capsys.readouterr().err == ""


def test_every_default_is_a_known_name():
    """A default that is not in KNOWN_KEYS would warn on every single run."""
    assert set(cloudxr_env.DEFAULT_ENV) <= cloudxr_env.KNOWN_KEYS
