# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest
import yaml
import zarr

from egocap.data_processing.update_episode_task_description import (
    MAX_EPISODE_SPAN,
    EpisodeRangeError,
    _parse_episode_arg,
    _update_session_config_yaml,
    update_episode_task_description,
)


def test_parse_episode_arg_single():
    assert list(_parse_episode_arg("5")) == [5]


def test_parse_episode_arg_range():
    assert list(_parse_episode_arg("2-7")) == [2, 3, 4, 5, 6, 7]


def test_parse_episode_arg_reversed_range_normalizes():
    assert list(_parse_episode_arg("7-2")) == [2, 3, 4, 5, 6, 7]


def test_parse_episode_arg_strips_whitespace():
    assert list(_parse_episode_arg(" 2 - 4 ")) == [2, 3, 4]


def test_parse_episode_arg_invalid_raises():
    with pytest.raises(EpisodeRangeError):
        _parse_episode_arg("not-a-range")
    with pytest.raises(EpisodeRangeError):
        _parse_episode_arg("")


def test_parse_episode_arg_refuses_a_range_no_session_could_hold_without_walking_it():
    assert _parse_episode_arg(f"5-{5 + MAX_EPISODE_SPAN - 1}") == range(5, 5 + MAX_EPISODE_SPAN)
    for spec in (f"5-{5 + MAX_EPISODE_SPAN}", "0-3000000000", "3000000000-0"):
        with pytest.raises(EpisodeRangeError, match=f"at most {MAX_EPISODE_SPAN} episodes"):
            _parse_episode_arg(spec)
    # More digits than Python converts at all is refused the same way, not as a different error.
    with pytest.raises(EpisodeRangeError, match="too long"):
        _parse_episode_arg("9" * 5000)


def test_update_session_config_yaml_missing_file_is_noop(tmp_path):
    _update_session_config_yaml(tmp_path, task_description="x", task_name="", studio="")
    assert not (tmp_path / "config.yaml").exists()


def test_update_session_config_yaml_updates_meta(tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text("meta:\n  operator_name: Alice\n", encoding="utf-8")
    _update_session_config_yaml(
        tmp_path, task_description="pour cereal", task_name="pour_cereal", studio="Kitchen"
    )
    data = yaml.safe_load(cfg.read_text(encoding="utf-8"))
    assert data["meta"]["task_description"] == "pour cereal"
    assert data["meta"]["task_name"] == "pour_cereal"
    assert data["meta"]["studio"] == "Kitchen"
    assert data["meta"]["operator_name"] == "Alice"  # preserved


def test_update_session_config_yaml_no_meta_section_creates_one(tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text("recording:\n  record_fps: 60\n", encoding="utf-8")
    _update_session_config_yaml(tmp_path, task_description="x", task_name="", studio="")
    data = yaml.safe_load(cfg.read_text(encoding="utf-8"))
    assert data["meta"]["task_description"] == "x"


def test_update_session_config_yaml_non_dict_top_level_becomes_dict(tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text("- just\n- a\n- list\n", encoding="utf-8")
    _update_session_config_yaml(tmp_path, task_description="x", task_name="", studio="")
    data = yaml.safe_load(cfg.read_text(encoding="utf-8"))
    assert data["meta"]["task_description"] == "x"


def test_update_session_config_yaml_malformed_yaml_swallowed(tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text("meta: [unbalanced\n", encoding="utf-8")
    # Should not raise -- malformed config is a no-op, not a crash.
    _update_session_config_yaml(tmp_path, task_description="x", task_name="", studio="")


def test_update_session_config_yaml_unreadable_swallowed(tmp_path, monkeypatch):
    cfg = tmp_path / "config.yaml"
    cfg.write_text("meta: {}\n", encoding="utf-8")

    real_open = open

    def boom(p, *a, **k):
        if str(p) == str(cfg):
            raise OSError("permission denied")
        return real_open(p, *a, **k)

    monkeypatch.setattr("builtins.open", boom)
    _update_session_config_yaml(tmp_path, task_description="x", task_name="", studio="")


def _make_session(tmp_path: Path, n_episodes: int = 3) -> Path:
    store = zarr.open(str(tmp_path), mode="a")
    for i in range(n_episodes):
        g = store.create_group(f"episode_{i}")
        g.attrs["episode_length"] = 10
        g.attrs["episode_duration"] = 1.0
    return tmp_path


def test_update_episode_task_description_updates_requested_episodes(tmp_path):
    _make_session(tmp_path, 3)
    result = update_episode_task_description(
        tmp_path, "0-1", task_description="pour cereal", task_name="pour", studio="Kitchen"
    )
    assert result.updated_episodes == ["episode_0", "episode_1"]
    assert result.missing_episodes == []
    assert result.studio_stored == "Kitchen"
    assert result.aggregates_refreshed is True

    store = zarr.open(str(tmp_path), mode="r")
    assert store["episode_0"].attrs["task_description"] == "pour cereal"
    assert store["episode_0"].attrs["task_name"] == "pour"
    assert store["episode_0"].attrs["studio"] == "Kitchen"
    assert "task_description" not in store["episode_2"].attrs
    # Two of three takes were corrected, so the session's own task is left alone; the studio is the session's.
    assert "task_description" not in store.attrs and "task_name" not in store.attrs
    assert store.attrs["studio"] == "Kitchen"


def test_correcting_every_take_also_relabels_the_session(tmp_path):
    _make_session(tmp_path, 3)
    (tmp_path / "config.yaml").write_text("meta: {}\n", encoding="utf-8")
    result = update_episode_task_description(tmp_path, "0-5", task_description="pour cereal", task_name="pour")
    assert result.updated_episodes == ["episode_0", "episode_1", "episode_2"] and result.config_yaml_updated
    store = zarr.open(str(tmp_path), mode="r")
    assert (store.attrs["task_description"], store.attrs["task_name"]) == ("pour cereal", "pour")
    assert yaml.safe_load((tmp_path / "config.yaml").read_text())["meta"] == {
        "task_description": "pour cereal",
        "task_name": "pour",
    }


def test_correcting_some_takes_leaves_the_sessions_config_task_alone(tmp_path):
    _make_session(tmp_path, 2)
    (tmp_path / "config.yaml").write_text("meta:\n  task_description: before\n", encoding="utf-8")
    update_episode_task_description(tmp_path, "1", task_description="after", task_name="t", studio="Kitchen")
    assert yaml.safe_load((tmp_path / "config.yaml").read_text())["meta"] == {
        "task_description": "before",
        "studio": "Kitchen",
    }


def test_a_corrected_take_carries_the_new_task_under_both_of_its_names(tmp_path):
    _make_session(tmp_path, 2)
    store = zarr.open(str(tmp_path), mode="a")
    store["episode_0"].attrs["instruction_language"] = "pour water"
    update_episode_task_description(tmp_path, "0-1", task_description="wipe the table")
    store = zarr.open(str(tmp_path), mode="r")
    assert store["episode_0"].attrs["instruction_language"] == "wipe the table"
    # An episode that never carried the second name does not gain it.
    assert "instruction_language" not in store["episode_1"].attrs


def test_update_episode_task_description_missing_episode_recorded(tmp_path):
    _make_session(tmp_path, 1)
    result = update_episode_task_description(tmp_path, "0-2", task_description="x")
    assert result.updated_episodes == ["episode_0"]
    assert result.missing_episodes == ["episode_1", "episode_2"]


def test_update_episode_task_description_no_studio_no_task_name(tmp_path):
    _make_session(tmp_path, 1)
    result = update_episode_task_description(tmp_path, "0", task_description="x")
    assert result.studio_stored == ""
    store = zarr.open(str(tmp_path), mode="r")
    assert "task_name" not in store["episode_0"].attrs
    assert "studio" not in store["episode_0"].attrs


def test_update_episode_task_description_relocate_requires_studio(tmp_path):
    _make_session(tmp_path, 1)
    with pytest.raises(ValueError, match="non-empty studio"):
        update_episode_task_description(tmp_path, "0", task_description="x", relocate=True)


def test_update_episode_task_description_relocate_moves_session(tmp_path):
    eod_root = tmp_path / "EOD"
    session = eod_root / "_unmatched" / "2026-01-01-10-00-00"
    _make_session(session, 1)
    result = update_episode_task_description(session, "0", task_description="x", studio="Kitchen", relocate=True)
    assert result.zarr_root == eod_root / "Kitchen" / "2026-01-01-10-00-00"
    assert result.zarr_root.is_dir()


def test_update_episode_task_description_missing_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        update_episode_task_description(tmp_path / "missing", "0", task_description="x")


def test_update_episode_task_description_updates_config_yaml_sidecar(tmp_path):
    _make_session(tmp_path, 1)
    (tmp_path / "config.yaml").write_text("meta: {}\n", encoding="utf-8")
    result = update_episode_task_description(
        tmp_path, "0", task_description="pour cereal", task_name="pour", studio="Kitchen"
    )
    assert result.config_yaml_updated is True
    data = yaml.safe_load((tmp_path / "config.yaml").read_text(encoding="utf-8"))
    assert data["meta"]["task_description"] == "pour cereal"


def test_a_partial_correction_relabels_only_the_named_take_of_the_aligned_copy(tmp_path):
    session = tmp_path / "session"
    _make_session(session, 2)
    aligned = _make_aligned_copy(session)
    update_episode_task_description(session, "0", task_description="pour cereal")
    store = zarr.open(str(aligned), mode="r")
    assert store["episode_0"].attrs["task_description"] == "pour cereal"
    assert "task_description" not in store.attrs and "studio" not in store.attrs
    assert yaml.safe_load((aligned / "config.yaml").read_text(encoding="utf-8"))["meta"] == {}


def _make_aligned_copy(session: Path) -> Path:
    aligned = session.with_name(f"{session.name}_aligned")
    store = zarr.open(str(aligned), mode="a")
    store.attrs["index_source"] = "head"
    store.create_group("episode_0").attrs["episode_length"] = 5
    (aligned / "config.yaml").write_text("meta: {}\n", encoding="utf-8")
    return aligned


def test_update_episode_task_description_updates_the_aligned_copy_beside_the_session(tmp_path):
    session = tmp_path / "session"
    _make_session(session, 2)
    aligned = _make_aligned_copy(session)
    update_episode_task_description(
        session, "0-1", task_description="pour cereal", task_name="pour", studio="Kitchen"
    )
    store = zarr.open(str(aligned), mode="r")
    for attrs in (store.attrs, store["episode_0"].attrs):
        assert attrs["task_description"] == "pour cereal"
        assert attrs["task_name"] == "pour" and attrs["studio"] == "Kitchen"
    assert "episode_1" not in store and store.attrs["index_source"] == "head"
    meta = yaml.safe_load((aligned / "config.yaml").read_text(encoding="utf-8"))["meta"]
    assert meta["task_description"] == "pour cereal"


def test_update_episode_task_description_leaves_the_aligned_name_and_studio_when_not_given(tmp_path):
    session = tmp_path / "session"
    _make_session(session, 1)
    aligned = _make_aligned_copy(session)
    update_episode_task_description(session, "0", task_description="x")
    attrs = zarr.open(str(aligned), mode="r").attrs
    assert attrs["task_description"] == "x" and "task_name" not in attrs and "studio" not in attrs


def test_update_episode_task_description_bad_range_with_relocate_raises_without_moving(tmp_path):
    # Regression: episodes must be parsed before relocating, so a malformed
    # range raises and leaves the session at its original location rather
    # than relocated with no episode updates applied.
    eod_root = tmp_path / "EOD"
    session = eod_root / "_unmatched" / "2026-01-01-10-00-00"
    _make_session(session, 1)
    with pytest.raises(EpisodeRangeError):
        update_episode_task_description(
            session, "not-a-range", task_description="x", studio="Kitchen", relocate=True
        )
    assert session.is_dir()
    assert not (eod_root / "Kitchen").exists()


def test_update_episode_task_description_aggregates_refresh_failure_recorded(tmp_path, monkeypatch):
    _make_session(tmp_path, 1)
    import egocap.data_processing.update_episode_task_description as mod

    def boom(*a, **k):
        raise RuntimeError("zarr aggregate refresh failed")

    monkeypatch.setattr(mod, "refresh_session_egocap_zarr_aggregates", boom)
    result = update_episode_task_description(tmp_path, "0", task_description="x")
    assert result.aggregates_refreshed is False


def test_update_episode_task_description_relocating_a_managed_session_raises_and_changes_nothing(tmp_path):
    session = tmp_path / "EOD" / "_unmatched" / "2026-01-01-10-00-00"
    _make_session(session, 1)
    (session / ".recording").mkdir()
    (session / ".recording" / "episode_0.json").write_text("{}", encoding="utf-8")
    (session / "config.yaml").write_text("meta: {}\n", encoding="utf-8")
    _make_aligned_copy(session)
    before = {str(f.relative_to(tmp_path)): f.read_bytes() for f in sorted(tmp_path.rglob("*")) if f.is_file()}
    with pytest.raises(OSError, match="fixed by its episode manifests"):
        update_episode_task_description(session, "0", task_description="x", studio="Kitchen", relocate=True)
    after = {str(f.relative_to(tmp_path)): f.read_bytes() for f in sorted(tmp_path.rglob("*")) if f.is_file()}
    assert after == before and not (tmp_path / "EOD" / "Kitchen").exists()


def test_a_failed_config_write_leaves_the_file_as_it_was_and_no_temporary_file(tmp_path, monkeypatch):
    cfg = tmp_path / "config.yaml"
    cfg.write_text("meta:\n  task_description: before\n", encoding="utf-8")

    def refuse(*_args, **_kwargs):
        raise yaml.YAMLError("cannot represent")

    monkeypatch.setattr(yaml, "dump", refuse)
    assert _update_session_config_yaml(tmp_path, task_description="after", task_name="", studio="") is False
    assert cfg.read_text(encoding="utf-8") == "meta:\n  task_description: before\n"
    assert sorted(path.name for path in tmp_path.iterdir()) == ["config.yaml"]


def test_the_config_is_replaced_whole_and_keeps_non_ascii_text(tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text("meta: {}\n", encoding="utf-8")
    assert _update_session_config_yaml(tmp_path, task_description="Café table", task_name="", studio="") is True
    assert "Café table" in cfg.read_text(encoding="utf-8")
    assert sorted(path.name for path in tmp_path.iterdir()) == ["config.yaml"]
