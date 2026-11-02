# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import json
import time
from pathlib import Path

from egocap.analytics.count_daily_trajectories import (
    _episode_index,
    _episode_recorded_date,
    _read_session_station_name,
    _read_zarr_json,
    compute_station_data_breakdown,
    count_trajectories,
    make_session_stats,
)


def _write_zarr_json(path: Path, attrs: dict) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "zarr.json").write_text(json.dumps({"attributes": attrs}), encoding="utf-8")


def _write_session_config(session_dir: Path, station_name: str | None) -> None:
    meta = f"meta:\n  egocap_station_name: {station_name}\n" if station_name else "meta: {}\n"
    (session_dir / "config.yaml").write_text(meta, encoding="utf-8")


def _build_session(
    data_dir: Path,
    studio: str,
    session_name: str,
    station_name: str,
    episodes: dict[int, dict],
    discarded: list[int] | None = None,
) -> Path:
    session_dir = data_dir / studio / session_name
    session_dir.mkdir(parents=True, exist_ok=True)
    _write_session_config(session_dir, station_name)
    _write_zarr_json(session_dir, {"discarded_episode_indices": discarded or []})
    for idx, attrs in episodes.items():
        _write_zarr_json(session_dir / f"episode_{idx}", attrs)
    return session_dir


def test_make_session_stats_defaults():
    stats = make_session_stats()
    assert stats["studio"] == ""
    assert stats["total"] == 0
    assert stats["descriptions"] == set()


def test_read_zarr_json_missing_file_returns_empty(tmp_path):
    assert _read_zarr_json(tmp_path / "missing.json") == {}


def test_read_zarr_json_malformed_returns_empty(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("not json", encoding="utf-8")
    assert _read_zarr_json(path) == {}


def test_read_zarr_json_no_attributes_key_returns_empty(tmp_path):
    path = tmp_path / "zarr.json"
    path.write_text(json.dumps({}), encoding="utf-8")
    assert _read_zarr_json(path) == {}


def test_read_zarr_json_returns_attributes(tmp_path):
    path = tmp_path / "zarr.json"
    path.write_text(json.dumps({"attributes": {"x": 1}}), encoding="utf-8")
    assert _read_zarr_json(path) == {"x": 1}


def test_episode_index_valid_and_invalid(tmp_path):
    assert _episode_index(tmp_path / "episode_7") == 7
    assert _episode_index(tmp_path / "episode_7.pending") is None
    assert _episode_index(tmp_path / "not_an_episode") is None


def test_episode_recorded_date_from_sidecar(tmp_path):
    episode_dir = tmp_path / "episode_0"
    episode_dir.mkdir()
    (episode_dir / "recorded_at.txt").write_text("2026-03-04T10:00:00\n", encoding="utf-8")
    assert _episode_recorded_date({}, episode_dir, tmp_path) == "2026-03-04"


def test_episode_recorded_date_sidecar_too_short_falls_through(tmp_path):
    episode_dir = tmp_path / "episode_0"
    episode_dir.mkdir()
    (episode_dir / "recorded_at.txt").write_text("202603", encoding="utf-8")
    attrs = {"recorded_at": "2026-05-06; 12-00-00"}
    assert _episode_recorded_date(attrs, episode_dir, tmp_path) == "2026-05-06"


def test_episode_recorded_date_from_zarr_attr(tmp_path):
    episode_dir = tmp_path / "episode_0"
    episode_dir.mkdir()
    attrs = {"recorded_at": "2026-05-06; 12-00-00"}
    assert _episode_recorded_date(attrs, episode_dir, tmp_path) == "2026-05-06"


def test_episode_recorded_date_from_mtime(tmp_path):
    episode_dir = tmp_path / "episode_0"
    episode_dir.mkdir()
    expected = time.strftime("%Y-%m-%d", time.localtime(episode_dir.stat().st_mtime))
    assert _episode_recorded_date({}, episode_dir, tmp_path) == expected


def test_episode_recorded_date_from_session_name_fallback(tmp_path, monkeypatch):
    episode_dir = tmp_path / "episode_0"
    episode_dir.mkdir()
    session_dir = tmp_path / "2026-01-02-03-04-05"

    real_stat = Path.stat

    def boom(self, *a, **k):
        if self == episode_dir:
            raise OSError("stat failed")
        return real_stat(self, *a, **k)

    monkeypatch.setattr(Path, "stat", boom)
    assert _episode_recorded_date({}, episode_dir, session_dir) == "2026-01-02"


def test_episode_recorded_date_no_resolution_returns_empty(tmp_path, monkeypatch):
    episode_dir = tmp_path / "episode_0"
    episode_dir.mkdir()
    session_dir = tmp_path / "not-a-timestamp"

    real_stat = Path.stat

    def boom(self, *a, **k):
        if self == episode_dir:
            raise OSError("stat failed")
        return real_stat(self, *a, **k)

    monkeypatch.setattr(Path, "stat", boom)
    assert _episode_recorded_date({}, episode_dir, session_dir) == ""


def test_episode_recorded_date_sidecar_unreadable_falls_through(tmp_path, monkeypatch):
    episode_dir = tmp_path / "episode_0"
    episode_dir.mkdir()
    sidecar = episode_dir / "recorded_at.txt"
    sidecar.write_text("2026-03-04", encoding="utf-8")

    real_read_text = Path.read_text

    def boom(self, *a, **k):
        if self == sidecar:
            raise OSError("unreadable")
        return real_read_text(self, *a, **k)

    monkeypatch.setattr(Path, "read_text", boom)
    attrs = {"recorded_at": "2026-05-06; 12-00-00"}
    assert _episode_recorded_date(attrs, episode_dir, tmp_path) == "2026-05-06"


def test_read_session_station_name_missing_config_returns_none(tmp_path):
    assert _read_session_station_name(tmp_path) is None


def test_read_session_station_name_found(tmp_path):
    _write_session_config(tmp_path, "station_01")
    assert _read_session_station_name(tmp_path) == "station_01"


def test_read_session_station_name_malformed_yaml_returns_none(tmp_path):
    (tmp_path / "config.yaml").write_text("meta: [unbalanced\n", encoding="utf-8")
    assert _read_session_station_name(tmp_path) is None


def test_read_session_station_name_no_meta_returns_none(tmp_path):
    (tmp_path / "config.yaml").write_text("recording:\n  record_fps: 60\n", encoding="utf-8")
    assert _read_session_station_name(tmp_path) is None


def test_read_session_station_name_meta_not_dict_returns_none(tmp_path):
    (tmp_path / "config.yaml").write_text("meta: not_a_dict\n", encoding="utf-8")
    assert _read_session_station_name(tmp_path) is None


def test_read_session_station_name_name_not_present_returns_none(tmp_path):
    (tmp_path / "config.yaml").write_text("meta:\n  other_field: x\n", encoding="utf-8")
    assert _read_session_station_name(tmp_path) is None


def test_read_session_station_name_unreadable_file_returns_none(tmp_path, monkeypatch):
    cfg = tmp_path / "config.yaml"
    cfg.write_text("meta:\n  egocap_station_name: X\n", encoding="utf-8")

    real_path_open = Path.open

    def boom(self, *a, **k):
        if self == cfg:
            raise OSError("permission denied")
        return real_path_open(self, *a, **k)

    monkeypatch.setattr(Path, "open", boom)
    assert _read_session_station_name(tmp_path) is None


def test_read_session_station_name_yaml_unavailable_returns_none(tmp_path, monkeypatch):
    # Exercises the lazy `import yaml` -> `except ImportError: return None`
    # branch by blocking the import for this call only.
    import builtins

    _write_session_config(tmp_path, "station_01")
    real_import = builtins.__import__

    def blocked_import(name, *args, **kwargs):
        if name == "yaml":
            raise ImportError("blocked for test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_import)
    assert _read_session_station_name(tmp_path) is None


def test_compute_station_data_breakdown_empty_station_or_missing_dir(tmp_path):
    assert compute_station_data_breakdown(tmp_path / "missing", "X", "2026-01-01") == {
        "total_seconds": 0.0,
        "valid_seconds": 0.0,
        "excluded_seconds": 0.0,
    }
    tmp_path.mkdir(exist_ok=True)
    assert compute_station_data_breakdown(tmp_path, "", "2026-01-01")["total_seconds"] == 0.0


def test_compute_station_data_breakdown_sums_matching_station_and_date(tmp_path):
    session_dir = _build_session(
        tmp_path,
        "Kitchen",
        "2026-01-01-10-00-00",
        "station_01",
        episodes={
            0: {"episode_duration": 5.0, "recorded_at": "2026-01-01; 10-00-00"},
            1: {"episode_duration": 3.0, "recorded_at": "2026-01-01; 10-05-00"},
        },
        discarded=[1],
    )
    breakdown = compute_station_data_breakdown(tmp_path, "station_01", "2026-01-01")
    assert breakdown["total_seconds"] == 8.0
    assert breakdown["valid_seconds"] == 5.0
    assert breakdown["excluded_seconds"] == 3.0
    assert session_dir.is_dir()


def test_compute_station_data_breakdown_skips_non_matching_station(tmp_path):
    _build_session(
        tmp_path,
        "Kitchen",
        "2026-01-01-10-00-00",
        "station_02",
        episodes={0: {"episode_duration": 5.0, "recorded_at": "2026-01-01; 10-00-00"}},
    )
    breakdown = compute_station_data_breakdown(tmp_path, "station_01", "2026-01-01")
    assert breakdown["total_seconds"] == 0.0


def test_compute_station_data_breakdown_skips_aligned_and_non_dirs(tmp_path):
    _build_session(
        tmp_path,
        "Kitchen",
        "2026-01-01-10-00-00_aligned",
        "station_01",
        episodes={0: {"episode_duration": 5.0, "recorded_at": "2026-01-01; 10-00-00"}},
    )
    studio_dir = tmp_path / "Kitchen"
    (studio_dir / "not_a_dir_file").write_text("", encoding="utf-8")
    (tmp_path / "top_level_file").write_text("", encoding="utf-8")
    breakdown = compute_station_data_breakdown(tmp_path, "station_01", "2026-01-01")
    assert breakdown["total_seconds"] == 0.0


def test_compute_station_data_breakdown_skips_non_episode_entries_and_date_mismatch(tmp_path):
    session_dir = _build_session(
        tmp_path,
        "Kitchen",
        "2026-01-01-10-00-00",
        "station_01",
        episodes={0: {"episode_duration": 5.0, "recorded_at": "2026-02-02; 10-00-00"}},
    )
    (session_dir / "not_an_episode_dir").mkdir()
    (session_dir / "episode_0" / "nested_file").write_text("", encoding="utf-8")
    breakdown = compute_station_data_breakdown(tmp_path, "station_01", "2026-01-01")
    assert breakdown["total_seconds"] == 0.0


def test_compute_station_data_breakdown_skips_episode_with_no_attrs(tmp_path):
    session_dir = _build_session(
        tmp_path,
        "Kitchen",
        "2026-01-01-10-00-00",
        "station_01",
        episodes={},
    )
    (session_dir / "episode_0").mkdir()  # no zarr.json -> empty attrs
    breakdown = compute_station_data_breakdown(tmp_path, "station_01", "2026-01-01")
    assert breakdown["total_seconds"] == 0.0


def test_count_trajectories_missing_dir_returns_empty(tmp_path):
    assert count_trajectories(tmp_path / "missing", "2026-01-01") == {}


def test_count_trajectories_counts_valid_and_excluded(tmp_path):
    _build_session(
        tmp_path,
        "Kitchen",
        "2026-01-01-10-00-00",
        "station_01",
        episodes={
            0: {
                "episode_duration": 5.0,
                "recorded_at": "2026-01-01; 10-00-00",
                "task_description": "pour cereal",
            },
            1: {"episode_duration": 3.0, "recorded_at": "2026-01-01; 10-05-00"},
        },
        discarded=[1],
    )
    result = count_trajectories(tmp_path, "2026-01-01", track_language=True)
    key = "Kitchen/2026-01-01-10-00-00"
    assert result[key]["total"] == 2
    assert result[key]["valid"] == 1
    assert result[key]["excluded"] == 1
    assert result[key]["valid_seconds"] == 5.0
    assert result[key]["excluded_seconds"] == 3.0
    assert result[key]["descriptions"] == {"pour cereal"}
    assert result[key]["studio"] == "Kitchen"
    assert result[key]["timestamp"] == "2026-01-01-10-00-00"


def test_count_trajectories_track_language_false_skips_descriptions(tmp_path):
    _build_session(
        tmp_path,
        "Kitchen",
        "2026-01-01-10-00-00",
        "station_01",
        episodes={
            0: {
                "episode_duration": 5.0,
                "recorded_at": "2026-01-01; 10-00-00",
                "task_description": "pour cereal",
            }
        },
    )
    result = count_trajectories(tmp_path, "2026-01-01", track_language=False)
    assert result["Kitchen/2026-01-01-10-00-00"]["descriptions"] == set()


def test_count_trajectories_date_mismatch_produces_no_session(tmp_path):
    _build_session(
        tmp_path,
        "Kitchen",
        "2026-01-01-10-00-00",
        "station_01",
        episodes={0: {"episode_duration": 5.0, "recorded_at": "2026-02-02; 10-00-00"}},
    )
    assert count_trajectories(tmp_path, "2026-01-01") == {}


def test_count_trajectories_skips_aligned_and_non_dir_entries(tmp_path):
    _build_session(
        tmp_path,
        "Kitchen",
        "2026-01-01-10-00-00_aligned",
        "station_01",
        episodes={0: {"episode_duration": 5.0, "recorded_at": "2026-01-01; 10-00-00"}},
    )
    (tmp_path / "Kitchen" / "not_a_dir_file").write_text("", encoding="utf-8")
    (tmp_path / "top_level_file").write_text("", encoding="utf-8")
    assert count_trajectories(tmp_path, "2026-01-01") == {}


def test_count_trajectories_skips_non_episode_dirs_and_empty_attrs(tmp_path):
    session_dir = _build_session(
        tmp_path,
        "Kitchen",
        "2026-01-01-10-00-00",
        "station_01",
        episodes={},
    )
    (session_dir / "not_an_episode").mkdir()
    (session_dir / "episode_x").mkdir()  # fails int() parse
    (session_dir / "episode_0").mkdir()  # no zarr.json
    assert count_trajectories(tmp_path, "2026-01-01") == {}
