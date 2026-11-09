# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Deployment station YAML loader: flat/merged layouts, validation, and the unset-path default."""

from pathlib import Path

import pytest

from capture_server import protocol as p
from egocap_web.station_config import (
    STATION_CONFIG_ENV,
    ResolvedStation,
    StationConfigError,
    load_station_config,
    station_config_path_from_env,
)


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "station.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_unset_path_returns_none(tmp_path):
    assert load_station_config(None) is None


def test_flat_file_with_station_name_uses_it_as_id(tmp_path):
    path = _write(
        tmp_path,
        """
meta:
  egocap_station_name: Station 01
  operator_name: Alex
recording:
  data_dir: ./recordings
""",
    )
    resolved = load_station_config(path)
    assert resolved == ResolvedStation(
        station_id="Station 01",
        station_label="Station 01",
        operator_name="Alex",
        auto_fill_operator_name=False,
        task_description=None,
        data_dir=tmp_path / "recordings",
        cameras=(),
        latency={},
        alignment_index=None,
    )


def test_flat_file_without_station_name_gets_generic_id(tmp_path):
    path = _write(tmp_path, "meta: {}\n")
    resolved = load_station_config(path)
    assert resolved.station_id == "station_01"
    assert resolved.station_label is None
    assert resolved.operator_name == ""
    assert resolved.auto_fill_operator_name is False


def test_merged_file_selects_first_entry_in_yaml_order(tmp_path):
    path = _write(
        tmp_path,
        """
stations:
  station_02:
    meta:
      egocap_station_name: Station 02
  station_01:
    meta:
      egocap_station_name: Station 01
""",
    )
    resolved = load_station_config(path)
    assert resolved.station_id == "station_02"
    assert resolved.station_label == "Station 02"


def test_merged_file_with_empty_stations_mapping_is_rejected(tmp_path):
    path = _write(tmp_path, "stations: {}\n")
    with pytest.raises(StationConfigError, match="nonempty mapping"):
        load_station_config(path)


def test_duplicate_top_level_key_is_rejected(tmp_path):
    path = _write(tmp_path, "meta:\n  operator_name: A\nmeta:\n  operator_name: B\n")
    with pytest.raises(StationConfigError, match="duplicate key"):
        load_station_config(path)


def test_duplicate_station_key_in_merged_file_is_rejected(tmp_path):
    path = _write(
        tmp_path,
        """
stations:
  station_01:
    meta: {}
  station_01:
    meta: {}
""",
    )
    with pytest.raises(StationConfigError, match="duplicate key"):
        load_station_config(path)


def test_missing_file_is_a_station_config_error(tmp_path):
    with pytest.raises(StationConfigError, match="cannot read"):
        load_station_config(tmp_path / "missing.yaml")


def test_malformed_yaml_is_a_station_config_error(tmp_path):
    path = _write(tmp_path, "meta: [unbalanced\n")
    with pytest.raises(StationConfigError, match="not valid YAML"):
        load_station_config(path)


def test_non_utf8_file_is_a_station_config_error(tmp_path):
    path = tmp_path / "station.yaml"
    path.write_bytes("meta:\n  operator_name: Álex\n".encode("latin-1"))
    with pytest.raises(StationConfigError, match="not valid UTF-8"):
        load_station_config(path)


def test_an_out_of_range_date_scalar_is_a_station_config_error(tmp_path):
    # Not a YAMLError: PyYAML's own timestamp constructor raises a plain ValueError from
    # datetime's own validation (month 13 is impossible) for a bare (unquoted) date scalar.
    path = _write(tmp_path, "meta:\n  operator_name: 2026-13-45\n")
    with pytest.raises(StationConfigError, match="not valid"):
        load_station_config(path)


def test_an_unhashable_mapping_key_is_a_station_config_error(tmp_path):
    path = _write(tmp_path, "meta:\n  ? [a, b]\n  : value\n")
    with pytest.raises(StationConfigError, match="not valid"):
        load_station_config(path)


def test_non_mapping_document_is_rejected(tmp_path):
    path = _write(tmp_path, "- just\n- a\n- list\n")
    with pytest.raises(StationConfigError, match="must be a mapping"):
        load_station_config(path)


@pytest.mark.parametrize(
    "yaml_text,message",
    [
        ("meta:\n  operator_name: 5\n", "must be a string"),
        ("recording:\n  data_dir: 5\n", "nonempty string"),
        ("meta: not-a-mapping\n", "must be a mapping"),
        ("recording: not-a-mapping\n", "must be a mapping"),
        ("meta:\n  auto_fill_operator_name: 1\n", "must be a boolean"),
    ],
)
def test_invalid_consumed_field_types_are_rejected(tmp_path, yaml_text, message):
    path = _write(tmp_path, yaml_text)
    with pytest.raises(StationConfigError, match=message):
        load_station_config(path)


def test_auto_fill_operator_name_can_be_opted_into(tmp_path):
    path = _write(tmp_path, "meta:\n  operator_name: Alex\n  auto_fill_operator_name: true\n")
    assert load_station_config(path).auto_fill_operator_name is True


def test_relative_data_dir_resolves_against_the_config_files_own_directory(tmp_path):
    sub = tmp_path / "deploy"
    sub.mkdir()
    path = sub / "station.yaml"
    path.write_text("recording:\n  data_dir: ../recordings\n", encoding="utf-8")
    resolved = load_station_config(path)
    assert resolved.data_dir == sub / "../recordings"
    assert resolved.data_dir.resolve() == tmp_path / "recordings"


def test_absolute_data_dir_is_used_as_is(tmp_path):
    path = _write(tmp_path, f"recording:\n  data_dir: {tmp_path / 'abs'}\n")
    resolved = load_station_config(path)
    assert resolved.data_dir == tmp_path / "abs"


def test_no_video_sources_is_an_explicit_empty_camera_selection(tmp_path):
    path = _write(tmp_path, "meta:\n  operator_name: Alex\n")
    resolved = load_station_config(path)
    assert resolved.cameras == ()


def test_video_sources_become_arbitrary_named_camera_selections(tmp_path):
    path = _write(
        tmp_path,
        """
video_sources:
- type: oak_camera
  name: some_custom_camera_name
- type: oak_camera
  name: another_camera
""",
    )
    resolved = load_station_config(path)
    assert resolved.cameras == (
        p.CameraSelection(name="some_custom_camera_name", type=p.CameraType.OAK_D_COLOR),
        p.CameraSelection(name="another_camera", type=p.CameraType.OAK_D_COLOR),
    )


def test_an_unsupported_video_source_type_is_a_station_config_error(tmp_path):
    path = _write(tmp_path, "video_sources:\n- type: some_other_camera\n  name: head_human_camera\n")
    with pytest.raises(StationConfigError, match="oak_camera"):
        load_station_config(path)


def test_an_unhashable_video_source_type_does_not_crash(tmp_path):
    path = _write(tmp_path, "video_sources:\n- type: [not, a, string]\n  name: head_human_camera\n")
    with pytest.raises(StationConfigError, match="oak_camera"):
        load_station_config(path)


def test_a_duplicate_video_source_name_is_a_station_config_error(tmp_path):
    path = _write(
        tmp_path,
        "video_sources:\n- type: oak_camera\n  name: dup\n- type: oak_camera\n  name: dup\n",
    )
    with pytest.raises(StationConfigError, match="duplicate"):
        load_station_config(path)


def test_too_many_video_sources_is_a_station_config_error(tmp_path):
    entries = "\n".join(f"- type: oak_camera\n  name: cam{i}" for i in range(4))
    path = _write(tmp_path, f"video_sources:\n{entries}\n")
    with pytest.raises(StationConfigError, match="at most 3"):
        load_station_config(path)


def test_station_config_path_from_env(monkeypatch, tmp_path):
    monkeypatch.delenv(STATION_CONFIG_ENV, raising=False)
    assert station_config_path_from_env() is None

    monkeypatch.setenv(STATION_CONFIG_ENV, "")
    assert station_config_path_from_env() is None

    monkeypatch.setenv(STATION_CONFIG_ENV, str(tmp_path / "x.yaml"))
    assert station_config_path_from_env() == tmp_path / "x.yaml"


_SHIPPED_TEMPLATE = (
    Path(__file__).resolve().parents[2] / "recording" / "egocap" / "stations" / "ego_stations.example.yaml"
)


def test_the_shipped_template_resolves_a_real_station_label():
    # A deployment that copies the shipped template as-is (the documented first step) must get a
    # real, distinct label -- not the generic development default every station would otherwise
    # share if meta.egocap_station_name were left blank.
    resolved = load_station_config(_SHIPPED_TEMPLATE)
    assert resolved.station_id == "station_01"
    assert resolved.station_label == "Station 01"
    assert resolved.auto_fill_operator_name is False


def _latency_yaml(**seconds):
    return "alignment:\n  tracking_latency_s:\n" + "".join(
        f"    {name}: {value}\n" for name, value in seconds.items()
    )


def test_tracking_source_latencies_are_resolved_in_seconds(tmp_path):
    path = _write(tmp_path, _latency_yaml(pico_controller=0.05, manus_glove=1))
    assert load_station_config(path).latency == {"pico_controller": 0.05, "manus_glove": 1.0}


def test_numeric_sources_latency_is_ignored(tmp_path):
    path = _write(
        tmp_path,
        "numeric_sources:\n  - name: pico_controller\n    latency: 0.1\n  - name: manus_glove\n  - just_a_string\n",  # noqa: E501
    )
    assert load_station_config(path).latency == {}


def test_absent_or_empty_tracking_latency_resolves_to_no_latency(tmp_path):
    assert load_station_config(_write(tmp_path, "alignment:\n  tracking_latency_s:\n")).latency == {}
    assert load_station_config(_write(tmp_path, "alignment: {}\n")).latency == {}


@pytest.mark.parametrize("value", ["true", "'0.1'", "-0.1", ".inf", ".nan", "[1]"])
def test_an_invalid_tracking_latency_is_a_station_config_error(tmp_path, value):
    path = _write(tmp_path, _latency_yaml(pico_controller=value))
    with pytest.raises(StationConfigError, match="tracking_latency_s.pico_controller must be"):
        load_station_config(path)


def test_a_tracking_latency_for_an_unknown_source_is_a_station_config_error(tmp_path):
    with pytest.raises(StationConfigError, match="unknown source 'head'"):
        load_station_config(_write(tmp_path, _latency_yaml(head=0.1)))


@pytest.mark.parametrize(
    ("text", "match"),
    [
        ("alignment: [1]\n", "'alignment' must be a mapping"),
        ("alignment:\n  tracking_latency_s: [0.1]\n", "tracking_latency_s must be a mapping"),
    ],
)
def test_a_non_mapping_alignment_or_tracking_latency_is_a_station_config_error(tmp_path, text, match):
    with pytest.raises(StationConfigError, match=match):
        load_station_config(_write(tmp_path, text))


_CAMERAS = "video_sources:\n  - type: oak_camera\n    name: head\n  - type: oak_camera\n    name: wrist\n"


def test_alignment_is_off_without_an_index_source_even_with_cameras(tmp_path):
    assert load_station_config(_write(tmp_path, _CAMERAS)).alignment_index is None


def test_alignment_index_source_can_be_named(tmp_path):
    path = _write(tmp_path, _CAMERAS + "alignment:\n  index_source: wrist\n")
    assert load_station_config(path).alignment_index == "wrist"


def test_alignment_can_be_disabled(tmp_path):
    path = _write(tmp_path, _CAMERAS + "alignment:\n  enabled: false\n  index_source: head\n")
    assert load_station_config(path).alignment_index is None


def test_alignment_is_off_without_an_alignment_section(tmp_path):
    assert load_station_config(_write(tmp_path, "meta: {}\n")).alignment_index is None


@pytest.mark.parametrize("section", ["[a, b]", "yes", "3"])
def test_a_non_mapping_alignment_section_is_a_station_config_error(tmp_path, section):
    with pytest.raises(StationConfigError, match="alignment"):
        load_station_config(_write(tmp_path, _CAMERAS + f"alignment: {section}\n"))
