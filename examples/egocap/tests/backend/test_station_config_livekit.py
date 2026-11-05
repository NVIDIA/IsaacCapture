# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""The optional ``livekit_preview`` section of the station YAML."""

from pathlib import Path

import pytest

from egocap_web.livekit_preview import LiveKitPreview, LiveKitTrack
from egocap_web.station_config import StationConfigError, load_station_config

VALID = """
meta: {}
livekit_preview:
  url: wss://lk.example:7881
  room: sensing-preview
  publisher: orin-sensing
  tracks:
    - name: Head left
      track: sensing-left
    - name: Head right
      track: sensing-right
"""


def _load(tmp_path: Path, text: str):
    path = tmp_path / "station.yaml"
    path.write_text(text, encoding="utf-8")
    return load_station_config(path)


def _with_section(section: str) -> str:
    return "meta: {}\nlivekit_preview:\n" + section


def test_valid_section_parses_into_a_preview_with_ordered_tracks(tmp_path):
    assert _load(tmp_path, VALID).livekit_preview == LiveKitPreview(
        url="wss://lk.example:7881",
        room="sensing-preview",
        publisher="orin-sensing",
        tracks=(LiveKitTrack("Head left", "sensing-left"), LiveKitTrack("Head right", "sensing-right")),
    )


def test_plain_ws_url_is_accepted(tmp_path):
    text = VALID.replace("wss://", "ws://")
    assert _load(tmp_path, text).livekit_preview.url == "ws://lk.example:7881"


def test_absent_section_means_no_preview_and_otherwise_unchanged_station(tmp_path):
    resolved = _load(tmp_path, "meta:\n  egocap_station_name: S\n")
    assert resolved.livekit_preview is None
    assert resolved.station_id == "S"
    assert resolved.cameras == ()


def test_livekit_preview_is_not_a_capture_server_camera(tmp_path):
    assert _load(tmp_path, VALID).cameras == ()


def test_four_tracks_are_allowed(tmp_path):
    tracks = "".join(f"    - name: N{i}\n      track: t{i}\n" for i in range(4))
    section = f"  url: ws://h\n  room: r\n  publisher: p\n  tracks:\n{tracks}"
    assert len(_load(tmp_path, _with_section(section)).livekit_preview.tracks) == 4


def _section(url="ws://h", room="r", publisher="p", tracks="    - {name: A, track: a}\n") -> str:
    lines = []
    for key, value in (("url", url), ("room", room), ("publisher", publisher)):
        if value is not None:
            lines.append(f"  {key}: {value!r}\n" if value == "" else f"  {key}: {value}\n")
    if tracks is not None:
        lines.append("  tracks:\n" + tracks)
    return _with_section("".join(lines))


@pytest.mark.parametrize(
    ("text", "message"),
    [
        pytest.param("meta: {}\nlivekit_preview: nope\n", "must be a mapping", id="scalar-section"),
        pytest.param("meta: {}\nlivekit_preview: [a]\n", "must be a mapping", id="list-section"),
        pytest.param(_section(url=None), r"url must be a nonempty string", id="missing-url"),
        pytest.param(_section(url=""), r"url must be a nonempty string", id="empty-url"),
        pytest.param(_section(url="http://h"), "ws:// or wss://", id="http-url"),
        pytest.param(_section(url="localhost:7880"), "ws:// or wss://", id="schemeless-url"),
        pytest.param(_section(room=None), r"room must be a nonempty string", id="missing-room"),
        pytest.param(_section(room=""), r"room must be a nonempty string", id="empty-room"),
        pytest.param(_section(publisher=None), r"publisher must be a nonempty string", id="missing-publisher"),
        pytest.param(_section(publisher=""), r"publisher must be a nonempty string", id="empty-publisher"),
        pytest.param(_section(tracks=None), r"tracks must be a nonempty list", id="missing-tracks"),
        pytest.param(_section(tracks="    []\n"), r"tracks must be a nonempty list", id="empty-tracks"),
        pytest.param(
            _with_section("  url: ws://h\n  room: r\n  publisher: p\n  tracks: x\n"),
            r"tracks must be a nonempty list",
            id="non-list-tracks",
        ),
        pytest.param(
            _section(tracks="".join(f"    - {{name: N{i}, track: t{i}}}\n" for i in range(5))),
            "at most 4",
            id="too-many-tracks",
        ),
        pytest.param(
            _section(tracks="    - just-a-string\n"), r"tracks\[0\] must be a mapping", id="non-mapping-track"
        ),
        pytest.param(
            _section(tracks="    - {track: a}\n"),
            r"tracks\[0\]\.name must be a nonempty string",
            id="missing-name",
        ),
        pytest.param(
            _section(tracks="    - {name: '', track: a}\n"), r"tracks\[0\]\.name must be", id="empty-name"
        ),
        pytest.param(
            _section(tracks="    - {name: A}\n"),
            r"tracks\[0\]\.track must be a nonempty string",
            id="missing-track",
        ),
        pytest.param(
            _section(tracks="    - {name: A, track: ''}\n"), r"tracks\[0\]\.track must be", id="empty-track"
        ),
        pytest.param(
            _section(tracks="    - {name: A, track: 5}\n"), r"tracks\[0\]\.track must be", id="non-string-track"
        ),
        pytest.param(
            _section(tracks="    - {name: A, track: a}\n    - {name: B, track: a}\n"),
            "duplicate track",
            id="duplicate-track",
        ),
        pytest.param(
            _section(tracks="    - {name: A, track: a}\n    - {name: A, track: b}\n"),
            "duplicate name",
            id="duplicate-name",
        ),
    ],
)
def test_invalid_section_is_a_station_config_error(tmp_path, text, message):
    with pytest.raises(StationConfigError, match=message):
        _load(tmp_path, text)


def test_error_names_the_station(tmp_path):
    with pytest.raises(StationConfigError, match="livekit_preview"):
        _load(tmp_path, "meta: {}\nlivekit_preview: 3\n")
