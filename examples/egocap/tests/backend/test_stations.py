# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""The station registry: its validation, its changes, and the file that keeps it."""

from __future__ import annotations

import json

import pytest

from egocap_web.stations import (
    MAX_STATION_NAME_CHARS,
    MAX_STATIONS,
    Station,
    StationError,
    StationNotFoundError,
    StationRegistry,
    StationStore,
    parse_registry,
    station_store_path,
)

BENCH = {"name": "Bench A", "camera_type": "oak_d", "camera_roles": {"ego": "cam_head", "wrist_left": "cam_left"}}
CART = {"name": "Cart B", "camera_roles": {"ego": "cam_top", "wrist_right": "cam_grip"}}


def _ids():
    counter = iter(range(1000))
    return lambda: f"st_{next(counter)}"


def _two_stations() -> StationRegistry:
    new_id = _ids()
    return StationRegistry().added(BENCH, new_id).added(CART, new_id)


def _refusal(action) -> str:
    with pytest.raises(StationError) as refused:
        action()
    return refused.value.public_message


def test_two_stations_with_different_cameras_round_trip_through_the_file(tmp_path):
    store = StationStore(tmp_path / "nested" / "stations.json")
    assert store.load() == StationRegistry()
    registry = _two_stations().selecting("st_1")
    store.save(registry)
    assert store.load() == registry
    assert registry.selected.name == "Cart B"
    assert registry.find("st_0").camera_roles == {"ego": "cam_head", "wrist_left": "cam_left", "wrist_right": None}
    assert json.loads(store.path.read_text())["stations"][1] == {
        "id": "st_1",
        "name": "Cart B",
        "camera_type": None,
        "camera_roles": {"ego": "cam_top", "wrist_left": None, "wrist_right": "cam_grip"},
        "headset_address": None,
    }
    assert sorted(path.name for path in store.path.parent.iterdir()) == ["stations.json"]


def test_a_camera_in_two_roles_is_refused_naming_the_camera_and_both_roles():
    duplicate = {"name": "Bench C", "camera_roles": {"ego": "cam_head", "wrist_left": "cam_head"}}
    assert _refusal(lambda: StationRegistry().added(duplicate)) == (
        "camera 'cam_head' cannot fill both ego and wrist_left"
    )
    # The same rule holds when an edit of one role collides with a role already saved.
    registry = _two_stations()
    assert "cannot fill both ego and wrist_right" in _refusal(
        lambda: registry.updated("st_1", {"camera_roles": {"ego": "cam_grip"}})
    )
    assert registry.find("st_1").camera_roles["ego"] == "cam_top"


@pytest.mark.parametrize(
    ("fields", "reason"),
    [
        ({"name": "bench a"}, "a station named 'Bench A' already exists"),
        ({"name": "  Bench   A "}, "a station named 'Bench A' already exists"),
        # Punctuation is dropped from the daily log's file name, so this would share Bench A's log.
        ({"name": "Bench_A"}, "a station named 'Bench A' already exists"),
        ({"name": "bench.a!"}, "a station named 'Bench A' already exists"),
        ({"name": "   "}, "station name cannot be empty"),
        ({}, "station name must be text"),
        ({"name": 7}, "station name must be text"),
        ({"name": "x" * (MAX_STATION_NAME_CHARS + 1)}, f"limited to {MAX_STATION_NAME_CHARS} characters"),
        ({"name": "Bay\x071"}, "control characters"),
        ({"name": "New", "camera_type": "webcam"}, "camera_type must be one of"),
        ({"name": "New", "camera_roles": {"overhead": "cam"}}, "unknown camera role(s): overhead"),
        ({"name": "New", "camera_roles": {"ego": ""}}, "camera_roles.ego must be a camera name or null"),
        ({"name": "New", "camera_roles": ["cam"]}, "camera_roles must be an object"),
        ({"name": "New", "headset_address": "headset.local"}, "'headset.local' is not an IP address"),
        ({"name": "New", "headset_address": 7}, "headset_address must be an IP address or null"),
        ({"name": "New", "colour": "red", "size": 1}, "unknown station field(s): colour, size"),
    ],
)
def test_an_invalid_new_station_is_refused_with_a_specific_reason(fields, reason):
    assert reason in _refusal(lambda: _two_stations().added(fields))


def test_a_change_that_is_not_an_object_is_refused():
    assert _refusal(lambda: StationRegistry().added(["Bench"])) == "a station change must be an object"
    assert _refusal(lambda: _two_stations().updated("st_0", "Bench")) == "a station change must be an object"


def test_a_name_is_tidied_and_a_legacy_body_token_is_ignored():
    registry = StationRegistry().added({"name": "  Bay \t 1 ", "control_token": "secret"}, _ids())
    assert registry.stations[0] == Station(station_id="st_0", name="Bay 1")


def test_a_new_id_that_is_taken_is_drawn_again():
    draws = iter(["st_0", "st_0", "st_9"])
    registry = StationRegistry().added(BENCH, lambda: next(draws)).added(CART, lambda: next(draws))
    assert [station.station_id for station in registry.stations] == ["st_0", "st_9"]


def test_a_real_id_is_generated_when_none_is_supplied():
    station_id = StationRegistry().added(BENCH).stations[0].station_id
    assert station_id.startswith("st_") and len(station_id) == 11


def test_the_registry_is_bounded():
    registry, new_id = StationRegistry(), _ids()
    for index in range(MAX_STATIONS):
        registry = registry.added({"name": f"Station {index}"}, new_id)
    assert f"at most {MAX_STATIONS} stations" in _refusal(lambda: registry.added({"name": "One more"}, new_id))


def test_an_update_changes_only_the_fields_it_names():
    registry = _two_stations().updated(
        "st_0", {"name": "Bench Z", "camera_type": None, "headset_address": " 192.168.004.20 ".replace("004", "4")}
    )
    station = registry.find("st_0")
    assert (station.name, station.camera_type, station.headset_address) == ("Bench Z", None, "192.168.4.20")
    assert station.camera_roles == {"ego": "cam_head", "wrist_left": "cam_left", "wrist_right": None}
    cleared = registry.updated("st_0", {"camera_roles": {"wrist_left": None}, "headset_address": None})
    assert cleared.find("st_0").camera_roles["wrist_left"] is None
    assert cleared.find("st_0").headset_address is None
    # Renaming a station to its own name, in another case, is not a duplicate of itself.
    assert registry.updated("st_0", {"name": "bench z"}).find("st_0").name == "bench z"
    assert registry.find("st_1") == _two_stations().find("st_1")


def test_an_update_is_refused_for_a_taken_name_or_bad_roles():
    registry = _two_stations()
    assert _refusal(lambda: registry.updated("st_0", {"name": "CART B"})) == (
        "a station named 'Cart B' already exists"
    )
    assert "camera_roles must be an object" in _refusal(lambda: registry.updated("st_0", {"camera_roles": "cam"}))


def test_an_unknown_station_is_not_found_for_every_change():
    registry = _two_stations()
    for action in (
        lambda: registry.updated("st_7", {"name": "X"}),
        lambda: registry.removed("st_7"),
        lambda: registry.selecting("st_7"),
    ):
        with pytest.raises(StationNotFoundError):
            action()


def test_selection_follows_removal():
    registry = _two_stations().selecting("st_0")
    assert registry.removed("st_1").selected_id == "st_0"
    emptied = registry.removed("st_0")
    assert emptied.selected is None and [station.name for station in emptied.stations] == ["Cart B"]
    assert registry.selecting(None).selected is None


def test_a_station_reports_why_it_cannot_record():
    bench = _two_stations().find("st_0")
    assert bench.problems(None) == []
    assert bench.problems(frozenset({"cam_head", "cam_left"})) == []
    assert bench.problems(frozenset({"cam_head"})) == [
        "camera 'cam_left' (left wrist) is not offered by the capture session"
    ]
    bare = Station(station_id="st_9", name="Bare")
    assert bare.problems(None) == ["no camera is assigned to the ego (head) role"]
    assert bare.problems(frozenset()) == ["no camera is assigned to the ego (head) role"]


@pytest.mark.parametrize(
    ("raw", "reason"),
    [
        ([], "the station registry must be an object"),
        ({"stations": {}}, "stations must be a list"),
        ({"stations": [{"name": "A"}] * (MAX_STATIONS + 1)}, "stations must be a list"),
        ({"stations": ["Bench"]}, "each station must be an object"),
        ({"stations": [{"name": "A"}]}, "each station needs an id"),
        ({"stations": [{"id": "", "name": "A"}]}, "each station needs an id"),
        ({"stations": [{"id": "a", "name": "A"}, {"id": "a", "name": "B"}]}, "station id 'a' appears twice"),
        ({"stations": [{"id": "a", "name": "A"}, {"id": "b", "name": "a"}]}, "a station named 'A' already exists"),
        ({"stations": [{"id": "a", "name": "A"}], "selected": "b"}, "selected station is not in the registry"),
        ({"stations": [{"id": "a", "name": "A"}], "selected": 1}, "selected station is not in the registry"),
    ],
)
def test_a_hand_edited_registry_must_hold_to_the_same_rules(raw, reason):
    assert reason in _refusal(lambda: parse_registry(raw))


def test_an_empty_object_is_an_empty_registry():
    assert parse_registry({}) == StationRegistry()


@pytest.mark.parametrize("content", [b"not json", b"\xff\xfe", b"[1, 2]"])
def test_an_unusable_file_is_refused_on_load_and_set_aside_on_the_next_change(tmp_path, content):
    store = StationStore(tmp_path / "stations.json")
    store.path.write_bytes(content)
    with pytest.raises(StationError):
        store.load()
    before, after = store.change(lambda registry: registry.added(BENCH, _ids()))
    assert before == StationRegistry() and store.load() == after
    assert (tmp_path / "stations.json.invalid").read_bytes() == content


def test_a_refused_change_leaves_the_file_as_it_was(tmp_path):
    store = StationStore(tmp_path / "stations.json")
    store.save(_two_stations())
    saved = store.path.read_text()
    with pytest.raises(StationError):
        store.change(lambda registry: registry.added({"name": "Bench A"}))
    assert store.path.read_text() == saved
    assert sorted(path.name for path in tmp_path.iterdir()) == ["stations.json"]


def test_the_registry_sits_beside_the_operator_settings(tmp_path):
    assert station_store_path(tmp_path / ".egocap" / "operator_settings.json") == (
        tmp_path / ".egocap" / "stations.json"
    )
