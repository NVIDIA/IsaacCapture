# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""The station service: when the station in use may change, and what a change does to the live session."""

from __future__ import annotations

import pytest

from egocap_web.station_service import STATION_CHANGING, TAKE_IN_PROGRESS, StationService, session_adapter
from egocap_web.stations import StationError, StationNotFoundError, StationStore

BENCH = {"name": "Bench A", "camera_roles": {"ego": "cam_head", "wrist_left": "cam_left"}}
CART = {"name": "Cart B", "camera_roles": {"ego": "cam_top"}}


class _Session:
    """A stand-in live session that records what the service asked of it."""

    def __init__(self):
        self.offered: frozenset[str] | None = frozenset({"cam_head", "cam_left", "cam_top"})
        self.blocked: str | None = None
        self.rotation_refusal: str | None = None
        self.rename_failure: OSError | None = None
        self.events: list[tuple[str, str]] = []

    def adapter(self):
        return session_adapter(
            offered_cameras=lambda: self.offered,
            change_blocked=lambda: self.blocked,
            show_label=lambda label: self.events.append(("show", label)),
            rotate_batch=self._rotate,
            renamed=self._renamed,
        )

    def _renamed(self, previous: str, name: str) -> None:
        if self.rename_failure is not None:
            raise self.rename_failure
        self.events.append(("renamed", f"{previous} -> {name}"))

    async def _rotate(self) -> None:
        if self.rotation_refusal is not None:
            raise StationError(self.rotation_refusal)
        self.events.append(("rotate", ""))


def _service(tmp_path, session: _Session | None = None) -> tuple[StationService, _Session]:
    session = session or _Session()
    return StationService(StationStore(tmp_path / "stations.json"), session.adapter(), "Deployment"), session


async def _with_two_stations(tmp_path) -> tuple[StationService, _Session, str, str]:
    service, session = _service(tmp_path)
    bench = await service.add(BENCH)
    cart = await service.add(CART)
    return service, session, bench.station_id, cart.station_id


async def test_an_empty_registry_keeps_the_deployment_label_and_blocks_nothing(tmp_path):
    service, session = _service(tmp_path)
    assert service.label() == "Deployment" and service.block_reason() is None
    assert service.listing(private=False) == {
        "stations": [],
        "selected": None,
        "roles": ["ego", "wrist_left", "wrist_right"],
        "camera_types": ["sensing_gmsl", "oak_d"],
        "label": "Deployment",
        "locked_reason": None,
    }
    assert session.events == []


async def test_adding_stations_changes_nothing_about_the_session_but_then_one_must_be_selected(tmp_path):
    service, session, bench_id, _ = await _with_two_stations(tmp_path)
    assert session.events == [] and service.label() == "Deployment"
    assert service.block_reason() == "select a station before recording"
    await service.select(bench_id)
    assert service.block_reason() is None


async def test_selecting_a_station_relabels_the_session_and_rotates_the_batch(tmp_path):
    service, session, bench_id, cart_id = await _with_two_stations(tmp_path)
    await service.select(bench_id)
    assert service.label() == "Bench A"
    # Rotated first: a session still open under the old label must never receive a take for the new one.
    assert session.events == [("rotate", ""), ("show", "Bench A")]
    session.events.clear()
    await service.select(cart_id)
    assert session.events == [("rotate", ""), ("show", "Cart B")]  # another station: nothing is carried over
    session.events.clear()
    await service.select(cart_id)  # no change of label, so nothing to apply
    assert session.events == []
    await service.select(None)
    assert service.label() == "Deployment" and session.events[-1] == ("show", "Deployment")


async def test_switching_is_blocked_during_a_take_and_allowed_afterwards(tmp_path):
    service, session, bench_id, cart_id = await _with_two_stations(tmp_path)
    await service.select(bench_id)
    session.events.clear()
    session.blocked = TAKE_IN_PROGRESS
    for change in (
        lambda: service.select(cart_id),
        lambda: service.select(None),
        lambda: service.update(bench_id, {"name": "Bench Z"}),
        lambda: service.update(bench_id, {"camera_type": "oak_d"}),
        lambda: service.remove(bench_id),
    ):
        with pytest.raises(StationError) as refused:
            await change()
        assert refused.value.public_message == TAKE_IN_PROGRESS
    assert service.registry.selected_id == bench_id and service.label() == "Bench A" and session.events == []
    assert service.listing(private=False)["locked_reason"] == TAKE_IN_PROGRESS
    # A station that is not in use is unaffected by the take, so it can still be edited, added or deleted.
    await service.update(cart_id, {"name": "Cart C"})
    await service.remove(cart_id)
    cart_id = (await service.add(CART)).station_id
    session.blocked = None
    await service.select(cart_id)
    assert service.label() == "Cart B"


async def test_a_take_is_held_for_the_moment_the_station_in_use_is_being_changed(tmp_path, monkeypatch):
    service, session, bench_id, cart_id = await _with_two_stations(tmp_path)
    during: list[tuple[str | None, str]] = []
    change = StationStore.change

    def writing(self, apply):
        # What a take admitted while the registry file is being written would be told, and labelled with.
        during.append((service.block_reason(), service.label()))
        return change(self, apply)

    monkeypatch.setattr(StationStore, "change", writing)
    await service.select(bench_id)
    assert during == [(STATION_CHANGING, "Deployment")]
    assert service.block_reason() is None and service.label() == "Bench A"
    # A change that leaves the label alone holds nothing.
    await service.remember_headset_address("10.0.0.42")
    assert during[-1] == (None, "Bench A")
    # A refused change releases the hold as well.
    session.rotation_refusal = "an episode is active"
    with pytest.raises(StationError):
        await service.select(cart_id)
    assert service.block_reason() is None


async def test_renaming_the_station_in_use_carries_its_day_to_the_new_name(tmp_path):
    service, session, bench_id, _ = await _with_two_stations(tmp_path)
    await service.select(bench_id)
    session.events.clear()
    await service.update(bench_id, {"name": "Bench Z"})
    assert service.label() == "Bench Z"
    # The same station under a new name: the batch rotates, the label changes, and its day follows it.
    assert session.events == [("rotate", ""), ("show", "Bench Z"), ("renamed", "Bench A -> Bench Z")]
    session.events.clear()
    await service.update(bench_id, {"camera_roles": {"wrist_left": None}})
    assert service.registry.selected.camera_roles == {"ego": "cam_head", "wrist_left": None, "wrist_right": None}
    assert session.events == []  # the label did not change


async def test_a_rename_is_kept_even_when_its_day_cannot_be_carried_over(tmp_path, caplog):
    service, session, bench_id, _ = await _with_two_stations(tmp_path)
    await service.select(bench_id)
    session.rename_failure = OSError("disk full")
    await service.update(bench_id, {"name": "Bench Z"})
    # The rename is saved and live; failing it now would leave the page and the registry disagreeing.
    assert (
        service.label() == "Bench Z" and StationStore(tmp_path / "stations.json").load().selected.name == "Bench Z"
    )
    assert "Could not carry the day's timeline" in caplog.text


async def test_the_headset_address_is_remembered_even_during_a_take(tmp_path):
    service, session, bench_id, _ = await _with_two_stations(tmp_path)
    await service.remember_headset_address("10.0.0.42")  # nothing in use: nothing to remember it for
    assert all(station.headset_address is None for station in service.registry.stations)
    await service.select(bench_id)
    session.events.clear()
    session.blocked = TAKE_IN_PROGRESS
    await service.remember_headset_address("10.0.0.42")
    assert service.registry.selected.headset_address == "10.0.0.42"
    assert StationStore(tmp_path / "stations.json").load().selected.headset_address == "10.0.0.42"
    assert session.events == []  # nothing about what is recorded changed


async def test_deleting_the_station_in_use_returns_to_the_deployment_label(tmp_path):
    service, session, bench_id, _ = await _with_two_stations(tmp_path)
    await service.select(bench_id)
    await service.remove(bench_id)
    assert service.registry.selected is None and service.label() == "Deployment"
    assert session.events[-2:] == [("rotate", ""), ("show", "Deployment")]
    assert service.block_reason() == "select a station before recording"


async def test_a_refused_rotation_refuses_the_whole_change(tmp_path):
    service, session, bench_id, _ = await _with_two_stations(tmp_path)
    session.rotation_refusal = "an episode is active"
    with pytest.raises(StationError):
        await service.select(bench_id)
    assert service.registry.selected is None and service.label() == "Deployment"
    assert StationStore(tmp_path / "stations.json").load().selected is None
    assert session.events == []


async def test_a_refused_change_never_rotates_the_batch(tmp_path):
    service, session, bench_id, _ = await _with_two_stations(tmp_path)
    await service.select(bench_id)
    session.events.clear()
    with pytest.raises(StationError):
        await service.update(bench_id, {"name": "Cart B"})
    with pytest.raises(StationNotFoundError):
        await service.select("st_missing")
    assert session.events == [] and service.label() == "Bench A"


async def test_the_station_in_use_blocks_recording_until_it_can_record(tmp_path):
    service, session, bench_id, _ = await _with_two_stations(tmp_path)
    await service.select(bench_id)
    session.offered = frozenset({"cam_head"})
    assert service.block_reason() == (
        "station 'Bench A': camera 'cam_left' (left wrist) is not offered by the capture session"
    )
    await service.update(bench_id, {"camera_roles": {"ego": None, "wrist_left": None}})
    assert service.block_reason() == "station 'Bench A': no camera is assigned to the ego (head) role"
    session.offered = None  # before a capture session is accepted only the assignment itself is judged
    await service.update(bench_id, {"camera_roles": {"ego": "cam_elsewhere"}})
    assert service.block_reason() is None


async def test_the_listing_shows_each_stations_problems_and_keeps_the_headset_address_private(tmp_path):
    service, session, bench_id, cart_id = await _with_two_stations(tmp_path)
    await service.update(cart_id, {"headset_address": "192.168.4.20", "camera_roles": {"ego": "cam_gone"}})
    await service.select(bench_id)
    public = service.listing(private=False)
    assert public["selected"] == bench_id and public["label"] == "Bench A"
    assert [station["problems"] for station in public["stations"]] == [
        [],
        ["camera 'cam_gone' (ego (head)) is not offered by the capture session"],
    ]
    assert all("headset_address" not in station for station in public["stations"])
    private = service.listing(private=True)
    assert [station["headset_address"] for station in private["stations"]] == [None, "192.168.4.20"]


async def test_the_registry_survives_a_restart(tmp_path):
    service, _, bench_id, _ = await _with_two_stations(tmp_path)
    await service.select(bench_id)
    restarted, session = _service(tmp_path)
    assert restarted.label() == "Bench A" and restarted.registry == service.registry
    assert session.events == []  # startup publishes the label itself; nothing is rotated


async def test_an_unreadable_registry_starts_empty_with_a_warning_until_it_is_replaced(tmp_path):
    (tmp_path / "stations.json").write_text("not json")
    service, _ = _service(tmp_path)
    assert service.registry.stations == () and service.block_reason() is None
    assert "unreadable" in service.listing(private=False)["warning"]
    await service.add(BENCH)
    assert "warning" not in service.listing(private=False)
    assert (tmp_path / "stations.json.invalid").read_text() == "not json"


async def test_a_registry_that_cannot_be_read_starts_empty_with_a_warning_and_is_not_set_aside(
    tmp_path, monkeypatch
):
    path = tmp_path / "stations.json"
    path.write_text('{"stations": [{"id": "st_1", "name": "Bench A"}]}')

    def denied(self):
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(StationStore, "load", denied)
    service, _ = _service(tmp_path)
    assert service.registry.stations == () and service.block_reason() is None
    assert service.listing(private=False)["warning"] == (
        "Saved stations cannot be read. Check the registry file's permissions."
    )
    assert path.exists() and not (tmp_path / "stations.json.invalid").exists()
