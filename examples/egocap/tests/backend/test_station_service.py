# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""The station service: when the station in use may change, and what a change does to the live session."""

from __future__ import annotations

import pytest

from egocap_web.station_service import TAKE_IN_PROGRESS, StationService, session_adapter
from egocap_web.stations import StationError, StationNotFoundError, StationStore

BENCH = {"name": "Bench A", "camera_roles": {"ego": "cam_head", "wrist_left": "cam_left"}}
CART = {"name": "Cart B", "camera_roles": {"ego": "cam_top"}}


class _Session:
    """A stand-in live session that records what the service asked of it."""

    def __init__(self):
        self.offered: frozenset[str] | None = frozenset({"cam_head", "cam_left", "cam_top"})
        self.blocked: str | None = None
        self.rotation_refusal: str | None = None
        self.events: list[tuple[str, str]] = []

    def adapter(self):
        return session_adapter(
            offered_cameras=lambda: self.offered,
            change_blocked=lambda: self.blocked,
            show_label=lambda label: self.events.append(("show", label)),
            rotate_batch=self._rotate,
            leave_station=lambda label: self.events.append(("leave", label)),
        )

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
    assert session.events == [("rotate", ""), ("leave", "Deployment"), ("show", "Bench A")]
    session.events.clear()
    await service.select(cart_id)
    assert session.events == [("rotate", ""), ("leave", "Bench A"), ("show", "Cart B")]
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
        lambda: service.update_selected({"camera_type": "oak_d"}),
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


async def test_editing_the_station_in_use_applies_at_once(tmp_path):
    service, session, bench_id, _ = await _with_two_stations(tmp_path)
    await service.select(bench_id)
    session.events.clear()
    await service.update(bench_id, {"name": "Bench Z"})
    assert service.label() == "Bench Z"
    assert session.events == [("rotate", ""), ("leave", "Bench A"), ("show", "Bench Z")]
    session.events.clear()
    updated = await service.update_selected({"camera_roles": {"wrist_left": None}})
    assert updated.camera_roles == {"ego": "cam_head", "wrist_left": None, "wrist_right": None}
    assert session.events == []  # the label did not change


async def test_editing_this_station_with_nothing_selected_is_left_to_the_caller(tmp_path):
    service, _, _, _ = await _with_two_stations(tmp_path)
    assert await service.update_selected({"camera_type": "oak_d"}) is None


async def test_deleting_the_station_in_use_returns_to_the_deployment_label(tmp_path):
    service, session, bench_id, _ = await _with_two_stations(tmp_path)
    await service.select(bench_id)
    await service.remove(bench_id)
    assert service.registry.selected is None and service.label() == "Deployment"
    assert session.events[-2:] == [("leave", "Bench A"), ("show", "Deployment")]
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
    await service.update_selected({"camera_roles": {"ego": None, "wrist_left": None}})
    assert service.block_reason() == "station 'Bench A': no camera is assigned to the ego (head) role"
    session.offered = None  # before a capture session is accepted only the assignment itself is judged
    await service.update_selected({"camera_roles": {"ego": "cam_elsewhere"}})
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
