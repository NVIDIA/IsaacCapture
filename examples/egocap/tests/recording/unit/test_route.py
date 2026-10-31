# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Studio resolution, group assignment, output numbering, group totals and the routed-path guards.

Everything runs on real directories and real JSON/zarr metadata files; no filesystem call is substituted.
"""

import datetime
import json
from pathlib import Path

import pytest

from egocap.recording.episode_layout import (
    ROUTE_ATTR,
    EpisodeManifest,
    final_dir,
    finalize_episode_dir,
    locate_episode,
    remove_owned_episode,
    route_id,
)
from egocap.recording.route import RoutingConfig, assign_output, refresh_group_totals, resolve_studio

NOW = datetime.datetime(2026, 10, 20, 9, 30, 0)
SESSION_ID = "session_1"


def _session(tmp_path: Path) -> Path:
    session = tmp_path / "out" / SESSION_ID
    (session / ".recording").mkdir(parents=True)
    return session


def _assign(session, *, studio="Kitchen", task="stack blocks", reserved=(), now=NOW):
    return assign_output(session, SESSION_ID, studio, task, "op", reserved, now)


def _attrs(path: Path) -> dict:
    return json.loads((path / "zarr.json").read_text())["attributes"]


def _episode(group: Path, index: int, **attributes) -> Path:
    path = group / f"episode_{index}"
    path.mkdir(parents=True)
    (path / "zarr.json").write_text(json.dumps({"attributes": attributes, "zarr_format": 3, "node_type": "group"}))
    return path


# --- studio resolution -----------------------------------------------------------------------


def test_the_session_studio_overrides_everything_and_uses_the_known_spelling():
    config = RoutingConfig(
        session_studio="dining room", known_studios=("Dining_Room",), studio_by_task=(("t", "X"),)
    )
    assert resolve_studio("t", config) == "Dining_Room"


def test_a_task_is_looked_up_by_its_normalized_description_else_it_is_unmatched():
    config = RoutingConfig(studio_by_task=(("Stack  the Blocks", "Kitchen"),), known_studios=("Kitchen",))
    assert resolve_studio("stack the blocks", config) == "Kitchen"
    assert resolve_studio("something else", config) == "_unmatched"
    assert resolve_studio("", RoutingConfig(unmatched_studio="Other")) == "Other"


@pytest.mark.parametrize(
    ("config", "problem"),
    [
        (RoutingConfig(unmatched_studio=" "), "unmatched_studio"),
        (RoutingConfig(studio_by_task=(("", "Kitchen"),)), "empty field"),
        (RoutingConfig(studio_by_task=(("a task", " "),)), "empty field"),
        (RoutingConfig(studio_by_task=(("A Task", "X"), ("a  task", "Y"))), "normalize to the same text"),
    ],
)
def test_routing_configuration_problems_are_named(config, problem):
    assert any(problem in p for p in config.problems())


def test_a_valid_routing_configuration_has_no_problems():
    assert RoutingConfig(studio_by_task=(("a", "X"), ("b", "Y"))).problems() == []


# --- group assignment and numbering ----------------------------------------------------------


def test_the_first_assignment_creates_a_stamped_group_and_slot_zero(tmp_path):
    session = _session(tmp_path)
    group, index = _assign(session)
    assert (group, index) == ("Kitchen/2026-10-20-09-30-00", 0)
    attrs = _attrs(tmp_path / "out" / group)
    assert attrs[ROUTE_ATTR] == route_id(SESSION_ID, group)
    assert attrs["studio"] == "Kitchen" and attrs["task_description"] == "stack blocks"
    assert attrs["discarded_episode_indices"] == [] and attrs["operator_name"] == "op"


def test_the_same_studio_task_and_day_reuse_the_group_and_advance_the_slot(tmp_path):
    session = _session(tmp_path)
    group, _ = _assign(session)
    _episode(tmp_path / "out" / group, 0)
    again, index = _assign(session, now=NOW.replace(hour=17))
    assert again == group and index == 1
    # A slot promised to another take of this group is skipped too; slots promised in other groups are not.
    assert _assign(session, reserved=[(group, 1), ("Other/x", 9)])[1] == 2


def test_a_new_day_or_task_starts_a_new_group(tmp_path):
    session = _session(tmp_path)
    first, _ = _assign(session)
    assert _assign(session, now=NOW + datetime.timedelta(days=1))[0] != first
    assert _assign(session, task="another task")[0] != first


def test_a_timestamp_collision_gets_a_suffix(tmp_path):
    session = _session(tmp_path)
    (tmp_path / "out" / "Kitchen" / "2026-10-20-09-30-00").mkdir(parents=True)
    group, _ = _assign(session)
    assert group == "Kitchen/2026-10-20-09-30-00_1"


def test_an_unrelated_existing_group_directory_is_a_collision_never_merged_into(tmp_path):
    session = _session(tmp_path)
    group, _ = _assign(session)
    stamp = tmp_path / "out" / group / "zarr.json"
    stamp.write_text(json.dumps({"attributes": {ROUTE_ATTR: "someone|else"}}))
    with pytest.raises(PermissionError, match="not a group owned"):
        _assign(session)


def test_an_interrupted_assignment_resumes_with_the_same_group(tmp_path):
    session = _session(tmp_path)
    group, _ = _assign(session)
    for entry in (tmp_path / "out" / group).iterdir():
        entry.unlink()  # the directory vanished after the route was recorded
    (tmp_path / "out" / group).rmdir()
    assert _assign(session)[0] == group
    assert (tmp_path / "out" / group / "zarr.json").is_file()


# --- containment and ownership: nothing is written before validation, damage fails clearly -----


def _tree(path: Path) -> set[str]:
    return {str(p.relative_to(path)) for p in path.rglob("*")}


@pytest.mark.parametrize("studio", ["..", ".", "...", ".hidden"])
def test_an_unsafe_studio_is_refused_before_anything_is_written(tmp_path, studio):
    session = _session(tmp_path)
    before = _tree(tmp_path)
    with pytest.raises(ValueError, match="not a safe directory name"):
        _assign(session, studio=studio)
    assert _tree(tmp_path) == before


def test_routing_configuration_rejects_unsafe_studio_names():
    for config in (
        RoutingConfig(session_studio=".."),
        RoutingConfig(known_studios=("ok", "...")),
        RoutingConfig(studio_by_task=(("a task", ".."),)),
        RoutingConfig(unmatched_studio="."),
    ):
        assert any("safe" in p for p in config.problems()), config


@pytest.mark.parametrize(
    "routes",
    [
        "not json",
        "[1, 2]",
        '{"k": 3}',
        '{"Kitchen|stack blocks|2026-10-20": "/abs/Kitchen/x"}',
        '{"Kitchen|stack blocks|2026-10-20": "../outside/x"}',
        '{"Kitchen|stack blocks|2026-10-20": "Dining/2026"}',
        '{"Kitchen|stack blocks|2026-10-20": "Kitchen/a/b"}',
        '{"Kitchen|stack blocks|2026-10-20": "Kitchen/.."}',
    ],
)
def test_damaged_recorded_routes_fail_clearly_before_anything_is_written(tmp_path, routes):
    session = _session(tmp_path)
    (session / ".recording" / "routes.json").write_text(routes)
    before = _tree(tmp_path)
    with pytest.raises(ValueError):
        _assign(session)
    assert _tree(tmp_path) == before


def test_a_symlinked_studio_directory_is_refused_before_anything_is_written(tmp_path):
    session = _session(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (tmp_path / "out" / "Kitchen").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        _assign(session)
    assert list(outside.iterdir()) == []


def test_a_recorded_group_that_is_a_symlink_is_refused(tmp_path):
    session = _session(tmp_path)
    group, _ = _assign(session)
    outside = tmp_path / "outside"
    outside.mkdir()
    real = tmp_path / "out" / group
    real.rename(tmp_path / "moved")
    real.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        _assign(session)
    assert list(outside.iterdir()) == []


def test_unreadable_or_unstamped_group_metadata_is_an_error_never_adopted(tmp_path):
    session = _session(tmp_path)
    group, _ = _assign(session)
    stamp = tmp_path / "out" / group / "zarr.json"
    stamp.write_text("not json")
    with pytest.raises(ValueError, match="unreadable or not valid JSON"):
        _assign(session)
    stamp.unlink()  # a recorded group directory without its stamp is not ours to adopt
    with pytest.raises(ValueError, match="unreadable"):
        _assign(session)
    assert not stamp.exists()


def test_two_sessions_choosing_the_same_group_name_each_get_their_own(tmp_path):
    first, second = _session(tmp_path), tmp_path / "out" / "session_2"
    (second / ".recording").mkdir(parents=True)
    a, _ = assign_output(first, SESSION_ID, "Kitchen", "stack blocks", "op", [], NOW)
    b, _ = assign_output(second, "session_2", "Kitchen", "stack blocks", "op", [], NOW)
    assert a == "Kitchen/2026-10-20-09-30-00" and b == "Kitchen/2026-10-20-09-30-00_1"
    assert _attrs(tmp_path / "out" / a)[ROUTE_ATTR] == route_id(SESSION_ID, a)  # the first claim is untouched
    assert _attrs(tmp_path / "out" / b)[ROUTE_ATTR] == route_id("session_2", b)


def test_new_directory_entries_are_flushed_before_a_manifest_can_point_at_them(tmp_path, monkeypatch):
    """The studio directory's parent (the output root) and the group's parent (the studio) are fsynced."""
    import egocap.recording.route as route

    flushed = []
    real = route.fsync_dir
    monkeypatch.setattr(route, "fsync_dir", lambda p: (flushed.append(Path(p)), real(p))[1])
    session = _session(tmp_path)
    _assign(session)
    out = tmp_path / "out"
    assert out in flushed and (out / "Kitchen") in flushed


def test_two_sessions_creating_the_same_new_studio_do_not_race(tmp_path):
    """The studio directory may already exist when a second session creates it: that is not an error."""
    first, second = _session(tmp_path), tmp_path / "out" / "session_2"
    (second / ".recording").mkdir(parents=True)
    (tmp_path / "out" / "Kitchen").mkdir()  # the other session won the creation
    assert assign_output(second, "session_2", "Kitchen", "stack blocks", "op", [], NOW)[0].startswith("Kitchen/")
    assert _assign(first)[0].startswith("Kitchen/")


def test_a_recorded_route_must_still_describe_the_group_it_names(tmp_path):
    """A routes.json entry for another task that points at this session's existing group is refused."""
    session = _session(tmp_path)
    group, _ = _assign(session, task="task A")
    routes_file = session / ".recording" / "routes.json"
    routes = json.loads(routes_file.read_text())
    routes["Kitchen|task B|2026-10-20"] = group  # task B now aliases task A's group
    routes_file.write_text(json.dumps(routes))
    before = _tree(tmp_path)
    with pytest.raises(PermissionError, match="holds 'task A'"):
        _assign(session, task="task B")
    assert _tree(tmp_path) == before
    # the same task under another studio name is refused the same way
    routes["Dining|task A|2026-10-20"] = group
    routes_file.write_text(json.dumps(routes))
    with pytest.raises(ValueError):
        _assign(session, studio="Dining", task="task A")


# --- group totals ----------------------------------------------------------------------------


def test_group_totals_are_derived_from_the_finalized_episodes_and_repeatable(tmp_path):
    session = _session(tmp_path)
    group_name, _ = _assign(session)
    group = tmp_path / "out" / group_name
    _episode(group, 0, episode_duration=1800.0, episode_length=100)
    _episode(group, 1, episode_duration=1800.0, episode_length=90, bad_episode=True, capture_qc_issues=["frozen"])
    _episode(group, 2, episode_duration=0.0, bad_episode=True)
    (group / "episode_3.pending").mkdir()  # a pending take is never counted
    (group / "notes.txt").write_text("not an episode")
    first = refresh_group_totals(group)
    assert first["num_episodes"] == 3 and first["total_data_time_hours"] == 1.0
    assert first["episode_lengths"] == {"0": 100, "1": 90}
    assert first["discarded_episode_indices"] == [1, 2]
    assert first["discarded_reasons"] == {"1": "frozen", "2": "episode flagged bad"}
    assert refresh_group_totals(group) == first  # deriving again changes nothing
    assert _attrs(group)[ROUTE_ATTR] == route_id(SESSION_ID, group_name)  # the stamp survives the update


def test_group_totals_fail_clearly_on_unreadable_episode_metadata(tmp_path):
    session = _session(tmp_path)
    group = tmp_path / "out" / _assign(session)[0]
    (group / "episode_0").mkdir()  # an episode directory with no readable metadata is not silently counted
    with pytest.raises(ValueError, match="unreadable"):
        refresh_group_totals(group)


# --- routed-path guards ----------------------------------------------------------------------


def _manifest(session, *, index=0, k=0, studio="Kitchen", group="Kitchen/2026-10-20-09-30-00"):
    return EpisodeManifest(
        session_id=SESSION_ID,
        episode_id=f"{SESSION_ID}/episode_{index}",
        episode_index=index,
        config_digest="d",
        task="stack blocks",
        operator="op",
        sources=[],
        pending_path=f"episode_{index}.pending",
        final_path=f"episode_{index}",
        studio=studio,
        assignment={"group": group, "slot": k},
    )


def _stamped_group(session, manifest):
    group = session.parent / manifest.assignment["group"]
    group.mkdir(parents=True)
    (group / "zarr.json").write_text(
        json.dumps({"attributes": {ROUTE_ATTR: route_id(SESSION_ID, manifest.assignment["group"])}})
    )
    return group


def _pending(session, manifest):
    pending = session / manifest.pending_path
    pending.mkdir()
    (pending / "zarr.json").write_text(json.dumps({"attributes": {"recording_episode_id": manifest.episode_id}}))
    return pending


def test_a_routed_episode_is_renamed_into_its_group_and_found_there_after_a_crash(tmp_path):
    session = _session(tmp_path)
    manifest = _manifest(session)
    group = _stamped_group(session, manifest)
    _pending(session, manifest)
    assert locate_episode(session, manifest) == "pending"
    final = finalize_episode_dir(session, manifest)
    assert final == group / "episode_0" and not (session / manifest.pending_path).exists()
    assert finalize_episode_dir(session, manifest) == final  # a repeat after the rename is a no-op
    assert locate_episode(session, manifest) == "final"  # the rename happened, the checkpoint did not
    assert final_dir(session, manifest) == final


def test_a_routed_final_episode_is_never_deleted_by_removal(tmp_path):
    session = _session(tmp_path)
    manifest = _manifest(session)
    _stamped_group(session, manifest)
    _pending(session, manifest)
    final = finalize_episode_dir(session, manifest)
    manifest.storage_phase = "CHECKED"
    with pytest.raises(PermissionError, match="finalized output"):
        remove_owned_episode(session, manifest)
    assert final.is_dir()


def test_a_routed_target_that_is_not_ours_is_a_collision_not_an_overwrite(tmp_path):
    session = _session(tmp_path)
    manifest = _manifest(session)
    group = _stamped_group(session, manifest)
    _pending(session, manifest)
    (group / "episode_0").mkdir()
    with pytest.raises(FileExistsError, match="both"):
        finalize_episode_dir(session, manifest)


@pytest.mark.parametrize(
    "changes",
    [
        {"assignment": {"group": "../outside", "slot": 0}},
        {"assignment": {"group": "Kitchen", "slot": 0}},  # not <studio>/<timestamp>
        {"assignment": {"group": "/abs/Kitchen/x", "slot": 0}},
        {"assignment": {"group": "Dining/2026-10-20-09-30-00", "slot": 0}},  # not under the frozen studio
        {"assignment": {"group": "Kitchen/..", "slot": 0}},  # traversal component
        {"assignment": {"group": "Kitchen/2026-10-20-09-30-00", "slot": -1}},
        {"assignment": {"group": "Kitchen/2026-10-20-09-30-00", "slot": "0"}},
        {"studio": None},
    ],
)
def test_noncanonical_routed_paths_are_refused(tmp_path, changes):
    session = _session(tmp_path)
    manifest = _manifest(session)
    for name, value in changes.items():
        setattr(manifest, name, value)
    with pytest.raises(ValueError):
        final_dir(session, manifest)


def test_a_group_not_stamped_for_this_session_is_refused(tmp_path):
    session = _session(tmp_path)
    manifest = _manifest(session)
    group = _stamped_group(session, manifest)
    (group / "zarr.json").write_text(json.dumps({"attributes": {ROUTE_ATTR: "other|group"}}))
    with pytest.raises(PermissionError, match="not a group owned"):
        final_dir(session, manifest)
    (group / "zarr.json").write_text("not json")
    with pytest.raises(PermissionError):
        final_dir(session, manifest)


def test_symlinked_studio_or_group_directories_are_refused(tmp_path):
    session = _session(tmp_path)
    manifest = _manifest(session)
    outside = tmp_path / "outside"
    outside.mkdir()
    (tmp_path / "out" / "Kitchen").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        final_dir(session, manifest)
    (tmp_path / "out" / "Kitchen").unlink()
    (tmp_path / "out" / "Kitchen").mkdir()
    (tmp_path / "out" / manifest.assignment["group"]).symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        final_dir(session, manifest)
    (tmp_path / "out" / manifest.assignment["group"]).unlink()
    (tmp_path / "out" / manifest.assignment["group"]).mkdir()
    (tmp_path / "out" / manifest.assignment["group"] / "episode_0").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        final_dir(session, manifest)


def test_an_unrouted_manifest_still_uses_the_session_layout(tmp_path):
    session = _session(tmp_path)
    manifest = _manifest(session)
    manifest.studio = manifest.assignment = None
    assert final_dir(session, manifest) == session / "episode_0"
