# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from egocap.studio.mid_train_data import (
    grasp_thumbnail_path,
    load_mid_train_grasps,
    load_mid_train_library,
    lookup_mid_train_description,
    lookup_mid_train_grasps,
    lookup_mid_train_task,
    parse_grasp_cell,
    to_lower_snake,
)

FIXTURE_CSV = Path(__file__).resolve().parents[1] / "fixtures" / "mid_train" / "grasp_tasks.csv"


def test_to_lower_snake():
    assert to_lower_snake("Unscrew Cap") == "unscrew_cap"
    assert to_lower_snake("  already_snake ") == "already_snake"
    assert to_lower_snake("") == ""
    assert to_lower_snake(None) == ""


def test_load_mid_train_library_from_fixture():
    library = load_mid_train_library(FIXTURE_CSV)
    assert "unscrew" in library
    assert ("unscrew_cap", "Unscrew the bottle cap.", "bottle") in library["unscrew"]
    # Compound "shake + pour" is listed under both component primitives.
    assert any(t[0] == "shake_then_pour" for t in library["shake"])
    assert any(t[0] == "shake_then_pour" for t in library["pour"])


def test_load_mid_train_library_missing_file_returns_empty(tmp_path):
    assert load_mid_train_library(tmp_path / "missing.csv") == {}


def test_load_mid_train_library_too_short_returns_empty(tmp_path):
    path = tmp_path / "short.csv"
    path.write_text("Task Name,Task Description,Primitive(s)\n", encoding="utf-8")
    assert load_mid_train_library(path) == {}


def test_load_mid_train_library_missing_header_returns_empty(tmp_path):
    path = tmp_path / "bad_header.csv"
    path.write_text("Foo,Bar\nx,y\n", encoding="utf-8")
    assert load_mid_train_library(path) == {}


def test_load_mid_train_library_unreadable_file_returns_empty(tmp_path, monkeypatch):
    path = tmp_path / "unreadable.csv"
    path.write_text("Task Name,Task Description,Primitive(s)\nx,y,z\n", encoding="utf-8")

    real_open = open

    def boom(p, *a, **k):
        if str(p) == str(path):
            raise OSError("permission denied")
        return real_open(p, *a, **k)

    monkeypatch.setattr("builtins.open", boom)
    assert load_mid_train_library(path) == {}


def test_load_mid_train_library_row_too_short_skipped(tmp_path):
    path = tmp_path / "short_row.csv"
    path.write_text("Task Name,Task Description,Primitive(s)\nonly_one_col\n", encoding="utf-8")
    assert load_mid_train_library(path) == {}


def test_load_mid_train_library_dedups_task_within_primitive(tmp_path):
    path = tmp_path / "dup.csv"
    path.write_text(
        "Task Name,Task Description,Primitive(s)\n"
        "unscrew_cap,First desc,unscrew\n"
        "unscrew_cap,Second desc,unscrew\n",
        encoding="utf-8",
    )
    library = load_mid_train_library(path)
    assert len(library["unscrew"]) == 1
    assert library["unscrew"][0][1] == "First desc"


def test_load_mid_train_library_skips_rows_missing_task_or_primitive(tmp_path):
    path = tmp_path / "empty_fields.csv"
    path.write_text(
        "Task Name,Task Description,Primitive(s)\n,desc,unscrew\ntask_name,desc,\n",
        encoding="utf-8",
    )
    assert load_mid_train_library(path) == {}


def test_lookup_mid_train_task_found_and_missing():
    library = load_mid_train_library(FIXTURE_CSV)
    found = lookup_mid_train_task("unscrew", "unscrew_cap", library)
    assert found == ("Unscrew the bottle cap.", "bottle")
    assert lookup_mid_train_task("unscrew", "no_such_task", library) is None
    assert lookup_mid_train_task("", "unscrew_cap", library) is None
    assert lookup_mid_train_task("unscrew", "", library) is None


def test_lookup_mid_train_task_default_library_param():
    assert lookup_mid_train_task("", "", None) is None


def test_lookup_mid_train_description_found_across_primitives():
    library = load_mid_train_library(FIXTURE_CSV)
    assert lookup_mid_train_description("unscrew_cap", library) == "Unscrew the bottle cap."
    assert lookup_mid_train_description("no_such_task", library) is None
    assert lookup_mid_train_description("", library) is None


def test_lookup_mid_train_description_default_library_param():
    assert lookup_mid_train_description("") is None


def test_parse_grasp_cell():
    assert parse_grasp_cell("Power Disk [10] (cap)") == ("Power Disk", 10, "cap")
    assert parse_grasp_cell("Quadpod [27] (peach)") == ("Quadpod", 27, "peach")
    assert parse_grasp_cell("No bracket here") is None
    assert parse_grasp_cell("") is None


def test_parse_grasp_cell_no_role():
    name, number, role = parse_grasp_cell("Bare Grasp [5]")
    assert (name, number, role) == ("Bare Grasp", 5, "")


def test_grasp_thumbnail_path_missing_returns_none(tmp_path, monkeypatch):
    import egocap.studio.mid_train_data as mod

    monkeypatch.setattr(mod, "GRASP_THUMBNAIL_DIR", tmp_path)
    assert grasp_thumbnail_path(10) is None


def test_grasp_thumbnail_path_found(tmp_path, monkeypatch):
    import egocap.studio.mid_train_data as mod

    monkeypatch.setattr(mod, "GRASP_THUMBNAIL_DIR", tmp_path)
    thumb = tmp_path / "10_power_disk.png"
    thumb.write_bytes(b"")
    assert grasp_thumbnail_path(10) == thumb


def test_load_mid_train_grasps_from_fixture():
    grasps = load_mid_train_grasps(FIXTURE_CSV)
    assert grasps["unscrew_cap"] == ("Power Disk [10] (cap)", "Quadpod [27] (peach)")
    assert grasps["shake_then_pour"] == ("Power Disk [10] (cap)", "")


def test_load_mid_train_grasps_missing_file_returns_empty(tmp_path):
    assert load_mid_train_grasps(tmp_path / "missing.csv") == {}


def test_load_mid_train_grasps_too_short_returns_empty(tmp_path):
    path = tmp_path / "short.csv"
    path.write_text("Task Name,Left Hand Grasp,Right Hand Grasp\n", encoding="utf-8")
    assert load_mid_train_grasps(path) == {}


def test_load_mid_train_grasps_missing_header_returns_empty(tmp_path):
    path = tmp_path / "bad_header.csv"
    path.write_text("Foo,Bar\nx,y\n", encoding="utf-8")
    assert load_mid_train_grasps(path) == {}


def test_load_mid_train_grasps_unreadable_file_returns_empty(tmp_path, monkeypatch):
    path = tmp_path / "unreadable.csv"
    path.write_text("Task Name,Left Hand Grasp,Right Hand Grasp\nx,y,z\n", encoding="utf-8")

    real_open = open

    def boom(p, *a, **k):
        if str(p) == str(path):
            raise OSError("permission denied")
        return real_open(p, *a, **k)

    monkeypatch.setattr("builtins.open", boom)
    assert load_mid_train_grasps(path) == {}


def test_load_mid_train_grasps_row_too_short_skipped(tmp_path):
    path = tmp_path / "short_row.csv"
    path.write_text("Task Name,Left Hand Grasp,Right Hand Grasp\nonly_one_col\n", encoding="utf-8")
    assert load_mid_train_grasps(path) == {}


def test_load_mid_train_grasps_skips_empty_task_name(tmp_path):
    path = tmp_path / "empty_task.csv"
    path.write_text("Task Name,Left Hand Grasp,Right Hand Grasp\n,left,right\n", encoding="utf-8")
    assert load_mid_train_grasps(path) == {}


def test_load_mid_train_grasps_first_occurrence_wins(tmp_path):
    path = tmp_path / "dup.csv"
    path.write_text(
        "Task Name,Left Hand Grasp,Right Hand Grasp\n"
        "task_a,first_left,first_right\n"
        "task_a,second_left,second_right\n",
        encoding="utf-8",
    )
    grasps = load_mid_train_grasps(path)
    assert grasps["task_a"] == ("first_left", "first_right")


def test_lookup_mid_train_grasps_found_and_missing():
    grasps = load_mid_train_grasps(FIXTURE_CSV)
    assert lookup_mid_train_grasps("unscrew_cap", grasps) == (
        "Power Disk [10] (cap)",
        "Quadpod [27] (peach)",
    )
    assert lookup_mid_train_grasps("no_such_task", grasps) is None
    assert lookup_mid_train_grasps("", grasps) is None


def test_lookup_mid_train_grasps_default_param():
    assert lookup_mid_train_grasps("") is None
