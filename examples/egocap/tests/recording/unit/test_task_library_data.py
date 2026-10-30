# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from egocap.studio import task_library_data
from egocap.studio.task_library_data import load_task_library, lookup_task_description

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "task_library"


def test_load_task_library_from_fixtures():
    library = load_task_library_from(FIXTURE_DIR)
    assert library["Kitchen"] == [
        ("pour_cereal", "Pour cereal into the bowl."),
        ("pour_milk", "Pour milk into the bowl."),
    ]
    assert library["Dining_Room"] == [
        ("place_napkin", "Place the napkin to the left of the plate."),
    ]
    assert "Should" not in library


def load_task_library_from(directory, monkeypatch=None):
    # load_task_library() reads the module-level TASK_CSV_DIR constant, so
    # tests monkeypatch it rather than passing a parameter (none exists).
    import pytest

    mp = pytest.MonkeyPatch()
    mp.setattr(task_library_data, "TASK_CSV_DIR", directory)
    try:
        return load_task_library()
    finally:
        mp.undo()


def test_load_task_library_missing_dir_returns_empty(tmp_path):
    missing = tmp_path / "does-not-exist"
    assert load_task_library_from(missing) == {}


def test_load_task_library_empty_dir_returns_empty(tmp_path):
    assert load_task_library_from(tmp_path) == {}


def test_load_task_library_malformed_csv_returns_empty(tmp_path, monkeypatch):
    # Invalid UTF-8 bytes raise UnicodeDecodeError inside the per-file parse
    # loop; load_task_library's own try/except catches it and returns {}
    # rather than propagating, since one bad CSV shouldn't crash the lookup.
    (tmp_path / "Bad.csv").write_bytes(b"title\n,,,\nStudio,Task,Description\n\xff\xfe,x,y\n")
    monkeypatch.setattr(task_library_data, "TASK_CSV_DIR", tmp_path)
    assert load_task_library() == {}


def test_load_task_library_csv_too_short_is_skipped(tmp_path, monkeypatch):
    (tmp_path / "Short.csv").write_text("title\n,,,\nStudio,Task,Description\n", encoding="utf-8")
    monkeypatch.setattr(task_library_data, "TASK_CSV_DIR", tmp_path)
    assert load_task_library() == {}


def test_load_task_library_missing_header_column_skipped(tmp_path, monkeypatch):
    (tmp_path / "NoHeader.csv").write_text(
        "title\n,,,\nStudio,Scenario,Description\nKitchen,x,desc\n", encoding="utf-8"
    )
    monkeypatch.setattr(task_library_data, "TASK_CSV_DIR", tmp_path)
    assert load_task_library() == {}


def test_load_task_library_row_missing_fields_skipped(tmp_path, monkeypatch):
    (tmp_path / "Partial.csv").write_text(
        "title\n,,,\nStudio,Scenario,Task,Description\nKitchen\n", encoding="utf-8"
    )
    monkeypatch.setattr(task_library_data, "TASK_CSV_DIR", tmp_path)
    assert load_task_library() == {}


def test_load_task_library_empty_studio_or_task_skipped(tmp_path, monkeypatch):
    (tmp_path / "Empty.csv").write_text(
        "title\n,,,\nStudio,Scenario,Task,Description\n,Scene,task,desc\nKitchen,Scene,,desc\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(task_library_data, "TASK_CSV_DIR", tmp_path)
    assert load_task_library() == {}


def test_lookup_task_description_found():
    library = {"Kitchen": [("pour_cereal", "Pour cereal into the bowl.")]}
    assert lookup_task_description(library, "Kitchen", "pour_cereal") == "Pour cereal into the bowl."


def test_lookup_task_description_canonicalizes_studio():
    library = {"Kitchen": [("pour_cereal", "Pour cereal into the bowl.")]}
    assert lookup_task_description(library, "kitchen", "pour_cereal") == "Pour cereal into the bowl."


def test_lookup_task_description_missing_studio_or_task_returns_none():
    library = {"Kitchen": [("pour_cereal", "desc")]}
    assert lookup_task_description(library, "Kitchen", "unknown_task") is None
    assert lookup_task_description(library, "Garage", "pour_cereal") is None


def test_lookup_task_description_empty_args_returns_none():
    assert lookup_task_description({}, "", "pour_cereal") is None
    assert lookup_task_description({}, "Kitchen", "") is None
