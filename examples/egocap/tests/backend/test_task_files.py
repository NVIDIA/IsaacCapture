"""The task-file reader on small fixtures and on the example catalog shipped in data/taskfiles/."""

from __future__ import annotations

from pathlib import Path

import pytest

from egocap_web import task_files as tf

REAL_ROOT = Path(__file__).resolve().parents[2] / "data" / "taskfiles"
STUDIO_HEADER = "Studio,Scenario,Task,Description\n"


def _write(root: Path, name: str, text: str, *, encoding: str = "utf-8") -> None:
    (root / name).write_bytes(text.encode(encoding))


def _studio(rows: str) -> str:
    return f"Title,,,\n,,,\n{STUDIO_HEADER}{rows}"


def test_the_shipped_catalog_parses_with_its_row_count():
    counts = {name: len(tf.read_task_file(REAL_ROOT, name).tasks) for name in tf.list_task_files(REAL_ROOT)}
    assert counts == {"Kitchen.csv": 11}
    first = tf.read_task_file(REAL_ROOT, "Kitchen.csv").tasks[0]
    assert first.studio == "Kitchen" and first.name and first.instruction and first.scenario


def test_listing_is_sorted_and_only_top_level_regular_csv_files(tmp_path):
    for name in ("b.csv", "a.csv", "é.csv", "notes.txt", ".csv", "a..csv", "back\\slash.csv"):
        _write(tmp_path, name, "x")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "c.csv").write_text("x")
    (tmp_path / "dir.csv").mkdir()
    (tmp_path / "link.csv").symlink_to(tmp_path / "a.csv")
    assert tf.list_task_files(tmp_path) == sorted(["a.csv", "b.csv", "é.csv"])


def test_an_empty_directory_lists_nothing_and_a_missing_root_is_unavailable(tmp_path):
    assert tf.list_task_files(tmp_path) == []
    with pytest.raises(tf.TaskFilesUnavailable):
        tf.list_task_files(tmp_path / "missing")
    with pytest.raises(tf.TaskFilesUnavailable):
        tf.read_task_file(tmp_path / "missing", "a.csv")


def test_studio_records_keep_order_duplicates_quotes_newlines_and_ignore_extra_columns(tmp_path):
    text = STUDIO_HEADER.replace("Description", "Description,Operator")
    text = (
        "Title,,,,\n,,,,\n" + text + 'Kitchen,Cleanup,Wash dishes,"Wash, rinse,\nand dry.",Sam\n'
        "\n"
        "Kitchen,Cleanup,Wash dishes,Again,Lee\n"
    )
    _write(tmp_path, "k.csv", text)
    contents = tf.read_task_file(tmp_path, "k.csv")
    assert contents.format == "studio_csv"
    assert [t.to_json() for t in contents.tasks] == [
        {
            "name": "Wash dishes",
            "instruction": "Wash, rinse,\nand dry.",
            "studio": "Kitchen",
            "scenario": "Cleanup",
        },
        {"name": "Wash dishes", "instruction": "Again", "studio": "Kitchen", "scenario": "Cleanup"},
    ]


def test_a_byte_order_mark_and_case_and_spaces_in_headers_are_accepted(tmp_path):
    _write(tmp_path, "k.csv", "﻿" + "T,,,\n,,,\n STUDIO , scenario,TASK,Description \nK,S,Do it,Now\n")
    assert tf.read_task_file(tmp_path, "k.csv").tasks[0].name == "Do it"


def test_mid_train_headers_on_the_first_record_with_native_name_and_primitive_rules(tmp_path):
    _write(
        tmp_path,
        "m.csv",
        ",Task Name,Task Description,Primitive(s),Left Hand Grasp,Right Hand Grasp,Props,Operator Name\n"
        '1,Unscrew Water-Bottle,Unscrew a cap,unscrew + shake,Large Diameter,Power Disk,"a, b",Sam\n',
    )
    contents = tf.read_task_file(tmp_path, "m.csv")
    assert contents.format == "mid_train_csv"
    assert contents.tasks[0].to_json() == {
        "name": "unscrew_water_bottle",
        "instruction": "Unscrew a cap",
        "props": "a, b",
        "left_grasp": "Large Diameter",
        "right_grasp": "Power Disk",
        "primitives": ["unscrew", "shake"],
    }
    assert "Sam" not in str(contents.to_json())


def test_mid_train_optional_columns_may_be_absent(tmp_path):
    _write(tmp_path, "m.csv", "Task Name,Task Description\nfold_towel,Fold a towel\n")
    assert tf.read_task_file(tmp_path, "m.csv").tasks[0].to_json() == {
        "name": "fold_towel",
        "instruction": "Fold a towel",
    }


@pytest.mark.parametrize(
    "text",
    [
        "",
        'Task Name,Task Description\nx,"unterminated\n',  # unterminated quote
        "just,some,cells\n1,2,3\n",
        "T,,,\n,,,\nStudio,Scenario,Task\nK,S,Do\n",  # missing a required header
        "T,,,\n,,,\nStudio,Task,Task,Description,Scenario\nK,Do,Do,Now,S\n",  # duplicate required header
        _studio("Kitchen,S,Do it\n"),  # incomplete record
        _studio("Kitchen,S,,Now\n"),  # empty task name
        _studio("Kitchen,S,Do it,\n"),  # empty instruction
        "Task Name,Task Description\n  ,Do it\n",  # empty Mid Train task name
        "Task Name,Task Description\n!!!,Do it\n",  # name with nothing left after normalising
    ],
)
def test_a_malformed_file_is_refused_rather_than_partially_returned(tmp_path, text):
    _write(tmp_path, "bad.csv", text)
    with pytest.raises(tf.MalformedTaskFile):
        tf.read_task_file(tmp_path, "bad.csv")


def test_non_utf8_bytes_are_malformed(tmp_path):
    _write(tmp_path, "bad.csv", "Task Name,Task Description\nx,é\n", encoding="latin-1")
    with pytest.raises(tf.MalformedTaskFile):
        tf.read_task_file(tmp_path, "bad.csv")


def test_the_size_bound_is_checked_on_the_bytes_read(tmp_path, monkeypatch):
    monkeypatch.setattr(tf, "MAX_FILE_BYTES", 64)
    _write(tmp_path, "big.csv", "Task Name,Task Description\n" + "x,y\n" * 100)
    with pytest.raises(tf.TaskFileTooLarge):
        tf.read_task_file(tmp_path, "big.csv")


@pytest.mark.parametrize(
    "name", ["", ".csv", "a.txt", "../a.csv", "sub/a.csv", "a\\b.csv", "a\0.csv", "..csv", "a", "Kitchen..csv"]
)
def test_names_that_are_not_plain_csv_filenames_are_invalid(tmp_path, name):
    with pytest.raises(tf.InvalidTaskFileName):
        tf.read_task_file(tmp_path, name)


def test_missing_symlinked_and_directory_names_are_not_found(tmp_path):
    real = tmp_path / "real.csv"
    real.write_text("Task Name,Task Description\nx,y\n")
    (tmp_path / "link.csv").symlink_to(real)
    (tmp_path / "dir.csv").mkdir()
    for name in ("absent.csv", "link.csv", "dir.csv"):
        with pytest.raises(tf.TaskFileNotFound):
            tf.read_task_file(tmp_path, name)


def test_reading_changes_no_files(tmp_path):
    _write(tmp_path, "k.csv", _studio("K,S,Do it,Now\n"))
    before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in tmp_path.iterdir()}
    tf.list_task_files(tmp_path)
    tf.read_task_file(tmp_path, "k.csv")
    assert before == {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in tmp_path.iterdir()}


def test_a_failure_while_resolving_the_path_is_storage_unavailable_not_a_crash(tmp_path, monkeypatch):
    _write(tmp_path, "k.csv", _studio("K,S,Do it,Now\n"))

    def failing(self, strict=False):
        raise OSError("filesystem changed underneath")

    monkeypatch.setattr(Path, "resolve", failing)
    with pytest.raises(tf.TaskFilesUnavailable):
        tf.read_task_file(tmp_path, "k.csv")


def test_the_librarys_summary_file_hidden_files_and_editor_leftovers_are_not_catalogs(tmp_path):
    for name in ("Kitchen.csv", "Summary.csv", "._Kitchen.csv", "~$Kitchen.csv", ".hidden.csv"):
        _write(tmp_path, name, "Task Name,Task Description\nx,y\n")
    assert tf.list_task_files(tmp_path) == ["Kitchen.csv"]
    for name in ("Summary.csv", "._Kitchen.csv", "~$Kitchen.csv", ".hidden.csv"):
        with pytest.raises(tf.TaskFileNotFound):  # well formed but not a catalog: absent, not a bad request
            tf.read_task_file(tmp_path, name)


def test_a_name_that_is_not_valid_utf8_is_not_listed(tmp_path):
    import os

    with open(os.path.join(os.fsencode(tmp_path), b"bad\xff.csv"), "w") as handle:
        handle.write("x")
    _write(tmp_path, "ok.csv", "Task Name,Task Description\nx,y\n")
    assert tf.list_task_files(tmp_path) == ["ok.csv"]


def test_a_catalog_deleted_after_listing_is_not_found_but_an_unreadable_one_is_unavailable(tmp_path, monkeypatch):
    monkeypatch.setattr(tf, "list_task_files", lambda root: ["gone.csv"])
    with pytest.raises(tf.TaskFileNotFound):
        tf.read_task_file(tmp_path, "gone.csv")
    _write(tmp_path, "locked.csv", "Task Name,Task Description\nx,y\n")
    monkeypatch.setattr(tf, "list_task_files", lambda root: ["locked.csv"])

    def denied(self, *args, **kwargs):
        raise PermissionError("denied")

    monkeypatch.setattr(Path, "open", denied)
    with pytest.raises(tf.TaskFilesUnavailable):
        tf.read_task_file(tmp_path, "locked.csv")


@pytest.mark.parametrize(
    "name", ["Unscrew Water-Bottle!", "  Fold  Towel ", "pick_Up-the.cup", "Ünï code", "123 go", "___x___"]
)
def test_task_names_normalise_exactly_as_the_recorders_mid_train_lookup_does(name):
    """The API's names must match what the recorder looks tasks up by, so the two functions are pinned together."""
    from egocap.studio.mid_train_data import to_lower_snake

    assert tf._lower_snake(name) == to_lower_snake(name)
