from pathlib import Path

from egocap.studio.studio_lookup import (
    _normalize_desc,
    build_description_to_studio,
    default_task_library_dir,
    known_studios_from_task_library_dir,
    lookup_studio,
)

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "task_library"


def test_build_description_to_studio_from_fixtures():
    mapping = build_description_to_studio(FIXTURE_DIR)
    assert mapping["pour cereal into the bowl"] == "Kitchen"
    assert mapping["place the napkin to the left of the plate"] == "Dining_Room"
    assert "should" not in mapping


def test_known_studios_from_fixtures():
    known = known_studios_from_task_library_dir(FIXTURE_DIR)
    assert known == frozenset({"Kitchen", "Dining_Room"})


def test_missing_dir_returns_empty():
    missing = FIXTURE_DIR / "does-not-exist"
    assert build_description_to_studio(missing) == {}
    assert known_studios_from_task_library_dir(missing) == frozenset()


def test_csv_too_short_is_skipped(tmp_path):
    (tmp_path / "Short.csv").write_text("title\n,,,\nStudio,Description\n", encoding="utf-8")
    assert build_description_to_studio(tmp_path) == {}


def test_csv_missing_header_is_skipped(tmp_path):
    (tmp_path / "NoHeader.csv").write_text("title\n,,,\nFoo,Bar\nx,y\n", encoding="utf-8")
    assert build_description_to_studio(tmp_path) == {}


def test_csv_row_missing_fields_is_skipped(tmp_path):
    (tmp_path / "Partial.csv").write_text(
        "title\n,,,\nStudio,Scenario,Task,Description\nKitchen\n", encoding="utf-8"
    )
    assert build_description_to_studio(tmp_path) == {}


def test_row_with_empty_studio_or_description_skipped(tmp_path):
    (tmp_path / "Empty.csv").write_text(
        "title\n,,,\nStudio,Scenario,Task,Description\n,Scene,Task,Desc\nStudio,Scene,Task,\n",
        encoding="utf-8",
    )
    assert build_description_to_studio(tmp_path) == {}


def test_default_task_library_dir_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("UMI_TASK_LIBRARY_DIR", str(tmp_path))
    assert default_task_library_dir() == tmp_path.resolve()


def test_default_task_library_dir_fallback(monkeypatch):
    monkeypatch.delenv("UMI_TASK_LIBRARY_DIR", raising=False)
    result = default_task_library_dir()
    assert result.name == "task_library"


def test_lookup_studio_empty_description_returns_none():
    assert lookup_studio("", {"x": "Kitchen"}) is None


def test_lookup_studio_normalizes_before_matching():
    mapping = {"pour cereal into the bowl": "Kitchen"}
    assert lookup_studio("  Pour cereal into the bowl.  ", mapping) == "Kitchen"


def test_lookup_studio_no_match_returns_none():
    assert lookup_studio("unrelated description", {"x": "Kitchen"}) is None


def test_unreadable_csv_is_skipped(tmp_path, monkeypatch):
    path = tmp_path / "Unreadable.csv"
    path.write_text("title\n,,,\nStudio,Description\nKitchen,x\n", encoding="utf-8")

    real_open = open

    def boom(p, *a, **k):
        if str(p) == str(path):
            raise OSError("permission denied")
        return real_open(p, *a, **k)

    monkeypatch.setattr("builtins.open", boom)
    assert build_description_to_studio(tmp_path) == {}


def test_known_studios_too_short_missing_header_and_short_row_skipped(tmp_path):
    (tmp_path / "Short.csv").write_text("title\n,,,\nStudio\n", encoding="utf-8")
    (tmp_path / "NoHeader.csv").write_text("title\n,,,\nFoo,Bar\nx,y\n", encoding="utf-8")
    (tmp_path / "Partial.csv").write_text("title\n,,,\nStudio,Scenario\n\n", encoding="utf-8")
    assert known_studios_from_task_library_dir(tmp_path) == frozenset()


def test_known_studios_empty_studio_skipped(tmp_path):
    (tmp_path / "Empty.csv").write_text("title\n,,,\nStudio,Scenario\n,Scene\n", encoding="utf-8")
    assert known_studios_from_task_library_dir(tmp_path) == frozenset()


def test_normalize_desc_empty_returns_empty():
    assert _normalize_desc("") == ""
    assert _normalize_desc(None) == ""


def test_build_description_to_studio_default_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("UMI_TASK_LIBRARY_DIR", str(FIXTURE_DIR))
    assert build_description_to_studio(None)["pour cereal into the bowl"] == "Kitchen"


def test_known_studios_default_dir(monkeypatch):
    monkeypatch.setenv("UMI_TASK_LIBRARY_DIR", str(FIXTURE_DIR))
    assert known_studios_from_task_library_dir(None) == frozenset({"Kitchen", "Dining_Room"})


def test_known_studios_unreadable_csv_is_skipped(tmp_path, monkeypatch):
    path = tmp_path / "Unreadable.csv"
    path.write_text("title\n,,,\nStudio\nKitchen\n", encoding="utf-8")

    real_open = open

    def boom(p, *a, **k):
        if str(p) == str(path):
            raise OSError("permission denied")
        return real_open(p, *a, **k)

    monkeypatch.setattr("builtins.open", boom)
    assert known_studios_from_task_library_dir(tmp_path) == frozenset()
