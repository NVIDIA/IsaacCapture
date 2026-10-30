import json

from egocap.recording.episode_error_log import ERROR_LOG_FILENAME, log_episode_errors


def test_log_episode_errors_appends_jsonl(tmp_path):
    log_episode_errors(tmp_path, 3, "live_abort", ["pico_controller: disconnected"])
    log_episode_errors(tmp_path, 3, "data_check", ["frozen stream"])

    path = tmp_path / ERROR_LOG_FILENAME
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2

    first = json.loads(lines[0])
    assert first["episode_id"] == 3
    assert first["kind"] == "live_abort"
    assert first["reasons"] == ["pico_controller: disconnected"]
    assert "ts" in first

    second = json.loads(lines[1])
    assert second["kind"] == "data_check"


def test_log_episode_errors_creates_parent_dir(tmp_path):
    session_root = tmp_path / "nested" / "session"
    log_episode_errors(session_root, 0, "data_check_error", ["crashed"])
    assert (session_root / ERROR_LOG_FILENAME).is_file()


def test_log_episode_errors_none_episode_id(tmp_path):
    log_episode_errors(tmp_path, None, "live_abort", ["reason"])
    record = json.loads((tmp_path / ERROR_LOG_FILENAME).read_text(encoding="utf-8"))
    assert record["episode_id"] is None


def test_log_episode_errors_coerces_reasons_to_str(tmp_path):
    log_episode_errors(tmp_path, 1, "live_abort", [ValueError("boom")])
    record = json.loads((tmp_path / ERROR_LOG_FILENAME).read_text(encoding="utf-8"))
    assert record["reasons"] == ["boom"]


def test_log_episode_errors_never_raises_on_failure(tmp_path):
    # session_root is a file, not a directory -- mkdir(parents=True) on it
    # raises, which log_episode_errors must swallow (never break recording).
    blocked = tmp_path / "not_a_dir"
    blocked.write_text("", encoding="utf-8")
    log_episode_errors(blocked / "session", 0, "live_abort", ["x"])
