# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Operator settings: the file store, the token-gated PUT, the tools registry, and the real application."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from egocap_web.app import app as real_app
from egocap_web.operator_settings import (
    CAMERA_TYPES,
    OperatorSettings,
    OperatorSettingsError,
    OperatorSettingsStore,
    operator_settings_path,
    router,
)

TOKEN = "t" * 43


@dataclass
class _Context:
    control_token: str


def _client(tmp_path: Path, with_store: bool = True) -> TestClient:
    small = FastAPI()
    small.include_router(router)
    if with_store:
        small.state.operator_settings_store = OperatorSettingsStore(tmp_path / "operator_settings.json")
        small.state.recording_context = _Context(control_token=TOKEN)
    return TestClient(small)


def test_store_defaults_when_nothing_is_saved_and_round_trips_a_value(tmp_path):
    store = OperatorSettingsStore(tmp_path / "nested" / "operator_settings.json")
    assert store.load() == OperatorSettings()
    store.save(OperatorSettings(camera_type="oak_d"))
    assert store.load() == OperatorSettings(camera_type="oak_d")
    assert json.loads(store.path.read_text()) == {
        "camera_type": "oak_d",
        "camera_roles": {"ego": None, "wrist_left": None, "wrist_right": None},
        "task_file": None,
        "preselected_task": None,
    }
    assert not store.path.with_name("operator_settings.json.tmp").exists()


@pytest.mark.parametrize("text", ["not json", "[1, 2]", '{"camera_type": "webcam"}'])
def test_store_rejects_an_unusable_file_instead_of_guessing(tmp_path, text):
    path = tmp_path / "operator_settings.json"
    path.write_text(text)
    with pytest.raises(OperatorSettingsError):
        OperatorSettingsStore(path).load()


def test_path_is_hidden_under_the_recordings_unless_overridden(tmp_path, monkeypatch):
    monkeypatch.delenv("EGOCAP_OPERATOR_SETTINGS_FILE", raising=False)
    assert operator_settings_path(tmp_path) == tmp_path / ".egocap" / "operator_settings.json"
    monkeypatch.setenv("EGOCAP_OPERATOR_SETTINGS_FILE", str(tmp_path / "elsewhere.json"))
    assert operator_settings_path(tmp_path) == tmp_path / "elsewhere.json"


def test_get_reports_defaults_and_the_accepted_types(tmp_path):
    response = _client(tmp_path).get("/api/operator-settings")
    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    assert response.json() == {
        "camera_type": None,
        "camera_types": list(CAMERA_TYPES),
        "camera_roles": {"ego": None, "wrist_left": None, "wrist_right": None},
        "roles": ["ego", "wrist_left", "wrist_right"],
        "task_file": None,
        "preselected_task": None,
    }


def test_put_changes_only_the_fields_it_names(tmp_path):
    client = _client(tmp_path)
    client.put("/api/operator-settings", json={"control_token": TOKEN, "camera_type": "oak_d"})
    tasks = client.put(
        "/api/operator-settings",
        json={"control_token": TOKEN, "task_file": "Kitchen.csv", "preselected_task": "wash_cup"},
    )
    assert tasks.status_code == 200, tasks.text
    body = tasks.json()
    assert (
        body["camera_type"] == "oak_d"
        and body["task_file"] == "Kitchen.csv"
        and body["preselected_task"] == "wash_cup"
    )
    stored = json.loads((tmp_path / "operator_settings.json").read_text())
    assert stored == {
        "camera_type": "oak_d",
        "camera_roles": {"ego": None, "wrist_left": None, "wrist_right": None},
        "task_file": "Kitchen.csv",
        "preselected_task": "wash_cup",
    }
    cleared = client.put("/api/operator-settings", json={"control_token": TOKEN, "preselected_task": None})
    assert cleared.json()["preselected_task"] is None and cleared.json()["task_file"] == "Kitchen.csv"


@pytest.mark.parametrize(
    "body",
    [
        {"task_file": "../x.csv"},
        {"task_file": "notes.txt"},
        {"task_file": 3},
        {"preselected_task": ""},
        {"preselected_task": 1},
    ],
)
def test_put_rejects_a_bad_task_setting(tmp_path, body):
    response = _client(tmp_path).put("/api/operator-settings", json={"control_token": TOKEN, **body})
    assert response.status_code == 400, response.text


def test_endpoints_answer_503_before_the_store_exists(tmp_path):
    client = _client(tmp_path, with_store=False)
    assert client.get("/api/operator-settings").status_code == 503
    assert (
        client.put("/api/operator-settings", json={"control_token": TOKEN, "camera_type": "oak_d"}).status_code
        == 503
    )


def test_put_needs_the_control_token_before_it_looks_at_values(tmp_path):
    client = _client(tmp_path)
    for body in (
        {"camera_type": "oak_d"},
        {"control_token": "x" * 43, "camera_type": "oak_d"},
        {"control_token": TOKEN[:-1] + "!", "camera_type": "bogus"},
    ):
        response = client.put("/api/operator-settings", json=body)
        assert response.status_code == 403, body
        assert response.json() == {"detail": "control token rejected"}
    assert client.get("/api/operator-settings").json()["camera_type"] is None


def test_put_rejects_a_bad_value_or_body_with_the_reason(tmp_path):
    client = _client(tmp_path)
    bad_value = client.put("/api/operator-settings", json={"control_token": TOKEN, "camera_type": "webcam"})
    assert bad_value.status_code == 400
    assert "camera_type must be one of" in bad_value.json()["detail"]
    assert client.put("/api/operator-settings", json=[TOKEN]).status_code == 400
    assert (
        client.put(
            "/api/operator-settings", content=b"{", headers={"content-type": "application/json"}
        ).status_code
        == 400
    )


def test_put_saves_and_clears_the_camera_type(tmp_path):
    client = _client(tmp_path)
    saved = client.put("/api/operator-settings", json={"control_token": TOKEN, "camera_type": "sensing_gmsl"})
    assert saved.status_code == 200 and saved.json()["camera_type"] == "sensing_gmsl"
    assert client.get("/api/operator-settings").json()["camera_type"] == "sensing_gmsl"
    assert (tmp_path / "operator_settings.json").exists()
    cleared = client.put("/api/operator-settings", json={"control_token": TOKEN, "camera_type": None})
    assert cleared.status_code == 200 and cleared.json()["camera_type"] is None


def test_an_unusable_file_shows_defaults_so_the_form_can_repair_it(tmp_path):
    (tmp_path / "operator_settings.json").write_text("nope")
    response = _client(tmp_path).get("/api/operator-settings")
    assert response.status_code == 200
    assert response.json()["camera_type"] is None
    assert "invalid" in response.json()["warning"]


def test_tools_are_listed_as_unavailable_with_a_reason_and_scope(tmp_path):
    body = _client(tmp_path).get("/api/tools").json()
    ids = [tool["id"] for tool in body["tools"]]
    assert ids == ["headset_calibration", "glove_calibration", "fisheye_rectification"]
    for tool in body["tools"]:
        assert tool["available"] is False and tool["unavailable_reason"]
    fisheye = body["tools"][2]
    assert fisheye["applies_to"] == "sensing_gmsl"
    assert "fisheye" in fisheye["description"].lower()


def test_the_real_application_honors_the_settings_path_override(monkeypatch, unused_tcp_port, tmp_path):
    monkeypatch.setenv("EGOCAP_CAPTURE_SERVER_URL", f"ws://127.0.0.1:{unused_tcp_port}")
    monkeypatch.setenv("EGOCAP_OPERATOR_SETTINGS_FILE", str(tmp_path / "op.json"))
    with TestClient(real_app) as client:
        assert client.get("/api/operator-settings").json()["camera_type"] is None
        # The real run's token is only printed to the backend's console, so the wrong one is all a test can send.
        assert (
            client.put("/api/operator-settings", json={"control_token": TOKEN, "camera_type": "oak_d"}).status_code
            == 403
        )
        assert client.get("/api/tools").status_code == 200


def test_camera_roles_start_unassigned_and_change_one_role_at_a_time(tmp_path):
    client = _client(tmp_path)
    body = client.get("/api/operator-settings").json()
    assert body["roles"] == ["ego", "wrist_left", "wrist_right"]
    assert body["camera_roles"] == {"ego": None, "wrist_left": None, "wrist_right": None}
    first = client.put("/api/operator-settings", json={"control_token": TOKEN, "camera_roles": {"ego": "Ego"}})
    assert first.status_code == 200, first.text
    assert first.json()["camera_roles"] == {"ego": "Ego", "wrist_left": None, "wrist_right": None}
    second = client.put(
        "/api/operator-settings",
        json={"control_token": TOKEN, "camera_roles": {"wrist_left": "Wrist left", "wrist_right": "Wrist right"}},
    )
    assert second.json()["camera_roles"] == {
        "ego": "Ego",
        "wrist_left": "Wrist left",
        "wrist_right": "Wrist right",
    }
    stored = json.loads((tmp_path / "operator_settings.json").read_text())
    assert stored["camera_roles"] == {"ego": "Ego", "wrist_left": "Wrist left", "wrist_right": "Wrist right"}
    # Swapping the wrists in one change is allowed: the check is on the result, not on each step.
    swapped = client.put(
        "/api/operator-settings",
        json={"control_token": TOKEN, "camera_roles": {"wrist_left": "Wrist right", "wrist_right": "Wrist left"}},
    )
    assert swapped.json()["camera_roles"]["wrist_left"] == "Wrist right"
    cleared = client.put("/api/operator-settings", json={"control_token": TOKEN, "camera_roles": {"ego": None}})
    assert cleared.json()["camera_roles"]["ego"] is None
    assert cleared.json()["camera_roles"]["wrist_left"] == "Wrist right"


@pytest.mark.parametrize(
    ("roles", "fragment"),
    [
        ({"head": "Ego"}, "unknown camera role"),
        ({"ego": ""}, "camera name or null"),
        ({"ego": 1}, "camera name or null"),
        ({"ego": "Cam", "wrist_left": "Cam"}, "two roles"),
        ("Ego", "must be an object"),
    ],
)
def test_camera_roles_reject_bad_assignments_without_storing_them(tmp_path, roles, fragment):
    client = _client(tmp_path)
    response = client.put("/api/operator-settings", json={"control_token": TOKEN, "camera_roles": roles})
    assert response.status_code == 400, response.text
    assert fragment in response.json()["detail"]
    assert client.get("/api/operator-settings").json()["camera_roles"]["ego"] is None


@pytest.mark.parametrize("raw", [b"nope", b"\xff", b'{"camera_type": "unknown"}'])
def test_authenticated_put_repairs_corrupt_storage(tmp_path, raw, caplog):
    path = tmp_path / "operator_settings.json"
    path.write_bytes(raw)
    client = _client(tmp_path)
    refused = client.put("/api/operator-settings", json={"camera_type": "oak_d"})
    assert refused.status_code == 403 and path.read_bytes() == raw
    saved = client.put("/api/operator-settings", json={"control_token": TOKEN, "camera_type": "oak_d"})
    assert saved.status_code == 200
    assert OperatorSettingsStore(path).load().camera_type == "oak_d"
    assert "Replacing invalid operator settings" in caplog.text


def test_overlapping_partial_updates_preserve_both_changes(tmp_path, monkeypatch):
    import time
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    client = _client(tmp_path)
    store = client.app.state.operator_settings_store
    original = store.load

    def slow_load():
        current = original()
        time.sleep(0.03)
        return current

    monkeypatch.setattr(store, "load", slow_load)
    start = Barrier(2)

    def save(changes):
        start.wait(timeout=5)
        return client.put("/api/operator-settings", json={"control_token": TOKEN, **changes})

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(save, {"camera_type": "oak_d"})
        second = pool.submit(save, {"camera_roles": {"ego": "Head"}})
        assert first.result(timeout=5).status_code == second.result(timeout=5).status_code == 200
    saved = original()
    assert saved.camera_type == "oak_d" and saved.camera_roles["ego"] == "Head"


def test_catalog_and_preselected_task_are_checked_and_catalog_changes_clear_the_task(tmp_path):
    client = _client(tmp_path)
    root = tmp_path / "catalogs"
    root.mkdir()
    (root / "A.csv").write_text("Task Name,Task Description\nfold,Fold a towel\n")
    (root / "B.csv").write_text("Task Name,Task Description\nstack,Stack the cups\n")
    client.app.state.task_file_root = root
    for changes in (
        {"task_file": "Missing.csv"},
        {"preselected_task": "fold"},
        {"task_file": "A.csv", "preselected_task": "unknown"},
    ):
        response = client.put("/api/operator-settings", json={"control_token": TOKEN, **changes})
        assert response.status_code == 400
    saved = client.put(
        "/api/operator-settings", json={"control_token": TOKEN, "task_file": "A.csv", "preselected_task": "fold"}
    )
    assert saved.status_code == 200
    switched = client.put("/api/operator-settings", json={"control_token": TOKEN, "task_file": "B.csv"})
    assert switched.status_code == 200 and switched.json()["preselected_task"] is None


def test_failed_replace_preserves_saved_settings_and_cleans_temporary_file(tmp_path, monkeypatch):
    from egocap_web import operator_settings

    store = OperatorSettingsStore(tmp_path / "settings.json")
    store.save(OperatorSettings(camera_type="oak_d"))

    def fail(*args):
        raise OSError("storage unavailable")

    monkeypatch.setattr(operator_settings.os, "replace", fail)
    with pytest.raises(OSError):
        store.save(OperatorSettings(camera_type="sensing_gmsl"))
    assert store.load().camera_type == "oak_d"
    assert sorted(path.name for path in tmp_path.iterdir()) == ["settings.json"]


def test_real_application_uses_hidden_settings_and_catalog_storage_by_default(
    monkeypatch, unused_tcp_port, tmp_path
):
    from egocap_web import app as app_module

    monkeypatch.setenv("EGOCAP_CAPTURE_SERVER_URL", f"ws://127.0.0.1:{unused_tcp_port}")
    monkeypatch.delenv("EGOCAP_OPERATOR_SETTINGS_FILE", raising=False)
    monkeypatch.delenv("EGOCAP_STATION_CONFIG", raising=False)
    monkeypatch.setattr(app_module, "_recording_output_root", lambda data_dir: tmp_path)
    with TestClient(real_app) as client:
        token = real_app.state.recording_context.control_token
        response = client.put("/api/operator-settings", json={"control_token": token, "camera_type": "oak_d"})
        assert response.status_code == 200
        assert real_app.state.operator_settings_store.path == tmp_path / ".egocap" / "operator_settings.json"
        assert real_app.state.imported_task_file_root == tmp_path / ".egocap" / "taskfiles"
        assert (tmp_path / ".egocap" / "operator_settings.json").is_file()
