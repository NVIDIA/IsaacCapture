# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""The task-file endpoints through the real application, plus the HTTP error mapping on a small app."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from egocap_web.app import app as real_app
from egocap_web.task_file_routes import router

REAL_ROOT = Path(__file__).resolve().parents[2] / "data" / "taskfiles"


def _client(root: Path) -> TestClient:
    small = FastAPI()
    small.include_router(router)
    small.state.task_file_root = root
    return TestClient(small)


def test_the_real_application_serves_json_not_the_spa_and_stays_up_for_capture(monkeypatch, unused_tcp_port):
    monkeypatch.setenv("EGOCAP_CAPTURE_SERVER_URL", f"ws://127.0.0.1:{unused_tcp_port}")  # nothing listens there
    monkeypatch.delenv("EGOCAP_TASK_FILES_DIR", raising=False)
    with TestClient(real_app) as client:  # runs the lifespan, with no capture server reachable
        listing = client.get("/api/task-files")
        assert listing.headers["content-type"].startswith("application/json")
        assert listing.headers["cache-control"] == "no-store"
        assert listing.json() == {"files": ["Kitchen.csv"]}
        kitchen = client.get("/api/task-files/Kitchen.csv")
        assert kitchen.status_code == 200 and kitchen.json()["format"] == "studio_csv"
        assert len(kitchen.json()["tasks"]) == 11
        assert client.get("/api/task-files/nope.csv").status_code == 404
        assert client.get("/api/capture-status").status_code == 200  # preview still answers


def test_listing_and_reading_through_http(tmp_path):
    (tmp_path / "m.csv").write_text("Task Name,Task Description\nfold_towel,Fold a towel\n")
    client = _client(tmp_path)
    assert client.get("/api/task-files").json() == {"files": ["m.csv"]}
    body = client.get("/api/task-files/m.csv").json()
    assert body == {
        "name": "m.csv",
        "format": "mid_train_csv",
        "tasks": [{"name": "fold_towel", "instruction": "Fold a towel"}],
    }


@pytest.mark.parametrize(
    ("setup", "name", "status"),
    [
        (lambda root: None, "a.txt", 400),
        (lambda root: None, "absent.csv", 404),
        (lambda root: (root / "bad.csv").write_text("a,b\n"), "bad.csv", 500),
    ],
)
def test_failures_map_to_status_codes_without_paths(tmp_path, setup, name, status):
    setup(tmp_path)
    response = _client(tmp_path).get(f"/api/task-files/{name}")
    assert response.status_code == status
    assert response.headers["cache-control"] == "no-store"
    assert str(tmp_path) not in response.text


def test_an_oversized_file_is_a_server_side_500_and_a_missing_root_is_503(tmp_path, monkeypatch):
    from egocap_web import task_files

    monkeypatch.setattr(task_files, "MAX_FILE_BYTES", 8)
    (tmp_path / "big.csv").write_text("Task Name,Task Description\n" + "x,y\n" * 10)
    assert _client(tmp_path).get("/api/task-files/big.csv").status_code == 500
    missing = _client(tmp_path / "missing")
    assert missing.get("/api/task-files").status_code == 503
    assert missing.get("/api/task-files/a.csv").status_code == 503
    assert str(tmp_path) not in missing.get("/api/task-files").text


def test_the_catalog_directory_can_be_pointed_elsewhere_by_the_environment(monkeypatch, tmp_path, unused_tcp_port):
    (tmp_path / "Other.csv").write_text("Task Name,Task Description\nfold_towel,Fold a towel\n")
    monkeypatch.setenv("EGOCAP_CAPTURE_SERVER_URL", f"ws://127.0.0.1:{unused_tcp_port}")
    monkeypatch.setenv("EGOCAP_TASK_FILES_DIR", str(tmp_path))
    with TestClient(real_app) as client:
        assert client.get("/api/task-files").json() == {"files": ["Other.csv"]}


def test_a_file_with_a_name_that_is_not_valid_utf8_does_not_break_the_listing(tmp_path):
    import os

    (tmp_path / "ok.csv").write_text("Task Name,Task Description\nx,y\n")
    with open(os.path.join(os.fsencode(tmp_path), b"bad\xff.csv"), "w") as handle:
        handle.write("x")
    response = _client(tmp_path).get("/api/task-files")
    assert response.status_code == 200 and response.json() == {"files": ["ok.csv"]}
