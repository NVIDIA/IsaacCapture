# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Operator-imported task catalogs: import by path or upload, the merged listing, shadowing, and removal."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from egocap_web import task_files
from egocap_web.task_file_routes import router

TOKEN = "t" * 43
MID_TRAIN = "Task Name,Task Description\nfold_towel,Fold a towel\nstack_cups,Stack the cups\n"
STUDIO = "Kitchen,,,\n,,,\nStudio,Scenario,Task,Description\nA,Sink,wash_cup,Wash the cup\n"


@dataclass
class _Context:
    control_token: str


def _client(tmp_path: Path, *, imported: bool = True, context: bool = True) -> TestClient:
    small = FastAPI()
    small.include_router(router)
    shipped = tmp_path / "shipped"
    shipped.mkdir(exist_ok=True)  # a test may build two clients over one directory
    (shipped / "Kitchen.csv").write_text(STUDIO)
    small.state.task_file_root = shipped
    if imported:
        small.state.imported_task_file_root = tmp_path / "imported"
    if context:
        small.state.recording_context = _Context(control_token=TOKEN)
    return TestClient(small)


def test_listing_merges_shipped_and_imported_and_marks_the_imports(tmp_path):
    client = _client(tmp_path)
    assert client.get("/api/task-files").json() == {"files": ["Kitchen.csv"], "imported": []}
    response = client.post(
        "/api/task-files/import", json={"control_token": TOKEN, "name": "Mid.csv", "content": MID_TRAIN}
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["file"]["format"] == "mid_train_csv" and [t["name"] for t in body["file"]["tasks"]] == [
        "fold_towel",
        "stack_cups",
    ]
    assert body["files"] == ["Kitchen.csv", "Mid.csv"] and body["imported"] == ["Mid.csv"]
    assert client.get("/api/task-files/Mid.csv").json()["tasks"][1]["instruction"] == "Stack the cups"


def test_import_by_path_reads_the_file_on_the_backend_machine(tmp_path):
    downloaded = tmp_path / "Downloads" / "Garage tasks.csv"
    downloaded.parent.mkdir()
    downloaded.write_text(MID_TRAIN)
    client = _client(tmp_path)
    response = client.post("/api/task-files/import", json={"control_token": TOKEN, "path": str(downloaded)})
    assert response.status_code == 200, response.text
    assert response.json()["imported"] == ["Garage tasks.csv"]
    assert (tmp_path / "imported" / "Garage tasks.csv").read_text() == MID_TRAIN
    missing = client.post(
        "/api/task-files/import", json={"control_token": TOKEN, "path": str(tmp_path / "nope.csv")}
    )
    assert missing.status_code == 404
    directory = client.post("/api/task-files/import", json={"control_token": TOKEN, "path": str(tmp_path)})
    assert directory.status_code == 400


def test_an_import_shadows_a_shipped_catalog_of_the_same_name_without_touching_it(tmp_path):
    client = _client(tmp_path)
    client.post(
        "/api/task-files/import", json={"control_token": TOKEN, "name": "Kitchen.csv", "content": MID_TRAIN}
    )
    assert client.get("/api/task-files").json() == {"files": ["Kitchen.csv"], "imported": ["Kitchen.csv"]}
    assert client.get("/api/task-files/Kitchen.csv").json()["format"] == "mid_train_csv"
    assert (tmp_path / "shipped" / "Kitchen.csv").read_text() == STUDIO
    # Removing the import uncovers the shipped catalog again.
    assert (
        client.request("DELETE", "/api/task-files/Kitchen.csv", json={"control_token": TOKEN}).status_code == 200
    )
    assert client.get("/api/task-files/Kitchen.csv").json()["format"] == "studio_csv"


@pytest.mark.parametrize(
    ("body", "status", "fragment"),
    [
        ({"name": "Mid.csv", "content": MID_TRAIN}, 403, "control token"),
        ({"control_token": "x" * 43, "name": "Mid.csv", "content": MID_TRAIN}, 403, "control token"),
        ({"control_token": TOKEN}, 400, "either path"),
        ({"control_token": TOKEN, "name": "notes.txt", "content": MID_TRAIN}, 400, "ending in .csv"),
        ({"control_token": TOKEN, "name": "../x.csv", "content": MID_TRAIN}, 400, "ending in .csv"),
        ({"control_token": TOKEN, "name": "Bad.csv", "content": "a,b\n1,2\n"}, 400, "neither a studio catalog"),
        ({"control_token": TOKEN, "name": "Bad.csv", "content": "Task Name,Task Description\nx,\n"}, 400, "empty"),
    ],
)
def test_a_refused_import_stores_nothing(tmp_path, body, status, fragment):
    client = _client(tmp_path)
    response = client.post("/api/task-files/import", json=body)
    assert response.status_code == status, response.text
    assert fragment in response.json()["detail"]
    assert not (tmp_path / "imported").exists() or list((tmp_path / "imported").iterdir()) == []


def test_an_oversized_upload_is_the_operators_mistake(tmp_path, monkeypatch):
    monkeypatch.setattr(task_files, "MAX_FILE_BYTES", 16)
    client = _client(tmp_path)
    response = client.post(
        "/api/task-files/import", json={"control_token": TOKEN, "name": "Mid.csv", "content": MID_TRAIN}
    )
    assert response.status_code == 400 and "larger than" in response.json()["detail"]


def test_removal_is_token_gated_and_only_for_imports(tmp_path):
    client = _client(tmp_path)
    client.post("/api/task-files/import", json={"control_token": TOKEN, "name": "Mid.csv", "content": MID_TRAIN})
    assert client.request("DELETE", "/api/task-files/Mid.csv", json={}).status_code == 403
    assert (
        client.request("DELETE", "/api/task-files/Kitchen.csv", json={"control_token": TOKEN}).status_code == 404
    )
    gone = client.request("DELETE", "/api/task-files/Mid.csv", json={"control_token": TOKEN})
    assert gone.status_code == 200 and gone.json() == {"files": ["Kitchen.csv"], "imported": []}
    assert client.request("DELETE", "/api/task-files/Mid.csv", json={"control_token": TOKEN}).status_code == 404


def test_without_an_imported_root_or_context_the_write_endpoints_say_so(tmp_path):
    no_root = _client(tmp_path, imported=False)
    assert (
        no_root.post(
            "/api/task-files/import", json={"control_token": TOKEN, "name": "M.csv", "content": MID_TRAIN}
        ).status_code
        == 503
    )
    assert no_root.request("DELETE", "/api/task-files/M.csv", json={"control_token": TOKEN}).status_code == 404
    assert no_root.get("/api/task-files").json() == {"files": ["Kitchen.csv"], "imported": []}
    no_context = _client(tmp_path, context=False)
    assert (
        no_context.post(
            "/api/task-files/import", json={"control_token": TOKEN, "name": "M.csv", "content": MID_TRAIN}
        ).status_code
        == 503
    )
