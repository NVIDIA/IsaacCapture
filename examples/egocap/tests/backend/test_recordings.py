# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""The recordings PoC endpoints: the opt-in gate, path-traversal rejection,
listing/download behavior, and the real app's page-serving mount."""

from __future__ import annotations

import importlib
import os
import re
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from egocap_web import app as app_module
from egocap_web import recordings as recordings_module


def _reload(monkeypatch, *, enabled: bool) -> object:
    """Re-imports recordings.py with the given env so its module-level
    ``if _enabled():`` route registration (and ``_FRONTEND_DIST`` mount) is
    re-evaluated -- the routes only exist on the module at import time, so a
    live env var change alone wouldn't affect an already-imported module."""
    monkeypatch.setenv("EGOCAP_RECORDINGS_ENABLED", "1" if enabled else "0")
    importlib.reload(recordings_module)
    return recordings_module


def _client(monkeypatch, *, enabled: bool, root: Path | None = None) -> TestClient:
    """``root``, when given, is set directly on the returned app's own
    ``state.recording_output_root`` -- the same attribute the real app's
    lifespan sets once at startup (see ``_recording_output_root`` in
    recordings.py)."""
    module = _reload(monkeypatch, enabled=enabled)
    small = FastAPI()
    small.include_router(module.router)
    if root is not None:
        small.state.recording_output_root = root
    return TestClient(small)


@pytest.fixture(autouse=True)
def _restore_module_state(monkeypatch):
    # Every test reloads egocap_web.recordings with its own env; always leave
    # it reloaded back to its real default (disabled, since that's what a
    # plain `import egocap_web.app` elsewhere in the suite expects) afterward.
    yield
    monkeypatch.delenv("EGOCAP_RECORDINGS_ENABLED", raising=False)
    importlib.reload(recordings_module)


def test_disabled_by_default_the_feature_has_no_routes_at_all(monkeypatch, tmp_path):
    client = _client(monkeypatch, enabled=False, root=tmp_path)
    assert client.get("/recordings/api/list").status_code == 404
    assert client.get("/recordings/files/anything").status_code == 404
    assert client.get("/recordings/").status_code == 404


def test_enabled_the_feature_is_reachable(monkeypatch, tmp_path):
    client = _client(monkeypatch, enabled=True, root=tmp_path)
    assert client.get("/recordings/api/list").status_code == 200


def test_missing_root_is_an_empty_listing_not_an_error(monkeypatch, tmp_path):
    client = _client(monkeypatch, enabled=True, root=tmp_path / "does-not-exist")
    response = client.get("/recordings/api/list")
    assert response.status_code == 200
    assert response.json() == {"path": "", "entries": []}


def test_missing_subpath_is_404(monkeypatch, tmp_path):
    client = _client(monkeypatch, enabled=True, root=tmp_path)
    assert client.get("/recordings/api/list/nope").status_code == 404


def test_listing_shape_dirs_first_ordering_and_no_store_header(monkeypatch, tmp_path):
    (tmp_path / "b_file.mp4").write_bytes(b"x")
    (tmp_path / "a_dir").mkdir()
    (tmp_path / "a_dir" / "inner.mp4").write_bytes(b"y")
    client = _client(monkeypatch, enabled=True, root=tmp_path)
    response = client.get("/recordings/api/list")
    assert response.headers["cache-control"] == "no-store"
    body = response.json()
    assert body["path"] == ""
    names = [e["name"] for e in body["entries"]]
    assert names == ["a_dir", "b_file.mp4"]  # dirs first, then alphabetical
    dir_entry, file_entry = body["entries"]
    assert dir_entry["is_dir"] is True and dir_entry["size"] is None
    assert file_entry["is_dir"] is False and file_entry["size"] == 1


def test_a_file_with_a_name_that_is_not_valid_utf8_does_not_break_the_listing(monkeypatch, tmp_path):
    (tmp_path / "ok.mp4").write_bytes(b"x")
    with open(os.path.join(os.fsencode(tmp_path), b"bad\xff.mp4"), "wb") as handle:
        handle.write(b"x")
    client = _client(monkeypatch, enabled=True, root=tmp_path)
    response = client.get("/recordings/api/list")
    assert response.status_code == 200
    assert [e["name"] for e in response.json()["entries"]] == ["ok.mp4"]


def test_the_requested_path_itself_being_hidden_404s_the_listing(monkeypatch, tmp_path):
    (tmp_path / "session").mkdir()
    (tmp_path / "session" / ".recording").mkdir()
    (tmp_path / "session" / ".recording" / "session.lock").write_bytes(b"x")
    (tmp_path / "session" / "episode_3.pending").mkdir()
    (tmp_path / "session" / "episode_3.pending" / "cam.mp4").write_bytes(b"x")
    client = _client(monkeypatch, enabled=True, root=tmp_path)
    # Listing a hidden directory directly must 404, not list its contents --
    # the entries are hidden from the parent listing precisely because
    # they're not meant to be browsed at all, not just unlisted.
    assert client.get("/recordings/api/list/session/.recording").status_code == 404
    assert client.get("/recordings/api/list/session/episode_3.pending").status_code == 404


def test_hidden_entries_are_excluded_from_listings_and_downloads(monkeypatch, tmp_path):
    (tmp_path / ".recording").mkdir()
    (tmp_path / ".recording" / "session.lock").write_bytes(b"x")
    (tmp_path / "episode_0.pending").mkdir()
    (tmp_path / "episode_0.pending" / "cam.mp4").write_bytes(b"x")
    (tmp_path / "episode_0").mkdir()
    (tmp_path / "episode_0" / "cam.mp4").write_bytes(b"done")
    client = _client(monkeypatch, enabled=True, root=tmp_path)
    names = [e["name"] for e in client.get("/recordings/api/list").json()["entries"]]
    assert names == ["episode_0"]  # .recording and episode_0.pending both hidden
    assert client.get("/recordings/files/.recording/session.lock").status_code == 404
    assert client.get("/recordings/files/episode_0.pending/cam.mp4").status_code == 404
    assert client.get("/recordings/files/episode_0/cam.mp4").status_code == 200


def test_path_traversal_is_rejected_on_both_routes(monkeypatch, tmp_path):
    # Real siblings of the root, not a nonexistent path like /etc/passwd --
    # a nonexistent target passes these assertions even with the
    # containment check deleted, because every failure mode (missing
    # subpath, not-a-file) already 404s on its own. These must only pass
    # because _resolve actually rejects the target.
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()
    outside_file = outside_dir / "secret.txt"
    outside_file.write_text("nope")
    root = tmp_path / "root"
    root.mkdir()
    client = _client(monkeypatch, enabled=True, root=root)

    assert client.get("/recordings/api/list/..%2Foutside").status_code == 404
    assert client.get(f"/recordings/api/list/{outside_dir}").status_code == 404  # absolute path
    assert client.get("/recordings/files/..%2Foutside%2Fsecret.txt").status_code == 404
    assert client.get(f"/recordings/files/{outside_file}").status_code == 404  # absolute path


def test_a_symlink_escaping_the_root_is_rejected(monkeypatch, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("nope")
    root = tmp_path / "root"
    root.mkdir()
    (root / "escape").symlink_to(outside)
    client = _client(monkeypatch, enabled=True, root=root)
    assert client.get("/recordings/api/list/escape").status_code == 404
    assert client.get("/recordings/files/escape/secret.txt").status_code == 404


def test_a_successful_download(monkeypatch, tmp_path):
    (tmp_path / "episode_0").mkdir()
    (tmp_path / "episode_0" / "cam.mp4").write_bytes(b"video-bytes")
    client = _client(monkeypatch, enabled=True, root=tmp_path)
    response = client.get("/recordings/files/episode_0/cam.mp4")
    assert response.status_code == 200
    assert response.content == b"video-bytes"
    assert "attachment" in response.headers["content-disposition"]


def test_recording_output_root_uses_app_state_when_set(monkeypatch, tmp_path):
    # app.state.recording_output_root is what the real app sets once at startup (see
    # app.py's lifespan); recordings.py's standalone router reads that same attribute
    # back -- proven through actual listing behavior, not just a direct attribute read.
    app_state_root = tmp_path / "from_app_state"
    app_state_root.mkdir()
    (app_state_root / "episode_5").mkdir()
    client = _client(monkeypatch, enabled=True, root=app_state_root)
    response = client.get("/recordings/api/list")
    assert response.status_code == 200
    assert {e["name"] for e in response.json()["entries"]} == {"episode_5"}


def test_the_real_application_serves_the_recordings_page_despite_the_router_mount(
    monkeypatch, unused_tcp_port, built_recordings_frontend_dist
):
    # built_recordings_frontend_dist runs the real `npm ci && npm run build`
    # for recordings_frontend/ into its real (gitignored) dist/ *before* the
    # reloads below -- recordings.py's `if _FRONTEND_DIST.is_dir():` route
    # registration happens at import/reload time, so the build has to land
    # first, not just before the app is used.
    recordings_dist = built_recordings_frontend_dist

    monkeypatch.setenv("EGOCAP_CAPTURE_SERVER_URL", f"ws://127.0.0.1:{unused_tcp_port}")  # nothing listens there
    _reload(monkeypatch, enabled=True)
    importlib.reload(app_module)
    try:
        with TestClient(app_module.app) as client:
            index = client.get("/recordings/")
            assert index.status_code == 200
            assert "text/html" in index.headers["content-type"]
            # Byte-for-byte, not just "some page came back" -- proves the
            # server delivered the actual built file, not a cached/stale one.
            assert index.content == (recordings_dist / "index.html").read_bytes()

            redirect = client.get("/recordings", follow_redirects=False)
            assert redirect.status_code == 307
            assert redirect.headers["location"] == "/recordings/"

            # A real asset the built page actually references, not just the
            # shell -- proves the frontend's `base: "/recordings/"` config
            # and this backend's mount prefix agree, not just that *some*
            # file under dist/ happens to be reachable.
            script_paths = re.findall(r'src="(/recordings/assets/[^"]+\.js)"', index.text)
            assert script_paths, "index.html referenced no /recordings/assets/*.js script"
            asset = client.get(script_paths[0])
            assert asset.status_code == 200
            asset_file = recordings_dist / script_paths[0].removeprefix("/recordings/")
            assert asset.content == asset_file.read_bytes()

            assert client.get("/recordings/assets/nonexistent.js").status_code == 404
    finally:
        # Leave both modules reloaded back to the disabled-by-default state
        # other test files importing them expect.
        monkeypatch.delenv("EGOCAP_RECORDINGS_ENABLED", raising=False)
        importlib.reload(recordings_module)
        importlib.reload(app_module)
