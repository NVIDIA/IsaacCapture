# SPDX-FileCopyrightText: Copyright (c) 2026 RealHand. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checks for the pinned RealHand robot-asset fetcher."""

import hashlib
import io

import pytest

from isaaccapture_examples.teleop import fetch_realhand_assets as _FETCHER


def test_default_assets_use_user_cache(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    assert _FETCHER._default_output_dir() == tmp_path / "isaaccapture/realhand"


def _sibling(path: str, size: int = 4, sha256: str | None = None) -> dict:
    sibling = {"rfilename": path, "size": size, "blobId": "b" * 40}
    if sha256 is not None:
        sibling["lfs"] = {"sha256": sha256, "size": size}
    return sibling


def test_selected_files_include_only_requested_model_and_licenses():
    root = _FETCHER.REMOTE_ASSET_ROOT
    metadata = {
        "siblings": [
            *[_sibling(path) for path in sorted(_FETCHER.LICENSE_FILES)],
            _sibling(f"{root}/p7_l6/P7_l6_bimanual.urdf"),
            _sibling(f"{root}/p7_l6/meshes/link.stl", sha256="a" * 64),
            _sibling(f"{root}/p7_l20/P7_L20_bimanual.urdf"),
            _sibling("examples/isaac_lab/calibrations/operator.yml"),
            _sibling("hardware/controller_mount.3mf"),
        ]
    }

    selected = _FETCHER._selected_files(metadata, {"l6"})
    selected_paths = {item.path for item in selected}

    assert _FETCHER.LICENSE_FILES <= selected_paths
    assert f"{root}/p7_l6/P7_l6_bimanual.urdf" in selected_paths
    assert f"{root}/p7_l6/meshes/link.stl" in selected_paths
    assert not any("p7_l20" in path for path in selected_paths)
    assert not any(path.endswith((".yml", ".3mf")) for path in selected_paths)


def test_repository_metadata_rejects_a_moved_revision(monkeypatch):
    monkeypatch.setattr(_FETCHER, "_request_json", lambda _url: {"sha": "0" * 40})

    with pytest.raises(RuntimeError, match="refusing an unpinned download"):
        _FETCHER._repository_metadata()


def test_non_lfs_file_without_blob_hash_is_rejected():
    root = _FETCHER.REMOTE_ASSET_ROOT
    metadata = {"siblings": [{"rfilename": f"{root}/p7_l6/mesh.stl", "size": 4}]}
    with pytest.raises(RuntimeError, match="Git blob hash"):
        _FETCHER._selected_files(metadata, {"l6"})


@pytest.mark.parametrize("is_lfs", (False, True))
def test_same_size_corruption_is_detected_and_repaired(tmp_path, monkeypatch, is_lfs):
    content = b"mesh"
    blob_id = hashlib.sha1(b"blob 4\0" + content, usedforsecurity=False).hexdigest()
    remote = _FETCHER.RemoteFile(
        "LICENSE",
        4,
        hashlib.sha256(content).hexdigest() if is_lfs else None,
        None if is_lfs else blob_id,
    )
    target = tmp_path / "LICENSE"
    target.write_bytes(b"oops")
    assert not _FETCHER._is_complete(target, remote)
    monkeypatch.setattr(
        _FETCHER.urllib.request, "urlopen", lambda *args, **kwargs: io.BytesIO(content)
    )
    assert _FETCHER._download(remote, tmp_path, False) == (target, True)
    assert target.read_bytes() == content
    assert _FETCHER._download(remote, tmp_path, False) == (target, False)
    monkeypatch.setattr(
        _FETCHER.urllib.request, "urlopen", lambda *args, **kwargs: io.BytesIO(b"evil")
    )
    with pytest.raises(RuntimeError, match=r"hash|SHA-256"):
        _FETCHER._download(remote, tmp_path, True)
    assert target.read_bytes() == content
    assert not list(tmp_path.glob("*.part-*"))


@pytest.mark.parametrize("path", ("../secret", "/absolute/path"))
def test_remote_path_rejects_traversal(path):
    with pytest.raises(ValueError, match="Unsafe repository path"):
        _FETCHER._normalize_remote_path(path)


def test_local_paths_preserve_models_and_license_layout(tmp_path):
    root = _FETCHER.REMOTE_ASSET_ROOT

    model = _FETCHER.RemoteFile(f"{root}/p7_o6/mesh.stl", 1, None)
    license_file = _FETCHER.RemoteFile("LICENSES/Apache-2.0.txt", 1, None)

    assert _FETCHER._local_path(model, tmp_path) == tmp_path / "p7_o6" / "mesh.stl"
    assert _FETCHER._local_path(license_file, tmp_path) == (
        tmp_path / "LICENSES" / "Apache-2.0.txt"
    )


def test_validate_model_checks_urdf_hash_and_meshes(tmp_path, monkeypatch):
    asset_dir = tmp_path / "p7_l6"
    mesh_path = asset_dir / "meshes" / "link.stl"
    mesh_path.parent.mkdir(parents=True)
    mesh_path.write_bytes(b"mesh")
    urdf_path = asset_dir / "P7_l6_bimanual.urdf"
    urdf_path.write_text(
        '<robot name="test"><link name="hand"><visual><geometry>'
        '<mesh filename="meshes/link.stl"/>'
        "</geometry></visual></link></robot>",
        encoding="utf-8",
    )
    expected_hash = hashlib.sha256(urdf_path.read_bytes()).hexdigest()
    monkeypatch.setitem(_FETCHER.URDF_SHA256, "l6", expected_hash)

    assert _FETCHER._validate_model(tmp_path, "l6") == urdf_path

    mesh_path.unlink()
    with pytest.raises(FileNotFoundError, match="references missing meshes"):
        _FETCHER._validate_model(tmp_path, "l6")
