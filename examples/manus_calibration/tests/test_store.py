# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path

import pytest
from isaaccapture_examples.manus_calibration import store as store_module
from isaaccapture_examples.manus_calibration.protocol import ExportedFile, Side
from isaaccapture_examples.manus_calibration.store import McalStore, StoreError

ATTEMPT = "cal-test"


def stage(
    store: McalStore, side: Side, data: bytes = b"vendor-bytes", **overrides
) -> ExportedFile:
    """Write a staged file the way the worker would and return its export record."""
    staging = store.directory / ".staging" / ATTEMPT
    staging.mkdir(parents=True, exist_ok=True)
    name = f"{ATTEMPT}_{side.value}.mcal"
    (staging / name).write_bytes(data)
    fields = {
        "side": side,
        "file": name,
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }
    return ExportedFile(**{**fields, **overrides})


@pytest.fixture
def store(tmp_path: Path) -> McalStore:
    store = McalStore(tmp_path / "cal")
    store.create_staging(ATTEMPT)
    return store


def final_files(store: McalStore) -> list[str]:
    return sorted(p.name for p in store.directory.glob("*.mcal"))


def test_staging_is_private_and_unique(tmp_path):
    store = McalStore(tmp_path / "cal")
    staging = store.create_staging(ATTEMPT)
    assert stat.S_IMODE(staging.stat().st_mode) == 0o700
    assert stat.S_IMODE((store.directory / ".staging").stat().st_mode) == 0o700
    with pytest.raises(StoreError):
        store.create_staging(ATTEMPT)
    with pytest.raises(StoreError):
        store.create_staging("../escape")


def test_publish_a_pair_makes_one_manifest_with_both_files(store):
    exports = {
        Side.LEFT: stage(store, Side.LEFT, b"L" * 100),
        Side.RIGHT: stage(store, Side.RIGHT, b"R" * 50),
    }
    published = store.publish(ATTEMPT, [Side.LEFT, Side.RIGHT], exports)
    assert [f.side for f in published.files] == [Side.LEFT, Side.RIGHT]
    assert published.path_of(Side.LEFT).read_bytes() == b"L" * 100
    assert published.path_of(Side.RIGHT).read_bytes() == b"R" * 50
    manifest = json.loads(published.manifest.read_text())
    assert manifest["set_id"] == published.set_id and len(manifest["files"]) == 2
    assert store.list_published() == [published]
    assert not list((store.directory / "manifests").glob(".*"))


def test_published_files_survive_discarding_staging(store):
    published = store.publish(
        ATTEMPT, [Side.LEFT], {Side.LEFT: stage(store, Side.LEFT)}
    )
    store.discard_staging(ATTEMPT)
    assert not (store.directory / ".staging" / ATTEMPT).exists()
    assert published.path_of(Side.LEFT).read_bytes() == b"vendor-bytes"
    assert store.list_published() == [published]


@pytest.mark.parametrize(
    "override",
    [
        {"bytes": 5},
        {"sha256": "0" * 64},
        {"file": "other.mcal"},
        {"file": "../x.mcal"},
        {"file": "a.txt"},
    ],
)
def test_a_mismatched_export_publishes_nothing(store, override):
    export = stage(store, Side.LEFT, **override)
    with pytest.raises(StoreError):
        store.publish(ATTEMPT, [Side.LEFT], {Side.LEFT: export})
    assert final_files(store) == [] and store.list_published() == []


def test_an_empty_export_is_refused(store):
    export = stage(store, Side.LEFT, b"")
    with pytest.raises(StoreError, match="empty"):
        store.publish(ATTEMPT, [Side.LEFT], {Side.LEFT: export})


def test_a_pair_with_one_bad_side_publishes_neither(store):
    exports = {
        Side.LEFT: stage(store, Side.LEFT),
        Side.RIGHT: stage(store, Side.RIGHT, bytes=1),
    }
    with pytest.raises(StoreError):
        store.publish(ATTEMPT, [Side.LEFT, Side.RIGHT], exports)
    assert final_files(store) == [] and store.list_published() == []


def test_missing_or_extra_sides_are_refused(store):
    left = stage(store, Side.LEFT)
    with pytest.raises(StoreError):
        store.publish(ATTEMPT, [Side.LEFT, Side.RIGHT], {Side.LEFT: left})
    with pytest.raises(StoreError):
        store.publish(
            ATTEMPT,
            [Side.LEFT],
            {Side.LEFT: left, Side.RIGHT: stage(store, Side.RIGHT)},
        )


def test_a_symlinked_staged_file_is_refused(store, tmp_path):
    outside = tmp_path / "outside.mcal"
    outside.write_bytes(b"secret")
    export = stage(store, Side.LEFT, b"secret")
    link = store.directory / ".staging" / ATTEMPT / export.file
    link.unlink()
    link.symlink_to(outside)
    with pytest.raises(StoreError):
        store.publish(ATTEMPT, [Side.LEFT], {Side.LEFT: export})
    assert final_files(store) == []


def test_a_manifest_failure_leaves_no_selectable_files(store, monkeypatch):
    real_link = os.link
    manifest_dir = str(store.directory / "manifests")

    def failing_link(src, dst, *args, **kwargs):
        if str(dst).startswith(manifest_dir):
            raise OSError(28, "No space left on device")
        return real_link(src, dst, *args, **kwargs)

    monkeypatch.setattr(store_module.os, "link", failing_link)
    exports = {Side.LEFT: stage(store, Side.LEFT), Side.RIGHT: stage(store, Side.RIGHT)}
    with pytest.raises(StoreError, match="manifest"):
        store.publish(ATTEMPT, [Side.LEFT, Side.RIGHT], exports)
    assert final_files(store) == [] and store.list_published() == []
    assert not list((store.directory / "manifests").glob("*"))


def test_a_link_failure_midway_removes_the_files_already_linked(store, monkeypatch):
    real_link = os.link
    calls = []

    def failing_link(src, dst, *args, **kwargs):
        calls.append(dst)
        if len(calls) == 2:
            raise OSError(5, "I/O error")
        return real_link(src, dst, *args, **kwargs)

    monkeypatch.setattr(store_module.os, "link", failing_link)
    exports = {Side.LEFT: stage(store, Side.LEFT), Side.RIGHT: stage(store, Side.RIGHT)}
    with pytest.raises(StoreError):
        store.publish(ATTEMPT, [Side.LEFT, Side.RIGHT], exports)
    assert final_files(store) == []


def test_existing_files_are_never_overwritten(store, monkeypatch):
    monkeypatch.setattr(store_module.secrets, "token_hex", lambda n: "0" * (2 * n))
    first = store.publish(
        ATTEMPT, [Side.LEFT], {Side.LEFT: stage(store, Side.LEFT, b"first")}
    )
    monkeypatch.setattr(store_module, "datetime", _FrozenDatetime(first.set_id))
    # A fresh staging directory: the first set's files are hard links into the old one.
    store.discard_staging(ATTEMPT)
    store.create_staging(ATTEMPT)
    with pytest.raises(StoreError, match="unique"):
        store.publish(
            ATTEMPT, [Side.LEFT], {Side.LEFT: stage(store, Side.LEFT, b"second")}
        )
    assert first.path_of(Side.LEFT).read_bytes() == b"first"
    assert store.list_published() == [first]


class _FrozenDatetime:
    """Stands in for the datetime class so every set id collides with ``set_id``."""

    def __init__(self, set_id: str) -> None:
        from datetime import datetime as real

        self._real = real
        self._stamp = real.strptime(set_id[:16], "%Y%m%dT%H%M%SZ")

    def now(self, tz=None):
        return self._stamp.replace(tzinfo=tz)

    def strptime(self, *args):
        return self._real.strptime(*args)


def test_tampered_or_unlisted_sets_are_not_listed(store):
    published = store.publish(
        ATTEMPT, [Side.LEFT], {Side.LEFT: stage(store, Side.LEFT)}
    )
    published.path_of(Side.LEFT).write_bytes(b"tampered")
    assert store.list_published() == []
    (store.directory / "manifests" / "junk.json").write_text("{not json")
    (store.directory / "orphan.mcal").write_bytes(b"no manifest")
    assert store.list_published() == []


def test_discard_staging_ignores_missing_and_symlinked_directories(store, tmp_path):
    store.discard_staging("never-created")
    target = tmp_path / "keep"
    target.mkdir()
    (target / "file").write_text("x")
    (store.directory / ".staging" / "linked").symlink_to(target)
    store.discard_staging("linked")
    assert (target / "file").exists()


def test_a_set_id_collision_undoes_the_partial_links_and_draws_another_id(
    store, monkeypatch
):
    stamp = "20260101T000000Z"
    monkeypatch.setattr(store_module, "datetime", _FrozenDatetime(f"{stamp}-00000000"))
    draws = iter(["aaaaaaaa", "bbbbbbbb"])
    monkeypatch.setattr(store_module.secrets, "token_hex", lambda n: next(draws))
    # The right side's first-choice name is taken, so the left link made first must be undone.
    squatter = store.directory / f"{stamp}-aaaaaaaa_right.mcal"
    squatter.write_bytes(b"not ours")
    exports = {Side.LEFT: stage(store, Side.LEFT), Side.RIGHT: stage(store, Side.RIGHT)}
    published = store.publish(ATTEMPT, [Side.LEFT, Side.RIGHT], exports)
    assert published.set_id == f"{stamp}-bbbbbbbb"
    assert squatter.read_bytes() == b"not ours"
    assert not (store.directory / f"{stamp}-aaaaaaaa_left.mcal").exists()
    assert store.list_published() == [published]


def test_a_staged_directory_is_not_a_regular_file(store):
    export = stage(store, Side.LEFT)
    staged = store.directory / ".staging" / ATTEMPT / export.file
    staged.unlink()
    staged.mkdir()
    with pytest.raises(StoreError, match="regular file"):
        store.publish(ATTEMPT, [Side.LEFT], {Side.LEFT: export})
    assert final_files(store) == []


def test_unsafe_names_are_refused_before_the_staging_directory_is_touched(store):
    for name in (
        "../escape.mcal",
        "dir/x.mcal",
        ".hidden.mcal",
        "x.txt",
        "x.mcal.part",
        "",
    ):
        export = (
            stage(store, Side.LEFT, file=name)
            if name
            else stage(store, Side.LEFT, file="")
        )
        with pytest.raises(StoreError, match="unexpected export name"):
            store.publish(ATTEMPT, [Side.LEFT], {Side.LEFT: export})
    assert final_files(store) == []


def test_staging_ids_are_validated_and_unique(tmp_path):
    store = McalStore(tmp_path / "cal")
    for bad in ("", "../x", "a/b", "x" * 65, "has space"):
        with pytest.raises(StoreError):
            store.create_staging(bad)
    store.create_staging("one")
    with pytest.raises(StoreError):
        store.create_staging("one")


def test_listing_ignores_temporary_and_malformed_manifests_quietly(store, caplog):
    published = store.publish(
        ATTEMPT, [Side.LEFT], {Side.LEFT: stage(store, Side.LEFT)}
    )
    manifests = store.directory / "manifests"
    (manifests / ".20260101T000000Z-00000000.tmp").write_text("{}")
    (manifests / ".hidden.json").write_text("{not json")
    with caplog.at_level("WARNING"):
        assert store.list_published() == [published]
    assert caplog.records == []


def test_a_manifest_that_does_not_match_its_name_or_version_is_ignored(store):
    published = store.publish(
        ATTEMPT, [Side.LEFT], {Side.LEFT: stage(store, Side.LEFT)}
    )
    body = json.loads(published.manifest.read_text())
    renamed = published.manifest.with_name("20260101T000000Z-00000000.json")
    renamed.write_text(json.dumps(body))
    future = dict(body, version=2)
    published.manifest.write_text(json.dumps(future))
    assert store.list_published() == []
