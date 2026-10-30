from pathlib import Path

import pytest
import zarr

from egocap.eod.eod_studio_session_move import (
    PENDING_SUFFIX,
    _episode_index_from_key,
    ensure_dataset_level_seed_fields,
    refresh_session_egocap_zarr_aggregates,
    relocate_session_to_canonical_studio,
)

KNOWN = frozenset({"Kitchen", "Dining_Room"})


def _session_dir(eod_root: Path, studio: str, ts: str = "2026-01-01-10-00-00") -> Path:
    d = eod_root / studio / ts
    d.mkdir(parents=True, exist_ok=True)
    return d


def test_relocate_empty_desired_studio_is_noop(tmp_path):
    session = _session_dir(tmp_path, "_unmatched")
    result = relocate_session_to_canonical_studio(session, "", KNOWN)
    assert result == session.resolve()


def test_relocate_non_session_dir_is_noop(tmp_path):
    not_a_session = tmp_path / "random_folder"
    not_a_session.mkdir()
    result = relocate_session_to_canonical_studio(not_a_session, "Kitchen", KNOWN)
    assert result == not_a_session


def test_relocate_missing_dir_is_noop(tmp_path):
    missing = tmp_path / "2026-01-01-10-00-00"
    result = relocate_session_to_canonical_studio(missing, "Kitchen", KNOWN)
    assert result == missing


def test_relocate_unmatched_target_is_noop(tmp_path):
    session = _session_dir(tmp_path, "Kitchen")
    # desired_studio_meta itself sanitizes/canonicalizes to "_unmatched" ->
    # the `want == _UNMATCHED` branch, a deliberate no-op (never route
    # a session *into* the unmatched bucket via this path).
    result = relocate_session_to_canonical_studio(session, "_unmatched", KNOWN)
    assert result == session
    assert session.is_dir()


def test_relocate_moves_from_unmatched_to_canonical_studio(tmp_path):
    session = _session_dir(tmp_path, "_unmatched")
    result = relocate_session_to_canonical_studio(session, "Kitchen", KNOWN)
    assert result == tmp_path / "Kitchen" / "2026-01-01-10-00-00"
    assert result.is_dir()
    assert not session.is_dir()


def test_relocate_already_in_canonical_studio_is_noop(tmp_path):
    session = _session_dir(tmp_path, "Kitchen")
    result = relocate_session_to_canonical_studio(session, "Kitchen", KNOWN)
    assert result == session
    assert session.is_dir()


def test_relocate_case_insensitive_match_is_noop(tmp_path):
    session = _session_dir(tmp_path, "Kitchen")
    result = relocate_session_to_canonical_studio(session, "kitchen", KNOWN)
    assert result == session


def test_relocate_wrong_studio_folder_moves(tmp_path):
    session = _session_dir(tmp_path, "Dining_Room")
    result = relocate_session_to_canonical_studio(session, "Kitchen", KNOWN)
    assert result == tmp_path / "Kitchen" / "2026-01-01-10-00-00"


def test_relocate_destination_collision_raises(tmp_path):
    session = _session_dir(tmp_path, "_unmatched")
    _session_dir(tmp_path, "Kitchen")  # pre-existing destination
    with pytest.raises(OSError, match="refuse overwrite"):
        relocate_session_to_canonical_studio(session, "Kitchen", KNOWN)


def test_relocate_have_equals_want_but_different_case_is_noop(tmp_path):
    # session folder is "kitchen" (lowercase); canonicalizing both "have" and
    # "want" against KNOWN resolves them to the same "Kitchen" spelling, even
    # though the literal directories differ -- the have==want check (not the
    # earlier identical-path check) is what makes this a no-op.
    session = _session_dir(tmp_path, "kitchen")
    result = relocate_session_to_canonical_studio(session, "Kitchen", KNOWN)
    assert result == session
    assert session.is_dir()


def test_relocate_default_known_studios_uses_task_library(tmp_path):
    session = _session_dir(tmp_path, "_unmatched")
    # known_studios=None falls back to TASK_LIBRARY.keys(), which is empty in
    # this environment (no task_library/ CSVs shipped) -- canonicalize just
    # sanitizes "Kitchen" since it's not a recognized known studio either way.
    result = relocate_session_to_canonical_studio(session, "Kitchen", None)
    assert result == tmp_path / "Kitchen" / "2026-01-01-10-00-00"


def test_episode_index_from_key():
    assert _episode_index_from_key("episode_7") == 7
    assert _episode_index_from_key("episode_7" + PENDING_SUFFIX) is None
    assert _episode_index_from_key("not_an_episode") is None
    assert _episode_index_from_key(123) is None
    assert _episode_index_from_key("episode_") is None
    assert _episode_index_from_key("episode_abc") is None


def test_ensure_dataset_level_seed_fields_sets_defaults(tmp_path):
    store = zarr.open(str(tmp_path), mode="a")
    ensure_dataset_level_seed_fields(store)
    assert store.attrs["discarded_episode_indices"] == []
    assert store.attrs["discarded_reasons"] == {}


def test_ensure_dataset_level_seed_fields_preserves_existing(tmp_path):
    store = zarr.open(str(tmp_path), mode="a")
    store.attrs["discarded_episode_indices"] = [1, 2]
    ensure_dataset_level_seed_fields(store)
    assert store.attrs["discarded_episode_indices"] == [1, 2]


def test_refresh_session_egocap_zarr_aggregates(tmp_path):
    store = zarr.open(str(tmp_path), mode="a")
    ep0 = store.create_group("episode_0")
    ep0.attrs["episode_length"] = 100
    ep0.attrs["episode_duration"] = 1800.0
    ep1 = store.create_group("episode_1")
    ep1.attrs["episode_length"] = 50
    ep1.attrs["episode_duration"] = 1800.0
    # A still-pending episode must not be counted as finalized.
    store.create_group(f"episode_2{PENDING_SUFFIX}")

    refresh_session_egocap_zarr_aggregates(tmp_path)

    store2 = zarr.open(str(tmp_path), mode="r")
    assert store2.attrs["num_episodes"] == 2
    assert store2.attrs["episode_lengths"] == {"0": 100, "1": 50}
    assert store2.attrs["total_data_time_hours"] == 1.0
    assert store2.attrs["discarded_episode_indices"] == []


def test_refresh_session_egocap_zarr_aggregates_missing_duration_and_length(tmp_path):
    store = zarr.open(str(tmp_path), mode="a")
    store.create_group("episode_0")  # no attrs at all
    refresh_session_egocap_zarr_aggregates(tmp_path)
    store2 = zarr.open(str(tmp_path), mode="r")
    assert store2.attrs["num_episodes"] == 1
    assert store2.attrs["episode_lengths"] == {}
    assert store2.attrs["total_data_time_hours"] == 0.0


def test_refresh_session_egocap_zarr_aggregates_no_episodes(tmp_path):
    zarr.open(str(tmp_path), mode="a")
    refresh_session_egocap_zarr_aggregates(tmp_path)
    store2 = zarr.open(str(tmp_path), mode="r")
    assert store2.attrs["num_episodes"] == 0
    assert store2.attrs["total_data_time_hours"] == 0.0


def test_refresh_session_egocap_zarr_aggregates_accepts_str_path(tmp_path):
    zarr.open(str(tmp_path), mode="a")
    refresh_session_egocap_zarr_aggregates(str(tmp_path))
    store2 = zarr.open(str(tmp_path), mode="r")
    assert store2.attrs["num_episodes"] == 0


def test_refresh_session_egocap_zarr_aggregates_ignores_non_digit_episode_key(tmp_path):
    store = zarr.open(str(tmp_path), mode="a")
    store.create_group("episode_abc")  # starts with episode_, not pending, but
    # _episode_index_from_key rejects the non-digit suffix -- excluded from
    # the finalized count rather than crashing on int("abc").
    refresh_session_egocap_zarr_aggregates(tmp_path)
    store2 = zarr.open(str(tmp_path), mode="r")
    assert store2.attrs["num_episodes"] == 0


def test_refresh_session_egocap_zarr_aggregates_keys_fallback_to_iteration(tmp_path, monkeypatch):
    import egocap.eod.eod_studio_session_move as mod

    store = zarr.open(str(tmp_path), mode="a")
    ep0 = store.create_group("episode_0")
    ep0.attrs["episode_length"] = 10
    ep0.attrs["episode_duration"] = 36.0

    real_open = mod.zarr.open

    class _NoKeysGroup:
        """Wraps a real zarr group but hides .keys() to force the
        AttributeError fallback (`names = list(store)`) in
        refresh_session_egocap_zarr_aggregates -- some store-like objects
        support iteration without exposing a dict-style .keys()."""

        def __init__(self, inner):
            self._inner = inner

        def __getattr__(self, name):
            if name == "keys":
                raise AttributeError("no keys() on this store-like object")
            return getattr(self._inner, name)

        def __iter__(self):
            return iter(self._inner)

        def __getitem__(self, key):
            return self._inner[key]

    def fake_open(p, mode="a"):
        return _NoKeysGroup(real_open(p, mode=mode))

    monkeypatch.setattr(mod.zarr, "open", fake_open)
    refresh_session_egocap_zarr_aggregates(tmp_path)

    store2 = zarr.open(str(tmp_path), mode="r")
    assert store2.attrs["num_episodes"] == 1
