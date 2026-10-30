# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Station config.yaml snapshot behavior: no real recorded episode needed (numeric-only, no camera/encoder)."""

import dataclasses
from pathlib import Path

import pytest
import yaml

from recording_service import (
    CalibrationAsset,
    ConfigurationError,
    Disposition,
    FieldSpec,
    NumericSample,
    NumericSourceSpec,
    QcConfig,
    RecordingService,
    RoutingConfig,
    ServiceConfig,
)

SOURCE = NumericSourceSpec("s", (FieldSpec("v", "float64"),))


def _config(output_root: Path, **overrides) -> ServiceConfig:
    base = ServiceConfig(
        output_root=output_root,
        numeric_sources=(SOURCE,),
        qc=QcConfig(required_sources=("s",)),
    )
    return dataclasses.replace(base, **overrides)


async def _kept_episode(config: ServiceConfig, own):
    """Record one minimal numeric-only take to completion and return (session, result)."""
    service = own(RecordingService(config))
    session = await service.open_session()
    episode = await service.start_episode("t")
    for i in range(3):
        service.submit_sample(episode.episode_id, NumericSample("s", float(i), float(i), {"v": 1.0}))
    result = await service.stop_episode(episode.episode_id)
    return session, result


async def test_station_label_writes_config_yaml_once_and_freezes_on_reopen(tmp_path, own):
    """A new session's config.yaml records the station once; a later station_label change never applies to it."""
    base = _config(tmp_path / "out", station_label="Station 01")
    service = own(RecordingService(base))
    session = await service.open_session()
    cfg_path = session.session_path / "config.yaml"
    assert yaml.safe_load(cfg_path.read_text()) == {"meta": {"egocap_station_name": "Station 01"}}
    await service.close()

    relabeled = dataclasses.replace(base, station_label="Station 02")
    service2 = own(RecordingService(relabeled))
    await service2.open_session(session.session_id)  # reopen the same session
    assert yaml.safe_load(cfg_path.read_text()) == {"meta": {"egocap_station_name": "Station 01"}}  # unchanged


async def test_no_station_label_writes_no_config_yaml(tmp_path, own):
    """An unset station_label (the ServiceConfig default) writes no config.yaml at all."""
    service = own(RecordingService(_config(tmp_path / "out")))
    session = await service.open_session()
    assert not (session.session_path / "config.yaml").exists()


async def test_routed_group_gets_a_copy_of_the_session_config_yaml(tmp_path, own):
    """The first take routed into a group carries the session's frozen config.yaml alongside it."""
    config = _config(tmp_path / "out", routing=RoutingConfig(), station_label="Station 01")
    session, result = await _kept_episode(config, own)
    assert result.disposition is Disposition.KEPT
    group_dir = Path(result.final_path).parent
    group_config = group_dir / "config.yaml"
    assert group_config.is_file()
    assert yaml.safe_load(group_config.read_text()) == yaml.safe_load(
        (session.session_path / "config.yaml").read_text()
    )


async def test_routed_group_gets_a_copy_of_preserved_calibration_assets(tmp_path, own):
    """The first take routed into a group also carries the session's preserved calibration files."""
    asset_source = tmp_path / "intrinsics.json"
    asset_source.write_text('{"fx": 1.0}')
    config = _config(
        tmp_path / "out",
        routing=RoutingConfig(),
        calibration_assets=(CalibrationAsset("head_camera_intrinsics.json", asset_source, required=False),),
    )
    session, result = await _kept_episode(config, own)
    assert result.disposition is Disposition.KEPT
    group_dir = Path(result.final_path).parent
    group_asset = group_dir / "head_camera_intrinsics.json"
    assert group_asset.is_file()
    assert group_asset.read_text() == (session.session_path / "head_camera_intrinsics.json").read_text()


async def test_routed_group_with_an_optional_calibration_asset_missing_everywhere_gets_no_copy(tmp_path, own):
    """A calibration asset never preserved for this session (optional and absent at creation) is never required
    at the group either -- same missing-is-OK policy, not newly enforced once routing is involved."""
    config = _config(
        tmp_path / "out",
        routing=RoutingConfig(),
        calibration_assets=(
            CalibrationAsset("head_camera_intrinsics.json", tmp_path / "missing.json", required=False),
        ),
    )
    session, result = await _kept_episode(config, own)
    assert result.disposition is Disposition.KEPT
    group_dir = Path(result.final_path).parent
    assert not (group_dir / "head_camera_intrinsics.json").exists()


async def test_seed_group_assets_rejects_a_conflicting_existing_config_yaml(tmp_path, own):
    """A group directory whose config.yaml does not match this session's own is a hard conflict, not a warning."""
    out = tmp_path / "out"
    config = _config(out, station_label="Station 01")
    service = own(RecordingService(config))
    session = await service.open_session()  # writes session's own config.yaml

    group_dir = out / "some-group"
    group_dir.mkdir()
    (group_dir / "config.yaml").write_text(yaml.safe_dump({"meta": {"egocap_station_name": "Station 99"}}))

    with pytest.raises(ConfigurationError, match="does not match this session"):
        service._seed_group_assets(session.session_path, "some-group")


async def test_seed_group_assets_rejects_a_conflicting_existing_calibration_asset(tmp_path, own):
    """A group directory whose calibration file does not match the session's preserved copy is also rejected."""
    out = tmp_path / "out"
    asset_source = tmp_path / "intrinsics.json"
    asset_source.write_text('{"fx": 1.0}')
    config = _config(
        out, calibration_assets=(CalibrationAsset("head_camera_intrinsics.json", asset_source, required=False),)
    )
    service = own(RecordingService(config))
    session = await service.open_session()  # preserves the calibration asset into the session

    group_dir = out / "some-group"
    group_dir.mkdir()
    (group_dir / "head_camera_intrinsics.json").write_text('{"fx": 9.9}')

    with pytest.raises(ConfigurationError, match="does not match this session"):
        service._seed_group_assets(session.session_path, "some-group")


async def test_a_second_take_routes_into_the_existing_group_and_does_not_touch_its_config_yaml(tmp_path, own):
    """The common path: two kept takes for the same station/task/day land in the same routed
    group. The second one must not be refused, and must not rewrite the group's config.yaml that
    the first one already seeded (scoped to config.yaml -- not a claim about calibration-asset
    idempotence, which the existing single-take tests above already cover separately)."""
    config = _config(tmp_path / "out", routing=RoutingConfig(), station_label="Station 01")
    service = own(RecordingService(config))
    await service.open_session()

    async def _kept_take(i: int):
        episode = await service.start_episode("t")
        service.submit_sample(episode.episode_id, NumericSample("s", float(i), float(i), {"v": 1.0}))
        return await service.stop_episode(episode.episode_id)

    first_result = await _kept_take(0)
    assert first_result.disposition is Disposition.KEPT
    group_dir = Path(first_result.final_path).parent
    group_config = group_dir / "config.yaml"
    seeded_at_first_take = group_config.read_text()

    second_result = await _kept_take(1)
    assert second_result.disposition is Disposition.KEPT
    assert Path(second_result.final_path).parent == group_dir  # same group, not a new one
    assert group_config.read_text() == seeded_at_first_take  # re-seeding changed nothing


async def test_seed_group_assets_accepts_a_group_config_the_eod_rename_backfill_has_since_edited(tmp_path, own):
    """The EOD Episode Rename backfill (update_episode_task_description.py) legitimately adds
    task_description/task_name/studio to a group's config.yaml meta after routing -- a later take
    into that same group must still succeed, since the station name (the only field this session
    snapshot and that check compare) is unchanged."""
    out = tmp_path / "out"
    config = _config(out, station_label="Station 01")
    service = own(RecordingService(config))
    session = await service.open_session()  # writes session's own config.yaml

    group_dir = out / "some-group"
    group_dir.mkdir()
    (group_dir / "config.yaml").write_text(
        yaml.safe_dump(
            {"meta": {"egocap_station_name": "Station 01", "task_description": "renamed by EOD backfill"}}
        )
    )

    service._seed_group_assets(session.session_path, "some-group")  # must not raise


async def test_seed_group_assets_refuses_a_preserved_asset_that_was_since_lost(tmp_path, own):
    """A calibration asset recorded as preserved at session creation, then deleted from the session
    directory by hand, is refused -- not silently treated as 'never had it', which would let the group
    seed proceed without a file the session's own calibration record says should exist."""
    out = tmp_path / "out"
    asset_source = tmp_path / "intrinsics.json"
    asset_source.write_text('{"fx": 1.0}')
    config = _config(
        out, calibration_assets=(CalibrationAsset("head_camera_intrinsics.json", asset_source, required=False),)
    )
    service = own(RecordingService(config))
    session = await service.open_session()  # preserves and records the asset's hash
    (session.session_path / "head_camera_intrinsics.json").unlink()  # lost after the fact

    with pytest.raises(ConfigurationError, match="preserved but is now missing or changed"):
        service._seed_group_assets(session.session_path, "some-group")
