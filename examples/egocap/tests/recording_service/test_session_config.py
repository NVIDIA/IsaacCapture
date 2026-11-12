# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""The session config.yaml's source declarations and the digest's treatment of a numeric frame type."""

import asyncio
import dataclasses
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import yaml
import zarr

import recording_service.recording_service as impl
from egocap.recording.alignment import align_episode, source_rules
from recording_service import (
    CaptureQcConfig,
    ConfigurationError,
    Disposition,
    FieldSpec,
    NumericSample,
    NumericSourceSpec,
    PixelFormat,
    QcConfig,
    RecordingService,
    RoutingConfig,
    ServiceConfig,
    StateError,
    VideoSample,
    VideoSourceSpec,
)

FIELDS = (FieldSpec("v", "float64"),)
DEFAULTS = {"video_frame_type": "RGBCameraFrame", "numeric_frame_type": "NumericFrame"}


def _config(output_root: Path, **overrides) -> ServiceConfig:
    base = ServiceConfig(
        output_root=output_root,
        numeric_sources=(
            NumericSourceSpec("tracker", FIELDS, "PicoTrackingFrame"),
            NumericSourceSpec("plain", FIELDS),
        ),
        video_sources=(VideoSourceSpec("cam", 64, 48, 30.0), VideoSourceSpec("wrist", 64, 48, 30.0)),
        qc=QcConfig(required_sources=("tracker",)),
    )
    return dataclasses.replace(base, **overrides)


async def _written(config: ServiceConfig, own) -> dict:
    """Open a new session and return its config.yaml as the standard loader reads it."""
    session = await own(RecordingService(config)).open_session()
    return yaml.safe_load((session.session_path / "config.yaml").read_text())


async def test_declares_every_source_with_the_defaults_and_only_the_known_frame_types(tmp_path, own):
    written = await _written(_config(tmp_path / "out"), own)
    assert written == {
        "recording": DEFAULTS,
        "video_sources": [{"name": "cam"}, {"name": "wrist"}],
        "numeric_sources": [{"name": "tracker", "frame_type": "PicoTrackingFrame"}, {"name": "plain"}],
    }


async def test_station_meta_is_written_only_with_a_label(tmp_path, own):
    labelled = await _written(_config(tmp_path / "a", station_label="Station 01"), own)
    assert labelled["meta"] == {"egocap_station_name": "Station 01"} and labelled["recording"] == DEFAULTS
    assert "meta" not in await _written(_config(tmp_path / "b"), own)
    assert "meta" not in await _written(_config(tmp_path / "c", station_label=""), own)


async def test_latency_is_written_only_for_sources_that_have_one_and_never_as_zero(tmp_path, own):
    config = _config(tmp_path / "out", latency={"tracker": 0.25, "cam": 0.0})
    written = await _written(config, own)
    assert written["numeric_sources"] == [
        {"name": "tracker", "frame_type": "PicoTrackingFrame", "latency": 0.25},
        {"name": "plain"},
    ]
    assert written["video_sources"] == [{"name": "cam", "latency": 0.0}, {"name": "wrist"}]  # explicit 0 is kept


async def test_no_latency_configured_writes_none(tmp_path, own):
    written = await _written(_config(tmp_path / "out", latency=None), own)
    assert not any("latency" in e for e in written["numeric_sources"] + written["video_sources"])


async def test_a_numeric_only_session_declares_an_empty_video_list(tmp_path, own):
    config = _config(tmp_path / "out", video_sources=())
    assert (await _written(config, own))["video_sources"] == []


async def test_reopening_a_session_keeps_its_config_yaml_and_still_works(tmp_path, own):
    config = _config(tmp_path / "out", latency={"tracker": 0.1})
    service = own(RecordingService(config))
    session = await service.open_session()
    path = session.session_path / "config.yaml"
    first = path.read_text()
    await service.close()

    changed = dataclasses.replace(config, latency={"tracker": 0.9}, station_label="Station 07")
    again = own(RecordingService(changed))
    reopened = await again.open_session(session.session_id)
    assert reopened.session_path == session.session_path and path.read_text() == first


async def test_a_reopened_session_records_its_next_take_under_the_frozen_config_yaml(tmp_path, own):
    config = _config(tmp_path / "out", video_sources=())
    service = own(RecordingService(config))
    session = await service.open_session()
    first = await service.start_episode("t")
    service.submit_sample(first.episode_id, NumericSample("tracker", 0.0, 0.0, {"v": 1.0}))
    await service.stop_episode(first.episode_id)
    await service.close()

    again = own(RecordingService(config))
    await again.open_session(session.session_id)
    assert not again.pending_episodes()
    second = await again.start_episode("t")
    assert second.episode_index == 1 and (session.session_path / "config.yaml").is_file()
    await again.discard_episode(second.episode_id)


async def test_a_routed_group_copy_carries_the_declarations(tmp_path, own):
    config = _config(
        tmp_path / "out",
        routing=RoutingConfig(),
        video_sources=(),
        numeric_sources=(NumericSourceSpec("tracker", FIELDS, "PicoTrackingFrame"),),
        latency={"tracker": 0.5},
        station_label=None,
    )
    service = own(RecordingService(config))
    session = await service.open_session()
    episode = await service.start_episode("t")
    for i in range(3):
        service.submit_sample(episode.episode_id, NumericSample("tracker", float(i), float(i), {"v": 1.0}))
    result = await service.stop_episode(episode.episode_id)
    assert result.disposition is Disposition.KEPT
    copied = yaml.safe_load((Path(result.final_path).parent / "config.yaml").read_text())
    assert copied == yaml.safe_load((session.session_path / "config.yaml").read_text())
    assert copied["numeric_sources"] == [{"name": "tracker", "frame_type": "PicoTrackingFrame", "latency": 0.5}]


def _pre_frame_type_digest(config: ServiceConfig) -> str:
    qc = dataclasses.asdict(config.qc)
    if config.qc.capture is None:
        del qc["capture"]
    else:
        qc["capture_algorithm_version"] = impl.ALGORITHM_VERSION
    canonical = {
        **({"routing": True} if config.routing is not None else {}),
        "numeric": [
            {"name": s.name, "fields": [dataclasses.asdict(f) for f in s.fields]} for s in config.numeric_sources
        ],
        "video": [dataclasses.asdict(s) for s in config.video_sources],
        "qc": qc,
        "latency": config.latency,
        "calibration": [a.name for a in config.calibration_assets],
    }
    return hashlib.sha256(json.dumps(canonical, sort_keys=True).encode()).hexdigest()


def test_a_take_recorded_before_frame_types_existed_keeps_its_digest():
    for extra in (
        {},
        {"routing": RoutingConfig()},
        {"qc": QcConfig(capture=CaptureQcConfig())},
        {"latency": {"a": 1.0}},
    ):
        config = ServiceConfig(
            Path("out"),
            (NumericSourceSpec("a", FIELDS),),
            (VideoSourceSpec("cam", 64, 48, 30.0),),
            **extra,
        )
        assert impl._digest(config) == _pre_frame_type_digest(config)
        assert impl._layout_digest(config) == _pre_frame_type_digest(
            dataclasses.replace(config, qc=dataclasses.replace(config.qc, capture=None))
        )


def test_the_digest_is_the_same_with_a_frame_type_unset_set_or_changed():
    def digest(frame_type):
        return impl._digest(ServiceConfig(Path("out"), (NumericSourceSpec("a", FIELDS, frame_type),)))

    assert digest(None) == digest("PicoTrackingFrame") == digest("ManusFrame")


def test_a_frame_type_does_not_change_the_layout_digest_but_a_field_does():
    def layout(fields, frame_type):
        return impl._layout_digest(ServiceConfig(Path("out"), (NumericSourceSpec("a", fields, frame_type),)))

    other = (FieldSpec("extra", "float64"),)
    assert layout(FIELDS, None) == layout(FIELDS, "ManusFrame")
    assert layout(FIELDS, None) != layout(other, None)


def test_other_source_fields_still_change_the_digest():
    def digest(name, fields):
        return impl._digest(ServiceConfig(Path("out"), (NumericSourceSpec(name, fields, "ManusFrame"),)))

    assert digest("a", FIELDS) != digest("b", FIELDS)
    assert digest("a", FIELDS) != digest("a", (FieldSpec("extra", "float64"),))


async def test_a_recorded_episode_aligns_from_its_own_config_yaml_and_arrays(tmp_path, own):
    config = ServiceConfig(
        output_root=tmp_path / "out",
        numeric_sources=(
            NumericSourceSpec(
                "pico_controller",
                (FieldSpec("head_pose", "float64", (4, 4)), FieldSpec("left_trigger", "float64")),
                "PicoTrackingFrame",
            ),
            NumericSourceSpec("manus_glove", (FieldSpec("left_hand_pose", "float64", (25, 4, 4)),), "ManusFrame"),
        ),
        video_sources=(VideoSourceSpec("cam", 64, 48, 30.0),),
        qc=QcConfig(required_sources=("pico_controller", "cam")),
    )
    service = own(RecordingService(config))
    session = await service.open_session()
    episode = await service.start_episode("t")
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    items = []
    for i in range(6):  # six camera frames at 30 Hz, bracketed by 60 Hz tracking samples
        t = 1.0 + i / 30
        items.append(VideoSample("cam", frame, PixelFormat.RGB8, t, t, np.eye(3)))
    for i in range(-2, 16):
        t = 1.0 + i / 60
        head = np.eye(4)
        head[0, 3] = t
        items.append(NumericSample("pico_controller", t, t, {"head_pose": head, "left_trigger": t}))
        items.append(NumericSample("manus_glove", t, t, {"left_hand_pose": np.full((25, 4, 4), t)}))
    for item in sorted(items, key=lambda x: x.capture_time):
        while service.submit_sample(episode.episode_id, item).status.name == "FULL":
            await asyncio.sleep(0)
    result = await service.stop_episode(episode.episode_id)
    assert result.disposition is Disposition.KEPT

    declared = yaml.safe_load((session.session_path / "config.yaml").read_text())
    root = zarr.open_group(str(result.final_path), mode="r")
    times = {name: root[name]["capture_time"][:] for name in ("cam", "pico_controller", "manus_glove")}
    fields = {
        "pico_controller": {k: root["pico_controller"][k][:] for k in ("head_pose", "left_trigger")},
        "manus_glove": {"left_hand_pose": root["manus_glove"]["left_hand_pose"][:]},
    }
    rules = source_rules(declared)
    aligned = align_episode("cam", times, ["cam"], fields, rules)

    count = int(aligned.valid_indices.sum())
    assert aligned.valid_indices.dtype == bool and aligned.valid_indices.shape == (6,) and count == 6
    assert aligned.sources["cam"].aligned_frame_indices.tolist() == list(range(6))
    pico = aligned.sources["pico_controller"].interpolated
    assert pico["head_pose"].shape == (count, 4, 4) and pico["left_trigger"].shape == (count,)
    assert np.allclose(pico["head_pose"][:, 0, 3], aligned.valid_timestamps)  # poses follow the linear ramp
    glove = aligned.sources["manus_glove"].interpolated["left_hand_pose"]
    assert glove.shape == (count, 25, 4, 4) and np.allclose(glove[:, 0, 0, 0], aligned.valid_timestamps)


async def _frozen_session(tmp_path, own, config=None):
    config = config or _config(tmp_path / "out", video_sources=())
    service = own(RecordingService(config))
    session = await service.open_session()
    await service.close()
    return config, session


def _with_frame_types(config, **types):
    sources = tuple(
        dataclasses.replace(s, frame_type=types.get(s.name, s.frame_type)) for s in config.numeric_sources
    )
    return dataclasses.replace(config, numeric_sources=sources)


async def test_reopening_with_the_same_or_default_equivalent_frame_types_succeeds(tmp_path, own):
    config, session = await _frozen_session(tmp_path, own)
    path = session.session_path / "config.yaml"
    before = path.read_bytes()
    reordered = dataclasses.replace(config, numeric_sources=config.numeric_sources[::-1])
    equivalent = _with_frame_types(config, plain="NumericFrame")  # the declared default, spelled out
    for candidate in (reordered, equivalent):
        again = own(RecordingService(candidate))
        await again.open_session(session.session_id)
        await again.close()
    assert path.read_bytes() == before


async def test_reopening_with_a_changed_frame_type_is_refused_and_leaves_the_session_alone(tmp_path, own):
    config, session = await _frozen_session(tmp_path, own)
    path = session.session_path / "config.yaml"
    before = path.read_bytes()
    for changed in (
        _with_frame_types(config, tracker="NumericFrame"),  # specialised -> default
        _with_frame_types(config, plain="PicoTrackingFrame"),  # default -> specialised
    ):
        again = own(RecordingService(changed))
        with pytest.raises(ConfigurationError, match="open a new session"):
            await again.open_session(session.session_id)
        assert path.read_bytes() == before
    # The refusal released the session lock: the unchanged configuration still opens it.
    ok = own(RecordingService(config))
    await ok.open_session(session.session_id)


async def test_a_session_without_a_config_yaml_reopens_but_cannot_start_a_take(tmp_path, own):
    config, session = await _frozen_session(tmp_path, own)
    (session.session_path / "config.yaml").unlink()
    again = own(RecordingService(config))
    await again.open_session(session.session_id)
    assert not again.pending_episodes()
    with pytest.raises(StateError, match="no frozen source declarations"):
        await again.start_episode("t")
    assert not list(session.session_path.glob("episode_*"))


@pytest.mark.parametrize(
    "text",
    [
        "numeric_sources: [1]\n",
        "[1, 2]\n",
        "",
        "recording: {}\n",
        "numeric_sources: [{name: tracker}, {name: tracker}]\n",
    ],
)
async def test_a_malformed_config_yaml_refuses_reopen_clearly(tmp_path, own, text):
    config, session = await _frozen_session(tmp_path, own)
    (session.session_path / "config.yaml").write_text(text)
    with pytest.raises(ConfigurationError, match="no readable source declarations"):
        await own(RecordingService(config)).open_session(session.session_id)


async def test_reopening_with_a_source_the_frozen_config_does_not_declare_is_refused(tmp_path, own):
    config, session = await _frozen_session(tmp_path, own)
    (session.session_path / "config.yaml").write_text(
        "numeric_sources: [{name: tracker, frame_type: PicoTrackingFrame}]\n"
    )
    with pytest.raises(ConfigurationError, match="'plain' is not declared"):
        await own(RecordingService(config)).open_session(session.session_id)
