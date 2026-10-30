# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Which configuration changes block finishing a pending take: synthetic sources, no recorded data needed.

A SEALED take is still to be judged, so the whole configuration (including the capture policy and algorithm
version) must match. A CHECKED take already carries its decision and policy, so only the layout-affecting parts
must match.
"""

from pathlib import Path

import pytest

import recording_service.recording_service as impl
from egocap.recording.episode_layout import EpisodeManifest
from egocap.recording.episode_store import FieldSpec, NumericSourceSpec, VideoSourceSpec
from recording_service import CaptureQcConfig, QcConfig, RecordingService, ServiceConfig

NUMERIC = NumericSourceSpec("tracker", (FieldSpec("pose", "float64", (3,)),))
CAMERA = VideoSourceSpec("cam", 64, 48, 30.0)


def _config(capture="default", numeric=(NUMERIC,)):
    if capture == "default":
        capture = CaptureQcConfig(camera_expected_fps=(("cam", None),), pose_streams=(("tracker", "pose"),))
    qc = QcConfig(required_sources=("tracker",), capture=capture)
    return ServiceConfig(Path("out"), numeric, (CAMERA,), qc=qc)


def _manifest(config, phase, *, legacy=False):
    """A manifest recorded under ``config``; ``legacy`` mimics one written before the layout digest existed."""
    config = RecordingService(config)._config  # the service digests its own resolved snapshot of the configuration
    return EpisodeManifest(
        session_id="s",
        episode_id="s/episode_0",
        episode_index=0,
        config_digest=impl._digest(config),
        task="t",
        operator="o",
        sources=[],
        pending_path="episode_0.pending",
        final_path="episode_0",
        storage_phase=phase,
        layout_digest=None if legacy else impl._layout_digest(config),
    )


def test_the_layout_digest_ignores_the_capture_policy_and_equals_the_legacy_digest_without_one():
    with_capture, without = _config(), _config(capture=None)
    assert impl._layout_digest(with_capture) == impl._digest(without)  # a take recorded before capture QC existed
    assert impl._digest(with_capture) != impl._digest(without)


def test_a_changed_threshold_or_algorithm_version_changes_the_full_digest_but_not_the_layout_digest(monkeypatch):
    base = _config()
    stricter = _config(capture=CaptureQcConfig(camera_expected_fps=(("cam", None),), freeze_seconds=0.5))
    assert impl._digest(base) != impl._digest(stricter)
    assert impl._layout_digest(base) == impl._layout_digest(stricter)
    before = impl._digest(base)
    monkeypatch.setattr(impl, "ALGORITHM_VERSION", impl.ALGORITHM_VERSION + 1)
    assert impl._digest(base) != before and impl._layout_digest(base) == impl._layout_digest(stricter)


def test_a_changed_source_declaration_changes_the_layout_digest():
    other = NumericSourceSpec("tracker", (FieldSpec("pose", "float64", (4,)),))
    assert impl._layout_digest(_config()) != impl._layout_digest(_config(numeric=(other,)))


@pytest.mark.parametrize(
    ("phase", "later_capture", "may_finish"),
    [
        ("SEALED", None, False),  # still to be judged: the capture policy in force must be the one recorded under
        ("SEALED", "same", True),
        ("CHECKED", None, True),  # decided: policy removed
        (
            "CHECKED",
            CaptureQcConfig(camera_expected_fps=(("cam", None),), freeze_seconds=0.5),
            True,
        ),  # stricter later
    ],
)
def test_only_a_sealed_take_needs_the_same_capture_policy_to_be_finished(phase, later_capture, may_finish):
    recorded = _config()
    later = _config(capture=recorded.qc.capture if later_capture == "same" else later_capture)
    assert RecordingService(later)._same_configuration(_manifest(recorded, phase)) is may_finish


def test_a_checked_take_is_still_blocked_by_a_changed_layout():
    other = NumericSourceSpec("tracker", (FieldSpec("pose", "float64", (4,)),))
    assert (
        RecordingService(_config(numeric=(other,)))._same_configuration(_manifest(_config(), "CHECKED")) is False
    )


def test_a_checked_take_written_before_the_layout_digest_existed_is_compared_by_its_old_digest():
    legacy = _manifest(_config(capture=None), "CHECKED", legacy=True)  # config_digest is the layout digest
    assert RecordingService(_config())._same_configuration(legacy) is True
    legacy_other = _manifest(_config(capture=None, numeric=()), "CHECKED", legacy=True)
    assert RecordingService(_config())._same_configuration(legacy_other) is False
