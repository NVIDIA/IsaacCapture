# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for resolved teleoperation runtime profiles."""

from types import SimpleNamespace

import pytest
from isaaccapture_examples.teleop_ros2 import session_config
from isaaccapture_examples.teleop_ros2.constants import (
    LEFT_WUJI_HAND_JOINT_NAMES,
    RIGHT_WUJI_HAND_JOINT_NAMES,
    TELEOP_MODES,
    WUJI_HAND_JOINT_COUNT,
    HandRetargeter,
    HandTrackingProvider,
    TeleopMode,
    resolve_hand_retargeter,
)
from isaaccapture.teleop_session_manager import SessionMode
from isaaccapture_examples.teleop_ros2.teleop_profiles import (
    TELEOP_PROFILE_SPECS,
    PublishType,
    TeleopProfile,
    TeleopProfileSpec,
    resolve_teleop_profile_spec,
    validate_session_result,
)


def _frame_for(profile_spec: TeleopProfileSpec) -> dict:
    return {key: object() for key in profile_spec.required_result_keys}


def test_every_profile_has_a_spec() -> None:
    assert tuple(mode.value for mode in TeleopMode) == TELEOP_MODES
    assert set(TELEOP_PROFILE_SPECS) == set(TeleopProfile)
    assert {spec.mode for spec in TELEOP_PROFILE_SPECS.values()} == set(TeleopMode)


@pytest.mark.parametrize(
    ("mode", "builder_name"),
    (
        (TeleopMode.CONTROLLER_TELEOP, "build_controller_teleop_config"),
        (TeleopMode.HAND_TELEOP, "build_hand_teleop_config"),
        (TeleopMode.CONTROLLER_RAW, "build_controller_raw_config"),
        (TeleopMode.FULL_BODY, "build_full_body_config"),
    ),
)
def test_session_config_dispatches_each_mode(
    monkeypatch, mode: TeleopMode, builder_name: str
) -> None:
    expected_config = object()
    monkeypatch.setattr(
        session_config,
        builder_name,
        lambda _params: expected_config,
    )

    assert (
        session_config.build_session_config(SimpleNamespace(mode=mode))
        is expected_config
    )


def test_controller_profile_spec_is_resolved_for_selected_retargeter() -> None:
    controller_spec = resolve_teleop_profile_spec(
        TeleopMode.CONTROLLER_TELEOP, HandRetargeter.TRIHAND
    )
    hands_spec = resolve_teleop_profile_spec(
        TeleopMode.CONTROLLER_TELEOP, HandRetargeter.DEXPILOT
    )
    wuji_spec = resolve_teleop_profile_spec(
        TeleopMode.CONTROLLER_TELEOP, HandRetargeter.WUJI
    )
    manus_spec = resolve_teleop_profile_spec(
        TeleopMode.CONTROLLER_TELEOP,
        HandRetargeter.WUJI,
        HandTrackingProvider.MANUS,
    )

    assert "hand_left" not in controller_spec.required_result_keys
    assert "hand_right" not in controller_spec.required_result_keys
    assert PublishType.HAND_POSES not in controller_spec.publish_types

    assert {"hand_left", "hand_right"} <= hands_spec.required_result_keys
    assert PublishType.HAND_POSES in hands_spec.publish_types
    assert PublishType.EE_FROM_CONTROLLERS in hands_spec.publish_types
    assert {"hand_left", "hand_right"} <= wuji_spec.required_result_keys
    assert PublishType.EE_FROM_CONTROLLERS in wuji_spec.publish_types
    assert PublishType.EE_FROM_HANDS not in wuji_spec.publish_types
    assert PublishType.EE_FROM_CONTROLLERS in manus_spec.publish_types
    assert PublishType.EE_FROM_HANDS not in manus_spec.publish_types
    assert manus_spec.apply_manus_controller_mount_offset
    assert (
        manus_spec
        is TELEOP_PROFILE_SPECS[TeleopProfile.CONTROLLER_TELEOP_WITH_HAND_MANUS_EE]
    )


@pytest.mark.parametrize(
    ("retargeter", "expected_profile"),
    (
        (HandRetargeter.TRIHAND, TeleopProfile.CONTROLLER_TELEOP),
        (
            HandRetargeter.DEXPILOT,
            TeleopProfile.CONTROLLER_TELEOP_WITH_HAND_CONTROLLER_EE,
        ),
        (
            HandRetargeter.PINK_IK,
            TeleopProfile.CONTROLLER_TELEOP_WITH_HAND_CONTROLLER_EE,
        ),
        (
            HandRetargeter.WUJI,
            TeleopProfile.CONTROLLER_TELEOP_WITH_HAND_CONTROLLER_EE,
        ),
    ),
)
def test_controller_profile_spec_resolution(
    retargeter: HandRetargeter, expected_profile: TeleopProfile
) -> None:
    profile_spec = resolve_teleop_profile_spec(TeleopMode.CONTROLLER_TELEOP, retargeter)
    assert profile_spec is TELEOP_PROFILE_SPECS[expected_profile]


@pytest.mark.parametrize(
    "retargeter",
    (HandRetargeter.DEXPILOT, HandRetargeter.PINK_IK, HandRetargeter.WUJI),
)
def test_controller_wuji_provider_uses_provider_wrist_for_every_retargeter(
    retargeter: HandRetargeter,
) -> None:
    profile_spec = resolve_teleop_profile_spec(
        TeleopMode.CONTROLLER_TELEOP,
        retargeter,
        HandTrackingProvider.WUJI,
    )

    assert (
        profile_spec
        is TELEOP_PROFILE_SPECS[TeleopProfile.CONTROLLER_TELEOP_WITH_HAND_WRIST_EE]
    )


@pytest.mark.parametrize(
    "retargeter",
    (HandRetargeter.DEXPILOT, HandRetargeter.PINK_IK, HandRetargeter.WUJI),
)
def test_controller_manus_provider_uses_known_good_transform_for_every_retargeter(
    retargeter: HandRetargeter,
) -> None:
    profile_spec = resolve_teleop_profile_spec(
        TeleopMode.CONTROLLER_TELEOP,
        retargeter,
        HandTrackingProvider.MANUS,
    )

    assert (
        profile_spec
        is TELEOP_PROFILE_SPECS[TeleopProfile.CONTROLLER_TELEOP_WITH_HAND_MANUS_EE]
    )
    assert profile_spec.apply_manus_controller_mount_offset


@pytest.mark.parametrize("profile", list(TeleopProfile))
def test_valid_session_result_is_accepted(profile: TeleopProfile) -> None:
    profile_spec = TELEOP_PROFILE_SPECS[profile]
    result = _frame_for(profile_spec)

    assert validate_session_result(result, profile_spec) is result


def test_session_result_reports_missing_and_unexpected_keys() -> None:
    profile_spec = TELEOP_PROFILE_SPECS[TeleopProfile.CONTROLLER_RAW]
    result = _frame_for(profile_spec)
    result.pop("controller_left")
    result["head"] = object()

    with pytest.raises(
        ValueError,
        match=r"missing keys: \['controller_left'\].*unexpected keys: \['head'\]",
    ):
        validate_session_result(result, profile_spec)


def test_mode_default_retargeters_are_resolved_centrally() -> None:
    assert (
        resolve_hand_retargeter(
            TeleopMode.CONTROLLER_TELEOP, HandRetargeter.MODE_DEFAULT
        )
        == HandRetargeter.TRIHAND
    )
    assert (
        resolve_hand_retargeter(TeleopMode.HAND_TELEOP, HandRetargeter.MODE_DEFAULT)
        == HandRetargeter.DEXPILOT
    )


def test_trihand_is_rejected_for_hand_teleop() -> None:
    with pytest.raises(ValueError, match="only valid with mode:=controller_teleop"):
        resolve_hand_retargeter(TeleopMode.HAND_TELEOP, HandRetargeter.TRIHAND)


@pytest.mark.parametrize(
    "provider",
    list(HandTrackingProvider),
)
def test_hand_teleop_always_uses_hand_wrist(
    provider: HandTrackingProvider,
) -> None:
    profile_spec = resolve_teleop_profile_spec(
        TeleopMode.HAND_TELEOP,
        HandRetargeter.DEXPILOT,
        provider,
    )

    assert profile_spec is TELEOP_PROFILE_SPECS[TeleopProfile.HAND_TELEOP]
    assert not profile_spec.apply_manus_controller_mount_offset


def test_managed_plugin_config_is_inferred_from_provider(tmp_path) -> None:
    manus_params = SimpleNamespace(
        hand_tracking_provider=HandTrackingProvider.MANUS,
        use_external_hand_tracking_plugin=False,
        session_mode=SessionMode.LIVE,
        plugin_search_paths=(tmp_path,),
    )
    wuji_params = SimpleNamespace(
        hand_tracking_provider=HandTrackingProvider.WUJI,
        use_external_hand_tracking_plugin=False,
        session_mode=SessionMode.LIVE,
        plugin_search_paths=(tmp_path,),
    )

    manus_config = session_config._resolve_hand_tracking_plugin_configs(manus_params)[0]
    wuji_config = session_config._resolve_hand_tracking_plugin_configs(wuji_params)[0]

    assert manus_config.plugin_name == "manus_hand_plugin"
    assert manus_config.plugin_args == ["--datasets=human"]
    assert wuji_config.plugin_name == "wuji_glove_plugin"
    assert wuji_config.plugin_args == []


def test_plugin_config_is_empty_for_external_provider_or_replay(tmp_path) -> None:
    external_params = SimpleNamespace(
        hand_tracking_provider=HandTrackingProvider.MANUS,
        use_external_hand_tracking_plugin=True,
        session_mode=SessionMode.LIVE,
        plugin_search_paths=(tmp_path,),
    )
    replay_params = SimpleNamespace(
        hand_tracking_provider=HandTrackingProvider.WUJI,
        use_external_hand_tracking_plugin=False,
        session_mode=SessionMode.REPLAY,
        plugin_search_paths=(tmp_path,),
    )

    assert session_config._resolve_hand_tracking_plugin_configs(external_params) == []
    assert session_config._resolve_hand_tracking_plugin_configs(replay_params) == []


def test_joint_alias_count_validation() -> None:
    session_config._validate_joint_name_alias_count("left_finger_joint_names", None, 2)
    session_config._validate_joint_name_alias_count(
        "left_finger_joint_names", ["a", "b"], 2
    )

    with pytest.raises(ValueError, match="must contain exactly 2"):
        session_config._validate_joint_name_alias_count(
            "left_finger_joint_names", ["a"], 2
        )


def test_wuji_default_joint_names_are_side_qualified_and_unique() -> None:
    assert WUJI_HAND_JOINT_COUNT == 20
    assert len(LEFT_WUJI_HAND_JOINT_NAMES) == WUJI_HAND_JOINT_COUNT
    assert len(RIGHT_WUJI_HAND_JOINT_NAMES) == WUJI_HAND_JOINT_COUNT
    names = LEFT_WUJI_HAND_JOINT_NAMES + RIGHT_WUJI_HAND_JOINT_NAMES
    assert len(set(names)) == 2 * WUJI_HAND_JOINT_COUNT
    assert names[0] == "left_thumb_j0"
    assert names[-1] == "right_pinky_j3"


def test_wuji_joint_alias_count_validation() -> None:
    session_config._validate_joint_name_alias_count(
        "left_finger_joint_names",
        [f"left_joint_{index}" for index in range(WUJI_HAND_JOINT_COUNT)],
        WUJI_HAND_JOINT_COUNT,
    )

    with pytest.raises(ValueError, match="must contain exactly 20"):
        session_config._validate_joint_name_alias_count(
            "left_finger_joint_names",
            ["left_joint"] * (WUJI_HAND_JOINT_COUNT - 1),
            WUJI_HAND_JOINT_COUNT,
        )
