# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Compile the real tracker against synthetic dependencies even without the vendor SDK.
set(_tracker_source "${PROJECT_SOURCE_DIR}/src/plugins/sharpa_avatar/core")
set(_double_include "${CMAKE_CURRENT_BINARY_DIR}/doubles")
foreach(_header IN ITEMS
    avatar_sdk/AvatarSDK.h
    deviceio_session/deviceio_session.hpp
    deviceio_trackers/haptic_command_reader_tracker.hpp
    openxr/openxr_platform.h
    oxr/oxr_session.hpp
    oxr_utils/oxr_time.hpp
    oxr_utils/math.hpp
    oxr_utils/os_time.hpp
    oxr_utils/pose_conversions.hpp
    plugin_utils/hand_injector.hpp
    plugin_utils/plugin_device_status_publisher.hpp
    plugin_utils/wrist_pose_source.hpp
    pusherio/schema_pusher.hpp
    flatbuffers/flatbuffers.h
    log_bridge/logger.hpp
    schema/joint_state_generated.h
)
    file(WRITE "${_double_include}/${_header}"
        "#pragma once\n#include \"${CMAKE_CURRENT_SOURCE_DIR}/sdk_doubles.hpp\"\n")
endforeach()

add_executable(sharpa_avatar_tracker_tests
    test_avatar_tracker.cpp
    "${PROJECT_SOURCE_DIR}/tests/cpp/plugins/sharpa_avatar/test_avatar_stream_health.cpp"
    "${_tracker_source}/avatar_hand_tracking_plugin.cpp"
    "${_tracker_source}/avatar_health.cpp"
)

target_compile_definitions(sharpa_avatar_tracker_tests PRIVATE AVATAR_SDK_CONFIG_PATH="synthetic")
target_include_directories(sharpa_avatar_tracker_tests PRIVATE "${_double_include}" "${_tracker_source}/inc")
target_link_libraries(sharpa_avatar_tracker_tests PRIVATE Catch2::Catch2WithMain)
catch_discover_tests(sharpa_avatar_tracker_tests TEST_PREFIX "sharpa_avatar_tracker_")
