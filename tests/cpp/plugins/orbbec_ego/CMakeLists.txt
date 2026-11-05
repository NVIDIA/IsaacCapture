# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_executable(ego_camera_tests
    test_ego_camera.cpp
    test_raw_calibration.cpp
)

target_link_libraries(ego_camera_tests
    PRIVATE
        ego::camera_core
        ego::mcap_support
        Catch2::Catch2WithMain
)

catch_discover_tests(ego_camera_tests ADD_TAGS_AS_LABELS)

add_test(
    NAME ego_script_tests
    COMMAND bash "${CMAKE_CURRENT_SOURCE_DIR}/test_ego_script.sh"
        "${CMAKE_SOURCE_DIR}/src/plugins/orbbec_ego/orbbec_ego.sh"
)

add_test(
    NAME ego_install_tests
    COMMAND bash "${CMAKE_CURRENT_SOURCE_DIR}/test_ego_install.sh"
        "${CMAKE_SOURCE_DIR}/src/plugins/orbbec_ego/install.sh"
)
