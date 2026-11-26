# SPDX-FileCopyrightText: Copyright (c) 2026 ORBBEC INC. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Exercise the private liveness policy without exporting a test-only public API.
set(_ego_whitebox_dir "${CMAKE_CURRENT_BINARY_DIR}/whitebox")
file(MAKE_DIRECTORY "${_ego_whitebox_dir}")
configure_file("${PROJECT_SOURCE_DIR}/src/plugins/orbbec_ego/core/frame_liveness.hpp"
    "${_ego_whitebox_dir}/frame_liveness.hpp" COPYONLY)
configure_file(test_video_liveness.cpp "${_ego_whitebox_dir}/test_video_liveness.cpp" COPYONLY)

add_executable(ego_camera_tests
    test_ego_camera.cpp
    test_raw_calibration.cpp
    test_video_liveness_hardware.cpp
    "${_ego_whitebox_dir}/test_video_liveness.cpp"
)

target_link_libraries(ego_camera_tests
    PRIVATE
        ego::camera_core
        ego::mcap_support
        Catch2::Catch2WithMain
)

catch_discover_tests(ego_camera_tests TEST_SPEC "~[hardware]" ADD_TAGS_AS_LABELS)

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
