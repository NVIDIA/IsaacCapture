# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_executable(ego_camera_tests
    test_ego_camera.cpp
    test_raw_calibration.cpp
    test_time_alignment.cpp
)

target_link_libraries(ego_camera_tests
    PRIVATE
        ego::camera_core
        ego::mcap_support
        Catch2::Catch2WithMain
)

catch_discover_tests(ego_camera_tests ADD_TAGS_AS_LABELS)
