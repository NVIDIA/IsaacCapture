# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_executable(pose_provider_tests
    fake_camera_input.hpp
    test_camera_input.cpp
    test_pose_provider.cpp
    test_stub_contract.cpp
    test_undistort.cpp
)

target_link_libraries(pose_provider_tests
    PRIVATE
        Catch2::Catch2WithMain
        pose_provider::pose_camera_input
        pose_provider::pose_engine_stub
        pose_provider::pose_provider_core
        pose_provider::pose_undistort
)

catch_discover_tests(pose_provider_tests ADD_TAGS_AS_LABELS)
