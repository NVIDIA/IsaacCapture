# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_executable(pose_provider_tests
    test_stub_contract.cpp
)

target_link_libraries(pose_provider_tests
    PRIVATE
        Catch2::Catch2WithMain
        pose_provider::pose_engine_stub
)

catch_discover_tests(pose_provider_tests ADD_TAGS_AS_LABELS)
