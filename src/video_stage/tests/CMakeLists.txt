# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_executable(video_stage_tests
    test_sine_warp.cpp
)

target_link_libraries(video_stage_tests PRIVATE video_stage::core Catch2::Catch2WithMain)

catch_discover_tests(video_stage_tests)
