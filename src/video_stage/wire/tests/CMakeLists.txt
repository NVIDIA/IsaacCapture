# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_executable(video_stage_wire_tests
    test_avs1.cpp
)

target_link_libraries(video_stage_wire_tests PRIVATE video_stage::wire Catch2::Catch2WithMain)

catch_discover_tests(video_stage_wire_tests)
