# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_executable(video_stage_sender_tests
    test_direct_sender.cpp
    test_direct_sender_target.cpp
)

target_link_libraries(video_stage_sender_tests PRIVATE video_stage::sender Catch2::Catch2WithMain)

catch_discover_tests(video_stage_sender_tests)
