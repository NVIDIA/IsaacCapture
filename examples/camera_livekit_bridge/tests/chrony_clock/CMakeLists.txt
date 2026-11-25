# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_executable(chrony_clock_tests test_chrony_capture_clock.cpp)

target_link_libraries(chrony_clock_tests
    PRIVATE
        camera_livekit_bridge::chrony_clock
        Threads::Threads
)

target_compile_options(chrony_clock_tests PRIVATE -Wall -Wextra)

add_test(NAME chrony_clock COMMAND chrony_clock_tests)
