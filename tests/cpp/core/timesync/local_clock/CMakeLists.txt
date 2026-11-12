# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.20)

add_executable(timesync_local_clock_tests test_local_clock.cpp)

target_link_libraries(timesync_local_clock_tests PRIVATE timesync::timesync)

target_compile_options(timesync_local_clock_tests PRIVATE -Wall -Wextra)

add_test(NAME timesync_local_clock COMMAND timesync_local_clock_tests)
