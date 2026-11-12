# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.20)

add_executable(timesync_capture_qualifier_tests test_capture_qualifier.cpp)

target_link_libraries(timesync_capture_qualifier_tests PRIVATE timesync::timesync)

target_compile_options(timesync_capture_qualifier_tests PRIVATE -Wall -Wextra)

add_test(NAME timesync_capture_qualifier COMMAND timesync_capture_qualifier_tests)
