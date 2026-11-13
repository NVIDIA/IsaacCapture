# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.20)

add_executable(timesync_chrony_tracking_tests test_chrony_tracking.cpp)

target_link_libraries(timesync_chrony_tracking_tests PRIVATE timesync::timesync)

target_compile_options(timesync_chrony_tracking_tests PRIVATE -Wall -Wextra)

add_test(NAME timesync_chrony_tracking COMMAND timesync_chrony_tracking_tests)
