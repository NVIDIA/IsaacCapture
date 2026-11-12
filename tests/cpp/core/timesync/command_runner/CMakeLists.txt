# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.20)

find_package(Threads REQUIRED)

add_executable(timesync_command_runner_tests test_command_runner.cpp)

target_link_libraries(timesync_command_runner_tests PRIVATE timesync::timesync Threads::Threads)

target_compile_options(timesync_command_runner_tests PRIVATE -Wall -Wextra)

add_test(NAME timesync_command_runner COMMAND timesync_command_runner_tests)
