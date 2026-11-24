# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.20)

add_executable(log_bridge_tests
    test_logger.cpp
)

target_link_libraries(log_bridge_tests PRIVATE
    log_bridge::log_bridge_core
    Catch2::Catch2WithMain
)

if(NOT WIN32)
    # The out-of-process half of this module's coverage; the pytest leaf at
    # tests/python/core/logging_config runs it.
    add_subdirectory(emitter)
endif()

message(STATUS "log_bridge_tests target enabled with Catch2")
# Creating any logger builds this process's own sinks: keep their file in the
# build tree, and off a forwarding socket an interactive shell may be exporting.
catch_discover_tests(log_bridge_tests ADD_TAGS_AS_LABELS
    PROPERTIES ENVIRONMENT "ISAACCAPTURE_LOG_DIR=${CMAKE_CURRENT_BINARY_DIR}/logs;ISAACCAPTURE_LOG_SOCKET="
)
