# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.20)

add_executable(plugin_manager_tests
    test_plugin_manager.cpp
)

target_link_libraries(plugin_manager_tests PRIVATE
    teleop_plugin_manager
    test_support::temporary_directory
    Catch2::Catch2WithMain
)

if(NOT WIN32)
    add_subdirectory(test_process)
    target_sources(plugin_manager_tests PRIVATE test_plugin_process.cpp)
    target_compile_definitions(plugin_manager_tests PRIVATE
        PLUGIN_MANAGER_TEST_PROCESS="$<TARGET_FILE:plugin_manager_test_process>"
    )
    add_dependencies(plugin_manager_tests plugin_manager_test_process)
endif()

set_target_properties(plugin_manager_tests PROPERTIES ISAACTELEOP_NEEDS_SOURCE_TREE TRUE)

message(STATUS "plugin_manager_tests target enabled with Catch2")
catch_discover_tests(plugin_manager_tests)
