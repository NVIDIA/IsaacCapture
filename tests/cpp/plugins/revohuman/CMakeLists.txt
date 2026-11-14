# SPDX-FileCopyrightText: Copyright (c) 2026 BrainCo. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_executable(revohuman_core_tests
    test_hand_model.cpp
    ptv1_neutral_points.inc
    test_pipeline.cpp
)

# hand_points.hpp is private to the core library.
target_include_directories(revohuman_core_tests PRIVATE
    ${CMAKE_SOURCE_DIR}/src/plugins/revohuman/core
)

target_link_libraries(revohuman_core_tests
    PRIVATE
        Catch2::Catch2WithMain
        revohuman_plugin::sdk
        revohuman_plugin::revohuman_core
        isaacteleop_schema
)

catch_discover_tests(revohuman_core_tests)
