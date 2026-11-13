# SPDX-FileCopyrightText: Copyright (c) 2026 BrainCo. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_executable(revohuman_core_tests
    test_hand_model.cpp
    new_geometry_fixture.inc
    test_pipeline.cpp
)

target_link_libraries(revohuman_core_tests
    PRIVATE
        Catch2::Catch2WithMain
        revohuman_plugin::sdk
        revohuman_plugin::revohuman_core
        isaacteleop_schema
)

catch_discover_tests(revohuman_core_tests)
