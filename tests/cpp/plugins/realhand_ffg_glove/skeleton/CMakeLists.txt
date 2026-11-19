# SPDX-FileCopyrightText: Copyright (c) 2026 RealHand. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_executable(realhand_ffg_skeleton_tests
    test_skeleton.cpp
    ${CMAKE_SOURCE_DIR}/src/plugins/realhand_ffg_glove/realhand_ffg_glove_skeleton.cpp
)
target_include_directories(realhand_ffg_skeleton_tests PRIVATE ${CMAKE_SOURCE_DIR}/src/plugins)
target_link_libraries(realhand_ffg_skeleton_tests PRIVATE Catch2::Catch2WithMain realhand_ffg_test_support)
catch_discover_tests(realhand_ffg_skeleton_tests TEST_PREFIX "realhand_ffg_skeleton_")
