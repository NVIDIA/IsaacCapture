# SPDX-FileCopyrightText: Copyright (c) 2026 RealHand. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_executable(realhand_ffg_options_tests test_options.cpp)
target_include_directories(realhand_ffg_options_tests PRIVATE ${CMAKE_SOURCE_DIR}/src/plugins)
target_link_libraries(realhand_ffg_options_tests PRIVATE Catch2::Catch2WithMain Teleop::plugin_utils)
catch_discover_tests(realhand_ffg_options_tests TEST_PREFIX "realhand_ffg_options_")
