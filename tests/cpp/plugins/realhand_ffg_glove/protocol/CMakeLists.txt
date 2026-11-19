# SPDX-FileCopyrightText: Copyright (c) 2026 RealHand. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_executable(realhand_ffg_protocol_tests
    test_protocol.cpp
    ${CMAKE_SOURCE_DIR}/src/plugins/realhand_ffg_glove/realhand_ffg_glove_protocol.cpp
)
target_include_directories(realhand_ffg_protocol_tests PRIVATE ${CMAKE_SOURCE_DIR}/src/plugins)
target_link_libraries(realhand_ffg_protocol_tests PRIVATE Catch2::Catch2WithMain )
catch_discover_tests(realhand_ffg_protocol_tests TEST_PREFIX "realhand_ffg_protocol_")
