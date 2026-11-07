# SPDX-FileCopyrightText: Copyright (c) 2026 RealHand. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_executable(realhand_ffg_reader_tests
    test_reader.cpp
    ${CMAKE_SOURCE_DIR}/src/plugins/realhand_ffg_glove/realhand_ffg_glove_reader.cpp
    ${CMAKE_SOURCE_DIR}/src/plugins/realhand_ffg_glove/realhand_ffg_glove_serial.cpp
    ${CMAKE_SOURCE_DIR}/src/plugins/realhand_ffg_glove/realhand_ffg_glove_protocol.cpp
)
target_include_directories(realhand_ffg_reader_tests PRIVATE ${CMAKE_SOURCE_DIR}/src/plugins)
target_link_libraries(realhand_ffg_reader_tests PRIVATE Catch2::Catch2WithMain Teleop::plugin_utils)
catch_discover_tests(realhand_ffg_reader_tests TEST_PREFIX "realhand_ffg_reader_")
