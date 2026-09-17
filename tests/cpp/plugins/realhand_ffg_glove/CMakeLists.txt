# SPDX-FileCopyrightText: Copyright (c) 2026 RealHand. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_executable(test_realhand_ffg_glove_protocol
    test_realhand_ffg_glove_protocol.cpp
    ${CMAKE_SOURCE_DIR}/src/plugins/realhand_ffg_glove/realhand_ffg_glove_protocol.cpp
)
target_include_directories(test_realhand_ffg_glove_protocol PRIVATE
    ${CMAKE_SOURCE_DIR}/src/plugins/realhand_ffg_glove
)
add_test(NAME realhand_ffg_glove_protocol COMMAND test_realhand_ffg_glove_protocol)
