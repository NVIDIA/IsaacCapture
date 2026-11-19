# SPDX-FileCopyrightText: Copyright (c) 2026 RealHand. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(realhand_ffg_test_support INTERFACE)
target_include_directories(realhand_ffg_test_support INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/inc
    ${CMAKE_SOURCE_DIR}/src/plugins
)
target_link_libraries(realhand_ffg_test_support INTERFACE Teleop::plugin_utils yaml-cpp::yaml-cpp)
