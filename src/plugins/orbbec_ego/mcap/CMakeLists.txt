# SPDX-FileCopyrightText: Copyright (c) 2026 ORBBEC INC. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(ego_mcap_support STATIC mcap_impl.cpp)
add_library(ego::mcap_support ALIAS ego_mcap_support)
target_link_libraries(ego_mcap_support PUBLIC mcap::mcap_core isaacteleop_schema)

target_include_directories(ego_mcap_support INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/inc>)
