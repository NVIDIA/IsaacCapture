# SPDX-FileCopyrightText: Copyright (c) 2026 ORBBEC INC. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(ego_mcap_merge_core STATIC merge.cpp)
add_library(ego::mcap_merge ALIAS ego_mcap_merge_core)
target_include_directories(ego_mcap_merge_core INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/inc>)
target_link_libraries(ego_mcap_merge_core PUBLIC ego::mcap_support)
