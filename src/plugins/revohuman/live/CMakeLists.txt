# SPDX-FileCopyrightText: Copyright (c) 2026 BrainCo. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Glove samples from the RevoHuman C SDK. Built from the header alone: whoever links it also links
# the SDK library (the plugin when it was found, the tests a fake of it).
add_library(revohuman_live STATIC
    live_source.cpp
    inc/revohuman_plugin/live_source.hpp
    rh_frame.cpp
    inc/revohuman_plugin/rh_frame.hpp
)

target_include_directories(revohuman_live
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/inc>
)

target_link_libraries(revohuman_live
    PUBLIC
        revohuman_plugin::revohuman_core
        revohuman_plugin::sdk_headers
        log_bridge::log_bridge_core
)

add_library(revohuman_plugin::revohuman_live ALIAS revohuman_live)
