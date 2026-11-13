# SPDX-FileCopyrightText: Copyright (c) 2026 BrainCo. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# The hand mapping and the per-hand push pipeline, apart from OpenXR so they are unit-tested without a runtime.
add_library(revohuman_core STATIC
    hand_model.cpp
    hand_fit.inc
    inc/revohuman_plugin/hand_model.hpp
    inc/revohuman_plugin/pipeline.hpp
    inc/revohuman_plugin/sdk.hpp
    inc/revohuman_plugin/types.hpp
)

target_include_directories(revohuman_core
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/inc>
)

target_link_libraries(revohuman_core
    PUBLIC
        isaacteleop_schema
        log_bridge::log_bridge_core
        revohuman_plugin::sdk
)

add_library(revohuman_plugin::revohuman_core ALIAS revohuman_core)
