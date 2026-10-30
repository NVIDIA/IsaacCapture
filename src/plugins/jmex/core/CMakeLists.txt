# SPDX-FileCopyrightText: Copyright (c) 2026 j-mex. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(jmex_plugin_core STATIC
    openxr_convention.cpp
    body_joint_mapping.cpp
    full_body_publisher.cpp
    device_clock.cpp
    plugin_args.cpp
    inc/jmex/openxr_convention.hpp
    inc/jmex/plugin_args.hpp
    inc/jmex/receiver_config.hpp
    inc/jmex/full_body_frame.hpp
    inc/jmex/body_joint_mapping.hpp
    inc/jmex/full_body_publisher.hpp
    inc/jmex/device_clock.hpp
)

target_include_directories(jmex_plugin_core
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/inc>
)

# Public headers include the generated full-body schema, so isaacteleop_schema propagates. Nothing
# else does: no public header includes OpenXR or the pusher -- the transport the publisher owns is
# behind this library's own headers.
target_link_libraries(jmex_plugin_core
    PUBLIC
        isaacteleop_schema
    PRIVATE
        pusherio::pusherio
        oxr::oxr_core
        oxr::oxr_utils
        log_bridge::log_bridge_core
)

set_target_properties(jmex_plugin_core PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN YES
)

add_library(isaac::jmex_plugin_core ALIAS jmex_plugin_core)
