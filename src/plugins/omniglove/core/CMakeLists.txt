# SPDX-FileCopyrightText: Copyright (c) 2026 Agibot Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

add_library(omniglove_plugin_core STATIC
    omniglove_hand_conversion.cpp
    omniglove_plugin.cpp
    inc/omniglove_plugin/omniglove_hand_conversion.hpp
    inc/omniglove_plugin/omniglove_plugin.hpp
)

target_include_directories(omniglove_plugin_core
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/inc>
)

target_link_libraries(omniglove_plugin_core
    PRIVATE
        OmnigloveSDK::api
    PUBLIC
        Teleop::plugin_utils
        deviceio::deviceio_session
        deviceio::deviceio_trackers
        oxr::oxr_core
        oxr::oxr_utils
        OpenXR::openxr_loader
        Teleop::openxr_extensions
        log_bridge::log_bridge_core
)

add_library(isaac::omniglove_plugin ALIAS omniglove_plugin_core)
