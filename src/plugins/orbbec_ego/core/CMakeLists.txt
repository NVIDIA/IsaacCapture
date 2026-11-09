# SPDX-FileCopyrightText: Copyright (c) 2026 ORBBEC INC. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set(ego_camera_core_sources
    ego_camera.cpp
    frame_liveness.hpp
    calibration.cpp
    calibration.hpp
    inc/ego_camera/calibration.hpp
    inc/ego_camera/raw_calibration.hpp
    device_diagnostics.cpp
    device_diagnostics.hpp
    device_controls.cpp
    device_controls.hpp
    frame_sink.cpp
    metadata_sink.cpp
    metadata_sink.hpp
    mcap_metadata_sink.cpp
    schema_metadata_sink.cpp
    sdk_timestamp.cpp
    inc/ego_camera/ego_camera.hpp
    inc/ego_camera/sdk_timestamp.hpp
)

if(ORBBEC_EGO_ENABLE_PREVIEW)
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(EGO_PREVIEW REQUIRED IMPORTED_TARGET sdl2 libavcodec libavutil libswscale)
    list(APPEND ego_camera_core_sources
        preview.cpp
        preview.hpp
    )
endif()

add_library(ego_camera_core STATIC ${ego_camera_core_sources})

target_include_directories(ego_camera_core
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/inc>
)

target_link_libraries(ego_camera_core
    PUBLIC
        isaacteleop_schema
    PRIVATE
        yaml-cpp::yaml-cpp
        ob::OrbbecSDK
        ego::mcap_support
        deviceio::deviceio_trackers
        log_bridge::log_bridge_core
        oxr::oxr_core
        pusherio::pusherio
        Teleop::plugin_utils
)

if(ORBBEC_EGO_ENABLE_PREVIEW)
    target_compile_definitions(ego_camera_core PRIVATE ORBBEC_EGO_ENABLE_PREVIEW=1)
    target_link_libraries(ego_camera_core PRIVATE PkgConfig::EGO_PREVIEW)
endif()

add_library(ego::camera_core ALIAS ego_camera_core)
