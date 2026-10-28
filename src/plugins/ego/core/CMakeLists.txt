# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set(ego_camera_core_sources
    time_alignment.cpp
    inc/ego_camera/time_alignment.hpp
    ego_camera.cpp
    calibration.cpp
    calibration.hpp
    inc/ego_camera/calibration.hpp
    inc/ego_camera/raw_calibration.hpp
    device_diagnostics.cpp
    device_diagnostics.hpp
    device_controls.cpp
    device_controls.hpp
    wav_writer.hpp
    frame_sink.cpp
    metadata_sink.cpp
    metadata_sink.hpp
    mcap_metadata_sink.cpp
    schema_metadata_sink.cpp
    inc/ego_camera/ego_camera.hpp
)

if(EGO_ENABLE_PREVIEW)
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(EGO_PREVIEW REQUIRED IMPORTED_TARGET sdl2 libavcodec libavutil libswscale)
    list(APPEND ego_camera_core_sources
        preview.cpp
        preview.hpp
    )
endif()

add_library(ego_camera_core STATIC ${ego_camera_core_sources})

if(NOT TARGET nlohmann_json::nlohmann_json)
    find_package(nlohmann_json CONFIG REQUIRED)
endif()
find_package(OpenSSL REQUIRED COMPONENTS Crypto)

target_include_directories(ego_camera_core
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/inc>
)

target_link_libraries(ego_camera_core
    PUBLIC
        isaacteleop_schema
    PRIVATE
        nlohmann_json::nlohmann_json
        yaml-cpp::yaml-cpp
        OpenSSL::Crypto
        ${CMAKE_DL_LIBS}
        ob::OrbbecSDK
        ego::mcap_support
        log_bridge::log_bridge_core
        oxr::oxr_core
        pusherio::pusherio
        Teleop::plugin_utils
)

if(EGO_ENABLE_PREVIEW)
    target_compile_definitions(ego_camera_core PRIVATE EGO_ENABLE_PREVIEW=1)
    target_link_libraries(ego_camera_core PRIVATE PkgConfig::EGO_PREVIEW)
endif()

add_library(ego::camera_core ALIAS ego_camera_core)
