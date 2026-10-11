# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

include(GNUInstallDirs)
option(EGO_ENABLE_PREVIEW "Build the EGO SDL/FFmpeg preview" OFF)
set(EGO_PLUGIN_OUTPUT_DIR "${CMAKE_BINARY_DIR}/src/plugins/ego_camera")
add_subdirectory(mcap)

if(BUILD_PLUGIN_EGO_CAMERA)
    if(NOT DEFINED ORBBEC_SDK_ROOT AND DEFINED ENV{ORBBEC_SDK_ROOT})
        set(ORBBEC_SDK_ROOT "$ENV{ORBBEC_SDK_ROOT}" CACHE PATH "Path to the OrbbecSDK v2 package root")
    endif()

    if(NOT ORBBEC_SDK_ROOT)
        message(FATAL_ERROR "BUILD_PLUGIN_EGO_CAMERA=ON requires ORBBEC_SDK_ROOT, the OrbbecSDK v2 package root.")
    endif()

    foreach(_ego_required
            "${ORBBEC_SDK_ROOT}/include/libobsensor/ObSensor.hpp"
            "${ORBBEC_SDK_ROOT}/lib/OrbbecSDKConfig.cmake"
            "${ORBBEC_SDK_ROOT}/lib/libOrbbecSDK.so.2"
            "${ORBBEC_SDK_ROOT}/lib/extensions")
        if(NOT EXISTS "${_ego_required}")
            message(FATAL_ERROR "Invalid ORBBEC_SDK_ROOT='${ORBBEC_SDK_ROOT}': missing ${_ego_required}")
        endif()
    endforeach()

    find_package(OrbbecSDK CONFIG REQUIRED
        PATHS "${ORBBEC_SDK_ROOT}/lib"
        NO_DEFAULT_PATH
    )

    set(EGO_PLUGIN_MANIFEST "${CMAKE_CURRENT_SOURCE_DIR}/plugin.yaml")

    # The SDK SONAME patch level is release-specific.  Package every libOrbbecSDK
    # symlink/real-library pair supplied by the selected SDK rather than assuming
    # the development machine's 2.9.0 filename.
    file(GLOB ORBBEC_SDK_LIBRARY_FILES "${ORBBEC_SDK_ROOT}/lib/libOrbbecSDK.so*")
    if(NOT ORBBEC_SDK_LIBRARY_FILES)
        message(FATAL_ERROR "Invalid ORBBEC_SDK_ROOT='${ORBBEC_SDK_ROOT}': no libOrbbecSDK.so* files found")
    endif()

    add_subdirectory(core)
    add_subdirectory(app)

    install(FILES
        "${CMAKE_CURRENT_SOURCE_DIR}/plugin.yaml"
        "${CMAKE_CURRENT_SOURCE_DIR}/ego.example.toml"
        "${CMAKE_CURRENT_SOURCE_DIR}/README.md"
        "${ORBBEC_SDK_ROOT}/lib/OrbbecSDKConfig.xml"
        ${ORBBEC_SDK_LIBRARY_FILES}
        DESTINATION plugins/ego
        COMPONENT ego
    )
    install(PROGRAMS
        "${CMAKE_CURRENT_SOURCE_DIR}/ego.sh"
        DESTINATION plugins/ego
        COMPONENT ego
    )
    install(DIRECTORY "${ORBBEC_SDK_ROOT}/lib/extensions"
        DESTINATION plugins/ego
        COMPONENT ego
    )
    install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/tools" DESTINATION plugins/ego COMPONENT ego)
endif()
