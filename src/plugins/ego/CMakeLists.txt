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

    # A cached package directory must not override a newly selected SDK root.
    set(OrbbecSDK_DIR "${ORBBEC_SDK_ROOT}/lib" CACHE PATH "Selected EGO SDK CMake package directory" FORCE)
    find_package(OrbbecSDK CONFIG REQUIRED
        PATHS "${ORBBEC_SDK_ROOT}/lib"
        NO_DEFAULT_PATH
    )

    # Recheck when the SDK changes inside an existing build tree.
    unset(EGO_SDK_HAS_REQUIRED_APIS CACHE)
    try_compile(EGO_SDK_HAS_REQUIRED_APIS
        "${CMAKE_CURRENT_BINARY_DIR}/sdk_check"
        "${CMAKE_CURRENT_SOURCE_DIR}/tools/sdk_check.cpp"
        LINK_LIBRARIES ob::OrbbecSDK
        CXX_STANDARD 17
        OUTPUT_VARIABLE _ego_sdk_check_output
    )
    if(NOT EGO_SDK_HAS_REQUIRED_APIS)
        message(FATAL_ERROR
            "The SDK at '${ORBBEC_SDK_ROOT}' lacks the required EGO APIs or cannot link them. "
            "Use the EGO-specific v2.10.6-EGO-20261019 build from "
            "https://github.com/orbbec/OrbbecSDK_v2-EGO/releases/tag/v2.10.6-EGO-20261019\n"
            "${_ego_sdk_check_output}")
    endif()

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
