# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ==============================================================================
# LiveKit
# ==============================================================================
# Each selected sensor is published as a LiveKit camera track. The SDK is a prebuilt package (headers and shared libraries), pinned to a
# release and checked against its hash; point LIVEKIT_SDK_DIR at an unpacked copy to use another.
set(_livekit_local ON)
if(NOT LIVEKIT_SDK_DIR)
    set(_livekit_local OFF)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
        set(_livekit_arch "arm64")
        set(_livekit_hash "e3a97f71943c0bf9fa4ebd70927ca2906b31dcd6c14b42312b1225b4b868e4ad")
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
        set(_livekit_arch "x64")
        set(_livekit_hash "5e0feb2e519c6ff131dd2d1c45149115057ae4e21a324fdb249e1789d5614e7b")
    else()
        message(FATAL_ERROR "No prebuilt LiveKit SDK for ${CMAKE_SYSTEM_PROCESSOR}; set LIVEKIT_SDK_DIR.")
    endif()
    include(FetchContent)
    FetchContent_Declare(
        livekit_sdk
        URL https://github.com/livekit/client-sdk-cpp/releases/download/v1.12.4/livekit-sdk-ubuntu-24.04-${_livekit_arch}-1.12.4.tar.gz
        URL_HASH SHA256=${_livekit_hash}
    )
    FetchContent_MakeAvailable(livekit_sdk)
    set(LIVEKIT_SDK_DIR "${livekit_sdk_SOURCE_DIR}")
endif()
# The package ships no licence files. The pinned download supplies the SDK's own; a local SDK uses
# LICENSE/NOTICE from its directory, or LIVEKIT_SDK_LICENSE_FILE / LIVEKIT_SDK_NOTICE_FILE, and needs no network.
foreach(_notice LICENSE NOTICE)
    set(_override "${LIVEKIT_SDK_${_notice}_FILE}")
    if(_override)
        set(_found "${_override}")
    elseif(_livekit_local)
        set(_found "")
        foreach(_candidate "${LIVEKIT_SDK_DIR}/${_notice}" "${LIVEKIT_SDK_DIR}/../${_notice}")
            if(NOT _found AND EXISTS "${_candidate}")
                set(_found "${_candidate}")
            endif()
        endforeach()
        if(NOT _found)
            message(FATAL_ERROR
                "LIVEKIT_SDK_DIR is set but ${_notice} was not found in it. Put the SDK's ${_notice} file there "
                "or pass -DLIVEKIT_SDK_${_notice}_FILE=/path/to/${_notice}.")
        endif()
    else()
        if(_notice STREQUAL "LICENSE")
            set(_notice_hash "09e8a9bcec8067104652c168685ab0931e7868f9c8284b66f5ae6edae5f1130b")
        else()
            set(_notice_hash "6d0f7c5e0deb284362296c5da110dd353831da2e67ca77cb5abc051b951b0fac")
        endif()
        FetchContent_Declare(
            livekit_sdk_${_notice}
            URL https://raw.githubusercontent.com/livekit/client-sdk-cpp/v1.12.4/${_notice}
            URL_HASH SHA256=${_notice_hash}
            DOWNLOAD_NO_EXTRACT TRUE
        )
        FetchContent_MakeAvailable(livekit_sdk_${_notice})
        # FetchContent publishes <lowercased name>_SOURCE_DIR.
        string(TOLOWER "${_notice}" _notice_lower)
        set(_found "${livekit_sdk_${_notice_lower}_SOURCE_DIR}/${_notice}")
    endif()
    set(SENSING_LIVEKIT_${_notice}_FILE "${_found}" CACHE INTERNAL "")
endforeach()
find_package(LiveKit CONFIG REQUIRED PATHS "${LIVEKIT_SDK_DIR}" NO_DEFAULT_PATH)
message(STATUS "LiveKit SDK=${LIVEKIT_SDK_DIR}")
# Cache, so tests/ (a sibling directory) can find the same SDK.
set(SENSING_LIVEKIT_SDK_DIR "${LIVEKIT_SDK_DIR}" CACHE INTERNAL "")



# The LiveKit SDK behind a small interface: join a room, publish a video track, submit H.264 access units.
# No LiveKit type appears in an installed header, and nothing here touches CUDA.
add_library(sensing_livekit STATIC
    livekit_session.cpp
    inc/sensing_livekit/livekit_session.hpp
    inc/sensing_livekit/video_track_sink.hpp
)

target_include_directories(sensing_livekit
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/inc>
)

target_link_libraries(sensing_livekit
    PUBLIC
        log_bridge::log_bridge_core
    PRIVATE
        LiveKit::livekit
)

add_library(sensing::livekit ALIAS sensing_livekit)
