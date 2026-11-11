# SPDX-FileCopyrightText: Copyright (c) 2026 Agibot Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

set(OMNIGLOVE_SDK_ROOT "" CACHE PATH "Explicit OmniGlove SDK release root; empty fetches the pinned SDK")
set(OMNIGLOVE_SDK_REF "a2f899cd45c51ae01c9bbca9b893e4afba41c0d7" CACHE STRING "Managed OmniGlove SDK commit or ref")

if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
    message(FATAL_ERROR "The OmniGlove SDK supports Linux only")
endif()
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64)$")
    set(_omniglove_platform linux-x86_64)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)$")
    set(_omniglove_platform linux-aarch64)
else()
    message(FATAL_ERROR "Unsupported OmniGlove SDK architecture: ${CMAKE_SYSTEM_PROCESSOR}")
endif()

if(OMNIGLOVE_SDK_ROOT)
    file(REAL_PATH "${OMNIGLOVE_SDK_ROOT}" _omniglove_sdk_root BASE_DIRECTORY "${CMAKE_SOURCE_DIR}")
else()
    include(FetchContent)
    FetchContent_Declare(omniglove_sdk
        GIT_REPOSITORY https://github.com/AgibotTech/agilink_omniglove_sdk.git
        GIT_TAG "${OMNIGLOVE_SDK_REF}"
        # Consume the prebuilt package, not the SDK's standalone examples/build.
        SOURCE_SUBDIR isaac-prebuilt-sdk
    )
    FetchContent_MakeAvailable(omniglove_sdk)
    set(_omniglove_sdk_root "${omniglove_sdk_SOURCE_DIR}")
endif()

if(NOT EXISTS "${_omniglove_sdk_root}/SHA256SUMS" OR NOT EXISTS "${_omniglove_sdk_root}/VERSION")
    message(FATAL_ERROR "Not a complete OmniGlove SDK release: ${_omniglove_sdk_root}")
endif()
# The pinned commit selects the managed release; its checksums detect corruption.
file(STRINGS "${_omniglove_sdk_root}/SHA256SUMS" _omniglove_checksums)
foreach(_omniglove_entry IN LISTS _omniglove_checksums)
    if(NOT _omniglove_entry MATCHES "^([0-9a-fA-F]+)  (.+)$")
        message(FATAL_ERROR "Malformed OmniGlove SDK checksum entry: ${_omniglove_entry}")
    endif()
    set(_omniglove_expected "${CMAKE_MATCH_1}")
    set(_omniglove_file "${_omniglove_sdk_root}/${CMAKE_MATCH_2}")
    if(NOT EXISTS "${_omniglove_file}")
        message(FATAL_ERROR "Missing OmniGlove SDK release file: ${_omniglove_file}")
    endif()
    file(SHA256 "${_omniglove_file}" _omniglove_actual)
    if(NOT _omniglove_actual STREQUAL _omniglove_expected)
        message(FATAL_ERROR "OmniGlove SDK checksum mismatch: ${_omniglove_file}")
    endif()
endforeach()

set(OmnigloveSDK_DIR "${_omniglove_sdk_root}/${_omniglove_platform}/cmake")

# Tests outside this plugin directory also consume the imported SDK target.
find_package(OmnigloveSDK CONFIG REQUIRED GLOBAL
    PATHS "${_omniglove_sdk_root}/${_omniglove_platform}/cmake"
    NO_DEFAULT_PATH
)
get_target_property(OMNIGLOVE_SDK_LIBRARY OmnigloveSDK::api IMPORTED_LOCATION)
set(OMNIGLOVE_SDK_RELEASE_ROOT "${_omniglove_sdk_root}")
set(OMNIGLOVE_SDK_GIT_COMMIT "unknown")
find_package(Git QUIET)
if(GIT_FOUND)
    execute_process(
        COMMAND "${GIT_EXECUTABLE}" -C "${_omniglove_sdk_root}" rev-parse --show-toplevel
        OUTPUT_VARIABLE _omniglove_git_root
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    # An unpacked SDK inside a project must not inherit the enclosing repo's commit.
    if(_omniglove_git_root STREQUAL _omniglove_sdk_root)
        execute_process(
            COMMAND "${GIT_EXECUTABLE}" -C "${_omniglove_sdk_root}" rev-parse HEAD
            OUTPUT_VARIABLE OMNIGLOVE_SDK_GIT_COMMIT
            OUTPUT_STRIP_TRAILING_WHITESPACE
            COMMAND_ERROR_IS_FATAL ANY
        )
    endif()
endif()
message(STATUS "OmniGlove SDK ${OmnigloveSDK_VERSION}: ${OMNIGLOVE_SDK_GIT_COMMIT}")
