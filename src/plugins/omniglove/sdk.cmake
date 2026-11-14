# SPDX-FileCopyrightText: Copyright (c) 2026 Agibot Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

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

include(FetchContent)
FetchContent_Declare(omniglove_sdk
    GIT_REPOSITORY https://github.com/AgibotTech/agilink_omniglove_sdk.git
    GIT_TAG a2f899cd45c51ae01c9bbca9b893e4afba41c0d7 # v1.0.0
    # This deliberately absent subdirectory prevents running the vendor's CMake.
    SOURCE_SUBDIR isaac-prebuilt-sdk
)
FetchContent_MakeAvailable(omniglove_sdk)
file(REAL_PATH "${omniglove_sdk_SOURCE_DIR}" _omniglove_sdk_root)

if(NOT EXISTS "${_omniglove_sdk_root}/SHA256SUMS" OR NOT EXISTS "${_omniglove_sdk_root}/VERSION")
    message(FATAL_ERROR "Not a complete OmniGlove SDK release: ${_omniglove_sdk_root}")
endif()
# The pinned commit selects the managed release; its checksums detect corruption.
file(STRINGS "${_omniglove_sdk_root}/SHA256SUMS" _omniglove_checksums)
if(NOT _omniglove_checksums)
    message(FATAL_ERROR "Empty OmniGlove SDK checksum manifest")
endif()
set(_omniglove_verified_files)
foreach(_omniglove_entry IN LISTS _omniglove_checksums)
    if(NOT _omniglove_entry MATCHES "^([0-9a-fA-F]+)  (.+)$")
        message(FATAL_ERROR "Malformed OmniGlove SDK checksum entry: ${_omniglove_entry}")
    endif()
    set(_omniglove_expected "${CMAKE_MATCH_1}")
    string(LENGTH "${_omniglove_expected}" _omniglove_digest_length)
    if(NOT _omniglove_digest_length EQUAL 64)
        message(FATAL_ERROR "Invalid OmniGlove SDK SHA256 digest: ${_omniglove_entry}")
    endif()
    file(REAL_PATH "${_omniglove_sdk_root}/${CMAKE_MATCH_2}" _omniglove_file)
    cmake_path(IS_PREFIX _omniglove_sdk_root "${_omniglove_file}" NORMALIZE _omniglove_inside_root)
    if(NOT _omniglove_inside_root)
        message(FATAL_ERROR "OmniGlove SDK checksum path escapes release root: ${_omniglove_file}")
    endif()
    if(NOT EXISTS "${_omniglove_file}")
        message(FATAL_ERROR "Missing OmniGlove SDK release file: ${_omniglove_file}")
    endif()
    file(SHA256 "${_omniglove_file}" _omniglove_actual)
    if(NOT _omniglove_actual STREQUAL _omniglove_expected)
        message(FATAL_ERROR "OmniGlove SDK checksum mismatch: ${_omniglove_file}")
    endif()
    list(APPEND _omniglove_verified_files "${_omniglove_file}")
endforeach()

function(omniglove_require_verified_file path)
    if(NOT EXISTS "${path}")
        message(FATAL_ERROR "Missing OmniGlove SDK release file: ${path}")
    endif()
    file(REAL_PATH "${path}" _omniglove_required_file)
    if(NOT _omniglove_required_file IN_LIST _omniglove_verified_files)
        message(FATAL_ERROR "OmniGlove SDK file is not covered by SHA256SUMS: ${path}")
    endif()
endfunction()

foreach(_omniglove_required IN ITEMS
    VERSION LICENSE THIRD_PARTY_NOTICES.md licenses/libusb-COPYING
    "${_omniglove_platform}/cmake/OmnigloveSDKConfig.cmake"
    "${_omniglove_platform}/cmake/OmnigloveSDKConfigVersion.cmake"
    "${_omniglove_platform}/include/omniglove/omniglove.h"
)
    omniglove_require_verified_file("${_omniglove_sdk_root}/${_omniglove_required}")
endforeach()

set(OmnigloveSDK_DIR "${_omniglove_sdk_root}/${_omniglove_platform}/cmake")

# Tests outside this plugin directory also consume the imported SDK target.
find_package(OmnigloveSDK CONFIG REQUIRED GLOBAL
    PATHS "${_omniglove_sdk_root}/${_omniglove_platform}/cmake"
    NO_DEFAULT_PATH
)
get_target_property(OMNIGLOVE_SDK_LIBRARY OmnigloveSDK::api IMPORTED_LOCATION)
set(OMNIGLOVE_SDK_RELEASE_ROOT "${_omniglove_sdk_root}")
omniglove_require_verified_file("${OMNIGLOVE_SDK_LIBRARY}")
find_library(_omniglove_system_libusb NAMES usb-1.0 libusb-1.0.so.0 REQUIRED)
message(STATUS "OmniGlove SDK ${OmnigloveSDK_VERSION}; system libusb: ${_omniglove_system_libusb}")
