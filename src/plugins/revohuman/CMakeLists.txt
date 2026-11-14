# SPDX-FileCopyrightText: Copyright (c) 2026 BrainCo. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# RevoHuman exoskeleton gloves: OpenXR hands, encoder joint states and tactile data, pushed via OpenXR.

if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
    message(STATUS "Skipping revohuman plugin (Linux only)")
    return()
endif()

# The pinned RevoHuman C SDK release. The hand fits in core/hand_fit.inc are for the hand models of this
# release, so the version, the hashes and the fits change together.
set(_revohuman_sdk_version "0.1.3")
set(_revohuman_sdk_sha256_x86_64 "ac5ecead08753aefabef246aacfea9b08eadafa0e2cf43e45b3c37978219781a")
set(_revohuman_sdk_sha256_aarch64 "f5d0f1df854df11cc826913a8fce60e23e8a98b81dfa033a6744084c351fb72a")

set(REVOHUMAN_SDK_ROOT "" CACHE PATH
    "Optional RevoHuman C SDK root (include/, lib/, LICENSE); if empty, CMake downloads the pinned release")

if(REVOHUMAN_SDK_ROOT)
    if(NOT IS_DIRECTORY "${REVOHUMAN_SDK_ROOT}")
        message(FATAL_ERROR
            "REVOHUMAN_SDK_ROOT does not exist: ${REVOHUMAN_SDK_ROOT}\n"
            "Clear REVOHUMAN_SDK_ROOT to let CMake download the pinned release, or set it to a release package."
        )
    endif()
    get_filename_component(_revohuman_sdk_dir "${REVOHUMAN_SDK_ROOT}" ABSOLUTE)
else()
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)$")
        set(_revohuman_sdk_arch "aarch64")
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64)$")
        set(_revohuman_sdk_arch "x86_64")
    else()
        message(STATUS "Skipping revohuman plugin: no RevoHuman C SDK release for ${CMAKE_SYSTEM_PROCESSOR}; "
                       "set REVOHUMAN_SDK_ROOT to an SDK built for it")
        return()
    endif()
    include(FetchContent)
    FetchContent_Declare(revohuman_sdk
        URL "https://github.com/BrainCoTech/brainco-revohuman-sdk/releases/download/c-sdk-v${_revohuman_sdk_version}/revohuman-c-sdk-${_revohuman_sdk_version}-linux-${_revohuman_sdk_arch}.tar.gz"
        URL_HASH "SHA256=${_revohuman_sdk_sha256_${_revohuman_sdk_arch}}"
    )
    # The release has no CMakeLists.txt, so this only downloads, checks and extracts it.
    FetchContent_MakeAvailable(revohuman_sdk)
    set(_revohuman_sdk_dir "${revohuman_sdk_SOURCE_DIR}")
endif()

# Never reuse a cached library path: the headers and the library must come from the same SDK.
unset(REVOHUMAN_SDK_LIBRARY CACHE)
find_library(REVOHUMAN_SDK_LIBRARY NAMES revohuman PATHS "${_revohuman_sdk_dir}/lib" NO_DEFAULT_PATH NO_CACHE)
if(NOT REVOHUMAN_SDK_LIBRARY OR NOT EXISTS "${_revohuman_sdk_dir}/include/revohuman.h"
   OR NOT EXISTS "${_revohuman_sdk_dir}/include/revohuman_fk.h" OR NOT EXISTS "${_revohuman_sdk_dir}/LICENSE")
    message(FATAL_ERROR "RevoHuman C SDK at ${_revohuman_sdk_dir} is incomplete: it needs lib/librevohuman.so, "
                        "include/revohuman.h, include/revohuman_fk.h and LICENSE. "
                        "Clear REVOHUMAN_SDK_ROOT to let CMake download the pinned release, "
                        "or set it to a release package.")
endif()
message(STATUS "RevoHuman C SDK: ${_revohuman_sdk_dir}")

add_library(revohuman_plugin::sdk UNKNOWN IMPORTED GLOBAL)
set_target_properties(revohuman_plugin::sdk PROPERTIES
    IMPORTED_LOCATION "${REVOHUMAN_SDK_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${_revohuman_sdk_dir}/include")

add_subdirectory(core)
add_subdirectory(app)

install(FILES plugin.yaml README.md DESTINATION plugins/revohuman COMPONENT revohuman)
