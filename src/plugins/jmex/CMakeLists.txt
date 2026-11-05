# SPDX-FileCopyrightText: Copyright (c) 2026 j-mex. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.20)

if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
    message(STATUS "Skipping jmex plugin (Linux only)")
    return()
endif()

# The MOXI Receiver SDK is obtained from j-mex rather than fetched; README.md says where to put it.
set(_JMEX_LOCAL_SDK_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/MOXIReceiverSDK")
set(_JMEX_MIN_SDK_VERSION 0.8.0)

if(NOT DEFINED JMEX_SDK_ROOT AND DEFINED ENV{JMEX_SDK_ROOT})
    set(JMEX_SDK_ROOT "$ENV{JMEX_SDK_ROOT}" CACHE PATH "Path to the MOXI Receiver SDK platform directory")
endif()

# An override is searched on its own first: a stale or mistyped one must be reported, not degrade
# into a silent build against whatever is unpacked in the plugin directory. The cached result of an
# earlier configure is dropped, or a since-edited JMEX_SDK_ROOT would never be looked at again.
if(JMEX_SDK_ROOT)
    unset(MOXIReceiverSDK_DIR CACHE)
    find_package(MOXIReceiverSDK ${_JMEX_MIN_SDK_VERSION} QUIET NO_DEFAULT_PATH PATHS "${JMEX_SDK_ROOT}")
    if(NOT TARGET Moxi::Receiver)
        message(WARNING "JMEX_SDK_ROOT='${JMEX_SDK_ROOT}' holds no MOXIReceiverSDK ${_JMEX_MIN_SDK_VERSION} or newer; "
                        "ignoring it and searching the usual paths")
    endif()
endif()

if(NOT TARGET Moxi::Receiver)
    find_package(MOXIReceiverSDK ${_JMEX_MIN_SDK_VERSION} QUIET PATHS
                 "${_JMEX_LOCAL_SDK_ROOT}/sdk/linux-x64"
                 "${_JMEX_LOCAL_SDK_ROOT}")
endif()

# core/ builds with or without the SDK; app/ is the part that needs it.
add_subdirectory(core)

if(NOT TARGET Moxi::Receiver)
    message(STATUS "Skipping jmex plugin build: MOXIReceiverSDK ${_JMEX_MIN_SDK_VERSION} or newer not found "
                   "(unpack the SDK into ${_JMEX_LOCAL_SDK_ROOT}, or set CMAKE_PREFIX_PATH or "
                   "JMEX_SDK_ROOT to <sdk>/sdk/linux-x64)")
    return()
endif()

# Report which SDK answered: a stale CMAKE_PREFIX_PATH looks the same as a deliberate one.
message(STATUS "jmex plugin: MOXIReceiverSDK ${MOXIReceiverSDK_VERSION} from ${MOXIReceiverSDK_DIR}")

add_subdirectory(app)

install(FILES plugin.yaml README.md DESTINATION plugins/jmex)
