# SPDX-FileCopyrightText: Copyright (c) 2026 j-mex. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
    message(STATUS "Skipping jmex plugin (Linux only)")
    return()
endif()

# core/ builds with or without the SDK; app/ is the part that needs it.
add_subdirectory(core)

if(NOT CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
    message(WARNING "BUILD_PLUGIN_JMEX=ON but the MOXI Receiver SDK is built for x86_64 only and this "
                    "build targets ${CMAKE_SYSTEM_PROCESSOR}; skipping the jmex plugin build.")
    return()
endif()

# The MOXI Receiver SDK is obtained from j-mex rather than fetched; README.md says where to put it.
set(_JMEX_LOCAL_SDK_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/MOXIReceiverSDK")
set(_JMEX_MIN_SDK_VERSION 0.8.1)

find_package(MOXIReceiverSDK ${_JMEX_MIN_SDK_VERSION} QUIET PATHS "${_JMEX_LOCAL_SDK_ROOT}/sdk/linux-x64")

if(NOT TARGET Moxi::Receiver)
    message(WARNING "BUILD_PLUGIN_JMEX=ON but no MOXIReceiverSDK ${_JMEX_MIN_SDK_VERSION} or newer was found; "
                    "skipping the jmex plugin build. Unpack the SDK into ${_JMEX_LOCAL_SDK_ROOT}, or set "
                    "CMAKE_PREFIX_PATH to <sdk>/sdk/linux-x64 or MOXIReceiverSDK_DIR to "
                    "<sdk>/sdk/linux-x64/lib/cmake/MOXIReceiverSDK.")
    return()
endif()

# A stale CMAKE_PREFIX_PATH looks the same as a deliberate one.
message(STATUS "jmex plugin: MOXIReceiverSDK ${MOXIReceiverSDK_VERSION} from ${MOXIReceiverSDK_DIR}")

add_subdirectory(app)

install(FILES plugin.yaml README.md DESTINATION plugins/jmex)
