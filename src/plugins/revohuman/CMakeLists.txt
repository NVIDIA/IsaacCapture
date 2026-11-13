# SPDX-FileCopyrightText: Copyright (c) 2026 BrainCo. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# RevoHuman exoskeleton gloves: OpenXR hands, encoder joint states and tactile data, pushed via OpenXR.

# Read by install.sh, which must not accept a build without the plugin.
set(REVOHUMAN_PLUGIN_AVAILABLE OFF CACHE INTERNAL "The RevoHuman plugin is built")

if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
    message(STATUS "Skipping revohuman plugin (Linux only)")
    return()
endif()

set(REVOHUMAN_SDK_ROOT "$ENV{REVOHUMAN_SDK_ROOT}" CACHE PATH "RevoHuman C SDK root (include/, lib/, LICENSE)")
if(NOT REVOHUMAN_SDK_ROOT)
    message(STATUS "REVOHUMAN_SDK_ROOT not set — skipping revohuman plugin build.")
    return()
endif()
# Never reuse a cached library path: the headers and the library must come from the same SDK root.
unset(REVOHUMAN_SDK_LIBRARY CACHE)
find_library(REVOHUMAN_SDK_LIBRARY NAMES revohuman PATHS "${REVOHUMAN_SDK_ROOT}/lib" NO_DEFAULT_PATH NO_CACHE)
if(NOT REVOHUMAN_SDK_LIBRARY OR NOT EXISTS "${REVOHUMAN_SDK_ROOT}/include/revohuman.h"
   OR NOT EXISTS "${REVOHUMAN_SDK_ROOT}/include/revohuman_fk.h" OR NOT EXISTS "${REVOHUMAN_SDK_ROOT}/LICENSE")
    message(FATAL_ERROR "RevoHuman C SDK at ${REVOHUMAN_SDK_ROOT} is incomplete: it needs lib/librevohuman.so, "
                        "include/revohuman.h, include/revohuman_fk.h and LICENSE. "
                        "Use a release package (install.sh downloads one).")
endif()

add_library(revohuman_plugin::sdk UNKNOWN IMPORTED GLOBAL)
set_target_properties(revohuman_plugin::sdk PROPERTIES
    IMPORTED_LOCATION "${REVOHUMAN_SDK_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${REVOHUMAN_SDK_ROOT}/include")
set(REVOHUMAN_PLUGIN_AVAILABLE ON CACHE INTERNAL "The RevoHuman plugin is built")

add_subdirectory(core)
add_subdirectory(app)

install(FILES plugin.yaml README.md DESTINATION plugins/revohuman COMPONENT revohuman)
