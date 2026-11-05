# SPDX-FileCopyrightText: Copyright (c) 2026 BrainCo. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# RevoHuman exoskeleton gloves: OpenXR hands, encoder joint states and tactile data with its
# TactileMetadata, pushed via OpenXR.

# Outcome of this configure for scripts such as install.sh, which must not accept a build without
# the plugin; OFF until the live source is known to link, so an early return below leaves it OFF.
set(REVOHUMAN_LIVE_AVAILABLE OFF CACHE INTERNAL "RevoHuman live source links against the C SDK")

if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
    message(STATUS "Skipping revohuman plugin (Linux only)")
    add_custom_target(revohuman_plugin
        COMMAND ${CMAKE_COMMAND} -E echo "Skipping revohuman: Linux only")
    return()
endif()

# The plugin needs the C SDK header (revohuman.h); the live source also needs its library.
# There is no default location: without the cache variable or the environment variable the
# plugin is skipped, as it is when the header is missing. install.sh always passes it.
set(REVOHUMAN_SDK_ROOT "" CACHE PATH "RevoHuman C SDK root (contains include/revohuman.h and lib/)")
if(NOT REVOHUMAN_SDK_ROOT AND DEFINED ENV{REVOHUMAN_SDK_ROOT})
    set(REVOHUMAN_SDK_ROOT "$ENV{REVOHUMAN_SDK_ROOT}")
endif()
set(_revohuman_no_sdk "")
if(NOT REVOHUMAN_SDK_ROOT)
    set(_revohuman_no_sdk "RevoHuman C SDK root not set (set -DREVOHUMAN_SDK_ROOT=/path/to/sdk or REVOHUMAN_SDK_ROOT)")
elseif(NOT EXISTS "${REVOHUMAN_SDK_ROOT}/include/revohuman.h")
    set(_revohuman_no_sdk "RevoHuman C SDK not found (set -DREVOHUMAN_SDK_ROOT=/path/to/sdk)")
endif()
if(_revohuman_no_sdk)
    message(STATUS "${_revohuman_no_sdk} — skipping revohuman plugin build.")
    return()
endif()
set(REVOHUMAN_SDK_INCLUDE_DIR "${REVOHUMAN_SDK_ROOT}/include")

add_library(revohuman_sdk_headers INTERFACE)
target_include_directories(revohuman_sdk_headers INTERFACE $<BUILD_INTERFACE:${REVOHUMAN_SDK_INCLUDE_DIR}>)
add_library(revohuman_plugin::sdk_headers ALIAS revohuman_sdk_headers)

# The library is used only when it links with every function the live source calls, so a partial
# SDK build skips the plugin executable instead of failing the build. Checked on every configure,
# and searched again whenever the SDK root changes (find_library alone keeps its first result).
if(NOT "${REVOHUMAN_SDK_ROOT}" STREQUAL "${_REVOHUMAN_SDK_LIBRARY_ROOT}")
    unset(REVOHUMAN_SDK_LIBRARY CACHE)
    set(_REVOHUMAN_SDK_LIBRARY_ROOT "${REVOHUMAN_SDK_ROOT}" CACHE INTERNAL "SDK root REVOHUMAN_SDK_LIBRARY was found in")
endif()
find_library(REVOHUMAN_SDK_LIBRARY NAMES revohuman PATHS "${REVOHUMAN_SDK_ROOT}/lib" NO_DEFAULT_PATH)
set(REVOHUMAN_LIVE_AVAILABLE OFF)
if(REVOHUMAN_SDK_LIBRARY)
    include(CheckCXXSourceCompiles)
    include(CMakePushCheckState)
    cmake_push_check_state(RESET)
    set(CMAKE_REQUIRED_INCLUDES "${REVOHUMAN_SDK_INCLUDE_DIR}")
    set(CMAKE_REQUIRED_LIBRARIES "${REVOHUMAN_SDK_LIBRARY}")
    set(CMAKE_REQUIRED_QUIET ON)
    unset(REVOHUMAN_SDK_LINKS CACHE)
    check_cxx_source_compiles([=[
#include <revohuman.h>
using Fn = void (*)();
#define RH_FN(f) reinterpret_cast<Fn>(&f)
int main(int argc, char**)
{
    const Fn used[] = {
        RH_FN(rh_init_options_init), RH_FN(rh_init), RH_FN(rh_shutdown), RH_FN(rh_version),
        RH_FN(rh_set_log_callback), RH_FN(rh_last_error), RH_FN(rh_last_error_code), RH_FN(rh_error_class_name),
        RH_FN(rh_kit_options_init), RH_FN(rh_kit_open),
        RH_FN(rh_kit_close), RH_FN(rh_kit_identity), RH_FN(rh_kit_glove), RH_FN(rh_glove_info),
        RH_FN(rh_joint_name), RH_FN(rh_joints_to_radians), RH_FN(rh_subscribe_options_init),
        RH_FN(rh_glove_subscribe), RH_FN(rh_sub_recv_glove), RH_FN(rh_sub_dropped), RH_FN(rh_sub_close),
        RH_FN(rh_glove_capture_ok), RH_FN(rh_glove_capture_cause), RH_FN(rh_scan_options_init),
        RH_FN(rh_scan), RH_FN(rh_scan_count), RH_FN(rh_scan_get), RH_FN(rh_scan_free),
    };
    return used[argc] == nullptr;
}
]=] REVOHUMAN_SDK_LINKS)
    cmake_pop_check_state()
    if(REVOHUMAN_SDK_LINKS)
        set(REVOHUMAN_LIVE_AVAILABLE ON)
        add_library(revohuman_plugin::sdk UNKNOWN IMPORTED)
        set_target_properties(revohuman_plugin::sdk PROPERTIES
            IMPORTED_LOCATION "${REVOHUMAN_SDK_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${REVOHUMAN_SDK_INCLUDE_DIR}")
        message(STATUS "RevoHuman C SDK library ${REVOHUMAN_SDK_LIBRARY}: live source available "
                       "(installed next to the plugin)")
    else()
        message(STATUS "RevoHuman C SDK library ${REVOHUMAN_SDK_LIBRARY} lacks functions the live source "
                       "calls — skipping the plugin executable")
    endif()
else()
    message(STATUS "RevoHuman C SDK library not found in ${REVOHUMAN_SDK_ROOT}/lib — skipping the plugin executable")
endif()
set(REVOHUMAN_LIVE_AVAILABLE "${REVOHUMAN_LIVE_AVAILABLE}"
    CACHE INTERNAL "RevoHuman live source links against the C SDK")

add_subdirectory(core)
add_subdirectory(live)
if(REVOHUMAN_LIVE_AVAILABLE)
    add_subdirectory(kinematics)
endif()
add_subdirectory(app)

# One component for the plugin, its manifest, README and the SDK library (app/), so install.sh installs
# exactly these; a plain `cmake --install` still installs every component. No plugin, no manifest.
if(REVOHUMAN_LIVE_AVAILABLE)
    install(FILES plugin.yaml README.md DESTINATION plugins/revohuman COMPONENT revohuman)
endif()
