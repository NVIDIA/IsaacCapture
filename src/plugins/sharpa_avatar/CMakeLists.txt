# SPDX-FileCopyrightText: Copyright (c) 2025-2026 Avatar SDK contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Sharpa Avatar gloves -> Isaac Teleop hand-tracking plugin.
#
# Enable with -DBUILD_PLUGIN_SHARPA_AVATAR=ON (see install.sh). Locate Avatar SDK:
#   1) vendor/avatar-sdk next to this CMakeLists (staged by vendor_avatar_sdk.sh)
#   2) AVATAR_SDK_ROOT / $ENV{AVATAR_SDK_ROOT}
#   3) /opt/avatar-sdk
#
# Installs libavatar_sdk (+ casadi/ipopt) into lib/, and sdk_config.json +
# hand_fk/ next to the plugin binary so runtime does not need a separate SDK deb.

cmake_minimum_required(VERSION 3.20)

include(GNUInstallDirs)

option(AVATAR_ADD_SDK_TO_BUILD_RPATH "Add Avatar SDK lib dir to BUILD_RPATH for plugin and tools" ON)

set(_AVATAR_VENDOR_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/vendor/avatar-sdk")
set(_AVATAR_DEFAULT_ROOT "/opt/avatar-sdk")

if(NOT DEFINED AVATAR_SDK_ROOT)
  if(EXISTS "${_AVATAR_VENDOR_ROOT}/include/avatar_sdk/AvatarSDK.h")
    set(AVATAR_SDK_ROOT "${_AVATAR_VENDOR_ROOT}" CACHE PATH "Path to Avatar SDK root (vendored or /opt/avatar-sdk)" FORCE)
  elseif(DEFINED ENV{AVATAR_SDK_ROOT})
    set(AVATAR_SDK_ROOT "$ENV{AVATAR_SDK_ROOT}" CACHE PATH "Path to Avatar SDK root (vendored or /opt/avatar-sdk)" FORCE)
  else()
    set(AVATAR_SDK_ROOT "${_AVATAR_DEFAULT_ROOT}" CACHE PATH "Path to Avatar SDK root (vendored or /opt/avatar-sdk)")
  endif()
elseif(NOT EXISTS "${AVATAR_SDK_ROOT}/include/avatar_sdk/AvatarSDK.h")
  if(EXISTS "${_AVATAR_VENDOR_ROOT}/include/avatar_sdk/AvatarSDK.h")
    message(STATUS "Cached AVATAR_SDK_ROOT='${AVATAR_SDK_ROOT}' is invalid; using vendored '${_AVATAR_VENDOR_ROOT}'")
    set(AVATAR_SDK_ROOT "${_AVATAR_VENDOR_ROOT}" CACHE PATH "Path to Avatar SDK root (vendored or /opt/avatar-sdk)" FORCE)
  endif()
endif()

get_filename_component(AVATAR_SDK_ROOT "${AVATAR_SDK_ROOT}" ABSOLUTE)

set(AVATAR_SDK_INCLUDE_DIR "${AVATAR_SDK_ROOT}/include" CACHE PATH "Avatar SDK include path" FORCE)
set(AVATAR_SDK_LIBRARY_DIR "${AVATAR_SDK_ROOT}/lib" CACHE PATH "Avatar SDK library path" FORCE)

set(AVATAR_SHOULD_BUILD ON)
set(AVATAR_SKIP_REASON "")

if(NOT EXISTS "${AVATAR_SDK_INCLUDE_DIR}/avatar_sdk/AvatarSDK.h")
  set(AVATAR_SHOULD_BUILD OFF)
  set(AVATAR_SKIP_REASON "Header not found at ${AVATAR_SDK_INCLUDE_DIR}/avatar_sdk/AvatarSDK.h (run src/plugins/sharpa_avatar/install.sh or vendor_avatar_sdk.sh)")
endif()

set(_AVATAR_LIB "")
if(AVATAR_SHOULD_BUILD)
  foreach(_cand
      "${AVATAR_SDK_LIBRARY_DIR}/libavatar_sdk_wrapper.so"
      "${AVATAR_SDK_LIBRARY_DIR}/libavatar_sdk.so")
    if(EXISTS "${_cand}")
      set(_AVATAR_LIB "${_cand}")
      break()
    endif()
  endforeach()
  if(NOT _AVATAR_LIB)
    set(AVATAR_SHOULD_BUILD OFF)
    set(AVATAR_SKIP_REASON "No libavatar_sdk[_wrapper].so found under ${AVATAR_SDK_LIBRARY_DIR}")
  endif()
endif()

if(NOT AVATAR_SHOULD_BUILD)
  message(FATAL_ERROR
    "Sharpa Avatar plugin: ${AVATAR_SKIP_REASON}\n"
    "  Run:  ${CMAKE_CURRENT_SOURCE_DIR}/install.sh\n"
    "  Or set AVATAR_SDK_ROOT to an installed Avatar SDK (headers + libavatar_sdk.so).")
endif()

message(STATUS "Avatar plugin: using SDK at ${AVATAR_SDK_ROOT} (lib: ${_AVATAR_LIB})")

add_library(AvatarSDK::AvatarSDK SHARED IMPORTED)
set_target_properties(AvatarSDK::AvatarSDK PROPERTIES
  IMPORTED_LOCATION "${_AVATAR_LIB}"
  INTERFACE_INCLUDE_DIRECTORIES "${AVATAR_SDK_INCLUDE_DIR}"
)

set(_ISAAC_AVATAR_BUILD_RPATH "$ORIGIN;$ORIGIN/../lib")
if(AVATAR_ADD_SDK_TO_BUILD_RPATH)
  list(APPEND _ISAAC_AVATAR_BUILD_RPATH "${AVATAR_SDK_LIBRARY_DIR}")
endif()
list(JOIN _ISAAC_AVATAR_BUILD_RPATH ";" _ISAAC_AVATAR_BUILD_RPATH_STR)

add_subdirectory(core)
add_library(isaac::avatar_plugin ALIAS avatar_plugin_core)
add_subdirectory(app)
add_subdirectory(tools)

# Runtime libraries next to other Isaac libs (plugin RPATH: $ORIGIN/../../lib).
file(GLOB _AVATAR_RUNTIME_LIBS
  "${AVATAR_SDK_LIBRARY_DIR}/libavatar_sdk.so"
  "${AVATAR_SDK_LIBRARY_DIR}/libavatar_sdk_wrapper.so"
  "${AVATAR_SDK_LIBRARY_DIR}/libcasadi.so*"
  "${AVATAR_SDK_LIBRARY_DIR}/libcasadi_nlpsol_ipopt.so"
  "${AVATAR_SDK_LIBRARY_DIR}/libipopt.so*"
  "${AVATAR_SDK_LIBRARY_DIR}/libsipopt.so*"
  "${AVATAR_SDK_LIBRARY_DIR}/libcoinmumps.so*"
  "${AVATAR_SDK_LIBRARY_DIR}/libcoinmetis.so*"
)
if(_AVATAR_RUNTIME_LIBS)
  install(FILES ${_AVATAR_RUNTIME_LIBS}
    DESTINATION ${CMAKE_INSTALL_LIBDIR}
    COMPONENT avatar
  )
endif()

# Config + hand_fk beside the plugin binary (hand_fk_config_root is relative).
set(_AVATAR_SHARE_DIR "${AVATAR_SDK_ROOT}/share")
if(EXISTS "${_AVATAR_SHARE_DIR}/sdk_config.json")
  install(FILES "${_AVATAR_SHARE_DIR}/sdk_config.json"
    DESTINATION plugins/sharpa_avatar
    COMPONENT avatar
  )
endif()
if(EXISTS "${_AVATAR_SHARE_DIR}/hand_fk")
  install(DIRECTORY "${_AVATAR_SHARE_DIR}/hand_fk"
    DESTINATION plugins/sharpa_avatar
    COMPONENT avatar
  )
endif()
if(EXISTS "${_AVATAR_SHARE_DIR}/wave-sdk")
  install(DIRECTORY "${_AVATAR_SHARE_DIR}/wave-sdk"
    DESTINATION plugins/sharpa_avatar
    COMPONENT avatar
  )
endif()
if(EXISTS "${_AVATAR_SHARE_DIR}/Version")
  install(FILES "${_AVATAR_SHARE_DIR}/Version"
    DESTINATION plugins/sharpa_avatar
    COMPONENT avatar
  )
endif()
