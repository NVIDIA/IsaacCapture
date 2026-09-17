# SPDX-FileCopyrightText: Copyright (c) 2025-2026 Avatar SDK contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
  message(STATUS "Skipping sharpa_avatar plugin (Linux only)")
  return()
endif()

include(GNUInstallDirs)

# SDK root precedence: -DAVATAR_SDK_ROOT > $AVATAR_SDK_ROOT > the default host path.
if(NOT DEFINED AVATAR_SDK_ROOT)
  if(DEFINED ENV{AVATAR_SDK_ROOT})
    set(AVATAR_SDK_ROOT "$ENV{AVATAR_SDK_ROOT}")
  else()
    set(AVATAR_SDK_ROOT "/opt/avatar-sdk")
  endif()
endif()
set(_AVATAR_SDK_INCLUDE_DIR "${AVATAR_SDK_ROOT}/include")
set(_AVATAR_SDK_LIBRARY_DIR "${AVATAR_SDK_ROOT}/lib")
set(_AVATAR_SDK_CONFIG "${AVATAR_SDK_ROOT}/share/sdk_config.json")
set(_AVATAR_VERSION_FILE "${AVATAR_SDK_ROOT}/share/Version")
set(_AVATAR_PLUGIN_BUILD_DIR "${CMAKE_CURRENT_BINARY_DIR}")

find_library(_AVATAR_SDK_LIBRARY
  NAMES avatar_sdk
  PATHS "${_AVATAR_SDK_LIBRARY_DIR}"
  NO_DEFAULT_PATH
  NO_CACHE
)

if(NOT EXISTS "${_AVATAR_SDK_INCLUDE_DIR}/avatar_sdk/AvatarSDK.h"
    OR NOT _AVATAR_SDK_LIBRARY
    OR NOT EXISTS "${_AVATAR_SDK_CONFIG}"
    OR NOT EXISTS "${_AVATAR_VERSION_FILE}")
  message(WARNING
    "BUILD_PLUGIN_SHARPA_AVATAR=ON but the Avatar SDK under ${AVATAR_SDK_ROOT} is missing; "
    "skipping this plugin. Run src/plugins/sharpa_avatar/install_avatar_sdk.sh.")
  return()
endif()

# Version pinning lives in install_avatar_sdk.sh --check: the default root is held
# to the full pin; a custom root is validated for layout and production build.
execute_process(
  COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/install_avatar_sdk.sh" --check "${AVATAR_SDK_ROOT}"
  RESULT_VARIABLE _AVATAR_SDK_CHECK_RESULT
  OUTPUT_VARIABLE _AVATAR_SDK_CHECK_OUTPUT
  ERROR_VARIABLE _AVATAR_SDK_CHECK_ERROR
  OUTPUT_STRIP_TRAILING_WHITESPACE
  ERROR_STRIP_TRAILING_WHITESPACE
)
if(NOT _AVATAR_SDK_CHECK_RESULT EQUAL 0)
  message(FATAL_ERROR "${_AVATAR_SDK_CHECK_ERROR}")
elseif(_AVATAR_SDK_CHECK_ERROR)
  message(WARNING "${_AVATAR_SDK_CHECK_ERROR}")
endif()
message(STATUS "${_AVATAR_SDK_CHECK_OUTPUT}")

add_library(AvatarSDK::AvatarSDK SHARED IMPORTED)
set_target_properties(AvatarSDK::AvatarSDK PROPERTIES
  IMPORTED_LOCATION "${_AVATAR_SDK_LIBRARY}"
  INTERFACE_INCLUDE_DIRECTORIES "${_AVATAR_SDK_INCLUDE_DIR}"
)

add_subdirectory(core)
add_subdirectory(app)
add_subdirectory(tools)

configure_file(plugin.yaml.in "${CMAKE_CURRENT_BINARY_DIR}/plugin.yaml" @ONLY)

install(
  FILES "${CMAKE_CURRENT_BINARY_DIR}/plugin.yaml" README.md
  DESTINATION plugins/sharpa_avatar
  COMPONENT avatar
)
