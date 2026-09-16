# SPDX-FileCopyrightText: Copyright (c) 2025-2026 Avatar SDK contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

include(GNUInstallDirs)

set(_AVATAR_SDK_ROOT "/opt/avatar-sdk")
set(_AVATAR_SDK_INCLUDE_DIR "${_AVATAR_SDK_ROOT}/include")
set(_AVATAR_SDK_LIBRARY_DIR "${_AVATAR_SDK_ROOT}/lib")
set(_AVATAR_SDK_CONFIG "${_AVATAR_SDK_ROOT}/share/sdk_config.json")
set(_AVATAR_PLUGIN_BUILD_DIR "${CMAKE_CURRENT_BINARY_DIR}")

if(NOT EXISTS "${_AVATAR_SDK_INCLUDE_DIR}/avatar_sdk/AvatarSDK.h")
  message(FATAL_ERROR
    "Avatar SDK header not found under ${_AVATAR_SDK_INCLUDE_DIR}. "
    "Run src/plugins/sharpa_avatar/install_avatar_sdk.sh.")
endif()

# Current production packages expose avatar_sdk; wrapper is a legacy fallback.
find_library(_AVATAR_SDK_LIBRARY
  NAMES avatar_sdk avatar_sdk_wrapper
  PATHS "${_AVATAR_SDK_LIBRARY_DIR}"
  NO_DEFAULT_PATH
  NO_CACHE
  REQUIRED
)

if(NOT EXISTS "${_AVATAR_SDK_CONFIG}")
  message(FATAL_ERROR "Avatar SDK configuration not found: ${_AVATAR_SDK_CONFIG}")
endif()
add_library(AvatarSDK::AvatarSDK SHARED IMPORTED)
set_target_properties(AvatarSDK::AvatarSDK PROPERTIES
  IMPORTED_LOCATION "${_AVATAR_SDK_LIBRARY}"
  INTERFACE_INCLUDE_DIRECTORIES "${_AVATAR_SDK_INCLUDE_DIR}"
)

add_subdirectory(core)
add_subdirectory(app)
add_subdirectory(tools)

configure_file(plugin.yaml "${CMAKE_CURRENT_BINARY_DIR}/plugin.yaml" COPYONLY)

install(
  FILES plugin.yaml README.md
  DESTINATION plugins/sharpa_avatar
  COMPONENT avatar
)
