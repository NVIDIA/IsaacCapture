# SPDX-FileCopyrightText: Copyright (c) 2025-2026 Avatar SDK contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
  message(STATUS "Skipping sharpa_avatar plugin (Linux only)")
  return()
endif()

include(GNUInstallDirs)

set(_AVATAR_SDK_ROOT "/opt/avatar-sdk")
set(_AVATAR_SDK_INCLUDE_DIR "${_AVATAR_SDK_ROOT}/include")
set(_AVATAR_SDK_LIBRARY_DIR "${_AVATAR_SDK_ROOT}/lib")
set(_AVATAR_SDK_CONFIG "${_AVATAR_SDK_ROOT}/share/sdk_config.json")
set(_AVATAR_VERSION_FILE "${_AVATAR_SDK_ROOT}/share/Version")
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
    "BUILD_PLUGIN_SHARPA_AVATAR=ON but the Avatar SDK under ${_AVATAR_SDK_ROOT} is missing; "
    "skipping this plugin. Run src/plugins/sharpa_avatar/install_avatar_sdk.sh.")
  return()
endif()

file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/install_avatar_sdk.sh" _AVATAR_SDK_PIN_LINES
  REGEX "^production_version=")
list(LENGTH _AVATAR_SDK_PIN_LINES _AVATAR_SDK_PIN_COUNT)
if(NOT _AVATAR_SDK_PIN_COUNT EQUAL 1)
  message(FATAL_ERROR "Could not read production_version from install_avatar_sdk.sh.")
endif()
list(GET _AVATAR_SDK_PIN_LINES 0 _AVATAR_SDK_PIN_LINE)
string(REGEX REPLACE "^production_version=\"([^\"]+)\"" "\\1" _AVATAR_SDK_PACKAGE_VERSION
  "${_AVATAR_SDK_PIN_LINE}")
if(_AVATAR_SDK_PACKAGE_VERSION STREQUAL "" OR _AVATAR_SDK_PACKAGE_VERSION STREQUAL "${_AVATAR_SDK_PIN_LINE}")
  message(FATAL_ERROR "Could not parse production_version from install_avatar_sdk.sh.")
endif()

set(_AVATAR_SDK_BUILD_TYPE "")
set(_AVATAR_SDK_VERSION_FIELD "")
file(STRINGS "${_AVATAR_VERSION_FILE}" _AVATAR_VERSION_LINES)
foreach(_AVATAR_VERSION_LINE IN LISTS _AVATAR_VERSION_LINES)
  if(_AVATAR_VERSION_LINE MATCHES "^BUILD_TYPE=(.*)$")
    set(_AVATAR_SDK_BUILD_TYPE "${CMAKE_MATCH_1}")
  elseif(_AVATAR_VERSION_LINE MATCHES "^VERSION=(.*)$")
    set(_AVATAR_SDK_VERSION_FIELD "${CMAKE_MATCH_1}")
  endif()
endforeach()
if(_AVATAR_SDK_VERSION_FIELD STREQUAL "" OR NOT _AVATAR_SDK_BUILD_TYPE STREQUAL "Production")
  message(FATAL_ERROR
    "Expected a production Avatar SDK, found ${_AVATAR_SDK_VERSION_FIELD} ${_AVATAR_SDK_BUILD_TYPE}. "
    "Run src/plugins/sharpa_avatar/install_avatar_sdk.sh.")
endif()

execute_process(
  COMMAND dpkg-query -W -f=\${Version} avatar-sdk
  RESULT_VARIABLE _AVATAR_DPKG_RESULT
  OUTPUT_VARIABLE _AVATAR_DPKG_VERSION
  OUTPUT_STRIP_TRAILING_WHITESPACE
  ERROR_QUIET
)
if(NOT _AVATAR_DPKG_RESULT EQUAL 0 OR NOT _AVATAR_DPKG_VERSION STREQUAL _AVATAR_SDK_PACKAGE_VERSION)
  message(FATAL_ERROR
    "Expected avatar-sdk ${_AVATAR_SDK_PACKAGE_VERSION}, but ${_AVATAR_DPKG_VERSION} is installed. "
    "Run src/plugins/sharpa_avatar/install_avatar_sdk.sh.")
endif()
message(STATUS "Avatar SDK ${_AVATAR_SDK_VERSION_FIELD} (${_AVATAR_SDK_BUILD_TYPE}), package ${_AVATAR_DPKG_VERSION}")

add_library(AvatarSDK::AvatarSDK SHARED IMPORTED)
set_target_properties(AvatarSDK::AvatarSDK PROPERTIES
  IMPORTED_LOCATION "${_AVATAR_SDK_LIBRARY}"
  INTERFACE_INCLUDE_DIRECTORIES "${_AVATAR_SDK_INCLUDE_DIR}"
)

# Parse the SDK's human landmark names for the OpenXR mapping.
include(FetchContent)
FetchContent_Declare(
  nlohmann_json
  URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
)
set(JSON_BuildTests OFF CACHE INTERNAL "")
isaac_teleop_third_party_scope_begin()
FetchContent_MakeAvailable(nlohmann_json)
isaac_teleop_third_party_scope_end()

add_subdirectory(core)
add_subdirectory(app)
add_subdirectory(tools)

configure_file(plugin.yaml "${CMAKE_CURRENT_BINARY_DIR}/plugin.yaml" COPYONLY)

install(
  FILES plugin.yaml README.md
  DESTINATION plugins/sharpa_avatar
  COMPONENT avatar
)
