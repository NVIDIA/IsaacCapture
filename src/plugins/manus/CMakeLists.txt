# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

include(GNUInstallDirs)

option(MANUS_ADD_SDK_TO_BUILD_RPATH "Add Manus SDK lib dir to BUILD_RPATH for plugin and tools" ON)

set(_MANUS_DEFAULT_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/ManusSDK")

if(NOT DEFINED MANUS_SDK_ROOT)
  if(DEFINED ENV{MANUS_SDK_ROOT})
    set(MANUS_SDK_ROOT "$ENV{MANUS_SDK_ROOT}" CACHE PATH "Path to Manus SDK root containing include/ and lib/" FORCE)
  else()
    set(MANUS_SDK_ROOT "${_MANUS_DEFAULT_ROOT}" CACHE PATH "Path to Manus SDK root containing include/ and lib/")
  endif()
elseif(NOT EXISTS "${MANUS_SDK_ROOT}/include/ManusSDK.h")
  if(EXISTS "${_MANUS_DEFAULT_ROOT}/include/ManusSDK.h")
    message(STATUS "Cached MANUS_SDK_ROOT='${MANUS_SDK_ROOT}' is invalid; falling back to local default '${_MANUS_DEFAULT_ROOT}'")
    set(MANUS_SDK_ROOT "${_MANUS_DEFAULT_ROOT}" CACHE PATH "Path to Manus SDK root containing include/ and lib/" FORCE)
  endif()
endif()

set(MANUS_SDK_INCLUDE_DIR "${MANUS_SDK_ROOT}/include" CACHE PATH "Manus SDK include path" FORCE)

set(_MANUS_LIBDIR_CANDIDATES
  "${MANUS_SDK_ROOT}/lib"
  "${MANUS_SDK_ROOT}/lib64"
)
set(_MANUS_LIBDIR "")
foreach(_cand IN LISTS _MANUS_LIBDIR_CANDIDATES)
  if(EXISTS "${_cand}")
    set(_MANUS_LIBDIR "${_cand}")
    break()
  endif()
endforeach()
if(NOT _MANUS_LIBDIR)
  set(_MANUS_LIBDIR "${MANUS_SDK_ROOT}/lib")
endif()
set(MANUS_SDK_LIBRARY_DIR "${_MANUS_LIBDIR}" CACHE PATH "Manus SDK library path" FORCE)

set(MANUS_SHOULD_BUILD ON)
set(MANUS_SKIP_REASON "")

if(NOT EXISTS "${MANUS_SDK_INCLUDE_DIR}/ManusSDK.h")
  set(MANUS_SHOULD_BUILD OFF)
  set(MANUS_SKIP_REASON "Headers not found in ${MANUS_SDK_INCLUDE_DIR}")
endif()

if(MANUS_SHOULD_BUILD)
  if(NOT EXISTS "${MANUS_SDK_LIBRARY_DIR}/libManusSDK.so")
    if(NOT EXISTS "${MANUS_SDK_LIBRARY_DIR}/libManusSDK_Integrated.so")
      set(MANUS_SHOULD_BUILD OFF)
      set(MANUS_SKIP_REASON "Libraries not found in ${MANUS_SDK_LIBRARY_DIR}")
    endif()
  endif()
endif()

set(_MANUS_LIB "")
if(MANUS_SHOULD_BUILD)
  set(_MANUS_CANDIDATES
    "${MANUS_SDK_LIBRARY_DIR}/libManusSDK_Integrated.so"
    "${MANUS_SDK_LIBRARY_DIR}/libManusSDK.so"
  )
  foreach(_cand IN LISTS _MANUS_CANDIDATES)
    if(EXISTS "${_cand}")
      set(_MANUS_LIB "${_cand}")
      break()
    endif()
  endforeach()
  if(NOT _MANUS_LIB)
    set(MANUS_SHOULD_BUILD OFF)
    set(MANUS_SKIP_REASON "No shared library found under ${MANUS_SDK_LIBRARY_DIR}")
  endif()
endif()

# The calibration engine and wire format need no Manus SDK, so they build, and their tests run, without it.
# POSIX file and process calls make it Linux-only.
if(UNIX AND NOT APPLE)
  add_subdirectory(calibration/core)
endif()

if(NOT MANUS_SHOULD_BUILD)
  message(STATUS "Skipping Manus plugin build: ${MANUS_SKIP_REASON}")
  add_custom_target(manus_plugin_core COMMAND ${CMAKE_COMMAND} -E echo "Skipping manus_plugin_core: ${MANUS_SKIP_REASON}")
  add_custom_target(manus_hand_plugin COMMAND ${CMAKE_COMMAND} -E echo "Skipping manus_hand_plugin: ${MANUS_SKIP_REASON}")
  add_custom_target(manus_hand_tracker_printer COMMAND ${CMAKE_COMMAND} -E echo "Skipping manus_hand_tracker_printer: ${MANUS_SKIP_REASON}")
  add_custom_target(manus_glove_calibrator COMMAND ${CMAKE_COMMAND} -E echo "Skipping manus_glove_calibrator: ${MANUS_SKIP_REASON}")
  return()
endif()

# Create imported library target for Manus SDK
# Note: The Manus SDK library doesn't have a SONAME, so we need to use IMPORTED_NO_SONAME
add_library(ManusSDK::ManusSDK SHARED IMPORTED)

get_filename_component(_MANUS_LIB_NAME "${_MANUS_LIB}" NAME)

set_target_properties(ManusSDK::ManusSDK PROPERTIES
  IMPORTED_LOCATION "${_MANUS_LIB}"
  IMPORTED_SONAME "${_MANUS_LIB_NAME}"
  IMPORTED_NO_SONAME TRUE
  INTERFACE_INCLUDE_DIRECTORIES "${MANUS_SDK_INCLUDE_DIR}"
)

set(_ISAAC_MANUS_BASE_BUILD_RPATH "$ORIGIN;$ORIGIN/../lib")
set(_ISAAC_MANUS_EFFECTIVE_BUILD_RPATH "${_ISAAC_MANUS_BASE_BUILD_RPATH}")
if(MANUS_ADD_SDK_TO_BUILD_RPATH)
  list(APPEND _ISAAC_MANUS_EFFECTIVE_BUILD_RPATH "${MANUS_SDK_LIBRARY_DIR}")
endif()
list(JOIN _ISAAC_MANUS_EFFECTIVE_BUILD_RPATH ";" _ISAAC_MANUS_EFFECTIVE_BUILD_RPATH_STR)

add_subdirectory(core)
add_library(isaac::manus_plugin ALIAS manus_plugin_core)
add_subdirectory(app)
add_subdirectory(tools)
add_subdirectory(calibration)

# Install Manus SDK shared library
install(FILES "${_MANUS_LIB}"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}
    COMPONENT manus
)
