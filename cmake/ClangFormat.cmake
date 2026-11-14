# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Clang-Format enforcement for Linux builds
# Defines targets:
#  - clang_format_check: fails if formatting changes would be applied
#  - clang_format_fix: applies formatting in-place

# Guard: only configure once
if (TARGET clang_format_check)
  return()
endif()

# Option to enable enforcement (default set by caller)
if (NOT DEFINED ENABLE_CLANG_FORMAT_CHECK)
  set(ENABLE_CLANG_FORMAT_CHECK OFF)
endif()

# Find clang-format binary
find_program(CLANG_FORMAT_EXE NAMES clang-format-14)

# Collect project source files (exclude external deps and build artifacts)
set(_cf_glob_dirs
  ${CMAKE_SOURCE_DIR}/src
  ${CMAKE_SOURCE_DIR}/examples
)
set(_cf_patterns
  *.h
  *.hh
  *.hpp
  *.hxx
  *.c
  *.cc
  *.cpp
  *.cxx
)

set(_cf_exclude_patterns
  ".*/build/.*"
  ".*/deps/.*"
  ".*/manus/ManusSDK/.*"
  ".*/wuji_glove/wuji-sdk-c-.*"
  ".*/node_modules/.*"
  ".*/third_party/.*"
  ".*/venv/.*"
  ".*/\\.venv/.*"
  ".*/__pycache__/.*"
)

set(CLANG_FORMAT_SOURCES)
foreach(dir IN LISTS _cf_glob_dirs)
  if (EXISTS ${dir})
    foreach(pat IN LISTS _cf_patterns)
      file(GLOB_RECURSE _found FOLLOW_SYMLINKS ${dir}/${pat})
      foreach(f IN LISTS _found)
        # Exclude patterns
        set(_skip FALSE)
        foreach(ex IN LISTS _cf_exclude_patterns)
          if (f MATCHES ${ex})
            set(_skip TRUE)
          endif()
        endforeach()
        if (NOT _skip)
          list(APPEND CLANG_FORMAT_SOURCES ${f})
        endif()
      endforeach()
    endforeach()
  endif()
endforeach()
list(REMOVE_DUPLICATES CLANG_FORMAT_SOURCES)

# Helper to build command line
function(_build_clang_format_cmd outvar fix)
  if (fix)
    set(_args -i)
  else()
    set(_args --dry-run -Werror)
  endif()
  # Prefer repo style if present
  if (EXISTS ${CMAKE_SOURCE_DIR}/.clang-format)
    list(APPEND _args -style=file)
  endif()
  set(_cmd ${CLANG_FORMAT_EXE} ${_args} ${CLANG_FORMAT_SOURCES})
  set(${outvar} "${_cmd}" PARENT_SCOPE)
endfunction()

# Define targets only if we have files
if (CLANG_FORMAT_SOURCES)
  if (CLANG_FORMAT_EXE)
    # Check target (optionally part of ALL)
    _build_clang_format_cmd(_cf_check_cmd FALSE)
    if (ENABLE_CLANG_FORMAT_CHECK)
      add_custom_target(clang_format_check ALL
        COMMAND ${_cf_check_cmd}
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        COMMENT "Enforcing clang-format: verifying formatting is clean")
    else()
      add_custom_target(clang_format_check
        COMMAND ${_cf_check_cmd}
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        COMMENT "Check clang-format formatting (not enforced)")
    endif()

    # Fix target
    _build_clang_format_cmd(_cf_fix_cmd TRUE)
    add_custom_target(clang_format_fix
      COMMAND ${_cf_fix_cmd}
      WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
      COMMENT "Applying clang-format to project sources")

    # Also expose as a CTest if testing is enabled
    if (BUILD_TESTING)
      add_test(NAME clang_format COMMAND ${_cf_check_cmd})
    endif()
  else()
    if (ENABLE_CLANG_FORMAT_CHECK)
      message(FATAL_ERROR "clang-format not found but ENABLE_CLANG_FORMAT_CHECK is ON. Install clang-format (e.g., apt install clang-format).")
    else()
      message(STATUS "clang-format not found; skipping clang-format targets. Install clang-format to enable formatting checks.")
    endif()
  endif()
else()
  message(STATUS "No C/C++ sources found for clang-format.")
endif()
