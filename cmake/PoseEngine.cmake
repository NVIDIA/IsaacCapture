# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Pose Provider engine adapters: each adapter registers from its own directory, and pose_engine_select() picks the
# one adapter that executables link. See src/plugins/pose_provider/README.md.

function(_pose_engine_check_distribution caller value)
    if(NOT value MATCHES "^(redistributable|user-built-only)$")
        message(FATAL_ERROR "${caller}: DISTRIBUTION must be redistributable or user-built-only, not '${value}'")
    endif()
endfunction()

# pose_engine_register(<target> DISTRIBUTION redistributable|user-built-only)
#
# Records <target> as an engine adapter. A user-built-only adapter adds no install rules, and a pip build (SKBUILD)
# with it fails, because the root install walk would package its executables. Adapters that do not use
# pose_engine_add() call this directly, so that the one-adapter check sees them.
function(pose_engine_register target)
    cmake_parse_arguments(PARSE_ARGV 1 arg "" "DISTRIBUTION" "")
    if(arg_UNPARSED_ARGUMENTS)
        message(FATAL_ERROR "pose_engine_register(${target}): unexpected arguments: ${arg_UNPARSED_ARGUMENTS}")
    endif()
    _pose_engine_check_distribution("pose_engine_register(${target})" "${arg_DISTRIBUTION}")
    get_property(selected GLOBAL PROPERTY POSE_ENGINE_SELECTED)
    if(selected)
        message(FATAL_ERROR "${target} registered after pose_engine_select(); add its directory before that call.")
    endif()
    if(arg_DISTRIBUTION STREQUAL "user-built-only" AND SKBUILD)
        message(FATAL_ERROR "${target} is user-built-only: it cannot be part of a pip build. "
                            "Build it with CMake directly.")
    endif()
    set_property(GLOBAL APPEND PROPERTY POSE_ENGINE_ADAPTERS "${target}")
    set_property(GLOBAL PROPERTY POSE_ENGINE_DISTRIBUTION_${target} "${arg_DISTRIBUTION}")
endfunction()

# pose_engine_add(<NAME>
#     TARGET <adapter target>
#     DISTRIBUTION redistributable|user-built-only
#     SCRIPT <what builds the engine library; named in errors>
#     [INCLUDE_DIRS <dir>...]   # use . for <NAME>_ROOT itself
#     [HEADERS <file>...]       # checked only
#     [LIBRARIES <file>...])    # for example lib/libfoo.so
#
# Paths are relative to <NAME>_ROOT. Declares POSE_ENGINE_<NAME> (default OFF) and <NAME>_ROOT. When the option is
# ON, checks that every listed path exists, registers TARGET, and sets POSE_ENGINE_<NAME>_INCLUDE_DIRS and
# POSE_ENGINE_<NAME>_LIBRARIES (absolute, in the order given) in the caller's scope. Linking those libraries puts
# their directories in the build RPATH of executables, not the install RPATH; their own dependencies must resolve
# through their own RPATH.
function(pose_engine_add name)
    cmake_parse_arguments(PARSE_ARGV 1 arg "" "TARGET;DISTRIBUTION;SCRIPT" "INCLUDE_DIRS;HEADERS;LIBRARIES")
    if(arg_UNPARSED_ARGUMENTS OR arg_KEYWORDS_MISSING_VALUES)
        message(FATAL_ERROR "pose_engine_add(${name}): unexpected arguments or empty keywords: "
                            "${arg_UNPARSED_ARGUMENTS} ${arg_KEYWORDS_MISSING_VALUES}")
    endif()
    if(NOT arg_TARGET OR NOT arg_SCRIPT)
        message(FATAL_ERROR "pose_engine_add(${name}): TARGET and SCRIPT are required")
    endif()
    if(NOT arg_INCLUDE_DIRS AND NOT arg_HEADERS AND NOT arg_LIBRARIES)
        message(FATAL_ERROR "pose_engine_add(${name}): list at least one of INCLUDE_DIRS, HEADERS or LIBRARIES")
    endif()
    _pose_engine_check_distribution("pose_engine_add(${name})" "${arg_DISTRIBUTION}")

    option(POSE_ENGINE_${name} "Build the ${name} Pose Provider engine adapter" OFF)
    set(${name}_ROOT "" CACHE PATH "Directory of the ${name} engine library")
    if(NOT POSE_ENGINE_${name})
        return()
    endif()

    set(root "${${name}_ROOT}")
    if(NOT root)
        message(FATAL_ERROR "POSE_ENGINE_${name} is ON, but ${name}_ROOT is not set. "
                            "Run ${arg_SCRIPT}, then set ${name}_ROOT to its output directory.")
    endif()
    if(NOT IS_ABSOLUTE "${root}")
        message(FATAL_ERROR "${name}_ROOT must be an absolute path, not '${root}'")
    endif()
    set(missing "")
    set(include_dirs "")
    foreach(dir IN LISTS arg_INCLUDE_DIRS)
        if(IS_DIRECTORY "${root}/${dir}")
            list(APPEND include_dirs "${root}/${dir}")
        else()
            list(APPEND missing "${dir}/")
        endif()
    endforeach()
    foreach(header IN LISTS arg_HEADERS)
        if(NOT EXISTS "${root}/${header}")
            list(APPEND missing "${header}")
        endif()
    endforeach()
    set(libraries "")
    foreach(library IN LISTS arg_LIBRARIES)
        if(EXISTS "${root}/${library}")
            list(APPEND libraries "${root}/${library}")
        else()
            list(APPEND missing "${library}")
        endif()
    endforeach()
    if(missing)
        # One path per line, so that CMake's message wrapping never splits a path.
        list(JOIN missing "\n  " missing)
        message(FATAL_ERROR "POSE_ENGINE_${name} is ON, but ${name}_ROOT (${root}) lacks:\n  ${missing}\n"
                            "Run ${arg_SCRIPT}, then set ${name}_ROOT to its output directory.")
    endif()

    pose_engine_register("${arg_TARGET}" DISTRIBUTION "${arg_DISTRIBUTION}")
    set(POSE_ENGINE_${name}_INCLUDE_DIRS "${include_dirs}" PARENT_SCOPE)
    set(POSE_ENGINE_${name}_LIBRARIES "${libraries}" PARENT_SCOPE)
endfunction()

# pose_engine_select(<out_var> <default>)
#
# Sets <out_var> to the one registered adapter, or to <default> when none is registered, and <out_var>_DISTRIBUTION
# to its distribution; <default> is redistributable. More than one adapter fails. Call it once, after every adapter
# directory has been added; a later registration fails.
function(pose_engine_select out_var default)
    set_property(GLOBAL PROPERTY POSE_ENGINE_SELECTED TRUE)
    get_property(adapters GLOBAL PROPERTY POSE_ENGINE_ADAPTERS)
    list(REMOVE_DUPLICATES adapters)
    list(LENGTH adapters count)
    if(count GREATER 1)
        message(FATAL_ERROR "More than one Pose Provider engine adapter is enabled (${adapters}). Enable one.")
    endif()
    if(count EQUAL 0)
        set(${out_var} "${default}" PARENT_SCOPE)
        set(${out_var}_DISTRIBUTION "redistributable" PARENT_SCOPE)
        return()
    endif()
    if(NOT TARGET "${adapters}")
        message(FATAL_ERROR "Pose Provider engine adapter ${adapters} is registered but its target does not exist.")
    endif()
    get_property(distribution GLOBAL PROPERTY POSE_ENGINE_DISTRIBUTION_${adapters})
    set(${out_var} "${adapters}" PARENT_SCOPE)
    set(${out_var}_DISTRIBUTION "${distribution}" PARENT_SCOPE)
endfunction()
