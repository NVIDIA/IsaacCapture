# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Configured by the pose_engine_* tests with fake engine adapters. Prints the selected adapter.
cmake_minimum_required(VERSION 3.24)
project(pose_engine_fixture NONE)

include(${HELPER})

# A complete engine library directory that a test can point <NAME>_ROOT at.
file(WRITE ${CMAKE_BINARY_DIR}/engine/include/fake_engine.h "")
file(WRITE ${CMAKE_BINARY_DIR}/engine/lib/libfake_engine.so "")

pose_engine_add(REDIST
    TARGET redist_adapter
    DISTRIBUTION redistributable
    SCRIPT build_redist.sh
    INCLUDE_DIRS include
    HEADERS include/fake_engine.h
    LIBRARIES lib/libfake_engine.so
)
if(POSE_ENGINE_REDIST)
    add_library(redist_adapter INTERFACE)
    pose_engine_register(redist_adapter DISTRIBUTION redistributable) # A repeated registration is one adapter.
endif()

pose_engine_add(USER_BUILT
    TARGET user_built_adapter
    DISTRIBUTION user-built-only
    SCRIPT build_user_built.sh
    LIBRARIES lib/libfake_engine.so
)
if(POSE_ENGINE_USER_BUILT)
    add_library(user_built_adapter INTERFACE)
endif()

# Lists paths that the engine library directory lacks.
pose_engine_add(PARTIAL
    TARGET partial_adapter
    DISTRIBUTION redistributable
    SCRIPT build_partial.sh
    INCLUDE_DIRS include absent_include
    HEADERS include/fake_engine.h include/absent_engine.h
    LIBRARIES lib/libfake_engine.so lib/libabsent_engine.so
)
if(POSE_ENGINE_PARTIAL)
    add_library(partial_adapter INTERFACE)
endif()

# An adapter that registers with plain CMake instead of pose_engine_add().
option(PLAIN_ADAPTER "Add an adapter that registers without the helper" OFF)
set(PLAIN_DISTRIBUTION "redistributable" CACHE STRING "DISTRIBUTION of the plain adapter")
if(PLAIN_ADAPTER)
    add_library(plain_adapter INTERFACE)
    pose_engine_register(plain_adapter DISTRIBUTION ${PLAIN_DISTRIBUTION})
endif()

option(MISSPELLED_KEYWORD "Call pose_engine_add() with HEADER instead of HEADERS" OFF)
if(MISSPELLED_KEYWORD)
    pose_engine_add(MISSPELLED TARGET misspelled_adapter DISTRIBUTION redistributable SCRIPT build.sh HEADER x.h)
endif()

add_library(stub_adapter INTERFACE)
pose_engine_select(selected stub_adapter)
message(STATUS "Selected adapter: ${selected} (${selected_DISTRIBUTION})")

option(LATE_ADAPTER "Register an adapter after pose_engine_select()" OFF)
if(LATE_ADAPTER)
    add_library(late_adapter INTERFACE)
    pose_engine_register(late_adapter DISTRIBUTION redistributable)
endif()
