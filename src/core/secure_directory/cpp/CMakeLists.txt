# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(secure_directory STATIC
    secure_directory.cpp
    inc/secure_directory/secure_directory.hpp
)
target_include_directories(secure_directory INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/inc>
)
add_library(secure_directory::secure_directory ALIAS secure_directory)

if(NOT WIN32)
    target_link_libraries(secure_directory PRIVATE posix_utils::posix_utils)
endif()
