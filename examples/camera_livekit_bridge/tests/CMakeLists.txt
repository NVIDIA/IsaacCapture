# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# One test executable per subdirectory (each with its own CMakeLists.txt); Python tests live in tests/python.
file(GLOB _test_dirs LIST_DIRECTORIES true RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/*)
foreach(_dir IN LISTS _test_dirs)
    if(IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/${_dir} AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${_dir}/CMakeLists.txt)
        add_subdirectory(${_dir})
    endif()
endforeach()
