# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.16)
project(argus_sender LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
include(CTest)
find_package(Threads REQUIRED)
include(cmake/YamlCpp.cmake)
set(ARGUS_SENDER_ROOT "${CMAKE_CURRENT_SOURCE_DIR}")
include(cmake/Tests.cmake)
