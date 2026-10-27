# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(deviceio_py_utils INTERFACE
    inc/deviceio_py_utils/session.hpp
)

target_include_directories(deviceio_py_utils INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/inc
)

# PyRemoteSession owns the TrackerSampleStreamReader (mcap/sample_stream.hpp) that its client feeds.
target_link_libraries(deviceio_py_utils INTERFACE deviceio::deviceio_session mcap::mcap_core)

add_library(deviceio::deviceio_py_utils ALIAS deviceio_py_utils)
