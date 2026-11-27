# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_executable(posix_utils_tests test_file_descriptor.cpp)
target_link_libraries(posix_utils_tests PRIVATE posix_utils::posix_utils Catch2::Catch2WithMain)
catch_discover_tests(posix_utils_tests ADD_TAGS_AS_LABELS)
