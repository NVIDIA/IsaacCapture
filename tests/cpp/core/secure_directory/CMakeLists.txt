# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_executable(secure_directory_tests test_secure_directory.cpp)
target_link_libraries(secure_directory_tests PRIVATE
    secure_directory::secure_directory
    test_support::temporary_directory
    Catch2::Catch2WithMain
)
catch_discover_tests(secure_directory_tests ADD_TAGS_AS_LABELS)
