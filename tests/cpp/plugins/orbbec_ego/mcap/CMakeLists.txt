# SPDX-FileCopyrightText: Copyright (c) 2026 ORBBEC INC. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_executable(ego_mcap_tests test_merge.cpp)
target_link_libraries(ego_mcap_tests PRIVATE ego::mcap_merge Catch2::Catch2WithMain)
catch_discover_tests(ego_mcap_tests ADD_TAGS_AS_LABELS)
