# SPDX-FileCopyrightText: Copyright (c) 2026 ORBBEC INC. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_executable(ego_media_tests test_export.cpp)
target_link_libraries(ego_media_tests PRIVATE ego::mcap_support Catch2::Catch2WithMain)
target_compile_definitions(ego_media_tests PRIVATE
    EGO_EXPORTER_PATH="$<TARGET_FILE:ego_mcap_export_media>")
add_dependencies(ego_media_tests ego_mcap_export_media)
catch_discover_tests(ego_media_tests ADD_TAGS_AS_LABELS)
