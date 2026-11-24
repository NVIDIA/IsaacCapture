# SPDX-FileCopyrightText: Copyright (c) 2026 mwang259. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(dglove_format INTERFACE)
target_include_directories(dglove_format INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/inc)
target_link_libraries(dglove_format INTERFACE isaacteleop_schema)
add_library(dglove::format ALIAS dglove_format)
