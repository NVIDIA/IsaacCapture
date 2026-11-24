# SPDX-FileCopyrightText: Copyright (c) 2026 mwang259. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(dayanglove_format INTERFACE)
target_include_directories(dayanglove_format INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/inc)
target_link_libraries(dayanglove_format INTERFACE isaacteleop_schema)
add_library(dayanglove::format ALIAS dayanglove_format)
