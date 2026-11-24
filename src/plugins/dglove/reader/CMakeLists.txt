# SPDX-FileCopyrightText: Copyright (c) 2026 mwang259. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_executable(dglove_reader main.cpp)
target_link_libraries(dglove_reader PRIVATE
    deviceio::deviceio_session deviceio::deviceio_trackers oxr::oxr_core dglove::format)

install(TARGETS dglove_reader RUNTIME DESTINATION plugins/dglove COMPONENT dglove)
