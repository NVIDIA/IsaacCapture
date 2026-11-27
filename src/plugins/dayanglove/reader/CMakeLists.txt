# SPDX-FileCopyrightText: Copyright (c) 2026 mwang259. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_executable(dayanglove_reader main.cpp)
target_link_libraries(dayanglove_reader PRIVATE
    deviceio::deviceio_session deviceio::deviceio_trackers oxr::oxr_core dayanglove::format)

install(TARGETS dayanglove_reader RUNTIME DESTINATION plugins/dayanglove COMPONENT dayanglove)
