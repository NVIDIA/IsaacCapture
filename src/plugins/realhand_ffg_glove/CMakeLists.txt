# SPDX-FileCopyrightText: Copyright (c) 2026 RealHand. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_executable(realhand_ffg_glove_plugin
    main.cpp
    realhand_ffg_glove_plugin.cpp
    realhand_ffg_glove_protocol.cpp
    realhand_ffg_glove_serial.cpp
)

target_link_libraries(realhand_ffg_glove_plugin PRIVATE
    Teleop::plugin_utils
    pusherio::pusherio
    oxr::oxr_core
    isaacteleop_schema
)

install(TARGETS realhand_ffg_glove_plugin
    RUNTIME DESTINATION plugins/realhand_ffg_glove
    COMPONENT realhand_ffg_glove
)
install(FILES plugin.yaml README.md
    DESTINATION plugins/realhand_ffg_glove
    COMPONENT realhand_ffg_glove
)
