# SPDX-FileCopyrightText: Copyright (c) 2026 j-mex. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_executable(jmex_plugin
    main.cpp
    receiver_session.cpp
    receiver_session.hpp
)

target_link_libraries(jmex_plugin
    PRIVATE
        isaac::jmex_plugin_core
        Moxi::Receiver
        log_bridge::log_bridge_core
)

install(TARGETS jmex_plugin RUNTIME DESTINATION plugins/jmex)
