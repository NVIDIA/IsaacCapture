# SPDX-FileCopyrightText: Copyright (c) 2026 j-mex. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_executable(jmex_plugin
    main.cpp
    receiver_session.cpp
    receiver_session.hpp
)

target_link_libraries(jmex_plugin
    PRIVATE
        jmex::jmex_plugin_core
        # Static (libmoxi_receiver.a): the installed executable loads no shared library from the SDK.
        Moxi::Receiver
        log_bridge::log_bridge_core
)

install(TARGETS jmex_plugin RUNTIME DESTINATION plugins/jmex)
