# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

find_package(Threads REQUIRED)

add_executable(soma_body_pusher soma_body_pusher.cpp)
target_link_libraries(soma_body_pusher PRIVATE
    pusherio::pusherio
    oxr::oxr_core
    isaacteleop_schema
    Threads::Threads
)
install(TARGETS soma_body_pusher
    RUNTIME DESTINATION examples/soma_body_publisher/cpp
)
