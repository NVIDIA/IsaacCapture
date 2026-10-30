// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tcp_socket.hpp"
#include "transport.hpp"

#include <chrono>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>

namespace remote_plugin_session
{

//! Where a client connects: plain "host[:port]" (TCP) or "ws://host[:port][/path]" (WebSocket).
struct RemoteAddress
{
    Endpoint endpoint;
    std::optional<std::string> ws_path; //!< set for ws://
};

//! Throws std::invalid_argument; wss:// is refused (terminate TLS in a proxy in front of the bridge).
RemoteAddress parse_remote(const std::string& text, uint16_t default_port);
//! Connects (and for ws:// completes the upgrade) within `timeout`; throws std::runtime_error.
std::unique_ptr<ITransport> connect_transport(const RemoteAddress& remote,
                                              std::chrono::milliseconds timeout = kConnectTimeout);

} // namespace remote_plugin_session
