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

//! Ports the bridge listens on (TCP and ws://) and clients dial when the address names none.
constexpr uint16_t kDefaultBridgePort = 47500;
constexpr uint16_t kDefaultBridgeWsPort = 47501;

//! Parses "host[:port]" or "[v6-host][:port]"; a missing port falls back to default_port. Throws
//! std::invalid_argument.
Endpoint parse_endpoint(const std::string& text, uint16_t default_port);

//! Where a client connects: plain "host[:port]" (TCP) or "ws://host[:port][/path]" (WebSocket).
struct RemoteAddress
{
    Endpoint endpoint;
    std::optional<std::string> ws_path; //!< set for ws://
};

//! Throws std::invalid_argument; wss:// is refused (terminate TLS in a proxy in front of the bridge).
RemoteAddress parse_remote(const std::string& text, uint16_t default_port);
//! parse_remote with the bridge's default port for the address's scheme.
RemoteAddress parse_bridge_remote(const std::string& text);
//! Connects (and for ws:// completes the upgrade) within `timeout`; throws std::runtime_error.
std::unique_ptr<ITransport> connect_transport(const RemoteAddress& remote,
                                              std::chrono::milliseconds timeout = kConnectTimeout);

} // namespace remote_plugin_session
