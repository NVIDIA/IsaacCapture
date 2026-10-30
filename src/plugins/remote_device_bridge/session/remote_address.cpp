// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "inc/remote_plugin_session/remote_address.hpp"

#include "inc/remote_plugin_session/tcp_transport.hpp"
#include "inc/remote_plugin_session/websocket_transport.hpp"

#include <stdexcept>
#include <utility>

namespace remote_plugin_session
{

RemoteAddress parse_remote(const std::string& text, uint16_t default_port)
{
    const std::string scheme = "ws://";
    if (text.rfind("wss://", 0) == 0)
    {
        throw std::invalid_argument("wss:// is not supported ('" + text + "'); terminate TLS in a proxy, use ws://");
    }
    RemoteAddress remote;
    if (text.rfind(scheme, 0) != 0)
    {
        remote.endpoint = parse_endpoint(text, default_port);
        return remote;
    }
    std::string authority = text.substr(scheme.size());
    const size_t slash = authority.find('/');
    remote.ws_path = slash == std::string::npos ? "/" : authority.substr(slash);
    if (slash != std::string::npos)
    {
        authority.resize(slash);
    }
    remote.endpoint = parse_endpoint(authority, default_port);
    return remote;
}

std::unique_ptr<ITransport> connect_transport(const RemoteAddress& remote, std::chrono::milliseconds timeout)
{
    TcpSocket socket = TcpSocket::connect(remote.endpoint, timeout);
    if (!remote.ws_path)
    {
        return std::make_unique<TcpTransport>(std::move(socket));
    }
    return std::make_unique<WebSocketTransport>(std::move(socket), WebSocketTransport::Role::Client,
                                                remote.endpoint.host + ":" + std::to_string(remote.endpoint.port),
                                                *remote.ws_path, timeout);
}

} // namespace remote_plugin_session
